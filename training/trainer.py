"""Joint NER and Sentiment Training Module.

Implements joint training with configurable loss weighting and
curriculum learning schedule.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .preprocessing import NER_LABELS, LABEL_TO_ID, ID_TO_LABEL

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Loss weights
    ner_loss_weight: float = 1.0
    sentiment_loss_weight: float = 0.5

    # Curriculum (optional): adjust weights over epochs
    use_curriculum: bool = True
    curriculum_ner_start: float = 1.0
    curriculum_ner_end: float = 0.5
    curriculum_sentiment_start: float = 0.3
    curriculum_sentiment_end: float = 1.0

    # Scheduler
    warmup_epochs: int = 1
    scheduler_type: str = "cosine"  # "cosine" or "linear"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_best_only: bool = True

    # Evaluation
    eval_every_n_steps: int = 100
    log_every_n_steps: int = 10

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Label smoothing for NER
    label_smoothing: float = 0.1

    # Mixed precision (not supported on MPS)
    use_amp: bool = False

    # Two-stage labeling: include sentiment_expanded_mentions in sentiment mask
    use_expanded_sentiment: bool = False

    # Global-attention dropout (closes train/inference regime gap).
    # With probability `global_attn_dropout_prob`, the per-sample global mask is
    # replaced with CLS-only (no entity-token globals). The effective probability
    # ramps from 0 to the target value over the first `warmup_frac` of total
    # optimizer steps; after that it stays at the target. Default 0 = disabled.
    global_attn_dropout_prob: float = 0.0
    global_attn_dropout_warmup_frac: float = 0.0

    # Dual-eval: run a second validation pass with CLS-only global attention
    # to monitor the inference regime. Disable to save time.
    eval_cls_only_regime: bool = True


@dataclass
class TrainingState:
    """Tracks training state."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_ner_f1: float = 0.0
    best_sentiment_mse: float = float("inf")
    best_cls_only_f1: float = 0.0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


class JointTrainer:
    """
    Trainer for joint NER and Sentiment learning.

    Supports:
    - Joint loss with configurable weighting
    - Curriculum learning (adjusting weights over training)
    - Mixed precision training (CUDA only)
    - Gradient accumulation
    - Checkpointing with best model tracking
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        label_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The FinancialEntitySentimentModel
            config: Training configuration
            label_weights: Optional class weights for NER loss
        """
        self.model = model
        self.config = config
        self.state = TrainingState()

        # Setup device
        self.device = self._setup_device(config.device)
        self.model = self.model.to(self.device)

        # Loss functions
        self.ner_criterion = nn.CrossEntropyLoss(
            weight=label_weights.to(self.device) if label_weights is not None else None,
            label_smoothing=config.label_smoothing,
            ignore_index=-100,  # Ignore padding
        )
        self.sentiment_criterion = nn.MSELoss(reduction="none")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler (will be set in train())
        self.scheduler = None

        # Mixed precision scaler (CUDA only)
        self.scaler = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Total training steps (set in train()) — used by the dropout curriculum
        self.total_training_steps = 0

        logger.info(f"Trainer initialized on device: {self.device}")

    def _setup_device(self, device_str: str) -> torch.device:
        """Setup compute device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def _get_curriculum_weights(self, epoch: int) -> Tuple[float, float]:
        """
        Get loss weights based on curriculum schedule.

        Early epochs: Focus on NER (simpler task)
        Later epochs: Increase sentiment weight

        Args:
            epoch: Current epoch

        Returns:
            (ner_weight, sentiment_weight) tuple
        """
        if not self.config.use_curriculum:
            return self.config.ner_loss_weight, self.config.sentiment_loss_weight

        progress = epoch / max(self.config.epochs - 1, 1)

        ner_weight = (
            self.config.curriculum_ner_start +
            (self.config.curriculum_ner_end - self.config.curriculum_ner_start) * progress
        )
        sentiment_weight = (
            self.config.curriculum_sentiment_start +
            (self.config.curriculum_sentiment_end - self.config.curriculum_sentiment_start) * progress
        )

        return ner_weight, sentiment_weight

    def _effective_dropout(self) -> float:
        """Current global-attention dropout probability with linear warmup.

        Ramps from 0 to `global_attn_dropout_prob` over the first
        `global_attn_dropout_warmup_frac` of total optimizer steps, then holds.
        Returns 0 if dropout is disabled or before training starts.
        """
        p_target = self.config.global_attn_dropout_prob
        if p_target <= 0:
            return 0.0
        if self.total_training_steps <= 0:
            return 0.0
        warmup_frac = self.config.global_attn_dropout_warmup_frac
        if warmup_frac <= 0:
            return p_target
        progress = self.state.global_step / (warmup_frac * self.total_training_steps)
        return min(progress, 1.0) * p_target

    def _build_global_attn_mask(
        self,
        input_ids: torch.Tensor,
        entity_masks: Optional[torch.Tensor],
        *,
        force_cls_only: bool = False,
        apply_dropout: bool = False,
    ) -> torch.Tensor:
        """Construct the encoder's global_attention_mask.

        Args:
            input_ids: (B, T) - used only for shape/device
            entity_masks: (B, num_entities, T) gold entity-token mask, or None.
            force_cls_only: If True, return CLS-only mask regardless of entity_masks.
                            Used for dual-eval to measure inference-regime metrics.
            apply_dropout: If True (training only), with per-sample probability
                            `_effective_dropout()`, replace that sample's
                            entity-aware mask with CLS-only. Trains the encoder
                            to be robust to the inference regime.

        Returns:
            global_attn: (B, T) {0,1} mask. Position 0 (CLS) is always global.
        """
        device = input_ids.device
        bs = input_ids.shape[0]

        # CLS-only baseline
        global_attn = torch.zeros_like(input_ids)
        global_attn[:, 0] = 1

        if force_cls_only or entity_masks is None:
            return global_attn

        # Entity-aware default (matches the existing model behavior)
        ep = entity_masks.sum(dim=1).clamp(0, 1).long()  # (B, T)
        entity_aware = global_attn | ep

        if not apply_dropout:
            return entity_aware

        p = self._effective_dropout()
        if p <= 0:
            return entity_aware

        # Per-sample dropout: with probability p, replace entity-aware with CLS-only
        drop = (torch.rand(bs, device=device) < p)  # (B,)
        # Broadcast (B,) -> (B, T)
        return torch.where(drop.unsqueeze(1), global_attn, entity_aware)

    def _compute_loss(
        self,
        ner_output,
        sentiment_preds: torch.Tensor,
        ner_labels: torch.Tensor,
        sentiment_targets: torch.Tensor,
        entity_mask_valid: torch.Tensor,
        ner_weight: float,
        sentiment_weight: float,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint loss.

        Args:
            ner_output: Either tensor (logits) or dict (CRF output)
            sentiment_preds: (batch, max_entities)
            ner_labels: (batch, seq_len)
            sentiment_targets: (batch, max_entities)
            entity_mask_valid: (batch, max_entities) - 1 for valid entities
            ner_weight: Weight for NER loss
            sentiment_weight: Weight for sentiment loss
            attention_mask: (batch, seq_len) for CRF

        Returns:
            (total_loss, loss_dict) tuple
        """
        # Handle CRF vs standard NER head
        if isinstance(ner_output, dict):
            # CRF head: loss is already computed
            ner_loss = ner_output.get("loss", torch.tensor(0.0, device=self.device))
            ner_logits = ner_output["logits"]
        else:
            # Standard NER head: compute cross-entropy loss
            ner_logits = ner_output
            batch_size, seq_len, num_labels = ner_logits.shape
            ner_logits_flat = ner_logits.view(-1, num_labels)
            ner_labels_flat = ner_labels.view(-1)
            ner_loss = self.ner_criterion(ner_logits_flat, ner_labels_flat)

        # Sentiment loss: only for valid entities
        sentiment_loss_all = self.sentiment_criterion(sentiment_preds, sentiment_targets)
        # Mask out invalid entities
        sentiment_loss_masked = sentiment_loss_all * entity_mask_valid

        # Average over valid entities only
        num_valid = entity_mask_valid.sum()
        if num_valid > 0:
            sentiment_loss = sentiment_loss_masked.sum() / num_valid
        else:
            sentiment_loss = torch.tensor(0.0, device=self.device)

        # Combine losses
        total_loss = ner_weight * ner_loss + sentiment_weight * sentiment_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "ner_loss": ner_loss.item(),
            "sentiment_loss": sentiment_loss.item() if num_valid > 0 else 0.0,
            "ner_weight": ner_weight,
            "sentiment_weight": sentiment_weight,
        }

        return total_loss, loss_dict

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        ner_weight: float,
        sentiment_weight: float,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch dictionary from data loader
            ner_weight: NER loss weight
            sentiment_weight: Sentiment loss weight

        Returns:
            Loss dictionary
        """
        self.model.train()

        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        ner_labels = batch["ner_labels"].to(self.device)
        entity_masks = batch["entity_masks"].to(self.device)
        sentiment_targets = batch["sentiment_scores"].to(self.device)
        entity_mask_valid = batch["entity_mask_valid"].to(self.device)

        # Build global attention mask with per-sample dropout (training regime)
        global_attn = self._build_global_attn_mask(
            input_ids, entity_masks, apply_dropout=True,
        )

        # Forward pass
        if self.scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                ner_output, sentiment_preds = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_masks=entity_masks,
                    global_attention_mask=global_attn,
                    ner_labels=ner_labels,  # Pass labels for CRF training
                )
                loss, loss_dict = self._compute_loss(
                    ner_output, sentiment_preds,
                    ner_labels, sentiment_targets, entity_mask_valid,
                    ner_weight, sentiment_weight, attention_mask,
                )

            # Backward with scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            ner_output, sentiment_preds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_masks=entity_masks,
                global_attention_mask=global_attn,
                ner_labels=ner_labels,  # Pass labels for CRF training
            )
            loss, loss_dict = self._compute_loss(
                ner_output, sentiment_preds,
                ner_labels, sentiment_targets, entity_mask_valid,
                ner_weight, sentiment_weight, attention_mask,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

        # Expose the dropout that was used this step (for diagnostics/logging)
        loss_dict["global_attn_dropout"] = self._effective_dropout()
        return loss_dict

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        global_attn_regime: str = "entity_aware",
        ner_weight: float = 1.0,
        sentiment_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader
            global_attn_regime: "entity_aware" (model default, training regime) or
                "cls_only" (no entity globals — measures e2e inference regime).
                Used for dual-eval after enabling global-attention dropout.

        Returns:
            Metrics dictionary (keys are unprefixed; caller may suffix them).
        """
        assert global_attn_regime in ("entity_aware", "cls_only"), \
            f"Unknown global_attn_regime: {global_attn_regime}"
        force_cls_only = (global_attn_regime == "cls_only")
        self.model.eval()

        total_ner_loss = 0.0
        total_sentiment_loss = 0.0
        total_samples = 0
        total_entities = 0

        # For NER metrics
        all_ner_preds = []
        all_ner_labels = []

        # For sentiment metrics
        all_sentiment_preds = []
        all_sentiment_targets = []

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            ner_labels = batch["ner_labels"].to(self.device)
            entity_masks = batch["entity_masks"].to(self.device)
            sentiment_targets = batch["sentiment_scores"].to(self.device)
            entity_mask_valid = batch["entity_mask_valid"].to(self.device)

            # Build global attention mask explicitly so we can force the regime.
            # entity_masks is still passed through for sentiment-head pooling.
            global_attn = self._build_global_attn_mask(
                input_ids, entity_masks,
                force_cls_only=force_cls_only, apply_dropout=False,
            )

            # Forward
            ner_output, sentiment_preds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_masks=entity_masks,
                global_attention_mask=global_attn,
            )

            # Get batch size from input
            batch_size = input_ids.shape[0]

            # Handle CRF vs standard NER output
            if isinstance(ner_output, dict):
                ner_logits = ner_output["logits"]
                # For evaluation, we don't pass labels to CRF
                # So compute loss manually
                if hasattr(self.model.ner_head, 'crf'):
                    ner_loss = -self.model.ner_head.crf(
                        ner_logits, ner_labels,
                        mask=attention_mask.bool(), reduction='mean'
                    )
                else:
                    _, seq_len, num_labels = ner_logits.shape
                    ner_loss = self.ner_criterion(
                        ner_logits.view(-1, num_labels),
                        ner_labels.view(-1),
                    )
            else:
                ner_logits = ner_output
                _, seq_len, num_labels = ner_logits.shape
                ner_loss = self.ner_criterion(
                    ner_logits.view(-1, num_labels),
                    ner_labels.view(-1),
                )
            total_ner_loss += ner_loss.item() * batch_size

            # Sentiment loss (only valid entities)
            sentiment_loss_all = self.sentiment_criterion(sentiment_preds, sentiment_targets)
            sentiment_loss_masked = sentiment_loss_all * entity_mask_valid
            num_valid = entity_mask_valid.sum().item()
            if num_valid > 0:
                total_sentiment_loss += sentiment_loss_masked.sum().item()
                total_entities += num_valid

            total_samples += batch_size

            # Collect predictions for metrics
            if isinstance(ner_output, dict) and "predictions" in ner_output:
                # Use CRF Viterbi predictions
                ner_preds = ner_output["predictions"]
            else:
                # Use argmax for standard head
                ner_preds = ner_logits.argmax(dim=-1)  # (batch, seq_len)

            # Mask out padding tokens
            valid_mask = attention_mask.bool()
            for i in range(batch_size):
                valid_tokens = valid_mask[i]
                all_ner_preds.extend(ner_preds[i][valid_tokens].cpu().tolist())
                all_ner_labels.extend(ner_labels[i][valid_tokens].cpu().tolist())

            # Collect sentiment predictions
            for i in range(batch_size):
                for j in range(entity_mask_valid.shape[1]):
                    if entity_mask_valid[i, j] > 0:
                        all_sentiment_preds.append(sentiment_preds[i, j].item())
                        all_sentiment_targets.append(sentiment_targets[i, j].item())

        # Compute metrics
        metrics = {
            "val_ner_loss": total_ner_loss / total_samples,
            "val_sentiment_loss": total_sentiment_loss / max(total_entities, 1),
            "val_total_loss": (
                ner_weight * total_ner_loss / total_samples +
                sentiment_weight * total_sentiment_loss / max(total_entities, 1)
            ),
        }

        # NER F1
        ner_metrics = compute_ner_metrics(all_ner_preds, all_ner_labels)
        metrics.update(ner_metrics)

        # Sentiment metrics
        if all_sentiment_preds:
            sentiment_metrics = compute_sentiment_metrics(
                all_sentiment_preds, all_sentiment_targets
            )
            metrics.update(sentiment_metrics)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training history dictionary
        """
        # Setup scheduler
        num_training_steps = len(train_loader) * self.config.epochs
        self.total_training_steps = num_training_steps  # for dropout curriculum
        num_warmup_steps = min(
            len(train_loader) * self.config.warmup_epochs,
            num_training_steps // 2  # Warmup can't be more than half
        )
        num_main_steps = max(num_training_steps - num_warmup_steps, 1)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(num_warmup_steps, 1),
        )

        if self.config.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_main_steps,
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_main_steps,
            )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[max(num_warmup_steps, 1)],
        )

        # Training history
        history = {
            "train_loss": [],
            "train_ner_loss": [],
            "train_sentiment_loss": [],
            "val_loss": [],
            "val_ner_f1": [],          # entity-aware regime (training regime)
            "val_ner_f1_cls_only": [], # CLS-only regime (e2e inference regime)
            "val_sentiment_mse": [],
            "val_sentiment_mse_cls_only": [],
            "learning_rate": [],
            "global_attn_dropout": [],
        }
        dropout_enabled = self.config.global_attn_dropout_prob > 0
        if dropout_enabled:
            logger.info(
                f"  Global-attn dropout: p_target={self.config.global_attn_dropout_prob}, "
                f"warmup_frac={self.config.global_attn_dropout_warmup_frac}"
            )

        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Device: {self.device}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            ner_weight, sentiment_weight = self._get_curriculum_weights(epoch)

            epoch_losses = []
            epoch_ner_losses = []
            epoch_sentiment_losses = []

            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            logger.info(f"  NER weight: {ner_weight:.3f}, Sentiment weight: {sentiment_weight:.3f}")

            for step, batch in enumerate(train_loader):
                loss_dict = self.train_step(batch, ner_weight, sentiment_weight)
                self.scheduler.step()

                epoch_losses.append(loss_dict["total_loss"])
                epoch_ner_losses.append(loss_dict["ner_loss"])
                epoch_sentiment_losses.append(loss_dict["sentiment_loss"])

                self.state.global_step += 1

                # Logging
                if (step + 1) % self.config.log_every_n_steps == 0:
                    avg_loss = sum(epoch_losses[-self.config.log_every_n_steps:]) / self.config.log_every_n_steps
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Step {step + 1}/{len(train_loader)}: "
                        f"loss={avg_loss:.4f}, "
                        f"ner={loss_dict['ner_loss']:.4f}, "
                        f"sent={loss_dict['sentiment_loss']:.4f}, "
                        f"lr={lr:.2e}"
                    )

            # Epoch summary
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_ner_loss = sum(epoch_ner_losses) / len(epoch_ner_losses)
            avg_sentiment_loss = sum(epoch_sentiment_losses) / len(epoch_sentiment_losses)

            history["train_loss"].append(avg_train_loss)
            history["train_ner_loss"].append(avg_ner_loss)
            history["train_sentiment_loss"].append(avg_sentiment_loss)
            history["learning_rate"].append(self.scheduler.get_last_lr()[0])
            history["global_attn_dropout"].append(self._effective_dropout())

            logger.info(f"  Epoch {epoch + 1} complete: train_loss={avg_train_loss:.4f}")

            # Validation — dual eval when dropout is enabled so we track both regimes
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, global_attn_regime="entity_aware", ner_weight=ner_weight, sentiment_weight=sentiment_weight)
                history["val_loss"].append(val_metrics["val_total_loss"])
                history["val_ner_f1"].append(val_metrics.get("ner_f1", 0.0))
                history["val_sentiment_mse"].append(val_metrics.get("sentiment_mse", 0.0))

                if dropout_enabled and self.config.eval_cls_only_regime:
                    cls_metrics = self.evaluate(val_loader, global_attn_regime="cls_only", ner_weight=ner_weight, sentiment_weight=sentiment_weight)
                    history["val_ner_f1_cls_only"].append(cls_metrics.get("ner_f1", 0.0))
                    history["val_sentiment_mse_cls_only"].append(cls_metrics.get("sentiment_mse", 0.0))
                    logger.info(
                        f"  Validation [entity-aware]: loss={val_metrics['val_total_loss']:.4f}, "
                        f"ner_f1={val_metrics.get('ner_f1', 0):.4f}, "
                        f"sentiment_mse={val_metrics.get('sentiment_mse', 0):.4f}"
                    )
                    logger.info(
                        f"  Validation [CLS-only]    : "
                        f"ner_f1={cls_metrics.get('ner_f1', 0):.4f}, "
                        f"sentiment_mse={cls_metrics.get('sentiment_mse', 0):.4f}  "
                        f"(dropout p={self._effective_dropout():.3f})"
                    )
                    # Save best inference-regime checkpoint independently
                    cls_f1 = cls_metrics.get("ner_f1", 0.0)
                    if cls_f1 > self.state.best_cls_only_f1:
                        self.state.best_cls_only_f1 = cls_f1
                        self.save_checkpoint("best_cls_only.pt")
                        logger.info("  Saved best CLS-only model!")
                else:
                    history["val_ner_f1_cls_only"].append(0.0)
                    history["val_sentiment_mse_cls_only"].append(0.0)
                    logger.info(
                        f"  Validation: loss={val_metrics['val_total_loss']:.4f}, "
                        f"ner_f1={val_metrics.get('ner_f1', 0):.4f}, "
                        f"sentiment_mse={val_metrics.get('sentiment_mse', 0):.4f}"
                    )

                # Save best model (training regime)
                if val_metrics["val_total_loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics["val_total_loss"]
                    if self.config.save_best_only:
                        self.save_checkpoint("best_model.pt")
                        logger.info("  Saved best model!")

            # Regular checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                if not self.config.save_best_only:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        elapsed = time.time() - start_time
        logger.info(f"\nTraining complete! Total time: {elapsed / 60:.1f} minutes")

        # Save final model
        self.save_checkpoint("final_model.pt")

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "state": self.state,
            "config": self.config,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str, model_weights_only: bool = False):
        """Load model checkpoint.

        Args:
            filename: Path to checkpoint file
            model_weights_only: If True, only load model weights (skip optimizer/scheduler/state).
                              Useful for transfer learning where optimizer state may cause issues.
        """
        # Handle both relative and absolute paths
        path = Path(filename)
        if not path.is_absolute():
            path = self.checkpoint_dir / filename
        # Use weights_only=False for checkpoints containing TrainingState
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if not model_weights_only:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint["scheduler_state_dict"] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.state = checkpoint["state"]
            logger.info(f"Checkpoint loaded from {path}")
        else:
            logger.info(f"Model weights loaded from {path} (optimizer/scheduler reset)")


def compute_ner_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute NER metrics (precision, recall, F1).

    Args:
        preds: Predicted label IDs
        labels: Ground truth label IDs

    Returns:
        Metrics dictionary
    """
    # Count true positives, false positives, false negatives per entity type
    entity_types = ["COMPANY", "TICKER", "PERSON", "ORG", "MONEY", "PERCENT", "DATE"]

    tp = {t: 0 for t in entity_types}
    fp = {t: 0 for t in entity_types}
    fn = {t: 0 for t in entity_types}

    for pred, label in zip(preds, labels):
        pred_label = ID_TO_LABEL.get(pred, "O")
        true_label = ID_TO_LABEL.get(label, "O")

        # Extract entity type (strip B-/I- prefix)
        pred_type = pred_label[2:] if pred_label != "O" else None
        true_type = true_label[2:] if true_label != "O" else None

        if pred_type and true_type and pred_type == true_type:
            tp[pred_type] += 1
        elif pred_type:
            fp[pred_type] += 1
        if true_type and pred_type != true_type:
            fn[true_type] += 1

    # Compute overall metrics
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "ner_precision": precision,
        "ner_recall": recall,
        "ner_f1": f1,
    }

    # Per-type F1 for diagnostic visibility (especially during CLS-only eval)
    for entity_type in entity_types:
        type_tp = tp[entity_type]
        type_fp = fp[entity_type]
        type_fn = fn[entity_type]
        type_p = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
        type_r = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
        type_f1 = 2 * type_p * type_r / (type_p + type_r) if (type_p + type_r) > 0 else 0.0
        metrics[f"ner_f1_{entity_type}"] = type_f1

    return metrics


def compute_sentiment_metrics(
    preds: List[float],
    targets: List[float],
) -> Dict[str, float]:
    """
    Compute sentiment metrics.

    Args:
        preds: Predicted sentiment scores
        targets: Ground truth sentiment scores

    Returns:
        Metrics dictionary
    """
    if not preds:
        return {"sentiment_mse": 0.0, "sentiment_mae": 0.0, "sentiment_corr": 0.0,
                "sentiment_ccc": 0.0, "sentiment_std_ratio": 0.0, "sentiment_sign_flip": 0.0,
                "sentiment_neutral_false_polar": 0.0, "sentiment_over_neutral": 0.0,
                "sentiment_very_neg_f1": 0.0, "sentiment_very_pos_f1": 0.0}

    preds_t = torch.tensor(preds)
    targets_t = torch.tensor(targets)

    # MSE
    mse = ((preds_t - targets_t) ** 2).mean().item()

    # MAE
    mae = (preds_t - targets_t).abs().mean().item()

    # Pearson correlation
    if len(preds) > 1:
        preds_mean = preds_t.mean()
        targets_mean = targets_t.mean()
        # population std (unbiased=False) to stay consistent with the population
        # covariance below — mixing unbiased std with population cov scales r by
        # (n-1)/n, which is visibly wrong on small per-type/per-arm slices.
        preds_std = preds_t.std(unbiased=False)
        targets_std = targets_t.std(unbiased=False)

        if preds_std > 0 and targets_std > 0:
            corr = ((preds_t - preds_mean) * (targets_t - targets_mean)).mean()
            corr = corr / (preds_std * targets_std)
            corr = corr.item()
        else:
            corr = 0.0
    else:
        corr = 0.0

    # --- extra metrics for the second-phase retrain (diagnose under-polarization) ---
    p = preds_t.float()
    t = targets_t.float()
    p_std = p.std(unbiased=False).item()
    t_std = t.std(unbiased=False).item()
    # CCC (concordance) — scale-sensitive, unlike Pearson
    if len(preds) > 1 and (p_std > 0 or t_std > 0):
        cov = ((p - p.mean()) * (t - t.mean())).mean()
        ccc = (2 * cov / (p.var(unbiased=False) + t.var(unbiased=False)
                          + (p.mean() - t.mean()) ** 2 + 1e-8)).item()
    else:
        ccc = 0.0
    std_ratio = (p_std / t_std) if t_std > 0 else 0.0
    # sign-flip rate among both-signed entities
    both = (t.abs() >= 0.1) & (p.abs() >= 0.1)
    sign_flip = ((torch.sign(t[both]) != torch.sign(p[both])).float().mean().item()
                 if both.any() else 0.0)
    # neutral false-polarization: gold incidental but model strong
    neutral = t.abs() < 0.1
    neutral_fp = ((p[neutral].abs() > 0.3).float().mean().item() if neutral.any() else 0.0)
    # over-neutrality: gold strong but model flat
    strong = t.abs() >= 0.4
    over_neutral = ((p[strong].abs() < 0.1).float().mean().item() if strong.any() else 0.0)

    def _bucket(x):  # -> tensor of bucket idx 0..4
        b = torch.full_like(x, 2)  # neutral
        b[x < -0.6] = 0; b[(x >= -0.6) & (x < -0.2)] = 1
        b[(x >= 0.2) & (x < 0.6)] = 3; b[x >= 0.6] = 4
        return b
    gb, pb = _bucket(t), _bucket(p)
    extreme = {}
    for name, idx in (("very_neg", 0), ("very_pos", 4)):
        tp = float(((gb == idx) & (pb == idx)).sum())
        fp = float(((gb != idx) & (pb == idx)).sum())
        fn = float(((gb == idx) & (pb != idx)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        extreme[f"{name}_f1"] = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0

    return {
        "sentiment_mse": mse,
        "sentiment_mae": mae,
        "sentiment_corr": corr,
        "sentiment_ccc": ccc,
        "sentiment_std_ratio": std_ratio,
        "sentiment_sign_flip": sign_flip,
        "sentiment_neutral_false_polar": neutral_fp,
        "sentiment_over_neutral": over_neutral,
        "sentiment_very_neg_f1": extreme["very_neg_f1"],
        "sentiment_very_pos_f1": extreme["very_pos_f1"],
    }

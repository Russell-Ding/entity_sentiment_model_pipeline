#!/usr/bin/env python3
"""Stage 3: Sentiment-Only Training with V2 Cross-Attention Head.

Loads a Stage 1 or Stage 2 checkpoint (encoder + NER), discards the
sentiment head weights, and trains a fresh SentimentHead (cross-attention)
from Xavier init with frozen encoder + NER.

Usage (on Colab):
    python scripts/training/train_stage3.py

    # Custom settings
    python scripts/training/train_stage3.py --batch-size 52 --epochs 10 --lr 5e-4

    # Quick test
    python scripts/training/train_stage3.py --max-train-samples 200 --epochs 2
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# transformers torch.load safety workaround
# ---------------------------------------------------------------------------
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import FinancialEntitySentimentModel
from models.sentiment_head import SentimentHead
from training.preprocessing import DataPreprocessor, LABEL_TO_ID
from training.dataset import EntitySentimentDataset, collate_fn, create_data_loaders
from training.trainer import compute_ner_metrics, compute_sentiment_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
# Ensure stdout handler flushes immediately (Colab buffers subprocess output)
for h in logging.getLogger().handlers:
    if hasattr(h, 'stream'):
        h.stream = sys.stdout
logging.getLogger().handlers[0].flush = lambda: sys.stdout.flush()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, train_loader, optimizer, scheduler, scaler, device,
    gradient_clip=1.0,
):
    """Train for one epoch. Sentiment loss computed in float32."""
    model.train()
    total_loss = 0.0
    total_sentiment_loss = 0.0
    num_batches = 0

    ner_criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ner_labels = batch["ner_labels"].to(device)
        entity_masks = batch["entity_masks"].to(device)
        sentiment_targets = batch["sentiment_scores"].to(device)
        entity_mask_valid = batch["entity_mask_valid"].to(device)

        # Forward in float16
        with torch.amp.autocast(device_type="cuda"):
            ner_output, sentiment_preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_masks=entity_masks,
                ner_labels=ner_labels,
            )

        # Sentiment loss in float32 — Pearson correlation is numerically
        # unstable in float16 and causes GradScaler to skip all updates
        valid_mask = entity_mask_valid.bool()
        valid_preds = sentiment_preds[valid_mask].float()
        valid_targets = sentiment_targets[valid_mask].float()
        if valid_preds.numel() > 0:
            sentiment_loss = model.sentiment_head.compute_loss(valid_preds, valid_targets)
        else:
            sentiment_loss = torch.tensor(0.0, device=device)

        loss = sentiment_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_sentiment_loss += sentiment_loss.item() if valid_preds.numel() > 0 else 0
        num_batches += 1

        if (step + 1) % 50 == 0:
            avg = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            logger.info(f"  Step {step+1}/{len(train_loader)}  loss={avg:.4f}  lr={lr:.2e}")

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_sentiment_loss": total_sentiment_loss / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model. Returns dict of metrics."""
    model.eval()

    all_ner_preds = []
    all_ner_labels = []
    all_sentiment_preds = []
    all_sentiment_targets = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ner_labels = batch["ner_labels"].to(device)
        entity_masks = batch["entity_masks"].to(device)
        sentiment_targets = batch["sentiment_scores"].to(device)
        entity_mask_valid = batch["entity_mask_valid"].to(device)

        with torch.amp.autocast(device_type="cuda"):
            ner_output, sentiment_preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_masks=entity_masks,
            )

        # float32 for metrics
        sentiment_preds = sentiment_preds.float()
        batch_size = input_ids.shape[0]

        # NER
        if isinstance(ner_output, dict) and "predictions" in ner_output:
            ner_pred_ids = ner_output["predictions"]
        else:
            ner_logits = ner_output if not isinstance(ner_output, dict) else ner_output["logits"]
            ner_pred_ids = ner_logits.argmax(dim=-1)

        valid_tok = attention_mask.bool().cpu()
        for i in range(batch_size):
            m = valid_tok[i]
            all_ner_preds.extend(ner_pred_ids[i][m].cpu().tolist())
            all_ner_labels.extend(ner_labels[i][m].cpu().tolist())

        # Sentiment
        for i in range(batch_size):
            for j in range(entity_mask_valid.shape[1]):
                if entity_mask_valid[i, j] > 0:
                    all_sentiment_preds.append(sentiment_preds[i, j].cpu().item())
                    all_sentiment_targets.append(sentiment_targets[i, j].cpu().item())

    metrics = compute_ner_metrics(all_ner_preds, all_ner_labels)
    if all_sentiment_preds:
        metrics.update(compute_sentiment_metrics(all_sentiment_preds, all_sentiment_targets))
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, history, local_path, drive_path=None):
    """Save to local first, then copy to Drive."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_metrics": val_metrics,
        "epoch": epoch,
        "history": history,
    }
    torch.save(ckpt, local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    logger.info(f"  Saved locally: {local_path} ({size_mb:.0f} MB)")

    if drive_path:
        try:
            shutil.copy2(local_path, drive_path)
            logger.info(f"  Copied to Drive: {drive_path}")
        except Exception as e:
            logger.warning(f"  Drive copy failed: {e} (local backup is safe)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 3: Sentiment-only training")
    parser.add_argument("--project-path", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--encoder-name", type=str, default="allenai/longformer-large-4096")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0=all")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0=all")
    parser.add_argument("--local-ckpt-dir", type=str, default="/content",
                        help="Local dir for checkpoints (fast, reliable)")
    parser.add_argument("--source-checkpoint", type=str, default=None,
                        help="Path to the checkpoint to load encoder + NER weights from. "
                             "Default: project/checkpoints/stage2_joint_large/best_model.pt. "
                             "For the dropout-trained pipeline, pass "
                             "project/checkpoints/stage1_ner_large_v2/best_cls_only.pt.")
    parser.add_argument("--drive-ckpt-dir", type=str, default=None,
                        help="Drive directory for final checkpoint outputs. "
                             "Default: project/checkpoints/stage3_sentiment_large.")
    args = parser.parse_args()

    project = Path(args.project_path)
    train_file = project / "data/labeled/final/train.jsonl"
    val_file = project / "data/labeled/final/val.jsonl"
    holdout_file = project / "data/labeled/final/holdout.jsonl"
    stage2_ckpt = (Path(args.source_checkpoint) if args.source_checkpoint
                   else project / "checkpoints/stage2_joint_large/best_model.pt")
    drive_ckpt_dir = (Path(args.drive_ckpt_dir) if args.drive_ckpt_dir
                      else project / "checkpoints/stage3_sentiment_large")
    log_dir = project / "logs"
    local_ckpt_dir = Path(args.local_ckpt_dir)

    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    local_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Log to LOCAL disk first (Drive FUSE drops log entries under heavy I/O)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_log = local_ckpt_dir / f"stage3_sentiment_{timestamp}.log"
    drive_log = log_dir / f"stage3_sentiment_{timestamp}.log"
    file_handler = logging.FileHandler(str(local_log))
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("STAGE 3: SENTIMENT-ONLY TRAINING (V2 Cross-Attention)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Config: batch={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    logger.info(f"Local checkpoint dir: {local_ckpt_dir}")
    logger.info(f"Drive checkpoint dir: {drive_ckpt_dir}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data...")
    preprocessor = DataPreprocessor(
        model_name=args.encoder_name,
        max_length=args.max_length,
        use_expanded_sentiment=True,
    )

    train_loader, val_loader = create_data_loaders(
        train_files=str(train_file),
        val_files=str(val_file),
        preprocessor=preprocessor,
        batch_size=args.batch_size,
    )
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # ------------------------------------------------------------------
    # 2. Build model, load source weights, discard sentiment head weights
    # ------------------------------------------------------------------
    logger.info("Building model...")
    model = FinancialEntitySentimentModel(
        encoder_name=args.encoder_name,
        hidden_size=args.hidden_size,
        use_crf_ner=True,
        max_length=args.max_length,
    )
    model = model.to(device)

    logger.info(f"Loading Stage 2 checkpoint: {stage2_ckpt}")
    ckpt = torch.load(str(stage2_ckpt), map_location=device, weights_only=False)
    stage2_state = ckpt["model_state_dict"]

    # Discard sentiment head weights from the source — Stage 3 retrains the
    # sentiment head from Xavier init for a clean fit on the frozen backbone.
    sent_keys = [k for k in stage2_state if k.startswith("sentiment_head.")]
    stage2_state = {k: v for k, v in stage2_state.items() if not k.startswith("sentiment_head.")}
    model.load_state_dict(stage2_state, strict=False)
    logger.info(f"Loaded encoder + NER. Discarded {len(sent_keys)} sentiment-head keys (fresh init).")

    stage2_metrics = ckpt.get("val_metrics", {})
    del ckpt
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Freeze encoder + NER
    # ------------------------------------------------------------------
    for name, param in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("ner_head."):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Verify sentiment head params require grad
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  Trainable: {name} ({param.numel():,})")

    # ------------------------------------------------------------------
    # 4. Optimizer, scheduler, scaler
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda")

    logger.info(f"Optimizer steps: {total_steps}")

    # ------------------------------------------------------------------
    # 5. Baseline evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating baseline (fresh V2 head)...")
    baseline = evaluate(model, val_loader, device)
    logger.info(f"Baseline: NER-F1={baseline.get('ner_f1',0):.4f}, "
                f"Sent-MSE={baseline.get('sentiment_mse',0):.4f}, "
                f"Sent-Corr={baseline.get('sentiment_corr',0):.4f}")

    # Quick sanity check: verify gradients flow (uses separate scaler to avoid
    # corrupting the training scaler's internal state)
    logger.info("Gradient sanity check...")
    model.train()
    sample_batch = next(iter(train_loader))
    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)
    entity_masks = sample_batch["entity_masks"].to(device)
    sentiment_targets = sample_batch["sentiment_scores"].to(device)
    entity_mask_valid = sample_batch["entity_mask_valid"].to(device)

    with torch.amp.autocast(device_type="cuda"):
        _, preds = model(input_ids=input_ids, attention_mask=attention_mask,
                         entity_masks=entity_masks)
    valid_p = preds[entity_mask_valid.bool()].float()
    valid_t = sentiment_targets[entity_mask_valid.bool()].float()
    if valid_p.numel() > 0:
        test_loss = model.sentiment_head.compute_loss(valid_p, valid_t)
        test_loss.backward()  # plain backward, no scaler involvement
        grad_ok = any(p.grad is not None and p.grad.abs().max() > 0
                      for p in model.sentiment_head.parameters())
        logger.info(f"  Test loss: {test_loss.item():.4f}, Gradients flowing: {grad_ok}")
        if not grad_ok:
            logger.error("FATAL: No gradients flowing to sentiment head! Aborting.")
            sys.exit(1)
    optimizer.zero_grad()

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    best_corr = -1.0
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(args.epochs):
        t0 = time.time()
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            gradient_clip=args.gradient_clip,
        )

        val_metrics = evaluate(model, val_loader, device)

        epoch_time = time.time() - t0
        gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Record
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_sentiment_loss"].append(train_metrics["train_sentiment_loss"])
        history["val_sentiment_mse"].append(val_metrics.get("sentiment_mse", 0))
        history["val_sentiment_mae"].append(val_metrics.get("sentiment_mae", 0))
        history["val_sentiment_corr"].append(val_metrics.get("sentiment_corr", 0))
        history["val_ner_f1"].append(val_metrics.get("ner_f1", 0))

        corr = val_metrics.get("sentiment_corr", 0)
        logger.info(
            f"  Results: loss={train_metrics['train_loss']:.4f}, "
            f"mse={val_metrics.get('sentiment_mse',0):.4f}, "
            f"corr={corr:.4f}, "
            f"ner_f1={val_metrics.get('ner_f1',0):.4f}, "
            f"time={epoch_time/60:.1f}min, gpu={gpu_gb:.1f}GB"
        )

        # Best model
        if corr > best_corr:
            best_corr = corr
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, dict(history),
                local_path=str(local_ckpt_dir / "best_model.pt"),
                drive_path=str(drive_ckpt_dir / "best_model.pt"),
            )
            logger.info(f"  New best! corr={best_corr:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{args.patience})")

        # Periodic checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, dict(history),
            local_path=str(local_ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"),
            drive_path=str(drive_ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"),
        )

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE! Best corr: {best_corr:.4f}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 7. Holdout evaluation
    # ------------------------------------------------------------------
    if holdout_file.exists():
        logger.info("Running holdout evaluation...")

        # Load best model
        best_path = local_ckpt_dir / "best_model.pt"
        if best_path.exists():
            best_ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info(f"Loaded best model (epoch {best_ckpt.get('epoch', -1)+1})")
            del best_ckpt
            torch.cuda.empty_cache()

        holdout_samples = preprocessor.process_file(str(holdout_file))
        holdout_dataset = EntitySentimentDataset(holdout_samples, max_entities_per_sample=10)
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0,
        )

        holdout_metrics = evaluate(model, holdout_loader, device)
        logger.info("HOLDOUT RESULTS:")
        logger.info(f"  NER F1:          {holdout_metrics.get('ner_f1', 0):.4f}")
        logger.info(f"  Sentiment MSE:   {holdout_metrics.get('sentiment_mse', 0):.4f}")
        logger.info(f"  Sentiment MAE:   {holdout_metrics.get('sentiment_mae', 0):.4f}")
        logger.info(f"  Sentiment Corr:  {holdout_metrics.get('sentiment_corr', 0):.4f}")

    # ------------------------------------------------------------------
    # 8. Final Drive sync
    # ------------------------------------------------------------------
    logger.info("Syncing all local checkpoints and log to Drive...")
    import glob
    for src_path in sorted(glob.glob(str(local_ckpt_dir / "*.pt"))):
        fname = os.path.basename(src_path)
        dst = str(drive_ckpt_dir / fname)
        try:
            shutil.copy2(src_path, dst)
            logger.info(f"  {fname} -> Drive OK")
        except Exception as e:
            logger.warning(f"  {fname} -> Drive FAILED: {e}")

    # Copy log file to Drive
    try:
        shutil.copy2(str(local_log), str(drive_log))
        logger.info(f"  Log copied to Drive: {drive_log}")
    except Exception as e:
        logger.warning(f"  Log copy failed: {e}")

    try:
        from google.colab import drive
        drive.flush_and_unmount()
        logger.info("Drive flushed.")
        drive.mount("/content/drive")
        logger.info("Drive remounted.")
        # Give Drive FUSE a moment to reconnect before listing
        time.sleep(5)
    except Exception as e:
        logger.warning(f"Drive flush/remount issue: {e}")

    # Verify checkpoints on Drive (best-effort — don't crash if Drive not ready)
    logger.info("Final checkpoint verification:")
    try:
        if drive_ckpt_dir.exists():
            for f in sorted(os.listdir(str(drive_ckpt_dir))):
                if f.endswith(".pt"):
                    size = os.path.getsize(str(drive_ckpt_dir / f)) / 1e6
                    logger.info(f"  {f:35s} {size:8.1f} MB")
        else:
            logger.warning(f"  Drive directory not accessible yet: {drive_ckpt_dir}")
            logger.info("  Local checkpoints are safe at /content/")
    except Exception as e:
        logger.warning(f"  Verification skipped: {e}")

    logger.info("Done. Terminating runtime.")

    # Terminate Colab runtime to stop billing
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception:
        pass  # Not on Colab, ignore


if __name__ == "__main__":
    main()

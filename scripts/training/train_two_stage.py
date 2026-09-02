#!/usr/bin/env python3
"""
Two-Stage Training: NER First, then Joint NER+Sentiment

Stage 1: Train NER only (sentiment_loss_weight=0)
  - Focus on entity boundary detection
  - Uses full 32K training dataset
  - Saves best NER model

Stage 2: Joint fine-tuning (curriculum learning)
  - Loads Stage 1 checkpoint
  - Gradually increases sentiment weight
  - Final model for production

Usage:
    python train_two_stage.py --stage 1  # Run Stage 1 only
    python train_two_stage.py --stage 2  # Run Stage 2 (requires Stage 1 complete)
    python train_two_stage.py --stage both  # Run both stages sequentially
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import FinancialEntitySentimentModel
from training import DataPreprocessor, create_data_loaders
from training.trainer import JointTrainer, TrainingConfig

# Setup logging
Path(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / f"two_stage_training_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage NER + Sentiment Training")

    # Stage selection
    parser.add_argument("--stage", type=str, default="both",
                       choices=["1", "2", "both"],
                       help="Which stage to run: 1 (NER only), 2 (joint), or both")

    # Data paths (Sonnet-relabeled final dataset)
    parser.add_argument("--train_file", type=str,
                       default=str(PROJECT_ROOT / "data/labeled/final/train.jsonl"))
    parser.add_argument("--val_file", type=str,
                       default=str(PROJECT_ROOT / "data/labeled/final/val.jsonl"))

    # Model
    parser.add_argument("--encoder_name", type=str, default="allenai/longformer-large-4096")
    parser.add_argument("--hidden_size", type=int, default=1024,
                       help="Encoder hidden size (base=768, large=1024)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_crf", action=argparse.BooleanOptionalAction, default=True,
                       help="Use CRF for NER head (slower but may improve accuracy)")

    # Training hyperparameters
    parser.add_argument("--stage1_epochs", type=int, default=5,
                       help="Epochs for Stage 1 (NER only)")
    parser.add_argument("--stage2_epochs", type=int, default=5,
                       help="Epochs for Stage 2 (joint)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--stage2_lr", type=float, default=1e-5,
                       help="Lower LR for Stage 2 fine-tuning")

    # Global-attention dropout (NEW — closes train/inference regime gap)
    parser.add_argument("--global_attn_dropout_prob", type=float, default=0.3,
                       help="Per-sample probability of replacing entity-aware "
                            "global attention with CLS-only during training. "
                            "0 = disabled (legacy behavior). Recommended: 0.3.")
    parser.add_argument("--stage1_dropout_warmup_frac", type=float, default=0.3,
                       help="Stage 1 dropout warmup over first fraction of "
                            "Stage-1 optimizer steps (0..1). Default 0.3 ramps "
                            "from 0 to target over first 30%% of training.")
    parser.add_argument("--stage2_dropout_warmup_frac", type=float, default=0.0,
                       help="Stage 2 dropout warmup. Default 0 = use full target "
                            "from start (Stage 1 already established the regime).")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True,
                       help="Mixed-precision training (CUDA only).")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False,
                       help="Enable gradient checkpointing on the encoder. Trades ~30%% "
                            "extra compute for ~3x activation-memory savings. Recommended "
                            "for Longformer-Large on <80 GB GPUs.")

    # Checkpoints (default to large variants to avoid clobbering archived runs)
    parser.add_argument("--stage1_checkpoint_dir", type=str,
                       default=str(PROJECT_ROOT / "checkpoints/stage1_ner_large_v2"))
    parser.add_argument("--stage2_checkpoint_dir", type=str,
                       default=str(PROJECT_ROOT / "checkpoints/stage2_joint_large_v2"))
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                       help="Path to Stage 1 checkpoint for Stage 2 (auto-detected if not set)")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"])

    return parser.parse_args()


def run_stage1(args, train_loader, val_loader):
    """
    Stage 1: Train NER only.

    - sentiment_loss_weight = 0
    - Focus on entity boundary detection
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: NER-ONLY TRAINING")
    logger.info("=" * 60)

    # Create model
    logger.info("Creating model...")
    model = FinancialEntitySentimentModel(
        encoder_name=args.encoder_name,
        hidden_size=args.hidden_size,
        use_crf_ner=args.use_crf,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Stage 1 config: NER only (no sentiment loss)
    config = TrainingConfig(
        epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,

        # NER only: disable sentiment
        ner_loss_weight=1.0,
        sentiment_loss_weight=0.0,

        # No curriculum for Stage 1
        use_curriculum=False,

        # Global-attention dropout (closes train/inference regime gap)
        global_attn_dropout_prob=args.global_attn_dropout_prob,
        global_attn_dropout_warmup_frac=args.stage1_dropout_warmup_frac,

        use_amp=args.use_amp,
        checkpoint_dir=args.stage1_checkpoint_dir,
        device=args.device,
    )

    logger.info(f"Stage 1 Config:")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  NER weight: {config.ner_loss_weight}")
    logger.info(f"  Sentiment weight: {config.sentiment_loss_weight} (disabled)")
    logger.info(f"  Global-attn dropout: p={config.global_attn_dropout_prob}, "
                f"warmup_frac={config.global_attn_dropout_warmup_frac}")

    # Create trainer
    trainer = JointTrainer(model=model, config=config)

    # Train
    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - start_time

    logger.info(f"\nStage 1 complete in {elapsed/60:.1f} minutes")
    logger.info(f"Best NER F1 (entity-aware): {max(history.get('val_ner_f1', [0])):.4f}")
    if history.get("val_ner_f1_cls_only"):
        cls_f1s = history["val_ner_f1_cls_only"]
        logger.info(f"Best NER F1 (CLS-only)    : {max(cls_f1s):.4f}")
        logger.info(f"  Final CLS-only F1       : {cls_f1s[-1]:.4f} "
                    f"(this is the e2e inference regime — gating value for retrain decisions)")

    return trainer, history


def run_stage2(args, train_loader, val_loader, stage1_checkpoint=None):
    """
    Stage 2: Joint NER + Sentiment fine-tuning.

    - Loads Stage 1 checkpoint
    - Uses curriculum learning
    - Gradually increases sentiment weight
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: JOINT NER + SENTIMENT FINE-TUNING")
    logger.info("=" * 60)

    # Find Stage 1 checkpoint
    if stage1_checkpoint is None:
        stage1_dir = Path(args.stage1_checkpoint_dir)
        best_model = stage1_dir / "best_model.pt"
        if best_model.exists():
            stage1_checkpoint = str(best_model)
        else:
            raise ValueError(f"No Stage 1 checkpoint found at {best_model}. Run Stage 1 first.")

    logger.info(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")

    # Create model
    model = FinancialEntitySentimentModel(
        encoder_name=args.encoder_name,
        hidden_size=args.hidden_size,
        use_crf_ner=args.use_crf,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Stage 2 config: Joint with curriculum
    config = TrainingConfig(
        epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        learning_rate=args.stage2_lr,  # Lower LR for fine-tuning

        # Initial weights (will be adjusted by curriculum)
        ner_loss_weight=1.0,
        sentiment_loss_weight=0.5,

        # Curriculum: start with NER focus, increase sentiment
        use_curriculum=True,
        curriculum_ner_start=1.0,
        curriculum_ner_end=0.5,
        curriculum_sentiment_start=0.3,
        curriculum_sentiment_end=1.0,

        # Global-attention dropout (Stage 2 keeps same target, no warmup by default —
        # Stage 1 already established the regime)
        global_attn_dropout_prob=args.global_attn_dropout_prob,
        global_attn_dropout_warmup_frac=args.stage2_dropout_warmup_frac,

        use_amp=args.use_amp,
        checkpoint_dir=args.stage2_checkpoint_dir,
        device=args.device,
    )

    logger.info(f"Stage 2 Config:")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Curriculum: NER {config.curriculum_ner_start}->{config.curriculum_ner_end}")
    logger.info(f"  Curriculum: Sentiment {config.curriculum_sentiment_start}->{config.curriculum_sentiment_end}")
    logger.info(f"  Global-attn dropout: p={config.global_attn_dropout_prob}, "
                f"warmup_frac={config.global_attn_dropout_warmup_frac}")

    # Create trainer and load Stage 1 weights
    trainer = JointTrainer(model=model, config=config)
    trainer.load_checkpoint(stage1_checkpoint, model_weights_only=True)

    # Train
    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - start_time

    logger.info(f"\nStage 2 complete in {elapsed/60:.1f} minutes")
    logger.info(f"Best NER F1 (entity-aware): {max(history.get('val_ner_f1', [0])):.4f}")
    if history.get("val_ner_f1_cls_only"):
        logger.info(f"Best NER F1 (CLS-only)    : {max(history.get('val_ner_f1_cls_only', [0])):.4f}")
    logger.info(f"Best Sentiment MSE: {min(history.get('val_sentiment_mse', [999])):.4f}")

    return trainer, history


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("TWO-STAGE TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Val file: {args.val_file}")

    # Create checkpoint directories
    Path(args.stage1_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.stage2_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Run stages
    if args.stage in ["1", "both"]:
        logger.info("\nLoading Stage 1 data (NER mentions only for global attention)...")
        preprocessor_stage1 = DataPreprocessor(
            model_name=args.encoder_name,
            max_length=args.max_length,
            use_expanded_sentiment=False,
        )
        train_loader_s1, val_loader_s1 = create_data_loaders(
            train_files=args.train_file,
            val_files=args.val_file,
            preprocessor=preprocessor_stage1,
            batch_size=args.batch_size,
        )
        logger.info(f"Train batches: {len(train_loader_s1)}")
        logger.info(f"Val batches: {len(val_loader_s1)}")
        stage1_trainer, stage1_history = run_stage1(args, train_loader_s1, val_loader_s1)

    if args.stage in ["2", "both"]:
        logger.info("\nLoading Stage 2 data (includes sentiment-expanded mentions)...")
        preprocessor_stage2 = DataPreprocessor(
            model_name=args.encoder_name,
            max_length=args.max_length,
            use_expanded_sentiment=True,
        )
        train_loader_s2, val_loader_s2 = create_data_loaders(
            train_files=args.train_file,
            val_files=args.val_file,
            preprocessor=preprocessor_stage2,
            batch_size=args.batch_size,
        )
        logger.info(f"Train batches: {len(train_loader_s2)}")
        logger.info(f"Val batches: {len(val_loader_s2)}")

        stage1_ckpt = args.stage1_checkpoint
        if args.stage == "both":
            # Use the checkpoint we just created
            stage1_ckpt = str(Path(args.stage1_checkpoint_dir) / "best_model.pt")

        stage2_trainer, stage2_history = run_stage2(
            args, train_loader_s2, val_loader_s2,
            stage1_checkpoint=stage1_ckpt
        )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if args.stage in ["2", "both"]:
        logger.info(f"\nFinal model saved to: {args.stage2_checkpoint_dir}/best_model.pt")
        logger.info("Ready for evaluation on holdout set!")


if __name__ == "__main__":
    main()

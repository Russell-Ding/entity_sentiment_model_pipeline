#!/usr/bin/env python3
"""Streamlined three-stage training pipeline (the May 2026 v2.0 approach).

This is the proven recipe that produced `trained_model/v2.0_20260517/`. Use
it whenever you want to retrain from scratch on new data — typically:
- Annotated more news articles? Replace train.jsonl / val.jsonl and rerun.
- Switched encoder size or max_length? Override flags and rerun.

Pipeline stages (each runs in a subprocess so GPU memory is released between):
  1. Stage 1 — NER-only with global-attention dropout (p=0.3, warmup 30%)
     → encoder + NER head trained to handle both CLS-only and entity-aware
       global attention regimes
     → output: best_cls_only.pt (highest CLS-only NER F1)
  2. Stage 2 — Joint NER + sentiment with curriculum
     → fine-tunes the same backbone with sentiment-head training enabled
     → output: best_model.pt + best_cls_only.pt
  3. Stage 3 — Fresh SentimentHead retrain on FROZEN Stage 1 backbone
     → IMPORTANT: loads STAGE 1's best_cls_only.pt, NOT Stage 2's. The
       Stage 2 joint training slightly degrades the CLS-only NER F1, and
       Stage 3 throws away the sentiment head anyway, so using Stage 1's
       backbone gives a stronger e2e result.
     → output: best_model.pt, checkpoint_epoch_*.pt

When complete, the script extracts the best epoch from Stage 3 and writes a
production-ready bundle to `trained_model/v<VERSION>_<DATE>/` with:
  - model.pt (just weights + training metadata, no optimizer state)
  - config.json
  - tokenizer/
  - MODEL_CARD.md (auto-generated from the run's metrics)

Usage on Colab (G4 95 GB GPU recommended; ~14 hours total):

    !python scripts/training/train_pipeline.py \\
        --version v3.0 \\
        --output-version-name v3.0_20260601 \\
        --run-id 20260601

Re-run a partial chain with --skip-stages 1,2 to skip already-done stages.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd, log_prefix=""):
    """Run a subprocess and stream its output to this script's stdout."""
    print(f"\n{log_prefix} >>> {' '.join(str(c) for c in cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"{log_prefix} subprocess failed with code {proc.returncode}")


def _validate_paths(train_data: Path, val_data: Path, encoder_name: str):
    if not train_data.exists():
        raise SystemExit(f"Train data not found: {train_data}")
    if not val_data.exists():
        raise SystemExit(f"Val data not found: {val_data}")
    # Encoder name is just a string; HF will download or fail at runtime.


def _package_final_model(
    version: str,
    version_name: str,
    run_id: str,
    stage3_local_dir: Path,
    stage3_drive_dir: Path,
    encoder_name: str,
    hidden_size: int,
    max_length: int,
    dropout_p: float,
    dropout_warmup_frac: float,
    train_data: Path,
    val_data: Path,
):
    """Find the best Stage 3 checkpoint and package it into trained_model/."""
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
    import torch

    # Find best Stage 3 epoch by val sentiment_corr (the file named best_model.pt
    # may be stale from a Drive-sync edge case, so scan all epoch ckpts)
    candidates = []
    for d in [stage3_local_dir, stage3_drive_dir]:
        if d.exists():
            for f in sorted(d.glob("checkpoint_epoch_*.pt")):
                try:
                    ck = torch.load(str(f), map_location="cpu", weights_only=False)
                    corr = ck.get("val_metrics", {}).get("sentiment_corr", -1.0)
                    candidates.append((corr, f, ck))
                except Exception as e:
                    print(f"  Skipping {f.name}: {e}", flush=True)
            break  # only scan the first dir that exists

    if not candidates:
        raise SystemExit("No Stage 3 epoch checkpoints found to package.")

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_corr, best_path, best_ck = candidates[0]
    print(f"\nPackaging final model: {best_path.name}  (val sent_corr={best_corr:.4f})", flush=True)

    # Production destination
    dest = PROJECT_ROOT / "trained_model" / version_name
    dest.mkdir(parents=True, exist_ok=True)

    # 1. model.pt — drop optimizer/scheduler, keep weights + minimal metadata
    val_metrics = best_ck.get("val_metrics", {})
    epoch_idx = best_ck.get("epoch")  # 0-indexed in train_stage3.py format
    payload = {
        "model_state_dict": best_ck["model_state_dict"],
        "epoch": epoch_idx,
        "val_metrics": val_metrics,
        "training_metadata": {
            "stage": "stage3_pipeline_retrain",
            "run_id": run_id,
            "trained_with_global_attn_dropout": True,
            "dropout_p": dropout_p,
            "dropout_warmup_frac": dropout_warmup_frac,
            "source_checkpoint": "stage1/best_cls_only.pt",
            "epoch_selected": (epoch_idx + 1) if epoch_idx is not None else None,
            "val_pearson_r": val_metrics.get("sentiment_corr"),
            "val_mse": val_metrics.get("sentiment_mse"),
            "val_ner_f1_entity_aware": val_metrics.get("ner_f1"),
            "encoder_name": encoder_name,
            "hidden_size": hidden_size,
            "max_length": max_length,
            "train_data": str(train_data),
            "val_data": str(val_data),
        },
    }
    torch.save(payload, dest / "model.pt")
    print(f"  Saved: {dest / 'model.pt'}  ({(dest / 'model.pt').stat().st_size / 1e9:.2f} GB)", flush=True)

    # 2. config.json
    label_map = {
        "O": 0, "B-COMPANY": 1, "I-COMPANY": 2, "B-TICKER": 3, "I-TICKER": 4,
        "B-PERSON": 5, "I-PERSON": 6, "B-ORG": 7, "I-ORG": 8,
        "B-MONEY": 9, "I-MONEY": 10, "B-PERCENT": 11, "I-PERCENT": 12,
        "B-DATE": 13, "I-DATE": 14,
    }
    config = {
        "encoder_name": encoder_name,
        "hidden_size": hidden_size,
        "max_length": max_length,
        "use_ner_head": True,
        "use_coref_head": False,
        "use_crf_ner": True,
        "num_ner_labels": 15,
        "ner_label_to_id": label_map,
        "sentiment_head": "cross_attention",
        "sentiment_score_range": [-0.95, 0.95],
        "trained_with_global_attn_dropout": True,
        "global_attn_dropout_prob": dropout_p,
        "global_attn_dropout_warmup_frac": dropout_warmup_frac,
    }
    with open(dest / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {dest / 'config.json'}", flush=True)

    # 3. Tokenizer — copy from the most recent prior version (Longformer tokenizer
    # is identical across base/large, so we can reuse any existing copy)
    tokenizer_sources = sorted((PROJECT_ROOT / "trained_model").glob("v*/tokenizer"))
    if tokenizer_sources:
        src = tokenizer_sources[-1]  # most recent
        if src.resolve() != (dest / "tokenizer").resolve():
            if (dest / "tokenizer").exists():
                shutil.rmtree(dest / "tokenizer")
            shutil.copytree(src, dest / "tokenizer")
            print(f"  Copied tokenizer from {src}", flush=True)

    # 4. MODEL_CARD.md
    sc = val_metrics.get("sentiment_corr", 0)
    sm = val_metrics.get("sentiment_mse", 0)
    nf = val_metrics.get("ner_f1", 0)
    card = f"""# Financial Entity Sentiment Model — {version}

## Model Version
- **Version**: {version}
- **Run ID**: {run_id}
- **Date packaged**: {datetime.now().strftime('%Y-%m-%d')}
- **Source training run**: Stage 1 + Stage 2 + Stage 3 (pipeline retrain)
- **Best epoch**: {(epoch_idx + 1) if epoch_idx is not None else '?'}

## Headline (val, gold-mask)
| Metric | Value |
|---|---|
| Sentiment Pearson r | {sc:.4f} |
| Sentiment MSE | {sm:.4f} |
| NER F1 (entity-aware) | {nf:.4f} |

## Architecture
- Encoder: `{encoder_name}` (hidden={hidden_size}, max_len={max_length})
- NER head: CRF, 15 BIO labels
- Sentiment head: cross-attention, output ∈ [-0.95, 0.95]

## Training
- Three-stage pipeline (`scripts/training/train_pipeline.py`)
- Global-attention dropout: p={dropout_p}, warmup {int(dropout_warmup_frac*100)}% of Stage 1 steps
- Train: `{train_data.name}`
- Val: `{val_data.name}`

## Loading

```python
import torch, sys
sys.path.insert(0, "path/to/entity_sentiment_model_pipeline")
from models.pipeline import FinancialEntitySentimentModel
from training.preprocessing import LABEL_TO_ID

model = FinancialEntitySentimentModel(
    encoder_name="{encoder_name}",
    hidden_size={hidden_size},
    num_ner_labels=15,
    use_ner_head=True,
    use_coref_head=False,
    use_crf_ner=True,
    ner_label_to_id=LABEL_TO_ID,
    max_length={max_length},
)
ck = torch.load("trained_model/{version_name}/model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()
```

For e2e inference, see `scripts/evaluation/evaluate_e2e_pipeline.py`.
"""
    with open(dest / "MODEL_CARD.md", "w") as f:
        f.write(card)
    print(f"  Saved: {dest / 'MODEL_CARD.md'}", flush=True)

    return dest


def main():
    p = argparse.ArgumentParser(
        description="Streamlined Stage 1+2+3 training pipeline (May-2026 v2.0 recipe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    p.add_argument("--train-data", type=str,
                   default=str(PROJECT_ROOT / "data/labeled/final/train.jsonl"))
    p.add_argument("--val-data", type=str,
                   default=str(PROJECT_ROOT / "data/labeled/final/val.jsonl"))

    # Model architecture
    p.add_argument("--encoder-name", type=str, default="allenai/longformer-large-4096")
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=2048)

    # Hyperparameters
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--stage2-epochs", type=int, default=5)
    p.add_argument("--stage3-epochs", type=int, default=10)
    p.add_argument("--batch-size-train", type=int, default=24,
                   help="Stages 1+2 batch size (Stage 3 uses its own --stage3-batch).")
    p.add_argument("--stage3-batch", type=int, default=80)
    p.add_argument("--stage1-lr", type=float, default=2e-5)
    p.add_argument("--stage2-lr", type=float, default=1e-5)
    p.add_argument("--stage3-lr", type=float, default=5e-4)
    p.add_argument("--global-attn-dropout-prob", type=float, default=0.3)
    p.add_argument("--stage1-dropout-warmup-frac", type=float, default=0.3)
    p.add_argument("--stage2-dropout-warmup-frac", type=float, default=0.0)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True,
                   help="Enable encoder gradient checkpointing (recommended for <80GB GPUs).")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    # Output organization
    p.add_argument("--run-id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                   help="Used to name intermediate checkpoint dirs.")
    p.add_argument("--version", type=str, default="v3.0",
                   help="Model version label for the final bundle.")
    p.add_argument("--output-version-name", type=str, default=None,
                   help="trained_model/<NAME>/ for the final bundle. "
                        "Default: <version>_<run-id>")
    p.add_argument("--local-ckpt-root", type=str, default="/content",
                   help="Local fast-disk root for checkpoints during training "
                        "(used as /content on Colab). Final bundle is always "
                        "written to trained_model/<NAME>/ in the project.")
    p.add_argument("--drive-ckpt-root", type=str,
                   default=str(PROJECT_ROOT / "checkpoints"))

    # Stage selection
    p.add_argument("--skip-stages", type=str, default="",
                   help="Comma-separated stages to skip (e.g. '1,2'). "
                        "Useful for resuming a partial run.")
    p.add_argument("--skip-packaging", action="store_true",
                   help="Don't build the final trained_model/ bundle.")

    args = p.parse_args()

    train_data = Path(args.train_data)
    val_data = Path(args.val_data)
    _validate_paths(train_data, val_data, args.encoder_name)

    skip = {int(s.strip()) for s in args.skip_stages.split(",") if s.strip()}
    version_name = args.output_version_name or f"{args.version}_{args.run_id}"

    # Per-run directories — clearly named so different runs don't clobber each other
    stage1_local = Path(args.local_ckpt_root) / f"stage1_{args.run_id}"
    stage2_local = Path(args.local_ckpt_root) / f"stage2_{args.run_id}"
    stage3_local = Path(args.local_ckpt_root) / f"stage3_{args.run_id}"
    stage1_drive = Path(args.drive_ckpt_root) / f"stage1_ner_{args.run_id}"
    stage2_drive = Path(args.drive_ckpt_root) / f"stage2_joint_{args.run_id}"
    stage3_drive = Path(args.drive_ckpt_root) / f"stage3_sentiment_{args.run_id}"
    for d in [stage1_local, stage2_local, stage3_local,
              stage1_drive, stage2_drive, stage3_drive]:
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80, flush=True)
    print("  STREAMLINED TRAINING PIPELINE  (May-2026 v2.0 recipe)", flush=True)
    print("=" * 80, flush=True)
    print(f"  Train data        : {train_data}", flush=True)
    print(f"  Val data          : {val_data}", flush=True)
    print(f"  Encoder           : {args.encoder_name}  (hidden={args.hidden_size}, max_len={args.max_length})", flush=True)
    print(f"  Dropout           : p={args.global_attn_dropout_prob}, S1 warmup {args.stage1_dropout_warmup_frac}", flush=True)
    print(f"  Output bundle     : trained_model/{version_name}/", flush=True)
    print(f"  Skip stages       : {sorted(skip) if skip else 'none'}", flush=True)
    print()

    common_train_args = [
        "--train_file", str(train_data),
        "--val_file", str(val_data),
        "--encoder_name", args.encoder_name,
        "--hidden_size", str(args.hidden_size),
        "--max_length", str(args.max_length),
        "--batch_size", str(args.batch_size_train),
        "--global_attn_dropout_prob", str(args.global_attn_dropout_prob),
        "--stage1_dropout_warmup_frac", str(args.stage1_dropout_warmup_frac),
        "--stage2_dropout_warmup_frac", str(args.stage2_dropout_warmup_frac),
        "--stage1_checkpoint_dir", str(stage1_local),
        "--stage2_checkpoint_dir", str(stage2_local),
    ]
    if args.gradient_checkpointing:
        common_train_args.append("--gradient_checkpointing")

    # -----------------------------------------------------------------
    # Stage 1 — NER-only with global-attention dropout
    # -----------------------------------------------------------------
    if 1 not in skip:
        cmd = [sys.executable, "scripts/training/train_two_stage.py",
               "--stage", "1",
               "--stage1_epochs", str(args.stage1_epochs),
               "--learning_rate", str(args.stage1_lr)] + common_train_args
        _run(cmd, log_prefix="[STAGE 1]")
    else:
        print("[STAGE 1] skipped", flush=True)

    stage1_best_cls = stage1_local / "best_cls_only.pt"
    if not stage1_best_cls.exists():
        # Fallback to best_model.pt
        candidate = stage1_local / "best_model.pt"
        if candidate.exists():
            stage1_best_cls = candidate
            print(f"[STAGE 1] best_cls_only.pt missing — falling back to best_model.pt", flush=True)
        else:
            raise SystemExit(f"Stage 1 checkpoint not found at {stage1_local}. Cannot proceed.")

    # -----------------------------------------------------------------
    # Stage 2 — Joint with curriculum
    # -----------------------------------------------------------------
    if 2 not in skip:
        cmd = [sys.executable, "scripts/training/train_two_stage.py",
               "--stage", "2",
               "--stage2_epochs", str(args.stage2_epochs),
               "--stage2_lr", str(args.stage2_lr),
               "--stage1_checkpoint", str(stage1_best_cls)] + common_train_args
        _run(cmd, log_prefix="[STAGE 2]")
    else:
        print("[STAGE 2] skipped", flush=True)

    # -----------------------------------------------------------------
    # Stage 3 — Fresh SentimentHead retrain on Stage 1's backbone
    #   IMPORTANT: input is stage1_best_cls (NOT Stage 2's best_model.pt)
    # -----------------------------------------------------------------
    if 3 not in skip:
        cmd = [sys.executable, "scripts/training/train_stage3.py",
               "--source-checkpoint", str(stage1_best_cls),
               "--drive-ckpt-dir", str(stage3_drive),
               "--local-ckpt-dir", str(stage3_local),
               "--encoder-name", args.encoder_name,
               "--hidden-size", str(args.hidden_size),
               "--max-length", str(args.max_length),
               "--batch-size", str(args.stage3_batch),
               "--epochs", str(args.stage3_epochs),
               "--lr", str(args.stage3_lr)]
        _run(cmd, log_prefix="[STAGE 3]")
    else:
        print("[STAGE 3] skipped", flush=True)

    elapsed_hours = (time.time() - t_start) / 3600
    print(f"\nAll stages complete in {elapsed_hours:.2f} hours.", flush=True)

    # -----------------------------------------------------------------
    # Package final model
    # -----------------------------------------------------------------
    if args.skip_packaging:
        print("Skipping production packaging (--skip-packaging set).", flush=True)
        return

    _package_final_model(
        version=args.version,
        version_name=version_name,
        run_id=args.run_id,
        stage3_local_dir=stage3_local,
        stage3_drive_dir=stage3_drive,
        encoder_name=args.encoder_name,
        hidden_size=args.hidden_size,
        max_length=args.max_length,
        dropout_p=args.global_attn_dropout_prob,
        dropout_warmup_frac=args.stage1_dropout_warmup_frac,
        train_data=train_data,
        val_data=val_data,
    )
    print(f"\nDone. Production model at: trained_model/{version_name}/", flush=True)


if __name__ == "__main__":
    main()

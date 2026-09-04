#!/usr/bin/env python3
"""Holdout Evaluation for Stage 3 Sentiment Model.

Supports the cross-attention SentimentHead (current architecture). Legacy V1
self-attention checkpoints are archived and not loadable here — use git
history pre-2026-05-17 if needed.

Usage:
    # Evaluate latest best_model.pt (auto-detect v1/v2)
    python scripts/evaluation/evaluate_holdout_stage3.py

    # Evaluate a specific checkpoint
    python scripts/evaluation/evaluate_holdout_stage3.py --checkpoint checkpoints/stage3_sentiment_large/checkpoint_epoch_5.pt

    # Quick test with subsample
    python scripts/evaluation/evaluate_holdout_stage3.py --max-samples 200 --device cpu
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# transformers torch.load safety check workaround
# ---------------------------------------------------------------------------
import transformers.utils.import_utils as _tiu
_tiu.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _tmu
_tmu.check_torch_load_is_safe = lambda: None

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import FinancialEntitySentimentModel
from training.preprocessing import (
    DataPreprocessor,
    NER_LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    SENTIMENT_ENTITY_TYPES,
)
from training.dataset import EntitySentimentDataset, collate_fn
from training.metrics import compute_ner_metrics, compute_sentiment_metrics

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
STAGE3_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model_20260418.pt"
STAGE2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "archive" / "stage2_joint_large" / "best_model.pt"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "labeled" / "final" / "holdout.jsonl"
ENCODER_NAME = "allenai/longformer-large-4096"
HIDDEN_SIZE = 1024
MAX_LENGTH = 2048
NUM_NER_LABELS = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SENTIMENT_BUCKETS = [
    ("very_negative", -1.0, -0.6),
    ("negative", -0.6, -0.2),
    ("neutral", -0.2, 0.2),
    ("positive", 0.2, 0.6),
    ("very_positive", 0.6, 1.01),
]


def bucket_name(score: float) -> str:
    for name, lo, hi in SENTIMENT_BUCKETS:
        if lo <= score < hi:
            return name
    return "very_positive" if score >= 1.0 else "very_negative"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested)


def load_model_with_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a SentimentHead (cross-attention) checkpoint.

    Old V1 self-attention checkpoints (Apr 2026 era) are no longer supported;
    they are archived under checkpoints/archive/ for reference only.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    if any(k.startswith("sentiment_head.entity_attention.") for k in state_dict):
        raise RuntimeError(
            "Checkpoint contains the legacy V1 self-attention sentiment head. "
            "This eval script only supports the V2 cross-attention head used "
            "in trained_model/v2.0_20260517/. Use git history (pre-2026-05-17) "
            "if you need to load V1 checkpoints."
        )

    model = FinancialEntitySentimentModel(
        encoder_name=ENCODER_NAME,
        hidden_size=HIDDEN_SIZE,
        num_ner_labels=NUM_NER_LABELS,
        use_ner_head=True,
        use_coref_head=False,
        use_crf_ner=True,
        ner_label_to_id=LABEL_TO_ID,
        device="cpu",
        max_length=MAX_LENGTH,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_metrics = checkpoint.get("val_metrics", {})
    epoch = checkpoint.get("epoch", "?")
    history = checkpoint.get("history", {})

    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, val_metrics, epoch, history


@torch.no_grad()
def run_holdout_evaluation(model, loader, device, n_samples, entity_type_map):
    """Run inference and collect predictions."""
    all_ner_preds = []
    all_ner_labels = []
    all_sentiment_preds = []
    all_sentiment_targets = []
    all_sentiment_entity_types = []
    per_label_tp = defaultdict(int)
    per_label_fp = defaultdict(int)
    per_label_fn = defaultdict(int)

    total_batches = len(loader)
    sample_cursor = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        if (batch_idx + 1) % max(total_batches // 20, 1) == 0 or batch_idx == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
            logger.info(f"  Batch {batch_idx + 1}/{total_batches}  "
                        f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ner_labels_batch = batch["ner_labels"].to(device)
        entity_masks = batch["entity_masks"].to(device)
        sentiment_targets = batch["sentiment_scores"].to(device)
        entity_mask_valid = batch["entity_mask_valid"].to(device)

        try:
            ner_output, sentiment_preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_masks=entity_masks,
            )
        except RuntimeError as e:
            if "mps" in str(e).lower():
                logger.warning(f"  MPS error on batch {batch_idx}, falling back to CPU")
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
                ner_labels_batch = ner_labels_batch.cpu()
                entity_masks = entity_masks.cpu()
                sentiment_targets = sentiment_targets.cpu()
                entity_mask_valid = entity_mask_valid.cpu()
                model_cpu = model.cpu()
                ner_output, sentiment_preds = model_cpu(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_masks=entity_masks,
                )
                model.to(device)
            else:
                raise

        bs = input_ids.shape[0]

        # NER predictions
        if isinstance(ner_output, dict) and "predictions" in ner_output:
            ner_pred_ids = ner_output["predictions"]
        else:
            ner_pred_ids = ner_output.argmax(dim=-1)

        valid_mask = attention_mask.bool().cpu()
        ner_pred_ids_cpu = ner_pred_ids.cpu()
        ner_labels_cpu = ner_labels_batch.cpu()

        for i in range(bs):
            tok_mask = valid_mask[i]
            preds_i = ner_pred_ids_cpu[i][tok_mask].tolist()
            labels_i = ner_labels_cpu[i][tok_mask].tolist()
            all_ner_preds.extend(preds_i)
            all_ner_labels.extend(labels_i)

            for p, l in zip(preds_i, labels_i):
                p_lab = ID_TO_LABEL.get(p, "O")
                l_lab = ID_TO_LABEL.get(l, "O")
                if p_lab != "O" and l_lab != "O" and p_lab == l_lab:
                    per_label_tp[p_lab] += 1
                elif p_lab != "O":
                    per_label_fp[p_lab] += 1
                if l_lab != "O" and p_lab != l_lab:
                    per_label_fn[l_lab] += 1

        # Sentiment predictions
        sentiment_preds_cpu = sentiment_preds.cpu()
        sentiment_targets_cpu = sentiment_targets.cpu()
        entity_mask_valid_cpu = entity_mask_valid.cpu()

        for i in range(bs):
            for j in range(entity_mask_valid_cpu.shape[1]):
                if entity_mask_valid_cpu[i, j] > 0:
                    all_sentiment_preds.append(sentiment_preds_cpu[i, j].item())
                    all_sentiment_targets.append(sentiment_targets_cpu[i, j].item())
                    sample_idx = sample_cursor + i
                    etype = entity_type_map.get((sample_idx, j), "UNKNOWN")
                    all_sentiment_entity_types.append(etype)

        sample_cursor += bs

    elapsed_total = time.time() - t_start
    logger.info(f"Inference complete in {elapsed_total:.1f}s "
                f"({elapsed_total / max(n_samples, 1):.2f}s/sample)")

    return {
        "all_ner_preds": all_ner_preds,
        "all_ner_labels": all_ner_labels,
        "all_sentiment_preds": all_sentiment_preds,
        "all_sentiment_targets": all_sentiment_targets,
        "all_sentiment_entity_types": all_sentiment_entity_types,
        "per_label_tp": per_label_tp,
        "per_label_fp": per_label_fp,
        "per_label_fn": per_label_fn,
        "elapsed": elapsed_total,
    }


def print_results(results, n_samples, val_metrics_saved, epoch):
    """Print comprehensive evaluation results."""
    all_ner_preds = results["all_ner_preds"]
    all_ner_labels = results["all_ner_labels"]
    all_sentiment_preds = results["all_sentiment_preds"]
    all_sentiment_targets = results["all_sentiment_targets"]
    all_sentiment_entity_types = results["all_sentiment_entity_types"]
    per_label_tp = results["per_label_tp"]
    per_label_fp = results["per_label_fp"]
    per_label_fn = results["per_label_fn"]

    print()
    print("=" * 80)
    print(f"  HOLDOUT EVALUATION  --  Stage 3 Sentiment Head")
    print(f"  Checkpoint epoch: {epoch}  |  {n_samples} holdout samples")
    print("=" * 80)

    # NER Metrics
    ner_metrics = compute_ner_metrics(all_ner_preds, all_ner_labels)
    print()
    print("-" * 60)
    print("  NER METRICS (Token-Level)")
    print("-" * 60)
    print(f"  Precision : {ner_metrics['ner_precision']:.4f}")
    print(f"  Recall    : {ner_metrics['ner_recall']:.4f}")
    print(f"  F1        : {ner_metrics['ner_f1']:.4f}")

    # Per-label NER
    print()
    print("-" * 60)
    print("  PER-LABEL NER BREAKDOWN")
    print("-" * 60)
    print(f"  {'Label':<14} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  {'-'*6} {'-'*6} {'-'*6}")

    all_labels_sorted = sorted(
        set(list(per_label_tp.keys()) + list(per_label_fp.keys()) + list(per_label_fn.keys())),
        key=lambda x: LABEL_TO_ID.get(x, 99),
    )
    for lab in all_labels_sorted:
        tp = per_label_tp.get(lab, 0)
        fp = per_label_fp.get(lab, 0)
        fn = per_label_fn.get(lab, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {lab:<14} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}  {tp:>6} {fp:>6} {fn:>6}")

    # Sentiment Metrics
    sent_metrics = {}
    print()
    print("-" * 60)
    print("  SENTIMENT METRICS (Overall)")
    print("-" * 60)

    if all_sentiment_preds:
        sent_metrics = compute_sentiment_metrics(all_sentiment_preds, all_sentiment_targets)
        print(f"  MSE         : {sent_metrics['sentiment_mse']:.6f}")
        print(f"  MAE         : {sent_metrics['sentiment_mae']:.6f}")
        print(f"  Pearson r   : {sent_metrics['sentiment_corr']:.6f}")
        print(f"  Total entities: {len(all_sentiment_preds)}")

        preds_t = torch.tensor(all_sentiment_preds)
        tgts_t = torch.tensor(all_sentiment_targets)
        print(f"  Prediction range : [{preds_t.min().item():.4f}, {preds_t.max().item():.4f}]")
        print(f"  Target range     : [{tgts_t.min().item():.4f}, {tgts_t.max().item():.4f}]")
        print(f"  Prediction mean  : {preds_t.mean().item():.4f}  std={preds_t.std().item():.4f}")
        print(f"  Target mean      : {tgts_t.mean().item():.4f}  std={tgts_t.std().item():.4f}")
    else:
        print("  No sentiment entities found!")

    # Per-entity-type sentiment
    print()
    print("-" * 60)
    print("  PER-ENTITY-TYPE SENTIMENT BREAKDOWN")
    print("-" * 60)

    etype_sent = defaultdict(lambda: {"preds": [], "targets": []})
    for pred, tgt, etype in zip(all_sentiment_preds, all_sentiment_targets, all_sentiment_entity_types):
        etype_sent[etype]["preds"].append(pred)
        etype_sent[etype]["targets"].append(tgt)

    print(f"  {'Type':<12} {'Count':>6} {'MSE':>10} {'MAE':>10} {'Corr':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for etype in sorted(etype_sent.keys()):
        data = etype_sent[etype]
        m = compute_sentiment_metrics(data["preds"], data["targets"])
        print(f"  {etype:<12} {len(data['preds']):>6} {m['sentiment_mse']:>10.6f} "
              f"{m['sentiment_mae']:>10.6f} {m['sentiment_corr']:>10.6f}")

    # Bucket confusion matrix
    print()
    print("-" * 60)
    print("  SENTIMENT BUCKET CONFUSION MATRIX")
    print("  (rows = true, columns = predicted)")
    print("-" * 60)

    bucket_names = [b[0] for b in SENTIMENT_BUCKETS]
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, tgt in zip(all_sentiment_preds, all_sentiment_targets):
        confusion[bucket_name(tgt)][bucket_name(pred)] += 1

    header = f"  {'True \\ Pred':<16}"
    for bn in bucket_names:
        header += f" {bn:>14}"
    header += f" {'Total':>8}"
    print(header)
    print(f"  {'-'*16}" + f" {'-'*14}" * len(bucket_names) + f" {'-'*8}")

    for true_b in bucket_names:
        row = f"  {true_b:<16}"
        row_total = 0
        for pred_b in bucket_names:
            cnt = confusion[true_b][pred_b]
            row_total += cnt
            row += f" {cnt:>14}"
        row += f" {row_total:>8}"
        print(row)

    grand_total = sum(sum(confusion[tb].values()) for tb in bucket_names)
    if grand_total > 0:
        diag_sum = sum(confusion[b][b] for b in bucket_names)
        bucket_acc = diag_sum / grand_total
        print(f"\n  Bucket accuracy (exact): {bucket_acc:.4f} ({diag_sum}/{grand_total})")

        adj_correct = 0
        for i, true_b in enumerate(bucket_names):
            for j, pred_b in enumerate(bucket_names):
                if abs(i - j) <= 1:
                    adj_correct += confusion[true_b][pred_b]
        adj_acc = adj_correct / grand_total
        print(f"  Adjacent bucket accuracy: {adj_acc:.4f} ({adj_correct}/{grand_total})")

    # Comparison with checkpoint val metrics
    if val_metrics_saved:
        print()
        print("-" * 60)
        print("  COMPARISON: VAL (checkpoint) vs HOLDOUT")
        print("-" * 60)
        print(f"  {'Metric':<24} {'Val (ckpt)':>14} {'Holdout':>14} {'Delta':>10}")
        print(f"  {'-'*24} {'-'*14} {'-'*14} {'-'*10}")

        comparisons = [
            ("NER F1", "ner_f1", ner_metrics, False),
            ("Sentiment MSE", "sentiment_mse", sent_metrics, True),
            ("Sentiment MAE", "sentiment_mae", sent_metrics, True),
            ("Sentiment Corr", "sentiment_corr", sent_metrics, False),
        ]
        for label, key, metrics_dict, lower_is_better in comparisons:
            val_v = val_metrics_saved.get(key, 0)
            ho_v = metrics_dict.get(key, 0)
            delta = ho_v - val_v
            sign = "+" if delta > 0 else ""
            print(f"  {label:<24} {val_v:>14.6f} {ho_v:>14.6f} {sign}{delta:>9.6f}")

    print()
    print("=" * 80)
    return ner_metrics, sent_metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 3 holdout evaluation (v1/v2 auto-detect)")
    parser.add_argument("--checkpoint", type=str, default=str(STAGE3_CHECKPOINT),
                        help="Path to checkpoint .pt file")
    parser.add_argument("--holdout", type=str, default=str(HOLDOUT_PATH))
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    # Load model
    model, val_metrics, epoch, history = load_model_with_checkpoint(
        args.checkpoint, device
    )

    # Print training history summary
    if history:
        print()
        print("-" * 60)
        print("  TRAINING HISTORY")
        print("-" * 60)
        corr_hist = history.get("val_sentiment_corr", [])
        mse_hist = history.get("val_sentiment_mse", [])
        for i, (c, m) in enumerate(zip(corr_hist, mse_hist)):
            marker = " <-- best" if i == epoch else ""
            print(f"  Epoch {i+1}: corr={c:.4f}  mse={m:.4f}{marker}")

    param_count = sum(p.numel() for p in model.parameters())
    sent_params = sum(p.numel() for p in model.sentiment_head.parameters())
    logger.info(f"Total params: {param_count:,}  |  Sentiment head: {sent_params:,}")

    # Preprocess holdout data
    logger.info(f"Preprocessing holdout: {args.holdout}")
    preprocessor = DataPreprocessor(
        model_name=ENCODER_NAME,
        max_length=MAX_LENGTH,
        use_expanded_sentiment=True,
    )
    holdout_samples = preprocessor.process_file(args.holdout)
    logger.info(f"Preprocessed {len(holdout_samples)} samples")

    if args.max_samples > 0 and len(holdout_samples) > args.max_samples:
        import random
        random.seed(42)
        holdout_samples = random.sample(holdout_samples, args.max_samples)
        logger.info(f"Sub-sampled to {len(holdout_samples)}")

    n_samples = len(holdout_samples)

    # Entity type map
    entity_type_map = {}
    for si, sample in enumerate(holdout_samples):
        for ei, ent in enumerate(sample.entities):
            entity_type_map[(si, ei)] = ent.get("type", "UNKNOWN")

    # DataLoader
    dataset = EntitySentimentDataset(holdout_samples, max_entities_per_sample=10)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    logger.info(f"DataLoader: {len(loader)} batches")

    # Run evaluation
    results = run_holdout_evaluation(model, loader, device, n_samples, entity_type_map)

    # Print results
    ner_metrics, sent_metrics = print_results(
        results, n_samples, val_metrics, epoch
    )


if __name__ == "__main__":
    main()

"""Metric functions shared by the trainer and the evaluation scripts.

Kept separate from trainer.py so the evaluation code can run without the
training loop (which is not part of the public release).
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .preprocessing import ID_TO_LABEL


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

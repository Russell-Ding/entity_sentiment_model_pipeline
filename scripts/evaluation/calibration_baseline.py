#!/usr/bin/env python3
"""Phase-0 calibration baseline (per Codex review) — runs on saved e2e predictions.

The current model under-polarizes (pred std 0.167 vs gold 0.275). Before any
retrain we test the cheapest possible fix: a post-hoc calibrator on the model's
existing predictions vs the new decisive labels.

  - affine     : pred' = a*pred + b   (least squares, fit set)
  - isotonic   : monotonic fit        (sklearn IsotonicRegression, clip)

Fit on one article-id split, evaluate on a disjoint split (no leakage), and
compare raw vs affine vs isotonic on the metrics that matter.

KEY INTERPRETATION: affine (a>0) does NOT change Pearson; isotonic (monotonic)
does NOT change Spearman. So calibration can only fix MAGNITUDE (std, MAE, bucket
recall), never RANKING. If calibration restores std/buckets but Pearson stays
~0.485, the 0.485->0.63 gap is a genuine ranking limit -> the retrain is justified
on correlation grounds. If calibration ALSO closes most of the gap, much of the
deficit was just slope and the retrain bar is higher / may be unnecessary.

Usage:
  python3 scripts/evaluation/calibration_baseline.py \
    --predictions outputs/eval_new_dataset/e2e_predictions_*.jsonl \
    --output outputs/eval_new_dataset/calibration_baseline.json
"""
from __future__ import annotations
import argparse, glob, hashlib, json
import numpy as np

BUCKETS = [("very_neg", -1.01, -0.6), ("neg", -0.6, -0.2), ("neutral", -0.2, 0.2),
           ("pos", 0.2, 0.6), ("very_pos", 0.6, 1.01)]


def bucket_of(s):
    for n, lo, hi in BUCKETS:
        if lo <= s < hi:
            return n
    return "very_pos" if s >= 0.6 else "very_neg"


def pearson(x, y):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(a):
        o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        cs = np.cumsum(cnt); st = cs - cnt
        return ((st + cs - 1) / 2.0)[inv]
    return pearson(rank(x), rank(y))


def split_by_id(ids, frac_fit=0.5):
    """Deterministic article-id hash split -> (fit_mask, eval_mask)."""
    h = np.array([int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 1000 for i in ids])
    fit = h < int(frac_fit * 1000)
    return fit, ~fit


def metrics(gold, pred):
    gold, pred = np.asarray(gold, float), np.asarray(pred, float)
    m = {
        "n": int(len(gold)),
        "pearson": round(pearson(pred, gold), 4),
        "spearman": round(spearman(pred, gold), 4),
        "pred_std": round(float(pred.std()), 4),
        "gold_std": round(float(gold.std()), 4),
        "std_ratio": round(float(pred.std() / gold.std()), 4) if gold.std() else float("nan"),
        "mae": round(float(np.abs(pred - gold).mean()), 4),
        "mean_bias": round(float(pred.mean() - gold.mean()), 4),
    }
    # over-neutrality + neutral false-polarization + sign flips
    strong = np.abs(gold) >= 0.4
    m["over_neutral_rate"] = round(float((np.abs(pred[strong]) < 0.1).mean()), 4) if strong.any() else None
    neutral = np.abs(gold) < 0.1
    m["neutral_false_polar_rate"] = round(float((np.abs(pred[neutral]) > 0.3).mean()), 4) if neutral.any() else None
    bs = (np.abs(gold) >= 0.1) & (np.abs(pred) >= 0.1)
    m["sign_flip_rate"] = round(float((np.sign(gold[bs]) != np.sign(pred[bs])).mean()), 4) if bs.any() else None
    # extreme-bucket precision/recall/F1 (Codex: recall alone is gameable)
    gb = np.array([bucket_of(s) for s in gold])
    pb = np.array([bucket_of(s) for s in pred])
    for ex in ("very_neg", "very_pos"):
        tp = int(((gb == ex) & (pb == ex)).sum())
        fp = int(((gb != ex) & (pb == ex)).sum())
        fn = int(((gb == ex) & (pb != ex)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        m[f"{ex}_P/R/F1"] = f"{prec:.2f}/{rec:.2f}/{f1:.2f}"
        m[f"{ex}_n_gold"] = tp + fn
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default="outputs/eval_new_dataset/calibration_baseline.json")
    ap.add_argument("--frac-fit", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted(glob.glob(args.predictions))
    if not files:
        raise SystemExit(f"no predictions match {args.predictions}")
    ids, gold, pred = [], [], []
    for fp in files:
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            for e in r.get("per_entity_results", []):
                ids.append(r["id"]); gold.append(float(e["gold_sentiment"]))
                pred.append(float(e["agg_pred_sentiment"]))
    ids = np.array(ids, dtype=object); gold = np.array(gold); pred = np.array(pred)
    fit, ev = split_by_id(ids, args.frac_fit)
    print(f"entities: {len(gold)}  fit={fit.sum()}  eval={ev.sum()}")
    if fit.sum() < 10 or ev.sum() < 10 or np.unique(pred[fit]).size < 3:
        raise SystemExit("Too few / near-constant predictions to fit a calibrator "
                         f"(fit={fit.sum()}, eval={ev.sum()}, unique_fit_preds={np.unique(pred[fit]).size}).")

    # ---- fit calibrators on FIT split ----
    a, b = np.polyfit(pred[fit], gold[fit], 1)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=-1.0, y_max=1.0)
    iso.fit(pred[fit], gold[fit])

    # ---- evaluate on EVAL split ----
    raw_m = metrics(gold[ev], pred[ev])
    aff_m = metrics(gold[ev], np.clip(a * pred[ev] + b, -1, 1))
    iso_m = metrics(gold[ev], iso.predict(pred[ev]))

    out = {
        "n_entities_total": int(len(gold)),
        "fit_n": int(fit.sum()), "eval_n": int(ev.sum()),
        "affine": {"a": round(float(a), 4), "b": round(float(b), 4)},
        "raw": raw_m, "affine_calibrated": aff_m, "isotonic_calibrated": iso_m,
    }
    import os
    d = os.path.dirname(args.output)
    if d:
        os.makedirs(d, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    cols = ["pearson", "spearman", "std_ratio", "mae", "mean_bias",
            "over_neutral_rate", "neutral_false_polar_rate", "sign_flip_rate",
            "very_neg_P/R/F1", "very_pos_P/R/F1"]
    print(f"\naffine fit:  pred' = {a:.3f}*pred + {b:+.3f}")
    print(f"\n{'metric':26} {'raw':>16} {'affine':>16} {'isotonic':>16}")
    for c in cols:
        print(f"{c:26} {str(raw_m.get(c)):>16} {str(aff_m.get(c)):>16} {str(iso_m.get(c)):>16}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

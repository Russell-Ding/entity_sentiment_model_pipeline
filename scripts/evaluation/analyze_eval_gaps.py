#!/usr/bin/env python3
"""Phase-3 gap / error analysis for the current model vs the new DeepSeek+Kimi labels.

Consumes the `--save-predictions` output of evaluate_e2e_pipeline.py
(e2e_predictions_*.jsonl), where each line has:
    per_entity_results: [{canonical_id, type, gold_sentiment, agg_pred_sentiment, n_matched_preds}]
    predicted_spans:    [{type, char_start, char_end, text, pred_sentiment, match_iou}]

Produces the actionable tables the retrain decision needs (per the Kimi review):
  - overall + per-type Pearson r AND Spearman rho, with bootstrap CIs
  - per-company (canonical_id) r for the most frequent entities
  - trivial baseline (predict-the-global-mean) MAE vs the model's MAE
  - over-neutrality rate (gold |s|>=0.4 but model ~0)  -> the lawsuit/probe blind spot
  - sign-flip rate, with concrete examples (text pulled from the final jsonl)
  - error-by-gold-magnitude bins
  - bucket confusion matrix (very_neg..very_pos)
  - matched-span IoU distribution + unmatched counts (the span-contamination check)
  - PERSON split: Kimi-rescored vs not (isolates the rescoring effect)
  - calibration table (binned mean pred vs mean gold), + PNG if matplotlib present

Writes <output-dir>/gap_analysis.json and gap_analysis.md (+ calibration.png).
Pure numpy; correlations implemented locally so no scipy dependency.

Usage:
    python3 scripts/evaluation/analyze_eval_gaps.py \
        --predictions outputs/eval_new_dataset/e2e_predictions_*.jsonl \
        --final data/labeled/deepseek_t1/labels_10ticker.final.jsonl \
        --person-scores data/labeled/deepseek_t1/person_scores.jsonl \
        --output-dir outputs/eval_new_dataset
"""
from __future__ import annotations
import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

SENT_TYPES = {"ORG", "TICKER", "PERSON", "COMPANY"}
BUCKETS = [("very_neg", -1.01, -0.6), ("neg", -0.6, -0.2), ("neutral", -0.2, 0.2),
           ("pos", 0.2, 0.6), ("very_pos", 0.6, 1.01)]


def bucket_of(s: float) -> str:
    for n, lo, hi in BUCKETS:
        if lo <= s < hi:
            return n
    return "very_pos" if s >= 0.6 else "very_neg"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _rank(a: np.ndarray) -> np.ndarray:
    # average ranks (ties shared) — for Spearman
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # resolve ties to average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def boot_ci(x: np.ndarray, y: np.ndarray, fn, n_boot=1000, seed=0):
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(x[idx], y[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fmt_ci(point, ci):
    if math.isnan(point):
        return "n/a"
    return f"{point:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def load_predictions(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No predictions files match: {pattern}")
    rows = []          # per covered entity
    iou_matched = []   # match_iou for matched spans (>0)
    n_pred_spans = 0
    n_unmatched_pred = 0
    for fp in files:
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            aid = r["id"]
            for e in r.get("per_entity_results", []):
                rows.append((aid, e.get("canonical_id"), e.get("type"),
                             float(e["gold_sentiment"]), float(e["agg_pred_sentiment"])))
            for s in r.get("predicted_spans", []):
                if s.get("type") not in SENT_TYPES:
                    continue
                n_pred_spans += 1
                iou = s.get("match_iou", 0.0) or 0.0
                if iou > 0:
                    iou_matched.append(iou)
                else:
                    n_unmatched_pred += 1
    return rows, np.array(iou_matched), n_pred_spans, n_unmatched_pred, files


def load_person_rescored(final_path: Path, person_scores_path: Path):
    """Return set of (article_id, canonical_id) for PERSON entities Kimi actually rescored."""
    if not person_scores_path or not person_scores_path.exists():
        return None
    kimi = {}  # id -> set(canonical_name)
    for line in open(person_scores_path):
        if not line.strip():
            continue
        r = json.loads(line)
        kimi[r["id"]] = set((r.get("scores") or {}).keys())
    rescored = set()
    text_by_id = {}
    ents_by_id = {}
    for line in open(final_path):
        if not line.strip():
            continue
        r = json.loads(line)
        aid = r["id"]
        text_by_id[aid] = r.get("text", "")
        ents_by_id[aid] = r.get("entities", [])
        names = kimi.get(aid)
        if not names:
            continue
        for e in r.get("entities", []):
            if e.get("type") == "PERSON" and e.get("canonical_name") in names:
                cid = e.get("canonical_id")
                if isinstance(cid, list):
                    cid = cid[0] if cid else None
                rescored.add((aid, cid))
    return rescored, text_by_id, ents_by_id


def ccc(x, y):
    """Concordance correlation (scale-sensitive, unlike Pearson)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    vx, vy = x.var(), y.var()
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    denom = vx + vy + (x.mean() - y.mean()) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def corr_block(g, p, label, seed=0):
    g, p = np.asarray(g, float), np.asarray(p, float)
    pr, sr = pearson(g, p), spearman(g, p)
    return {
        "label": label, "n": int(len(g)),
        "pearson": pr, "pearson_ci": boot_ci(g, p, pearson, seed=seed),
        "spearman": sr, "spearman_ci": boot_ci(g, p, spearman, seed=seed + 1),
        "ccc": ccc(p, g),
        "std_ratio": round(float(p.std() / g.std()), 4) if g.std() else float("nan"),
        "mae": float(np.abs(g - p).mean()) if len(g) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="glob for e2e_predictions_*.jsonl")
    ap.add_argument("--final", default="data/labeled/deepseek_t1/labels_10ticker.final.jsonl")
    ap.add_argument("--person-scores", default="data/labeled/deepseek_t1/person_scores.jsonl")
    ap.add_argument("--output-dir", default="outputs/eval_new_dataset")
    ap.add_argument("--top-companies", type=int, default=15)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, iou_matched, n_pred_spans, n_unmatched_pred, files = load_predictions(args.predictions)
    if not rows:
        raise SystemExit("No per-entity results found in predictions.")
    person_info = load_person_rescored(Path(args.final), Path(args.person_scores))
    rescored_set, text_by_id, ents_by_id = (person_info if person_info else (None, {}, {}))

    aid = np.array([r[0] for r in rows])
    cid = np.array([r[1] for r in rows], dtype=object)
    typ = np.array([r[2] for r in rows], dtype=object)
    gold = np.array([r[3] for r in rows], float)
    pred = np.array([r[4] for r in rows], float)

    report = {"n_files": len(files), "n_covered_entities": len(rows)}

    # ---- overall + per-type ----
    report["overall"] = corr_block(gold, pred, "overall")
    report["per_type"] = []
    for t in ["ORG", "TICKER", "PERSON", "COMPANY"]:
        m = typ == t
        if m.sum() >= 5:
            report["per_type"].append(corr_block(gold[m], pred[m], t))

    # ---- per-company (most frequent canonical_ids) ----
    counts = defaultdict(int)
    for c in cid:
        counts[c] += 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:args.top_companies]
    report["per_company"] = []
    for c, n in top:
        if n < 5:
            continue
        m = cid == c
        report["per_company"].append(corr_block(gold[m], pred[m], str(c)))

    # ---- trivial baseline: predict the global mean ----
    mean_pred = float(gold.mean())
    report["baseline"] = {
        "predict_mean_value": mean_pred,
        "predict_mean_mae": float(np.abs(gold - mean_pred).mean()),
        "model_mae": float(np.abs(gold - pred).mean()),
        "note": "a constant predictor has Pearson r undefined (0); any positive r beats it. "
                "MAE comparison shows whether the model beats predicting the mean.",
    }

    # ---- over-neutrality (the lawsuit/probe blind spot) ----
    strong = np.abs(gold) >= 0.4
    model_flat = np.abs(pred) < 0.1
    report["over_neutrality"] = {
        "n_gold_strong": int(strong.sum()),
        "n_model_flat_on_strong": int((strong & model_flat).sum()),
        "rate": float((strong & model_flat).mean() if strong.any() else 0.0)
                if strong.sum() else 0.0,
        "rate_among_strong": float((model_flat[strong]).mean()) if strong.any() else float("nan"),
    }

    # ---- sign flips ----
    both_signed = (np.abs(gold) >= 0.1) & (np.abs(pred) >= 0.1)
    flip = both_signed & (np.sign(gold) != np.sign(pred))
    report["sign_flip"] = {
        "n_both_signed": int(both_signed.sum()),
        "n_flips": int(flip.sum()),
        "rate_among_both_signed": float(flip[both_signed].mean()) if both_signed.any() else float("nan"),
    }

    # ---- error by gold magnitude ----
    mag_bins = [(0.0, 0.2, "incidental"), (0.2, 0.4, "mild"), (0.4, 0.6, "moderate"),
                (0.6, 1.01, "strong")]
    report["error_by_magnitude"] = []
    ag = np.abs(gold)
    for lo, hi, name in mag_bins:
        m = (ag >= lo) & (ag < hi)
        if m.sum() == 0:
            continue
        report["error_by_magnitude"].append({
            "band": name, "range": [lo, hi], "n": int(m.sum()),
            "mae": float(np.abs(gold[m] - pred[m]).mean()),
            "mean_gold": float(gold[m].mean()), "mean_pred": float(pred[m].mean()),
        })

    # ---- bucket confusion ----
    gb = [bucket_of(s) for s in gold]
    pb = [bucket_of(s) for s in pred]
    names = [n for n, _, _ in BUCKETS]
    conf = {gn: {pn: 0 for pn in names} for gn in names}
    for a, b in zip(gb, pb):
        conf[a][b] += 1
    exact = sum(conf[n][n] for n in names) / len(gb)
    report["bucket_confusion"] = {"matrix": conf, "exact_acc": float(exact)}

    # extreme-bucket precision/recall/F1 (recall alone is gameable — Codex)
    report["extreme_bucket_f1"] = {}
    for ex in ("very_neg", "very_pos"):
        tp = conf[ex][ex]
        fp = sum(conf[g_][ex] for g_ in names) - tp
        fn = sum(conf[ex][p_] for p_ in names) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        report["extreme_bucket_f1"][ex] = {
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "n_gold": tp + fn,
        }

    # ---- IoU distribution + span contamination ----
    if len(iou_matched):
        report["iou_distribution"] = {
            "n_matched": int(len(iou_matched)),
            "mean": float(iou_matched.mean()),
            "pct_iou_lt_0.7": float((iou_matched < 0.7).mean()),
            "pct_iou_eq_1.0": float((iou_matched >= 0.999).mean()),
            "p10": float(np.percentile(iou_matched, 10)),
            "p50": float(np.percentile(iou_matched, 50)),
        }
    report["span_coverage"] = {
        "n_pred_sentiment_spans": n_pred_spans,
        "n_unmatched_pred_spans": n_unmatched_pred,
        "pct_pred_unmatched": float(n_unmatched_pred / n_pred_spans) if n_pred_spans else 0.0,
    }

    # ---- PERSON rescored vs not ----
    if rescored_set is not None:
        pm = typ == "PERSON"
        is_res = np.array([(aid[i], cid[i]) in rescored_set for i in range(len(rows))])
        for sub, lbl in [(pm & is_res, "PERSON_kimi_rescored"), (pm & ~is_res, "PERSON_not_rescored")]:
            if sub.sum() >= 5:
                report.setdefault("person_split", []).append(corr_block(gold[sub], pred[sub], lbl))

    # ---- concrete examples (over-neutral + sign flips) ----
    def examples(mask, n):
        out_ex = []
        idxs = np.where(mask)[0]
        for i in idxs[: n]:
            a = aid[i]
            ents = ents_by_id.get(a, [])
            txt = text_by_id.get(a, "")
            out_ex.append({
                "id": str(a), "canonical_id": str(cid[i]), "type": str(typ[i]),
                "gold": round(float(gold[i]), 3), "pred": round(float(pred[i]), 3),
                "text_head": txt[:240].replace("\n", " "),
            })
        return out_ex
    report["examples_over_neutral"] = examples(strong & model_flat, args.examples)
    report["examples_sign_flip"] = examples(flip, args.examples)

    # ---- calibration table (+ optional plot) ----
    edges = np.linspace(-1, 1, 11)
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gold >= lo) & (gold < hi) if hi < 1.0 else (gold >= lo) & (gold <= hi)
        if m.sum() == 0:
            continue
        calib.append({"gold_bin": [round(lo, 1), round(hi, 1)], "n": int(m.sum()),
                      "mean_gold": float(gold[m].mean()), "mean_pred": float(pred[m].mean())})
    report["calibration"] = calib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gx = [c["mean_gold"] for c in calib]
        py = [c["mean_pred"] for c in calib]
        plt.figure(figsize=(5, 5))
        plt.plot([-1, 1], [-1, 1], "k--", alpha=0.4, label="ideal")
        plt.plot(gx, py, "o-", label="model")
        plt.xlabel("mean gold sentiment (new labels)"); plt.ylabel("mean predicted")
        plt.title("Calibration: model vs new labels"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(out / "calibration.png", dpi=120)
        plt.close()
        report["calibration_plot"] = str(out / "calibration.png")
    except Exception as e:
        report["calibration_plot"] = f"(matplotlib unavailable: {e})"

    # ---- write JSON + markdown ----
    (out / "gap_analysis.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, out / "gap_analysis.md")
    print(f"Wrote {out/'gap_analysis.json'} and {out/'gap_analysis.md'}")
    print(f"Overall: Pearson {fmt_ci(report['overall']['pearson'], report['overall']['pearson_ci'])}  "
          f"Spearman {fmt_ci(report['overall']['spearman'], report['overall']['spearman_ci'])}  "
          f"N={report['overall']['n']}")


def write_markdown(rep, path):
    L = ["# Gap Analysis — current model vs new DeepSeek+Kimi labels\n",
         f"Covered entities: **{rep['n_covered_entities']:,}** "
         f"(from {rep['n_files']} prediction file(s))\n",
         "## Headline correlations\n",
         "| scope | N | Pearson r [95% CI] | Spearman ρ [95% CI] | MAE |",
         "|---|---:|---|---|---:|"]
    o = rep["overall"]
    L.append(f"| **overall** | {o['n']:,} | {fmt_ci(o['pearson'], o['pearson_ci'])} | "
             f"{fmt_ci(o['spearman'], o['spearman_ci'])} | {o['mae']:.3f} |")
    for b in rep.get("per_type", []):
        L.append(f"| {b['label']} | {b['n']:,} | {fmt_ci(b['pearson'], b['pearson_ci'])} | "
                 f"{fmt_ci(b['spearman'], b['spearman_ci'])} | {b['mae']:.3f} |")
    bl = rep["baseline"]
    L += ["\n## Trivial baseline",
          f"- predict-the-mean ({bl['predict_mean_value']:+.3f}) MAE = **{bl['predict_mean_mae']:.3f}**; "
          f"model MAE = **{bl['model_mae']:.3f}** → model {'beats' if bl['model_mae']<bl['predict_mean_mae'] else 'does NOT beat'} the mean predictor.",
          f"- {bl['note']}"]
    on = rep["over_neutrality"]; sf = rep["sign_flip"]
    L += ["\n## Systematic gaps",
          f"- **Over-neutrality**: of {on['n_gold_strong']:,} entities with gold |s|≥0.4, "
          f"the model is ~flat (|pred|<0.1) on **{on['n_model_flat_on_strong']:,}** "
          f"(**{on['rate_among_strong']*100:.1f}%**) — the lawsuit/probe blind spot.",
          f"- **Sign flips**: among {sf['n_both_signed']:,} both-signed entities, "
          f"**{sf['n_flips']:,}** flip sign (**{sf['rate_among_both_signed']*100:.1f}%**)."]
    L += ["\n### Error by gold magnitude", "| band | N | MAE | mean gold | mean pred |",
          "|---|---:|---:|---:|---:|"]
    for e in rep.get("error_by_magnitude", []):
        L.append(f"| {e['band']} | {e['n']:,} | {e['mae']:.3f} | {e['mean_gold']:+.3f} | {e['mean_pred']:+.3f} |")
    iou = rep.get("iou_distribution")
    L += ["\n## Span contamination check (DeepSeek spans, not human-gold)"]
    if iou:
        L.append(f"- matched-span IoU: mean {iou['mean']:.3f}, median {iou['p50']:.3f}; "
                 f"{iou['pct_iou_lt_0.7']*100:.0f}% of matches have IoU<0.7 "
                 f"(partial-overlap → possibly different sentiment target).")
    sc = rep["span_coverage"]
    L.append(f"- {sc['n_unmatched_pred_spans']:,}/{sc['n_pred_sentiment_spans']:,} "
             f"({sc['pct_pred_unmatched']*100:.0f}%) predicted sentiment-spans matched no gold entity.")
    ps = rep.get("person_split")
    if ps:
        L += ["\n## PERSON: Kimi-rescored vs not", "| subset | N | Pearson r | Spearman ρ |",
              "|---|---:|---|---|"]
        for b in ps:
            L.append(f"| {b['label']} | {b['n']:,} | {fmt_ci(b['pearson'], b['pearson_ci'])} | "
                     f"{fmt_ci(b['spearman'], b['spearman_ci'])} |")
    bc = rep["bucket_confusion"]
    names = [n for n, _, _ in BUCKETS]
    L += [f"\n## Bucket confusion (exact acc {bc['exact_acc']*100:.1f}%)",
          "| gold ↓ / pred → | " + " | ".join(names) + " |",
          "|---|" + "---|" * len(names)]
    for gn in names:
        L.append(f"| {gn} | " + " | ".join(str(bc['matrix'][gn][pn]) for pn in names) + " |")
    if rep.get("per_company"):
        L += ["\n## Per-company (top by frequency)", "| entity | N | Pearson r [CI] |", "|---|---:|---|"]
        for b in rep["per_company"]:
            L.append(f"| {b['label']} | {b['n']:,} | {fmt_ci(b['pearson'], b['pearson_ci'])} |")
    for title, key in [("Over-neutral examples", "examples_over_neutral"),
                       ("Sign-flip examples", "examples_sign_flip")]:
        exs = rep.get(key, [])
        if exs:
            L.append(f"\n### {title}")
            for e in exs:
                L.append(f"- **{e['canonical_id']}** ({e['type']}) gold={e['gold']} pred={e['pred']} "
                         f"— _{e['text_head']}…_")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

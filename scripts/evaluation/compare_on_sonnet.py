#!/usr/bin/env python3
"""Compare a retrained checkpoint vs v2.0 on the Sonnet holdout (gold-grade NER + sentiment).

The per-arm test split uses DeepSeek-extracted spans, so its NER F1 is only a
coverage diagnostic. The Sonnet holdout (data/labeled/final/holdout_relabeled.jsonl)
carries gold-grade NER + sentiment and is the exact benchmark v2.0's published
numbers came from — so it's the apples-to-apples way to confirm whether armC's
encoder fine-tuning actually degraded NER (catastrophic forgetting) and whether
the sentiment gain holds cross-distribution.

Reads two evaluate_e2e_pipeline.py metrics JSONs (same schema: ner_metrics,
sentiment_on_covered, per_type_sentiment, coverage):
  --v20    default trained_model/v2.0_20260517/evaluation_results_e2e.json
  --arm    the new run's e2e_metrics_*.json (produced by running the e2e eval on
           the Sonnet holdout with the armC checkpoint)

Usage:
  python3 scripts/evaluation/compare_on_sonnet.py \
      --arm outputs/retrain_armC__v1_sonnet/e2e_metrics_*.json
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
V20_DEFAULT = PROJECT / "trained_model/v2.0_20260517/evaluation_results_e2e.json"


def load(p):
    files = sorted(glob.glob(str(p)))
    if not files:
        raise SystemExit(f"no metrics file matches: {p}")
    return json.load(open(files[-1]))


def row(label, m):
    ner = m.get("ner_metrics", {})
    st = ner.get("overall_sentiment_types", {})
    pt = ner.get("per_type", {})
    soc = m.get("sentiment_on_covered", {})
    cov = m.get("coverage", {})
    psent = m.get("per_type_sentiment", {})
    return {
        "label": label,
        "ner_sent_f1": st.get("f1"), "ner_sent_p": st.get("precision"), "ner_sent_r": st.get("recall"),
        "ner_org_f1": pt.get("ORG", {}).get("f1"),
        "ner_ticker_f1": pt.get("TICKER", {}).get("f1"),
        "ner_person_f1": pt.get("PERSON", {}).get("f1"),
        "cov": cov.get("coverage_rate"),
        "sent_r": soc.get("pearson_r"), "sent_mae": soc.get("mae"),
        "sent_pred_std": soc.get("pred_std"), "sent_gold_std": soc.get("gold_std"),
        "sent_org_r": psent.get("ORG", {}).get("pearson_r"),
        "sent_ticker_r": psent.get("TICKER", {}).get("pearson_r"),
        "sent_person_r": psent.get("PERSON", {}).get("pearson_r"),
    }


def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20", default=str(V20_DEFAULT))
    ap.add_argument("--arm", required=True, help="glob for the retrained run's e2e_metrics_*.json on the Sonnet holdout")
    ap.add_argument("--arm-label", default="armC")
    ap.add_argument("--output", default=str(PROJECT / "outputs/sonnet_crosscheck.md"))
    args = ap.parse_args()

    a = row("v2.0", load(args.v20))
    b = row(args.arm_label, load(args.arm))

    def delta(k):
        if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
            d = b[k] - a[k]
            return f"{d:+.3f}"
        return "—"

    groups = [
        ("NER (gold spans) — degradation check", [
            ("sent-types F1", "ner_sent_f1"), ("  precision", "ner_sent_p"), ("  recall", "ner_sent_r"),
            ("ORG F1", "ner_org_f1"), ("TICKER F1", "ner_ticker_f1"), ("PERSON F1", "ner_person_f1")]),
        ("Sentiment (cross-distribution)", [
            ("coverage", "cov"), ("Pearson r", "sent_r"), ("MAE", "sent_mae"),
            ("pred_std", "sent_pred_std"), ("gold_std", "sent_gold_std"),
            ("ORG r", "sent_org_r"), ("TICKER r", "sent_ticker_r"), ("PERSON r", "sent_person_r")]),
    ]
    lines = ["# Sonnet-holdout cross-check — {} vs v2.0".format(args.arm_label), "",
             "Gold-grade NER + sentiment, same benchmark as v2.0's published numbers.",
             "NER drop here = real catastrophic forgetting from encoder fine-tuning.", ""]
    print("\n".join(lines))
    for title, rows in groups:
        hdr = f"\n## {title}\n\n| metric | v2.0 | {args.arm_label} | Δ |\n|---|---|---|---|"
        print(hdr); lines.append(hdr)
        for name, k in rows:
            line = f"| {name} | {f(a[k])} | {f(b[k])} | {delta(k)} |"
            print(line); lines.append(line)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Label-noise audit: surface high-confidence model<->label disagreements for eyeballing.

Codex's guardrail before promoting: when the trained model is CONFIDENT and disagrees
strongly with the gold label, is it a MODEL error or a LABEL error? Many label errors =
the training set is noisy and the 0.643 partly reflects fitting that noise.

Reads control's e2e predictions on the test split (per_entity_results: gold + pred per
covered entity) + the source jsonl for article text. Buckets the worst disagreements:
  - SIGN FLIP   : |gold|>=0.3 and |pred|>=0.3 and opposite signs
  - MODEL_HOT   : |pred|>=0.5 but gold ~0 (model sees sentiment the label calls incidental)
  - MODEL_FLAT  : |gold|>=0.5 but |pred|<0.1 (model misses sentiment the label has)
  - BIG_GAP     : |pred-gold|>=0.6 (any direction)
Samples N per bucket with a text window around the entity for adjudication.

Usage:
  python3 scripts/evaluation/audit_disagreements.py \
    --predictions outputs/retrain_control/e2e_predictions_*.jsonl \
    --source data/labeled/deepseek_t1/splits/test.jsonl --n 6
"""
from __future__ import annotations
import argparse, glob, json, random


def load_source(path):
    by_id = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        by_id[r["id"]] = r
    return by_id


def entity_window(rec, canonical_id, width=260):
    """A text snippet around the entity's first mention (for context)."""
    text = rec.get("text", "")
    for e in rec.get("entities", []):
        cid = e.get("canonical_id")
        cid = cid[0] if isinstance(cid, list) else cid
        if cid != canonical_id:
            continue
        ms = e.get("ner_mentions") or e.get("coref_mentions") or []
        if ms and isinstance(ms[0].get("start_char"), int):
            s = ms[0]["start_char"]
            lo = max(0, s - width // 2)
            return ("…" if lo else "") + text[lo:lo + width].replace("\n", " ") + "…"
    return text[:width].replace("\n", " ") + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=6, help="samples per bucket")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--output", default="outputs/control_label_audit.md")
    args = ap.parse_args()

    src = load_source(args.source)
    buckets = {"SIGN_FLIP": [], "MODEL_HOT": [], "MODEL_FLAT": [], "BIG_GAP": []}
    n_entities = 0
    for fp in sorted(glob.glob(args.predictions)):
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            rec = src.get(r["id"])
            if rec is None:
                continue
            for e in r.get("per_entity_results", []):
                n_entities += 1
                g, p = float(e["gold_sentiment"]), float(e["agg_pred_sentiment"])
                cid, typ = e.get("canonical_id"), e.get("type")
                item = {"id": r["id"], "cid": cid, "type": typ, "gold": round(g, 2),
                        "pred": round(p, 2), "gap": round(p - g, 2)}
                if abs(g) >= 0.3 and abs(p) >= 0.3 and (g > 0) != (p > 0):
                    buckets["SIGN_FLIP"].append(item)
                elif abs(p) >= 0.5 and abs(g) < 0.1:
                    buckets["MODEL_HOT"].append(item)
                elif abs(g) >= 0.5 and abs(p) < 0.1:
                    buckets["MODEL_FLAT"].append(item)
                elif abs(p - g) >= 0.6:
                    buckets["BIG_GAP"].append(item)

    rng = random.Random(args.seed)
    L = ["# Label-noise audit — control vs gold (DeepSeek/Kimi) on the test split", "",
         f"Covered entities: {n_entities:,}. Buckets are the worst disagreements; "
         "adjudicate each: is the **model** wrong (model limit) or the **label** wrong (training noise)?", ""]
    desc = {
        "SIGN_FLIP": "opposite signs, both strong",
        "MODEL_HOT": "model strong, label ~0 (model sees stake the label calls incidental)",
        "MODEL_FLAT": "label strong, model ~0 (model misses sentiment the label has)",
        "BIG_GAP": "|pred-gold| >= 0.6",
    }
    for b, items in buckets.items():
        L.append(f"\n## {b} — {desc[b]}  ({len(items):,} total, {100*len(items)/max(n_entities,1):.1f}%)")
        for it in rng.sample(items, min(args.n, len(items))):
            win = entity_window(src[it["id"]], it["cid"])
            L.append(f"- **{it['cid']}** ({it['type']}) gold={it['gold']} pred={it['pred']} "
                     f"gap={it['gap']}  [{it['id']}]\n  _{win}_")
    summary = {b: len(v) for b, v in buckets.items()}
    L.insert(3, f"**Disagreement rates:** " +
             ", ".join(f"{b} {100*n/max(n_entities,1):.1f}%" for b, n in summary.items()) + "\n")
    open(args.output, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

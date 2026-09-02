#!/usr/bin/env python3
"""Overlay Kimi person scores onto the cleaned labels -> final training set.

Reads:
  data/labeled/deepseek_t1/labels_10ticker.cleaned.jsonl   (points 1+2 applied)
  data/labeled/deepseek_t1/person_scores.jsonl             (Kimi point-3 rescore)
Writes:
  data/labeled/deepseek_t1/labels_10ticker.final.jsonl

For every PERSON entity, if Kimi returned a score for that canonical_name in that
article, replace sentiment_score with the Kimi value. ORG/TICKER untouched.
Persons in articles Kimi rejected (content-policy) or didn't cover keep their
existing DeepSeek score. Idempotent; original files untouched.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CLEAN = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.cleaned.jsonl"
SCORES = PROJECT / "data/labeled/deepseek_t1/person_scores.jsonl"
FINAL = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.final.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, default=CLEAN)
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--output", type=Path, default=FINAL)
    args = ap.parse_args()

    by_id: dict[str, dict] = {}
    if args.scores.exists():
        for line in args.scores.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            by_id[r["id"]] = r.get("scores", {})
    print(f"loaded Kimi scores for {len(by_id)} articles")

    n_art = n_persons = n_updated = 0
    with args.clean.open(encoding="utf-8") as f, args.output.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            n_art += 1
            r = json.loads(line)
            scores = by_id.get(r["id"], {})
            for e in r.get("entities", []):
                if e.get("type") != "PERSON":
                    continue
                n_persons += 1
                name = e.get("canonical_name")
                if name in scores:
                    new = scores[name]
                    if e.get("sentiment_score") != new:
                        n_updated += 1
                    e["sentiment_score"] = new
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"articles: {n_art}  person entities: {n_persons}  scores updated: {n_updated}")
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()

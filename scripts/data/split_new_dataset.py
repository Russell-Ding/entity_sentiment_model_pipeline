#!/usr/bin/env python3
"""Split the new decisive dataset into train/val/test for the second-phase retrain.

Splits `labels_10ticker.final.jsonl` (30,829 articles) BY ARTICLE ID so no entity
leaks across splits, with near-duplicate removal first (Bloomberg reruns share
near-identical text under distinct ids — Codex flagged this leakage path).

  - dedup: drop articles whose normalized text-prefix hash already appeared
           (keeps the first occurrence).
  - split: deterministic md5(id) hash → train/val/test ≈ 90/5/5.

Outputs to data/labeled/deepseek_t1/splits/{train,val,test}.jsonl and verifies
zero id + text-prefix overlap across the three splits.

Usage:
  python3 scripts/data/split_new_dataset.py
  python3 scripts/data/split_new_dataset.py --val-frac 0.05 --test-frac 0.05
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT / "data/labeled/deepseek_t1/labels_10ticker.final.jsonl"
DEFAULT_OUT = PROJECT / "data/labeled/deepseek_t1/splits"


def text_key(text: str) -> str:
    """Normalized prefix hash for near-duplicate detection."""
    norm = " ".join((text or "")[:500].lower().split())
    return hashlib.md5(norm.encode()).hexdigest()


def bucket(article_id: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(str(article_id).encode()).hexdigest(), 16) % 10000
    if h < test_frac * 10000:
        return "test"
    if h < (test_frac + val_frac) * 10000:
        return "val"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seen_text: set[str] = set()
    n_in = n_dup = 0
    writers = {s: (args.out_dir / f"{s}.jsonl").open("w", encoding="utf-8")
               for s in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0}
    ids = {"train": set(), "val": set(), "test": set()}
    tkeys = {"train": set(), "val": set(), "test": set()}

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            tk = text_key(r.get("text", ""))
            if tk in seen_text:        # near-duplicate → drop
                n_dup += 1
                continue
            seen_text.add(tk)
            s = bucket(r["id"], args.val_frac, args.test_frac)
            writers[s].write(line if line.endswith("\n") else line + "\n")
            counts[s] += 1
            ids[s].add(r["id"])
            tkeys[s].add(tk)

    for w in writers.values():
        w.close()

    # Verify disjointness
    id_overlap = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
    tk_overlap = (tkeys["train"] & tkeys["val"]) | (tkeys["train"] & tkeys["test"]) | (tkeys["val"] & tkeys["test"])

    print(f"input articles      : {n_in}")
    print(f"near-dup dropped     : {n_dup}")
    print(f"kept                 : {sum(counts.values())}")
    for s in ("train", "val", "test"):
        print(f"  {s:5}: {counts[s]:6,}  -> {args.out_dir / (s + '.jsonl')}")
    print(f"id overlap across splits   : {len(id_overlap)}")
    print(f"text-key overlap across splits: {len(tk_overlap)}")
    assert not id_overlap and not tk_overlap, "LEAKAGE across splits!"
    print("OK — splits are disjoint by id and text-prefix.")


if __name__ == "__main__":
    main()

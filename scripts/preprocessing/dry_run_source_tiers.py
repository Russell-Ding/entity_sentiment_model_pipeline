#!/usr/bin/env python3
"""1% dry-run of source rules on the full bulk corpus.

Streams every <TICKER>.jsonl.gz, samples articles at the given rate,
applies rules in-memory, and reports:
- Per-rule hit count + fraction
- "no_rule" fallback rate
- Final tier distribution
- Comparison vs the 5K Kimi sample (to detect corpus drift)

Does NOT write any output files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("dry_run")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing"))

from apply_source_tiers import load_rules, apply_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="1% dry-run of source rules")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rules-file", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=0.01,
                        help="Fraction of articles to sample (default 0.01 = 1%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kimi-labels", type=Path,
                        default=PROJECT_ROOT / "outputs" / "source_classification_kimi_v2.jsonl",
                        help="Kimi-labeled sample for drift comparison")
    args = parser.parse_args()

    random.seed(args.seed)
    rules = load_rules(args.rules_file)
    logger.info(f"Loaded {len(rules)} rules")

    files = sorted(args.input_dir.glob("*.jsonl.gz"))
    logger.info(f"Streaming {len(files)} ticker files (sampling {args.sample_rate:.1%})")

    rule_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    content_type_counter: Counter = Counter()
    sampled = 0
    total = 0

    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                total += 1
                if random.random() >= args.sample_rate:
                    continue
                if not line.strip():
                    continue
                rec = json.loads(line)
                updates = apply_rules(rec, rules)
                rule_counter[updates["rule_name"] or "_no_rule"] += 1
                tier_counter[updates["final_source_tier"]] += 1
                content_type_counter[updates["content_type"]] += 1
                sampled += 1

    logger.info(f"Scanned {total:,} articles total, sampled {sampled:,} ({sampled/total*100:.3f}%)")
    print()
    print("=" * 78)
    print("RULE-MATCH DISTRIBUTION (sampled)")
    print("=" * 78)
    for name, count in sorted(rule_counter.items(), key=lambda x: -x[1]):
        pct = count / sampled * 100
        print(f"  {name:32s} {count:6d}  ({pct:5.2f}%)")

    print()
    print("=" * 78)
    print("FINAL TIER DISTRIBUTION (sampled)")
    print("=" * 78)
    for tier in sorted(tier_counter):
        count = tier_counter[tier]
        pct = count / sampled * 100
        print(f"  T{tier}: {count:6d}  ({pct:5.2f}%)")

    print()
    print("=" * 78)
    print("CONTENT_TYPE DISTRIBUTION (sampled)")
    print("=" * 78)
    for ct, count in sorted(content_type_counter.items(), key=lambda x: -x[1]):
        pct = count / sampled * 100
        print(f"  {ct:32s} {count:6d}  ({pct:5.2f}%)")

    # Drift check vs Kimi sample
    if args.kimi_labels.exists():
        print()
        print("=" * 78)
        print("DRIFT CHECK vs Kimi-labeled 5K sample")
        print("=" * 78)
        kimi_ct = Counter()
        kimi_total = 0
        with open(args.kimi_labels) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                cls = r.get("haiku_classification") or {}
                ct = cls.get("content_type")
                if ct:
                    kimi_ct[ct] += 1
                    kimi_total += 1
        print(f"  Kimi sample: {kimi_total} classified articles")
        print(f"  Dry-run sample: {sampled} articles")
        print()
        print(f"  {'category':32s} {'kimi%':>8s} {'dryrun%':>8s} {'drift':>8s}")
        print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
        # Union of categories
        all_cats = set(kimi_ct.keys()) | set(content_type_counter.keys())
        rows = []
        for ct in all_cats:
            kimi_pct = kimi_ct.get(ct, 0) / kimi_total * 100 if kimi_total else 0
            dry_pct = content_type_counter.get(ct, 0) / sampled * 100 if sampled else 0
            drift = dry_pct - kimi_pct
            rows.append((ct, kimi_pct, dry_pct, drift))
        # Sort by abs drift descending
        rows.sort(key=lambda r: -abs(r[3]))
        for ct, k, d, drift in rows:
            flag = " ⚠" if abs(drift) >= 5 else ""
            print(f"  {ct:32s} {k:7.2f}% {d:7.2f}% {drift:+7.2f}%{flag}")


if __name__ == "__main__":
    main()

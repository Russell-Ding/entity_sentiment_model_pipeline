#!/usr/bin/env python3
"""Build tier-specific training subsets from retiered article files.

Reads retiered <TICKER>.jsonl.gz files, applies quality filters, performs
global cross-ticker deduplication, optional per-ticker capping, optional
per-year capping, and emits training subset JSONL files.

Usage:
    python scripts/preprocessing/build_training_subsets.py \
        --input-dir data/raw/eodhd_bulk_20260517/news_retiered \
        --output-dir data/labeled/training_subsets \
        --min-content-length 200 \
        --exclude-truncated \
        --max-articles-per-ticker 7000
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("build_subsets")


def load_records(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all records from input directory."""
    records: List[Dict[str, Any]] = []
    for gz_path in sorted(input_dir.glob("*.jsonl.gz")):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_duplicate_flags(path: Path | None) -> Dict[str, bool]:
    """Load duplicate flags sidecar JSON."""
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept both {id: bool} and list of ids
    if isinstance(data, dict):
        return {k: bool(v) for k, v in data.items()}
    if isinstance(data, list):
        return {k: True for k in data}
    return {}


def passes_quality_filters(
    rec: Dict[str, Any],
    min_content_length: int,
    exclude_truncated: bool,
    max_symbols_count: int | None,
    min_title_length: int,
) -> bool:
    """Return True if record passes quality filters."""
    if rec.get("content_length", 0) < min_content_length:
        return False
    if exclude_truncated and rec.get("is_truncated", False):
        return False
    title = rec.get("title", "")
    if len(title) < min_title_length:
        return False
    if max_symbols_count is not None:
        symbols_count = rec.get("symbols_count", 0)
        if isinstance(symbols_count, int) and symbols_count > max_symbols_count:
            return False
    return True


def cap_per_ticker(
    records: List[Dict[str, Any]], max_per_ticker: int | None
) -> List[Dict[str, Any]]:
    """Uniformly sample records so no ticker exceeds max_per_ticker."""
    if max_per_ticker is None:
        return records
    by_ticker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_ticker[r.get("primary_ticker", "")].append(r)
    capped: List[Dict[str, Any]] = []
    for ticker, recs in by_ticker.items():
        if len(recs) > max_per_ticker:
            recs = random.sample(recs, max_per_ticker)
        capped.extend(recs)
    return capped


def apply_year_cap(
    records: List[Dict[str, Any]], max_per_year: int | None
) -> List[Dict[str, Any]]:
    """Uniformly sample records so no year exceeds max_per_year."""
    if max_per_year is None:
        return records
    by_year: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        date = r.get("date", "")
        year = date[:4] if isinstance(date, str) and len(date) >= 4 else "unknown"
        by_year[year].append(r)
    capped: List[Dict[str, Any]] = []
    for year, recs in by_year.items():
        if len(recs) > max_per_year:
            recs = random.sample(recs, max_per_year)
        capped.extend(recs)
    return capped


def log_year_distribution(records: List[Dict[str, Any]], label: str) -> None:
    """Log article counts per year."""
    year_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        date = r.get("date", "")
        year = date[:4] if isinstance(date, str) and len(date) >= 4 else "unknown"
        year_counts[year] += 1
    logger.info(f"Year distribution ({label}):")
    for year in sorted(year_counts):
        pct = year_counts[year] / len(records) * 100 if records else 0
        logger.info(f"  {year}: {year_counts[year]:,} ({pct:.2f}%)")


def build_subsets(
    records: List[Dict[str, Any]],
    output_dir: Path,
    min_content_length: int,
    exclude_truncated: bool,
    max_symbols_count: int | None,
    min_title_length: int,
    max_articles_per_ticker: int | None,
    max_per_year: int | None,
) -> None:
    """Build and write tier-specific training subsets."""
    # Quality filter
    quality_records = [
        r
        for r in records
        if passes_quality_filters(
            r, min_content_length, exclude_truncated, max_symbols_count, min_title_length
        )
    ]
    logger.info(
        f"Quality filter: {len(quality_records):,} / {len(records):,} passed "
        f"(min_length={min_content_length}, exclude_truncated={exclude_truncated}, "
        f"max_symbols_count={max_symbols_count}, min_title_length={min_title_length})"
    )
    log_year_distribution(quality_records, "after quality filter")

    # Per-ticker cap
    ticker_capped = cap_per_ticker(quality_records, max_articles_per_ticker)
    if max_articles_per_ticker is not None:
        logger.info(
            f"Per-ticker cap ({max_articles_per_ticker:,}): "
            f"{len(ticker_capped):,} / {len(quality_records):,} retained"
        )
        log_year_distribution(ticker_capped, "after per-ticker cap")

    # Per-year cap
    year_capped = apply_year_cap(ticker_capped, max_per_year)
    if max_per_year is not None:
        logger.info(
            f"Per-year cap ({max_per_year:,}): "
            f"{len(year_capped):,} / {len(ticker_capped):,} retained"
        )
        log_year_distribution(year_capped, "after per-year cap")

    # Cross-ticker deduplication by url_canonical
    seen_urls: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for rec in year_capped:
        url = rec.get("url_canonical", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(rec)
    logger.info(f"Cross-ticker dedup: {len(deduped):,} / {len(year_capped):,} unique")
    if duplicate_flags and not args.exclude_duplicates:
        flagged_deduped = sum(1 for r in deduped if duplicate_flags.get(r.get("id", ""), False))
        logger.info(f"  Flagged duplicates within deduped set: {flagged_deduped:,}")
    log_year_distribution(deduped, "after dedup")

    # Tier-based subsets
    tier1 = [r for r in deduped if r.get("final_source_tier", 3) == 1]
    tier1_plus_2 = [r for r in deduped if r.get("final_source_tier", 3) <= 2]
    all_tiers = deduped

    subsets = {
        "train_tier1.jsonl": tier1,
        "train_tier1_plus_2.jsonl": tier1_plus_2,
        "train_all_tiers.jsonl": all_tiers,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, subset_records in subsets.items():
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in subset_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(subset_records):,} records to {out_path}")

    # Source breakdown for logging
    source_counts: Dict[str, int] = defaultdict(int)
    for rec in deduped:
        source_counts[rec.get("detected_source", "unknown")] += 1

    logger.info("Source distribution in final deduped corpus:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
        pct = count / len(deduped) * 100 if deduped else 0
        logger.info(f"  {source}: {count:,} ({pct:.2f}%)")

    # Tier distribution
    tier_counts: Dict[int, int] = defaultdict(int)
    for rec in deduped:
        tier_counts[rec.get("final_source_tier", 3)] += 1
    logger.info("Tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        pct = count / len(deduped) * 100 if deduped else 0
        logger.info(f"  Tier {tier}: {count:,} ({pct:.2f}%)")

    # Per-ticker distribution summary
    ticker_counts: Dict[str, int] = defaultdict(int)
    for rec in deduped:
        ticker_counts[rec.get("primary_ticker", "")] += 1
    counts = sorted(ticker_counts.values())
    if counts:
        logger.info("Per-ticker article count summary (after all filters):")
        logger.info(f"  Mean: {sum(counts) / len(counts):.0f}")
        logger.info(f"  Median: {counts[len(counts) // 2]:,}")
        logger.info(f"  Min: {counts[0]:,}")
        logger.info(f"  Max: {counts[-1]:,}")
        p90_idx = int(len(counts) * 0.9)
        logger.info(f"  p90: {counts[min(p90_idx, len(counts) - 1)]:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tier-specific training subsets")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with retiered <TICKER>.jsonl.gz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write subset JSONL files",
    )
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=200,
        help="Minimum content length in characters (default: 200)",
    )
    parser.add_argument(
        "--exclude-truncated",
        action="store_true",
        default=True,
        help="Exclude articles ending with [...truncated] (default: True)",
    )
    parser.add_argument(
        "--include-truncated",
        dest="exclude_truncated",
        action="store_false",
        help="Include truncated articles",
    )
    parser.add_argument(
        "--max-symbols-count",
        type=int,
        default=10,
        help="Exclude articles tagged with more than N symbols (default: 10, set to 0 to disable)",
    )
    parser.add_argument(
        "--min-title-length",
        type=int,
        default=5,
        help="Minimum title length in characters (default: 5)",
    )
    parser.add_argument(
        "--max-articles-per-ticker",
        type=int,
        default=7000,
        help="Maximum articles per ticker after quality filters (default: 7000, set to 0 to disable)",
    )
    parser.add_argument(
        "--max-per-year",
        type=int,
        default=None,
        help="Maximum articles per publication year (default: no cap)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--duplicate-flags",
        type=Path,
        default=None,
        help="Optional JSON sidecar from check_training_duplicates.py",
    )
    parser.add_argument(
        "--exclude-duplicates",
        action="store_true",
        default=False,
        help="Exclude articles flagged as likely training duplicates",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    random.seed(args.seed)

    max_symbols_count = args.max_symbols_count if args.max_symbols_count > 0 else None
    max_articles_per_ticker = (
        args.max_articles_per_ticker if args.max_articles_per_ticker > 0 else None
    )

    logger.info(f"Loading records from {args.input_dir}...")
    records = load_records(args.input_dir)
    logger.info(f"Loaded {len(records):,} total records")

    duplicate_flags = load_duplicate_flags(args.duplicate_flags)
    if duplicate_flags:
        flagged = sum(1 for r in records if duplicate_flags.get(r.get("id", ""), False))
        logger.info(f"Duplicate flags loaded: {flagged:,} / {len(records):,} flagged")
        if args.exclude_duplicates:
            before = len(records)
            records = [r for r in records if not duplicate_flags.get(r.get("id", ""), False)]
            logger.info(f"Excluded duplicates: {len(records):,} / {before:,} retained")

    log_year_distribution(records, "raw input")

    build_subsets(
        records,
        args.output_dir,
        args.min_content_length,
        args.exclude_truncated,
        max_symbols_count,
        args.min_title_length,
        max_articles_per_ticker,
        args.max_per_year,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Enrich an inference JSONL with ticker resolution and per-ticker aggregation.

Reads `outputs/inference/<TICKER>.t1_sentiment.jsonl` (one record per article,
each with an `entities` array of per-surface-form sentiment) and writes a new
JSONL with two additions per record:

  1. Each existing entity row gets a `ticker` field (string or null).
  2. A new `ticker_sentiments` array, one row per resolved ticker:
       {
         "ticker": "AAPL",
         "company": "Apple Inc.",
         "sentiment": 0.013,         # mention-count-weighted mean
         "sentiment_std": 0.07,      # weighted std (across contributing aliases)
         "n_aliases": 3,             # surface forms that resolved to this ticker
         "total_mentions": 15,       # sum of mentions across aliases
         "source_aliases": ["Apple", "Apple Inc.", "Apple Inc"]
       }

The per-article surface-form `entities` array is preserved unchanged (other than
the new `ticker` field), so you keep the granular signal AND get the
ticker-aggregated view.

Usage:
    python3 scripts/postprocessing/enrich_with_tickers.py \\
        --input outputs/inference/AAPL.US.t1_sentiment.jsonl \\
        --output outputs/inference/AAPL.US.t1_sentiment_enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "postprocessing"))
from ticker_alias_resolver import TickerAliasResolver  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enrich")


def aggregate_to_tickers(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse per-surface-form entities to per-ticker aggregated sentiment.

    Mention-count-weighted mean for sentiment so a 10-mention "Apple" outweighs
    a 1-mention "Apple Inc." in the per-ticker average. Returns one row per
    resolved ticker, sorted by total_mentions desc.
    """
    by_ticker: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "company": "",
        "sources": set(),  # set of resolver sources contributing (sp500/extras/etc.)
        "n_aliases": 0,
        "total_mentions": 0,
        "weighted_sum": 0.0,
        "weighted_count": 0,
        "weighted_sq_sum": 0.0,
        "source_aliases": [],
    })

    for e in entities:
        t = e.get("ticker")
        if not t:
            continue
        n = int(e.get("num_mentions", 0))
        if n <= 0:
            continue
        s = e.get("sentiment")
        agg = by_ticker[t]
        if not agg["company"]:
            agg["company"] = e.get("ticker_company", "")
        agg["n_aliases"] += 1
        agg["total_mentions"] += n
        agg["source_aliases"].append(e["canonical_id"])
        if s is not None:
            agg["weighted_sum"] += float(s) * n
            agg["weighted_count"] += n
            agg["weighted_sq_sum"] += float(s) ** 2 * n

    out = []
    for tk, agg in by_ticker.items():
        wc = agg["weighted_count"]
        if wc > 0:
            mean = agg["weighted_sum"] / wc
            var = max(0.0, agg["weighted_sq_sum"] / wc - mean ** 2)
            std = var ** 0.5
            mean_r: Optional[float] = round(mean, 4)
            std_r: Optional[float] = round(std, 4)
        else:
            mean_r = None
            std_r = None
        out.append({
            "ticker": tk,
            "company": agg["company"],
            "sentiment": mean_r,
            "sentiment_std": std_r,
            "n_aliases": agg["n_aliases"],
            "total_mentions": agg["total_mentions"],
            "source_aliases": agg["source_aliases"],
        })
    out.sort(key=lambda r: -r["total_mentions"])
    return out


def enrich_record(rec: Dict[str, Any], resolver: TickerAliasResolver) -> Dict[str, Any]:
    """Add `ticker` to each entity and append `ticker_sentiments` array.

    NOTE: Mutates `rec["entities"]` in place. For streaming JSONL usage this is
    fine; if calling as a library on existing objects, shallow-copy first.
    """
    entities = rec.get("entities", [])
    for e in entities:
        res = resolver.resolve(e.get("canonical_id", ""), e.get("entity_type", "ORG"))
        if res:
            e["ticker"] = res.ticker
            e["ticker_company"] = res.company
            e["ticker_source"] = res.source
        else:
            e["ticker"] = None
    ticker_rows = aggregate_to_tickers(entities)
    rec["ticker_sentiments"] = ticker_rows
    rec["n_tickers_resolved"] = len(ticker_rows)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Add ticker resolution + per-ticker aggregation to inference JSONL")
    parser.add_argument("--input", type=Path, required=True,
                        help="Inference JSONL (output of infer_entity_sentiment.py)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Enriched JSONL output path")
    parser.add_argument("--sp500-csv", type=Path, default=None,
                        help="Override SP500 seed CSV path")
    parser.add_argument("--extras-csv", type=Path, default=None,
                        help="Override extras CSV path")
    args = parser.parse_args()

    kwargs = {}
    if args.sp500_csv:    kwargs["sp500_csv"] = args.sp500_csv
    if args.extras_csv:   kwargs["extras_csv"] = args.extras_csv
    resolver = TickerAliasResolver(**kwargs)
    logger.info(f"Resolver: {len(resolver.ticker_set):,} tickers, "
                f"{len(resolver.alias_map):,} aliases, "
                f"{len(resolver.blocklist):,} blocklist")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_articles = 0
    n_entities_total = 0
    n_entities_resolved = 0
    n_tickers_total = 0
    n_articles_with_ticker = 0

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            enriched = enrich_record(rec, resolver)
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            n_articles += 1
            ents = enriched.get("entities", [])
            n_entities_total += len(ents)
            n_entities_resolved += sum(1 for e in ents if e.get("ticker"))
            tks = enriched.get("ticker_sentiments", [])
            n_tickers_total += len(tks)
            if tks:
                n_articles_with_ticker += 1

            if n_articles % 1000 == 0:
                logger.info(f"  {n_articles:,} articles processed")

    logger.info(f"Done. {n_articles:,} articles -> {args.output}")
    logger.info(f"  Entity rows resolved to ticker:        {n_entities_resolved:6,d} / {n_entities_total:,d} "
                f"({n_entities_resolved/max(n_entities_total,1)*100:5.1f}%)")
    logger.info(f"  Articles with >=1 resolved ticker:     {n_articles_with_ticker:6,d} / {n_articles:,d} "
                f"({n_articles_with_ticker/max(n_articles,1)*100:5.1f}%)")
    logger.info(f"  Total ticker_sentiment rows emitted:   {n_tickers_total:6,d}  "
                f"(avg {n_tickers_total/max(n_articles,1):.2f}/article)")


if __name__ == "__main__":
    main()

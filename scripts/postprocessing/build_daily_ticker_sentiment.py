"""Aggregate per-article ticker sentiment into per-(ticker, trade_date) daily series.

Inputs
------
outputs/inference/{TICKER}.t1_sentiment_enriched.jsonl
  Each line is one article with a `ticker_sentiments` list of
  {ticker, company, sentiment, sentiment_std, n_aliases, total_mentions, source_aliases}.

Outputs
-------
outputs/inference/daily_ticker_sentiment_long.jsonl
  One JSON line per (ticker, trade_date) with weighted mean/std and counts.
outputs/inference/daily_ticker_sentiment_wide.csv
  Wide pivot: rows = trade_date, cols = top-N tickers by article count, vals = sentiment_mean.

Dedup
-----
A ticker mentioned in multiple primary-ticker feeds (e.g. AAPL appears in both AAPL.US
and MSFT.US feeds for the same article_id) is counted ONCE per (ticker, article_id).
The first record encountered wins; we track which primary feeds contributed via
`n_primary_feeds`.

Weighting
---------
sentiment_mean is weighted by `total_mentions`. This matches how the per-article
ticker_sentiments are themselves built (mention-count weighted across aliases) and
gives heavily-discussed articles more pull. sentiment_std is the weighted population
standard deviation across the contributing article-level sentiment values.

Excluded sources (P0)
---------------------
MT Newswires T1 articles in this corpus are 100% paywall stubs (298-302 chars,
body is just a lead sentence + "PREMIUM Upgrade to read..." boilerplate). They
carry near-zero extractable body signal, so they are excluded from the daily
aggregation by default (--exclude-sources). The articles still exist in the raw
enriched files; this only drops them from the daily series.

Coverage-gap flag (P0)
----------------------
Each daily row carries `preceded_by_gap` / `gap_weekdays`. A gap is the number of
weekdays strictly between this observation and the prior observation for the same
ticker. Weekdays (not calendar days) are used so a normal Fri->Mon weekend does
not register as a gap. `preceded_by_gap` is True when gap_weekdays >= GAP_THRESHOLD.
This lets downstream know a sentiment reading follows a T1 coverage blackout
(e.g. NVDA had zero T1 coverage during its pivotal May 2023 earnings week).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "outputs" / "inference"

PRIMARY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "V",
]

LONG_OUT = INFERENCE_DIR / "daily_ticker_sentiment_long.jsonl"
WIDE_OUT = INFERENCE_DIR / "daily_ticker_sentiment_wide.csv"
TOP_N_WIDE = 20

# content_type values to drop from the daily series (paywall stubs / no body signal).
DEFAULT_EXCLUDED_SOURCES = {"mt_newswires"}
# A run of this many weekdays with no observation flags the next row as post-gap.
GAP_THRESHOLD = 3


def weekdays_between(d1: date, d2: date) -> int:
    """Count weekdays (Mon-Fri) strictly between d1 and d2 (exclusive of both).

    Fri -> Mon returns 0 (normal weekend). Fri -> Tue returns 1 (e.g. a Monday
    holiday). A multi-week blackout returns the full weekday count.
    """
    if d2 <= d1:
        return 0
    count = 0
    cur = d1.toordinal() + 1
    end = d2.toordinal()
    while cur < end:
        if date.fromordinal(cur).weekday() < 5:  # 0-4 = Mon-Fri
            count += 1
        cur += 1
    return count


def iter_enriched(path: Path, corrupt_counter: dict) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt lines but don't blow up the run.
                corrupt_counter[path.name] = corrupt_counter.get(path.name, 0) + 1
                continue


def weighted_mean_std(values: list[float], weights: list[float]) -> tuple[float, float]:
    """Mention-count weighted mean and population std.

    Returns (0.0, 0.0) when total weight is 0 (defensive; callers normally guard).
    For a single point we return (value, 0.0) — std is undefined with one sample.
    """
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0, 0.0
    mean = sum(v * w for v, w in zip(values, weights)) / total_w
    if len(values) == 1:
        return mean, 0.0
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return mean, math.sqrt(max(var, 0.0))


def build_daily(excluded_sources: set[str] | None = None,
                gap_threshold: int = GAP_THRESHOLD) -> tuple[list[dict], dict]:
    # key: (ticker, trade_date) -> list of contributions
    #   each contribution: {article_id, sentiment, weight, primary_feed, mentions}
    # dedup_index: (ticker, trade_date, article_id) -> index into by_key[...] list
    # (O(1) lookup; Kimi review flagged the prior O(n) scan as a perf nit.)
    excluded_sources = excluded_sources if excluded_sources is not None else set(DEFAULT_EXCLUDED_SOURCES)
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    dedup_index: dict[tuple[str, str, str], int] = {}

    files_read = 0
    files_missing: list[str] = []
    articles_seen = 0
    articles_excluded = 0
    corrupt_lines: dict[str, int] = {}
    skipped_nonfinite = 0
    skipped_unparseable = 0
    skipped_baddate = 0

    for primary in PRIMARY_TICKERS:
        path = INFERENCE_DIR / f"{primary}.US.t1_sentiment_enriched.jsonl"
        if not path.exists():
            files_missing.append(path.name)
            continue
        files_read += 1
        for art in iter_enriched(path, corrupt_lines):
            articles_seen += 1
            # Drop low-signal sources (paywall stubs) before they reach aggregation.
            content_type = (art.get("content_type") or "").strip().lower()
            if content_type in excluded_sources:
                articles_excluded += 1
                continue
            article_id = art.get("article_id")
            trade_date_raw = art.get("trade_date")
            if not article_id or not trade_date_raw:
                continue
            # Normalize date string at ingestion so the aggregation key is always
            # YYYY-MM-DD. This prevents the same logical day from splitting into
            # multiple rows if upstream formatting drifts.
            try:
                trade_date = datetime.strptime(str(trade_date_raw), "%Y-%m-%d").strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                skipped_baddate += 1
                continue
            primary_feed = art.get("primary_ticker", f"{primary}.US")
            for ts in art.get("ticker_sentiments", []) or []:
                ticker = ts.get("ticker")
                if not ticker:
                    continue
                # Normalize ticker (defensive — upstream is usually clean).
                ticker = str(ticker).strip().upper()
                if not ticker:
                    continue
                sentiment_raw = ts.get("sentiment")
                mentions_raw = ts.get("total_mentions") or 0
                if sentiment_raw is None or mentions_raw is None:
                    continue
                # Robust numeric coercion — never crash on a bad upstream record.
                try:
                    sentiment = float(sentiment_raw)
                    mentions = int(mentions_raw)
                except (TypeError, ValueError):
                    skipped_unparseable += 1
                    continue
                if mentions <= 0:
                    continue
                if not math.isfinite(sentiment):
                    # NaN/Inf would silently poison the daily mean & std.
                    skipped_nonfinite += 1
                    continue
                dedup_key = (ticker, trade_date, article_id)
                prior_idx = dedup_index.get(dedup_key)
                if prior_idx is not None:
                    # Same article seen via a different primary feed — just record
                    # that this feed also covered it. First-wins on sentiment/mentions
                    # (documented behavior; see module docstring).
                    by_key[(ticker, trade_date)][prior_idx]["primary_feeds"].add(primary_feed)
                    continue
                contribs = by_key[(ticker, trade_date)]
                dedup_index[dedup_key] = len(contribs)
                contribs.append({
                    "article_id": article_id,
                    "sentiment": sentiment,
                    "mentions": mentions,
                    "primary_feeds": {primary_feed},
                })

    rows: list[dict] = []
    for (ticker, trade_date), contribs in by_key.items():
        sentiments = [c["sentiment"] for c in contribs]
        weights = [float(c["mentions"]) for c in contribs]
        mean, std = weighted_mean_std(sentiments, weights)
        primary_feeds: set[str] = set()
        for c in contribs:
            primary_feeds.update(c["primary_feeds"])
        rows.append({
            "ticker": ticker,
            "trade_date": trade_date,
            "sentiment_mean": round(mean, 6),
            "sentiment_std": round(std, 6),
            "n_articles": len(contribs),
            "total_mentions": int(sum(weights)),
            "n_primary_feeds": len(primary_feeds),
            "primary_feeds": sorted(primary_feeds),
            # Per Kimi review: downstream cannot otherwise distinguish "only one
            # article" (std forced to 0) from a genuinely tight multi-article day.
            "single_article_day": len(contribs) == 1,
        })

    rows.sort(key=lambda r: (r["trade_date"], r["ticker"]))

    # Coverage-gap flag: per ticker, mark each row with the weekday gap since the
    # prior observation for that ticker. A gap >= threshold means the reading
    # follows a T1 coverage blackout (the prior data point is stale).
    n_flagged_gaps = 0
    prev_date: dict[str, date] = {}
    for r in rows:  # rows are date-sorted, so per-ticker order is chronological
        tk = r["ticker"]
        # All trade_dates were validated and normalized to YYYY-MM-DD at ingestion,
        # so parsing here is safe without a try/except.
        cur = datetime.strptime(r["trade_date"], "%Y-%m-%d").date()
        prior = prev_date.get(tk)
        if prior is None:
            r["gap_weekdays"] = None  # first observation for this ticker
            r["preceded_by_gap"] = False
        else:
            gap = weekdays_between(prior, cur)
            r["gap_weekdays"] = gap
            r["preceded_by_gap"] = gap >= gap_threshold
            if r["preceded_by_gap"]:
                n_flagged_gaps += 1
        prev_date[tk] = cur

    stats = {
        "files_read": files_read,
        "files_missing": files_missing,
        "articles_seen": articles_seen,
        "articles_excluded": articles_excluded,
        "excluded_sources": sorted(excluded_sources),
        "n_rows": len(rows),
        "n_flagged_gaps": n_flagged_gaps,
        "gap_threshold": gap_threshold,
        "corrupt_lines": corrupt_lines,
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_unparseable": skipped_unparseable,
        "skipped_baddate": skipped_baddate,
    }
    return rows, stats


def write_long(rows: list[dict]) -> None:
    LONG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with LONG_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_wide(rows: list[dict], top_n: int = TOP_N_WIDE) -> list[str]:
    # Rank tickers by total article count.
    article_count: dict[str, int] = defaultdict(int)
    mention_count: dict[str, int] = defaultdict(int)
    for r in rows:
        article_count[r["ticker"]] += r["n_articles"]
        mention_count[r["ticker"]] += r["total_mentions"]
    top_tickers = [
        t for t, _ in sorted(article_count.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    ]

    # date -> ticker -> sentiment_mean
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["ticker"] in top_tickers:
            by_date[r["trade_date"]][r["ticker"]] = r["sentiment_mean"]

    WIDE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with WIDE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + top_tickers)
        for date in sorted(by_date.keys()):
            row = [date]
            for t in top_tickers:
                v = by_date[date].get(t)
                row.append("" if v is None else f"{v:.6f}")
            writer.writerow(row)
    return top_tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-article ticker sentiment into a daily series")
    parser.add_argument("--exclude-sources", type=str, default=",".join(sorted(DEFAULT_EXCLUDED_SOURCES)),
                        help="Comma-separated content_type values to drop (paywall stubs). "
                             "Pass empty string to keep all sources.")
    parser.add_argument("--gap-threshold", type=int, default=GAP_THRESHOLD,
                        help="Weekday gap >= this flags a row as following a coverage blackout.")
    args = parser.parse_args()

    excluded = {s.strip().lower() for s in args.exclude_sources.split(",") if s.strip()}

    rows, stats = build_daily(excluded_sources=excluded, gap_threshold=args.gap_threshold)
    write_long(rows)
    top_tickers = write_wide(rows)

    # Summary
    print("=" * 70)
    print("Daily ticker sentiment aggregation")
    print("=" * 70)
    print(f"Files read       : {stats['files_read']} / {len(PRIMARY_TICKERS)}")
    if stats["files_missing"]:
        print(f"Files missing    : {', '.join(stats['files_missing'])}")
    print(f"Articles scanned : {stats['articles_seen']:,}")
    if stats["excluded_sources"]:
        print(f"Excluded sources : {', '.join(stats['excluded_sources'])}  "
              f"({stats['articles_excluded']:,} articles dropped)")
    print(f"(ticker, date)   : {stats['n_rows']:,} rows")
    print(f"Coverage gaps    : {stats['n_flagged_gaps']:,} rows flagged "
          f"(>= {stats['gap_threshold']} weekday blackout before the reading)")
    if stats["corrupt_lines"]:
        total_corrupt = sum(stats["corrupt_lines"].values())
        print(f"Corrupt lines    : {total_corrupt} ({stats['corrupt_lines']})")
    if stats["skipped_nonfinite"]:
        print(f"Skipped NaN/Inf  : {stats['skipped_nonfinite']:,}")
    if stats["skipped_unparseable"]:
        print(f"Skipped unparseable sentiment: {stats['skipped_unparseable']:,}")
    if stats["skipped_baddate"]:
        print(f"Skipped bad date : {stats['skipped_baddate']:,}")
    if rows:
        dates = [r["trade_date"] for r in rows]
        print(f"Date range       : {min(dates)}  ->  {max(dates)}")

    # Top 10 tickers by total mentions
    mentions: dict[str, int] = defaultdict(int)
    articles: dict[str, int] = defaultdict(int)
    for r in rows:
        mentions[r["ticker"]] += r["total_mentions"]
        articles[r["ticker"]] += r["n_articles"]
    top10 = sorted(mentions.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop 10 tickers by total mentions:")
    print(f"  {'ticker':<10}{'mentions':>12}{'articles':>12}")
    for t, m in top10:
        print(f"  {t:<10}{m:>12,}{articles[t]:>12,}")

    print(f"\nWide CSV top-{len(top_tickers)} tickers: {', '.join(top_tickers)}")
    print(f"\nWrote: {LONG_OUT}")
    print(f"Wrote: {WIDE_OUT}")


if __name__ == "__main__":
    main()

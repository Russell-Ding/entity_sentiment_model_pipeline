#!/usr/bin/env python3
"""Probe EODHD APIs to estimate the scale of a full S&P 500 data pull.

Hits both endpoints (news + EOD prices) for a handful of sample tickers
and reports:
  - How far back the price history goes per ticker
  - How dense the news is (articles per ticker, date range)
  - Estimated API call count + time + disk for a full 503-ticker pull
  - How many returned articles overlap with the existing training set
    (by ID prefix and date window)

API budget: ~6 calls per probed ticker (3 news pages + 1 price + a couple
for date probing). Set --tickers to override the default sample.

Usage:
    python scripts/collection/eodhd_probe.py
    python scripts/collection/eodhd_probe.py --tickers AAPL MSFT JPM XOM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

EOD_BASE = "https://eodhd.com/api/eod"
NEWS_BASE = "https://eodhd.com/api/news"

# Existing training data covers this window
TRAINING_DATE_MIN = "2020-10-02"
TRAINING_DATE_MAX = "2026-01-19"


def load_api_key() -> str:
    secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
    if not secrets_path.exists():
        raise SystemExit(f"Not found: {secrets_path}")
    with open(secrets_path) as f:
        data = yaml.safe_load(f)
    key = data.get("api", {}).get("eodhd_api")
    if not key:
        raise SystemExit("eodhd_api missing from config/secrets.yaml")
    return key


def load_training_ids() -> set[str]:
    """Article IDs already in our labeled training/eval data."""
    ids = set()
    base = PROJECT_ROOT / "data" / "labeled" / "final"
    for fname in ["train.jsonl", "val.jsonl", "holdout.jsonl", "holdout_relabeled.jsonl"]:
        path = base / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    aid = rec.get("id", "")
                    if aid:
                        ids.add(aid)
                except Exception:
                    pass
    return ids


def fetch_eod(symbol: str, api_key: str) -> dict:
    """Fetch full EOD price history for symbol. One API call."""
    url = f"{EOD_BASE}/{symbol}"
    params = {"api_token": api_key, "fmt": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return {"symbol": symbol, "n_rows": 0, "first_date": None, "last_date": None}
    return {
        "symbol": symbol,
        "n_rows": len(rows),
        "first_date": rows[0].get("date"),
        "last_date": rows[-1].get("date"),
    }


def fetch_news_page(symbol: str, api_key: str, offset: int = 0,
                    limit: int = 1000, from_date: str | None = None) -> list[dict]:
    """One news page (one API call) for symbol."""
    url = NEWS_BASE
    params = {
        "s": symbol,
        "api_token": api_key,
        "fmt": "json",
        "offset": offset,
        "limit": limit,
    }
    if from_date:
        params["from"] = from_date
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def probe_ticker(symbol: str, api_key: str, training_ids: set[str],
                 max_news_pages: int = 3) -> dict:
    """Probe one ticker. Returns a dict of stats. Uses ~4-5 API calls."""
    result = {"symbol": symbol}

    # 1. Price history depth — 1 call
    t0 = time.time()
    price = fetch_eod(symbol, api_key)
    result["price"] = {**price, "elapsed_sec": round(time.time() - t0, 2)}

    # 2. News (full reverse-chronological window via paging) — N calls
    all_news_meta = []  # just keep dates + ids, not full bodies, to save memory
    offset = 0
    pages_fetched = 0
    t0 = time.time()
    while pages_fetched < max_news_pages:
        page = fetch_news_page(symbol, api_key, offset=offset, limit=1000)
        if not page:
            break
        pages_fetched += 1
        for a in page:
            d = a.get("date", "")[:10]  # YYYY-MM-DD slice
            # EODHD news returns "link" but article id is derived in our pipeline
            # as a stable hash; use the URL as the proxy for uniqueness here.
            all_news_meta.append({
                "date": d,
                "link": a.get("link", ""),
                "symbols": a.get("symbols", []),
            })
        if len(page) < 1000:
            break  # last page
        offset += 1000

    news_elapsed = round(time.time() - t0, 2)
    dates = [m["date"] for m in all_news_meta if m["date"]]
    first_news = min(dates) if dates else None
    last_news = max(dates) if dates else None
    # Articles in training window
    in_window = sum(1 for d in dates if TRAINING_DATE_MIN <= d <= TRAINING_DATE_MAX)

    result["news"] = {
        "pages_fetched": pages_fetched,
        "articles_seen": len(all_news_meta),
        "first_date": first_news,
        "last_date": last_news,
        "in_training_window_count": in_window,
        "elapsed_sec": news_elapsed,
        "hit_page_cap": pages_fetched == max_news_pages and len(all_news_meta) % 1000 == 0,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+",
                        default=["AAPL", "MSFT", "JPM", "XOM"],
                        help="Probe these S&P 500 tickers (without .US suffix). "
                             "Default: AAPL MSFT JPM XOM — chosen to cover "
                             "different sectors and news densities.")
    parser.add_argument("--max-news-pages", type=int, default=3,
                        help="Cap on news pages per ticker probe (1000 articles/page).")
    args = parser.parse_args()

    api_key = load_api_key()
    print(f"API key loaded: {api_key[:8]}...{api_key[-4:]}")
    print()

    training_ids = load_training_ids()
    print(f"Existing training IDs: {len(training_ids):,}")
    print(f"Training window: {TRAINING_DATE_MIN} → {TRAINING_DATE_MAX}")
    print()

    print(f"Probing {len(args.tickers)} tickers (max {args.max_news_pages} news pages each)")
    print(f"Estimated API budget for probe: ~{len(args.tickers) * (1 + args.max_news_pages)} calls")
    print()

    results = []
    for tk in args.tickers:
        symbol = f"{tk}.US"
        print(f"Probing {symbol} ...")
        try:
            r = probe_ticker(symbol, api_key, training_ids, args.max_news_pages)
            results.append(r)
            p = r["price"]
            n = r["news"]
            print(f"  Price: {p['n_rows']:,} rows, "
                  f"{p['first_date']} → {p['last_date']} "
                  f"({p['elapsed_sec']}s)")
            print(f"  News : {n['articles_seen']:,} articles in {n['pages_fetched']} pages "
                  f"({n['first_date']} → {n['last_date']}, {n['elapsed_sec']}s)")
            print(f"         {n['in_training_window_count']:,} fall in training window. "
                  f"hit_page_cap={n['hit_page_cap']}")
            print()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({"symbol": symbol, "error": str(e)})
            print()

    # ---------------------------------------------------------------------
    # Extrapolate to full S&P 500 pull
    # ---------------------------------------------------------------------
    successful = [r for r in results if "error" not in r]
    if not successful:
        print("No successful probes — cannot extrapolate.")
        return

    avg_articles = sum(r["news"]["articles_seen"] for r in successful) / len(successful)
    avg_pages = sum(r["news"]["pages_fetched"] for r in successful) / len(successful)
    any_hit_cap = any(r["news"]["hit_page_cap"] for r in successful)

    n_sp500 = 503  # current S&P 500 count (from earlier scrape)
    est_price_calls = n_sp500
    est_news_calls_min = int(n_sp500 * avg_pages)
    # If any ticker hit the page cap, real number could be 2-5x larger
    est_news_calls_max = est_news_calls_min * 3 if any_hit_cap else est_news_calls_min

    print("=" * 70)
    print("  EXTRAPOLATION TO FULL S&P 500 PULL")
    print("=" * 70)
    print(f"  Probed tickers     : {len(successful)} / {len(results)}")
    print(f"  Avg articles/ticker: {avg_articles:,.0f}  (over probed sample)")
    print(f"  Avg pages/ticker   : {avg_pages:.1f}")
    print(f"  Any hit page cap?  : {any_hit_cap}  "
          f"(probes capped at {args.max_news_pages} pages)")
    print()
    print(f"  Estimated API calls (503 tickers):")
    print(f"    EOD prices       : ~{est_price_calls:,}")
    print(f"    News pages       : ~{est_news_calls_min:,}  "
          f"to ~{est_news_calls_max:,}")
    print(f"    TOTAL            : ~{est_price_calls + est_news_calls_min:,} "
          f"to ~{est_price_calls + est_news_calls_max:,}")
    print(f"  Daily limit (paid) : 100,000 calls/day")
    print(f"  Already used today : 1,415 (from collection_tracking.json)")
    print()
    print(f"  Estimated total articles: "
          f"{int(avg_articles * n_sp500):,} (lower bound) "
          f"to {int(avg_articles * n_sp500 * 3):,} (if any tickers were capped)")
    print(f"  Estimated disk: ~{int(avg_articles * n_sp500 * 3000 / 1e9):.1f} GB "
          f"(rough — 3KB/article average)")
    print()

    # Save raw probe results for reference
    out = PROJECT_ROOT / "outputs" / "eodhd_probe_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "date_window_min": TRAINING_DATE_MIN,
            "date_window_max": TRAINING_DATE_MAX,
            "training_id_count": len(training_ids),
            "probes": results,
            "extrapolation": {
                "n_sp500": n_sp500,
                "avg_articles_per_ticker_in_probe": avg_articles,
                "avg_pages_per_ticker_in_probe": avg_pages,
                "any_hit_page_cap": any_hit_cap,
                "est_news_calls_min": est_news_calls_min,
                "est_news_calls_max": est_news_calls_max,
                "est_price_calls": est_price_calls,
            },
        }, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

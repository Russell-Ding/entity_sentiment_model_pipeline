#!/usr/bin/env python3
"""Fetch the current S&P 500 component list from Wikipedia.

Saves a CSV with ticker, company name, GICS sector, sub-industry, and date
added to the index. Used as the canonical universe for downstream EODHD pulls.

The Wikipedia table is the de-facto reference for the live S&P 500
composition — Bloomberg and major data providers cite it. It updates as
companies are added/removed.

Usage:
    python scripts/collection/fetch_sp500_universe.py
    # → writes data/raw/sp500_components_<DATE>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500() -> pd.DataFrame:
    """Scrape the live S&P 500 table from Wikipedia.

    Returns a DataFrame with one row per current constituent.
    """
    # Wikipedia blocks the default urllib UA — fetch HTML ourselves first.
    import io
    import requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        raise RuntimeError("No tables found on Wikipedia S&P 500 page")

    df = tables[0]
    # Standardize column names — Wikipedia varies slightly over time
    df.columns = [c.strip() for c in df.columns]
    # Common forms we've seen: "Symbol" or "Ticker symbol" or "Symbol "
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("symbol", "ticker symbol", "ticker"):
            rename_map[col] = "ticker"
        elif cl in ("security", "company"):
            rename_map[col] = "company"
        elif "gics sector" in cl and "sub" not in cl:
            rename_map[col] = "sector"
        elif "gics sub" in cl or "industry" in cl:
            rename_map[col] = "sub_industry"
        elif cl == "date added" or "date first added" in cl:
            rename_map[col] = "date_added"
        elif cl == "cik":
            rename_map[col] = "cik"
        elif "founded" in cl:
            rename_map[col] = "founded"
    df = df.rename(columns=rename_map)

    # Keep just the useful columns
    keep = [c for c in ["ticker", "company", "sector", "sub_industry",
                        "date_added", "cik", "founded"] if c in df.columns]
    df = df[keep].copy()

    # EODHD uses "TICKER.US" for US listings, and tickers with dots use "-"
    # in some sources. Wikipedia uses "BRK.B" -> EODHD uses "BRK-B.US"
    df["eodhd_symbol"] = df["ticker"].str.replace(".", "-", regex=False) + ".US"

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--date-suffix", type=str,
                        default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sp500_components_{args.date_suffix}.csv"

    print(f"Fetching S&P 500 list from {WIKI_URL} ...")
    df = fetch_sp500()
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print()
    print("Sector distribution:")
    if "sector" in df.columns:
        for sector, n in df["sector"].value_counts().items():
            print(f"  {sector:40s} {n:4d}")
    print()
    print("Sample (5 rows):")
    sample_cols = [c for c in ["ticker", "eodhd_symbol", "company", "sector"]
                   if c in df.columns]
    print(df[sample_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()

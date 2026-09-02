"""Step 1 — Build the point-in-time (ticker, trade_date) panel.

Reads every outputs/inference/*_enriched.jsonl, explodes `ticker_sentiments` so
each row is (signal_ticker, trade_date, article), then aggregates to ONE row per
(ticker, trade_date). Joins each ticker's prices and attaches forward returns and
controls measured strictly AFTER the signal date.

No look-ahead, by construction:
  * The signal at day t aggregates only articles whose `trade_date == t`
    (`trade_date` already rolls post-publication news to the next tradable
    session, so it is the point-in-time-correct signal date).
  * Forward returns fwd_ret_h are close[t+h]/close[t]-1 using ADJUSTED close, i.e.
    they begin at t's close and end strictly after t. The signal is "known at t's
    close"; the long-short backtest adds a further next-open execution lag.

Output: signal_strategy/outputs/panel.parquet  (+ panel_baseline.parquet with the
EODHD article-level polarity aggregated the same way, for the Step-5 baseline).
"""
from __future__ import annotations

import gzip
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import data_utils as du


# --------------------------------------------------------------------------- #
# Parse enriched inference -> long (article x signal_ticker) rows
# --------------------------------------------------------------------------- #
def parse_enriched(enriched_glob: str) -> pd.DataFrame:
    """Explode ticker_sentiments across all enriched files.

    Dedups by article_id (an article lives in its primary_ticker's file, but we
    guard against any cross-file duplication so a single article never
    double-counts).
    """
    files = sorted(glob.glob(enriched_glob))
    if not files:
        raise FileNotFoundError(f"no enriched files matched {enriched_glob}")
    seen_articles: set[str] = set()
    rows: list[dict] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                aid = r.get("article_id")
                if aid in seen_articles:
                    continue
                seen_articles.add(aid)
                td = r.get("trade_date")
                tier = r.get("source_tier")
                tier1 = 1.0 if tier == 1 else 0.0
                for ts in r.get("ticker_sentiments", []):
                    tk = ts.get("ticker")
                    if not tk or td is None:
                        continue
                    sent = ts.get("sentiment")
                    if sent is None:
                        continue
                    rows.append({
                        "ticker": tk,
                        "trade_date": td,
                        "article_id": aid,
                        "sentiment": float(sent),
                        "total_mentions": float(ts.get("total_mentions", 0) or 0),
                        "tier1": tier1,
                    })
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    print(f"  parsed {len(files)} files, {len(seen_articles):,} unique articles, "
          f"{len(df):,} (article x ticker) rows")
    return df


def aggregate_signal(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (ticker, trade_date)."""
    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["total_mentions"].to_numpy()
        s = g["sentiment"].to_numpy()
        wsum = w.sum()
        sent_mw = float((s * w).sum() / wsum) if wsum > 0 else float(s.mean())
        return pd.Series({
            "sent_mw": sent_mw,
            "sent_mean": float(s.mean()),
            "n_articles": int(len(g)),
            "n_mentions": float(wsum),
            "sent_disp": float(np.std(s, ddof=0)) if len(g) > 1 else 0.0,
            "tier1_frac": float(g["tier1"].mean()),
        })

    out = long_df.groupby(["ticker", "trade_date"], sort=False).apply(agg).reset_index()
    print(f"  aggregated to {len(out):,} (ticker, trade_date) signal rows")
    return out


# --------------------------------------------------------------------------- #
# Attach prices: forward returns + controls, trading-calendar aware
# --------------------------------------------------------------------------- #
def attach_prices(sig: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    prices_dir = du.rp(cfg, cfg["paths"]["prices_dir"])
    suffix = cfg["universe"]["price_suffix"]
    horizons = cfg["horizons"]
    mom_s, mom_l = cfg["controls"]["mom_short"], cfg["controls"]["mom_long"]

    out_parts = []
    n_no_price, n_no_bar = 0, 0
    for tk, g in sig.groupby("ticker", sort=False):
        pf = du.load_price_frame(prices_dir, f"{tk}{suffix}")
        if pf is None or pf.empty:
            n_no_price += len(g)
            continue
        ac = pf["adjusted_close"].to_numpy()
        op = pf["open"].to_numpy()
        vol = pf["volume"].to_numpy()
        pos = {d: i for i, d in enumerate(pf.index)}
        n = len(pf)

        g = g.copy()
        idx = g["trade_date"].map(pos)          # integer position of the signal bar
        valid = idx.notna()
        n_no_bar += int((~valid).sum())
        g = g[valid].copy()
        if g.empty:
            continue
        i = idx[valid].astype(int).to_numpy()

        g["adjusted_close"] = ac[i]
        g["open_t"] = op[i]
        g["dollar_volume"] = ac[i] * vol[i]
        g["size"] = np.log(np.where(g["dollar_volume"] > 0, g["dollar_volume"], np.nan))

        # forward returns (close-to-close, adjusted) for each horizon
        for h in horizons:
            f = i + h
            ok = f < n
            simple = np.full(len(i), np.nan)
            logr = np.full(len(i), np.nan)
            simple[ok] = ac[f[ok]] / ac[i[ok]] - 1.0
            logr[ok] = np.log(ac[f[ok]] / ac[i[ok]])
            g[f"fwd_ret_{h}"] = simple
            g[f"fwd_logret_{h}"] = logr

        # past-return momentum controls
        for lag, name in ((mom_s, f"mom_{mom_s}"), (mom_l, f"mom_{mom_l}")):
            b = i - lag
            ok = b >= 0
            m = np.full(len(i), np.nan)
            m[ok] = ac[i[ok]] / ac[b[ok]] - 1.0
            g[name] = m

        out_parts.append(g)

    panel = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    if n_no_price:
        print(f"  dropped {n_no_price:,} signal rows with no price file")
    if n_no_bar:
        print(f"  dropped {n_no_bar:,} signal rows with no price bar on trade_date")
    return panel


def print_coverage(panel: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} coverage ===")
    print(f"  rows: {len(panel):,}")
    print(f"  unique tickers: {panel['ticker'].nunique():,}")
    print(f"  date range: {panel['trade_date'].min().date()} -> {panel['trade_date'].max().date()}")
    per_day = panel.groupby("trade_date")["ticker"].nunique()
    print(f"  trading days: {len(per_day):,}")
    print(f"  median names/day: {per_day.median():.0f}  (p10={per_day.quantile(.1):.0f}, "
          f"p90={per_day.quantile(.9):.0f})")
    for h in (1, 5, 20):
        c = panel[f"fwd_ret_{h}"].notna().mean()
        print(f"  fwd_ret_{h} non-null: {c:.1%}")


# --------------------------------------------------------------------------- #
# Baseline panel: EODHD article-level polarity, aggregated identically
# --------------------------------------------------------------------------- #
def build_baseline_panel(cfg: dict, model_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw EODHD `eodhd_sentiment.polarity` per (primary_ticker, trade_date).

    News archives are per-primary-ticker and lack a precomputed trade_date, so we
    roll each article's publication date to the next tradable session in that
    ticker's own price calendar (same point-in-time rule as the model panel).
    Used only as the Step-5 baseline to isolate the entity-level v2.1 edge.
    """
    news_files = sorted(glob.glob(du.rp(cfg, cfg["paths"]["news_glob"])))
    prices_dir = du.rp(cfg, cfg["paths"]["prices_dir"])
    suffix = cfg["universe"]["price_suffix"]
    horizons = cfg["horizons"]

    rows = []
    for fp in news_files:
        base = os.path.basename(fp).replace(".jsonl.gz", "")     # e.g. ABBV.US
        tk = base.replace(suffix, "")
        pf = du.load_price_frame(prices_dir, base)
        if pf is None or pf.empty:
            continue
        cal = pf.index
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                es = r.get("eodhd_sentiment") or {}
                pol = es.get("polarity")
                if pol is None or not r.get("date"):
                    continue
                pub = pd.to_datetime(r["date"]).tz_localize(None).normalize()
                # next tradable session >= publication date
                j = cal.searchsorted(pub, side="left")
                if j >= len(cal):
                    continue
                rows.append({"ticker": tk, "trade_date": cal[j], "polarity": float(pol)})
    bdf = pd.DataFrame(rows)
    if bdf.empty:
        return bdf
    agg = bdf.groupby(["ticker", "trade_date"], sort=False).agg(
        eodhd_pol=("polarity", "mean"), n_articles_base=("polarity", "size")).reset_index()

    # join the already-computed forward returns/controls from the model panel
    keep = ["ticker", "trade_date", "adjusted_close", "dollar_volume", "size",
            "mom_21", "mom_252"] + \
           [f"fwd_ret_{h}" for h in horizons] + [f"fwd_logret_{h}" for h in horizons]
    panel_b = agg.merge(model_panel[keep], on=["ticker", "trade_date"], how="inner")
    return panel_b


def main():
    cfg = du.load_config()
    out_dir = du.rp(cfg, cfg["paths"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] parsing enriched inference ...")
    long_df = parse_enriched(du.rp(cfg, cfg["paths"]["enriched_glob"]))
    print("[2/4] aggregating signal ...")
    sig = aggregate_signal(long_df)
    print("[3/4] attaching prices (forward returns + controls) ...")
    panel = attach_prices(sig, cfg)
    print_coverage(panel, "model panel")
    panel.to_parquet(du.rp(cfg, cfg["paths"]["panel"]), index=False)
    print(f"  wrote {cfg['paths']['panel']}")

    print("\n[4/4] building EODHD baseline panel ...")
    panel_b = build_baseline_panel(cfg, panel)
    if not panel_b.empty:
        print_coverage(panel_b.assign(**{}), "baseline panel")
        panel_b.to_parquet(du.rp(cfg, cfg["paths"]["panel_baseline"]), index=False)
        print(f"  wrote {cfg['paths']['panel_baseline']}")
    else:
        print("  (no baseline rows built)")


if __name__ == "__main__":
    main()

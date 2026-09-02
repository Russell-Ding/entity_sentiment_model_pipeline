"""MOC feasibility study — can we enter at the SIGNAL close and hold to the next close?

Motivation: the first pass showed all the alpha lives in the overnight gap after the
signal close (paper close-entry H=1 gross Sharpe 2.93), but `trade_date` assigns
after-hours news (~19% of articles, incl. the 4-5pm earnings spike) to the SAME
calendar day — so the paper backtest peeked at news that did not exist at the 4pm
close. This script tests the honest version of "trade the close":

  1. Re-assign every article to its first FEASIBLE close: the first trading day d
     with publication time <= d 15:45 ET (NYSE MOC submission cutoff, minus margin).
  2. Rebuild the sent_mw signal on feasible dates.
  3. Re-run the H=1 close->close long-short (enter MOC at t close, exit MOC at t+1
     close) and decompose day-1 spread into overnight vs intraday.
  4. Contrast with (a) the original look-ahead paper run and (b) a late-news-only
     signal traded at the next open (is after-hours news tradable at the open?).

Usage: python3 signal_strategy/src/moc_feasibility.py   (needs panel.parquet built)
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import data_utils as du
from build_panel import aggregate_signal
from backtest_ls import (load_prices, day_weights, run_backtest, perf_stats,
                         net_series, overnight_decomposition, evaluate)

ET = ZoneInfo("America/New_York")
CUTOFF_MIN = 15 * 60 + 45          # 15:45 ET — inference + MOC submission margin


# --------------------------------------------------------------------------- #
def trading_calendar(cfg) -> pd.DatetimeIndex:
    """NYSE session days from a full-history, always-traded name (AAPL)."""
    prices_dir = du.rp(cfg, cfg["paths"]["prices_dir"])
    pf = du.load_price_frame(prices_dir, "AAPL.US")
    return pf.index


def parse_with_feasible_dates(enriched_glob: str, cal: pd.DatetimeIndex):
    """Explode ticker_sentiments; assign eff_date = first trading day whose 15:45 ET
    cutoff is at/after publication. Also tag whether eff_date == original trade_date
    (feasible same-day) vs pushed later (was look-ahead for close entry)."""
    cal_np = cal.to_numpy()
    files = sorted(glob.glob(enriched_glob))
    seen: set[str] = set()
    rows: list[dict] = []
    n_art = n_pushed = n_badts = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                aid = r.get("article_id")
                if aid in seen:
                    continue
                seen.add(aid)
                td, ds = r.get("trade_date"), r.get("date")
                if td is None or ds is None:
                    continue
                try:
                    pub = datetime.fromisoformat(ds)
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=ZoneInfo("UTC"))
                    pub = pub.astimezone(ET)
                except ValueError:
                    n_badts += 1
                    continue
                n_art += 1
                # first calendar session d with pub <= d 15:45 ET
                d0 = np.datetime64(pub.date())
                i = np.searchsorted(cal_np, d0)
                if i < len(cal_np) and cal_np[i] == d0 \
                        and pub.hour * 60 + pub.minute <= CUTOFF_MIN:
                    eff = cal_np[i]
                else:
                    j = np.searchsorted(cal_np, d0, side="right")
                    if j >= len(cal_np):
                        continue                       # beyond price history
                    eff = cal_np[j]
                eff_ts = pd.Timestamp(eff)
                pushed = eff_ts > pd.Timestamp(td)
                n_pushed += pushed
                for ts in r.get("ticker_sentiments", []):
                    tk, sent = ts.get("ticker"), ts.get("sentiment")
                    if not tk or sent is None:
                        continue
                    rows.append({
                        "ticker": tk, "trade_date": eff_ts, "article_id": aid,
                        "sentiment": float(sent),
                        "total_mentions": float(ts.get("total_mentions", 0) or 0),
                        "tier1": 1.0, "pushed": bool(pushed),
                    })
    df = pd.DataFrame(rows)
    print(f"  {n_art:,} articles ({n_badts} bad timestamps); "
          f"{n_pushed:,} ({n_pushed / n_art * 100:.1f}%) pushed to a later close "
          f"(not knowable by 15:45 ET on their original trade_date)")
    return df


def main():
    cfg = du.load_config()
    out_dir = du.rp(cfg, cfg["paths"]["out_dir"])
    sig = cfg["signal"]["primary"]
    suffix = cfg["universe"]["price_suffix"]
    min_names = cfg["universe"]["min_names_per_day"]

    cal = trading_calendar(cfg)
    print("parsing enriched files with 15:45 ET feasibility cutoff ...")
    long_df = parse_with_feasible_dates(du.rp(cfg, cfg["paths"]["enriched_glob"]), cal)

    print("aggregating feasible-signal panel ...")
    feas = aggregate_signal(long_df[["ticker", "trade_date", "sentiment",
                                     "total_mentions", "tier1"]])
    feas["dollar_volume"] = np.nan            # equal weighting only

    # late-news-only signal: articles NOT knowable at their original close
    late = long_df[long_df["pushed"]]
    print(f"aggregating late-news-only panel ({late['article_id'].nunique():,} articles) ...")
    late_panel = aggregate_signal(late[["ticker", "trade_date", "sentiment",
                                        "total_mentions", "tier1"]])
    late_panel["dollar_volume"] = np.nan

    # original (look-ahead) panel for the same-machinery comparison
    orig = pd.read_parquet(du.rp(cfg, cfg["paths"]["panel"]))

    tickers = {f"{t}{suffix}" for t in
               set(feas["ticker"]) | set(orig["ticker"]) | set(late_panel["ticker"])}
    print(f"loading prices for {len(tickers)} tickers ...")
    prices, master = load_prices(cfg, tickers)

    results = {"cutoff_et": "15:45", "H": 1}

    print("\nday-1 spread decomposition, FEASIBLE signal (Q5-Q1, bps):")
    dec = overnight_decomposition(feas, prices, sig, min_names, suffix, 1)
    results["decomposition_feasible"] = dec
    print(f"  close->close (what MOC entry captures) : {dec['spread_day1_closeclose_bps']:7.1f} bps")
    print(f"  overnight gap close->open              : {dec['spread_day1_overnight_bps']:7.1f} bps")
    print(f"  intraday open->close                   : {dec['spread_day1_open_close_bps']:7.1f} bps")

    runs = {
        "feasible_moc":  (feas, "close", "FEASIBLE signal, MOC close->close (implementable)"),
        "original_paper": (orig, "close", "ORIGINAL signal, close entry (has look-ahead)"),
        "late_next_open": (late_panel, "open", "LATE news only, next-open entry"),
    }
    print(f"\nH=1 long-short (equal weight, Q5-Q1):")
    print(f"  {'variant':<44} | {'gross_Sh':>8} {'bps/d':>6} | {'net5_Sh':>7} {'net10_Sh':>8} {'turn/yr':>8}")
    for key, (panel, entry, label) in runs.items():
        wmap = day_weights(panel, sig, "equal", min_names, suffix)
        bt = run_backtest(wmap, prices, master, 1, entry=entry)
        g = perf_stats(bt["gross"], bt["turnover"])
        n5 = perf_stats(net_series(bt, 5), bt["turnover"])
        n10 = perf_stats(net_series(bt, 10), bt["turnover"])
        results[key] = {"label": label, "gross": g, "net5bps": n5, "net10bps": n10,
                        "eval": evaluate(bt, cfg, key)}
        print(f"  {label:<44} | {g['sharpe']:>8.2f} {g['mean_daily_bps']:>6.2f} | "
              f"{n5['sharpe']:>7.2f} {n10['sharpe']:>8.2f} {g['ann_turnover']:>7.1f}x")
        bt.to_csv(os.path.join(out_dir, f"ls_daily_{key}.csv"))

    with open(os.path.join(out_dir, "moc_feasibility.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote signal_strategy/outputs/moc_feasibility.json")


if __name__ == "__main__":
    main()

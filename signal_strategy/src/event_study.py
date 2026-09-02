"""Step 4 — Event study: where does the alpha concentrate?

Events = ticker-days in the TOP sentiment quintile vs the BOTTOM quintile (ranked
cross-sectionally each day on sent_mw). Abnormal return = stock daily return minus
the equal-weight universe return that day. We trace mean cumulative abnormal return
(CAR) from the signal close (tau=0) out to tau=+20 trading days, and read off the
window where the long-minus-short spread peaks -> default holding period H.

Entry convention: the signal is known at tau=0's close, so CAR(0)=0 and CAR(tau)
accumulates abnormal returns over the strictly-following sessions (t -> t+tau).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_utils as du

MAXTAU = 20


def build_universe_returns(cfg: dict) -> tuple[dict, pd.Series]:
    """Per-ticker daily simple returns (adjusted) + equal-weight market return series."""
    prices_dir = du.rp(cfg, cfg["paths"]["prices_dir"])
    rets = {}
    for fp in sorted(glob.glob(os.path.join(prices_dir, "*.csv"))):
        base = os.path.basename(fp).replace(".csv", "")        # e.g. ABBV.US
        pf = du.load_price_frame(prices_dir, base)
        if pf is None or pf.empty:
            continue
        r = pf["adjusted_close"].pct_change()
        rets[base] = r
    mkt = pd.DataFrame(rets).mean(axis=1)                       # equal-weight universe
    mkt.name = "mkt"
    return rets, mkt


def assign_quintiles(panel: pd.DataFrame, sig: str, min_names: int) -> pd.DataFrame:
    """Per-day quintile label (0..4) on sig; days with too few names dropped."""
    def q(g):
        s = g[sig].dropna()
        if len(s) < max(min_names, 5):
            return pd.Series(np.nan, index=g.index)
        try:
            lab = pd.qcut(g[sig].rank(method="first"), 5, labels=False)
        except ValueError:
            return pd.Series(np.nan, index=g.index)
        return lab
    panel = panel.copy()
    panel["q"] = panel.groupby("trade_date", group_keys=False).apply(q)
    return panel


def car_paths(events: pd.DataFrame, rets: dict, mkt: pd.Series, suffix: str):
    """Mean CAR path (tau=0..MAXTAU) across events. Abnormal = ticker ret - mkt ret."""
    # precompute abnormal-return arrays + position maps per ticker
    abn_cache, pos_cache = {}, {}
    paths = []
    for _, row in events.iterrows():
        tk_us = f"{row['ticker']}{suffix}"
        if tk_us not in rets:
            continue
        if tk_us not in abn_cache:
            r = rets[tk_us]
            abn = (r - mkt.reindex(r.index)).to_numpy()
            abn_cache[tk_us] = abn
            pos_cache[tk_us] = {d: i for i, d in enumerate(r.index)}
        abn = abn_cache[tk_us]
        pos = pos_cache[tk_us]
        i = pos.get(row["trade_date"])
        if i is None or i + MAXTAU >= len(abn):
            continue
        seg = abn[i + 1: i + 1 + MAXTAU]                        # tau=1..MAXTAU
        car = np.concatenate([[0.0], np.cumsum(seg)])          # tau=0..MAXTAU
        paths.append(car)
    if not paths:
        return None, 0
    arr = np.vstack(paths)
    return arr, arr.shape[0]


def main():
    cfg = du.load_config()
    panel = pd.read_parquet(du.rp(cfg, cfg["paths"]["panel"]))
    out_dir = du.rp(cfg, cfg["paths"]["out_dir"])
    sig = cfg["signal"]["primary"]
    suffix = cfg["universe"]["price_suffix"]
    min_names = cfg["universe"]["min_names_per_day"]

    print("building universe daily returns + equal-weight market ...")
    rets, mkt = build_universe_returns(cfg)
    print(f"  {len(rets)} tickers, market series {mkt.index.min().date()}..{mkt.index.max().date()}")

    panel = assign_quintiles(panel, sig, min_names)
    top = panel[panel["q"] == 4]
    bot = panel[panel["q"] == 0]
    print(f"  top-quintile events: {len(top):,}   bottom-quintile events: {len(bot):,}")

    top_arr, n_top = car_paths(top, rets, mkt, suffix)
    bot_arr, n_bot = car_paths(bot, rets, mkt, suffix)
    tau = np.arange(0, MAXTAU + 1)
    car_top = top_arr.mean(0)
    car_bot = bot_arr.mean(0)
    spread = car_top - car_bot
    # SE of the spread at each tau (independent-events approx)
    se_spread = np.sqrt(top_arr.var(0, ddof=1) / n_top + bot_arr.var(0, ddof=1) / n_bot)
    tstat = np.divide(spread, se_spread, out=np.zeros_like(spread), where=se_spread > 0)

    H = int(tau[np.argmax(spread)])
    print(f"\nCAR spread (top-bottom) peaks at tau=+{H}: {spread[H]*1e4:.1f} bps "
          f"(t={tstat[H]:.2f})")
    for t in (1, 5, 10, 20):
        print(f"  tau=+{t:>2}: top {car_top[t]*1e4:>7.1f} bps | bot {car_bot[t]*1e4:>7.1f} bps "
              f"| spread {spread[t]*1e4:>7.1f} bps (t={tstat[t]:.2f})")

    # --- plot ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(tau, car_top * 1e4, label=f"top Q5 (n={n_top:,})", color="#2a7")
    ax[0].plot(tau, car_bot * 1e4, label=f"bottom Q1 (n={n_bot:,})", color="#c44")
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_title("Mean CAR by sentiment quintile")
    ax[0].set_xlabel("trading days after signal (tau)"); ax[0].set_ylabel("CAR (bps)")
    ax[0].legend()
    ax[1].plot(tau, spread * 1e4, color="#338", label="Q5 - Q1 spread")
    ax[1].fill_between(tau, (spread - 1.96 * se_spread) * 1e4, (spread + 1.96 * se_spread) * 1e4,
                       color="#338", alpha=.15, label="95% CI")
    ax[1].axhline(0, color="k", lw=.6); ax[1].axvline(H, color="#888", ls="--", lw=.8)
    ax[1].set_title(f"Long-short CAR spread (peak at tau=+{H})")
    ax[1].set_xlabel("trading days after signal (tau)"); ax[1].set_ylabel("spread (bps)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "event_study_car.png"), dpi=110)
    print(f"\nwrote {cfg['paths']['out_dir']}/event_study_car.png")

    out = {
        "n_top": n_top, "n_bottom": n_bot, "peak_tau": H,
        "tau": tau.tolist(),
        "car_top_bps": (car_top * 1e4).round(2).tolist(),
        "car_bottom_bps": (car_bot * 1e4).round(2).tolist(),
        "spread_bps": (spread * 1e4).round(2).tolist(),
        "spread_tstat": np.round(tstat, 3).tolist(),
        "recommended_H": H,
    }
    with open(os.path.join(out_dir, "event_study.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {cfg['paths']['out_dir']}/event_study.json   recommended H={H}")


if __name__ == "__main__":
    main()

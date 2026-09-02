"""Step 3 — Fama-MacBeth cross-sectional regressions.

Each day, regress the forward return on the standardized signal plus controls:

    fwd_ret_h ~ sent_mw + size + mom_21 + mom_252        (predictors z-scored daily)

Collect the daily slope vector, average over days, and report Newey-West (lag 5)
t-stats on the time series of slopes. The question: is sentiment priced AFTER
controlling for size and momentum? The `sent_mw` slope is the answer, in return
units per 1 cross-sectional SD of signal.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import data_utils as du


PREDICTORS = ["sent_mw", "size", "mom_21", "mom_252"]
NW_LAG = 5


def daily_slopes(panel: pd.DataFrame, ret_col: str, min_names: int) -> pd.DataFrame:
    """Return a DataFrame of daily OLS coefficients (index=date, cols=[const]+PREDICTORS)."""
    recs = {}
    cols = ["const"] + PREDICTORS
    for d, g in panel.groupby("trade_date"):
        sub = g[[ret_col] + PREDICTORS].dropna()
        if len(sub) < max(min_names, len(PREDICTORS) + 2):
            continue
        X = np.column_stack([du.zscore(sub[p]).to_numpy() for p in PREDICTORS])
        X = np.column_stack([np.ones(len(sub)), X])           # intercept
        y = sub[ret_col].to_numpy()
        # guard against a degenerate (collinear) day
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        recs[d] = beta
    return pd.DataFrame.from_dict(recs, orient="index", columns=cols).sort_index()


def summarize(slopes: pd.DataFrame) -> dict:
    out = {}
    for c in slopes.columns:
        mu, se, t = du.newey_west_mean_tstat(slopes[c].to_numpy(), lag=NW_LAG)
        out[c] = {"mean": float(mu), "nw_se": float(se), "nw_tstat": float(t),
                  "n_days": int(slopes[c].notna().sum())}
    return out


def main():
    cfg = du.load_config()
    panel = pd.read_parquet(du.rp(cfg, cfg["paths"]["panel"]))
    out_dir = du.rp(cfg, cfg["paths"]["out_dir"])
    min_names = cfg["universe"]["min_names_per_day"]
    horizons = cfg["horizons"]

    results = {"predictors": PREDICTORS, "nw_lag": NW_LAG, "horizons": {}}
    print("Fama-MacBeth: fwd_ret_h ~ " + " + ".join(PREDICTORS) +
          "   (predictors z-scored daily)\n")
    print(f"slopes are mean daily coefficient (return per 1 cross-sectional SD), "
          f"Newey-West(lag {NW_LAG}) t in [].\n")
    for h in horizons:
        slopes = daily_slopes(panel, f"fwd_ret_{h}", min_names)
        s = summarize(slopes)
        results["horizons"][h] = s
        n = s["sent_mw"]["n_days"]
        print(f"--- h={h}  ({n} days) ---")
        for c in ["sent_mw", "size", "mom_21", "mom_252"]:
            bp = s[c]["mean"] * 1e4
            print(f"  {c:>8}: {bp:>8.2f} bps  [t={s[c]['nw_tstat']:>6.2f}]")
        print()

    with open(os.path.join(out_dir, "fama_macbeth.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {cfg['paths']['out_dir']}/fama_macbeth.json")

    print("\n--- CONCLUSION ---")
    for h in horizons:
        s = results["horizons"][h]["sent_mw"]
        sig = "priced" if abs(s["nw_tstat"]) >= 1.96 else "NOT priced (|t|<1.96)"
        print(f"  h={h:>2}: sent_mw = {s['mean']*1e4:.2f} bps/SD, t={s['nw_tstat']:.2f} -> {sig} after controls")


if __name__ == "__main__":
    main()

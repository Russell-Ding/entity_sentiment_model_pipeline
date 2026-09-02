"""Step 2 — Information Coefficient (the cheap go/no-go).

For each trading day, compute the cross-sectional Spearman rank correlation
between the signal (`sent_mw`) and each forward-return horizon. Summarize the
daily IC series: mean IC, IC stdev, IC IR (mean/std), Newey-West t-stat, and the
fraction of days positive. Also split by tier-1 news share.

Gate: if IC ~ 0 / IR < ~0.05 across horizons, say so plainly.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_utils as du


def daily_ic(panel: pd.DataFrame, sig_col: str, ret_col: str, min_names: int) -> pd.Series:
    ics = {}
    for d, g in panel.groupby("trade_date"):
        sub = g[[sig_col, ret_col]].dropna()
        if len(sub) < min_names:
            continue
        ic = du.spearman_ic(sub, sig_col, ret_col)
        if not np.isnan(ic):
            ics[d] = ic
    return pd.Series(ics).sort_index()


def summarize_ic(ic: pd.Series) -> dict:
    mu, se, t = du.newey_west_mean_tstat(ic.to_numpy())
    sd = ic.std(ddof=0)
    return {
        "n_days": int(ic.size),
        "mean_ic": float(mu),
        "ic_std": float(sd),
        "ic_ir": float(mu / sd) if sd > 0 else float("nan"),
        "nw_tstat": float(t),
        "pct_days_positive": float((ic > 0).mean()),
    }


def main():
    cfg = du.load_config()
    panel = pd.read_parquet(du.rp(cfg, cfg["paths"]["panel"]))
    out_dir = du.rp(cfg, cfg["paths"]["out_dir"])
    sig = cfg["signal"]["primary"]
    min_names = cfg["universe"]["min_names_per_day"]
    horizons = cfg["horizons"]

    results = {"signal": sig, "min_names_per_day": min_names, "horizons": {}}
    ic_series = {}
    print(f"Information Coefficient — Spearman({sig}, fwd_ret_h), per day\n")
    print(f"{'h':>3} {'n_days':>7} {'mean_IC':>9} {'IC_std':>8} {'IC_IR':>7} "
          f"{'NW_t':>7} {'%pos':>6}")
    for h in horizons:
        ic = daily_ic(panel, sig, f"fwd_ret_{h}", min_names)
        ic_series[h] = ic
        s = summarize_ic(ic)
        results["horizons"][h] = s
        print(f"{h:>3} {s['n_days']:>7} {s['mean_ic']:>9.4f} {s['ic_std']:>8.4f} "
              f"{s['ic_ir']:>7.3f} {s['nw_tstat']:>7.2f} {s['pct_days_positive']:>6.1%}")

    # --- conditioning split: news intensity ---
    # NOTE: all enriched inference is tier-1 (tier1_frac == 1.0 everywhere), so the
    # brief's tier-1 split is degenerate here. We instead split on corroboration:
    # ticker-days with a single article vs multiple articles (does corroborated
    # news flow carry more signal?).
    print("\nBy news intensity (ticker-days with 1 article vs >=2), h=5:")
    results["all_tier1"] = True
    results["intensity_split"] = {}
    for label, mask in (("multi_article", panel["n_articles"] >= 2),
                        ("single_article", panel["n_articles"] == 1)):
        sub = panel[mask]
        ic = daily_ic(sub, sig, "fwd_ret_5", min_names)
        s = summarize_ic(ic)
        results["intensity_split"][label] = s
        print(f"  {label:>14} (h=5): mean_IC {s['mean_ic']:.4f}  IR {s['ic_ir']:.3f}  "
              f"NW_t {s['nw_tstat']:.2f}  n_days={s['n_days']}")

    # --- plot: IC by horizon (bar) + cumulative IC at h=5 ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hs = list(horizons)
    means = [results["horizons"][h]["mean_ic"] for h in hs]
    irs = [results["horizons"][h]["ic_ir"] for h in hs]
    ax[0].bar([str(h) for h in hs], means, color="#3b7", alpha=.8)
    ax[0].set_title("Mean daily IC by horizon")
    ax[0].set_xlabel("forward horizon (trading days)")
    ax[0].set_ylabel("mean Spearman IC")
    ax[0].axhline(0, color="k", lw=.6)
    for i, ir in enumerate(irs):
        ax[0].text(i, means[i], f"IR {ir:.2f}", ha="center", va="bottom", fontsize=8)
    ic5 = ic_series[5]
    ax[1].plot(ic5.index, ic5.cumsum(), color="#26a")
    ax[1].set_title("Cumulative daily IC (h=5)")
    ax[1].set_xlabel("date"); ax[1].set_ylabel("cumulative IC")
    ax[1].axhline(0, color="k", lw=.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ic_by_horizon.png"), dpi=110)
    print(f"\nwrote {cfg['paths']['out_dir']}/ic_by_horizon.png")

    # save daily IC series + summary
    pd.DataFrame({f"ic_h{h}": ic_series[h] for h in horizons}).to_csv(
        os.path.join(out_dir, "ic_daily.csv"))
    with open(os.path.join(out_dir, "ic_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {cfg['paths']['out_dir']}/ic_summary.json, ic_daily.csv")

    # --- verdict ---
    best = max(horizons, key=lambda h: abs(results["horizons"][h]["ic_ir"]))
    ir = results["horizons"][best]["ic_ir"]
    print("\n--- GATE ---")
    if abs(ir) < 0.05:
        print(f"IC IR peaks at {ir:.3f} (h={best}) — below ~0.05. Weak/no signal.")
    else:
        print(f"IC IR peaks at {ir:.3f} (h={best}); NW t={results['horizons'][best]['nw_tstat']:.2f}. "
              f"Signal present -> proceed.")


if __name__ == "__main__":
    main()

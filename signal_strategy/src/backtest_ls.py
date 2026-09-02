"""Step 5 — Quintile long-short backtest (realistic execution + costs).

Each day, sort the universe cross-sectionally on the signal into quintiles; go
LONG the top quintile (Q5) and SHORT the bottom (Q1). Realism:

  * Execution lag: positions are entered at the NEXT session's OPEN after the
    signal date (we never trade on the signal bar). The first holding day's return
    is therefore open->close of t+1; the post-signal overnight gap is NOT captured.
  * Overlapping H-day holding via a fractional book: each day 1/H of the book is
    the freshly-formed sub-portfolio and H-1/H is older sub-portfolios still in
    their holding window (Jegadeesh-Titman style). Gross exposure ~ 2 (1 long, 1
    short), dollar-neutral.
  * Weighting: equal-weight and value-weight (by dollar volume) variants.
  * Costs: charged on day-over-day change in target book weights; reported gross
    and net at 5 and 10 bps per side (conservative — ignores intra-name netting
    drift, so a slight over-estimate of turnover).

Reports annualized return / vol / Sharpe, max drawdown, turnover, mean-daily-return
Newey-West t-stat, hit rate — full sample and out-of-sample (last 30% of dates).
Reruns the identical machinery on the EODHD article-level polarity baseline to
isolate the entity-level v2.1 edge.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import data_utils as du

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Price machinery
# --------------------------------------------------------------------------- #
def load_prices(cfg, tickers_us):
    """Return {ticker_us: {'dates': index, 'pos': map, 'c2c': arr, 'oc': arr}}."""
    prices_dir = du.rp(cfg, cfg["paths"]["prices_dir"])
    out = {}
    master = None
    for tk in sorted(tickers_us):
        pf = du.load_price_frame(prices_dir, tk)
        if pf is None or pf.empty:
            continue
        ac = pf["adjusted_close"].to_numpy()
        op = pf["open"].to_numpy()
        cl = pf["close"].to_numpy()
        c2c = np.empty(len(ac)); c2c[0] = np.nan; c2c[1:] = ac[1:] / ac[:-1] - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            oc = np.where(op > 0, cl / op - 1.0, np.nan)        # open->close, raw
        out[tk] = {"dates": pf.index, "pos": {d: i for i, d in enumerate(pf.index)},
                   "c2c": c2c, "oc": oc}
        master = pf.index if master is None else master.union(pf.index)
    return out, master


# --------------------------------------------------------------------------- #
# Signal -> per-day leg weights
# --------------------------------------------------------------------------- #
def day_weights(panel, sig, weighting, min_names, suffix):
    """For each trade_date, dict {ticker_us: weight}; long Q5 sum +1, short Q1 sum -1."""
    wmap = {}
    for d, g in panel.groupby("trade_date"):
        s = g[[sig, "ticker", "dollar_volume"]].dropna(subset=[sig])
        if len(s) < max(min_names, 10):
            continue
        try:
            q = pd.qcut(s[sig].rank(method="first"), 5, labels=False)
        except ValueError:
            continue
        longs, shorts = s[q == 4], s[q == 0]
        w = {}
        for leg, sign in ((longs, +1.0), (shorts, -1.0)):
            if weighting == "value":
                dv = leg["dollar_volume"].clip(lower=0).to_numpy()
                tot = dv.sum()
                ww = dv / tot if tot > 0 else np.full(len(leg), 1.0 / len(leg))
            else:
                ww = np.full(len(leg), 1.0 / len(leg))
            for tk, wi in zip(leg["ticker"].to_numpy(), ww):
                w[f"{tk}{suffix}"] = sign * wi
        wmap[d] = w
    return wmap


# --------------------------------------------------------------------------- #
# Core overlapping-book backtest
# --------------------------------------------------------------------------- #
def run_backtest(wmap, prices, master, H, entry="open"):
    """Return DataFrame indexed by date with columns gross, turnover (sum|dW|).

    entry='open'  -> realistic next-open execution: the first holding day's return
                     is open[t+1]->close[t+1] (post-signal overnight gap NOT captured).
    entry='close' -> infeasible "paper" execution at the signal close: first day is
                     close[t]->close[t+1] (captures the overnight gap). The contrast
                     between the two isolates how much alpha is overnight-only.
    """
    master = pd.DatetimeIndex(sorted(master))
    signal_days = set(wmap)
    # Clip the calendar to the active signal window (the price union spans decades
    # before any news exists; those zero-signal days would otherwise dominate and
    # destroy the Sharpe). Keep one session before the first signal so its entry
    # lands, and H sessions after the last signal so the book unwinds.
    sig_sorted = sorted(signal_days)
    lo = max(0, master.searchsorted(sig_sorted[0]) - 1)
    hi = min(len(master), master.searchsorted(sig_sorted[-1]) + H + 2)
    master = master[lo:hi]

    def ret(tk, d, first):
        p = prices.get(tk)
        if p is None:
            return np.nan
        i = p["pos"].get(d)
        if i is None:
            return np.nan
        return p["oc"][i] if (first and entry == "open") else p["c2c"][i]

    active = []                      # list of [weights_dict, age]
    prevW = {}
    rows = {}
    for p in range(len(master)):
        d = master[p]
        # new sub-portfolio enters today if the PREVIOUS session was a signal day
        if p > 0 and master[p - 1] in signal_days:
            active.append([wmap[master[p - 1]], 0])
        # daily gross return = mean over active sub-portfolios (cash drag if <H)
        gross = 0.0
        for sp in active:
            sp[1] += 1                                          # age 1..H
            first = sp[1] == 1
            r = 0.0
            for tk, wi in sp[0].items():
                rr = ret(tk, d, first)
                if np.isfinite(rr):
                    r += wi * rr
            gross += r
        gross /= H
        # target book weights (for turnover/costs)
        W = {}
        for sp in active:
            for tk, wi in sp[0].items():
                W[tk] = W.get(tk, 0.0) + wi / H
        keys = set(W) | set(prevW)
        turnover = sum(abs(W.get(k, 0.0) - prevW.get(k, 0.0)) for k in keys)
        prevW = W
        # retire finished sub-portfolios
        active = [sp for sp in active if sp[1] < H]
        rows[d] = (gross, turnover)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["gross", "turnover"])
    return df.sort_index()


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def perf_stats(daily_net: pd.Series, turnover: pd.Series) -> dict:
    r = daily_net.dropna()
    if len(r) < 20:
        return {}
    mu, vol = r.mean(), r.std(ddof=0)
    ann_ret = mu * TRADING_DAYS
    ann_vol = vol * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    _, _, t = du.newey_west_mean_tstat(r.to_numpy(), lag=10)
    return {
        "ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe),
        "max_drawdown": float(dd), "hit_rate": float((r > 0).mean()),
        "mean_daily_bps": float(mu * 1e4), "nw_tstat": float(t),
        "avg_oneway_turnover": float(turnover.mean() / 2.0),
        "ann_turnover": float(turnover.mean() / 2.0 * TRADING_DAYS),
        "n_days": int(len(r)),
    }


def net_series(bt: pd.DataFrame, bps: float) -> pd.Series:
    return bt["gross"] - (bps / 1e4) * bt["turnover"]


def overnight_decomposition(panel, prices, sig, min_names, suffix, H):
    """Why paper != tradable: split the day-1 Q5-Q1 spread into the overnight gap
    (close[t]->open[t+1], uncapturable at next-open) vs the intraday open->close,
    and compare the H-day spread under close-entry vs open-entry. Returns bps."""
    def q(g):
        if len(g) < max(min_names, 10):
            return pd.Series(np.nan, index=g.index)
        try:
            return pd.qcut(g[sig].rank(method="first"), 5, labels=False)
        except ValueError:
            return pd.Series(np.nan, index=g.index)
    panel = panel.copy()
    panel["q"] = panel.groupby("trade_date", group_keys=False).apply(q)
    acc = {0: [], 4: []}
    for _, r in panel[panel["q"].isin([0, 4])].iterrows():
        p = prices.get(f"{r['ticker']}{suffix}")
        if p is None:
            continue
        i = p["pos"].get(r["trade_date"])
        if i is None or i + H >= len(p["c2c"]):
            continue
        c2c, oc = p["c2c"], p["oc"]
        # cumulative close-entry (paper) vs open-entry (tradable) H-day return
        paper = np.nansum(c2c[i + 1:i + 1 + H])
        trad = oc[i + 1] + np.nansum(c2c[i + 2:i + 1 + H])
        overnight = c2c[i + 1] - oc[i + 1]                      # close[t]->open[t+1]
        acc[int(r["q"])].append((c2c[i + 1], oc[i + 1], overnight, paper, trad))
    a4, a0 = np.array(acc[4]), np.array(acc[0])
    sp = lambda col: float((np.nanmean(a4[:, col]) - np.nanmean(a0[:, col])) * 1e4)
    return {
        "n_top": len(a4), "n_bottom": len(a0),
        "spread_day1_closeclose_bps": sp(0),   # event-study style (incl. overnight)
        "spread_day1_open_close_bps": sp(1),   # tradable intraday day 1
        "spread_day1_overnight_bps": sp(2),    # the uncapturable gap
        "spread_Hday_paper_close_entry_bps": sp(3),
        "spread_Hday_tradable_open_entry_bps": sp(4),
    }


def evaluate(bt, cfg, label):
    """Stats at each cost level, full-sample and OOS (last 30%)."""
    cutoff = du.split_dates(pd.Series(bt.index), cfg["backtest"]["train_frac"])
    out = {"full": {}, "oos": {}}
    for bps in [0] + cfg["execution"]["costs_bps"]:
        net = net_series(bt, bps)
        out["full"][f"{bps}bps"] = perf_stats(net, bt["turnover"])
        oos = bt[bt.index > cutoff]
        out["oos"][f"{bps}bps"] = perf_stats(net_series(oos, bps), oos["turnover"])
    out["oos_start"] = str(cutoff.date())
    return out


# --------------------------------------------------------------------------- #
def main():
    cfg = du.load_config()
    out_dir = du.rp(cfg, cfg["paths"]["out_dir"])
    sig = cfg["signal"]["primary"]
    suffix = cfg["universe"]["price_suffix"]
    min_names = cfg["universe"]["min_names_per_day"]

    # holding period from the event study (fallback to config default)
    H = cfg["execution"]["default_hold"]
    es = os.path.join(out_dir, "event_study.json")
    if os.path.exists(es):
        H = max(2, int(json.load(open(es))["recommended_H"]))
    print(f"holding period H = {H} (from event study)\n")

    panel = pd.read_parquet(du.rp(cfg, cfg["paths"]["panel"]))
    panel_b = pd.read_parquet(du.rp(cfg, cfg["paths"]["panel_baseline"]))

    # union of tickers we'll ever trade (both signals) -> load prices once
    tickers = {f"{t}{suffix}" for t in panel["ticker"].unique()} | \
              {f"{t}{suffix}" for t in panel_b["ticker"].unique()}
    print(f"loading prices for {len(tickers)} tickers ...")
    prices, master = load_prices(cfg, tickers)

    results = {"H": H, "signal": sig}
    print("execution decomposition (Q5-Q1 spread, bps) ...")
    decomp = overnight_decomposition(panel, prices, sig, min_names, suffix, H)
    results["execution_decomposition"] = decomp
    print(f"  day-1 close->close (paper, incl overnight): {decomp['spread_day1_closeclose_bps']:7.1f} bps")
    print(f"  day-1 overnight gap (UNcapturable)        : {decomp['spread_day1_overnight_bps']:7.1f} bps")
    print(f"  day-1 open->close (tradable intraday)     : {decomp['spread_day1_open_close_bps']:7.1f} bps")
    print(f"  {H}-day paper  (close-entry)               : {decomp['spread_Hday_paper_close_entry_bps']:7.1f} bps")
    print(f"  {H}-day tradable (open-entry)              : {decomp['spread_Hday_tradable_open_entry_bps']:7.1f} bps\n")
    wmap = day_weights(panel, sig, "equal", min_names, suffix)

    # ---- core deliverable: H x execution sweep (equal weight) ----------------
    # paper  = signal-close execution (captures the overnight gap; INFEASIBLE).
    # open   = realistic next-open execution.
    # The contrast is the whole story: paper alpha is real but overnight-only and
    # cost-fragile; tradable alpha is absent.
    print("H x execution sweep (equal weight, Q5-Q1):\n")
    print(f"  {'H':>2} {'entry':>6} | {'gross_Sh':>8} {'gross_bps':>9} | "
          f"{'net5_Sh':>7} {'net10_Sh':>8} {'net10_ann':>9} {'turn/yr':>8}")
    sweep = {}
    H_list = sorted({1, 5, H})
    for hh in H_list:
        for entry in ("close", "open"):
            bt = run_backtest(wmap, prices, master, hh, entry=entry)
            g = perf_stats(bt["gross"], bt["turnover"])
            n5 = perf_stats(net_series(bt, 5), bt["turnover"])
            n10 = perf_stats(net_series(bt, 10), bt["turnover"])
            sweep[f"H{hh}_{entry}"] = {"gross": g, "net5bps": n5, "net10bps": n10}
            print(f"  {hh:>2} {entry:>6} | {g['sharpe']:>8.2f} {g['mean_daily_bps']:>9.2f} | "
                  f"{n5['sharpe']:>7.2f} {n10['sharpe']:>8.2f} {n10['ann_return']*100:>8.1f}% "
                  f"{g['ann_turnover']:>7.1f}x")
    results["sweep_equal"] = sweep

    # ---- the brief's required variants at the event-study H -------------------
    # full stats (full + OOS, all cost levels) for: EW & VW next-open (realistic),
    # EW paper (close exec), and the EODHD baseline.
    bt_open = run_backtest(wmap, prices, master, H, entry="open")
    bt_paper = run_backtest(wmap, prices, master, H, entry="close")
    results["v21_equal_open"] = evaluate(bt_open, cfg, "ew_open")
    results["v21_equal_paper_closeexec"] = evaluate(bt_paper, cfg, "ew_paper")
    bt_open.to_csv(os.path.join(out_dir, "ls_daily_equal.csv"))
    bt_paper.to_csv(os.path.join(out_dir, "ls_daily_equal_paper.csv"))

    wmap_v = day_weights(panel, sig, "value", min_names, suffix)
    bt_val = run_backtest(wmap_v, prices, master, H, entry="open")
    results["v21_value_open"] = evaluate(bt_val, cfg, "vw_open")

    # ---- baseline: EODHD article-level polarity (equal weight, next-open) ----
    panel_b = panel_b.rename(columns={"eodhd_pol": "sig_base"})
    wmap_b = day_weights(panel_b, "sig_base", "equal", min_names, suffix)
    bt_b = run_backtest(wmap_b, prices, master, H, entry="open")
    bt_b_paper = run_backtest(wmap_b, prices, master, H, entry="close")
    results["baseline_eodhd_open"] = evaluate(bt_b, cfg, "base_open")
    results["baseline_eodhd_paper_closeexec"] = evaluate(bt_b_paper, cfg, "base_paper")
    bt_b.to_csv(os.path.join(out_dir, "ls_daily_baseline.csv"))

    print(f"\nat H={H}, 10bps, full sample:")
    for k, lbl in (("v21_equal_paper_closeexec", "v2.1 EW PAPER (close exec)"),
                    ("v21_equal_open", "v2.1 EW tradable (open exec)"),
                    ("v21_value_open", "v2.1 VW tradable (open exec)"),
                    ("baseline_eodhd_paper_closeexec", "EODHD PAPER (close exec)"),
                    ("baseline_eodhd_open", "EODHD tradable (open exec)")):
        s = results[k]["full"]["10bps"]
        o = results[k]["oos"]["10bps"]
        print(f"  {lbl:<30} Sharpe {s['sharpe']:>5.2f}  ann {s['ann_return']*100:>6.1f}%  "
              f"DD {s['max_drawdown']*100:>6.1f}%  | OOS Sharpe {o['sharpe']:>5.2f}")

    # ---- equity curve: paper-gross vs tradable-gross vs tradable-net + baseline ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for series, lab, c, ls in (
        (bt_paper["gross"], f"v2.1 PAPER close-exec, GROSS (Sharpe {sweep[f'H{H}_close']['gross']['sharpe']:.2f})", "#2a7", "-"),
        (bt_open["gross"], "v2.1 tradable open-exec, GROSS", "#e80", "-"),
        (net_series(bt_open, 10), "v2.1 tradable open-exec, NET 10bps", "#c44", "-"),
        (net_series(bt_b, 10), "EODHD baseline, NET 10bps", "#88a", "--")):
        eq = (1 + series.dropna()).cumprod()
        ax.plot(eq.index, eq.values, label=lab, color=c, ls=ls)
    cutoff = du.split_dates(pd.Series(bt_open.index), cfg["backtest"]["train_frac"])
    ax.axvline(cutoff, color="#888", ls=":", lw=1)
    ax.text(cutoff, ax.get_ylim()[1], " OOS ->", va="top", fontsize=8, color="#666")
    ax.set_title(f"Quintile long-short: paper signal is real, tradable signal is not (H={H})")
    ax.set_ylabel("growth of $1 (log)"); ax.set_xlabel("date"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.axhline(1, color="k", lw=.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ls_equity_curve.png"), dpi=110)
    print(f"\nwrote {cfg['paths']['out_dir']}/ls_equity_curve.png")

    with open(os.path.join(out_dir, "backtest.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {cfg['paths']['out_dir']}/backtest.json")


if __name__ == "__main__":
    main()

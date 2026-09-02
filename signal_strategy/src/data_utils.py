"""Shared utilities for the sentiment->returns signal study.

Price loading, trading-calendar-aware forward returns, Newey-West HAC t-stats,
and small stats helpers. Kept dependency-light (pandas/numpy/scipy) so the whole
study reproduces offline.
"""
from __future__ import annotations

import os
import glob
import yaml
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Config / paths
# --------------------------------------------------------------------------- #
def repo_root() -> str:
    """Repo root = two levels up from this file (signal_strategy/src/ -> repo)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = repo_root()
    return cfg


def rp(cfg: dict, rel: str) -> str:
    """Resolve a repo-relative path to absolute."""
    return os.path.join(cfg["_root"], rel)


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def load_price_frame(prices_dir: str, ticker_us: str) -> pd.DataFrame | None:
    """Load one <ticker>.US.csv as a date-indexed, ascending DataFrame.

    Returns None if the file is missing. Columns: open, high, low, close,
    adjusted_close, volume. Index = trading dates (pandas Timestamp), sorted,
    deduplicated (last wins).
    """
    fp = os.path.join(prices_dir, f"{ticker_us}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    if df.empty or "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    for c in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Newey-West HAC standard error for a 1-D mean (the Fama-MacBeth / IC case)
# --------------------------------------------------------------------------- #
def newey_west_mean_tstat(x: np.ndarray, lag: int | None = None) -> tuple[float, float, float]:
    """Mean, NW-HAC standard error, and t-stat for the mean of a time series.

    Standard Fama-MacBeth inference: we have a time series of daily estimates
    (slopes or daily ICs); their average is the estimate, and we need a t-stat
    robust to autocorrelation (overlapping forward returns induce it).

    Bartlett kernel. lag defaults to floor(4*(n/100)^(2/9)) (Newey-West rule).
    Returns (mean, se, tstat). NaNs dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return (float(np.mean(x)) if n else np.nan, np.nan, np.nan)
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    gamma0 = np.dot(e, e) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(e[k:], e[:-k]) / n
        var += 2.0 * w * gamma_k
    # variance of the mean
    se = np.sqrt(max(var, 0.0) / n)
    t = mu / se if se > 0 else np.nan
    return mu, se, t


# --------------------------------------------------------------------------- #
# Cross-sectional helpers
# --------------------------------------------------------------------------- #
def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional standardization; constant/empty -> zeros."""
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def spearman_ic(g: pd.DataFrame, sig_col: str, ret_col: str) -> float:
    """Cross-sectional Spearman rank correlation for one day's slice."""
    sub = g[[sig_col, ret_col]].dropna()
    if len(sub) < 5 or sub[sig_col].nunique() < 2 or sub[ret_col].nunique() < 2:
        return np.nan
    return sub[sig_col].rank().corr(sub[ret_col].rank())


def split_dates(dates: pd.Series, train_frac: float) -> pd.Timestamp:
    """Return the cutoff date so that ~train_frac of unique dates fall <= cutoff."""
    uniq = np.sort(pd.unique(dates))
    if len(uniq) == 0:
        return None
    idx = min(int(np.floor(len(uniq) * train_frac)), len(uniq) - 1)
    return pd.Timestamp(uniq[idx])

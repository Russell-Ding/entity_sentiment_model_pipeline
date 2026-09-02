"""Map per-article news sentiment to leakage-free forward stock returns.

Produces a joined (ticker, effective_trade_date) dataset with daily sentiment and
THREE forward return windows, all computed from daily OHLC bars. No execution
window is privileged — downstream analysis decides.

Leakage-safe alignment
-----------------------
Each article's UTC timestamp is converted to US/Eastern (DST-aware via zoneinfo),
then assigned an `effective_trade_date` (the first session whose 16:00 ET CLOSE a
daily-bar strategy could still trade on):

    news before 16:00 ET on a trading day   -> effective_trade_date = that day
    news at/after 16:00 ET                   -> next trading day
    news on a weekend / exchange holiday     -> next trading day

A position entered at effective_trade_date T's close therefore uses only
information known by T's close. The trading calendar is taken directly from the
price CSV's date index (so it always matches the prices we join to, and needs no
external calendar dependency). Caveat: NYSE half-days (1:00 PM ET early closes,
~4/year) are treated with the standard 16:00 cutoff; the handful of 13:00-16:00
articles on those days is <0.1% of the corpus.

Return windows (T = effective_trade_date, T1 = next trading session)
--------------------------------------------------------------------
  close_to_close : adj_close[T1] / adj_close[T] - 1        (overnight + next full day)
  overnight      : adj_open[T1]  / adj_close[T] - 1        (the gap reaction)
  intraday_next  : adj_close[T1] / adj_open[T1]  - 1       (next-day intraday drift)

(1 + overnight) * (1 + intraday_next) == (1 + close_to_close) exactly — verified.

Only `adjusted_close` is split/dividend-adjusted in the source data; raw open is
adjusted with the same-day factor f = adjusted_close / close before use.

Outputs
-------
outputs/inference/sentiment_returns_long.jsonl   one row per (ticker, eff_date)
outputs/inference/sentiment_returns.csv          flat CSV of the same

This script ONLY builds and verifies the joined dataset. Statistical evaluation
(panel regression, IC, etc.) is a separate step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "outputs" / "inference"
PRICE_DIR = REPO_ROOT / "data" / "raw" / "eodhd_bulk_20260518" / "prices"

PRIMARY_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]
ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)  # standard NYSE close; half-days not special-cased (see docstring)

DEFAULT_EXCLUDED_SOURCES = {"mt_newswires"}  # paywall stubs; consistent with daily aggregation

# Known upstream coverage gap (Bloomberg+AP dark) — flag, don't drop, here.
BLACKOUT_START = "2024-11-01"
BLACKOUT_END = "2024-12-31"

LONG_OUT = INFERENCE_DIR / "sentiment_returns_long.jsonl"
CSV_OUT = INFERENCE_DIR / "sentiment_returns.csv"


# ---------------------------------------------------------------------------
# Price loading + trading calendar (derived from the price index itself)
# ---------------------------------------------------------------------------

class PriceSeries:
    """Adjusted OHLC for one ticker, with a next-trading-day lookup.

    The trading calendar IS the set of dates present in the CSV, so it is exactly
    consistent with the bars we compute returns from.
    """

    def __init__(self, ticker: str):
        path = PRICE_DIR / f"{ticker}.US.csv"
        if not path.exists():
            raise FileNotFoundError(f"Price CSV not found: {path}")
        self.ticker = ticker
        # date -> dict(adj_open, adj_close)
        self.bars: dict[str, dict] = {}
        self.dates: list[str] = []
        with path.open() as f:
            for row in csv.DictReader(f):
                d = row["date"]
                if d in self.bars:
                    continue  # skip duplicate dates (would corrupt next_trading_day)
                try:
                    o = float(row["open"]); c = float(row["close"]); ac = float(row["adjusted_close"])
                except (ValueError, KeyError, TypeError):
                    continue
                # Guard both close and adjusted_close == 0 (delisted/penny edge): a zero
                # adj_close would later raise ZeroDivisionError in the return computation.
                if c <= 0 or ac <= 0 or not (math.isfinite(o) and math.isfinite(c) and math.isfinite(ac)):
                    continue
                f_adj = ac / c  # same-day split/div factor
                self.bars[d] = {"adj_open": o * f_adj, "adj_close": ac}
                self.dates.append(d)
        self.dates.sort()
        self._date_to_idx = {d: i for i, d in enumerate(self.dates)}

    def is_trading_day(self, d: str) -> bool:
        return d in self.bars

    def next_trading_day(self, d: str, inclusive: bool = False) -> Optional[str]:
        """Smallest trading date >= d (inclusive) or > d (exclusive). None if past end."""
        import bisect
        i = bisect.bisect_left(self.dates, d)
        if inclusive and i < len(self.dates) and self.dates[i] == d:
            return self.dates[i]
        if not inclusive and i < len(self.dates) and self.dates[i] == d:
            i += 1
        elif i < len(self.dates) and self.dates[i] != d:
            pass  # bisect_left already points at first date > d when d not present
        return self.dates[i] if i < len(self.dates) else None


def effective_trade_date(utc_ts: str, prices: PriceSeries) -> Optional[str]:
    """Leakage-safe session a position could be entered at, given a UTC timestamp."""
    try:
        dt = datetime.fromisoformat(utc_ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    et = dt.astimezone(ET)
    d = et.date().isoformat()
    if prices.is_trading_day(d) and et.time() < MARKET_CLOSE:
        return d
    # after-close on a trading day, OR weekend/holiday -> next trading session
    return prices.next_trading_day(d, inclusive=False)


# ---------------------------------------------------------------------------
# Sentiment aggregation keyed on effective_trade_date
# ---------------------------------------------------------------------------

def iter_enriched(path: Path, corrupt: dict) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                corrupt[path.name] = corrupt.get(path.name, 0) + 1


def build(excluded_sources: set[str]) -> tuple[list[dict], dict]:
    prices: dict[str, PriceSeries] = {}
    for tk in PRIMARY_TICKERS:
        try:
            prices[tk] = PriceSeries(tk)
        except FileNotFoundError:
            pass

    # (ticker, eff_date) -> aggregation accumulator
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "w_sum": 0.0, "w_cnt": 0, "w_sq": 0.0,
        "n_articles": 0, "total_mentions": 0,
        "article_ids": set(), "session_buckets": defaultdict(int),
    })

    stats = {
        "files_read": 0, "files_missing": [], "articles_seen": 0,
        "articles_excluded": 0, "no_eff_date": 0, "no_price_ticker": 0,
        "corrupt_lines": {}, "session_buckets": defaultdict(int),
        "shifted_to_next_session": 0,
    }

    for primary in PRIMARY_TICKERS:
        path = INFERENCE_DIR / f"{primary}.US.t1_sentiment_enriched.jsonl"
        if not path.exists():
            stats["files_missing"].append(path.name)
            continue
        stats["files_read"] += 1
        for art in iter_enriched(path, stats["corrupt_lines"]):
            stats["articles_seen"] += 1
            ct = (art.get("content_type") or "").strip().lower()
            if ct in excluded_sources:
                stats["articles_excluded"] += 1
                continue
            utc_ts = art.get("date")
            article_id = art.get("article_id")
            if not utc_ts or not article_id:
                continue
            for ts in art.get("ticker_sentiments", []) or []:
                ticker = ts.get("ticker")
                if not ticker:
                    continue
                # Normalize to dash class-share form (BRK.B / "BRK B" -> BRK-B) so
                # primary-ticker rows are never silently dropped to no_price_ticker.
                ticker = str(ticker).strip().upper().replace(".", "-").replace(" ", "-")
                if ticker not in prices:
                    stats["no_price_ticker"] += 1
                    continue
                eff = effective_trade_date(utc_ts, prices[ticker])
                if eff is None:
                    stats["no_eff_date"] += 1
                    continue
                # session bucket diagnostic (relative to the article's own UTC date).
                # Use the parsed UTC date, not a string slice, so format drift can't
                # mislabel the shift count.
                try:
                    naive_date = datetime.fromisoformat(utc_ts).date().isoformat()
                except (ValueError, TypeError):
                    naive_date = str(utc_ts)[:10]
                if eff != naive_date:
                    stats["shifted_to_next_session"] += 1
                sent_raw = ts.get("sentiment")
                ment_raw = ts.get("total_mentions") or 0
                if sent_raw is None:
                    continue
                try:
                    sent = float(sent_raw); ment = int(ment_raw)
                except (TypeError, ValueError):
                    continue
                if ment <= 0 or not math.isfinite(sent):
                    continue
                key = (ticker, eff)
                a = agg[key]
                if article_id in a["article_ids"]:
                    continue  # dedup across feeds
                a["article_ids"].add(article_id)
                a["w_sum"] += sent * ment
                a["w_cnt"] += ment
                a["w_sq"] += sent * sent * ment
                a["n_articles"] += 1
                a["total_mentions"] += ment

    # Build rows: join to prices, compute the 3 forward returns
    rows: list[dict] = []
    n_no_t1 = 0
    decomp_ok = 0
    decomp_bad = 0
    for (ticker, eff), a in agg.items():
        ps = prices[ticker]
        if not ps.is_trading_day(eff):
            # eff is always a trading day by construction; defensive guard
            continue
        t1 = ps.next_trading_day(eff, inclusive=False)
        wc = a["w_cnt"]
        mean = a["w_sum"] / wc if wc else None
        std = math.sqrt(max(a["w_sq"] / wc - mean ** 2, 0.0)) if wc else None

        c_t = ps.bars[eff]["adj_close"]
        cc = on = idn = None
        if t1 is not None:
            c_t1 = ps.bars[t1]["adj_close"]
            o_t1 = ps.bars[t1]["adj_open"]
            # c_t and c_t1 are > 0 by construction (filtered on load). Guard o_t1
            # defensively in case a bad open slipped through.
            if c_t > 0 and o_t1 > 0:
                cc = c_t1 / c_t - 1.0
                on = o_t1 / c_t - 1.0
                idn = c_t1 / o_t1 - 1.0
                # verify multiplicative decomposition
                if abs((1 + on) * (1 + idn) - (1 + cc)) < 1e-9:
                    decomp_ok += 1
                else:
                    decomp_bad += 1
        else:
            n_no_t1 += 1

        in_blackout = BLACKOUT_START <= eff <= BLACKOUT_END
        rows.append({
            "ticker": ticker,
            "effective_trade_date": eff,
            "next_trade_date": t1,
            "sentiment_mean": round(mean, 6) if mean is not None else None,
            "sentiment_std": round(std, 6) if std is not None else None,
            "n_articles": a["n_articles"],
            "total_mentions": a["total_mentions"],
            "ret_close_to_close": round(cc, 8) if cc is not None else None,
            "ret_overnight": round(on, 8) if on is not None else None,
            "ret_intraday_next": round(idn, 8) if idn is not None else None,
            "in_coverage_blackout": in_blackout,
        })

    rows.sort(key=lambda r: (r["effective_trade_date"], r["ticker"]))
    stats["n_rows"] = len(rows)
    stats["rows_without_t1"] = n_no_t1
    stats["decomp_ok"] = decomp_ok
    stats["decomp_bad"] = decomp_bad
    return rows, stats


def write_outputs(rows: list[dict]) -> None:
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with LONG_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["ticker", "effective_trade_date", "next_trade_date", "sentiment_mean",
            "sentiment_std", "n_articles", "total_mentions", "ret_close_to_close",
            "ret_overnight", "ret_intraday_next", "in_coverage_blackout"]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map news sentiment to leakage-free forward returns")
    ap.add_argument("--exclude-sources", type=str, default=",".join(sorted(DEFAULT_EXCLUDED_SOURCES)),
                    help="content_type values to drop (paywall stubs). Empty string keeps all.")
    args = ap.parse_args()
    excluded = {s.strip().lower() for s in args.exclude_sources.split(",") if s.strip()}

    rows, stats = build(excluded)
    write_outputs(rows)

    print("=" * 72)
    print("Sentiment -> forward-return mapping (leakage-safe, effective_trade_date)")
    print("=" * 72)
    print(f"Files read              : {stats['files_read']} / {len(PRIMARY_TICKERS)}")
    if stats["files_missing"]:
        print(f"Files missing           : {', '.join(stats['files_missing'])}")
    print(f"Articles scanned        : {stats['articles_seen']:,}")
    print(f"Excluded (paywall stubs): {stats['articles_excluded']:,}")
    print(f"Shifted to next session : {stats['shifted_to_next_session']:,}  "
          f"(after-close / weekend news rolled forward — leakage prevention)")
    print(f"(ticker, eff_date) rows : {stats['n_rows']:,}")
    print(f"Rows without T+1 price  : {stats['rows_without_t1']:,}  (most recent days; no forward bar yet)")
    print(f"Decomposition check     : {stats['decomp_ok']:,} OK, {stats['decomp_bad']:,} mismatched "
          f"(overnight x intraday == close-to-close)")
    if stats["corrupt_lines"]:
        print(f"Corrupt lines           : {sum(stats['corrupt_lines'].values())}")
    if rows:
        dates = [r["effective_trade_date"] for r in rows]
        print(f"Date range              : {min(dates)} -> {max(dates)}")
    print(f"\nWrote: {LONG_OUT}")
    print(f"Wrote: {CSV_OUT}")


if __name__ == "__main__":
    main()

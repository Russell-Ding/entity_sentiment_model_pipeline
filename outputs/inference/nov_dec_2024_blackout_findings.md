# Nov-Dec 2024 T1 Coverage Blackout — Findings (2026-05-31)

## Summary

**A system-wide T1 (wire-grade) coverage blackout in Nov-Dec 2024, caused by an upstream EODHD data gap — not a tiering or model bug.** Bloomberg and AP wire content vanished entirely from the raw feed for two months; the tiering logic is working correctly.

## Scope: system-wide, not NVDA-specific

8 of 10 tickers dropped to **exactly zero T1 articles** in both Nov and Dec 2024:

| Ticker | Oct T1 | Nov T1 | Dec T1 |
|---|---:|---:|---:|
| AAPL | 32 | **0** | **0** |
| MSFT | 22 | **0** | **0** |
| GOOGL | 49 | **0** | **0** |
| AMZN | 19 | **0** | **0** |
| NVDA | 32 | **0** | **0** |
| TSLA | 40 | **0** | **0** |
| BRK-B | 12 | **0** | **0** |
| JPM | 26 | **0** | **0** |
| META | 48 | 16 | 19 |
| V | 36 | 31 | 36 |

Raw article volume stayed substantial (NVDA had 226 raw in Nov), so the articles exist — none were tiered T1.

## Root cause: upstream wire-feed outage

Combined T1 by source, Sep 2024 → Mar 2025:

| Month | Total T1 | Reuters | Bloomberg | AP |
|---|---:|---:|---:|---:|
| 2024-09 | 430 | 226 | 173 | 31 |
| 2024-10 | 316 | 160 | 125 | 20 |
| **2024-11** | **47** | 47 | **0** | **0** |
| **2024-12** | **55** | 55 | **0** | **0** |
| 2025-01 | 305 | 145 | 76 | 9 |
| 2025-02 | 410 | 176 | 147 | 24 |

- **Bloomberg and AP went completely dark** in Nov-Dec 2024 and recovered in Jan 2025.
- **Reuters was reduced** (47, 55 vs normal 125-226) but partial.
- In the raw data for the blackout months, the only detected sources are Motley Fool, Yahoo Finance, Benzinga, Business Wire — all correctly classified T3. No Reuters/Bloomberg/AP wire content is present to tier.
- The collection script (`collect_eodhd_sp500_bulk.py`) does no source filtering on fetch, so this is a provider-side gap, not a client bug (Kimi-confirmed).

## The META/V "exceptions" are illusory

META (16/19) and V (31/36) appear to survive, but their Nov-Dec T1 articles are off-topic `reuters.com` URL matches caught by the `reuters_url` rule — e.g. "Chilean copper production up 9.8% in November", "Poland Spring litigation", "Amazon accused of mismanaging forfeited [assets]". These are not real META/V coverage; they're the known reuters-URL false-positive tail.

## Action taken

**Document only** — no data fix. The blackout is 1.5 years old (dump is May 2026); EODHD's historical API almost certainly cannot backfill Nov-Dec 2024 wire content. The P0 `coverage_gap` flag (added to `build_daily_ticker_sentiment.py`) automatically marks the first post-blackout reading in Jan 2025 for affected tickers, so downstream analysis knows those readings follow a coverage gap.

**Kimi verdict:** Diagnosis sound (upstream gap, not client bug). Action = document only; the coverage_gap flag handles the rebound. No re-pull worth attempting.

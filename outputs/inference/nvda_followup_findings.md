# NVDA Time-Series Follow-Up Investigations (2026-05-31)

Investigates the three concerns Kimi raised on the original NVDA time-series review.

---

## (a) May 24, 2023 earnings: data coverage gap, not model failure

**Finding: The May 2023 anomaly is a missing-data problem.**

Daily NVDA self-sentiment, May 22 – June 2, 2023:

| Date | T1 Articles | Notes |
|---|---|---|
| 2023-05-22 | 2 | pre-earnings |
| 2023-05-23 to 2023-05-29 | **0** | **earnings + reaction window — entirely absent** |
| 2023-05-30 | 5 | coverage resumes |
| 2023-05-31 | 8 | |
| 2023-06-01 | 18 | post-event T1 explosion |
| 2023-06-02 | 9 | |

NVDA Q1 FY24 earnings dropped after-market May 24, 2023; the stock jumped +24% on May 25 on $11B guidance vs $7.2B consensus. **The single most important week in NVDA's history has zero T1 wire coverage in our dataset.** The raw EODHD bulk shows T2/T3 articles on those days (Yahoo Finance, Yahoo SEO Listicles, Investing.com) but no T1 (Reuters/Bloomberg/AP) entries.

**Pre/post comparison** (excluding the gap):
- Pre-earnings (May 1-23): mention-weighted +0.170
- Post-earnings (May 25 - Jun 15): mention-weighted +0.265
- Delta: **+0.095** (clear positive shift)

**Kimi confirmed:** "Spot-checked the raw retiered data. Not a tiering bug — no wire prefixes appear in any article content on those days. Data coverage gap, not aggregation or model error."

---

## (b) Mention-weighted vs article-mean: small bias, doesn't explain May 2023

**Finding: Mention-weighting is systematically higher by ~0.03, but direction-preserving (r=0.94). Does NOT explain May 2023.**

Across 48 NVDA-months with ≥10 articles:
- **Pearson correlation between methods: 0.9377**
- Mean absolute divergence: 0.048
- Months where |diff| > 0.05: 20/48
- Months where |diff| > 0.10: 4/48 (2021-04, 2022-03, 2022-08, 2025-06)

**May 2023 specifically:**
- Article-mean: +0.156
- Mention-weighted: +0.140
- Diff: -0.016 (negligible)

Big-divergence months tend to be event-heavy periods where a few long high-mention articles dominate the count. Worth flagging in time-series exports but not a correctness bug.

---

## (c) Source-mix stability: 2 real regime changes

**Finding: Two non-trivial shifts confirmed; 2025 volume jump is partly artifactual.**

Year-over-year cosine similarity on T1 source distribution:
| Year transition | Cosine sim | Interpretation |
|---|---|---|
| 2020→2021 | 0.929 | sourcing onboarding (small 2020 sample) |
| 2021→2022 | **0.999** | stable |
| 2022→2023 | 0.921 | Reuters share rose 48%→65%, Bloomberg dropped 50%→28% |
| 2023→2024 | 0.986 | stable |
| **2024→2025** | **0.907** | **biggest shift — MT Newswires entered at 21% of T1** |

**NVDA T1 article volume:**
- 2023: 50.8 articles/month
- 2024: 66.1 articles/month
- 2025: **284.2 articles/month** (4.3× jump)

Of 2025 NVDA T1 articles, 21.5% are from MT Newswires (a source not in T1 pre-2025). So the volume jump is partly real (AI narrative intensified) and partly artifactual (new source added).

---

## Kimi Review (2026-05-31)

> **Sound** — all three reads confirmed. Spot-checked the raw eodhd_bulk_20260518 data; tables validate exactly.

**Additional issues Kimi flagged:**

1. **Nov-Dec 2024 is a TOTAL T1 blackout for NVDA** (and possibly all tickers). NVDA has 226 (Nov) and 212 (Dec) raw articles but **zero T1**. Suggests a two-month premium-source outage worth scoping across all 10 tickers.

2. **MT Newswires stubs are paywall blurbs with no real text** (~300 chars, no actual wire content). Their *intent* is T1 but their *signal content* is near-zero. Recommends demoting them to T2 or dropping entirely.

3. **August 2025 may be anomalously high** even within 2025 — possible duplicate/backfill issue. (Note: my data shows Aug 2025 at 378 NVDA articles, which is high but within the 2025 baseline of ~284/month, not a 4× outlier.)

## Recommended actions (P0 → P2)

| Priority | Action |
|---|---|
| **P0** | Treat May 2023 as a known coverage gap, not a model bug. Add a `coverage_gap` flag to time-series exports when T1 count = 0 for ≥3 consecutive days. |
| **P0** | Decide MT Newswires policy before doing YoY comparisons. Recommendation: demote to T2 — the stubs have no extractable text. |
| **P1** | Scope the Nov-Dec 2024 blackout across all 10 tickers (is it NVDA-only or a broader outage?). |
| **P2** | Recompute source-mix cosine after MT Newswires exclusion to isolate the real source-identity shift. |

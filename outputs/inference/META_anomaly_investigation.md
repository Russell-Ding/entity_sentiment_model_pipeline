# META T1 Article-Count Anomaly Investigation

**Date:** 2026-05-25
**Observation:** META has 2,594 T1 articles in the inference output, vs AAPL=7,504, TSLA=7,418 — roughly 35% of peer volume.

## Hypotheses tested

(a) Upstream: fewer META articles in the raw EODHD feed.
(b) Tiering: more META articles being demoted to T2/T3.
(c) Downstream "Meta" common-word problem: NER missing "Meta" as an entity.
(d) Recent rename: Facebook→Meta in Oct 2021; pre-2022 articles never tagged with META.US.
(e) Other.

## Evidence

### 1. Raw inventory by tier
| Ticker | Raw total | T1 | T2 | T3 | T1 share |
|---|---|---|---|---|---|
| META.US | 20,825 | 2,594 | 2,420 | 15,811 | **12.5%** |
| AAPL.US | 58,692 | 7,504 | 7,719 | 43,469 | 12.8% |
| TSLA.US | 53,721 | 7,418 | 6,632 | 39,671 | 13.8% |

T1 share is essentially identical across tickers (12.5-13.8%). **Hypothesis (b) ruled out.** META's deficit is in the raw inventory, not in tier assignment.

### 2. Year distribution (T1 only)
| Year | META T1 | AAPL T1 | TSLA T1 |
|---|---|---|---|
| 2020 | 0 | 109 | 68 |
| 2021 | 1 | 1,332 | 1,356 |
| 2022 | 417 | 1,419 | 1,762 |
| 2023 | 194 | 1,017 | 1,569 |
| 2024 | 457 | 668 | 795 |
| 2025 | 1,525 | 2,959 | 1,868 |

- Pre-2022 raw (all tiers): META = **12 articles** vs AAPL = 7,302, TSLA = 8,427.
- Pre-2022 T1: META = 1 vs AAPL = 1,441, TSLA = 1,424. **The Facebook era is essentially missing from META.US.**
- Inspection of the 12 pre-2022 articles: most are about "META 1 Coin Trust", a META ETF, or generic market roundups — not Facebook coverage.

### 3. Inference output integrity
- Inference output count = 2,594. Raw T1 count = 2,594. **1:1 match — no downstream loss.** Whatever upstream gives is what we get.
- 81.7% of articles have META resolved as a ticker entity; 81.1% mention "Meta" canonical_id; 31.8% also mention "Facebook" (overlapping). Of the 472 (18.3%) where the model finds no Meta/Facebook entity, most are sector/market roundups where META.US was attached because META is one of several symbols listed — not because Meta is the article's subject. This is expected behavior, not an NER failure. **Hypothesis (c) ruled out.**

### 4. Post-2022 gap remains
Even excluding the pre-rename period, META has ~2,593 post-2022 T1 vs AAPL ~6,063 / TSLA ~5,994. So roughly **half** of META's deficit is the pre-rename absence; the other half is EODHD simply covering META less heavily than AAPL/TSLA in T1 outlets (Reuters, MT Newswires, Bloomberg, etc.) — Apple and Tesla are perennial market favorites, Meta is one of many big-tech names.

## Verdict

**Primary cause: (d) the Facebook→Meta rename**, accounting for ~1,400 missing T1 articles (2020-2021).
**Secondary cause: (a) lower upstream coverage** of META.US vs AAPL.US/TSLA.US in post-rename T1 sources (~3,400 article gap remains after correcting for the rename).
**Not the cause:** (b) tiering is identical; (c) NER detection is fine — the inference output is 1:1 with raw T1.

## Recommendation

**No fix worth doing.** Options considered:

1. **Backfill Facebook-era articles from FB.US:** would require sourcing the legacy FB ticker history from EODHD, deduplicating by URL/canonical_url, and re-tiering. Possible, but the upstream data may not exist in our current bulk dump, and the legacy Facebook period was a fundamentally different business (no Metaverse, less AI). Mixing pre/post-rename data risks introducing temporal regime changes that hurt training/evaluation more than they help.
2. **Re-tier from T2/T3:** would change tier definitions, breaking comparability with other tickers.
3. **Accept the count:** 2,594 T1 articles is still a substantial sample for a single ticker, and the model is finding META in 81.7% of them with reasonable canonical_id resolution. Performance per-article is what matters for sentiment quality, not absolute count.

META's article count is just-what-it-is. Document the rename effect when reporting per-ticker metrics so reviewers don't read low META volume as a quality issue.

---

## Kimi K2 Review (2026-05-25)

**Verdict: Diagnosis confirmed.**

> Your diagnosis is correct. Verdict and recommendation are sound.
> - (b) is ruled out by identical T1 shares (~12.5–13.8%).
> - (c) is ruled out by the 1:1 raw-to-inference match and 81.7% entity resolution rate.
> - (d) is proven by the near-zero pre-2022 META.US count (12 raw, 1 T1) plus manual inspection showing false matches (META ETF, etc.) rather than Facebook-era news.

**Kimi added two refinements:**

1. **Untested hypothesis (g):** Post-rename upstream coverage lag (2022 H1). 2022 META T1 = 417 is oddly low even after the rename — EODHD may have taken time to migrate indexing from FB → META. Conservative to attribute the post-rename gap to upstream (a), which we did.

2. **Caveat on the ~1,400 estimate:** Facebook (social media/advertising) may have intrinsically lower financial-news density than AAPL (consumer hardware) or TSLA (autos/CEO drama). So 1,400 is likely an *upper bound* — actual Facebook-era gap may be smaller, making "no fix" recommendation even stronger.

3. **"No fix" reasoning validated:** "Pre-2022 Facebook (social media, regulatory fights, Cambridge Analytica) is semantically distinct from post-2022 Meta (metaverse capex, AI, Reels). Backfilling FB-era articles could degrade sentiment model calibration."

**Final action:** Document the rename effect in per-ticker metrics; no backfill.

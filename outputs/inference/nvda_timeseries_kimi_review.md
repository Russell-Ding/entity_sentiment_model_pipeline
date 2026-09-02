# NVDA Time-Series Kimi Review (2026-05-25)

## Overall Verdict
The **broad regime shift is probably real** (pre-AI baseline vs. 2023-2024 AI boom), but **monthly granularity is noisy enough that specific inflection points are unreliable**. Weak-to-moderate signal, not strong.

## Critical Findings

### 1. May 2023 RED FLAG
NVDA's seminal AI earnings call was May 24, 2023 — the moment that defined the AI narrative. Our model shows:
- April 2023: +0.26
- **May 2023: +0.14 (local MINIMUM)** ← should have been the peak
- June 2023: +0.23

With 30 articles / 132 mentions, this is not sample-size noise. Kimi's hypothesis: the monthly aggregation buries the post-May-24 spike because most May articles are pre-earnings preview pieces. **Needs daily granularity to verify.**

### 2. ChatGPT inflection (Nov 2022) is unreliable
Nov 2022 = +0.13, Dec 2022 = +0.12 (only 4 articles!). The "inflection" started in Oct (+0.086), not Nov. Better story: "sentiment recovered from late-2022 crypto/GPU crash, AI demand added a boost mid-2023" — not a clean ChatGPT marker.

### 3. Mention-weighted mean may be systematically biased
Articles with repeated NVDA mentions get weighted higher. Repeat-mention articles tend to be promotional ("NVDA is a buy" thrice). Balanced analytical pieces mention the ticker once. **Recommendation: re-run with simple article-level mean and compare.**

### 4. <10 article filter creates selection bias
Months with <10 articles are dropped. 2020 has only 5 months of data; 2021 missing Feb/Mar/Jul. The pre-2022 baseline is built on a non-continuous, possibly biased sample.

### 5. 2025 volume explosion
2024 monthly volume ~50-100. Late 2025 jumps to 200-400/month. Could reflect:
- Genuine surge in NVDA coverage (plausible — the AI boom intensified)
- OR data source mix changing over time (different outlets, different baselines)

Need to verify source composition is stable before reading 2025 softness (Apr -0.09) as narrative.

## Validated Findings

- **Aug 2022 (-0.18)** aligns with crypto crash + Ethereum merge killing GPU mining demand. Looks well-calibrated.
- **Dec 2021 (-0.12)** aligns with ARM-deal regulatory scrutiny + Omicron tech correction. Plausible.
- **Magnitudes are reasonable.** Monthly stds of 0.20-0.30 imply individual articles span a healthy range. Compression at the monthly aggregate level is just CLT, not model failure.

## Recommended Follow-Up

Before publishing NVDA narrative validation:
1. Check daily granularity around May 24, 2023 — did the model spike post-earnings?
2. Re-run monthly aggregation with article-level mean (not mention-weighted) — compare
3. Verify source composition stability across years before interpreting 2025 trends

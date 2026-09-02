# Can model-derived sentiment drive a trading signal? — Literature review

**Purpose.** First-pass survey of prior research on (1) whether news sentiment predicts
stock returns, and (2) how practitioners turn sentiment scores into a tradable signal.
Goal is to pick a first modeling approach to test on our own data.

**Our starting ingredients (already in this repo):**

- **v2.1 sentiment model** (`trained_model/v2.1_20260620/`): entity-level sentiment for
  ORG / TICKER / PERSON in financial news, scored −0.95…0.95, *implication for the entity*
  (not article tone). Model card notes **TICKER sentiment is strongest (r ≈ 0.81)** and is
  "most suitable for ticker→returns signals."
- **Pre-computed inference** (`outputs/inference/*_enriched.jsonl`): 501 tickers, per-article
  entity + aggregated `ticker_sentiments`, each tagged with a `trade_date` (~2021–2025).
- **Daily prices** (`data/raw/eodhd_bulk_20260518/prices/*.csv`): OHLCV + `adjusted_close`.

So we already have the two halves a signal needs — point-in-time ticker sentiment and
daily returns. This review is about how to connect them defensibly.

Scope per request: **academic literature on news sentiment & returns** + **signal
construction methods**. (ML predictor architectures noted only where directly relevant.)

---

## Part A — Does news sentiment predict returns? (the academic case)

The short answer from ~20 years of research: **yes, but the effect is small, short-lived,
and mostly a cross-sectional / drift phenomenon**, not a market-timing magic wand.

### A1. Foundational papers

- **Tetlock (2007), "Giving Content to Investor Sentiment."** The origin point. Pessimism
  in the WSJ "Abreast of the Market" column predicts downward price pressure followed by
  reversion to fundamentals, plus higher trading volume. Magnitude is tiny: a one-SD
  pessimism shock moves the conditional DJIA return by ~**5.5 bps**. Establishes that
  *tone* carries information beyond hard news.
- **Tetlock, Saar-Tsechansky & Macskassy (2008), "More Than Words."** Moves from market-level
  to **firm-level**: negative words in company-specific news predict lower earnings and lower
  returns, and **prices under-react** to the negative tone (i.e. there is drift to trade).
  This is the conceptual basis for an entity/ticker-level signal like ours.
- **García (2013), "Sentiment During Recessions."** The predictive content of news sentiment
  is **concentrated in recessions** — the effect is state-dependent, not constant. Important
  for regime-aware backtests.
- **Baker & Wurgler (2006), "Investor Sentiment and the Cross-Section of Stock Returns."**
  Sentiment matters most for hard-to-value / hard-to-arbitrage stocks (small, young, volatile,
  non-dividend-paying). Tells us *where* in the cross-section to expect the strongest effect.

### A2. Text-based return prediction (the quant/ML bridge)

- **Ke, Kelly & Xiu, "Predicting Returns with Text Data" (NBER w26186).** A supervised
  topic-model (SESTM) that learns sentiment *from realized returns* rather than a fixed
  dictionary. A long-short portfolio on the resulting score earns large paper Sharpe ratios;
  key lesson is that **learning the sentiment→return mapping end-to-end beats off-the-shelf
  lexicons.** Directly relevant: our v2.1 score is a learned signal, not a dictionary count.
- **Fed working paper, "Predicting Stock Returns from News Stories" (2016).** Firms with
  good news over a week subsequently outperform firms with bad news, and **predictability
  persists ≥ 3 months** — evidence of slow information diffusion (drift) we can harvest.

### A3. The LLM era (most relevant to our setup)

- **Kirtac & Germano (2024), "Sentiment trading with large language models"** (*Finance
  Research Letters*; arXiv 2412.19245). 965k US news articles 2010–2023. A GPT-3-class model
  (OPT) hits 74.4% directional accuracy vs FinBERT 72.2% and Loughran-McDonald dictionary
  50.1%. A long-short strategy on the LLM score reports **Sharpe ~3.05 after 10 bps costs**.
  Treat the headline Sharpe with heavy skepticism (see Part C), but the **ranking of methods
  is the real takeaway: learned/contextual > FinBERT > dictionary.**
- **GPT-3.5 vs RavenPack comparisons.** LLM-derived sentiment matches a commercial vendor's
  returns; score correlation ~0.59 — agreement on direction, disagreement on magnitude. Our
  v2.1 (continuous, magnitude-calibrated, std-ratio ~1.0) is positioned to exploit magnitude.

### A4. Why entity-level sentiment specifically (our model's edge)

- **SEntFiN 1.0 (entity-aware financial sentiment)** and **EFSA (event-level financial
  sentiment)** and **TABFSA (targeted aspect-based financial sentiment).** Core finding: one
  headline often carries *different* sentiment for *different* entities ("Apple wins bid to
  pause rival's ban" is +Apple / −rival). Article-level sentiment blends these and loses
  signal. **Entity/ticker-resolved sentiment is strictly more informative for a per-stock
  signal** — which is exactly what v2.1 produces and where its measured ticker r ≈ 0.81 lives.
  Caveat from our own model card: it can mis-sign legal-outcome news ("won a lawsuit" scored
  negative), so a legal-news guardrail is worth testing.

**Part A bottom line:** the predictive effect is real but small per-article, decays fast, is
strongest in the cross-section / drift and in hard-to-arbitrage names, and is best captured
by a *learned, entity-resolved* score (us) rather than a dictionary.

---

## Part B — Signal construction methods (turning scores into a tradable signal)

This is the part that matters most for us — the model already works; the question is the
*wrapper* around it. Standard pipeline in the literature:

### B1. Aggregate scores into a per-ticker, point-in-time series

- One article → many entity scores. Aggregate to a **per-ticker, per-day** value (mean is
  standard; consider mention-count or source-tier weighting — we have `source_tier` and
  `total_mentions`). Common window: **previous market open → today's market open**, averaging
  articles in the window, so the value is *known before* the bar you trade on.
- Decisions to test: equal-weight vs mention-weighted; cap/winsorize extremes; include a
  *volume-of-news* feature (article count often predicts magnitude/volatility separately).

### B2. Normalize / neutralize before ranking

- Cross-sectionally **standardize** the daily sentiment (z-score across the universe each day)
  so the signal is comparable across dates and regimes.
- Optionally **neutralize** to sector / size / market beta so you're not just buying a sector
  tilt. Baker-Wurgler implies the residual signal is cleaner in small/volatile names.

### B3. Build the position — two dominant academic recipes

| Method | How | Evidence / typical effect |
|---|---|---|
| **Portfolio sorts (quintile/decile long-short)** | Each day/week, sort the universe on the sentiment score, long top bucket, short bottom, rebalance. | News-sentiment long-short ≈ **70 bps/month (~8.7% annualized)**; news-disagreement variant ≈ 5.4%/yr with ~4.5% 4-factor alpha. The workhorse, easy to interpret. |
| **Fama-MacBeth cross-sectional regression** | Regress forward returns on the sentiment score (+ controls) each period; average the slopes. | Standard significance test for "is the signal priced after controlling for size/value/momentum?" Use this to *validate*, sorts to *trade*. |

Start here before any ML predictor — a sort + Fama-MacBeth tells you if there's signal at all.

### B4. Choose the horizon (event-study / drift framing)

- Frame each news day as an **event**; measure **cumulative abnormal returns (CAR)** over
  trading-day windows after it. This reveals *when* the alpha lives (open auction? day 1?
  drift over 1–3 weeks?).
- The relevant anomaly is **post-earnings/news-announcement drift (PEAD)** — prices under-react
  and drift for days-to-weeks. Text-based drift ("PEAD.txt", Philly Fed) shows the news *tone*
  itself predicts drift. **Optimize the holding period** (test 1/5/10/20-day exits) rather than
  assuming one.

### B5. Mind the decay and the clock

- **Sentiment alpha decays fast.** Signal half-life ≈ `ln(0.5)/ln(φ)` (φ = lag-1
  autocorrelation); news signals are short-lived, with predictive power concentrated near the
  **next open** (overnight news → next-day open CAR correlation ≈ 0.57 for news). Execution
  timing (trade the open auction vs midday) materially changes capture.
- **Turnover is high** — equal-weight news-sentiment strategies run ~**90–94% turnover**, so
  the strategy lives or dies on transaction costs (Part C).

**Part B bottom line:** aggregate to point-in-time per-ticker daily → cross-sectionally
standardize/neutralize → **quintile long-short + Fama-MacBeth** to confirm signal → event-study
to pick the holding window → respect fast decay and high turnover.

---

## Part C — Pitfalls & validation (where sentiment backtests die)

Every impressive Sharpe above (1.88, 2.0, 3.05) should be read against these. Most published
"amazing" results shrink hard out-of-sample.

- **Look-ahead / point-in-time bias** — the #1 killer. The sentiment value must use only
  articles timestamped *before* the trade bar. We have `date` and `trade_date`; confirm
  `trade_date` rolls post-publication news to the next tradable session and never peeks.
- **Survivorship / index reconstitution** — our 500-odd tickers look like *today's* S&P 500;
  testing 2021–2025 on current constituents bakes in survivorship. Use point-in-time membership
  or acknowledge the bias.
- **Transaction costs & turnover** — at ~90% turnover, results are extremely cost-sensitive.
  The FinBERT tech long-short kept Sharpe ~2.0 *only* after modeling 5 bps; always report net.
- **Crowding / alpha decay** — published sentiment signals are widely traded; live alpha is a
  fraction of backtest. Commercial-vendor (RavenPack) overlap means part of our edge may already
  be priced.
- **Multiple testing** — testing many windows/weightings inflates the best Sharpe by luck.
  Hold out a final test period; pre-register the rule.
- **Model-specific error modes** — from our own model card: legal-outcome mis-signs, under-read
  list/secondary mentions, NER coverage ~0.42–0.49 (we only score entities NER detects). These
  add noise/bias to the signal that a clean backtest must tolerate.

---

## Part D — Recommendation: what to test first on our data

A deliberately simple, hard-to-fool baseline before any fancy predictor:

1. **Build the panel.** From `outputs/inference/*_enriched.jsonl`, take per-article
   `ticker_sentiments.sentiment`, aggregate to **per-ticker daily** (mention-weighted mean,
   previous-open→open window). Join to forward returns from `prices/*.csv` `adjusted_close`.
2. **Confirm there's signal (cheap, decisive).** Compute the daily **Information Coefficient**
   (rank corr of sentiment vs next-day/next-5-day return) and run **Fama-MacBeth** with size /
   momentum controls. If IC and the slope are ~0, stop and rethink before building a strategy.
3. **Event study for the horizon.** CAR in [0, +20] trading days after high- vs low-sentiment
   days → read off where the alpha concentrates and set the holding period.
4. **Trade it: quintile long-short**, daily cross-sectional standardized sentiment, **next-open
   execution**, value-weighted, with **5–10 bps costs and turnover reported**. Compare to a
   FinBERT/article-level sentiment baseline to prove the *entity-level* edge is real.
5. **Only then** consider an ML predictor (gradient boosting / panel model) combining sentiment
   with news-volume, source-tier, and dispersion features — but the literature (Ke-Kelly-Xiu)
   says the learned *score* is most of the value; the wrapper rarely needs to be exotic.

**Why this order:** the academic record says the effect is small and decays fast, so the first
job is to *detect and size* it cleanly (IC + Fama-MacBeth + event study), not to maximize a
backtest Sharpe. The portfolio sort is the field-standard way to trade it once detected.

---

## References

**Foundational sentiment & returns**
- Tetlock (2007), Giving Content to Investor Sentiment — https://www.researchgate.net/publication/4992763_Giving_Content_to_Investor_Sentiment_The_Role_of_Media_in_the_Stock_Market
- García (2013), Sentiment During Recessions — https://leeds-faculty.colorado.edu/garcia/media_v33.pdf
- Baker & Wurgler (2006), Investor Sentiment and the Cross-Section of Stock Returns — https://pages.stern.nyu.edu/~jwurgler/papers/sentiment.pdf

**Text-based return prediction**
- Ke, Kelly & Xiu, Predicting Returns with Text Data (NBER w26186) — https://www.nber.org/system/files/working_papers/w26186/revisions/w26186.rev1.pdf
- Fed, Predicting Stock Returns from News Stories (2016) — https://www.federalreserve.gov/econresdata/feds/2016/files/2016048pap.pdf

**LLM-era sentiment trading**
- Kirtac & Germano (2024), Sentiment Trading with Large Language Models — https://arxiv.org/abs/2412.19245 · https://www.sciencedirect.com/science/article/pii/S1544612324002575
- GPT-3.5 vs RavenPack sentiment strategy — https://philippdubach.com/posts/trading-on-market-sentiment/
- Backtesting Sentiment Signals for Trading (arXiv 2507.03350) — https://arxiv.org/pdf/2507.03350
- Dynamic Asset Pricing: FinBERT + Fama-French 5-factor (arXiv 2505.01432) — https://arxiv.org/pdf/2505.01432

**Entity / aspect-level financial sentiment**
- SEntFiN 1.0: Entity-Aware Sentiment for Financial News — https://arxiv.org/pdf/2305.12257
- EFSA: Event-Level Financial Sentiment Analysis — https://arxiv.org/html/2404.08681
- TABFSA: Targeted Aspect-based Financial Sentiment — https://dl.acm.org/doi/10.1145/3580480
- Entity-Level Sentiment Classification in Finance (arXiv 2301.03136) — https://arxiv.org/pdf/2301.03136

**Signal construction, drift & timing**
- Post–Earnings-Announcement Drift (overview) — https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift
- PEAD.txt: Post-Earnings-Announcement Drift Using Text (Philly Fed) — https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/2021/wp21-07.pdf
- Overnight Sentiment and Intraday Return Dynamics (QuantPedia) — https://quantpedia.com/overnight-sentiment-and-the-intraday-return-dynamics/
- Does Overnight News Explain Overnight Returns? (arXiv 2507.04481) — https://arxiv.org/pdf/2507.04481

**Backtesting pitfalls**
- The Critical Pitfalls of Backtesting Trading Strategies — https://starqube.com/backtesting-investment-strategies/
- Look-Ahead Bias Prevention in Quantitative Trading — https://quantjourney.substack.com/p/advanced-look-ahead-bias-prevention
- Backtesting Mistakes That Kill Quant Strategies — https://hedgefundalpha.com/education/backtesting-mistakes-kill-quant-strategies-guide/

---
*Compiled 2026-06-20. First-pass review; effect sizes are as reported by each source and are
pre-cost / in-sample unless noted — treat headline Sharpe ratios skeptically (see Part C).*

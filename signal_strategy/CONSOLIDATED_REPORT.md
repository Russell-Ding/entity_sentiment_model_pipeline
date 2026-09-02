# Can the v2.1 Sentiment Model Trade Stocks? — Consolidated Report

> **Author's note.** Consolidated write-up of `RESULTS.md` and the two external
> reviews. Study designed and directed by Russell Ding; analysis executed with
> Claude Code; independent reviews by Codex and Kimi. "You" in the text refers
> to the author. All conclusions verified by the author.

**Three-agent analysis: Claude (study author) · Codex/GPT (reviewer) · Kimi (reviewer + replication)**
*Compiled 2026-08-09 · Source docs: `signal_strategy/RESULTS.md`, `signal_strategy/notes/reviews/`*

---

## 1. Executive summary

**Question.** The v2.1 entity-sentiment model reads financial news and scores how
positive or negative each article is for each company mentioned. Can those scores
predict stock returns and drive a trading strategy?

**Answer, agreed by all three agents: No.** The model **explains** stock moves —
it does not **predict** them. The strong early results turned out to be a subtle
form of look-ahead bias: about 25% of articles (including the 4–6 pm earnings
spike) were dated to a trading day whose 4 pm close had *already happened* before
the article was published. The market had already reacted; the "prediction" was
really the model agreeing with a move that had already occurred. Once the data is
restricted to news genuinely knowable before the close, predictive power is zero
at every horizon, and every executable trading variant loses money.

**What survives — also agreed by all three agents:**

1. **The model is an accurate measurement instrument.** Its scores line up
   strongly with how the market actually reacts to news. That is valuable for
   monitoring, attribution, and analysis — just not for forecasting returns.
2. **One real predictive result: volatility.** The *magnitude* of sentiment
   (ignoring direction) predicts how much a stock will move the next day, even
   after controlling for recent volatility. Kimi independently verified this
   with an extra control and it held up.

**The pivot both reviewers endorse:** use the model for risk/volatility overlays,
news monitoring, overnight P&L attribution, and a model-QA feedback loop — not
for directional trading.

---

## 2. The data and the strategy design

### Ingredients

| Component | Detail |
|---|---|
| News + sentiment | 118,033 unique tier-1 articles, scored by the v2.1 model into per-ticker sentiment |
| Universe | 502 tickers (current S&P 500 members) |
| Period | 2020-03-23 to 2025-12-31 (1,441 trading days) |
| Prices | EODHD daily OHLC + adjusted close |
| Panel | 60,888 ticker-days; median 45 names per day with news |

### Signal construction

Each article's per-ticker sentiment scores were combined into one number per
ticker per day: **`sent_mw`**, a mention-weighted average (an article that talks
about Apple ten times counts more for Apple than one that mentions it once).
Multiple articles on the same ticker-day are averaged.

### The strategy tested

The classic cross-sectional recipe: every day, **rank** all tickers with news by
sentiment, **buy the top 20%** (most positive news) and **short the bottom 20%**
(most negative), dollar-neutral, hold for H days, and measure whether the
long side beats the short side. Two execution assumptions were compared:

- **"Paper"** — enter at the close of the signal day itself (turns out to be
  infeasible; this is where the look-ahead lived).
- **"Tradable"** — enter at the next morning's open (realistic).

---

## 3. The analysis pipeline (what was actually run)

Five steps, each a standard technique, run in sequence
(`signal_strategy/src/run_all.py`):

**Step 1 — Build the panel** (`build_panel.py`). Join news sentiment to prices;
compute forward returns at 1, 5, and 20 days.

**Step 2 — Information Coefficient (IC)** (`ic_analysis.py`). Each day, take the
rank correlation between today's sentiment and the *next* N days' return, then
average across all days. An IC of zero means the signal knows nothing; even a
good daily equity signal is typically only 0.02–0.05. The t-statistic (NW t,
adjusted for autocorrelation) says whether the average is statistically
distinguishable from zero (roughly, |t| > 2 is significant).

**Step 3 — Fama-MacBeth regression** (`fama_macbeth.py`). Same question but as a
daily regression with control variables (size proxy, 1-month and 12-month
momentum), to check the sentiment effect isn't just momentum or size in disguise.
The slope is "extra bps of next-day return per 1 standard deviation of sentiment."

**Step 4 — Event study** (`event_study.py`). Line up all the extreme-sentiment
days at t=0 and trace the average cumulative abnormal return for the next 20
days, top quintile vs bottom quintile. Shows the *shape* of the reaction and how
long to hold.

**Step 5 — Long-short backtest** (`backtest_ls.py`). The actual strategy: daily
quintile portfolios, both execution modes, transaction costs at 5 and 10 bps per
side, turnover accounting, out-of-sample split (last 30% of dates).

Then two follow-ups triggered by your question about MOC execution:

**Follow-up A — MOC feasibility** (`moc_feasibility.py`). Audited every
article's actual publication timestamp (converted to New York time) against a
15:45 ET market-on-close order cutoff, rebuilt the signal using only genuinely
knowable news, and re-ran the backtest.

**Follow-up B — Feasible IC rerun.** Rebuilt the statistical tests on the
corrected panel (`panel_feasible.parquet`) to see whether the "predictive
content" itself survived. It did not (Section 5).

---

## 4. First-pass results (later shown to be contaminated)

These are the numbers as first computed. They looked like a clean success story —
which is exactly why they're worth seeing before the correction.

**IC (Step 2):**

| Horizon | Mean IC | NW t | % days positive |
|--:|--:|--:|--:|
| 1 day | +0.0204 | 3.65 | 54.7% |
| 5 days | +0.0166 | 2.92 | 53.8% |
| 20 days | +0.0096 | 1.48 | 53.7% |

**Fama-MacBeth (Step 3):** sentiment worth **+10.3 bps per standard deviation**
for next-day returns (t = 6.7), surviving size and momentum controls at 1–5 days.

**Event study (Step 4):** top-vs-bottom quintile spread of **+35.6 bps by day 1**
(t = 10.1), peaking at +39.9 bps around day 8, then flat. The positive-news side
drifts up persistently; the negative-news side mean-reverts.

**Backtest (Step 5):** this is where the story cracked open:

| H | Entry | Gross Sharpe | Gross bps/day |
|--:|:--|--:|--:|
| 1 | Signal-day close ("paper") | **+2.93** | +25.4 |
| 1 | Next open ("tradable") | **−0.72** | −4.5 |
| 8 | Signal-day close | +0.80 | +2.5 |
| 8 | Next open | −0.41 | −1.2 |

The decomposition showed why: of the +36 bps day-one spread, **+40.9 bps is the
overnight gap** (signal-day close to next open) and **−4.9 bps** is the tradable
intraday part. The entire "edge" sat in a window you cannot trade with daily
bars. First-pass verdict at this point: *"genuinely predictive but untradable."*

That verdict was wrong, in an instructive way.

---

## 5. The discovery: it was look-ahead, not prediction

Your MOC question — "can we enter at the signal-day close and exit the next
close?" — forced an audit of *when each article was actually published* versus
which trading day it was assigned to. Finding:

> **~25% of articles were assigned to a trading day whose close happened
> *before* the article existed.** The biggest chunk is the 4:00–6:00 pm
> earnings-release spike: an earnings report published Monday 5 pm was stamped
> `trade_date = Monday`, so the "prediction" measured Monday's close against
> Tuesday's — but by Monday's close the article didn't exist, and by Tuesday's
> open the market had fully reacted to it.

So the model wasn't forecasting the market. The model was reading Monday-evening
news and the market was reading the same news Monday evening — and the panel
construction made that agreement look like foresight.

**Corrected results** (signal restricted to news knowable by 15:45 ET on day t):

| Horizon | Original IC (contaminated) | Feasible IC | Feasible NW t |
|--:|--:|--:|--:|
| 1 day | +0.0204 (t 3.65) | −0.0067 | −1.39 |
| 5 days | +0.0166 (t 2.92) | −0.0020 | −0.39 |
| 20 days | +0.0096 (t 1.48) | +0.0025 | +0.41 |

**Zero predictive content at every horizon.** And the executable strategies:

| Variant (H=1, equal weight) | Gross Sharpe | Gross bps/day |
|---|--:|--:|
| Original signal, close entry (look-ahead) | +2.93 | +25.4 |
| **Feasible signal, MOC close-to-close (implementable)** | **−0.55** | **−3.7** |
| After-hours news, traded at next open | −0.18 | −1.3 |

The feasible signal's overnight gap is +1.5 bps (vs +40.9 contaminated) — news
published during the trading day is already in the price by the close, and
after-hours news is fully repriced by the next open. **The market absorbs this
news faster than any daily-bar strategy can act, at every entry point.**

**The one survivor: volatility.** Sentiment *magnitude* |sent_mw| (feasible
version) predicts next-day *absolute* returns with rank-IC **+0.024 (NW t 5.3)**
after controlling for trailing 21-day realized volatility. Intense news tone
today means a bigger move tomorrow — direction unknown. That is a usable
non-directional signal (risk management, position sizing, options).

---

## 6. External review 1 — Codex (GPT)

Codex reviewed the study read-only (RESULTS.md + all source code) *before* the
feasible-IC rerun existed, and independently reached the same central verdict:

> "The headline claim of 'real predictive content' is not established … [it]
> should be withdrawn" until the IC / Fama-MacBeth / event-study tests are
> rebuilt on timestamp-feasible panels.

That is exactly what Follow-up B then confirmed — the claim died when retested.

**Codex's issue list (ranked by its severity):**

| # | Severity | Issue | Plain-language meaning |
|--:|:--|:--|:--|
| 1 | Critical | IC/FM/event-study all use the contaminated panel | The "prediction" evidence shared the same look-ahead as the backtest — confirmed |
| 2 | High | Event-study t=10.1 assumes independent events | Thousands of overlapping same-day/same-ticker events counted as independent; the true t is far lower |
| 3 | High | "Size" control is same-day dollar volume | It's liquidity, not market cap — and same-day volume *reacts to the news*, so it's a contaminated control |
| 4 | High | Not genuinely out-of-sample | The holding period H=8 was picked using the full sample; the 70/30 "OOS" split is a report, not a holdout |
| 5 | High | Survivorship + missing returns | Universe is *today's* S&P 500 (losers excluded); missing price bars silently count as zero — worst in the short (bad-news) leg |
| 6 | Med-high | Overnight decomposition mixes price bases | Adjusted close-to-close minus raw open-to-close, combined additively — not an exact identity around dividends/splits |
| 7 | Med-high | Calendar/timestamp assumptions unaudited | Naive timestamps assumed UTC; AAPL price dates used as the NYSE calendar; half-days ignored |
| 8 | Medium | Forced quintiles on tiny cross-sections | As few as 10 names → 2-stock legs; ties broken by row order |
| 9 | Medium | Fixed HAC lag 5 for 20-day overlapping returns | Statistical error bars too narrow at long horizons |
| 10 | Medium | Cost model illustrative, not conservative | No spreads, borrow, drift, or auction impact — moot for a negative-gross strategy |
| 11 | Medium | COVID regime concentration | Sample starts at the March-2020 low; no year-by-year breakout was reported |

Codex's bottom line on scope: the honest claim is **"no detected edge," not
"edge is exactly zero"** — the feasible MOC Sharpe of −0.55 carries t ≈ −1.4.

**Codex's suggested next tests** (top of its 12-item list): rebuild every
statistic on point-in-time panels at multiple cutoffs (9:30 / 15:45 / 16:00 /
next open); bucket the original results by publication hour to isolate the
contamination directly; placebo date-shifts (a contaminated signal peaks at the
wrongly-assigned date); portfolio-level HAC inference instead of event-level
t-stats; year-by-year breakouts with frozen rules.

---

## 7. External review 2 — Kimi

Kimi went a step further than reviewing: it **re-ran the numbers itself from the
artifacts on disk**. Its verdict:

> "The study's updated conclusion is correct and the self-diagnosis is genuine,
> not cosmetic … the original 'predictive content' was overwhelmingly
> look-ahead."

**Independent replications (the most valuable part):**

- **A sharper contamination test.** Instead of rebuilding the panel, Kimi simply
  *dropped* the ~25% of ticker-days the feasibility rule reassigns and re-ran
  the IC on what remained: it collapsed from +0.0204 (t 3.65) to **+0.0037
  (t 0.68)**. Over 80% of the headline IC came from the after-hours subset.
- **The volatility result survived a harder test.** Kimi added a control the
  study didn't use — same-day absolute return — on top of trailing volatility.
  Result: partial rank-IC **+0.019 (NW t 3.7)**, stable in every year 2021–2025.
- **Sanity checks passed:** timezone/DST handling verified clean on ~60k
  timestamps; the contamination is *not* a COVID artifact (the original IC is
  positive every single year 2020–2025).

**New problems Kimi found that Codex missed:**

1. **The "late news at next open" test entered one session too late.** The
   late-news panel dated articles to the first knowable *close*, and the
   backtest then entered the *following* open — so Monday-evening news was
   traded at Wednesday's open, not Tuesday's. Kimi ran the correct Tuesday-open
   version itself: also ≈ 0 (−4.9 bps day-1 open-to-close, Sharpe −0.72). The
   conclusion stands, but the test as originally labeled was the wrong test.
2. **The decisive correction isn't reproducible.** `panel_feasible.parquet` and
   the feasible-IC/volatility tables were produced by ad-hoc uncommitted
   scripts. Kimi's recompute gives feasible 1-day IC −0.0006 (t −0.10) vs the
   reported −0.0067 (t −1.39) — same verdict (≈ 0), different exact numbers.
   And Fama-MacBeth + the event study were **never re-run** on the corrected
   panel, so those sections remain contaminated as printed.
3. **A hidden reversal hint.** The pooled feasible IC of ≈ 0 averages away two
   years — 2023 and 2024 — where it is significantly *negative* (t ≈ −2 each):
   possibly a short-horizon post-news reversal effect. Probably noise, but cheap
   to test and untested.
4. **One feasible trading channel was never tested:** pre-market news (~28% of
   articles, published 5–9 am ET) traded at *that same day's* open. The current
   code can only enter the session *after* the signal day, so this — the last
   daily-bar stone unturned — was structurally impossible to express.
5. Smaller: the 15:45 rule ignores 1 pm half-day closes (~2 days/yr); ~7k
   articles timestamped 0–5 am ET deserve a publication-vs-crawl-time audit.

---

## 8. Where the three agents agree and disagree

**Full agreement (high confidence):**

| Point | Claude | Codex | Kimi |
|:--|:--|:--|:--|
| Directional daily trading: dead | yes — tested | yes — predicted it | yes — replicated it |
| Original "prediction" was look-ahead | yes — found via MOC audit | yes — flagged before the rerun | yes — quantified: >80% of IC |
| Model is a good *measurement* instrument | yes — | yes — | yes — |
| Volatility signal is real and the top surviving use | yes — t 5.3 | (post-dates its review) | yes — replicated, t 3.7 with extra control |
| Event-study t=10 overstated (non-independent events) | accepted | yes — flagged | yes — flagged |
| "Size" control is post-treatment dollar volume | accepted | yes — flagged | yes — flagged |
| Weak OOS / multiple-testing discipline | accepted | yes — flagged | yes — flagged (extends it to the vol result) |

**Points of tension (worth knowing, none change the verdict):**

- **Exact feasible-IC numbers.** Claude reported −0.0067 (t −1.39); Kimi's
  recompute from the same parquet gives −0.0006 (t −0.10). Both are "zero";
  the discrepancy exists because the correction was ad hoc, not scripted —
  Kimi's reproducibility criticism, demonstrated on itself.
- **How dead is trading?** Codex and Kimi both stress the negative results are
  "no detected edge," not proof of zero (t-stats around −1.4). And Kimi keeps
  one door ajar: the untested pre-market-news-at-open channel, plus the 2023–24
  reversal hint. Weak priors on both, but they are honest open items rather
  than tested failures.
- **Half-corrected record.** Kimi is right that Fama-MacBeth and the event
  study were never re-run on the feasible panel — the correction so far rests
  on the IC and the backtest only.

---

## 9. Recommendations (merged from all three agents)

### If you want to close out the trading question completely

1. **Commit a `feasible_analysis.py`** that rebuilds `panel_feasible.parquet`
   and re-runs IC + Fama-MacBeth + event study on it, so the correction is a
   runnable script, not a claim. *(Kimi's #1; Codex's #1.)*
2. **Test pre-market news at the same day's open** — the one untested feasible
   channel (~28% of articles). *(Kimi.)*
3. **Check the 2023–24 negative-IC reversal hint** with a proper short-horizon
   reversal test. *(Kimi.)*
4. If any of those show life, apply the full methodology hardening: portfolio-
   level HAC inference, lagged controls, point-in-time universe, frozen rules
   with a one-shot holdout. *(Codex.)*

### The productive pivot: non-trading uses (priority order)

1. **Volatility / risk overlay** — the surviving predictive result. Feed
   |sentiment|, dispersion, and news intensity into volatility forecasts,
   position sizing, VaR, or options strategies. Gate on hardening first:
   earnings-day control, overnight-vs-intraday split of the predicted move,
   a true out-of-sample freeze. *(All three agents' top pick.)*
2. **Overnight P&L attribution** — the model demonstrably explains overnight
   gaps; entity-level granularity lets you attribute a portfolio's overnight
   move to specific companies and articles. Turns the study's fatal flaw into
   the product. *(Kimi #2, Codex #4.)*
3. **News monitoring and alerts** — flag sentiment shocks and deterioration on
   held names; rank the daily news queue by relevance, extremity, and
   corroboration (multi-article days carry more signal — the study's own
   intensity split). Latency-insensitive when framed as a review queue.
4. **Model-QA flywheel** — the panel already holds model sentiment, vendor
   polarity, and realized market reaction side by side; disagreements between
   the three surface mislabeled or hard articles for the next retrain. Directly
   feeds the parent project. *(Codex #7–8, Kimi #4.)*
5. **Slow fundamental features** — weekly/monthly sentiment level, dispersion,
   and negative-tail frequency for slower models. Lower priority: the feasible
   IC is ≈ 0 even at 20 days, so pitch these as features, not signals.

### Explicitly deprioritized

Any directional daily-bar strategy — close entry, next-open entry, MOC — all
tested, all negative, and the mechanism is understood: **the market prices this
news faster than daily bars can act.**

---

## Appendix — mini-glossary

- **bps** — basis points; 1 bp = 0.01%.
- **IC (Information Coefficient)** — daily rank correlation between the signal
  and subsequent returns, averaged over days. 0 = no skill; 0.02–0.05 is
  typical for real (weak) equity signals.
- **NW t** — Newey-West t-statistic: significance adjusted for the fact that
  daily results overlap/correlate. Roughly, |t| > 2 = statistically significant.
- **Fama-MacBeth** — run one cross-sectional regression per day, then average
  the daily slopes; the standard way to test a return predictor with controls.
- **Sharpe ratio** — annualized return divided by annualized risk. > 1 is good;
  negative means it loses money.
- **MOC** — market-on-close order; must be submitted by ~15:45–15:50 ET, which
  is why 15:45 is the study's "knowability" cutoff.
- **Look-ahead bias** — using information in a backtest that was not actually
  available at the assumed decision time. The central villain of this study.
- **Overnight gap** — price change from one day's 4 pm close to the next
  morning's 9:30 open; where most earnings reactions happen, and untradable
  with daily-bar entries.

**File map:** study details `RESULTS.md` · code `src/` · reviewer prompts and
full raw reviews `notes/reviews/{review_prompt.md, codex_review_raw.txt,
kimi_review_raw.txt}` · corrected panel `outputs/panel_feasible.parquet` ·
MOC study `outputs/moc_feasibility.json`.

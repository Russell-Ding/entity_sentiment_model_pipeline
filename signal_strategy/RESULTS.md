# Sentiment → Returns Signal — First Pass Results

> **Author's note.** This study was designed and directed by Russell Ding as an
> independent research project. The analysis code was written with Claude Code
> under the author's direction, and the results were independently reviewed by
> two other model-based reviewers (Codex and Kimi; raw reviews in
> `notes/reviews/`). The document below is kept in its original working form —
> including the first-pass conclusion that later proved wrong — because the
> correction is the point. All conclusions were verified by the author.

**Verdict (updated after the MOC follow-up and external review):** The v2.1
entity-sentiment signal **does not predict returns — it explains them.** The
first-pass headline ("genuinely predictive but untradable") was wrong: the IC /
Fama-MacBeth / event-study evidence was computed on a panel where ~25% of articles
(incl. the 4–6pm earnings spike) carry a `trade_date` whose close *precedes*
publication — the same look-ahead that inflated the close-entry backtest (gross
Sharpe 2.93). Rebuilt with an honest 15:45 ET knowability cutoff, the 1-day IC
collapses from +0.020 (t 3.65) to **−0.007 (t −1.4)** and is ≈ 0 at every horizon;
the implementable MOC close→close strategy has gross Sharpe **−0.55**; after-hours
news traded at the next open is also ≈ 0. What survives:
(a) the signal aligns strongly with the *contemporaneous* overnight reaction —
evidence of **measurement quality**, useful for attribution and monitoring, not
alpha; (b) sentiment **magnitude** genuinely predicts next-day **volatility**
beyond trailing realized vol (rank-IC +0.024, NW t 5.3) — the one live predictive
result, and it is non-directional. **No tradable directional edge at daily
granularity.** (Sections 2–4 below report the original, contaminated numbers for
the record — read them with the follow-up sections.)

Reproduce: `python3 signal_strategy/src/run_all.py` (or `--skip-panel`). All numbers
below come from `signal_strategy/outputs/`.

---

## Coverage (point-in-time panel)

- **60,888 ticker-days**, **502 tickers**, **2020-03-23 → 2025-12-31** (1,441 trading days).
- Median **45 names/day** (p10 = 8, p90 = 64) — a comfortable cross-section to rank.
- Source: 118,033 unique tier-1 articles → `ticker_sentiments` exploded to (ticker,
  trade_date), mention-weighted into `sent_mw`. Joined to EODHD adjusted prices.
- **No look-ahead:** the signal for day *t* uses only news with `trade_date ≤ t`;
  forward returns begin strictly after *t*; the backtest adds a next-open lag.
  (Built in `build_panel.py`; the field `trade_date` already rolls post-publication
  news to the next tradable session.)

## Step 2 — Information Coefficient (daily cross-sectional Spearman, sent_mw vs fwd ret)

| horizon | mean IC | IC IR | NW t | % days +ve |
|--------:|--------:|------:|-----:|-----------:|
| 1d  | 0.0204 | **0.097** | **3.65** | 54.7% |
| 5d  | 0.0166 | 0.080 | 2.92 | 53.8% |
| 20d | 0.0096 | 0.045 | 1.48 | 53.7% |

Signal is real and decays with horizon — strongest at 1 day, insignificant by 20.
Corroborated (multi-article) ticker-days carry slightly more signal than single-
article days (IC 0.0175 vs 0.0137 at 5d). *All inference is tier-1, so the brief's
tier-1 split is degenerate and was replaced by this news-intensity split.*

## Step 3 — Fama-MacBeth (fwd_ret ~ sent_mw + size + mom_21 + mom_252, z-scored daily)

| horizon | sent_mw slope | NW t | priced after controls? |
|--------:|--------------:|-----:|:----------------------:|
| 1d  | **10.29 bps/SD** | **6.70** | yes |
| 5d  | **11.23 bps/SD** | **3.70** | yes |
| 20d | 4.81 bps/SD | 0.88 | no |

Sentiment survives size and momentum controls cleanly at 1–5 days. Detection is
unambiguous.

## Step 4 — Event study (top vs bottom sentiment quintile, CAR vs equal-weight universe)

Long-minus-short cumulative abnormal return: **+35.6 bps by t+1 (t = 10.1)**,
peaking at **+39.9 bps at t+8**, then flat. Asymmetric and informative: the top
quintile drifts up persistently (+33.9 bps by t+20) while the bottom mean-reverts
back toward zero (−1.9 bps). Read-off holding period **H = 8** (used below; the
conclusion is identical at H = 1 or 5). See `event_study_car.png`.

## Step 5 — Long-short backtest, and why the signal is untradable

Daily quintile sort, long Q5 / short Q1, dollar-neutral, overlapping H-day book.
Two execution modes isolate the problem:
- **paper** = enter at the signal *close* (captures the overnight gap; **infeasible**).
- **open**  = enter at the *next session's open* (**realistic**).

**Execution decomposition of the day-1 Q5−Q1 spread (`backtest.json`):**

| component | spread |
|---|--:|
| close[t] → close[t+1] (paper, incl. overnight) | +36.0 bps |
| **close[t] → open[t+1] (overnight gap — uncapturable)** | **+40.9 bps** |
| open[t+1] → close[t+1] (tradable, intraday) | **−4.9 bps** |
| 8-day hold, paper (close-entry) | +38.8 bps |
| 8-day hold, tradable (open-entry) | **−2.1 bps** |

The whole edge is the overnight gap. Entering at the open captures none of it.

**H × execution sweep, equal-weight (`sweep_equal`):**

| H | entry | gross Sharpe | gross bps/day | net@5bps Sharpe | net@10bps Sharpe | turnover/yr |
|--:|:--|--:|--:|--:|--:|--:|
| 1 | paper (close) | **2.93** | 25.4 | 1.12 | −0.69 | 395× |
| 1 | tradable (open) | −0.72 | −4.5 | −3.22 | −5.70 | 395× |
| 5 | paper (close) | 1.14 | 4.3 | 0.28 | −0.58 | 82× |
| 5 | tradable (open) | −0.48 | −1.7 | −1.41 | −2.33 | 82× |
| 8 | paper (close) | 0.80 | 2.5 | 0.15 | −0.50 | 51× |
| 8 | tradable (open) | −0.41 | −1.2 | −1.09 | −1.77 | 51× |

Two independent nails in the coffin:
1. **Tradable (next-open) gross Sharpe is negative at every H** — no alpha to harvest
   once the overnight move is gone.
2. **Even the infeasible paper signal doesn't clear realistic costs.** Its gross
   Sharpe is high (2.93 at H=1) but the alpha is so day-1-concentrated that capturing
   it requires ~400×/yr turnover; it survives 5 bps (Sharpe 1.12) but not 10 bps
   (−0.69). Stretching the hold to lower turnover (H=8, 51×) bleeds the gross to
   2.5 bps/day, which also dies after costs.

**Variants at H = 8, 10 bps, full / OOS (last 30%, from 2024-04-25):**

| strategy | Sharpe | ann. return | max DD | OOS Sharpe |
|---|--:|--:|--:|--:|
| v2.1 EW — paper (close exec, infeasible) | −0.50 | −4.0% | −28.9% | −0.05 |
| v2.1 EW — tradable (open exec) | −1.77 | −13.3% | −55.3% | −1.90 |
| v2.1 VW — tradable (open exec) | −0.96 | −15.5% | −64.2% | −1.73 |
| EODHD baseline — paper (close exec) | −0.62 | −5.1% | −29.9% | −0.95 |
| EODHD baseline — tradable (open exec) | −0.85 | −6.7% | −35.2% | −1.25 |

**Baseline comparison:** the entity-level v2.1 signal has a clearly *stronger paper
edge* than article-level EODHD polarity (v2.1 H=1 paper gross Sharpe 2.93 vs the
baseline's weaker close-exec curve), so the model *does* add cross-sectional ranking
value over off-the-shelf polarity. But both are overnight-gap phenomena and neither
is tradable at the next open. The model is better; the opportunity still isn't there.

See `ls_equity_curve.png`: the paper-gross curve compounds strongly upward while the
tradable-net curve drifts down.

---

## Guardrails / honesty

- **Look-ahead / point-in-time:** signal uses only `trade_date ≤ t`; the next-open
  lag is explicit (`execution.lag_days`). The *only* configuration that "works" is the
  one we flag as infeasible (trading the signal bar's close).
- **The overnight gap is the crux, not a footnote.** Headline event-study/IC numbers
  use close-to-close returns and therefore *include* a return you cannot trade. We
  report it both ways; the tradable number is the honest one.
- **Costs / turnover:** always reported net at 5 and 10 bps/side. The cost model is
  conservative (turnover from day-over-day target weights, ignoring intra-name
  netting), so realistic costs are *no better* than shown.
- **Survivorship:** the ~500 names are today's S&P 500 members — a forward-looking
  universe. This biases *long-only* returns upward; for a dollar-neutral L/S it is
  second-order, but the bias is real and not claimed away.
- **OOS discipline:** rule (H, quintiles, weighting) read off the event study; final
  stats also reported on a held-out last 30% of dates. OOS does not rescue it.
- **Known model error modes (model card):** legal-outcome mis-signs, under-read
  secondary mentions, NER coverage ≈ 0.42–0.49 → noisy ticker-days, kept (not
  silently filtered). These add noise but are not why the signal fails — the failure
  is execution timing, which is upstream of label noise.

## Follow-up — can we enter at the signal close (MOC) and exit the next close?

Tested (`moc_feasibility.py`, outputs in `moc_feasibility.json`). The idea: if the
signal is computable before 4pm, submit market-on-close orders at the signal-day
close and capture the overnight gap legitimately. **It fails, and the diagnosis
reframes the whole study:**

- **`trade_date` assigns after-hours news to the same day.** Timestamp audit
  (publication time in ET vs assigned date): ~25% of articles — including the
  4–6pm earnings-release spike — were **not publicly available by the 15:45 ET
  MOC cutoff** of their assigned day. The "paper close-entry" Sharpe of 2.93 was
  therefore **look-ahead**, not a missed opportunity.
- **Rebuilt the signal honestly** (each article assigned to the first close by
  which it was knowable, 15:45 ET cutoff) and re-ran H=1 close→close:

| variant (H=1, equal weight) | gross Sharpe | gross bps/day |
|---|--:|--:|
| Original signal, close entry (**look-ahead**) | 2.93 | +25.4 |
| **Feasible signal, MOC close→close (implementable)** | **−0.55** | **−3.7** |
| Late (after-hours) news only, next-open entry | −0.18 | −1.3 |

- The feasible signal's day-1 overnight gap is **+1.5 bps** (vs +40.9 with
  look-ahead): news published *during* the trading day is already in the price by
  the close. The after-hours news that drives the gap can't be traded at the prior
  close (it doesn't exist yet) and is fully repriced by the next open (last row).

**Conclusion: the market prices this news faster than a daily-bar strategy can act
at every entry point available.** Intraday knowable → priced by the close.
After-hours → priced by the open.

## Follow-up 2 — the "predictive content" claim does not survive either

The contamination above infects Sections 2–4, not just the backtest: for the ~25%
late-assigned articles, `fwd_ret_1` (close t → close t+1) *contains the market's
own reaction to the article*, so IC/FM/event-study "prediction" is partly
contemporaneous alignment. Rerunning the IC on the feasibility-corrected panel
(`panel_feasible.parquet`, signal = news knowable by 15:45 ET on day t):

| horizon | original IC (contaminated) | feasible IC | feasible NW t |
|--:|--:|--:|--:|
| 1d  | +0.0204 (t 3.65) | **−0.0067** | −1.39 |
| 5d  | +0.0166 (t 2.92) | −0.0020 | −0.39 |
| 20d | +0.0096 (t 1.48) | +0.0025 | +0.41 |

Zero predictive content at every horizon once knowability is enforced. The correct
reading of the strong original numbers is that **the model accurately measures the
sentiment the market itself prices in** — a validation of the model as a
*measurement* instrument, and a negative result for it as a *forecasting* one.

**One predictive result does survive the feasibility correction: volatility.**
|sent_mw| (feasible) predicts next-day absolute returns with cross-sectional
rank-IC **+0.024 (NW t 5.3)** after residualizing on trailing 21-day realized vol
(raw, uncontaminated pilot: +0.05–0.06, t ≈ 12). News tone intensity knows
something about tomorrow's risk that yesterday's vol doesn't — a usable,
non-directional signal for risk management or vol-aware strategies.

## External review (Codex; full text in `notes/reviews/`)

An independent read-only review by Codex (GPT) reached the same central verdict
before seeing the feasible-IC rerun: *"the headline claim of 'real predictive
content' is not established … [it] should be withdrawn"* until IC/FM/event tests
are rebuilt point-in-time — which the table above now does. Other issues it
flagged, accepted and noted here rather than silently fixed:

- **Event-study t ≈ 10 is overstated** — events overlap in time/ticker and are not
  independent; date-clustered or portfolio-series HAC inference is the right way.
- **The "size" control is same-day dollar volume**, not market cap — and same-day
  volume is post-treatment (reacts to the news itself). Should be lagged/renamed.
- **OOS discipline is weak** — H was chosen on the full sample; the 70/30 split is
  a report, not a true holdout. Multiple configurations were inspected untracked.
- **Survivorship + missing-return handling**: current-constituent universe, and
  missing price bars contribute silent zeros without leg re-normalization —
  most dangerous in the short (bad-news) quintile.
- **Decomposition arithmetic**: adjusted close-close vs raw open-close bases are
  mixed and combined additively, not multiplicatively; half-days/DST untested.
- Smaller: forced quintiles on tiny/tied cross-sections, fixed NW lag 5 for 20-day
  overlapping returns, cost model is illustrative rather than conservative,
  2020-COVID regime concentration unexamined.

None of these change the sign of the conclusion (they mostly make the *original*
evidence weaker still), but they bound what this study can claim: **"no detected
edge," not "edge is exactly zero"** — the feasible MOC Sharpe −0.55 has t ≈ −1.4.

## External review 2 (Kimi; full text in `notes/reviews/`)

Kimi went further than a read: it **independently re-ran the core numbers** from
the artifacts on disk. Where it confirms:

- **Look-ahead confirmed, and quantified more sharply.** Its restriction test —
  drop only the ticker-days the feasibility rule reassigns, keep everything else —
  collapses the 1-day IC from +0.0204 (t 3.65) to **+0.0037 (t 0.68)**: over 80%
  of the headline IC came from the ~25% after-hours subset.
- **The volatility result survives a control this study didn't use.** Adding
  same-day |return| on top of trailing vol, Kimi still gets partial rank-IC
  **+0.019 (NW t 3.7)**, stable across 2021–2025. It rates the vol overlay the
  top surviving use, same as here.
- Timezone/DST handling verified clean; the contamination is *not* a COVID
  artifact (original IC is positive in every year 2020–2025).

New issues it found that Codex did not:

- **The "late news at next open" test enters one session too late.** The late-only
  panel dates news to the first *knowable close*, and the backtest then enters the
  *following* open — i.e. Monday-evening news is traded Wednesday, not Tuesday.
  The correct Tuesday-open test (original panel, `entry='open'`, pushed rows) also
  gives ≈0 (−4.9 bps day-1 open→close), so the conclusion stands — but the
  labeled evidence for it in the MOC table was the wrong test.
- **Reproducibility gap on the decisive correction.** `panel_feasible.parquet`
  and the feasible-IC/vol tables were built ad hoc, with no committed script;
  Kimi's recompute gets feasible 1-d IC −0.0006 (t −0.10) vs the −0.0067 reported
  above. Same verdict (≈0), but the exact numbers are not regenerable from the
  repo — and FM + the event study were never re-run on the feasible panel, so
  Sections 3–4 remain contaminated as printed.
- **A reversal hint the pooled t hides:** feasible IC is significantly *negative*
  in 2023 and 2024 (t ≈ −2 each year). Probably noise, but a short-horizon
  news-conditioned reversal test is cheap and untested.
- **One genuinely untested feasible channel remains: pre-market news (~28% of
  articles, 5–9 am ET) traded at that same day's open.** The current machinery
  can only enter the session *after* the signal day, so this was never tested.
  It is the last daily-bar stone unturned.
- Smaller: the 15:45 rule ignores 1 pm half-day closes (~2 days/yr); ~7k articles
  timestamped 0–5 am ET warrant a publication-vs-crawl-time audit.

## If there's a next pass

For **trading**, two channels remain untested: pre-market news traded at that
same day's open (feasible with daily bars — Kimi's point above), and intraday
reaction speed in the minutes after publication, which daily OHLC cannot
evaluate. The prior on both is now weak, since the feasible signal shows no
predictive content even at the close.

The productive pivot is **non-trading uses**, where the model's demonstrated
strength (accurate contemporaneous measurement, entity-level granularity) is the
asset rather than the liability:
1. **Volatility / risk signal** — the surviving predictive result. Feed |sent|,
   dispersion, and negative-tail flags into vol forecasts, position sizing, VaR
   overlays, or options strategies.
2. **Portfolio news monitoring & alerts** — flag sentiment shocks/deterioration on
   held names for human review; triage the daily news queue by entity relevance,
   sentiment extremity, and corroboration.
3. **Overnight P&L attribution** — the model explains overnight gaps well by
   construction; link moves to the specific entities/articles that drove them.
4. **Slow fundamental features** — weekly/monthly sentiment level, innovation,
   dispersion, and tail frequency as inputs to slower models (must be re-tested
   point-in-time, but they don't compete on news latency).
5. **Model QA loop** — use divergence between model sentiment, vendor polarity,
   and realized reaction to mine mislabeled/hard articles for the next retrain.

# Financial Entity Sentiment → Trading Signal

**An end-to-end, independently built research pipeline:** a Longformer-based model that reads
financial news and scores sentiment *per entity* (companies, tickers, people), trained by
distillation from ~30k LLM-labelled articles — followed by a rigorous test of whether those
scores carry a tradable signal for S&P 500 stocks.

The short answer to the trading question is **no** — and the project is organised around
showing *how* that answer was reached honestly, including the look-ahead bias that made
the first pass look like a Sharpe-2.9 strategy, the point-in-time correction that killed
it, and the one result that survived (sentiment intensity predicts next-day volatility).

**Start with the report:** [`docs/project_report.pdf`](docs/project_report.pdf) covers the
model structure, the training procedure, the sentiment-prediction validation, the return and
volatility tests, and (in the appendix) the labelling prompts. This README is the map of the
repository.

> Everything here was built on my own time, on my own hardware/Colab, with my own data
> subscription. No employer code or data is involved.

---

## Headline results

**Model (entity-level sentiment, in-distribution test — 1,558 articles / 8,078 entities)**

| Metric | v2.0 | **v2.1** |
|---|---|---|
| Pearson r (pred vs. label) | 0.485 | **0.643** [0.626, 0.660] |
| Concordance corr. (CCC) | 0.416 | **0.641** |
| pred/label std ratio | 0.57 | **1.03** — magnitude compression fixed |
| TICKER / ORG / PERSON r | 0.62 / 0.49 / 0.49 | **0.81 / 0.63 / 0.68** |
| Entity coverage of the end-to-end pipeline (same tagger) | 48.9% | 48.4% |

(The v2.0 column is measured on the full new-label set, v2.1 on the held-out test split;
see report §5.)

**Signal study (502 S&P 500 names, 2020-03 → 2025-12, 60,888 ticker-days; scores from the
v2.0 checkpoint)**

| Test | Naïve panel | Point-in-time panel (15:45 ET cutoff) |
|---|---|---|
| 1-day rank IC (Newey-West t) | +0.020 (t 3.7) | **−0.007 (t −1.4) ≈ 0** |
| Fama-MacBeth slope, 1d | +10.3 bps/σ (t 6.7) | — (contaminated; see report) |
| L/S quintile, H=1, close entry | Sharpe **+2.93** (look-ahead) | Sharpe **−0.55** (implementable MOC) |
| \|sentiment\| → next-day \|return\|, controlling for trailing vol | — | **rank-IC +0.024 (t 5.3)** — survives |

Full details: report §6, [`signal_strategy/RESULTS.md`](signal_strategy/RESULTS.md) and the
consolidated three-reviewer write-up
[`signal_strategy/CONSOLIDATED_REPORT.md`](signal_strategy/CONSOLIDATED_REPORT.md).

---

## Part 1 — The NLP model

**Task.** Given a full news article, find every financial entity (ORG, TICKER, PERSON, plus
MONEY / PERCENT / DATE spans) and output a sentiment score in (−0.95, 0.95) for each
sentiment-bearing entity. The score is the *implication for that entity*, not the tone of
the article — a lawsuit is negative for the defendant and neutral for the court.

**Architecture** ([`models/`](models/), report §3):

```
article ──► Longformer-large-4096 (shared encoder, 2048 tokens, global attention on entity tokens)
                 │
        ┌────────┴────────┐
   NER head (CRF,       Sentiment head (cross-attention:
   15 BIO labels)       entity tokens query the whole document) ──► tanh · 0.95
```

~439M parameters, 4.8M in the sentiment head. At inference the encoder runs twice per
article: once with CLS-only global attention to tag entities, once with global attention on
the predicted entity tokens to score them.

**Training** (report §4; the training code itself is not included): three stages on a single
Colab GPU — NER-only with *global-attention dropout* (the fix for a train/test attention-regime
mismatch that had collapsed end-to-end NER F1 to 0.05), joint training with a curriculum, then a
sentiment-head-only stage — followed by a head-only retrain (v2.1) on more decisive labels with a
magnitude-weighted Huber + (1 − CCC) + sign-penalty loss that replaced MSE + (1 − Pearson).

**Labels** (report §2, prompts in the appendix and in [`data_label_criteria/`](data_label_criteria/)):
LLM-generated entity spans and scores — Claude Haiku (v1.0), Claude Sonnet relabels (v2.0),
DeepSeek with a *decisive* prompt plus Kimi PERSON re-scoring (v2.1) — with near-duplicate
removal, article-id-disjoint splits and a label-noise audit.

**Evaluation** ([`scripts/evaluation/`](scripts/evaluation/), report §5): end-to-end evaluation
(raw text → NER → sentiment, IoU-matched to gold) in
[`evaluate_e2e_pipeline.py`](scripts/evaluation/evaluate_e2e_pipeline.py); per-version numbers
in [`trained_model/*/MODEL_CARD.md`](trained_model/) and
[`outputs/`](outputs/). Known limits: NER recall is the bottleneck (entity coverage ~0.42–0.49
end to end); the model still anchors on negative keywords when the entity *won* a dispute;
secondary mentions are under-read; the CRF transition parameters never trained in v2.x (the
tagger is effectively emission-only — see the model cards).

---

## Part 2 — Does the sentiment predict returns?

[`signal_strategy/`](signal_strategy/) — a self-contained study (report §6), reproducible with
`python3 signal_strategy/src/run_all.py` given the inference outputs and EODHD prices.

**Setup.** 118,033 tier-1 articles scored by the v2.0 checkpoint → per-ticker, per-day
mention-weighted sentiment (`sent_mw`) → joined to EODHD adjusted prices for 502 names,
2020-03-23 → 2025-12-31. Forward returns at 1 / 5 / 20 days. (The tier-1 set still contains
~14.5k paywall stubs from one wire service that were marked for demotion but never re-tiered;
the per-ticker daily series filters them, this panel does not.)

**Method** (each step its own script in [`signal_strategy/src/`](signal_strategy/src/)):
information coefficient with Newey-West t-stats; Fama-MacBeth regressions with size and
momentum controls; event study; quintile long-short backtest with next-open vs. close entry
and 5/10 bps costs; a market-on-close feasibility audit that checks every article's
publication time against a 15:45 ET cutoff.

**What happened.** The first pass looked like a clean success (IC +0.020, a +36 bps day-one
quintile spread, close-entry Sharpe 2.9), but the tradable next-open version was negative — the
whole edge sat in the overnight gap. Auditing timestamps showed that ~25% of articles, including
the 4–6 pm earnings spike, had been assigned a `trade_date` whose close preceded publication.
Rebuilt with an honest knowability cutoff, the IC is ≈ 0 at every horizon and every daily-bar
execution tested loses money: **the market absorbs this news faster than daily bars can act.**
One channel — pre-market news traded at that same day's open — was never tested, so the claim is
"no detected edge", not "edge is exactly zero". Two things survive: the model is a good
*measurement* instrument (its scores align with the contemporaneous overnight reaction), and
sentiment **magnitude** predicts next-day **volatility** beyond trailing realised vol (rank-IC
+0.024, NW t 5.3; +0.019, t 3.7 with same-day |return| as an extra control) — a promising
candidate feature, not a hardened one.

**Review.** The study was reviewed read-only by two independent model-based reviewers (Codex
and Kimi); Codex diagnosed the timing problem before the corrected numbers existed, and Kimi
re-ran the key numbers from the artifacts on disk and replicated the volatility result. Raw
reviews in
[`signal_strategy/notes/reviews/`](signal_strategy/notes/reviews/); reconciled view in
[`CONSOLIDATED_REPORT.md`](signal_strategy/CONSOLIDATED_REPORT.md). The open items there
(commit the feasible-panel script, test pre-market news at the open, check the 2023–24
reversal hint, point-in-time membership, re-score with v2.1) are the honest to-do list.

---

## Repository layout

```
docs/                   project_report.{tex,pdf}; report_assets/ (ASCII copies of the prompts for the appendix)
models/                 Longformer encoder, CRF NER head, coref wrapper, cross-attention sentiment head, pipeline
training/               preprocessing (label schema, mention→token alignment), dataset (batching), metrics
scripts/
  evaluation/           end-to-end eval, gold-mask eval, calibration baseline, gap analysis, arm comparison, audit
  inference/            batch inference over retiered news → per-article entity sentiment (default checkpoint v2.0)
  postprocessing/       ticker alias resolution, per-article/daily ticker sentiment, leakage-safe returns mapping
  analysis/             inspect_ner_head.py — reproduces the CRF-transition diagnostics
signal_strategy/        the signal study: src/, outputs/ (metrics + plots), reports, literature review, reviews
trained_model/          model cards, configs and e2e metrics for v1.0 / v2.0 / v2.1 (weights not included)
outputs/                e2e evaluation summaries, calibration, gap analysis, label audit, corpus-coverage notes
data_label_criteria/    the labelling spec and prompts (entity spec v2, decisive prompt, PERSON prompt)
data/reference/         alias table and non-company blocklist used for ticker resolution
notebooks/              Colab launchers for inference and the two evaluations (outputs stripped)
```

**Not included in this repository:** the news and price data (EODHD terms), the LLM label
files, model weights (~1.8 GB per version), the inference outputs and panels derived from the
news data, and the data-collection, labelling and model-training code. The label
specification and prompts, the model cards with their metrics, and all evaluation and
signal-study code are here, so every number in the report can be traced to the script that
produced it.

## Using the code

```bash
pip install -r requirements.txt

# Score a news file with a checkpoint (weights not included)
python3 scripts/inference/infer_entity_sentiment.py --input <articles.jsonl.gz> \
    --output <out.jsonl> --checkpoint trained_model/v2.1_20260620/model.pt

# End-to-end evaluation against a labelled file
python3 scripts/evaluation/evaluate_e2e_pipeline.py --help

# Reproduce the CRF diagnostics quoted in the model cards
python3 scripts/analysis/inspect_ner_head.py --help

# Signal study (needs outputs/inference/*_enriched.jsonl and EODHD prices; see signal_strategy/src/config.yaml)
python3 signal_strategy/src/run_all.py
```

Load the model in Python with `models.pipeline.FinancialEntitySentimentModel` (see the usage
block in [`trained_model/v2.0_20260517/MODEL_CARD.md`](trained_model/v2.0_20260517/MODEL_CARD.md)).

## Corrections log

- **2026-09-03** — The `signal_strategy/` scores were produced by the v2.0 checkpoint, not
  v2.1 (correction notes at the top of the study documents). The v2.1 sentiment head was
  warm-started from the v2.0 head, not re-initialised. The CRF transition parameters in v2.0
  and v2.1 never trained. Details in the model cards and report §4.

## License

Code and documentation: [MIT](LICENSE). Data and model weights are outside the license and
are not redistributed.

## Author

Russell Ding — [GitHub](https://github.com/Russell-Ding). Built as an independent project;
the analysis code was written with Claude Code under my direction and independently reviewed
by other models, with all conclusions verified by me.

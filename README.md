# Financial Entity Sentiment → Trading Signal

**An end-to-end, independently built research pipeline:** a Longformer-based model that reads
financial news and scores sentiment *per entity* (companies, tickers, people), trained on
~30k LLM-labeled articles — followed by a rigorous test of whether those scores carry a
tradable signal for S&P 500 stocks.

The short answer to the trading question is **no** — and the repo is organised around
showing *how* that answer was reached honestly, including the look-ahead bias that made
the first pass look like a Sharpe-2.9 strategy, the point-in-time correction that killed
it, and the one result that survived (sentiment intensity predicts next-day volatility).

> Everything here was built on my own time, on my own hardware/Colab, with my own data
> subscription. No employer code or data is involved.

---

## Headline results

**Model (v2.1, entity-level sentiment, in-distribution test — 1,558 articles / 8,078 entities)**

| Metric | v2.0 | **v2.1** |
|---|---|---|
| Pearson r (pred vs. label) | 0.485 | **0.643** [0.626, 0.660] |
| Concordance corr. (CCC) | 0.416 | **0.641** |
| pred/label std ratio | 0.57 | **1.03** — magnitude compression fixed |
| TICKER / ORG / PERSON r | 0.62 / 0.49 / 0.49 | **0.81 / 0.63 / 0.68** |

**Signal study (502 S&P 500 names, 2020-03 → 2025-12, 60,888 ticker-days)**

| Test | Naïve panel | Point-in-time panel (15:45 ET cutoff) |
|---|---|---|
| 1-day rank IC (Newey-West t) | +0.020 (t 3.7) | **−0.007 (t −1.4) ≈ 0** |
| Fama-MacBeth slope, 1d | +10.3 bps/σ (t 6.7) | — (contaminated; see report) |
| L/S quintile, H=1, close entry | Sharpe **+2.93** (look-ahead) | Sharpe **−0.55** (implementable MOC) |
| \|sentiment\| → next-day \|return\|, controlling for trailing vol | — | **rank-IC +0.024 (t 5.3)** — survives |

Full details: [`signal_strategy/RESULTS.md`](signal_strategy/RESULTS.md) and the consolidated
three-reviewer write-up [`signal_strategy/CONSOLIDATED_REPORT.md`](signal_strategy/CONSOLIDATED_REPORT.md).

---

## Part 1 — The NLP model

### Task

Given a full news article, find every financial entity (COMPANY, TICKER, PERSON, ORG, plus
MONEY / PERCENT / DATE spans) and output a sentiment score in [−0.95, 0.95] for each
sentiment-bearing entity. The score is the *implication for that entity*, not the tone of
the article — an article about a lawsuit is negative for the defendant and neutral for the
court.

### Data

* **Corpus:** 1.69M EODHD news articles covering the S&P 500 universe
  ([`scripts/collection/`](scripts/collection/)).
* **Source tiering:** every article is tagged T1/T2/T3 + a content type (earnings release,
  analyst call, market wrap, syndicated PR, …) by an LLM-labeled 5k sample distilled into
  regex rules ([`docs/NEWS_CLASSIFICATION_PIPELINE.md`](docs/NEWS_CLASSIFICATION_PIPELINE.md),
  [`outputs/source_rules_v4.json`](outputs/source_rules_v4.json)). Tiers drive sampling and
  loss weighting.
* **Labels:** LLM-generated entity spans + sentiment via knowledge distillation (Claude Haiku
  for v1.0, Claude Sonnet relabels for v2.0, DeepSeek with a *decisive* prompt + Kimi for
  PERSON re-scoring for v2.1), with quality
  guardians, near-duplicate dedup and article-id-disjoint train/val/test splits
  ([`data_label_criteria/`](data_label_criteria/), [`scripts/labeling/`](scripts/labeling/)).
  v2.1 trains on 27,015 / 1,558 / 1,558 articles.
* **Data are not redistributed** (EODHD terms). [`data/DATASET_README.md`](data/DATASET_README.md)
  documents the schema and composition; the collection and labeling scripts reproduce it
  with your own keys.

### Architecture ([`models/`](models/))

```
article ──► Longformer-large-4096 (shared encoder, 2048 tokens, global attention on entity tokens)
                 │
        ┌────────┴────────┐
   NER head (CRF,       Sentiment head (V2 cross-attention:
   15 BIO labels)       entity tokens query the whole document) ──► tanh · 0.95
```

* Longformer over ModernBERT/BERT because articles routinely exceed 512 tokens and the
  sentiment for an entity often lives far from its mention.
* NER is a CRF-decoded BIO tagger; the sentiment head attends from each entity's mention
  tokens over the full document. ~439M parameters, 4.8M in the sentiment head.
* Design spec: [`docs/architecture_spec.md`](docs/architecture_spec.md).

### Training — three stages ([`scripts/training/train_pipeline.py`](scripts/training/train_pipeline.py))

| Stage | Trains | Key idea |
|---|---|---|
| 1 · NER | encoder + NER head | **Global-attention dropout** (p=0.3, warm-up 30%) — the fix for a train/test regime mismatch that had collapsed end-to-end NER F1 from 0.74 (gold entity positions) to 0.05 (no gold positions). Randomly hiding the entity-aware global attention during training forces the encoder to work from CLS-only attention too. [`docs/global_attn_dropout_proposal.pptx`](docs/global_attn_dropout_proposal.pptx) |
| 2 · Joint | all | curriculum: NER weight 1.0→0.5, sentiment 0.3→1.0 |
| 3 · Sentiment | sentiment head only (encoder + NER frozen) | v2.1 loss: **magnitude-weighted Huber + (1 − CCC) + sign penalty**, replacing MSE + (1 − Pearson). Pearson is scale-invariant so it never penalised the head's ⅓-magnitude "hedging"; CCC does. Derivation: [`docs/ccc_huber_loss_explained.pdf`](docs/ccc_huber_loss_explained.pdf) |

The v2.1 retrain was run as a **5-arm ablation** (frozen encoder vs. unfreezing the top-2
layers, loss variants). The unfrozen arm scored marginally higher in-distribution but was
rejected for NER catastrophic forgetting (PERSON F1 −6.8%) and calibration drift
off-distribution — [`docs/retrain_comparison.md`](docs/retrain_comparison.md).

Runs on a single Colab GPU via the launchers in [`notebooks/`](notebooks/);
[`docs/RETRAIN_GUIDE.md`](docs/RETRAIN_GUIDE.md) is the operational runbook.

### Evaluation and known limits

* End-to-end evaluation (raw text → NER → sentiment, IoU-matched to gold) in
  [`scripts/evaluation/evaluate_e2e_pipeline.py`](scripts/evaluation/evaluate_e2e_pipeline.py);
  per-version numbers in [`trained_model/*/MODEL_CARD.md`](trained_model/).
* A label-noise audit of high-confidence model↔label disagreements found ~1 clear label error
  in 24 adjudicated cases — the gains are real, not fitted noise
  ([`outputs/control_label_audit.md`](outputs/control_label_audit.md)).
* **Honest limits:** NER recall is the bottleneck (entity coverage ~0.48 e2e); the model still
  anchors on negative keywords when the entity *won* a dispute; secondary/list-style mentions
  are under-read. Cross-distribution, sentiment degrades gracefully (r 0.61, spread still
  calibrated, +0.06 mean bias).

---

## Part 2 — Does the sentiment predict returns?

[`signal_strategy/`](signal_strategy/) — a self-contained study, reproducible with
`python3 signal_strategy/src/run_all.py`.

### Setup

118,033 tier-1 articles scored by v2.1 → per-ticker, per-day mention-weighted sentiment
(`sent_mw`) → joined to EODHD adjusted prices for 502 names, 2020-03-23 → 2025-12-31
(median 45 names/day with news). Forward returns at 1 / 5 / 20 days.

### Method (standard cross-sectional toolkit, each step its own script)

1. **Information coefficient** — daily Spearman rank IC, Newey-West t-stats, IC-IR.
2. **Fama-MacBeth** — daily cross-sectional regressions with size and 1m/12m momentum controls.
3. **Event study** — cumulative abnormal returns around extreme-sentiment days, top vs. bottom quintile.
4. **Long-short backtest** — daily quintile portfolios, next-open vs. close entry, 5/10 bps costs, turnover, 70/30 date split.
5. **MOC feasibility audit** — every article's publication timestamp converted to New York time and tested against a 15:45 ET market-on-close cutoff.
6. **Literature grounding** — [`research/literature_review.md`](signal_strategy/research/literature_review.md).

### What happened

The first pass looked like a clean success: IC +0.020 (t 3.7), Fama-MacBeth +10 bps/σ
surviving controls, a +36 bps day-one quintile spread, and a close-entry backtest with gross
Sharpe 2.9. But the *tradable* next-open version was **negative** — the whole edge sat in the
overnight gap.

Auditing publication timestamps explained it: **~25% of articles (including the 4–6 pm
earnings-release spike) had been assigned a `trade_date` whose close preceded publication.**
The model wasn't forecasting Tuesday; it was reading Monday-evening news that Monday's
close couldn't know and Tuesday's open had already priced. Rebuilt with an honest
knowability cutoff, the IC is ≈ 0 at every horizon and every executable daily-bar variant
(MOC close-to-close, next-open for after-hours news) loses money. **The market absorbs
this news faster than daily bars can act.**

Two things survive: (a) the model is a good *measurement* instrument — its scores align
strongly with the contemporaneous overnight reaction, which is useful for attribution and
monitoring; (b) sentiment **magnitude** predicts next-day **volatility** beyond trailing
realised vol (rank-IC +0.024, NW t 5.3; +0.019, t 3.7 with same-day |return| as an extra
control) — a non-directional signal suited to risk overlays and position sizing.

### Review process

The study was reviewed read-only by two independent model-based reviewers (Codex and Kimi);
Kimi re-ran the key numbers from the artifacts on disk. Both independently reached the
look-ahead diagnosis, confirmed the volatility result, and flagged further weaknesses
(event-study independence, a post-treatment size control, survivorship in the universe, the
correction not being a committed script). The raw reviews are in
[`signal_strategy/notes/reviews/`](signal_strategy/notes/reviews/) and the reconciled view in
[`CONSOLIDATED_REPORT.md`](signal_strategy/CONSOLIDATED_REPORT.md). The open items there are
the honest to-do list, not a claim of closure.

---

## Repository layout

```
models/                 Longformer encoder, CRF NER head, coref wrapper, V2 sentiment head, pipeline
training/               dataset / preprocessing / trainer
scripts/
  collection/           EODHD bulk news + price collection, S&P 500 universe
  preprocessing/        source tiering, training-subset construction
  labeling/             LLM labeling (DeepSeek, Kimi), PERSON re-scoring, quality guardian
  training/             3-stage pipeline, sentiment-arm ablations
  evaluation/           end-to-end eval, calibration baseline, arm comparison, disagreement audit
  inference/            batch inference over retiered news → per-article entity/ticker sentiment
  postprocessing/       ticker alias resolution, daily ticker sentiment, returns mapping
data_label_criteria/    the labeling specs & prompts (entity spec v2, decisive relabeling prompt, PERSON prompt)
data_collection/        earlier collectors (NewsAPI, Yahoo, SEC EDGAR)
config/                 config loader + secrets template
docs/                   architecture spec, retrain guide, news-classification pipeline, loss derivation,
                        retrain plan & ablation comparison, global-attention-dropout proposal
trained_model/          model cards, configs and e2e metrics for v1.0 / v2.0 / v2.1 (weights not included)
outputs/                training curves, calibration, label audit, source rules, e2e metric summaries
notebooks/              Colab launchers (outputs stripped)
signal_strategy/        the signal study: src/, outputs/ (metrics + plots), reports, literature review, reviews
```

---

## Reproducing

```bash
pip install -r requirements.txt
cp config/secrets.yaml.template config/secrets.yaml     # or export EODHD_API_KEY / ANTHROPIC_API_KEY …

# 1. Data (needs an EODHD subscription)
python3 scripts/collection/fetch_sp500_universe.py
python3 scripts/collection/collect_eodhd_sp500_bulk.py --help
python3 scripts/preprocessing/apply_source_tiers.py --help

# 2. Labels (DeepSeek / Kimi API keys)
bash scripts/labeling/run_deepseek_labeling.sh
python3 scripts/data/split_new_dataset.py --help

# 3. Train (Colab GPU) — see docs/RETRAIN_GUIDE.md and notebooks/full_pipeline_launcher.ipynb
python3 scripts/training/train_pipeline.py --help
python3 scripts/training/train_sentiment_arm.py --arm control

# 4. Evaluate / infer
python3 scripts/evaluation/evaluate_e2e_pipeline.py --help
python3 scripts/inference/infer_entity_sentiment.py --input <retiered.jsonl> --output <out.jsonl>

# 5. Signal study
python3 signal_strategy/src/run_all.py            # ~7 min; --skip-panel to reuse panel.parquet
```

**Not included in this repo:** raw news and price data (EODHD terms), LLM label files,
model weights (~1.8 GB per version), and the inference outputs / panels derived from the
news data. Everything needed to regenerate them is here.

## License

Code and documentation: [MIT](LICENSE). Data and model weights are outside the license and
are not redistributed.

## Author

Russell Ding — [GitHub](https://github.com/Russell-Ding). Built as an independent project;
the analysis code was written with Claude Code under my direction and independently reviewed
by other models, with all conclusions verified by me.

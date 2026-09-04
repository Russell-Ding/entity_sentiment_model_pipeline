# Model Card — v2.1_20260620 (Financial Entity Sentiment)

**Lineage:** v2.0_20260517 → **v2.1** = sentiment head retrained on the new decisive
labels with a redesigned loss. **Encoder and NER head are frozen and byte-identical
to v2.0.**

## What changed vs v2.0 and why

v2.0's sentiment head **under-polarized**: it predicted ~⅓ of the true magnitude
(pred std 0.167 vs gold 0.275) and almost never emitted strong scores. Diagnosis
(see `docs/second_phase_retrain_plan.md`): the loss was the bottleneck — pure masked
MSE on a ~59%-neutral label distribution drives mean-collapse, and the `(1−Pearson)`
term is scale-invariant so it never penalizes magnitude. A post-hoc calibration test
confirmed it was a **discrimination** limit, not a fixable slope.

v2.1 retrains **only the sentiment head** (frozen encoder + NER) on the
DeepSeek+Kimi decisive label set with a new loss:

    L = weighted-Huber( w=clip(1+3|y|, max 3) )  +  (1 − CCC)  +  sign-penalty(|y|≥0.4)

- **CCC** (concordance correlation) replaces `(1−Pearson)` — it is scale-*sensitive*,
  so it punishes magnitude compression directly.
- **Magnitude-weighted Huber** up-weights strong-sentiment entities (robust to noisy
  LLM extremes) so the neutral mass can't drive mean-collapse.
- See `docs/ccc_huber_loss_explained.pdf` for the full derivation.

This is the **`control` arm** of a 5-arm ablation. An arm that additionally unfroze
the top-2 encoder layers (`armC`) scored marginally higher on the in-distribution
test (Pearson 0.690) but was **rejected**: it caused NER catastrophic forgetting
(PERSON F1 −6.8% on gold) and cross-distribution sentiment calibration drift, for no
ranking gain on foreign data. Freezing the encoder (this model) avoids both.

## Training

- **Base:** v2.0_20260517 checkpoint (encoder + NER frozen; sentiment head **warm-started
  from the v2.0 head** and retrained — an earlier version of this card said "re-init"; see
  Corrections below).
- **Loss:** `ccc_huber` (`models/sentiment_head.py::configure_loss`); select best by CCC.
- **Data:** `data/labeled/deepseek_t1/splits/` — 30,131 T1 articles (after near-dup
  dedup) split **by article id** into train 27,015 / val 1,558 / test 1,558 (disjoint).
  Labels: DeepSeek v4-flash decisive prompt; PERSON entities re-scored by Kimi.
- **Recipe:** head LR 5e-4, AdamW, cosine schedule, batch 48, best epoch 10 (val CCC 0.618).
- Script: `scripts/training/train_sentiment_arm.py --arm control`.

## Evaluation

### Sentiment — in-distribution test split (1,558 articles, 8,078 covered entities)
| metric | v2.0 | **v2.1** |
|---|---|---|
| **Pearson r** | 0.485 | **0.643** [0.626, 0.660] |
| Spearman ρ | 0.451 | **0.600** |
| CCC | 0.416 | **0.641** |
| pred/gold std ratio | 0.57 | **1.03** (compression fixed) |
| MAE | 0.209 | **0.196** |
| sign-flip rate | 0.215 | **0.151** |
| very_neg / very_pos F1 | 0.00 / 0.01 | **0.13 / 0.37** |
| ORG / TICKER / PERSON r | 0.486 / 0.622 / 0.493 | **0.627 / 0.807 / 0.678** |

Baselines it clears: current model (0.485), affine-calibration floor (0.483), and the
old Sonnet-holdout reference (0.628).

### NER — UNCHANGED (frozen encoder)
On the gold Sonnet holdout, v2.1 NER is **byte-identical to v2.0** (all deltas 0.000):
sentiment-types F1 **0.459** (P 0.713, R 0.338); ORG 0.462, TICKER 0.445, PERSON 0.458.

### Cross-distribution sanity (gold Sonnet holdout, NOT the training distribution)
| | v2.0 (home) | v2.1 |
|---|---|---|
| Pearson r | 0.628 | 0.606 |
| pred/gold std ratio | 0.73 | **1.02** (well-calibrated) |
| mean_pred vs gold (0.106) | 0.105 | 0.168 (mild +0.06 bias) |
| MAE | 0.192 | 0.234 |

Sentiment **degrades gracefully** off-distribution: spread stays calibrated, with a
mild positive mean-bias. If absolute scores are consumed on a non-decisive
distribution, a per-deployment affine offset trivially corrects the +0.06 shift.

## Known limitations (label-noise audit, `outputs/control_label_audit.md`)

A spot-audit of high-confidence model↔label disagreements found the **labels are
sound** (≈1 clear label error in ~24 adjudicated cases — so the gain is real, not
fitted noise). The residual errors are **model behaviors**, not data:

1. **"Won a lawsuit/dispute" can still score negative** — the model anchors on
   negative keywords (ban / sue / probe / antitrust) even when the entity *won*
   (e.g. "Apple wins bid to pause Watch ban"). Watch this in legal-outcome news.
2. **Under-reads list-style / secondary mentions** (market-wrap blurbs, secondary
   clauses) — a single-pass coverage limit; ~3.7% of entities under-scored.
3. **Over-reacts to charged context when the entity is incidental** (e.g. a company
   named only as a benchmark or a partner).

Also inherits v2.0's e2e NER recall bottleneck (coverage ~0.42–0.49): sentiment is
only produced for entities the NER head detects.

## Intended use / scope

Entity-level sentiment for **ORG / TICKER / PERSON** in financial news, scored
−0.95…0.95 (implication for the entity, not article tone). Tuned for the **decisive**
label regime (lawsuits/probes/misses negative; indices and incidental mentions 0.0).
TICKER sentiment is strongest (r 0.81) and most suitable for ticker→returns signals.

## Files
`model.pt` (weights + metadata), `config.json`, `tokenizer/`,
`evaluation_results_e2e.json` (in-distribution test). Load with the same path as
v2.0 in `scripts/evaluation/evaluate_e2e_pipeline.py` / `scripts/inference/`.

## Corrections (2026-09-03)

Two facts established while documenting the model structure; neither changes any
number above.

1. **The sentiment head was warm-started, not re-initialised.** The retrain script
   loads the full v2.0 state dict and never resets the head; the run ledger's
   pre-training baseline (val CCC 0.4147) is the v2.0 head's own score. The v2.0
   Stage-3 head, by contrast, really was initialised fresh.
2. **The CRF transition scores never trained (v2.0 and v2.1 alike).** All 225
   transition and 30 start/end scores lie within ±0.11 (the library's init range) and
   the −1e4 BIO constraints are absent, because the Stage-1/2 training script built
   the head without the label map and the constraints written at load time are
   overwritten by the checkpoint. Emissions dominate (Viterbi = per-token argmax on
   >99% of tokens), so the tagger is effectively emission-only with span repair by the
   lenient decoder. Reproduce with `scripts/analysis/inspect_ner_head.py`. A future
   retrain should pass `ner_label_to_id` at model construction and give the CRF
   parameters their own learning rate.

**Note on the signal study:** the `signal_strategy/` scores were produced by the
v2.0 checkpoint, not this one (see the correction note in `signal_strategy/RESULTS.md`).

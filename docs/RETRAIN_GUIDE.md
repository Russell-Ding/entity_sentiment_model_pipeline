# Retrain Guide

**Last updated:** 2026-05-17
**Audience:** Future Russell, Claude, Kimi, or any agent helping with this project.
**Goal:** Run the streamlined three-stage retrain pipeline that produces a new
production model when training data changes.

---

## TL;DR — How to retrain

You have new training data and want a fresh model. Here's the entire workflow:

```bash
# 1. Make sure the data files are at:
#    data/labeled/final/train.jsonl
#    data/labeled/final/val.jsonl
#    (and optionally holdout files for evaluation)

# 2. Open notebooks/full_pipeline_launcher.ipynb on Colab with G4 95GB GPU.

# 3. Run cells 1-4. Wait ~12-14 hours.

# 4. Cell 5 verifies the bundle was created at trained_model/v<VERSION>_<DATE>/

# 5. Cell 6 (optional) immediately runs the e2e benchmark.

# 6. Cell 7 terminates the runtime.
```

Done. The rest of this guide explains the pieces and how to recover when
things go wrong.

---

## What the pipeline does

Three stages, all driven by `scripts/training/train_pipeline.py`:

| Stage | What trains | Loss | Key trick |
|---|---|---|---|
| **1. NER-only** | Encoder + NER head, all parameters | NER (CRF) only | **Global-attention dropout** (p=0.3, warmup 30%) — randomly hides entity-aware globals 30% of the time so the encoder learns to produce useful representations from CLS-only attention too. This is what makes e2e inference work. |
| **2. Joint** | Encoder + NER + sentiment, all parameters | Curriculum: NER weight 1.0→0.5, sentiment 0.3→1.0 | Continues global-attn dropout, lower LR (1e-5) |
| **3. Sentiment retrain** | **Only sentiment head** (~5M params trainable) — encoder + NER FROZEN | Pure sentiment (MSE + Pearson r) | **Loads STAGE 1's `best_cls_only.pt`, not Stage 2's.** See "Why" below. |

After Stage 3, `train_pipeline.py` automatically packages the best epoch into a
production bundle at `trained_model/v<VERSION>_<DATE>/`.

### Why Stage 3 loads Stage 1's checkpoint (not Stage 2's)

Stage 2's joint training slightly degrades the encoder's CLS-only NER F1
(from ~0.58 to ~0.53) because it's also optimizing for sentiment. Since
Stage 3 **discards the sentiment head and re-initializes it fresh** anyway,
the only thing Stage 3 cares about from its input checkpoint is the
encoder + NER weights — and Stage 1's are better for the e2e inference
regime. This was tested empirically in the May-2026 retrain.

### Why global-attention dropout is the core technique

The model originally collapsed in e2e (NER F1 = 0.05 vs 0.74 with gold
masks) because the encoder was always trained with global attention on
gold entity tokens, but at inference there are no gold entities — only
CLS gets global attention. Dropout teaches the encoder to handle both
regimes.

---

## Prerequisites

### Hardware
- **GPU**: G4 with 95 GB HBM memory (most reliable).
  - A100 80GB also works at slightly smaller batch sizes.
  - A100 40GB: requires reducing `--batch-size-train` to 8 and possibly
    enabling `--gradient-checkpointing` (already default).
- **Disk**: Each stage saves ~5 GB of checkpoints to `/content/` on Colab.
  Need ~25 GB free. Colab's default disk has 100+ GB so plenty.
- **Time**: 12-14 hours wall-clock on G4. Fits in one Colab Pro+ session
  (24 hr limit). Risky on Colab Pro (12 hr limit) — if you only have Pro,
  run Stage 1 alone first, then Stage 2+3 in a fresh session.

### Software
- Colab notebook installs deps automatically (cell 3): `transformers`,
  `torch`, `pytorch-crf`. No additional setup needed.

### Data layout

```
data/labeled/final/
├── train.jsonl              ← required
├── val.jsonl                ← required
├── holdout.jsonl            ← optional, used by evaluate_holdout_stage3.py
└── holdout_relabeled.jsonl  ← optional, used by evaluate_e2e_pipeline.py (benchmark)
```

Each JSONL row has:
```json
{
  "id": "unique_article_id",
  "text": "Full article text...",
  "entities": [
    {
      "canonical_id": "AAPL",
      "canonical_name": "Apple Inc.",
      "type": "ORG",
      "sentiment_score": 0.3,
      "ner_mentions": [{"text": "Apple", "start_char": 45, "end_char": 50}],
      "coref_mentions": [],
      "sentiment_expanded_mentions": []
    }
  ],
  "metadata": {"source": "nasdaq.com", "date": "2024-06-15"}
}
```

See `data/DATASET_README.md` for full schema details.

---

## Launching the retrain

### Option A — Notebook (recommended)

Open `notebooks/full_pipeline_launcher.ipynb` on Colab:
1. **Cell 1**: mount Drive + verify GPU (`!nvidia-smi` should show 95 GB)
2. **Cell 2**: set `PROJECT_PATH` to your Drive mount of this repo. Sets
   `RUN_ID` to a timestamp, `VERSION` to "v3.0" (edit if you want a
   different version label), and `OUTPUT_VERSION_NAME` to
   `<VERSION>_<DATE>`.
3. **Cell 3**: pip install
4. **Cell 4**: runs the full pipeline. The big one.
5. **Cell 5**: verifies the final bundle was created
6. **Cell 6** (optional): immediately runs the e2e benchmark
7. **Cell 7**: terminates the runtime to stop billing

### Option B — Command line

If running outside the notebook (e.g., from a Colab terminal):

```bash
python scripts/training/train_pipeline.py \
    --run-id 20260601 \
    --version v3.0 \
    --output-version-name v3.0_20260601 \
    --local-ckpt-root /content \
    --drive-ckpt-root /path/to/project/checkpoints
```

The script's `--help` lists every parameter you can override (training
data paths, hyperparameters, encoder choice, etc.).

---

## What to monitor during training

### Stage 1 (~3-4 hr)

End of each epoch logs **both regimes**:

```
Validation [entity-aware]: ner_f1=0.76, sentiment_mse=0.11
Validation [CLS-only]    : ner_f1=0.55, sentiment_mse=0.11  (dropout p=0.300)
```

**Gating signal**: `Validation [CLS-only]: ner_f1` should be **≥ 0.40 by
end of epoch 1, ≥ 0.50 by end of training**. If it stays near 0 after
epoch 1, kill the run — something is wrong with the data or the dropout
setting.

Best checkpoint saved as `best_cls_only.pt` (used as Stage 3 input).

### Stage 2 (~3-4 hr)

Same dual-eval logging. Expect:
- Entity-aware F1: stable around 0.76
- **CLS-only F1: may drop ~0.05 from Stage 1's peak** — this is normal
  (curriculum trades NER quality for sentiment training)
- Sentiment MSE: should drop from 0.11 → ~0.10

Don't kill the run if CLS-only F1 drops a bit. Only kill if it goes below
~0.30 (catastrophic regression).

### Stage 3 (~3-4 hr)

End of each epoch logs:
```
Results: loss=..., mse=..., corr=..., ner_f1=0.5841, time=..., gpu=...
```

**`ner_f1` is constant** (encoder frozen) — sanity check that nothing
broke the freeze.

**`corr` (Pearson r) climbs over epochs**. Want to see ≥ 0.50 by epoch 8-10
(the May-2026 reference reached 0.5105 at epoch 9).

### Final packaging

After Stage 3 completes, the script:
1. Scans all `checkpoint_epoch_*.pt` files in Stage 3's local dir
2. Picks the one with highest val `sentiment_corr`
3. Strips optimizer state, saves as `trained_model/<VERSION_NAME>/model.pt`
4. Writes config.json and MODEL_CARD.md with the run's metrics

---

## Common failures and fixes

### "CUDA out of memory" at Stage 1 step 0

Encoder size + batch size combination too large.

**Fix:**
```
--batch-size-train 16    # (was 24)
# or even
--batch-size-train 8
```

Gradient checkpointing is on by default. If you turned it off, turn it
back on:
```
--gradient-checkpointing
```

### Stage 1 CLS-only F1 stays near zero after epoch 1

Either training data is degenerate, or dropout is too aggressive.

**Fix:** Lower the dropout probability or shorten the warmup:
```
--global-attn-dropout-prob 0.2 \
--stage1-dropout-warmup-frac 0.5
```

### Drive checkpoint sync issue: `best_model.pt` is stale

Google Drive's FUSE filesystem sometimes drops in-place file overwrites.
Symptom: `best_model.pt` on Drive contains epoch-1 weights, but
`checkpoint_epoch_N.pt` for each N is correct.

**Detection:** The packaging step in `train_pipeline.py` already handles
this — it scans all `checkpoint_epoch_*.pt` files and picks the best by
metric, NOT by filename. So this bug doesn't affect the final bundle.

**If you need to manually load the best model from a stage's archive**,
verify the metrics by checking `val_metrics["sentiment_corr"]` inside the
file, not by trusting the `best_model.pt` filename. Example:

```python
import torch
ck = torch.load("checkpoints/stage3_X/best_model.pt", map_location="cpu", weights_only=False)
print(ck["val_metrics"]["sentiment_corr"])  # if low, file is stale; scan epoch files
```

### Colab disconnected mid-training

Re-mount Drive, run cell 2 to restore paths, and use `--skip-stages` to
skip stages that already completed.

For example, if Stage 1 and Stage 2 finished but Stage 3 was interrupted:

```bash
python scripts/training/train_pipeline.py \
    --run-id <SAME_RUN_ID_AS_BEFORE> \
    --skip-stages 1,2
```

The same `--run-id` ensures the new invocation reuses Stage 1's checkpoint
dir at `/content/stage1_<run_id>/` (Drive copy too).

### "credential propagation was unsuccessful" when re-mounting Drive

Run `drive.mount('/content/drive', force_remount=True)` and click through
the OAuth prompt again.

### A data file has malformed entities (logged as warnings)

Stage 1 logs include lines like:
```
WARNING Line 2538: Processing error - 'list' object has no attribute 'get'
```

Not a concern. The preprocessor skips that single article and continues.
Typical retention: ~99.9% (16 out of 17,587 dropped in the May-2026 run).

---

## After training: evaluation

Two eval scripts available:

### Gold-mask evaluation (sentiment quality given perfect NER input)
```bash
python scripts/evaluation/evaluate_holdout_stage3.py \
    --checkpoint trained_model/v3.0_<DATE>/model.pt \
    --holdout data/labeled/final/holdout.jsonl
```

### End-to-end evaluation (raw text → NER → sentiment)
```bash
python scripts/evaluation/evaluate_e2e_pipeline.py \
    --checkpoint trained_model/v3.0_<DATE>/model.pt \
    --benchmark data/labeled/final/holdout_relabeled.jsonl \
    --output-dir outputs/e2e_evaluation \
    --local-output-dir /content \
    --ner-mode single-pass \
    --inference-batch-size 16
```

Cell 6 of the launcher notebook runs this automatically.

---

## What "good" looks like (May-2026 reference numbers)

For the v2.0 model trained with this exact pipeline:

| Eval | Metric | Value |
|---|---|---|
| Val (gold-mask) | Sentiment Pearson r | 0.5105 |
| Val (gold-mask) | Sentiment MSE | 0.0802 |
| Val (gold-mask) | NER F1 (entity-aware) | 0.7649 |
| Holdout (gold-mask) | Sentiment Pearson r | 0.5252 |
| Holdout (gold-mask) | NER F1 | 0.7366 |
| **E2E** | NER F1 (sentiment-bearing) | 0.4586 |
| **E2E** | Entity coverage | 0.4223 |
| **E2E** | Sentiment Pearson r | 0.6279 (~0.60 effective) |
| **E2E** | Bucket adjacent accuracy | 0.9478 |

After a retrain with more data, expect **at least these numbers**, and
ideally improvements on coverage and sentiment Pearson r.

---

## Where the files live

```
entity_sentiment_model_pipeline/
├── RETRAIN_GUIDE.md                 ← this file
├── notebooks/
│   └── full_pipeline_launcher.ipynb ← THE entry point for Colab
├── scripts/
│   ├── training/
│   │   ├── train_pipeline.py        ← orchestrator (calls the two below)
│   │   ├── train_two_stage.py       ← Stage 1+2 (don't run directly)
│   │   └── train_stage3.py          ← Stage 3 (don't run directly)
│   └── evaluation/
│       ├── evaluate_e2e_pipeline.py
│       └── evaluate_holdout_stage3.py
├── data/labeled/final/              ← input data
├── trained_model/
│   ├── v1.0_20260116/               ← OLD (Jan 2026, COMPANY-only)
│   ├── v2.0_20260517/                ← CURRENT PRODUCTION (May 2026)
│   └── v3.0_<NEW_DATE>/             ← what a new run will produce
└── checkpoints/
    ├── archive/                     ← historical training artifacts
    └── <stage1|2|3>_<run_id>/       ← latest-run intermediates (transient)
```

---

## Architecture quick-reference

### Inference (two passes per article)

```
Pass 1 (NER):
    encoder(text, global_attn=CLS_only)  →  hidden_states_1
    ner_head(hidden_states_1)            →  BIO labels  →  predicted entity spans

Pass 2 (Sentiment):
    global_attn = CLS | predicted_entity_tokens
    encoder(text, global_attn)           →  hidden_states_2
    sentiment_head(hidden_states_2,
                   predicted_entity_masks) →  per-entity sentiment ∈ [-0.95, 0.95]
```

### Why two passes
- Pass 1 uses CLS-only globals because we don't know entities yet
- Pass 2 uses entity-aware globals (matches sentiment head training regime)
- Both passes are supported by the encoder thanks to Stage 1+2 dropout training

### Model components (in `models/`)
- `encoder.py` — Longformer-Large wrapper (1024 hidden, 24 layers,
  optional gradient checkpointing)
- `ner_head_crf.py` — CRF-based BIO classifier, 15 labels
- `sentiment_head.py` — `SentimentHead` (cross-attention,
  ~5M params, output ∈ [-0.95, 0.95])
- `pipeline.py` — `FinancialEntitySentimentModel` glues them together

---

## Troubleshooting workflow for agents

If you (Claude, Kimi, or another assistant) are helping the user retrain
and something goes wrong, follow this order:

1. **Read the log**: `logs/two_stage_training_*.log` for Stage 1+2,
   `/content/stage3_*/stage3_sentiment_*.log` for Stage 3.
2. **Identify which stage failed** from the log timestamps.
3. **Match the error to "Common failures" above** for known fixes.
4. **If OOM**: lower batch size. Numbers above are for 95 GB GPU.
5. **If Drive sync issue**: use `--skip-stages` with the same `--run-id`
   to resume from where it died.
6. **If the metrics never improve**: check the training data file is
   the right one and not empty / corrupted.

---

## Maintenance notes

### When to bump the version number
- New training data → v3.0
- Architecture change → vN.0
- Hyperparameter tuning on same data → v2.1, v2.2, ...

### When to update this guide
- Hyperparameter defaults change in `train_pipeline.py`
- New failure modes are discovered
- The pipeline structure changes (e.g., a Stage 4 is added)

### Source of truth
- This guide is the human-readable summary.
- `scripts/training/train_pipeline.py --help` is the authoritative list
  of available flags.
- The most recent `trained_model/v*_*/MODEL_CARD.md` documents what was
  actually run for that release.

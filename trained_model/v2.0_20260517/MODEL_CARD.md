# Financial Entity Sentiment Model — v2.0

## Model Version
- **Version**: v2.0
- **Date**: 2026-05-17
- **Model file**: `model.pt` (1.76 GB)

## Headline Numbers

### End-to-End Evaluation (raw text → NER → sentiment, no gold annotations)
Run on `data/labeled/final/holdout_relabeled.jsonl` (6,750 articles, Sonnet-relabeled).

| Metric | Value |
|---|---|
| **NER F1 (sentiment-bearing types)** | **0.4586**  (P=0.71, R=0.34) |
| **Entity coverage** (gold entities found) | **42.2%**  (27,566 / 65,270) |
| **Sentiment Pearson r** (on covered entities) | **0.6279** (≈0.60 effective after self-selection adjustment) |
| **Sentiment MAE** | 0.1919 |
| **Joint accuracy ≤ 0.2** | 0.6370 |
| **Joint accuracy ≤ 0.4** | 0.8688 |
| **Bucket adjacent accuracy** | **0.9478** |

Per-entity-type sentiment Pearson r on covered entities:
| Type | N | Pearson r |
|---|---|---|
| TICKER | 5,150 | **0.7726** |
| ORG | 18,539 | 0.5984 |
| PERSON | 3,877 | 0.4124 |

### Gold-mask Evaluation (sentiment quality with perfect NER input)
Run on `data/labeled/final/holdout.jsonl` (9,371 articles, original labels).

| Metric | Value |
|---|---|
| Sentiment Pearson r | 0.5252 |
| Sentiment MSE | 0.0870 |
| NER F1 | 0.7366 |

### Validation (best epoch, gold-mask)
| Metric | Value |
|---|---|
| Sentiment Pearson r | 0.5105 |
| Sentiment MSE | 0.0802 |
| NER F1 | 0.7649 |

## Architecture
- **Encoder**: `allenai/longformer-large-4096` (435M params, 1024 hidden, 24 layers)
- **NER head**: CRF-based BIO classifier (15 labels: 7 entity types × {B-, I-} + O)
- **Sentiment head**: V2 cross-attention (entity tokens query → full document attends as keys/values), scaled tanh output ∈ [-0.95, 0.95]
- **Total params**: 439M
- **Trainable params (Stage 3 only)**: 4.8M (V2 sentiment head)

## Training Story

This model was trained with a **global-attention dropout** strategy to close a
train/test regime mismatch present in earlier versions. The previous model
(`best_model_20260418.pt`, archived) had NER F1 = 0.74 with gold entity
positions but only 0.05 in end-to-end inference — because the encoder was
always trained with global attention on gold entity tokens, but at inference
no gold entities are available.

### Three-stage training pipeline (May 2026)

**Stage 1 — NER-only training (`scripts/training/train_two_stage.py --stage 1`)**
- 5 epochs, batch=24, LR=2e-5, AdamW, mixed precision, gradient checkpointing
- All parameters trainable
- Per-sample global-attention dropout: probability 0.3, ramped from 0 over first 30% of steps
- Result: CLS-only NER F1 = 0.5841 at epoch 2 (vs ~0.05 baseline)
- Checkpoint used as input for Stage 2: `best_cls_only.pt` (epoch 2)

**Stage 2 — Joint NER + sentiment with curriculum (`--stage 2`)**
- 5 epochs, batch=24, LR=1e-5
- Curriculum: NER weight 1.0 → 0.5, sentiment weight 0.3 → 1.0
- Global-attention dropout p=0.3 (no warmup — already established in Stage 1)
- Result: encoder + NER slightly drifted from CLS-only optimum (CLS F1 dropped to 0.53)
- Decision: Stage 3 loaded from Stage 1's checkpoint, not Stage 2's (better encoder for the e2e regime; Stage 3 throws away the sentiment head anyway)

**Stage 3 — Fresh V2 sentiment head retrain (`scripts/training/train_stage3.py`)**
- 10 epochs, batch=80, LR=5e-4, AdamW
- Encoder + NER frozen (only V2 sentiment head trainable)
- V2 head initialized fresh (Xavier); previous V2 weights from Stage 2 discarded
- No global-attention dropout in Stage 3 (sentiment only sees entity-aware encoder at inference)
- Loss: combined MSE + (1 − Pearson r) on valid entities
- Best epoch: 9 (val Pearson r = 0.5105)

### Training data
- **Train**: 17,587 articles from `data/labeled/final/train.jsonl` (Sonnet-relabeled)
- **Val**: 2,130 articles from `data/labeled/final/val.jsonl`
- **E2E benchmark**: `data/labeled/final/holdout_relabeled.jsonl` (6,750 Sonnet-relabeled)
- **Gold-mask benchmark**: `data/labeled/final/holdout.jsonl` (9,371 original labels)

## Inference

The model is designed for a **two-pass end-to-end inference**:

```
Pass 1 (NER):
    encoder(text, global_attention_mask=CLS_only)  →  hidden_states
    ner_head(hidden_states)  →  BIO labels  →  predicted entity spans

Pass 2 (Sentiment):
    global_attn = CLS | predicted_entity_tokens
    encoder(text, global_attention_mask=global_attn)  →  hidden_states
    sentiment_head(hidden_states, predicted_entity_masks)  →  per-entity sentiment ∈ [-0.95, 0.95]
```

The encoder is robust to both `CLS-only` (Pass 1) and `CLS + entity tokens`
(Pass 2) global attention regimes thanks to the Stage 1+2 dropout training.

## Usage

```python
import torch
import sys
sys.path.insert(0, "path/to/entity_sentiment_model_pipeline")

from models.pipeline import FinancialEntitySentimentModel
from training.preprocessing import LABEL_TO_ID

# Build model with matching architecture
model = FinancialEntitySentimentModel(
    encoder_name="allenai/longformer-large-4096",
    hidden_size=1024,
    num_ner_labels=15,
    use_ner_head=True,
    use_coref_head=False,
    use_crf_ner=True,
    ner_label_to_id=LABEL_TO_ID,
    max_length=2048,
)

# Load weights
checkpoint = torch.load("trained_model/v2.0_20260517/model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.cuda()

# For e2e inference on a text, use scripts/evaluation/evaluate_e2e_pipeline.py
# (look at predict_articles_batched() for the batched implementation)
```

## Recommended Use Cases

✅ **Production-ready for:**
- Headline / per-entity sentiment lookup ("how does this article feel about AAPL?")
- TICKER-driven trading signals (Pearson r = 0.77 for tickers specifically)
- Sentiment dashboards with adjacent-bucket tolerance
- Information retrieval ranking by sentiment

⚠️ **Use with caution:**
- Position-sizing using exact bucket boundaries (consider Platt scaling first)
- PERSON sentiment (Pearson r = 0.41 — could be label noise or structural)
- High-coverage aggregations (only 42% of gold entities found)

❌ **Not recommended for:**
- Tasks requiring complete entity coverage (consider follow-on training with predicted-mask sentiment in Stage 3, or coref-augmented inference)

## Files
- `model.pt` — Model weights + metadata (1.76 GB)
- `config.json` — Architecture configuration
- `MODEL_CARD.md` — This file
- `evaluation_results_e2e.json` — Full e2e evaluation metrics
- `tokenizer/` — Longformer tokenizer files

## Known Limitations

0. **CRF transitions untrained (found 2026-09-03).** The CRF's transition and
   start/end scores are still at their random initial values (all within ±0.11) and
   the BIO constraints were never installed, because `train_two_stage.py` built the
   head without the label map. Viterbi equals per-token argmax on >99% of tokens; the
   tagger is effectively emission-only and BIO validity comes from the lenient span
   decoder. Reproduce with `scripts/analysis/inspect_ner_head.py`.
1. **NER recall is conservative (0.34 e2e)** — model is high-precision, low-recall. Trade-off from global-attention dropout that bought us inference-regime robustness.
2. **PERCENT NER overfires** (P=0.19) — the head fires on every "%" sign without context. Not sentiment-bearing so doesn't affect product, but indicates brittle lexical-cue behavior.
3. **PERSON sentiment quality lower than ORG/TICKER** (r=0.41 vs 0.60/0.77) — root cause unclear (label noise vs structural).
4. **Self-selection bias** in e2e Pearson r — ~3-6 points of inflation because the 42% of entities the NER finds may be linguistically easier sentiment cases. Effective r ≈ 0.60.

## Changelog
- **v2.0 (2026-05-17)**: Production release. Global-attention dropout retrain. NER F1 in e2e went from 0.05 (v1 era) to 0.46. Sentiment Pearson r in e2e: 0.63 (≈0.60 effective).
- **v1.0 (2026-01-16)**: Initial release with sentiment head trained. NER quality limited. *Archived in `checkpoints/archive/`.*

## Reproducibility

Source checkpoint: `checkpoints/stage3_sentiment_large_v2retrain/checkpoint_epoch_9.pt`
(archived after this release).
Full training artifacts (Stage 1, Stage 2, all Stage 3 epochs): `checkpoints/archive/v2_dropout_run/`.

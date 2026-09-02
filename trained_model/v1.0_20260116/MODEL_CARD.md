# Financial Entity Sentiment Model

## Model Version
- **Version**: v1.0
- **Date**: 2026-01-16
- **Model File**: model.pt

## Architecture
- **Encoder**: allenai/longformer-base-4096
- **NER Head**: CRF (Conditional Random Field)
- **Sentiment Head**: Entity-focused attention pooling + regression

## Training Data
- **Training samples**: 2,133 (cleaned)
- **Validation samples**: 235 (cleaned)
- **Entity types**: COMPANY (primary)
- **Sentiment labels**: Generated via Claude Haiku knowledge distillation [-1, 1]

## Evaluation Metrics (on cleaned validation set)

### Sentiment Performance (Ground Truth Entities)
| Metric | Value |
|--------|-------|
| RMSE | 0.1866 |
| MAE | 0.1609 |
| Pearson r | 0.9264 |
| Spearman r | 0.8772 |
| Missing entities | 14.9% |

### Full Pipeline (CRF NER + Sentiment)
| Metric | Value |
|--------|-------|
| RMSE | 0.1761 |
| MAE | 0.1457 |
| Pearson r | 0.9167 |
| Spearman r | 0.8730 |

### NER Performance
| Metric | Value |
|--------|-------|
| F1 | 0.069 |
| Note | NER needs retraining on cleaned data |

## Usage

```python
from models.pipeline import FinancialEntitySentimentModel

model = FinancialEntitySentimentModel(
    encoder_name="allenai/longformer-base-4096",
    max_length=2048,
    use_crf_ner=True,
    device="mps"  # or "cuda" or "cpu"
)

# Load weights
checkpoint = torch.load("trained_model/v1.0_20260116/model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Analyze text
results = model.analyze(
    text="Apple Inc. reported record quarterly revenue...",
    target_entities=["Apple Inc."],
    use_coref=False
)
```

## Files
- `model.pt` - Model weights and state dict
- `config.json` - Model configuration
- `tokenizer/` - Longformer tokenizer files

## Known Limitations
1. NER (CRF) was trained on noisy position data - needs retraining
2. Only COMPANY entity type in training data
3. Sentiment labels from knowledge distillation (Haiku), not human annotated

## Changelog
- v1.0 (2026-01-16): Initial release with sentiment head trained, CRF NER needs improvement

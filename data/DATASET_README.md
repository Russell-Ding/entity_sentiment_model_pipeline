# Dataset Documentation

**Last Updated:** 2026-01-22

This document describes the datasets used for training, validating, and evaluating the Entity Sentiment Model.

---

## Quick Reference

| Dataset | File | Samples | Purpose |
|---------|------|---------|---------|
| **Training** | `labeled/cleaned/train_cleaned_v3.jsonl` | 28,764 | Train the model |
| **Validation** | `labeled/cleaned/val_cleaned_v3.jsonl` | 3,195 | Tune hyperparameters, prevent overfitting |
| **Holdout** | `labeled/holdout/eodhd_non_yahoo_holdout_v4.jsonl` | 9,103 | Final unbiased evaluation |

---

## 1. Training Dataset

**File:** `data/labeled/cleaned/train_cleaned_v3.jsonl`
**Size:** 209 MB | **Samples:** 28,764 | **Entities:** 346,779

### Source Composition

| Source | Articles | Percentage |
|--------|----------|------------|
| **EODHD API** (aggregated news) | 26,631 | 92.6% |
| **Yahoo Finance** (direct) | 2,124 | 7.4% |
| Other (NewsAPI, News Daily) | 9 | <0.1% |

### EODHD News Domain Breakdown

The EODHD API aggregates news from multiple financial news sources:

| News Domain | ~Articles | % of EODHD |
|-------------|-----------|------------|
| finance.yahoo.com | ~10,000 | 34% |
| nasdaq.com | ~7,500 | 25% |
| globenewswire.com | ~4,200 | 14% |
| seekingalpha.com | ~4,000 | 14% |
| investing.com | ~2,000 | 7% |
| reuters.com | ~1,800 | 6% |

### Entity Distribution

| Entity Type | Count | Has Sentiment |
|-------------|-------|---------------|
| PERCENT | 162,783 | No (null) |
| MONEY | 103,919 | No (null) |
| DATE | 32,632 | No (null) |
| ORG | 25,535 | Yes [-1.0, 1.0] |
| TICKER | 21,122 | Yes [-1.0, 1.0] |
| PERSON | 788 | Yes [-1.0, 1.0] |

### Purpose

- Used during model training to learn entity recognition and sentiment patterns
- 90/10 split was applied to create training and validation sets
- Contains diverse financial news articles from 2020-2025

---

## 2. Validation Dataset

**File:** `data/labeled/cleaned/val_cleaned_v3.jsonl`
**Size:** 22 MB | **Samples:** 3,195 | **Entities:** 37,995

### Source Composition

Same source distribution as training data (split from same pool).

### Purpose

- Used **during training** to:
  - Monitor model performance on unseen data
  - Tune hyperparameters (learning rate, batch size, etc.)
  - Implement early stopping to prevent overfitting
- **Can be used multiple times** during training iterations

---

## 3. Holdout Dataset

**File:** `data/labeled/holdout/eodhd_non_yahoo_holdout_v4.jsonl`
**Size:** 73 MB | **Samples:** 9,103 | **Entities:** 163,457

### Source Composition

| News Domain | Articles | Percentage |
|-------------|----------|------------|
| nasdaq.com | ~3,200 | 35% |
| globenewswire.com | ~2,500 | 27% |
| seekingalpha.com | ~2,000 | 22% |
| investing.com | ~800 | 9% |
| reuters.com | ~600 | 7% |

**Critical:** This dataset contains **ONLY non-Yahoo sources** to ensure no overlap with training data.

### Entity Distribution

| Entity Type | Count | Has Sentiment |
|-------------|-------|---------------|
| PERCENT | 63,600 | No (null) |
| ORG | 38,430 | Yes [-1.0, 1.0] |
| TICKER | 26,712 | Yes [-1.0, 1.0] |
| MONEY | 21,299 | No (null) |
| DATE | 13,416 | No (null) |

### Purpose

- **Final evaluation only** - used ONCE after model training is complete
- Tests model generalization to unseen news sources
- Provides unbiased performance metrics
- **NEVER use for training or hyperparameter tuning**

### Why Holdout Matters

1. **Prevents data leakage** - Model never sees this data during development
2. **Tests generalization** - Different sources than training data
3. **Unbiased evaluation** - Only used once for final metrics
4. **Contamination status: CLEAN** - Never used in any training process

---

## Data Quality & Validation

All datasets have been validated and cleaned by Claude Opus agents. The following checks passed:

| Validation Check | Status |
|------------------|--------|
| TICKER labels (1-5 uppercase letters only) | PASS |
| ORG labels (organizations like NYSE, SEC, ECB correctly typed) | PASS |
| Character indices (`text[start:end] == mention`) | PASS |
| Sentiment scores (ORG/TICKER/PERSON: [-1.0, 1.0]) | PASS |
| Null sentiment (MONEY/PERCENT/DATE: null) | PASS |
| No bad single-char mentions (U.S., T&C fragments removed) | PASS |

---

## Entity Labeling Rules

### Entity Types

| Type | Description | Sentiment |
|------|-------------|-----------|
| **TICKER** | Stock symbols (AAPL, MSFT, GOOGL) - 1-5 uppercase letters | Required |
| **ORG** | Organization names (Apple Inc., Federal Reserve, NYSE) | Required |
| **PERSON** | People's names (Elon Musk, Warren Buffett) | Required |
| **MONEY** | Currency amounts ($1.5 billion, 50 million euros) | Null |
| **PERCENT** | Percentages (5%, 10.5 percent) | Null |
| **DATE** | Dates and time periods (Q3 2024, January 15) | Null |

### Known ORG Acronyms (NOT Tickers)

These uppercase acronyms are organizations, not stock symbols:

```
NYSE, SEC, ECB, FOMC, IMF, BOE, CFTC, FED, CNBC, OECD, BOJ, FINRA,
OCC, PBOC, EU, ICE, HUD, NATO, OPEC, EPA, FDA, CDC, WHO, UN, UK,
USA, US, BBC, CNN, ABC, NBC, CBS, PBS, NPR, AP, LSE, TSX, ASX,
HKEX, CME, DOJ, FTC, IRS, GAO, AI, CEO, COO
```

### Sentiment Score Range

- **-1.0** = Very negative (scandal, bankruptcy, crash)
- **0.0** = Neutral (factual mention, no opinion)
- **+1.0** = Very positive (record profits, breakthrough)

---

## File Format

All datasets are JSONL (JSON Lines) format. Each line is a JSON object:

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
      "ner_mentions": [
        {
          "text": "Apple",
          "start_char": 45,
          "end_char": 50
        }
      ],
      "coref_mentions": []
    }
  ],
  "metadata": {
    "source": "nasdaq.com",
    "date": "2024-06-15"
  }
}
```

---

## How to Use

### Training a Model

```python
train_file = "data/labeled/cleaned/train_cleaned_v3.jsonl"
val_file = "data/labeled/cleaned/val_cleaned_v3.jsonl"

# Load and train
with open(train_file) as f:
    train_data = [json.loads(line) for line in f]
```

### Final Evaluation (After Training Complete)

```python
# Only use ONCE after model is finalized
holdout_file = "data/labeled/holdout/eodhd_non_yahoo_holdout_v4.jsonl"

with open(holdout_file) as f:
    holdout_data = [json.loads(line) for line in f]

# Evaluate model performance
predictions = model.predict(holdout_data)
metrics = calculate_metrics(predictions, holdout_data)
```

---

## Archive

Intermediate and source files are preserved in `data/labeled/archive/` and `data/archive/`:

| Archive Location | Contents |
|------------------|----------|
| `labeled/archive/cleaned_intermediate/` | Previous versions (v1, v2, harmonized) |
| `labeled/archive/holdout_intermediate/` | Previous holdout versions and chunks |
| `labeled/archive/haiku_raw_backup/` | Original Haiku-labeled source files |
| `labeled/archive/unused_sources/` | Labeled data not merged (Polygon, SEC) |
| `archive/raw_backup/` | Raw unlabeled source data |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-01-15 | Initial labeled dataset |
| v2 | 2026-01-21 | Fixed TICKER/ORG labels, sentiment nulls |
| **v3** | 2026-01-22 | Full validation pass, production ready |
| **v4** (holdout) | 2026-01-22 | Removed bad single-char mentions |

---

## Contact

For questions about this dataset, refer to the project documentation or labeling guides in `/training_data/`.

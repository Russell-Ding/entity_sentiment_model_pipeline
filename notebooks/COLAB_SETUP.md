# Google Colab Training Setup

## Quick Start

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload Notebook**:
   - File → Upload notebook
   - Select `colab_training_setup.ipynb` from this folder

3. **Enable GPU**:
   - Runtime → Change runtime type
   - Select **T4 GPU** (or better)
   - Click Save

4. **Run All Cells**: Runtime → Run all

## What the Notebook Does

### Stage 1: NER-Only Training (~30-60 min)
- Trains entity boundary detection (COMPANY, TICKER, PERSON, etc.)
- Uses 28,764 training articles
- Saves checkpoint to Google Drive

### Stage 2: Joint Fine-tuning (~30-60 min)
- Loads Stage 1 checkpoint
- Trains NER + Sentiment together with curriculum learning
- Saves final model to Google Drive

## Configuration

Edit these in the notebook's CONFIG cell:

```python
CONFIG = {
    # Increase batch_size if GPU memory allows (T4 = 4-8, A100 = 16+)
    "stage1_batch_size": 4,

    # Increase max_length for longer articles (up to 4096)
    "max_length": 2048,

    # Adjust epochs based on convergence
    "stage1_epochs": 5,
    "stage2_epochs": 5,
}
```

## Expected Results

After training completes:

| Metric | Expected Range |
|--------|----------------|
| NER F1 | 0.70 - 0.85 |
| Sentiment MSE | 0.10 - 0.25 |
| Sentiment Correlation | 0.50 - 0.75 |

## Output Files

Saved to Google Drive:
```
checkpoints/
├── stage1_ner/
│   ├── best_model.pt     # Best NER model
│   └── final_model.pt    # Last epoch
└── stage2_joint/
    ├── best_model.pt     # Best joint model (USE THIS)
    └── final_model.pt    # Last epoch

outputs/
└── training_curves.png   # Training visualization
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 2
- Reduce `max_length` to 1024

### Slow Training
- Ensure GPU is enabled (Runtime → Change runtime type)
- Check `!nvidia-smi` shows GPU

### Drive Not Mounting
- Re-run the mount cell
- Ensure you're logged into Google account

### Module Import Errors
- Verify PROJECT_PATH is correct
- Run the pip install cell

## Training Data

| Dataset | Articles | Source |
|---------|----------|--------|
| Training | 28,764 | EODHD + Yahoo |
| Validation | 3,195 | EODHD + Yahoo |
| Holdout | 9,103 | Non-Yahoo only |

The holdout set (NASDAQ, Reuters, etc.) is used for uncontaminated cross-source evaluation.

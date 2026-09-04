"""Training module for Financial Entity Sentiment Model."""

from .preprocessing import (
    DataPreprocessor,
    ProcessedSample,
    NER_LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    char_span_to_token_span,
    validate_ner_labels,
    get_entity_statistics,
)

from .dataset import (
    EntitySentimentDataset,
    collate_fn,
    create_data_loaders,
    get_label_weights,
)

from .metrics import (
    compute_ner_metrics,
    compute_sentiment_metrics,
)

try:  # the training loop (trainer.py) is not part of the public release
    from .trainer import (
        JointTrainer,
        TrainingConfig,
        TrainingState,
    )
except ImportError:  # pragma: no cover
    JointTrainer = TrainingConfig = TrainingState = None  # type: ignore[assignment]

__all__ = [
    # Preprocessing
    "DataPreprocessor",
    "ProcessedSample",
    "NER_LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "char_span_to_token_span",
    "validate_ner_labels",
    "get_entity_statistics",
    # Dataset
    "EntitySentimentDataset",
    "collate_fn",
    "create_data_loaders",
    "get_label_weights",
    # Trainer
    "JointTrainer",
    "TrainingConfig",
    "TrainingState",
    "compute_ner_metrics",
    "compute_sentiment_metrics",
]

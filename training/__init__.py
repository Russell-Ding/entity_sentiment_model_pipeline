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

from .trainer import (
    JointTrainer,
    TrainingConfig,
    TrainingState,
    compute_ner_metrics,
    compute_sentiment_metrics,
)

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

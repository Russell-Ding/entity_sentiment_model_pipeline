"""PyTorch Dataset for joint NER and Sentiment training."""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import (
    DataPreprocessor,
    ProcessedSample,
    LABEL_TO_ID,
    NER_LABELS,
)

logger = logging.getLogger(__name__)


class EntitySentimentDataset(Dataset):
    """
    Dataset for joint NER and Sentiment training.

    Each sample contains:
    - input_ids: Tokenized text
    - attention_mask: Attention mask
    - ner_labels: BIO labels for NER
    - entity_masks: List of masks for each entity
    - sentiment_scores: List of sentiment scores for each entity
    """

    def __init__(
        self,
        samples: List[ProcessedSample],
        max_entities_per_sample: int = 10,
    ):
        """
        Initialize dataset.

        Args:
            samples: List of ProcessedSample objects
            max_entities_per_sample: Maximum entities to keep per sample
        """
        self.samples = samples
        self.max_entities_per_sample = max_entities_per_sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Basic tensors
        input_ids = torch.tensor(sample.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(sample.attention_mask, dtype=torch.long)
        ner_labels = torch.tensor(sample.ner_labels, dtype=torch.long)

        # Entity masks and sentiment scores
        entities = sample.entities[:self.max_entities_per_sample]

        if entities:
            entity_masks = torch.tensor(
                [e["entity_mask"] for e in entities],
                dtype=torch.float,
            )
            sentiment_scores = torch.tensor(
                [e["sentiment_score"] for e in entities],
                dtype=torch.float,
            )
            num_entities = len(entities)
        else:
            # No entities - create dummy tensors
            seq_len = len(sample.input_ids)
            entity_masks = torch.zeros(1, seq_len, dtype=torch.float)
            sentiment_scores = torch.zeros(1, dtype=torch.float)
            num_entities = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ner_labels": ner_labels,
            "entity_masks": entity_masks,
            "sentiment_scores": sentiment_scores,
            "num_entities": num_entities,
            "sample_id": sample.id,
        }

    @classmethod
    def from_jsonl(
        cls,
        file_paths: Union[str, List[str]],
        preprocessor: Optional[DataPreprocessor] = None,
        max_entities_per_sample: int = 10,
    ) -> "EntitySentimentDataset":
        """
        Load dataset from JSONL file(s).

        Args:
            file_paths: Path or list of paths to JSONL files
            preprocessor: DataPreprocessor instance (creates one if not provided)
            max_entities_per_sample: Maximum entities per sample

        Returns:
            EntitySentimentDataset instance
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if preprocessor is None:
            preprocessor = DataPreprocessor()

        all_samples = []
        for path in file_paths:
            samples = preprocessor.process_file(path)
            all_samples.extend(samples)

        logger.info(f"Loaded {len(all_samples)} samples from {len(file_paths)} files")

        return cls(all_samples, max_entities_per_sample)

    @classmethod
    def from_processed_jsonl(
        cls,
        file_paths: Union[str, List[str]],
        max_entities_per_sample: int = 10,
    ) -> "EntitySentimentDataset":
        """
        Load from pre-processed JSONL (already has token positions).

        Args:
            file_paths: Path or list of paths to processed JSONL files
            max_entities_per_sample: Maximum entities per sample

        Returns:
            EntitySentimentDataset instance
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        all_samples = []
        for path in file_paths:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    sample = ProcessedSample(
                        id=data.get("id", ""),
                        input_ids=data["input_ids"],
                        attention_mask=data["attention_mask"],
                        ner_labels=data["ner_labels"],
                        entities=data.get("entities", []),
                        metadata=data.get("metadata", {}),
                    )
                    all_samples.append(sample)

        logger.info(f"Loaded {len(all_samples)} processed samples")

        return cls(all_samples, max_entities_per_sample)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Handles variable number of entities per sample by padding.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched tensors
    """
    batch_size = len(batch)
    seq_len = batch[0]["input_ids"].shape[0]

    # Stack basic tensors (same shape across batch)
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    ner_labels = torch.stack([b["ner_labels"] for b in batch])

    # Find max entities in batch
    max_entities = max(b["entity_masks"].shape[0] for b in batch)
    max_entities = max(max_entities, 1)  # At least 1

    # Pad entity masks and sentiment scores
    entity_masks = torch.zeros(batch_size, max_entities, seq_len)
    sentiment_scores = torch.zeros(batch_size, max_entities)
    entity_mask_valid = torch.zeros(batch_size, max_entities)  # Track valid entities

    for i, b in enumerate(batch):
        num_ent = b["entity_masks"].shape[0]
        if b["num_entities"] > 0:
            entity_masks[i, :num_ent] = b["entity_masks"]
            sentiment_scores[i, :num_ent] = b["sentiment_scores"]
            entity_mask_valid[i, :b["num_entities"]] = 1

    # Collect sample IDs
    sample_ids = [b["sample_id"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "entity_masks": entity_masks,
        "sentiment_scores": sentiment_scores,
        "entity_mask_valid": entity_mask_valid,
        "sample_ids": sample_ids,
    }


def create_data_loaders(
    train_files: Union[str, List[str]],
    val_files: Optional[Union[str, List[str]]] = None,
    batch_size: int = 4,
    val_split: float = 0.1,
    num_workers: int = 0,
    preprocessor: Optional[DataPreprocessor] = None,
    max_entities_per_sample: int = 10,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders.

    Args:
        train_files: Training file path(s)
        val_files: Validation file path(s), or None to split from train
        batch_size: Batch size
        val_split: Validation split ratio (if val_files is None)
        num_workers: Number of data loading workers
        preprocessor: DataPreprocessor instance
        max_entities_per_sample: Maximum entities per sample
        seed: Random seed for splitting

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if preprocessor is None:
        preprocessor = DataPreprocessor()

    # Load all training samples
    if isinstance(train_files, str):
        train_files = [train_files]

    all_samples = []
    for path in train_files:
        samples = preprocessor.process_file(path)
        all_samples.extend(samples)

    logger.info(f"Total samples: {len(all_samples)}")

    # Split into train/val if no separate val files
    if val_files is None:
        random.seed(seed)
        random.shuffle(all_samples)
        val_size = int(len(all_samples) * val_split)
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]
    else:
        train_samples = all_samples
        if isinstance(val_files, str):
            val_files = [val_files]
        val_samples = []
        for path in val_files:
            samples = preprocessor.process_file(path)
            val_samples.extend(samples)

    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Create datasets
    train_dataset = EntitySentimentDataset(train_samples, max_entities_per_sample)
    val_dataset = EntitySentimentDataset(val_samples, max_entities_per_sample) if val_samples else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def get_label_weights(samples: List[ProcessedSample]) -> torch.Tensor:
    """
    Compute class weights for NER labels based on inverse frequency.

    Args:
        samples: List of processed samples

    Returns:
        Tensor of shape (num_labels,) with class weights
    """
    label_counts = [0] * len(NER_LABELS)

    for sample in samples:
        for label_id in sample.ner_labels:
            if 0 <= label_id < len(label_counts):
                label_counts[label_id] += 1

    total = sum(label_counts)
    weights = []

    for count in label_counts:
        if count > 0:
            # Inverse frequency, capped to avoid extreme weights
            weight = min(total / (len(NER_LABELS) * count), 10.0)
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)

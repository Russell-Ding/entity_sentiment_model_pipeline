"""Preprocessing module for converting Haiku labels to training format.

Converts character positions to Longformer token positions and generates
BIO labels for NER training.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
from transformers import LongformerTokenizerFast

logger = logging.getLogger(__name__)

# NER label mappings (BIO scheme)
NER_LABELS = [
    "O",           # 0 - Outside
    "B-COMPANY",   # 1
    "I-COMPANY",   # 2
    "B-TICKER",    # 3
    "I-TICKER",    # 4
    "B-PERSON",    # 5
    "I-PERSON",    # 6
    "B-ORG",       # 7
    "I-ORG",       # 8
    "B-MONEY",     # 9
    "I-MONEY",     # 10
    "B-PERCENT",   # 11
    "I-PERCENT",   # 12
    "B-DATE",      # 13
    "I-DATE",      # 14
]

LABEL_TO_ID = {label: i for i, label in enumerate(NER_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(NER_LABELS)}

# Entity types that get sentiment scores
SENTIMENT_ENTITY_TYPES = {"COMPANY", "TICKER", "PERSON", "ORG"}


@dataclass
class ProcessedSample:
    """A single processed training sample."""
    id: str
    input_ids: List[int]
    attention_mask: List[int]
    ner_labels: List[int]
    entities: List[Dict]  # Each has entity_mask, sentiment_score, canonical_id
    metadata: Dict = field(default_factory=dict)


def char_span_to_token_span(
    offset_mapping: List[Tuple[int, int]],
    start_char: int,
    end_char: int,
) -> Optional[Tuple[int, int]]:
    """
    Convert character span to token span using offset mapping.

    Args:
        offset_mapping: List of (start_char, end_char) for each token
        start_char: Start character position
        end_char: End character position (exclusive)

    Returns:
        (start_token, end_token) tuple or None if not found
    """
    start_token = None
    end_token = None

    for idx, (token_start, token_end) in enumerate(offset_mapping):
        # Skip special tokens [CLS], [SEP], padding (offset 0,0)
        if token_start == token_end == 0:
            continue

        # Find first token that overlaps with our span
        if start_token is None and token_end > start_char:
            start_token = idx

        # Find last token that overlaps with our span
        if token_start < end_char:
            end_token = idx + 1

    if start_token is not None and end_token is not None:
        return (start_token, end_token)
    return None


class DataPreprocessor:
    """Preprocesses Haiku-labeled data for Longformer training."""

    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        use_expanded_sentiment: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            model_name: Longformer model name for tokenizer
            max_length: Maximum sequence length
            use_expanded_sentiment: If True, include sentiment_expanded_mentions
                                   in sentiment mask (for two-stage labeling)
        """
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length
        self.use_expanded_sentiment = use_expanded_sentiment

    def process_sample(self, sample: Dict) -> Optional[ProcessedSample]:
        """
        Process a single Haiku-labeled sample.

        Args:
            sample: Dictionary with text, entities, metadata

        Returns:
            ProcessedSample or None if processing fails
        """
        text = sample.get("text", "")
        if not text.strip():
            return None

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        offset_mapping = encoding["offset_mapping"][0].tolist()

        # Initialize NER labels (all O)
        ner_labels = [LABEL_TO_ID["O"]] * len(input_ids)

        processed_entities = []

        for entity in sample.get("entities", []):
            entity_type = entity.get("type", "")
            canonical_id = entity.get("canonical_id", "")
            sentiment_score = entity.get("sentiment_score")

            # Skip if missing required fields
            if not entity_type or not canonical_id:
                continue

            # Process NER mentions (get BIO labels)
            ner_mentions = entity.get("ner_mentions", [])
            for mention in ner_mentions:
                start_char = mention.get("start_char")
                end_char = mention.get("end_char")

                if start_char is None or end_char is None:
                    continue

                span = char_span_to_token_span(offset_mapping, start_char, end_char)
                if span is None:
                    continue

                start_tok, end_tok = span

                # Assign B- label to first token
                b_label = f"B-{entity_type}"
                if b_label in LABEL_TO_ID:
                    ner_labels[start_tok] = LABEL_TO_ID[b_label]

                # Assign I- labels to remaining tokens
                i_label = f"I-{entity_type}"
                if i_label in LABEL_TO_ID:
                    for tok_idx in range(start_tok + 1, end_tok):
                        if tok_idx < len(ner_labels):
                            ner_labels[tok_idx] = LABEL_TO_ID[i_label]

            # Build entity mask for sentiment (includes coref_mentions)
            # Only for sentiment-bearing entity types
            if entity_type in SENTIMENT_ENTITY_TYPES and sentiment_score is not None:
                entity_mask = [0] * len(input_ids)

                # Include all mentions (ner + coref)
                all_mentions = ner_mentions + entity.get("coref_mentions", [])

                # Optionally include sentiment_expanded_mentions (two-stage schema)
                if self.use_expanded_sentiment:
                    all_mentions = all_mentions + entity.get("sentiment_expanded_mentions", [])

                for mention in all_mentions:
                    start_char = mention.get("start_char")
                    end_char = mention.get("end_char")

                    if start_char is None or end_char is None:
                        continue

                    span = char_span_to_token_span(offset_mapping, start_char, end_char)
                    if span is None:
                        continue

                    start_tok, end_tok = span
                    for tok_idx in range(start_tok, end_tok):
                        if tok_idx < len(entity_mask):
                            entity_mask[tok_idx] = 1

                # Only add entity if mask has at least one 1
                if sum(entity_mask) > 0:
                    processed_entities.append({
                        "canonical_id": canonical_id,
                        "type": entity_type,
                        "sentiment_score": sentiment_score,
                        "entity_mask": entity_mask,
                    })

        return ProcessedSample(
            id=sample.get("id", ""),
            input_ids=input_ids,
            attention_mask=attention_mask,
            ner_labels=ner_labels,
            entities=processed_entities,
            metadata=sample.get("metadata", {}),
        )

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> List[ProcessedSample]:
        """
        Process a JSONL file of Haiku-labeled samples.

        Args:
            input_path: Path to input JSONL file
            output_path: Optional path to save processed samples

        Returns:
            List of ProcessedSample objects
        """
        input_path = Path(input_path)
        processed_samples = []

        with open(input_path) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    sample = json.loads(line)
                    processed = self.process_sample(sample)
                    if processed:
                        processed_samples.append(processed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                except Exception as e:
                    logger.warning(f"Line {line_num}: Processing error - {e}")

        logger.info(f"Processed {len(processed_samples)} samples from {input_path}")

        # Save if output path provided
        if output_path:
            self._save_processed(processed_samples, output_path)

        return processed_samples

    def _save_processed(self, samples: List[ProcessedSample], output_path: str):
        """Save processed samples to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in samples:
                data = {
                    "id": sample.id,
                    "input_ids": sample.input_ids,
                    "attention_mask": sample.attention_mask,
                    "ner_labels": sample.ner_labels,
                    "entities": sample.entities,
                    "metadata": sample.metadata,
                }
                f.write(json.dumps(data) + "\n")

        logger.info(f"Saved {len(samples)} processed samples to {output_path}")


def validate_ner_labels(ner_labels: List[int]) -> List[str]:
    """
    Validate BIO sequence and return any errors.

    Args:
        ner_labels: List of label IDs

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for i, label_id in enumerate(ner_labels):
        label = ID_TO_LABEL.get(label_id, "UNKNOWN")

        if label.startswith("I-"):
            # I- must follow B- or I- of same type
            if i == 0:
                errors.append(f"Position {i}: {label} cannot be first token")
                continue

            prev_label = ID_TO_LABEL.get(ner_labels[i-1], "O")
            entity_type = label[2:]  # Remove "I-"

            valid_prev = [f"B-{entity_type}", f"I-{entity_type}"]
            if prev_label not in valid_prev:
                errors.append(
                    f"Position {i}: {label} follows invalid {prev_label}"
                )

    return errors


def get_entity_statistics(samples: List[ProcessedSample]) -> Dict:
    """
    Compute statistics over processed samples.

    Args:
        samples: List of processed samples

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_samples": len(samples),
        "total_entities": 0,
        "entities_by_type": {},
        "sentiment_distribution": {
            "negative": 0,  # < -0.3
            "slightly_negative": 0,  # -0.3 to -0.1
            "neutral": 0,  # -0.1 to 0.1
            "slightly_positive": 0,  # 0.1 to 0.3
            "positive": 0,  # > 0.3
        },
        "avg_entities_per_sample": 0,
        "samples_with_sentiment": 0,
    }

    for sample in samples:
        for entity in sample.entities:
            stats["total_entities"] += 1

            entity_type = entity.get("type", "UNKNOWN")
            stats["entities_by_type"][entity_type] = (
                stats["entities_by_type"].get(entity_type, 0) + 1
            )

            sentiment = entity.get("sentiment_score")
            if sentiment is not None:
                if sentiment < -0.3:
                    stats["sentiment_distribution"]["negative"] += 1
                elif sentiment < -0.1:
                    stats["sentiment_distribution"]["slightly_negative"] += 1
                elif sentiment <= 0.1:
                    stats["sentiment_distribution"]["neutral"] += 1
                elif sentiment <= 0.3:
                    stats["sentiment_distribution"]["slightly_positive"] += 1
                else:
                    stats["sentiment_distribution"]["positive"] += 1

        if sample.entities:
            stats["samples_with_sentiment"] += 1

    if stats["total_samples"] > 0:
        stats["avg_entities_per_sample"] = (
            stats["total_entities"] / stats["total_samples"]
        )

    return stats

"""Named Entity Recognition Head Module.

Token classification head for identifying financial entities using BIO tagging.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


# Entity type definitions with BIO tagging
ENTITY_LABELS = [
    "O",           # Outside any entity
    "B-COMPANY",   # Beginning of company name
    "I-COMPANY",   # Inside company name
    "B-TICKER",    # Beginning of ticker symbol
    "I-TICKER",    # Inside ticker (rare, usually single token)
    "B-PERSON",    # Beginning of person name
    "I-PERSON",    # Inside person name
    "B-ORG",       # Beginning of organization
    "I-ORG",       # Inside organization
    "B-MONEY",     # Beginning of monetary value
    "I-MONEY",     # Inside monetary value
    "B-PERCENT",   # Beginning of percentage
    "I-PERCENT",   # Inside percentage
    "B-DATE",      # Beginning of date
    "I-DATE",      # Inside date
]

LABEL_TO_ID = {label: i for i, label in enumerate(ENTITY_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}


class NERHead(nn.Module):
    """
    Token classification head for Named Entity Recognition.

    Implements BIO tagging scheme for financial entity types:
    - COMPANY: Apple Inc., Goldman Sachs
    - TICKER: AAPL, GS
    - PERSON: Tim Cook, Jamie Dimon
    - ORG: Federal Reserve, SEC
    - MONEY: $1.5 billion
    - PERCENT: 15%, 2.5 percentage points
    - DATE: Q3 2024, fiscal year 2025

    Architecture:
        encoder_output -> Dropout -> Linear -> logits

    Attributes:
        num_labels: Number of BIO labels (15 = 7 types * 2 + O)
        hidden_size: Input hidden dimension from encoder
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 15,
        dropout_prob: float = 0.1,
    ):
        """
        Initialize NER head.

        Args:
            hidden_size: Input dimension from encoder
            num_labels: Number of output labels (BIO scheme)
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Store label mappings
        self.label_to_id = LABEL_TO_ID
        self.id_to_label = ID_TO_LABEL

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token classification.

        Args:
            encoder_output: (batch, seq_len, hidden_size) from encoder

        Returns:
            logits: (batch, seq_len, num_labels) classification logits
        """
        x = self.dropout(encoder_output)
        logits = self.classifier(x)
        return logits

    def predict(
        self,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get predicted label IDs.

        Args:
            encoder_output: (batch, seq_len, hidden_size) from encoder
            attention_mask: (batch, seq_len) to mask padding tokens

        Returns:
            predictions: (batch, seq_len) predicted label IDs
        """
        logits = self.forward(encoder_output)
        predictions = logits.argmax(dim=-1)

        # Mask padding tokens to O label
        if attention_mask is not None:
            predictions = predictions * attention_mask

        return predictions

    def decode_entities(
        self,
        predictions: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Dict]]:
        """
        Decode BIO predictions to entity spans.

        Args:
            predictions: (batch, seq_len) label IDs
            input_ids: (batch, seq_len) token IDs
            tokenizer: Tokenizer for decoding tokens
            attention_mask: (batch, seq_len) to identify valid tokens

        Returns:
            List of entity lists per batch item, each entity is a dict with:
                - text: Entity string
                - type: Entity type (COMPANY, TICKER, etc.)
                - start: Start token position
                - end: End token position
        """
        batch_size = predictions.shape[0]
        all_entities = []

        for batch_idx in range(batch_size):
            entities = []
            current_entity = None
            current_type = None
            start_pos = None

            seq_len = predictions.shape[1]
            if attention_mask is not None:
                valid_len = attention_mask[batch_idx].sum().item()
            else:
                valid_len = seq_len

            for pos in range(int(valid_len)):
                label_id = predictions[batch_idx, pos].item()
                label = self.id_to_label[label_id]

                if label.startswith("B-"):
                    # Save previous entity if exists
                    if current_entity is not None:
                        token_ids = input_ids[batch_idx, start_pos:pos].tolist()
                        text = tokenizer.decode(token_ids, skip_special_tokens=True)
                        entities.append({
                            "text": text.strip(),
                            "type": current_type,
                            "start": start_pos,
                            "end": pos,
                        })

                    # Start new entity
                    current_type = label[2:]  # Remove "B-" prefix
                    start_pos = pos
                    current_entity = True

                elif label.startswith("I-"):
                    # Continue entity only if type matches
                    entity_type = label[2:]
                    if current_entity is None or entity_type != current_type:
                        # Invalid I- tag, treat as O
                        if current_entity is not None:
                            token_ids = input_ids[batch_idx, start_pos:pos].tolist()
                            text = tokenizer.decode(token_ids, skip_special_tokens=True)
                            entities.append({
                                "text": text.strip(),
                                "type": current_type,
                                "start": start_pos,
                                "end": pos,
                            })
                        current_entity = None
                        current_type = None
                        start_pos = None
                    # Otherwise continue accumulating

                else:  # O label
                    # End current entity if exists
                    if current_entity is not None:
                        token_ids = input_ids[batch_idx, start_pos:pos].tolist()
                        text = tokenizer.decode(token_ids, skip_special_tokens=True)
                        entities.append({
                            "text": text.strip(),
                            "type": current_type,
                            "start": start_pos,
                            "end": pos,
                        })
                    current_entity = None
                    current_type = None
                    start_pos = None

            # Handle entity at end of sequence
            if current_entity is not None:
                token_ids = input_ids[batch_idx, start_pos:int(valid_len)].tolist()
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                entities.append({
                    "text": text.strip(),
                    "type": current_type,
                    "start": start_pos,
                    "end": int(valid_len),
                })

            all_entities.append(entities)

        return all_entities

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for NER training.

        Args:
            logits: (batch, seq_len, num_labels) prediction logits
            labels: (batch, seq_len) ground truth label IDs
            attention_mask: (batch, seq_len) to mask padding tokens

        Returns:
            loss: Scalar loss value
        """
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        # Reshape for loss computation
        # logits: (batch * seq_len, num_labels)
        # labels: (batch * seq_len,)
        batch_size, seq_len, num_labels = logits.shape
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        loss = loss_fn(logits_flat, labels_flat)
        loss = loss.view(batch_size, seq_len)

        # Apply attention mask
        if attention_mask is not None:
            loss = loss * attention_mask
            loss = loss.sum() / attention_mask.sum()
        else:
            loss = loss.mean()

        return loss

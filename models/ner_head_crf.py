"""NER Head with Conditional Random Fields (CRF).

Implements a more advanced NER architecture using CRF for better
sequence labeling with BIO tag constraints.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from torchcrf import CRF


# BIO tag constraints for CRF
def create_bio_constraints(num_labels: int, label_to_id: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Create BIO tag constraints for CRF.

    Rules:
    - I-X can only follow B-X or I-X of the same type
    - B-X can follow any tag
    - O can follow any tag

    Args:
        num_labels: Total number of labels
        label_to_id: Mapping from label names to IDs

    Returns:
        List of (from_tag, to_tag) tuples representing impossible transitions
    """
    id_to_label = {v: k for k, v in label_to_id.items()}
    impossible_transitions = []

    for from_id in range(num_labels):
        for to_id in range(num_labels):
            from_label = id_to_label.get(from_id, "")
            to_label = id_to_label.get(to_id, "")

            # I-X can only follow B-X or I-X of same type
            if to_label.startswith("I-"):
                entity_type = to_label[2:]  # Remove "I-"
                valid_prev = [f"B-{entity_type}", f"I-{entity_type}"]

                if from_label not in valid_prev:
                    impossible_transitions.append((from_id, to_id))

    return impossible_transitions


class NERHeadCRF(nn.Module):
    """
    NER Head with Conditional Random Fields.

    Architecture:
        Input → Linear → CRF → BIO Tags

    Benefits over simple classification:
    - Enforces BIO tag sequence constraints
    - Models dependencies between adjacent tags
    - Better handles class imbalance through sequence modeling
    - Prevents invalid tag sequences (e.g., I-COMPANY after O)

    Attributes:
        linear: Linear layer to project hidden states to tag scores
        crf: Conditional Random Field layer
        num_labels: Number of BIO tags
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        label_to_id: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize CRF-based NER head.

        Args:
            hidden_size: Hidden dimension from encoder
            num_labels: Number of BIO labels
            label_to_id: Optional mapping for BIO constraints
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.label_to_id = label_to_id or {}

        # Linear projection layer
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

        # Set up BIO constraints if label mapping provided
        if label_to_id:
            self._setup_bio_constraints()

        # Initialize weights
        self._init_weights()

    def _setup_bio_constraints(self):
        """Setup BIO tag transition constraints."""
        impossible_transitions = create_bio_constraints(self.num_labels, self.label_to_id)

        # Set impossible transitions to very negative values
        if impossible_transitions:
            with torch.no_grad():
                for from_id, to_id in impossible_transitions:
                    self.crf.transitions[from_id, to_id] = -10000.0

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize linear layer
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CRF NER.

        Args:
            hidden_states: (batch, seq_len, hidden_size) from encoder
            labels: (batch, seq_len) BIO label IDs for training
            attention_mask: (batch, seq_len) attention mask

        Returns:
            Dictionary containing:
            - logits: (batch, seq_len, num_labels) tag scores
            - loss: CRF loss (if labels provided)
            - predictions: (batch, seq_len) predicted tag IDs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Apply dropout and linear projection
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)  # (batch, seq_len, num_labels)

        results = {"logits": logits}

        # Create mask for CRF (exclude padding tokens)
        if attention_mask is None:
            crf_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        else:
            crf_mask = attention_mask.bool()

        # Training mode: compute CRF loss
        if labels is not None and self.training:
            # CRF loss (negative log-likelihood)
            crf_loss = self.crf(logits, labels, mask=crf_mask, reduction='mean')
            results["loss"] = -crf_loss  # CRF returns log-likelihood, we want loss

        # Inference: Viterbi decoding for best sequence
        predictions = self.crf.decode(logits, mask=crf_mask)

        # Convert predictions to tensor (CRF.decode returns list)
        pred_tensor = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
        for i, pred_seq in enumerate(predictions):
            seq_len_i = len(pred_seq)
            pred_tensor[i, :seq_len_i] = torch.tensor(pred_seq, device=hidden_states.device)

        results["predictions"] = pred_tensor

        return results

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
            predictions: (batch, seq_len) predicted label IDs
            input_ids: (batch, seq_len) input token IDs
            tokenizer: Tokenizer for converting IDs to tokens
            attention_mask: (batch, seq_len) attention mask

        Returns:
            List of entity lists per batch item
        """
        batch_entities = []
        id_to_label = {v: k for k, v in self.label_to_id.items()} if self.label_to_id else {}

        for i in range(predictions.shape[0]):
            entities = []
            current_entity = None

            # Get valid sequence length
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
            else:
                valid_len = predictions.shape[1]

            for j in range(valid_len):
                pred_id = predictions[i, j].item()
                label = id_to_label.get(pred_id, "O")

                if label.startswith("B-"):
                    # Start new entity
                    if current_entity:
                        entities.append(current_entity)

                    entity_type = label[2:]  # Remove "B-"
                    token_id = input_ids[i, j].item()
                    token_text = tokenizer.decode([token_id])

                    current_entity = {
                        "type": entity_type,
                        "text": token_text,
                        "start": j,
                        "end": j + 1,
                    }

                elif label.startswith("I-") and current_entity:
                    # Continue entity
                    entity_type = label[2:]
                    if current_entity["type"] == entity_type:
                        token_id = input_ids[i, j].item()
                        token_text = tokenizer.decode([token_id])
                        current_entity["text"] += token_text
                        current_entity["end"] = j + 1
                    else:
                        # Type mismatch - end current entity and ignore this token
                        entities.append(current_entity)
                        current_entity = None

                else:
                    # O tag or invalid sequence - end current entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            # Add final entity if exists
            if current_entity:
                entities.append(current_entity)

            batch_entities.append(entities)

        return batch_entities

    def get_transition_matrix(self) -> torch.Tensor:
        """Get the learned CRF transition matrix."""
        return self.crf.transitions.data

    def print_transitions(self, id_to_label: Optional[Dict[int, str]] = None):
        """Print learned transition probabilities."""
        transitions = self.get_transition_matrix()

        if id_to_label is None:
            id_to_label = {i: f"TAG_{i}" for i in range(self.num_labels)}

        print("CRF Transition Matrix (from → to):")
        print("=" * 50)

        # Header
        header = "From\\To".ljust(12)
        for to_id in range(self.num_labels):
            header += f"{id_to_label.get(to_id, str(to_id)):>8}"
        print(header)
        print("-" * len(header))

        # Rows
        for from_id in range(self.num_labels):
            row = f"{id_to_label.get(from_id, str(from_id)):10}  "
            for to_id in range(self.num_labels):
                score = transitions[from_id, to_id].item()
                row += f"{score:>8.2f}"
            print(row)


class CRFLoss(nn.Module):
    """
    Wrapper for CRF loss with additional features.
    """

    def __init__(
        self,
        num_labels: int,
        label_to_id: Optional[Dict[str, int]] = None,
        ignore_index: int = -100,
    ):
        """
        Initialize CRF loss.

        Args:
            num_labels: Number of labels
            label_to_id: Label mapping for constraints
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()

        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.crf = CRF(num_labels, batch_first=True)

        # Setup constraints
        if label_to_id:
            impossible_transitions = create_bio_constraints(num_labels, label_to_id)
            with torch.no_grad():
                for from_id, to_id in impossible_transitions:
                    self.crf.transitions[from_id, to_id] = -10000.0

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CRF loss.

        Args:
            logits: (batch, seq_len, num_labels)
            labels: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            CRF loss (negative log-likelihood)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # Handle ignore_index
        if self.ignore_index != -100:
            # Mask out ignored positions
            valid_mask = (labels != self.ignore_index) & attention_mask
        else:
            valid_mask = attention_mask

        # Compute CRF loss
        log_likelihood = self.crf(logits, labels, mask=valid_mask, reduction='mean')

        return -log_likelihood


# Utility function to update existing model
def upgrade_to_crf_ner(
    model,
    label_to_id: Dict[str, int],
    hidden_size: int = 768,
    dropout: float = 0.1,
):
    """
    Upgrade existing model to use CRF-based NER head.

    Args:
        model: Existing FinancialEntitySentimentModel
        label_to_id: BIO label mapping
        hidden_size: Hidden dimension
        dropout: Dropout rate

    Returns:
        Model with upgraded CRF NER head
    """
    # Replace NER head with CRF version
    model.ner_head = NERHeadCRF(
        hidden_size=hidden_size,
        num_labels=len(label_to_id),
        label_to_id=label_to_id,
        dropout=dropout,
    )

    # Move to same device
    if hasattr(model, '_device'):
        model.ner_head = model.ner_head.to(model._device)

    return model
"""Shared Longformer Encoder Module.

The encoder provides contextualized token representations with configurable
global attention for entity-focused processing.
"""

import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from typing import List, Tuple, Optional


class LongformerEncoder(nn.Module):
    """
    Shared Longformer encoder with runtime-configurable global attention.

    Longformer uses a combination of local sliding window attention and
    global attention. We configure entity tokens to have global attention,
    allowing them to attend to all tokens in the document and vice versa.

    Attributes:
        model: The underlying Longformer model
        tokenizer: Associated tokenizer
        hidden_size: Output hidden dimension (768 for base)
        max_length: Maximum sequence length (4096)
    """

    MODEL_NAME = "allenai/longformer-base-4096"

    def __init__(
        self,
        model_name: str = None,
        hidden_size: int = 768,
        max_length: int = 4096,
        local_attention_window: int = 512,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize the Longformer encoder.

        Args:
            model_name: HuggingFace model identifier
            hidden_size: Hidden dimension size
            max_length: Maximum sequence length
            local_attention_window: Size of local attention window
            gradient_checkpointing: If True, trade ~30% extra compute for
                ~3x activation-memory savings. Enables larger batch sizes
                on Longformer-Large at seq=2048.
        """
        super().__init__()

        model_name = model_name or self.MODEL_NAME

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.local_attention_window = local_attention_window

        # Load pretrained model and tokenizer
        self.model = LongformerModel.from_pretrained(model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)

        if gradient_checkpointing:
            # HF transformers requires use_cache=False alongside checkpointing
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

    def create_global_attention_mask(
        self,
        input_ids: torch.Tensor,
        entity_positions: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        """
        Create attention mask where entity tokens attend globally.

        Global attention allows specific tokens to attend to ALL other tokens
        in the sequence, and ALL tokens attend to them. This is critical for
        capturing sentiment expressed anywhere in the document about entities.

        Args:
            input_ids: (batch, seq_len) token IDs
            entity_positions: List of (start, end) tuples per batch item.
                            Each tuple defines a span of entity tokens.

        Returns:
            global_attention_mask: (batch, seq_len) with 1s for global attention
        """
        batch_size, seq_len = input_ids.shape
        global_attention_mask = torch.zeros(
            batch_size, seq_len,
            dtype=torch.long,
            device=input_ids.device
        )

        # CLS token always gets global attention
        global_attention_mask[:, 0] = 1

        # Entity positions get global attention
        if entity_positions is not None:
            for batch_idx, positions in enumerate(entity_positions):
                if positions is not None:
                    for start, end in positions:
                        global_attention_mask[batch_idx, start:end] = 1

        return global_attention_mask

    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        padding: str = "max_length",
    ) -> dict:
        """
        Tokenize input text.

        Args:
            text: Input text string
            return_tensors: Format of returned tensors
            truncation: Whether to truncate to max_length
            padding: Padding strategy

        Returns:
            Dictionary containing input_ids and attention_mask
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            max_length=self.max_length,
            truncation=truncation,
            padding=padding,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) standard attention mask
            global_attention_mask: (batch, seq_len) global attention positions
            output_attentions: Whether to return attention weights

        Returns:
            last_hidden_state: (batch, seq_len, hidden_size) contextualized embeddings
        """
        # Default global attention on CLS if not provided
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
        )

        return outputs.last_hidden_state

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the CLS token embedding for document-level representation.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            global_attention_mask: (batch, seq_len) global attention positions

        Returns:
            cls_embedding: (batch, hidden_size) document embedding
        """
        hidden_states = self.forward(
            input_ids, attention_mask, global_attention_mask
        )
        return hidden_states[:, 0, :]  # CLS token is at position 0

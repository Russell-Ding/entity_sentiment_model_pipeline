"""Financial Entity Sentiment Analysis Pipeline.

Complete model combining encoder and all task heads for entity-level
sentiment analysis on financial news articles.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from .encoder import LongformerEncoder
from .ner_head import NERHead
from .ner_head_crf import NERHeadCRF
from .coref_head import CorefHead, DEFAULT_TICKER_MAP
from .sentiment_head import SentimentHead
from .utils import get_device, device_to_str, get_device_info


class FinancialEntitySentimentModel(nn.Module):
    """
    Complete pipeline for entity-level sentiment analysis on financial news.

    Architecture:
        Input Article
             │
             ▼
        ┌─────────────────────────────┐
        │    Longformer Encoder       │
        │  (shared representations)   │
        └─────────────────────────────┘
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
    ┌─────┐ ┌─────┐ ┌─────┐
    │ NER │ │Coref│ │Sent.│
    │Head │ │Head │ │Head │
    └─────┘ └─────┘ └─────┘
       │     │     │
       └─────┼─────┘
             ▼
        Entity-level Sentiment Scores

    Components:
        - Longformer encoder: Shared contextualized representations with
          runtime-configurable global attention for entity tokens
        - NER head: Optional token classification for entity detection
        - Coreference head: FastCoref wrapper for mention linking
        - Sentiment head: Attention-based regression to [-1, 1] scores

    Attributes:
        encoder: Shared Longformer encoder
        ner_head: Named entity recognition head (optional)
        coref_head: Coreference resolution head
        sentiment_head: Entity sentiment regression head
    """

    def __init__(
        self,
        encoder_name: str = "allenai/longformer-base-4096",
        hidden_size: int = 768,
        num_attention_heads: int = 8,
        num_ner_labels: int = 15,
        use_ner_head: bool = True,
        use_coref_head: bool = True,
        use_crf_ner: bool = False,
        ner_label_to_id: Optional[Dict[str, int]] = None,
        device: Optional[Union[str, torch.device]] = None,
        ticker_map: Optional[Dict[str, List[str]]] = None,
        max_length: int = 4096,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize the complete pipeline model.

        Args:
            encoder_name: HuggingFace model identifier for encoder
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads in sentiment head
            num_ner_labels: Number of NER labels (BIO scheme)
            use_ner_head: Whether to include NER head
            use_coref_head: Whether to include coreference head
            device: Compute device. Options:
                - None or "auto": Auto-detect (cuda > mps > cpu)
                - "cuda" or "cuda:0": Use NVIDIA GPU
                - "mps": Use Apple Silicon GPU
                - "cpu": Use CPU
            ticker_map: Optional ticker-company mappings for coreference
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()

        # Resolve device with auto-detection
        resolved_device = get_device(device)
        self._device = resolved_device
        self._device_str = device_to_str(resolved_device)

        self.use_ner_head = use_ner_head
        self.use_coref_head = use_coref_head
        self.use_crf_ner = use_crf_ner
        self.hidden_size = hidden_size
        self.encoder_name = encoder_name
        self.max_length = max_length

        # Shared encoder
        self.encoder = LongformerEncoder(
            model_name=encoder_name,
            hidden_size=hidden_size,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Task heads
        if use_ner_head:
            if use_crf_ner:
                self.ner_head = NERHeadCRF(
                    hidden_size=hidden_size,
                    num_labels=num_ner_labels,
                    label_to_id=ner_label_to_id,
                    dropout=0.1,
                )
            else:
                self.ner_head = NERHead(
                    hidden_size=hidden_size,
                    num_labels=num_ner_labels,
                )

        if use_coref_head:
            # CorefHead handles its own device management
            self.coref_head = CorefHead(device=self._device_str)
            if ticker_map:
                self.coref_head.set_ticker_map(ticker_map)
            else:
                self.coref_head.set_ticker_map(DEFAULT_TICKER_MAP)

        self.sentiment_head = SentimentHead(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
        )

    @property
    def tokenizer(self):
        """Access tokenizer from encoder."""
        return self.encoder.tokenizer

    def _find_entity_token_positions(
        self,
        input_ids: torch.Tensor,
        entity_tokens: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Find all occurrences of entity tokens in input sequence.

        Args:
            input_ids: (seq_len,) token IDs
            entity_tokens: List of token IDs for the entity

        Returns:
            List of (start, end) token positions
        """
        positions = []
        input_list = input_ids.tolist()
        entity_len = len(entity_tokens)

        for i in range(len(input_list) - entity_len + 1):
            if input_list[i:i + entity_len] == entity_tokens:
                positions.append((i, i + entity_len))

        return positions

    def _create_entity_mask(
        self,
        seq_len: int,
        positions: List[Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create binary mask for entity token positions.

        Args:
            seq_len: Sequence length
            positions: List of (start, end) positions
            device: Target device

        Returns:
            mask: (seq_len,) binary mask with 1s at entity positions
        """
        mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
        for start, end in positions:
            mask[start:end] = 1.0
        return mask

    def _expand_positions_with_coref(
        self,
        text: str,
        entity: str,
        base_positions: List[Tuple[int, int]],
        input_ids: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Expand entity positions with coreference mentions.

        Args:
            text: Original text
            entity: Target entity string
            base_positions: Initial entity token positions
            input_ids: (seq_len,) token IDs

        Returns:
            All token positions including coreferences
        """
        if not self.use_coref_head:
            return base_positions

        # Get all coreferent mentions
        all_mentions = self.coref_head.expand_entity_mentions(
            text, entity, use_augmentation=True
        )

        # Find token positions for each mention
        all_positions = list(base_positions)
        for mention in all_mentions:
            if mention.lower() == entity.lower():
                continue
            mention_tokens = self.tokenizer.encode(
                mention, add_special_tokens=False
            )
            positions = self._find_entity_token_positions(input_ids, mention_tokens)
            all_positions.extend(positions)

        # Remove duplicates and sort
        all_positions = sorted(set(all_positions), key=lambda x: x[0])
        return all_positions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_masks: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        ner_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for joint NER and sentiment training.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) standard attention mask
            entity_masks: (batch, num_entities, seq_len) binary masks for each entity
            global_attention_mask: (batch, seq_len) for Longformer global attention
            ner_labels: (batch, seq_len) NER labels for CRF training (optional)

        Returns:
            ner_output: Either tensor (logits) or dict (CRF output)
            sentiment_scores: (batch, num_entities) sentiment scores per entity
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Create global attention mask if not provided
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # Always include CLS token
            # Add global attention to entity positions if provided
            if entity_masks is not None:
                # Aggregate all entity positions
                entity_positions = entity_masks.sum(dim=1).clamp(0, 1).long()
                global_attention_mask = global_attention_mask | entity_positions

        # Encode
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # NER predictions (token-level)
        if self.use_ner_head:
            if self.use_crf_ner:
                # CRF head returns dict with logits, loss, and predictions
                ner_output = self.ner_head(
                    encoder_output,
                    labels=ner_labels,
                    attention_mask=attention_mask
                )
            else:
                # Standard linear head
                ner_output = self.ner_head(encoder_output)
        else:
            # Return dummy NER output if head not enabled
            ner_output = torch.zeros(
                batch_size, seq_len, 15,  # 15 NER labels
                device=device,
            )

        # Sentiment predictions (entity-level)
        if entity_masks is not None:
            num_entities = entity_masks.shape[1]
            sentiment_scores = torch.zeros(batch_size, num_entities, device=device)

            # Process each entity's mask
            for i in range(num_entities):
                entity_mask = entity_masks[:, i, :]  # (batch, seq_len)
                scores = self.sentiment_head(encoder_output, entity_mask)
                sentiment_scores[:, i] = scores
        else:
            # No entities - return zeros
            sentiment_scores = torch.zeros(batch_size, 1, device=device)

        return ner_output, sentiment_scores

    def forward_ner(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ner_labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for NER only.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            ner_labels: (batch, seq_len) NER labels for CRF training (optional)

        Returns:
            ner_output: Either tensor (logits) or dict (CRF output)
        """
        if not self.use_ner_head:
            raise ValueError("NER head not enabled")

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.use_crf_ner:
            return self.ner_head(
                encoder_output,
                labels=ner_labels,
                attention_mask=attention_mask
            )
        else:
            return self.ner_head(encoder_output)

    @torch.no_grad()
    def analyze(
        self,
        text: str,
        target_entities: Optional[List[str]] = None,
        use_coref: bool = True,
    ) -> Dict[str, Dict]:
        """
        Full inference pipeline for a single article.

        Args:
            text: Article text
            target_entities: List of entities to analyze.
                           If None and NER head enabled, detects entities.
            use_coref: Whether to expand entities with coreference

        Returns:
            Dict mapping entity names to results:
                {
                    "entity_name": {
                        "sentiment_score": float,
                        "mentions": List[Tuple[int, int]],
                        "num_mentions": int
                    }
                }
        """
        self.eval()
        device = next(self.parameters()).device

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding="max_length",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Get entities
        if target_entities is None:
            if self.use_ner_head:
                # Detect entities with NER
                encoder_output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                if self.use_crf_ner:
                    ner_output = self.ner_head(encoder_output, attention_mask=attention_mask)
                    predictions = ner_output["predictions"]
                else:
                    ner_logits = self.ner_head(encoder_output)
                    predictions = ner_logits.argmax(dim=-1)
                entities_list = self.ner_head.decode_entities(
                    predictions, input_ids, self.tokenizer, attention_mask
                )
                target_entities = [e["text"] for e in entities_list[0]]
            else:
                return {"error": "No entities provided and NER head not enabled"}

        if not target_entities:
            return {"entities": [], "message": "No entities found or provided"}

        results = {}

        for entity in target_entities:
            # Find base entity positions
            # Try without leading space first
            entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False)
            base_positions = self._find_entity_token_positions(
                input_ids[0], entity_tokens
            )

            # If not found, try with leading space (RoBERTa/Longformer tokenization quirk)
            if not base_positions:
                entity_tokens_with_space = self.tokenizer.encode(" " + entity, add_special_tokens=False)
                base_positions = self._find_entity_token_positions(
                    input_ids[0], entity_tokens_with_space
                )

            if not base_positions:
                results[entity] = {
                    "sentiment_score": None,
                    "error": "Entity not found in text",
                }
                continue

            # Expand with coreference
            if use_coref and self.use_coref_head:
                all_positions = self._expand_positions_with_coref(
                    text, entity, base_positions, input_ids[0]
                )
            else:
                all_positions = base_positions

            # Create masks
            seq_len = input_ids.shape[1]
            entity_mask = self._create_entity_mask(seq_len, all_positions, device)
            entity_mask = entity_mask.unsqueeze(0)

            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # CLS token
            for start, end in all_positions:
                global_attention_mask[:, start:end] = 1

            # Forward pass
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )

            sentiment_score = self.sentiment_head(encoder_output, entity_mask)

            results[entity] = {
                "sentiment_score": sentiment_score.item(),
                "mentions": all_positions,
                "num_mentions": len(all_positions),
            }

        return results

    @torch.no_grad()
    def batch_analyze(
        self,
        articles: List[Dict[str, Union[str, List[str]]]],
        use_coref: bool = True,
    ) -> List[Dict]:
        """
        Analyze multiple articles.

        Args:
            articles: List of {"text": str, "entities": List[str]}
            use_coref: Whether to use coreference expansion

        Returns:
            List of result dicts per article
        """
        results = []
        for article in articles:
            result = self.analyze(
                text=article["text"],
                target_entities=article.get("entities"),
                use_coref=use_coref,
            )
            results.append(result)
        return results

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    def save_pretrained(self, save_path: str) -> None:
        """
        Save model weights and configuration.

        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), save_path / "model.pt")

        # Save config
        config = {
            "encoder_name": self.encoder_name,
            "hidden_size": self.hidden_size,
            "use_ner_head": self.use_ner_head,
            "use_coref_head": self.use_coref_head,
            "use_crf_ner": self.use_crf_ner,
            "num_ner_labels": self.ner_head.num_labels if self.use_ner_head else 15,
            "ner_label_to_id": self.ner_head.label_to_id if (self.use_ner_head and self.use_crf_ner) else None,
        }
        import json
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / "tokenizer")

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        device: Optional[Union[str, torch.device]] = None,
        exclude_ner_head: bool = False,
        exclude_sentiment_head: bool = False,
        use_crf_ner: Optional[bool] = None,
        ner_label_to_id: Optional[Dict[str, int]] = None,
        num_ner_labels: Optional[int] = None,
    ):
        """
        Load model from saved checkpoint.

        Args:
            load_path: Directory containing saved model
            device: Device to load to. Options:
                - None or "auto": Auto-detect (cuda > mps > cpu)
                - "cuda" or "cuda:0": Use NVIDIA GPU
                - "mps": Use Apple Silicon GPU
                - "cpu": Use CPU
            exclude_ner_head: If True, don't load NER head weights from checkpoint.
                Useful when switching from standard NER to CRF NER head.
            exclude_sentiment_head: If True, don't load sentiment head weights from
                checkpoint. Useful when migrating from v1 to v2 sentiment head.
            use_crf_ner: Override config to use CRF NER head. If None, uses saved config.
            ner_label_to_id: Override label mapping for CRF constraints.
            num_ner_labels: Override number of NER labels.

        Returns:
            Loaded model instance
        """
        load_path = Path(load_path)

        # Resolve device
        resolved_device = get_device(device)

        # Load config
        import json
        with open(load_path / "config.json") as f:
            config = json.load(f)

        # Allow overriding NER config for architecture changes
        final_use_crf = use_crf_ner if use_crf_ner is not None else config.get("use_crf_ner", False)
        final_num_labels = num_ner_labels if num_ner_labels is not None else config.get("num_ner_labels", 15)
        final_label_to_id = ner_label_to_id if ner_label_to_id is not None else config.get("ner_label_to_id")

        # Create model
        model = cls(
            encoder_name=config.get("encoder_name", "allenai/longformer-base-4096"),
            hidden_size=config["hidden_size"],
            use_ner_head=config["use_ner_head"],
            use_coref_head=config["use_coref_head"],
            use_crf_ner=final_use_crf,
            num_ner_labels=final_num_labels,
            ner_label_to_id=final_label_to_id,
            device=resolved_device,
        )

        # Load weights
        state_dict = torch.load(
            load_path / "model.pt",
            map_location=resolved_device,
            weights_only=True,
        )

        # Optionally exclude head weights (for architecture changes)
        if exclude_ner_head:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ner_head.")}
        if exclude_sentiment_head:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("sentiment_head.")}

        if exclude_ner_head or exclude_sentiment_head:
            model.load_state_dict(state_dict, strict=False)
            excluded = []
            if exclude_ner_head:
                excluded.append("NER")
            if exclude_sentiment_head:
                excluded.append("sentiment")
            print(f"Loaded checkpoint excluding {' and '.join(excluded)} head weights. Excluded heads initialized fresh.")
        else:
            model.load_state_dict(state_dict)

        model.to(resolved_device)
        return model

    def freeze_except_ner(self) -> None:
        """
        Freeze all parameters except the NER head.
        Useful for fine-tuning only the NER head on new data.
        """
        for name, param in self.named_parameters():
            if name.startswith("ner_head."):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Frozen all except NER head: {trainable:,} / {total:,} parameters trainable")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for full model training."""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")

    def freeze_encoder(self) -> None:
        """Freeze only the encoder, keep all heads trainable."""
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Frozen encoder: {trainable:,} / {total:,} parameters trainable")

    @staticmethod
    def get_available_devices() -> dict:
        """
        Get information about available compute devices.

        Returns:
            dict: Device availability info including recommended device
        """
        return get_device_info()

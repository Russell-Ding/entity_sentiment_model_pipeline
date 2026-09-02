"""Coreference Resolution Head Module.

Wrapper around FastCoref for linking entity mentions (pronouns, aliases, abbreviations).
Includes financial domain augmentation with ticker-company mappings.
"""

import json
import warnings
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union

from .utils import get_device, device_to_str


class CorefHead:
    """
    Coreference resolution head using FastCoref.

    Links entity mentions across the document including:
    - Pronouns (he, she, it, they)
    - Noun phrases (the company, the CEO)
    - Abbreviations and aliases

    Also provides financial domain augmentation via ticker-company mappings
    to link "AAPL" with "Apple Inc." and "Apple".

    Note:
        FastCoref may not fully support MPS. When MPS is specified,
        the model will attempt to use it but may fall back to CPU
        if issues are encountered.

    Attributes:
        model: FastCoref model instance
        ticker_map: Mapping from tickers to company names
        device: Compute device
    """

    def __init__(
        self,
        device: Optional[Union[str, "torch.device"]] = None,
        ticker_map_path: Optional[str] = None,
    ):
        """
        Initialize coreference head.

        Args:
            device: Compute device for FastCoref. Options:
                - None or "auto": Auto-detect (cuda > mps > cpu)
                - "cuda" or "cuda:0": Use NVIDIA GPU
                - "mps": Use Apple Silicon GPU (may fall back to CPU)
                - "cpu": Use CPU
            ticker_map_path: Path to ticker-company mapping JSON file
        """
        # Resolve device
        import torch
        resolved_device = get_device(device)
        self._device_type = resolved_device.type

        # FastCoref expects device as string
        # Note: FastCoref may not support MPS well, we'll handle fallback in model loading
        if self._device_type == "mps":
            # Try MPS first, but be prepared to fall back
            self._coref_device = "mps"
            self._fallback_to_cpu = True
        elif self._device_type == "cuda":
            self._coref_device = device_to_str(resolved_device)
            self._fallback_to_cpu = False
        else:
            self._coref_device = "cpu"
            self._fallback_to_cpu = False

        self.device = self._coref_device
        self.ticker_map: Dict[str, List[str]] = {}
        self._model = None  # Lazy loading

        if ticker_map_path:
            self.load_ticker_map(ticker_map_path)

    @property
    def model(self):
        """Lazy load FastCoref model."""
        if self._model is None:
            try:
                from fastcoref import FCoref
            except ImportError:
                raise ImportError(
                    "fastcoref is required for coreference resolution. "
                    "Install with: pip install fastcoref"
                )

            try:
                self._model = FCoref(device=self._coref_device)
            except Exception as e:
                # Handle MPS fallback
                if self._fallback_to_cpu and self._coref_device == "mps":
                    warnings.warn(
                        f"FastCoref failed to initialize on MPS: {e}. "
                        "Falling back to CPU for coreference resolution."
                    )
                    self._coref_device = "cpu"
                    self.device = "cpu"
                    self._model = FCoref(device="cpu")
                else:
                    raise

        return self._model

    def load_ticker_map(self, path: str) -> None:
        """
        Load ticker to company name mappings.

        Expected JSON format:
        {
            "AAPL": ["Apple Inc.", "Apple"],
            "GOOGL": ["Alphabet Inc.", "Google"],
            ...
        }

        Args:
            path: Path to JSON file
        """
        with open(path) as f:
            self.ticker_map = json.load(f)

    def set_ticker_map(self, ticker_map: Dict[str, List[str]]) -> None:
        """
        Set ticker map directly.

        Args:
            ticker_map: Dict mapping ticker symbols to company name variants
        """
        self.ticker_map = ticker_map

    def predict(self, texts: List[str]) -> List[List[List[str]]]:
        """
        Get coreference clusters for texts.

        Args:
            texts: List of document strings

        Returns:
            List of cluster lists per document.
            Each cluster is a list of coreferent mention strings.
        """
        predictions = self.model.predict(texts=texts)
        return [pred.get_clusters() for pred in predictions]

    def predict_with_spans(
        self,
        texts: List[str],
    ) -> List[List[List[Tuple[int, int, str]]]]:
        """
        Get coreference clusters with character spans.

        Args:
            texts: List of document strings

        Returns:
            List of cluster lists per document.
            Each cluster is a list of (start, end, text) tuples.
        """
        predictions = self.model.predict(texts=texts)
        results = []

        for pred in predictions:
            clusters = pred.get_clusters(as_strings=False)
            text = pred.text
            cluster_spans = []

            for cluster in clusters:
                spans = []
                for start, end in cluster:
                    mention = text[start:end]
                    spans.append((start, end, mention))
                cluster_spans.append(spans)

            results.append(cluster_spans)

        return results

    def augment_with_tickers(
        self,
        coref_clusters: List[List[str]],
    ) -> List[List[str]]:
        """
        Augment coreference clusters with ticker-company mappings.

        If a cluster contains a ticker or company name from the mapping,
        adds all related variants to that cluster.

        Args:
            coref_clusters: List of coreference clusters

        Returns:
            Augmented clusters with ticker-company links
        """
        augmented = []

        for cluster in coref_clusters:
            expanded_cluster: Set[str] = set(cluster)

            for mention in cluster:
                # Check if mention is a known ticker
                mention_upper = mention.upper()
                if mention_upper in self.ticker_map:
                    expanded_cluster.update(self.ticker_map[mention_upper])
                    expanded_cluster.add(mention_upper)

                # Check if mention matches any company name
                for ticker, names in self.ticker_map.items():
                    for name in names:
                        if (mention.lower() == name.lower() or
                            mention.lower() in name.lower() or
                            name.lower() in mention.lower()):
                            expanded_cluster.add(ticker)
                            expanded_cluster.update(names)
                            break

            augmented.append(list(expanded_cluster))

        return augmented

    def get_entity_cluster(
        self,
        entity: str,
        coref_clusters: List[List[str]],
    ) -> Optional[List[str]]:
        """
        Find the coreference cluster containing a specific entity.

        Args:
            entity: Target entity string
            coref_clusters: List of coreference clusters

        Returns:
            The cluster containing the entity, or None if not found
        """
        entity_lower = entity.lower()

        for cluster in coref_clusters:
            for mention in cluster:
                if (entity_lower == mention.lower() or
                    entity_lower in mention.lower() or
                    mention.lower() in entity_lower):
                    return cluster

        return None

    def expand_entity_mentions(
        self,
        text: str,
        entity: str,
        use_augmentation: bool = True,
    ) -> List[str]:
        """
        Get all mentions that refer to the given entity.

        Args:
            text: Document text
            entity: Target entity
            use_augmentation: Whether to apply ticker-company augmentation

        Returns:
            List of all coreferent mentions for the entity
        """
        # Get coreference clusters
        clusters = self.predict([text])[0]

        # Augment with ticker mappings
        if use_augmentation:
            clusters = self.augment_with_tickers(clusters)

        # Find entity's cluster
        entity_cluster = self.get_entity_cluster(entity, clusters)

        if entity_cluster:
            return entity_cluster
        else:
            # Return just the entity if no cluster found
            return [entity]

    def find_mention_positions(
        self,
        text: str,
        mentions: List[str],
    ) -> List[Tuple[int, int]]:
        """
        Find character positions of all mentions in text.

        Args:
            text: Document text
            mentions: List of mention strings to find

        Returns:
            List of (start, end) character positions
        """
        positions = []
        text_lower = text.lower()

        for mention in mentions:
            mention_lower = mention.lower()
            start = 0

            while True:
                pos = text_lower.find(mention_lower, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(mention)))
                start = pos + 1

        # Sort by position and remove overlaps
        positions = sorted(set(positions), key=lambda x: x[0])
        return positions


# Default ticker mappings for common stocks
DEFAULT_TICKER_MAP = {
    "AAPL": ["Apple Inc.", "Apple"],
    "GOOGL": ["Alphabet Inc.", "Google", "Alphabet"],
    "GOOG": ["Alphabet Inc.", "Google", "Alphabet"],
    "MSFT": ["Microsoft Corporation", "Microsoft"],
    "AMZN": ["Amazon.com Inc.", "Amazon"],
    "META": ["Meta Platforms Inc.", "Meta", "Facebook"],
    "TSLA": ["Tesla Inc.", "Tesla"],
    "NVDA": ["NVIDIA Corporation", "NVIDIA", "Nvidia"],
    "JPM": ["JPMorgan Chase & Co.", "JPMorgan", "JP Morgan"],
    "GS": ["Goldman Sachs Group Inc.", "Goldman Sachs", "Goldman"],
    "BAC": ["Bank of America Corporation", "Bank of America", "BofA"],
    "WMT": ["Walmart Inc.", "Walmart"],
    "JNJ": ["Johnson & Johnson"],
    "V": ["Visa Inc.", "Visa"],
    "MA": ["Mastercard Incorporated", "Mastercard"],
    "DIS": ["The Walt Disney Company", "Disney", "Walt Disney"],
    "NFLX": ["Netflix Inc.", "Netflix"],
    "INTC": ["Intel Corporation", "Intel"],
    "AMD": ["Advanced Micro Devices Inc.", "AMD"],
    "CRM": ["Salesforce Inc.", "Salesforce"],
}

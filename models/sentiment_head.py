"""Entity-Focused Sentiment Regression Head.

Produces a continuous sentiment score in [-0.95, 0.95] for each entity using
cross-attention from entity-token queries to the full document context.

Architecture:
    encoder_output ─┬─ extract entity tokens ──→ Q  ─┐
                    │                                 │
                    └─ full sequence ──→ K, V ────────┤
                                                      │
                          CrossAttention(Q, K, V) ────┘
                                      │
                          Mean-pool over entity queries + residual
                                      │
                                  LayerNorm
                                      │
                          FC1 (H → H/2) → GELU → Dropout
                          FC2 (H/2 → H/8) → GELU → Dropout
                          FC3 (H/8 → 1) → scaled tanh
                                      │
                              score ∈ [-0.95, 0.95]

The cross-attention pattern lets each entity token gather sentiment signal
from context words like "surged", "sued", "profitable" anywhere in the
document. The attention matrix is (n_entity_tokens × seq_len), typically
40-400× smaller than full self-attention over the sequence.

Loss: combined MSE + (1 - Pearson r) on valid entities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SentimentHead(nn.Module):
    """Entity cross-attention sentiment regression head.

    Replaces the legacy self-attention V1 head that restricted keys to entity
    positions only (which prevented entities from seeing sentiment-bearing
    context). Removed in favor of this cross-attention design after the
    Apr 2026 audit.

    Attributes:
        hidden_size: Encoder output dimension.
        num_heads: Cross-attention heads.
        intermediate_size: First MLP layer size (default hidden_size // 2).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size or hidden_size // 2

        # Cross-attention: entity queries attend to full context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Layer norms
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # Deeper regression MLP with bottleneck
        self.fc1 = nn.Linear(hidden_size, self.intermediate_size)        # H → H/2
        self.fc2 = nn.Linear(self.intermediate_size, hidden_size // 8)   # H/2 → H/8
        self.fc3 = nn.Linear(hidden_size // 8, 1)                        # H/8 → 1
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        """Xavier init for FC layers, zero bias on output layer."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)  # Start predicting ~0 (neutral)

    def _extract_entity_tokens(
        self,
        encoder_output: torch.Tensor,
        entity_position_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Extract entity token embeddings into a dense (non-padded) tensor.

        Args:
            encoder_output: (batch, seq_len, hidden_size)
            entity_position_mask: (batch, seq_len) binary mask

        Returns:
            entity_tokens: (batch, max_entity_tokens, hidden_size)
            entity_padding_mask: (batch, max_entity_tokens) True=pad position
            max_entity_tokens: int
        """
        batch_size, seq_len, hidden = encoder_output.shape
        device = encoder_output.device

        counts = entity_position_mask.sum(dim=1).long()
        max_entity_tokens = max(counts.max().item(), 1)

        entity_tokens = torch.zeros(batch_size, max_entity_tokens, hidden, device=device)
        entity_padding_mask = torch.ones(batch_size, max_entity_tokens, dtype=torch.bool, device=device)

        flat_indices = entity_position_mask.bool().nonzero(as_tuple=False)
        if flat_indices.shape[0] > 0:
            batch_ids = flat_indices[:, 0]
            seq_ids = flat_indices[:, 1]
            # Within-sample running index for scatter
            cum = torch.zeros(batch_size, dtype=torch.long, device=device)
            within_idx = torch.empty_like(batch_ids)
            for i in range(batch_ids.shape[0]):
                b = batch_ids[i]
                within_idx[i] = cum[b]
                cum[b] += 1
            entity_tokens[batch_ids, within_idx] = encoder_output[batch_ids, seq_ids]
            entity_padding_mask[batch_ids, within_idx] = False

        return entity_tokens, entity_padding_mask, max_entity_tokens

    def forward(
        self,
        encoder_output: torch.Tensor,
        entity_position_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-attention from entity tokens to full context.

        Args:
            encoder_output: (batch, seq_len, hidden_size) from encoder
            entity_position_mask: (batch, seq_len) binary mask, 1 for entity tokens
            return_attention: Whether to return attention weights

        Returns:
            sentiment_score: (batch,) continuous values in [-0.95, 0.95]
            attention_weights: (batch, n_entity_tokens, seq_len) if return_attention
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Empty masks: return neutral sentiment
        has_entities = entity_position_mask.sum(dim=1) > 0
        if not has_entities.any():
            if return_attention:
                return torch.zeros(batch_size, device=device), None
            return torch.zeros(batch_size, device=device)

        # Extract entity token embeddings → queries
        entity_tokens, entity_pad_mask, n_ent = self._extract_entity_tokens(
            encoder_output, entity_position_mask
        )

        # Normalize queries and context separately
        queries = self.query_norm(entity_tokens)
        context = self.context_norm(encoder_output)

        # Cross-attention: entities attend to full document
        attn_output, attn_weights = self.cross_attention(
            query=queries,
            key=context,
            value=context,
            key_padding_mask=None,
            need_weights=return_attention,
        )

        # Residual + masked mean pool
        attn_output = attn_output + entity_tokens
        pool_mask = (~entity_pad_mask).float().unsqueeze(-1)
        pooled = (attn_output * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1e-9)

        # Post-attention layer norm
        pooled = self.post_attn_norm(pooled)

        # Regression MLP
        x = self.activation(self.fc1(pooled))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        score = self.fc3(x).squeeze(-1)

        # Scaled tanh: keep outputs in [-0.95, 0.95] to avoid gradient saturation
        sentiment_score = 0.95 * torch.tanh(score)

        if return_attention:
            return sentiment_score, attn_weights
        return sentiment_score

    def configure_loss(self, mode: str = "mse_pearson", **kw) -> None:
        """Set the sentiment loss recipe (used by the second-phase retrain ablation).

        mode="mse_pearson" (default): legacy 0.5*MSE + 0.5*(1-Pearson) — Arm A, and
            the recipe the v2.0 model was trained with. Leaves behavior unchanged.
        mode="ccc_huber": weighted-Huber + lambda*(1-CCC) + optional sign penalty —
            the new recipe (Arm B/C). CCC (concordance) is NOT scale-invariant, so it
            penalizes magnitude compression (unlike Pearson); the magnitude weight
            up-weights strong-sentiment entities; the sign penalty guards |y|>=thr.
        """
        self._loss_mode = mode
        self._huber_delta = kw.get("huber_delta", 0.1)
        self._mag_alpha = kw.get("mag_alpha", 3.0)      # weight = clip(1+alpha|y|, max)
        self._mag_wmax = kw.get("mag_wmax", 3.0)
        self._huber_weight = kw.get("huber_weight", 1.0)
        self._ccc_weight = kw.get("ccc_weight", 1.0)
        self._sign_weight = kw.get("sign_weight", 0.5)
        self._sign_thr = kw.get("sign_thr", 0.4)

    @staticmethod
    def _ccc(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concordance correlation coefficient (penalizes corr + mean shift + scale)."""
        vp, vy = pred - pred.mean(), y - y.mean()
        cov = (vp * vy).mean()
        return 2 * cov / (pred.var(unbiased=False) + y.var(unbiased=False)
                          + (pred.mean() - y.mean()) ** 2 + 1e-8)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Sentiment loss. Recipe selected via configure_loss(); defaults to the
        legacy MSE + (1 - Pearson) so existing callers/checkpoints are unchanged.
        """
        mode = getattr(self, "_loss_mode", "mse_pearson")

        if mode == "ccc_huber":
            # Magnitude-weighted Huber (robust to noisy LLM extremes; up-weights
            # strong-sentiment entities so the neutral mass can't drive mean-collapse).
            w = torch.clamp(1.0 + self._mag_alpha * labels.abs(), max=self._mag_wmax)
            hub = F.huber_loss(predictions, labels, delta=self._huber_delta, reduction="none")
            wh = (w * hub).sum() / w.sum().clamp_min(1e-8)
            loss = self._huber_weight * wh
            # CCC term — scale-sensitive, fixes the under-polarization Pearson misses.
            if predictions.shape[0] > 2:
                loss = loss + self._ccc_weight * (1.0 - self._ccc(predictions, labels))
            # Asymmetric sign penalty on strong-sentiment entities (wrong sign on a
            # lawsuit/probe is worse than a magnitude miss).
            if self._sign_weight > 0:
                strong = labels.abs() >= self._sign_thr
                if strong.any():
                    loss = loss + self._sign_weight * torch.relu(
                        -(predictions[strong] * labels[strong])).mean()
            return loss

        # --- legacy default: 0.5*MSE + 0.5*(1 - Pearson) ---
        mse = nn.MSELoss(reduction=reduction)(predictions, labels)
        if predictions.shape[0] > 2:
            vx = predictions - predictions.mean()
            vy = labels - labels.mean()
            denom = vx.norm() * vy.norm()
            if denom > 1e-8:
                pearson = (vx * vy).sum() / denom
                return 0.5 * mse + 0.5 * (1.0 - pearson)
        return mse


# Backward-compatible alias for code that still imports `SentimentHeadV2`.
# Kept temporarily so external scripts don't break; remove after Q3 2026.
SentimentHeadV2 = SentimentHead

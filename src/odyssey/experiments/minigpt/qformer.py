"""Educational Q-Former: learnable queries cross-attend to frozen vision features.

Used as a bridge between a CLIP vision encoder and a causal language model.
Each layer applies self-attention on queries, optional cross-attention to image
patches, then a feed-forward block — all with residual connections.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QFormerConfig:
    """Hyperparameters for the scratch Q-Former."""

    num_queries: int = 32
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 6
    encoder_hidden_size: int = 768
    ffn_dim: int = 3072
    cross_attention_freq: int = 1
    dropout: float = 0.0


class QFormerSelfAttention(nn.Module):
    """Multi-head self-attention over the learnable query tokens."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: (batch, num_queries, hidden_size)"""
        batch, seq_len, _ = queries.shape
        q = self.q_proj(queries).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(queries).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(queries).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)


class QFormerCrossAttention(nn.Module):
    """Queries attend to image patch features from the vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        encoder_hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(encoder_hidden_size, hidden_size)
        self.v_proj = nn.Linear(encoder_hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """queries: (batch, num_queries, hidden_size); image_embeds: (batch, num_patches, encoder_hidden_size)"""
        batch, num_queries, _ = queries.shape
        num_patches = image_embeds.shape[1]

        q = self.q_proj(queries).view(batch, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(image_embeds).view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(image_embeds).view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, num_queries, -1)
        return self.out_proj(out)


class QFormerFFN(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, hidden_size: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class QFormerLayer(nn.Module):
    """One Q-Former block: self-attn → cross-attn (optional) → FFN."""

    def __init__(self, cfg: QFormerConfig, *, use_cross_attention: bool):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.self_attn_norm = nn.LayerNorm(cfg.hidden_size)
        self.self_attn = QFormerSelfAttention(cfg.hidden_size, cfg.num_heads, cfg.dropout)

        if use_cross_attention:
            self.cross_attn_norm = nn.LayerNorm(cfg.hidden_size)
            self.cross_attn = QFormerCrossAttention(
                cfg.hidden_size,
                cfg.encoder_hidden_size,
                cfg.num_heads,
                cfg.dropout,
            )

        self.ffn_norm = nn.LayerNorm(cfg.hidden_size)
        self.ffn = QFormerFFN(cfg.hidden_size, cfg.ffn_dim, cfg.dropout)

    def forward(self, queries: torch.Tensor, image_embeds: torch.Tensor | None = None) -> torch.Tensor:
        residual = queries
        queries = self.self_attn_norm(queries)
        queries = self.self_attn(queries) + residual

        if self.use_cross_attention:
            if image_embeds is None:
                raise ValueError("image_embeds required when cross-attention is enabled")
            residual = queries
            queries = self.cross_attn_norm(queries)
            queries = self.cross_attn(queries, image_embeds) + residual

        residual = queries
        queries = self.ffn_norm(queries)
        queries = self.ffn(queries) + residual
        return queries


class QFormer(nn.Module):
    """Stack of Q-Former layers with learnable query embeddings."""

    def __init__(self, cfg: QFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.query_tokens = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList(
            [
                QFormerLayer(cfg, use_cross_attention=(i % cfg.cross_attention_freq == 0))
                for i in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

    @property
    def num_queries(self) -> int:
        return self.cfg.num_queries

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """image_embeds: (batch, num_patches, encoder_hidden_size) → (batch, num_queries, hidden_size)"""
        batch = image_embeds.shape[0]
        queries = self.query_tokens.expand(batch, -1, -1)
        for layer in self.layers:
            queries = layer(queries, image_embeds)
        return self.norm(queries)

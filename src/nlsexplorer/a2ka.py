"""Reusable Attention to Key Areas (A2KA) module."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Attention(nn.Module):
    """One basic attention unit used by A2KA."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.atten_Matrix = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        rate_matrix = self.relu(self.atten_Matrix(embedding))
        attention = F.softmax(rate_matrix, dim=1)
        pooled = (embedding * attention).sum(1) / math.sqrt(rate_matrix.size(1))
        return self.layer_norm(pooled), attention


class FeedForwardNorm(nn.Module):
    """Feed-forward residual normalization used by the generic A2KA module."""

    def __init__(
        self,
        hidden_dim: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, source: Tensor) -> Tensor:
        residual = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm(source + self.dropout2(residual))


class A2KA(nn.Module):
    """Configurable A2KA network.

    Parameters
    ----------
    hidden_dim:
        Final dimension of every input token embedding.
    config:
        Number of basic attention units in each layer. ``hidden_dim`` must be
        divisible by every value so that unit projections concatenate back to
        the original embedding dimension.
    """

    def __init__(self, hidden_dim: int, config: Sequence[int]) -> None:
        super().__init__()
        self.Att_config = tuple(int(value) for value in config)
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        if not self.Att_config:
            raise ValueError("config must contain at least one layer")
        if any(value <= 0 for value in self.Att_config):
            raise ValueError("every config value must be positive")
        if any(hidden_dim % value for value in self.Att_config):
            raise ValueError("hidden_dim must be divisible by every config value")

        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim
        self.Att_li = nn.ModuleList(
            [
                nn.ModuleList([Attention(hidden_dim) for _ in range(units)])
                for units in self.Att_config
            ]
        )
        self.AAt = Attention(hidden_dim)
        self.pro_li = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim // units) for _ in range(units)]
                )
                for units in self.Att_config
            ]
        )
        self.project_li = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in self.Att_config]
        )
        self.FI_li = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in self.Att_config]
        )
        set_dim = hidden_dim // 4
        self.Set_li = nn.ModuleList(
            [nn.Linear(hidden_dim, set_dim) for _ in self.Att_config]
        )
        self.hidden2p = nn.Linear(len(self.Att_config) * set_dim + hidden_dim, 1)
        self.norm_li = nn.ModuleList(
            [FeedForwardNorm(hidden_dim) for _ in self.Att_config]
        )

    def forward(self, embedding: Tensor) -> tuple[Tensor, list[Tensor]]:
        if embedding.ndim != 3:
            raise ValueError(
                "A2KA expects [batch, sequence_length, hidden_dim] input"
            )
        if embedding.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected hidden_dim={self.hidden_dim}, got {embedding.size(-1)}"
            )

        batch_size = embedding.size(0)
        token_embedding = embedding
        layer_vectors: list[Tensor] = []
        layer_attention: list[Tensor] = []

        for layer_index, units in enumerate(self.Att_config):
            vectors: list[Tensor] = []
            attentions: list[Tensor] = []
            for unit_index in range(units):
                vector, attention = self.Att_li[layer_index][unit_index](token_embedding)
                vectors.append(F.relu(self.pro_li[layer_index][unit_index](vector)))
                attentions.append(attention)

            stacked_attention = torch.stack(attentions).squeeze(3)
            summed_attention = stacked_attention.sum(0).unsqueeze(2)
            layer_attention.append(summed_attention)
            projected_attention = self.project_li[layer_index](summed_attention)
            token_embedding = token_embedding * projected_attention + token_embedding
            token_embedding = self.norm_li[layer_index](token_embedding)

            stacked_vectors = torch.stack(vectors).transpose(0, 1).permute(0, 2, 1)
            merged = stacked_vectors.reshape(batch_size, -1)
            merged = F.relu(self.FI_li[layer_index](merged)) + merged
            layer_vectors.append(F.relu(self.Set_li[layer_index](merged)))

        features = torch.stack(layer_vectors).transpose(0, 1).reshape(batch_size, -1)
        pooled, _ = self.AAt(token_embedding)
        probability = torch.sigmoid(
            self.hidden2p(self.dropout(torch.cat((pooled, features), dim=1)))
        )
        return probability, layer_attention

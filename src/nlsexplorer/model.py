"""The checkpoint-compatible NLSExplorer A2KA inference model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DEFAULT_HIDDEN_DIM = 1280
DEFAULT_ATTENTION_CONFIG = (64,) * 16


class Attention(nn.Module):
    """One basic attention unit used by the published NLSExplorer model.

    The training code declared an additional ``Embedding(500, hidden_dim)`` named
    ``ll``, but never used it in ``forward``. It accounted for 656 million of the
    model's 719 million parameters. The inference model intentionally omits it.
    ``checkpoint.is_legacy_unused_key`` removes only those unused weights before
    strict state-dict validation.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.atten_Matrix = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        rate_matrix = self.relu(self.atten_Matrix(embedding))
        attention_rate = F.softmax(rate_matrix, dim=1)
        sequence_length = rate_matrix.size(1)
        summary = (embedding * attention_rate).sum(1) / math.sqrt(sequence_length)
        return self.layer_norm(summary), attention_rate


class NLSExplorerModel(nn.Module):
    """A2KA binary classifier matching the published checkpoint's forward pass."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        attention_config: Sequence[int] = DEFAULT_ATTENTION_CONFIG,
    ) -> None:
        super().__init__()
        config = tuple(int(units) for units in attention_config)
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        if not config or any(units <= 0 for units in config):
            raise ValueError("attention_config must contain positive integers")
        if any(hidden_dim % units for units in config):
            raise ValueError(
                "hidden_dim must be divisible by every attention_config value"
            )

        # Keep the historical attribute names so the published state dict remains
        # directly compatible after removal of the unused *.ll.weight entries.
        self.Att_config = list(config)
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim

        self.Att_li = nn.ModuleList(
            [
                nn.ModuleList([Attention(hidden_dim) for _ in range(units)])
                for units in config
            ]
        )
        self.AAt = Attention(hidden_dim)
        self.pro_li = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(hidden_dim, hidden_dim // units)
                        for _ in range(units)
                    ]
                )
                for units in config
            ]
        )
        self.project_li = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in config]
        )
        self.FI_li = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in config]
        )

        set_dim = hidden_dim // 4
        self.Set_li = nn.ModuleList(
            [nn.Linear(hidden_dim, set_dim) for _ in config]
        )
        classifier_input_dim = len(config) * set_dim + hidden_dim
        self.hidden2p = nn.Linear(classifier_input_dim, 1)
        self.norm_li = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in config])

    def forward(
        self, embedding: Tensor
    ) -> tuple[Tensor, list[Tensor], list[list[Tensor]], Tensor]:
        if embedding.ndim != 3:
            raise ValueError(
                "embedding must have shape (batch, sequence_length, hidden_dim)"
            )
        if embedding.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected embedding dimension {self.hidden_dim}, "
                f"got {embedding.size(-1)}"
            )
        if embedding.size(1) == 0:
            raise ValueError("embedding sequence_length must be greater than zero")

        batch_size = embedding.size(0)
        layer_vectors: list[Tensor] = []
        layer_attention: list[Tensor] = []
        basic_unit_attention: list[list[Tensor]] = []
        transformed = embedding

        for layer_index, units in enumerate(self.Att_config):
            unit_vectors: list[Tensor] = []
            unit_attention: list[Tensor] = []
            for unit_index in range(units):
                vector, attention = self.Att_li[layer_index][unit_index](transformed)
                vector = F.relu(self.pro_li[layer_index][unit_index](vector))
                unit_vectors.append(vector)
                unit_attention.append(attention)

            basic_unit_attention.append(unit_attention)
            stacked_attention = torch.stack(unit_attention).squeeze(3)
            summed_attention = stacked_attention.sum(0).unsqueeze(2)
            layer_attention.append(summed_attention)
            projected_attention = self.project_li[layer_index](summed_attention)

            transformed = transformed * projected_attention + transformed
            transformed = self.norm_li[layer_index](transformed)

            stacked_vectors = torch.stack(unit_vectors).transpose(0, 1)
            flattened = stacked_vectors.permute(0, 2, 1).reshape(batch_size, -1)
            flattened = F.relu(self.FI_li[layer_index](flattened)) + flattened
            layer_vectors.append(F.relu(self.Set_li[layer_index](flattened)))

        selected_features = torch.stack(layer_vectors).transpose(0, 1)
        selected_features = selected_features.reshape(batch_size, -1)
        final_summary, final_attention = self.AAt(transformed)
        classifier_input = torch.cat((final_summary, selected_features), dim=1)
        probability = torch.sigmoid(self.hidden2p(self.dropout(classifier_input)))
        return probability, layer_attention, basic_unit_attention, final_attention


# The historical analysis scripts called the same architecture LSTMTagger even
# though it does not contain an LSTM. Preserve the import name for users migrating
# notebooks while exposing the descriptive NLSExplorerModel name as the primary API.
LSTMTagger = NLSExplorerModel

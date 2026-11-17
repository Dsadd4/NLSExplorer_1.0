"""Stable sequence-to-prediction API for NLSExplorer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from .checkpoint import load_model
from .embedding import ESM1bEmbedder
from .regions import (
    Region,
    aggregate_basic_unit_attention,
    aggregate_layer_attention,
    attention_to_regions,
    load_selective_bau_indices,
)


class EmbeddingProvider(Protocol):
    def embed(
        self, sequences: Sequence[str], *, batch_size: int = 1
    ) -> tuple[list[str], list[Tensor]]: ...


@dataclass(frozen=True)
class Prediction:
    identifier: str
    sequence: str
    nuclear_probability: float
    attention: tuple[float, ...]
    predicted_regions: tuple[Region, ...]

    def to_dict(self, *, one_based_regions: bool = True) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "sequence": self.sequence,
            "nuclear_probability": self.nuclear_probability,
            "predicted_regions": [
                region.to_dict(one_based=one_based_regions)
                for region in self.predicted_regions
            ],
            "attention": list(self.attention),
        }


class NLSExplorerPredictor:
    """Run ESM-1b embedding and the published NLSExplorer classifier."""

    def __init__(
        self,
        model: nn.Module,
        embedder: EmbeddingProvider,
        *,
        device: str | torch.device | None = None,
        selected_bau_indices: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.model = model.eval()
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.model.to(self.device)
        self.embedder = embedder
        self.selected_bau_indices = selected_bau_indices

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "auto",
        esm_checkpoint: str | Path | None = None,
    ) -> NLSExplorerPredictor:
        model = load_model(checkpoint, device=device)
        model_device = next(model.parameters()).device
        embedder = ESM1bEmbedder(device=model_device, model_path=esm_checkpoint)
        return cls(
            model,
            embedder,
            device=model_device,
            selected_bau_indices=load_selective_bau_indices(),
        )

    def predict(
        self,
        sequences: Sequence[str],
        *,
        identifiers: Sequence[str] | None = None,
        batch_size: int = 1,
        attention_fraction: float = 0.10,
        merge_gap: int = 2,
        min_region_length: int = 4,
    ) -> list[Prediction]:
        if identifiers is None:
            identifiers = [f"sequence_{index + 1}" for index in range(len(sequences))]
        if len(identifiers) != len(sequences):
            raise ValueError("identifiers and sequences must have the same length")

        normalized, embeddings = self.embedder.embed(sequences, batch_size=batch_size)
        results: list[Prediction] = []
        with torch.inference_mode():
            for identifier, sequence, embedding in zip(
                identifiers, normalized, embeddings, strict=True
            ):
                output = self.model(embedding.unsqueeze(0).to(self.device))
                probability, layer_attention, basic_unit_attention = output[:3]
                if self.selected_bau_indices is None:
                    raw_attention = aggregate_layer_attention(layer_attention)[0]
                else:
                    raw_attention = aggregate_basic_unit_attention(
                        basic_unit_attention,
                        self.selected_bau_indices,
                    )[0]
                normalized_attention = torch.softmax(raw_attention, dim=0)
                regions = attention_to_regions(
                    raw_attention,
                    fraction=attention_fraction,
                    merge_gap=merge_gap,
                    min_length=min_region_length,
                )
                results.append(
                    Prediction(
                        identifier=str(identifier),
                        sequence=sequence,
                        nuclear_probability=float(probability[0, 0].cpu()),
                        attention=tuple(float(value) for value in normalized_attention.cpu()),
                        predicted_regions=tuple(regions),
                    )
                )
        return results

"""Convert NLSExplorer attention profiles into deterministic candidate regions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class Region:
    """A zero-based, half-open candidate sequence region."""

    start: int
    end: int
    score: float

    def to_dict(self, *, one_based: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if one_based:
            result["start"] = self.start + 1
            result["end"] = self.end
        return result


def aggregate_layer_attention(layer_attention: list[Tensor]) -> Tensor:
    """Sum the per-layer BAU attention returned by ``NLSExplorerModel``."""

    if not layer_attention:
        raise ValueError("layer_attention must contain at least one tensor")
    return torch.stack(layer_attention).sum(0).squeeze(-1)


def load_selective_bau_indices() -> tuple[tuple[int, ...], ...]:
    """Load the selective BAU profile used by the published analysis script."""

    resource = files("nlsexplorer").joinpath("data/selective_bau_mask.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return tuple(tuple(int(index) for index in layer) for layer in payload["selected_unit_indices"])


def aggregate_basic_unit_attention(
    basic_unit_attention: list[list[Tensor]],
    selected_indices: tuple[tuple[int, ...], ...],
) -> Tensor:
    """Aggregate only the BAUs selected by the published 80% profile."""

    if len(basic_unit_attention) != len(selected_indices):
        raise ValueError(
            "selective BAU profile does not match the model's number of layers"
        )
    selected: list[Tensor] = []
    for layer_index, unit_indices in enumerate(selected_indices):
        units = basic_unit_attention[layer_index]
        for unit_index in unit_indices:
            if unit_index >= len(units):
                raise ValueError(
                    "selective BAU profile references a unit outside the model configuration"
                )
            selected.append(units[unit_index])
    if not selected:
        raise ValueError("selective BAU profile contains no selected units")
    return torch.stack(selected).sum(0).squeeze(-1)


def _expand_region(start: int, end: int, length: int, minimum: int) -> tuple[int, int]:
    while end - start < minimum and (start > 0 or end < length):
        if start > 0:
            start -= 1
        if end - start < minimum and end < length:
            end += 1
    return start, end


def attention_to_regions(
    attention: Tensor,
    *,
    fraction: float = 0.10,
    merge_gap: int = 2,
    min_length: int = 4,
) -> list[Region]:
    """Select high-attention positions and group them into candidate regions.

    This keeps the position ranking and gap-merging behavior used in the
    original research script, but removes its random region extension. The
    output is reproducible and uses zero-based, half-open coordinates.
    """

    profile = torch.as_tensor(attention, dtype=torch.float32).flatten().cpu()
    if not profile.numel():
        return []
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    if min_length <= 0:
        raise ValueError("min_length must be positive")

    probabilities = torch.softmax(profile, dim=0)
    selected_count = min(profile.numel(), max(1, int(fraction * profile.numel())))
    positions = torch.topk(probabilities, selected_count).indices.sort().values.tolist()

    groups: list[tuple[int, int]] = []
    start = positions[0]
    previous = positions[0]
    for position in positions[1:]:
        if position - previous > merge_gap:
            groups.append((start, previous + 1))
            start = position
        previous = position
    groups.append((start, previous + 1))

    regions: list[Region] = []
    for start, end in groups:
        start, end = _expand_region(start, end, profile.numel(), min_length)
        regions.append(
            Region(start=start, end=end, score=float(probabilities[start:end].sum()))
        )
    return regions

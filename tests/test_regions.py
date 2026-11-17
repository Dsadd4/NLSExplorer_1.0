import pytest
import torch

from nlsexplorer.regions import (
    aggregate_basic_unit_attention,
    aggregate_layer_attention,
    attention_to_regions,
    load_selective_bau_indices,
)


def test_aggregate_layer_attention():
    layers = [torch.ones(2, 4, 1), torch.full((2, 4, 1), 2.0)]
    aggregate = aggregate_layer_attention(layers)
    assert aggregate.shape == (2, 4)
    assert torch.equal(aggregate, torch.full((2, 4), 3.0))


def test_published_selective_bau_profile_and_aggregation():
    indices = load_selective_bau_indices()
    assert len(indices) == 16
    assert sum(len(layer) for layer in indices) == 339

    units = [[torch.full((1, 3, 1), float(index + 1)) for index in range(3)]]
    aggregate = aggregate_basic_unit_attention(units, ((0, 2),))
    assert torch.equal(aggregate, torch.full((1, 3), 4.0))


def test_attention_to_regions_is_deterministic_and_half_open():
    attention = torch.tensor([0.0, 8.0, 7.0, 0.0, 0.0, 0.0, 9.0, 0.0])
    regions = attention_to_regions(
        attention,
        fraction=0.375,
        merge_gap=2,
        min_length=2,
    )
    assert [(region.start, region.end) for region in regions] == [(1, 3), (5, 7)]
    assert all(region.score > 0 for region in regions)
    assert regions[0].to_dict(one_based=True)["start"] == 2
    assert regions[0].to_dict(one_based=True)["end"] == 3


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_attention_to_regions_validates_fraction(fraction):
    with pytest.raises(ValueError):
        attention_to_regions(torch.ones(4), fraction=fraction)

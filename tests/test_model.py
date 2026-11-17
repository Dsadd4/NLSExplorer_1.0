from __future__ import annotations

import pytest
import torch

from nlsexplorer.device import DeviceUnavailableError, resolve_device
from nlsexplorer.model import NLSExplorerModel


def test_small_model_forward_shapes_and_probability_range() -> None:
    torch.manual_seed(7)
    model = NLSExplorerModel(hidden_dim=8, attention_config=(2, 2)).eval()
    embedding = torch.randn(3, 5, 8)

    probability, layer_attention, unit_attention, final_attention = model(embedding)

    assert probability.shape == (3, 1)
    assert torch.all((probability >= 0) & (probability <= 1))
    assert len(layer_attention) == 2
    assert all(item.shape == (3, 5, 1) for item in layer_attention)
    assert len(unit_attention) == 2
    assert all(len(layer) == 2 for layer in unit_attention)
    assert final_attention.shape == (3, 5, 1)


def test_model_uses_published_state_dict_names_without_unused_ll() -> None:
    model = NLSExplorerModel(hidden_dim=8, attention_config=(2,))
    keys = set(model.state_dict())

    assert "Att_li.0.0.atten_Matrix.weight" in keys
    assert "Att_li.0.0.layer_norm.weight" in keys
    assert "AAt.atten_Matrix.weight" in keys
    assert "hidden2p.weight" in keys
    assert not any(key.endswith(".ll.weight") for key in keys)


@pytest.mark.parametrize(
    ("hidden_dim", "config"),
    [(0, (2,)), (8, ()), (8, (0,)), (7, (2,))],
)
def test_model_rejects_invalid_architecture(
    hidden_dim: int, config: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        NLSExplorerModel(hidden_dim=hidden_dim, attention_config=config)


def test_model_rejects_invalid_embedding_shape() -> None:
    model = NLSExplorerModel(hidden_dim=8, attention_config=(2,))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(5, 8))
    with pytest.raises(ValueError, match="expected embedding dimension"):
        model(torch.randn(1, 5, 7))
    with pytest.raises(ValueError, match="greater than zero"):
        model(torch.randn(1, 0, 8))


def test_resolve_device_cpu_and_unavailable_cuda() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        with pytest.raises(DeviceUnavailableError, match="CUDA was requested"):
            resolve_device("cuda")

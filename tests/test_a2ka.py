import pytest
import torch

from nlsexplorer import A2KA


def test_a2ka_cpu_forward_shapes():
    torch.manual_seed(1)
    model = A2KA(hidden_dim=8, config=[2, 4]).eval()
    embedding = torch.randn(3, 5, 8)

    with torch.inference_mode():
        probability, layer_attention = model(embedding)

    assert probability.shape == (3, 1)
    assert len(layer_attention) == 2
    assert all(attention.shape == (3, 5, 1) for attention in layer_attention)
    assert torch.all((probability >= 0) & (probability <= 1))
    assert not any(key.endswith(".ll.weight") for key in model.state_dict())


@pytest.mark.parametrize(
    ("hidden_dim", "config"),
    [(0, [1]), (1, [1]), (8, []), (8, [0]), (8, [3])],
)
def test_a2ka_rejects_invalid_configuration(hidden_dim, config):
    with pytest.raises(ValueError):
        A2KA(hidden_dim=hidden_dim, config=config)


def test_a2ka_rejects_wrong_input_shape():
    model = A2KA(hidden_dim=8, config=[2])
    with pytest.raises(ValueError, match="batch, sequence_length, hidden_dim"):
        model(torch.randn(5, 8))
    with pytest.raises(ValueError, match="expected hidden_dim=8"):
        model(torch.randn(1, 5, 7))

import torch

from nlsexplorer.model import NLSExplorerModel
from nlsexplorer.predictor import NLSExplorerPredictor


class FakeEmbedder:
    def embed(self, sequences, *, batch_size=1):
        torch.manual_seed(2)
        return list(sequences), [torch.randn(len(sequence), 8) for sequence in sequences]


def test_predictor_runs_without_esm_or_large_checkpoint():
    torch.manual_seed(1)
    model = NLSExplorerModel(hidden_dim=8, attention_config=(2,)).eval()
    predictor = NLSExplorerPredictor(model, FakeEmbedder(), device="cpu")

    results = predictor.predict(
        ["MSSAKRR", "AAAA"],
        identifiers=["p1", "p2"],
        attention_fraction=0.25,
    )

    assert [result.identifier for result in results] == ["p1", "p2"]
    assert all(0 <= result.nuclear_probability <= 1 for result in results)
    assert [len(result.attention) for result in results] == [7, 4]
    assert all(result.predicted_regions for result in results)
    assert results[0].to_dict()["predicted_regions"][0]["start"] >= 1

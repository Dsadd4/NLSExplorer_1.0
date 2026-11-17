import pytest
import torch

from nlsexplorer.embedding import MAX_SEQUENCE_LENGTH, ESM1bEmbedder, normalize_sequence


class FakeAlphabet:
    def get_batch_converter(self):
        def convert(data):
            maximum = max(len(sequence) for _, sequence in data)
            tokens = torch.zeros((len(data), maximum + 2), dtype=torch.long)
            return [label for label, _ in data], [sequence for _, sequence in data], tokens

        return convert


class FakeESM(torch.nn.Module):
    def forward(self, tokens, repr_layers, return_contacts):
        batch, length = tokens.shape
        representation = torch.arange(batch * length * 8, dtype=torch.float32).reshape(
            batch, length, 8
        )
        return {"representations": {repr_layers[0]: representation.to(tokens.device)}}


def test_normalize_sequence():
    assert normalize_sequence(" ms sa krr\n") == "MSSAKRR"
    with pytest.raises(ValueError, match="empty"):
        normalize_sequence("  \n")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_sequence("ABC-")


def test_embedder_uses_injected_model_without_network():
    embedder = ESM1bEmbedder(device="cpu")
    embedder._model = FakeESM().eval()
    embedder._alphabet = FakeAlphabet()

    sequences, embeddings = embedder.embed(["MSSAK", "KRR"], batch_size=2)

    assert sequences == ["MSSAK", "KRR"]
    assert [tuple(embedding.shape) for embedding in embeddings] == [(5, 8), (3, 8)]
    assert all(embedding.device.type == "cpu" for embedding in embeddings)


def test_embedder_truncates_to_esm_limit():
    embedder = ESM1bEmbedder(device="cpu")
    embedder._model = FakeESM().eval()
    embedder._alphabet = FakeAlphabet()

    with pytest.warns(UserWarning, match="ESM-1b limit"):
        sequences, embeddings = embedder.embed(["A" * (MAX_SEQUENCE_LENGTH + 1)])

    assert len(sequences[0]) == MAX_SEQUENCE_LENGTH
    assert embeddings[0].shape[0] == MAX_SEQUENCE_LENGTH

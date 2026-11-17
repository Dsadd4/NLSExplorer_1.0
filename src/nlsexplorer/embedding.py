"""Protein-language-model embeddings used by NLSExplorer inference."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from .device import resolve_device

ESM1B_REPRESENTATION_LAYER = 33
MAX_SEQUENCE_LENGTH = 1022
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
ESM1B_FILENAME = "esm1b_t33_650M_UR50S.pt"
ESM1B_REGRESSION_FILENAME = "esm1b_t33_650M_UR50S-contact-regression.pt"
ESM1B_URL = (
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt"
)
ESM1B_REGRESSION_URL = (
    "https://dl.fbaipublicfiles.com/fair-esm/regression/"
    "esm1b_t33_650M_UR50S-contact-regression.pt"
)


def normalize_sequence(sequence: str) -> str:
    """Normalize whitespace/case and validate a protein sequence."""

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("protein sequence is empty")
    invalid = sorted(set(normalized) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(
            "protein sequence contains unsupported characters: " + ", ".join(invalid)
        )
    return normalized


class ESM1bEmbedder:
    """Lazy ESM-1b residue embedding provider.

    ``fair-esm`` and the 650M-parameter ESM-1b model are loaded only when
    :meth:`embed` is first called. Pass ``model_path`` to run fully offline.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "auto",
        model_path: str | Path | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model_path = Path(model_path).expanduser() if model_path else None
        self._model = None
        self._alphabet = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            esm = importlib.import_module("esm")
        except ImportError as error:
            raise RuntimeError(
                "ESM-1b is required for sequence inference. Install it with "
                "`python -m pip install -e '.[inference]'`."
            ) from error

        if self.model_path is None:
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        else:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ESM checkpoint not found: {self.model_path}")
            # fair-esm 2.0 predates PyTorch 2.6's weights_only default change.
            # Official ESM checkpoints contain an argparse.Namespace, so an
            # explicitly trusted local ESM checkpoint must use the full loader.
            model_data = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )
            regression_path = self.model_path.with_suffix("").with_name(
                f"{self.model_path.stem}-contact-regression.pt"
            )
            regression_data = (
                torch.load(
                    regression_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if regression_path.is_file()
                else None
            )
            model, alphabet = esm.pretrained.load_model_and_alphabet_core(
                self.model_path.stem,
                model_data,
                regression_data,
            )
        self._model = model.eval().to(self.device)
        self._alphabet = alphabet

    def embed(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 1,
    ) -> tuple[list[str], list[Tensor]]:
        """Return normalized sequences and per-residue ESM-1b embeddings."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        normalized: list[str] = []
        for sequence in sequences:
            cleaned = normalize_sequence(sequence)
            if len(cleaned) > MAX_SEQUENCE_LENGTH:
                warnings.warn(
                    f"sequence length {len(cleaned)} exceeds the ESM-1b limit; "
                    f"using the first {MAX_SEQUENCE_LENGTH} residues",
                    stacklevel=2,
                )
                cleaned = cleaned[:MAX_SEQUENCE_LENGTH]
            normalized.append(cleaned)
        if not normalized:
            raise ValueError("at least one protein sequence is required")

        self._load()
        assert self._model is not None
        assert self._alphabet is not None
        converter = self._alphabet.get_batch_converter()
        embeddings: list[Tensor] = []
        with torch.inference_mode():
            for offset in range(0, len(normalized), batch_size):
                batch = normalized[offset : offset + batch_size]
                data = [(str(offset + index), sequence) for index, sequence in enumerate(batch)]
                _, _, tokens = converter(data)
                output = self._model(
                    tokens.to(self.device),
                    repr_layers=[ESM1B_REPRESENTATION_LAYER],
                    return_contacts=False,
                )
                token_embeddings = output["representations"][ESM1B_REPRESENTATION_LAYER]
                for index, sequence in enumerate(batch):
                    embeddings.append(token_embeddings[index, 1 : len(sequence) + 1])
        return normalized, embeddings

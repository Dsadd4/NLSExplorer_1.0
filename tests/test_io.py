import json
from pathlib import Path

import pytest

from nlsexplorer.io import read_csv_sequences, read_fasta, write_predictions


def test_read_fasta_and_csv(tmp_path: Path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">p1 first\nMSSA\nKRR\n>p2\nAAAA\n")
    assert [(record.identifier, record.sequence) for record in read_fasta(fasta)] == [
        ("p1", "MSSAKRR"),
        ("p2", "AAAA"),
    ]

    csv_file = tmp_path / "input.csv"
    csv_file.write_text("id,sequence\np1,MSSAKRR\np2,AAAA\n")
    assert [(record.identifier, record.sequence) for record in read_csv_sequences(csv_file)] == [
        ("p1", "MSSAKRR"),
        ("p2", "AAAA"),
    ]


def test_csv_requires_sequence_column(tmp_path: Path):
    path = tmp_path / "input.csv"
    path.write_text("id,protein\np1,MSSA\n")
    with pytest.raises(ValueError, match="sequence"):
        read_csv_sequences(path)


def test_write_json_predictions(tmp_path: Path):
    path = tmp_path / "predictions.json"
    payload = [
        {
            "id": "p1",
            "sequence": "MSSA",
            "nuclear_probability": 0.5,
            "predicted_regions": [],
            "attention": [],
        }
    ]
    write_predictions(payload, output=path, output_format="json")
    assert json.loads(path.read_text()) == payload

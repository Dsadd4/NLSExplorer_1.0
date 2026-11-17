from pathlib import Path

import torch

from nlsexplorer.cli import main


def test_check_model_cli(tmp_path: Path, capsys):
    checkpoint = tmp_path / "model.pt"
    torch.save({"weight": torch.ones(1)}, checkpoint)

    assert main(["check-model", str(checkpoint)]) == 0
    output = capsys.readouterr().out
    assert "kind: intact" in output


def test_check_model_cli_reports_extracted_checkpoint(tmp_path: Path, capsys):
    extracted = tmp_path / "model"
    (extracted / "data").mkdir(parents=True)
    (extracted / "data.pkl").write_bytes(b"metadata")
    (extracted / "data" / "0").write_bytes(b"storage")
    (extracted / "version").write_text("3")

    assert main(["check-model", str(extracted)]) == 2
    output = capsys.readouterr().out
    assert "kind: extracted" in output

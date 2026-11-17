from __future__ import annotations

import json

import pytest

from nlsexplorer.scnls import SCNLS, main, read_sequences_csv


def test_readme_style_api_returns_structured_patterns() -> None:
    result = SCNLS(["KKKKRRR"], 3, 3, 1, top_k=5)

    assert result.sequence_count == 1
    assert result.workers_used == 1
    assert list(result.gapped_patterns.items()) == [
        ("K**R", 3),
        ("K***R", 3),
        ("KK**R", 3),
        ("K*K", 2),
        ("K*R", 2),
    ]
    assert list(result.continuous_patterns.items()) == [
        ("K", 4),
        ("KK", 3),
        ("R", 3),
        ("KKK", 2),
        ("RR", 2),
    ]


def test_top_k_is_exact() -> None:
    result = SCNLS(["KKKKRRR"], 3, 3, 1, top_k=3)

    assert len(result.gapped_patterns) == 3
    assert len(result.continuous_patterns) == 3


def test_workers_are_capped_to_sequence_count() -> None:
    result = SCNLS(["KRR"], 3, 3, 20)

    assert result.workers_used == 1


@pytest.mark.parametrize(
    ("sequences", "k", "max_gap", "workers", "message"),
    [
        ([], 3, 3, 1, "at least one"),
        ([""], 3, 3, 1, "sequence 1 is empty"),
        (["KRR"], 0, 3, 1, "k must be"),
        (["KRR"], 3, 0, 1, "max_gap must be"),
        (["KRR"], 3, 3, 0, "workers must be"),
    ],
)
def test_invalid_api_inputs_are_rejected(
    sequences: list[str],
    k: int,
    max_gap: int,
    workers: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SCNLS(sequences, k, max_gap, workers)


def test_csv_reader_reports_missing_column(tmp_path) -> None:
    csv_path = tmp_path / "sequences.csv"
    csv_path.write_text("id,sequence\n1,KRR\n", encoding="utf-8")

    with pytest.raises(ValueError, match="available columns: id, sequence"):
        read_sequences_csv(csv_path, "Recommended Segment")


def test_sequence_cli_prints_json(capsys) -> None:
    exit_code = main(
        [
            "--sequence",
            "KKKKRRR",
            "--k",
            "3",
            "--max-gap",
            "3",
            "--workers",
            "8",
            "--top-k",
            "2",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workers_used"] == 1
    assert len(payload["gapped_patterns"]) == 2
    assert len(payload["continuous_patterns"]) == 2


def test_csv_cli_uses_named_column(tmp_path, capsys) -> None:
    csv_path = tmp_path / "sequences.csv"
    csv_path.write_text(
        "id,protein\nfirst,KKKKRRR\nsecond,KRR\n", encoding="utf-8"
    )

    assert (
        main(
            [
                "--csv",
                str(csv_path),
                "--sequence-column",
                "protein",
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Gapped patterns" in output
    assert "Continuous patterns" in output


def test_legacy_readme_cli_is_supported(capsys) -> None:
    assert (
        main(
            [
                "--mode",
                "s",
                "--material",
                "KKKKRRR",
                "--maxgap",
                "3",
                "--kths",
                "3",
                "--processor",
                "1",
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    assert "K**R\t3" in capsys.readouterr().out


def test_cli_reports_missing_csv_without_traceback(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(SystemExit, match="2"):
        main(["--csv", str(missing)])

    error = capsys.readouterr().err
    assert "CSV file does not exist" in error
    assert "Traceback" not in error


def test_cli_rejects_nonpositive_workers(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--sequence", "KRR", "--workers", "0"])

    assert "must be a positive integer" in capsys.readouterr().err

"""Sequence input and prediction output helpers."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class SequenceRecord:
    identifier: str
    sequence: str


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    """Read one or more FASTA records."""

    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    records: list[SequenceRecord] = []
    identifier: str | None = None
    sequence_lines: list[str] = []
    with fasta_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    if not sequence_lines:
                        raise ValueError(f"FASTA record {identifier!r} has no sequence")
                    records.append(SequenceRecord(identifier, "".join(sequence_lines)))
                identifier = line[1:].strip().split(maxsplit=1)[0]
                if not identifier:
                    raise ValueError(f"FASTA header on line {line_number} is empty")
                sequence_lines = []
            elif identifier is None:
                raise ValueError(
                    f"FASTA sequence appears before the first header on line {line_number}"
                )
            else:
                sequence_lines.append(line)
    if identifier is not None:
        if not sequence_lines:
            raise ValueError(f"FASTA record {identifier!r} has no sequence")
        records.append(SequenceRecord(identifier, "".join(sequence_lines)))
    if not records:
        raise ValueError(f"FASTA file contains no records: {fasta_path}")
    return records


def read_csv_sequences(
    path: str | Path,
    *,
    sequence_column: str = "sequence",
    id_column: str | None = "id",
) -> list[SequenceRecord]:
    """Read protein sequences from CSV without requiring pandas."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain a {sequence_column!r} column; found {reader.fieldnames}"
            )
        records: list[SequenceRecord] = []
        for row_number, row in enumerate(reader, start=2):
            sequence = (row.get(sequence_column) or "").strip()
            if not sequence:
                raise ValueError(f"CSV row {row_number} has an empty sequence")
            identifier = (
                (row.get(id_column) or "").strip()
                if id_column and id_column in row
                else ""
            )
            records.append(SequenceRecord(identifier or f"sequence_{row_number - 1}", sequence))
    if not records:
        raise ValueError(f"CSV file contains no sequence rows: {csv_path}")
    return records


def write_predictions(
    predictions: list[dict[str, Any]],
    *,
    output: str | Path | None = None,
    output_format: str = "json",
) -> None:
    """Write prediction records as JSON or TSV."""

    handle: TextIO
    should_close = False
    if output is None:
        handle = sys.stdout
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", encoding="utf-8", newline="")
        should_close = True
    try:
        if output_format == "json":
            json.dump(predictions, handle, indent=2)
            handle.write("\n")
        elif output_format == "tsv":
            fieldnames = [
                "id",
                "sequence",
                "nuclear_probability",
                "predicted_regions",
                "attention",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for prediction in predictions:
                row = dict(prediction)
                row["predicted_regions"] = json.dumps(row["predicted_regions"])
                row["attention"] = json.dumps(row["attention"])
                writer.writerow(row)
        else:
            raise ValueError("output_format must be 'json' or 'tsv'")
    finally:
        if should_close:
            handle.close()

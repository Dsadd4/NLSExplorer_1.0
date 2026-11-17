"""Search and collect continuous and discontinuous sequence patterns.

This module contains the dependency-free SCNLS implementation. The public
``SCNLS`` function is compatible with the original README example while
returning structured data instead of printing from library code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class SCNLSResult:
    """Structured output from an SCNLS run."""

    gapped_patterns: dict[str, int]
    continuous_patterns: dict[str, int]
    sequence_count: int
    workers_used: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _normalize_sequences(sequences: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for index, sequence in enumerate(sequences, start=1):
        if not isinstance(sequence, str):
            raise TypeError(f"sequence {index} must be a string")
        sequence = sequence.strip()
        if not sequence:
            raise ValueError(f"sequence {index} is empty")
        normalized.append(sequence)
    if not normalized:
        raise ValueError("at least one non-empty sequence is required")
    return normalized


def generate_combinations(length: int, k: int) -> list[tuple[int, ...]]:
    """Generate SCNLS segment/gap length combinations.

    ``length`` is the number of alternating retained/gap components and each
    component length ranges from 1 through ``k`` (inclusive).
    """

    if length <= 0 or k <= 0:
        raise ValueError("length and k must be positive integers")
    return list(product(range(1, k + 1), repeat=length))


def calculate_entropy_with_expectation(
    sequence: str,
    unknown_symbol: str = "*",
    amino_acids: str = AMINO_ACIDS,
) -> float:
    """Calculate expected Shannon entropy for a pattern containing gaps."""

    if not sequence:
        raise ValueError("sequence must not be empty")
    if not unknown_symbol:
        raise ValueError("unknown_symbol must not be empty")
    if not amino_acids:
        raise ValueError("amino_acids must not be empty")

    known_counts = Counter(char for char in sequence if char != unknown_symbol)
    expected_unknown_count = sequence.count(unknown_symbol) / len(amino_acids)
    total_length = len(sequence)
    entropy = 0.0
    for amino_acid in amino_acids:
        frequency = known_counts.get(amino_acid, 0) + expected_unknown_count
        probability = frequency / total_length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def _count_gapped_patterns(
    sequence: str,
    combinations: Sequence[tuple[int, ...]],
    entropy_threshold: float,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for combination in combinations:
        span = sum(combination)
        for start in range(len(sequence)):
            if start + span > len(sequence):
                continue
            current = start
            parts: list[str] = []
            for index, component_length in enumerate(combination):
                if index % 2 == 0:
                    parts.append(sequence[current : current + component_length])
                else:
                    parts.append("*" * component_length)
                current += component_length
            pattern = "".join(parts)
            if calculate_entropy_with_expectation(pattern) > entropy_threshold:
                counts[pattern] += 1
    return counts


def _count_chunk(
    arguments: tuple[list[str], list[tuple[int, ...]], float],
) -> Counter[str]:
    sequences, combinations, entropy_threshold = arguments
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(
            _count_gapped_patterns(sequence, combinations, entropy_threshold)
        )
    return counts


def _split_evenly(sequences: list[str], workers: int) -> list[list[str]]:
    chunk_size = math.ceil(len(sequences) / workers)
    return [
        sequences[start : start + chunk_size]
        for start in range(0, len(sequences), chunk_size)
    ]


def mine_gapped_patterns(
    sequences: Iterable[str],
    k: int,
    max_gap: int,
    workers: int = 1,
    entropy_threshold: float = 0.0,
) -> tuple[dict[str, int], int]:
    """Mine discontinuous patterns and return counts plus workers used."""

    normalized = _normalize_sequences(sequences)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if max_gap <= 0:
        raise ValueError("max_gap must be a positive integer")
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    if entropy_threshold < 0:
        raise ValueError("entropy_threshold must be zero or greater")

    workers_used = min(workers, len(normalized))
    combinations = generate_combinations(max_gap, k)
    chunks = _split_evenly(normalized, workers_used)
    arguments = [
        (chunk, combinations, entropy_threshold)
        for chunk in chunks
    ]

    if workers_used == 1:
        chunk_counts = [_count_chunk(arguments[0])]
    else:
        with multiprocessing.Pool(processes=workers_used) as pool:
            chunk_counts = pool.map(_count_chunk, arguments)

    combined: Counter[str] = Counter()
    for counts in chunk_counts:
        combined.update(counts)

    # The original algorithm strips boundary gaps and retains the highest
    # count when several raw patterns map to the same visible pattern.
    collapsed: dict[str, int] = {}
    for pattern, count in sorted(
        combined.items(), key=lambda item: item[1], reverse=True
    ):
        visible_pattern = pattern.strip("*")
        if visible_pattern and visible_pattern not in collapsed:
            collapsed[visible_pattern] = count
    return collapsed, workers_used


def get_continuous_patterns(sequence: str) -> dict[str, int]:
    """Count all non-empty continuous substrings in one sequence."""

    if not sequence:
        raise ValueError("sequence must not be empty")
    counts: Counter[str] = Counter()
    for start in range(len(sequence)):
        for end in range(start + 1, len(sequence) + 1):
            counts[sequence[start:end]] += 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def mine_continuous_patterns(sequences: Iterable[str]) -> dict[str, int]:
    """Count continuous substrings across multiple sequences."""

    normalized = _normalize_sequences(sequences)
    combined: Counter[str] = Counter()
    for sequence in normalized:
        combined.update(get_continuous_patterns(sequence))
    return dict(sorted(combined.items(), key=lambda item: item[1], reverse=True))


def _take_top(patterns: dict[str, int], top_k: int) -> dict[str, int]:
    return dict(list(patterns.items())[:top_k])


def SCNLS(
    sequences: Iterable[str],
    k: int,
    max_gap: int,
    workers: int = 1,
    *,
    entropy_threshold: float = 0.0,
    top_k: int = 10,
) -> SCNLSResult:
    """Run SCNLS and return the top discontinuous and continuous patterns.

    Parameters are positional-compatible with the historical README example::

        result = SCNLS(sequences, k, max_gap, workers)
    """

    normalized = _normalize_sequences(sequences)
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    gapped, workers_used = mine_gapped_patterns(
        normalized,
        k=k,
        max_gap=max_gap,
        workers=workers,
        entropy_threshold=entropy_threshold,
    )
    continuous = mine_continuous_patterns(normalized)
    return SCNLSResult(
        gapped_patterns=_take_top(gapped, top_k),
        continuous_patterns=_take_top(continuous, top_k),
        sequence_count=len(normalized),
        workers_used=workers_used,
    )


def read_sequences_csv(path: str | Path, column: str) -> list[str]:
    """Read sequences from a named CSV column."""

    csv_path = Path(path)
    try:
        handle = csv_path.open(newline="", encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise ValueError(f"CSV file does not exist: {csv_path}") from error
    except OSError as error:
        raise ValueError(f"cannot open CSV file {csv_path}: {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            available = ", ".join(fieldnames) if fieldnames else "none"
            raise ValueError(
                f"CSV column {column!r} was not found; available columns: {available}"
            )
        sequences = [
            (row.get(column) or "").strip()
            for row in reader
            if (row.get(column) or "").strip()
        ]
    if not sequences:
        raise ValueError(f"CSV column {column!r} contains no non-empty sequences")
    return sequences


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine continuous and discontinuous patterns with SCNLS."
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--sequence",
        action="append",
        help="Sequence to analyze; repeat the option for multiple sequences.",
    )
    inputs.add_argument("--csv", type=Path, help="CSV file containing sequences.")
    parser.add_argument(
        "--sequence-column",
        default="Recommended Segment",
        help="CSV column containing sequences (default: %(default)s).",
    )

    # Historical CLI compatibility: --mode s/f --material VALUE.
    parser.add_argument("--mode", choices=("s", "f"), help=argparse.SUPPRESS)
    parser.add_argument("--material", help=argparse.SUPPRESS)

    parser.add_argument(
        "--k",
        "--kths",
        dest="k",
        type=_positive_int,
        default=3,
        help="Maximum retained/gap component length (default: %(default)s).",
    )
    parser.add_argument(
        "--max-gap",
        "--maxgap",
        dest="max_gap",
        type=_positive_int,
        default=3,
        help="Number of alternating retained/gap components (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        "--processor",
        dest="workers",
        type=_positive_int,
        default=1,
        help="Worker processes; capped at the sequence count (default: %(default)s).",
    )
    parser.add_argument(
        "--entropy-threshold",
        "--entropythreshold",
        dest="entropy_threshold",
        type=_nonnegative_float,
        default=0.0,
        help="Minimum expected Shannon entropy (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=10,
        help="Number of patterns to report per category (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s).",
    )
    parser.add_argument("--output", type=Path, help="Write output to this file.")
    return parser


def _resolve_cli_sequences(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[str]:
    modern_input_selected = args.sequence is not None or args.csv is not None
    legacy_input_selected = args.mode is not None or args.material is not None
    if modern_input_selected and legacy_input_selected:
        parser.error("use either --sequence/--csv or legacy --mode/--material, not both")

    if args.sequence is not None:
        return _normalize_sequences(args.sequence)
    if args.csv is not None:
        try:
            return read_sequences_csv(args.csv, args.sequence_column)
        except ValueError as error:
            parser.error(str(error))

    if args.mode is None and args.material is None:
        parser.error("one of --sequence, --csv, or --mode with --material is required")
    if args.mode is None or args.material is None:
        parser.error("legacy usage requires both --mode and --material")
    if args.mode == "s":
        return _normalize_sequences([args.material])
    try:
        return read_sequences_csv(args.material, args.sequence_column)
    except ValueError as error:
        parser.error(str(error))


def _format_text(result: SCNLSResult) -> str:
    lines = ["Gapped patterns", "pattern\tcount"]
    lines.extend(
        f"{pattern}\t{count}" for pattern, count in result.gapped_patterns.items()
    )
    lines.extend(("", "Continuous patterns", "pattern\tcount"))
    lines.extend(
        f"{pattern}\t{count}"
        for pattern, count in result.continuous_patterns.items()
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """SCNLS command-line entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    sequences = _resolve_cli_sequences(parser, args)
    try:
        result = SCNLS(
            sequences,
            k=args.k,
            max_gap=args.max_gap,
            workers=args.workers,
            entropy_threshold=args.entropy_threshold,
            top_k=args.top_k,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    if args.format == "json":
        rendered = json.dumps(result.to_dict(), indent=2) + "\n"
    else:
        rendered = _format_text(result)

    if args.output is None:
        print(rendered, end="")
    else:
        try:
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot write output file {args.output}: {error}")
    return 0


__all__ = [
    "SCNLS",
    "SCNLSResult",
    "calculate_entropy_with_expectation",
    "generate_combinations",
    "get_continuous_patterns",
    "main",
    "mine_continuous_patterns",
    "mine_gapped_patterns",
    "read_sequences_csv",
]

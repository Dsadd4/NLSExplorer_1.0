"""Compatibility entry point for the packaged SCNLS implementation."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nlsexplorer.scnls import (
    SCNLS,
    SCNLSResult,
    calculate_entropy_with_expectation,
    generate_combinations,
    get_continuous_patterns,
    main,
    mine_continuous_patterns,
    mine_gapped_patterns,
    read_sequences_csv,
)


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


if __name__ == "__main__":
    raise SystemExit(main())

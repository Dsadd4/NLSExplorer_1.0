"""Command-line interface for NLSExplorer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .checkpoint import (
    DEFAULT_CHECKPOINT_URL,
    CheckpointKind,
    download_checkpoint,
    inspect_checkpoint,
    optimize_checkpoint,
    repack_extracted_checkpoint,
)
from .embedding import (
    ESM1B_FILENAME,
    ESM1B_REGRESSION_FILENAME,
    ESM1B_REGRESSION_URL,
    ESM1B_URL,
)
from .io import SequenceRecord, read_csv_sequences, read_fasta, write_predictions
from .predictor import NLSExplorerPredictor


def default_checkpoint_path() -> Path:
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    return cache_root / "nlsexplorer" / "NLS_loc_modeltes.pt"


def default_esm_path() -> Path:
    return default_checkpoint_path().parent / ESM1B_FILENAME


def _progress(downloaded: int, total: int | None) -> None:
    if total:
        percent = 100 * downloaded / total
        message = (
            f"\rDownloading checkpoint: {downloaded / 1e9:.2f}/"
            f"{total / 1e9:.2f} GB ({percent:.1f}%)"
        )
    else:
        message = f"\rDownloading checkpoint: {downloaded / 1e9:.2f} GB"
    print(message, end="", file=sys.stderr, flush=True)


def _download(args: argparse.Namespace) -> int:
    destination = Path(args.output).expanduser()
    result = download_checkpoint(
        destination,
        url=args.url,
        overwrite=args.overwrite,
        progress=_progress,
    )
    print(file=sys.stderr)
    print(result)
    return 0


def _download_esm(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory).expanduser()
    model_path = output_directory / ESM1B_FILENAME
    regression_path = output_directory / ESM1B_REGRESSION_FILENAME
    download_checkpoint(
        model_path,
        url=ESM1B_URL,
        overwrite=args.overwrite,
        progress=_progress,
        validator=None,
    )
    print(file=sys.stderr)
    download_checkpoint(
        regression_path,
        url=ESM1B_REGRESSION_URL,
        overwrite=args.overwrite,
        progress=_progress,
        validator=None,
    )
    print(file=sys.stderr)
    print(model_path)
    return 0


def _check(args: argparse.Namespace) -> int:
    inspection = inspect_checkpoint(args.checkpoint)
    print(f"path: {inspection.path}")
    print(f"kind: {inspection.kind.value}")
    print(f"message: {inspection.message}")
    if inspection.member_count is not None:
        print(f"members: {inspection.member_count}")
    if inspection.storage_count is not None:
        print(f"tensor storages: {inspection.storage_count}")
    return 0 if inspection.kind is CheckpointKind.INTACT else 2


def _recover(args: argparse.Namespace) -> int:
    result = repack_extracted_checkpoint(
        args.source,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(result)
    return 0


def _optimize(args: argparse.Namespace) -> int:
    result = optimize_checkpoint(
        args.source,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(result)
    return 0


def _prediction_records(args: argparse.Namespace) -> list[SequenceRecord]:
    if args.sequence:
        return [
            SequenceRecord(f"sequence_{index + 1}", sequence)
            for index, sequence in enumerate(args.sequence)
        ]
    if args.fasta:
        return read_fasta(args.fasta)
    return read_csv_sequences(
        args.csv,
        sequence_column=args.sequence_column,
        id_column=args.id_column,
    )


def _predict(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        if not args.download:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Run `nlsexplorer download-model` "
                "or add --download. The downloaded file must not be extracted."
            )
        download_checkpoint(checkpoint, progress=_progress)
        print(file=sys.stderr)

    esm_checkpoint = Path(args.esm_checkpoint).expanduser() if args.esm_checkpoint else None
    if esm_checkpoint is None and args.download:
        esm_checkpoint = default_esm_path()
        if not esm_checkpoint.exists():
            namespace = argparse.Namespace(
                output_directory=str(esm_checkpoint.parent),
                overwrite=False,
            )
            _download_esm(namespace)

    records = _prediction_records(args)
    predictor = NLSExplorerPredictor.from_checkpoint(
        checkpoint,
        device=args.device,
        esm_checkpoint=esm_checkpoint,
    )
    predictions = predictor.predict(
        [record.sequence for record in records],
        identifiers=[record.identifier for record in records],
        batch_size=args.batch_size,
        attention_fraction=args.attention_fraction,
    )
    write_predictions(
        [prediction.to_dict() for prediction in predictions],
        output=args.output,
        output_format=args.format,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlsexplorer",
        description="Nuclear localization prediction and interpretable A2KA analysis.",
    )
    parser.add_argument("--version", action="version", version="NLSExplorer 1.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download-model", help="Download the lean NLSExplorer checkpoint with resume support."
    )
    download.add_argument("--output", default=str(default_checkpoint_path()))
    download.add_argument(
        "--url",
        default=DEFAULT_CHECKPOINT_URL,
    )
    download.add_argument("--overwrite", action="store_true")
    download.set_defaults(handler=_download)

    download_esm = commands.add_parser(
        "download-esm", help="Download ESM-1b and contact weights with resume support."
    )
    download_esm.add_argument(
        "--output-directory",
        default=str(default_checkpoint_path().parent),
    )
    download_esm.add_argument("--overwrite", action="store_true")
    download_esm.set_defaults(handler=_download_esm)

    check = commands.add_parser(
        "check-model", help="Identify intact, incomplete, or accidentally extracted checkpoints."
    )
    check.add_argument("checkpoint")
    check.set_defaults(handler=_check)

    recover = commands.add_parser(
        "recover-checkpoint", help="Repack an accidentally extracted checkpoint directory."
    )
    recover.add_argument("source")
    recover.add_argument("--output")
    recover.add_argument("--overwrite", action="store_true")
    recover.set_defaults(handler=_recover)

    optimize = commands.add_parser(
        "optimize-checkpoint",
        help="Remove only the published model's forward-unused ll weights.",
    )
    optimize.add_argument("source")
    optimize.add_argument("--output")
    optimize.add_argument("--overwrite", action="store_true")
    optimize.set_defaults(handler=_optimize)

    predict = commands.add_parser(
        "predict", help="Predict nuclear localization and candidate NLS regions."
    )
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sequence", action="append", help="Protein sequence; repeatable.")
    inputs.add_argument("--fasta", help="Input FASTA file.")
    inputs.add_argument("--csv", help="Input CSV file.")
    predict.add_argument("--sequence-column", default="sequence")
    predict.add_argument("--id-column", default="id")
    predict.add_argument("--checkpoint", default=str(default_checkpoint_path()))
    predict.add_argument("--download", action="store_true")
    predict.add_argument("--esm-checkpoint", help="Local ESM-1b checkpoint for offline use.")
    predict.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    predict.add_argument("--batch-size", type=int, default=1)
    predict.add_argument("--attention-fraction", type=float, default=0.10)
    predict.add_argument("--output")
    predict.add_argument("--format", choices=("json", "tsv"), default="json")
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

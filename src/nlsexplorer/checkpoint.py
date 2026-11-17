"""Download, inspect, recover, and load NLSExplorer checkpoints."""

from __future__ import annotations

import http.client
import os
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .device import resolve_device
from .model import (
    DEFAULT_ATTENTION_CONFIG,
    DEFAULT_HIDDEN_DIM,
    NLSExplorerModel,
)

LEGACY_CHECKPOINT_URL = (
    "http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/NLS_loc_modeltes"
)
DEFAULT_CHECKPOINT_URL = (
    "https://github.com/Dsadd4/NLSExplorer_1.0/releases/download/"
    "v1.1.0/NLSExplorer_A2KA_v1.1.0.pt"
)


class CheckpointKind(str, Enum):
    INTACT = "intact"
    EXTRACTED = "extracted"
    INNER_DATA_PICKLE = "inner_data_pickle"
    INVALID = "invalid"
    MISSING = "missing"


@dataclass(frozen=True)
class CheckpointInspection:
    path: Path
    kind: CheckpointKind
    message: str
    member_count: int | None = None
    storage_count: int | None = None


class CheckpointError(RuntimeError):
    """Base class for actionable checkpoint errors."""


class CheckpointFormatError(CheckpointError):
    """Raised when a path is not an intact PyTorch checkpoint."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when checkpoint keys do not match the requested model."""


class _RetryableDownloadError(CheckpointError):
    """Internal signal for an interrupted but resumable transfer."""


def _extracted_layout(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data.pkl").is_file()
        and (path / "data").is_dir()
        and (path / "version").is_file()
    )


def inspect_checkpoint(path: str | os.PathLike[str]) -> CheckpointInspection:
    """Classify an intact checkpoint or a commonly mis-extracted layout."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return CheckpointInspection(
            resolved, CheckpointKind.MISSING, f"Checkpoint does not exist: {resolved}"
        )
    if _extracted_layout(resolved):
        storage_count = sum(child.is_file() for child in (resolved / "data").iterdir())
        return CheckpointInspection(
            resolved,
            CheckpointKind.EXTRACTED,
            "This is an extracted torch.save archive. Repack the directory or "
            "download the original NLS_loc_modeltes file; do not load data.pkl directly.",
            member_count=storage_count + 2,
            storage_count=storage_count,
        )
    if resolved.is_file() and resolved.name == "data.pkl" and _extracted_layout(
        resolved.parent
    ):
        storage_count = sum(
            child.is_file() for child in (resolved.parent / "data").iterdir()
        )
        return CheckpointInspection(
            resolved,
            CheckpointKind.INNER_DATA_PICKLE,
            "data.pkl is only checkpoint metadata and cannot be loaded alone. "
            "Load the original checkpoint file or repack its parent directory.",
            member_count=storage_count + 2,
            storage_count=storage_count,
        )
    if not resolved.is_file() or not zipfile.is_zipfile(resolved):
        return CheckpointInspection(
            resolved,
            CheckpointKind.INVALID,
            "The path is not a PyTorch ZIP checkpoint or an extracted checkpoint directory.",
        )

    try:
        with zipfile.ZipFile(resolved) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as error:
        return CheckpointInspection(
            resolved, CheckpointKind.INVALID, f"Invalid ZIP checkpoint: {error}"
        )

    data_pickle = [name for name in names if name.endswith("/data.pkl")]
    version = [name for name in names if name.endswith("/version")]
    roots = {name.rsplit("/data.pkl", 1)[0] for name in data_pickle}
    storage_names = [
        name
        for name in names
        if any(name.startswith(f"{root}/data/") for root in roots)
    ]
    if len(data_pickle) == 1 and len(version) == 1 and storage_names:
        return CheckpointInspection(
            resolved,
            CheckpointKind.INTACT,
            "Intact PyTorch checkpoint.",
            member_count=len(names),
            storage_count=len(storage_names),
        )
    return CheckpointInspection(
        resolved,
        CheckpointKind.INVALID,
        "ZIP file does not contain the data.pkl, data/*, and version records "
        "required by a torch.save checkpoint.",
        member_count=len(names),
        storage_count=len(storage_names),
    )


def require_intact_checkpoint(path: str | os.PathLike[str]) -> Path:
    inspection = inspect_checkpoint(path)
    if inspection.kind is CheckpointKind.INTACT:
        return inspection.path
    raise CheckpointFormatError(inspection.message)


def repack_extracted_checkpoint(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Rebuild a torch-loadable ZIP64 file from an extracted checkpoint directory."""

    source_path = Path(source).expanduser().resolve()
    inspection = inspect_checkpoint(source_path)
    if inspection.kind is CheckpointKind.INNER_DATA_PICKLE:
        source_path = source_path.parent
        inspection = inspect_checkpoint(source_path)
    if inspection.kind is not CheckpointKind.EXTRACTED:
        raise CheckpointFormatError(
            f"Expected a directory containing data.pkl, data/, and version: {source_path}"
        )

    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else source_path.with_name(f"{source_path.name}.pt")
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.part")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with zipfile.ZipFile(
            partial_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for file_path in sorted(source_path.rglob("*")):
                if file_path.is_file():
                    member = Path("archive") / file_path.relative_to(source_path)
                    archive.write(file_path, member.as_posix())
        require_intact_checkpoint(partial_path)
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return output_path


def _download_total(response: Any, resumed_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return int(content_length) + resumed_bytes
    return None


def download_checkpoint(
    destination: str | os.PathLike[str],
    *,
    url: str = DEFAULT_CHECKPOINT_URL,
    chunk_size: int = 8 * 1024 * 1024,
    progress: Callable[[int, int | None], None] | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> Path:
    """Download with HTTP Range resume and automatic recovery from stalled transfers."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_delay < 0:
        raise ValueError("retry_delay must be non-negative")
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if not overwrite:
            require_intact_checkpoint(destination_path)
            return destination_path
        destination_path.unlink()

    partial_path = destination_path.with_name(f"{destination_path.name}.part")
    failed_attempts = 0
    while True:
        resumed_bytes = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"User-Agent": "NLSExplorer checkpoint downloader"}
        if resumed_bytes:
            headers["Range"] = f"bytes={resumed_bytes}-"

        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
            with response:
                status = getattr(response, "status", response.getcode())
                if resumed_bytes and status != 206:
                    resumed_bytes = 0
                    mode = "wb"
                else:
                    mode = "ab" if resumed_bytes else "wb"
                total = _download_total(response, resumed_bytes)
                downloaded = resumed_bytes
                with partial_path.open(mode) as output_file:
                    while True:
                        block = response.read(chunk_size)
                        if not block:
                            break
                        output_file.write(block)
                        downloaded += len(block)
                        if progress is not None:
                            progress(downloaded, total)

            if total is not None and downloaded != total:
                raise _RetryableDownloadError(
                    f"transfer stopped at {downloaded} of {total} bytes"
                )
            break
        except urllib.error.HTTPError as error:
            if error.code == 416 and resumed_bytes:
                content_range = error.headers.get("Content-Range", "")
                total_text = content_range.rsplit("/", 1)[-1]
                if total_text.isdigit() and resumed_bytes == int(total_text):
                    require_intact_checkpoint(partial_path)
                    partial_path.replace(destination_path)
                    return destination_path
            if error.code not in {408, 429} and error.code < 500:
                raise CheckpointError(
                    f"Checkpoint download failed: HTTP {error.code}"
                ) from error
            retryable_error: Exception = error
        except (
            TimeoutError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            ConnectionError,
            _RetryableDownloadError,
        ) as error:
            retryable_error = error

        if failed_attempts >= max_retries:
            raise CheckpointError(
                f"Checkpoint download failed after {failed_attempts + 1} attempts; "
                f"rerun to resume {partial_path}. Last error: {retryable_error}"
            ) from retryable_error
        failed_attempts += 1
        if retry_delay:
            time.sleep(retry_delay * failed_attempts)

    require_intact_checkpoint(partial_path)
    partial_path.replace(destination_path)
    return destination_path


def is_legacy_unused_key(key: str) -> bool:
    """Return whether a state-dict key is an unused legacy length embedding."""

    return key.endswith(".ll.weight")


def optimize_checkpoint(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a lean checkpoint without the legacy, forward-unused ``ll`` weights."""

    source_path = require_intact_checkpoint(source)
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else source_path.with_name(f"{source_path.stem}.lean.pt")
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.part")
    partial_path.unlink(missing_ok=True)

    payload = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = _extract_state_dict(payload)
    filtered = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not is_legacy_unused_key(key)
    )
    if not filtered:
        raise CheckpointCompatibilityError("No active model weights remain after filtering.")
    try:
        torch.save(filtered, partial_path)
        require_intact_checkpoint(partial_path)
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return output_path


def _extract_state_dict(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError(
            "Checkpoint must contain a PyTorch state_dict mapping."
        )
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        payload = payload["state_dict"]
    if not payload or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in payload.items()
    ):
        raise CheckpointCompatibilityError(
            "Checkpoint does not contain a plain string-to-tensor state_dict."
        )
    return payload


def load_model(
    checkpoint: str | os.PathLike[str],
    *,
    device: str | torch.device | None = "auto",
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    attention_config: Sequence[int] = DEFAULT_ATTENTION_CONFIG,
    mmap: bool = True,
) -> NLSExplorerModel:
    """Load the published state dict into the lean, checkpoint-compatible model."""

    checkpoint_path = require_intact_checkpoint(checkpoint)
    target_device = resolve_device(device)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=mmap,
    )
    state_dict = _extract_state_dict(payload)
    filtered_state_dict = {
        key: value for key, value in state_dict.items() if not is_legacy_unused_key(key)
    }

    with torch.device("meta"):
        model = NLSExplorerModel(
            hidden_dim=hidden_dim, attention_config=attention_config
        )
    incompatible = model.load_state_dict(
        filtered_state_dict, strict=False, assign=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:8]) or "none"
        unexpected = ", ".join(incompatible.unexpected_keys[:8]) or "none"
        raise CheckpointCompatibilityError(
            "Checkpoint does not match NLSExplorerModel after removing only the "
            f"unused *.ll.weight entries. Missing: {missing}. Unexpected: {unexpected}."
        )

    if target_device.type != "cpu":
        model = model.to(target_device)
    model.eval()
    return model

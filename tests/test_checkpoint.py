from __future__ import annotations

import io
import threading
import zipfile
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from nlsexplorer.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointFormatError,
    CheckpointKind,
    download_checkpoint,
    inspect_checkpoint,
    load_model,
    optimize_checkpoint,
    repack_extracted_checkpoint,
)
from nlsexplorer.model import NLSExplorerModel


def _legacy_state_dict(hidden_dim: int = 8, config: tuple[int, ...] = (2,)):
    torch.manual_seed(11)
    model = NLSExplorerModel(hidden_dim=hidden_dim, attention_config=config).eval()
    state = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())
    for layer_index, units in enumerate(config):
        for unit_index in range(units):
            state[
                f"Att_li.{layer_index}.{unit_index}.ll.weight"
            ] = torch.randn(500, hidden_dim)
    state["AAt.ll.weight"] = torch.randn(500, hidden_dim)
    return model, state


def test_inspect_repack_and_load_extracted_checkpoint(tmp_path: Path) -> None:
    reference_model, state = _legacy_state_dict()
    intact = tmp_path / "NLS_loc_modeltes"
    torch.save(state, intact)

    intact_info = inspect_checkpoint(intact)
    assert intact_info.kind is CheckpointKind.INTACT
    assert intact_info.storage_count == len(state)

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(intact) as archive:
        archive.extractall(extracted)
        root_name = archive.namelist()[0].split("/", 1)[0]
    archive_contents = extracted / root_name

    extracted_info = inspect_checkpoint(archive_contents)
    assert extracted_info.kind is CheckpointKind.EXTRACTED
    data_pickle_info = inspect_checkpoint(archive_contents / "data.pkl")
    assert data_pickle_info.kind is CheckpointKind.INNER_DATA_PICKLE

    repacked = repack_extracted_checkpoint(archive_contents)
    assert inspect_checkpoint(repacked).kind is CheckpointKind.INTACT

    loaded_model = load_model(
        repacked,
        device="cpu",
        hidden_dim=8,
        attention_config=(2,),
    )
    embedding = torch.randn(2, 4, 8)
    with torch.inference_mode():
        expected = reference_model(embedding)
        observed = loaded_model(embedding)
    assert torch.equal(expected[0], observed[0])
    for expected_layer, observed_layer in zip(expected[1], observed[1], strict=True):
        assert torch.equal(expected_layer, observed_layer)


def test_load_model_rejects_any_nonlegacy_key_mismatch(tmp_path: Path) -> None:
    _, state = _legacy_state_dict()
    state["unexpected.weight"] = torch.ones(1)
    path = tmp_path / "bad.pt"
    torch.save(state, path)

    with pytest.raises(CheckpointCompatibilityError, match="Unexpected"):
        load_model(path, device="cpu", hidden_dim=8, attention_config=(2,))


def test_optimize_checkpoint_removes_only_unused_ll_weights(tmp_path: Path) -> None:
    _, state = _legacy_state_dict()
    source = tmp_path / "legacy.pt"
    output = tmp_path / "lean.pt"
    torch.save(state, source)

    optimize_checkpoint(source, output)
    lean_state = torch.load(output, map_location="cpu", weights_only=True)

    assert not any(key.endswith(".ll.weight") for key in lean_state)
    assert set(lean_state) == {
        key for key in state if not key.endswith(".ll.weight")
    }
    load_model(output, device="cpu", hidden_dim=8, attention_config=(2,))


def test_load_model_explains_extracted_directory(tmp_path: Path) -> None:
    extracted = tmp_path / "NLS_loc_modeltes"
    (extracted / "data").mkdir(parents=True)
    (extracted / "data.pkl").write_bytes(b"metadata")
    (extracted / "data" / "0").write_bytes(b"tensor")
    (extracted / "version").write_text("3")

    with pytest.raises(CheckpointFormatError, match="extracted torch.save archive"):
        load_model(extracted, device="cpu", hidden_dim=8, attention_config=(2,))


def test_download_checkpoint_resumes_part_file(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    torch.save({"weight": torch.arange(256)}, buffer)
    content = buffer.getvalue()

    class RangeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            start = (
                int(range_header.removeprefix("bytes=").removesuffix("-"))
                if range_header
                else 0
            )
            body = content[start:]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            if range_header:
                content_range = f"bytes {start}-{len(content) - 1}/{len(content)}"
                self.send_header("Content-Range", content_range)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = tmp_path / "checkpoint.pt"
        partial = tmp_path / "checkpoint.pt.part"
        partial.write_bytes(content[:1000])
        progress: list[tuple[int, int | None]] = []
        result = download_checkpoint(
            destination,
            url=f"http://127.0.0.1:{server.server_port}/checkpoint",
            chunk_size=257,
            progress=lambda downloaded, total: progress.append((downloaded, total)),
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.read_bytes() == content
    assert not partial.exists()
    assert progress[-1] == (len(content), len(content))


def test_download_checkpoint_retries_and_resumes_dropped_connection(
    tmp_path: Path,
) -> None:
    buffer = io.BytesIO()
    torch.save({"weight": torch.arange(2048)}, buffer)
    content = buffer.getvalue()

    class DroppedConnectionHandler(BaseHTTPRequestHandler):
        request_count = 0

        def do_GET(self) -> None:
            type(self).request_count += 1
            range_header = self.headers.get("Range")
            start = (
                int(range_header.removeprefix("bytes=").removesuffix("-"))
                if range_header
                else 0
            )
            body = content[start:]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            if range_header:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{len(content) - 1}/{len(content)}",
                )
            self.end_headers()
            if type(self).request_count == 1:
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DroppedConnectionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = tmp_path / "checkpoint.pt"
        result = download_checkpoint(
            destination,
            url=f"http://127.0.0.1:{server.server_port}/checkpoint",
            chunk_size=257,
            max_retries=2,
            retry_delay=0,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert DroppedConnectionHandler.request_count == 2
    assert result.read_bytes() == content


def test_download_accepts_legacy_non_zip_file_without_format_validator(
    tmp_path: Path,
) -> None:
    content = b"legacy-pytorch-checkpoint" * 128

    class LegacyHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), LegacyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = tmp_path / "legacy.pt"
        result = download_checkpoint(
            destination,
            url=f"http://127.0.0.1:{server.server_port}/legacy",
            validator=None,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.read_bytes() == content

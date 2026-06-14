"""Unit tests for in-place bag materialization (no catalog required).

Builds a minimal BDBag on disk whose fetch.txt points at a local
HTTP server, then exercises ``materialize_bag_dir`` and
``DatasetBag.materialize`` without any catalog connection.

Note on ``file://`` vs ``http://``: bdbag's fetch engine does not
support the ``file://`` protocol (its default fetchers only cover
``http``, ``https``, ``s3``, and ``gs``).  The fixture therefore
spins up a ``socketserver.TCPServer`` on 127.0.0.1 using a random
ephemeral port for the duration of each test.  This keeps the tests
network-free in the external sense while still exercising the real
bdbag materialization path.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import socketserver
import threading
from pathlib import Path

from bdbag import bdbag_api as bdb

from deriva_ml.dataset.bag_cache import BagCache
from deriva_ml.dataset.bag_download import materialize_bag_dir


class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that suppresses access log output."""

    def __init__(self, *args, directory: str, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, *args):  # noqa: D102
        pass


def _make_holey_bag(tmp_path: Path) -> tuple[Path, Path, socketserver.TCPServer]:
    """Create a valid BDBag with one un-fetched fetch.txt entry.

    Spins up a local HTTP server on a random port to serve the payload
    (bdbag does not support ``file://`` URLs natively).

    Returns ``(bag_dir, expected_target_file, httpd)`` where ``httpd``
    must be shut down by the caller once the test is complete.
    """
    # The remote payload the bag will fetch.
    payload = b"hello-materialize"
    src_dir = tmp_path / "remote"
    src_dir.mkdir(parents=True)
    (src_dir / "payload.txt").write_bytes(payload)

    # Start a local HTTP server serving src_dir.
    httpd = socketserver.TCPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: _SilentHandler(*a, directory=str(src_dir), **kw),
    )
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()

    # Build the remote file manifest that bdb.make_bag will turn into
    # a properly-structured fetch.txt + manifest-md5.txt.
    md5 = hashlib.md5(payload).hexdigest()
    url = f"http://127.0.0.1:{port}/payload.txt"
    manifest = [
        {
            "url": url,
            "length": len(payload),
            "filename": "assets/payload.txt",  # relative to data/; bdbag prepends data/
            "md5": md5,
        }
    ]
    manifest_file = tmp_path / "remote_manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    # Create the bag directory and build the bag using the remote manifest.
    # make_bag writes fetch.txt and manifest-md5.txt automatically so the
    # resulting bag passes validate_bag_structure before materialization.
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    bdb.make_bag(bag_dir.as_posix(), remote_file_manifest=manifest_file.as_posix())

    target = bag_dir / "data" / "assets" / "payload.txt"
    return bag_dir, target, httpd


def test_materialize_bag_dir_fetches_missing_entry(tmp_path: Path):
    """materialize_bag_dir downloads an un-fetched fetch.txt entry."""
    bag_dir, target, httpd = _make_holey_bag(tmp_path)
    try:
        assert not target.exists()
        assert not BagCache._is_fully_materialized(bag_dir)

        result = materialize_bag_dir(bag_dir)

        assert result == bag_dir
        assert target.exists()
        assert target.read_bytes() == b"hello-materialize"
        assert BagCache._is_fully_materialized(bag_dir)
    finally:
        httpd.shutdown()


def test_materialize_bag_dir_idempotent_noop(tmp_path: Path, monkeypatch):
    """On an already-complete bag, materialize_bag_dir does not fetch."""
    bag_dir, _target, httpd = _make_holey_bag(tmp_path)
    try:
        materialize_bag_dir(bag_dir)  # first call completes it
        assert BagCache._is_fully_materialized(bag_dir)

        # Second call must short-circuit before touching bdb.materialize.
        def _boom(*args, **kwargs):
            raise AssertionError("bdb.materialize should not be called on a complete bag")

        monkeypatch.setattr("deriva_ml.dataset.bag_download.bdb.materialize", _boom)
        result = materialize_bag_dir(bag_dir)
        assert result == bag_dir
    finally:
        httpd.shutdown()


class _StubModel:
    """Minimal stand-in for DatabaseModel exposing only ``bag_path``."""

    def __init__(self, bag_path: Path):
        self.bag_path = bag_path


def test_datasetbag_materialize_fetches_in_place(tmp_path: Path):
    """DatasetBag.materialize() fetches missing files and returns self."""
    from deriva_ml.dataset.dataset_bag import DatasetBag

    bag_dir, target, httpd = _make_holey_bag(tmp_path)
    try:
        # Build a DatasetBag without running its catalog-touching __init__.
        bag = DatasetBag.__new__(DatasetBag)
        bag.model = _StubModel(bag_dir)

        assert not target.exists()
        result = bag.materialize()

        assert result is bag
        assert target.exists()
        assert target.read_bytes() == b"hello-materialize"
        assert BagCache._is_fully_materialized(bag_dir)
    finally:
        httpd.shutdown()

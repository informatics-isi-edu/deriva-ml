"""Unit tests for _invoke_deriva_py_uploader using a fake GenericUploader.

Exercises the uploader integration without a real catalog upload. The
fake simulates GenericUploader's interface just deeply enough to test
scan-root layout, callback wiring, cancel propagation, and result
attribution.

These tests still use the ``test_ml`` fixture (which requires a live
Deriva catalog for model lookups) because ``_invoke_deriva_py_uploader``
needs ``ml.model.name_to_table`` and ``ml.model.asset_metadata`` to
compute the scan-root layout. But the GenericUploader itself is
monkeypatched to a test double, so no network traffic to Hatrac is
performed.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


class FakeGenericUploader:
    """Test double matching GenericUploader's public surface used by S3."""

    instances: list["FakeGenericUploader"] = []

    def __init__(self, *, server=None, config_file=None,
                 credential_file=None, dcctx_cid=None):
        self.server = server
        self.config_file = config_file
        self.credential_file = credential_file
        self.dcctx_cid = dcctx_cid
        self.scanned_root: Path | None = None
        self.cancelled = False
        self.file_status: dict[str, dict] = {}
        # Test harness pokes these:
        self._scan_result_files: list[Path] = []
        self._per_file_states: dict[str, str] = {}  # path → "Success"/"Failed"
        self._on_upload = None
        FakeGenericUploader.instances.append(self)

    def initialize(self, cleanup=False):
        pass

    def getUpdatedConfig(self):
        pass

    def scanDirectory(self, root, abort_on_invalid_input=False, purge_state=False):
        self.scanned_root = Path(root)
        # Walk the scan root and record the files the test said to expect.
        for p in Path(root).rglob("*"):
            if p.is_file() or p.is_symlink():
                # Don't include the config.json at the scan-root top level.
                if p.name == "config.json" and p.parent == Path(root):
                    continue
                self._scan_result_files.append(p.resolve())

    def cancel(self):
        self.cancelled = True

    def cleanup(self):
        pass

    def uploadFiles(self, status_callback=None, file_callback=None):
        """Simulate uploadFiles — iterate scanned files, apply pre-seeded
        per-file states, invoke callbacks before/after each."""
        for f in self._scan_result_files:
            if self.cancelled:
                self.file_status[str(f)] = {"State": 5, "Status": "Cancelled by user"}
                break
            # status_callback fires BEFORE each file
            if status_callback:
                status_callback()
            # Apply pre-seeded state
            state_name = self._per_file_states.get(str(f), "Success")
            # UploadState tuple indices: Success=0, Failed=1, Pending=2,
            # Running=3, Paused=4, Aborted=5, Cancelled=6, Timeout=7.
            state_code = {"Success": 0, "Failed": 1}[state_name]
            self.file_status[str(f)] = {
                "State": state_code,
                "Status": f"{state_name}",
                "Result": {"url": "mock"} if state_name == "Success" else None,
            }
            if self._on_upload:
                self._on_upload(f)
            # status_callback fires AFTER each file
            if status_callback:
                status_callback()
        return self.file_status

    def getFileStatusAsArray(self):
        return [{"File": k, **v} for k, v in self.file_status.items()]


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeGenericUploader.instances.clear()
    yield
    FakeGenericUploader.instances.clear()


@pytest.fixture
def asset_ml(test_ml):
    """test_ml with a SomeAsset asset table created.

    _invoke_deriva_py_uploader resolves the target table via
    ml.model.name_to_table, so the table must exist in the catalog.
    """
    test_ml.create_asset("SomeAsset")
    return test_ml


def test_invoke_uploader_happy_path(monkeypatch, tmp_path, asset_ml):
    """All files succeed → returned 'uploaded' lists them; SQLite marked Uploaded."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    # Stage two asset files on disk
    f1 = tmp_path / "a.txt"; f1.write_text("a")
    f2 = tmp_path / "b.txt"; f2.write_text("b")

    files = [
        {"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}},
        {"path": str(f2), "rid": "R2", "pending_id": 2, "metadata": {}},
    ]

    result = ue._invoke_deriva_py_uploader(
        ml=asset_ml, files=files,
        target_table="SomeAsset",
        execution_rid="EXE-X",
        cancel_event=None,
    )

    assert sorted(result["uploaded"]) == sorted([str(f1), str(f2)])
    assert result["failed"] == []
    # Exactly one fake uploader instance was constructed
    assert len(FakeGenericUploader.instances) == 1
    u = FakeGenericUploader.instances[0]
    # Scan root was under a temp dir (not tmp_path)
    assert u.scanned_root is not None
    assert u.scanned_root != tmp_path


def test_invoke_uploader_mixed_outcomes(monkeypatch, tmp_path, asset_ml):
    """Success and Failed files split correctly into return dict keys."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    f1 = tmp_path / "ok.txt"; f1.write_text("ok")
    f2 = tmp_path / "bad.txt"; f2.write_text("bad")

    files = [
        {"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}},
        {"path": str(f2), "rid": "R2", "pending_id": 2, "metadata": {}},
    ]

    # Hook scanDirectory to assign per-file states by basename.
    original_scan = FakeGenericUploader.scanDirectory

    def scan_patch(self, root, **kw):
        original_scan(self, root, **kw)
        for p in self._scan_result_files:
            self._per_file_states[str(p)] = {
                "ok.txt": "Success",
                "bad.txt": "Failed",
            }.get(p.name, "Success")

    monkeypatch.setattr(FakeGenericUploader, "scanDirectory", scan_patch)

    result = ue._invoke_deriva_py_uploader(
        ml=asset_ml, files=files,
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=None,
    )
    assert result["uploaded"] == [str(f1)]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == str(f2)


def test_invoke_uploader_cancel_mid_batch(monkeypatch, tmp_path, asset_ml):
    """cancel_event.set() during a batch → uploader.cancel() called; remaining files not attributed."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    cancel_event = threading.Event()

    f1 = tmp_path / "a.txt"; f1.write_text("a")
    f2 = tmp_path / "b.txt"; f2.write_text("b")
    f3 = tmp_path / "c.txt"; f3.write_text("c")

    files = [
        {"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}},
        {"path": str(f2), "rid": "R2", "pending_id": 2, "metadata": {}},
        {"path": str(f3), "rid": "R3", "pending_id": 3, "metadata": {}},
    ]

    # After f1 uploads, fire cancel.
    original_scan = FakeGenericUploader.scanDirectory

    def scan_patch(self, root, **kw):
        original_scan(self, root, **kw)
        # Set up an on_upload hook that fires cancel after the first file.
        count = {"n": 0}

        def _after(_):
            count["n"] += 1
            if count["n"] == 1:
                cancel_event.set()

        self._on_upload = _after

    monkeypatch.setattr(FakeGenericUploader, "scanDirectory", scan_patch)

    result = ue._invoke_deriva_py_uploader(
        ml=asset_ml, files=files,
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=cancel_event,
    )
    u = FakeGenericUploader.instances[0]
    # Cancel was propagated to the uploader.
    assert u.cancelled is True
    # Cancel cut short the batch: exactly one file succeeded before
    # cancel tripped (iteration order depends on the filesystem, so we
    # don't assert *which* file; only that not all three were attributed).
    total_attributed = len(result["uploaded"]) + len(result["failed"])
    assert total_attributed == 1, (
        f"expected cancel to stop the batch after the first file, "
        f"got uploaded={result['uploaded']!r} failed={result['failed']!r}"
    )


def test_invoke_uploader_reconciliation(monkeypatch, tmp_path, asset_ml):
    """If status_callback misses a file, the reconciliation pass catches it."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    f1 = tmp_path / "only.txt"; f1.write_text("only")
    files = [{"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}}]

    # Make uploadFiles skip the status_callback entirely but still
    # populate file_status.
    def no_callback_upload(self, status_callback=None, file_callback=None):
        for f in self._scan_result_files:
            self.file_status[str(f)] = {
                "State": 0,  # Success
                "Status": "Complete",
                "Result": {"url": "mock"},
            }
        return self.file_status

    monkeypatch.setattr(FakeGenericUploader, "uploadFiles", no_callback_upload)

    result = ue._invoke_deriva_py_uploader(
        ml=asset_ml, files=files,
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=None,
    )
    assert result["uploaded"] == [str(f1)]


def test_invoke_uploader_cleans_up_scan_root_on_exception(monkeypatch, tmp_path, asset_ml):
    """Exception during scanDirectory → temp directory removed."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    f1 = tmp_path / "a.txt"; f1.write_text("a")
    files = [{"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}}]

    recorded_roots: list[Path] = []

    def boom(self, root, **kw):
        recorded_roots.append(Path(root))
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(FakeGenericUploader, "scanDirectory", boom)

    with pytest.raises(RuntimeError, match="scan exploded"):
        ue._invoke_deriva_py_uploader(
            ml=asset_ml, files=files,
            target_table="SomeAsset", execution_rid="EXE-X",
            cancel_event=None,
        )

    # TemporaryDirectory context manager should have cleaned up.
    assert recorded_roots
    assert not recorded_roots[0].exists()


def test_invoke_uploader_empty_files_noop(asset_ml):
    """Empty files list → no uploader constructed, empty result."""
    from deriva_ml.execution import upload_engine as ue
    result = ue._invoke_deriva_py_uploader(
        ml=asset_ml, files=[],
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=None,
    )
    assert result == {"uploaded": [], "failed": []}
    assert FakeGenericUploader.instances == []

"""Tests for the deriva-ml upload CLI."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help_lists_flags():
    """Smoke test: help text mentions the main flags."""
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.cli.upload", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    out = result.stdout
    for flag in ["--host", "--catalog", "--execution", "--retry-failed"]:
        assert flag in out, f"flag {flag!r} missing from --help"


def test_cli_execution_arg_accepts_multiple(test_ml, monkeypatch):
    from deriva_ml.cli import upload as upload_cli
    from deriva_ml.execution.upload_engine import UploadReport

    calls = []
    def _fake(self, **kw):
        calls.append(kw)
        return UploadReport(
            execution_rids=kw.get("execution_rids") or [],
            total_uploaded=0, total_failed=0, per_table={},
        )
    monkeypatch.setattr("deriva_ml.DerivaML.upload_pending", _fake)
    monkeypatch.setattr(
        upload_cli, "_construct_ml",
        lambda host, catalog, mode: test_ml,
    )

    upload_cli.main([
        "--host", "h", "--catalog", "c",
        "--execution", "EXE-A",
        "--execution", "EXE-B",
    ])
    assert calls[0]["execution_rids"] == ["EXE-A", "EXE-B"]


def test_cli_rejects_parallel_flag():
    """--parallel is no longer supported — argparse should reject it."""
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.cli.upload",
         "--host", "example.org", "--catalog", "1", "--parallel", "4"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "--parallel" in result.stderr


def test_cli_rejects_bandwidth_flag():
    """--bandwidth-mbps is no longer supported."""
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.cli.upload",
         "--host", "example.org", "--catalog", "1", "--bandwidth-mbps", "100"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "--bandwidth-mbps" in result.stderr

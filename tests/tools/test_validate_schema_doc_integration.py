"""Integration tests — run the validator on the real repo files.

If this test fails, docs/reference/schema.md has drifted from
src/deriva_ml/schema/create_schema.py. Fix one or both.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_validator_runs_clean_on_current_repo():
    """The authoritative doc and code agree. This IS the CI gate."""
    from deriva_ml.tools.validate_schema_doc import (
        diff_schemas, load_from_code, load_from_doc,
    )

    doc_path = REPO_ROOT / "docs" / "reference" / "schema.md"
    code_path = REPO_ROOT / "src" / "deriva_ml" / "schema" / "create_schema.py"

    expected = load_from_doc(doc_path)
    actual = load_from_code(code_path)
    mismatches = diff_schemas(expected=expected, actual=actual)

    assert mismatches == [], (
        "schema.md and create_schema.py disagree:\n"
        + "\n".join(f"  - {m.kind.value}: {m.detail}" for m in mismatches)
    )


def test_cli_exits_zero_on_current_repo():
    """Invoking the CLI against the real paths returns exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.tools.validate_schema_doc"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "agree" in result.stdout

"""Grep gate: fail if any Phase-2-Subsystem-1a legacy Status references remain.

Fails loudly if the library still contains lowercase ExecutionStatus
member references, legacy Status enum usage, or legacy value strings
that Phase 2 Subsystem 1a was supposed to purge.

Catches missed migrations before merge.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns that, if found in src/deriva_ml/, indicate an incomplete
# migration. The validator runs each via grep -rEn.
FORBIDDEN_PATTERNS = [
    # Legacy Status enum identifiers
    r"from deriva_ml\.core\.enums import.*\bStatus\b",
    r"from \.enums import.*\bStatus\b",
    r"^class Status\b",
    # Legacy Status member refs (whole-word so we don't catch
    # PendingRowStatus.failed etc.)
    r"\bStatus\.(pending|running|completed|initializing|aborted|failed|created)\b",
    # Lowercase ExecutionStatus member refs
    r"ExecutionStatus\.(created|running|stopped|failed|pending_upload|uploaded|aborted)\b",
]

# Files that legitimately reference these patterns and should be
# excluded (this very test file matches its own patterns).
ALLOW_FILES = {
    "tests/test_migration_complete.py",
}


def test_no_legacy_status_references_in_src():
    """Fails if any FORBIDDEN_PATTERNS match any file under src/deriva_ml/."""
    hits: list[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        cmd = [
            "grep",
            "-rEn",
            "--include=*.py",
            pattern,
            str(REPO_ROOT / "src" / "deriva_ml"),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
        # grep returns 0 on hit, 1 on no-hit, 2 on error.
        if result.returncode == 0 and result.stdout.strip():
            hits.append(f"pattern={pattern!r}\n{result.stdout}")
    assert not hits, (
        "Legacy Status references found in src/deriva_ml/:\n\n"
        + "\n".join(hits)
    )


def test_executionstatus_values_are_title_case_canonical():
    """The 7 canonical ExecutionStatus members exist with expected values."""
    from deriva_ml.execution.state_store import ExecutionStatus

    expected = {
        "Created": "Created",
        "Running": "Running",
        "Stopped": "Stopped",
        "Failed": "Failed",
        "Pending_Upload": "Pending_Upload",
        "Uploaded": "Uploaded",
        "Aborted": "Aborted",
    }
    actual = {m.name: m.value for m in ExecutionStatus}
    assert actual == expected

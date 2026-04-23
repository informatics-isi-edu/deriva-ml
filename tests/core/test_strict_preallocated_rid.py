"""Tests for the set_strict_preallocated_rid / is_strict_preallocated_rid helpers.

See Bug E.2 — the helpers manage the annotation
``tag:isrd.isi.edu,2026:strict-preallocated-rid`` on asset tables,
which opts the table into strict-mode RID checks during upload.
"""
from __future__ import annotations

import pytest

from deriva_ml.core.mixins.annotation import STRICT_PREALLOCATED_RID_TAG


class TestStrictPreallocatedRid:
    """Tests for strict-preallocated-RID annotation helpers."""

    def test_is_strict_default_false(self, test_ml):
        """An asset table without the annotation returns False."""
        # Execution_Asset is a built-in asset table with no strict annotation.
        assert test_ml.is_strict_preallocated_rid("Execution_Asset") is False

    def test_set_strict_true_updates_annotation(self, test_ml):
        """set_strict_preallocated_rid(True) applies the annotation."""
        # Use Execution_Asset (built-in asset table).
        test_ml.set_strict_preallocated_rid("Execution_Asset", strict=True)
        table = test_ml.model.name_to_table("Execution_Asset")
        assert table.annotations.get(STRICT_PREALLOCATED_RID_TAG) == {"strict": True}
        # is_strict_preallocated_rid now returns True.
        assert test_ml.is_strict_preallocated_rid("Execution_Asset") is True
        # Cleanup so other tests see a clean slate.
        test_ml.set_strict_preallocated_rid("Execution_Asset", strict=False)

    def test_set_strict_false_removes_annotation(self, test_ml):
        """set_strict_preallocated_rid(False) removes the annotation."""
        # Apply first, then remove.
        test_ml.set_strict_preallocated_rid("Execution_Asset", strict=True)
        assert test_ml.is_strict_preallocated_rid("Execution_Asset") is True
        test_ml.set_strict_preallocated_rid("Execution_Asset", strict=False)
        table = test_ml.model.name_to_table("Execution_Asset")
        assert STRICT_PREALLOCATED_RID_TAG not in table.annotations
        assert test_ml.is_strict_preallocated_rid("Execution_Asset") is False

    def test_set_strict_raises_for_non_asset_table(self, test_ml):
        """Non-asset tables (e.g., Workflow) cannot have strict annotation."""
        from deriva_ml.core.exceptions import DerivaMLTableTypeError
        with pytest.raises(DerivaMLTableTypeError):
            test_ml.set_strict_preallocated_rid("Workflow", strict=True)

    def test_is_strict_handles_malformed_annotation(self, test_ml):
        """If the annotation is set to a non-dict value, returns False."""
        table = test_ml.model.name_to_table("Execution_Asset")
        # Planting a malformed annotation directly (not via the helper).
        table.annotations[STRICT_PREALLOCATED_RID_TAG] = "bogus"
        try:
            assert test_ml.is_strict_preallocated_rid("Execution_Asset") is False
        finally:
            # Cleanup.
            table.annotations.pop(STRICT_PREALLOCATED_RID_TAG, None)

    def test_is_strict_false_when_strict_key_missing(self, test_ml):
        """{"other": true} without "strict" key returns False."""
        table = test_ml.model.name_to_table("Execution_Asset")
        table.annotations[STRICT_PREALLOCATED_RID_TAG] = {"other": True}
        try:
            assert test_ml.is_strict_preallocated_rid("Execution_Asset") is False
        finally:
            table.annotations.pop(STRICT_PREALLOCATED_RID_TAG, None)

    def test_is_strict_false_when_strict_is_false(self, test_ml):
        """{"strict": false} explicitly returns False."""
        table = test_ml.model.name_to_table("Execution_Asset")
        table.annotations[STRICT_PREALLOCATED_RID_TAG] = {"strict": False}
        try:
            assert test_ml.is_strict_preallocated_rid("Execution_Asset") is False
        finally:
            table.annotations.pop(STRICT_PREALLOCATED_RID_TAG, None)

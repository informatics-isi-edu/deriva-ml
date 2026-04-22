"""Unit tests for DerivaModel.asset_metadata_columns."""
from __future__ import annotations

import pytest


def test_asset_metadata_columns_returns_column_objects(test_ml):
    """For an asset table with metadata cols, returns sorted Column objects."""
    # Use Execution_Metadata which is a known asset table with known columns.
    # We're not checking specific columns here; we're checking shape.
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    assert isinstance(cols, list)
    for c in cols:
        assert hasattr(c, "name")
        assert hasattr(c, "nullok")


def test_asset_metadata_columns_sorted_by_name(test_ml):
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    names = [c.name for c in cols]
    assert names == sorted(names)


def test_asset_metadata_columns_excludes_standard_asset_columns(test_ml):
    """Must not include Filename, URL, Length, MD5, Description, or system columns."""
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    names = {c.name for c in cols}
    for forbidden in ("Filename", "URL", "Length", "MD5", "Description",
                      "RID", "RCT", "RMT", "RCB", "RMB"):
        assert forbidden not in names


def test_asset_metadata_columns_matches_asset_metadata_set(test_ml):
    """The column-object list names must equal asset_metadata()'s set."""
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    names = {c.name for c in cols}
    assert names == test_ml.model.asset_metadata("Execution_Metadata")


def test_asset_metadata_columns_raises_for_non_asset_table(test_ml):
    from deriva_ml.core.exceptions import DerivaMLTableTypeError
    with pytest.raises(DerivaMLTableTypeError):
        test_ml.model.asset_metadata_columns("Workflow")   # not an asset

"""Unit tests for _validate_pending_asset_metadata."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _col(name: str, nullok: bool = True):
    """Build a mock Column with name + nullok."""
    c = MagicMock()
    c.name = name
    c.nullok = nullok
    return c


def _fake_model(columns_by_table: dict[str, list]):
    """Build a fake DerivaModel.asset_metadata_columns() for the given
    mapping of table_name -> list of mock Column objects."""
    model = MagicMock()
    model.asset_metadata_columns.side_effect = lambda t: columns_by_table.get(t, [])
    return model


def _fake_manifest(entries: dict):
    """Build a fake AssetManifest.pending_assets() that returns the
    given dict of {key: AssetEntry}."""
    manifest = MagicMock()
    manifest.pending_assets.return_value = entries
    return manifest


def _entry(asset_table: str, schema: str = "test-schema", metadata: dict | None = None):
    from deriva_ml.asset.manifest import AssetEntry
    return AssetEntry(
        asset_table=asset_table,
        schema=schema,
        metadata=metadata or {},
    )


def test_empty_manifest_returns_none():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({})
    manifest = _fake_manifest({})
    # Should not raise.
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_asset_with_no_metadata_columns_passes():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({"Execution_Asset": []})
    manifest = _fake_manifest({
        "Execution_Asset/f.bin": _entry("Execution_Asset"),
    })
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_all_required_metadata_present_passes():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={"Acquisition_Time": "2026-01-01"}),
    })
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_missing_single_not_null_column_raises():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    msg = str(ei.value)
    assert "Image/a.png" in msg
    assert "Acquisition_Time" in msg


def test_missing_multiple_columns_aggregated():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [
            _col("Acquisition_Date", nullok=False),
            _col("Acquisition_Time", nullok=False),
        ],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    msg = str(ei.value)
    # Both columns listed, sorted order
    assert "Acquisition_Date" in msg
    assert "Acquisition_Time" in msg
    assert msg.index("Acquisition_Date") < msg.index("Acquisition_Time")


def test_missing_across_multiple_assets_aggregated():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
        "Plate": [_col("Well_Position", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),
        "Plate/p.json": _entry("Plate", metadata={}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    msg = str(ei.value)
    assert "Image/a.png" in msg
    assert "Plate/p.json" in msg
    assert "Acquisition_Time" in msg
    assert "Well_Position" in msg


def test_nullable_missing_is_not_an_error():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({
        "Image": [_col("Description_Note", nullok=True)],  # nullable
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),  # no value supplied
    })
    # Nullable missing is fine — sentinel path handles it.
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_explicit_none_value_for_not_null_column_raises():
    """Explicit None for a NOT-NULL column is treated same as absent — raises.

    This is the Bug C root cause: {"col": None} would previously flow
    through downstream staging as the string "None" and fail the catalog
    insert. The validator must reject it at upload time instead.
    """
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={"Acquisition_Time": None}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    assert "Acquisition_Time" in str(ei.value)
    assert "Image/a.png" in str(ei.value)


def test_explicit_none_value_for_nullable_column_passes():
    """Explicit None for a NULLABLE column is fine — sentinel path handles it."""
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({
        "Image": [_col("Description_Note", nullok=True)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={"Description_Note": None}),
    })
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_error_message_is_deterministic():
    """Same manifest twice → byte-identical error messages (sorted)."""
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [
            _col("Zeta", nullok=False),
            _col("Alpha", nullok=False),
        ],
    })
    manifest = _fake_manifest({
        "Image/z.png": _entry("Image", metadata={}),
        "Image/a.png": _entry("Image", metadata={}),
    })
    msgs = []
    for _ in range(2):
        try:
            _validate_pending_asset_metadata(model, manifest)
        except DerivaMLValidationError as e:
            msgs.append(str(e))
    assert msgs[0] == msgs[1]
    # Sorted: Alpha before Zeta, a.png before z.png
    assert msgs[0].index("Image/a.png") < msgs[0].index("Image/z.png")

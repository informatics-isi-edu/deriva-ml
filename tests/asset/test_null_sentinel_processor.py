"""Unit tests for NullSentinelProcessor."""
from __future__ import annotations


def test_single_sentinel_converted_to_none():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata = {"Acquisition_Time": "__NULL__", "Acquisition_Date": "2026-01-01"}
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata["Acquisition_Time"] is None
    assert metadata["Acquisition_Date"] == "2026-01-01"


def test_multiple_sentinels_in_metadata_all_converted():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata = {"a": "__NULL__", "b": "__NULL__", "c": "real"}
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata["a"] is None
    assert metadata["b"] is None
    assert metadata["c"] == "real"


def test_non_sentinel_values_unchanged():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata = {
        "x": "hello",
        "y": 42,
        "z": None,         # already None — stays None
        "q": "__NUL__",    # almost-sentinel — stays as-is
        "r": "__NULLISH__",
    }
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata["x"] == "hello"
    assert metadata["y"] == 42
    assert metadata["z"] is None
    assert metadata["q"] == "__NUL__"
    assert metadata["r"] == "__NULLISH__"


def test_empty_metadata_is_no_op():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata: dict = {}
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata == {}


def test_asset_table_upload_spec_includes_null_sentinel_processor_when_metadata_present(test_ml):
    """Upload spec for an asset table with metadata wires the processor."""
    from deriva_ml import BuiltinTypes, ColumnDefinition
    from deriva_ml.dataset.upload import asset_table_upload_spec

    # Create an asset table with a metadata column — expect the processor wired.
    test_ml.create_asset(
        "NullSentinelMetaAsset",
        column_defs=[ColumnDefinition(name="foo", type=BuiltinTypes.int4)],
    )
    spec = asset_table_upload_spec(test_ml.model, "NullSentinelMetaAsset")
    pre = spec.get("pre_processors", [])
    types = [p.get("processor_type") for p in pre]
    assert any(
        t and t.endswith("NullSentinelProcessor") for t in types
    ), f"pre_processors must wire NullSentinelProcessor; got {pre}"


def test_asset_table_upload_spec_omits_pre_processors_when_no_metadata(test_ml):
    """Asset tables with zero metadata columns don't need the processor."""
    from deriva_ml.dataset.upload import asset_table_upload_spec

    # Create an asset table with no metadata columns — no processor expected.
    test_ml.create_asset("NullSentinelNoMetaAsset")
    spec = asset_table_upload_spec(test_ml.model, "NullSentinelNoMetaAsset")
    assert "pre_processors" not in spec or not spec["pre_processors"]

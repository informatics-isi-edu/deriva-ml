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

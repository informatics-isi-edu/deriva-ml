"""Unit tests for denormalization-specific exception classes."""

from __future__ import annotations

from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeError,
    DerivaMLDenormalizeMultiLeaf,
    DerivaMLDenormalizeNoSink,
    DerivaMLDenormalizeUnrelatedAnchor,
    DerivaMLException,
)


class TestDenormalizeExceptionHierarchy:
    """All denormalize exceptions inherit from DerivaMLDenormalizeError."""

    def test_multi_leaf_inherits(self) -> None:
        err = DerivaMLDenormalizeMultiLeaf(["A", "B"], ["A", "B", "C"])
        assert isinstance(err, DerivaMLDenormalizeError)
        assert isinstance(err, DerivaMLException)

    def test_ambiguous_path_inherits(self) -> None:
        err = DerivaMLDenormalizeAmbiguousPath("X", "Y", [["X", "Y"], ["X", "Z", "Y"]], ["Z"])
        assert isinstance(err, DerivaMLDenormalizeError)

    def test_downstream_leaf_inherits(self) -> None:
        err = DerivaMLDenormalizeDownstreamLeaf("Subject", ["Image"])
        assert isinstance(err, DerivaMLDenormalizeError)

    def test_unrelated_anchor_inherits(self) -> None:
        err = DerivaMLDenormalizeUnrelatedAnchor(["Foo"], ["Image", "Subject"])
        assert isinstance(err, DerivaMLDenormalizeError)

    def test_no_sink_inherits(self) -> None:
        err = DerivaMLDenormalizeNoSink("cycle detected in FK graph")
        assert isinstance(err, DerivaMLDenormalizeError)


class TestDenormalizeExceptionMessages:
    """Exception messages include the specific fields needed to fix the problem."""

    def test_multi_leaf_message_includes_candidates(self) -> None:
        err = DerivaMLDenormalizeMultiLeaf(["Subject", "Diagnosis"], ["Subject", "Diagnosis"])
        assert "Subject" in str(err)
        assert "Diagnosis" in str(err)
        assert "row_per" in str(err)

    def test_ambiguous_path_message_includes_paths(self) -> None:
        err = DerivaMLDenormalizeAmbiguousPath(
            "Image",
            "Subject",
            [["Image", "Subject"], ["Image", "Observation", "Subject"]],
            ["Observation"],
        )
        assert "Image" in str(err)
        assert "Subject" in str(err)
        assert "Observation" in str(err)
        assert "include_tables" in str(err)
        assert "via" in str(err)

    def test_downstream_leaf_message_includes_tables(self) -> None:
        err = DerivaMLDenormalizeDownstreamLeaf("Subject", ["Image", "Diagnosis"])
        assert "Subject" in str(err)
        assert "Image" in str(err)
        assert "Diagnosis" in str(err)
        assert "aggregation" in str(err).lower()

    def test_unrelated_anchor_message_includes_ignore_option(self) -> None:
        err = DerivaMLDenormalizeUnrelatedAnchor(["Foo"], ["Image"])
        assert "Foo" in str(err)
        assert "ignore_unrelated_anchors" in str(err)


class TestDenormalizeExceptionAttributes:
    """Exceptions expose structured fields for programmatic consumers."""

    def test_multi_leaf_fields(self) -> None:
        err = DerivaMLDenormalizeMultiLeaf(["A", "B"], ["A", "B"])
        assert err.candidates == ["A", "B"]
        assert err.include_tables == ["A", "B"]

    def test_ambiguous_path_fields(self) -> None:
        err = DerivaMLDenormalizeAmbiguousPath("X", "Y", [["X", "Y"], ["X", "Z", "Y"]], ["Z"])
        assert err.from_table == "X"
        assert err.to_table == "Y"
        assert err.paths == [["X", "Y"], ["X", "Z", "Y"]]
        assert err.suggested_intermediates == ["Z"]

    def test_downstream_leaf_fields(self) -> None:
        err = DerivaMLDenormalizeDownstreamLeaf("Subject", ["Image"])
        assert err.row_per == "Subject"
        assert err.downstream_tables == ["Image"]

    def test_unrelated_anchor_fields(self) -> None:
        err = DerivaMLDenormalizeUnrelatedAnchor(["Foo"], ["Image"])
        assert err.unrelated_tables == ["Foo"]
        assert err.include_tables == ["Image"]

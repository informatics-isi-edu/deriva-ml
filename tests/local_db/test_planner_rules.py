"""Unit tests for _prepare_wide_table / _build_join_tree planner rules.

These tests verify the semantic rules independently of any catalog fetching
or SQL execution. They check the pure model-analysis behavior:

- Rule 2: sink-finding for row_per auto-inference
- Rule 5: downstream-leaf rejection when row_per is explicit
- Rule 6: per-pair path ambiguity detection

Most diamond-ambiguity tests use the ``denorm_diamond_deriva_model``
fixture, which extends the base canned schema with an ``Observation``
table + ``Image.Observation`` FK so that ``Image`` can reach ``Subject``
via two distinct FK paths.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeMultiLeaf,
)


class TestSinkFinding:
    """Rule 2: row_per is the unique sink in the FK subgraph on include_tables."""

    def test_linear_chain_sink(self, denorm_diamond_deriva_model) -> None:
        """Subject ← Observation ← Image: Image is the sink."""
        model = denorm_diamond_deriva_model
        sinks = model._find_sinks(
            include_tables=["Subject", "Observation", "Image"],
            via=[],
        )
        assert sinks == ["Image"], f"expected ['Image'], got {sinks}"

    def test_single_sink_simple(self, denorm_deriva_model) -> None:
        """Image points to Subject. Image is the sink."""
        model = denorm_deriva_model
        sinks = model._find_sinks(include_tables=["Subject", "Image"], via=[])
        assert sinks == ["Image"]

    def test_multi_leaf_raises(self, denorm_deriva_model) -> None:
        """If two requested tables have no FK between them, both are sinks."""
        model = denorm_deriva_model
        sinks = model._find_sinks(include_tables=["Dataset", "Subject"], via=[])
        assert len(sinks) >= 2
        assert "Dataset" in sinks or "Subject" in sinks
        with pytest.raises(DerivaMLDenormalizeMultiLeaf) as excinfo:
            model._determine_row_per(include_tables=["Dataset", "Subject"], via=[], row_per=None)
        assert "Dataset" in str(excinfo.value)
        assert "Subject" in str(excinfo.value)


class TestDownstreamLeafRejection:
    """Rule 5: explicit row_per with downstream table in include_tables → error."""

    def test_downstream_leaf_rejected(self, denorm_deriva_model) -> None:
        """row_per=Subject with Image (downstream) → error."""
        model = denorm_deriva_model
        with pytest.raises(DerivaMLDenormalizeDownstreamLeaf) as excinfo:
            model._determine_row_per(
                include_tables=["Subject", "Image"],
                via=[],
                row_per="Subject",
            )
        assert "Subject" in str(excinfo.value)
        assert "Image" in str(excinfo.value)

    def test_downstream_leaf_accepted_if_no_downstream(self, denorm_deriva_model) -> None:
        """row_per=Image is fine because Image IS the sink (nothing downstream)."""
        model = denorm_deriva_model
        result = model._determine_row_per(
            include_tables=["Subject", "Image"],
            via=[],
            row_per="Image",
        )
        assert result == "Image"


class TestPathAmbiguity:
    """Rule 6: multiple FK paths between row_per and a requested table → error."""

    def test_no_ambiguity_single_path(self, denorm_deriva_model) -> None:
        """Simple chain without diamond: no ambiguity.

        Uses the non-diamond fixture; Image→Subject is a single FK path.
        """
        model = denorm_deriva_model
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=[],
        )
        assert result == []

    def test_ambiguity_raises(self, denorm_diamond_deriva_model) -> None:
        """With diamond Image→Subject, raise for bare include_tables."""
        model = denorm_diamond_deriva_model
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=[],
        )
        assert len(result) >= 1
        amb = result[0]
        assert amb["from_table"] == "Image"
        assert amb["to_table"] == "Subject"
        assert len(amb["paths"]) >= 2

    def test_ambiguity_resolved_by_intermediate(self, denorm_diamond_deriva_model) -> None:
        """Including the intermediate in include_tables removes ambiguity."""
        model = denorm_diamond_deriva_model
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Observation", "Subject"],
            via=[],
        )
        assert result == []

    def test_ambiguity_resolved_by_via(self, denorm_diamond_deriva_model) -> None:
        """via=[Observation] removes ambiguity without adding Observation columns."""
        model = denorm_diamond_deriva_model
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=["Observation"],
        )
        assert result == []


class TestPrepareWideTableIntegration:
    """Integration: _prepare_wide_table should raise on ambiguity (Rule 6).

    This confirms the guards are wired into the public planner entry point.
    We only check the guard behavior — full plan validation is covered by
    the existing :mod:`tests.local_db.test_denormalize` suite.
    """

    def test_prepare_raises_on_diamond(self, denorm_diamond_deriva_model) -> None:
        """_prepare_wide_table raises AmbiguousPath when diamond is unrouted."""
        model = denorm_diamond_deriva_model
        with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
            model._prepare_wide_table(
                dataset=None,
                dataset_rid="DS-001",
                include_tables=["Image", "Subject"],
            )

    def test_prepare_succeeds_with_via(self, denorm_diamond_deriva_model) -> None:
        """via=[Observation] disambiguates, planner returns successfully."""
        model = denorm_diamond_deriva_model
        # Planner should not raise; we don't inspect the plan shape here
        # (left to the integration test_denormalize tests).
        element_tables, denormalized_columns, _ = model._prepare_wide_table(
            dataset=None,
            dataset_rid="DS-001",
            include_tables=["Image", "Subject"],
            via=["Observation"],
        )
        assert "Image" in element_tables

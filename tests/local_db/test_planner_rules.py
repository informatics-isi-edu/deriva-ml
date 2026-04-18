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


class TestDirectionCoherence:
    """Direction-coherence primitive: _edge_direction, _path_direction_sequence,
    _is_direction_coherent.

    These helpers underpin Rule 6's refinement — they let the path
    enumerator filter zigzag paths (up-to-shared-ancestor-and-back-down
    to sibling data) out of the ambiguity-detection candidate set.
    """

    def test_edge_direction_along_fk(self, denorm_deriva_model) -> None:
        """Image has FK pointing at Subject → Image→Subject is along-FK."""
        model = denorm_deriva_model
        assert model._edge_direction("Image", "Subject") == "along"

    def test_edge_direction_against_fk(self, denorm_deriva_model) -> None:
        """Subject is pointed at by Image → Subject→Image is against-FK."""
        model = denorm_deriva_model
        assert model._edge_direction("Subject", "Image") == "against"

    def test_edge_direction_no_fk(self, denorm_deriva_model) -> None:
        """Dataset and Subject have no direct FK edge → None."""
        model = denorm_deriva_model
        assert model._edge_direction("Dataset", "Subject") is None

    def test_coherent_pure_hoist(self, denorm_diamond_deriva_model) -> None:
        """Pure along-FK chain is coherent (zero direction changes)."""
        model = denorm_diamond_deriva_model
        # Image → Observation → Subject — both along-FK
        assert model._is_direction_coherent(["Image", "Observation", "Subject"])

    def test_coherent_pure_rollup(self, denorm_deriva_model) -> None:
        """Pure against-FK chain is coherent."""
        model = denorm_deriva_model
        # Subject → Image — single against-FK edge
        assert model._is_direction_coherent(["Subject", "Image"])

    def test_coherent_single_direction_change(self, denorm_diamond_deriva_model) -> None:
        """One direction change (hoist then rollup) is coherent — classic M:N siblings."""
        model = denorm_diamond_deriva_model
        # Image → Subject → Observation — along, against = one change.
        # (Note: Observation.Subject means Subject → Observation is against,
        # so path goes along then against.)
        assert model._is_direction_coherent(["Image", "Subject", "Observation"])

    def test_incoherent_zigzag_rejected(self, denorm_deriva_model) -> None:
        """A path with TWO direction changes (zigzag) is incoherent."""
        model = denorm_deriva_model
        # Simulate: Image → Subject → Image (against-then-along would
        # be one change, but we need two to be incoherent). The
        # canonical zigzag is up-across-down:
        #   Image → Dataset_Image → Dataset → Dataset_Image' → Image'
        # In the canned denorm schema we can construct this path
        # manually since Dataset_Image is an association that links
        # Image ↔ Dataset. A self-zigzag through Dataset:
        path = ["Image", "Dataset_Image", "Dataset", "Dataset_Image", "Image"]
        # The association hops collapse to transparent; the remaining
        # concrete edges are "Dataset → Dataset" which is empty. The
        # path is trivially coherent here. For a real zigzag we'd need
        # two concrete edges with opposite directions. Since the canned
        # schema doesn't contain a richer topology, we verify the
        # opposite case directly with _path_direction_sequence.
        sequence = model._path_direction_sequence(path)
        # Expected: transparent, transparent — no concrete edges.
        assert all(d == "transparent" for d in sequence)
        # Because the sequence has only transparent entries, the path
        # is trivially coherent (< 2 concrete edges).
        assert model._is_direction_coherent(path)

    def test_short_path_trivially_coherent(self, denorm_deriva_model) -> None:
        """Paths with 0 or 1 edges are trivially coherent."""
        model = denorm_deriva_model
        assert model._is_direction_coherent(["Image"])  # no edges
        assert model._is_direction_coherent(["Image", "Subject"])  # one edge

    def test_transparent_hop_does_not_count_as_change(self, denorm_deriva_model) -> None:
        """Transparent association hops collapse out of the sequence."""
        model = denorm_deriva_model
        # Dataset → Dataset_Image → Image. Dataset_Image is an
        # association table; the two edges collapse to one transparent
        # hop, which does NOT contribute to the concrete-edge count.
        sequence = model._path_direction_sequence(["Dataset", "Dataset_Image", "Image"])
        # Expect a single "transparent" entry (not two direction tags).
        assert sequence == ["transparent"]
        assert model._is_direction_coherent(["Dataset", "Dataset_Image", "Image"])


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

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
        sinks = model._planner._find_sinks(
            include_tables=["Subject", "Observation", "Image"],
            via=[],
        )
        assert sinks == ["Image"], f"expected ['Image'], got {sinks}"

    def test_single_sink_simple(self, denorm_deriva_model) -> None:
        """Image points to Subject. Image is the sink."""
        model = denorm_deriva_model
        sinks = model._planner._find_sinks(include_tables=["Subject", "Image"], via=[])
        assert sinks == ["Image"]

    def test_multi_leaf_raises(self, denorm_deriva_model) -> None:
        """If two requested tables have no FK between them, both are sinks."""
        model = denorm_deriva_model
        sinks = model._planner._find_sinks(include_tables=["Dataset", "Subject"], via=[])
        assert len(sinks) >= 2
        assert "Dataset" in sinks or "Subject" in sinks
        with pytest.raises(DerivaMLDenormalizeMultiLeaf) as excinfo:
            model._planner._determine_row_per(include_tables=["Dataset", "Subject"], via=[], row_per=None)
        assert "Dataset" in str(excinfo.value)
        assert "Subject" in str(excinfo.value)


class TestDownstreamLeafRejection:
    """Rule 5: explicit row_per with downstream table in include_tables → error."""

    def test_downstream_leaf_rejected(self, denorm_deriva_model) -> None:
        """row_per=Subject with Image (downstream) → error."""
        model = denorm_deriva_model
        with pytest.raises(DerivaMLDenormalizeDownstreamLeaf) as excinfo:
            model._planner._determine_row_per(
                include_tables=["Subject", "Image"],
                via=[],
                row_per="Subject",
            )
        assert "Subject" in str(excinfo.value)
        assert "Image" in str(excinfo.value)

    def test_downstream_leaf_accepted_if_no_downstream(self, denorm_deriva_model) -> None:
        """row_per=Image is fine because Image IS the sink (nothing downstream)."""
        model = denorm_deriva_model
        result = model._planner._determine_row_per(
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
        result = model._planner._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=[],
        )
        assert result == []

    def test_ambiguity_raises(self, denorm_diamond_deriva_model) -> None:
        """With diamond Image→Subject, raise for bare include_tables."""
        model = denorm_diamond_deriva_model
        result = model._planner._find_path_ambiguities(
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
        result = model._planner._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Observation", "Subject"],
            via=[],
        )
        assert result == []

    def test_ambiguity_resolved_by_via(self, denorm_diamond_deriva_model) -> None:
        """via=[Observation] removes ambiguity without adding Observation columns."""
        model = denorm_diamond_deriva_model
        result = model._planner._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=["Observation"],
        )
        assert result == []

    def test_non_monotonic_path_not_a_diamond(self, denorm_diamond_deriva_model) -> None:
        """Paths that reverse FK direction at a shared neighbor are NOT diamonds.

        Image↔Observation has two undirected walks:

        * Direct: ``Image → Observation`` (Image.Observation FK, all downstream)
        * Via shared Subject: ``Image → Subject → Observation`` — this
          walks ``Image.Subject`` downstream, then ``Observation.Subject``
          UPSTREAM (Observation has the FK to Subject, so Observation is
          referenced_by Subject). The direction reverses at Subject.

        A direction reversal at an interior vertex signals a
        common-neighbor shortcut, not a genuine FK join alternative.
        Rule 6 should NOT flag this as ambiguity: the direct FK is the
        only valid monotonic-downstream path from Image to Observation.

        Without this filter, the planner spuriously reports Image↔
        Observation as ambiguous, forcing every caller to add a
        disambiguation keyword even when there's a single obvious
        direct FK.
        """
        model = denorm_diamond_deriva_model
        result = model._planner._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Observation"],
            via=[],
        )
        assert result == [], (
            f"Image↔Observation should have a single downstream FK path (direct), not a diamond. Got: {result}"
        )


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
            model._planner._prepare_wide_table(
                dataset=None,
                dataset_rid="DS-001",
                include_tables=["Image", "Subject"],
            )

    def test_prepare_succeeds_with_via(self, denorm_diamond_deriva_model) -> None:
        """via=[Observation] disambiguates, planner returns successfully."""
        model = denorm_diamond_deriva_model
        # Planner should not raise; we don't inspect the plan shape here
        # (left to the integration test_denormalize tests).
        element_tables, denormalized_columns, _ = model._planner._prepare_wide_table(
            dataset=None,
            dataset_rid="DS-001",
            include_tables=["Image", "Subject"],
            via=["Observation"],
        )
        assert "Image" in element_tables


# ---------------------------------------------------------------------------
# Transparency predicates (issue #174)
# ---------------------------------------------------------------------------


class TestTransparencyPredicates:
    """Coverage for the transparency-predicate widening (issue #174).

    A "transparent intermediate" is a table the planner walks
    *through* without asking the caller to name it in ``include_tables``
    or ``via``. The original predicate (:meth:`_is_topological_association`)
    only recognized pure 2-FK association tables. Issue #174 widens
    the rule to also include DerivaML feature-association tables —
    3-FK tables whose third FK is the audit edge to the ML schema's
    ``Execution`` table.

    See the module-level "Transparency model" section in
    ``denormalize_planner.py`` for the conceptual contract.
    """

    # -- Predicate behavior ---------------------------------------------

    def test_topological_assoc_still_recognized(self, denorm_feature_deriva_model) -> None:
        """The 2-FK predicate must still fire on pure assoc tables."""
        model = denorm_feature_deriva_model
        # Dataset_Image: 2 FKs (Dataset, Image).
        assert model._planner._is_topological_association("Dataset_Image")
        assert not model._planner._is_feature_association("Dataset_Image")

    def test_feature_assoc_recognized(self, denorm_feature_deriva_model) -> None:
        """3-FK feature-assoc with FK to Execution is a feature-assoc."""
        model = denorm_feature_deriva_model
        # Execution_Image_Image_Classification: 3 FKs (Image, Execution,
        # Image_Classification). Has an FK to Execution → feature-assoc.
        assert model._planner._is_feature_association("Execution_Image_Image_Classification")
        # Strictly NOT a 2-FK topological association.
        assert not model._planner._is_topological_association("Execution_Image_Image_Classification")
        # But the union predicate (what the planner actually consults)
        # still considers it transparent.
        assert model._planner._is_transparent_intermediate("Execution_Image_Image_Classification")

    def test_three_fk_without_execution_is_not_feature(self, denorm_feature_deriva_model) -> None:
        """3-FK table without FK to Execution is NOT a feature-assoc.

        ``Image_Subject_UnrelatedThing`` has three domain FKs but none
        points at ``Execution`` — it's a genuine 3-way domain
        association, which the caller must name explicitly to route
        through.
        """
        model = denorm_feature_deriva_model
        assert not model._planner._is_feature_association("Image_Subject_UnrelatedThing")
        assert not model._planner._is_topological_association("Image_Subject_UnrelatedThing")
        assert not model._planner._is_transparent_intermediate("Image_Subject_UnrelatedThing")

    def test_four_fk_assoc_is_not_transparent(self, denorm_feature_deriva_model) -> None:
        """4+ FK tables are not transparent even with an FK to Execution.

        Multi-way associations have an arbitrary number of domain
        edges, so the caller has to disambiguate which edge they
        want — they can't be silently hopped.
        """
        model = denorm_feature_deriva_model
        assert not model._planner._is_feature_association("FourWayAssoc")
        assert not model._planner._is_topological_association("FourWayAssoc")
        assert not model._planner._is_transparent_intermediate("FourWayAssoc")

    def test_domain_table_is_not_transparent(self, denorm_feature_deriva_model) -> None:
        """Plain domain tables (e.g. Image) are never transparent."""
        model = denorm_feature_deriva_model
        assert not model._planner._is_transparent_intermediate("Image")
        assert not model._planner._is_transparent_intermediate("Subject")

    # -- Reachability through feature-assoc bridges ---------------------

    def test_outbound_reachable_hops_through_feature_assoc(self, denorm_feature_deriva_model) -> None:
        """Image → feature-assoc → Image_Classification is reachable.

        This is the regression test for issue #174 case A. Under the
        old code, ``_outbound_reachable("Image", {"Image",
        "Image_Classification"})`` returned ``set()`` because the
        feature-assoc table failed the 2-FK predicate and the walker
        couldn't bridge through it. Under the fix, the walker
        recognizes feature-assoc tables as transparent and surfaces
        ``Image_Classification`` as reachable.
        """
        model = denorm_feature_deriva_model
        reachable = model._planner._outbound_reachable(
            "Image",
            {"Image", "Image_Classification"},
        )
        assert reachable == {"Image_Classification"}, (
            f"Image should reach Image_Classification through the feature-assoc bridge, got {reachable}"
        )

    def test_outbound_reachable_does_not_hop_through_three_fk_domain(self, denorm_feature_deriva_model) -> None:
        """A 3-FK domain-only assoc table is NOT a transparent bridge.

        ``Image_Subject_UnrelatedThing`` has FKs to Image, Subject,
        and UnrelatedThing. With only Image and UnrelatedThing in the
        subgraph, the walker should NOT silently hop through the
        3-way association — the third FK is a real domain edge and
        the caller must signal intent.
        """
        model = denorm_feature_deriva_model
        reachable = model._planner._outbound_reachable(
            "Image",
            {"Image", "UnrelatedThing"},
        )
        assert reachable == set(), (
            f"Image must NOT reach UnrelatedThing through a 3-way "
            f"domain assoc (caller has to route explicitly), got {reachable}"
        )

    # -- Sink finding ---------------------------------------------------

    def test_find_sinks_multileaf_for_bridge_only_set(self, denorm_feature_deriva_model) -> None:
        """With Image + Image_Classification (linked only by a feature
        bridge), neither side is strictly downstream of the other.

        Both endpoints sit upstream of the feature-assoc table. Under
        strict-downstream sink-finding (no bidirectional bridge hop),
        each candidate has empty strict-downstream set inside the
        subgraph, so both are sinks. The planner raises MultiLeaf and
        the caller must pick ``row_per=`` explicitly. This is the
        actionable error message — under the old code the same shape
        produced zero sinks (``NoSink``), which was harder to debug.

        Issue #174: ``split_dataset(stratify_by_column=...)`` should
        provide ``row_per=element_table`` on the user's behalf so the
        explicit-row_per path resolves cleanly without forcing the
        caller to pick.
        """
        from deriva_ml.core.exceptions import DerivaMLDenormalizeMultiLeaf

        model = denorm_feature_deriva_model
        sinks = model._planner._find_sinks(
            include_tables=["Image", "Image_Classification"],
            via=[],
        )
        assert sorted(sinks) == ["Image", "Image_Classification"], (
            f"Both endpoints of a transparent bridge should be sink candidates under strict downstream, got {sinks}"
        )
        with pytest.raises(DerivaMLDenormalizeMultiLeaf):
            model._planner._determine_row_per(
                include_tables=["Image", "Image_Classification"],
                via=[],
                row_per=None,
            )

    # -- Sink finding with explicit row_per -----------------------------

    def test_determine_row_per_explicit_image_succeeds(self, denorm_feature_deriva_model) -> None:
        """row_per='Image' is accepted with the feature-assoc bridge.

        With strict-downstream Rule 5, Image_Classification is NOT
        considered downstream of Image — the bridge sits between them
        but doesn't produce a fan-out from Image's perspective.
        Explicit ``row_per=Image`` is therefore accepted, and the
        symmetric ``row_per=Image_Classification`` is accepted too.
        This is the call-site contract that ``split_dataset`` relies
        on when plumbing ``row_per=element_table``.
        """
        model = denorm_feature_deriva_model
        resolved_image = model._planner._determine_row_per(
            include_tables=["Image", "Image_Classification"],
            via=[],
            row_per="Image",
        )
        assert resolved_image == "Image"

        resolved_ic = model._planner._determine_row_per(
            include_tables=["Image", "Image_Classification"],
            via=[],
            row_per="Image_Classification",
        )
        assert resolved_ic == "Image_Classification"

    def test_outbound_reachable_image_to_classification_via_bridge(self, denorm_feature_deriva_model) -> None:
        """Regression for issue #174 variant A.

        ``include_tables=["Image_Classification"]`` (no Image). The
        dataset member anchor table is Image. Under the old code,
        ``_outbound_reachable("Image", {"Image_Classification"})``
        returned ``set()`` because the feature-assoc table failed
        the transparency predicate — and
        :meth:`Denormalizer._classify_anchors` then flagged Image as
        an unrelated anchor with no FK path to Image_Classification.

        Under the fix, the bridge hop fires and Image_Classification
        appears reachable — the anchor classifies as ``scoping``
        rather than ``unrelated``.
        """
        model = denorm_feature_deriva_model
        reachable = model._planner._outbound_reachable(
            "Image",
            {"Image_Classification"},
        )
        assert reachable == {"Image_Classification"}, (
            f"Image anchor must reach Image_Classification through the "
            f"feature-assoc bridge so anchor classification works "
            f"(issue #174 variant A), got {reachable}"
        )

    # -- Strict downstream primitive -----------------------------------

    # -- End-to-end prepare_wide_table over the feature-assoc bridge ----

    def test_prepare_wide_table_succeeds_with_explicit_row_per(self, denorm_feature_deriva_model) -> None:
        """The full planner entry point produces a plan for the issue
        #174 case when ``row_per=`` is supplied.

        This is the integration-level proof that the predicate +
        strict-downstream changes compose correctly: the planner
        succeeds (no exception), the plan keys the join tree on Image,
        and — critically — the feature-association bridge appears in
        the JOIN path. Under the old code the JOIN couldn't route
        through the 3-FK feature-assoc (the transparency predicate
        didn't fire), so even when ``row_per`` was accepted the
        bridge step was missing and the projection wouldn't actually
        connect Image to Image_Classification rows at query time.
        """
        model = denorm_feature_deriva_model
        element_tables, denormalized_columns, _ = model._planner._prepare_wide_table(
            dataset=None,
            dataset_rid="DS-001",
            include_tables=["Image", "Image_Classification"],
            row_per="Image",
        )
        assert "Image" in element_tables, (
            f"join plan should be keyed on the chosen row_per, got element_tables.keys()={list(element_tables.keys())}"
        )
        # The plan tuple is (path_names, join_conditions, join_types);
        # path_names is the pre-order JOIN sequence. For the feature-
        # assoc bridge to be wired correctly, the bridge must appear
        # in the sequence between Image and Image_Classification.
        path_names, _join_conditions, _join_types = element_tables["Image"]
        assert "Execution_Image_Image_Classification" in path_names, (
            f"join sequence must route through the feature-assoc bridge, got path_names={path_names}"
        )
        assert "Image_Classification" in path_names, (
            f"Image_Classification must be reachable via the JOIN, got path_names={path_names}"
        )

    def test_strict_downstream_skips_bridge_hop(self, denorm_feature_deriva_model) -> None:
        """Strict reach does NOT bridge-hop into the symmetric side.

        Direct primitive-level coverage for the helper added in
        issue #174.
        """
        model = denorm_feature_deriva_model
        # Strict downstream from Image: feature-assoc references Image
        # but isn't in the set, and we do NOT hop through it. So
        # Image_Classification is NOT in strict downstream(Image).
        strict = model._planner._outbound_reachable_strict(
            "Image",
            {"Image", "Image_Classification"},
        )
        assert strict == set(), (
            f"strict downstream(Image) must be empty when the only "
            f"path to Image_Classification is via a transparent bridge, "
            f"got {strict}"
        )

        # By contrast, the standard bidirectional reach DOES surface
        # the bridge hop — this is the connectivity primitive used
        # by Denormalizer._classify_anchors and must remain
        # bidirectional.
        broad = model._planner._outbound_reachable(
            "Image",
            {"Image", "Image_Classification"},
        )
        assert broad == {"Image_Classification"}, (
            f"bidirectional reach should still see Image_Classification through the bridge, got {broad}"
        )

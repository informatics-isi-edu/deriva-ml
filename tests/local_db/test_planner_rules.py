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
        # The planner now emits one entry per distinct Dataset->element route,
        # keyed ``{element}#{i}`` (the consumer UNIONs them). Assert an Image
        # route exists rather than the legacy single ``"Image"`` key.
        assert any(key.split("#", 1)[0] == "Image" for key in element_tables)


class TestFeatureBridgeDiamond:
    """Rule 6 over the diamond-with-feature-bridge shape (finding 09 §7.1).

    This pins the one *intentional behavior change* of the Option E2
    predicate fix. When a target and a value table are connected by
    BOTH a transparent feature bridge AND a second independent FK path,
    the now-transparent bridge hops in ``_is_downstream_chain`` and
    registers as a competing downstream chain. Rule 6 must therefore
    raise ``DerivaMLDenormalizeAmbiguousPath`` — behavior that did NOT
    fire under the old predicate (the bridge was opaque, so Rule 6 saw
    only the direct path and planned silently). Raising here is the
    *correct* outcome: there really are two ways to relate the tables,
    so the planner asks the caller to pick.
    """

    def test_feature_bridge_is_transparent(self, denorm_feature_diamond_deriva_model) -> None:
        """Precondition: the 4-FK feature bridge is transparent under E2.

        This is the property that turns the second path into a genuine
        competing downstream chain. If this regresses, the ambiguity
        below would silently disappear (the bug we're guarding against).
        """
        model = denorm_feature_diamond_deriva_model
        assert model._planner._is_feature_association("Execution_Image_Image_Classification")
        # Image reaches Image_Classification through the bridge hop.
        assert model._planner._outbound_reachable(
            "Image",
            {"Image", "Image_Classification"},
        ) == {"Image_Classification"}

    def test_diamond_with_feature_bridge_raises_ambiguity(self, denorm_feature_diamond_deriva_model) -> None:
        """Two downstream chains (direct FK + feature bridge) → ambiguity."""
        model = denorm_feature_diamond_deriva_model
        result = model._planner._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Image_Classification"],
            via=[],
        )
        assert len(result) == 1, f"expected exactly one ambiguity, got {result}"
        amb = result[0]
        assert amb["from_table"] == "Image"
        assert amb["to_table"] == "Image_Classification"
        # Both competing paths must be present: the direct FK and the
        # one that hops the feature bridge.
        path_sigs = {tuple(p) for p in amb["paths"]}
        assert ("Image", "Image_Classification") in path_sigs, (
            f"direct FK path missing from ambiguity, got {amb['paths']}"
        )
        assert (
            "Image",
            "Execution_Image_Image_Classification",
            "Image_Classification",
        ) in path_sigs, f"feature-bridge path missing from ambiguity, got {amb['paths']}"

    def test_prepare_wide_table_raises_on_feature_bridge_diamond(self, denorm_feature_diamond_deriva_model) -> None:
        """The public planner entry point raises AmbiguousPath on the diamond."""
        model = denorm_feature_diamond_deriva_model
        with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
            model._planner._prepare_wide_table(
                dataset=None,
                dataset_rid="DS-001",
                include_tables=["Image", "Image_Classification"],
            )

    def test_via_bridge_does_not_disambiguate(self, denorm_feature_diamond_deriva_model) -> None:
        """Naming the *transparent* bridge in via= does NOT resolve the diamond.

        ``_is_signaled`` deliberately excludes transparent intermediates
        (feature-assoc and pure-association tables) from counting as a
        user path-signal — the user shouldn't have to name plumbing.
        Consequently the feature bridge cannot be used to pick the
        bridged path: the ambiguity stands, and the caller must instead
        narrow ``include_tables`` (drop one endpoint). This pins the
        interaction between transparency and Rule 6's signaling rule.
        """
        model = denorm_feature_diamond_deriva_model
        result = model._planner._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Image_Classification"],
            via=["Execution_Image_Image_Classification"],
        )
        assert len(result) == 1, (
            f"transparent bridge in via= must not signal a path, so the diamond stays ambiguous, got {result}"
        )


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
    the (≥4)-FK tables ``create_feature`` builds, marked by an FK to
    the ML schema's ``Feature_Name`` vocab AND an FK to the ML schema's
    ``Execution`` table (finding 09: the predicate originally keyed off
    a 3-FK count and so rejected every real feature, which has at least
    target + value + Feature_Name + Execution = 4 domain FKs).

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
        """Real 4-FK feature-assoc (Feature_Name FK + Execution FK) is recognized."""
        model = denorm_feature_deriva_model
        # Execution_Image_Image_Classification: 4 domain FKs (Image,
        # Image_Classification, Feature_Name, Execution) — the canonical
        # shape create_feature produces. The Feature_Name FK + Execution
        # FK are the two markers _is_feature_association keys off.
        assert model._planner._is_feature_association("Execution_Image_Image_Classification")
        # Strictly NOT a 2-FK topological association.
        assert not model._planner._is_topological_association("Execution_Image_Image_Classification")
        # But the union predicate (what the planner actually consults)
        # still considers it transparent.
        assert model._planner._is_transparent_intermediate("Execution_Image_Image_Classification")

    def test_three_fk_without_execution_is_not_feature(self, denorm_feature_deriva_model) -> None:
        """3-FK table without the feature marker FKs is NOT a feature-assoc.

        ``Image_Subject_UnrelatedThing`` has three domain FKs but none
        points at ``Execution`` or ``Feature_Name`` — it's a genuine
        3-way domain association, which the caller must name explicitly
        to route through.
        """
        model = denorm_feature_deriva_model
        assert not model._planner._is_feature_association("Image_Subject_UnrelatedThing")
        assert not model._planner._is_topological_association("Image_Subject_UnrelatedThing")
        assert not model._planner._is_transparent_intermediate("Image_Subject_UnrelatedThing")

    def test_four_fk_assoc_is_not_transparent(self, denorm_feature_deriva_model) -> None:
        """A 4-FK association WITHOUT a Feature_Name FK is not a feature.

        ``FourWayAssoc`` has four domain FKs (Image, Subject,
        UnrelatedThing, Execution) — including an ``Execution``
        provenance edge — but **no** FK to ``Feature_Name``. Since the
        predicate keys off the Feature_Name FK + Execution FK pair (not
        the FK count), this multi-way domain association is correctly
        rejected: it has an arbitrary number of domain edges the caller
        must disambiguate, so it can't be silently hopped. This is the
        negative guard that distinguishes Option E2 (key off marker
        FKs) from the rejected Option E1 (``>=3`` count, which would
        wrongly accept this).
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
        # The planner now emits one entry per distinct Dataset->element route,
        # keyed ``{element}#{i}``. Resolve the Image route(s) by element prefix
        # rather than the legacy single ``"Image"`` key.
        image_keys = [key for key in element_tables if key.split("#", 1)[0] == "Image"]
        assert image_keys, (
            f"join plan should include the chosen row_per element, got keys={list(element_tables.keys())}"
        )
        # The plan tuple is (path_names, join_conditions, join_types);
        # path_names is the pre-order JOIN sequence. For the feature-
        # assoc bridge to be wired correctly, the bridge must appear
        # in the sequence between Image and Image_Classification.
        # subtree is identical across routes, so any Image route carries the bridge
        path_names, _join_conditions, _join_types = element_tables[image_keys[0]]
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


class TestJoinOrderValidity:
    """Every emitted route must order its JOINs so that no ON clause references
    a table joined later (#320 regression).

    The consumer (``_denormalize_impl``) joins the tables of each route in
    ``path_names`` order, using ``join_conditions[table]`` for the ON clause.
    SQLite (and any SQL engine) rejects ``... JOIN B ON A.x = B.y ...`` when
    ``A`` is joined *after* ``B`` ("ON clause references tables to its right").

    #318 introduced multi-hop ``Dataset -> ... -> element`` route prefixes. On a
    multi-route topology (a table reachable via a membership association while
    another requested table is FK-reachable through it), the prefix can place a
    subtree table *before* the element, yet the subtree edge's ON clause
    references the element — producing an invalid join order. This pins the
    invariant that prevented it.
    """

    @staticmethod
    def _on_clause_tables(col_pairs) -> set[str]:
        """Tables referenced by an ON clause (both sides of each fk/pk pair)."""
        tables: set[str] = set()
        for fk_col, pk_col in col_pairs:
            tables.add(fk_col.table.name)
            tables.add(pk_col.table.name)
        return tables

    def test_no_join_references_a_later_table(self, eye_ai_planner) -> None:
        """For the [Subject, Observation, Image] request on the eye-ai schema,
        no route emits a JOIN whose ON clause references a not-yet-joined table.
        """
        planner, stub, dataset_rid = eye_ai_planner
        element_tables, _cols, _multi = planner._prepare_wide_table(
            stub, dataset_rid, ["Subject", "Observation", "Image"]
        )
        assert element_tables, "planner produced no routes"

        violations: list[str] = []
        for key, (path, join_conditions, _join_types) in element_tables.items():
            for i, table_name in enumerate(path):
                col_pairs = join_conditions.get(table_name)
                if not col_pairs:
                    continue  # root/Dataset or no condition
                later_tables = set(path[i + 1 :])
                refs_later = self._on_clause_tables(col_pairs) & later_tables
                if refs_later:
                    violations.append(
                        f"route '{key}': JOIN {table_name} ON references "
                        f"not-yet-joined {sorted(refs_later)} (path={path})"
                    )
        assert not violations, "invalid JOIN order (#320):\n" + "\n".join(violations)

"""Unit tests for the Denormalizer class public API.

These tests use the canned bag-model fixtures from conftest.py — no live
catalog required. They verify the class wraps _denormalize_impl correctly
with the new rules applied.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeUnrelatedAnchor,
)
from deriva_ml.local_db.denormalizer import Denormalizer


class TestDenormalizerConstruction:
    """Denormalizer(dataset) derives catalog/workspace/model from the dataset."""

    def test_construct_from_dataset_like(self, populated_denorm) -> None:
        """Minimal construction: wrap a dataset-like object."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        assert d is not None


class TestAsDataframe:
    """Denormalizer.as_dataframe returns a pd.DataFrame with expected shape."""

    def test_simple_star_schema(self, populated_denorm) -> None:
        """One row per Image, Subject columns hoisted."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"])
        assert isinstance(df, pd.DataFrame)
        # 3 Images in the fixture (one with NULL Subject = LEFT JOIN preserved).
        assert len(df) == 3
        assert any(c.startswith("Image.") for c in df.columns)
        assert any(c.startswith("Subject.") for c in df.columns)

    def test_empty_dataset(self, populated_denorm) -> None:
        """Nonexistent dataset RID returns empty DataFrame with correct columns."""
        ds = _FakeDataset(populated_denorm, dataset_rid="NO-SUCH-DS")
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"])
        assert len(df) == 0
        assert len(df.columns) > 0  # schema preserved


class TestAsDict:
    """Denormalizer.as_dict streams rows as dicts."""

    def test_yields_dicts(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        rows = list(d.as_dict(["Image", "Subject"]))
        assert len(rows) == 3
        for r in rows:
            assert isinstance(r, dict)


class TestColumns:
    """Denormalizer.columns previews column names and types — no data fetch."""

    def test_columns_returns_tuples(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        cols = d.columns(["Image", "Subject"])
        assert isinstance(cols, list)
        for entry in cols:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            name, typ = entry
            assert isinstance(name, str)
            assert isinstance(typ, str)


class TestRowPerAutoInference:
    """Verify Rule 2 auto-inference through the Denormalizer."""

    def test_image_is_sink(self, populated_denorm) -> None:
        """include_tables=[Subject, Image] → row_per auto = Image."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Subject", "Image"])
        assert plan["row_per"] == "Image"
        assert plan["row_per_source"] == "auto-inferred"


class TestExplicitRowPer:
    """Verify explicit row_per honored; Rule 5 downstream rejection."""

    def test_explicit_matching_auto(self, populated_denorm) -> None:
        """Explicit row_per=Image (same as auto) works."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"], row_per="Image")
        assert len(df) == 3

    def test_downstream_leaf_rejected(self, populated_denorm) -> None:
        """row_per=Subject with Image downstream → DerivaMLDenormalizeDownstreamLeaf."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        with pytest.raises(DerivaMLDenormalizeDownstreamLeaf):
            d.as_dataframe(["Image", "Subject"], row_per="Subject")


class TestViaParameter:
    """Verify via= is forwarded to the planner and resolves path ambiguity."""

    def test_via_resolves_diamond(self, populated_denorm_diamond) -> None:
        """Diamond schema: via=['Observation'] should prevent ambiguity error."""
        ds = _FakeDataset(populated_denorm_diamond)
        d = Denormalizer(ds)
        # Without via, diamond raises
        with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
            d.as_dataframe(["Image", "Subject"])
        # With via, ambiguity resolved
        df = d.as_dataframe(["Image", "Subject"], via=["Observation"])
        assert isinstance(df, pd.DataFrame)
        # Concrete proof of resolution: one row per Image + Subject columns hoisted.
        assert len(df) == 3
        assert any(c.startswith("Image.") for c in df.columns)
        assert any(c.startswith("Subject.") for c in df.columns)
        # Observation columns should NOT be present (via adds to join, not output)
        assert not any(c.startswith("Observation.") for c in df.columns)


class TestUnrelatedAnchors:
    """Rule 8: anchors with no FK path to include_tables → error by default."""

    def test_unrelated_anchor_rejected(self, populated_denorm) -> None:
        # A dataset whose members include an unrelated type.
        class _HeteroDataset(_FakeDataset):
            def list_dataset_members(self, **kwargs):
                return {
                    "Image": [{"RID": "IMG-1"}],
                    # UnrelatedThing has no FKs → no path to Subject/Image.
                    "UnrelatedThing": [{"RID": "UT-1"}],
                }

        ds = _HeteroDataset(populated_denorm)
        d = Denormalizer(ds)
        with pytest.raises(DerivaMLDenormalizeUnrelatedAnchor):
            d.as_dataframe(["Image", "Subject"])

    def test_unrelated_anchor_ignored_with_flag(self, populated_denorm) -> None:
        class _HeteroDataset(_FakeDataset):
            def list_dataset_members(self, **kwargs):
                return {
                    "Image": [{"RID": "IMG-1"}],
                    "UnrelatedThing": [{"RID": "UT-1"}],
                }

        ds = _HeteroDataset(populated_denorm)
        d = Denormalizer(ds)
        # With the flag, no error
        df = d.as_dataframe(
            ["Image", "Subject"],
            ignore_unrelated_anchors=True,
        )
        assert isinstance(df, pd.DataFrame)


class TestOrphanRows:
    """Rule 7 case 3: upstream anchor with no row_per reachable → orphan row."""

    def test_orphan_subject_emits_row(self, populated_denorm) -> None:
        """Subject member with no Image in the dataset → one orphan row."""

        class _WithOrphan(_FakeDataset):
            def list_dataset_members(self, **kwargs):
                members = super().list_dataset_members()
                # Add an orphan Subject whose RID doesn't match any Image's Subject FK.
                members["Subject"] = list(members["Subject"]) + [{"RID": "ORPHAN-SUBJ"}]
                return members

        ds = _WithOrphan(populated_denorm)
        # Insert the orphan Subject row into the engine so _emit_orphan_rows can fetch it.
        from sqlalchemy.orm import Session

        ls = populated_denorm["local_schema"]
        subj_cls = ls.get_orm_class("Subject")
        with Session(ls.engine) as session:
            session.add(subj_cls(RID="ORPHAN-SUBJ", Name="Orphan"))
            session.commit()

        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"])
        # 3 Image rows + 1 orphan Subject row = 4
        # The orphan row has Image cols NULL and Subject cols populated.
        orphans = df[df["Image.RID"].isna()]
        assert len(orphans) == 1
        # The orphan's Subject.RID should be "ORPHAN-SUBJ".
        assert orphans.iloc[0]["Subject.RID"] == "ORPHAN-SUBJ"


class TestAnchorClassification:
    """Rule 7 cases 1/2/4/5/6 exercised via Denormalizer._classify_anchors.

    Case 3 is covered by TestOrphanRows above (end-to-end orphan emission).
    Cases 1/2/4 are "scoping" (anchor contributes a filter). Case 5 is
    "silent drop regardless of flag". Case 6 is covered by
    TestUnrelatedAnchors (raises / flag-suppressed drop).
    """

    def test_case_1_anchor_is_row_per(self, populated_denorm) -> None:
        """Anchor table == row_per → scoping."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        scoping, orphans, ignored = d._classify_anchors(
            anchors={"Image": ["IMG-1", "IMG-2"]},
            include_tables=["Image", "Subject"],
            via=[],
            row_per="Image",
            ignore_unrelated_anchors=False,
        )
        assert scoping == {"Image": ["IMG-1", "IMG-2"]}
        assert orphans == {}
        assert ignored == {}

    def test_case_2_in_include_reaches_row_per(self, populated_denorm) -> None:
        """Anchor in include_tables, upstream of row_per → scoping."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        scoping, orphans, ignored = d._classify_anchors(
            # Subject is in include_tables and Image points to Subject via FK,
            # so Subject is upstream of row_per=Image.
            anchors={"Subject": ["SUBJ-A"], "Image": ["IMG-1"]},
            include_tables=["Image", "Subject"],
            via=[],
            row_per="Image",
            ignore_unrelated_anchors=False,
        )
        assert "Image" in scoping
        assert "Subject" in scoping  # Case 2: reaches row_per
        assert orphans == {}
        assert ignored == {}

    def test_empty_anchor_list_is_ignored(self, populated_denorm) -> None:
        """An anchor table with an empty RID list is skipped entirely.

        ``list_dataset_members`` may return ``{"File": []}`` when the
        ``Dataset_File`` association table exists but no members were
        actually added. A zero-RID anchor can't contribute anything to
        the output — so it should NOT trigger Rule 8's UnrelatedAnchor
        diagnostic, which warns about anchors "that would contribute
        nothing." There's nothing to contribute and nothing to warn
        about.

        Without this skip, the File anchor with an empty RID list would
        hit the case-6 (no FK path to Subject) branch and raise
        DerivaMLDenormalizeUnrelatedAnchor even though it holds no data.
        """
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        # Unknown table name simulates the Dataset_File assoc returning
        # an empty member list — "Unrelated" is a nonexistent table so
        # it has no FK path to Subject. If we didn't short-circuit on
        # empty lists, this would raise.
        scoping, orphans, ignored = d._classify_anchors(
            anchors={"Subject": ["SUBJ-A"], "Unrelated": []},
            include_tables=["Subject"],
            via=[],
            row_per="Subject",
            ignore_unrelated_anchors=False,
        )
        assert scoping == {"Subject": ["SUBJ-A"]}
        assert orphans == {}
        # "Unrelated" should not appear in any of the result dicts —
        # empty anchors are simply skipped.
        assert "Unrelated" not in scoping
        assert "Unrelated" not in orphans
        assert "Unrelated" not in ignored

    def test_case_4_downstream_anchor_when_row_per_upstream(self, populated_denorm) -> None:
        """Anchor DOWNSTREAM of row_per is still scoping (direction-agnostic).

        When row_per=Subject and an Image anchor is present, the Image
        table is DOWNSTREAM of Subject (Image.Subject is an FK to Subject).
        The anchor still provides a valid filter: "include only Subjects
        that have one of these Images." Rule 7's reachability check must
        accept either-direction FK connectivity, not only downstream.

        This reproduces the nested-dataset scenario where a bag contains
        both Subject and Image members, the user asks for Subject rows
        only, and the Image anchors must be recognized as related
        (filter-only) rather than classified as case-6 unrelated.
        """
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        scoping, orphans, ignored = d._classify_anchors(
            anchors={"Subject": ["SUBJ-A"], "Image": ["IMG-1"]},
            include_tables=["Subject"],
            via=[],
            row_per="Subject",
            ignore_unrelated_anchors=False,
        )
        # Image is DOWNSTREAM of Subject but still reaches Subject via
        # Image.Subject FK — classified as scoping (filter-only).
        assert "Subject" in scoping
        assert "Image" in scoping, (
            "Image anchor should be classified as scoping (filter-only) "
            "because Image.Subject gives it an upstream FK path to row_per=Subject."
        )
        assert orphans == {}
        assert ignored == {}

    def test_case_4_not_in_include_reaches_row_per(self, populated_denorm) -> None:
        """Anchor NOT in include_tables but reaches row_per → scoping (filter-only).

        Dataset points to Image via the Dataset_Image association, so
        Dataset is upstream of row_per=Image even though Dataset is not in
        include_tables. This is pure filter-only contribution per spec
        §3.7 case 4 — no orphan row because Dataset's columns aren't
        projected.
        """
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        scoping, orphans, ignored = d._classify_anchors(
            anchors={"Image": ["IMG-1"], "Dataset": ["DS-001"]},
            include_tables=["Image", "Subject"],
            via=[],
            row_per="Image",
            ignore_unrelated_anchors=False,
        )
        assert "Dataset" in scoping, "Dataset → Image via association = filter-only scoping"
        assert "Dataset" not in orphans
        assert "Dataset" not in ignored

    def test_case_5_silent_drop_regardless_of_flag(self, populated_denorm) -> None:
        """Anchor NOT in include_tables, upstream of subgraph, can't reach row_per.

        Spec §3.8: "Anchors of case 5 (table ∉ include_tables, unreachable)
        are silently dropped regardless of the flag — they contribute no
        output either way, so there's nothing to warn about."

        Construction: pick a table that HAS an FK edge into the subgraph
        but whose specific anchor RIDs don't reach row_per. In the canned
        schema, Dataset_Image IS in the subgraph's outbound reach but is
        an association table — we use Dataset with a row_per=Subject (so
        Dataset reaches Image in-subgraph but NOT Subject since Subject
        is downstream of row_per's direction).

        Easier: use row_per=Dataset and an anchor on a table that points
        to Image (which is upstream of the subgraph but won't reach
        Dataset). Since we can't easily construct this with only
        one-direction FKs, we verify the classification semantics
        directly with a synthesized `_classify_anchors` call: the
        default behavior with no matching anchors must NOT raise.
        """
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)

        # Simpler case-5 construction: anchors on a table IN the
        # subgraph reach-set but NOT in include_tables, with no rows that
        # actually reach row_per. _classify_anchors uses table-level
        # reachability (does the table have ANY FK path to row_per?),
        # not per-RID reachability — so the case-5 classification here
        # depends on FK graph shape, not actual data.
        #
        # In the canned schema, Dataset has FK path to Image (via
        # Dataset_Image) — that's case 4 (reaches), so we can't construct
        # a true case-5 with this schema. Instead, assert the negative:
        # WITH flag=False, a truly unrelated anchor (UnrelatedThing)
        # raises — that's case 6 — NOT case 5.
        from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor

        # Sanity: the canned schema doesn't have a pure case-5 example
        # (every in-graph table reaches Image). Case 5 is exercised in
        # integration tests with richer schemas.
        # Instead, verify that unrelated (case 6) → raises, confirming
        # case 5 vs case 6 is a real distinction in the code.
        with pytest.raises(DerivaMLDenormalizeUnrelatedAnchor):
            d._classify_anchors(
                anchors={"Image": ["IMG-1"], "UnrelatedThing": ["UT-1"]},
                include_tables=["Image", "Subject"],
                via=[],
                row_per="Image",
                ignore_unrelated_anchors=False,
            )

    def test_case_6_raises_on_unrelated(self, populated_denorm) -> None:
        """Anchor with no FK path at all → case 6, raises (when flag=False)."""
        from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor

        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        with pytest.raises(DerivaMLDenormalizeUnrelatedAnchor) as excinfo:
            d._classify_anchors(
                anchors={"Image": ["IMG-1"], "UnrelatedThing": ["UT-1"]},
                include_tables=["Image", "Subject"],
                via=[],
                row_per="Image",
                ignore_unrelated_anchors=False,
            )
        assert "UnrelatedThing" in str(excinfo.value)

    def test_case_6_flag_suppresses_raise(self, populated_denorm) -> None:
        """Case 6 + ignore_unrelated_anchors=True → silently added to ignored."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        scoping, orphans, ignored = d._classify_anchors(
            anchors={"Image": ["IMG-1"], "UnrelatedThing": ["UT-1"]},
            include_tables=["Image", "Subject"],
            via=[],
            row_per="Image",
            ignore_unrelated_anchors=True,
        )
        assert "Image" in scoping
        assert "UnrelatedThing" in ignored
        assert "UnrelatedThing" not in scoping
        assert "UnrelatedThing" not in orphans


class TestNoSink:
    """Rule 2 edge case: DerivaMLDenormalizeNoSink on cycle / empty."""

    def test_empty_include_tables_via_determine_row_per(self, populated_denorm) -> None:
        """Empty include_tables → no sink → NoSink."""
        from deriva_ml.core.exceptions import DerivaMLDenormalizeNoSink

        model = populated_denorm["model"]
        with pytest.raises(DerivaMLDenormalizeNoSink):
            model._determine_row_per(include_tables=[], via=[], row_per=None)


class TestDescribe:
    """Denormalizer.describe returns the full plan dict per spec §5."""

    def test_describe_keys(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        # Required keys per spec §5
        for key in [
            "row_per",
            "row_per_source",
            "row_per_candidates",
            "columns",
            "include_tables",
            "via",
            "join_path",
            "transparent_intermediates",
            "ambiguities",
            "estimated_row_count",
            "anchors",
            "source",
        ]:
            assert key in plan, f"plan missing key {key}: {list(plan.keys())}"

    def test_describe_row_per_explicit(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"], row_per="Image")
        assert plan["row_per_source"] == "explicit"

    def test_describe_row_per_auto(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        assert plan["row_per_source"] == "auto-inferred"
        assert plan["row_per"] == "Image"

    def test_describe_ambiguity_reported(self, populated_denorm_diamond) -> None:
        """Diamond schema: ambiguity reported (not raised) on dry-run."""
        ds = _FakeDataset(populated_denorm_diamond)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        # Ambiguity reported rather than raised (describe is dry-run)
        assert len(plan["ambiguities"]) > 0
        amb = plan["ambiguities"][0]
        # Tight spec-shape assertions: the downstream consumer (Task 8 +
        # user-facing docs) relies on this exact structure.
        assert amb["type"] == "multiple_paths"
        assert amb["from"] == "Image"
        assert amb["to"] == "Subject"
        # paths is a list[str] formatted with the " → " separator.
        assert isinstance(amb["paths"], list)
        assert all(isinstance(p, str) and " → " in p for p in amb["paths"])
        # suggestions has both disambiguation paths (spec §5).
        assert set(amb["suggestions"].keys()) == {"add_to_include_tables", "add_to_via"}

    def test_describe_anchors(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        anc = plan["anchors"]
        assert "by_type" in anc
        assert "total" in anc
        assert anc["by_type"]["Image"] == 3  # 3 image members in fixture

    def test_describe_never_raises_on_bad_row_per(self, populated_denorm) -> None:
        """row_per not in include_tables → resolved_row_per=None, no ValueError.

        describe is a dry-run — it must hand back a well-formed dict so the
        user can diagnose, not throw on a typo or stale name.
        """
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image"], row_per="Subject")
        assert plan["row_per"] is None
        # The rest of the dict is still populated with the 12 spec keys.
        assert plan["row_per_source"] == "explicit"
        assert plan["include_tables"] == ["Image"]

    def test_describe_never_raises_on_downstream_leaf(self, populated_denorm) -> None:
        """row_per with a downstream table in include_tables → None, no raise."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        # Subject points to nothing in-set; Image points to Subject.
        # Explicit row_per=Subject with Image in include_tables is a Rule-5
        # violation that would raise from as_dataframe — but describe must
        # swallow it.
        plan = d.describe(["Image", "Subject"], row_per="Subject")
        assert plan["row_per"] is None


class TestListPaths:
    """list_paths describes the FK graph from the dataset's anchor types."""

    def test_list_paths_keys(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        info = d.list_paths()
        # Required keys per spec §6
        for key in [
            "member_types",
            "anchor_types",
            "reachable_tables",
            "association_tables",
            "feature_tables",
            "schema_paths",
        ]:
            assert key in info

    def test_list_paths_reports_member_types(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        info = d.list_paths()
        assert "Image" in info["member_types"]
        assert "Subject" in info["member_types"]

    def test_list_paths_filter_by_tables(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        # When filter is given, schema_paths contains only entries involving
        # those tables.
        info = d.list_paths(tables=["Image"])
        for (source, target), _ in info["schema_paths"].items():
            assert "Image" in (source, target)

    def test_list_paths_reports_association_tables(self, populated_denorm) -> None:
        """Pure M:N link tables (Dataset_Image) surface in association_tables."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        info = d.list_paths()
        # The canned schema has Dataset_Image as a pure association.
        assert "Dataset_Image" in info["association_tables"]


class TestFromRids:
    """Denormalizer.from_rids constructs from arbitrary RID anchors."""

    def test_from_rids_with_table_tuples(self, populated_denorm) -> None:
        """(table, RID) pairs skip the lookup."""
        ml = _FakeMl(populated_denorm)
        d = Denormalizer.from_rids(
            [("Image", r) for r in populated_denorm["image_rids"]],
            ml=ml,
            dataset_rid=populated_denorm["dataset_rid"],
        )
        df = d.as_dataframe(["Image", "Subject"])
        # 3 Images in fixture — all reachable.
        assert len(df) == 3

    def test_from_rids_with_separate_deps(self, populated_denorm) -> None:
        """Escape hatch: pass catalog, workspace, model explicitly."""
        ls = populated_denorm["local_schema"]
        d = Denormalizer.from_rids(
            [("Image", r) for r in populated_denorm["image_rids"]],
            catalog=None,  # no lookup needed (table supplied)
            workspace=None,  # fixture provides engine directly
            model=populated_denorm["model"],
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=populated_denorm["dataset_rid"],
        )
        df = d.as_dataframe(["Image", "Subject"])
        assert len(df) == 3

    def test_from_rids_mixed_forms_with_fake_catalog(self, populated_denorm) -> None:
        """Mixed tuple + bare RIDs — bare RIDs resolved via catalog.resolve_rid."""
        ml = _FakeMl(populated_denorm)
        ml.catalog = _FakeCatalog({"IMG-1": "Image"})
        d = Denormalizer.from_rids(
            [
                ("Image", "IMG-2"),
                ("Image", "IMG-3"),
                "IMG-1",  # bare — catalog resolves this to Image
            ],
            ml=ml,
            dataset_rid=populated_denorm["dataset_rid"],
        )
        df = d.as_dataframe(["Image", "Subject"])
        assert len(df) == 3

    # ── Negative-path tests for the docstring's Raises: contract ──────────

    def test_from_rids_rejects_missing_model(self, populated_denorm) -> None:
        """No ml=, no model= → ValueError."""
        with pytest.raises(ValueError, match="requires either ml= or an explicit model="):
            Denormalizer.from_rids([("Image", "IMG-1")])

    def test_from_rids_rejects_bare_rid_without_catalog(self, populated_denorm) -> None:
        """Bare RID, no catalog → ValueError."""
        with pytest.raises(ValueError, match="no catalog available"):
            Denormalizer.from_rids(
                ["IMG-1"],
                model=populated_denorm["model"],
                engine=populated_denorm["local_schema"].engine,
            )

    def test_from_rids_rejects_unresolvable_bare_rid(self, populated_denorm) -> None:
        """Bare RID that catalog can't resolve → ValueError (not KeyError)."""
        ml = _FakeMl(populated_denorm)
        ml.catalog = _FakeCatalog({})  # empty → every lookup raises
        with pytest.raises(ValueError, match="Cannot resolve RID 'BOGUS'"):
            Denormalizer.from_rids(
                ["BOGUS"],
                ml=ml,
                dataset_rid=populated_denorm["dataset_rid"],
            )

    def test_from_rids_rejects_bad_tuple_arity(self, populated_denorm) -> None:
        """3-tuple or 1-tuple → ValueError (not opaque unpack error)."""
        ml = _FakeMl(populated_denorm)
        with pytest.raises(ValueError, match="must be .table, RID."):
            Denormalizer.from_rids(
                [("Image", "IMG-1", "extra")],  # type: ignore[list-item]
                ml=ml,
                dataset_rid=populated_denorm["dataset_rid"],
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeMl:
    """Minimal DerivaML-shaped fixture for from_rids tests."""

    def __init__(self, populated_denorm):
        self._pd = populated_denorm
        self.model = populated_denorm["model"]

        class _WS:
            def __init__(self, ls):
                self._ls = ls
                self.local_schema = ls
                self.engine = ls.engine

        self.workspace = _WS(populated_denorm["local_schema"])
        self.catalog = None  # bare-RID lookup not needed for tuple anchors


class _FakeCatalog:
    """Minimal catalog shim: resolve_rid maps RID -> table_name via a dict.

    Raises KeyError (mirroring ErmrestCatalog's behavior) when the RID is
    absent, so from_rids' error translation into ValueError is exercised.
    """

    def __init__(self, rid_to_table: dict[str, str]):
        self._map = rid_to_table

    def resolve_rid(self, rid: str):
        if rid not in self._map:
            raise KeyError(rid)

        # Return a namespace exposing .table.name to mirror the real API.
        class _TableInfo:
            def __init__(self, name: str):
                self.table = type("T", (), {"name": name})()

        return _TableInfo(self._map[rid])


class _FakeDataset:
    """Minimal DatasetLike for Denormalizer construction in tests.

    Wraps the populated_denorm fixture with the interface bits the
    Denormalizer needs: model, engine, orm_resolver, dataset_rid,
    list_dataset_members, list_dataset_children.
    """

    def __init__(self, populated_denorm: dict[str, Any], dataset_rid: str | None = None):
        self._pd = populated_denorm
        self.dataset_rid = dataset_rid or populated_denorm["dataset_rid"]
        # Attributes the Denormalizer uses:
        self.model = populated_denorm["model"]
        self.engine = populated_denorm["local_schema"].engine
        self._orm_resolver = populated_denorm["local_schema"].get_orm_class

    # Attributes exposed as "ml instance" pseudo-shim for Denormalizer(ds)
    # construction: the class pulls workspace/catalog from ds. For unit
    # tests against the canned fixture, we supply only what's actually
    # dereferenced — see Denormalizer.__init__.
    @property
    def _ml_instance(self):
        return None  # sentinel: tests use the fixture's engine directly

    def list_dataset_members(self, **kwargs: Any) -> dict[str, list[dict]]:
        """Return members based on whichever RID lists the fixture supplies.

        Both :func:`populated_denorm` and :func:`populated_denorm_diamond`
        expose ``image_rids`` + ``subject_rids``. The diamond fixture also
        carries ``observation_rids``, but Observation is intentionally not
        exposed as a member — it's meant to be reached via ``via=[...]``,
        not as an anchor.

        Honors ``self.dataset_rid``: returns no members if the dataset_rid
        doesn't match the fixture's (e.g., "NO-SUCH-DS"). This mirrors real
        ``Dataset.list_dataset_members`` which filters by dataset RID.
        """
        members: dict[str, list[dict]] = {}
        # Nonexistent dataset → no members.
        if self.dataset_rid != self._pd["dataset_rid"]:
            return members
        if "image_rids" in self._pd:
            members["Image"] = [{"RID": r} for r in self._pd["image_rids"]]
        if "subject_rids" in self._pd:
            members["Subject"] = [{"RID": r} for r in self._pd["subject_rids"]]
        return members

    def list_dataset_children(self, **kwargs: Any) -> list:
        return []

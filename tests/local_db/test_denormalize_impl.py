"""Tests for ``_denormalize_impl`` — the low-level primitive called by ``Denormalizer``."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from deriva_ml.local_db.denormalize import DenormalizeResult, _denormalize_impl


class TestDenormalize:
    """Core denormalization tests using the populated_denorm fixture."""

    def test_simple_denormalize(self, populated_denorm: dict[str, Any]) -> None:
        """Join Image + Subject through Dataset_Image, verify all rows appear."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
        )

        assert isinstance(result, DenormalizeResult)
        # 3 images in the dataset; IMG-3 has NULL Subject so LEFT JOIN keeps it
        assert result.row_count == 3

        # Verify column names contain expected prefixes
        col_names = [name for name, _ in result.columns]
        assert any("Image.Filename" in c for c in col_names)
        assert any("Subject.Name" in c for c in col_names)
        assert any("Image.RID" in c for c in col_names)

    def test_empty_dataset(self, populated_denorm: dict[str, Any]) -> None:
        """A dataset with no members returns zero rows but correct columns."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]

        # Use a nonexistent dataset RID
        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid="NO-SUCH-DS",
            include_tables=["Image", "Subject"],
        )

        assert result.row_count == 0
        # Column metadata should still be populated
        assert len(result.columns) > 0

    def test_left_join_null_subject(self, populated_denorm: dict[str, Any]) -> None:
        """Image with NULL Subject FK appears in result with NULL Subject cols."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
        )

        rows = list(result.iter_rows())
        # Find the row for IMG-3 (the one with NULL Subject)
        null_row = [r for r in rows if r.get("Image.Filename") == "c.png"]
        assert len(null_row) == 1
        assert null_row[0].get("Subject.Name") is None

    def test_to_dataframe(self, populated_denorm: dict[str, Any]) -> None:
        """to_dataframe returns a DataFrame with correct shape."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == result.row_count
        assert len(df.columns) == len(result.columns)

    def test_to_dataframe_empty(self, populated_denorm: dict[str, Any]) -> None:
        """to_dataframe on empty result returns DataFrame with correct columns."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid="NO-SUCH-DS",
            include_tables=["Image", "Subject"],
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert len(df.columns) == len(result.columns)

    def test_iter_rows(self, populated_denorm: dict[str, Any]) -> None:
        """iter_rows yields dicts with the correct keys."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
        )

        rows = list(result.iter_rows())
        assert len(rows) == result.row_count
        # Every row should be a dict
        for row in rows:
            assert isinstance(row, dict)

    def test_dataset_children_rids(self, populated_denorm: dict[str, Any]) -> None:
        """Passing dataset_children_rids expands the WHERE clause."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]

        # DS-001 is the only dataset with data.  Passing it as a child with
        # a fake parent should still return rows.
        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid="FAKE-PARENT",
            include_tables=["Image", "Subject"],
            dataset_children_rids=["DS-001"],
        )

        # Should see the 3 images associated with DS-001
        assert result.row_count == 3

    def test_single_table_include(self, populated_denorm: dict[str, Any]) -> None:
        """Including only Image (no Subject) still works."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image"],
        )

        assert result.row_count == 3
        col_names = [name for name, _ in result.columns]
        assert any("Image.Filename" in c for c in col_names)
        # Subject table columns (e.g., Subject.Name, Subject.RID) should NOT
        # appear, though Image.Subject (the FK column) will be there.
        assert not any(c.startswith("Subject.") for c in col_names)


class TestEmptyColumnSpecs:
    """Test that denormalize handles the no-column-specs edge case gracefully."""

    def test_empty_include_tables_returns_empty_result(self, populated_denorm: dict[str, Any]) -> None:
        """_prepare_wide_table with no matching columns → empty DenormalizeResult."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        # An empty include_tables list should produce no column_specs and no rows.
        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=[],
        )

        assert isinstance(result, DenormalizeResult)
        assert result.row_count == 0
        # columns list may be empty or contain no data columns
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Regression tests — these close the gap that let C1 ship:
# the prior test suite pre-populated ORM tables in a fixture, so denormalize()
# "passed" even though it never fetched rows. These tests start with an
# empty LocalSchema and verify that source="catalog" actually populates rows
# via PagedFetcher.
# ---------------------------------------------------------------------------


class TestCatalogSource:
    """Verify source='catalog' actually fetches rows via the paged client.

    THIS IS THE TEST THAT WOULD HAVE CAUGHT THE C1 BUG.

    Unlike the ``populated_denorm`` fixture which pre-inserts rows into the
    LocalSchema, these tests start with a fresh empty schema and supply a
    fake paged client. The denormalizer must walk the join plan, fetch rows
    via the client, and then run the SQL join — exactly what production
    callers need.
    """

    def test_catalog_source_fetches_and_joins(
        self,
        denorm_deriva_model: Any,
        denorm_local_schema: Any,
    ) -> None:
        """An empty LocalSchema returns rows when source='catalog'."""
        # Import FakePagedClient from the paged_fetcher tests.
        from tests.local_db.test_paged_fetcher import FakePagedClient

        ls = denorm_local_schema
        model = denorm_deriva_model
        ds_rid = "DS-CAT-001"

        # Provide the rows that a real ERMrest catalog would return, keyed
        # by ERMrest-qualified "schema:table" names.
        fake = FakePagedClient(
            rows_by_table={
                "deriva-ml:Dataset": [{"RID": ds_rid, "Description": "t"}],
                "deriva-ml:Dataset_Image": [
                    {"RID": "DI-1", "Dataset": ds_rid, "Image": "IMG-A"},
                    {"RID": "DI-2", "Dataset": ds_rid, "Image": "IMG-B"},
                ],
                "isa:Image": [
                    {"RID": "IMG-A", "Filename": "a.png", "Subject": "S-1"},
                    {"RID": "IMG-B", "Filename": "b.png", "Subject": "S-2"},
                ],
                "isa:Subject": [
                    {"RID": "S-1", "Name": "Alice"},
                    {"RID": "S-2", "Name": "Bob"},
                ],
            }
        )

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
            source="catalog",
            paged_client=fake,
        )

        # KEY ASSERTION: rows come back despite starting from an empty DB.
        # This is the assertion the prior test suite should have had.
        assert result.row_count == 2, (
            "source='catalog' must fetch rows via the paged client; "
            "returning 0 rows means PagedFetcher was never called."
        )

        rows = list(result.iter_rows())
        filenames = {r["Image.Filename"] for r in rows}
        assert filenames == {"a.png", "b.png"}

        # Verify the client was actually consulted (not just bypassed).
        assert any(req[0] == "fetch_rid_batch" for req in fake.requests), (
            "Expected fetch_rid_batch calls on the paged client"
        )

    def test_catalog_source_requires_paged_client(self, populated_denorm: dict[str, Any]) -> None:
        """source='catalog' without paged_client raises ValueError."""
        with pytest.raises(ValueError, match="paged_client"):
            _denormalize_impl(
                model=populated_denorm["model"],
                engine=populated_denorm["local_schema"].engine,
                orm_resolver=populated_denorm["local_schema"].get_orm_class,
                dataset_rid=populated_denorm["dataset_rid"],
                include_tables=["Image"],
                source="catalog",
                # paged_client omitted — must error
            )

    def test_local_source_does_not_require_paged_client(self, populated_denorm: dict[str, Any]) -> None:
        """source='local' (default) works without a paged_client."""
        # Just verify it doesn't raise — rows come from the fixture.
        result = _denormalize_impl(
            model=populated_denorm["model"],
            engine=populated_denorm["local_schema"].engine,
            orm_resolver=populated_denorm["local_schema"].get_orm_class,
            dataset_rid=populated_denorm["dataset_rid"],
            include_tables=["Image"],
            source="local",
        )
        assert result.row_count > 0

    def test_slice_source_does_not_require_paged_client(self, populated_denorm: dict[str, Any]) -> None:
        """source='slice' works without a paged_client (rows from attached slice)."""
        result = _denormalize_impl(
            model=populated_denorm["model"],
            engine=populated_denorm["local_schema"].engine,
            orm_resolver=populated_denorm["local_schema"].get_orm_class,
            dataset_rid=populated_denorm["dataset_rid"],
            include_tables=["Image"],
            source="slice",
        )
        assert result.row_count > 0


class TestDatasetOrmGuard:
    """C3: missing Dataset ORM raises a clear RuntimeError, not a cryptic crash."""

    def test_missing_dataset_orm_raises_runtime_error(self, populated_denorm: dict[str, Any]) -> None:
        """If orm_resolver returns None for 'Dataset', raise clear error."""
        real_resolver = populated_denorm["local_schema"].get_orm_class

        def broken_resolver(name: str) -> Any:
            if name == "Dataset":
                return None
            return real_resolver(name)

        with pytest.raises(RuntimeError, match="Dataset ORM"):
            _denormalize_impl(
                model=populated_denorm["model"],
                engine=populated_denorm["local_schema"].engine,
                orm_resolver=broken_resolver,
                dataset_rid=populated_denorm["dataset_rid"],
                include_tables=["Image"],
            )


class TestRowCompletenessInvariant:
    """SC-06 / RB-02 / TC-03 regression: same table, two paths, two parametrizations.

    Spec §6 step 3 requires the local cache to contain the union of rows
    every path's ``(table, rid_column, rids)`` tuple would fetch — the
    *row-completeness invariant*. Before the fix, ``_populate_from_catalog_inner``
    keyed its ``processed`` set on table name only, so the second walk's
    fetch was silently skipped when two element paths converged on the
    same table.

    These tests pin the invariant by hand-constructing a two-path
    ``join_tables`` dict and asserting both fetches actually fire (and
    that the resulting Image set is the union of what each path
    legitimately asks for).
    """

    def _build_two_path_scenario(
        self,
        denorm_feature_deriva_model: Any,
        denorm_local_schema_feature: Any,
    ) -> dict[str, Any]:
        """Set up a fixture where two element paths converge on Image.

        The pre-populated state is what ``_collect_fk_values`` will read
        from the engine to decide which Image RIDs each path needs.

        Path A: ``Dataset → Dataset_Image → Image`` — pulls IMG-A1, IMG-A2.
        Path B: ``Dataset → Execution_Image_Image_Classification → Image``
            — pulls IMG-B1, IMG-B2 (a disjoint set; this is what makes
            the bug visible).

        Returns a dict carrying everything the test needs: the
        hand-crafted ``join_tables``, the engine, the resolver, the
        ``FakePagedClient`` (preloaded with rows for *all* Image RIDs so
        a correct walk pulls the union; a buggy walk pulls only one
        path's worth).
        """
        from sqlalchemy.orm import Session

        from tests.local_db.test_paged_fetcher import FakePagedClient

        ls = denorm_local_schema_feature
        model = denorm_feature_deriva_model

        ds_rid = "DS-001"
        path_a_image_rids = ["IMG-A1", "IMG-A2"]
        path_b_image_rids = ["IMG-B1", "IMG-B2"]
        exec_rid = "EXE-1"
        cls_rid = "CLS-cat"

        # Pre-populate the engine with the "prior tables" each path
        # reads to derive its target RIDs. Path A reads Dataset_Image;
        # path B reads Execution_Image_Image_Classification.
        with Session(ls.engine) as session:
            ds_cls = ls.get_orm_class("Dataset")
            session.add(ds_cls(RID=ds_rid, Description="t"))

            di_cls = ls.get_orm_class("Dataset_Image")
            for img in path_a_image_rids:
                session.add(di_cls(RID=f"DI-{img}", Dataset=ds_rid, Image=img))

            exe_cls = ls.get_orm_class("Execution")
            session.add(exe_cls(RID=exec_rid, Description="run"))

            cls_cls = ls.get_orm_class("Image_Classification")
            session.add(cls_cls(RID=cls_rid, Name="cat"))

            # Feature_Name is now an FK to deriva-ml.Feature_Name.Name
            # (the real 4-FK feature shape), so the term must exist
            # before the feature rows reference it.
            fn_cls = ls.get_orm_class("Feature_Name")
            session.add(fn_cls(RID="FN-default", Name="default"))

            feat_cls = ls.get_orm_class("Execution_Image_Image_Classification")
            for img in path_b_image_rids:
                session.add(
                    feat_cls(
                        RID=f"EIIC-{img}",
                        Feature_Name="default",
                        Image=img,
                        Execution=exec_rid,
                        Image_Classification=cls_rid,
                    )
                )
            session.commit()

        # Look up the actual ERMrest Column objects so we can build
        # join_conditions whose `.table.name` / `.name` match what
        # ``_collect_fk_values`` and ``_col_table_name`` expect.
        image_tbl = model.name_to_table("Image")
        dataset_tbl = model.name_to_table("Dataset")
        dataset_image_tbl = model.name_to_table("Dataset_Image")
        exec_tbl = model.name_to_table("Execution")
        feat_tbl = model.name_to_table("Execution_Image_Image_Classification")

        image_rid_col = image_tbl.columns["RID"]
        dataset_rid_col = dataset_tbl.columns["RID"]
        di_image_col = dataset_image_tbl.columns["Image"]
        di_dataset_col = dataset_image_tbl.columns["Dataset"]
        feat_image_col = feat_tbl.columns["Image"]
        exec_rid_col = exec_tbl.columns["RID"]
        feat_exec_col = feat_tbl.columns["Execution"]

        # Hand-crafted join_tables: two element paths both ending at
        # Image, but reaching it via different intermediate tables.
        # ``_collect_fk_values`` will read Dataset_Image for path A and
        # Execution_Image_Image_Classification for path B, producing
        # disjoint Image RID sets.
        join_tables = {
            "Image_via_DatasetImage": (
                ["Dataset", "Dataset_Image", "Image"],
                {
                    "Dataset_Image": {(di_dataset_col, dataset_rid_col)},
                    "Image": {(di_image_col, image_rid_col)},
                },
                {"Dataset_Image": "inner", "Image": "inner"},
            ),
            "Image_via_Feature": (
                ["Dataset", "Execution", "Execution_Image_Image_Classification", "Image"],
                {
                    # Synthetic edge: planner-internal, we don't need Dataset→Execution
                    # FK conditions here because the test calls the *inner* walk
                    # directly and only the table-targeted conditions matter.
                    "Execution": set(),
                    "Execution_Image_Image_Classification": {(feat_exec_col, exec_rid_col)},
                    "Image": {(feat_image_col, image_rid_col)},
                },
                {
                    "Execution": "inner",
                    "Execution_Image_Image_Classification": "inner",
                    "Image": "inner",
                },
            ),
        }

        # FakePagedClient with rows for the union — a correct walk will
        # pull all four; a buggy walk (today) pulls only path A's two.
        all_images = path_a_image_rids + path_b_image_rids
        fake = FakePagedClient(
            rows_by_table={
                "deriva-ml:Dataset": [{"RID": ds_rid, "Description": "t"}],
                "isa:Image": [{"RID": r, "Filename": f"{r}.png", "Subject": None} for r in all_images],
                # Path A's prior table (Dataset_Image) is already in engine — fetcher
                # will still issue a fetch for it; provide rows so the call doesn't
                # crash. The same applies to path B's intermediate tables.
                "deriva-ml:Dataset_Image": [
                    {"RID": f"DI-{r}", "Dataset": ds_rid, "Image": r} for r in path_a_image_rids
                ],
                "deriva-ml:Execution": [{"RID": exec_rid, "Description": "run"}],
                "deriva-ml:Execution_Image_Image_Classification": [
                    {
                        "RID": f"EIIC-{r}",
                        "Feature_Name": "default",
                        "Image": r,
                        "Execution": exec_rid,
                        "Image_Classification": cls_rid,
                    }
                    for r in path_b_image_rids
                ],
            }
        )

        return {
            "model": model,
            "engine": ls.engine,
            "orm_resolver": ls.get_orm_class,
            "join_tables": join_tables,
            "fake_client": fake,
            "dataset_rid": ds_rid,
            "path_a_image_rids": path_a_image_rids,
            "path_b_image_rids": path_b_image_rids,
            "local_schema": ls,
        }

    def test_two_paths_both_fetch_their_target_rids(
        self,
        denorm_feature_deriva_model: Any,
        denorm_local_schema_feature: Any,
    ) -> None:
        """Both element paths' Image fetches must fire (row-completeness invariant).

        With the bug, ``processed`` adds ``"Image"`` after path A's fetch
        and path B's fetch is silently skipped. The Image RIDs only
        reachable via path B are absent from the local cache.

        With the fix, the dedup key is ``(table, rid_column, frozenset(rids))``,
        so the two paths' distinct parametrizations both issue a fetch.

        The assertion that fails on main is the final
        ``image_rids_in_engine == set(union)`` — IMG-B1/IMG-B2 are missing
        because path B's fetch never ran.
        """
        from sqlalchemy import select as sa_select

        from deriva_ml.local_db.denormalize import (
            _foreign_keys_off,
            _populate_from_catalog_inner,
        )
        from deriva_ml.local_db.paged_fetcher import PagedFetcher

        scenario = self._build_two_path_scenario(denorm_feature_deriva_model, denorm_local_schema_feature)

        fetcher = PagedFetcher(client=scenario["fake_client"], engine=scenario["engine"])

        table_to_schema = {
            "Dataset": "deriva-ml",
            "Dataset_Image": "deriva-ml",
            "Image": "isa",
            "Execution": "deriva-ml",
            "Execution_Image_Image_Classification": "deriva-ml",
        }

        with _foreign_keys_off(scenario["engine"]):
            _populate_from_catalog_inner(
                fetcher=fetcher,
                engine=scenario["engine"],
                orm_resolver=scenario["orm_resolver"],
                table_to_schema=table_to_schema,
                join_tables=scenario["join_tables"],
                dataset_rid_list=[scenario["dataset_rid"]],
            )

        # Collect all Image fetches the fake client saw.
        image_fetch_rids: list[set[str]] = []
        for kind, params in scenario["fake_client"].requests:
            if kind == "fetch_rid_batch" and params["table"] == "isa:Image":
                image_fetch_rids.append(set(params["rids"]))

        union_requested = set().union(*image_fetch_rids) if image_fetch_rids else set()
        expected_union = set(scenario["path_a_image_rids"] + scenario["path_b_image_rids"])

        # The invariant: every (table, rid_column, rids) tuple in the
        # plan must have its rows in the local cache. The cheapest way
        # to surface this is: the union of all Image fetches must
        # include every Image RID either path's parametrization
        # legitimately asks for.
        assert union_requested == expected_union, (
            f"Row-completeness invariant violated: path B's Image fetch was "
            f"skipped. Image RIDs requested across all fetches: {union_requested}. "
            f"Expected union of both paths: {expected_union}. "
            f"Missing: {expected_union - union_requested}."
        )

        # And the engine must end up holding the union.
        image_orm = scenario["orm_resolver"]("Image")
        with scenario["engine"].connect() as conn:
            image_rids_in_engine = {
                row[0] for row in conn.execute(sa_select(image_orm.__table__.columns["RID"])).fetchall()
            }
        assert image_rids_in_engine == expected_union, (
            f"Local cache missing rows: engine has {image_rids_in_engine}, expected union {expected_union}."
        )

    def test_duplicate_parametrization_is_still_deduped(
        self,
        denorm_feature_deriva_model: Any,
        denorm_local_schema_feature: Any,
    ) -> None:
        """Two paths with the SAME (table, rid_column, rids) tuple → one fetch.

        The fix preserves the optimization for true duplicates — if both
        paths converge on Image via the same rid_column with the same
        rid set, only one fetch should fire. This pins the dedup half of
        the contract (the other half being row-completeness).
        """
        from deriva_ml.local_db.denormalize import (
            _foreign_keys_off,
            _populate_from_catalog_inner,
        )
        from deriva_ml.local_db.paged_fetcher import PagedFetcher

        scenario = self._build_two_path_scenario(denorm_feature_deriva_model, denorm_local_schema_feature)

        # Replace path B's join_conditions[Image] with an identical
        # parametrization to path A's — same (rid_column, source) so
        # _collect_fk_values returns the same (rid_column, rids).
        model = scenario["model"]
        image_rid_col = model.name_to_table("Image").columns["RID"]
        di_image_col = model.name_to_table("Dataset_Image").columns["Image"]

        # Rewire path B's Image conditions to point at Dataset_Image too
        # (artificial — the point is just that both paths produce the
        # same (rid_column, frozenset(rids)) tuple for Image).
        path_b = scenario["join_tables"]["Image_via_Feature"]
        path_b[1]["Image"] = {(di_image_col, image_rid_col)}

        fetcher = PagedFetcher(client=scenario["fake_client"], engine=scenario["engine"])
        table_to_schema = {
            "Dataset": "deriva-ml",
            "Dataset_Image": "deriva-ml",
            "Image": "isa",
            "Execution": "deriva-ml",
            "Execution_Image_Image_Classification": "deriva-ml",
        }

        with _foreign_keys_off(scenario["engine"]):
            _populate_from_catalog_inner(
                fetcher=fetcher,
                engine=scenario["engine"],
                orm_resolver=scenario["orm_resolver"],
                table_to_schema=table_to_schema,
                join_tables=scenario["join_tables"],
                dataset_rid_list=[scenario["dataset_rid"]],
            )

        image_fetches = [
            params
            for kind, params in scenario["fake_client"].requests
            if kind == "fetch_rid_batch" and params["table"] == "isa:Image"
        ]
        # Exactly one Image fetch: the two paths produced an identical
        # (rid_column, rids) tuple and the second is correctly deduped.
        assert len(image_fetches) == 1, (
            f"Expected 1 Image fetch for identical parametrizations, got {len(image_fetches)}: {image_fetches}"
        )


def test_denormalize_result_extend() -> None:
    """DenormalizeResult.extend appends rows and updates row_count."""
    from deriva_ml.local_db.denormalize import DenormalizeResult

    base = DenormalizeResult(
        columns=[("A.RID", "text")],
        row_count=2,
        _rows=[{"A.RID": "1-A"}, {"A.RID": "1-B"}],
    )
    extended = base.extend([{"A.RID": "1-C"}, {"A.RID": "1-D"}])
    assert extended.row_count == 4
    assert extended.columns == base.columns
    assert [r["A.RID"] for r in extended.iter_rows()] == ["1-A", "1-B", "1-C", "1-D"]
    # Does not mutate the original
    assert base.row_count == 2


class TestCacheAgeSeconds:
    """SC-03 / spec §6 freshness caveat: ``DenormalizeResult.cache_age_seconds``.

    The contract (PR #231 §6): ``DenormalizeResult`` carries
    ``cache_age_seconds: float | None``:

    - ``None`` for ``source != "catalog"`` (no live fetch happened).
    - For ``source == "catalog"``: the wall-clock delta between the
      denormalize call's completion and the *earliest* fetch in the
      local cache that participated in this result's SQL JOIN.

    These tests pin the contract at the ``_denormalize_impl`` boundary.
    """

    def test_denormalize_result_carries_cache_age_seconds(
        self,
        denorm_deriva_model: Any,
        denorm_local_schema: Any,
    ) -> None:
        """A successful catalog fetch produces a non-negative cache_age_seconds."""
        from tests.local_db.test_paged_fetcher import FakePagedClient

        ls = denorm_local_schema
        model = denorm_deriva_model
        ds_rid = "DS-CA-001"

        fake = FakePagedClient(
            rows_by_table={
                "deriva-ml:Dataset": [{"RID": ds_rid, "Description": "t"}],
                "deriva-ml:Dataset_Image": [
                    {"RID": "DI-1", "Dataset": ds_rid, "Image": "IMG-A"},
                ],
                "isa:Image": [
                    {"RID": "IMG-A", "Filename": "a.png", "Subject": "S-1"},
                ],
                "isa:Subject": [
                    {"RID": "S-1", "Name": "Alice"},
                ],
            }
        )

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
            source="catalog",
            paged_client=fake,
        )

        assert result.cache_age_seconds is not None, (
            "source='catalog' with non-empty join plan must surface a freshness signal"
        )
        assert result.cache_age_seconds >= 0, f"cache_age_seconds must be non-negative, got {result.cache_age_seconds}"

    def test_denormalize_result_cache_age_is_none_for_local_source(
        self,
        populated_denorm: dict[str, Any],
    ) -> None:
        """source='local' (no live fetch) reports cache_age_seconds=None.

        The spec carve-out (PR #231 §6): ``None`` for ``source='bag'``
        (no live fetch happened). ``source='local'`` is the in-tree
        equivalent — rows are pre-populated, no PagedFetcher is
        constructed, no ledger exists.
        """
        result = _denormalize_impl(
            model=populated_denorm["model"],
            engine=populated_denorm["local_schema"].engine,
            orm_resolver=populated_denorm["local_schema"].get_orm_class,
            dataset_rid=populated_denorm["dataset_rid"],
            include_tables=["Image", "Subject"],
            source="local",
        )

        assert result.cache_age_seconds is None, (
            "source='local' must not produce a cache_age_seconds — no live fetch happened"
        )

    def test_denormalize_result_cache_age_is_none_for_slice_source(
        self,
        populated_denorm: dict[str, Any],
    ) -> None:
        """source='slice' also produces cache_age_seconds=None (no live fetch).

        Spec PR #231 §6 names ``source='bag'`` explicitly. ``source='slice'``
        is the production equivalent that ``DatasetBag.get_denormalized_as_dataframe``
        uses; same carve-out applies.
        """
        result = _denormalize_impl(
            model=populated_denorm["model"],
            engine=populated_denorm["local_schema"].engine,
            orm_resolver=populated_denorm["local_schema"].get_orm_class,
            dataset_rid=populated_denorm["dataset_rid"],
            include_tables=["Image"],
            source="slice",
        )

        assert result.cache_age_seconds is None, (
            "source='slice' must not produce a cache_age_seconds — no live fetch happened"
        )

    def test_denormalize_result_cache_age_reflects_oldest_fetch(
        self,
        denorm_deriva_model: Any,
        denorm_local_schema: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cache_age_seconds reports age of the *oldest* participating fetch.

        Scenario: simulate two fetches in the same call separated by N
        seconds of wall-clock time. The reported ``cache_age_seconds``
        must be at least N (the delta from the first fetch to "now"),
        not the delta from the most recent fetch.

        We drive this deterministically by monkey-patching
        ``time.monotonic`` in the paged_fetcher and denormalize modules
        to return successive values from a controlled list — no actual
        ``sleep`` calls (which would slow the suite and add jitter).
        """
        from tests.local_db.test_paged_fetcher import FakePagedClient

        ls = denorm_local_schema
        model = denorm_deriva_model
        ds_rid = "DS-CA-AGE-001"

        fake = FakePagedClient(
            rows_by_table={
                "deriva-ml:Dataset": [{"RID": ds_rid, "Description": "t"}],
                "deriva-ml:Dataset_Image": [
                    {"RID": "DI-1", "Dataset": ds_rid, "Image": "IMG-A"},
                ],
                "isa:Image": [
                    {"RID": "IMG-A", "Filename": "a.png", "Subject": "S-1"},
                ],
                "isa:Subject": [
                    {"RID": "S-1", "Name": "Alice"},
                ],
            }
        )

        # Monkey-patched clock: each call returns the next value in the
        # sequence. The fetcher calls ``time.monotonic`` once per distinct
        # ``record_fetch_start``; ``_denormalize_impl`` calls it once at
        # the end via ``_compute_cache_age_seconds``. We pin those reads
        # to a deterministic sequence so the assertion below is exact.
        # Sequence: 100 (Dataset fetch), 110 (Dataset_Image), 120 (Image),
        # 130 (Subject), then 145 for the final "now" — i.e. the oldest
        # fetch is 45s in the past at the time of result construction.
        clock_values = iter([100.0, 110.0, 120.0, 130.0, 145.0])

        def _fake_monotonic() -> float:
            return next(clock_values)

        # Patch where ``time.monotonic`` is actually called — once in
        # paged_fetcher.PagedFetcher.record_fetch_start, once in
        # denormalize._compute_cache_age_seconds. Patching the underlying
        # ``time`` module on both module surfaces is the cleanest seam.
        from deriva_ml.local_db import denormalize as denorm_mod
        from deriva_ml.local_db import paged_fetcher as pf_mod

        monkeypatch.setattr(pf_mod.time, "monotonic", _fake_monotonic)
        monkeypatch.setattr(denorm_mod.time, "monotonic", _fake_monotonic)

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", "Subject"],
            source="catalog",
            paged_client=fake,
        )

        # Oldest fetch at t=100, "now" at t=145 → age = 45s exactly.
        # The implementation does at most 4 ``record_fetch_start`` calls
        # (Dataset, Dataset_Image, Image, Subject) plus one final
        # ``time.monotonic()`` read for the cache_age_seconds computation —
        # 5 total reads, matching our 5-value sequence. If the planner
        # ever changes shape, this test will surface it as a StopIteration
        # from the iterator (a clear signal the contract needs an update).
        assert result.cache_age_seconds == 45.0, (
            f"Expected cache_age_seconds reporting oldest fetch age (45.0), got {result.cache_age_seconds}"
        )


class TestCollectFkValuesPartialEngineState:
    """TC-08 / spec §7 F4: ``_collect_fk_values`` reads what's IN the engine.

    The audit (TC-08) names a fragility: ``_collect_fk_values`` queries
    the local engine (``SELECT DISTINCT pull_col FROM other_table``) for
    the RIDs to feed the next fetch. If a prior tighter-scoped
    denormalize populated the engine with only a subset of the
    "other_table" rows, the new (broader) denormalize inherits the
    tighter scope silently — there's no second consultation of the
    server to broaden the fetch.

    This is the F4 behavior the spec names. The test below documents it
    by setting up partial engine state, calling ``_collect_fk_values``,
    and asserting the returned RID list reflects ONLY what's in the
    engine. This is the current contract — and is the fragility F4
    names. If the implementation ever changes to consult the server
    instead, this test will fail and the spec entry should be revised
    accordingly.
    """

    def test_returns_only_rids_present_in_engine(
        self,
        denorm_deriva_model: Any,
        denorm_local_schema: Any,
    ) -> None:
        """Partial engine state → ``_collect_fk_values`` returns the
        partial set, not the broader server set.

        Documents F4 behavior: engine state determines fetch scope.
        This is the fragility named in spec §7. If the engine has only
        K of N FK source rows, ``_collect_fk_values`` returns K values
        — even though against a full catalog the join needs all N.
        """
        from sqlalchemy.orm import Session

        from deriva_ml.local_db.denormalize import _collect_fk_values

        ls = denorm_local_schema
        model = denorm_deriva_model

        # Set up PARTIAL engine state: insert Dataset + only 2 of 4 hypothetical
        # Dataset_Image rows. Against a "full catalog" the join would pull all
        # 4 Images; with only 2 in the engine, _collect_fk_values can only
        # report the 2 it sees.
        ds_rid = "DS-PE-001"
        partial_image_rids = ["IMG-A", "IMG-B"]
        broader_image_rids = ["IMG-A", "IMG-B", "IMG-C", "IMG-D"]  # what catalog has

        with Session(ls.engine) as session:
            ds_cls = ls.get_orm_class("Dataset")
            session.add(ds_cls(RID=ds_rid, Description="partial-state"))
            di_cls = ls.get_orm_class("Dataset_Image")
            for img in partial_image_rids:
                session.add(di_cls(RID=f"DI-{img}", Dataset=ds_rid, Image=img))
            session.commit()

        # Build the join condition the planner would have produced for
        # the Image fetch leg: Dataset_Image.Image -> Image.RID.
        dataset_image_tbl = model.name_to_table("Dataset_Image")
        image_tbl = model.name_to_table("Image")
        di_image_col = dataset_image_tbl.columns["Image"]
        image_rid_col = image_tbl.columns["RID"]

        conditions = {(di_image_col, image_rid_col)}

        values, filter_col = _collect_fk_values(
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            conditions=conditions,
            target_table_name="Image",
        )

        # Filter column on Image is "RID" (the PK side of the FK).
        assert filter_col == "RID"
        # The contract: values reflect engine state, not the broader catalog.
        assert set(values) == set(partial_image_rids), (
            f"Expected _collect_fk_values to return engine-state values "
            f"{set(partial_image_rids)} (the rows actually in the local "
            f"Dataset_Image), got {set(values)}. F4 behavior changed?"
        )
        # And NOT the broader set the catalog would have provided.
        assert set(values) != set(broader_image_rids), (
            "Sanity check on the test premise: engine should NOT contain "
            "the broader catalog set; if it does, the partial-state "
            "precondition is broken."
        )

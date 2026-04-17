"""Tests for ``_denormalize_impl`` — the low-level primitive called by ``Denormalizer``."""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

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


class TestDatasetVersionWarning:
    """Tests for the 'version' parameter warning path on Dataset delegates.

    These tests exercise the warning logic directly via a mock Dataset so that
    we don't need a live catalog connection.
    """

    def _make_mock_dataset(self, local_schema_built: bool = True):
        """Build a minimal mock of the Dataset object's dependencies."""
        # Build a mock workspace
        mock_ws = MagicMock()
        if local_schema_built:
            mock_ws.local_schema = MagicMock()
            mock_ws.local_schema.get_orm_class = MagicMock(return_value=None)
            mock_ws.engine = MagicMock()
        else:
            mock_ws.local_schema = None

        mock_ml = MagicMock()
        mock_ml.workspace = mock_ws

        mock_dataset = MagicMock()
        mock_dataset._ml_instance = mock_ml
        mock_dataset.dataset_rid = "DS-MOCK"
        mock_dataset.list_dataset_children.return_value = []
        return mock_dataset, mock_ws

    def test_version_warning_dataframe(self, populated_denorm: dict[str, Any]) -> None:
        """denormalize_as_dataframe emits UserWarning when version != None."""
        from deriva_ml.dataset.dataset import Dataset

        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]

        mock_dataset, mock_ws = self._make_mock_dataset(local_schema_built=True)
        mock_ws.engine = ls.engine
        mock_ws.local_schema.get_orm_class = ls.get_orm_class
        mock_dataset._ml_instance.model = model

        stub_result = DenormalizeResult(columns=[], row_count=0, _rows=[])
        # The function body does `from deriva_ml.local_db.denormalize import _denormalize_impl`,
        # so we must patch the name in that module, not in dataset.dataset.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("deriva_ml.local_db.denormalize._denormalize_impl", return_value=stub_result):
                Dataset.denormalize_as_dataframe(mock_dataset, ["Image"], version="1.0.0")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "version" in str(user_warnings[0].message).lower()

    def test_no_warning_when_version_is_none(self, populated_denorm: dict[str, Any]) -> None:
        """denormalize_as_dataframe does NOT warn when version is None."""
        from deriva_ml.dataset.dataset import Dataset

        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]

        mock_dataset, mock_ws = self._make_mock_dataset(local_schema_built=True)
        mock_ws.engine = ls.engine
        mock_ws.local_schema.get_orm_class = ls.get_orm_class
        mock_dataset._ml_instance.model = model

        stub_result = DenormalizeResult(columns=[], row_count=0, _rows=[])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("deriva_ml.local_db.denormalize._denormalize_impl", return_value=stub_result):
                Dataset.denormalize_as_dataframe(mock_dataset, ["Image"], version=None)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_version_warning_dict(self, populated_denorm: dict[str, Any]) -> None:
        """denormalize_as_dict also emits UserWarning when version != None."""
        from deriva_ml.dataset.dataset import Dataset

        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]

        mock_dataset, mock_ws = self._make_mock_dataset(local_schema_built=True)
        mock_ws.engine = ls.engine
        mock_ws.local_schema.get_orm_class = ls.get_orm_class
        mock_dataset._ml_instance.model = model

        stub_result = DenormalizeResult(columns=[], row_count=0, _rows=[])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("deriva_ml.local_db.denormalize._denormalize_impl", return_value=stub_result):
                # denormalize_as_dict is a generator; must exhaust it.
                list(Dataset.denormalize_as_dict(mock_dataset, ["Image"], version="2.0.0"))

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "version" in str(user_warnings[0].message).lower()


class TestLocalSchemaNoneGuard:
    """Tests for the local_schema is None guard in Dataset delegates."""

    def _make_mock_dataset_no_schema(self):
        """Return a mock Dataset whose workspace has local_schema=None."""
        mock_ws = MagicMock()
        mock_ws.local_schema = None

        mock_ml = MagicMock()
        mock_ml.workspace = mock_ws

        mock_dataset = MagicMock()
        mock_dataset._ml_instance = mock_ml
        mock_dataset.dataset_rid = "DS-MOCK"
        mock_dataset.list_dataset_children.return_value = []
        return mock_dataset

    def test_dataframe_raises_when_no_schema(self) -> None:
        """denormalize_as_dataframe raises RuntimeError if local_schema is None."""
        from deriva_ml.dataset.dataset import Dataset

        mock_dataset = self._make_mock_dataset_no_schema()

        with pytest.raises(RuntimeError, match="local_schema not built"):
            Dataset.denormalize_as_dataframe(mock_dataset, ["Image"])

    def test_dict_raises_when_no_schema(self) -> None:
        """denormalize_as_dict raises RuntimeError if local_schema is None."""
        from deriva_ml.dataset.dataset import Dataset

        mock_dataset = self._make_mock_dataset_no_schema()

        with pytest.raises(RuntimeError, match="local_schema not built"):
            # Must trigger the generator to see the RuntimeError
            list(Dataset.denormalize_as_dict(mock_dataset, ["Image"]))


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

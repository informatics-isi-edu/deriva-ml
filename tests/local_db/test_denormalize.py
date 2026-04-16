"""Tests for the unified denormalization engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from deriva_ml.local_db.denormalize import DenormalizeResult, denormalize


class TestDenormalize:
    """Core denormalization tests using the populated_denorm fixture."""

    def test_simple_denormalize(self, populated_denorm: dict[str, Any]) -> None:
        """Join Image + Subject through Dataset_Image, verify all rows appear."""
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        result = denormalize(
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
        result = denormalize(
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

        result = denormalize(
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

        result = denormalize(
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

        result = denormalize(
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

        result = denormalize(
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
        result = denormalize(
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

        result = denormalize(
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

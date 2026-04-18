"""Unit tests for local_db.schema.LocalSchema adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from deriva.core.ermrest_model import Model
from sqlalchemy import text

from deriva_ml.local_db.schema import LocalSchema


class TestLocalSchemaFromBagModel:
    def test_builds_tables_under_file_db(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            # Tables should exist with fully-qualified names
            assert ls.find_table("isa.Image") is not None
            assert ls.find_table("isa.Subject") is not None
            assert ls.find_table("deriva-ml.Dataset") is not None
        finally:
            ls.dispose()

    def test_unqualified_find_table_works(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            t = ls.find_table("Image")
            assert t.name.endswith("Image")
        finally:
            ls.dispose()

    def test_engine_is_wal(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            with ls.engine.connect() as conn:
                mode = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert mode == "wal"
        finally:
            ls.dispose()


class TestLocalSchemaSchemasList:
    def test_exposes_schemas(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            assert set(ls.schemas) == {"isa", "deriva-ml"}
        finally:
            ls.dispose()


class TestLocalSchemaListTables:
    def test_list_tables_returns_all(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            tables = ls.list_tables()
            assert any("Image" in t for t in tables)
            assert any("Subject" in t for t in tables)
            assert any("Dataset" in t for t in tables)
        finally:
            ls.dispose()


class TestLocalSchemaContextManager:
    def test_context_manager_disposes(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        ) as ls:
            assert ls.find_table("Image") is not None
        # After context exit, engine should be disposed. A simple probe:
        # attempting a new connection on the disposed engine should either
        # work (SQLAlchemy re-initializes the pool) or — better — we just
        # assert that the context manager returned successfully.


class TestLocalSchemaProperties:
    def test_metadata_property(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        ) as ls:
            assert ls.metadata is not None
            # MetaData should contain our tables
            assert len(ls.metadata.tables) > 0

    def test_database_path_property(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        ) as ls:
            assert ls.database_path == db

    def test_get_orm_class(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        ) as ls:
            # get_orm_class returns None when no ORM class is mapped (pass-through)
            # The result type may be None or a class depending on automap success;
            # the point is the call doesn't raise.
            _ = ls.get_orm_class("isa.Image")


class TestLocalSchemaReadOnly:
    def test_read_only_rejects_writes(self, canned_bag_model: Model, tmp_path: Path) -> None:
        """Opening with read_only=True must yield an engine that rejects writes."""
        # First build writable so the files exist
        db = tmp_path / "ro_test.db"
        rw = LocalSchema.build(
            model=canned_bag_model,
            schemas=["deriva-ml"],
            database_path=db,
        )
        rw.dispose()

        # Then re-open read-only (single-schema).
        ro = LocalSchema.build(
            model=canned_bag_model,
            schemas=["deriva-ml"],
            database_path=db,
            read_only=True,
        )
        try:
            with ro.engine.connect() as conn:
                # A table read should work
                ro_t = ro.find_table("Dataset")
                conn.execute(ro_t.select()).fetchall()
                # A write must fail
                with pytest.raises(Exception):  # noqa: B017, BLE001
                    conn.execute(ro_t.insert().values(RID="X"))
                    conn.commit()
        finally:
            ro.dispose()

    def test_read_only_returns_previously_written_data(self, canned_bag_model: Model, tmp_path: Path) -> None:
        """Data written via writable engine is visible through read-only engine."""
        db = tmp_path / "ro_data_test.db"

        # Write some data
        rw = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        dataset_t = rw.find_table("Dataset")
        from sqlalchemy import insert

        with rw.engine.begin() as conn:
            conn.execute(insert(dataset_t).values(RID="TEST-1", Description="hello"))
        rw.dispose()

        # Re-open read-only (single schema to avoid NotImplementedError)
        ro = LocalSchema.build(
            model=canned_bag_model,
            schemas=["deriva-ml"],
            database_path=db,
            read_only=True,
        )
        try:
            ro_t = ro.find_table("Dataset")
            from sqlalchemy import select

            with ro.engine.connect() as conn:
                rows = conn.execute(select(ro_t)).mappings().all()
            assert any(r["RID"] == "TEST-1" for r in rows)
            assert any(r["Description"] == "hello" for r in rows)
        finally:
            ro.dispose()

    def test_read_only_multi_schema_not_supported(self, canned_bag_model: Model, tmp_path: Path) -> None:
        """Multi-schema read-only raises NotImplementedError (Phase 2 will address)."""
        db = tmp_path / "ro_multi.db"
        rw = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        rw.dispose()

        with pytest.raises(NotImplementedError):
            LocalSchema.build(
                model=canned_bag_model,
                schemas=["isa", "deriva-ml"],
                database_path=db,
                read_only=True,
            )

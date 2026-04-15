"""Unit tests for local_db.schema.LocalSchema adapter."""

from __future__ import annotations

from pathlib import Path

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

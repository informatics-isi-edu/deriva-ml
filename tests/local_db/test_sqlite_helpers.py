"""Unit tests for local_db.sqlite_helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from deriva_ml.local_db import sqlite_helpers as sh


class TestCreateWalEngine:
    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert result == "wal"
        finally:
            engine.dispose()

    def test_synchronous_normal(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            with engine.connect() as conn:
                # synchronous=NORMAL is integer 1 in SQLite
                result = conn.execute(text("PRAGMA synchronous")).scalar()
            assert result == 1
        finally:
            engine.dispose()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "sub" / "db.sqlite"
        assert not db.parent.exists()
        engine = sh.create_wal_engine(db)
        try:
            assert db.parent.exists()
        finally:
            engine.dispose()

    def test_read_only_mode(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        # First create the DB in RW mode
        rw = sh.create_wal_engine(db)
        with rw.connect() as conn:
            conn.execute(text("CREATE TABLE t (x INT)"))
            conn.commit()
        rw.dispose()

        # Then open read-only
        ro = sh.create_wal_engine(db, read_only=True)
        try:
            with ro.connect() as conn:
                # Reads work
                conn.execute(text("SELECT * FROM t")).fetchall()
                # Writes fail
                with pytest.raises(Exception):
                    conn.execute(text("INSERT INTO t VALUES (1)"))
                    conn.commit()
        finally:
            ro.dispose()


class TestAttachDetach:
    def test_attach_makes_tables_visible(self, tmp_path: Path) -> None:
        main_db = tmp_path / "main.sqlite"
        slice_db = tmp_path / "slice.sqlite"

        # Create main DB
        eng_main = sh.create_wal_engine(main_db)
        with eng_main.connect() as conn:
            conn.execute(text("CREATE TABLE main_t (x INT)"))
            conn.execute(text("INSERT INTO main_t VALUES (1)"))
            conn.commit()
        eng_main.dispose()

        # Create slice DB
        eng_slice = sh.create_wal_engine(slice_db)
        with eng_slice.connect() as conn:
            conn.execute(text("CREATE TABLE slice_t (y INT)"))
            conn.execute(text("INSERT INTO slice_t VALUES (99)"))
            conn.commit()
        eng_slice.dispose()

        # Attach slice into main
        eng = sh.create_wal_engine(main_db)
        try:
            with eng.connect() as conn:
                sh.attach_database(conn, slice_db, "slice")
                # Can see slice.slice_t
                result = conn.execute(text("SELECT y FROM slice.slice_t")).scalar()
                assert result == 99
                sh.detach_database(conn, "slice")
        finally:
            eng.dispose()


class TestSchemaMeta:
    def test_initial_version_is_1(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            version = sh.ensure_schema_meta(engine, expected_version=1)
            assert version == 1
        finally:
            engine.dispose()

    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            sh.ensure_schema_meta(engine, expected_version=1)
            # Second call should be a no-op
            v = sh.ensure_schema_meta(engine, expected_version=1)
            assert v == 1
        finally:
            engine.dispose()

    def test_raises_on_higher_version(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            # Pretend the DB was created by a newer deriva-ml
            sh.ensure_schema_meta(engine, expected_version=2)
            with pytest.raises(sh.SchemaVersionError):
                sh.ensure_schema_meta(engine, expected_version=1)
        finally:
            engine.dispose()

"""Unit tests for local_db.workspace.Workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from deriva_ml.local_db.workspace import Workspace


class TestWorkspaceCreation:
    def test_creates_working_db_file(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="host.example.org", catalog_id="1")
        # File is created lazily when an engine is first requested
        _ = ws.engine
        assert ws.working_db_path.parent.is_dir()
        assert ws.working_db_path.is_file()
        ws.close()

    def test_wal_mode(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="host", catalog_id="1")
        try:
            with ws.engine.connect() as conn:
                assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        finally:
            ws.close()

    def test_two_workspaces_different_catalogs_do_not_collide(self, tmp_path: Path) -> None:
        ws1 = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws2 = Workspace(working_dir=tmp_path, hostname="h", catalog_id="2")
        try:
            assert ws1.working_db_path != ws2.working_db_path
        finally:
            ws1.close()
            ws2.close()

    def test_root_property(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="myhost", catalog_id="42")
        root = ws.root
        assert root == tmp_path / "catalogs" / "myhost__42"
        ws.close()

    def test_slice_db_path_method(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        p = ws.slice_db_path("my_slice")
        assert p == tmp_path / "catalogs" / "h__1" / "slices" / "my_slice" / "slice.db"
        ws.close()

    def test_context_manager_protocol(self, tmp_path: Path) -> None:
        with Workspace(working_dir=tmp_path, hostname="h", catalog_id="1") as ws:
            _ = ws.engine
            assert ws.working_db_path.is_file()
        # After exiting the context manager, engine should be gone
        assert ws._engine is None
        assert ws._closed

    def test_engine_raises_after_close(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = ws.engine


class TestAttachSlice:
    def test_attach_context_manager(self, tmp_path: Path) -> None:
        # Create a real slice.db with a tiny table
        slice_db = tmp_path / "catalogs" / "h__1" / "slices" / "s1" / "slice.db"
        slice_db.parent.mkdir(parents=True, exist_ok=True)
        from deriva_ml.local_db.sqlite_helpers import create_wal_engine

        eng = create_wal_engine(slice_db)
        with eng.connect() as conn:
            conn.execute(text("CREATE TABLE t (x INT)"))
            conn.execute(text("INSERT INTO t VALUES (1), (2), (3)"))
            conn.commit()
        eng.dispose()

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with ws.attach_slice("s1") as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM slice.t")).scalar()
                assert count == 3
            # After leaving the context, 'slice' should no longer be attached.
            with ws.engine.connect() as conn:
                with pytest.raises(Exception):
                    conn.execute(text("SELECT COUNT(*) FROM slice.t"))
        finally:
            ws.close()

    def test_attach_missing_slice_raises(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with pytest.raises(FileNotFoundError):
                with ws.attach_slice("nonexistent"):
                    pass
        finally:
            ws.close()


class TestLegacyWorkingDataView:
    def test_cache_table_roundtrip(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            df = pd.DataFrame({"x": [1, 2, 3]})
            view.cache_table("mytable", df)
            got = view.read_table("mytable")
            assert list(got["x"]) == [1, 2, 3]
            assert view.has_table("mytable")
            assert "mytable" in view.list_tables()
        finally:
            ws.close()

    def test_query(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            df = pd.DataFrame({"a": [10, 20, 30]})
            view.cache_table("qtable", df)
            result = view.query("SELECT SUM(a) AS total FROM qtable")
            assert result["total"].iloc[0] == 60
        finally:
            ws.close()

    def test_drop_table(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            df = pd.DataFrame({"v": [1, 2]})
            view.cache_table("todelete", df)
            assert view.has_table("todelete")
            view.drop_table("todelete")
            assert not view.has_table("todelete")
        finally:
            ws.close()

    def test_clear(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            view.cache_table("t1", pd.DataFrame({"x": [1]}))
            view.cache_table("t2", pd.DataFrame({"y": [2]}))
            assert view.has_table("t1")
            assert view.has_table("t2")
            view.clear()
            # After clear, user tables should be gone; schema_meta may remain
            user_tables = [t for t in view.list_tables() if t != "schema_meta"]
            assert user_tables == []
        finally:
            ws.close()

    def test_workspace_property(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            assert view.workspace is ws
        finally:
            ws.close()

    def test_has_table_returns_false_before_db_exists(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            # The DB file doesn't exist yet (engine not yet accessed)
            assert not ws.working_db_path.exists()
            assert not view.has_table("anything")
        finally:
            ws.close()

    def test_list_tables_returns_empty_before_db_exists(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            assert not ws.working_db_path.exists()
            assert view.list_tables() == []
        finally:
            ws.close()

    def test_read_table_raises_when_missing(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            with pytest.raises(ValueError, match="not found"):
                view.read_table("does_not_exist")
        finally:
            ws.close()


class TestWorkspaceClose:
    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws.close()
        ws.close()  # Should not raise


class TestDerivaMLIntegration:
    def test_ml_working_data_uses_workspace_path(self, tmp_path: Path) -> None:
        """DerivaML.working_data should write to catalogs/{host}__{cat}/working.db."""
        from deriva_ml import DerivaML
        import pandas as pd

        ml = DerivaML.__new__(DerivaML)
        ml.working_dir = tmp_path
        ml.host_name = "example.org"
        ml.catalog_id = "9"

        wd = ml.working_data
        wd.cache_table("demo", pd.DataFrame({"x": [1]}))

        expected = tmp_path / "catalogs" / "example.org__9" / "working.db"
        assert expected.is_file()

"""Unit tests for local_db.workspace.Workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from deriva_ml.local_db.workspace import Workspace


class TestWorkspaceCreation:
    def test_creates_working_db_file(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="host.example.org", catalog_id="1")
        # File is created lazily when an engine is first requested
        _ = ws.engine
        assert ws.working_db_path.is_dir()
        assert (ws.working_db_path / "main.db").is_file()
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

    def test_slice_db_path_method_returns_directory(self, tmp_path: Path) -> None:
        """slice_db_path is now an alias for the slice directory."""
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        result = ws.slice_db_path("my_slice")
        assert result == tmp_path / "catalogs" / "h__1" / "slices" / "my_slice"
        ws.close()

    def test_context_manager_protocol(self, tmp_path: Path) -> None:
        with Workspace(working_dir=tmp_path, hostname="h", catalog_id="1") as ws:
            _ = ws.engine
            assert (ws.working_db_path / "main.db").is_file()
        # After exiting the context manager, engine should be gone
        assert ws._engine is None
        assert ws._closed

    def test_engine_raises_after_close(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = ws.engine

    def test_engine_is_memoized(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            assert ws.engine is ws.engine
        finally:
            ws.close()

    def test_manifest_store_is_memoized(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            assert ws.manifest_store() is ws.manifest_store()
        finally:
            ws.close()


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


class TestMultiSchemaSliceAttach:
    """Tests for multi-schema (directory-based) slice attachment."""

    def test_attach_multi_schema_slice(self, tmp_path: Path, canned_bag_model) -> None:
        """Multi-schema slice: each per-schema file is ATTACH'd under slice_{stem}."""
        from sqlalchemy import insert

        from deriva_ml.local_db.schema import LocalSchema

        # Build a multi-schema slice directory
        s_dir = tmp_path / "catalogs" / "h__1" / "slices" / "ms1"
        s_dir.mkdir(parents=True)
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=s_dir,
        )
        image_t = ls.find_table("isa.Image")
        with ls.engine.begin() as conn:
            conn.execute(insert(image_t).values(RID="SLICE-IMG-1", Filename="test.jpg"))
        ls.dispose()

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            with ws.attach_slice("ms1") as conn:
                result = conn.execute(text("SELECT RID FROM slice_isa.Image")).fetchall()
                assert len(result) == 1
                assert result[0][0] == "SLICE-IMG-1"
        finally:
            ws.close()

    def test_attach_detaches_all_on_exit(self, tmp_path: Path, canned_bag_model) -> None:
        """After context exit, all slice schemas are detached."""
        from deriva_ml.local_db.schema import LocalSchema

        s_dir = tmp_path / "catalogs" / "h__1" / "slices" / "ms1"
        s_dir.mkdir(parents=True)
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=s_dir,
        )
        ls.dispose()

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            with ws.attach_slice("ms1") as conn:
                pass  # just enter and exit
            # After exit, slice_isa should not be accessible
            with ws.engine.connect() as conn:
                with pytest.raises(Exception):
                    conn.execute(text("SELECT * FROM slice_isa.Image"))
        finally:
            ws.close()

    def test_missing_slice_dir_raises(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with pytest.raises(FileNotFoundError):
                with ws.attach_slice("nonexistent"):
                    pass
        finally:
            ws.close()

    def test_legacy_single_file_slice_still_works(self, tmp_path: Path) -> None:
        """A slice with only slice.db (Phase 1 layout) still works under alias 'slice'."""
        from deriva_ml.local_db.sqlite_helpers import create_wal_engine

        s_dir = tmp_path / "catalogs" / "h__1" / "slices" / "legacy1"
        s_dir.mkdir(parents=True)
        eng = create_wal_engine(s_dir / "slice.db")
        with eng.connect() as conn:
            conn.execute(text("CREATE TABLE t (x INT)"))
            conn.execute(text("INSERT INTO t VALUES (42)"))
            conn.commit()
        eng.dispose()

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with ws.attach_slice("legacy1") as conn:
                val = conn.execute(text("SELECT x FROM slice.t")).scalar()
                assert val == 42
        finally:
            ws.close()

    def test_empty_slice_dir_raises(self, tmp_path: Path) -> None:
        """A slice directory with no .db file raises FileNotFoundError."""
        s_dir = tmp_path / "catalogs" / "h__1" / "slices" / "empty1"
        s_dir.mkdir(parents=True)

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with pytest.raises(FileNotFoundError, match="expected main.db or slice.db"):
                with ws.attach_slice("empty1"):
                    pass
        finally:
            ws.close()


class TestLocalSchema:
    def test_local_schema_is_none_before_build(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            assert ws.local_schema is None
        finally:
            ws.close()

    def test_build_local_schema_from_model(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(
                model=canned_bag_model,
                schemas=["isa", "deriva-ml"],
            )
            assert ws.local_schema is not None
            assert ws.orm_class("Image") is not None
            assert ws.orm_class("Dataset") is not None
        finally:
            ws.close()

    def test_local_schema_creates_per_schema_files(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(
                model=canned_bag_model,
                schemas=["isa", "deriva-ml"],
            )
            working_dir = tmp_path / "catalogs" / "h__1" / "working"
            assert (working_dir / "isa.db").is_file()
            assert (working_dir / "deriva-ml.db").is_file()
        finally:
            ws.close()

    def test_rebuild_schema_disposes_and_recreates(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            first = ws.local_schema
            ws.rebuild_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            second = ws.local_schema
            assert first is not second
        finally:
            ws.close()

    def test_orm_class_returns_none_for_unknown(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            result = ws.orm_class("NonexistentTable")
            assert result is None
        finally:
            ws.close()

    def test_orm_class_none_without_schema(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            assert ws.orm_class("Image") is None
        finally:
            ws.close()

    def test_engine_unified_with_local_schema(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            assert ws.engine is ws.local_schema.engine
        finally:
            ws.close()

    def test_manifest_store_works_after_schema_build(self, tmp_path: Path, canned_bag_model) -> None:
        """ManifestStore should work on the unified engine after schema build."""
        from deriva_ml.asset.manifest import AssetEntry

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            store = ws.manifest_store()
            store.add_asset("EX1", "Image/a.jpg", AssetEntry(asset_table="Image", schema="isa"))
            rows = store.list_assets("EX1")
            assert "Image/a.jpg" in rows
        finally:
            ws.close()

    def test_manifest_store_before_schema_build(self, tmp_path: Path) -> None:
        """ManifestStore should work even before schema build."""
        from deriva_ml.asset.manifest import AssetEntry

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            store = ws.manifest_store()
            store.add_asset("EX1", "Image/a.jpg", AssetEntry(asset_table="Image", schema="isa"))
            rows = store.list_assets("EX1")
            assert "Image/a.jpg" in rows
        finally:
            ws.close()


class TestWorkspaceClose:
    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws.close()
        ws.close()  # Should not raise


class TestWorkspaceCachedReads:
    def test_cached_table_read(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            # Insert test data
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert

            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))
                conn.execute(insert(subject_t).values(RID="S2", Name="Bob"))

            result = ws.cached_table_read("Subject", source="local")
            assert result.row_count == 2
            df = result.to_dataframe()
            assert len(df) == 2
            assert "Name" in df.columns
        finally:
            ws.close()

    def test_cached_table_read_uses_cache(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert

            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))

            r1 = ws.cached_table_read("Subject", source="local")
            r2 = ws.cached_table_read("Subject", source="local")
            assert r1.cache_key == r2.cache_key
        finally:
            ws.close()

    def test_list_cached_results(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert

            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))

            ws.cached_table_read("Subject", source="local")
            results = ws.list_cached_results()
            assert len(results) >= 1
        finally:
            ws.close()

    def test_invalidate_cache(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert

            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))

            cr = ws.cached_table_read("Subject", source="local")
            ws.invalidate_cache(cache_key=cr.cache_key)
            assert len(ws.list_cached_results()) == 0
        finally:
            ws.close()

    def test_cached_table_read_requires_schema(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with pytest.raises(RuntimeError, match="local_schema not built"):
                ws.cached_table_read("Subject")
        finally:
            ws.close()

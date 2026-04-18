"""Unit tests for local_db.manifest_store.ManifestStore."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect

from deriva_ml.asset.manifest import AssetEntry, FeatureEntry
from deriva_ml.local_db.manifest_store import ManifestStore
from deriva_ml.local_db.workspace import Workspace


@pytest.fixture
def store(tmp_path: Path):
    ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
    s = ManifestStore(ws.engine)
    s.ensure_schema()
    yield s
    ws.close()


class TestEnsureSchema:
    def test_creates_tables(self, store: ManifestStore) -> None:
        tables = inspect(store._engine).get_table_names()
        assert "execution_state__assets" in tables
        assert "execution_state__features" in tables

    def test_idempotent(self, store: ManifestStore) -> None:
        store.ensure_schema()
        store.ensure_schema()  # second call should be a no-op


class TestAssetCrud:
    def test_add_and_get_asset(self, store: ManifestStore) -> None:
        entry = AssetEntry(
            asset_table="Image",
            schema="isa",
            asset_types=["Training"],
            metadata={"Subject": "2-ABC"},
            description="test",
        )
        store.add_asset("4SP", "Image/scan.jpg", entry)
        got = store.get_asset("4SP", "Image/scan.jpg")
        assert got.asset_table == "Image"
        assert got.asset_types == ["Training"]
        assert got.metadata == {"Subject": "2-ABC"}
        assert got.status == "pending"

    def test_list_assets_for_execution(self, store: ManifestStore) -> None:
        e1 = AssetEntry(asset_table="Image", schema="isa")
        e2 = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("exec-A", "Image/a.jpg", e1)
        store.add_asset("exec-B", "Image/b.jpg", e2)
        a_only = store.list_assets("exec-A")
        assert set(a_only.keys()) == {"Image/a.jpg"}

    def test_mark_uploaded(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.mark_asset_uploaded("4SP", "Image/x.jpg", rid="1-RID")
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.status == "uploaded"
        assert got.rid == "1-RID"
        assert got.uploaded_at is not None

    def test_mark_failed(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.mark_asset_failed("4SP", "Image/x.jpg", error="nope")
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.status == "failed"
        assert got.error == "nope"

    def test_update_metadata(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa", metadata={})
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.update_asset_metadata("4SP", "Image/x.jpg", {"Subject": "ABC"})
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.metadata == {"Subject": "ABC"}

    def test_update_types(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa", asset_types=["A"])
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.update_asset_types("4SP", "Image/x.jpg", ["B", "C"])
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.asset_types == ["B", "C"]

    def test_get_missing_asset_raises(self, store: ManifestStore) -> None:
        with pytest.raises(KeyError):
            store.get_asset("4SP", "nonexistent")

    def test_add_asset_replaces_existing(self, store: ManifestStore) -> None:
        e1 = AssetEntry(asset_table="Image", schema="isa", description="v1")
        store.add_asset("4SP", "Image/x.jpg", e1)
        e2 = AssetEntry(asset_table="Image", schema="isa", description="v2")
        store.add_asset("4SP", "Image/x.jpg", e2)
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.description == "v2"

    def test_update_metadata_missing_raises(self, store: ManifestStore) -> None:
        with pytest.raises(KeyError):
            store.update_asset_metadata("4SP", "nope", {})

    def test_mark_uploaded_missing_raises(self, store: ManifestStore) -> None:
        with pytest.raises(KeyError):
            store.mark_asset_uploaded("4SP", "nope", rid="X")

    def test_mark_failed_after_uploaded(self, store: ManifestStore) -> None:
        """Marking a previously-uploaded asset as failed should work (last write wins)."""
        entry = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.mark_asset_uploaded("4SP", "Image/x.jpg", rid="1-RID")
        store.mark_asset_failed("4SP", "Image/x.jpg", error="re-upload failed")
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.status == "failed"
        assert got.error == "re-upload failed"
        # The uploaded_at and rid from the previous upload should be preserved
        # (mark_failed only sets status, error, updated_at)
        assert got.rid == "1-RID"

    def test_update_types_missing_raises(self, store: ManifestStore) -> None:
        with pytest.raises(KeyError):
            store.update_asset_types("4SP", "nonexistent", ["A"])


class TestFeatureCrud:
    def test_add_and_list(self, store: ManifestStore) -> None:
        f = FeatureEntry(
            feature_name="Diagnosis",
            target_table="Image",
            schema="isa",
            values_path="/some/path.csv",
            asset_columns={},
            status="pending",
        )
        store.add_feature("4SP", "Diagnosis", f)
        got = store.list_features("4SP")
        assert "Diagnosis" in got
        assert got["Diagnosis"].target_table == "Image"

    def test_list_features_empty(self, store: ManifestStore) -> None:
        got = store.list_features("nonexistent")
        assert got == {}

    def test_add_feature_replaces_existing(self, store: ManifestStore) -> None:
        f1 = FeatureEntry(feature_name="D", target_table="Image", schema="isa", values_path="/v1.csv", status="pending")
        store.add_feature("4SP", "D", f1)
        f2 = FeatureEntry(feature_name="D", target_table="Image", schema="isa", values_path="/v2.csv", status="pending")
        store.add_feature("4SP", "D", f2)
        got = store.list_features("4SP")
        assert got["D"].values_path == "/v2.csv"


class TestStatusFilters:
    def test_pending_only(self, store: ManifestStore) -> None:
        e1 = AssetEntry(asset_table="Image", schema="isa")
        e2 = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "a", e1)
        store.add_asset("4SP", "b", e2)
        store.mark_asset_uploaded("4SP", "b", "1-RID")
        pending = store.pending_assets("4SP")
        assert set(pending.keys()) == {"a"}

    def test_uploaded_only(self, store: ManifestStore) -> None:
        e1 = AssetEntry(asset_table="Image", schema="isa")
        e2 = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "a", e1)
        store.add_asset("4SP", "b", e2)
        store.mark_asset_uploaded("4SP", "b", "1-RID")
        uploaded = store.uploaded_assets("4SP")
        assert set(uploaded.keys()) == {"b"}

    def test_no_assets_returns_empty(self, store: ManifestStore) -> None:
        assert store.pending_assets("nonexistent") == {}
        assert store.uploaded_assets("nonexistent") == {}

"""Integration test: JSON manifest migration on workspace open."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deriva_ml.local_db.workspace import Workspace


@pytest.fixture
def legacy_manifest_dir(tmp_path: Path) -> Path:
    """Seed a legacy execution directory with an asset-manifest.json."""
    exec_dir = tmp_path / "execution" / "2-ABC"
    exec_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "execution_rid": "2-ABC",
        "created_at": "2026-04-15T00:00:00Z",
        "assets": {
            "Image/scan.jpg": {
                "asset_table": "Image",
                "schema": "isa",
                "asset_types": ["Training"],
                "metadata": {"Subject": "S-1"},
                "description": "",
                "status": "pending",
                "rid": None,
                "uploaded_at": None,
                "error": None,
            }
        },
        "features": {},
    }
    (exec_dir / "asset-manifest.json").write_text(json.dumps(manifest))
    return exec_dir


class TestManifestImport:
    def test_imports_legacy_manifests(self, tmp_path: Path, legacy_manifest_dir: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 1

            store = ws.manifest_store()
            rows = store.list_assets("2-ABC")
            assert "Image/scan.jpg" in rows
            assert rows["Image/scan.jpg"].asset_types == ["Training"]

            # Sidecar file created
            assert (legacy_manifest_dir / "asset-manifest.json.migrated.json").is_file()
            # Original removed
            assert not (legacy_manifest_dir / "asset-manifest.json").exists()
        finally:
            ws.close()

    def test_idempotent(self, tmp_path: Path, legacy_manifest_dir: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count1 = ws.import_legacy_manifests()
            count2 = ws.import_legacy_manifests()
            assert count1 == 1
            assert count2 == 0  # Already migrated
        finally:
            ws.close()

    def test_no_manifests_no_sidecar(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 0
        finally:
            ws.close()

    def test_multiple_manifests_imported(self, tmp_path: Path) -> None:
        """Two different execution directories each with a manifest."""
        for exec_rid in ["X-1", "X-2"]:
            d = tmp_path / "execution" / exec_rid
            d.mkdir(parents=True)
            (d / "asset-manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "execution_rid": exec_rid,
                        "assets": {
                            "Image/a.jpg": {
                                "asset_table": "Image",
                                "schema": "isa",
                                "status": "pending",
                            }
                        },
                        "features": {},
                    }
                )
            )

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 2
            store = ws.manifest_store()
            assert "Image/a.jpg" in store.list_assets("X-1")
            assert "Image/a.jpg" in store.list_assets("X-2")
        finally:
            ws.close()

    def test_malformed_json_skipped_with_warning(self, tmp_path: Path) -> None:
        """A malformed JSON file should be skipped, not crash the import."""
        d = tmp_path / "execution" / "BAD"
        d.mkdir(parents=True)
        (d / "asset-manifest.json").write_text("NOT JSON {{{")

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 0
            # The file should still exist (not renamed — failed to parse)
            assert (d / "asset-manifest.json").exists()
        finally:
            ws.close()

    def test_missing_execution_rid_falls_back_to_parent_name(self, tmp_path: Path) -> None:
        """When the JSON has no execution_rid key, use the parent directory name."""
        d = tmp_path / "execution" / "PARENT-DIR-RID"
        d.mkdir(parents=True)
        (d / "asset-manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    # No "execution_rid" key
                    "assets": {
                        "Image/a.jpg": {
                            "asset_table": "Image",
                            "schema": "isa",
                            "status": "pending",
                        }
                    },
                    "features": {},
                }
            )
        )

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 1
            rows = ws.manifest_store().list_assets("PARENT-DIR-RID")
            assert "Image/a.jpg" in rows
        finally:
            ws.close()

    def test_feature_entries_in_legacy_manifest_are_silently_skipped(self, tmp_path: Path) -> None:
        """Features in the legacy manifest are silently skipped (file-based path retired)."""
        d = tmp_path / "execution" / "F-1"
        d.mkdir(parents=True)
        (d / "asset-manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "execution_rid": "F-1",
                    "assets": {},
                    "features": {
                        "Diagnosis": {
                            "feature_name": "Diagnosis",
                            "target_table": "Image",
                            "schema": "isa",
                            "values_path": "/path/to/values.csv",
                            "asset_columns": {},
                            "status": "pending",
                        }
                    },
                }
            )
        )

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            # The manifest is still migrated (sidecar is created) even though
            # the legacy "features" entries are ignored — the migration counts assets.
            count = ws.import_legacy_manifests()
            assert count == 1
            # No feature_records should have been created
            records = ws.manifest_store().list_feature_records("F-1")
            assert records == []
        finally:
            ws.close()

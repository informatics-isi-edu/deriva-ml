"""Tests for AssetManifest, AssetRecord, and AssetFilePath metadata behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deriva_ml.asset.manifest import AssetManifest, AssetEntry, FeatureEntry
from deriva_ml.asset.aux_classes import AssetFilePath


# =============================================================================
# AssetManifest Tests
# =============================================================================


class TestAssetManifest:
    """Tests for the AssetManifest persistence layer."""

    def test_create_empty_manifest(self, tmp_path):
        """Test creating a new empty manifest."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")

        assert mp.exists()
        assert manifest.execution_rid == "4SP"
        assert manifest.assets == {}
        assert manifest.features == {}

    def test_add_asset(self, tmp_path):
        """Test adding an asset entry."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")

        entry = AssetEntry(
            asset_table="Image",
            schema="test-schema",
            asset_types=["Training_Data"],
            metadata={"Subject": "2-DEF"},
        )
        manifest.add_asset("Image/scan.jpg", entry)

        assert "Image/scan.jpg" in manifest.assets
        assert manifest.assets["Image/scan.jpg"].asset_table == "Image"
        assert manifest.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}
        assert manifest.assets["Image/scan.jpg"].status == "pending"

    def test_manifest_persists_to_disk(self, tmp_path):
        """Test that manifest survives reload (crash recovery)."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
            asset_types=["Training_Data"],
            metadata={"Subject": "2-DEF"},
        ))

        # Reload from disk (simulates crash recovery)
        manifest2 = AssetManifest(mp, "4SP")
        assert "Image/scan.jpg" in manifest2.assets
        assert manifest2.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}

    def test_mark_uploaded(self, tmp_path):
        """Test marking an asset as uploaded."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        manifest.mark_uploaded("Image/scan.jpg", "1-ABC")

        entry = manifest.assets["Image/scan.jpg"]
        assert entry.status == "uploaded"
        assert entry.rid == "1-ABC"
        assert entry.uploaded_at is not None

    def test_mark_failed(self, tmp_path):
        """Test marking an asset as failed."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        manifest.mark_failed("Image/scan.jpg", "upload timeout")

        entry = manifest.assets["Image/scan.jpg"]
        assert entry.status == "failed"
        assert entry.error == "upload timeout"

    def test_pending_and_uploaded_filters(self, tmp_path):
        """Test filtering assets by status."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")

        manifest.add_asset("Image/a.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest.add_asset("Image/b.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest.add_asset("Image/c.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest.mark_uploaded("Image/a.jpg", "1-A")

        pending = manifest.pending_assets()
        uploaded = manifest.uploaded_assets()

        assert len(pending) == 2
        assert len(uploaded) == 1
        assert "Image/a.jpg" in uploaded
        assert "Image/b.jpg" in pending

    def test_update_metadata(self, tmp_path):
        """Test updating metadata for an existing asset."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        manifest.update_asset_metadata("Image/scan.jpg", {"Subject": "2-DEF"})
        assert manifest.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}

        # Verify persisted
        manifest2 = AssetManifest(mp, "4SP")
        assert manifest2.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}

    def test_update_asset_types(self, tmp_path):
        """Test updating asset types."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        manifest.update_asset_types("Image/scan.jpg", ["Training_Data", "Labeled"])
        assert manifest.assets["Image/scan.jpg"].asset_types == ["Training_Data", "Labeled"]

    def test_missing_key_raises(self, tmp_path):
        """Test that operations on non-existent keys raise KeyError."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")

        with pytest.raises(KeyError):
            manifest.mark_uploaded("nonexistent", "1-A")

        with pytest.raises(KeyError):
            manifest.update_asset_metadata("nonexistent", {})

    def test_add_feature(self, tmp_path):
        """Test adding a feature entry."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")

        manifest.add_feature("Diagnosis", FeatureEntry(
            feature_name="Diagnosis",
            target_table="Image",
            schema="test-schema",
            values_path="features/Image/Diagnosis/Diagnosis.jsonl",
        ))

        assert "Diagnosis" in manifest.features
        assert manifest.features["Diagnosis"].target_table == "Image"

    def test_manifest_json_format(self, tmp_path):
        """Test that the manifest JSON is well-formed and readable."""
        mp = tmp_path / "asset-manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
            asset_types=["Training_Data"],
        ))

        with open(mp) as f:
            data = json.load(f)

        assert data["version"] == 1
        assert data["execution_rid"] == "4SP"
        assert "Image/scan.jpg" in data["assets"]
        assert data["assets"]["Image/scan.jpg"]["status"] == "pending"

    def test_resume_after_crash(self, tmp_path):
        """Test resume scenario: some assets uploaded, some pending."""
        mp = tmp_path / "asset-manifest.json"

        # First "session" — register and partially upload
        manifest1 = AssetManifest(mp, "4SP")
        manifest1.add_asset("Image/a.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest1.add_asset("Image/b.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest1.add_asset("Image/c.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest1.mark_uploaded("Image/a.jpg", "1-A")
        # "Crash" — manifest1 goes out of scope

        # Second "session" — reload and check state
        manifest2 = AssetManifest(mp, "4SP")
        pending = manifest2.pending_assets()
        uploaded = manifest2.uploaded_assets()

        assert len(pending) == 2
        assert len(uploaded) == 1
        assert uploaded["Image/a.jpg"].rid == "1-A"


# =============================================================================
# AssetFilePath Tests
# =============================================================================


class TestAssetFilePathBehavior:
    """Tests that AssetFilePath still behaves as a Path object."""

    def test_behaves_as_path(self, tmp_path):
        """Test that AssetFilePath can be used like a regular Path."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello")

        afp = AssetFilePath(
            asset_path=file_path,
            asset_table="Execution_Asset",
            file_name="test.txt",
            asset_metadata={},
            asset_types=["Test"],
        )

        # Path operations should work
        assert afp.exists()
        assert afp.name == "test.txt"
        assert afp.read_text() == "hello"
        assert afp.suffix == ".txt"
        assert afp.stem == "test"

    def test_write_and_read(self, tmp_path):
        """Test writing to and reading from AssetFilePath."""
        file_path = tmp_path / "output.txt"
        afp = AssetFilePath(
            asset_path=file_path,
            asset_table="Model",
            file_name="output.txt",
            asset_metadata={},
            asset_types=["Model_Weights"],
        )

        afp.write_text("model weights")
        assert afp.read_text() == "model weights"

    def test_path_parent_and_resolve(self, tmp_path):
        """Test that parent and resolve work correctly."""
        file_path = tmp_path / "subdir" / "test.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("data")

        afp = AssetFilePath(
            asset_path=file_path,
            asset_table="Image",
            file_name="test.txt",
            asset_metadata={},
            asset_types=["Image"],
        )

        # with_segments returns plain Path (required for Path subclass compat)
        parent = afp.parent
        assert isinstance(parent, Path)

    def test_asset_attributes_preserved(self, tmp_path):
        """Test that asset-specific attributes are preserved."""
        file_path = tmp_path / "scan.jpg"

        afp = AssetFilePath(
            asset_path=file_path,
            asset_table="Image",
            file_name="scan.jpg",
            asset_metadata={"Subject": "2-DEF"},
            asset_types=["Training_Data", "Labeled"],
            asset_rid="1-ABC",
        )

        assert afp.asset_table == "Image"
        assert afp.file_name == "scan.jpg"
        assert afp.asset_metadata == {"Subject": "2-DEF"}
        assert afp.asset_types == ["Training_Data", "Labeled"]
        assert afp.asset_rid == "1-ABC"
        assert afp.asset_name == "Image"  # backward compat alias

    def test_metadata_property_without_manifest(self, tmp_path):
        """Test metadata property when not bound to manifest."""
        afp = AssetFilePath(
            asset_path=tmp_path / "file.txt",
            asset_table="Image",
            file_name="file.txt",
            asset_metadata={"Subject": "2-DEF"},
            asset_types=["Image"],
        )

        assert afp.metadata == {"Subject": "2-DEF"}

    def test_metadata_property_with_manifest(self, tmp_path):
        """Test metadata property with manifest binding."""
        mp = tmp_path / "manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/file.txt", AssetEntry(
            asset_table="Image", schema="test-schema",
            metadata={"Subject": "2-DEF"},
        ))

        afp = AssetFilePath(
            asset_path=tmp_path / "file.txt",
            asset_table="Image",
            file_name="file.txt",
            asset_metadata={"Subject": "2-DEF"},
            asset_types=["Image"],
        )
        afp._bind_manifest(manifest, "Image/file.txt")

        # Update metadata via setter
        afp.metadata = {"Subject": "3-GHI", "Date": "2026-01-15"}

        # Both local and manifest should be updated
        assert afp.asset_metadata == {"Subject": "3-GHI", "Date": "2026-01-15"}
        assert manifest.assets["Image/file.txt"].metadata == {"Subject": "3-GHI", "Date": "2026-01-15"}

    def test_set_asset_types_with_manifest(self, tmp_path):
        """Test set_asset_types updates manifest."""
        mp = tmp_path / "manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/file.txt", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        afp = AssetFilePath(
            asset_path=tmp_path / "file.txt",
            asset_table="Image",
            file_name="file.txt",
            asset_metadata={},
            asset_types=["Image"],
        )
        afp._bind_manifest(manifest, "Image/file.txt")

        afp.set_asset_types(["Training_Data", "Labeled"])

        assert afp.asset_types == ["Training_Data", "Labeled"]
        assert manifest.assets["Image/file.txt"].asset_types == ["Training_Data", "Labeled"]


# =============================================================================
# AssetRecord Tests
# =============================================================================


class TestAssetRecord:
    """Tests for AssetRecord base class and dynamic generation."""

    def test_base_record_creation(self):
        """Test creating a base AssetRecord."""
        from deriva_ml.asset.asset_record import AssetRecord

        record = AssetRecord()
        assert record.model_dump() == {}

    def test_base_record_rejects_extra_fields(self):
        """Test that base record rejects unknown fields."""
        from deriva_ml.asset.asset_record import AssetRecord

        with pytest.raises(Exception):  # Pydantic ValidationError
            AssetRecord(unknown_field="value")

    def test_column_type_mapping(self):
        """Test that column types map correctly."""
        from deriva_ml.asset.asset_record import _map_column_type

        assert _map_column_type("text") is str
        assert _map_column_type("int4") is int
        assert _map_column_type("float8") is float
        assert _map_column_type("boolean") is bool
        assert _map_column_type("date") is str
        assert _map_column_type("timestamp") is str

    def test_record_model_dump(self):
        """Test that model_dump produces a clean dict for manifest storage."""
        from pydantic import create_model
        from deriva_ml.asset.asset_record import AssetRecord

        TestRecord = create_model(
            "TestRecord",
            __base__=AssetRecord,
            Subject=(str, ...),
            Score=(float, None),
        )

        record = TestRecord(Subject="2-DEF", Score=0.95)
        dump = record.model_dump()
        assert dump == {"Subject": "2-DEF", "Score": 0.95}

    def test_record_optional_fields(self):
        """Test that optional fields default to None."""
        from typing import Optional
        from pydantic import create_model
        from deriva_ml.asset.asset_record import AssetRecord

        TestRecord = create_model(
            "TestRecord",
            __base__=AssetRecord,
            Name=(str, ...),
            Notes=(Optional[str], None),
        )

        record = TestRecord(Name="test")
        assert record.Notes is None
        assert record.model_dump() == {"Name": "test", "Notes": None}


# =============================================================================
# Integration: AssetRecord with AssetFilePath metadata
# =============================================================================


class TestAssetRecordWithFilePath:
    """Test using AssetRecord with AssetFilePath metadata setter."""

    def test_set_metadata_from_record(self, tmp_path):
        """Test setting AssetFilePath metadata from an AssetRecord."""
        from pydantic import create_model
        from deriva_ml.asset.asset_record import AssetRecord

        ImageAsset = create_model(
            "ImageAsset",
            __base__=AssetRecord,
            Subject=(str, ...),
            Acquisition_Date=(str, ...),
        )

        mp = tmp_path / "manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        afp = AssetFilePath(
            asset_path=tmp_path / "scan.jpg",
            asset_table="Image",
            file_name="scan.jpg",
            asset_metadata={},
            asset_types=["Image"],
        )
        afp._bind_manifest(manifest, "Image/scan.jpg")

        # Set metadata via AssetRecord
        record = ImageAsset(Subject="2-DEF", Acquisition_Date="2026-01-15")
        afp.metadata = record

        assert afp.metadata == {"Subject": "2-DEF", "Acquisition_Date": "2026-01-15"}
        assert manifest.assets["Image/scan.jpg"].metadata == {
            "Subject": "2-DEF",
            "Acquisition_Date": "2026-01-15",
        }

    def test_set_metadata_from_dict(self, tmp_path):
        """Test setting metadata from a plain dict (backward compat)."""
        mp = tmp_path / "manifest.json"
        manifest = AssetManifest(mp, "4SP")
        manifest.add_asset("Image/scan.jpg", AssetEntry(
            asset_table="Image", schema="test-schema",
        ))

        afp = AssetFilePath(
            asset_path=tmp_path / "scan.jpg",
            asset_table="Image",
            file_name="scan.jpg",
            asset_metadata={},
            asset_types=["Image"],
        )
        afp._bind_manifest(manifest, "Image/scan.jpg")

        afp.metadata = {"Subject": "2-DEF"}
        assert afp.metadata == {"Subject": "2-DEF"}

"""Tests for AssetManifest, AssetRecord, and AssetFilePath metadata behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from deriva_ml.asset.aux_classes import AssetFilePath
from deriva_ml.asset.manifest import AssetEntry, AssetManifest
from deriva_ml.local_db.manifest_store import ManifestStore

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store(tmp_path):
    """Create a ManifestStore backed by a temp SQLite DB."""
    engine = create_engine(f"sqlite:///{tmp_path / 'ws.sqlite'}", future=True)
    s = ManifestStore(engine)
    s.ensure_schema()
    yield s
    engine.dispose()


@pytest.fixture
def manifest(store):
    """Create an AssetManifest backed by the store."""
    return AssetManifest(store, "4SP")


# =============================================================================
# AssetManifest Tests
# =============================================================================


class TestAssetManifest:
    """Tests for the AssetManifest persistence layer."""

    def test_create_empty_manifest(self, manifest):
        """Test creating a new empty manifest."""
        assert manifest.execution_rid == "4SP"
        assert manifest.assets == {}

    def test_add_asset(self, manifest):
        """Test adding an asset entry."""
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

    def test_manifest_persists_across_instances(self, store):
        """Test that manifest data is visible to a second instance (crash recovery)."""
        manifest1 = AssetManifest(store, "4SP")
        manifest1.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
                asset_types=["Training_Data"],
                metadata={"Subject": "2-DEF"},
            ),
        )

        # Second instance over the same store should see the data
        manifest2 = AssetManifest(store, "4SP")
        assert "Image/scan.jpg" in manifest2.assets
        assert manifest2.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}

    def test_mark_uploaded(self, manifest):
        """Test marking an asset as uploaded."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

        manifest.mark_uploaded("Image/scan.jpg", "1-ABC")

        entry = manifest.assets["Image/scan.jpg"]
        assert entry.status == "uploaded"
        assert entry.rid == "1-ABC"
        assert entry.uploaded_at is not None

    def test_mark_failed(self, manifest):
        """Test marking an asset as failed."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

        manifest.mark_failed("Image/scan.jpg", "upload timeout")

        entry = manifest.assets["Image/scan.jpg"]
        assert entry.status == "failed"
        assert entry.error == "upload timeout"

    def test_pending_and_uploaded_filters(self, manifest):
        """Test filtering assets by status."""
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

    def test_update_metadata(self, store):
        """Test updating metadata for an existing asset."""
        manifest = AssetManifest(store, "4SP")
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

        manifest.update_asset_metadata("Image/scan.jpg", {"Subject": "2-DEF"})
        assert manifest.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}

        # Verify a new instance sees the update
        manifest2 = AssetManifest(store, "4SP")
        assert manifest2.assets["Image/scan.jpg"].metadata == {"Subject": "2-DEF"}

    def test_update_asset_types(self, manifest):
        """Test updating asset types."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

        manifest.update_asset_types("Image/scan.jpg", ["Training_Data", "Labeled"])
        assert manifest.assets["Image/scan.jpg"].asset_types == ["Training_Data", "Labeled"]

    def test_missing_key_raises(self, manifest):
        """Test that operations on non-existent keys raise KeyError."""
        with pytest.raises(KeyError):
            manifest.mark_uploaded("nonexistent", "1-A")

        with pytest.raises(KeyError):
            manifest.update_asset_metadata("nonexistent", {})

    def test_get_asset_point_query(self, manifest):
        """``get_asset`` returns a single entry without fetching the rest.

        Regression test for the perf fix that swapped
        ``manifest.assets.get(key)`` (full SELECT for the execution) for
        a point query in ``AssetFilePath.metadata``. Stage three rows
        and ensure ``get_asset`` returns just the requested one with
        the right metadata.
        """
        manifest.add_asset(
            "Image/a.jpg",
            AssetEntry(asset_table="Image", schema="s", metadata={"Subject": "2-A"}),
        )
        manifest.add_asset(
            "Image/b.jpg",
            AssetEntry(asset_table="Image", schema="s", metadata={"Subject": "2-B"}),
        )
        manifest.add_asset(
            "Image/c.jpg",
            AssetEntry(asset_table="Image", schema="s", metadata={"Subject": "2-C"}),
        )

        got = manifest.get_asset("Image/b.jpg")
        assert got is not None
        assert got.metadata == {"Subject": "2-B"}
        assert got.asset_table == "Image"

    def test_get_asset_missing_returns_none(self, manifest):
        """``get_asset`` returns None for missing keys (caller-friendly).

        ``AssetFilePath.metadata`` is a property that reads on every
        access; raising KeyError for an unknown key would surface as a
        confusing crash from a simple attribute read. None is the
        expected sentinel — the property falls back to ``asset_metadata``.
        """
        assert manifest.get_asset("never/added.jpg") is None

    def test_to_json_format(self, manifest):
        """Test that to_json returns a well-formed dict."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
                asset_types=["Training_Data"],
            ),
        )

        data = manifest.to_json()

        # ``version`` was deleted in the catalog-asset P1 sweep
        # (audit D1) — the field was never read on load.
        assert "version" not in data
        assert data["execution_rid"] == "4SP"
        assert "Image/scan.jpg" in data["assets"]
        assert data["assets"]["Image/scan.jpg"]["status"] == "pending"

    def test_resume_after_crash(self, store):
        """Test resume scenario: some assets uploaded, some pending."""
        # First "session" — register and partially upload
        manifest1 = AssetManifest(store, "4SP")
        manifest1.add_asset("Image/a.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest1.add_asset("Image/b.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest1.add_asset("Image/c.jpg", AssetEntry(asset_table="Image", schema="s"))
        manifest1.mark_uploaded("Image/a.jpg", "1-A")
        # "Crash" — manifest1 goes out of scope

        # Second "session" — new instance sees same store state
        manifest2 = AssetManifest(store, "4SP")
        pending = manifest2.pending_assets()
        uploaded = manifest2.uploaded_assets()

        assert len(pending) == 2
        assert len(uploaded) == 1
        assert uploaded["Image/a.jpg"].rid == "1-A"

    def test_description_stored_in_entry(self, manifest):
        """Test that description is stored in asset entry."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
                description="A test scan image",
            ),
        )

        assert manifest.assets["Image/scan.jpg"].description == "A test scan image"

    def test_description_persists_across_instances(self, store):
        """Test that description survives reload (simulates crash recovery)."""
        manifest = AssetManifest(store, "4SP")
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
                description="Persisted description",
            ),
        )

        # New instance over the same store
        manifest2 = AssetManifest(store, "4SP")
        assert manifest2.assets["Image/scan.jpg"].description == "Persisted description"

    def test_description_none_by_default(self, manifest):
        """Test that description defaults to None when not provided."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

        assert manifest.assets["Image/scan.jpg"].description is None

    def test_description_in_json_format(self, manifest):
        """Test that description appears in to_json output."""
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
                description="JSON visible description",
            ),
        )

        data = manifest.to_json()
        assert data["assets"]["Image/scan.jpg"]["description"] == "JSON visible description"


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

    def test_metadata_property_with_manifest(self, store, tmp_path):
        """Test metadata property with manifest binding."""
        manifest = AssetManifest(store, "4SP")
        manifest.add_asset(
            "Image/file.txt",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
                metadata={"Subject": "2-DEF"},
            ),
        )

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

    def test_set_asset_types_with_manifest(self, store, tmp_path):
        """Test set_asset_types updates manifest."""
        manifest = AssetManifest(store, "4SP")
        manifest.add_asset(
            "Image/file.txt",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

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

    def test_asset_record_class_builds_pydantic_model_from_asset_table(self):
        """``asset_record_class`` reads an asset table and emits a typed Pydantic model.

        Coverage gap from audit P1 B4: the factory function had no
        end-to-end test against a mocked model. This test pins:

        * The returned class is a Pydantic ``BaseModel`` subclass
          named ``<TableName>AssetRecord``.
        * Only metadata-column names (the set returned by
          ``model.asset_metadata(table)``) become fields; system
          + standard-asset columns are skipped via the same set
          intersection.
        * Field types are mapped by ERMrest type (text → str,
          int4 → int, float8 → float, with nullable columns
          becoming ``Optional[T]`` defaulting to ``None``).
        * Constructed instances reject unknown fields (the base's
          ``extra = "forbid"`` is inherited).
        """
        from unittest.mock import MagicMock

        from deriva_ml.asset.asset_record import asset_record_class

        def _fake_column(name: str, typename: str, nullok: bool = False) -> MagicMock:
            col = MagicMock()
            col.name = name
            col.type.typename = typename
            col.nullok = nullok
            col.default = None
            return col

        # Standard-asset + system columns the factory skips —
        # ``model.asset_metadata`` excludes these, so the factory's
        # ``if col.name not in metadata_col_names: continue`` filter
        # drops them.
        skip_columns = [
            _fake_column(n, "text")
            for n in (
                "RID",
                "RCT",
                "RMT",
                "RCB",
                "RMB",
                "URL",
                "Filename",
                "Length",
                "MD5",
                "Description",
            )
        ]
        # Metadata columns the factory should pick up.
        metadata_columns = [
            _fake_column("Subject", "text"),  # required str
            _fake_column("Acquisition_Year", "int4"),  # required int
            _fake_column("Confidence", "float8", nullok=True),  # optional float
        ]
        metadata_names = {c.name for c in metadata_columns}

        fake_table = MagicMock()
        fake_table.name = "Image"
        fake_table.columns = skip_columns + metadata_columns

        fake_model = MagicMock()
        fake_model.name_to_table.return_value = fake_table
        # The factory walks ``model.asset_metadata(name)`` to find
        # which columns are metadata-vs-standard. Return the set of
        # metadata column names for our fake table.
        fake_model.asset_metadata.return_value = metadata_names

        ImageAssetRecord = asset_record_class(fake_model, "Image")
        fake_model.name_to_table.assert_called_once_with("Image")
        fake_model.asset_metadata.assert_called_once_with("Image")

        # Class shape.
        assert ImageAssetRecord.__name__ == "ImageAssetRecord"

        # Happy-path construction with all required fields.
        rec = ImageAssetRecord(Subject="2-DEF", Acquisition_Year=2026)
        assert rec.Subject == "2-DEF"
        assert rec.Acquisition_Year == 2026
        # Nullable column defaults to None.
        assert rec.Confidence is None

        # Optional field can be set.
        rec_with = ImageAssetRecord(Subject="2-DEF", Acquisition_Year=2026, Confidence=0.97)
        assert rec_with.Confidence == 0.97

        # Unknown fields rejected (extra="forbid" inherited from AssetRecord).
        # In particular, standard-asset columns like ``URL`` /
        # ``Filename`` must NOT appear as fields on the generated
        # class — the metadata filter must have dropped them.
        with pytest.raises(Exception):  # Pydantic ValidationError
            ImageAssetRecord(Subject="2-DEF", Acquisition_Year=2026, Bogus="x")
        with pytest.raises(Exception):  # standard-asset column must be filtered out
            ImageAssetRecord(
                Subject="2-DEF",
                Acquisition_Year=2026,
                URL="/hatrac/...",
            )


# =============================================================================
# Integration: AssetRecord with AssetFilePath metadata
# =============================================================================


class TestAssetRecordWithFilePath:
    """Test using AssetRecord with AssetFilePath metadata setter."""

    def test_set_metadata_from_record(self, store, tmp_path):
        """Test setting AssetFilePath metadata from an AssetRecord."""
        from pydantic import create_model

        from deriva_ml.asset.asset_record import AssetRecord

        ImageAsset = create_model(
            "ImageAsset",
            __base__=AssetRecord,
            Subject=(str, ...),
            Acquisition_Date=(str, ...),
        )

        manifest = AssetManifest(store, "4SP")
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

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

    def test_set_metadata_from_dict(self, store, tmp_path):
        """Test setting metadata from a plain dict (backward compat)."""
        manifest = AssetManifest(store, "4SP")
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="test-schema",
            ),
        )

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


class TestJsonDefault:
    """Tests for the _json_default serializer used by to_json()."""

    def test_datetime_serialized(self) -> None:
        from datetime import datetime, timezone

        from deriva_ml.asset.manifest import _json_default

        dt = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert _json_default(dt) == "2026-04-15T12:00:00+00:00"

    def test_date_serialized(self) -> None:
        from datetime import date

        from deriva_ml.asset.manifest import _json_default

        d = date(2026, 4, 15)
        assert _json_default(d) == "2026-04-15"

    def test_path_serialized(self) -> None:
        from pathlib import Path

        from deriva_ml.asset.manifest import _json_default

        p = Path("/tmp/test.txt")
        assert _json_default(p) == "/tmp/test.txt"

    def test_unknown_type_raises(self) -> None:
        from deriva_ml.asset.manifest import _json_default

        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_to_json_with_datetime_metadata(self, store, manifest) -> None:
        """to_json() + json.dumps with _json_default handles datetime metadata."""
        import json
        from datetime import datetime, timezone

        from deriva_ml.asset.manifest import AssetEntry, _json_default

        # Store the asset with plain string metadata (DB can't serialize datetime directly)
        manifest.add_asset(
            "Image/scan.jpg",
            AssetEntry(
                asset_table="Image",
                schema="isa",
                metadata={"date": "2026-01-01T00:00:00+00:00"},
            ),
        )
        # Build a JSON structure that includes a datetime object (simulating catalog data)
        data = manifest.to_json()
        # Inject a datetime into the structure to test the serializer path
        data["assets"]["Image/scan.jpg"]["metadata"]["ts"] = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = json.dumps(data, default=_json_default)
        assert "2026-01-01" in result

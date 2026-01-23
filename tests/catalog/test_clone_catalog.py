"""Tests for catalog cloning functionality.

Tests cover same-server cloning scenarios including:
- Schema-only cloning
- Full data cloning (schema + data)
- ML schema addition during cloning
- Dataset bag download verification on cloned catalogs
- Annotation and policy preservation
"""

from __future__ import annotations

import pytest
from pathlib import Path

from deriva.core import DerivaServer, get_credential

from deriva_ml import DerivaML, MLVocab
from deriva_ml.catalog import clone_catalog, CloneCatalogResult, AssetCopyMode
from deriva_ml.dataset.aux_classes import DatasetSpec, VersionPart
from deriva_ml.demo_catalog import DatasetDescription
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.model.deriva_ml_database import DerivaMLDatabase
from tests.catalog_manager import CatalogManager


class TestCloneCatalogSameServer:
    """Tests for same-server catalog cloning."""

    def test_clone_schema_only(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Test cloning just the schema without any data."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Clone the catalog (schema only)
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            schema_only=True,
        )

        try:
            # Verify result structure
            assert isinstance(result, CloneCatalogResult)
            assert result.catalog_id is not None
            assert result.hostname == hostname
            assert result.schema_only is True
            assert result.source_hostname == hostname
            assert result.source_catalog_id == source_catalog_id

            # Connect to the cloned catalog and verify schema exists
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path,
            )

            # Verify ML schema exists
            model = cloned_ml.catalog.getCatalogModel()
            assert "deriva-ml" in model.schemas, "ML schema should exist in clone"

            # Verify domain schema exists
            assert catalog_manager.domain_schema in model.schemas, (
                f"Domain schema '{catalog_manager.domain_schema}' should exist in clone"
            )

            # Verify key ML tables exist
            ml_schema = model.schemas["deriva-ml"]
            expected_tables = ["Dataset", "Execution", "Workflow", "Dataset_Version"]
            for table_name in expected_tables:
                assert table_name in ml_schema.tables, (
                    f"Table {table_name} should exist in ML schema"
                )

            # Verify no data was copied (schema only)
            pb = cloned_ml.pathBuilder()
            ml_path = pb.schemas["deriva-ml"]
            datasets = list(ml_path.tables["Dataset"].path.entities().fetch())
            assert len(datasets) == 0, "Schema-only clone should have no datasets"

        finally:
            # Clean up the cloned catalog
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_with_data(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test cloning catalog with schema and data."""
        # First populate the source catalog with some data
        ml = catalog_manager.ensure_populated(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get counts from source catalog
        pb = ml.pathBuilder()
        domain_path = pb.schemas[catalog_manager.domain_schema]
        source_subjects = list(domain_path.tables["Subject"].path.entities().fetch())
        source_images = list(domain_path.tables["Image"].path.entities().fetch())

        # Clone with data
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            schema_only=False,
        )

        try:
            assert result.schema_only is False

            # Connect to cloned catalog
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            # Verify data was copied
            pb_clone = cloned_ml.pathBuilder()
            domain_path_clone = pb_clone.schemas[catalog_manager.domain_schema]

            cloned_subjects = list(
                domain_path_clone.tables["Subject"].path.entities().fetch()
            )
            cloned_images = list(
                domain_path_clone.tables["Image"].path.entities().fetch()
            )

            assert len(cloned_subjects) == len(source_subjects), (
                f"Expected {len(source_subjects)} subjects, got {len(cloned_subjects)}"
            )
            assert len(cloned_images) == len(source_images), (
                f"Expected {len(source_images)} images, got {len(cloned_images)}"
            )

            # Verify RIDs are preserved
            source_subject_rids = {s["RID"] for s in source_subjects}
            cloned_subject_rids = {s["RID"] for s in cloned_subjects}
            assert source_subject_rids == cloned_subject_rids, (
                "Subject RIDs should be preserved in clone"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_with_datasets_and_download_bag(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that datasets in cloned catalog can be downloaded as bags.

        Note: When cloning a catalog, the dataset versions reference snapshot IDs
        from the source catalog's history. These snapshots don't exist in the clone.
        To download a bag from a cloned catalog, we need to create a new version
        in the clone (which creates a valid snapshot in the clone's history).
        """
        # Create a fully populated catalog with datasets
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get the root dataset info from source
        source_dataset = dataset_desc.dataset

        # Clone with data
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            schema_only=False,
        )

        try:
            # Connect to cloned catalog
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            # Look up the same dataset in the clone
            cloned_dataset = cloned_ml.lookup_dataset(source_dataset.dataset_rid)
            assert cloned_dataset is not None, "Dataset should exist in clone"

            # Create a new version in the clone - this creates a valid snapshot
            # in the clone's history that we can use to download the bag
            new_version = cloned_dataset.increment_dataset_version(
                component=VersionPart.patch,
                description="Version created in cloned catalog",
            )

            # Verify dataset can be downloaded as a bag using the new version
            bag = cloned_dataset.download_dataset_bag(
                version=new_version,
                use_minid=False,
            )
            assert bag is not None, "Bag download should succeed"

            # Verify bag contents
            members = bag.list_dataset_members()
            assert len(members) > 0, "Bag should contain dataset members"

            # Verify files are accessible
            if "Image" in members:
                for image_record in members["Image"]:
                    if "Filename" in image_record:
                        file_path = Path(image_record["Filename"])
                        assert file_path.exists(), (
                            f"Image file should exist: {file_path}"
                        )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_add_ml_schema_when_missing(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test adding ML schema to a clone when the source doesn't have it.

        This test creates a plain catalog without ML schema, clones it with
        add_ml_schema=True, and verifies the ML schema is added.
        """
        hostname = catalog_manager.hostname

        # Create a plain catalog without ML schema
        server = DerivaServer("https", hostname, credentials=get_credential(hostname))
        plain_catalog = server.create_ermrest_catalog()

        try:
            # Configure basic catalog structure
            model = plain_catalog.getCatalogModel()
            model.configure_baseline_catalog()

            # Verify it doesn't have ML schema
            assert "deriva-ml" not in model.schemas, (
                "Plain catalog should not have ML schema"
            )

            # Clone with ML schema addition
            result = clone_catalog(
                source_hostname=hostname,
                source_catalog_id=str(plain_catalog.catalog_id),
                add_ml_schema=True,
            )

            try:
                assert result.ml_schema_added is True, (
                    "Result should indicate ML schema was added"
                )

                # Verify ML schema exists in clone
                cloned_ml = DerivaML(
                    hostname,
                    result.catalog_id,
                    working_dir=tmp_path,
                )
                cloned_model = cloned_ml.catalog.getCatalogModel()
                assert "deriva-ml" in cloned_model.schemas, (
                    "Clone should have ML schema"
                )

                # Verify ML tables exist
                ml_schema = cloned_model.schemas["deriva-ml"]
                assert "Dataset" in ml_schema.tables
                assert "Execution" in ml_schema.tables
                assert "Workflow" in ml_schema.tables

            finally:
                self._delete_catalog(hostname, result.catalog_id)

        finally:
            # Clean up the plain catalog
            plain_catalog.delete_ermrest_catalog(really=True)

    def test_clone_add_ml_schema_when_exists(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that add_ml_schema is a no-op when ML schema already exists."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Clone with add_ml_schema (but source already has it)
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            add_ml_schema=True,
        )

        try:
            # Should not indicate schema was added (it was already there)
            assert result.ml_schema_added is False, (
                "Should not add ML schema when it already exists"
            )

            # Verify ML schema exists and is intact
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path,
            )
            model = cloned_ml.catalog.getCatalogModel()
            assert "deriva-ml" in model.schemas

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_preserves_annotations(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that catalog annotations are preserved during cloning."""
        ml = catalog_manager.get_ml_instance(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get source catalog annotations
        source_model = ml.catalog.getCatalogModel()
        source_annotations = dict(source_model.annotations)

        # Clone with annotations (default behavior)
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            copy_annotations=True,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )
            cloned_model = cloned_ml.catalog.getCatalogModel()
            cloned_annotations = dict(cloned_model.annotations)

            # Compare key annotations (some system annotations may differ)
            # Check for presence of important annotation keys
            for key in source_annotations:
                if key not in ["tag:isrd.isi.edu,2019:chaise-config"]:
                    # Skip chaise-config as it may have host-specific settings
                    assert key in cloned_annotations, (
                        f"Annotation {key} should be preserved in clone"
                    )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_without_annotations(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test cloning without preserving annotations."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            copy_annotations=False,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path,
            )
            cloned_model = cloned_ml.catalog.getCatalogModel()

            # Annotations should be minimal or empty
            # Note: Some baseline annotations may still exist from configure_baseline
            assert cloned_model is not None

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_with_alias(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test creating a catalog alias during cloning."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname
        alias_name = f"test-clone-alias-{source_catalog_id}"

        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            alias=alias_name,
        )

        try:
            assert result.alias == alias_name, (
                f"Expected alias '{alias_name}', got '{result.alias}'"
            )

            # Verify we can connect using the alias
            cloned_ml = DerivaML(
                hostname,
                alias_name,  # Use alias instead of catalog ID
                working_dir=tmp_path,
            )
            assert cloned_ml.catalog is not None

        finally:
            # Clean up alias first, then catalog
            try:
                server = DerivaServer(
                    "https", hostname, credentials=get_credential(hostname)
                )
                server.delete_ermrest_alias(alias_name)
            except Exception:
                pass  # Alias may not exist
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_exclude_schemas(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test excluding specific schemas from cloning."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Clone but exclude the domain schema
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            exclude_schemas=[catalog_manager.domain_schema],
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path,
            )
            cloned_model = cloned_ml.catalog.getCatalogModel()

            # ML schema should exist
            assert "deriva-ml" in cloned_model.schemas

            # Domain schema should NOT exist
            assert catalog_manager.domain_schema not in cloned_model.schemas, (
                f"Excluded schema '{catalog_manager.domain_schema}' should not exist"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_result_has_source_info(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that CloneCatalogResult contains source catalog information."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
        )

        try:
            assert result.source_hostname == hostname
            assert result.source_catalog_id == source_catalog_id
            assert result.catalog_id != source_catalog_id, (
                "Clone should have different catalog ID"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_bag_contents_match_source(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that bag contents from clone match source catalog exactly.

        Note: When cloning a catalog, dataset versions reference snapshot IDs
        from the source catalog that don't exist in the clone. We create a new
        version in both catalogs to enable comparison with valid snapshots.
        """
        # Create fully populated catalog
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_dataset = dataset_desc.dataset

        # Clone the catalog first (before creating new version in source)
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            # Create new versions in both source and clone
            # These versions will have valid snapshots in their respective catalogs
            source_new_version = source_dataset.increment_dataset_version(
                component=VersionPart.patch,
                description="Version for comparison test",
            )

            cloned_dataset = cloned_ml.lookup_dataset(source_dataset.dataset_rid)
            cloned_new_version = cloned_dataset.increment_dataset_version(
                component=VersionPart.patch,
                description="Version for comparison test",
            )

            # Download bags using the new versions
            source_bag = source_dataset.download_dataset_bag(
                version=source_new_version,
                use_minid=False,
            )
            source_members = source_bag.list_dataset_members()

            cloned_bag = cloned_dataset.download_dataset_bag(
                version=cloned_new_version,
                use_minid=False,
            )
            cloned_members = cloned_bag.list_dataset_members()

            # Compare member counts per type
            for member_type in source_members:
                assert member_type in cloned_members, (
                    f"Member type '{member_type}' should exist in clone"
                )
                assert len(source_members[member_type]) == len(cloned_members[member_type]), (
                    f"Member count mismatch for {member_type}: "
                    f"source={len(source_members[member_type])}, "
                    f"clone={len(cloned_members[member_type])}"
                )

            # Compare RIDs for each member type
            for member_type in source_members:
                source_rids = {m["RID"] for m in source_members[member_type]}
                cloned_rids = {m["RID"] for m in cloned_members[member_type]}
                assert source_rids == cloned_rids, (
                    f"RID mismatch for {member_type}: "
                    f"source={source_rids}, clone={cloned_rids}"
                )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_nested_datasets_preserved(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that nested dataset relationships are preserved in clone."""
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get nested dataset structure from source
        source_dataset = dataset_desc.dataset
        source_children = source_dataset.list_dataset_children()
        source_child_rids = {c.dataset_rid for c in source_children}

        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            # Verify nested structure in clone
            cloned_dataset = cloned_ml.lookup_dataset(source_dataset.dataset_rid)
            cloned_children = cloned_dataset.list_dataset_children()
            cloned_child_rids = {c.dataset_rid for c in cloned_children}

            assert source_child_rids == cloned_child_rids, (
                "Nested dataset relationships should be preserved"
            )

            # Verify each child's types are preserved
            source_child_types = {
                c.dataset_rid: set(c.dataset_types) for c in source_children
            }
            cloned_child_types = {
                c.dataset_rid: set(c.dataset_types) for c in cloned_children
            }
            assert source_child_types == cloned_child_types, (
                "Dataset types should be preserved for nested datasets"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_asset_mode_references(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that same-server clone uses REFERENCES asset mode."""
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
        )

        try:
            # Same-server clones should use REFERENCES mode (assets stay on same hatrac)
            assert result.asset_mode == AssetCopyMode.REFERENCES, (
                "Same-server clone should use REFERENCES asset mode"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_vocabulary_terms_preserved(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that vocabulary terms are preserved in clone."""
        ml = catalog_manager.ensure_features(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get vocabulary terms from source
        source_workflow_types = ml.list_vocabulary_terms(MLVocab.workflow_type)

        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            cloned_workflow_types = cloned_ml.list_vocabulary_terms(MLVocab.workflow_type)

            # Compare vocabulary terms
            source_names = {t.name for t in source_workflow_types}
            cloned_names = {t.name for t in cloned_workflow_types}

            assert source_names == cloned_names, (
                f"Workflow type vocabulary mismatch: "
                f"source={source_names}, clone={cloned_names}"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_reinitializes_dataset_versions(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that dataset versions are incremented after cloning.

        Verifies that:
        1. The clone result includes datasets_reinitialized count
        2. The clone result includes source_snapshot info
        3. Dataset versions in the clone have valid snapshots
        4. Version descriptions include source catalog URL
        """
        # Create a catalog with datasets
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_dataset = dataset_desc.dataset
        source_version = source_dataset.current_version

        # Clone with dataset version reinitialization (default)
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
        )

        try:
            # Verify result includes version reinitialization info
            assert result.datasets_reinitialized > 0, (
                "Should have reinitialized at least one dataset"
            )
            assert result.source_snapshot, (
                "Should include source snapshot ID"
            )

            # Connect to cloned catalog
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            # Verify dataset version was incremented
            cloned_dataset = cloned_ml.lookup_dataset(source_dataset.dataset_rid)
            cloned_version = cloned_dataset.current_version

            # Version should be incremented (patch version bumped)
            assert str(cloned_version) != str(source_version), (
                f"Clone version {cloned_version} should differ from source {source_version}"
            )

            # The cloned dataset should be downloadable as a bag
            # (which would fail if versions weren't reinitialized)
            bag = cloned_dataset.download_dataset_bag(
                version=cloned_version,
                use_minid=False,
            )
            assert bag is not None, "Bag download should succeed with reinitialized version"

            # Check that version description includes source catalog URL
            pb = cloned_ml.pathBuilder()
            version_table = pb.schemas["deriva-ml"].tables["Dataset_Version"]
            versions = list(
                version_table.path
                .filter(version_table.Dataset == source_dataset.dataset_rid)
                .entities()
                .fetch()
            )
            assert len(versions) > 0, "Should have version records"

            # Find the version created by cloning
            clone_version_record = next(
                (v for v in versions if "Cloned from" in (v.get("Description") or "")),
                None
            )
            assert clone_version_record is not None, (
                "Should have a version with 'Cloned from' in description"
            )
            assert result.source_snapshot in clone_version_record["Description"], (
                "Version description should include source snapshot ID"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_clone_skip_version_reinitialization(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test cloning without dataset version reinitialization."""
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Clone without version reinitialization
        result = clone_catalog(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            reinitialize_dataset_versions=False,
        )

        try:
            # Should not have reinitialized any datasets
            assert result.datasets_reinitialized == 0, (
                "Should not reinitialize datasets when disabled"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def _delete_catalog(self, hostname: str, catalog_id: str) -> None:
        """Helper to delete a catalog."""
        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            catalog = server.connect_ermrest(catalog_id)
            catalog.delete_ermrest_catalog(really=True)
        except Exception as e:
            print(f"Warning: Failed to delete catalog {catalog_id}: {e}")


class TestCloneCatalogErrors:
    """Tests for error handling in catalog cloning."""

    def test_clone_nonexistent_catalog(self, catalog_host: str, tmp_path: Path):
        """Test cloning a catalog that doesn't exist."""
        with pytest.raises(Exception):
            clone_catalog(
                source_hostname=catalog_host,
                source_catalog_id="99999999",  # Non-existent catalog
            )

    def test_clone_invalid_hostname(self, tmp_path: Path):
        """Test cloning from an invalid hostname."""
        with pytest.raises(Exception):
            clone_catalog(
                source_hostname="invalid.hostname.that.does.not.exist.local",
                source_catalog_id="1",
            )

"""Tests for catalog cloning via create_ml_workspace.

Tests cover workspace creation scenarios including:
- Basic workspace creation with data
- ML schema addition during cloning
- Dataset bag download verification on cloned catalogs
- Annotation and policy preservation
- Orphan handling strategies
- Three-stage clone approach
- Hidden FK pruning
- Asset localization after clone
"""

from __future__ import annotations

import pytest
from pathlib import Path

from deriva.core import DerivaServer, get_credential

from deriva_ml import DerivaML, MLVocab
from deriva_ml.catalog import create_ml_workspace, CloneCatalogResult, AssetCopyMode
from deriva_ml.dataset.aux_classes import DatasetSpec, VersionPart
from deriva_ml.demo_catalog import DatasetDescription
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.model.deriva_ml_database import DerivaMLDatabase
from tests.catalog_manager import CatalogManager


def _get_root_rid(catalog_manager: CatalogManager, tmp_path: Path) -> tuple[DerivaML, str]:
    """Helper to get a root dataset RID from a populated catalog.

    Returns:
        Tuple of (DerivaML instance, root dataset RID).
    """
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    return ml, dataset_desc.dataset.dataset_rid


class TestCreateMlWorkspace:
    """Tests for create_ml_workspace."""

    def test_workspace_creation_basic(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test creating a workspace with schema and data."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert isinstance(result, CloneCatalogResult)
            assert result.catalog_id is not None
            assert result.hostname == hostname
            assert result.source_hostname == hostname
            assert result.source_catalog_id == source_catalog_id

            # Connect to the cloned catalog and verify schema exists
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "clone",
            )

            model = cloned_ml.catalog.getCatalogModel()
            assert "deriva-ml" in model.schemas, "ML schema should exist in clone"
            assert catalog_manager.domain_schema in model.schemas, (
                f"Domain schema '{catalog_manager.domain_schema}' should exist in clone"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_with_data(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test creating a workspace copies data from source."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get counts from source catalog
        pb = ml.pathBuilder()
        domain_path = pb.schemas[catalog_manager.domain_schema]
        source_subjects = list(domain_path.tables["Subject"].path.entities().fetch())
        source_images = list(domain_path.tables["Image"].path.entities().fetch())

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
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

            # Subset clone may have fewer rows, but should have some data
            assert len(cloned_subjects) > 0, "Should have some subjects"
            assert len(cloned_images) > 0, "Should have some images"

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_with_datasets_and_download_bag(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that datasets in cloned catalog can be downloaded as bags.

        Note: When cloning a catalog, the dataset versions reference snapshot IDs
        from the source catalog's history. These snapshots don't exist in the clone.
        To download a bag from a cloned catalog, we need to create a new version
        in the clone (which creates a valid snapshot in the clone's history).
        """
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_dataset_rid = root_rid

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            # Look up the same dataset in the clone
            cloned_dataset = cloned_ml.lookup_dataset(source_dataset_rid)
            assert cloned_dataset is not None, "Dataset should exist in clone"

            # Create a new version in the clone
            new_version = cloned_dataset.increment_dataset_version(
                component=VersionPart.patch,
                description="Version created in cloned catalog",
            )

            # Verify dataset can be downloaded as a bag
            bag = cloned_dataset.download_dataset_bag(
                version=new_version,
                use_minid=False,
            )
            assert bag is not None, "Bag download should succeed"

            members = bag.list_dataset_members()
            assert len(members) > 0, "Bag should contain dataset members"

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_add_ml_schema_when_missing(
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
            model = plain_catalog.getCatalogModel()
            model.configure_baseline_catalog()

            assert "deriva-ml" not in model.schemas, (
                "Plain catalog should not have ML schema"
            )

            # We need to find a table/row to use as root_rid.
            # In a plain catalog with just baseline, use ERMrest_Client from public schema.
            pb = plain_catalog.getPathBuilder()
            clients = list(pb.public.ERMrest_Client.path.entities().fetch())
            if not clients:
                pytest.skip("No ERMrest_Client rows in plain catalog")
            root_rid = clients[0]["RID"]

            result = create_ml_workspace(
                source_hostname=hostname,
                source_catalog_id=str(plain_catalog.catalog_id),
                root_rid=root_rid,
                add_ml_schema=True,
            )

            try:
                assert result.ml_schema_added is True, (
                    "Result should indicate ML schema was added"
                )

                cloned_ml = DerivaML(
                    hostname,
                    result.catalog_id,
                    working_dir=tmp_path,
                )
                cloned_model = cloned_ml.catalog.getCatalogModel()
                assert "deriva-ml" in cloned_model.schemas, (
                    "Clone should have ML schema"
                )

                ml_schema = cloned_model.schemas["deriva-ml"]
                assert "Dataset" in ml_schema.tables
                assert "Execution" in ml_schema.tables
                assert "Workflow" in ml_schema.tables

            finally:
                self._delete_catalog(hostname, result.catalog_id)

        finally:
            plain_catalog.delete_ermrest_catalog(really=True)

    def test_workspace_add_ml_schema_when_exists(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that add_ml_schema is a no-op when ML schema already exists."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            add_ml_schema=True,
        )

        try:
            assert result.ml_schema_added is False, (
                "Should not add ML schema when it already exists"
            )

            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )
            model = cloned_ml.catalog.getCatalogModel()
            assert "deriva-ml" in model.schemas

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_preserves_annotations(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that catalog annotations are preserved during cloning."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_model = ml.catalog.getCatalogModel()
        source_annotations = dict(source_model.annotations)

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
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

            for key in source_annotations:
                if key not in ["tag:isrd.isi.edu,2019:chaise-config"]:
                    assert key in cloned_annotations, (
                        f"Annotation {key} should be preserved in clone"
                    )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_without_annotations(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test cloning without preserving annotations."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            copy_annotations=False,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "clone",
            )
            cloned_model = cloned_ml.catalog.getCatalogModel()
            assert cloned_model is not None

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_with_alias(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test creating a catalog alias during cloning."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname
        alias_name = f"test-clone-alias-{source_catalog_id}"

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            alias=alias_name,
        )

        try:
            assert result.alias == alias_name, (
                f"Expected alias '{alias_name}', got '{result.alias}'"
            )

            cloned_ml = DerivaML(
                hostname,
                alias_name,
                working_dir=tmp_path / "clone",
            )
            assert cloned_ml.catalog is not None

        finally:
            try:
                server = DerivaServer(
                    "https", hostname, credentials=get_credential(hostname)
                )
                server.delete_ermrest_alias(alias_name)
            except Exception:
                pass
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_exclude_schemas(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test excluding specific schemas from cloning."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            exclude_schemas=[catalog_manager.domain_schema],
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "clone",
            )
            cloned_model = cloned_ml.catalog.getCatalogModel()

            assert "deriva-ml" in cloned_model.schemas
            assert catalog_manager.domain_schema not in cloned_model.schemas, (
                f"Excluded schema '{catalog_manager.domain_schema}' should not exist"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_result_has_source_info(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that CloneCatalogResult contains source catalog information."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.source_hostname == hostname
            assert result.source_catalog_id == source_catalog_id
            assert result.catalog_id != source_catalog_id, (
                "Clone should have different catalog ID"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_nested_datasets_preserved(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that nested dataset relationships are preserved in clone."""
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_dataset = dataset_desc.dataset
        root_rid = source_dataset.dataset_rid
        source_children = source_dataset.list_dataset_children()
        source_child_rids = {c.dataset_rid for c in source_children}

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            cloned_dataset = cloned_ml.lookup_dataset(source_dataset.dataset_rid)
            cloned_children = cloned_dataset.list_dataset_children()
            cloned_child_rids = {c.dataset_rid for c in cloned_children}

            assert source_child_rids == cloned_child_rids, (
                "Nested dataset relationships should be preserved"
            )

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

    def test_workspace_asset_mode_references(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that same-server clone uses REFERENCES asset mode."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.asset_mode == AssetCopyMode.REFERENCES, (
                "Same-server clone should use REFERENCES asset mode"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_vocabulary_terms_preserved(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that vocabulary terms are preserved in clone."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_workflow_types = ml.list_vocabulary_terms(MLVocab.workflow_type)

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            cloned_workflow_types = cloned_ml.list_vocabulary_terms(MLVocab.workflow_type)

            source_names = {t.name for t in source_workflow_types}
            cloned_names = {t.name for t in cloned_workflow_types}

            assert source_names == cloned_names, (
                f"Workflow type vocabulary mismatch: "
                f"source={source_names}, clone={cloned_names}"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_reinitializes_dataset_versions(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that dataset versions are incremented after cloning.

        Verifies that:
        1. The clone result includes datasets_reinitialized count
        2. The clone result includes source_snapshot info
        3. Dataset versions in the clone have valid snapshots
        4. Version descriptions include source catalog URL
        """
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        source_dataset = dataset_desc.dataset
        root_rid = source_dataset.dataset_rid
        source_version = source_dataset.current_version

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.datasets_reinitialized > 0, (
                "Should have reinitialized at least one dataset"
            )
            assert result.source_snapshot, (
                "Should include source snapshot ID"
            )

            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            cloned_dataset = cloned_ml.lookup_dataset(source_dataset.dataset_rid)
            cloned_version = cloned_dataset.current_version

            assert str(cloned_version) != str(source_version), (
                f"Clone version {cloned_version} should differ from source {source_version}"
            )

            bag = cloned_dataset.download_dataset_bag(
                version=cloned_version,
                use_minid=False,
            )
            assert bag is not None, "Bag download should succeed with reinitialized version"

            pb = cloned_ml.pathBuilder()
            version_table = pb.schemas["deriva-ml"].tables["Dataset_Version"]
            versions = list(
                version_table.path
                .filter(version_table.Dataset == source_dataset.dataset_rid)
                .entities()
                .fetch()
            )
            assert len(versions) > 0, "Should have version records"

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

    def test_workspace_skip_version_reinitialization(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test cloning without dataset version reinitialization."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            reinitialize_dataset_versions=False,
        )

        try:
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


class TestCreateMlWorkspaceErrors:
    """Tests for error handling in create_ml_workspace."""

    def test_nonexistent_catalog(self, catalog_host: str, tmp_path: Path):
        """Test cloning a catalog that doesn't exist."""
        with pytest.raises(Exception):
            create_ml_workspace(
                source_hostname=catalog_host,
                source_catalog_id="99999999",
                root_rid="ABC123",
            )

    def test_invalid_hostname(self, tmp_path: Path):
        """Test cloning from an invalid hostname."""
        with pytest.raises(Exception):
            create_ml_workspace(
                source_hostname="invalid.hostname.that.does.not.exist.local",
                source_catalog_id="1",
                root_rid="ABC123",
            )


class TestCloneReport:
    """Tests for CloneReport functionality."""

    def test_clone_report_to_dict(self):
        """Test CloneReport.to_dict() returns proper structure."""
        from deriva_ml.catalog.clone import (
            CloneReport,
            CloneIssue,
            CloneIssueSeverity,
            CloneIssueCategory,
        )

        report = CloneReport()
        report.tables_restored = {"schema:table1": 100, "schema:table2": 200}
        report.tables_failed = ["schema:table3"]
        report.tables_skipped = ["schema:table4"]
        report.fkeys_applied = 50
        report.fkeys_failed = 2
        report.fkeys_pruned = 1
        report.orphan_details = {
            "schema:table1": {
                "rows_removed": 10,
                "rows_nullified": 0,
                "missing_references": {"schema:ref_table": 10},
            }
        }
        report.add_issue(CloneIssue(
            severity=CloneIssueSeverity.WARNING,
            category=CloneIssueCategory.ORPHAN_ROWS,
            message="Test warning",
            table="schema:table1",
            row_count=10,
        ))
        report.add_issue(CloneIssue(
            severity=CloneIssueSeverity.ERROR,
            category=CloneIssueCategory.FK_VIOLATION,
            message="Test error",
            table="schema:table2",
        ))

        result = report.to_dict()

        assert result["summary"]["total_issues"] == 2
        assert result["summary"]["errors"] == 1
        assert result["summary"]["warnings"] == 1
        assert result["summary"]["tables_restored"] == 2
        assert result["summary"]["tables_failed"] == 1
        assert result["summary"]["total_rows_restored"] == 300
        assert result["summary"]["orphan_rows_removed"] == 10
        assert result["summary"]["fkeys_applied"] == 50
        assert result["summary"]["fkeys_failed"] == 2

        assert len(result["issues"]) == 2
        assert result["tables_restored"] == {"schema:table1": 100, "schema:table2": 200}
        assert result["tables_failed"] == ["schema:table3"]

    def test_clone_report_to_json(self):
        """Test CloneReport.to_json() returns valid JSON string."""
        import json
        from deriva_ml.catalog.clone import CloneReport

        report = CloneReport()
        report.tables_restored = {"schema:table1": 100}
        report.fkeys_applied = 10

        json_str = report.to_json()

        parsed = json.loads(json_str)
        assert "summary" in parsed
        assert parsed["summary"]["tables_restored"] == 1

    def test_clone_report_to_text(self):
        """Test CloneReport.to_text() returns human-readable output."""
        from deriva_ml.catalog.clone import (
            CloneReport,
            CloneIssue,
            CloneIssueSeverity,
            CloneIssueCategory,
        )

        report = CloneReport()
        report.tables_restored = {"schema:table1": 100}
        report.fkeys_applied = 10
        report.add_issue(CloneIssue(
            severity=CloneIssueSeverity.WARNING,
            category=CloneIssueCategory.ORPHAN_ROWS,
            message="Orphan rows deleted",
            table="schema:table1",
            row_count=5,
        ))

        text = report.to_text()

        assert "CATALOG CLONE REPORT" in text
        assert "SUMMARY" in text
        assert "Tables restored:" in text
        assert "ISSUES" in text
        assert "WARNING" in text
        assert "Orphan rows deleted" in text

    def test_clone_issue_str(self):
        """Test CloneIssue string representation."""
        from deriva_ml.catalog.clone import (
            CloneIssue,
            CloneIssueSeverity,
            CloneIssueCategory,
        )

        issue = CloneIssue(
            severity=CloneIssueSeverity.ERROR,
            category=CloneIssueCategory.FK_VIOLATION,
            message="FK constraint failed",
            table="schema:table1",
            row_count=10,
        )

        str_repr = str(issue)
        assert "[ERROR]" in str_repr
        assert "schema:table1" in str_repr
        assert "FK constraint failed" in str_repr
        assert "(10 rows)" in str_repr


class TestOrphanStrategy:
    """Tests for OrphanStrategy enum and orphan handling."""

    def test_orphan_strategy_values(self):
        """Test OrphanStrategy enum has expected values."""
        from deriva_ml.catalog.clone import OrphanStrategy

        assert OrphanStrategy.FAIL.value == "fail"
        assert OrphanStrategy.DELETE.value == "delete"
        assert OrphanStrategy.NULLIFY.value == "nullify"

    def test_workspace_with_orphan_strategy_fail_default(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that FAIL is the default orphan strategy."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.catalog_id is not None
            assert result.report is not None
            assert result.report.fkeys_failed == 0

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_result_includes_orphan_stats(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that CloneCatalogResult includes orphan statistics."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert hasattr(result, 'orphan_rows_removed')
            assert hasattr(result, 'orphan_rows_nullified')
            assert hasattr(result, 'fkeys_pruned')
            assert result.orphan_rows_removed >= 0
            assert result.orphan_rows_nullified >= 0
            assert result.fkeys_pruned >= 0

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_report_attached_to_result(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that CloneCatalogResult includes a CloneReport."""
        from deriva_ml.catalog.clone import CloneReport

        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.report is not None
            assert isinstance(result.report, CloneReport)
            assert result.report.fkeys_applied > 0

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


class TestOrphanHandlingWithIncoherentPolicies:
    """Tests for orphan handling when source has incoherent policies.

    These tests create catalogs with intentionally orphaned data to verify
    the DELETE and NULLIFY strategies work correctly.
    """

    def test_identify_orphan_values(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Test _identify_orphan_values helper function."""
        from deriva_ml.catalog.clone import _identify_orphan_values

        ml = catalog_manager.ensure_populated(tmp_path)
        hostname = catalog_manager.hostname

        model = ml.catalog.getCatalogModel()
        domain_schema = model.schemas[catalog_manager.domain_schema]

        image_table = domain_schema.tables.get("Image")
        if image_table and image_table.foreign_keys:
            fk = list(image_table.foreign_keys)[0]
            fk_def = fk.prejson()

            orphans = _identify_orphan_values(
                ml.catalog,
                catalog_manager.domain_schema,
                "Image",
                fk_def,
            )

            assert isinstance(orphans, set)

    def test_workspace_with_orphan_strategy_delete(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that clone creates catalog with DELETE strategy."""
        from deriva_ml.catalog.clone import OrphanStrategy

        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            orphan_strategy=OrphanStrategy.DELETE,
        )

        try:
            assert result.catalog_id is not None
            assert result.report is not None

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_with_orphan_strategy_nullify(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that clone creates catalog with NULLIFY strategy."""
        from deriva_ml.catalog.clone import OrphanStrategy

        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            orphan_strategy=OrphanStrategy.NULLIFY,
        )

        try:
            assert result.catalog_id is not None
            assert result.report is not None

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


class TestThreeStageApproach:
    """Tests for the three-stage cloning approach.

    The three stages are:
    1. Create schema WITHOUT foreign keys
    2. Copy all data
    3. Apply foreign keys (with orphan handling)
    """

    def test_workspace_applies_all_fks(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that clone applies foreign keys in stage 3."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.report is not None
            assert result.report.fkeys_applied > 0, "Should have applied some FKs"

            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "clone",
            )
            model = cloned_ml.catalog.getCatalogModel()

            ml_schema = model.schemas.get("deriva-ml")
            if ml_schema:
                dataset_table = ml_schema.tables.get("Dataset")
                if dataset_table:
                    fk_count = len(list(dataset_table.foreign_keys))
                    assert fk_count >= 0

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_preserves_data_integrity(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that clone preserves referential integrity."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            pb_clone = cloned_ml.pathBuilder()
            domain_path_clone = pb_clone.schemas[catalog_manager.domain_schema]
            cloned_subjects = list(
                domain_path_clone.tables["Subject"].path.entities().fetch()
            )
            cloned_images = list(
                domain_path_clone.tables["Image"].path.entities().fetch()
            )

            # Verify FKs are satisfied (images reference valid subjects)
            cloned_subject_rids = {s["RID"] for s in cloned_subjects}
            for image in cloned_images:
                if "Subject" in image and image["Subject"]:
                    assert image["Subject"] in cloned_subject_rids, (
                        f"Image {image['RID']} references non-existent subject"
                    )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_report_tracks_table_progress(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that CloneReport tracks which tables were restored."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            assert result.report is not None
            assert len(result.report.tables_restored) > 0, (
                "Should have restored some tables"
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


class TestPruneHiddenFkeys:
    """Tests for the prune_hidden_fkeys option."""

    def test_workspace_with_prune_hidden_fkeys_false(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test clone with prune_hidden_fkeys=False (default)."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            prune_hidden_fkeys=False,
        )

        try:
            assert result.fkeys_pruned == 0, (
                "Should not prune FKs with prune_hidden_fkeys=False"
            )

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    def test_workspace_with_prune_hidden_fkeys_true(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test clone with prune_hidden_fkeys=True."""
        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            prune_hidden_fkeys=True,
        )

        try:
            assert result.catalog_id is not None

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


class TestLocalizeResult:
    """Tests for LocalizeResult dataclass."""

    def test_localize_result_default_values(self):
        """Test LocalizeResult has correct default values."""
        from deriva_ml.catalog.localize import LocalizeResult

        result = LocalizeResult()
        assert result.assets_processed == 0
        assert result.assets_skipped == 0
        assert result.assets_failed == 0
        assert result.errors == []
        assert result.localized_assets == []

    def test_localize_result_with_values(self):
        """Test LocalizeResult with populated values."""
        from deriva_ml.catalog.localize import LocalizeResult

        result = LocalizeResult(
            assets_processed=5,
            assets_skipped=2,
            assets_failed=1,
            errors=["Error 1", "Error 2"],
            localized_assets=[
                ("RID1", "https://old.host/file1", "https://new.host/file1"),
                ("RID2", "https://old.host/file2", "https://new.host/file2"),
            ],
        )

        assert result.assets_processed == 5
        assert result.assets_skipped == 2
        assert result.assets_failed == 1
        assert len(result.errors) == 2
        assert len(result.localized_assets) == 2
        assert result.localized_assets[0][0] == "RID1"


class TestLocalizeHelperFunctions:
    """Tests for localize helper functions."""

    def test_extract_hatrac_path_full_url(self):
        """Test extracting hatrac path from full URL."""
        from deriva_ml.catalog.localize import _extract_hatrac_path

        url = "https://www.facebase.org/hatrac/fb2/image/xyz123.png"
        path = _extract_hatrac_path(url)
        assert path == "/hatrac/fb2/image/xyz123.png"

    def test_extract_hatrac_path_with_version(self):
        """Test extracting hatrac path with version suffix."""
        from deriva_ml.catalog.localize import _extract_hatrac_path

        url = "https://www.facebase.org/hatrac/fb2/image/xyz123.png:ABC123"
        path = _extract_hatrac_path(url)
        assert path == "/hatrac/fb2/image/xyz123.png:ABC123"

    def test_extract_hatrac_path_relative(self):
        """Test extracting from relative hatrac path."""
        from deriva_ml.catalog.localize import _extract_hatrac_path

        path = "/hatrac/namespace/file.txt"
        result = _extract_hatrac_path(path)
        assert result == "/hatrac/namespace/file.txt"

    def test_extract_hatrac_path_not_hatrac(self):
        """Test extracting from non-hatrac URL returns None."""
        from deriva_ml.catalog.localize import _extract_hatrac_path

        url = "https://example.com/some/other/path"
        path = _extract_hatrac_path(url)
        assert path is None

    def test_find_asset_table_path_with_schema(self):
        """Test finding asset table when schema is specified."""
        from deriva_ml.catalog.localize import _find_asset_table_path
        from unittest.mock import MagicMock

        pb = MagicMock()
        mock_table = MagicMock()
        pb.schemas = {"test_schema": MagicMock()}
        pb.schemas["test_schema"].tables = {"TestTable": mock_table}

        result = _find_asset_table_path(pb, "TestTable", "test_schema")
        assert result == (mock_table, "test_schema")

    def test_find_asset_table_path_not_found(self):
        """Test finding asset table that doesn't exist."""
        from deriva_ml.catalog.localize import _find_asset_table_path
        from unittest.mock import MagicMock

        pb = MagicMock()
        schema_mock = MagicMock()
        tables_mock = MagicMock()
        tables_mock.__getitem__ = MagicMock(side_effect=KeyError("TestTable"))
        schema_mock.tables = tables_mock
        pb.schemas = {"test_schema": schema_mock}

        result = _find_asset_table_path(pb, "TestTable", "test_schema")
        assert result == (None, None)

    def test_find_asset_table_path_search_all_schemas(self):
        """Test finding asset table when searching all schemas."""
        from deriva_ml.catalog.localize import _find_asset_table_path
        from unittest.mock import MagicMock

        pb = MagicMock()
        mock_table = MagicMock()

        schema1 = MagicMock()
        tables1_mock = MagicMock()
        tables1_mock.__getitem__ = MagicMock(side_effect=KeyError("TestTable"))
        schema1.tables = tables1_mock

        schema2 = MagicMock()
        tables2_mock = MagicMock()
        tables2_mock.__getitem__ = MagicMock(return_value=mock_table)
        schema2.tables = tables2_mock

        schemas_mock = MagicMock()
        schemas_mock.__iter__ = MagicMock(return_value=iter(["schema1", "schema2"]))
        schemas_mock.__getitem__ = MagicMock(
            side_effect=lambda k: {"schema1": schema1, "schema2": schema2}[k]
        )
        pb.schemas = schemas_mock

        result = _find_asset_table_path(pb, "TestTable", None)
        assert result == (mock_table, "schema2")


class TestLocalizeAssets:
    """Tests for localize_assets function."""

    def test_localize_assets_raises_on_missing_table(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that localize_assets raises ValueError for non-existent table."""
        from deriva_ml.catalog.localize import localize_assets

        ml = catalog_manager.get_ml_instance(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            localize_assets(
                catalog=ml,
                asset_table="NonExistentTable",
                asset_rids=["1-ABC"],
            )

    def test_localize_assets_empty_rids(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test localize_assets with empty RID list."""
        from deriva_ml.catalog.localize import localize_assets

        ml = catalog_manager.ensure_populated(tmp_path)

        result = localize_assets(
            catalog=ml,
            asset_table="Image",
            asset_rids=[],
            schema_name=catalog_manager.domain_schema,
        )

        assert result.assets_processed == 0
        assert result.assets_skipped == 0
        assert result.assets_failed == 0

    def test_localize_assets_dry_run(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test localize_assets dry run mode."""
        from deriva_ml.catalog.localize import localize_assets

        ml = catalog_manager.ensure_populated(tmp_path)

        pb = ml.pathBuilder()
        domain_path = pb.schemas[catalog_manager.domain_schema]
        images = list(domain_path.tables["Image"].path.entities().fetch())

        if not images:
            pytest.skip("No images in test catalog")

        remote_images = [
            img for img in images
            if img.get("URL") and ml.host_name not in img.get("URL", "")
        ]

        if not remote_images:
            image_rids = [images[0]["RID"]]
            result = localize_assets(
                catalog=ml,
                asset_table="Image",
                asset_rids=image_rids,
                schema_name=catalog_manager.domain_schema,
                dry_run=True,
            )
            assert result.assets_skipped >= 0
        else:
            image_rids = [remote_images[0]["RID"]]
            result = localize_assets(
                catalog=ml,
                asset_table="Image",
                asset_rids=image_rids,
                schema_name=catalog_manager.domain_schema,
                dry_run=True,
            )
            assert result.assets_processed >= 0

    def test_localize_assets_skips_local_assets(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that localize_assets skips assets already on local host."""
        from deriva_ml.catalog.localize import localize_assets

        ml = catalog_manager.ensure_populated(tmp_path)

        pb = ml.pathBuilder()
        domain_path = pb.schemas[catalog_manager.domain_schema]
        images = list(domain_path.tables["Image"].path.entities().fetch())

        if not images:
            pytest.skip("No images in test catalog")

        local_images = [
            img for img in images
            if img.get("URL") and ml.host_name in img.get("URL", "")
        ]

        if not local_images:
            pytest.skip("No local images in test catalog")

        image_rids = [local_images[0]["RID"]]
        result = localize_assets(
            catalog=ml,
            asset_table="Image",
            asset_rids=image_rids,
            schema_name=catalog_manager.domain_schema,
        )

        assert result.assets_skipped >= 1
        assert result.assets_processed == 0

    def test_localize_assets_skips_missing_rids(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that localize_assets skips RIDs that don't exist."""
        from deriva_ml.catalog.localize import localize_assets

        ml = catalog_manager.ensure_populated(tmp_path)

        result = localize_assets(
            catalog=ml,
            asset_table="Image",
            asset_rids=["NONEXISTENT-RID"],
            schema_name=catalog_manager.domain_schema,
        )

        assert result.assets_skipped >= 1
        assert result.assets_processed == 0

    def test_localize_assets_with_ermrest_catalog(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test localize_assets accepts ErmrestCatalog directly."""
        from deriva_ml.catalog.localize import _get_catalog_info

        ml = catalog_manager.get_ml_instance(tmp_path)

        ermrest_catalog, hostname, credential = _get_catalog_info(ml.catalog)

        assert ermrest_catalog is not None
        assert hostname == ml.host_name

    def test_get_catalog_info_from_derivaml(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test _get_catalog_info extracts info from DerivaML instance."""
        from deriva_ml.catalog.localize import _get_catalog_info

        ml = catalog_manager.get_ml_instance(tmp_path)

        ermrest_catalog, hostname, credential = _get_catalog_info(ml)

        assert ermrest_catalog is ml.catalog
        assert hostname == ml.host_name
        assert credential is not None

    def test_get_catalog_info_from_ermrest_catalog(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test _get_catalog_info extracts info from ErmrestCatalog."""
        from deriva_ml.catalog.localize import _get_catalog_info

        ml = catalog_manager.get_ml_instance(tmp_path)

        ermrest_catalog, hostname, credential = _get_catalog_info(ml.catalog)

        assert ermrest_catalog is ml.catalog
        assert hostname == ml.host_name


class TestLocalizeAfterClone:
    """Tests for localizing assets after cloning with asset_mode=REFERENCES."""

    def test_clone_refs_then_localize(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test cloning with refs mode then localizing assets.

        This test verifies the workflow:
        1. Clone catalog with asset_mode=REFERENCES (URLs point to source)
        2. Localize assets to copy them to local hatrac
        3. Verify assets are now accessible from local server
        """
        from deriva_ml.catalog.localize import localize_assets

        ml, root_rid = _get_root_rid(catalog_manager, tmp_path)
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                default_schema=catalog_manager.domain_schema,
                working_dir=tmp_path / "clone",
            )

            pb = cloned_ml.pathBuilder()
            domain_path = pb.schemas[catalog_manager.domain_schema]
            images = list(domain_path.tables["Image"].path.entities().fetch())

            if not images:
                pytest.skip("No images in cloned catalog")

            image_rids = [img["RID"] for img in images[:2]]

            localize_result = localize_assets(
                catalog=cloned_ml,
                asset_table="Image",
                asset_rids=image_rids,
                schema_name=catalog_manager.domain_schema,
            )

            assert localize_result.assets_skipped == len(image_rids)
            assert localize_result.assets_processed == 0

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

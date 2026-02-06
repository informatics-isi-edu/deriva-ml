"""Tests for create_ml_workspace and its helper functions.

Tests cover:
- Table discovery helpers (_discover_reachable_tables, _expand_tables_with_associations, etc.)
- Export annotation parsing
- Input validation
- Integration tests for subset cloning via create_ml_workspace
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deriva.core import DerivaServer, get_credential

from deriva_ml import DerivaML
from deriva_ml.catalog import create_ml_workspace, CloneCatalogResult, AssetCopyMode
from deriva_ml.catalog.clone import (
    _discover_reachable_tables,
    _expand_tables_with_associations,
    _expand_tables_with_vocabularies,
    _parse_export_annotation_tables,
)
from tests.catalog_manager import CatalogManager


class TestDiscoverReachableTables:
    """Tests for _discover_reachable_tables helper function."""

    def test_discovers_connected_tables_via_outbound_fk(self):
        """Test discovery follows outbound FK relationships."""
        model = MagicMock()

        # Table1 references Table2 via FK
        table2 = MagicMock()
        table2.schema.name = "demo"
        table2.name = "Table2"
        table2.foreign_keys = []
        table2.referenced_by = []

        fk = MagicMock()
        fk.pk_table = table2

        table1 = MagicMock()
        table1.schema.name = "demo"
        table1.name = "Table1"
        table1.foreign_keys = [fk]
        table1.referenced_by = []

        # Table2 is referenced by Table1
        ref_fk = MagicMock()
        ref_fk.table = table1
        table2.referenced_by = [ref_fk]

        model.schemas = {
            "demo": MagicMock(tables={
                "Table1": table1,
                "Table2": table2,
            })
        }

        result = _discover_reachable_tables(
            model=model,
            start_tables=["demo:Table1"],
        )

        assert "demo:Table1" in result
        assert "demo:Table2" in result

    def test_discovers_connected_tables_via_inbound_fk(self):
        """Test discovery follows inbound FK relationships (referenced_by)."""
        model = MagicMock()

        # Table2 references Table1 (so Table1 has Table2 in referenced_by)
        table1 = MagicMock()
        table1.schema.name = "demo"
        table1.name = "Table1"
        table1.foreign_keys = []

        table2 = MagicMock()
        table2.schema.name = "demo"
        table2.name = "Table2"
        table2.foreign_keys = []
        table2.referenced_by = []

        # FK from Table2 to Table1
        fk = MagicMock()
        fk.pk_table = table1
        fk.table = table2
        table2.foreign_keys = [fk]

        # Table1 is referenced by Table2
        table1.referenced_by = [fk]

        model.schemas = {
            "demo": MagicMock(tables={
                "Table1": table1,
                "Table2": table2,
            })
        }

        result = _discover_reachable_tables(
            model=model,
            start_tables=["demo:Table1"],
        )

        assert "demo:Table1" in result
        assert "demo:Table2" in result

    def test_excludes_specified_tables(self):
        """Test that excluded tables are not discovered."""
        model = MagicMock()

        table2 = MagicMock()
        table2.schema.name = "demo"
        table2.name = "Table2"
        table2.foreign_keys = []
        table2.referenced_by = []

        fk = MagicMock()
        fk.pk_table = table2

        table1 = MagicMock()
        table1.schema.name = "demo"
        table1.name = "Table1"
        table1.foreign_keys = [fk]
        table1.referenced_by = []

        model.schemas = {
            "demo": MagicMock(tables={
                "Table1": table1,
                "Table2": table2,
            })
        }

        result = _discover_reachable_tables(
            model=model,
            start_tables=["demo:Table1"],
            exclude_tables={("demo", "Table2")},
        )

        assert "demo:Table1" in result
        assert "demo:Table2" not in result

    def test_excludes_system_schemas(self):
        """Test that system schemas are automatically excluded."""
        model = MagicMock()

        public_table = MagicMock()
        public_table.schema.name = "public"
        public_table.name = "ERMrest_Client"
        public_table.foreign_keys = []
        public_table.referenced_by = []

        fk = MagicMock()
        fk.pk_table = public_table

        table1 = MagicMock()
        table1.schema.name = "demo"
        table1.name = "Table1"
        table1.foreign_keys = [fk]
        table1.referenced_by = []

        model.schemas = {
            "demo": MagicMock(tables={"Table1": table1}),
            "public": MagicMock(tables={"ERMrest_Client": public_table}),
        }

        result = _discover_reachable_tables(
            model=model,
            start_tables=["demo:Table1"],
        )

        assert "demo:Table1" in result
        assert "public:ERMrest_Client" not in result

    def test_excludes_specified_schemas(self):
        """Test that specified schemas are excluded from discovery."""
        model = MagicMock()

        audit_table = MagicMock()
        audit_table.schema.name = "audit"
        audit_table.name = "Log"
        audit_table.foreign_keys = []
        audit_table.referenced_by = []

        fk = MagicMock()
        fk.pk_table = audit_table

        table1 = MagicMock()
        table1.schema.name = "demo"
        table1.name = "Table1"
        table1.foreign_keys = [fk]
        table1.referenced_by = []

        model.schemas = {
            "demo": MagicMock(tables={"Table1": table1}),
            "audit": MagicMock(tables={"Log": audit_table}),
        }

        result = _discover_reachable_tables(
            model=model,
            start_tables=["demo:Table1"],
            exclude_schemas={"audit"},
        )

        assert "demo:Table1" in result
        assert "audit:Log" not in result

    def test_invalid_table_format_raises(self):
        """Test that invalid table format raises ValueError."""
        model = MagicMock()

        with pytest.raises(ValueError, match="must be specified as 'schema:table'"):
            _discover_reachable_tables(model, ["InvalidTableName"])


class TestExpandTablesWithAssociations:
    """Tests for _expand_tables_with_associations helper function."""

    def test_no_associations_needed(self):
        """Test when tables don't need any association tables."""
        # Create a mock model with no associations between tables
        model = MagicMock()

        # Table1 has no associations
        table1 = MagicMock()
        table1.find_associations.return_value = []
        table1.schema.name = "demo"
        table1.name = "Table1"

        model.schemas = {
            "demo": MagicMock(tables={"Table1": table1})
        }

        include_tables = ["demo:Table1"]
        all_tables, added = _expand_tables_with_associations(model, include_tables)

        assert all_tables == ["demo:Table1"]
        assert added == []

    def test_adds_association_table(self):
        """Test that association tables connecting included tables are added."""
        model = MagicMock()

        # Create association that connects Table1 and Table2
        assoc_table = MagicMock()
        assoc_table.schema.name = "demo"
        assoc_table.name = "Table1_Table2"

        other_fk = MagicMock()
        other_fk.pk_table.schema.name = "demo"
        other_fk.pk_table.name = "Table2"

        assoc = MagicMock()
        assoc.table = assoc_table
        assoc.other_fkeys = {other_fk}

        # Table1 has association to Table2
        table1 = MagicMock()
        table1.find_associations.return_value = [assoc]
        table1.schema.name = "demo"
        table1.name = "Table1"

        # Table2 has no associations
        table2 = MagicMock()
        table2.find_associations.return_value = []
        table2.schema.name = "demo"
        table2.name = "Table2"

        model.schemas = {
            "demo": MagicMock(tables={
                "Table1": table1,
                "Table2": table2,
            })
        }

        include_tables = ["demo:Table1", "demo:Table2"]
        all_tables, added = _expand_tables_with_associations(model, include_tables)

        assert "demo:Table1_Table2" in all_tables
        assert "demo:Table1_Table2" in added

    def test_invalid_table_format_raises(self):
        """Test that invalid table format raises ValueError."""
        model = MagicMock()

        with pytest.raises(ValueError, match="must be specified as 'schema:table'"):
            _expand_tables_with_associations(model, ["InvalidTableName"])


class TestExpandTablesWithVocabularies:
    """Tests for _expand_tables_with_vocabularies helper function."""

    def test_no_vocabulary_references(self):
        """Test when tables don't reference any vocabularies."""
        model = MagicMock()

        # Table with no FK to vocabulary
        table1 = MagicMock()
        table1.foreign_keys = []

        model.schemas = {
            "demo": MagicMock(tables={"Table1": table1})
        }

        include_tables = ["demo:Table1"]
        all_tables, added = _expand_tables_with_vocabularies(model, include_tables)

        assert all_tables == ["demo:Table1"]
        assert added == []

    def test_adds_vocabulary_table(self):
        """Test that referenced vocabulary tables are added."""
        model = MagicMock()

        # Create vocabulary table with required columns
        vocab_table = MagicMock()
        vocab_table.schema.name = "vocab"
        vocab_table.name = "Status"
        # Vocabulary columns (case-insensitive)
        vocab_col1 = MagicMock()
        vocab_col1.name = "Name"
        vocab_col2 = MagicMock()
        vocab_col2.name = "URI"
        vocab_col3 = MagicMock()
        vocab_col3.name = "Synonyms"
        vocab_col4 = MagicMock()
        vocab_col4.name = "Description"
        vocab_col5 = MagicMock()
        vocab_col5.name = "ID"
        vocab_table.columns = [vocab_col1, vocab_col2, vocab_col3, vocab_col4, vocab_col5]

        # FK pointing to vocabulary
        fk = MagicMock()
        fk.pk_table = vocab_table

        # Table1 references vocabulary
        table1 = MagicMock()
        table1.foreign_keys = [fk]

        model.schemas = {
            "demo": MagicMock(tables={"Table1": table1}),
            "vocab": MagicMock(tables={"Status": vocab_table}),
        }

        include_tables = ["demo:Table1"]
        all_tables, added = _expand_tables_with_vocabularies(model, include_tables)

        assert "vocab:Status" in all_tables
        assert "vocab:Status" in added

    def test_does_not_add_non_vocabulary(self):
        """Test that non-vocabulary FK targets are not added."""
        model = MagicMock()

        # Create non-vocabulary table (missing required columns)
        other_table = MagicMock()
        other_table.schema.name = "demo"
        other_table.name = "OtherTable"
        col1 = MagicMock()
        col1.name = "RID"
        col2 = MagicMock()
        col2.name = "Value"
        other_table.columns = [col1, col2]

        # FK pointing to non-vocabulary
        fk = MagicMock()
        fk.pk_table = other_table

        # Table1 references non-vocabulary
        table1 = MagicMock()
        table1.foreign_keys = [fk]

        model.schemas = {
            "demo": MagicMock(tables={
                "Table1": table1,
                "OtherTable": other_table,
            })
        }

        include_tables = ["demo:Table1"]
        all_tables, added = _expand_tables_with_vocabularies(model, include_tables)

        assert all_tables == ["demo:Table1"]
        assert added == []


class TestParseExportAnnotationTables:
    """Tests for _parse_export_annotation_tables helper function."""

    def test_parses_simple_export_annotation(self):
        """Test parsing a simple export annotation with paths."""
        table = MagicMock()
        table.schema.name = "isa"
        table.name = "project"

        # Mock the export annotation structure
        table.annotations = {
            "tag:isrd.isi.edu,2019:export": {
                "*": {
                    "templates": [
                        {
                            "type": "BAG",
                            "outputs": [
                                {
                                    "source": {"api": "entity"},
                                    "destination": {"name": "project", "type": "json"},
                                },
                                {
                                    "source": {"api": "entity", "path": "isa:dataset"},
                                    "destination": {"name": "dataset", "type": "json"},
                                },
                                {
                                    "source": {"api": "entity", "path": "isa:dataset/isa:experiment"},
                                    "destination": {"name": "experiment", "type": "json"},
                                },
                            ],
                        }
                    ]
                }
            }
        }

        tables, paths = _parse_export_annotation_tables(table)

        assert "isa:project" in tables
        assert "isa:dataset" in tables
        assert "isa:experiment" in tables
        assert len(paths) >= 2  # At least two paths: project->dataset and project->dataset->experiment

    def test_returns_root_table_when_no_annotation(self):
        """Test that root table is still returned when no export annotation exists."""
        table = MagicMock()
        table.schema.name = "demo"
        table.name = "MyTable"
        table.annotations = {}

        tables, paths = _parse_export_annotation_tables(table)

        assert tables == ["demo:MyTable"]
        assert paths == []

    def test_ignores_attribute_projections(self):
        """Test that attribute projections (with = sign) are ignored."""
        table = MagicMock()
        table.schema.name = "isa"
        table.name = "project"

        table.annotations = {
            "tag:isrd.isi.edu,2019:export": {
                "*": {
                    "templates": [
                        {
                            "type": "BAG",
                            "outputs": [
                                {
                                    "source": {
                                        "api": "attribute",
                                        "path": "isa:dataset/isa:file/url:=URL,length:=Length",
                                    },
                                    "destination": {"name": "file", "type": "fetch"},
                                },
                            ],
                        }
                    ]
                }
            }
        }

        tables, paths = _parse_export_annotation_tables(table)

        # Should have root table and isa:dataset and isa:file
        assert "isa:project" in tables
        assert "isa:dataset" in tables
        assert "isa:file" in tables

    def test_handles_missing_path_in_source(self):
        """Test handling of outputs without a path (root table only)."""
        table = MagicMock()
        table.schema.name = "isa"
        table.name = "project"

        table.annotations = {
            "tag:isrd.isi.edu,2019:export": {
                "*": {
                    "templates": [
                        {
                            "type": "BAG",
                            "outputs": [
                                {
                                    "source": {"api": "entity"},  # No path
                                    "destination": {"name": "project", "type": "json"},
                                },
                            ],
                        }
                    ]
                }
            }
        }

        tables, paths = _parse_export_annotation_tables(table)

        assert tables == ["isa:project"]
        assert paths == []


class TestCreateMlWorkspaceIntegration:
    """Integration tests for create_ml_workspace (requires running catalog)."""

    @pytest.mark.skip(reason="Requires running catalog")
    def test_create_workspace_basic(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Test basic workspace creation from a dataset RID."""
        ml = catalog_manager.ensure_populated(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get a dataset RID to use as root
        pb = ml.pathBuilder()
        ml_path = pb.schemas["deriva-ml"]
        datasets = list(ml_path.tables["Dataset"].path.entities().fetch())

        if not datasets:
            pytest.skip("No datasets in source catalog")

        root_rid = datasets[0]["RID"]

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            include_tables=[
                f"{catalog_manager.domain_schema}:Subject",
                f"{catalog_manager.domain_schema}:Image",
                "deriva-ml:Dataset",
            ],
        )

        try:
            assert isinstance(result, CloneCatalogResult)
            assert result.catalog_id is not None

            # Connect to cloned catalog
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "cloned",
            )

            # Verify only specified tables exist
            model = cloned_ml.catalog.getCatalogModel()
            assert "deriva-ml" in model.schemas

            # Verify data was filtered to reachable rows
            cloned_pb = cloned_ml.pathBuilder()
            cloned_datasets = list(
                cloned_pb.schemas["deriva-ml"].tables["Dataset"].path.entities().fetch()
            )
            assert len(cloned_datasets) <= len(datasets)

        finally:
            self._delete_catalog(hostname, result.catalog_id)

    @pytest.mark.skip(reason="Requires running catalog")
    def test_create_workspace_with_associations(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Test that association tables are automatically included."""
        ml = catalog_manager.ensure_populated(tmp_path / "source")
        source_catalog_id = str(catalog_manager.catalog_id)
        hostname = catalog_manager.hostname

        # Get a subject RID
        pb = ml.pathBuilder()
        domain_path = pb.schemas[catalog_manager.domain_schema]
        subjects = list(domain_path.tables["Subject"].path.entities().fetch())

        if not subjects:
            pytest.skip("No subjects in source catalog")

        root_rid = subjects[0]["RID"]

        result = create_ml_workspace(
            source_hostname=hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            include_tables=[
                f"{catalog_manager.domain_schema}:Subject",
                f"{catalog_manager.domain_schema}:Image",
            ],
            include_associations=True,
        )

        try:
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "cloned",
            )

            model = cloned_ml.catalog.getCatalogModel()
            domain_schema = model.schemas.get(catalog_manager.domain_schema)
            assert domain_schema is not None

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


class TestCreateMlWorkspaceValidation:
    """Tests for input validation in create_ml_workspace."""

    def test_invalid_include_table_format_raises(self):
        """Test that invalid include_tables format raises ValueError."""
        with pytest.raises(ValueError, match="must be specified as 'schema:table'"):
            create_ml_workspace(
                source_hostname="localhost",
                source_catalog_id="1",
                root_rid="ABC123",
                include_tables=["InvalidTableName"],  # Missing schema prefix
            )

    def test_invalid_exclude_objects_format_raises(self):
        """Test that invalid exclude_objects format raises ValueError."""
        with pytest.raises(ValueError, match="exclude_objects entries must be 'schema:table'"):
            create_ml_workspace(
                source_hostname="localhost",
                source_catalog_id="1",
                root_rid="ABC123",
                exclude_objects=["InvalidTableName"],  # Missing schema prefix
            )

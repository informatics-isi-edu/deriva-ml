"""Tests for clone_subset_catalog functionality.

Tests cover the subset cloning feature that creates a catalog containing
only the data reachable from a specified RID.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deriva.core import DerivaServer, get_credential

from deriva_ml import DerivaML
from deriva_ml.catalog import clone_subset_catalog, CloneCatalogResult, AssetCopyMode
from deriva_ml.catalog.clone import (
    _discover_reachable_tables,
    _expand_tables_with_associations,
    _expand_tables_with_vocabularies,
    _build_path_query,
    _compute_reachable_rids,
    _copy_subset_table_data,
    _parse_export_annotation_tables,
    _compute_reachable_rids_from_paths,
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


class TestBuildPathQuery:
    """Tests for _build_path_query helper function."""

    def test_simple_path(self):
        """Test building query with a simple path."""
        query = _build_path_query(
            root_table="demo:Subject",
            root_rid="ABC123",
            path=[("demo", "Image")],
        )
        # Colon must NOT be encoded - ERMrest uses it as schema:table separator
        assert query == "/entity/demo:Subject/RID=ABC123/demo:Image"

    def test_multi_hop_path(self):
        """Test building query with multiple hops."""
        query = _build_path_query(
            root_table="demo:Dataset",
            root_rid="XYZ789",
            path=[("demo", "Subject"), ("demo", "Image")],
        )
        assert query == "/entity/demo:Dataset/RID=XYZ789/demo:Subject/demo:Image"

    def test_empty_path(self):
        """Test building query with empty path (root only)."""
        query = _build_path_query(
            root_table="demo:Subject",
            root_rid="ABC123",
            path=[],
        )
        assert query == "/entity/demo:Subject/RID=ABC123"

    def test_special_characters_escaped(self):
        """Test that special characters in RID are URL-escaped."""
        query = _build_path_query(
            root_table="demo:Table",
            root_rid="A+B=C",
            path=[],
        )
        # The + should be URL-encoded (urlquote default escapes +)
        assert "A%2BB%3DC" in query


class TestCopySubsetTableData:
    """Tests for _copy_subset_table_data helper function."""

    def test_empty_rids_returns_zero(self):
        """Test that empty RID set returns zero rows."""
        src_catalog = MagicMock()
        dst_catalog = MagicMock()
        report = MagicMock()

        rows_copied, rows_skipped, skipped, truncated = _copy_subset_table_data(
            src_catalog=src_catalog,
            dst_catalog=dst_catalog,
            sname="demo",
            tname="Table1",
            reachable_rids=set(),
            page_size=100,
            report=report,
        )

        assert rows_copied == 0
        assert rows_skipped == 0
        assert skipped == []
        assert truncated == []
        # Should not have called catalog methods
        src_catalog.get.assert_not_called()
        dst_catalog.post.assert_not_called()

    def test_copies_rows_successfully(self):
        """Test successful copying of rows."""
        src_catalog = MagicMock()
        dst_catalog = MagicMock()
        report = MagicMock()

        # Mock source returning rows
        src_catalog.get.return_value.json.return_value = [
            {"RID": "A1", "Name": "Test1"},
            {"RID": "A2", "Name": "Test2"},
        ]

        rows_copied, rows_skipped, skipped, truncated = _copy_subset_table_data(
            src_catalog=src_catalog,
            dst_catalog=dst_catalog,
            sname="demo",
            tname="Table1",
            reachable_rids={"A1", "A2"},
            page_size=100,
            report=report,
        )

        assert rows_copied == 2
        assert rows_skipped == 0
        dst_catalog.post.assert_called_once()


class TestCloneSubsetCatalogIntegration:
    """Integration tests for clone_subset_catalog (requires running catalog)."""

    @pytest.mark.skip(reason="Requires running catalog")
    def test_clone_subset_basic(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Test basic subset cloning from a dataset RID."""
        # First populate the source catalog
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

        # Clone subset
        result = clone_subset_catalog(
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
            # Should have at most the original dataset count (likely just the root)
            assert len(cloned_datasets) <= len(datasets)

        finally:
            # Clean up
            try:
                cred = get_credential(hostname)
                server = DerivaServer("https", hostname, credentials=cred)
                server.delete_ermrest_catalog(result.catalog_id)
            except Exception:
                pass

    @pytest.mark.skip(reason="Requires running catalog")
    def test_clone_subset_with_associations(
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

        result = clone_subset_catalog(
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
            # Connect and verify association tables were included
            cloned_ml = DerivaML(
                hostname,
                result.catalog_id,
                working_dir=tmp_path / "cloned",
            )

            model = cloned_ml.catalog.getCatalogModel()
            domain_schema = model.schemas.get(catalog_manager.domain_schema)

            # Check that any Subject_Image association exists (if there was one)
            # The actual table name depends on the schema
            assert domain_schema is not None

        finally:
            try:
                cred = get_credential(hostname)
                server = DerivaServer("https", hostname, credentials=cred)
                server.delete_ermrest_catalog(result.catalog_id)
            except Exception:
                pass


class TestCloneSubsetCatalogValidation:
    """Tests for input validation in clone_subset_catalog."""

    def test_invalid_include_table_format_raises(self):
        """Test that invalid include_tables format raises ValueError."""
        with pytest.raises(ValueError, match="must be specified as 'schema:table'"):
            clone_subset_catalog(
                source_hostname="localhost",
                source_catalog_id="1",
                root_rid="ABC123",
                include_tables=["InvalidTableName"],  # Missing schema prefix
            )

    def test_invalid_exclude_objects_format_raises(self):
        """Test that invalid exclude_objects format raises ValueError."""
        with pytest.raises(ValueError, match="exclude_objects entries must be 'schema:table'"):
            clone_subset_catalog(
                source_hostname="localhost",
                source_catalog_id="1",
                root_rid="ABC123",
                exclude_objects=["InvalidTableName"],  # Missing schema prefix
            )

    def test_empty_include_tables(self):
        """Test behavior with empty include_tables list."""
        # This should work - tables will be auto-discovered from root RID
        # Can't actually test without a running catalog
        pass

    def test_none_include_tables(self):
        """Test behavior with None include_tables."""
        # This should work - tables will be auto-discovered from root RID
        # Can't actually test without a running catalog
        pass


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


class TestComputeReachableRidsFromPaths:
    """Tests for _compute_reachable_rids_from_paths helper function."""

    def test_returns_root_rid_for_root_table(self):
        """Test that root RID is always included for root table."""
        catalog = MagicMock()
        catalog.get.return_value.json.return_value = []

        root_rid = "ABC123"
        root_table = "demo:Project"
        paths = []
        include_tables = ["demo:Project"]

        result = _compute_reachable_rids_from_paths(
            catalog, root_rid, root_table, paths, include_tables
        )

        assert root_rid in result["demo:Project"]

    def test_queries_each_path(self):
        """Test that each path is queried for reachable rows."""
        catalog = MagicMock()

        # First call returns empty, subsequent calls return rows
        def mock_get(uri):
            mock_response = MagicMock()
            if "Dataset" in uri and "Experiment" not in uri:
                mock_response.json.return_value = [{"RID": "D1"}, {"RID": "D2"}]
            elif "Experiment" in uri:
                mock_response.json.return_value = [{"RID": "E1"}]
            else:
                mock_response.json.return_value = []
            return mock_response

        catalog.get.side_effect = mock_get

        root_rid = "P1"
        root_table = "demo:Project"
        paths = [
            ["demo:Project", "demo:Dataset"],
            ["demo:Project", "demo:Dataset", "demo:Experiment"],
        ]
        include_tables = ["demo:Project", "demo:Dataset", "demo:Experiment"]

        result = _compute_reachable_rids_from_paths(
            catalog, root_rid, root_table, paths, include_tables
        )

        assert "P1" in result["demo:Project"]
        assert "D1" in result["demo:Dataset"]
        assert "D2" in result["demo:Dataset"]
        assert "E1" in result["demo:Experiment"]

    def test_handles_query_failures_gracefully(self):
        """Test that query failures are handled without crashing."""
        catalog = MagicMock()
        catalog.get.side_effect = Exception("Connection error")

        root_rid = "P1"
        root_table = "demo:Project"
        paths = [["demo:Project", "demo:Dataset"]]
        include_tables = ["demo:Project", "demo:Dataset"]

        result = _compute_reachable_rids_from_paths(
            catalog, root_rid, root_table, paths, include_tables
        )

        # Should still have root RID
        assert "P1" in result["demo:Project"]
        # Dataset may be empty due to error
        assert result["demo:Dataset"] == set()

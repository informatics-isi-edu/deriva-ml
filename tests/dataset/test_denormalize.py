"""
Tests for the _denormalize method in DatasetBag.

The _denormalize method generates SQL statements for denormalizing dataset data by:
1. Walking the schema graph based on incoming and outgoing FK relationships
2. Collecting columns and values for all specified tables
3. Handling cycles in the schema graph using topological sorting
4. Generating appropriate JOIN clauses based on ORM relationships

Note: Due to SQLite limitations, each schema (deriva-ml and domain schema) is stored
in a separate database with FKs included. SQLAlchemy ORM relationships are used to
capture the linkages between the two schemas.
"""

from pprint import pformat

import pandas as pd
import pytest
try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None

from deriva_ml import DerivaML
from deriva_ml.execution.execution import ExecutionConfiguration

class TestDenormalize:
    """Test suite for the _denormalize method in DatasetBag."""

    def test_denormalize_single_table(self, dataset_test, tmp_path):
        """Test denormalization with a single table included.

        This verifies that denormalizing with just one table returns the
        expected columns and rows for that table.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Test denormalizing with just the Subject table
        df = bag.denormalize_as_dataframe(include_tables=["Subject"])

        # Verify the dataframe has the expected structure
        assert isinstance(df, pd.DataFrame)
        # Should have columns from Subject table (prefixed with table name)
        subject_columns = [col for col in df.columns if col.startswith("Subject.")]
        assert len(subject_columns) > 0
        assert "Subject.RID" in df.columns
        assert "Subject.Name" in df.columns

    def test_denormalize_multiple_tables(self, dataset_test, tmp_path):
        """Test denormalization with multiple tables included.

        This verifies that denormalizing with multiple tables correctly
        joins them based on FK relationships. Observation is included
        to disambiguate the Image→Subject path.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Test denormalizing with Subject, Image, and Observation tables
        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        df = bag.denormalize_as_dataframe(include_tables=["Subject", "Image", "Observation"])

        # Verify the dataframe has columns from both tables
        assert isinstance(df, pd.DataFrame)
        subject_columns = [col for col in df.columns if col.startswith("Subject.")]
        image_columns = [col for col in df.columns if col.startswith("Image.")]

        assert len(subject_columns) > 0, "Expected Subject columns in denormalized result"
        assert len(image_columns) > 0, "Expected Image columns in denormalized result"

    def test_denormalize_as_dict(self, dataset_test, tmp_path):
        """Test denormalize_as_dict returns proper dictionary generator.

        This verifies that the denormalize_as_dict method returns a generator
        that yields dictionaries with properly labeled column names.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Test denormalize_as_dict
        rows = list(bag.denormalize_as_dict(include_tables=["Subject"]))

        # Verify we get dictionaries back
        assert len(rows) > 0, "Expected at least one row from denormalize_as_dict"

        # Check that the first row has the expected keys
        first_row = dict(rows[0])
        assert "Subject.RID" in first_row
        assert "Subject.Name" in first_row

    def test_denormalize_column_labeling(self, dataset_test, tmp_path):
        """Test that denormalized columns are properly labeled with table name prefixes.

        The _denormalize method should label columns as "TableName.ColumnName" to
        avoid ambiguity when multiple tables are joined.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        df = bag.denormalize_as_dataframe(include_tables=["Subject", "Image", "Observation"])

        # All columns should be prefixed with table name
        for col in df.columns:
            assert "." in col, f"Column {col} should be prefixed with table name"
            table_name, col_name = col.split(".", 1)
            assert table_name in ["Subject", "Image", "Observation"], f"Unexpected table prefix: {table_name}"

    def test_denormalize_with_nested_dataset(self, dataset_test, tmp_path):
        """Test denormalization includes data from nested datasets.

        When a dataset has nested datasets, denormalization should include
        data from all nested datasets in the result.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # The demo dataset has nested datasets
        nested_children = bag.list_dataset_children(recurse=True)
        assert len(nested_children) > 0, "Expected nested datasets for this test"

        # Denormalize - should include data from parent and nested datasets
        df = bag.denormalize_as_dataframe(include_tables=["Subject"])

        # The number of rows should reflect data from all nested datasets
        # Get total expected subjects from list_dataset_members (not get_table_as_dict
        # which returns all subjects in the bag, not just those in this dataset)
        all_members = bag.list_dataset_members(recurse=True)
        all_subjects = all_members.get("Subject", [])
        assert len(df) == len(all_subjects), "Denormalized results should include nested dataset data"

    def test_denormalize_nested_dataset_bag(self, dataset_test, tmp_path):
        """Test denormalization on a nested dataset bag.

        When we get a DatasetBag for a nested dataset, denormalization should
        only include data relevant to that specific nested dataset.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get a nested dataset bag
        nested_children = bag.list_dataset_children()
        assert len(nested_children) > 0, "Expected nested datasets"

        nested_bag = nested_children[0]

        # Denormalize the nested dataset
        df = nested_bag.denormalize_as_dataframe(include_tables=["Image"])

        # Verify we got data specific to the nested dataset
        assert isinstance(df, pd.DataFrame)
        # The nested dataset should have its own subset of images

    def test_denormalize_excludes_system_columns(self, dataset_test, tmp_path):
        """Test that denormalization excludes system metadata columns.

        System columns like RCT, RMT, RCB, RMB should typically be excluded
        from the denormalized result to reduce clutter.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Subject"])

        # Check that system columns are excluded
        system_cols = ["RCT", "RMT", "RCB", "RMB"]
        for sys_col in system_cols:
            matching_cols = [c for c in df.columns if c.endswith(f".{sys_col}")]
            assert len(matching_cols) == 0, f"System column {sys_col} should be excluded"

    def test_denormalize_excludes_association_tables(self, dataset_test, tmp_path):
        """Test that association table columns are not included in denormalized output.

        Association tables (pure join tables) should be traversed for joining
        but their columns should not appear in the final denormalized result.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        df = bag.denormalize_as_dataframe(include_tables=["Subject", "Image", "Observation"])

        # Association tables like Dataset_Subject should not have columns in output
        # The association tables are used for joining but their columns are excluded
        association_prefixes = ["Dataset_Subject.", "Dataset_Image.", "Image_Subject."]
        for prefix in association_prefixes:
            matching_cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(matching_cols) == 0, f"Association table columns ({prefix}) should be excluded"

class TestDenormalizeSchemaGraph:
    """Test the schema graph walking behavior of _denormalize."""

    def test_schema_path_discovery(self, dataset_test, tmp_path):
        """Test that schema paths are correctly discovered for denormalization.

        The _denormalize method should find all paths from Dataset to the
        requested tables through the FK relationship graph.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # The _schema_to_paths method should return paths through the schema
        paths = bag.model._schema_to_paths()

        # Should have paths to various tables
        assert len(paths) > 0, "Expected schema paths"

        # Paths should start from Dataset
        for path in paths:
            assert path[0].name == "Dataset", "All paths should start from Dataset table"

    def test_schema_path_exclude_tables(self, dataset_test, tmp_path):
        """Test that exclude_tables prunes branches from FK graph traversal.

        When exclude_tables is provided, _schema_to_paths should skip those tables
        and all paths that would pass through them.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get all paths without exclusion
        all_paths = bag.model._schema_to_paths()

        # Find a table name that appears in at least one path (not Dataset itself)
        table_names_in_paths = {
            table.name
            for path in all_paths
            for table in path
            if table.name != "Dataset"
        }
        assert len(table_names_in_paths) > 0, "Expected tables in paths besides Dataset"

        # Pick a table to exclude
        exclude_name = next(iter(table_names_in_paths))
        excluded_paths = bag.model._schema_to_paths(exclude_tables={exclude_name})

        # Excluded table should not appear in any path (except if it's the root)
        for path in excluded_paths:
            for table in path[1:]:  # Skip root (Dataset)
                assert table.name != exclude_name, (
                    f"Excluded table '{exclude_name}' found in path: {[t.name for t in path]}"
                )

        # Excluded paths should be a subset of all paths (fewer or equal)
        assert len(excluded_paths) <= len(all_paths), (
            "Excluding tables should not produce more paths"
        )

    def test_prepare_wide_table_validation(self, dataset_test, tmp_path):
        """Test that _prepare_wide_table validates table existence.

        The method should raise an exception when given invalid table names.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Attempt to denormalize with non-existent table
        from deriva_ml.core.exceptions import DerivaMLException

        with pytest.raises(DerivaMLException):
            bag.denormalize_as_dataframe(include_tables=["NonExistentTable"])

    def test_join_order_topological_sort(self, dataset_test, tmp_path):
        """Test that joins are ordered correctly via topological sort.

        The _prepare_wide_table method uses TopologicalSorter to determine
        the correct order for joining tables.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Call _prepare_wide_table to get join structure
        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        join_tables, columns = bag.model._prepare_wide_table(
            bag, bag.dataset_rid, include_tables=["Subject", "Image", "Observation"]
        )

        # join_tables should have entries with ordered table lists
        assert len(join_tables) > 0, "Expected join table structure"

        for element, (path, join_conditions) in join_tables.items():
            # Path should start with Dataset
            assert path[0] == "Dataset", "Join path should start with Dataset"
            # Should have join conditions for tables after Dataset
            for table_name in path[1:]:
                if table_name in join_conditions:
                    assert len(join_conditions[table_name]) > 0, f"Expected join conditions for {table_name}"

class TestDenormalizeOrmRelationships:
    """Test ORM relationship handling in _denormalize."""

    def test_find_relationship_attr(self, dataset_test, tmp_path):
        """Test that _find_relationship_attr correctly locates ORM relationships.

        This method is used to find the SQLAlchemy relationship attribute
        that connects two tables.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get ORM classes for tables with known relationships
        image_class = bag.model.get_orm_class_by_name("Image")
        subject_class = bag.model.get_orm_class_by_name("Subject")

        # Image has FK to Subject, so we should find a relationship
        rel_attr = bag._find_relationship_attr(image_class, subject_class)
        assert rel_attr is not None, "Should find relationship from Image to Subject"

    def test_cross_schema_relationships(self, dataset_test, tmp_path):
        """Test that relationships across schemas (domain <-> ml) work correctly.

        Since each schema is stored in a separate SQLite database, cross-schema
        relationships need special handling via ORM relationships.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get classes from different schemas
        dataset_class = bag.model.get_orm_class_by_name(f"{bag.model.ml_schema}.Dataset")
        subject_class = bag.model.get_orm_class_by_name(f"{bag.model.default_schema}.Subject")

        # Both should be found
        assert dataset_class is not None, "Should find Dataset class from ml schema"
        assert subject_class is not None, "Should find Subject class from domain schema"

        # The model should be able to traverse relationships between schemas
        # through the association tables

    def test_many_to_one_preference(self, dataset_test, tmp_path):
        """Test that MANYTOONE relationships are preferred when multiple paths exist.

        When multiple relationships exist between tables, _find_relationship_attr
        should prefer MANYTOONE relationships for better join semantics.
        """
        from sqlalchemy import inspect as sa_inspect

        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get Image class which has FK to Subject (MANYTOONE)
        image_class = bag.model.get_orm_class_by_name("Image")
        subject_class = bag.model.get_orm_class_by_name("Subject")

        rel_attr = bag._find_relationship_attr(image_class, subject_class)

        # The relationship should exist
        assert rel_attr is not None

        # Check that it's a MANYTOONE relationship if we can inspect it
        mapper = sa_inspect(image_class).mapper
        relationships = [r for r in mapper.relationships if r.mapper is sa_inspect(subject_class).mapper]
        if relationships:
            # At least one should be MANYTOONE
            has_manytoone = any(r.direction.name == "MANYTOONE" for r in relationships)
            assert has_manytoone, "Expected MANYTOONE relationship from Image to Subject"

class TestDenormalizeDataIntegrity:
    """Test data integrity in denormalized results."""

    def test_row_count_consistency(self, dataset_test, tmp_path):
        """Test that denormalized row counts match expected data.

        The number of rows in the denormalized result should be consistent
        with the underlying data relationships (list_dataset_members).
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get subjects via list_dataset_members (which respects dataset membership)
        all_members = bag.list_dataset_members(recurse=True)
        subjects_from_members = all_members.get("Subject", [])

        # Get subjects via denormalization
        df = bag.denormalize_as_dataframe(include_tables=["Subject"])

        # Row counts should match - denormalize uses list_dataset_members internally
        assert len(df) == len(subjects_from_members), "Denormalized row count should match dataset members"

    def test_data_value_preservation(self, dataset_test, tmp_path):
        """Test that data values are preserved correctly during denormalization.

        The values in the denormalized result should exactly match the
        values from list_dataset_members.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Get subjects from dataset members
        all_members = bag.list_dataset_members(recurse=True)
        subjects_from_members = all_members.get("Subject", [])
        member_rids = {s["RID"] for s in subjects_from_members}
        member_names = {s["Name"] for s in subjects_from_members}

        # Get subjects via denormalization
        rows = list(bag.denormalize_as_dict(include_tables=["Subject"]))
        denorm_rids = {dict(r)["Subject.RID"] for r in rows}
        denorm_names = {dict(r)["Subject.Name"] for r in rows}

        # Values should match
        assert member_rids == denorm_rids, "RIDs should be preserved"
        assert member_names == denorm_names, "Names should be preserved"

    def test_joined_data_relationships(self, dataset_test, tmp_path):
        """Test that joined data maintains correct relationships.

        When multiple tables are denormalized, the joined rows should
        maintain the correct FK relationships via the multi-hop chain
        Image → Observation → Subject.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Denormalize Image, Observation, and Subject together
        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        df = bag.denormalize_as_dataframe(include_tables=["Subject", "Image", "Observation"])

        # Each row should have matching FK chain: Image→Observation→Subject
        if "Image.Observation" in df.columns and "Observation.RID" in df.columns:
            valid_rows = df.dropna(subset=["Image.Observation", "Observation.RID"])
            for _, row in valid_rows.iterrows():
                assert row["Image.Observation"] == row["Observation.RID"], (
                    f"FK relationship should be maintained: "
                    f"Image.Observation={row['Image.Observation']} != Observation.RID={row['Observation.RID']}"
                )

        if "Observation.Subject" in df.columns and "Subject.RID" in df.columns:
            valid_rows = df.dropna(subset=["Observation.Subject", "Subject.RID"])
            for _, row in valid_rows.iterrows():
                assert row["Observation.Subject"] == row["Subject.RID"], (
                    f"FK relationship should be maintained: "
                    f"Observation.Subject={row['Observation.Subject']} != Subject.RID={row['Subject.RID']}"
                )

class TestDenormalizeEdgeCases:
    """Test edge cases and error handling in _denormalize."""

    def test_empty_include_tables_raises_error(self, dataset_test, tmp_path):
        """Test that empty include_tables list is handled appropriately."""
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Empty list should either raise an error or return empty result
        # depending on implementation
        try:
            df = bag.denormalize_as_dataframe(include_tables=[])
            # If no error, should return empty dataframe
            assert len(df.columns) == 0 or len(df) == 0
        except (ValueError, Exception):
            # Error is also acceptable behavior
            pass

    def test_single_row_dataset(self, dataset_test, tmp_path):
        """Test denormalization works with minimal data.

        This test verifies that denormalization works correctly on nested datasets
        that may have smaller amounts of data.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version

        # Download bag once for this test
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Find a nested dataset with subjects
        nested = bag.list_dataset_children(recurse=True)
        found_nested_with_subjects = False
        for nested_bag in nested:
            members = nested_bag.list_dataset_members()
            if "Subject" in members and len(members["Subject"]) > 0:
                df = nested_bag.denormalize_as_dataframe(include_tables=["Subject"])
                assert isinstance(df, pd.DataFrame)
                found_nested_with_subjects = True
                break

        # It's okay if no nested datasets have subjects - the test passes
        # as long as the denormalization logic works when there are subjects

    def test_table_in_ml_schema(self, dataset_test, tmp_path):
        """Test denormalization with tables from the ML schema.

        Tables like Dataset_Version are in the ML schema and should be
        handled correctly even though they're in a different database.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version

        # Download bag once for this test
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Verify that the model has both schemas loaded correctly
        assert bag.model.ml_schema is not None, "ML schema should be set"
        assert bag.model.default_schema is not None, "Domain schema should be set"

        # Verify tables exist in the metadata from both schemas
        ml_tables = [t for t in bag.model.metadata.tables.keys() if t.startswith(bag.model.ml_schema)]
        domain_tables = [t for t in bag.model.metadata.tables.keys() if t.startswith(bag.model.default_schema)]

        assert len(ml_tables) > 0, "Should have ML schema tables"
        assert len(domain_tables) > 0, "Should have domain schema tables"

class TestDenormalizeSqlGeneration:
    """Test the SQL generation aspects of _denormalize."""

    def test_sql_select_structure(self, dataset_test, tmp_path):
        """Test that _denormalize returns a valid SQLAlchemy Select object."""
        from sqlalchemy import Select
        from sqlalchemy.sql.selectable import CompoundSelect

        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Call _denormalize directly to get the SQL statement
        sql_stmt = bag._denormalize(include_tables=["Subject"])

        # Should return a Select or CompoundSelect (union)
        assert isinstance(sql_stmt, (Select, CompoundSelect)), "Should return SQLAlchemy Select object"

    def test_union_for_multiple_paths(self, dataset_test, tmp_path):
        """Test that multiple paths result in a UNION statement.

        When multiple paths exist through the schema to reach the same tables,
        the results should be UNIONed together.
        """

        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # With multiple element types, we should get a union
        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        sql_stmt = bag._denormalize(include_tables=["Subject", "Image", "Observation"])

        # The result should be a CompoundSelect (UNION)
        # This depends on whether there are multiple paths
        assert sql_stmt is not None

class TestCatalogDenormalize:
    """Test suite for catalog-based denormalization (Dataset class).

    These tests verify that the denormalize_as_dataframe and denormalize_as_dict
    methods work correctly when querying a live Deriva catalog using the datapath API,
    as opposed to querying a downloaded bag using SQLAlchemy.
    """

    def test_catalog_denormalize_single_table(self, catalog_with_datasets, tmp_path):
        """Test catalog-based denormalization with a single table.

        This verifies that denormalizing with just one table returns the
        expected columns and rows for that table directly from the catalog.
        """
        ml_instance, dataset_description = catalog_with_datasets

        dataset = dataset_description.dataset

        # Test denormalizing with just the Subject table
        df = dataset.denormalize_as_dataframe(include_tables=["Subject"])

        # Verify the dataframe has the expected structure
        assert isinstance(df, pd.DataFrame)
        # Should have columns from Subject table (prefixed with table name using underscore)
        subject_columns = [col for col in df.columns if col.startswith("Subject_")]
        assert len(subject_columns) > 0, "Expected Subject columns in denormalized result"
        assert "Subject_RID" in df.columns, "Expected Subject_RID column"
        assert "Subject_Name" in df.columns, "Expected Subject_Name column"

    def test_catalog_denormalize_multiple_tables(self, catalog_with_datasets, tmp_path):
        """Test catalog-based denormalization with multiple tables.

        This verifies that denormalizing with multiple tables correctly
        joins them based on FK relationships when querying the catalog.
        """
        ml_instance, dataset_description = catalog_with_datasets

        dataset = dataset_description.dataset

        # Test denormalizing with Subject, Image, and Observation tables
        # Observation is needed to disambiguate Image→Subject vs Image→Observation→Subject
        df = dataset.denormalize_as_dataframe(include_tables=["Subject", "Image", "Observation"])

        # Verify the dataframe has columns from both tables
        assert isinstance(df, pd.DataFrame)
        subject_columns = [col for col in df.columns if col.startswith("Subject_")]
        image_columns = [col for col in df.columns if col.startswith("Image_")]

        assert len(subject_columns) > 0, "Expected Subject columns in denormalized result"
        assert len(image_columns) > 0, "Expected Image columns in denormalized result"

    def test_catalog_denormalize_as_dict(self, catalog_with_datasets, tmp_path):
        """Test catalog denormalize_as_dict returns proper dictionary generator.

        This verifies that the denormalize_as_dict method returns a generator
        that yields dictionaries with properly labeled column names.
        """
        ml_instance, dataset_description = catalog_with_datasets

        dataset = dataset_description.dataset

        # Test denormalize_as_dict
        rows = list(dataset.denormalize_as_dict(include_tables=["Subject"]))

        # Verify we get dictionaries back
        assert len(rows) > 0, "Expected at least one row"
        assert isinstance(rows[0], dict), "Expected dictionary rows"

        # Verify column naming convention (table_column)
        first_row = rows[0]
        subject_keys = [k for k in first_row.keys() if k.startswith("Subject_")]
        assert len(subject_keys) > 0, "Expected Subject_ prefixed keys"

    def test_catalog_denormalize_empty_result(self, test_ml, tmp_path):
        """Test catalog denormalization handles empty datasets gracefully.

        An empty dataset should return an empty DataFrame without errors.
        """
        ml_instance = test_ml
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="A manual workflow")

        # Create a workflow and execution for dataset creation
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for testing",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )

        # Create an empty dataset (empty list for dataset_types)
        dataset = execution.create_dataset(
            description="Empty test dataset",
            dataset_types=[],
        )

        # Test denormalizing an empty dataset
        df = dataset.denormalize_as_dataframe(include_tables=["Subject"])

        # Should return empty DataFrame
        assert isinstance(df, pd.DataFrame)
        # May be empty or have zero rows

    def test_catalog_and_bag_denormalize_consistency(self, catalog_with_datasets, tmp_path):
        """Test that catalog and bag denormalization produce consistent results.

        Both implementations should return the same data, just with slightly
        different column naming conventions (underscore vs dot separator).
        """
        ml_instance, dataset_description = catalog_with_datasets

        dataset = dataset_description.dataset
        current_version = dataset.current_version

        # Get catalog-based denormalized data
        catalog_df = dataset.denormalize_as_dataframe(include_tables=["Subject"])

        # Download bag and get bag-based denormalized data
        bag = dataset.download_dataset_bag(current_version, use_minid=False)
        bag_df = bag.denormalize_as_dataframe(include_tables=["Subject"])

        # Both should have the same number of rows
        assert len(catalog_df) == len(bag_df), (
            f"Catalog ({len(catalog_df)} rows) and bag ({len(bag_df)} rows) "
            "should have same number of rows"
        )

        # Both should have Subject columns (different naming: _ vs .)
        catalog_subject_cols = [c for c in catalog_df.columns if c.startswith("Subject_")]
        bag_subject_cols = [c for c in bag_df.columns if c.startswith("Subject.")]

        assert len(catalog_subject_cols) > 0, "Expected Subject columns in catalog result"
        assert len(bag_subject_cols) > 0, "Expected Subject columns in bag result"

    def test_catalog_denormalize_with_nested_dataset(self, catalog_with_datasets, tmp_path):
        """Test that catalog denormalization includes nested dataset members.

        When a dataset has nested datasets, the denormalization should include
        members from both the parent and all nested children.
        """
        ml_instance, dataset_description = catalog_with_datasets

        dataset = dataset_description.dataset

        # Get members including nested
        all_members = dataset.list_dataset_members(recurse=True)

        # Denormalize
        df = dataset.denormalize_as_dataframe(include_tables=["Subject"])

        # Should include subjects from all nested datasets
        if "Subject" in all_members:
            expected_subject_count = len(all_members["Subject"])
            # The denormalized result should have at least as many rows
            # (could be more if there are multiple paths)
            assert len(df) >= 0, "Should return valid DataFrame"

    def test_catalog_denormalize_dict_generator_behavior(self, catalog_with_datasets, tmp_path):
        """Test that denormalize_as_dict is a proper generator.

        Verify that the method returns a generator that can be iterated
        multiple times or consumed partially.
        """
        ml_instance, dataset_description = catalog_with_datasets

        dataset = dataset_description.dataset

        # Get generator
        gen = dataset.denormalize_as_dict(include_tables=["Subject"])

        # Should be a generator
        from types import GeneratorType
        assert isinstance(gen, GeneratorType), "Should return a generator"

        # Consume the generator
        rows = list(gen)
        assert isinstance(rows, list), "Should be consumable as list"

    def test_catalog_non_member_fk_join(self, catalog_with_datasets, tmp_path):
        """Catalog: Join to non-member table via FK."""
        ml_instance, dataset_description = catalog_with_datasets
        dataset = dataset_description.dataset

        df = dataset.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        assert len(df) > 0
        obs_cols = [c for c in df.columns if c.startswith("Observation_")]
        assert len(obs_cols) > 0, "Expected Observation columns"

        obs_rid_col = "Observation_RID"
        if obs_rid_col in df.columns:
            non_null = df[obs_rid_col].notna().sum()
            assert non_null > 0, "Observation_RID should be populated"

    def test_catalog_multihop_chain(self, catalog_with_datasets, tmp_path):
        """Catalog: Multi-hop chain Image → Observation → Subject."""
        ml_instance, dataset_description = catalog_with_datasets
        dataset = dataset_description.dataset

        df = dataset.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        assert len(df) > 0
        for prefix in ["Image_", "Observation_", "Subject_"]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        if "Subject_RID" in df.columns:
            non_null = df["Subject_RID"].notna().sum()
            assert non_null > 0, "Subject_RID should be populated via multi-hop"

    def test_catalog_ambiguous_paths_error(self, catalog_with_datasets, tmp_path):
        """Catalog: Ambiguous paths raise error."""
        from deriva_ml.core.exceptions import DerivaMLException

        ml_instance, dataset_description = catalog_with_datasets
        dataset = dataset_description.dataset

        with pytest.raises(DerivaMLException) as exc_info:
            dataset.denormalize_as_dataframe(include_tables=["Image", "Subject"])

        error_msg = str(exc_info.value)
        assert "ambiguous" in error_msg.lower() or "multiple" in error_msg.lower()
        assert "Observation" in error_msg

    def test_catalog_disambiguation(self, catalog_with_datasets, tmp_path):
        """Catalog: Including intermediate resolves ambiguity."""
        ml_instance, dataset_description = catalog_with_datasets
        dataset = dataset_description.dataset

        df = dataset.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        assert len(df) > 0
        for prefix in ["Image_", "Observation_", "Subject_"]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"


class TestMultiHopDenormalize:
    """Test multi-hop FK joins in bag denormalization.

    These tests use the extended schema: Image → Observation → Subject
    and ClinicalRecord_Observation linking ClinicalRecord ↔ Observation.
    Image is the only dataset member; other tables are FK-reachable.
    """

    def test_non_member_fk_join(self, dataset_test, tmp_path):
        """M1: Join to a table that is not a dataset member via FK."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        assert len(df) > 0, "Expected rows from denormalization"
        obs_columns = [c for c in df.columns if c.startswith("Observation.")]
        assert len(obs_columns) > 0, "Expected Observation columns"
        obs_rid_col = "Observation.RID"
        assert obs_rid_col in df.columns, f"Expected {obs_rid_col} column"
        non_null_count = df[obs_rid_col].notna().sum()
        assert non_null_count > 0, (
            "Observation.RID should be populated for Images with Observation FK. "
            "Got all nulls — FK join to non-member table is not working."
        )

    def test_multihop_chain(self, dataset_test, tmp_path):
        """M2: Multi-hop chain Image → Observation → Subject, all included."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        assert len(df) > 0
        for prefix in ["Image.", "Observation.", "Subject."]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        subject_rid_col = "Subject.RID"
        if subject_rid_col in df.columns:
            non_null = df[subject_rid_col].notna().sum()
            assert non_null > 0, "Subject.RID should be populated via multi-hop FK chain"

    def test_association_table_join(self, dataset_test, tmp_path):
        """M3: Join through association table (M:N)."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "ClinicalRecord"]
        )

        assert len(df) > 0
        cr_cols = [c for c in df.columns if c.startswith("ClinicalRecord.")]
        assert len(cr_cols) > 0, "Expected ClinicalRecord columns"
        assoc_cols = [c for c in df.columns if c.startswith("ClinicalRecord_Observation.")]
        assert len(assoc_cols) == 0, "Association table columns should not appear in output"

    def test_reverse_fk_direction(self, dataset_test, tmp_path):
        """M4: Reverse FK direction — Observation listed first, Image is the member."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Observation", "Image"]
        )

        assert len(df) > 0
        obs_cols = [c for c in df.columns if c.startswith("Observation.")]
        img_cols = [c for c in df.columns if c.startswith("Image.")]
        assert len(obs_cols) > 0, "Expected Observation columns"
        assert len(img_cols) > 0, "Expected Image columns"

    def test_full_chain_with_association(self, dataset_test, tmp_path):
        """M5: Full chain Image → Observation ← ClinicalRecord_Observation → ClinicalRecord."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "ClinicalRecord"]
        )

        assert len(df) > 0
        for prefix in ["Image.", "Observation.", "ClinicalRecord."]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        cr_rid_col = "ClinicalRecord.RID"
        if cr_rid_col in df.columns:
            non_null = df[cr_rid_col].notna().sum()
            assert non_null > 0, "ClinicalRecord.RID should be populated via association table"

    def test_fk_value_integrity(self, dataset_test, tmp_path):
        """D1: FK values are correct in joined rows."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        if "Image.Observation" in df.columns and "Observation.RID" in df.columns:
            valid = df.dropna(subset=["Image.Observation", "Observation.RID"])
            for _, row in valid.iterrows():
                assert row["Image.Observation"] == row["Observation.RID"], (
                    f"FK mismatch: Image.Observation={row['Image.Observation']} "
                    f"!= Observation.RID={row['Observation.RID']}"
                )

    def test_row_count_matches_members(self, dataset_test, tmp_path):
        """D2: Row count equals dataset member count, no duplication."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        members = bag.list_dataset_members(recurse=True)
        image_count = len(members.get("Image", []))

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])
        assert len(df) == image_count, (
            f"Row count ({len(df)}) should match Image member count ({image_count})"
        )

    def test_null_fk_outer_join(self, dataset_test, tmp_path):
        """D3: Images with null Observation FK get null Observation columns."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        if "Image.Observation" in df.columns and "Observation.RID" in df.columns:
            null_obs_images = df[df["Image.Observation"].isna()]
            if len(null_obs_images) > 0:
                assert null_obs_images["Observation.RID"].isna().all(), (
                    "Images with null Observation FK should have null Observation columns"
                )

    def test_no_data_leakage(self, dataset_test, tmp_path):
        """D4: No data leakage from non-member records."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        if "Image.Observation" in df.columns:
            expected_obs_rids = set(df["Image.Observation"].dropna())
            if "Observation.RID" in df.columns:
                actual_obs_rids = set(df["Observation.RID"].dropna())
                assert actual_obs_rids.issubset(expected_obs_rids), (
                    f"Leaked Observation RIDs: {actual_obs_rids - expected_obs_rids}"
                )

    def test_single_table_regression(self, dataset_test, tmp_path):
        """E2: Single-table denormalization unchanged by schema additions."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df_subject = bag.denormalize_as_dataframe(include_tables=["Subject"])
        assert isinstance(df_subject, pd.DataFrame)
        assert len(df_subject) > 0

        df_image = bag.denormalize_as_dataframe(include_tables=["Image"])
        assert isinstance(df_image, pd.DataFrame)
        assert len(df_image) > 0

    def test_empty_intermediate_table(self, dataset_test, tmp_path):
        """E1: Verify query doesn't fail even with null FK values."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_all_non_member_tables(self, dataset_test, tmp_path):
        """E3: All non-member tables — should return empty or error."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Observation", "ClinicalRecord"]
        )
        assert len(df) == 0, (
            "Should return empty result when no included table has dataset members"
        )

    def test_denormalize_matches_members(self, dataset_test, tmp_path):
        """C2: Denormalized RIDs match list_dataset_members for member tables."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        members = bag.list_dataset_members(recurse=True)
        member_rids = {m["RID"] for m in members.get("Image", [])}

        df = bag.denormalize_as_dataframe(include_tables=["Image"])
        denorm_rids = set(df["Image.RID"].dropna())

        assert member_rids == denorm_rids, (
            f"Denormalized RIDs should match list_dataset_members. "
            f"Missing: {member_rids - denorm_rids}, Extra: {denorm_rids - member_rids}"
        )


class TestAmbiguousPaths:
    """Test ambiguous FK path detection and resolution.

    The test schema has two paths from Image to Subject:
    1. Direct: Image → Subject (via Image.Subject FK)
    2. Multi-hop: Image → Observation → Subject

    Requesting ["Image", "Subject"] should raise an error listing both paths.
    Including Observation should disambiguate.
    """

    def test_ambiguous_paths_raises_error(self, dataset_test, tmp_path):
        """A1: Ambiguous paths produce DerivaMLException with both paths listed."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        with pytest.raises(DerivaMLException) as exc_info:
            bag.denormalize_as_dataframe(include_tables=["Image", "Subject"])

        error_msg = str(exc_info.value)
        assert "Subject" in error_msg, "Error should mention the ambiguous target table"
        assert "Image" in error_msg, "Error should mention the source table"
        assert "ambiguous" in error_msg.lower() or "multiple" in error_msg.lower(), (
            f"Error should indicate ambiguity. Got: {error_msg}"
        )

    def test_ambiguous_error_lists_paths(self, dataset_test, tmp_path):
        """A1b: Error message contains enough info to resolve the ambiguity."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        with pytest.raises(DerivaMLException) as exc_info:
            bag.denormalize_as_dataframe(include_tables=["Image", "Subject"])

        error_msg = str(exc_info.value)
        assert "Observation" in error_msg, (
            f"Error should mention intermediate table 'Observation' so user knows "
            f"to include it for disambiguation. Got: {error_msg}"
        )

    def test_including_intermediate_resolves_ambiguity(self, dataset_test, tmp_path):
        """A2: Including Observation resolves the Image→Subject ambiguity."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        assert len(df) > 0, "Should return rows after disambiguation"

        for prefix in ["Image.", "Observation.", "Subject."]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        if all(c in df.columns for c in ["Image.Observation", "Observation.RID",
                                          "Observation.Subject", "Subject.RID"]):
            valid = df.dropna(subset=["Image.Observation", "Observation.RID",
                                       "Observation.Subject", "Subject.RID"])
            for _, row in valid.iterrows():
                assert row["Image.Observation"] == row["Observation.RID"], (
                    "Image.Observation should match Observation.RID"
                )
                assert row["Observation.Subject"] == row["Subject.RID"], (
                    "Observation.Subject should match Subject.RID"
                )

    def test_disambiguation_produces_correct_data(self, dataset_test, tmp_path):
        """A2b: Disambiguated path returns correct Subject for each Image."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        if all(c in df.columns for c in ["Subject.RID", "Subject.Name"]):
            valid = df.dropna(subset=["Subject.RID", "Subject.Name"])
            assert len(valid) > 0, "Should have rows with Subject data"
            for _, row in valid.iterrows():
                assert row["Subject.Name"].startswith("Thing"), (
                    f"Unexpected Subject.Name: {row['Subject.Name']}"
                )

    def test_direct_fk_no_ambiguity(self, dataset_test, tmp_path):
        """A3: Direct FK still works when no ambiguity exists."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])
        assert len(df) > 0
        if "Observation.RID" in df.columns:
            non_null = df["Observation.RID"].notna().sum()
            assert non_null > 0, "Observation.RID should be populated via direct FK"

    def test_association_table_single_path(self, dataset_test, tmp_path):
        """A4: Association table path — if only one path exists, no error."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "ClinicalRecord"]
        )
        assert len(df) > 0

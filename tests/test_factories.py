"""Tests for the testing factory functions.

These tests verify that the factory functions in tests/factories.py work correctly.
"""


from deriva_ml import DerivaML, MLVocab
from deriva_ml.dataset.aux_classes import DatasetVersion
from tests.factories import (
    assert_dataset_has_members,
    assert_dataset_version,
    make_dataset,
    make_dataset_spec,
    make_execution,
    make_execution_config,
    make_nested_datasets,
    make_table_rows,
    make_test_file,
    make_test_files,
    make_vocabulary_term,
    make_workflow,
)


class TestDatasetFactories:
    """Test dataset-related factory functions."""

    def test_make_dataset_default(self, deriva_catalog, tmp_path):
        """Test creating a dataset with default values."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        dataset = make_dataset(ml)

        assert dataset is not None
        assert dataset.description == "Test Dataset"
        assert dataset.dataset_types == ["Testing"]

    def test_make_dataset_custom(self, deriva_catalog, tmp_path):
        """Test creating a dataset with custom values."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        dataset = make_dataset(
            ml,
            description="Custom Dataset",
            dataset_types=["Training", "Validation"],
        )

        assert dataset.description == "Custom Dataset"
        assert set(dataset.dataset_types) == {"Training", "Validation"}

    def test_make_dataset_spec(self, deriva_catalog, tmp_path):
        """Test creating a DatasetSpec."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        dataset = make_dataset(ml)
        spec = make_dataset_spec(dataset.dataset_rid, version="1.0.0")

        assert spec.rid == dataset.dataset_rid
        assert spec.version == DatasetVersion(1, 0, 0)
        assert spec.materialize is True

    def test_make_nested_datasets(self, deriva_catalog, tmp_path):
        """Test creating nested datasets."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        # depth=2 creates root + 2 children (no grandchildren)
        root = make_nested_datasets(ml, depth=2, children_per_level=2)

        assert root is not None
        children = root.list_dataset_children()
        assert len(children) == 2

        # With depth=2, children don't have their own children
        # So all_descendants == direct children
        all_descendants = root.list_dataset_children(recurse=True)
        assert len(all_descendants) == 2


class TestExecutionFactories:
    """Test execution-related factory functions."""

    def test_make_execution_config_default(self):
        """Test creating an ExecutionConfiguration with defaults."""
        config = make_execution_config()

        assert config.description == "Test Execution"
        assert config.datasets == []
        assert config.assets == []

    def test_make_execution_config_custom(self, deriva_catalog, tmp_path):
        """Test creating an ExecutionConfiguration with custom values."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        dataset = make_dataset(ml)
        spec = make_dataset_spec(dataset.dataset_rid)

        config = make_execution_config(
            description="Custom Execution",
            datasets=[spec],
        )

        assert config.description == "Custom Execution"
        assert len(config.datasets) == 1
        assert config.datasets[0].rid == dataset.dataset_rid

    def test_make_workflow(self, deriva_catalog, tmp_path):
        """Test creating a workflow."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        workflow = make_workflow(ml, name="Test Pipeline")

        assert workflow is not None
        assert workflow.name == "Test Pipeline"

    def test_make_execution(self, deriva_catalog, tmp_path):
        """Test creating an execution."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        execution = make_execution(ml)

        assert execution is not None
        # Use context manager to properly enter/exit
        with execution.execute() as exe:
            # Status may be Initializing or Running depending on timing
            assert exe.status.value in ("Initializing", "Running")


class TestVocabularyFactories:
    """Test vocabulary-related factory functions."""

    def test_make_vocabulary_term(self, deriva_catalog, tmp_path):
        """Test adding a vocabulary term."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        rid = make_vocabulary_term(
            ml,
            MLVocab.dataset_type,
            "CustomType",
            description="A custom dataset type",
        )

        assert rid is not None
        term = ml.lookup_term(MLVocab.dataset_type, "CustomType")
        assert term.name == "CustomType"

    def test_make_vocabulary_term_skip_existing(self, deriva_catalog, tmp_path):
        """Test that skip_if_exists works correctly."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        # Create term first time
        rid1 = make_vocabulary_term(ml, MLVocab.dataset_type, "UniqueType")

        # Create same term again - should not raise
        rid2 = make_vocabulary_term(ml, MLVocab.dataset_type, "UniqueType")

        assert rid1 == rid2


class TestTableDataFactories:
    """Test table data factory functions."""

    def test_make_table_rows(self, deriva_catalog, tmp_path):
        """Test inserting table rows."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        rids = make_table_rows(
            ml,
            "Subject",
            [{"Name": "TestSubject1"}, {"Name": "TestSubject2"}],
        )

        assert len(rids) == 2
        assert all(rid is not None for rid in rids)


class TestFileFactories:
    """Test file-related factory functions."""

    def test_make_test_file(self, tmp_path):
        """Test creating a test file."""
        file_path = make_test_file(tmp_path)

        assert file_path.exists()
        assert file_path.name == "test_file.txt"
        assert file_path.read_text() == "Test content"

    def test_make_test_file_custom(self, tmp_path):
        """Test creating a test file with custom values."""
        file_path = make_test_file(
            tmp_path,
            filename="data.csv",
            content="col1,col2\n1,2\n3,4",
        )

        assert file_path.name == "data.csv"
        assert "col1,col2" in file_path.read_text()

    def test_make_test_files(self, tmp_path):
        """Test creating multiple test files."""
        files = make_test_files(tmp_path, count=5)

        assert len(files) == 5
        assert all(f.exists() for f in files)
        assert all(f.suffix == ".txt" for f in files)


class TestAssertionHelpers:
    """Test assertion helper functions."""

    def test_assert_dataset_version(self, deriva_catalog, tmp_path):
        """Test version assertion helper."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        dataset = make_dataset(ml)

        # Should not raise
        assert_dataset_version(dataset, dataset.current_version)

        # Test string version
        version_str = str(dataset.current_version)
        assert_dataset_version(dataset, version_str)

    def test_assert_dataset_has_members_empty(self, deriva_catalog, tmp_path):
        """Test member count assertion with empty dataset."""
        ml = DerivaML(
            deriva_catalog.hostname,
            deriva_catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        dataset = make_dataset(ml)

        # Empty dataset should have no members
        assert_dataset_has_members(dataset, {"Subject": 0})

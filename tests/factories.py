"""Testing factory functions for DerivaML.

This module provides factory functions for creating test objects with sensible
defaults. These factories simplify test setup by providing easy-to-use functions
that create properly configured objects for testing.

Factory Functions:
    make_dataset: Create a Dataset with customizable attributes
    make_execution: Create an Execution for testing workflows
    make_execution_config: Create an ExecutionConfiguration
    make_dataset_spec: Create a DatasetSpec for version testing
    make_vocabulary_term: Add a vocabulary term for testing

Mock Helpers:
    MockCatalogContext: Context manager for mocking catalog operations

Example:
    >>> from tests.factories import make_dataset, make_execution_config
    >>>
    >>> # Create a simple dataset
    >>> dataset = make_dataset(ml, description="Test dataset")
    >>>
    >>> # Create a dataset with specific types
    >>> dataset = make_dataset(ml, dataset_types=["Training", "Test"])
    >>>
    >>> # Create an execution configuration
    >>> config = make_execution_config(description="My test run")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from deriva_ml.core.definitions import MLVocab, RID
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.execution.execution_configuration import ExecutionConfiguration

if TYPE_CHECKING:
    from deriva_ml import DerivaML
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.workflow import Workflow


# =============================================================================
# Dataset Factories
# =============================================================================


def make_dataset(
    ml: DerivaML,
    description: str = "Test Dataset",
    dataset_types: list[str] | None = None,
    execution_rid: RID | None = None,
    version: DatasetVersion | str | None = None,
    ensure_types: bool = True,
) -> Dataset:
    """Create a Dataset for testing.

    This factory creates a dataset with sensible defaults, automatically
    ensuring that any required vocabulary terms exist.

    Args:
        ml: DerivaML instance to create the dataset in.
        description: Dataset description. Defaults to "Test Dataset".
        dataset_types: List of dataset type terms. Defaults to ["Testing"].
        execution_rid: Optional execution to associate with the dataset.
        version: Initial version. Defaults to None (uses catalog default).
        ensure_types: If True, automatically add missing dataset type terms
                     to the vocabulary. Defaults to True.

    Returns:
        The created Dataset object.

    Example:
        >>> dataset = make_dataset(ml)
        >>> dataset = make_dataset(ml, description="Training data", dataset_types=["Training"])
    """
    if dataset_types is None:
        dataset_types = ["Testing"]

    # Ensure vocabulary terms exist
    if ensure_types:
        for dtype in dataset_types:
            try:
                ml.lookup_term(MLVocab.dataset_type, dtype)
            except Exception:
                ml.add_term(
                    MLVocab.dataset_type,
                    dtype,
                    description=f"Test dataset type: {dtype}",
                )

    return ml.create_dataset(
        description=description,
        dataset_types=dataset_types,
        execution_rid=execution_rid,
        version=version,
    )


def make_dataset_spec(
    rid: RID,
    version: DatasetVersion | str = "0.1.0",
    materialize: bool = True,
) -> DatasetSpec:
    """Create a DatasetSpec for testing version-specific operations.

    Args:
        rid: The dataset RID.
        version: The version to specify. If string, parsed as semver.
                Defaults to "0.1.0".
        materialize: Whether to materialize the dataset. Defaults to True.

    Returns:
        A DatasetSpec configured for testing.

    Example:
        >>> spec = make_dataset_spec("1-ABC", version="1.0.0")
        >>> spec = make_dataset_spec(dataset.dataset_rid, version=dataset.current_version)
    """
    if isinstance(version, str):
        version = DatasetVersion.parse(version)

    return DatasetSpec(
        rid=rid,
        version=version,
        materialize=materialize,
    )


def make_nested_datasets(
    ml: DerivaML,
    depth: int = 2,
    children_per_level: int = 2,
    base_description: str = "Nested Dataset",
) -> Dataset:
    """Create a hierarchy of nested datasets for testing.

    Creates a parent dataset with nested child datasets to the specified depth.
    Useful for testing recursive dataset operations.

    Args:
        ml: DerivaML instance.
        depth: How many levels of nesting. Defaults to 2.
        children_per_level: Number of children at each level. Defaults to 2.
        base_description: Base description for datasets. Defaults to "Nested Dataset".

    Returns:
        The root (parent) Dataset containing the hierarchy.

    Example:
        >>> root = make_nested_datasets(ml, depth=3, children_per_level=2)
        >>> children = root.list_dataset_children(recurse=True)
    """

    def create_level(level: int, parent_desc: str) -> Dataset:
        ds = make_dataset(ml, description=f"{parent_desc} L{level}")
        if level < depth:
            child_rids = []
            for i in range(children_per_level):
                child = create_level(level + 1, f"{parent_desc} L{level} C{i + 1}")
                child_rids.append(child.dataset_rid)
            if child_rids:
                ds.add_dataset_members(child_rids)
        return ds

    return create_level(1, base_description)


# =============================================================================
# Execution Factories
# =============================================================================


def make_execution_config(
    description: str = "Test Execution",
    datasets: list[DatasetSpec] | None = None,
    assets: list[RID] | None = None,
    workflow: RID | None = None,
    **extra_params: Any,
) -> ExecutionConfiguration:
    """Create an ExecutionConfiguration for testing.

    Args:
        description: Execution description. Defaults to "Test Execution".
        datasets: List of DatasetSpecs to include. Defaults to empty list.
        assets: List of asset RIDs to include. Defaults to empty list.
        workflow: Workflow RID. Defaults to None.
        **extra_params: Additional parameters to pass to ExecutionConfiguration.

    Returns:
        An ExecutionConfiguration ready for use.

    Example:
        >>> config = make_execution_config()
        >>> config = make_execution_config(
        ...     description="Training run",
        ...     datasets=[make_dataset_spec("1-ABC")],
        ... )
    """
    return ExecutionConfiguration(
        description=description,
        datasets=datasets or [],
        assets=assets or [],
        workflow=workflow,
        **extra_params,
    )


def make_workflow(
    ml: DerivaML,
    name: str = "Test Workflow",
    workflow_type: str = "Testing",
    ensure_type: bool = True,
) -> Workflow:
    """Create a Workflow for testing.

    Args:
        ml: DerivaML instance.
        name: Workflow name. Defaults to "Test Workflow".
        workflow_type: Workflow type term. Defaults to "Testing".
        ensure_type: If True, automatically add the workflow type term
                    if it doesn't exist. Defaults to True.

    Returns:
        The created Workflow object.

    Example:
        >>> workflow = make_workflow(ml)
        >>> workflow = make_workflow(ml, name="Training Pipeline", workflow_type="ML")
    """
    # Ensure workflow type exists
    if ensure_type:
        try:
            ml.lookup_term(MLVocab.workflow_type, workflow_type)
        except Exception:
            ml.add_term(
                MLVocab.workflow_type,
                workflow_type,
                description=f"Test workflow type: {workflow_type}",
            )

    return ml.create_workflow(
        name=name,
        workflow_type=workflow_type,
    )


def make_execution(
    ml: DerivaML,
    config: ExecutionConfiguration | None = None,
    workflow: Workflow | RID | None = None,
    dry_run: bool = False,
    auto_workflow: bool = True,
) -> Execution:
    """Create an Execution for testing.

    This factory creates an execution context that can be used for testing
    execution workflows. If no workflow is provided and auto_workflow is True,
    a test workflow is automatically created.

    Args:
        ml: DerivaML instance.
        config: ExecutionConfiguration. Defaults to a basic test config.
        workflow: Workflow or RID to use. If None and auto_workflow is True,
                 creates a test workflow automatically.
        dry_run: If True, don't persist changes to catalog. Defaults to False.
        auto_workflow: If True and no workflow provided, create one automatically.
                      Defaults to True.

    Returns:
        An Execution ready for testing (not yet entered as context manager).

    Example:
        >>> execution = make_execution(ml)
        >>> with execution.execute() as exe:
        ...     # perform test operations
        ...     pass
    """
    if config is None:
        config = make_execution_config()

    if workflow is None and auto_workflow:
        workflow = make_workflow(ml)

    return ml.create_execution(
        configuration=config,
        workflow=workflow,
        dry_run=dry_run,
    )


# =============================================================================
# Vocabulary Helpers
# =============================================================================


def make_vocabulary_term(
    ml: DerivaML,
    vocabulary: str | MLVocab,
    term_name: str,
    description: str | None = None,
    synonyms: list[str] | None = None,
    skip_if_exists: bool = True,
) -> RID:
    """Add a vocabulary term for testing.

    Args:
        ml: DerivaML instance.
        vocabulary: Vocabulary table name or MLVocab enum.
        term_name: Name of the term to add.
        description: Term description. Defaults to auto-generated.
        synonyms: Optional list of synonyms. Defaults to None.
        skip_if_exists: If True, don't raise error if term exists.
                       Defaults to True.

    Returns:
        The RID of the created (or existing) term.

    Example:
        >>> rid = make_vocabulary_term(ml, MLVocab.dataset_type, "Training")
        >>> rid = make_vocabulary_term(
        ...     ml, "Asset_Type", "Model",
        ...     description="ML model file",
        ...     synonyms=["model", "weights"],
        ... )
    """
    vocab_name = vocabulary.value if isinstance(vocabulary, MLVocab) else vocabulary

    if skip_if_exists:
        try:
            term = ml.lookup_term(vocab_name, term_name)
            return term.RID
        except Exception:
            pass

    if description is None:
        description = f"Test term: {term_name}"

    return ml.add_term(
        vocab_name,
        term_name,
        description=description,
        synonyms=synonyms,
    )


# =============================================================================
# Table Data Factories
# =============================================================================


def make_table_rows(
    ml: DerivaML,
    table_name: str,
    rows: list[dict[str, Any]],
    schema: str | None = None,
) -> list[RID]:
    """Insert test rows into a table.

    Args:
        ml: DerivaML instance.
        table_name: Name of the table to insert into.
        rows: List of row dictionaries to insert.
        schema: Schema name. Defaults to domain_schema.

    Returns:
        List of RIDs for the inserted rows.

    Example:
        >>> rids = make_table_rows(ml, "Subject", [
        ...     {"Name": "Subject1"},
        ...     {"Name": "Subject2"},
        ... ])
    """
    if schema is None:
        schema = ml.domain_schema

    pb = ml.pathBuilder()
    table = pb.schemas[schema].tables[table_name]
    results = table.insert(rows)
    return [r["RID"] for r in results]


# =============================================================================
# Test File Helpers
# =============================================================================


def make_test_file(
    directory: Path,
    filename: str = "test_file.txt",
    content: str = "Test content",
) -> Path:
    """Create a test file for asset upload testing.

    Args:
        directory: Directory to create the file in.
        filename: Name of the file. Defaults to "test_file.txt".
        content: File content. Defaults to "Test content".

    Returns:
        Path to the created file.

    Example:
        >>> test_file = make_test_file(tmp_path)
        >>> test_file = make_test_file(tmp_path, "data.csv", "col1,col2\\n1,2")
    """
    file_path = directory / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def make_test_files(
    directory: Path,
    count: int = 3,
    prefix: str = "test",
    extension: str = ".txt",
) -> list[Path]:
    """Create multiple test files for batch testing.

    Args:
        directory: Directory to create files in.
        count: Number of files to create. Defaults to 3.
        prefix: Filename prefix. Defaults to "test".
        extension: File extension. Defaults to ".txt".

    Returns:
        List of paths to created files.

    Example:
        >>> files = make_test_files(tmp_path, count=5)
        >>> files = make_test_files(tmp_path, prefix="image", extension=".png")
    """
    files = []
    for i in range(count):
        filename = f"{prefix}_{i + 1}{extension}"
        content = f"Test content for file {i + 1}"
        files.append(make_test_file(directory, filename, content))
    return files


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_dataset_has_members(
    dataset: Dataset,
    expected_types: dict[str, int],
    recurse: bool = False,
) -> None:
    """Assert that a dataset has expected member counts.

    Args:
        dataset: The dataset to check.
        expected_types: Dict mapping member type names to expected counts.
        recurse: Whether to check recursively. Defaults to False.

    Raises:
        AssertionError: If member counts don't match expectations.

    Example:
        >>> assert_dataset_has_members(dataset, {"Subject": 5, "Image": 10})
    """
    members = dataset.list_dataset_members(recurse=recurse)
    for member_type, expected_count in expected_types.items():
        actual_count = len(members.get(member_type, []))
        assert actual_count == expected_count, (
            f"Expected {expected_count} {member_type} members, got {actual_count}"
        )


def assert_dataset_version(
    dataset: Dataset,
    expected_version: DatasetVersion | str,
) -> None:
    """Assert that a dataset has the expected version.

    Args:
        dataset: The dataset to check.
        expected_version: Expected version (DatasetVersion or string).

    Raises:
        AssertionError: If version doesn't match.

    Example:
        >>> assert_dataset_version(dataset, "1.0.0")
        >>> assert_dataset_version(dataset, DatasetVersion(1, 2, 0))
    """
    if isinstance(expected_version, str):
        expected_version = DatasetVersion.parse(expected_version)

    assert dataset.current_version == expected_version, (
        f"Expected version {expected_version}, got {dataset.current_version}"
    )

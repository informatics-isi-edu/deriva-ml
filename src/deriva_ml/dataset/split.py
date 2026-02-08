"""Generic dataset splitting for DerivaML.

This module provides functions to split a DerivaML dataset into training
and testing subsets with full provenance tracking. It works with any
DerivaML catalog and any registered element type.

The splitting API follows scikit-learn conventions (``test_size``,
``train_size``, ``shuffle``, ``seed``, ``stratify``) while integrating
with DerivaML's dataset hierarchy, execution provenance, and versioning.

Splitting Strategies:
    Random (default):
        Shuffles members and splits at the train_size boundary.
        No denormalization required.

    Stratified:
        Maintains class distribution across splits using scikit-learn's
        stratified splitting. Requires specifying a column to stratify by
        from the denormalized DataFrame.

    Custom:
        Users can provide a ``SelectionFunction`` callable for arbitrary
        selection logic (balanced labels, filtered subsets, etc.).

Example:
    Simple random 80/20 split::

        from deriva_ml import DerivaML
        from deriva_ml.dataset.split import split_dataset

        ml = DerivaML("localhost", "9")
        result = split_dataset(ml, "28D0", test_size=0.2, seed=42)

    Stratified split::

        result = split_dataset(
            ml, "28D0",
            test_size=0.2,
            stratify_by_column="Image_Classification_Image_Class",
            include_tables=["Image", "Image_Classification"],
        )

    Custom selection function::

        def my_selector(df, train_size, test_size, seed):
            # Custom logic...
            return train_indices, test_indices

        result = split_dataset(
            ml, "28D0",
            test_size=100,
            selection_fn=my_selector,
            include_tables=["Image", "Image_Classification"],
        )

See Also:
    - ``sklearn.model_selection.train_test_split``
    - ``Dataset.denormalize_as_dataframe``
    - ``Dataset.list_dataset_members``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from deriva_ml import DerivaML

from deriva_ml.execution import ExecutionConfiguration

logger = logging.getLogger("deriva_ml")


# =============================================================================
# Selection Function Protocol and Built-in Implementations
# =============================================================================


@runtime_checkable
class SelectionFunction(Protocol):
    """Protocol for custom train/test selection functions.

    A selection function receives the denormalized dataset DataFrame and
    returns a tuple of ``(train_indices, test_indices)`` as integer arrays
    indexing into the DataFrame rows.

    The function is responsible for:

    - Deciding which records go into training vs testing
    - Ensuring the sizes match the requested train_size/test_size
    - Implementing any balancing or stratification logic

    Args:
        df: Denormalized DataFrame from ``dataset.denormalize_as_dataframe()``.
            Columns are prefixed with table names (e.g., ``Image_RID``,
            ``Image_Classification_Image_Class``).
        train_size: Number of records for the training set.
        test_size: Number of records for the testing set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_indices, test_indices)`` as numpy arrays of
        integer indices into the DataFrame.

    Example:
        >>> def balanced_selector(df, train_size, test_size, seed):
        ...     rng = np.random.default_rng(seed)
        ...     # ... balance classes ...
        ...     return train_indices, test_indices
    """

    def __call__(
        self,
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def random_split(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split.

    Shuffles the DataFrame indices and splits at the train_size boundary.

    Args:
        df: Source DataFrame.
        train_size: Number of training records.
        test_size: Number of testing records.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_indices, test_indices)``.
    """
    rng = np.random.default_rng(seed)
    total_needed = train_size + test_size
    indices = np.arange(len(df))
    rng.shuffle(indices)
    indices = indices[:total_needed]
    return indices[:train_size], indices[train_size : train_size + test_size]


def stratified_split(stratify_column: str) -> SelectionFunction:
    """Create a stratified selection function.

    Returns a selection function that maintains the class distribution
    of the specified column across train and test sets. Delegates to
    scikit-learn's ``train_test_split`` for the actual stratification.

    Args:
        stratify_column: Column name in the denormalized DataFrame to
            stratify by (e.g., ``Image_Classification_Image_Class``).

    Returns:
        A ``SelectionFunction`` that performs stratified splitting.

    Example:
        >>> selector = stratified_split("Image_Classification_Image_Class")
        >>> train_idx, test_idx = selector(df, 400, 100, seed=42)
    """

    def _stratified_split(
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split as sklearn_split

        total_needed = train_size + test_size
        if total_needed > len(df):
            raise ValueError(
                f"Requested {total_needed} samples but dataset has {len(df)} records"
            )

        if stratify_column not in df.columns:
            available = [c for c in df.columns if not c.startswith("_")]
            raise ValueError(
                f"Column '{stratify_column}' not found in denormalized DataFrame. "
                f"Available columns: {available}"
            )

        indices = np.arange(len(df))

        # If we need a subset of the data, first do a stratified sample
        if total_needed < len(df):
            _, subset_indices = sklearn_split(
                indices,
                test_size=total_needed,
                stratify=df[stratify_column].values,
                random_state=seed,
            )
            sub_df = df.iloc[subset_indices]
        else:
            subset_indices = indices
            sub_df = df

        # Split the subset into train/test with stratification
        test_fraction = test_size / total_needed
        train_idx, test_idx = sklearn_split(
            np.arange(len(sub_df)),
            test_size=test_fraction,
            stratify=sub_df[stratify_column].values,
            random_state=seed,
        )

        return subset_indices[train_idx], subset_indices[test_idx]

    return _stratified_split


# =============================================================================
# Size Resolution Helper
# =============================================================================


def _resolve_sizes(
    total: int,
    test_size: float | int,
    train_size: float | int | None,
) -> tuple[int, int]:
    """Convert fractional or absolute sizes to absolute counts.

    Follows scikit-learn's convention: if both are fractions, they should
    sum to <= 1.0. If one is None, it's computed as the complement.

    Args:
        total: Total number of records in the dataset.
        test_size: Fraction (0-1) or absolute count of test samples.
        train_size: Fraction (0-1) or absolute count of train samples,
            or None for complement of test_size.

    Returns:
        Tuple of ``(train_count, test_count)`` as integers.

    Raises:
        ValueError: If sizes are invalid or exceed total.
    """
    # Convert test_size
    if isinstance(test_size, float) and 0 < test_size < 1:
        test_count = int(round(total * test_size))
    elif isinstance(test_size, (int, float)) and test_size >= 1:
        test_count = int(test_size)
    else:
        raise ValueError(
            f"test_size must be a float in (0, 1) or an int >= 1, got {test_size}"
        )

    # Convert train_size
    if train_size is None:
        train_count = total - test_count
    elif isinstance(train_size, float) and 0 < train_size < 1:
        train_count = int(round(total * train_size))
    elif isinstance(train_size, (int, float)) and train_size >= 1:
        train_count = int(train_size)
    else:
        raise ValueError(
            f"train_size must be a float in (0, 1), an int >= 1, or None, "
            f"got {train_size}"
        )

    if train_count + test_count > total:
        raise ValueError(
            f"Requested train_size={train_count} + test_size={test_count} = "
            f"{train_count + test_count} exceeds total dataset size of {total}"
        )

    if train_count <= 0 or test_count <= 0:
        raise ValueError(
            f"Both train and test must have at least 1 sample. "
            f"Got train_size={train_count}, test_size={test_count}"
        )

    return train_count, test_count


# =============================================================================
# Vocabulary Helpers
# =============================================================================


def _ensure_workflow_type(ml: DerivaML, workflow_type: str = "Dataset_Split") -> None:
    """Ensure the workflow type exists in the catalog.

    Args:
        ml: Connected DerivaML instance.
        workflow_type: Workflow type term name.
    """
    existing_types = {t.name for t in ml.list_vocabulary_terms("Workflow_Type")}
    if workflow_type not in existing_types:
        logger.info(f"Creating {workflow_type} workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name=workflow_type,
            description="Workflow for splitting datasets into training/testing subsets",
        )


def _ensure_dataset_types(ml: DerivaML) -> None:
    """Ensure required dataset types exist.

    Args:
        ml: Connected DerivaML instance.
    """
    required_types = {
        "Training": "A dataset subset used for model training",
        "Testing": "A dataset subset used for model testing/evaluation",
        "Split": "A dataset that contains nested dataset splits",
        "Labeled": "A dataset containing records with ground truth labels",
        "Unlabeled": "A dataset containing records without ground truth labels",
    }

    existing_terms = {t.name for t in ml.list_vocabulary_terms("Dataset_Type")}

    for type_name, description in required_types.items():
        if type_name not in existing_terms:
            try:
                ml.add_term(
                    table="Dataset_Type",
                    term_name=type_name,
                    description=description,
                )
                logger.info(f"  Added dataset type: {type_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not add dataset type {type_name}: {e}")


# =============================================================================
# Core Split Function
# =============================================================================


def split_dataset(
    ml: DerivaML,
    source_dataset_rid: str,
    *,
    # scikit-learn compatible parameters
    test_size: float | int = 0.2,
    train_size: float | int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    stratify_by_column: str | None = None,
    # DerivaML-specific parameters
    split_description: str = "",
    training_types: list[str] | None = None,
    testing_types: list[str] | None = None,
    element_table: str | None = None,
    include_tables: list[str] | None = None,
    selection_fn: SelectionFunction | None = None,
    workflow_type: str = "Dataset_Split",
    dry_run: bool = False,
) -> dict[str, str]:
    """Split a DerivaML dataset into training and testing subsets.

    Creates a new dataset hierarchy in the catalog::

        Split (parent, type: "Split")
        +-- Training (child, type: "Training", + training_types)
        +-- Testing (child, type: "Testing", + testing_types)

    All operations are performed within an execution context for
    full provenance tracking.

    This function is generic and works with any DerivaML dataset
    that has registered element types.

    Args:
        ml: Connected DerivaML instance.
        source_dataset_rid: RID of the source dataset to split.
        test_size: If float (0-1), fraction of data for testing.
            If int, absolute number of test samples. Default: 0.2.
        train_size: If float (0-1), fraction of data for training.
            If int, absolute number of training samples.
            If None, complement of test_size. Default: None.
        shuffle: Whether to shuffle before splitting. Default: True.
            Ignored when using stratified or custom selection functions
            (they handle their own shuffling).
        seed: Random seed for reproducibility. Default: 42.
        stratify_by_column: Column name for stratified splitting.
            Must be a column in the denormalized DataFrame (prefixed
            with table name, e.g., ``Image_Classification_Image_Class``).
            Mutually exclusive with ``selection_fn``.
        split_description: Description for the parent Split dataset.
        training_types: Additional dataset types for the training set
            beyond "Training" (e.g., ``["Labeled"]``). Default: None.
        testing_types: Additional dataset types for the testing set
            beyond "Testing" (e.g., ``["Labeled"]``). Default: None.
        element_table: Name of the element table to split (e.g., "Image").
            If None, auto-detected from the source dataset's members.
        include_tables: Tables to include when denormalizing for the
            selection function. Required when using ``stratify_by_column``
            or a custom ``selection_fn``.
        selection_fn: Custom selection function conforming to the
            ``SelectionFunction`` protocol. Mutually exclusive with
            ``stratify_by_column``.
        workflow_type: Workflow type vocabulary term. Default: "Dataset_Split".
        dry_run: If True, return what would happen without modifying catalog.

    Returns:
        Dictionary with keys:

        - ``split``: RID of the parent Split dataset
        - ``split_version``: Version string of the Split dataset
        - ``training``: RID of the Training dataset
        - ``training_version``: Version string of the Training dataset
        - ``testing``: RID of the Testing dataset
        - ``testing_version``: Version string of the Testing dataset
        - ``source``: RID of the source dataset
        - ``train_count``: Number of training samples
        - ``test_count``: Number of testing samples

    Raises:
        ValueError: If sizes are invalid, dataset has no members, or
            parameters conflict.

    Example:
        Simple random 80/20 split::

            from deriva_ml import DerivaML
            from deriva_ml.dataset.split import split_dataset

            ml = DerivaML("localhost", "9")
            result = split_dataset(ml, "28D0", test_size=0.2, seed=42)
            print(f"Training: {result['training']} ({result['train_count']} samples)")
            print(f"Testing:  {result['testing']} ({result['test_count']} samples)")

        Fixed-count split with labeled types::

            result = split_dataset(
                ml, "28D0",
                test_size=100,
                train_size=400,
                seed=42,
                training_types=["Labeled"],
                testing_types=["Labeled"],
            )

        Stratified split preserving class distribution::

            result = split_dataset(
                ml, "28D0",
                test_size=0.2,
                stratify_by_column="Image_Classification_Image_Class",
                include_tables=["Image", "Image_Classification"],
            )

        Custom selection function for balanced sampling::

            import numpy as np

            def balanced_selector(df, train_size, test_size, seed):
                rng = np.random.default_rng(seed)
                label_col = "Image_Classification_Image_Class"
                classes = df[label_col].unique()
                train_idx, test_idx = [], []
                for cls in classes:
                    cls_indices = df.index[df[label_col] == cls].to_numpy()
                    rng.shuffle(cls_indices)
                    per_class_train = train_size // len(classes)
                    per_class_test = test_size // len(classes)
                    train_idx.extend(cls_indices[:per_class_train])
                    test_idx.extend(cls_indices[per_class_train:per_class_train + per_class_test])
                return np.array(train_idx), np.array(test_idx)

            result = split_dataset(
                ml, "28D0",
                test_size=100,
                selection_fn=balanced_selector,
                include_tables=["Image", "Image_Classification"],
            )

        Dry run to preview the split plan without modifying the catalog::

            result = split_dataset(
                ml, "28D0",
                test_size=0.2,
                dry_run=True,
            )
            print(f"Would create: {result['train_count']} train, "
                  f"{result['test_count']} test")

        Use returned RIDs to create a hydra-zen configuration::

            from deriva_ml.dataset import DatasetSpecConfig

            result = split_dataset(ml, "28D0", test_size=0.2, seed=42)
            # Use the split dataset in a configuration
            split_config = DatasetSpecConfig(
                rid=result["split"],
                version=result["split_version"],
            )
    """
    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    if stratify_by_column and selection_fn:
        raise ValueError(
            "stratify_by_column and selection_fn are mutually exclusive. "
            "Use one or the other."
        )

    if stratify_by_column and not include_tables:
        raise ValueError(
            "include_tables is required when using stratify_by_column. "
            "Specify the tables needed for denormalization "
            "(e.g., include_tables=['Image', 'Image_Classification'])."
        )

    if selection_fn and not include_tables:
        raise ValueError(
            "include_tables is required when using a custom selection_fn. "
            "Specify the tables needed for denormalization."
        )

    # -------------------------------------------------------------------------
    # Look up source dataset and get members
    # -------------------------------------------------------------------------
    logger.info(f"Looking up source dataset: {source_dataset_rid}")
    source_ds = ml.lookup_dataset(source_dataset_rid)

    logger.info("Listing dataset members...")
    members = source_ds.list_dataset_members(recurse=True)

    # Auto-detect element table if not specified
    if element_table is None:
        candidate_tables = [
            table_name
            for table_name, records in members.items()
            if table_name != "Dataset" and len(records) > 0
        ]
        if not candidate_tables:
            raise ValueError(
                f"Source dataset {source_dataset_rid} has no members. "
                "Cannot split an empty dataset."
            )
        if len(candidate_tables) > 1:
            raise ValueError(
                f"Source dataset has members in multiple tables: {candidate_tables}. "
                "Specify element_table to choose which one to split."
            )
        element_table = candidate_tables[0]

    if element_table not in members or not members[element_table]:
        raise ValueError(
            f"Source dataset {source_dataset_rid} has no members in "
            f"table '{element_table}'. Available tables with members: "
            f"{[t for t, r in members.items() if r and t != 'Dataset']}"
        )

    member_records = members[element_table]
    total = len(member_records)
    logger.info(f"Found {total} members in table '{element_table}'")

    # -------------------------------------------------------------------------
    # Compute absolute sizes
    # -------------------------------------------------------------------------
    train_count, test_count = _resolve_sizes(total, test_size, train_size)
    logger.info(f"Split sizes: train={train_count}, test={test_count} (total={total})")

    # -------------------------------------------------------------------------
    # Determine selection strategy and get train/test RIDs
    # -------------------------------------------------------------------------
    use_denormalization = stratify_by_column is not None or selection_fn is not None

    if use_denormalization:
        logger.info(f"Denormalizing dataset with tables: {include_tables}")
        df = source_ds.denormalize_as_dataframe(include_tables)
        logger.info(
            f"Denormalized DataFrame: {len(df)} rows, {len(df.columns)} columns"
        )

        if stratify_by_column:
            logger.info(f"Using stratified split on column: {stratify_by_column}")
            selector = stratified_split(stratify_by_column)
        else:
            logger.info("Using custom selection function")
            selector = selection_fn

        train_indices, test_indices = selector(df, train_count, test_count, seed)

        # Map indices back to RIDs
        rid_column = f"{element_table}_RID"
        if rid_column not in df.columns:
            rid_column = "RID"
            if rid_column not in df.columns:
                raise ValueError(
                    f"Cannot find RID column. Tried '{element_table}_RID' and 'RID'. "
                    f"Available columns: {list(df.columns)}"
                )

        train_rids = df.iloc[train_indices][rid_column].tolist()
        test_rids = df.iloc[test_indices][rid_column].tolist()

    else:
        all_rids = [record["RID"] for record in member_records]

        if shuffle:
            rng = np.random.default_rng(seed)
            indices = np.arange(len(all_rids))
            rng.shuffle(indices)
            all_rids = [all_rids[i] for i in indices]

        train_rids = all_rids[:train_count]
        test_rids = all_rids[train_count : train_count + test_count]

    logger.info(
        f"Selected {len(train_rids)} training and {len(test_rids)} testing RIDs"
    )

    # -------------------------------------------------------------------------
    # Dry run
    # -------------------------------------------------------------------------
    if dry_run:
        strategy = (
            f"stratified by {stratify_by_column}" if stratify_by_column else "random"
        )
        if selection_fn:
            strategy = "custom selection function"
        return {
            "split": "(dry run)",
            "split_version": "(dry run)",
            "training": "(dry run)",
            "training_version": "(dry run)",
            "testing": "(dry run)",
            "testing_version": "(dry run)",
            "source": source_dataset_rid,
            "train_count": train_count,
            "test_count": test_count,
            "strategy": strategy,
            "element_table": element_table,
        }

    # -------------------------------------------------------------------------
    # Ensure vocabulary terms exist
    # -------------------------------------------------------------------------
    _ensure_workflow_type(ml, workflow_type)
    _ensure_dataset_types(ml)

    # -------------------------------------------------------------------------
    # Create execution and dataset hierarchy
    # -------------------------------------------------------------------------
    strategy_desc = (
        f"stratified by {stratify_by_column}" if stratify_by_column else "random"
    )
    if selection_fn:
        strategy_desc = "custom selection function"

    auto_description = (
        f"Train/test split of dataset {source_dataset_rid} "
        f"({strategy_desc}, train={train_count}, test={test_count}, seed={seed})"
    )

    logger.info("Creating workflow and execution...")
    workflow = ml.create_workflow(
        name=f"Dataset Split: {source_dataset_rid}",
        workflow_type=workflow_type,
        description="Split dataset into training and testing subsets",
    )

    config = ExecutionConfiguration(
        workflow=workflow,
        description=split_description or auto_description,
    )

    train_types = ["Training"] + (training_types or [])
    test_types = ["Testing"] + (testing_types or [])

    with ml.create_execution(config) as exe:
        logger.info(f"  Execution RID: {exe.execution_rid}")

        # Save split parameters as config artifact
        split_params = {
            "source_dataset_rid": source_dataset_rid,
            "test_size": test_size,
            "train_size": train_size,
            "train_count": train_count,
            "test_count": test_count,
            "shuffle": shuffle,
            "seed": seed,
            "stratify_by_column": stratify_by_column,
            "element_table": element_table,
            "include_tables": include_tables,
            "training_types": train_types,
            "testing_types": test_types,
            "strategy": strategy_desc,
        }
        params_file = Path(exe.working_dir) / "split_config.json"
        params_file.write_text(json.dumps(split_params, indent=2))
        logger.info(f"  Saved split parameters to {params_file}")

        # Create parent Split dataset
        split_ds = exe.create_dataset(
            description=split_description or auto_description,
            dataset_types=["Split"],
        )
        logger.info(f"  Created Split dataset: {split_ds.dataset_rid}")

        # Create Training dataset
        training_ds = exe.create_dataset(
            description=(
                f"Training subset ({train_count} samples) of {source_dataset_rid} "
                f"({strategy_desc}, seed={seed})"
            ),
            dataset_types=train_types,
        )
        logger.info(f"  Created Training dataset: {training_ds.dataset_rid}")

        # Create Testing dataset
        testing_ds = exe.create_dataset(
            description=(
                f"Testing subset ({test_count} samples) of {source_dataset_rid} "
                f"({strategy_desc}, seed={seed})"
            ),
            dataset_types=test_types,
        )
        logger.info(f"  Created Testing dataset: {testing_ds.dataset_rid}")

        # Link children to parent
        split_ds.add_dataset_members(
            [training_ds.dataset_rid, testing_ds.dataset_rid], validate=False
        )
        logger.info("  Linked Training and Testing to Split dataset")

        # Add members to training dataset
        logger.info(f"  Adding {len(train_rids)} members to Training dataset...")
        batch_size = 500
        for i in range(0, len(train_rids), batch_size):
            batch = train_rids[i : i + batch_size]
            training_ds.add_dataset_members({element_table: batch}, validate=False)
            added = min(i + batch_size, len(train_rids))
            if added % 2000 == 0 or added >= len(train_rids):
                logger.info(f"    Added {added}/{len(train_rids)}")

        # Add members to testing dataset
        logger.info(f"  Adding {len(test_rids)} members to Testing dataset...")
        for i in range(0, len(test_rids), batch_size):
            batch = test_rids[i : i + batch_size]
            testing_ds.add_dataset_members({element_table: batch}, validate=False)
            added = min(i + batch_size, len(test_rids))
            if added % 2000 == 0 or added >= len(test_rids):
                logger.info(f"    Added {added}/{len(test_rids)}")

    # Upload execution outputs (after context manager exits)
    logger.info("Uploading execution outputs...")
    exe.upload_execution_outputs(clean_folder=True)

    # -------------------------------------------------------------------------
    # Build result with versions
    # -------------------------------------------------------------------------
    split_ds_info = ml.lookup_dataset(split_ds.dataset_rid)
    training_ds_info = ml.lookup_dataset(training_ds.dataset_rid)
    testing_ds_info = ml.lookup_dataset(testing_ds.dataset_rid)

    return {
        "split": split_ds.dataset_rid,
        "split_version": str(split_ds_info.current_version),
        "training": training_ds.dataset_rid,
        "training_version": str(training_ds_info.current_version),
        "testing": testing_ds.dataset_rid,
        "testing_version": str(testing_ds_info.current_version),
        "source": source_dataset_rid,
        "train_count": train_count,
        "test_count": test_count,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """CLI entry point for ``deriva-ml-split-dataset``.

    Parses command-line arguments, connects to a DerivaML catalog, and
    splits the specified dataset into training and testing subsets.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    import argparse
    import sys
    import textwrap

    parser = argparse.ArgumentParser(
        description="Split a DerivaML dataset into training/testing subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
            # Simple random 80/20 split
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0

            # Stratified split by class label
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0 \\
                --stratify-by-column Image_Classification_Image_Class \\
                --include-tables Image,Image_Classification

            # Fixed-count split
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0 --train-size 400 --test-size 100

            # Dry run (show plan without modifying catalog)
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0 --dry-run

        For more information, see:
            https://github.com/informatics-isi-edu/deriva-ml
        """),
    )

    # Connection parameters
    parser.add_argument(
        "--hostname", required=True,
        help="Deriva server hostname (e.g., localhost, ml.derivacloud.org)",
    )
    parser.add_argument(
        "--catalog-id", required=True,
        help="Catalog ID to connect to",
    )
    parser.add_argument(
        "--domain-schema",
        help="Domain schema name (auto-detected if not provided)",
    )

    # Source dataset
    parser.add_argument(
        "--dataset-rid", required=True,
        help="RID of the source dataset to split",
    )

    # Split parameters (scikit-learn conventions)
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test set size as fraction (0-1) or absolute count (default: 0.2)",
    )
    parser.add_argument(
        "--train-size", type=float, default=None,
        help="Train set size as fraction (0-1) or absolute count "
        "(default: complement of test-size)",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Do not shuffle before splitting",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--stratify-by-column",
        help="Column name in denormalized DataFrame for stratified splitting "
        "(e.g., Image_Classification_Image_Class). Requires --include-tables.",
    )

    # DerivaML parameters
    parser.add_argument(
        "--element-table",
        help="Element table to split (e.g., Image). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--include-tables",
        help="Comma-separated tables for denormalization "
        "(e.g., Image,Image_Classification). Required for stratified splitting.",
    )
    parser.add_argument(
        "--training-types", default="Labeled",
        help="Comma-separated additional dataset types for training set "
        "(default: Labeled)",
    )
    parser.add_argument(
        "--testing-types", default="Labeled",
        help="Comma-separated additional dataset types for testing set "
        "(default: Labeled)",
    )
    parser.add_argument(
        "--description", default="",
        help="Description for the parent split dataset",
    )
    parser.add_argument(
        "--workflow-type", default="Dataset_Split",
        help="Workflow type vocabulary term (default: Dataset_Split)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without modifying catalog",
    )
    parser.add_argument(
        "--show-urls", action="store_true",
        help="Show Chaise web interface URLs for created datasets",
    )

    args = parser.parse_args()

    # Configure logging
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger("deriva_ml").addHandler(handler)
    logging.getLogger("deriva_ml").setLevel(logging.INFO)

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    try:
        from deriva_ml import DerivaML

        # Connect
        logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(args.catalog_id),
            domain_schemas={args.domain_schema} if args.domain_schema else None,
            check_auth=True,
        )
        logger.info(f"Connected, domain schema: {ml.default_schema}")

        # Parse comma-separated lists
        include_tables = (
            [t.strip() for t in args.include_tables.split(",")]
            if args.include_tables else None
        )
        training_types = (
            [t.strip() for t in args.training_types.split(",")]
            if args.training_types else None
        )
        testing_types = (
            [t.strip() for t in args.testing_types.split(",")]
            if args.testing_types else None
        )

        # Run the split
        result = split_dataset(
            ml=ml,
            source_dataset_rid=args.dataset_rid,
            test_size=args.test_size,
            train_size=args.train_size,
            shuffle=not args.no_shuffle,
            seed=args.seed,
            stratify_by_column=args.stratify_by_column,
            split_description=args.description,
            training_types=training_types,
            testing_types=testing_types,
            element_table=args.element_table,
            include_tables=include_tables,
            workflow_type=args.workflow_type,
            dry_run=args.dry_run,
        )

        # Print summary
        if args.dry_run:
            print(f"\n{'='*60}")
            print("  DRY RUN - No changes will be made")
            print(f"{'='*60}")
            print(f"  Source dataset:  {result['source']}")
            print(f"  Element table:   {result.get('element_table', 'auto-detect')}")
            print(f"  Strategy:        {result.get('strategy', 'random')}")
            print(f"  Seed:            {args.seed}")
            print(f"  Training size:   {result['train_count']}")
            print(f"  Testing size:    {result['test_count']}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("  SPLIT COMPLETE")
            print(f"{'='*60}")
            print(f"  Source dataset:  {result['source']}")
            print(f"  Split dataset:   {result['split']} (v{result['split_version']})")
            print(f"  Training:        {result['training']} (v{result['training_version']})")
            print(f"  Testing:         {result['testing']} (v{result['testing_version']})")

            if args.show_urls:
                print()
                print("  Chaise URLs:")
                for name in ["split", "training", "testing"]:
                    try:
                        url = ml.cite(result[name], current=True)
                        print(f"    {name}: {url}")
                    except Exception:
                        pass

            print(f"{'='*60}\n")

        return 0

    except Exception as e:
        logger.error(f"Split failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

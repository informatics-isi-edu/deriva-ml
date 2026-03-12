"""Generic dataset splitting for DerivaML.

This module provides functions to split a DerivaML dataset into training,
testing, and optionally validation subsets with full provenance tracking.
It works with any DerivaML catalog and any registered element type.

The splitting API follows scikit-learn conventions (``test_size``,
``train_size``, ``val_size``, ``shuffle``, ``seed``, ``stratify``) while
integrating with DerivaML's dataset hierarchy, execution provenance, and
versioning.

Splitting Strategies:
    Random (default):
        Shuffles members and splits at the partition boundaries.
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

    Three-way train/val/test split::

        result = split_dataset(
            ml, "28D0",
            test_size=0.2,
            val_size=0.1,
            seed=42,
        )

    Stratified split::

        result = split_dataset(
            ml, "28D0",
            test_size=0.2,
            stratify_by_column="Image_Classification_Image_Class",
            include_tables=["Image", "Image_Classification"],
        )

    Custom selection function::

        def my_selector(df, partition_sizes, seed):
            # Custom logic...
            return {"Training": train_indices, "Testing": test_indices}

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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from deriva_ml import DerivaML

from deriva_ml.execution import ExecutionConfiguration

logger = logging.getLogger("deriva_ml")


# =============================================================================
# Result Models
# =============================================================================


class PartitionInfo(BaseModel):
    """Information about a single partition (Training, Testing, or Validation)."""

    rid: str
    version: str
    count: int


class SplitResult(BaseModel):
    """Result of a dataset split operation."""

    source: str
    split: PartitionInfo
    training: PartitionInfo
    testing: PartitionInfo
    validation: PartitionInfo | None = None
    strategy: str
    element_table: str
    seed: int
    dry_run: bool = False


# =============================================================================
# Selection Function Protocol and Built-in Implementations
# =============================================================================


@runtime_checkable
class SelectionFunction(Protocol):
    """Protocol for custom partition selection functions.

    A selection function receives the denormalized dataset DataFrame and
    returns a dict mapping partition names to integer index arrays into
    the DataFrame rows.

    The function is responsible for:

    - Deciding which records go into each partition
    - Ensuring the sizes match the requested partition_sizes
    - Implementing any balancing or stratification logic

    Args:
        df: Denormalized DataFrame from ``dataset.denormalize_as_dataframe()``.
            Columns are prefixed with table names (e.g., ``Image_RID``,
            ``Image_Classification_Image_Class``).
        partition_sizes: Dict mapping partition names (e.g., "Training",
            "Testing", "Validation") to the number of records for each.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping partition names to numpy arrays of integer indices
        into the DataFrame.

    Example:
        >>> def balanced_selector(df, partition_sizes, seed):
        ...     rng = np.random.default_rng(seed)
        ...     # ... balance classes ...
        ...     return {"Training": train_indices, "Testing": test_indices}
    """

    def __call__(
        self,
        df: pd.DataFrame,
        partition_sizes: dict[str, int],
        seed: int,
    ) -> dict[str, np.ndarray]: ...


def random_split(
    df: pd.DataFrame,
    partition_sizes: dict[str, int],
    seed: int,
) -> dict[str, np.ndarray]:
    """Random split into N partitions.

    Shuffles the DataFrame indices and splits at partition boundaries.

    Args:
        df: Source DataFrame.
        partition_sizes: Dict mapping partition names to counts.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping partition names to index arrays.
    """
    rng = np.random.default_rng(seed)
    total_needed = sum(partition_sizes.values())
    indices = np.arange(len(df))
    rng.shuffle(indices)
    indices = indices[:total_needed]

    result = {}
    offset = 0
    for name, size in partition_sizes.items():
        result[name] = indices[offset : offset + size]
        offset += size
    return result


def stratified_split(
    stratify_column: str,
    missing: str = "error",
) -> SelectionFunction:
    """Create a stratified selection function.

    Returns a selection function that maintains the class distribution
    of the specified column across all partitions. Delegates to
    scikit-learn's ``train_test_split`` for the actual stratification.

    For two-way splits, performs a single stratified split. For three-way
    splits (Training/Validation/Testing), first separates the test set,
    then splits the remainder into training and validation.

    Args:
        stratify_column: Column name in the denormalized DataFrame to
            stratify by (e.g., ``Image_Classification_Image_Class``).
        missing: Policy for handling null/NaN values in the stratify column.
            - ``"error"`` (default): Raise ``ValueError`` if any values
              are missing. Reports the count and percentage of nulls.
            - ``"drop"``: Silently exclude rows with missing values from
              the split. Only rows with valid stratify values are assigned
              to partitions.
            - ``"include"``: Treat null/NaN as a distinct class label
              (``"__missing__"``). Missing-value rows are distributed
              across partitions proportionally like any other class.

    Returns:
        A ``SelectionFunction`` that performs stratified splitting.

    Raises:
        ValueError: If ``missing="error"`` and the stratify column
            contains null values.

    Example:
        >>> selector = stratified_split("Image_Classification_Image_Class")
        >>> partitions = selector(df, {"Training": 400, "Testing": 100}, seed=42)

        >>> # Drop rows with missing labels
        >>> selector = stratified_split("Diagnosis_Label", missing="drop")
        >>> partitions = selector(df, {"Training": 300, "Testing": 100}, seed=42)
    """
    if missing not in ("error", "drop", "include"):
        raise ValueError(
            f"missing must be 'error', 'drop', or 'include', got '{missing}'"
        )

    def _stratified_split(
        df: pd.DataFrame,
        partition_sizes: dict[str, int],
        seed: int,
    ) -> dict[str, np.ndarray]:
        from sklearn.model_selection import train_test_split as sklearn_split

        total_needed = sum(partition_sizes.values())

        if stratify_column not in df.columns:
            available = [c for c in df.columns if not c.startswith("_")]
            raise ValueError(
                f"Column '{stratify_column}' not found in denormalized DataFrame. "
                f"Available columns: {available}"
            )

        # Handle missing values in the stratify column
        null_mask = df[stratify_column].isna()
        null_count = null_mask.sum()

        if null_count > 0:
            null_pct = null_count / len(df) * 100
            if missing == "error":
                raise ValueError(
                    f"Column '{stratify_column}' has {null_count} missing values "
                    f"({null_pct:.1f}% of {len(df)} rows). "
                    f"Use stratify_missing='drop' to exclude these rows, "
                    f"or 'include' to treat nulls as a separate class."
                )
            elif missing == "drop":
                logger.info(
                    f"Dropping {null_count} rows ({null_pct:.1f}%) with missing "
                    f"values in '{stratify_column}'"
                )
                df = df[~null_mask].reset_index(drop=True)
            elif missing == "include":
                logger.info(
                    f"Treating {null_count} missing values ({null_pct:.1f}%) in "
                    f"'{stratify_column}' as class '__missing__'"
                )
                df = df.copy()
                df[stratify_column] = df[stratify_column].fillna("__missing__")

        if total_needed > len(df):
            raise ValueError(
                f"Requested {total_needed} samples but dataset has {len(df)} records"
                + (f" (after dropping {null_count} rows with missing values)"
                   if null_count > 0 and missing == "drop" else "")
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

        # Partition names in the order we'll peel them off
        partition_names = list(partition_sizes.keys())

        if len(partition_names) == 2:
            # Two-way split: single stratified split
            test_name = partition_names[1]
            train_name = partition_names[0]
            test_fraction = partition_sizes[test_name] / total_needed
            train_idx, test_idx = sklearn_split(
                np.arange(len(sub_df)),
                test_size=test_fraction,
                stratify=sub_df[stratify_column].values,
                random_state=seed,
            )
            return {
                train_name: subset_indices[train_idx],
                test_name: subset_indices[test_idx],
            }
        else:
            # Three-way split: peel off Testing first, then split remainder
            # into Training and Validation.
            test_size = partition_sizes["Testing"]
            test_fraction = test_size / total_needed
            remainder_idx, test_idx = sklearn_split(
                np.arange(len(sub_df)),
                test_size=test_fraction,
                stratify=sub_df[stratify_column].values,
                random_state=seed,
            )

            remainder_df = sub_df.iloc[remainder_idx]
            remainder_total = partition_sizes["Training"] + partition_sizes["Validation"]
            val_fraction = partition_sizes["Validation"] / remainder_total
            train_idx, val_idx = sklearn_split(
                np.arange(len(remainder_df)),
                test_size=val_fraction,
                stratify=remainder_df[stratify_column].values,
                random_state=seed,
            )

            return {
                "Training": subset_indices[remainder_idx[train_idx]],
                "Validation": subset_indices[remainder_idx[val_idx]],
                "Testing": subset_indices[test_idx],
            }

    return _stratified_split


# =============================================================================
# Size Resolution Helper
# =============================================================================


def _resolve_sizes(
    total: int,
    test_size: float | int,
    train_size: float | int | None = None,
    val_size: float | int | None = None,
) -> dict[str, int]:
    """Convert fractional or absolute sizes to absolute counts.

    Follows scikit-learn's convention: if both are fractions, they should
    sum to <= 1.0. If train_size is None, it's computed as the complement.

    Args:
        total: Total number of records in the dataset.
        test_size: Fraction (0-1) or absolute count of test samples.
        train_size: Fraction (0-1) or absolute count of train samples,
            or None for complement of test_size (minus val_size if provided).
        val_size: Fraction (0-1) or absolute count of validation samples,
            or None for no validation split.

    Returns:
        Dict mapping partition names ("Training", "Testing", and optionally
        "Validation") to integer counts.

    Raises:
        ValueError: If sizes are invalid or exceed total.
    """

    def _to_count(size: float | int, name: str) -> int:
        if isinstance(size, float) and 0 < size < 1:
            return int(round(total * size))
        elif isinstance(size, (int, float)) and size >= 1:
            return int(size)
        else:
            raise ValueError(
                f"{name} must be a float in (0, 1) or an int >= 1, got {size}"
            )

    test_count = _to_count(test_size, "test_size")

    val_count = 0
    if val_size is not None:
        val_count = _to_count(val_size, "val_size")

    if train_size is None:
        train_count = total - test_count - val_count
    else:
        train_count = _to_count(train_size, "train_size")

    if train_count + test_count + val_count > total:
        raise ValueError(
            f"Requested train_size={train_count} + test_size={test_count}"
            + (f" + val_size={val_count}" if val_size is not None else "")
            + f" = {train_count + test_count + val_count}"
            + f" exceeds total dataset size of {total}"
        )

    if train_count <= 0:
        raise ValueError(
            f"Training set must have at least 1 sample. Got train_size={train_count}"
        )
    if test_count <= 0:
        raise ValueError(
            f"Test set must have at least 1 sample. Got test_size={test_count}"
        )
    if val_size is not None and val_count <= 0:
        raise ValueError(
            f"Validation set must have at least 1 sample. Got val_size={val_count}"
        )

    result = {"Training": train_count, "Testing": test_count}
    if val_size is not None:
        result["Validation"] = val_count
    return result


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
        "Validation": "A dataset subset used for model validation during training",
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
    val_size: float | int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    stratify_by_column: str | None = None,
    stratify_missing: str = "error",
    # DerivaML-specific parameters
    split_description: str = "",
    training_types: list[str] | None = None,
    testing_types: list[str] | None = None,
    validation_types: list[str] | None = None,
    element_table: str | None = None,
    include_tables: list[str] | None = None,
    selection_fn: SelectionFunction | None = None,
    workflow_type: str = "Dataset_Split",
    dry_run: bool = False,
) -> SplitResult:
    """Split a DerivaML dataset into training, testing, and optionally validation subsets.

    Creates a new dataset hierarchy in the catalog::

        Split (parent, type: "Split")
        +-- Training (child, type: "Training", + training_types)
        +-- Validation (child, type: "Validation", + validation_types)  # if val_size
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
            If None, complement of test_size (and val_size). Default: None.
        val_size: If float (0-1), fraction of data for validation.
            If int, absolute number of validation samples.
            If None, no validation split is created (two-way split).
            Default: None.
        shuffle: Whether to shuffle before splitting. Default: True.
            Ignored when using stratified or custom selection functions
            (they handle their own shuffling).
        seed: Random seed for reproducibility. Default: 42.
        stratify_by_column: Column name for stratified splitting.
            Must be a column in the denormalized DataFrame (prefixed
            with table name, e.g., ``Image_Classification_Image_Class``).
            Mutually exclusive with ``selection_fn``.
        stratify_missing: Policy for null values in the stratify column.
            ``"error"`` (default) raises if any nulls exist,
            ``"drop"`` excludes rows with nulls,
            ``"include"`` treats nulls as a separate class.
            Only used when ``stratify_by_column`` is set.
        split_description: Description for the parent Split dataset.
        training_types: Additional dataset types for the training set
            beyond "Training" (e.g., ``["Labeled"]``). Default: None.
        testing_types: Additional dataset types for the testing set
            beyond "Testing" (e.g., ``["Labeled"]``). Default: None.
        validation_types: Additional dataset types for the validation set
            beyond "Validation" (e.g., ``["Labeled"]``). Default: None.
            Ignored when val_size is None.
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
        SplitResult with partition info for split, training, testing,
        and optionally validation datasets.

    Raises:
        ValueError: If sizes are invalid, dataset has no members, or
            parameters conflict.

    Example:
        Simple random 80/20 split::

            from deriva_ml import DerivaML
            from deriva_ml.dataset.split import split_dataset

            ml = DerivaML("localhost", "9")
            result = split_dataset(ml, "28D0", test_size=0.2, seed=42)
            print(f"Training: {result.training.rid} ({result.training.count} samples)")
            print(f"Testing:  {result.testing.rid} ({result.testing.count} samples)")

        Three-way train/val/test split::

            result = split_dataset(
                ml, "28D0",
                test_size=0.2,
                val_size=0.1,
                seed=42,
            )
            print(f"Validation: {result.validation.rid} ({result.validation.count} samples)")

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

        Stratified split dropping rows with missing labels::

            result = split_dataset(
                ml, "28D0",
                test_size=0.2,
                stratify_by_column="Image_Classification_Image_Class",
                stratify_missing="drop",
                include_tables=["Image", "Image_Classification"],
            )

        Custom selection function for balanced sampling::

            import numpy as np

            def balanced_selector(df, partition_sizes, seed):
                rng = np.random.default_rng(seed)
                label_col = "Image_Classification_Image_Class"
                classes = df[label_col].unique()
                result = {name: [] for name in partition_sizes}
                for cls in classes:
                    cls_indices = df.index[df[label_col] == cls].to_numpy()
                    rng.shuffle(cls_indices)
                    offset = 0
                    for name, size in partition_sizes.items():
                        per_class = size // len(classes)
                        result[name].extend(cls_indices[offset:offset + per_class])
                        offset += per_class
                return {name: np.array(idx) for name, idx in result.items()}

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
            print(f"Would create: {result.training.count} train, "
                  f"{result.testing.count} test")

        Use returned RIDs to create a hydra-zen configuration::

            from deriva_ml.dataset import DatasetSpecConfig

            result = split_dataset(ml, "28D0", test_size=0.2, seed=42)
            split_config = DatasetSpecConfig(
                rid=result.split.rid,
                version=result.split.version,
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
    partition_sizes = _resolve_sizes(total, test_size, train_size, val_size)
    size_summary = ", ".join(f"{k}={v}" for k, v in partition_sizes.items())
    logger.info(f"Split sizes: {size_summary} (total={total})")

    # -------------------------------------------------------------------------
    # Determine selection strategy and get partition RIDs
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
            selector = stratified_split(stratify_by_column, missing=stratify_missing)
        else:
            logger.info("Using custom selection function")
            selector = selection_fn

        partition_indices = selector(df, partition_sizes, seed)

        # Map indices back to RIDs
        rid_column = f"{element_table}_RID"
        if rid_column not in df.columns:
            rid_column = "RID"
            if rid_column not in df.columns:
                raise ValueError(
                    f"Cannot find RID column. Tried '{element_table}_RID' and 'RID'. "
                    f"Available columns: {list(df.columns)}"
                )

        partition_rids = {
            name: df.iloc[indices][rid_column].tolist()
            for name, indices in partition_indices.items()
        }

    else:
        all_rids = [record["RID"] for record in member_records]

        if shuffle:
            rng = np.random.default_rng(seed)
            indices = np.arange(len(all_rids))
            rng.shuffle(indices)
            all_rids = [all_rids[i] for i in indices]

        partition_rids = {}
        offset = 0
        for name, size in partition_sizes.items():
            partition_rids[name] = all_rids[offset : offset + size]
            offset += size

    for name, rids in partition_rids.items():
        logger.info(f"Selected {len(rids)} {name} RIDs")

    # -------------------------------------------------------------------------
    # Compute strategy description
    # -------------------------------------------------------------------------
    strategy_desc = (
        f"stratified by {stratify_by_column}" if stratify_by_column else "random"
    )
    if selection_fn:
        strategy_desc = "custom selection function"

    # -------------------------------------------------------------------------
    # Dry run
    # -------------------------------------------------------------------------
    if dry_run:
        result = SplitResult(
            source=source_dataset_rid,
            split=PartitionInfo(rid="(dry run)", version="(dry run)", count=0),
            training=PartitionInfo(
                rid="(dry run)",
                version="(dry run)",
                count=partition_sizes["Training"],
            ),
            testing=PartitionInfo(
                rid="(dry run)",
                version="(dry run)",
                count=partition_sizes["Testing"],
            ),
            validation=(
                PartitionInfo(
                    rid="(dry run)",
                    version="(dry run)",
                    count=partition_sizes["Validation"],
                )
                if "Validation" in partition_sizes
                else None
            ),
            strategy=strategy_desc,
            element_table=element_table,
            seed=seed,
            dry_run=True,
        )
        return result

    # -------------------------------------------------------------------------
    # Ensure vocabulary terms exist
    # -------------------------------------------------------------------------
    _ensure_workflow_type(ml, workflow_type)
    _ensure_dataset_types(ml)

    # -------------------------------------------------------------------------
    # Create execution and dataset hierarchy
    # -------------------------------------------------------------------------
    partitions_desc = ", ".join(f"{k}={v}" for k, v in partition_sizes.items())
    auto_description = (
        f"Split of dataset {source_dataset_rid} "
        f"({strategy_desc}, {partitions_desc}, seed={seed})"
    )

    logger.info("Creating workflow and execution...")
    workflow = ml.create_workflow(
        name=f"Dataset Split: {source_dataset_rid}",
        workflow_type=workflow_type,
        description="Split dataset into training/testing/validation subsets",
    )

    config = ExecutionConfiguration(
        workflow=workflow,
        description=split_description or auto_description,
    )

    train_types = ["Training"] + (training_types or [])
    test_types = ["Testing"] + (testing_types or [])
    val_types = ["Validation"] + (validation_types or []) if val_size is not None else []

    with ml.create_execution(config) as exe:
        logger.info(f"  Execution RID: {exe.execution_rid}")

        # Save split parameters as config artifact
        split_params = {
            "source_dataset_rid": source_dataset_rid,
            "test_size": test_size,
            "train_size": train_size,
            "val_size": val_size,
            "partition_sizes": partition_sizes,
            "shuffle": shuffle,
            "seed": seed,
            "stratify_by_column": stratify_by_column,
            "stratify_missing": stratify_missing,
            "element_table": element_table,
            "include_tables": include_tables,
            "training_types": train_types,
            "testing_types": test_types,
            "validation_types": val_types if val_types else None,
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
                f"Training subset ({partition_sizes['Training']} samples) of "
                f"{source_dataset_rid} ({strategy_desc}, seed={seed})"
            ),
            dataset_types=train_types,
        )
        logger.info(f"  Created Training dataset: {training_ds.dataset_rid}")

        # Create Validation dataset (if requested)
        validation_ds = None
        if val_size is not None:
            validation_ds = exe.create_dataset(
                description=(
                    f"Validation subset ({partition_sizes['Validation']} samples) of "
                    f"{source_dataset_rid} ({strategy_desc}, seed={seed})"
                ),
                dataset_types=val_types,
            )
            logger.info(f"  Created Validation dataset: {validation_ds.dataset_rid}")

        # Create Testing dataset
        testing_ds = exe.create_dataset(
            description=(
                f"Testing subset ({partition_sizes['Testing']} samples) of "
                f"{source_dataset_rid} ({strategy_desc}, seed={seed})"
            ),
            dataset_types=test_types,
        )
        logger.info(f"  Created Testing dataset: {testing_ds.dataset_rid}")

        # Link children to parent
        child_rids = [training_ds.dataset_rid, testing_ds.dataset_rid]
        if validation_ds is not None:
            child_rids.insert(1, validation_ds.dataset_rid)
        split_ds.add_dataset_members(child_rids, validate=False)
        logger.info("  Linked child datasets to Split dataset")

        # Add members to each partition
        batch_size = 500
        for part_name, ds in [
            ("Training", training_ds),
            ("Validation", validation_ds),
            ("Testing", testing_ds),
        ]:
            if ds is None:
                continue
            rids = partition_rids[part_name]
            logger.info(f"  Adding {len(rids)} members to {part_name} dataset...")
            for i in range(0, len(rids), batch_size):
                batch = rids[i : i + batch_size]
                ds.add_dataset_members({element_table: batch}, validate=False)
                added = min(i + batch_size, len(rids))
                if added % 2000 == 0 or added >= len(rids):
                    logger.info(f"    Added {added}/{len(rids)}")

    # Upload execution outputs (after context manager exits)
    logger.info("Uploading execution outputs...")
    exe.upload_execution_outputs(clean_folder=True)

    # -------------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------------
    split_ds_info = ml.lookup_dataset(split_ds.dataset_rid)
    training_ds_info = ml.lookup_dataset(training_ds.dataset_rid)
    testing_ds_info = ml.lookup_dataset(testing_ds.dataset_rid)

    validation_info = None
    if validation_ds is not None:
        validation_ds_info = ml.lookup_dataset(validation_ds.dataset_rid)
        validation_info = PartitionInfo(
            rid=validation_ds.dataset_rid,
            version=str(validation_ds_info.current_version),
            count=partition_sizes["Validation"],
        )

    return SplitResult(
        source=source_dataset_rid,
        split=PartitionInfo(
            rid=split_ds.dataset_rid,
            version=str(split_ds_info.current_version),
            count=0,
        ),
        training=PartitionInfo(
            rid=training_ds.dataset_rid,
            version=str(training_ds_info.current_version),
            count=partition_sizes["Training"],
        ),
        testing=PartitionInfo(
            rid=testing_ds.dataset_rid,
            version=str(testing_ds_info.current_version),
            count=partition_sizes["Testing"],
        ),
        validation=validation_info,
        strategy=strategy_desc,
        element_table=element_table,
        seed=seed,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """CLI entry point for ``deriva-ml-split-dataset``.

    Parses command-line arguments, connects to a DerivaML catalog, and
    splits the specified dataset into training, testing, and optionally
    validation subsets.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    import argparse
    import sys
    import textwrap

    parser = argparse.ArgumentParser(
        description="Split a DerivaML dataset into training/testing/validation subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
            # Simple random 80/20 split
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0

            # Three-way train/val/test split
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0 --val-size 0.1

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
        "--val-size", type=float, default=None,
        help="Validation set size as fraction (0-1) or absolute count "
        "(default: None, no validation split)",
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
    parser.add_argument(
        "--stratify-missing",
        choices=["error", "drop", "include"],
        default="error",
        help="Policy for null values in the stratify column: "
        "'error' (default) raises, 'drop' excludes nulls, "
        "'include' treats nulls as a separate class.",
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
        "--validation-types", default="Labeled",
        help="Comma-separated additional dataset types for validation set "
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
        validation_types = (
            [t.strip() for t in args.validation_types.split(",")]
            if args.validation_types else None
        )

        # Run the split
        result = split_dataset(
            ml=ml,
            source_dataset_rid=args.dataset_rid,
            test_size=args.test_size,
            train_size=args.train_size,
            val_size=args.val_size,
            shuffle=not args.no_shuffle,
            seed=args.seed,
            stratify_by_column=args.stratify_by_column,
            stratify_missing=args.stratify_missing,
            split_description=args.description,
            training_types=training_types,
            testing_types=testing_types,
            validation_types=validation_types,
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
            print(f"  Source dataset:  {result.source}")
            print(f"  Element table:   {result.element_table}")
            print(f"  Strategy:        {result.strategy}")
            print(f"  Seed:            {result.seed}")
            print(f"  Training size:   {result.training.count}")
            if result.validation:
                print(f"  Validation size: {result.validation.count}")
            print(f"  Testing size:    {result.testing.count}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("  SPLIT COMPLETE")
            print(f"{'='*60}")
            print(f"  Source dataset:  {result.source}")
            print(f"  Split dataset:   {result.split.rid} (v{result.split.version})")
            print(f"  Training:        {result.training.rid} (v{result.training.version})")
            if result.validation:
                print(f"  Validation:      {result.validation.rid} (v{result.validation.version})")
            print(f"  Testing:         {result.testing.rid} (v{result.testing.version})")

            if args.show_urls:
                print()
                print("  Chaise URLs:")
                for name, info in [
                    ("split", result.split),
                    ("training", result.training),
                    ("validation", result.validation),
                    ("testing", result.testing),
                ]:
                    if info is None:
                        continue
                    try:
                        url = ml.cite(info.rid, current=True)
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

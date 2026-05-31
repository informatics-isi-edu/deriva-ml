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
    ``split_dataset`` runs inside an Execution the caller has already
    opened. The caller's workflow identifies the code making the
    splitting decision; deriva-ml never invents a workflow on the
    caller's behalf, so this function is safe to call from
    environments without a git checkout (notebook kernels, MCP
    servers, scheduled jobs) as long as the caller has wired up a
    workflow with honest provenance::

        from deriva_ml import DerivaML
        from deriva_ml.dataset.split import split_dataset
        from deriva_ml.execution import ExecutionConfiguration

        ml = DerivaML("localhost", "9")

        workflow = ml.create_workflow(
            name="My splitting script",
            workflow_type="Dataset_Split",
            description="80/20 train/test for sleep-stage classifier v3",
        )
        config = ExecutionConfiguration(workflow=workflow)

        with ml.create_execution(config) as exe:
            result = split_dataset(ml, "28D0", exe, test_size=0.2, seed=42)
        exe.commit_output_assets(clean_folder=True)

    Three-way train/val/test split (same execution, reuse ``exe``)::

        result = split_dataset(
            ml, "28D0", exe,
            test_size=0.2,
            val_size=0.1,
            seed=42,
        )

    Stratified split::

        result = split_dataset(
            ml, "28D0", exe,
            test_size=0.2,
            stratify_by_column="Image_Class.Name",
            include_tables=["Image", "Image_Class"],
        )

    Custom selection function::

        def my_selector(df, partition_sizes, seed):
            # Custom logic...
            return {"Training": train_indices, "Testing": test_indices}

        result = split_dataset(
            ml, "28D0", exe,
            test_size=100,
            selection_fn=my_selector,
            include_tables=["Image", "Image_Classification"],
        )

See Also:
    - ``sklearn.model_selection.train_test_split``
    - ``Dataset.get_denormalized_as_dataframe``
    - ``Dataset.list_dataset_members``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from deriva_ml import DerivaML
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.execution import Execution

from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)
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
        df: Denormalized DataFrame from ``dataset.get_denormalized_as_dataframe()``.
            Columns use dot notation ``Table.column`` (e.g., ``Image.RID``,
            ``Image_Class.Name``) — see :func:`denormalize_column_name`.
        partition_sizes: Dict mapping partition names (e.g., "Training",
            "Testing", "Validation") to the number of records for each.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping partition names to numpy arrays of integer indices
        into the DataFrame.

    Example:
        >>> def balanced_selector(df, partition_sizes, seed):  # doctest: +SKIP
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
            stratify by, in dot notation (e.g., ``Image_Class.Name``).
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
        >>> selector = stratified_split("Image_Class.Name")  # doctest: +SKIP
        >>> partitions = selector(df, {"Training": 400, "Testing": 100}, seed=42)  # doctest: +SKIP

        >>> # Drop rows with missing labels
        >>> selector = stratified_split("Image_Class.Name", missing="drop")  # doctest: +SKIP
        >>> partitions = selector(df, {"Training": 300, "Testing": 100}, seed=42)  # doctest: +SKIP
    """
    if missing not in ("error", "drop", "include"):
        raise ValueError(f"missing must be 'error', 'drop', or 'include', got '{missing}'")

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
                f"Column '{stratify_column}' not found in denormalized DataFrame. Available columns: {available}"
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
                logger.info(f"Dropping {null_count} rows ({null_pct:.1f}%) with missing values in '{stratify_column}'")
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
                + (
                    f" (after dropping {null_count} rows with missing values)"
                    if null_count > 0 and missing == "drop"
                    else ""
                )
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
            raise ValueError(f"{name} must be a float in (0, 1) or an int >= 1, got {size}")

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
        raise ValueError(f"Training set must have at least 1 sample. Got train_size={train_count}")
    if test_count <= 0:
        raise ValueError(f"Test set must have at least 1 sample. Got test_size={test_count}")
    if val_size is not None and val_count <= 0:
        raise ValueError(f"Validation set must have at least 1 sample. Got val_size={val_count}")

    result = {"Training": train_count, "Testing": test_count}
    if val_size is not None:
        result["Validation"] = val_count
    return result


# =============================================================================
# Vocabulary Helpers
# =============================================================================


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


def _validate_split_inputs(
    *,
    stratify_by_column: str | None,
    selection_fn: SelectionFunction | None,
    include_tables: list[str] | None,
    row_per: str | None = None,
    element_table: str | None = None,
    partition_by: Literal["element", "row"] | None = None,
) -> Literal["element", "row"]:
    """Validate the mutually-exclusive + required-companion arg constraints.

    Four rules:

    1. ``stratify_by_column`` and ``selection_fn`` are mutually
       exclusive — both produce per-element partitioning decisions
       so allowing both would let the caller silently choose
       whichever the implementation checked first.
    2. ``stratify_by_column`` requires ``include_tables`` (it
       drives a denormalization that needs explicit table scope).
    3. ``selection_fn`` requires ``include_tables`` for the same
       reason.
    4. ``partition_by`` is required whenever ``row_per`` is set and
       differs from ``element_table`` — the (row_per !=
       element_table) shape is exactly the silent-leakage case
       (one element_table RID can have multiple denormalized rows
       that the selector may scatter across partitions), so the
       caller must declare intent explicitly. When ``row_per`` is
       ``None`` or equals ``element_table`` the partition unit is
       unambiguous and ``partition_by`` auto-defaults to
       ``"element"``.

    Args:
        stratify_by_column: As passed to :func:`split_dataset`.
        selection_fn: As passed to :func:`split_dataset`.
        include_tables: As passed to :func:`split_dataset`.
        row_per: As passed to :func:`split_dataset`. Used only to
            decide whether ``partition_by`` is required.
        element_table: As passed to :func:`split_dataset`. Compared
            against ``row_per`` to decide ambiguity.
        partition_by: As passed to :func:`split_dataset`.

    Returns:
        The effective ``partition_by`` value after defaulting. When
        the caller passed ``"element"`` or ``"row"`` it is returned
        as-is. When the caller passed ``None`` and ambiguity is
        absent (``row_per`` is ``None`` or equals ``element_table``),
        returns ``"element"``. When ``partition_by`` is required and
        the caller passed ``None``, raises ``ValueError`` instead of
        returning.

    Raises:
        ValueError: When the mutual-exclusion / requires rules are
            violated, or when ``partition_by`` is required and not
            supplied, or when ``partition_by`` is not one of the
            allowed string values.

    Note:
        ``ValueError`` rather than a ``DerivaMLException`` to match
        the existing surface: these are argument-shape errors the
        caller should fix at the call site, not catalog-side
        failures.
    """
    if stratify_by_column and selection_fn:
        raise ValueError("stratify_by_column and selection_fn are mutually exclusive. Use one or the other.")

    if stratify_by_column and not include_tables:
        raise ValueError(
            "include_tables is required when using stratify_by_column. "
            "Specify the tables needed for denormalization "
            "(e.g., include_tables=['Image', 'Image_Class'])."
        )

    if selection_fn and not include_tables:
        raise ValueError(
            "include_tables is required when using a custom selection_fn. "
            "Specify the tables needed for denormalization."
        )

    if partition_by is not None and partition_by not in ("element", "row"):
        raise ValueError(f"partition_by must be 'element', 'row', or None, got {partition_by!r}.")

    # Decide whether the (row_per, element_table) pair is ambiguous.
    # Ambiguity exists only when row_per is set AND differs from the
    # element_table — that's the shape where one element_table RID
    # can have multiple denormalized rows that the selector may
    # scatter across partitions (the silent-leakage case).
    row_per_differs = row_per is not None and row_per != element_table

    if partition_by is None:
        if row_per_differs:
            raise ValueError(
                "partition_by is required when row_per != element_table "
                f"(got row_per={row_per!r}, element_table={element_table!r}). "
                "This (row_per, element_table) shape is ambiguous: the "
                "denormalized dataframe has multiple rows per "
                f"{element_table or 'element'} RID, and the selector "
                "may scatter them across partitions — silently putting "
                "the same RID in train and test.\n\n"
                "Pick the intent explicitly:\n"
                "  partition_by='element' — dedupe rows per "
                f"{element_table or 'element'} RID before partitioning; "
                "partitions are disjoint at the element-RID level. "
                "Requires within-element agreement on selector-read "
                "columns (stratify_by_column).\n"
                "  partition_by='row'     — partition rows directly; "
                f"{element_table or 'element'} RIDs may appear in "
                "multiple partitions (per-annotation statistics, "
                "intentional row-level granularity)."
            )
        # Unambiguous case: auto-default to "element".
        return "element"

    return partition_by


def _dedupe_for_element_partition(
    df: pd.DataFrame,
    *,
    rid_column: str,
    element_table: str | None,
    stratify_by_column: str | None,
    stratify_missing: str,
    using_selection_fn: bool,
) -> pd.DataFrame:
    """Reduce the denormalized df to one row per element_table RID.

    Two steps:

    1. Verify within-element agreement on selector-read columns.
       For stratified splits that means ``stratify_by_column``;
       for custom ``selection_fn`` we skip (the read set is opaque)
       and the docstring warns the caller they own this check.
       NaN-handling follows ``stratify_missing`` — ``"error"`` raises
       on any NaN in a group, ``"drop"`` excludes whole groups whose
       members are NaN, ``"include"`` treats NaN as a sentinel that
       must match.
    2. Dedupe with stable, seed-deterministic ordering — sort by
       ``rid_column`` to make iteration order reproducible, then
       ``drop_duplicates(keep="first")`` on ``rid_column``. When
       within-element values agree (guaranteed by step 1), the
       first row encountered is representative.

    Args:
        df: Denormalized DataFrame from
            :meth:`Dataset.get_denormalized_as_dataframe`.
        rid_column: Name of the column carrying element_table RIDs
            (e.g., ``"Image.RID"``).
        element_table: Element table name (used in error messages).
        stratify_by_column: Column the stratified selector will
            read, or ``None`` when using ``selection_fn``.
        stratify_missing: NaN policy as passed to
            :func:`split_dataset` (``"error"`` / ``"drop"`` /
            ``"include"``). Only applied during the within-element
            check; doesn't override the selector's own NaN policy.
        using_selection_fn: ``True`` when the caller passed a custom
            ``selection_fn``. Skips the within-element check —
            the selector's read set is opaque so we can't enforce
            uniformity, and the docstring documents this is the
            caller's responsibility.

    Returns:
        DataFrame with one row per ``rid_column`` value.

    Raises:
        ValueError: When ``stratify_by_column`` is set and some
            element_table RID has disagreeing values across its
            rows. The error message names the consensus-feature
            pattern as the deriva-ml-shape fix.
    """
    if stratify_by_column and not using_selection_fn:
        if stratify_by_column not in df.columns:
            # Defer this error to the selector, which prints
            # the available columns helpfully.
            pass
        else:
            check_df = df[[rid_column, stratify_by_column]].copy()

            # NaN handling: match stratify_missing policy at the
            # group level.
            null_mask = check_df[stratify_by_column].isna()
            if null_mask.any():
                if stratify_missing == "error":
                    null_count = int(null_mask.sum())
                    raise ValueError(
                        f"Column '{stratify_by_column}' has {null_count} missing values. "
                        "Use stratify_missing='drop' to exclude these rows, "
                        "or 'include' to treat nulls as a separate class."
                    )
                elif stratify_missing == "drop":
                    # Drop any group that has ANY NaN — even partial
                    # NaN agreement within a group is uninformative
                    # for stratification.
                    bad_rids = set(check_df.loc[null_mask, rid_column])
                    check_df = check_df[~check_df[rid_column].isin(bad_rids)].copy()
                    df = df[~df[rid_column].isin(bad_rids)].copy()
                elif stratify_missing == "include":
                    check_df[stratify_by_column] = check_df[stratify_by_column].fillna("__missing__")

            # Within-element uniformity check.
            disagreement = check_df.groupby(rid_column)[stratify_by_column].nunique(dropna=False).loc[lambda s: s > 1]
            if len(disagreement) > 0:
                offenders = list(disagreement.index)[:5]
                raise ValueError(
                    f"split_dataset cannot partition by element when stratify column "
                    f"{stratify_by_column!r} has disagreeing values for the same "
                    f"{element_table or 'element'} RID. This usually means you're "
                    "stratifying on a multi-annotator feature without consensus "
                    "resolution.\n\n"
                    "The deriva-ml pattern for this is a separate consensus feature "
                    "that records the resolved label per element (e.g., "
                    f"'{element_table or 'Element'}_Classification_Consensus' written "
                    "by your adjudication workflow). Stratify on the consensus "
                    "feature, not the raw annotator feature.\n\n"
                    "Alternatively, pass partition_by='row' if you intend per-row "
                    "partitioning and accept that the same "
                    f"{element_table or 'element'} RID may appear in multiple "
                    "partitions.\n\n"
                    f"Offending RIDs (first {len(offenders)} of "
                    f"{len(disagreement)}): {offenders}"
                )

    # Stable, seed-deterministic dedupe. mergesort preserves the
    # original row order within equal sort keys, and keep="first"
    # then picks the first encountered occurrence. The result is
    # deterministic for a given input df.
    deduped = df.sort_values(by=rid_column, kind="mergesort").drop_duplicates(subset=[rid_column], keep="first")
    return deduped.reset_index(drop=True)


def _compute_partitions(
    *,
    source_ds: "Dataset",
    source_dataset_rid: str,
    element_table: str | None,
    test_size: float | int,
    train_size: float | int | None,
    val_size: float | int | None,
    shuffle: bool,
    seed: int,
    stratify_by_column: str | None,
    stratify_missing: str,
    include_tables: list[str] | None,
    selection_fn: SelectionFunction | None,
    row_per: str | None,
    via: list[str] | None,
    ignore_unrelated_anchors: bool,
    partition_by: Literal["element", "row"] = "element",
) -> tuple[
    dict[str, list[str]],
    dict[str, int],
    str,
    str,
]:
    """Resolve the source dataset members into per-partition RID lists.

    Pure read path — no catalog writes. Suitable for dry-run mode
    (the caller skips :func:`_create_split_hierarchy` and builds
    the dry-run :class:`SplitResult` directly from the outputs of
    this helper).

    Steps:

    1. Look up the source dataset and list its members.
    2. Auto-detect the element table when ``element_table is
       None`` (single-candidate-table check).
    3. Validate that ``element_table`` has members.
    4. Resolve absolute sizes via :func:`_resolve_sizes`.
    5. Build partition RID lists — either via the
       denormalization + selector path (stratified / custom) or
       the random-shuffle path.
    6. Compute the strategy description string.

    Args:
        source_ds: The looked-up source :class:`Dataset`.
        source_dataset_rid: The source RID. Used in error
            messages.
        element_table: Caller-specified element table (or
            ``None`` for auto-detect).
        test_size, train_size, val_size, shuffle, seed,
            stratify_by_column, stratify_missing, include_tables,
            selection_fn, row_per, via,
            ignore_unrelated_anchors: As passed to
            :func:`split_dataset`.
        partition_by: Effective partition unit after defaulting in
            :func:`_validate_split_inputs`. ``"element"`` (default)
            dedupes the denormalized dataframe to one row per
            ``element_table`` RID before partitioning, enforces
            within-element agreement on selector-read columns, and
            asserts element-RID disjointness at the end.
            ``"row"`` partitions denormalized rows directly with no
            dedupe — element RIDs may legitimately appear in
            multiple partitions.

    Returns:
        Four-tuple ``(partition_rids, partition_sizes,
        strategy_desc, element_table)``:

        - ``partition_rids`` — ``{name: [rid, ...]}`` for each
          partition.
        - ``partition_sizes`` — ``{name: int}`` — total RID count
          per partition. Includes ``"Validation"`` only when
          ``val_size`` is set.
        - ``strategy_desc`` — human-readable strategy
          (``"random"``, ``"stratified by ..."``, ``"custom
          selection function"``).
        - ``element_table`` — the resolved element table name
          (may differ from the input when auto-detected).

    Raises:
        ValueError: If the source has no members, multiple
            candidate tables when no ``element_table`` is given,
            or the chosen ``element_table`` has no members.
    """
    logger.info("Listing dataset members...")
    members = source_ds.list_dataset_members(recurse=True)

    if element_table is None:
        candidate_tables = [
            table_name for table_name, records in members.items() if table_name != "Dataset" and len(records) > 0
        ]
        if not candidate_tables:
            raise ValueError(f"Source dataset {source_dataset_rid} has no members. Cannot split an empty dataset.")
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

    partition_sizes = _resolve_sizes(total, test_size, train_size, val_size)
    size_summary = ", ".join(f"{k}={v}" for k, v in partition_sizes.items())
    logger.info(f"Split sizes: {size_summary} (total={total})")

    use_denormalization = stratify_by_column is not None or selection_fn is not None

    if use_denormalization:
        # Default row_per to the element table when stratifying, so
        # the natural "one row per element" cardinality lines up with
        # how the partitions are built downstream (RID lookups happen
        # via the {element_table}.RID column). Callers who want a
        # different row_per (e.g., one row per feature value when the
        # feature-assoc table is in include_tables) can pass it
        # explicitly. See issue #174 for the motivating case.
        effective_row_per = row_per if row_per is not None else element_table
        logger.info(
            f"Denormalizing dataset with tables: {include_tables} "
            f"(row_per={effective_row_per}, via={via or []}, "
            f"partition_by={partition_by!r})"
        )
        df = source_ds.get_denormalized_as_dataframe(
            include_tables,
            row_per=effective_row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )
        logger.info(f"Denormalized DataFrame: {len(df)} rows, {len(df.columns)} columns")

        # Resolve the element_table RID column once — both the
        # element-level dedupe and the partition-RID extraction need it.
        rid_column = f"{element_table}.RID"
        if rid_column not in df.columns:
            rid_column = "RID"
            if rid_column not in df.columns:
                raise ValueError(
                    f"Cannot find RID column. Tried '{element_table}.RID' and 'RID'. "
                    f"Available columns: {list(df.columns)}"
                )

        if partition_by == "element":
            # Element-mode contract: the selector sees one row per
            # element_table RID. Two steps:
            #
            # 1. Verify within-element agreement on every column the
            #    selector will read. For stratified splits that's
            #    just stratify_by_column; for selection_fn the read
            #    set is opaque, so we skip and document that the
            #    caller is responsible.
            # 2. Dedupe to one row per element_table RID with stable
            #    seed-deterministic ordering — sort by element_RID
            #    then drop_duplicates(keep="first").
            df = _dedupe_for_element_partition(
                df,
                rid_column=rid_column,
                element_table=element_table,
                stratify_by_column=stratify_by_column,
                stratify_missing=stratify_missing,
                using_selection_fn=selection_fn is not None,
            )
            logger.info(f"After element-level dedupe: {len(df)} rows")
            # Re-resolve partition sizes against the deduped row count.
            partition_sizes = _resolve_sizes(len(df), test_size, train_size, val_size)
            size_summary = ", ".join(f"{k}={v}" for k, v in partition_sizes.items())
            logger.info(f"Adjusted split sizes after dedupe: {size_summary} (total={len(df)})")

        if stratify_by_column:
            logger.info(f"Using stratified split on column: {stratify_by_column}")
            selector = stratified_split(stratify_by_column, missing=stratify_missing)
        else:
            logger.info("Using custom selection function")
            selector = selection_fn

        partition_indices = selector(df, partition_sizes, seed)

        partition_rids = {name: df.iloc[indices][rid_column].tolist() for name, indices in partition_indices.items()}

        if partition_by == "element":
            # Defensive invariant — after a correct dedupe the
            # selector sees disjoint rows (one per element_RID), so
            # mapping indices → RIDs preserves disjointness. If this
            # ever fires, the dedupe logic regressed or a selector
            # ignored its input contract; either case is a bug, not
            # bad user input — hence ``assert``, not ``ValueError``.
            seen: dict[str, str] = {}
            for name, rids in partition_rids.items():
                for r in rids:
                    assert r not in seen, (
                        "partition_by='element' disjointness invariant violated: "
                        f"RID {r!r} appears in both {seen.get(r)!r} and {name!r}. "
                        "Internal correctness bug — the element-level dedupe "
                        "failed to produce one row per element_table RID, or "
                        "the selector returned overlapping indices."
                    )
                    seen[r] = name
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

    # Strategy description.
    if selection_fn:
        strategy_desc = "custom selection function"
    elif stratify_by_column:
        strategy_desc = f"stratified by {stratify_by_column}"
    else:
        strategy_desc = "random"

    return partition_rids, partition_sizes, strategy_desc, element_table


def _create_split_hierarchy(
    *,
    ml: DerivaML,
    execution: Execution,
    source_dataset_rid: str,
    partition_rids: dict[str, list[str]],
    partition_sizes: dict[str, int],
    strategy_desc: str,
    element_table: str,
    seed: int,
    split_description: str,
    training_types: list[str] | None,
    testing_types: list[str] | None,
    validation_types: list[str] | None,
    val_size: float | int | None,
    split_params: dict[str, Any],
) -> SplitResult:
    """Create the parent Split + child Training/Testing(/Validation) datasets.

    Catalog-writing path — invoked only when ``dry_run=False``.
    Pre-extraction this was the second half of
    :func:`split_dataset` (~140 LOC).

    Args:
        ml: Connected :class:`DerivaML`.
        execution: Live :class:`Execution` owning the split.
        source_dataset_rid: Source dataset (recorded in the
            auto-description).
        partition_rids: Per-partition RID lists from
            :func:`_compute_partitions`.
        partition_sizes: Per-partition counts.
        strategy_desc: Strategy string for the description.
        element_table: Element table the partitions came from.
        seed: Seed value (recorded in descriptions + result).
        split_description: Caller's description (or empty, in
            which case ``auto_description`` is used).
        training_types, testing_types, validation_types: Extra
            dataset types beyond the built-in
            ``"Training"`` / ``"Testing"`` / ``"Validation"``.
        val_size: Whether to create the Validation child. The
            value itself is only checked for ``is None``.
        split_params: Full parameter dict written as the
            ``split_config.json`` artifact.

    Returns:
        :class:`SplitResult` with the newly-created RIDs and
        their current versions.
    """
    partitions_desc = ", ".join(f"{k}={v}" for k, v in partition_sizes.items())
    auto_description = f"Split of dataset {source_dataset_rid} ({strategy_desc}, {partitions_desc}, seed={seed})"

    logger.info("Splitting inside caller's execution %s", execution.execution_rid)

    train_types = ["Training"] + (training_types or [])
    test_types = ["Testing"] + (testing_types or [])
    val_types = ["Validation"] + (validation_types or []) if val_size is not None else []

    # Save split parameters as config artifact. The caller's execution
    # is responsible for uploading this on its own
    # ``commit_output_assets``; we never call upload here.
    params_file = Path(execution.working_dir) / "split_config.json"
    params_file.write_text(json.dumps(split_params, indent=2))
    logger.info(f"  Saved split parameters to {params_file}")

    # Create parent Split dataset.
    split_ds = execution.create_dataset(
        description=split_description or auto_description,
        dataset_types=["Split"],
    )
    logger.info(f"  Created Split dataset: {split_ds.dataset_rid}")

    # Create Training dataset.
    training_ds = execution.create_dataset(
        description=(
            f"Training subset ({partition_sizes['Training']} samples) of "
            f"{source_dataset_rid} ({strategy_desc}, seed={seed})"
        ),
        dataset_types=train_types,
    )
    logger.info(f"  Created Training dataset: {training_ds.dataset_rid}")

    # Create Validation dataset (if requested).
    validation_ds = None
    if val_size is not None:
        validation_ds = execution.create_dataset(
            description=(
                f"Validation subset ({partition_sizes['Validation']} samples) of "
                f"{source_dataset_rid} ({strategy_desc}, seed={seed})"
            ),
            dataset_types=val_types,
        )
        logger.info(f"  Created Validation dataset: {validation_ds.dataset_rid}")

    # Create Testing dataset.
    testing_ds = execution.create_dataset(
        description=(
            f"Testing subset ({partition_sizes['Testing']} samples) of "
            f"{source_dataset_rid} ({strategy_desc}, seed={seed})"
        ),
        dataset_types=test_types,
    )
    logger.info(f"  Created Testing dataset: {testing_ds.dataset_rid}")

    # Link children to parent.
    child_rids = [training_ds.dataset_rid, testing_ds.dataset_rid]
    if validation_ds is not None:
        child_rids.insert(1, validation_ds.dataset_rid)
    split_ds.add_dataset_members(child_rids, validate=False)
    logger.info("  Linked child datasets to Split dataset")

    # Record the source dataset as an INPUT consumed by this execution.
    # The split datasets above are OUTPUTS (this execution authored
    # them); the source is not nested under them and is not a
    # ``Dataset_Dataset`` parent of the Split. Recording the source as
    # an execution input is what makes the derivation walkable —
    # ``execution.list_input_datasets()`` and lineage walks can then
    # reach the source from the split, and vice versa — without the bag
    # download that declaring it in ``ExecutionConfiguration.datasets``
    # would force. The call is idempotent and a no-op under dry-run
    # (this helper only runs on the live path, but the guard is there
    # regardless).
    execution.add_input_dataset(source_dataset_rid)
    logger.info("  Recorded source dataset %s as execution input", source_dataset_rid)

    # Add members to each partition (batched).
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

    # Build result.
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


def split_dataset(
    ml: DerivaML,
    source_dataset_rid: str,
    execution: Execution,
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
    dry_run: bool = False,
    # Denormalization-control parameters (issue #174)
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
    # Partition-unit selector
    partition_by: Literal["element", "row"] | None = None,
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

    Provenance — the source dataset's relationship to the split:
        The new Split is a **standalone, self-contained** dataset
        hierarchy. The ``source_dataset_rid`` you pass in is **NOT**
        a parent of the Split and the Split is **NOT** nested under
        the source: there is no ``Dataset_Dataset`` edge between them,
        and ``source.list_dataset_children()`` /
        ``list_dataset_relations(source)`` will **not** list the Split.
        That is intentional — the source is an *input* the split
        *consumed*, not a container the split lives inside (nesting
        the Split under the source would re-partition the source's own
        members and flip the source's version on every split).

        The derivation is instead recorded as **execution provenance**:
        ``split_dataset`` registers ``source_dataset_rid`` as an input
        of ``execution`` (via :meth:`Execution.add_input_dataset`), and
        the Split / Training / Testing / Validation datasets as that
        execution's outputs. So the walkable path is
        ``source -> (input of) -> execution -> (output) -> split``:
        ``execution.list_input_datasets()`` returns the source, and a
        lineage walk (``deriva_ml_get_lineage``) reaches the splits
        from the source and vice versa. The ``SplitResult.source``
        field returned by this call also carries the source RID for
        immediate use.

        Membership consequence: the Training / Testing / Validation
        partitions are carved from the source's elements, so they
        **share element rows with the source** (and, in a two-way
        split, ``Training`` ∪ ``Testing`` reconstructs the source's
        element set). The train/eval relationship therefore lives in
        *shared membership*, not in a parent/child lineage edge —
        evaluating a model trained on the source against one of these
        partitions would leak. Reason about overlap via member sets,
        not via the dataset hierarchy.

    Args:
        ml: Connected DerivaML instance.
        source_dataset_rid: RID of the source dataset to split.
        execution: A live :class:`Execution` the caller has already
            opened (typically via ``with ml.create_execution(config) as
            exe:``). All datasets created by this split — the parent
            Split row and the Training / Validation / Testing children —
            are attributed to *this* execution, which in turn is
            attributed to the execution's workflow. The caller owns
            execution provenance: their workflow URL and checksum
            identify the code making the splitting decision, and
            deriva-ml never invents a workflow on the caller's behalf.
            The caller is responsible for committing the execution
            (``exe.commit_output_assets()`` / context-manager exit).
            ``split_dataset`` will write a ``split_config.json``
            artifact into ``exe.working_dir`` that the caller's upload
            will pick up.
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
            Must be a column in the denormalized DataFrame using dot notation
            (e.g., ``Image_Class.Name``). Use
            :meth:`Dataset.list_denormalized_columns` to discover available columns.
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
        dry_run: If True, return what would happen without modifying catalog.
        row_per: Explicit leaf table for denormalization (passed
            through to :meth:`Dataset.get_denormalized_as_dataframe`).
            When ``stratify_by_column`` or ``selection_fn`` is set and
            ``row_per`` is None, defaults to ``element_table`` — the
            natural anchor when partitioning element rows. Set
            explicitly to override (e.g., when projecting a feature
            value table's columns through a feature-association
            bridge and you want one row per feature value). When
            ``row_per != element_table`` the partition unit becomes
            ambiguous; ``partition_by`` must then be set explicitly.
        via: Tables forced into the join chain without contributing
            columns (denormalizer ``via=`` parameter). Useful to
            disambiguate path ambiguity (Rule 6) without polluting
            the output column list.
        ignore_unrelated_anchors: If True, silently drop dataset
            anchors whose table has no FK path to any requested
            table. Pass-through to the denormalizer (Rule 8) — useful
            when the source dataset has heterogeneous member tables
            and only a subset participates in the split.
        partition_by: Explicit declaration of the partition unit
            when ``row_per`` is set and differs from ``element_table``.
            Either ``"element"`` (one element_table RID per
            partition; dedupe rows before partitioning; enforces
            within-element agreement on the stratify column) or
            ``"row"`` (one denormalized row per partition; element
            RIDs may legitimately appear in multiple partitions).
            Auto-defaults to ``"element"`` when ``row_per`` is
            ``None`` or equals ``element_table`` (the unambiguous
            case). Required — no default — when ``row_per`` is set
            and differs from ``element_table``. See the
            "When to use ``partition_by='element'`` vs
            ``partition_by='row'``" section below.

    When to use ``partition_by='element'`` vs ``partition_by='row'``:
        The (``row_per``, ``element_table``) pair encodes two
        independent choices that the old API conflated:

        - ``element_table`` — what catalog entity does each partition
          collect (Image, Subject, Trial, ...).
        - ``row_per`` — how does the denormalized dataframe shape
          its rows (one per element_table RID, one per
          feature-value, one per visit, ...).

        When ``row_per`` equals ``element_table`` (or is unset) the
        two intents collapse: one element RID = one row, the
        selector partitions rows, and the resulting partitions are
        naturally disjoint at the element level. This is the
        unambiguous case and ``partition_by`` auto-defaults to
        ``"element"``.

        When ``row_per`` differs from ``element_table`` the same
        element RID can have multiple denormalized rows (the 1:N
        feature case). The selector now faces a real architectural
        choice the caller must make explicitly:

        ``partition_by="element"`` — partition the *elements*. The
        dataframe is deduplicated to one row per element_table RID
        before the selector runs. Partitions are guaranteed
        disjoint at the element-RID level. Use this when downstream
        consumers (training loaders, ROC analysis, accuracy
        metrics) operate at the element level — every reasonable ML
        evaluation does. Requires within-element agreement on any
        selector-read column: stratifying on
        ``Image_Classification.Image_Class`` only makes sense if
        every Image_RID has one class. When multiple annotators
        disagree per image, resolve them upstream (the deriva-ml
        pattern is a separate consensus feature that records the
        resolved label per element, written by your adjudication
        workflow) and stratify on the consensus feature, not on
        the raw annotator rows. ``split_dataset`` enforces this
        with a within-element uniformity check that names the
        offending RIDs.

        ``partition_by="row"`` — partition the *rows*. No dedupe,
        no uniformity check. Element RIDs may appear in multiple
        partitions; this is the expected shape for legitimate
        per-row use cases such as per-annotation statistics (each
        annotator-image pair scored independently) or time-series
        splits within a subject. The caller is responsible for
        ensuring partition disjointness at whatever granularity
        downstream consumers actually need.

        Migration note: callers that previously relied on the
        implicit-row-partition behavior of
        ``row_per=<feature_table>`` get a ``ValueError`` at the
        call site directing them to choose. Adding
        ``partition_by="row"`` restores the prior behavior;
        ``partition_by="element"`` switches to the safer
        per-element semantics (and almost always what the caller
        meant).

    Returns:
        SplitResult with partition info for split, training, testing,
        and optionally validation datasets.

    Raises:
        ValueError: If sizes are invalid, dataset has no members, or
            parameters conflict.

    Example:
        ``split_dataset`` always runs inside an Execution the caller has
        already opened — the ``execution`` argument is required. Every
        example below assumes ``exe`` is the live execution from::

            from deriva_ml import DerivaML
            from deriva_ml.dataset.split import split_dataset
            from deriva_ml.execution import ExecutionConfiguration

            ml = DerivaML("localhost", "9")
            workflow = ml.create_workflow(
                name="My splitting script",
                workflow_type="Dataset_Split",
            )
            config = ExecutionConfiguration(workflow=workflow)

        Simple random 80/20 split::

            with ml.create_execution(config) as exe:
                result = split_dataset(ml, "28D0", exe, test_size=0.2, seed=42)
            print(f"Training: {result.training.rid} ({result.training.count} samples)")
            print(f"Testing:  {result.testing.rid} ({result.testing.count} samples)")

        Three-way train/val/test split::

            result = split_dataset(
                ml, "28D0", exe,
                test_size=0.2,
                val_size=0.1,
                seed=42,
            )
            print(f"Validation: {result.validation.rid} ({result.validation.count} samples)")

        Fixed-count split with labeled types::

            result = split_dataset(
                ml, "28D0", exe,
                test_size=100,
                train_size=400,
                seed=42,
                training_types=["Labeled"],
                testing_types=["Labeled"],
            )

        Stratified split preserving class distribution (one row per
        Image, projecting the Image_Class vocab term as a column)::

            # Image and Image_Class are linked by the feature-
            # association table Execution_Image_Image_Classification,
            # which is a transparent bridge for the denormalizer.
            # Pass the **vocab/value table** (``Image_Class``) in
            # ``include_tables``, not the feature-name shorthand
            # (``Image_Classification``): the shorthand resolves to
            # the feature-association table, which is downstream of
            # Image and would trip Rule 5 against the auto-defaulted
            # ``row_per="Image"``. Stratify on the dotted column
            # against the vocab table.
            result = split_dataset(
                ml, "28D0", exe,
                test_size=0.2,
                stratify_by_column="Image_Class.Name",
                include_tables=["Image", "Image_Class"],
                element_table="Image",
                partition_by="element",
            )

        Override ``row_per`` to project one row per feature value
        instead — *per-annotation* statistics. Because ``row_per``
        differs from ``element_table``, ``partition_by`` must be set
        explicitly. ``"row"`` accepts that the same Image RID may
        appear in multiple partitions (its multiple annotation
        rows can land independently); ``"element"`` would dedupe
        to one row per Image before partitioning and would raise
        if annotators disagreed::

            # Per-annotation statistics — element RIDs may legitimately
            # appear in multiple partitions because each annotator-image
            # pair is its own observation. The feature-name shorthand
            # ``Image_Classification`` resolves to the feature-
            # association table; setting ``row_per`` to that table
            # explicitly makes the per-observation intent visible.
            # Stratify on the FK column on the feature-association
            # table (the resolver does not pull the vocab table into
            # the join when the shorthand is used with an explicit
            # feature-assoc ``row_per``).
            result = split_dataset(
                ml, "28D0", exe,
                test_size=0.2,
                stratify_by_column="Execution_Image_Image_Classification.Image_Class",
                include_tables=["Image", "Image_Classification"],
                row_per="Execution_Image_Image_Classification",
                partition_by="row",
            )

        Note: to get "one row per element with a feature value
        projected as a column," pass the vocab/value table in
        ``include_tables`` (as in the first stratified example
        above), not the feature-name shorthand. Rule 5 of the
        denormalizer rejects the shorthand combined with
        ``row_per=<element>`` because the feature-association table
        the shorthand resolves to is strictly downstream of the
        element — aggregation is not supported. To partition by
        feature *observation* instead (per-annotation statistics),
        use the shorthand together with an explicit
        ``row_per=<feature-assoc-table>`` and ``partition_by="row"``
        as in the second example above.

        Stratified split dropping rows with missing labels::

            result = split_dataset(
                ml, "28D0", exe,
                test_size=0.2,
                stratify_by_column="Image_Class.Name",
                stratify_missing="drop",
                include_tables=["Image", "Image_Class"],
                element_table="Image",
                partition_by="element",
            )

        Custom selection function for balanced sampling::

            import numpy as np

            def balanced_selector(df, partition_sizes, seed):
                rng = np.random.default_rng(seed)
                label_col = "Image_Class.Name"
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
                ml, "28D0", exe,
                test_size=100,
                selection_fn=balanced_selector,
                include_tables=["Image", "Image_Class"],
                element_table="Image",
                partition_by="element",
            )

        Dry run to preview the split plan without modifying the catalog::

            result = split_dataset(
                ml, "28D0", exe,
                test_size=0.2,
                dry_run=True,
            )
            print(f"Would create: {result.training.count} train, "
                  f"{result.testing.count} test")

        Use returned RIDs to create a hydra-zen configuration::

            from deriva_ml.dataset import DatasetSpecConfig

            result = split_dataset(ml, "28D0", exe, test_size=0.2, seed=42)
            split_config = DatasetSpecConfig(
                rid=result.split.rid,
                version=result.split.version,
            )

        Train directly from the split partitions (composition with
        framework adapters)::

            result = split_dataset(ml, "28D0", exe, test_size=0.2, seed=42)
            train_bag = ml.lookup_dataset(result.training.rid).download_dataset_bag(
                version=result.training.version
            )
            test_bag = ml.lookup_dataset(result.testing.rid).download_dataset_bag(
                version=result.testing.version
            )
            train_ds = train_bag.as_torch_dataset(
                element_type="Image",
                sample_loader=PIL.Image.open,
                targets=["Glaucoma_Grade"],
            )
            # Each partition bag feeds independently into PyTorch / TensorFlow;
            # the split hierarchy IS the train/val/test partitioning.

    See Also:
        ``DatasetBag.as_torch_dataset``, ``DatasetBag.as_tf_dataset``:
            Build framework-native datasets from any partition bag; same
            ``targets`` / ``target_transform`` / ``missing`` vocabulary.
        ``DatasetBag.restructure_assets``:
            Class-folder layout for ``ImageFolder``-style consumers.
    """
    # Post Ds-split extraction this function dispatches to three
    # helpers (above):
    #
    # 1. ``_validate_split_inputs`` — argument-shape checks.
    # 2. ``_compute_partitions`` — pure read path (members, sizes,
    #    selection); used by both dry-run and live paths.
    # 3. ``_create_split_hierarchy`` — catalog-writing path
    #    (parent/child datasets, member assignment).

    effective_partition_by = _validate_split_inputs(
        stratify_by_column=stratify_by_column,
        selection_fn=selection_fn,
        include_tables=include_tables,
        row_per=row_per,
        element_table=element_table,
        partition_by=partition_by,
    )

    logger.info(f"Looking up source dataset: {source_dataset_rid}")
    source_ds = ml.lookup_dataset(source_dataset_rid)

    partition_rids, partition_sizes, strategy_desc, element_table = _compute_partitions(
        source_ds=source_ds,
        source_dataset_rid=source_dataset_rid,
        element_table=element_table,
        test_size=test_size,
        train_size=train_size,
        val_size=val_size,
        shuffle=shuffle,
        seed=seed,
        stratify_by_column=stratify_by_column,
        stratify_missing=stratify_missing,
        include_tables=include_tables,
        selection_fn=selection_fn,
        row_per=row_per,
        via=via,
        ignore_unrelated_anchors=ignore_unrelated_anchors,
        partition_by=effective_partition_by,
    )

    # Dry-run early return — no catalog writes.
    if dry_run:
        return SplitResult(
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

    # Ensure dataset-type vocabulary terms exist (Training, Testing,
    # Validation, Split, Labeled, Unlabeled). Workflow-type vocabulary is
    # the caller's concern — they registered the workflow that owns
    # ``execution`` and chose its type.
    _ensure_dataset_types(ml)

    train_types = ["Training"] + (training_types or [])
    test_types = ["Testing"] + (testing_types or [])
    val_types = ["Validation"] + (validation_types or []) if val_size is not None else []

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
        "row_per": row_per,
        "via": via,
        "ignore_unrelated_anchors": ignore_unrelated_anchors,
        "partition_by": effective_partition_by,
        "training_types": train_types,
        "testing_types": test_types,
        "validation_types": val_types if val_types else None,
        "strategy": strategy_desc,
    }

    return _create_split_hierarchy(
        ml=ml,
        execution=execution,
        source_dataset_rid=source_dataset_rid,
        partition_rids=partition_rids,
        partition_sizes=partition_sizes,
        strategy_desc=strategy_desc,
        element_table=element_table,
        seed=seed,
        split_description=split_description,
        training_types=training_types,
        testing_types=testing_types,
        validation_types=validation_types,
        val_size=val_size,
        split_params=split_params,
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

            # Stratified split by class label (stratify on the vocab
            # table's Name column, reached transparently through the
            # Image_Classification feature)
            deriva-ml-split-dataset --hostname localhost --catalog-id 9 \\
                --dataset-rid 28D0 \\
                --stratify-by-column Image_Class.Name \\
                --include-tables Image,Image_Class

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
        "--hostname",
        required=True,
        help="Deriva server hostname (e.g., localhost, ml.derivacloud.org)",
    )
    parser.add_argument(
        "--catalog-id",
        required=True,
        help="Catalog ID to connect to",
    )
    parser.add_argument(
        "--domain-schema",
        help="Domain schema name (auto-detected if not provided)",
    )

    # Source dataset
    parser.add_argument(
        "--dataset-rid",
        required=True,
        help="RID of the source dataset to split",
    )

    # Split parameters (scikit-learn conventions)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size as fraction (0-1) or absolute count (default: 0.2)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=None,
        help="Train set size as fraction (0-1) or absolute count (default: complement of test-size)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help="Validation set size as fraction (0-1) or absolute count (default: None, no validation split)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle before splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--stratify-by-column",
        help="Column name in denormalized DataFrame (dot notation) for stratified "
        "splitting (e.g., Image_Class.Name). Requires --include-tables.",
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
        help="Comma-separated tables for denormalization (e.g., Image,Image_Class). Required for stratified splitting.",
    )
    parser.add_argument(
        "--row-per",
        help="Explicit leaf table for denormalization. Defaults to --element-table when stratifying (issue #174).",
    )
    parser.add_argument(
        "--via",
        help="Comma-separated tables forced into the join chain without "
        "contributing columns (denormalizer via= parameter). Use to "
        "disambiguate path ambiguity (Rule 6) without polluting output.",
    )
    parser.add_argument(
        "--ignore-unrelated-anchors",
        action="store_true",
        help="Silently drop dataset anchors with no FK path to any requested table (denormalizer Rule 8 escape hatch).",
    )
    parser.add_argument(
        "--partition-by",
        choices=["element", "row"],
        default=None,
        help="Partition unit: 'element' dedupes per element_table RID before "
        "partitioning (disjoint at the element level); 'row' partitions "
        "denormalized rows directly (element RIDs may overlap). Required when "
        "--row-per is set and differs from --element-table; auto-defaults to "
        "'element' otherwise.",
    )
    parser.add_argument(
        "--training-types",
        default="Labeled",
        help="Comma-separated additional dataset types for training set (default: Labeled)",
    )
    parser.add_argument(
        "--testing-types",
        default="Labeled",
        help="Comma-separated additional dataset types for testing set (default: Labeled)",
    )
    parser.add_argument(
        "--validation-types",
        default="Labeled",
        help="Comma-separated additional dataset types for validation set (default: Labeled)",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Description for the parent split dataset",
    )
    parser.add_argument(
        "--workflow-type",
        default="Dataset_Split",
        help="Workflow type vocabulary term (default: Dataset_Split)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without modifying catalog",
    )
    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Show Chaise web interface URLs for created datasets",
    )

    args = parser.parse_args()

    # Configure logging
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger = get_logger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    try:
        from deriva_ml import DerivaML
        from deriva_ml.execution import ExecutionConfiguration

        # Connect
        logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(args.catalog_id),
            domain_schemas={args.domain_schema} if args.domain_schema else None,
        )
        logger.info(f"Connected, domain schema: {ml.default_schema}")

        # Parse comma-separated lists
        include_tables = [t.strip() for t in args.include_tables.split(",")] if args.include_tables else None
        training_types = [t.strip() for t in args.training_types.split(",")] if args.training_types else None
        testing_types = [t.strip() for t in args.testing_types.split(",")] if args.testing_types else None
        validation_types = [t.strip() for t in args.validation_types.split(",")] if args.validation_types else None
        via = [t.strip() for t in args.via.split(",")] if args.via else None

        # Dry-run: skip workflow/execution overhead entirely. split_dataset's
        # dry-run path doesn't touch the catalog and doesn't need a live
        # execution -- pass a sentinel so the type-check is satisfied and the
        # early-return at the top of split_dataset fires before any execution
        # methods are called.
        if args.dry_run:
            result = split_dataset(
                ml=ml,
                source_dataset_rid=args.dataset_rid,
                execution=None,  # type: ignore[arg-type]  -- dry-run returns before use
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
                row_per=args.row_per,
                via=via,
                ignore_unrelated_anchors=args.ignore_unrelated_anchors,
                partition_by=args.partition_by,
                dry_run=True,
            )
        else:
            # The CLI itself is the caller -- it lives in a git checkout
            # of deriva-ml, so its workflow URL/checksum come from this
            # script's git context (via the Workflow validator's
            # built-in introspection). The MCP server, by contrast,
            # would never reach this code path -- it opens its own
            # execution from a caller-supplied workflow_rid.
            workflow = ml.create_workflow(
                name=f"deriva-ml-split-dataset CLI: {args.dataset_rid}",
                workflow_type=args.workflow_type,
                description="Split dataset via the deriva-ml-split-dataset CLI",
            )
            with ml.create_execution(
                ExecutionConfiguration(
                    workflow=workflow,
                    description=args.description or f"Split of {args.dataset_rid}",
                )
            ) as exe:
                result = split_dataset(
                    ml=ml,
                    source_dataset_rid=args.dataset_rid,
                    execution=exe,
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
                    row_per=args.row_per,
                    via=via,
                    ignore_unrelated_anchors=args.ignore_unrelated_anchors,
                    dry_run=False,
                )
            exe.commit_output_assets(clean_folder=True)

        # Print summary
        if args.dry_run:
            print(f"\n{'=' * 60}")
            print("  DRY RUN - No changes will be made")
            print(f"{'=' * 60}")
            print(f"  Source dataset:  {result.source}")
            print(f"  Element table:   {result.element_table}")
            print(f"  Strategy:        {result.strategy}")
            print(f"  Seed:            {result.seed}")
            print(f"  Training size:   {result.training.count}")
            if result.validation:
                print(f"  Validation size: {result.validation.count}")
            print(f"  Testing size:    {result.testing.count}")
            print(f"{'=' * 60}\n")
        else:
            print(f"\n{'=' * 60}")
            print("  SPLIT COMPLETE")
            print(f"{'=' * 60}")
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

            print(f"{'=' * 60}\n")

        return 0

    except Exception as e:
        logger.error(f"Split failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

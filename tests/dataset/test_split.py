"""Tests for dataset splitting functionality.

Tests cover:
- _resolve_sizes: size conversion logic
- random_split: deterministic random splitting
- stratified_split: class-distribution-preserving splits
- split_dataset: integration tests with a live catalog
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deriva_ml import DerivaML, MLVocab
from deriva_ml.dataset.split import (
    PartitionInfo,
    SelectionFunction,
    SplitResult,
    _resolve_sizes,
    random_split,
    split_dataset,
    stratified_split,
)
from deriva_ml.execution import ExecutionConfiguration

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")


# =============================================================================
# Unit Tests: _resolve_sizes
# =============================================================================


class TestResolveSizes:
    """Tests for the _resolve_sizes helper function."""

    def test_fraction_test_size(self):
        """Test fractional test_size with complement train_size."""
        result = _resolve_sizes(100, test_size=0.2, train_size=None)
        assert result == {"Training": 80, "Testing": 20}

    def test_integer_test_size(self):
        """Test absolute integer test_size."""
        result = _resolve_sizes(100, test_size=30, train_size=None)
        assert result == {"Training": 70, "Testing": 30}

    def test_both_fractions(self):
        """Test both sizes as fractions."""
        result = _resolve_sizes(100, test_size=0.3, train_size=0.5)
        assert result == {"Training": 50, "Testing": 30}

    def test_both_integers(self):
        """Test both sizes as absolute integers."""
        result = _resolve_sizes(100, test_size=20, train_size=60)
        assert result == {"Training": 60, "Testing": 20}

    def test_fraction_rounding(self):
        """Test that fractional sizes round correctly."""
        result = _resolve_sizes(10, test_size=0.3, train_size=None)
        assert result == {"Training": 7, "Testing": 3}

    def test_exceeds_total_raises(self):
        """Test that sizes exceeding total raise ValueError."""
        with pytest.raises(ValueError, match="exceeds total dataset size"):
            _resolve_sizes(100, test_size=60, train_size=60)

    def test_zero_test_size_raises(self):
        """Test that zero-resulting sizes raise ValueError."""
        with pytest.raises(ValueError):
            _resolve_sizes(100, test_size=0.0, train_size=None)

    def test_negative_test_size_raises(self):
        """Test that negative test_size raises ValueError."""
        with pytest.raises(ValueError):
            _resolve_sizes(100, test_size=-0.1, train_size=None)

    def test_small_dataset(self):
        """Test with a very small dataset."""
        result = _resolve_sizes(2, test_size=1, train_size=1)
        assert result == {"Training": 1, "Testing": 1}

    def test_train_size_complement(self):
        """Test that train_size=None gives the full complement."""
        result = _resolve_sizes(1000, test_size=0.1, train_size=None)
        assert result == {"Training": 900, "Testing": 100}

    def test_three_way_fractions(self):
        """Test three-way split with fractional sizes."""
        result = _resolve_sizes(100, test_size=0.2, val_size=0.1)
        assert result == {"Training": 70, "Testing": 20, "Validation": 10}

    def test_three_way_integers(self):
        """Test three-way split with absolute sizes."""
        result = _resolve_sizes(100, test_size=20, val_size=10, train_size=60)
        assert result == {"Training": 60, "Testing": 20, "Validation": 10}

    def test_three_way_complement(self):
        """Test three-way split with train_size as complement."""
        result = _resolve_sizes(100, test_size=20, val_size=10)
        assert result == {"Training": 70, "Testing": 20, "Validation": 10}

    def test_three_way_exceeds_total_raises(self):
        """Test that three-way sizes exceeding total raise ValueError."""
        with pytest.raises(ValueError, match="exceeds total dataset size"):
            _resolve_sizes(100, test_size=50, val_size=30, train_size=30)

    def test_val_size_zero_raises(self):
        """Test that zero val_size raises ValueError."""
        with pytest.raises(ValueError):
            _resolve_sizes(100, test_size=0.2, val_size=0.0)

    def test_no_val_size_returns_two_partitions(self):
        """Test that val_size=None returns only Training and Testing."""
        result = _resolve_sizes(100, test_size=0.2)
        assert "Validation" not in result
        assert set(result.keys()) == {"Training", "Testing"}

    def test_val_size_returns_three_partitions(self):
        """Test that val_size provided returns three partitions."""
        result = _resolve_sizes(100, test_size=0.2, val_size=0.1)
        assert set(result.keys()) == {"Training", "Testing", "Validation"}


# =============================================================================
# Unit Tests: random_split
# =============================================================================


class TestRandomSplit:
    """Tests for the random_split selection function."""

    def test_correct_sizes(self):
        """Test that output arrays have correct sizes."""
        df = pd.DataFrame({"x": range(100)})
        result = random_split(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        assert len(result["Training"]) == 70
        assert len(result["Testing"]) == 30

    def test_no_overlap(self):
        """Test that train and test indices don't overlap."""
        df = pd.DataFrame({"x": range(100)})
        result = random_split(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        assert len(set(result["Training"]) & set(result["Testing"])) == 0

    def test_deterministic(self):
        """Test that same seed produces same result."""
        df = pd.DataFrame({"x": range(100)})
        r1 = random_split(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        r2 = random_split(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        np.testing.assert_array_equal(r1["Training"], r2["Training"])
        np.testing.assert_array_equal(r1["Testing"], r2["Testing"])

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        df = pd.DataFrame({"x": range(100)})
        r1 = random_split(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        r2 = random_split(df, partition_sizes={"Training": 70, "Testing": 30}, seed=99)
        assert not np.array_equal(r1["Training"], r2["Training"])

    def test_subset_of_total(self):
        """Test splitting a subset of a larger dataset."""
        df = pd.DataFrame({"x": range(100)})
        result = random_split(df, partition_sizes={"Training": 20, "Testing": 10}, seed=42)
        assert len(result["Training"]) == 20
        assert len(result["Testing"]) == 10
        all_indices = np.concatenate([result["Training"], result["Testing"]])
        assert all(0 <= i < 100 for i in all_indices)

    def test_valid_indices(self):
        """Test that all indices are valid DataFrame indices."""
        df = pd.DataFrame({"x": range(50)})
        result = random_split(df, partition_sizes={"Training": 30, "Testing": 20}, seed=42)
        all_idx = np.concatenate([result["Training"], result["Testing"]])
        assert all(0 <= i < len(df) for i in all_idx)

    def test_conforms_to_protocol(self):
        """Test that random_split conforms to SelectionFunction protocol."""
        assert isinstance(random_split, SelectionFunction)

    def test_three_way_split(self):
        """Test three-way random split."""
        df = pd.DataFrame({"x": range(100)})
        result = random_split(
            df,
            partition_sizes={"Training": 60, "Validation": 20, "Testing": 20},
            seed=42,
        )
        assert len(result["Training"]) == 60
        assert len(result["Validation"]) == 20
        assert len(result["Testing"]) == 20

        # No overlap between any pair
        all_sets = [set(result[k]) for k in result]
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                assert len(all_sets[i] & all_sets[j]) == 0

    def test_three_way_valid_indices(self):
        """Test that all indices in three-way split are valid."""
        df = pd.DataFrame({"x": range(50)})
        result = random_split(
            df,
            partition_sizes={"Training": 30, "Validation": 10, "Testing": 10},
            seed=42,
        )
        all_idx = np.concatenate(list(result.values()))
        assert all(0 <= i < len(df) for i in all_idx)


# =============================================================================
# Unit Tests: stratified_split
# =============================================================================


@requires_sklearn
class TestStratifiedSplit:
    """Tests for the stratified_split factory function."""

    def test_correct_sizes(self):
        """Test that output arrays have correct sizes."""
        df = pd.DataFrame({
            "label": ["A"] * 50 + ["B"] * 50,
            "value": range(100),
        })
        selector = stratified_split("label")
        result = selector(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        assert len(result["Training"]) == 70
        assert len(result["Testing"]) == 30

    def test_maintains_distribution(self):
        """Test that class distribution is preserved."""
        df = pd.DataFrame({
            "label": ["A"] * 60 + ["B"] * 40,
            "value": range(100),
        })
        selector = stratified_split("label")
        result = selector(df, partition_sizes={"Training": 80, "Testing": 20}, seed=42)

        train_labels = df.iloc[result["Training"]]["label"]
        test_labels = df.iloc[result["Testing"]]["label"]

        train_a_frac = (train_labels == "A").sum() / len(train_labels)
        test_a_frac = (test_labels == "A").sum() / len(test_labels)

        assert abs(train_a_frac - 0.6) < 0.1
        assert abs(test_a_frac - 0.6) < 0.1

    def test_no_overlap(self):
        """Test that train and test indices don't overlap."""
        df = pd.DataFrame({
            "label": ["A"] * 50 + ["B"] * 50,
        })
        selector = stratified_split("label")
        result = selector(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        assert len(set(result["Training"]) & set(result["Testing"])) == 0

    def test_deterministic(self):
        """Test that same seed produces same result."""
        df = pd.DataFrame({
            "label": ["A"] * 50 + ["B"] * 50,
        })
        selector = stratified_split("label")
        r1 = selector(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        r2 = selector(df, partition_sizes={"Training": 70, "Testing": 30}, seed=42)
        np.testing.assert_array_equal(sorted(r1["Training"]), sorted(r2["Training"]))
        np.testing.assert_array_equal(sorted(r1["Testing"]), sorted(r2["Testing"]))

    def test_missing_column_raises(self):
        """Test that missing stratification column raises ValueError."""
        df = pd.DataFrame({"x": range(10)})
        selector = stratified_split("nonexistent")
        with pytest.raises(ValueError, match="not found in denormalized DataFrame"):
            selector(df, partition_sizes={"Training": 7, "Testing": 3}, seed=42)

    def test_exceeds_total_raises(self):
        """Test that requesting more than available raises ValueError."""
        df = pd.DataFrame({"label": ["A"] * 5 + ["B"] * 5})
        selector = stratified_split("label")
        with pytest.raises(ValueError, match="Requested .* samples but dataset has"):
            selector(df, partition_sizes={"Training": 8, "Testing": 5}, seed=42)

    def test_subset_of_total(self):
        """Test stratified split with fewer samples than total."""
        df = pd.DataFrame({
            "label": ["A"] * 50 + ["B"] * 50,
        })
        selector = stratified_split("label")
        result = selector(df, partition_sizes={"Training": 30, "Testing": 10}, seed=42)
        assert len(result["Training"]) == 30
        assert len(result["Testing"]) == 10

    def test_conforms_to_protocol(self):
        """Test that returned function conforms to SelectionFunction protocol."""
        selector = stratified_split("label")
        assert isinstance(selector, SelectionFunction)

    def test_multiclass(self):
        """Test with more than two classes."""
        df = pd.DataFrame({
            "label": ["A"] * 30 + ["B"] * 30 + ["C"] * 30 + ["D"] * 10,
        })
        selector = stratified_split("label")
        result = selector(df, partition_sizes={"Training": 80, "Testing": 20}, seed=42)
        assert len(result["Training"]) == 80
        assert len(result["Testing"]) == 20

        # All classes should be represented in both splits
        train_classes = set(df.iloc[result["Training"]]["label"])
        test_classes = set(df.iloc[result["Testing"]]["label"])
        assert train_classes == {"A", "B", "C", "D"}
        assert test_classes == {"A", "B", "C", "D"}

    def test_three_way_split(self):
        """Test three-way stratified split."""
        df = pd.DataFrame({
            "label": ["A"] * 60 + ["B"] * 40,
        })
        selector = stratified_split("label")
        result = selector(
            df,
            partition_sizes={"Training": 60, "Validation": 20, "Testing": 20},
            seed=42,
        )
        assert len(result["Training"]) == 60
        assert len(result["Validation"]) == 20
        assert len(result["Testing"]) == 20

        # No overlap between any pair
        all_sets = [set(result[k]) for k in result]
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                assert len(all_sets[i] & all_sets[j]) == 0

    def test_three_way_maintains_distribution(self):
        """Test that three-way split preserves class distribution."""
        df = pd.DataFrame({
            "label": ["A"] * 60 + ["B"] * 40,
        })
        selector = stratified_split("label")
        result = selector(
            df,
            partition_sizes={"Training": 60, "Validation": 20, "Testing": 20},
            seed=42,
        )

        for name in ["Training", "Validation", "Testing"]:
            labels = df.iloc[result[name]]["label"]
            a_frac = (labels == "A").sum() / len(labels)
            assert abs(a_frac - 0.6) < 0.15, f"{name} has A fraction {a_frac}, expected ~0.6"


# =============================================================================
# Integration Tests: split_dataset
# =============================================================================


class TestSplitDataset:
    """Integration tests for split_dataset with a live catalog."""

    @staticmethod
    def _setup_splittable_dataset(ml: DerivaML) -> str:
        """Create a dataset with enough members to split.

        Creates a table 'SplitTestItem' with 12 records and a dataset
        containing all of them.

        Returns:
            RID of the created dataset.
        """
        from deriva_ml import BuiltinTypes, ColumnDefinition, TableDefinition

        # Create a table for split testing
        ml.model.create_table(
            TableDefinition(
                name="SplitTestItem",
                columns=[
                    ColumnDefinition(name="Name", type=BuiltinTypes.text),
                    ColumnDefinition(name="Category", type=BuiltinTypes.text),
                ],
            )
        )
        ml.add_dataset_element_type("SplitTestItem")

        # Insert records with categories for stratification testing
        table_path = (
            ml.catalog.getPathBuilder()
            .schemas[ml.default_schema]
            .tables["SplitTestItem"]
        )
        records = [
            {"Name": f"Item{i}", "Category": "A" if i < 6 else "B"}
            for i in range(12)
        ]
        table_path.insert(records)
        item_rids = [r["RID"] for r in table_path.entities().fetch()]

        # Create workflow and dataset
        ml.add_term(MLVocab.workflow_type, "Setup", description="Setup workflow")
        ml.add_term("Dataset_Type", "Source", description="Source dataset")
        workflow = ml.create_workflow(
            name="Setup Workflow",
            workflow_type="Setup",
            description="Creating test data",
        )
        execution = ml.create_execution(
            ExecutionConfiguration(description="Setup", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types=["Source"],
            description="Test dataset for splitting",
        )
        dataset.add_dataset_members({"SplitTestItem": item_rids})

        return dataset.dataset_rid

    def test_basic_random_split(self, test_ml):
        """Test basic 80/20 random split."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=0.25,
            seed=42,
        )

        assert isinstance(result, SplitResult)
        assert result.source == source_rid
        assert result.training.count == 9
        assert result.testing.count == 3
        assert result.validation is None

        # Verify the datasets were actually created
        split_ds = ml.lookup_dataset(result.split.rid)
        assert split_ds is not None
        training_ds = ml.lookup_dataset(result.training.rid)
        assert training_ds is not None
        testing_ds = ml.lookup_dataset(result.testing.rid)
        assert testing_ds is not None

    def test_split_creates_hierarchy(self, test_ml):
        """Test that split creates correct parent-child hierarchy."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(ml, source_rid, test_size=3, seed=42)

        # Verify parent-child relationships
        split_ds = ml.lookup_dataset(result.split.rid)
        children = split_ds.list_dataset_children()
        child_rids = {c.dataset_rid for c in children}
        assert result.training.rid in child_rids
        assert result.testing.rid in child_rids

    def test_split_member_counts(self, test_ml):
        """Test that train and test datasets have correct member counts."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=4,
            train_size=8,
            seed=42,
        )

        # Check training dataset members
        training_ds = ml.lookup_dataset(result.training.rid)
        train_members = training_ds.list_dataset_members()
        assert len(train_members.get("SplitTestItem", [])) == 8

        # Check testing dataset members
        testing_ds = ml.lookup_dataset(result.testing.rid)
        test_members = testing_ds.list_dataset_members()
        assert len(test_members.get("SplitTestItem", [])) == 4

    def test_split_no_overlap(self, test_ml):
        """Test that training and testing sets have no overlapping members."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(ml, source_rid, test_size=4, seed=42)

        training_ds = ml.lookup_dataset(result.training.rid)
        testing_ds = ml.lookup_dataset(result.testing.rid)

        train_rids = {
            r["RID"] for r in training_ds.list_dataset_members().get("SplitTestItem", [])
        }
        test_rids = {
            r["RID"] for r in testing_ds.list_dataset_members().get("SplitTestItem", [])
        }

        assert len(train_rids & test_rids) == 0

    def test_split_deterministic(self, test_ml):
        """Test that same seed produces same member assignments."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result1 = split_dataset(ml, source_rid, test_size=4, seed=42)
        training_ds1 = ml.lookup_dataset(result1.training.rid)
        train_rids1 = {
            r["RID"]
            for r in training_ds1.list_dataset_members().get("SplitTestItem", [])
        }

        # Create another split with same parameters on same source
        result2 = split_dataset(ml, source_rid, test_size=4, seed=42)
        training_ds2 = ml.lookup_dataset(result2.training.rid)
        train_rids2 = {
            r["RID"]
            for r in training_ds2.list_dataset_members().get("SplitTestItem", [])
        }

        assert train_rids1 == train_rids2

    def test_split_dataset_types(self, test_ml):
        """Test that dataset types are applied correctly."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=4,
            seed=42,
            training_types=["Labeled"],
            testing_types=["Labeled"],
        )

        split_ds = ml.lookup_dataset(result.split.rid)
        training_ds = ml.lookup_dataset(result.training.rid)
        testing_ds = ml.lookup_dataset(result.testing.rid)

        assert "Split" in split_ds.dataset_types
        assert "Training" in training_ds.dataset_types
        assert "Testing" in testing_ds.dataset_types
        assert "Labeled" in training_ds.dataset_types
        assert "Labeled" in testing_ds.dataset_types

    def test_dry_run(self, test_ml):
        """Test dry run mode returns plan without modifying catalog."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        initial_datasets = list(ml.find_datasets())
        initial_count = len(initial_datasets)

        result = split_dataset(
            ml, source_rid,
            test_size=4,
            seed=42,
            dry_run=True,
        )

        assert isinstance(result, SplitResult)
        assert result.dry_run is True
        assert result.split.rid == "(dry run)"
        assert result.training.rid == "(dry run)"
        assert result.testing.rid == "(dry run)"
        assert result.source == source_rid
        assert result.training.count == 8
        assert result.testing.count == 4

        # Verify no new datasets were created
        current_datasets = list(ml.find_datasets())
        assert len(current_datasets) == initial_count

    def test_custom_selection_function(self, test_ml):
        """Test split with a custom selection function."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        def first_n_selector(
            df: pd.DataFrame,
            partition_sizes: dict[str, int],
            seed: int,
        ) -> dict[str, np.ndarray]:
            """Deterministic selector: allocate indices in order."""
            indices = np.arange(len(df))
            result = {}
            offset = 0
            for name, size in partition_sizes.items():
                result[name] = indices[offset : offset + size]
                offset += size
            return result

        result = split_dataset(
            ml, source_rid,
            test_size=4,
            seed=42,
            selection_fn=first_n_selector,
            include_tables=["SplitTestItem"],
        )

        assert result.training.count == 8
        assert result.testing.count == 4

    def test_mutually_exclusive_params(self, test_ml):
        """Test that stratify_by_column and selection_fn are mutually exclusive."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        def dummy_fn(df, partition_sizes, seed):
            return {"Training": np.array([0]), "Testing": np.array([1])}

        with pytest.raises(ValueError, match="mutually exclusive"):
            split_dataset(
                ml, source_rid,
                test_size=4,
                stratify_by_column="Category",
                selection_fn=dummy_fn,
                include_tables=["SplitTestItem"],
            )

    def test_stratify_requires_include_tables(self, test_ml):
        """Test that stratify_by_column requires include_tables."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        with pytest.raises(ValueError, match="include_tables is required"):
            split_dataset(
                ml, source_rid,
                test_size=4,
                stratify_by_column="SplitTestItem_Category",
            )

    def test_selection_fn_requires_include_tables(self, test_ml):
        """Test that selection_fn requires include_tables."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        def dummy_fn(df, partition_sizes, seed):
            return {"Training": np.array([0]), "Testing": np.array([1])}

        with pytest.raises(ValueError, match="include_tables is required"):
            split_dataset(
                ml, source_rid,
                test_size=4,
                selection_fn=dummy_fn,
            )

    def test_empty_dataset_raises(self, test_ml):
        """Test that splitting an empty dataset raises ValueError."""
        ml = test_ml

        # Create an empty dataset (no members)
        ml.add_term(MLVocab.workflow_type, "Setup", description="Setup workflow")
        ml.add_term("Dataset_Type", "Source", description="Source dataset")
        workflow = ml.create_workflow(
            name="Setup Workflow",
            workflow_type="Setup",
            description="Setup",
        )
        execution = ml.create_execution(
            ExecutionConfiguration(description="Setup", workflow=workflow)
        )
        empty_ds = execution.create_dataset(
            dataset_types=["Source"],
            description="Empty dataset",
        )

        with pytest.raises(ValueError, match="no members"):
            split_dataset(ml, empty_ds.dataset_rid, test_size=0.2)

    def test_provenance_tracking(self, test_ml):
        """Test that split creates proper execution provenance."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(ml, source_rid, test_size=4, seed=42)

        # The split dataset should have execution history
        split_ds = ml.lookup_dataset(result.split.rid)
        history = split_ds.dataset_history()
        assert len(history) > 0

    def test_versions_returned(self, test_ml):
        """Test that result includes version strings."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(ml, source_rid, test_size=4, seed=42)

        # Versions should be valid semver-like strings
        for info in [result.split, result.training, result.testing]:
            parts = str(info.version).split(".")
            assert len(parts) == 3

    def test_no_shuffle(self, test_ml):
        """Test splitting without shuffle preserves order."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=4,
            shuffle=False,
            seed=42,
        )

        assert result.training.count == 8
        assert result.testing.count == 4

    # =========================================================================
    # Three-way split integration tests
    # =========================================================================

    def test_three_way_split(self, test_ml):
        """Test three-way train/val/test split."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=2,
            val_size=2,
            seed=42,
        )

        assert isinstance(result, SplitResult)
        assert result.training.count == 8
        assert result.validation is not None
        assert result.validation.count == 2
        assert result.testing.count == 2

        # Verify all three datasets were created
        assert ml.lookup_dataset(result.training.rid) is not None
        assert ml.lookup_dataset(result.validation.rid) is not None
        assert ml.lookup_dataset(result.testing.rid) is not None

    def test_three_way_hierarchy(self, test_ml):
        """Test that three-way split creates correct parent-child hierarchy."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=2,
            val_size=2,
            seed=42,
        )

        split_ds = ml.lookup_dataset(result.split.rid)
        children = split_ds.list_dataset_children()
        child_rids = {c.dataset_rid for c in children}
        assert result.training.rid in child_rids
        assert result.validation.rid in child_rids
        assert result.testing.rid in child_rids
        assert len(child_rids) == 3

    def test_three_way_no_overlap(self, test_ml):
        """Test that all three partitions have no overlapping members."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=2,
            val_size=2,
            seed=42,
        )

        train_rids = {
            r["RID"] for r in
            ml.lookup_dataset(result.training.rid).list_dataset_members().get("SplitTestItem", [])
        }
        val_rids = {
            r["RID"] for r in
            ml.lookup_dataset(result.validation.rid).list_dataset_members().get("SplitTestItem", [])
        }
        test_rids = {
            r["RID"] for r in
            ml.lookup_dataset(result.testing.rid).list_dataset_members().get("SplitTestItem", [])
        }

        assert len(train_rids & val_rids) == 0
        assert len(train_rids & test_rids) == 0
        assert len(val_rids & test_rids) == 0

    def test_three_way_types(self, test_ml):
        """Test that three-way split applies correct dataset types."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=2,
            val_size=2,
            seed=42,
            training_types=["Labeled"],
            validation_types=["Labeled"],
            testing_types=["Labeled"],
        )

        training_ds = ml.lookup_dataset(result.training.rid)
        validation_ds = ml.lookup_dataset(result.validation.rid)
        testing_ds = ml.lookup_dataset(result.testing.rid)

        assert "Training" in training_ds.dataset_types
        assert "Labeled" in training_ds.dataset_types
        assert "Validation" in validation_ds.dataset_types
        assert "Labeled" in validation_ds.dataset_types
        assert "Testing" in testing_ds.dataset_types
        assert "Labeled" in testing_ds.dataset_types

    def test_three_way_dry_run(self, test_ml):
        """Test three-way dry run includes validation info."""
        ml = test_ml
        source_rid = self._setup_splittable_dataset(ml)

        result = split_dataset(
            ml, source_rid,
            test_size=2,
            val_size=2,
            seed=42,
            dry_run=True,
        )

        assert result.dry_run is True
        assert result.validation is not None
        assert result.validation.count == 2
        assert result.validation.rid == "(dry run)"

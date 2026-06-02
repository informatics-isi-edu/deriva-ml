"""Integration tests for the ``subsample`` primitive.

Spec 2026-06-01 §6 E — the new TestSubsample suite. Covers the full
``subsample`` contract: happy path, stratified sampling, dry-run,
single-output shape (no parent Split), provenance shape (source as
execution input, no Dataset_Dataset edge), Subsample tag application,
role-types-don't-propagate from the source, ``dataset_types`` argument
honored (with defensive dedupe), and deterministic-by-seed.

Tests use the same ``test_ml`` fixture and the ``SplitTestItem``
fixture builder pattern that the sibling split-dataset suite uses;
they require a live Deriva catalog (``DERIVA_HOST``).
"""

from __future__ import annotations

import pytest

from deriva_ml import DerivaML, MLVocab
from deriva_ml.dataset.split import (
    SubsampleResult,
    _validate_subsample_inputs,
    subsample,
)
from deriva_ml.execution import ExecutionConfiguration

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")


# =============================================================================
# Unit tests — argument validation (no catalog needed)
# =============================================================================


class TestValidateSubsampleInputs:
    """``_validate_subsample_inputs`` argument-shape rules."""

    def test_default_partition_by_element_when_row_per_omitted(self):
        result = _validate_subsample_inputs(
            size=100,
            stratify_by_column=None,
            include_tables=None,
            row_per=None,
            element_table="Image",
            partition_by=None,
        )
        assert result == "element"

    def test_partition_by_required_when_row_per_differs_from_element_table(self):
        with pytest.raises(ValueError) as exc:
            _validate_subsample_inputs(
                size=100,
                stratify_by_column="Image_Class.Name",
                include_tables=["Image", "Image_Class"],
                row_per="Execution_Image_Image_Class",
                element_table="Image",
                partition_by=None,
            )
        msg = str(exc.value)
        assert "partition_by" in msg
        assert "'element'" in msg
        assert "'row'" in msg

    def test_stratify_without_include_tables_raises(self):
        with pytest.raises(ValueError, match="include_tables is required"):
            _validate_subsample_inputs(
                size=100,
                stratify_by_column="Image_Class.Name",
                include_tables=None,
                row_per=None,
                element_table="Image",
                partition_by=None,
            )

    def test_size_negative_raises(self):
        with pytest.raises(ValueError, match="size must be"):
            _validate_subsample_inputs(
                size=-1,
                stratify_by_column=None,
                include_tables=None,
                row_per=None,
                element_table="Image",
                partition_by=None,
            )

    def test_size_zero_raises(self):
        with pytest.raises(ValueError, match="size must be"):
            _validate_subsample_inputs(
                size=0,
                stratify_by_column=None,
                include_tables=None,
                row_per=None,
                element_table="Image",
                partition_by=None,
            )

    def test_size_fraction_at_zero_raises(self):
        with pytest.raises(ValueError, match="size must be"):
            _validate_subsample_inputs(
                size=0.0,
                stratify_by_column=None,
                include_tables=None,
                row_per=None,
                element_table="Image",
                partition_by=None,
            )

    def test_size_fraction_at_one_raises(self):
        with pytest.raises(ValueError, match="size must be"):
            _validate_subsample_inputs(
                size=1.0,
                stratify_by_column=None,
                include_tables=None,
                row_per=None,
                element_table="Image",
                partition_by=None,
            )

    def test_size_fractional_in_range_ok(self):
        result = _validate_subsample_inputs(
            size=0.5,
            stratify_by_column=None,
            include_tables=None,
            row_per=None,
            element_table="Image",
            partition_by=None,
        )
        assert result == "element"


# =============================================================================
# Integration tests — TestSubsample (spec §6 E)
# =============================================================================


class TestSubsample:
    """Integration tests for ``subsample`` with a live catalog."""

    @staticmethod
    def _setup_subsamplable_dataset(ml: DerivaML) -> str:
        """Create a dataset with enough members and class diversity to subsample.

        Mirrors ``TestSplitDataset._setup_splittable_dataset`` but with
        more rows (24) so stratified subsampling has at least 2 rows
        per class to draw from.

        Returns:
            RID of the created source dataset.
        """
        from deriva_ml import BuiltinTypes, ColumnDefinition, TableDefinition

        ml.model.create_table(
            TableDefinition(
                name="SubsampleTestItem",
                columns=[
                    ColumnDefinition(name="Name", type=BuiltinTypes.text),
                    ColumnDefinition(name="Category", type=BuiltinTypes.text),
                ],
            )
        )
        ml.add_dataset_element_type("SubsampleTestItem")

        table_path = ml.catalog.getPathBuilder().schemas[ml.default_schema].tables["SubsampleTestItem"]
        records = [
            {"Name": f"Item{i}", "Category": ["A", "B", "C", "D"][i % 4]}
            for i in range(24)
        ]
        table_path.insert(records)
        item_rids = [r["RID"] for r in table_path.entities().fetch()]

        ml.add_term(MLVocab.workflow_type, "Setup", description="Setup workflow")
        ml.add_term(MLVocab.workflow_type, "Subsample", description="Subsample workflow")
        ml.add_term("Dataset_Type", "Source", description="Source dataset")
        workflow = ml.create_workflow(
            name="Subsample Setup Workflow",
            workflow_type="Setup",
            description="Creating subsample test data",
        )
        execution = ml.create_execution(ExecutionConfiguration(description="Setup", workflow=workflow))
        dataset = execution.create_dataset(
            dataset_types=["Source"],
            description="Test dataset for subsampling",
        )
        dataset.add_dataset_members({"SubsampleTestItem": item_rids})

        return dataset.dataset_rid

    @staticmethod
    def _subsample_in_execution(ml: DerivaML, source_rid: str, **kwargs) -> SubsampleResult:
        """Open a subsample execution, run ``subsample``, and commit."""
        workflow = ml.create_workflow(
            name="Test Subsample Workflow",
            workflow_type="Subsample",
            description="Subsampling workflow for tests",
        )
        with ml.create_execution(ExecutionConfiguration(workflow=workflow, description="Test subsample")) as exe:
            result = subsample(ml, source_rid, exe, **kwargs)
        exe.commit_output_assets(clean_folder=True)
        return result

    # ------ happy path ------

    def test_basic_random_subsample(self, test_ml):
        """A basic random subsample produces one dataset with the requested count."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(ml, source_rid, size=8, seed=42)

        assert isinstance(result, SubsampleResult)
        assert result.source == source_rid
        assert result.subsample.count == 8
        assert result.strategy == "random"
        # The new dataset exists.
        ds = ml.lookup_dataset(result.subsample.rid)
        assert ds is not None
        members = ds.list_dataset_members().get("SubsampleTestItem", [])
        assert len(members) == 8

    def test_fractional_size(self, test_ml):
        """A float ``size`` in (0, 1) is treated as a fraction."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(ml, source_rid, size=0.25, seed=42)
        # 24 * 0.25 = 6
        assert result.subsample.count == 6

    # ------ stratified ------

    @requires_sklearn
    def test_stratified_subsample(self, test_ml):
        """Stratified subsample preserves class proportions."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(
            ml,
            source_rid,
            size=8,
            seed=42,
            stratify_by_column="SubsampleTestItem.Category",
            include_tables=["SubsampleTestItem"],
            element_table="SubsampleTestItem",
        )

        assert "stratified" in result.strategy
        ds = ml.lookup_dataset(result.subsample.rid)
        # The 8 sampled records should cover all 4 categories
        # (2 per category) since the source is perfectly balanced 6/6/6/6.
        members = ds.list_dataset_members().get("SubsampleTestItem", [])
        assert len(members) == 8

    # ------ dry-run ------

    def test_dry_run_returns_plan_without_writes(self, test_ml):
        """dry_run=True returns a SubsampleResult plan without mutating the catalog."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        before = len(list(ml.find_datasets()))

        result = self._subsample_in_execution(ml, source_rid, size=8, seed=42, dry_run=True)

        assert isinstance(result, SubsampleResult)
        assert result.dry_run is True
        assert result.subsample.rid == "(dry run)"
        assert result.subsample.version == "(dry run)"
        assert result.subsample.count == 8
        assert result.source == source_rid

        # No catalog mutation — datasets count is unchanged.
        after = len(list(ml.find_datasets()))
        assert before == after

    def test_dry_run_records_no_input_edge(self, test_ml):
        """dry_run=True must not record the source as an execution input.

        Same contract as ``split_dataset`` — the dry-run path must not
        write the source-input edge.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        workflow = ml.create_workflow(
            name="Dry-Run Subsample Workflow",
            workflow_type="Subsample",
            description="Dry-run subsample input-edge test",
        )
        with ml.create_execution(
            ExecutionConfiguration(workflow=workflow, description="Dry-run subsample")
        ) as exe:
            subsample(ml, source_rid, exe, size=4, seed=42, dry_run=True)
            input_rids = {ds.dataset_rid for ds in exe.list_input_datasets()}

        assert source_rid not in input_rids, (
            "dry_run subsample must not record the source as an execution input"
        )

    # ------ single-output / no parent ------

    def test_single_output_no_parent_split(self, test_ml):
        """subsample produces a single dataset — no parent Split, no Dataset_Dataset edges.

        This is the structural contract that distinguishes ``subsample``
        from a one-sided ``split_dataset``.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(ml, source_rid, size=8, seed=42)

        sub_ds = ml.lookup_dataset(result.subsample.rid)
        # No parent Dataset_Dataset edges — the subsample is standalone.
        parents = sub_ds.list_dataset_parents()
        assert len(parents) == 0, (
            "subsample must not create a Dataset_Dataset parent edge; "
            f"got parents={[p.dataset_rid for p in parents]}"
        )
        # No children either (a subsample is a leaf).
        children = sub_ds.list_dataset_children()
        assert len(children) == 0

    # ------ provenance ------

    def test_source_recorded_as_execution_input(self, test_ml):
        """The source is recorded as an INPUT of the subsample's execution.

        Same provenance shape as ``split_dataset``. The subsample's
        producing-execution path leads back to the source via the
        ``Dataset_Execution`` association; no ``Dataset_Dataset`` edge
        is involved.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        workflow = ml.create_workflow(
            name="Provenance Subsample Workflow",
            workflow_type="Subsample",
            description="Subsample provenance test",
        )
        with ml.create_execution(
            ExecutionConfiguration(workflow=workflow, description="Subsample provenance")
        ) as exe:
            result = subsample(ml, source_rid, exe, size=8, seed=42)
            input_rids = {ds.dataset_rid for ds in exe.list_input_datasets()}
        exe.commit_output_assets(clean_folder=True)

        # The source is an input...
        assert source_rid in input_rids, (
            f"source {source_rid} should be an input of the subsample execution; "
            f"got inputs {input_rids}"
        )
        # ...and the subsample (this execution authored it) is NOT an input.
        assert result.subsample.rid not in input_rids

    def test_no_dataset_dataset_edge_from_source(self, test_ml):
        """Source dataset has no Dataset_Dataset child edge pointing at the subsample.

        Pins the inverse of the provenance contract: the derivation
        relationship lives in the execution graph, never in
        ``Dataset_Dataset``.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(ml, source_rid, size=8, seed=42)

        src_ds = ml.lookup_dataset(source_rid)
        child_rids = {c.dataset_rid for c in src_ds.list_dataset_children()}
        assert result.subsample.rid not in child_rids, (
            "subsample must not be a Dataset_Dataset child of its source"
        )

    # ------ Subsample tag application ------

    def test_subsample_tag_always_applied(self, test_ml):
        """The output dataset is always tagged with ``Subsample``."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(ml, source_rid, size=8, seed=42)

        ds = ml.lookup_dataset(result.subsample.rid)
        assert "Subsample" in ds.dataset_types

    def test_dataset_types_argument_honored(self, test_ml):
        """Caller-supplied ``dataset_types`` are added to the output dataset."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(
            ml,
            source_rid,
            size=8,
            seed=42,
            dataset_types=["Training", "Labeled"],
        )

        ds = ml.lookup_dataset(result.subsample.rid)
        # Both the caller-supplied types AND the Subsample origin tag are present.
        assert "Training" in ds.dataset_types
        assert "Labeled" in ds.dataset_types
        assert "Subsample" in ds.dataset_types

    def test_dataset_types_dedupes_subsample_tag(self, test_ml):
        """Spec §7 R4: passing ``Subsample`` in dataset_types does not double-tag.

        Defensive: the caller may pass ``dataset_types=["Subsample"]``
        thinking they need to be explicit. The implementation dedupes
        so the output carries Subsample exactly once.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result = self._subsample_in_execution(
            ml,
            source_rid,
            size=8,
            seed=42,
            dataset_types=["Subsample", "Labeled"],
        )

        ds = ml.lookup_dataset(result.subsample.rid)
        # Subsample appears exactly once (set membership; the catalog
        # itself stores each type as a row, but the de-dupe in
        # ``subsample`` prevents the duplicate insert).
        types = ds.dataset_types
        assert types.count("Subsample") == 1
        assert "Labeled" in types

    # ------ role-types-don't-propagate ------

    def test_role_types_dont_propagate_from_source(self, test_ml):
        """The subsample's role types come from ``dataset_types``, not the source.

        Source tagged ``Testing``; subsample with ``dataset_types=["Training"]``
        should carry ``Training`` (not Testing). The source's role tag is
        never inherited.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        # Tag the source as a Testing corpus.
        ml.add_term(MLVocab.dataset_type, "Testing", description="Test role tag")
        ml.lookup_dataset(source_rid).add_dataset_type("Testing")

        result = self._subsample_in_execution(
            ml,
            source_rid,
            size=8,
            seed=42,
            dataset_types=["Training"],
        )

        ds = ml.lookup_dataset(result.subsample.rid)
        assert "Training" in ds.dataset_types
        # And the source's Testing tag must NOT have leaked onto the subsample.
        assert "Testing" not in ds.dataset_types, (
            f"subsample must not inherit role-axis types from the source; "
            f"source was Testing, got subsample types {ds.dataset_types}"
        )

    def test_subsample_does_not_mutate_source_dataset_types(self, test_ml):
        """``subsample`` never modifies the source's ``dataset_types``."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        before = set(ml.lookup_dataset(source_rid).dataset_types)
        self._subsample_in_execution(
            ml,
            source_rid,
            size=8,
            seed=42,
            dataset_types=["Training", "Labeled"],
        )
        after = set(ml.lookup_dataset(source_rid).dataset_types)
        assert before == after, (
            f"subsample must not mutate the source's dataset_types; "
            f"before={before}, after={after}"
        )

    # ------ determinism ------

    def test_deterministic_by_seed(self, test_ml):
        """Same seed → same RID list."""
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result_a = self._subsample_in_execution(ml, source_rid, size=8, seed=42)
        rids_a = {
            r["RID"]
            for r in ml.lookup_dataset(result_a.subsample.rid)
            .list_dataset_members()
            .get("SubsampleTestItem", [])
        }

        result_b = self._subsample_in_execution(ml, source_rid, size=8, seed=42)
        rids_b = {
            r["RID"]
            for r in ml.lookup_dataset(result_b.subsample.rid)
            .list_dataset_members()
            .get("SubsampleTestItem", [])
        }

        assert rids_a == rids_b, "Same seed must produce the same subsample"

    def test_different_seeds_produce_different_subsamples(self, test_ml):
        """Two different seeds should produce different subsamples.

        Probabilistic — for size=8 of 24 the collision rate is
        1/C(24,8) ≈ 1/735K, so the test is effectively deterministic.
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        result_a = self._subsample_in_execution(ml, source_rid, size=8, seed=42)
        rids_a = {
            r["RID"]
            for r in ml.lookup_dataset(result_a.subsample.rid)
            .list_dataset_members()
            .get("SubsampleTestItem", [])
        }

        result_c = self._subsample_in_execution(ml, source_rid, size=8, seed=99)
        rids_c = {
            r["RID"]
            for r in ml.lookup_dataset(result_c.subsample.rid)
            .list_dataset_members()
            .get("SubsampleTestItem", [])
        }

        assert rids_a != rids_c, "Different seeds should produce different subsamples"

    # ------ config artifact ------

    def test_writes_subsample_config_json(self, test_ml):
        """A ``subsample_config.json`` artifact lands in execution.working_dir."""
        from pathlib import Path

        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        workflow = ml.create_workflow(
            name="Config Subsample Workflow",
            workflow_type="Subsample",
            description="Subsample config artifact test",
        )
        with ml.create_execution(
            ExecutionConfiguration(workflow=workflow, description="Config subsample")
        ) as exe:
            subsample(
                ml,
                source_rid,
                exe,
                size=8,
                seed=42,
                description="Test subsample with config",
            )
            params_file = Path(exe.working_dir) / "subsample_config.json"
            assert params_file.exists(), (
                f"subsample must write subsample_config.json to "
                f"execution.working_dir; not found at {params_file}"
            )
            # The file is valid JSON and carries the source + sample size.
            import json

            params = json.loads(params_file.read_text())
            assert params["source_dataset_rid"] == source_rid
            assert params["sample_size"] == 8
            assert params["seed"] == 42
        exe.commit_output_assets(clean_folder=True)

    # ------ size exceeds source ------

    def test_size_exceeds_source_raises(self, test_ml):
        """Requesting more samples than the source contains raises ValueError.

        Comes through ``_resolve_sizes`` inside ``_compute_partitions``
        (we pass the size as ``test_size`` with no ``train_size``).
        """
        ml = test_ml
        source_rid = self._setup_subsamplable_dataset(ml)

        with pytest.raises(ValueError):
            self._subsample_in_execution(ml, source_rid, size=100, seed=42)

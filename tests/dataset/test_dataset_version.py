from pprint import pformat

try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None

from deriva_ml import BuiltinTypes, ColumnDefinition, MLVocab, TableDefinition
from deriva_ml.dataset.aux_classes import DatasetVersion, VersionPart
from deriva_ml.execution.execution import ExecutionConfiguration


class TestDatasetVersion:
    def test_dataset_version_simple(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
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

        dataset = execution.create_dataset(
            dataset_types=type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        v0 = dataset.current_version
        assert "1.0.0" == str(v0)
        # Use the private force-bump primitive: this test verifies the
        # release-segment arithmetic and graph propagation without
        # going through the dev → release lifecycle.
        v1 = dataset._increment_dataset_version(component=VersionPart.minor)
        assert "1.1.0" == str(v1)
        assert "1.1.0" == str(dataset.current_version)

    def test_dataset_version_history(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
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

        dataset = execution.create_dataset(
            dataset_types=type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        assert 1 == len(dataset.dataset_history())
        v1 = dataset._increment_dataset_version(component=VersionPart.minor)
        assert 2 == len(dataset.dataset_history())

    def test_dataset_version(self, dataset_test, tmp_path):
        dataset_description = dataset_test.dataset_description
        ml_instance = dataset_description.dataset._ml_instance
        nested_datasets = [ml_instance.lookup_dataset(ds) for ds in dataset_description.member_rids.get("Dataset", [])]
        datasets = [
            ml_instance.lookup_dataset(dataset)
            for nested_description in dataset_description.members.get("Dataset", [])
            for dataset in nested_description.member_rids.get("Dataset", [])
        ]
        ic(datasets)
        _versions = {
            "d0": dataset_description.dataset.current_version,
            "d1": [ds.current_version for ds in nested_datasets],
            "d2": [ds.current_version for ds in datasets],
        }
        # Use the private force-bump primitive: this test verifies graph
        # propagation, which is _increment_dataset_version's specialty.
        # The public release() path operates on a single dataset only.
        nested_datasets[0]._increment_dataset_version(VersionPart.major)
        new_versions = {
            "d0": dataset_description.dataset.current_version,
            "d1": [ds.current_version for ds in nested_datasets],
            "d2": [ds.current_version for ds in datasets],
        }
        ic(_versions)
        ic(new_versions)
        assert new_versions["d0"].major == 2
        assert new_versions["d2"][0].major == 2


class TestMarkDev:
    """Integration tests for ``Dataset.mark_dev`` and the dev-row lifecycle.

    Covers ADR-0003's lazy-mutable-dev-row contract: first call after a
    release creates the dev row at ``.dev1``; subsequent calls advance
    ``.devN`` in place; ``Description`` is replaced (not appended);
    ``current_version`` and ``dataset_history`` reflect the dev state.
    """

    def _setup_dataset(self, ml_instance):
        """Helper: register vocabulary, create workflow+execution, return a fresh dataset."""
        ml_instance.add_term("Dataset_Type", "DevTest", description="A test")
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="A manual workflow")
        workflow = ml_instance.create_workflow(
            name="MarkDev Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for mark_dev tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="MarkDev Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types="DevTest",
            description="Dataset for mark_dev tests",
            version=DatasetVersion(0, 4, 0),
        )
        return dataset, execution

    def test_first_mark_dev_creates_dev1(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        assert str(dataset.current_version) == "0.4.0"

        dataset.mark_dev("Picked up classifier output")

        assert str(dataset.current_version) == "0.4.0.post1.dev1"
        assert dataset.current_version.is_devrelease

    def test_subsequent_mark_dev_advances_devN(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("First drift")
        dataset.mark_dev("Second drift")
        dataset.mark_dev("Third drift")

        assert str(dataset.current_version) == "0.4.0.post1.dev3"

    def test_dev_row_snapshot_is_null(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("Some drift")

        history = dataset.dataset_history()
        dev_entries = [h for h in history if h.dataset_version.is_devrelease]
        assert len(dev_entries) == 1
        assert dev_entries[0].snapshot is None

    def test_dev_row_count_stays_one(self, test_ml):
        """Mutable dev row: advancing .devN updates the same row."""
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("First")
        dataset.mark_dev("Second")
        dataset.mark_dev("Third")

        history = dataset.dataset_history()
        dev_entries = [h for h in history if h.dataset_version.is_devrelease]
        assert len(dev_entries) == 1, "expected exactly one dev row across calls"

    def test_description_is_replaced_not_appended(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("First description")
        dataset.mark_dev("Second description")

        history = dataset.dataset_history()
        dev = next(h for h in history if h.dataset_version.is_devrelease)
        assert dev.description == "Second description"
        assert "First description" not in (dev.description or "")

    def test_history_includes_dev_row(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        history_before = dataset.dataset_history()
        assert len(history_before) == 1

        dataset.mark_dev("Some drift")

        history_after = dataset.dataset_history()
        assert len(history_after) == 2
        assert any(h.dataset_version.is_devrelease for h in history_after)

    def test_history_is_sorted_ascending(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        # Make a few releases plus a dev row.
        dataset._increment_dataset_version(VersionPart.minor, "first bump")
        dataset._increment_dataset_version(VersionPart.minor, "second bump")
        dataset.mark_dev("then dev")

        history = dataset.dataset_history()
        labels = [str(h.dataset_version) for h in history]
        # Released ones first, dev label last (sorts after its anchor release).
        assert labels == sorted(labels, key=lambda s: DatasetVersion.parse(s))
        # And the last entry is the dev label.
        assert history[-1].dataset_version.is_devrelease

    def test_mark_dev_with_execution_attaches_execution(self, test_ml):
        dataset, execution = self._setup_dataset(test_ml)
        dataset.mark_dev("With execution", execution=execution)

        history = dataset.dataset_history()
        dev = next(h for h in history if h.dataset_version.is_devrelease)
        assert dev.execution_rid == execution.execution_rid

    def test_mark_dev_without_execution_leaves_null(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("Without execution")

        history = dataset.dataset_history()
        dev = next(h for h in history if h.dataset_version.is_devrelease)
        # DatasetHistory normalises empty/missing execution_rid to None.
        assert dev.execution_rid is None


class TestMutationsLandOnDev:
    """Integration tests for PR 4: member and type mutations flip to dev.

    Verifies the behavior change committed in ADR-0003 / PR 4: every
    mutation that today bumped to a released version now lands on a dev
    version instead. The dev counter advances per call (Q18), and
    no-op input doesn't advance.
    """

    def _setup_dataset_with_table(self, ml_instance):
        """Helper: register an element-type table and a fresh dataset.

        Returns (dataset, test_rids) where test_rids are five rows
        in the registered element type.
        """
        ml_instance.add_term(MLVocab.dataset_type, "DevDataMutation", description="A test type")
        ml_instance.add_term(MLVocab.dataset_type, "AnotherType", description="A second test type")
        ml_instance.add_term(MLVocab.workflow_type, "Manual Workflow", description="Manual workflow")
        ml_instance.model.create_table(
            TableDefinition(
                name="MutationTestItem",
                columns=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("MutationTestItem")
        table_path = ml_instance.catalog.getPathBuilder().schemas[ml_instance.default_schema].tables["MutationTestItem"]
        table_path.insert([{"Col1": f"Item{i}"} for i in range(5)])
        test_rids = [r["RID"] for r in table_path.entities().fetch()]

        workflow = ml_instance.create_workflow(
            name="Mutation Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for mutation tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Mutation Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types=["DevDataMutation"],
            description="Dataset for mutation tests",
            version=DatasetVersion(0, 4, 0),
        )
        return dataset, test_rids

    def test_add_dataset_members_flips_to_dev(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        # Sanity check the starting state — released at 0.4.0.
        assert str(dataset.current_version) == "0.4.0"

        dataset.add_dataset_members({"MutationTestItem": test_rids[:2]})

        new_version = dataset.current_version
        assert new_version.is_devrelease, f"Expected dev version, got {new_version}"
        assert str(new_version) == "0.4.0.post1.dev1"

    def test_add_dataset_members_advances_devN_on_subsequent_calls(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"MutationTestItem": test_rids[:1]})
        dataset.add_dataset_members({"MutationTestItem": test_rids[1:2]})
        dataset.add_dataset_members({"MutationTestItem": test_rids[2:3]})

        assert str(dataset.current_version) == "0.4.0.post1.dev3"

    def test_add_dataset_members_with_empty_input_is_noop(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)
        version_before = dataset.current_version

        dataset.add_dataset_members({"MutationTestItem": []})

        version_after = dataset.current_version
        # No row was inserted, so the dev counter does not advance.
        assert version_after == version_before
        assert not version_after.is_devrelease

    def test_add_dataset_type_flips_to_dev(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)
        assert not dataset.current_version.is_devrelease

        dataset.add_dataset_type("AnotherType")

        new_version = dataset.current_version
        assert new_version.is_devrelease
        assert str(new_version) == "0.4.0.post1.dev1"

    def test_add_dataset_type_for_existing_type_is_noop(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)
        version_before = dataset.current_version

        # The dataset was created with type "DevDataMutation"; re-adding it is a no-op.
        dataset.add_dataset_type("DevDataMutation")

        version_after = dataset.current_version
        assert version_after == version_before
        assert not version_after.is_devrelease

    def test_remove_dataset_type_flips_to_dev(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)
        # First add a second type so we have something to remove without
        # leaving the dataset typeless.
        dataset.add_dataset_types(["AnotherType"])
        # That advanced to .dev1; now remove it.
        dataset.remove_dataset_type("AnotherType")

        new_version = dataset.current_version
        assert new_version.is_devrelease
        # First add was .dev1; remove was .dev2.
        assert str(new_version) == "0.4.0.post1.dev2"

    def test_remove_dataset_type_for_absent_type_is_noop(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)
        version_before = dataset.current_version

        # "AnotherType" was added to the vocabulary but never associated
        # with this dataset, so removing it is a no-op.
        dataset.remove_dataset_type("AnotherType")

        version_after = dataset.current_version
        assert version_after == version_before
        assert not version_after.is_devrelease

    def test_force_bump_via_internal_helper(self, test_ml):
        """``_increment_dataset_version`` (private, force-bump) still produces released rows.

        After PR 5, the public release path is :meth:`Dataset.release`,
        which requires a dev row. ``_increment_dataset_version`` is
        kept as a private internal primitive for callers that need to
        force-bump without a dev period (e.g., catalog clone).
        """
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)

        new_version = dataset._increment_dataset_version(VersionPart.minor)

        assert not new_version.is_devrelease
        assert str(new_version) == "0.5.0"


class TestRelease:
    """Integration tests for ``Dataset.release`` per ADR-0003 / PR 5.

    Verifies:
    - release() promotes the dev row in place (released label, stamped Snapshot,
      replaced Description, overwritten Execution).
    - release() errors when no dev row exists.
    - release() honors the ``bump`` argument (major / minor / patch).
    - release() returns the new released ``DatasetVersion``.
    """

    def _setup_dataset_with_dev(self, ml_instance):
        """Helper: create a fresh dataset and flip it to dev via mark_dev.

        Returns (dataset, execution, dev_row_rid).
        """
        ml_instance.add_term("Dataset_Type", "ReleaseTest", description="A test type")
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="Manual workflow")
        workflow = ml_instance.create_workflow(
            name="Release Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for release tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Release Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types="ReleaseTest",
            description="Dataset for release tests",
            version=DatasetVersion(0, 4, 0),
        )
        dataset.mark_dev("initial drift")
        # Find the dev row's RID for in-place promotion verification.
        dev_row = next(h for h in dataset.dataset_history() if h.dataset_version.is_devrelease)
        return dataset, execution, dev_row.version_rid

    def test_release_minor_bump(self, test_ml):
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)
        assert str(dataset.current_version) == "0.4.0.post1.dev1"

        v = dataset.release(bump=VersionPart.minor, description="release notes")

        assert str(v) == "0.5.0"
        assert not v.is_devrelease
        assert str(dataset.current_version) == "0.5.0"

    def test_release_major_bump(self, test_ml):
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)

        v = dataset.release(bump=VersionPart.major, description="major change")

        assert str(v) == "1.0.0"

    def test_release_patch_bump(self, test_ml):
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)

        v = dataset.release(bump=VersionPart.patch, description="patch change")

        assert str(v) == "0.4.1"

    def test_release_promotes_in_place(self, test_ml):
        """Per ADR-0003 / Q12: the dev row's RID is preserved across promotion."""
        dataset, _execution, dev_rid = self._setup_dataset_with_dev(test_ml)

        dataset.release(bump=VersionPart.minor, description="promoted")

        # The released row should have the same RID as the dev row
        # (UPDATE in place, not INSERT a new row + DELETE the old).
        history = dataset.dataset_history()
        released_at_05 = [h for h in history if str(h.dataset_version) == "0.5.0"]
        assert len(released_at_05) == 1
        assert released_at_05[0].version_rid == dev_rid

    def test_release_stamps_snapshot(self, test_ml):
        """Released rows have a non-NULL Snapshot; dev rows had NULL."""
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)

        dataset.release(bump=VersionPart.minor, description="snap test")

        released = next(h for h in dataset.dataset_history() if str(h.dataset_version) == "0.5.0")
        assert released.snapshot is not None
        assert released.snapshot != ""

    def test_release_replaces_description(self, test_ml):
        """Per Q12: release replaces the dev row's accumulated description."""
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)
        # Add another mark_dev to accumulate description.
        dataset.mark_dev("more drift")

        dataset.release(bump=VersionPart.minor, description="release notes only")

        released = next(h for h in dataset.dataset_history() if str(h.dataset_version) == "0.5.0")
        assert released.description == "release notes only"
        assert "drift" not in (released.description or "")

    def test_release_overwrites_execution(self, test_ml):
        """release()'s execution arg overwrites whatever the dev row had."""
        dataset, original_execution, _dev_rid = self._setup_dataset_with_dev(test_ml)
        # Create a different execution to attach at release time.
        ml_instance = dataset._ml_instance
        workflow = ml_instance.create_workflow(
            name="Release Workflow 2",
            workflow_type="Manual Workflow",
            description="A second workflow",
        )
        release_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Release-time execution", workflow=workflow)
        )

        dataset.release(
            bump=VersionPart.minor,
            description="with release execution",
            execution=release_execution,
        )

        released = next(h for h in dataset.dataset_history() if str(h.dataset_version) == "0.5.0")
        assert released.execution_rid == release_execution.execution_rid
        # Not the original mark_dev execution.
        assert released.execution_rid != original_execution.execution_rid

    def test_release_without_execution_leaves_null(self, test_ml):
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)

        dataset.release(bump=VersionPart.minor, description="no execution")

        released = next(h for h in dataset.dataset_history() if str(h.dataset_version) == "0.5.0")
        # DatasetHistory normalises empty/missing execution_rid to None.
        assert released.execution_rid is None

    def test_release_errors_on_no_dev_row(self, test_ml):
        """release() with no dev period to promote raises a clear error."""
        import pytest

        # Use the mark_dev helper *without* mark_dev — fresh released-only dataset.
        ml_instance = test_ml
        ml_instance.add_term("Dataset_Type", "ReleaseTest", description="A test type")
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="Manual workflow")
        workflow = ml_instance.create_workflow(
            name="Release Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for release tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Release Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types="ReleaseTest",
            description="Dataset with no dev period",
            version=DatasetVersion(0, 4, 0),
        )
        # Sanity: dataset is at a released version, no dev row.
        assert not dataset.current_version.is_devrelease

        with pytest.raises(Exception) as exc_info:
            dataset.release(bump=VersionPart.minor, description="should fail")
        # Error message points the user at mark_dev as the resolution.
        assert "mark_dev" in str(exc_info.value)

    def test_release_then_mark_dev_creates_new_dev_period(self, test_ml):
        """After release, the next mark_dev creates a fresh .dev1 anchored at the new release."""
        dataset, _execution, _dev_rid = self._setup_dataset_with_dev(test_ml)
        dataset.release(bump=VersionPart.minor, description="0.5.0 release")
        assert str(dataset.current_version) == "0.5.0"

        dataset.mark_dev("new drift")

        # Anchored at 0.5.0 (the new last-released), not 0.4.0.
        assert str(dataset.current_version) == "0.5.0.post1.dev1"

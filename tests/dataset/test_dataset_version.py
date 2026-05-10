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
        # Bump to a new released version through the dev-versioning lifecycle:
        # mark_dev declares a dev period, release promotes it.
        dataset.mark_dev(description="bump to 1.1.0")
        v1 = dataset.release(bump=VersionPart.minor, description="1.1.0")
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
        dataset.mark_dev(description="bump")
        # After mark_dev, history has the initial release plus the dev row.
        assert 2 == len(dataset.dataset_history())
        # Release promotes the dev row in place — history count stays at 2.
        v1 = dataset.release(bump=VersionPart.minor, description="1.1.0")
        assert 2 == len(dataset.dataset_history())

    # NOTE: a `test_dataset_version` test was deleted in this PR.
    # It had verified that bumping a child dataset's version cascaded up to
    # the parent's. Graph cascading was a property of
    # ``_increment_dataset_version`` only; ``release()`` operates on a single
    # dataset (per ADR-0003 / Branch C). With ``_increment_dataset_version``
    # removed, there is no public API path that cascades. If parent-version
    # propagation is wanted later, a new explicit API should be designed
    # (probably as an explicit ``release_with_descendants(...)`` opt-in,
    # not an implicit cascade).


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
        # Make a few releases plus a dev row, all through the public lifecycle.
        dataset.mark_dev("first bump")
        dataset.release(bump=VersionPart.minor, description="first bump")
        dataset.mark_dev("second bump")
        dataset.release(bump=VersionPart.minor, description="second bump")
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

    # The previous test_force_bump_via_internal_helper was deleted in this
    # PR alongside the removal of Dataset._increment_dataset_version. Public
    # callers must use mark_dev() + release() now. The catalog-clone code
    # was the last remaining caller of the private primitive and was
    # rewritten to go through the public lifecycle.


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


class TestDriftDetection:
    """Integration tests for ``is_dirty`` / ``release_diff`` / ``compare_versions`` per ADR-0003 / PR 6.

    Covers the drift-detection trio. All three share one internal walk
    that filters reachable rows by ``RMT`` time predicate. The trio
    differs only in the predicate and what's returned (bool vs.
    per-table counts).
    """

    def _setup_dataset_with_table(self, ml_instance):
        """Helper: create a fresh dataset with an element-type table.

        Returns (dataset, test_rids).
        """
        ml_instance.add_term(MLVocab.dataset_type, "DriftTest", description="A test type")
        ml_instance.add_term(MLVocab.workflow_type, "Manual Workflow", description="Manual workflow")
        ml_instance.model.create_table(
            TableDefinition(
                name="DriftTestItem",
                columns=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("DriftTestItem")
        table_path = ml_instance.catalog.getPathBuilder().schemas[ml_instance.default_schema].tables["DriftTestItem"]
        table_path.insert([{"Col1": f"Item{i}"} for i in range(5)])
        test_rids = [r["RID"] for r in table_path.entities().fetch()]

        workflow = ml_instance.create_workflow(
            name="Drift Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for drift tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Drift Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types=["DriftTest"],
            description="Dataset for drift tests",
            version=DatasetVersion(0, 4, 0),
        )
        return dataset, test_rids

    def test_is_dirty_false_immediately_after_creation(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)

        assert dataset.is_dirty() is False

    def test_is_dirty_true_after_mutation(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DriftTestItem": test_rids[:2]})

        assert dataset.is_dirty() is True

    def test_is_dirty_false_after_release(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DriftTestItem": test_rids[:2]})
        dataset.release(bump=VersionPart.minor, description="0.5.0")

        assert dataset.is_dirty() is False

    def test_release_diff_empty_when_clean(self, test_ml):
        dataset, _test_rids = self._setup_dataset_with_table(test_ml)

        assert dataset.release_diff() == {}

    def test_release_diff_reports_added_members(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)

        dataset.add_dataset_members({"DriftTestItem": test_rids[:3]})

        diff = dataset.release_diff()
        # The Dataset_DriftTestItem association table should appear as drifted.
        # (Or the DriftTestItem table itself, depending on path traversal.)
        assert diff, f"Expected non-empty drift, got {diff}"
        # All values are positive counts.
        assert all(v > 0 for v in diff.values())

    def test_compare_versions_between_two_releases(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DriftTestItem": test_rids[:2]})
        dataset.release(bump=VersionPart.minor, description="v0.5.0 with 2 members")
        # Now at 0.5.0 released. Add more members and release again.
        dataset.add_dataset_members({"DriftTestItem": test_rids[2:5]})
        dataset.release(bump=VersionPart.minor, description="v0.6.0 with 5 members")

        diff = dataset.compare_versions("0.5.0", "0.6.0")
        assert diff, f"Expected non-empty diff between 0.5.0 and 0.6.0, got {diff}"
        # All values are positive.
        assert all(v > 0 for v in diff.values())

    def test_compare_versions_argument_order_is_symmetric(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DriftTestItem": test_rids[:2]})
        dataset.release(bump=VersionPart.minor, description="v0.5.0")
        dataset.add_dataset_members({"DriftTestItem": test_rids[2:5]})
        dataset.release(bump=VersionPart.minor, description="v0.6.0")

        ab = dataset.compare_versions("0.5.0", "0.6.0")
        ba = dataset.compare_versions("0.6.0", "0.5.0")

        assert ab == ba, "compare_versions should be symmetric in argument order"

    def test_compare_versions_same_version_returns_empty(self, test_ml):
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DriftTestItem": test_rids[:2]})
        dataset.release(bump=VersionPart.minor, description="v0.5.0")

        # Comparing the same version against itself yields nothing.
        diff = dataset.compare_versions("0.5.0", "0.5.0")
        assert diff == {}

    def test_compare_versions_with_unknown_version_raises(self, test_ml):
        import pytest

        dataset, _test_rids = self._setup_dataset_with_table(test_ml)

        with pytest.raises(Exception) as exc_info:
            dataset.compare_versions("0.4.0", "9.9.9")
        assert "9.9.9" in str(exc_info.value)

    def test_compare_versions_dev_label_resolves_to_live(self, test_ml):
        """A dev label that matches the current dev row resolves to live state."""
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DriftTestItem": test_rids[:2]})
        # Now at 0.4.0.post1.dev1 (dev). Compare to 0.4.0 — should show drift.
        current = str(dataset.current_version)

        diff = dataset.compare_versions("0.4.0", current)
        assert diff, f"Expected non-empty diff between 0.4.0 and {current}, got {diff}"


class TestDownloadAtDevVersion:
    """Integration tests for ``download_dataset_bag`` against dev labels (issue #89).

    Pre-fix, downloading at a dev label crashed with a Pydantic
    ValidationError because the snapshot is NULL on dev rows. After the
    fix, dev labels resolve to live catalog state per ADR-0003 / Q20.
    """

    def _setup_dataset_with_table(self, ml_instance):
        """Helper: create a fresh dataset with an element-type table populated.

        Returns (dataset, test_rids).
        """
        ml_instance.add_term(MLVocab.dataset_type, "DownloadDevTest", description="A test type")
        ml_instance.add_term(MLVocab.workflow_type, "Manual Workflow", description="Manual workflow")
        ml_instance.model.create_table(
            TableDefinition(
                name="DownloadDevItem",
                columns=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("DownloadDevItem")
        table_path = ml_instance.catalog.getPathBuilder().schemas[ml_instance.default_schema].tables["DownloadDevItem"]
        table_path.insert([{"Col1": f"Item{i}"} for i in range(3)])
        test_rids = [r["RID"] for r in table_path.entities().fetch()]

        workflow = ml_instance.create_workflow(
            name="DownloadDev Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for dev-download tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="DownloadDev Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types=["DownloadDevTest"],
            description="Dataset for dev-download tests",
            version=DatasetVersion(0, 4, 0),
        )
        return dataset, test_rids

    def test_download_dev_version_succeeds(self, test_ml, tmp_path):
        """Downloading at a dev label produces a valid bag (no ValidationError)."""
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DownloadDevItem": test_rids[:2]})
        # Dataset is now at 0.4.0.post1.dev1 (dev).
        assert dataset.current_version.is_devrelease

        # Pre-fix this raised a Pydantic ValidationError ('<rid>@None').
        bag = dataset.download_dataset_bag(dataset.current_version, use_minid=False)
        assert bag is not None

    def test_dev_minid_has_bare_rid(self, test_ml, tmp_path):
        """A DatasetMinid for a dev version uses a bare RID (no @snaptime)."""
        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DownloadDevItem": test_rids[:2]})
        assert dataset.current_version.is_devrelease

        minid = dataset._get_dataset_minid(
            version=dataset.current_version,
            create=True,
            use_minid=False,
        )
        # Bare RID — no @snapshot suffix.
        assert "@" not in minid.version_rid
        assert minid.dataset_snapshot is None

    def test_use_minid_rejected_for_dev_version(self, test_ml, tmp_path):
        """use_minid=True is not allowed for dev versions (dev labels aren't citable)."""
        import pytest

        dataset, test_rids = self._setup_dataset_with_table(test_ml)
        dataset.add_dataset_members({"DownloadDevItem": test_rids[:2]})
        assert dataset.current_version.is_devrelease

        # Probe the dev-MINID rejection directly via the internal method —
        # ``download_dataset_bag`` has an earlier S3-bucket check that fires
        # first in test environments without S3 configured.
        with pytest.raises(Exception) as exc_info:
            dataset._get_dataset_minid(
                version=dataset.current_version,
                create=True,
                use_minid=True,
            )
        msg = str(exc_info.value)
        # Error mentions dev-version citability and points at release()
        # as the resolution path.
        assert "MINID" in msg or "minid" in msg.lower()
        assert "release" in msg.lower()

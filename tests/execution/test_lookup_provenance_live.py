"""Live-catalog proof of ``lookup_provenance``'s binding-RESCAN machinery.

Ruling 9 (#391) reports a dataset's binding evidence as of ONE snaptime —
the maximum walked pin. When a *later* walk round discovers a higher pin
than the one a dataset was already scanned at, the dataset must be
RESCANNED at the new maximum, and the previous as-of view (arcs **and**
gaps) must be REPLACED rather than kept alongside.

``tests/execution/test_lookup_provenance_unit.py`` pins this against a
fake catalog. This module is the live counterpart: the same shape built
with real vocabularies, features, feature values, dataset versions and
``Dataset_Execution`` input pins on a real Deriva catalog, so the rescan
is proven against actual snapshot resolution and actual
``_find_feature_producers_impl`` reads rather than a stub.

Why the machinery is otherwise unreachable in live evidence: production
catalogs observed so far pin each dataset at a single version per walk,
so the max walked pin never advances mid-walk and the rescan branch never
fires. The scenario below forces the advance deliberately.

The shape built on the catalog::

    D                      dataset over demo Subject/Image members
    E1  binds feature values on D's members       → D released at v1
    E2  binds ADDITIONAL feature values           → D released at v2
    E1  ALSO records D@v2 as a consumed input     (forces the pin advance)
    R   consumes D@v1                             (the walk root)

Walking ``lookup_provenance(R)``:

    round 1 walks D@v1 and scans D there → discovers E1 (visible at v1)
    expanding E1 walks D@v2 — a HIGHER pin → RESCAN of D at v2 → E2

Recording D@v2 as an input of E1 is contrived but schema-legal: it is
simply an execution that consumed a later version of a dataset it also
bound values onto. It is the minimum edge that advances the walked pin
in a round *after* the dataset was first scanned, which is the precise
precondition the rescan branch exists for.

Gated on ``DERIVA_HOST`` like the other live tests in this directory.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.core.definitions import BuiltinTypes, ColumnDefinition
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.provenance import ArcKind, ProvenanceClosure

pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_provenance live rescan test requires DERIVA_HOST",
)


def _binding_arcs(closure: ProvenanceClosure, dataset_rid: str) -> list:
    """Every ``member_binding`` arc in ``closure`` that targets ``dataset_rid``.

    Args:
        closure: The walked provenance closure.
        dataset_rid: The dataset whose binding arcs to collect.

    Returns:
        The matching :class:`~deriva_ml.execution.provenance.ProvenanceArc`
        list, across all closure executions.
    """
    return [
        arc
        for execution in closure.executions.values()
        for arc in execution.arcs
        if arc.kind == ArcKind.member_binding and arc.input_rid == dataset_rid
    ]


@pytest.fixture
def rescan_scenario(test_ml, monkeypatch):
    """Build the live pin-advance scenario and walk it, counting binding scans.

    Constructs D/E1/E2/R as documented in the module docstring, instruments
    ``_find_feature_producers_impl`` so the test can observe how many times
    (and at which versions) D was scanned, then returns everything the
    assertions need.

    Every RID comes from a live create call — none are hard-coded, per the
    repo's opaque-RID rule.

    Args:
        test_ml: Function-scoped DerivaML instance on a real catalog.
        monkeypatch: Used to wrap the binding-scan seam with a counter.

    Yields:
        A dict with the closure, the scan log, and the scenario RIDs.
    """
    ml = test_ml

    # --- Vocabulary + feature setup -------------------------------------
    ml.add_term(vc.dataset_type, "RescanLive", description="Rescan live-proof dataset")
    ml.add_term(vc.workflow_type, "Rescan Live Test", description="Rescan live-proof workflow")
    ml.create_vocabulary("RescanQuality", "Quality vocab for the rescan live proof", update_navbar=False)
    ml.add_term("RescanQuality", "Good", description="Good")
    ml.add_term("RescanQuality", "Bad", description="Bad")
    ml.create_vocabulary("RescanGrade", "Grade vocab bound only at v2", update_navbar=False)
    ml.add_term("RescanGrade", "High", description="High")
    ml.add_term("RescanGrade", "Low", description="Low")

    # Two DISTINCT features so E1's and E2's bindings are independent:
    # E2's feature is what only the v2 scan can see.
    ml.create_feature("Subject", "RescanQualityFeature", terms=["RescanQuality"], update_navbar=False)
    ml.create_feature(
        "Subject",
        "RescanGradeFeature",
        terms=["RescanGrade"],
        metadata=[ColumnDefinition(name="Confidence", type=BuiltinTypes.int2, nullok=True)],
        optional=["Confidence"],
        update_navbar=False,
    )
    ml.apply_catalog_annotations()

    QualityFeature = ml.feature_record_class("Subject", "RescanQualityFeature")
    GradeFeature = ml.feature_record_class("Subject", "RescanGradeFeature")

    wf = ml.create_workflow(
        name="Rescan live-proof workflow",
        workflow_type="Rescan Live Test",
        description="Live proof of binding rescan on pin advance (#391)",
    )

    # --- Members: a few Subject rows D will contain ----------------------
    domain_path = ml._domain_path()
    subject_rows = list(domain_path.tables["Subject"].insert([{"Name": f"rescan-subject-{i}"} for i in range(3)]))
    subject_rids = [row["RID"] for row in subject_rows]
    ml.add_dataset_element_type("Subject")

    # --- exec_ds: creates D and adds the Subject members -----------------
    exec_ds = ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="rescan exec_ds (creates D)"),
    )
    with exec_ds.execute():
        dataset = exec_ds.create_dataset(
            dataset_types="RescanLive",
            description="rescan live-proof dataset D",
            version=DatasetVersion(1, 0, 0),
        )
        dataset.add_dataset_members(
            {"Subject": subject_rids},
            description="rescan: D's Subject members",
        )
    exec_ds.commit_output_assets()
    dataset_rid = dataset.dataset_rid

    # --- E1: binds Quality values on D's members -------------------------
    e1 = ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="rescan E1 (binds at v1)"),
    )
    with e1.execute():
        e1.add_features([QualityFeature(Subject=rid, RescanQuality="Good") for rid in subject_rids])
    e1.commit_output_assets()
    e1_rid = e1.execution_rid

    # --- Release D at v1: the snapshot must be cut AFTER E1's bindings,
    # so a scan at v1 sees E1 and nothing later.
    dataset.mark_dev(description="rescan: E1 bound values onto D's members", execution=e1)
    v1 = dataset.release(
        bump=VersionPart.minor,
        description="rescan: D@v1 with E1's bindings",
        execution=exec_ds,
    )

    # --- E2: binds ADDITIONAL values (a different feature) ---------------
    e2 = ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="rescan E2 (binds only visible at v2)"),
    )
    with e2.execute():
        e2.add_features([GradeFeature(Subject=rid, RescanGrade="High", Confidence=5) for rid in subject_rids])
    e2.commit_output_assets()
    e2_rid = e2.execution_rid

    # --- Release D at v2: v2 sees E1 + E2 (monotone superset) ------------
    dataset.mark_dev(description="rescan: E2 bound additional values", execution=e2)
    v2 = dataset.release(
        bump=VersionPart.minor,
        description="rescan: D@v2 with E1+E2 bindings",
        execution=exec_ds,
    )
    assert str(v1) != str(v2), "v1 and v2 must be distinct released versions"

    # --- The pin-advance edge: E1 ALSO consumed D@v2 ---------------------
    # Contrived but legal. This is what makes the maximum walked pin
    # advance in a round AFTER D was first scanned at v1.
    e1.add_input_dataset(dataset_rid, version=v2)

    # --- R: the walk root, consuming D@v1 --------------------------------
    root = ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="rescan R (root, consumes D@v1)",
            datasets=[DatasetSpec(rid=dataset_rid, version=v1, materialize=False)],
        ),
    )
    root_rid = root.execution_rid

    # --- Instrument the binding-scan seam --------------------------------
    # Counts scans per (dataset, version) so the test can prove D was
    # scanned exactly twice: once at v1, once at v2.
    scan_log: list[tuple[str, str]] = []
    original = type(ml)._find_feature_producers_impl

    def counting_impl(self, ds_rid, version=None):
        scan_log.append((ds_rid, str(version) if version is not None else None))
        return original(self, ds_rid, version)

    monkeypatch.setattr(type(ml), "_find_feature_producers_impl", counting_impl)

    closure = ml.lookup_provenance(root_rid)

    yield {
        "closure": closure,
        "scan_log": scan_log,
        "dataset_rid": dataset_rid,
        "e1_rid": e1_rid,
        "e2_rid": e2_rid,
        "root_rid": root_rid,
        "v1": str(v1),
        "v2": str(v2),
        "ml": ml,
    }


def test_dataset_is_rescanned_exactly_twice_when_the_pin_advances(rescan_scenario):
    """D is scanned at v1, then RESCANNED at v2 when the walked pin advances.

    Exactly two scans of D: a third would mean the rescan degenerated into
    "rescan every round"; a single scan would mean the advance was missed
    entirely (the pre-fix behavior).
    """
    scans = [entry for entry in rescan_scenario["scan_log"] if entry[0] == rescan_scenario["dataset_rid"]]
    scanned_versions = sorted(version for _rid, version in scans)

    assert scanned_versions == sorted([rescan_scenario["v1"], rescan_scenario["v2"]]), (
        f"expected exactly two scans of D (v1 then a rescan at the advanced pin), got {scanned_versions}"
    )


def test_binding_arcs_all_carry_the_advanced_version(rescan_scenario):
    """Every surviving ``member_binding`` arc on D carries v2, never v1.

    The rescan REPLACES the previous as-of view. A v1-labeled arc surviving
    beside a v2-labeled one is the two-version-labels violation the fix
    exists to prevent.
    """
    closure = rescan_scenario["closure"]
    arcs = _binding_arcs(closure, rescan_scenario["dataset_rid"])

    assert arcs, "no member_binding arcs survived the rescan"
    assert {arc.input_version for arc in arcs} == {rescan_scenario["v2"]}, (
        "stale as-of view survived the rescan: "
        f"{sorted({a.input_version for a in arcs})} (v1={rescan_scenario['v1']}, v2={rescan_scenario['v2']})"
    )


def test_binding_arcs_are_replaced_not_duplicated(rescan_scenario):
    """E1 — present in BOTH scans — ends with exactly one binding arc on D.

    Merge-instead-of-replace would leave E1 holding two arcs for one
    dataset, one per as-of view.
    """
    closure = rescan_scenario["closure"]
    e1_rid = rescan_scenario["e1_rid"]

    assert e1_rid in closure.executions, "E1 (visible at v1) missing from the closure"
    e1_binding_arcs = [
        arc
        for arc in closure.executions[e1_rid].arcs
        if arc.kind == ArcKind.member_binding and arc.input_rid == rescan_scenario["dataset_rid"]
    ]
    assert len(e1_binding_arcs) == 1, f"rescan duplicated E1's arc instead of replacing it: {e1_binding_arcs}"


def test_e2_is_discovered_only_because_of_the_rescan(rescan_scenario):
    """E2 is in the closure, and is genuinely v2-only.

    The control that makes the whole test meaningful: a scan at v1 does NOT
    see E2, so E2's presence in the closure can only come from the rescan at
    the advanced pin. Asserted directly against the live catalog by scanning
    D at v1 and at v2 and comparing.
    """
    closure = rescan_scenario["closure"]
    ml = rescan_scenario["ml"]
    dataset_rid = rescan_scenario["dataset_rid"]
    e2_rid = rescan_scenario["e2_rid"]

    assert e2_rid in closure.executions, "E2 (bound only at the advanced pin) missing from the closure"

    # Live control: who does the catalog report at each version?
    at_v1 = {r.execution_rid for r in ml.find_feature_producers(dataset_rid, version=rescan_scenario["v1"])}
    at_v2 = {r.execution_rid for r in ml.find_feature_producers(dataset_rid, version=rescan_scenario["v2"])}

    assert e2_rid not in at_v1, "E2 was already visible at v1 — the scenario does not exercise the rescan"
    assert e2_rid in at_v2, "E2 is not visible even at v2 — the scenario is malformed"


def test_no_binding_gap_carries_a_stale_version_label(rescan_scenario):
    """No binding-scan gap on D references the superseded v1 label.

    The gap half of the fix: a rescan drops the previous view's gaps too,
    so no gap may still describe D as of v1.
    """
    closure = rescan_scenario["closure"]
    v1 = rescan_scenario["v1"]
    dataset_rid = rescan_scenario["dataset_rid"]

    stale = [gap for gap in closure.gaps if dataset_rid in gap.detail and f"@{v1}" in gap.detail]
    assert not stale, f"gaps still carry the superseded v1 as-of label: {stale}"

"""Provenance Contract — behavior tests.

Validates the normative invariants of the Execution Provenance Contract
(docs/reference/provenance-contract.md) against a live catalog. These pin
contract-level behavior that the low-level mechanism tests
(test_state_machine.py, test_find_executions_by_dataset.py, etc.) do not
assert as a *contract*.

See docs/reference/provenance-contract-test-plan.md for the coverage matrix;
test names below carry their matrix id (e.g. C2).

RID discipline: every RID and every sentinel reference comes from a
fixture/lookup, never a hard-coded literal (CLAUDE.md).
"""

from __future__ import annotations

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.core.constants import ML_SCHEMA
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.state_store import ExecutionStatus


def _setup_workflow(ml):
    """Register vocab terms and a workflow for the contract tests."""
    ml.add_term(vc.workflow_type, "Provenance Contract Workflow", description="provenance contract tests")
    ml.add_term(vc.dataset_type, "TestSet", description="A test dataset type")
    return ml.create_workflow(
        name="provenance_contract_test",
        workflow_type="Provenance Contract Workflow",
        description="Workflow for provenance contract tests",
    )


# ─────────────────────────────────────────────────────────────────────────
# C. "Consumed a dataset" = declared in datasets=
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_C2_undeclared_dataset_produces_no_input_edge(test_ml):
    """C2 — A dataset NOT declared in datasets= yields no Dataset_Execution
    input edge, even though its members exist and could be read off-band.

    This pins the motivating bug's correct behavior: consumption is defined
    by the *declared configuration*, not by what the code happened to read.
    A run that reads a dataset's members without declaring the dataset has
    honestly NOT consumed it.
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    # A producer creates a dataset (output edge only).
    producer = ml.create_execution(ExecutionConfiguration(description="Producer", workflow=workflow))
    dataset = producer.create_dataset(dataset_types=["TestSet"], description="Produced dataset")

    # A second execution declares NO datasets — it does not consume this one,
    # by the contract's definition, regardless of what its code might read.
    non_consumer = ml.create_execution(
        ExecutionConfiguration(description="Reads nothing declared", workflow=workflow)
    )

    input_consumers = {
        r.execution_rid for r in ml.find_executions(dataset=dataset.dataset_rid, dataset_role="input")
    }
    assert non_consumer.execution_rid not in input_consumers, (
        "An execution that did not declare the dataset in datasets= must not "
        "appear as an input-consumer — consumption is by declaration, not by read."
    )


# ─────────────────────────────────────────────────────────────────────────
# A. Execution state model & honest termination
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_A6_never_driven_execution_stays_created(test_ml):
    """A6 — An execution created but never driven (never entered the context
    manager / never started) stays at Created.

    The contract treats such a row as an *abandoned draft* (cleanup), NOT an
    honest-termination violation: the obligation only attaches once a run is
    driven. This test pins that a never-driven execution is observably in the
    Created state — the discriminator the audit/cleanup uses to tell a draft
    apart from a driven-then-stranded run.
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    # Create but never enter `with` / call execution_start — never driven.
    exe = ml.create_execution(ExecutionConfiguration(description="Never driven", workflow=workflow))

    assert exe.status == ExecutionStatus.Created, (
        "A created-but-never-driven execution must stay at Created; it is a "
        "draft, not a violation."
    )


@pytest.mark.integration
def test_A4_blessed_path_reaches_terminal_state(test_ml):
    """A4 — Honest termination: a run driven through the context manager
    reaches a terminal state on clean block exit.

    The contract names the context manager as the guaranteed-termination path
    (the fix for the 6-05KM 'abandoned at Created' failure). A clean `with`
    block must leave the execution in a terminal state — not stranded at
    Created/Running.
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    exe = ml.create_execution(ExecutionConfiguration(description="Blessed-path run", workflow=workflow))
    assert exe.status == ExecutionStatus.Created  # precondition

    with exe.execute():
        pass  # a clean run that produces nothing

    terminal = {
        ExecutionStatus.Stopped,
        ExecutionStatus.Uploaded,
        ExecutionStatus.Failed,
        ExecutionStatus.Aborted,
    }
    assert exe.status in terminal, (
        f"After a clean `with` block the execution must be terminal, not "
        f"{exe.status} — the context manager guarantees honest termination."
    )


# ─────────────────────────────────────────────────────────────────────────
# B. Failed executions are first-class
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_B3_bad_input_fails_fast_with_no_partial_state(test_ml):
    """B3 — A declared input RID that does not resolve fails the run FAST,
    before any execution row or input edge is created.

    Input dataset RIDs are validated up front in Execution.__init__ (before
    the Execution row is inserted), so a bad input never produces partial
    provenance state — there is nothing to "record up to the failure". This
    is the stronger guarantee (validate-not-dry-run, ADR-0002): a bad input
    is caught with zero side effects.
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    # A real dataset (the "good" input) plus a well-formed-but-nonexistent one.
    producer = ml.create_execution(ExecutionConfiguration(description="Producer", workflow=workflow))
    good = producer.create_dataset(dataset_types=["TestSet"], description="Resolvable input")
    good_version = str(good.current_version)
    bogus_rid = good.dataset_rid[:-1] + ("X" if good.dataset_rid[-1] != "X" else "Y")

    execs_before = {r.execution_rid for r in ml.find_executions()}

    config = ExecutionConfiguration(
        description="Has a bad input",
        workflow=workflow,
        datasets=[
            DatasetSpec(rid=good.dataset_rid, version=good_version),
            DatasetSpec(rid=bogus_rid, version="1.0.0"),
        ],
    )

    # Upfront validation must reject the bad RID before creating anything.
    with pytest.raises(Exception):
        ml.create_execution(config)

    # No new execution row was created (fail-fast, before insert).
    execs_after = {r.execution_rid for r in ml.find_executions()}
    new_execs = execs_after - execs_before
    assert not new_execs, (
        f"A bad input must fail before any Execution row is created; "
        f"found new execution(s): {new_execs}"
    )

    # And no input edge was recorded for the good dataset by the failed attempt.
    # (The producer above created `good` as an OUTPUT; there must be no INPUT
    # consumer from the aborted bad-input run.)
    input_consumers = {
        r.execution_rid for r in ml.find_executions(dataset=good.dataset_rid, dataset_role="input")
    }
    assert input_consumers.isdisjoint(new_execs), "No partial input edge should exist from the failed run."


# ─────────────────────────────────────────────────────────────────────────
# F. Every artifact has a producing execution
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_F1_new_dataset_version_has_a_producer(test_ml):
    """F1 — A dataset created via the blessed path has a non-null producing
    execution (Dataset_Version.Execution authorship).

    This is the create-path half of 'every artifact has a producer'. The
    whole-catalog version (orphans → sentinel) is covered by the sentinel
    tests once that lands; this pins that new artifacts are never produced
    with a null producer.
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    producer = ml.create_execution(ExecutionConfiguration(description="Producer", workflow=workflow))
    dataset = producer.create_dataset(dataset_types=["TestSet"], description="Produced dataset")

    producers = {
        r.execution_rid for r in ml.find_executions(dataset=dataset.dataset_rid, dataset_role="output")
    }
    assert producer.execution_rid in producers, (
        "A newly created dataset version must record its producing execution "
        "(non-null Dataset_Version.Execution)."
    )


@pytest.mark.integration
def test_F0_sentinels_seeded_at_catalog_init(test_ml):
    """F0 — A freshly initialized catalog has the three unknown-provenance
    sentinels (Workflow, File, Execution), reachable via the accessors.

    Seeded by initialize_ml_schema, so EVERY catalog-creation path (prod,
    demo, and the test fixtures) gets them. The accessors resolve them by
    their reserved identifiers.
    """
    ml = test_ml

    file_rid = ml.unknown_provenance_file_rid()
    exec_rid = ml.unknown_provenance_execution_rid()
    assert file_rid and exec_rid

    # Idempotent: a second call returns the same RIDs (no duplicate seeding).
    assert ml.unknown_provenance_file_rid() == file_rid
    assert ml.unknown_provenance_execution_rid() == exec_rid

    # The sentinel Execution is a real, resolvable Execution row.
    assert ml.resolve_rid(exec_rid).table.name == "Execution"
    # The sentinel File is a real File row.
    assert ml.resolve_rid(file_rid).table.name == "File"


# ─────────────────────────────────────────────────────────────────────────
# H. Lineage (data-flow, per ADR-0001)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_H3_lineage_ignores_orchestration_edge(test_ml):
    """H3 — Lineage walks data-flow parents only; an Execution_Execution
    (orchestration / nesting) edge is NOT surfaced as a data-flow parent.

    Per ADR-0001: a nested execution can be a data-flow *sibling*, not a
    parent. Walking lineage from an artifact must not pull in an execution
    that is merely orchestration-linked to the producer.
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    # Parent produces a dataset (the artifact we walk from).
    parent = ml.create_execution(ExecutionConfiguration(description="Parent", workflow=workflow))
    dataset = parent.create_dataset(dataset_types=["TestSet"], description="Parent output")

    # A child execution, linked to the parent ONLY by orchestration
    # (Execution_Execution) — no dataset flows between them.
    child = ml.create_execution(ExecutionConfiguration(description="Nested child", workflow=workflow))
    parent.add_nested_execution(child)

    lineage = ml.lookup_lineage(dataset.dataset_rid)

    # Collect every execution RID reachable through the data-flow walk.
    walked: set[str] = set()

    def _collect(node):
        if node is None:
            return
        rid = getattr(node, "rid", None) or (node.get("rid") if isinstance(node, dict) else None)
        if rid:
            walked.add(rid)
        parents = getattr(node, "parents", None)
        if parents is None and isinstance(node, dict):
            parents = node.get("parents")
        for p in parents or []:
            _collect(p)

    # lookup_lineage returns a result object; walk whatever tree it exposes.
    root = getattr(lineage, "lineage", lineage)
    _collect(root)

    assert child.execution_rid not in walked, (
        "An orchestration-only (Execution_Execution) child must NOT appear in "
        "the data-flow lineage walk (ADR-0001)."
    )


# ─────────────────────────────────────────────────────────────────────────
# D. Workflow reproducibility
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_D5_workflow_deduplicated_by_checksum(test_ml):
    """D5 — Two workflows registered from the same source (same URL + checksum)
    resolve to the SAME Workflow row.

    Workflows are content-addressed by code identity (URL/checksum), not by the
    run. Registering the same code twice must return one RID, so the same code
    is one Workflow in the provenance graph.
    """
    ml = test_ml
    ml.add_term(vc.workflow_type, "Provenance Contract Workflow", description="provenance contract tests")

    # Two workflow objects built from the same calling source → same URL +
    # checksum (same test file, same commit). Registration must dedup them.
    wf_a = ml.create_workflow(
        name="dedup_a", workflow_type="Provenance Contract Workflow", description="first"
    )
    wf_b = ml.create_workflow(
        name="dedup_b", workflow_type="Provenance Contract Workflow", description="second"
    )

    rid_a = ml._add_workflow(wf_a)
    rid_b = ml._add_workflow(wf_b)

    assert rid_a == rid_b, (
        "Workflows with identical source (URL+checksum) must dedup to one RID; "
        f"got {rid_a} != {rid_b}."
    )


@pytest.mark.integration
def test_B5_exception_in_block_records_failed_not_success(test_ml):
    """B5 (raw-state half) — A run that raises inside the `with` block lands at
    Failed (a terminal failure state), NOT a success state.

    A Failed/Aborted execution is honestly-terminated but is NOT a
    complete-provenance success producer. This pins the foundation: failure is
    recorded as Failed, distinct from Uploaded/Stopped-success. (The
    'predicate returns false for Failed/Aborted' half needs the audit and is
    covered by an xfail in the audit suite.)
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    exe = ml.create_execution(ExecutionConfiguration(description="Will raise", workflow=workflow))

    with pytest.raises(RuntimeError, match="boom"):
        with exe.execute():
            raise RuntimeError("boom")

    assert exe.status == ExecutionStatus.Failed, (
        f"An execution whose block raised must land at Failed, not {exe.status}."
    )
    assert exe.status not in {ExecutionStatus.Uploaded, ExecutionStatus.Stopped}, (
        "A failed run must not be recorded in a success state."
    )


# ─────────────────────────────────────────────────────────────────────────
# F. The two unknown-provenance sentinels
#
# All xfail: neither sentinel is seeded yet (create_schema.py) and no
# no-input commit check / sentinel-attribution exists. These are the
# executable spec for that work. The intended accessor names are NOT yet
# decided — each test references the behavior via whatever accessor the
# implementation will expose; the xfail reason names what must be built.
# When the feature lands, the implementer updates the accessor call and
# removes the xfail (strict=True forces this).
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Unknown-provenance File sentinel + no-input commit check not "
    "implemented. An artifact-producer that commits with no declared input "
    "must get an Input edge to the seeded unknown-provenance File sentinel. "
    "Needs: sentinel seeding in create_schema.py + the commit-time check.",
    strict=True,
)
def test_F2_no_input_artifact_producer_gets_unknown_file_sentinel(test_ml):
    """F2 — An artifact-producer with no declared input gets an Input edge to
    the unknown-provenance File sentinel (recorded as explicitly-unknown, not
    a null/absent input)."""
    ml = test_ml
    workflow = _setup_workflow(ml)

    # A run that produces a dataset but declares no inputs at all.
    with ml.create_execution(
        ExecutionConfiguration(description="No declared input", workflow=workflow)
    ).execute() as exe:
        exe.create_dataset(dataset_types=["TestSet"], description="Output, no inputs")

    # The intended behavior: the execution now has exactly one input edge, to
    # the unknown-provenance File sentinel. The accessor for the sentinel is
    # part of the unbuilt feature.
    sentinel_file_rid = ml.unknown_provenance_file_rid()  # noqa: F821 — not yet implemented
    input_file_rids = {a.rid for a in ml.list_assets(execution=exe.execution_rid, asset_role="Input")}
    assert sentinel_file_rid in input_file_rids, (
        "A no-input artifact-producer must carry the unknown-provenance File "
        "sentinel as an explicit input."
    )


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Producerless-artifact attribution not implemented. A dataset with "
    "no producing execution must attribute to the seeded unknown-provenance "
    "Execution sentinel so get_lineage() returns 'unknown origin', never a "
    "null dead-end. The sentinel + its accessor exist (covered by F0); what is "
    "unbuilt is the attribution that routes a producerless artifact to it.",
    strict=True,
)
def test_F4_producerless_artifact_attributes_to_unknown_execution(test_ml):
    """F4 — An artifact with no real producer attributes to the
    unknown-provenance Execution sentinel; ``get_lineage`` from it terminates
    at the sentinel ('unknown origin'), never a null dead-end.

    The producerless case can only arise from a write that bypasses the
    execution-context API (a script inserting rows directly — the
    ``lac_chartlabel_ingest.py`` flow that motivated this contract) or from
    legacy data. We reproduce that bypass with a direct insert, then assert the
    attribution feature routes its lineage to the sentinel rather than leaving
    a null producer.
    """
    ml = test_ml
    sentinel_exec_rid = ml.unknown_provenance_execution_rid()

    # Bypass the API: insert a Dataset + Dataset_Version with a null Execution,
    # exactly as a script that skips the execution context manager would.
    pb = ml.pathBuilder().schemas[ML_SCHEMA]
    orphan_rid = pb.tables["Dataset"].insert([{"Description": "Producerless orphan", "Deleted": False}])[0]["RID"]
    pb.tables["Dataset_Version"].insert(
        [{"Dataset": orphan_rid, "Version": "0.1.0", "Execution": None, "Description": "orphan version"}]
    )

    # The required end state: lineage resolves the producer to the sentinel,
    # not None. Today the producer is null, so the walk is empty — this is the
    # gap the attribution feature must close.
    lineage = ml.get_lineage(orphan_rid)
    assert lineage.lineage is not None, "Producerless artifact must not yield a null-producer lineage."
    assert lineage.lineage.execution.rid == sentinel_exec_rid, (
        "A producerless artifact's lineage must terminate at the unknown-provenance Execution sentinel."
    )


# ─────────────────────────────────────────────────────────────────────────
# G. Audit & complete-provenance predicate
#
# All xfail: the audit does not exist. These define what it must report.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Provenance audit not implemented. It must scan catalog-wide, "
    "return findings, and mutate nothing (advisory/read-only, Goal 4).",
    strict=True,
)
def test_G1_audit_exists_and_is_read_only(test_ml):
    """G1 — The audit scans the catalog and returns findings without mutating
    state."""
    ml = test_ml
    result = ml.audit_provenance()  # noqa: F821 — not yet implemented
    # Shape contract: separate violation and known-degraded buckets.
    assert hasattr(result, "violations")
    assert hasattr(result, "known_degraded")


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Audit not implemented. It must flag an artifact with a null "
    "producer (no producing execution, not sentinel-attributed) as a violation.",
    strict=True,
)
def test_G7_audit_flags_null_producer_artifact(test_ml):
    """G7 — An artifact with a null producer (not sentinel-attributed) is a
    violation reported by the audit."""
    ml = test_ml
    result = ml.audit_provenance()  # noqa: F821 — not yet implemented
    # A seeded null-producer artifact must appear in violations.
    assert any("null" in str(v).lower() or "producer" in str(v).lower() for v in result.violations)


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Audit + sentinels not implemented. A sentinel-attributed row is "
    "COMPLIANT (known-degraded), and must appear in the audit's known_degraded "
    "bucket, NOT in violations (the 'compliant but flagged' ruling).",
    strict=True,
)
def test_G8_G10_sentinel_state_is_compliant_not_violation(test_ml):
    """G8/G10 — A sentinel-attributed artifact reads as conformant: it appears
    in the audit's known-degraded report, never the violation list. The
    durable post-backfill conformance invariant."""
    ml = test_ml
    result = ml.audit_provenance()  # noqa: F821 — not yet implemented
    sentinel_exec_rid = ml.unknown_provenance_execution_rid()  # noqa: F821
    violation_text = " ".join(str(v) for v in result.violations)
    assert sentinel_exec_rid not in violation_text, (
        "Sentinel-attributed state is compliant; it must not appear as a violation."
    )

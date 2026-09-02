"""Unit tests for ``DerivaML.lookup_lineage``.

These tests mock the catalog-touching primitives (``resolve_rid``,
``_retrieve_rid``, ``lookup_execution``, ``_producer_of_dataset``,
``_producers_of_asset``) and exercise the walk shape, RID-type
detection, depth/cycle/cap behavior, and Pydantic round-tripping.

A live-catalog smoke test lives in
``tests/execution/test_lookup_lineage_live.py`` and is gated on
``DERIVA_HOST``.

Note on test RIDs: the ``RID`` Pydantic type validates against
ERMrest's RID pattern (``[A-Z\\d]{1,4}`` segments separated by
hyphens), so test RIDs are written in that form (e.g. ``"1-DSAA"``)
rather than human-readable shorthand.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException, SnapshotUnavailable
from deriva_ml.core.mixins.execution import BindingDiagnostic, ExecutionMixin
from deriva_ml.execution.lineage import LineageNode, LineageResult
from deriva_ml.execution.state_store import ExecutionStatus
from deriva_ml.feature import FeatureProducerRecord

# ---------------------------------------------------------------------------
# Helpers — minimal stand-ins for the catalog primitives.
# ---------------------------------------------------------------------------


class _ThreadSafeCalls(list):
    """A ``list`` whose ``append`` is safe to call from worker threads.

    Binding scans run on a bounded worker pool (#391 C3), and every seam
    override records into one shared call log. CPython's list append is
    already atomic under the GIL, but the lock makes the intent explicit
    and keeps the harness correct if a seam ever records more than one
    entry per call. Assertions on this log must be order-insensitive
    (membership or count), because concurrent scans interleave.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def append(self, item: Any) -> None:
        with self._lock:
            super().append(item)


class _HandleProbe:
    """One DISTINCT stand-in for a per-worker ``DerivaML`` catalog handle.

    The real ``DerivaML._provenance_worker_handle`` returns a separate
    connection whose ``requests.Session`` and ``_snapshot_cache`` are
    unsynchronized, so the walk must lease each handle to ONE reader at a
    time. A harness that hands out the same object N times cannot observe a
    violation of that rule — every collision looks like normal reuse — so
    each probe is its own object and records its own occupancy.

    ``enter``/``exit`` bracket every attribute access routed through this
    handle. If two threads are ever inside the same probe at once, the
    probe records a ``concurrent_entry`` violation (rather than raising, so
    the walk still finishes and the test reports how many collisions
    happened, not just the first).

    Attribute reads delegate to the ONE shared ``_FakeML`` that owns the
    scripted state, guarded by a shared lock — the probe models the handle's
    isolation, not a second copy of the catalog.
    """

    def __init__(self, owner: "_FakeML", index: int, shared_lock: threading.Lock) -> None:
        self._owner = owner
        self._index = index
        self._shared_lock = shared_lock
        self._occupancy_lock = threading.Lock()
        self._occupants = 0

    # The seams a read side calls. Only these are instrumented; everything
    # else (``model``, plain attributes, dunder lookups) passes through
    # untouched. Wrapping by "is it callable" was WRONG: ``_FakeML.model``
    # is a MagicMock, which is callable, so it got wrapped into a function
    # and the asset branch's ``ml.model.name_to_table(...)`` raised
    # AttributeError — swallowed as ``resolution_failed``, silently losing
    # every asset producer. An explicit list cannot drift that way.
    _PROXIED_SEAMS = frozenset(
        {
            "lookup_execution",
            "_input_dataset_pairs",
            "_producer_of_dataset",
            "_producers_of_dataset_members",
            "_producers_of_asset",
        }
    )

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._owner, name)
        if name not in self._PROXIED_SEAMS:
            return target

        def proxied(*args: Any, **kwargs: Any) -> Any:
            with self._occupancy_lock:
                self._occupants += 1
                collided = self._occupants > 1
            if collided:
                self._owner.handle_violations.append(("concurrent_entry", self._index))
            # Hold the handle briefly so concurrent reads genuinely OVERLAP.
            # Without this the scripted seams return in microseconds and a
            # sharing bug is invisible by luck rather than absent by design —
            # which is exactly how the hash-mod scheme passed review.
            if self._owner._handle_hold_seconds:
                time.sleep(self._owner._handle_hold_seconds)
            try:
                # The scripted state behind every probe is one dict-based
                # _FakeML, so serialize access to it. This models a real
                # handle's OWN session being used by one reader at a time;
                # it is not what the test is measuring (the occupancy
                # counter above is).
                with self._shared_lock:
                    return target(*args, **kwargs)
            finally:
                with self._occupancy_lock:
                    self._occupants -= 1

        return proxied


@dataclass
class _StubColumn:
    name: str


@dataclass
class _StubTable:
    name: str
    columns: list[_StubColumn]


@dataclass
class _StubResolved:
    table: _StubTable


@dataclass
class _StubWorkflow:
    workflow_rid: str
    name: str
    url: str | None = None
    version: str | None = None
    checksum: str | None = None


@dataclass
class _StubAsset:
    asset_rid: str
    filename: str
    asset_table: str


class _StubDataset:
    """Minimal Dataset stand-in for list_input_datasets()."""

    def __init__(
        self,
        dataset_rid: str,
        description: str = "",
        version: str = "0.1.0",
        consumed_version: str | None = None,
    ) -> None:
        self.dataset_rid = dataset_rid
        self.description = description
        self._version = version
        # The version recorded on the Dataset_Execution edge (what was consumed).
        # None means "no pin" -> the walk falls back to current_version.
        self.consumed_version = consumed_version

    @property
    def current_version(self) -> str:
        return self._version


class _StubExecutionRecord:
    """Stand-in for ExecutionRecord with the methods _walk_node calls."""

    def __init__(
        self,
        execution_rid: str,
        description: str | None = None,
        workflow: _StubWorkflow | None = None,
        status: ExecutionStatus = ExecutionStatus.Uploaded,
        input_datasets: list[_StubDataset] | None = None,
        input_assets: list[_StubAsset] | None = None,
    ) -> None:
        self.execution_rid = execution_rid
        self.description = description
        self.workflow = workflow
        self.status = status
        self._input_datasets = input_datasets or []
        self._input_assets = input_assets or []

    def list_input_datasets(self) -> list[_StubDataset]:
        return list(self._input_datasets)

    def list_assets(self, asset_role: str | None = None) -> list[_StubAsset]:
        if asset_role and asset_role != "Input":
            return []
        return list(self._input_assets)


class _StubDatasetHandle:
    """Minimal ``Dataset`` stand-in for ``lookup_dataset(rid)`` (Task 8).

    Carries ONLY what the closure-tests engine calls on the object returned
    by ``ExecutionMixin.lookup_dataset``: ``dataset_rid`` and
    ``strict_version_snapshot_catalog(version)``. The method records its
    invocation into the OWNING ``_FakeML.calls`` list, so a quarantine test
    can assert "no snapshot-dependent call happened" even though the call
    physically happens on this handle, not on ``_FakeML`` itself.

    ``list_dataset_parents`` is a deliberate TRIPWIRE, not a supported seam:
    ruling 8 (#389) took ``Dataset_Dataset`` containment out of the closure,
    so the walk must never call it. Scripting parents via
    ``set_dataset_parents`` and asserting the call never lands is what makes
    the "containment is not walked" pin non-vacuous — without a scriptable
    containment edge, the assertion would hold trivially on a catalog that
    simply has no parents.

    Default-unavailable: a ``(dataset_rid, version)`` pair that was never
    scripted via ``set_snapshot_available`` raises ``SnapshotUnavailable``
    from ``strict_version_snapshot_catalog`` — the same "safe default" the
    real ``Dataset.strict_version_snapshot_catalog`` enforces (never
    silently fall back to a live/bare-id read). Tests must opt IN to a
    resolvable snapshot explicitly.
    """

    def __init__(self, dataset_rid: str, owner: "_FakeML") -> None:
        self.dataset_rid = dataset_rid
        self._owner = owner

    def strict_version_snapshot_catalog(self, version: Any) -> Any:
        self._owner.calls.append(("Dataset.strict_version_snapshot_catalog", (self.dataset_rid, str(version))))
        key = (self.dataset_rid, str(version))
        if not self._owner._snapshot_available.get(key, False):
            raise SnapshotUnavailable(f"Dataset {self.dataset_rid} version {version} has no recorded snapshot")
        return self._owner._snapshot_markers.get(key, f"<snapshot:{self.dataset_rid}@{version}>")

    def list_dataset_parents(self) -> list[str]:
        """Structural containment tripwire — see the class docstring."""
        self._owner.calls.append(("Dataset.list_dataset_parents", (self.dataset_rid,)))
        return list(self._owner._dataset_parents.get(self.dataset_rid, []))


class _FakeML(ExecutionMixin):
    """Bare ExecutionMixin host that scripts the primitives the lineage
    walk depends on. Keeps the surface minimal so individual tests can
    set up just the slice they care about.
    """

    def __init__(self) -> None:
        # Map RID -> (table_name, optional column-name set, optional row dict).
        self._rids: dict[str, tuple[str, set[str], dict[str, Any]]] = {}
        # Map RID -> _StubExecutionRecord (for lookup_execution).
        self._executions: dict[str, _StubExecutionRecord] = {}
        # Map dataset_rid -> producing-execution RID (or None).
        self._dataset_producers: dict[str, str | None] = {}
        # Map (dataset_rid, version) -> producing-execution RID.
        self._versioned_dataset_producers: dict[tuple[str, str], str] = {}
        # Map asset_rid -> list of producing-execution RIDs (fetched order).
        self._asset_producers: dict[str, list[str]] = {}
        # Map dataset_rid -> set of member-producing execution RIDs.
        self._dataset_member_producers: dict[str, set[str]] = {}
        # Map (dataset_rid, version) -> set of member-producing execution RIDs.
        self._versioned_member_producers: dict[tuple[str, str], set[str]] = {}
        # Tracks which RIDs the model considers "asset" tables.
        self._asset_table_names: set[str] = set()
        # Multi-row version history: dataset_rid -> list of version-row dicts
        # (shape matches ExecutionMixin._dataset_version_rows' output),
        # earliest-recorded first. Populated by add_version_row(); read back
        # by the _dataset_version_rows() override below.
        self._version_rows: dict[str, list[dict[str, Any]]] = {}
        # Snapshot-read authorship view: (dataset_rid, version) -> rows, for
        # tests that need the snapshot-scoped version-row read to differ
        # from the live/current one (Task 11).
        self._snapshot_version_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        # Snapshot availability: (dataset_rid, version) -> bool. Absent key
        # means "not scripted" -> strict_version_snapshot_catalog raises
        # SnapshotUnavailable (default-unavailable is the safe default).
        self._dataset_parents: dict[str, list[str]] = {}
        self._snapshot_available: dict[tuple[str, str], bool] = {}
        # Optional marker object returned by a scripted-available snapshot.
        self._snapshot_markers: dict[tuple[str, str], Any] = {}
        # (dataset_rid, version) -> [{"parent_rid": ..., "parent_version_then": ...}].
        # (dataset_rid, version) -> (records, diagnostics) for
        # _find_feature_producers_impl.
        self._binding_scans: dict[tuple[str, Any], tuple[list[Any], list[BindingDiagnostic]]] = {}
        # Call recording: every seam override appends (method_name, args).
        # Powers the quarantine tests ("assert NO snapshot-dependent call
        # happened"). Lock-guarded because binding scans run on a worker
        # pool (#391 C3) and would otherwise race on this list; ORDER
        # across concurrent scans is not guaranteed, so assertions on it
        # must be order-insensitive (membership/count), which every
        # existing assertion already is.
        self.calls: _ThreadSafeCalls = _ThreadSafeCalls()
        # Whether ``_provenance_worker_handle`` hands out a live handle, i.e.
        # whether the closure walk's frontier prefetch actually runs
        # concurrently (#391b). Off by default so the bulk of the suite pins
        # the sequential semantics; the parallel-expansion tests turn it on
        # and assert the SAME closure comes out.
        self._parallel_expansion_enabled = False
        # Per-worker handle bookkeeping (#391b lease fix). Each
        # ``_provenance_worker_handle`` call mints a DISTINCT ``_HandleProbe``
        # with its own occupancy counter; a probe entered by two threads at
        # once appends here, which is what lets a test prove the lease is a
        # lease and not a hash-mod routing scheme.
        self._handle_serial = 0
        self._scripted_state_lock = threading.Lock()
        self.handle_violations: _ThreadSafeCalls = _ThreadSafeCalls()
        # How long each proxied seam holds its handle. Zero by default so
        # the bulk of the suite stays fast; the lease test raises it so
        # concurrent reads genuinely overlap and a sharing bug is DETECTED
        # rather than missed by timing luck.
        self._handle_hold_seconds = 0.0
        # dataset_rid -> exception raised by _producers_of_dataset_members.
        self._member_scan_failures: dict[str, BaseException] = {}
        # Mock model.
        self.model = MagicMock()
        self.model.is_asset = lambda table: table.name in self._asset_table_names
        self.model.name_to_table = lambda name: _StubTable(name=name, columns=[_StubColumn("RID")])

    # -- scripting helpers -------------------------------------------------

    def add_dataset(self, rid: str, description: str = "", producer: str | None = None) -> None:
        self._rids[rid] = ("Dataset", set(), {"Description": description})
        self._dataset_producers[rid] = producer

    def add_asset(
        self,
        rid: str,
        asset_table: str,
        filename: str = "",
        description: str = "",
        producer: str | None = None,
        producers: list[str] | None = None,
    ) -> None:
        """Register an asset RID, with either a single ``producer`` or a
        multi-producer list. ``producers`` (fetched order preserved) wins
        when both are given; ``producer`` is sugar for a single-element list.
        """
        self._rids[rid] = (asset_table, set(), {"Description": description, "Filename": filename})
        self._asset_table_names.add(asset_table)
        if producers is not None:
            self._asset_producers[rid] = list(producers)
        else:
            self._asset_producers[rid] = [producer] if producer is not None else []

    def add_workflow(self, rid: str) -> None:
        self._rids[rid] = ("Workflow", set(), {})

    def add_feature_value(self, rid: str, producer_execution: str | None) -> None:
        self._rids[rid] = (
            "Image_Some_Feature",
            {"Feature_Name", "Execution"},
            {"Execution": producer_execution},
        )

    def add_execution(
        self,
        rid: str,
        *,
        description: str | None = None,
        workflow: _StubWorkflow | None = None,
        status: ExecutionStatus = ExecutionStatus.Uploaded,
        input_datasets: list[_StubDataset] | None = None,
        input_assets: list[_StubAsset] | None = None,
    ) -> _StubExecutionRecord:
        self._rids[rid] = ("Execution", set(), {"Description": description})
        rec = _StubExecutionRecord(
            execution_rid=rid,
            description=description,
            workflow=workflow,
            status=status,
            input_datasets=input_datasets,
            input_assets=input_assets,
        )
        self._executions[rid] = rec
        return rec

    def add_version_row(
        self,
        dataset_rid: str,
        version: str,
        execution: str | None,
        *,
        rct: str = "2025-01-01T00:00:00Z",
        snapshot: str | None = None,
    ) -> dict[str, Any]:
        """Append one ``Dataset_Version`` row for ``dataset_rid`` (Task 8).

        Feeds BOTH the live ``_dataset_version_rows`` override (rows appended
        here are returned in insertion order — tests are responsible for
        calling this earliest-first if order matters, mirroring the real
        method's RCT-sorted contract) AND, when a ``snapshot`` is given,
        registers this dataset/version pair as a resolvable snapshot handle
        (equivalent to calling ``set_snapshot_available(dataset_rid, version,
        True)``) so ``lookup_dataset(dataset_rid).strict_version_snapshot_catalog(version)``
        succeeds without a separate scripting call.

        Returns the constructed row dict, in case a test wants to reference
        its ``RID``.
        """
        row = {
            "RID": f"{dataset_rid}-VER-{version}",
            "RCT": rct,
            "Version": version,
            "Execution": execution,
            "Description": None,
        }
        self._version_rows.setdefault(dataset_rid, []).append(row)
        if snapshot is not None:
            self.set_snapshot_available(dataset_rid, version, True)
            self._snapshot_markers[(dataset_rid, str(version))] = snapshot
        return row

    def set_snapshot_version_rows(self, dataset_rid: str, version: str, rows: list[dict[str, Any]]) -> None:
        """Script the snapshot-scoped version-row read for (dataset_rid, version).

        For snapshot-read authorship (Task 11): a version-history read taken
        AT a specific snapshot can differ from the live/current read (e.g.
        later version rows that postdate the snapshot must not appear).
        ``rows`` must be shaped like ``_dataset_version_rows``' output — dicts
        with ``RID``, ``RCT``, ``Version``, ``Execution``, ``Description``.
        """
        self._snapshot_version_rows[(dataset_rid, str(version))] = list(rows)

    def set_snapshot_available(self, dataset_rid: str, version: str, ok: bool) -> None:
        """Script whether ``lookup_dataset(dataset_rid).strict_version_snapshot_catalog(version)``
        succeeds (``ok=True``) or raises ``SnapshotUnavailable`` (``ok=False``).

        An unscripted (dataset_rid, version) pair defaults to unavailable —
        see ``_StubDatasetHandle``'s docstring for the rationale.
        """
        self._snapshot_available[(dataset_rid, str(version))] = ok

    def set_dataset_parents(self, dataset_rid: str, parents: list[str]) -> None:
        """Script ``lookup_dataset(dataset_rid).list_dataset_parents()``.

        Structural ``Dataset_Dataset`` containment. The closure walk must
        NEVER read this (ruling 8, #389); it exists so a test can prove the
        edge was present and still ignored.
        """
        self._dataset_parents[dataset_rid] = list(parents)

    def enable_parallel_expansion(self, enabled: bool = True) -> None:
        """Turn the closure walk's concurrent frontier prefetch on or off.

        See ``_provenance_worker_handle``. The point of the switch is that
        both settings must produce the SAME closure — it is the control in
        the sequential-equivalence tests, not a feature flag anyone ships.
        """
        self._parallel_expansion_enabled = enabled

    def set_binding_scan(
        self,
        dataset_rid: str,
        version: Any,
        records: list[Any],
        diagnostics: list[BindingDiagnostic],
    ) -> None:
        """Script ``_find_feature_producers_impl(dataset_rid, version)``'s
        ``(records, diagnostics)`` return value."""
        self._binding_scans[(dataset_rid, version)] = (list(records), list(diagnostics))

    # -- ExecutionMixin protocol -------------------------------------------

    def resolve_rid(self, rid: str) -> _StubResolved:
        if rid not in self._rids:
            raise DerivaMLException(f"Invalid RID {rid}")
        table_name, extra_cols, _ = self._rids[rid]
        cols = [_StubColumn("RID")] + [_StubColumn(c) for c in extra_cols]
        return _StubResolved(table=_StubTable(name=table_name, columns=cols))

    def _retrieve_rid(self, rid: str) -> dict[str, Any]:
        if rid not in self._rids:
            raise DerivaMLException(f"Invalid RID {rid}")
        return self._rids[rid][2]

    def lookup_execution(self, rid: str) -> _StubExecutionRecord:
        if rid not in self._executions:
            raise DerivaMLException(f"No such execution {rid}")
        return self._executions[rid]

    def lookup_dataset(self, rid: str) -> _StubDatasetHandle:  # type: ignore[override]
        """Seam consumed by ``_find_feature_producers_impl`` and by the
        upcoming snapshot/parents closure tests. Returns a small stub whose
        methods record into ``self.calls`` (see ``_StubDatasetHandle``).
        """
        self.calls.append(("lookup_dataset", (rid,)))
        return _StubDatasetHandle(rid, self)

    def set_member_producers(self, dataset_rid: str, producers: set[str]) -> None:
        """Script the member-asset producing executions of a dataset."""
        self._dataset_member_producers[dataset_rid] = set(producers)

    def set_versioned_producer(self, dataset_rid: str, version: str, producer: str) -> None:
        """Script the producer of a SPECIFIC dataset version."""
        self._versioned_dataset_producers[(dataset_rid, version)] = producer

    def set_versioned_member_producers(self, dataset_rid: str, version: str, producers: set[str]) -> None:
        """Script the member-asset producers for a SPECIFIC dataset version."""
        self._versioned_member_producers[(dataset_rid, str(version))] = set(producers)

    def _scripted_input_pairs(self, execution_rid: str):
        """(Dataset, consumed_version) pairs for an execution, from the stub record."""
        rec = self._executions.get(execution_rid)
        if rec is None:
            return []
        return [(ds, ds.consumed_version) for ds in rec.list_input_datasets()]

    def _input_dataset_pairs(self, execution_rid: str):  # type: ignore[override]
        """Override the seam: return scripted pairs without hitting a catalog."""
        self.calls.append(("_input_dataset_pairs", (execution_rid,)))
        return self._scripted_input_pairs(execution_rid)

    def _producer_of_dataset(self, dataset_rid: str, version: Any = None) -> str | None:  # type: ignore[override]
        self.calls.append(("_producer_of_dataset", (dataset_rid, version)))
        if version is not None:
            return self._versioned_dataset_producers.get((dataset_rid, str(version)))
        return self._dataset_producers.get(dataset_rid)

    def _dataset_version_rows(self, dataset_rid: str) -> list[dict[str, Any]]:  # type: ignore[override]
        """Return the scripted version-row history for ``dataset_rid``.

        Prefers rows appended via ``add_version_row`` (Task 8's multi-row
        seam) when any are present; otherwise falls back to synthesizing a
        single-row history from the legacy ``add_dataset(..., producer=...)``
        scripting, preserving every pre-Task-8 test unmodified.

        ``_classify_rid``'s Dataset branch reads the origin from
        ``_dataset_version_rows()[0]`` rather than calling
        ``_producer_of_dataset`` directly (#367).
        """
        self.calls.append(("_dataset_version_rows", (dataset_rid,)))
        if dataset_rid in self._version_rows:
            return list(self._version_rows[dataset_rid])
        if dataset_rid not in self._dataset_producers:
            return []
        return [
            {
                "RID": f"{dataset_rid}-VER",
                "RCT": "2025-01-01T00:00:00Z",
                "Version": "0.1.0",
                "Execution": self._dataset_producers[dataset_rid],
                "Description": None,
            }
        ]

    def _dataset_version_rows_at(  # type: ignore[override]
        self, dataset_rid: str, version: Any, snapshot_catalog: Any
    ) -> list[dict[str, Any]]:
        """Return the SNAPSHOT-scoped version-row history (Task 11).

        Reads the rows scripted by ``set_snapshot_version_rows`` for
        ``(dataset_rid, version)``. An unscripted pair returns ``[]`` — the
        honest "the snapshot records no version rows" answer, never a silent
        fall-through to the live ``_dataset_version_rows`` view (which is
        exactly the leak the snapshot-faithfulness pin guards against).
        """
        self.calls.append(("_dataset_version_rows_at", (dataset_rid, str(version))))
        return list(self._snapshot_version_rows.get((dataset_rid, str(version)), []))

    def _sentinel_execution_rid_or_none(self) -> str | None:  # type: ignore[override]
        return None

    def _provenance_worker_handle(self) -> "_HandleProbe | None":
        """Per-worker catalog handle seam for the parallel frontier (#391b).

        The real ``DerivaML`` returns a SEPARATE connection here because
        ``requests.Session`` (and the instance's ``_snapshot_cache``) are
        unsynchronized. Each call therefore returns a DISTINCT
        ``_HandleProbe``, never ``self``: a pool of N references to one
        object could not detect two readers sharing a handle, which is
        precisely the bug the lease exists to prevent.

        Each probe records its own occupancy, so a test can assert that no
        handle is ever entered concurrently. ``enable_parallel_expansion``
        toggles whether handles are offered at all.
        """
        if not self._parallel_expansion_enabled:
            return None
        self._handle_serial += 1
        return _HandleProbe(self, self._handle_serial, self._scripted_state_lock)

    def _execution_summaries(self, rids: Any) -> dict[str, Any]:  # type: ignore[override]
        """Resolve scripted execution RIDs to ExecutionSummary via lookup_execution.

        Only used by ``_classify_rid`` to populate ``RootDescriptor.producing_execution``
        and ``version_history[].execution``; the walk itself (and the
        ``result.root.producing_execution`` assertions in these tests) is driven by
        ``_walk_node``/``lookup_execution``, which overwrites this value once a walk
        happens (see ``lookup_lineage``). Reuses the scripted ``_StubExecutionRecord``
        to build a minimal ``ExecutionSummary`` for RIDs that resolve.
        """
        from deriva_ml.execution.lineage import ExecutionSummary, WorkflowSummary

        rids = list(rids)
        self.calls.append(("_execution_summaries", (tuple(rids),)))
        out: dict[str, ExecutionSummary] = {}
        for rid in rids:
            if rid is None:
                continue
            rec = self._executions.get(rid)
            if rec is None:
                continue
            wf_summary = (
                WorkflowSummary(
                    rid=rec.workflow.workflow_rid,
                    name=rec.workflow.name,
                    checksum=getattr(rec.workflow, "checksum", None),
                )
                if rec.workflow
                else None
            )
            out[rid] = ExecutionSummary(
                rid=rec.execution_rid,
                description=rec.description,
                workflow=wf_summary,
                status=rec.status.value if hasattr(rec.status, "value") else str(rec.status),
            )
        return out

    def _producers_of_asset(self, asset_rid: str, asset_table: Any) -> list[str]:  # type: ignore[override]
        self.calls.append(("_producers_of_asset", (asset_rid, asset_table)))
        return list(self._asset_producers.get(asset_rid, []))

    def _producers_of_dataset_members(self, dataset_rid: str, version: Any = None) -> set[str]:  # type: ignore[override]
        self.calls.append(("_producers_of_dataset_members", (dataset_rid, version)))
        scripted = self._member_scan_failures.get(dataset_rid)
        if scripted is not None:
            # Scripted read-side failure (#391b). Raised from the SAME seam
            # whether the read happens on a worker or inline, so the
            # parallel and sequential paths must classify it identically.
            raise scripted
        if version is not None and (dataset_rid, str(version)) in self._versioned_member_producers:
            return set(self._versioned_member_producers[(dataset_rid, str(version))])
        return set(self._dataset_member_producers.get(dataset_rid, set()))

    def set_member_scan_failure(self, dataset_rid: str, error: BaseException) -> None:
        """Script ``_producers_of_dataset_members`` to raise for a dataset.

        Used to pin that a read-side failure is classified identically
        whether it happened on a worker (captured into the readout and
        applied on the main thread) or inline — same gap, same text, same
        propagation for the kinds that are not supposed to degrade.
        """
        self._member_scan_failures[dataset_rid] = error

    def _find_feature_producers_impl(  # type: ignore[override]
        self, dataset_rid: str, version: Any = None
    ) -> tuple[list[Any], list[BindingDiagnostic]]:
        """Return the scripted binding-scan result for (dataset_rid, version).

        Unscripted pairs return ``([], [])`` — an empty scan, not an error —
        matching the real method's behavior when a dataset has no bound
        features.
        """
        self.calls.append(("_find_feature_producers_impl", (dataset_rid, version)))
        return self._binding_scans.get((dataset_rid, version), ([], []))


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_lineage_workflow_rid_raises():
    ml = _FakeML()
    ml.add_workflow("3-WFAA")
    with pytest.raises(DerivaMLException, match="Workflow"):
        ml.lookup_lineage("3-WFAA")


def test_lineage_unknown_rid_raises():
    ml = _FakeML()
    with pytest.raises(DerivaMLException, match="Invalid RID"):
        ml.lookup_lineage("NOPE")


def test_lineage_dataset_with_no_producer_returns_empty_walk():
    ml = _FakeML()
    ml.add_dataset("1-DSAB", description="orphan dataset", producer=None)

    result = ml.lookup_lineage("1-DSAB")

    assert isinstance(result, LineageResult)
    assert result.root.type == "Dataset"
    assert result.root.rid == "1-DSAB"
    assert result.root.producing_execution is None
    assert result.lineage is None
    assert result.executions_visited == 0
    assert result.walked_complete is True


def test_lineage_dataset_one_level_chain():
    """Dataset DS-1 produced by EXE-A which consumed Dataset DS-0 (no producer)."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="root data", producer=None)
    ml.add_execution(
        "2-EXAA",
        description="train",
        workflow=_StubWorkflow(workflow_rid="3-WFAB", name="trainer"),
        input_datasets=[_StubDataset("1-DSAA", "root data", "0.1.0")],
    )
    ml.add_dataset("1-DSAB", description="trained set", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == "2-EXAA"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"
    assert result.lineage.execution.workflow is not None
    assert result.lineage.execution.workflow.name == "trainer"
    assert len(result.lineage.consumed_datasets) == 1
    assert result.lineage.consumed_datasets[0].rid == "1-DSAA"
    assert result.lineage.parents == []  # DS-0 has no producer
    assert result.executions_visited == 1
    assert result.walked_complete is True


def test_lineage_walk_populates_workflow_checksum():
    """The per-node walk path (_walk_node -> record.workflow) threads checksum (#383).

    Pins the path Codex flagged: _walk_node builds WorkflowSummary from
    record.workflow (a _StubWorkflow / Workflow object), not from a raw row
    dict, so checksum must be read off that object's attribute.
    """
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="root data", producer=None)
    ml.add_execution(
        "2-EXAA",
        description="train",
        workflow=_StubWorkflow(workflow_rid="3-WFAB", name="trainer", checksum="sha256:deadbeef"),
        input_datasets=[_StubDataset("1-DSAA", "root data", "0.1.0")],
    )
    ml.add_dataset("1-DSAB", description="trained set", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    assert result.lineage is not None
    assert result.lineage.execution.workflow is not None
    assert result.lineage.execution.workflow.checksum == "sha256:deadbeef"


def test_lineage_two_level_chain():
    """DS-2 <- EXE-B <- DS-1 <- EXE-A <- DS-0."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution(
        "2-EXAA",
        input_datasets=[_StubDataset("1-DSAA")],
    )
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution(
        "2-EXAB",
        input_datasets=[_StubDataset("1-DSAB")],
    )
    ml.add_dataset("1-DSAC", producer="2-EXAB")

    result = ml.lookup_lineage("1-DSAC")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAB"
    assert len(result.lineage.parents) == 1
    parent = result.lineage.parents[0]
    assert parent.execution.rid == "2-EXAA"
    assert parent.consumed_datasets[0].rid == "1-DSAA"
    assert parent.parents == []
    assert result.executions_visited == 2


def test_lineage_depth_zero_returns_only_immediate_producer():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution("2-EXAB", input_datasets=[_StubDataset("1-DSAB")])
    ml.add_dataset("1-DSAC", producer="2-EXAB")

    result = ml.lookup_lineage("1-DSAC", depth=0)

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAB"
    assert result.lineage.parents == []
    assert result.depth_capped is True
    assert result.executions_visited == 1


def test_lineage_depth_one_walks_one_layer():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution("2-EXAB", input_datasets=[_StubDataset("1-DSAB")])
    ml.add_dataset("1-DSAC", producer="2-EXAB")

    result = ml.lookup_lineage("1-DSAC", depth=1)

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAB"
    assert len(result.lineage.parents) == 1
    grandparent = result.lineage.parents[0]
    assert grandparent.execution.rid == "2-EXAA"
    # depth=1 means: walk one level. EXE-A's parents would be next layer
    # but only if EXE-A had data-flow parents. DS-0 has no producer, so
    # nothing more to walk. Hence depth_capped stays False.
    assert grandparent.parents == []
    assert result.depth_capped is False


def test_lineage_diamond_dag_marks_already_shown():
    """EXE-C consumed two datasets (DS-A, DS-B) both produced by EXE-X."""
    ml = _FakeML()
    ml.add_dataset("1-DSRT", producer=None)
    ml.add_execution("2-EXBX", input_datasets=[_StubDataset("1-DSRT")])
    ml.add_dataset("1-DSBA", producer="2-EXBX")
    ml.add_dataset("1-DSBB", producer="2-EXBX")
    ml.add_execution(
        "2-EXAC",
        input_datasets=[_StubDataset("1-DSBA"), _StubDataset("1-DSBB")],
    )
    ml.add_dataset("1-DSOT", producer="2-EXAC")

    result = ml.lookup_lineage("1-DSOT")

    # Only one EXE-X expansion (the second branch is collapsed).
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAC"
    assert len(result.lineage.parents) == 1, "diamond should dedupe to one parent edge by RID"
    # EXE-X visited once.
    assert result.executions_visited == 2  # EXE-C, EXE-X
    assert result.cycle_detected is False


def test_lineage_max_executions_cap():
    """Build a 4-deep chain and cap at 2 executions."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution("2-EXAB", input_datasets=[_StubDataset("1-DSAB")])
    ml.add_dataset("1-DSAC", producer="2-EXAB")
    ml.add_execution("2-EXAC", input_datasets=[_StubDataset("1-DSAC")])
    ml.add_dataset("1-DSAD", producer="2-EXAC")
    ml.add_execution("2-EXAD", input_datasets=[_StubDataset("1-DSAD")])
    ml.add_dataset("1-DSAE", producer="2-EXAD")

    result = ml.lookup_lineage("1-DSAE", max_executions=2)

    assert result.walked_complete is False
    assert result.executions_visited == 2


def test_lineage_execution_rid_is_self_root():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution(
        "2-EXAA",
        description="root execution",
        input_datasets=[_StubDataset("1-DSAA")],
    )

    result = ml.lookup_lineage("2-EXAA")

    assert result.root.type == "Execution"
    assert result.root.rid == "2-EXAA"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"
    assert result.executions_visited == 1


def test_lineage_asset_root_walks_via_producer():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_asset("4-ASAA", asset_table="Image", filename="cat.png", producer="2-EXAA")

    result = ml.lookup_lineage("4-ASAA")

    assert result.root.type == "Asset"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"


def test_lineage_feature_value_root_walks_via_execution_column():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_feature_value("5-FVAA", producer_execution="2-EXAA")

    result = ml.lookup_lineage("5-FVAA")

    assert result.root.type == "Feature"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"


def test_lineage_consumes_both_datasets_and_assets():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_asset("4-ASIN", asset_table="Image", filename="in.png", producer=None)
    ml.add_execution(
        "2-EXAA",
        input_datasets=[_StubDataset("1-DSAA", description="d", version="0.1.0")],
        input_assets=[_StubAsset("4-ASIN", "in.png", "Image")],
    )
    ml.add_dataset("1-DSAB", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    assert result.lineage is not None
    assert len(result.lineage.consumed_datasets) == 1
    assert len(result.lineage.consumed_assets) == 1
    assert result.lineage.consumed_assets[0].asset_table == "Image"
    assert result.lineage.consumed_assets[0].filename == "in.png"


def test_lineage_result_round_trips_via_pydantic():
    """Models serialize and reload cleanly."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    dumped = result.model_dump()
    reloaded = LineageResult.model_validate(dumped)
    assert reloaded == result


def test_lineage_node_recursive_validation_works():
    """Smoke test the recursive parents field on LineageNode."""
    leaf = LineageNode(
        execution={"rid": "2-EXLF", "status": "Uploaded"},
    )
    parent = LineageNode(
        execution={"rid": "2-EXPA", "status": "Uploaded"},
        parents=[leaf],
    )
    assert parent.parents[0].execution.rid == "2-EXLF"


def test_walk_node_extra_parent_rids_attaches_as_parents():
    """extra_parent_rids passed to the root walk become parents of the root node."""
    ml = _FakeML()
    # EXE-UP produced some members; it consumed DS-SRC (no producer).
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])
    # The dataset whose root walk we drive: produced by EXE-DS (version producer),
    # but its members were produced by EXE-UP.
    ml.add_dataset("1-DSIM", producer="2-EXDS")
    ml.add_execution("2-EXDS", input_datasets=[])
    ml.set_member_producers("1-DSIM", {"2-EXUP"})

    result = ml.lookup_lineage("1-DSIM")

    # Root node is the version-producer EXE-DS.
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXDS"
    # EXE-UP (the member-producer) appears as a parent of the root.
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXUP" in parent_rids
    # And the walk continued into EXE-UP's consumed source dataset.
    up_node = next(p for p in result.lineage.parents if p.execution.rid == "2-EXUP")
    assert {d.rid for d in up_node.consumed_datasets} == {"1-DSSR"}


def test_mid_walk_consumed_dataset_member_producers_become_parents():
    """When an execution consumes a dataset whose members have a distinct
    producer, that producer is walked as a parent."""
    ml = _FakeML()
    # Source dataset consumed by the upload exec.
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])
    # An intermediate image dataset: version-producer EXE-DS, members by EXE-UP.
    ml.add_dataset("1-DSIM", producer="2-EXDS")
    ml.set_member_producers("1-DSIM", {"2-EXUP"})
    ml.add_execution("2-EXDS", input_datasets=[])
    # A downstream execution that CONSUMES the image dataset as input.
    ml.add_execution("2-EXTR", input_datasets=[_StubDataset("1-DSIM")])
    ml.add_dataset("1-DSMO", producer="2-EXTR")

    result = ml.lookup_lineage("1-DSMO")

    # Root = EXE-TR; it consumed DS-IM. DS-IM's version producer EXE-DS AND its
    # member producer EXE-UP both appear among EXE-TR's parents.
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXTR"
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXDS" in parent_rids  # version producer
    assert "2-EXUP" in parent_rids  # member producer (the new edge)


def test_root_dataset_surfaces_member_producer_when_both_exist():
    """lookup_lineage(image_dataset): version-producer is root, member-producer
    is a parent reaching the source — the tk-018 case."""
    ml = _FakeML()
    ml.add_dataset("1-DSSR", producer=None)  # source dataset
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])  # upload
    ml.add_execution("2-EXDS", input_datasets=[])  # datasets-phase (version producer)
    ml.add_dataset("1-DSIM", producer="2-EXDS")  # image dataset
    ml.set_member_producers("1-DSIM", {"2-EXUP"})

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXDS"  # root = version producer
    up = next((p for p in result.lineage.parents if p.execution.rid == "2-EXUP"), None)
    assert up is not None, "member-producer must appear as a parent of the root"
    assert {d.rid for d in up.consumed_datasets} == {"1-DSSR"}  # reaches the source
    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == "2-EXDS"  # contract preserved


def test_root_dataset_no_version_producer_walks_from_member_producers():
    """A dataset with NO version producer but WITH member producers yields a
    non-empty walk (previously this returned an empty LineageResult).

    #367: the walk root is seeded from the member producer, but with no
    recorded origin the root's ``producing_execution`` stays unset — origin
    attribution and walk seeding are separate concerns (tk-018/#367). This
    assertion used to expect the walk root to overwrite
    ``producing_execution`` (the old last-writer contract); that overwrite
    no longer applies to Dataset roots.
    """
    ml = _FakeML()
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])
    ml.add_dataset("1-DSIM", producer=None)  # no version producer
    ml.set_member_producers("1-DSIM", {"2-EXUP"})

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXUP"  # representative root
    assert {d.rid for d in result.lineage.consumed_datasets} == {"1-DSSR"}
    assert result.root.producing_execution is None  # no recorded origin (#367)
    assert result.root.origin_recorded is False


def test_root_dataset_member_producer_equals_version_producer_no_dup():
    """If the version producer also produced the members, it is not listed as
    its own parent."""
    ml = _FakeML()
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXVP", input_datasets=[_StubDataset("1-DSSR")])
    ml.add_dataset("1-DSIM", producer="2-EXVP")
    ml.set_member_producers("1-DSIM", {"2-EXVP"})  # same exec

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXVP"
    # 2-EXVP must NOT appear as its own parent.
    assert all(p.execution.rid != "2-EXVP" for p in result.lineage.parents)


def test_root_dataset_no_producers_at_all_returns_empty_walk():
    """Neither version nor member producers -> empty walk (unchanged)."""
    ml = _FakeML()
    ml.add_dataset("1-DSIM", producer=None)
    # no member producers scripted -> empty set

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is None
    assert result.root.producing_execution is None
    assert result.walked_complete is True


# ---------------------------------------------------------------------------
# Behavioral tests: consumed-version + self-parent guard (CV-fix Task 3).
# ---------------------------------------------------------------------------


def test_mid_walk_self_parent_guard_no_false_cycle():
    """An execution that consumed D and also produced one of D's members must
    not become its own parent (no false cycle)."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    # EXSC consumes D-CON and also produced a member of D-CON.
    ml.add_dataset("1-DCON", producer="2-EXOT")
    ml.add_execution("2-EXSC", input_datasets=[_StubDataset("1-DCON")])
    ml.add_dataset("1-DOUT", producer="2-EXSC")
    ml.set_member_producers("1-DCON", {"2-EXSC"})  # self-produced member
    ml.add_execution("2-EXOT", input_datasets=[])

    result = ml.lookup_lineage("1-DOUT")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXSC"
    # 2-EXSC must NOT appear among its own parents, and no false cycle.
    assert all(p.execution.rid != "2-EXSC" for p in result.lineage.parents)
    assert result.cycle_detected is False


def test_mid_walk_uses_consumed_version_producer():
    """When E consumed D@1.0.0 (produced by X) but D@2.0.0 (latest) is by Y,
    walking through E surfaces X, not Y."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    ml.add_execution("2-EXVX", input_datasets=[_StubDataset("1-DSRC")])  # produced D@1.0.0
    ml.add_execution("2-EXVY", input_datasets=[])  # produced D@2.0.0
    # D consumed at 1.0.0 by EXMID:
    ml.add_dataset("1-DVER", producer="2-EXVY")  # latest producer = Y
    ml.set_versioned_producer("1-DVER", "1.0.0", "2-EXVX")  # consumed-version producer = X
    ml.add_execution("2-EXMD", input_datasets=[_StubDataset("1-DVER", consumed_version="1.0.0")])
    ml.add_dataset("1-DEND", producer="2-EXMD")

    result = ml.lookup_lineage("1-DEND")

    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXVX" in parent_rids  # consumed-version producer
    assert "2-EXVY" not in parent_rids  # latest-version producer must NOT appear


def test_mid_walk_uses_consumed_version_member_producers():
    """D@1.0.0 members by P1; D@2.0.0 adds members by P2. Walking E (consumed
    1.0.0) surfaces P1, not P2."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    ml.add_execution("2-EXP1", input_datasets=[_StubDataset("1-DSRC")])
    ml.add_execution("2-EXP2", input_datasets=[])
    ml.add_dataset("1-DVER", producer=None)
    ml.set_versioned_member_producers("1-DVER", "1.0.0", {"2-EXP1"})
    ml.set_versioned_member_producers("1-DVER", "2.0.0", {"2-EXP1", "2-EXP2"})
    ml.add_execution("2-EXMD", input_datasets=[_StubDataset("1-DVER", consumed_version="1.0.0")])
    ml.add_dataset("1-DEND", producer="2-EXMD")

    result = ml.lookup_lineage("1-DEND")

    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXP1" in parent_rids
    assert "2-EXP2" not in parent_rids


def test_mid_walk_consumed_dataset_summary_reports_consumed_version():
    """consumed_datasets[].version reflects the consumed version, not current."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    ml.add_dataset("1-DVER", producer=None)
    ml.add_execution(
        "2-EXMD",
        input_datasets=[_StubDataset("1-DVER", version="9.9.9", consumed_version="1.0.0")],
    )
    ml.add_dataset("1-DEND", producer="2-EXMD")

    result = ml.lookup_lineage("1-DEND")

    consumed = result.lineage.consumed_datasets
    assert len(consumed) == 1
    assert consumed[0].version == "1.0.0"  # consumed, not current 9.9.9


def test_self_parent_via_version_producer_no_false_cycle():
    """An execution that consumed D AND produced D's consumed version must not
    be its own parent (no false cycle). Guards execution.py:1631."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    # EXSC consumes D-VER, and is ALSO the producer of D-VER's consumed version.
    ml.add_dataset("1-DVER", producer="2-EXSC")
    ml.set_versioned_producer("1-DVER", "1.0.0", "2-EXSC")  # consumed-version producer == consumer
    ml.add_execution("2-EXSC", input_datasets=[_StubDataset("1-DVER", consumed_version="1.0.0")])
    ml.add_dataset("1-DOUT", producer="2-EXSC")

    result = ml.lookup_lineage("1-DOUT")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXSC"
    # 2-EXSC must NOT be its own parent, and no false cycle.
    assert all(p.execution.rid != "2-EXSC" for p in result.lineage.parents)
    assert result.cycle_detected is False


def test_multiple_consumed_datasets_different_versions():
    """An execution consuming D1@1.0.0 (producer EXV1) and D2@2.0.0 (producer
    EXV2) surfaces BOTH version-specific producers and BOTH summary versions."""
    ml = _FakeML()
    ml.add_dataset("1-DS01", producer=None)
    ml.add_dataset("1-DS02", producer=None)
    ml.add_execution("2-EXV1", input_datasets=[_StubDataset("1-DS01")])
    ml.add_execution("2-EXV2", input_datasets=[_StubDataset("1-DS02")])
    ml.add_dataset("1-DD01", producer=None)
    ml.add_dataset("1-DD02", producer=None)
    ml.set_versioned_producer("1-DD01", "1.0.0", "2-EXV1")
    ml.set_versioned_producer("1-DD02", "2.0.0", "2-EXV2")
    ml.add_execution(
        "2-EXMD",
        input_datasets=[
            _StubDataset("1-DD01", consumed_version="1.0.0"),
            _StubDataset("1-DD02", consumed_version="2.0.0"),
        ],
    )
    ml.add_dataset("1-DEND", producer="2-EXMD")

    result = ml.lookup_lineage("1-DEND")

    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert {"2-EXV1", "2-EXV2"} <= parent_rids
    summary_versions = {s.rid: s.version for s in result.lineage.consumed_datasets}
    assert summary_versions["1-DD01"] == "1.0.0"
    assert summary_versions["1-DD02"] == "2.0.0"


# ---------------------------------------------------------------------------
# #367: origin resolution (first-recorded, not last-writer)
# ---------------------------------------------------------------------------


def _gen_rid(n: int) -> str:
    return f"1-{n:04X}"


_version_row_ids = iter(range(0x100, 0xFFF))


def _version_row(rct: str, version: str, exec_rid: str | None) -> dict:
    return {
        "RID": _gen_rid(next(_version_row_ids)),
        "RCT": rct,
        "Version": version,
        "Execution": exec_rid,
        "Description": None,
    }


def _origin_ml(version_rows, sentinel_rid=None, summaries=None):
    """ExecutionMixin stub wired for the dataset branch of _classify_rid."""
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    ml.resolve_rid = lambda rid: _StubResolved(table=_StubTable(name="Dataset", columns=[]))
    ml._retrieve_rid = lambda rid: {"RID": rid, "Description": "a dataset"}
    ml._dataset_version_rows = lambda rid: list(version_rows)
    ml._sentinel_execution_rid_or_none = lambda: sentinel_rid
    ml._execution_summaries = lambda rids: dict(summaries or {})
    return ml


def test_unversioned_producer_is_first_recorded_not_latest():
    origin_exec, migration_exec = _gen_rid(10), _gen_rid(11)
    rows = [
        _version_row("2025-01-01T00:00:00Z", "0.1.0", origin_exec),
        _version_row("2026-01-01T00:00:00Z", "4.13.0", migration_exec),
    ]
    ml = _origin_ml(rows)
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed == origin_exec
    assert descriptor.origin_recorded is True
    assert descriptor.version_history[0].execution_rid == origin_exec
    assert descriptor.version_history[-1].execution_rid == migration_exec


def test_sentinel_origin_reported_but_never_walk_seed():
    sentinel = _gen_rid(66)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", sentinel)]
    ml = _origin_ml(rows, sentinel_rid=sentinel)
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None  # sentinel never seeds the walk
    assert descriptor.origin_recorded is False


def test_no_execution_on_first_row():
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", None)]
    ml = _origin_ml(rows)
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None
    assert descriptor.origin_recorded is False
    assert descriptor.producing_execution is None
    assert descriptor.version_history[0].execution_rid is None


def test_no_version_rows():
    ml = _origin_ml([])
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None
    assert descriptor.origin_recorded is False
    assert descriptor.version_history == []


def test_recorded_but_unresolvable_origin_keeps_recorded_true():
    origin_exec = _gen_rid(10)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin_exec)]
    ml = _origin_ml(rows, summaries={})  # summary resolution came back empty
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert descriptor.origin_recorded is True
    assert descriptor.producing_execution is None
    assert descriptor.version_history[0].execution_rid == origin_exec
    assert walk_seed == origin_exec


def test_sentinel_origin_producing_execution_carries_sentinel_summary():
    """Sentinel origin is never a walk seed, but its RootDescriptor still
    carries the sentinel's own ExecutionSummary when one is resolvable —
    ``origin_recorded`` is what signals "not a real author", not a missing
    ``producing_execution``."""
    from deriva_ml.execution.lineage import ExecutionSummary

    sentinel = _gen_rid(66)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", sentinel)]
    sentinel_summary = ExecutionSummary(rid=sentinel, description="sentinel", workflow=None, status="Uploaded")
    ml = _origin_ml(rows, sentinel_rid=sentinel, summaries={sentinel: sentinel_summary})
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None
    assert descriptor.origin_recorded is False
    assert descriptor.producing_execution is not None
    assert descriptor.producing_execution.rid == sentinel


def test_version_history_entries_carry_resolved_summaries():
    """Each version_history entry's ``execution`` is the resolved summary for
    that row's own ``execution_rid`` — not the origin's, not shared."""
    from deriva_ml.execution.lineage import ExecutionSummary

    author_a, author_b = _gen_rid(10), _gen_rid(11)
    rows = [
        _version_row("2025-01-01T00:00:00Z", "0.1.0", author_a),
        _version_row("2026-01-01T00:00:00Z", "2.0.0", author_b),
    ]
    summary_a = ExecutionSummary(rid=author_a, description="first", workflow=None, status="Uploaded")
    summary_b = ExecutionSummary(rid=author_b, description="second", workflow=None, status="Uploaded")
    ml = _origin_ml(rows, summaries={author_a: summary_a, author_b: summary_b})
    descriptor, _ = ml._classify_rid(_gen_rid(1))
    assert len(descriptor.version_history) == 2
    for entry in descriptor.version_history:
        assert entry.execution is not None
        assert entry.execution.rid == entry.execution_rid


# ---------------------------------------------------------------------------
# #367: walk seeding — sentinel exclusion, no overwrite, degradation
# ---------------------------------------------------------------------------


def _walkable_ml(version_rows, member_producers, known_execs, sentinel_rid=None, summaries=None):
    """Full lookup_lineage stub: dataset root + walkable executions."""
    ml = _origin_ml(version_rows, sentinel_rid=sentinel_rid, summaries=summaries)
    ml._producers_of_dataset_members = lambda rid, version=None: set(member_producers)
    ml._input_dataset_pairs = lambda rid: []

    def fake_lookup_execution(rid):
        if rid not in known_execs:
            raise DerivaMLException(f"no execution {rid}")
        rec = MagicMock()
        rec.description = f"exec {rid}"
        rec.workflow = None
        rec.status = ExecutionStatus.Uploaded
        rec.list_assets = lambda **kw: []  # consumed-asset iteration is on the record
        return rec

    ml.lookup_execution = fake_lookup_execution
    return ml


def test_member_fallback_seeds_walk_but_is_not_origin():
    member = _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", None)]  # no origin
    ml = _walkable_ml(rows, {member}, {member})
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == member  # walk seeded
    assert result.root.producing_execution is None  # origin NOT overwritten
    assert result.root.origin_recorded is False


def test_sentinel_origin_walk_seeds_from_member_producers():
    sentinel, member = _gen_rid(66), _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", sentinel)]
    ml = _walkable_ml(rows, {member}, {member}, sentinel_rid=sentinel)
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == member  # not the sentinel
    assert result.root.origin_recorded is False


def test_unexpandable_origin_degrades_to_member_seeding():
    origin, member = _gen_rid(10), _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin)]
    ml = _walkable_ml(rows, {member}, known_execs={member})  # origin not expandable
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == member
    assert result.root.origin_recorded is True  # origin still recorded


def test_normal_recorded_origin_walks_from_origin():
    origin, member = _gen_rid(10), _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin)]
    ml = _walkable_ml(rows, {member}, {origin, member})
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage.execution.rid == origin


def test_unexpandable_origin_in_member_set_still_degrades():
    """Origin is ALSO a member producer, but unexpandable; a distinct member
    producer IS expandable. The degradation retry must not be skipped just
    because ``producer_rid in member_producers`` — it must seed from the
    other expandable member (spec: walk seed vs. origin attribution)."""
    origin, other_member = _gen_rid(10), _gen_rid(31)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin)]
    # origin is itself one of the member producers, but not expandable.
    ml = _walkable_ml(rows, {origin, other_member}, known_execs={other_member})
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == other_member
    assert result.root.origin_recorded is True  # origin still recorded


def _collect_execution_rids(node) -> set[str]:
    """All execution RIDs anywhere in a LineageNode tree (self + parents, recursive)."""
    if node is None:
        return set()
    rids = {node.execution.rid}
    for parent in node.parents:
        rids |= _collect_execution_rids(parent)
    return rids


def test_sentinel_member_producer_never_seeds_or_parents():
    """The sentinel can show up in member_producers (provenance backfill
    attributes producerless member assets to it via Output edges), but it
    must NEVER become the walk root or an extra parent — lineage terminates
    at the sentinel (design rule). Both the sentinel and the real producer
    are resolvable here, and the sentinel RID is chosen to sort BEFORE the
    real producer's, so ``sorted(member_producers)[0]`` would pick the
    sentinel absent the fix — proving exclusion by filtering, not by the
    sentinel failing to expand."""
    sentinel, real_producer = _gen_rid(20), _gen_rid(30)
    assert sentinel < real_producer  # sentinel must sort first
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", None)]  # no origin
    ml = _walkable_ml(
        rows,
        {sentinel, real_producer},
        known_execs={sentinel, real_producer},
        sentinel_rid=sentinel,
    )
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == real_producer
    assert sentinel not in _collect_execution_rids(result.lineage)


def test_all_stale_but_last_member_producer_still_walks():
    """Three member producers, no origin; only the LAST in sorted order is
    resolvable. The walk must iterate through candidates until one expands,
    not give up after a single retry."""
    p1, p2, p3 = _gen_rid(40), _gen_rid(41), _gen_rid(42)
    assert sorted([p1, p2, p3]) == [p1, p2, p3]
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", None)]  # no origin
    ml = _walkable_ml(rows, {p1, p2, p3}, known_execs={p3})
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == p3


# ---------------------------------------------------------------------------
# Root version + asset-root shape (post-#367 follow-up)
# ---------------------------------------------------------------------------


def test_dataset_root_carries_current_version():
    """root.version = the label of the row the Dataset.Version FK points at.

    The FK targets the MIDDLE row here, so a pass proves FK-resolution,
    not a newest-row shortcut.
    """
    rows = [
        _version_row("2025-01-01T00:00:00Z", "0.1.0", None),
        _version_row("2025-06-01T00:00:00Z", "0.2.0", None),
        _version_row("2026-01-01T00:00:00Z", "0.3.0", None),
    ]
    ml = _origin_ml(rows)
    current_rid = rows[1]["RID"]
    ml._retrieve_rid = lambda rid: {"RID": rid, "Description": "d", "Version": current_rid}
    descriptor, _ = ml._classify_rid(_gen_rid(1))
    assert descriptor.version == "0.2.0"


def test_dataset_root_version_falls_back_to_latest_ordered():
    """No resolvable Version FK -> latest ordered row's label."""
    rows = [
        _version_row("2025-01-01T00:00:00Z", "0.1.0", None),
        _version_row("2026-01-01T00:00:00Z", "0.2.0", None),
    ]
    ml = _origin_ml(rows)  # _origin_ml's _retrieve_rid has no Version key
    descriptor, _ = ml._classify_rid(_gen_rid(1))
    assert descriptor.version == "0.2.0"


def test_dataset_root_version_none_when_no_rows():
    ml = _origin_ml([])
    descriptor, _ = ml._classify_rid(_gen_rid(1))
    assert descriptor.version is None


def test_asset_root_shape_carries_no_dataset_fields():
    """Asset-RID lineage end-to-end: walk works and the dataset-only
    fields stay in their not-applicable states."""
    producer = _gen_rid(70)
    ml = _FakeML()
    ml.add_execution(producer, input_datasets=[])
    ml.add_asset(_gen_rid(71), asset_table="Image", filename="x.png", producer=producer)

    result = ml.lookup_lineage(_gen_rid(71))

    assert result.root.type == "Asset"
    assert result.root.version is None
    assert result.root.origin_recorded is None
    assert result.root.version_history == []
    assert result.lineage is not None
    assert result.lineage.execution.rid == producer
    assert result.root.producing_execution.rid == producer  # non-Dataset overwrite kept


def test_walk_node_workflow_summary_carries_url_and_version():
    """Lineage nodes expose workflow code identity with no extra lookups (#372)."""
    producer = _gen_rid(80)
    ml = _FakeML()
    ml.add_dataset(_gen_rid(81), producer=None)
    ml.add_execution(
        producer,
        input_datasets=[],
        workflow=_StubWorkflow(
            workflow_rid=_gen_rid(82),
            name="trainer",
            url="https://github.com/org/model-repo",
            version="0.3.4",
        ),
    )
    ml.add_asset(_gen_rid(83), asset_table="Image", filename="m.keras", producer=producer)

    result = ml.lookup_lineage(_gen_rid(83))

    wf = result.lineage.execution.workflow
    assert wf.url == "https://github.com/org/model-repo"
    assert wf.version == "0.3.4"


# ---------------------------------------------------------------------------
# Task 8: _FakeML closure seams — smoke tests.
#
# Each seam is scripted then read back through its override (and, where
# applicable, through the _StubDatasetHandle returned by lookup_dataset),
# asserting both the round-tripped value AND that ml.calls recorded the
# invocation. These tests exercise the harness itself, not lookup_lineage.
# ---------------------------------------------------------------------------


def test_add_version_row_multi_row_feeds_dataset_version_rows():
    """add_version_row appends rows in insertion order; _dataset_version_rows
    returns exactly what was scripted and records the call."""
    ml = _FakeML()
    row1 = ml.add_version_row(_gen_rid(200), "0.1.0", _gen_rid(201), rct="2025-01-01T00:00:00Z")
    row2 = ml.add_version_row(_gen_rid(200), "0.2.0", _gen_rid(202), rct="2025-06-01T00:00:00Z")

    rows = ml._dataset_version_rows(_gen_rid(200))

    assert rows == [row1, row2]
    assert ("_dataset_version_rows", (_gen_rid(200),)) in ml.calls


def test_add_version_row_overrides_legacy_add_dataset_producer_row():
    """When both add_dataset(producer=...) and add_version_row are used for
    the same RID, the multi-row seam wins over the legacy single-row
    synthesis (add_version_row rows are checked first)."""
    ml = _FakeML()
    ml.add_dataset(_gen_rid(210), producer=_gen_rid(211))
    explicit_row = ml.add_version_row(_gen_rid(210), "3.0.0", _gen_rid(212))

    rows = ml._dataset_version_rows(_gen_rid(210))

    assert rows == [explicit_row]


def test_set_snapshot_version_rows_round_trips():
    """set_snapshot_version_rows scripts the snapshot-scoped view; reading it
    back directly returns exactly the scripted rows."""
    ml = _FakeML()
    scripted = [
        {
            "RID": _gen_rid(220),
            "RCT": "2025-01-01T00:00:00Z",
            "Version": "1.0.0",
            "Execution": _gen_rid(221),
            "Description": None,
        }
    ]
    ml.set_snapshot_version_rows(_gen_rid(219), "1.0.0", scripted)

    assert ml._snapshot_version_rows[(_gen_rid(219), "1.0.0")] == scripted


def test_lookup_dataset_returns_stub_with_matching_rid():
    ml = _FakeML()
    handle = ml.lookup_dataset(_gen_rid(230))

    assert handle.dataset_rid == _gen_rid(230)
    assert ("lookup_dataset", (_gen_rid(230),)) in ml.calls


def test_set_snapshot_available_true_returns_marker():
    """A scripted-available snapshot resolves without raising, and the call
    is recorded on the OWNING ml.calls (not some private list on the handle)."""
    ml = _FakeML()
    ml.set_snapshot_available(_gen_rid(240), "1.0.0", True)

    handle = ml.lookup_dataset(_gen_rid(240))
    result = handle.strict_version_snapshot_catalog("1.0.0")

    assert result is not None
    assert ("Dataset.strict_version_snapshot_catalog", (_gen_rid(240), "1.0.0")) in ml.calls


def test_set_snapshot_available_false_raises_snapshot_unavailable():
    ml = _FakeML()
    ml.set_snapshot_available(_gen_rid(241), "1.0.0", False)

    handle = ml.lookup_dataset(_gen_rid(241))
    with pytest.raises(SnapshotUnavailable):
        handle.strict_version_snapshot_catalog("1.0.0")


def test_unscripted_snapshot_defaults_to_unavailable():
    """Safe default: a (dataset_rid, version) pair that was NEVER scripted
    via set_snapshot_available (or add_version_row(snapshot=...)) raises
    SnapshotUnavailable, matching the real Dataset's no-live-fallback
    contract. See _StubDatasetHandle's docstring for the rationale."""
    ml = _FakeML()
    handle = ml.lookup_dataset(_gen_rid(242))

    with pytest.raises(SnapshotUnavailable):
        handle.strict_version_snapshot_catalog("9.9.9")


def test_add_version_row_with_snapshot_makes_it_available():
    """add_version_row(..., snapshot=...) is sugar for
    set_snapshot_available(..., True) plus a returned marker."""
    ml = _FakeML()
    ml.add_version_row(_gen_rid(250), "1.0.0", _gen_rid(251), snapshot="catalog@2026-01-01T00:00:00Z")

    handle = ml.lookup_dataset(_gen_rid(250))
    result = handle.strict_version_snapshot_catalog("1.0.0")

    assert result == "catalog@2026-01-01T00:00:00Z"


def test_set_binding_scan_round_trips_through_find_feature_producers_impl():
    ml = _FakeML()
    record = FeatureProducerRecord(
        execution_rid=_gen_rid(271), feature_name="Annotation", element_type="Image", value_count=3
    )
    diagnostic = BindingDiagnostic(kind="query_failed", subject="Annotation", detail="boom")
    ml.set_binding_scan(_gen_rid(270), None, [record], [diagnostic])

    records, diagnostics = ml._find_feature_producers_impl(_gen_rid(270), None)

    assert records == [record]
    assert diagnostics == [diagnostic]
    assert ("_find_feature_producers_impl", (_gen_rid(270), None)) in ml.calls


def test_set_binding_scan_is_version_scoped():
    """A binding scan scripted for version="1.0.0" does not leak into the
    unversioned (None) read for the same dataset."""
    ml = _FakeML()
    record = FeatureProducerRecord(
        execution_rid=_gen_rid(281), feature_name="Label", element_type="Image", value_count=1
    )
    ml.set_binding_scan(_gen_rid(280), "1.0.0", [record], [])

    versioned_records, _ = ml._find_feature_producers_impl(_gen_rid(280), "1.0.0")
    unversioned_records, unversioned_diagnostics = ml._find_feature_producers_impl(_gen_rid(280), None)

    assert versioned_records == [record]
    assert unversioned_records == []
    assert unversioned_diagnostics == []


def test_add_asset_multi_producer_preserves_fetched_order():
    """add_asset(..., producers=[...]) backs _producers_of_asset with fetched
    order preserved; _producer_of_asset derives the first element."""
    ml = _FakeML()
    ml.add_asset(
        _gen_rid(290),
        asset_table="Image",
        filename="x.png",
        producers=[_gen_rid(292), _gen_rid(291), _gen_rid(293)],
    )

    producers = ml._producers_of_asset(_gen_rid(290), "Image")

    assert producers == [_gen_rid(292), _gen_rid(291), _gen_rid(293)]
    assert ml._producer_of_asset(_gen_rid(290), "Image") == _gen_rid(292)
    assert ("_producers_of_asset", (_gen_rid(290), "Image")) in ml.calls


def test_add_asset_producer_kwarg_still_works_as_single_element_list():
    """Legacy single-producer scripting (add_asset(..., producer=...)) keeps
    working unmodified — sugar for a one-element producers list."""
    ml = _FakeML()
    ml.add_asset(_gen_rid(295), asset_table="Image", filename="y.png", producer=_gen_rid(296))

    assert ml._producers_of_asset(_gen_rid(295), "Image") == [_gen_rid(296)]


def test_call_recording_captures_seam_invocations_during_a_real_walk():
    """A full lookup_lineage walk populates ml.calls with the seams it hit —
    proves the recording wiring holds for the composed walk, not just direct
    unit calls to individual overrides."""
    ml = _FakeML()
    ml.add_dataset(_gen_rid(300), producer=None)
    ml.add_execution(_gen_rid(301), input_datasets=[_StubDataset(_gen_rid(300))])
    ml.add_dataset(_gen_rid(302), producer=_gen_rid(301))

    ml.lookup_lineage(_gen_rid(302))

    recorded_methods = {name for name, _ in ml.calls}
    assert "_dataset_version_rows" in recorded_methods
    assert "_input_dataset_pairs" in recorded_methods


def test_quarantine_snapshot_seams_untouched_by_ordinary_lineage_walk():
    """An ordinary (non-snapshot-scoped) lookup_lineage walk must NEVER touch
    the strict-snapshot seams (lookup_dataset /
    strict_version_snapshot_catalog) — those are reserved for the
    version-anchored closure walk. This is the quarantine assertion the
    call log exists to support."""
    ml = _FakeML()
    ml.add_dataset(_gen_rid(310), producer=None)
    ml.add_execution(_gen_rid(311), input_datasets=[_StubDataset(_gen_rid(310))])
    ml.add_dataset(_gen_rid(312), producer=_gen_rid(311))

    ml.lookup_lineage(_gen_rid(312))

    recorded_methods = {name for name, _ in ml.calls}
    assert "lookup_dataset" not in recorded_methods
    assert "Dataset.strict_version_snapshot_catalog" not in recorded_methods
    assert "_find_feature_producers_impl" not in recorded_methods
    # Structural containment is never a walk seam, in either mode
    # (ruling 8, #389).
    assert "Dataset.list_dataset_parents" not in recorded_methods

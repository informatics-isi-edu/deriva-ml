"""Tests for the lease orchestrator — composes rid_lease + state_store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine


class _MockLeaseCatalog:
    def __init__(self, *, prefix="RID-", fail_after=None):
        self.prefix = prefix
        self.fail_after = fail_after  # After N successful posts, start failing
        self.post_calls: list[list[dict]] = []

    def post(self, path: str, json=None, **_kw):
        self.post_calls.append(json)
        if self.fail_after is not None and len(self.post_calls) > self.fail_after:
            raise RuntimeError("simulated post failure")
        class _R:
            def __init__(self, bodies, prefix, offset):
                self._bodies = bodies
                self._prefix = prefix
                self._offset = offset
            def json(self):
                return [
                    {"RID": f"{self._prefix}{self._offset + i}", "ID": b["ID"]}
                    for i, b in enumerate(self._bodies)
                ]
        offset = sum(len(p) for p in self.post_calls[:-1])
        return _R(json, self.prefix, offset)


def _setup_store(tmp_path):
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    return store


def test_acquire_leases_happy_path(tmp_path):
    from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pending_ids = [
        store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        )
        for i in range(3)
    ]

    cat = _MockLeaseCatalog(prefix="R-")
    acquire_leases_for_execution(
        store=store, catalog=cat, execution_rid="EXE-A",
        pending_ids=pending_ids,
    )

    rows = store.list_pending_rows(execution_rid="EXE-A")
    assert len(rows) == 3
    for r in rows:
        assert r["status"] == "leased"
        assert r["rid"] is not None
        assert r["rid"].startswith("R-")
        assert r["leased_at"] is not None


def test_acquire_leases_skips_already_leased(tmp_path):
    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    already_leased = store.insert_pending_row(
        execution_rid="EXE-A", key="already",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        rid="EXISTING-RID",
        status=PendingRowStatus.leased,
    )
    to_lease = store.insert_pending_row(
        execution_rid="EXE-A", key="new",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    cat = _MockLeaseCatalog(prefix="R-")
    acquire_leases_for_execution(
        store=store, catalog=cat, execution_rid="EXE-A",
        pending_ids=[already_leased, to_lease],
    )

    # Only ONE token was POSTed — the already-leased row was skipped.
    assert len(cat.post_calls) == 1
    assert len(cat.post_calls[0]) == 1


def test_acquire_leases_reverts_on_post_failure(tmp_path):
    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    ids = [
        store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        )
        for i in range(2)
    ]

    cat = _MockLeaseCatalog(fail_after=0)  # fail on first POST
    with pytest.raises(RuntimeError):
        acquire_leases_for_execution(
            store=store, catalog=cat, execution_rid="EXE-A",
            pending_ids=ids,
        )

    # Rows must be back in 'staged' — the leasing→staged revert ran.
    rows = store.list_pending_rows(execution_rid="EXE-A")
    for r in rows:
        assert r["status"] == str(PendingRowStatus.staged), \
            "orchestrator must revert on POST failure"
        assert r["lease_token"] is None


class _MockCatalogWithGet:
    """Mock exposing GET for querying ERMrest_RID_Lease by token."""
    def __init__(self, *, rows_by_id: dict[str, str] | None = None):
        self.rows_by_id = rows_by_id or {}  # token → RID
        self.get_paths: list[str] = []

    def get(self, path: str, **_kw):
        self.get_paths.append(path)
        # Parse out the ID filter from the URL (ID=T1;ID=T2...).
        import re
        ids_param = re.search(r"ID=([^&]+)", path)
        tokens = ids_param.group(1).split(";") if ids_param else []
        tokens = [re.sub(r"^ID=", "", t) for t in tokens if t]
        rows = [
            {"ID": t, "RID": self.rows_by_id[t]}
            for t in tokens
            if t in self.rows_by_id
        ]
        class _R:
            def __init__(self, rows): self._rows = rows
            def json(self): return self._rows
        return _R(rows)


def test_reconcile_pending_leases_adopts_server_assigned(tmp_path):
    """Row status='leasing' whose token exists on the server →
    promoted to 'leased' with the server's RID."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T-FOUND")

    cat = _MockCatalogWithGet(rows_by_id={"T-FOUND": "R-LATE"})
    reconcile_pending_leases(store=store, catalog=cat, execution_rid="EXE-A")

    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.leased)
    assert row["rid"] == "R-LATE"


def test_reconcile_pending_leases_reverts_missing(tmp_path):
    """Row status='leasing' whose token doesn't exist on the server →
    reverted to 'staged'."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T-ORPHAN")

    cat = _MockCatalogWithGet(rows_by_id={})  # nothing on server
    reconcile_pending_leases(store=store, catalog=cat, execution_rid="EXE-A")

    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.staged)
    assert row["lease_token"] is None


def test_reconcile_pending_leases_chunks_large_batches(tmp_path, monkeypatch):
    """With PENDING_ROWS_LEASE_CHUNK=2 and 5 leasing rows, the orchestrator
    issues ceil(5/2)=3 GET requests and transitions all 5 rows out of
    'leasing' (mix of adopt/revert)."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus

    # Patch the binding in lease_orchestrator (the import was moved to
    # module top-of-file, so this is where the reconcile loop reads it).
    monkeypatch.setattr(
        "deriva_ml.execution.lease_orchestrator.PENDING_ROWS_LEASE_CHUNK", 2
    )

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    tokens = [f"TK-{i}" for i in range(5)]
    for i, tok in enumerate(tokens):
        pid = store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        )
        store.mark_pending_leasing(pid, lease_token=tok)

    # Mix: tokens 0, 2, 4 found on server; 1, 3 missing.
    rows_by_id = {"TK-0": "R-0", "TK-2": "R-2", "TK-4": "R-4"}
    cat = _MockCatalogWithGet(rows_by_id=rows_by_id)

    reconcile_pending_leases(store=store, catalog=cat, execution_rid="EXE-A")

    # ceil(5/2) = 3 chunks → 3 GET calls.
    assert len(cat.get_paths) == 3

    # All 5 rows should be out of 'leasing'.
    rows = store.list_pending_rows(execution_rid="EXE-A")
    assert len(rows) == 5
    statuses = {r["key"]: r["status"] for r in rows}
    rids = {r["key"]: r["rid"] for r in rows}
    for i in (0, 2, 4):
        assert statuses[f"k{i}"] == str(PendingRowStatus.leased)
        assert rids[f"k{i}"] == f"R-{i}"
    for i in (1, 3):
        assert statuses[f"k{i}"] == str(PendingRowStatus.staged)
        assert rids[f"k{i}"] is None


def test_reconcile_pending_leases_workspace_wide(tmp_path):
    """execution_rid=None reconciles across all executions."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus, ExecutionStatus
    from deriva_ml.core.connection_mode import ConnectionMode

    store = _setup_store(tmp_path)  # creates EXE-A
    now = datetime.now(timezone.utc)
    # Add a second execution.
    store.insert_execution(
        rid="EXE-B", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-B",
        created_at=now, last_activity=now,
    )
    pid_a = store.insert_pending_row(
        execution_rid="EXE-A", key="ka",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    pid_b = store.insert_pending_row(
        execution_rid="EXE-B", key="kb",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid_a, lease_token="TA")
    store.mark_pending_leasing(pid_b, lease_token="TB")

    cat = _MockCatalogWithGet(rows_by_id={"TA": "R-A", "TB": "R-B"})
    reconcile_pending_leases(store=store, catalog=cat, execution_rid=None)

    for rid, expected_r in [("EXE-A", "R-A"), ("EXE-B", "R-B")]:
        row = store.list_pending_rows(execution_rid=rid)[0]
        assert row["status"] == str(PendingRowStatus.leased)
        assert row["rid"] == expected_r


def test_workspace_open_reconciles_leases(test_ml, monkeypatch):
    """On DerivaML construction (online), startup sweep runs."""
    calls: list[str | None] = []

    def _spy(*, store, catalog, execution_rid):
        calls.append(execution_rid)

    from deriva_ml.execution import lease_orchestrator
    monkeypatch.setattr(lease_orchestrator, "reconcile_pending_leases", _spy)

    # Creating a DerivaML instance should trigger the sweep.
    # test_ml is already constructed; construct another one to observe.
    from deriva_ml import DerivaML
    DerivaML(
        hostname=test_ml.host_name,
        catalog_id=test_ml.catalog_id,
        working_dir=test_ml.working_dir,
    )
    # Sweep scoped workspace-wide (execution_rid=None).
    assert None in calls


def test_offline_workspace_skips_lease_reconcile(monkeypatch, tmp_path):
    """In offline mode there's no server to query — skip the sweep.

    S4 implemented real offline mode: ``DerivaML(mode=offline)`` reads
    a pre-populated schema cache at ``<working_dir>/schema-cache.json``
    and sets ``self.catalog = CatalogStub()``. The lease-reconcile
    hook in ``__init__`` is gated on ``self._mode is online``, so in
    offline mode it simply isn't called. This test plants a minimal
    cache file, constructs offline, and asserts the spy was never hit.

    No network stubs needed — offline mode doesn't touch
    ``DerivaServer``, ``get_credential``, or ``DerivaModel.from_catalog``.
    """
    from deriva_ml.execution import lease_orchestrator
    calls: list[str | None] = []
    def _spy(**_kw): calls.append(_kw.get("execution_rid"))
    monkeypatch.setattr(lease_orchestrator, "reconcile_pending_leases", _spy)

    # Plant a minimal schema cache so offline init succeeds without
    # a live catalog. The schema just needs to parse; no tables needed.
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    cache.write(
        snapshot_id="fake-snap",
        hostname="offline.example",
        catalog_id="1",
        ml_schema="deriva-ml",
        schema={
            "schemas": {
                "deriva-ml": {
                    "schema_name": "deriva-ml",
                    "tables": {},
                    "annotations": {},
                    "comment": None,
                },
            },
            "acls": {},
            "annotations": {},
        },
    )

    from deriva_ml import ConnectionMode, DerivaML
    DerivaML(
        hostname="offline.example",
        catalog_id="1",
        working_dir=str(tmp_path),
        mode=ConnectionMode.offline,
    )
    assert calls == []

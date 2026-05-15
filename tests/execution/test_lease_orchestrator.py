"""Tests for the lease orchestrator's crash-recovery path.

Originally also covered ``acquire_leases_for_execution``; that
production-dead acquire path was retired in the Phase 3 cleanup
(audit §1.6). Remaining tests exercise
:func:`reconcile_pending_leases`, which survives as a vestigial
no-op for the production call sites in ``core/base.py`` and
``core/mixins/execution.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine


def _setup_store(tmp_path):
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStateStore, ExecutionStatus

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A",
        workflow_rid=None,
        description=None,
        config_json="{}",
        status=ExecutionStatus.Running,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        created_at=now,
        last_activity=now,
    )
    return store


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
        rows = [{"ID": t, "RID": self.rows_by_id[t]} for t in tokens if t in self.rows_by_id]

        class _R:
            def __init__(self, rows):
                self._rows = rows

            def json(self):
                return self._rows

        return _R(rows)


def test_reconcile_pending_leases_adopts_server_assigned(tmp_path):
    """Row status='leasing' whose token exists on the server →
    promoted to 'leased' with the server's RID."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid="EXE-A",
        key="k1",
        target_schema="s",
        target_table="t",
        metadata_json="{}",
        created_at=now,
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
        execution_rid="EXE-A",
        key="k1",
        target_schema="s",
        target_table="t",
        metadata_json="{}",
        created_at=now,
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
    monkeypatch.setattr("deriva_ml.execution.lease_orchestrator.PENDING_ROWS_LEASE_CHUNK", 2)

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    tokens = [f"TK-{i}" for i in range(5)]
    for i, tok in enumerate(tokens):
        pid = store.insert_pending_row(
            execution_rid="EXE-A",
            key=f"k{i}",
            target_schema="s",
            target_table="t",
            metadata_json="{}",
            created_at=now,
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
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import ExecutionStatus, PendingRowStatus

    store = _setup_store(tmp_path)  # creates EXE-A
    now = datetime.now(timezone.utc)
    # Add a second execution.
    store.insert_execution(
        rid="EXE-B",
        workflow_rid=None,
        description=None,
        config_json="{}",
        status=ExecutionStatus.Running,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-B",
        created_at=now,
        last_activity=now,
    )
    pid_a = store.insert_pending_row(
        execution_rid="EXE-A",
        key="ka",
        target_schema="s",
        target_table="t",
        metadata_json="{}",
        created_at=now,
    )
    pid_b = store.insert_pending_row(
        execution_rid="EXE-B",
        key="kb",
        target_schema="s",
        target_table="t",
        metadata_json="{}",
        created_at=now,
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

    def _spy(**_kw):
        calls.append(_kw.get("execution_rid"))

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

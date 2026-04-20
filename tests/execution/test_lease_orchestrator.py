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
        config_json="{}", status=ExecutionStatus.running,
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

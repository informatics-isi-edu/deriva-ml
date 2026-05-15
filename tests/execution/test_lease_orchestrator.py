"""Tests for the lease-orchestrator call sites.

The lease-orchestrator's pending-rows machinery (``acquire`` /
``reconcile``) was retired in the Phase 3 cleanup per
``docs/design/deriva-ml-audit-2026-05-phase3-execution.md`` §1.5 /
§1.6. :func:`reconcile_pending_leases` survives as a no-op stub so
the two production call sites — workspace open in
``core/base.py`` and ``resume_execution`` in
``core/mixins/execution.py`` — continue to compile.

The tests below pin those call sites: workspace open triggers the
sweep (online mode), offline mode skips it. They use a spy so they
don't depend on the function's (now no-op) body.
"""

from __future__ import annotations


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

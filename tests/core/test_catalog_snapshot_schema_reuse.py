"""Unit tests for catalog_snapshot() schema reuse (no redundant /schema fetch).

These tests verify the performance fix from
docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md:
a snapshot DerivaML reuses the live instance's already-parsed schema
instead of re-fetching /schema from the server.
"""

from __future__ import annotations

import pytest

from deriva_ml import DerivaML


@pytest.fixture
def live_ml(catalog_manager, tmp_path):
    """A populated DerivaML instance against the test catalog."""
    catalog_manager.ensure_populated(tmp_path)
    return catalog_manager.get_ml_instance(tmp_path)


def test_live_instance_retains_schema_json(live_ml):
    """_init_online stores the parsed schema dict on the instance for reuse."""
    assert hasattr(live_ml, "_schema_json")
    assert isinstance(live_ml._schema_json, dict)
    # ermrest /schema payloads have a top-level "schemas" key.
    assert "schemas" in live_ml._schema_json


def test_init_online_validates_reused_schema_against_catalog(live_ml, monkeypatch):
    """When reuse_schema_json is supplied, _init_online validates it against the
    catalog's actual schema rather than trusting it blindly.

    The reuse fast-path used to adopt reuse_schema_json unconditionally,
    skipping getCatalogSchema() entirely. That is unsafe for a snapshot whose
    schema differs from the live instance's (e.g. a snapshot taken before a
    schema migration): the model would carry tables the snapshot catalog
    doesn't have, and a pathBuilder query against the snapshot 409s. The fix
    revalidates via getCatalogSchema (ETag-cheap) so the model matches the
    catalog. So exactly one revalidation request is expected — not zero.
    """
    from deriva.core.ermrest_catalog import ErmrestCatalog

    calls = {"n": 0}
    real = ErmrestCatalog.getCatalogSchema

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", counting)

    DerivaML(
        live_ml.host_name,
        live_ml.catalog_id,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
        reuse_schema_json=live_ml._schema_json,
    )
    # The model must come from the catalog's real schema (validated), not from
    # blindly trusting the reused dict. One revalidation fetch is correct.
    assert calls["n"] >= 1, (
        "reused schema must be validated against the catalog (getCatalogSchema "
        "called at least once), not adopted blindly"
    )


def test_init_online_model_matches_catalog_not_stale_reuse(live_ml, monkeypatch):
    """REGRESSION: a reused schema that DIFFERS from the catalog must not win.

    Reproduces the snapshot-predates-migration bug: the caller hands a reused
    schema containing a table the connected catalog does NOT have. The built
    model must reflect the CATALOG (table absent), not the stale reused dict
    (table present) — otherwise a later pathBuilder query 409s on a table the
    snapshot lacks.
    """
    import copy

    from deriva.core.ermrest_catalog import ErmrestCatalog

    real = ErmrestCatalog.getCatalogSchema
    # The catalog's REAL schema (what the server returns).
    catalog_schema = real(live_ml.catalog)
    ml_schema_name = live_ml.ml_schema
    assert "_PhantomTable_" not in catalog_schema["schemas"][ml_schema_name]["tables"]

    # A STALE reused schema: same as live PLUS a phantom table the catalog
    # does not have (simulating a table added after this snapshot's snaptime).
    # Build a MINIMAL valid table definition (RID column only, no foreign
    # keys) so it parses cleanly — copying an existing table would duplicate
    # its FK constraint names and crash the parser before the assertion.
    stale = copy.deepcopy(catalog_schema)
    phantom = {
        "schema_name": ml_schema_name,
        "table_name": "_PhantomTable_",
        "kind": "table",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "ermrest_rid"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"], "names": [[ml_schema_name, "_PhantomTable__RIDkey"]]}],
        "foreign_keys": [],
        "annotations": {},
        "comment": None,
    }
    stale["schemas"][ml_schema_name]["tables"]["_PhantomTable_"] = phantom

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", real)

    instance = DerivaML(
        live_ml.host_name,
        live_ml.catalog_id,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
        reuse_schema_json=stale,
    )
    # The model must match the catalog (no phantom), proving the stale reuse
    # was validated/discarded — not trusted.
    model_tables = set(instance.model.model.schemas[ml_schema_name].tables)
    assert "_PhantomTable_" not in model_tables, (
        "stale reused schema's phantom table leaked into the model — reuse was "
        "trusted blindly instead of validated against the catalog (the bug)"
    )


def _a_snapshot_id(live_ml):
    """Resolve a real snapshot id in the COMPOUND ``<catalog_id>@<snaptime>``
    form that production (``_version_snapshot_catalog_id``) always uses.

    A bare snaptime has no ``@``, so deriva-py would treat it as a *catalog
    id* and connect to a non-existent catalog — which only appeared to work
    before because the old code never queried ``/schema`` on that bogus
    connection. The schema-validation fix does query it, so the id must be
    the real compound form.
    """
    raw = live_ml.catalog.get("/").json()["snaptime"]
    # catalog_id may already be compound (e.g. on a snapshot-derived
    # instance); use only the bare catalog id portion to avoid a double "@".
    bare_catalog_id = str(live_ml.catalog_id).split("@", 1)[0]
    return f"{bare_catalog_id}@{raw}"


def test_catalog_snapshot_validates_schema_against_snapshot(live_ml, monkeypatch):
    """catalog_snapshot() validates the reused schema against the snapshot.

    The reused schema is no longer trusted blindly — it is validated against
    the snapshot catalog's own /schema (ETag-cheap: a 304 when the snapshot
    schema equals live). So a single getCatalogSchema call is expected on the
    snapshot, not zero. This is the correctness-over-zero-fetch tradeoff: a
    snapshot predating a schema migration would otherwise get a model that
    doesn't match its catalog.
    """
    from deriva.core.ermrest_catalog import ErmrestCatalog

    calls = {"n": 0}
    real = ErmrestCatalog.getCatalogSchema

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", counting)

    snap = live_ml.catalog_snapshot(_a_snapshot_id(live_ml))
    assert snap is not None
    assert calls["n"] >= 1, "catalog_snapshot must validate the reused schema against the snapshot"


def test_catalog_snapshot_memoized_per_id(live_ml):
    """Repeated catalog_snapshot() for the same snapshot id returns the same object."""
    sid = _a_snapshot_id(live_ml)
    first = live_ml.catalog_snapshot(sid)
    second = live_ml.catalog_snapshot(sid)
    assert first is second


def test_catalog_snapshot_cache_holds_one_entry_per_id(live_ml):
    """catalog_snapshot caches exactly one instance per snapshot id."""
    sid = _a_snapshot_id(live_ml)
    a = live_ml.catalog_snapshot(sid)
    assert len(live_ml._snapshot_cache) == 1
    b = live_ml.catalog_snapshot(sid)
    assert a is b
    assert len(live_ml._snapshot_cache) == 1


def _model_fingerprint(model) -> dict:
    """A structural fingerprint: {schema: {table: sorted(column names)}}."""
    fp: dict[str, dict[str, list[str]]] = {}
    for sname, schema in model.model.schemas.items():
        fp[sname] = {
            tname: sorted(c.name for c in table.columns)
            for tname, table in schema.tables.items()
        }
    return fp


def test_reused_schema_model_matches_fetched(live_ml):
    """The schema-reusing snapshot model is structurally identical to a fetched one."""
    from deriva.core.ermrest_catalog import ErmrestSnapshot

    # _a_snapshot_id already returns the compound "<catalog_id>@<snaptime>"
    # form that catalog_snapshot / _version_snapshot_catalog_id expect.
    compound_sid = _a_snapshot_id(live_ml)

    reused = live_ml.catalog_snapshot(compound_sid)

    # Build the same snapshot WITHOUT reuse — force a real getCatalogSchema.
    fetched = DerivaML(
        live_ml.host_name,
        compound_sid,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
    )

    # Both must be genuinely snapshot-pinned, not live-catalog connections.
    assert isinstance(reused.catalog, ErmrestSnapshot)
    assert isinstance(fetched.catalog, ErmrestSnapshot)

    assert _model_fingerprint(reused.model) == _model_fingerprint(fetched.model)


# ---------------------------------------------------------------------------
# pathBuilder cache invalidation at in-place create sites
#
# pathBuilder() builds its wrapper from the held model (self.model.model) and
# caches it keyed on inner-model identity. deriva-py's getPathBuilder() purges
# its own schema cache on every schema-mutating POST; by bypassing it we take
# on the obligation to invalidate self._path_builder_cache at EVERY site that
# mutates the inner model in place. These tests pin that obligation for the
# create methods that warm the cache (via add_term -> pathBuilder) BEFORE the
# in-place create_table: without the invalidation, a later same-instance read
# of the new table through pathBuilder returns a stale wrapper -> KeyError.
# ---------------------------------------------------------------------------


def _pathbuilder_table_names(ml) -> set[str]:
    """All table names visible through a fresh pathBuilder() wrapper."""
    pb = ml.pathBuilder()
    return {t for s in pb.schemas.values() for t in s.tables}


def test_pathbuilder_cache_invalidated_after_create_feature(populated_catalog):
    """create_feature must invalidate the warmed pathBuilder cache.

    Warm the cache, create a feature on the demo ``Subject`` table (which
    runs ``add_term`` -> pathBuilder, then an in-place ``create_table`` for
    the feature association table), then read through a fresh pathBuilder.
    The new feature association table must be visible — a stale cached
    wrapper would omit it.
    """
    import uuid

    ml = populated_catalog
    # Warm the cache so a stale wrapper would be returned without invalidation.
    ml.pathBuilder()

    suffix = uuid.uuid4().hex[:6].upper()
    vocab_name = f"PbVocab{suffix}"
    feature_name = f"PbFeat{suffix}"
    ml.create_vocabulary(vocab_name, "Vocab for pathBuilder invalidation test")
    ml.add_term(vocab_name, "Good", description="ok")

    ml.create_feature("Subject", feature_name, terms=[vocab_name])

    # The feature association table is Execution_Subject_<feature_name>.
    all_tables = _pathbuilder_table_names(ml)
    assert any(feature_name in t for t in all_tables), (
        f"feature table for {feature_name} not visible through pathBuilder "
        "-> stale cache (create_feature did not invalidate _path_builder_cache)"
    )


def test_pathbuilder_cache_invalidated_after_create_asset(populated_catalog):
    """create_asset must invalidate the warmed pathBuilder cache.

    Same property as create_feature, exercised through the simpler
    create_asset API: add_term warms the cache, then in-place create_table
    calls add the asset table and its association tables.
    """
    import uuid

    ml = populated_catalog
    ml.pathBuilder()  # warm

    asset_name = f"PbAsset{uuid.uuid4().hex[:6].upper()}"
    ml.create_asset(asset_name, comment="Asset for pathBuilder invalidation test")

    all_tables = _pathbuilder_table_names(ml)
    assert asset_name in all_tables, (
        f"asset table {asset_name} not visible through pathBuilder "
        "-> stale cache (create_asset did not invalidate _path_builder_cache)"
    )


def test_pathbuilder_cache_invalidated_after_add_dataset_element_type(populated_catalog):
    """add_dataset_element_type must invalidate the warmed pathBuilder cache.

    The no-workspace branch neither refreshes nor (previously) invalidated;
    the unconditional clear after the in-place create_table covers both the
    workspace and no-workspace paths.
    """
    ml = populated_catalog
    ml.pathBuilder()  # warm

    ml.add_dataset_element_type("Subject")

    # The association table linking Dataset to Subject must be reachable.
    all_tables = _pathbuilder_table_names(ml)
    assert any("Dataset" in t and "Subject" in t for t in all_tables), (
        "Dataset_Subject association not visible through pathBuilder "
        "-> stale cache (add_dataset_element_type did not invalidate cache)"
    )


# ---------------------------------------------------------------------------
# P4: perf + identity guards
#
# pathBuilder() is built from the held model via datapath.from_model — no
# /schema fetch on every call.  The wrapper is cached keyed on inner-model
# identity; refresh_model() rebinds the inner model, invalidating the cache.
# ---------------------------------------------------------------------------


def _count_schema_gets(monkeypatch) -> dict:
    """Patch DerivaBinding.get to count /schema requests."""
    import deriva.core.deriva_binding as db

    counter = {"schema": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        if isinstance(path, str) and path.split("?")[0].endswith("/schema"):
            counter["schema"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)
    return counter


def test_live_pathbuilder_no_schema_fetch(live_ml, monkeypatch):
    """ml.pathBuilder() builds from the held model with zero /schema GETs."""
    counter = _count_schema_gets(monkeypatch)
    live_ml.pathBuilder()
    live_ml.pathBuilder()
    live_ml.pathBuilder()
    assert counter["schema"] == 0, (
        f"pathBuilder() issued {counter['schema']} /schema GETs; expected 0 "
        "(wrapper is built from the in-memory model)"
    )


def test_live_pathbuilder_cached_identity(live_ml):
    """Repeated pathBuilder() returns the same wrapper until the model rebinds."""
    pb1 = live_ml.pathBuilder()
    pb2 = live_ml.pathBuilder()
    assert pb1 is pb2  # cached on inner-model identity
    live_ml.model.refresh_model()  # rebinds inner model -> cache invalidates
    pb3 = live_ml.pathBuilder()
    assert pb3 is not pb1  # rebuilt after refresh


# ---------------------------------------------------------------------------
# P5: write-through + snapshot-pinning guards
# ---------------------------------------------------------------------------


def test_model_built_wrapper_writes_reach_catalog(populated_catalog):
    """An insert via the model-built pathBuilder lands a real row."""
    import uuid

    ml = populated_catalog
    vname = "WriteVocab" + uuid.uuid4().hex[:6].upper()
    ml.create_vocabulary(vname, "write-through test")
    term = "term_" + uuid.uuid4().hex[:6]
    # add_term goes through pathBuilder().schemas[schema_name].tables[vname].insert(...)
    # where schema_name is resolved from the vocab table's own schema attribute.
    ml.add_term(vname, term, description="x")
    # Locate the vocab table's schema so we read back through the correct path.
    vocab_table = ml.model.name_to_table(vname)
    schema_name = vocab_table.schema.name
    # Read it back through a fresh pathBuilder.
    pb = ml.pathBuilder()
    rows = list(pb.schemas[schema_name].tables[vname].entities().fetch())
    names = {r.get("Name") for r in rows}
    assert term in names, f"inserted term {term!r} not found in {names}"


def test_snapshot_pathbuilder_reads_are_snapshot_pinned(live_ml):
    """A snapshot instance's pathBuilder routes data reads to the @snaptime URI."""
    # _a_snapshot_id already returns the compound form.
    compound = _a_snapshot_id(live_ml)
    snap = live_ml.catalog_snapshot(compound)
    pb = snap.pathBuilder()
    base = pb._wrapped_catalog._server_uri
    assert "@" in base, f"snapshot pathBuilder base URI not pinned: {base}"

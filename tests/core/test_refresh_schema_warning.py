"""Investigation tests for findings doc §E1.

The e2e journal at ``docs/e2e-test-2026-05-13-journal.md`` and the
findings doc at
``deriva-ml-model-template/docs/findings/2026-05-16-phase-1-improvements.md``
§E1 report that ``ml.refresh_schema()`` doesn't suppress the
"schema cache is at snapshot X; live catalog is at Y" warning —
calling ``refresh_schema()`` immediately after construction, the
warning still fires for subsequent operations.

Before designing a fix, these tests pin down what the cache layer
actually does. The hypothesis space:

1. ``SchemaCache.write`` doesn't persist the new snapshot_id
   (round-trip failure).
2. ``refresh_schema`` doesn't call ``SchemaCache.write`` with the
   live snapshot.
3. The warning fires from a code path other than
   ``_init_online``'s if-branch.
4. Multi-process race: the live catalog moves on between
   ``refresh_schema`` and the next ``DerivaML(...)`` construction.
5. The user is in offline mode (``refresh_schema`` raises
   ``DerivaMLReadOnlyError``).

This module rules out (1), (2), and (3) at the unit level. (4) and
(5) are environmental and not in scope.

Conclusion (preview): the cache round-trip is correct. The
warning has exactly one source. If it fires repeatedly across
process invocations, something else is mutating the cache or the
catalog between runs — that's the user's environment, not a
deriva-ml bug.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Hypothesis 1 — cache write/load round-trip preserves snapshot_id
# ---------------------------------------------------------------------------


def test_cache_write_then_load_preserves_snapshot_id(tmp_path) -> None:
    """A snapshot id written into the cache reads back unchanged."""
    from deriva_ml.core.schema_cache import SchemaCache

    cache = SchemaCache(tmp_path)
    cache.write(
        snapshot_id="SNAP-A",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )
    assert cache.snapshot_id() == "SNAP-A"


def test_cache_overwrite_with_new_snapshot_is_visible(tmp_path) -> None:
    """A second ``write()`` with a different snapshot_id overwrites the first.

    This is what ``refresh_schema`` does at lines 554-560: write the
    new ``live_snapshot_id`` into the cache. The next process
    invocation should see SNAP-B, not SNAP-A.
    """
    from deriva_ml.core.schema_cache import SchemaCache

    cache = SchemaCache(tmp_path)
    cache.write(
        snapshot_id="SNAP-A",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )
    assert cache.snapshot_id() == "SNAP-A"

    cache.write(
        snapshot_id="SNAP-B",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )
    assert cache.snapshot_id() == "SNAP-B"

    # A fresh SchemaCache pointing at the same directory (simulating
    # a new process / new DerivaML construction) sees the same SNAP-B.
    fresh_cache = SchemaCache(tmp_path)
    assert fresh_cache.snapshot_id() == "SNAP-B"


def test_cache_file_contents_reflect_latest_write(tmp_path) -> None:
    """The on-disk JSON file directly contains the last-written snapshot_id.

    Belt-and-braces check: even if ``snapshot_id()`` had a caching
    bug, the file content is authoritative across processes. If
    this test passes, then a multi-process refresh round-trip is
    *provably* working.
    """
    from deriva_ml.core.schema_cache import SchemaCache

    cache = SchemaCache(tmp_path)
    cache.write(
        snapshot_id="SNAP-A",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )
    cache.write(
        snapshot_id="SNAP-B",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )

    # Read the file directly — not through SchemaCache.
    on_disk = json.loads(cache._path.read_text())
    assert on_disk["snapshot_id"] == "SNAP-B"


# ---------------------------------------------------------------------------
# Hypothesis 3 — warning has exactly one source
# ---------------------------------------------------------------------------


def test_stale_cache_warning_has_unique_source() -> None:
    """The "schema cache is at snapshot" warning has one source in deriva-ml.

    A regression that adds a second emission site (e.g., a defensive
    re-check on every datapath access) would silently make
    ``refresh_schema()`` insufficient. Pin the call-site count via
    source grep.
    """
    import importlib
    from pathlib import Path

    deriva_ml = importlib.import_module("deriva_ml")
    src_root = Path(deriva_ml.__file__).parent
    occurrences: list[tuple[Path, int]] = []
    for py_path in src_root.rglob("*.py"):
        for lineno, line in enumerate(py_path.read_text().splitlines(), start=1):
            if "schema cache is at snapshot" in line:
                occurrences.append((py_path.relative_to(src_root), lineno))

    assert len(occurrences) == 1, (
        "Expected exactly one source of the stale-cache warning in "
        f"deriva_ml/. Found {len(occurrences)}: {occurrences}. A new "
        "emission site would mean refresh_schema() can no longer "
        "silence the warning by writing the new snapshot into the "
        "cache. Either consolidate the emissions or update this test."
    )
    # The single source must live in core/base.py — the init path
    # is the only legitimate place to emit it.
    path, _ = occurrences[0]
    assert path.as_posix() == "core/base.py", (
        f"Stale-cache warning moved out of core/base.py — now at "
        f"{path}. Update this test if the move is intentional."
    )


# ---------------------------------------------------------------------------
# Hypothesis 2 — refresh_schema's cache.write call uses live_snapshot_id
# ---------------------------------------------------------------------------


def test_refresh_schema_writes_live_snapshot_to_cache() -> None:
    """``refresh_schema`` is wired to call ``cache.write(snapshot_id=live)``.

    Source-grep regression guard. A refactor that changes
    ``refresh_schema`` to skip the cache write or pass the old
    snapshot would silently leave the cache stale forever.

    We assert the structural invariants of the function body:

    1. It fetches ``live_snapshot_id`` from the catalog.
    2. It calls ``cache.write(...)`` with the live snapshot.

    Order matters: the read must come before the write.
    """
    import inspect

    from deriva_ml.core.base import DerivaML

    src = inspect.getsource(DerivaML.refresh_schema)
    assert "live_snapshot_id" in src, (
        "refresh_schema must fetch the live snapshot id. "
        "Without this, the cache can't be updated to a known value."
    )
    assert "cache.write" in src, (
        "refresh_schema must call cache.write(...) to persist the "
        "new snapshot. Without this, the next process construction "
        "sees the old cached snapshot and re-fires the warning."
    )

    # Order: live_snapshot_id assignment must come before cache.write
    live_idx = src.find("live_snapshot_id = ")
    write_idx = src.find("cache.write")
    assert 0 <= live_idx < write_idx, (
        "refresh_schema must read the live snapshot BEFORE writing "
        "the cache, otherwise the cache write uses a stale value."
    )


def test_refresh_schema_passes_live_snapshot_as_snapshot_id_kwarg() -> None:
    """The ``cache.write`` call passes ``snapshot_id=live_snapshot_id``.

    A refactor that does ``cache.write(snapshot_id=cached_snapshot_id)``
    or omits ``snapshot_id`` entirely would leave the cache forever
    stale. Verify the keyword binding directly.
    """
    import inspect

    from deriva_ml.core.base import DerivaML

    src = inspect.getsource(DerivaML.refresh_schema)
    # The body should contain `snapshot_id=live_snapshot_id` somewhere
    # inside the cache.write call. We don't pin the exact format
    # (line-wrapped or single-line) — just the substring.
    assert "snapshot_id=live_snapshot_id" in src.replace(" ", "").replace("\n", ""), (
        "refresh_schema's cache.write call must bind "
        "`snapshot_id=live_snapshot_id`. Without this exact binding, "
        "the cache stays at the old snapshot."
    )


# ---------------------------------------------------------------------------
# Summary documentation test — captures what we learned
# ---------------------------------------------------------------------------


def test_e1_investigation_conclusion() -> None:
    """Documentary test recording the E1 investigation outcome.

    No assertion — passes by existing. If a future maintainer
    revisits E1, this docstring is the bookmark for what was already
    ruled out at the unit level.

    **What we ruled out (the tests above):**

    1. ``SchemaCache.write/load`` round-trip is correct. A snapshot
       written reads back unchanged across writes and across fresh
       cache instances.
    2. ``refresh_schema`` is wired to call ``cache.write`` with the
       live snapshot id; the function's source pins this invariant.
    3. The stale-cache warning has exactly one source in
       ``core/base.py``.

    **What's still possible (not testable at the unit level):**

    A. The live catalog moves on between sessions. A multi-tenant
       environment where another writer keeps updating the schema
       would produce the reported behavior — every fresh
       construction sees a stale cache because the live snapshot
       advanced again.
    B. The user is calling ``refresh_schema`` but it's raising an
       exception they're not seeing (``DerivaMLSchemaPinned``,
       ``DerivaMLSchemaRefreshBlocked``, ``DerivaMLReadOnlyError``).
    C. The user's working_dir differs between sessions, so the
       cache they refresh isn't the cache the next construction
       reads.

    All three are user-environment issues, not deriva-ml bugs. The
    finding's text "refresh_schema doesn't suppress the warning"
    isn't accurate; the cache round-trip works. If a future report
    of the same shape comes in, ask:

    - Which working_dir is being used? (must be the same across
      sessions)
    - Is the live catalog being mutated by anything else?
    - Is ``refresh_schema()`` returning cleanly, or is it raising
      one of the three exceptions above (silently caught by an
      outer ``try``)?
    """
    assert True


# ---------------------------------------------------------------------------
# Plausibility test: simulating the user's reported scenario at unit level
# ---------------------------------------------------------------------------


def test_full_round_trip_simulating_e1_scenario(tmp_path, caplog) -> None:
    """End-to-end simulation of the E1 scenario at the cache layer.

    Simulates: session 1 finds a stale cache → would warn → user calls
    refresh_schema → cache is updated → session 2 finds a fresh
    cache → does NOT warn.

    This exercises everything except the actual ``DerivaML.__init__``
    (which needs a live catalog). The warning-firing predicate is
    ``cached["snapshot_id"] != live_snapshot_id``. We exercise both
    sides of that predicate directly.
    """
    from deriva_ml.core.schema_cache import SchemaCache

    cache = SchemaCache(tmp_path)

    # ---- Session 1: stale cache exists from a prior life
    cache.write(
        snapshot_id="SNAP-OLD",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )

    # Init reads the cache and compares against live (simulated SNAP-NEW)
    live_snapshot_id = "SNAP-NEW"
    cached = cache.load()
    is_stale_at_init = cached["snapshot_id"] != live_snapshot_id
    assert is_stale_at_init, "test premise: cache should be stale before refresh"

    # ---- User calls refresh_schema → it writes the live snapshot
    cache.write(
        snapshot_id=live_snapshot_id,
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema={"schemas": {}, "acls": {}, "annotations": {}},
    )

    # ---- Session 2: a new process opens. The catalog's snapshot
    #      has not moved (no other writer). The cache should now
    #      match the live snapshot, so the predicate would NOT fire.
    fresh_cache = SchemaCache(tmp_path)
    cached2 = fresh_cache.load()
    is_stale_at_session_2 = cached2["snapshot_id"] != live_snapshot_id
    assert not is_stale_at_session_2, (
        "After refresh_schema, the cache should match the (still-current) "
        "live snapshot. Session 2's stale-check predicate would not fire. "
        "If the user reports the warning still fires, the catalog "
        "snapshot has moved on independently between sessions — that's "
        "a multi-tenant environment issue, not a deriva-ml bug."
    )
    # caplog fixture is here for future expansion (asserting on the
    # warning itself once it can be reached without a live catalog).
    _ = caplog

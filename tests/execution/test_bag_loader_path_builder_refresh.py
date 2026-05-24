"""Regression tests for the BagCatalogLoader stale-path-builder bug.

See ``docs/bugs/2026-05-16-bag-loader-stale-path-builder.md`` for
the full bug report. Symptom in production:
``commit_output_assets`` raised ``KeyError`` on a destination
table that exists in the catalog but isn't in the cached
path-builder, because ``ErmrestCatalog.getPathBuilder()`` memoized
per-catalog-instance and didn't see schema mutations made earlier
in the same process.

**Fix lineage:**

1. **Initial deriva-ml-side workaround** in
   ``load_execution_bag`` — force ``getPathBuilder(refresh=True)``
   immediately before constructing ``BagCatalogLoader``.
2. **First upstream fix** in deriva-py's
   ``BagCatalogLoader._ensure_path_builder`` — same idea,
   centralized in the loader.
3. **Second upstream fix** (T1, deriva-py ed5ee69, deriva-ml
   commit ``0f14de7e``): the staleness problem was solved at the
   ``ErmrestCatalog`` layer. Schema mutations through deriva-py
   now automatically invalidate the catalog's schema + path-
   builder caches via the ``_invalidate_schema_cache_if_schema_mutation``
   hook. ``getCatalogSchema()`` and ``getPathBuilder()`` always
   return current state; the explicit ``refresh=True`` flag is
   no longer needed because nothing can become stale relative
   to in-process mutations.

**What this test guards now:** the *existence* of the
auto-invalidation hook on ``ErmrestCatalog``. A deriva-py
refactor that removes or renames that hook without an
equivalent guarantee elsewhere would re-introduce the original
bug. We can't behaviorally test the contract without a live
catalog, but we can pin the API-level invariant: the catalog
class exposes a way to invalidate its schema cache, and
``getCatalogSchema`` / ``getPathBuilder`` exist as the read
entry points that consume it.

If a future deriva-py refactor renames the hook, update this
test to match — and verify the staleness contract still holds
end-to-end via the live-catalog integration suite.
"""

from __future__ import annotations


def test_ermrest_catalog_has_schema_invalidation_hook() -> None:
    """deriva-py's ``ErmrestCatalog`` exposes a schema-cache invalidation hook.

    Per deriva-ml commit 0f14de7e and the deriva-py pin at ed5ee69
    (T1: route all schema reads through deriva-py's
    getCatalogSchema), schema mutations through deriva-py
    automatically invalidate the catalog's schema + path-builder
    caches. The hook lives on ``ErmrestCatalog`` —
    ``_invalidate_schema_cache_if_schema_mutation``.

    A deriva-py change that removes this hook (or its
    functional equivalent — the prefix-purge cache layer) would
    re-introduce the stale-path-builder ``KeyError`` from the
    original 2026-05-16 bug. If this test fails because the
    hook was renamed, update the name; if it fails because the
    hook is genuinely gone with no replacement, that's a
    regression to flag upstream.
    """
    from deriva.core.ermrest_catalog import ErmrestCatalog

    assert hasattr(ErmrestCatalog, "_invalidate_schema_cache_if_schema_mutation"), (
        "deriva-py's ErmrestCatalog is expected to expose "
        "`_invalidate_schema_cache_if_schema_mutation` so schema "
        "mutations through the binding auto-purge the catalog's "
        "schema + path-builder caches. Without this (or an "
        "equivalent replacement), the fresh-catalog upload flow "
        "(create_ml_catalog → create_asset → commit_output_assets) "
        "will regress to KeyError on the newly-created table. "
        "See docs/bugs/2026-05-16-bag-loader-stale-path-builder.md "
        "and deriva-ml commit 0f14de7e for the full story."
    )


def test_ermrest_catalog_exposes_schema_read_entry_points() -> None:
    """The schema-read entry points the auto-invalidation supports are present.

    ``getCatalogSchema`` (parsed-dict access) and ``getPathBuilder``
    (datapath wrapper) are the two read-side APIs that consume
    the catalog's schema cache. The invalidation hook is only
    useful if these read paths exist and route through the cache.

    If either is renamed, deriva-ml's bag-pipeline (which calls
    both) breaks; update this test to match and verify the
    rename is reflected throughout deriva-ml.
    """
    from deriva.core.ermrest_catalog import ErmrestCatalog

    assert hasattr(ErmrestCatalog, "getCatalogSchema"), (
        "deriva-py's ErmrestCatalog is expected to expose "
        "`getCatalogSchema` as the parsed-schema read entry point. "
        "If this is renamed, update both this test and the "
        "deriva-ml call sites that consume it "
        "(notably execution/bag_commit.py)."
    )
    assert hasattr(ErmrestCatalog, "getPathBuilder"), (
        "deriva-py's ErmrestCatalog is expected to expose `getPathBuilder` as the datapath wrapper entry point."
    )

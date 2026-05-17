"""Regression tests for the BagCatalogLoader stale-path-builder bug.

See ``docs/bugs/2026-05-16-bag-loader-stale-path-builder.md`` for
the full bug report. Symptom in production:
``upload_execution_outputs`` raised ``KeyError`` on a destination
table that exists in the catalog but isn't in the cached
path-builder, because ``ErmrestCatalog.getPathBuilder()`` memoizes
per-catalog-instance and didn't see schema mutations made earlier
in the same process.

**Fix lineage:**

1. Initial deriva-ml-side workaround in
   :func:`~deriva_ml.execution.bag_commit.load_execution_bag` —
   force ``getPathBuilder(refresh=True)`` immediately before
   constructing the ``BagCatalogLoader``.
2. Upstream fix landed in deriva-py's
   ``BagCatalogLoader._ensure_path_builder`` (calls
   ``getPathBuilder(refresh=True)`` itself on first build).
3. The deriva-ml workaround became redundant once the upstream
   fix shipped; it was removed alongside the deriva-py pin bump.

**What this test pins now:** the **upstream contract** — that
deriva-py's ``BagCatalogLoader`` does call
``getPathBuilder(refresh=True)`` on the catalog it's given. A
regression in deriva-py that quietly stops calling
``refresh=True`` would re-introduce the production failure. The
deriva-ml-side test guards the version-pinned upstream behavior.

Unit-level: introspects ``BagCatalogLoader``'s source code for
the load-bearing pattern. No live server, no bag, no mocks
beyond ``inspect``.
"""

from __future__ import annotations

import inspect


def test_bag_catalog_loader_refreshes_path_builder() -> None:
    """deriva-py's ``BagCatalogLoader`` must call ``getPathBuilder(refresh=True)``.

    The class's lazy path-builder accessor (``_ensure_path_builder``
    in current deriva-py) is responsible for fetching the
    destination's path-builder with ``refresh=True`` so schema
    mutations made in-process — like the
    ``create_ml_catalog → ml.create_asset(...) → upload`` flow —
    are visible to the loader.

    Regression guard for
    ``docs/bugs/2026-05-16-bag-loader-stale-path-builder.md``: if
    a future deriva-py change quietly drops ``refresh=True`` from
    this code path, every fresh-catalog upload starts producing
    ``KeyError`` on in-process-created tables again.
    """
    from deriva.bag.catalog_loader import BagCatalogLoader

    src = inspect.getsource(BagCatalogLoader)
    assert "getPathBuilder(refresh=True)" in src, (
        "deriva.bag.catalog_loader.BagCatalogLoader is expected to "
        "call `getPathBuilder(refresh=True)` so the destination's "
        "in-process schema mutations are visible. The current "
        "deriva-py pin doesn't carry that call — which means a "
        "fresh-catalog upload flow (create_ml_catalog → "
        "create_asset → upload_execution_outputs) will fail with "
        "KeyError on the newly-created table. "
        "See docs/bugs/2026-05-16-bag-loader-stale-path-builder.md "
        "for the full report. Pin a deriva-py version that includes "
        "the upstream `_ensure_path_builder` fix, or reinstate the "
        "deriva-ml-side workaround in "
        "deriva_ml.execution.bag_commit.load_execution_bag."
    )


def test_bag_catalog_loader_path_builder_helper_exists() -> None:
    """The upstream fix landed as a ``_ensure_path_builder`` helper.

    deriva-py centralizes path-builder access through a single
    helper that handles the refresh-on-first-build invariant. A
    refactor that re-inlines path-builder access at multiple call
    sites without keeping the refresh contract is the kind of
    regression we want to catch.

    The helper name is a load-bearing marker for "the contract is
    being honored in one place" — verify it's still there.
    """
    from deriva.bag.catalog_loader import BagCatalogLoader

    src = inspect.getsource(BagCatalogLoader)
    assert "_ensure_path_builder" in src, (
        "Expected deriva-py's BagCatalogLoader to centralize "
        "path-builder access through a `_ensure_path_builder` "
        "helper. If the helper is renamed, update this test; if "
        "it's gone, verify the refresh=True invariant is still "
        "honored at every call site (and update the previous test "
        "accordingly)."
    )

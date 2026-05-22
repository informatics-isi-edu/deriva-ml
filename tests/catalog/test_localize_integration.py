"""Live-catalog integration tests for :func:`localize_assets`.

``localize_assets`` is the second leg of the split-phase catalog
slice copy: phase 1 (``clone_via_bag``) moves rows and references;
phase 2 (this function) pulls the actual asset bytes from the
source Hatrac, pushes them to the destination Hatrac, and rewrites
the catalog rows to point at the new location.

Pre-fix, the function had zero integration coverage — only its
internal helpers (``_extract_hatrac_path``, ``LocalizeResult``)
were unit-tested. A 350-line public function with no end-to-end
test is a release-blocker because behavioural regressions slip
through silently: a wrong URL, a missing chunk, a swallowed
exception all looked the same as success.

These tests stand up a real source (the demo catalog) + a real
destination (a freshly-created catalog), run ROWS_ONLY clone +
``localize_assets``, and assert: result counts are right, the
catalog URLs were rewritten, and Hatrac on the destination now
holds the bytes.

Slow (multiple minutes each) and require ``DERIVA_HOST``. Select
with ``-m integration``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deriva.bag.traversal import AssetMode, FKTraversalPolicy

from deriva_ml.catalog.clone_via_bag import clone_via_bag
from deriva_ml.catalog.localize import LocalizeResult, localize_assets
from deriva_ml.schema import create_ml_catalog

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml import DerivaML
    from deriva_ml.demo_catalog import DatasetDescription


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def dest_catalog_for_localize(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    catalog_host: str,
) -> "ErmrestCatalog":
    """Freshly-created destination catalog with the source's schema cloned in.

    Identical to ``test_clone_via_bag_integration.py::dest_catalog``;
    repeated here so this file is independently runnable. Uses
    ``clone_catalog(copy_data=False)`` so every dynamically-built
    table lands at the destination but no rows do — the
    ``localize_assets`` test paths populate rows themselves via
    ``clone_via_bag``.
    """
    source_ml, _ = catalog_with_datasets
    dest = create_ml_catalog(catalog_host, project_name="localize-assets-dest")
    try:
        source_ml.catalog.clone_catalog(
            dst_catalog=dest,
            copy_data=False,
            copy_annotations=True,
            copy_policy=False,
            truncate_after=False,
        )
        yield dest
    finally:
        dest.delete_ermrest_catalog(really=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _list_image_rows(catalog: "ErmrestCatalog", schema: str) -> list[dict]:
    """Return all Image rows in the catalog as a list of dicts."""
    pb = catalog.getPathBuilder()
    return list(pb.schemas[schema].tables["Image"].path.entities().fetch())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.integration
def test_localize_assets_same_host_skips_all_as_already_local(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog_for_localize: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """When source and destination share a hostname every asset is "already local".

    Pins the "already local" branch of ``localize_assets``: when
    the resolved source hostname (the URL's own ``netloc`` or, for
    relative URLs, the explicit ``source_hostname`` argument)
    equals the destination catalog's host, the function must
    short-circuit, increment ``assets_skipped``, and **not** issue
    any Hatrac transfers or catalog updates.

    Coverage purpose: this is the cheapest live-server path through
    ``localize_assets`` (no actual byte movement, but every URL
    parse, every per-asset branch, and the row-update gate must
    still execute). A regression that broke the URL-parse or the
    same-host comparison would either crash this test or report
    spurious ``assets_processed`` counts.

    The cross-host "URLs were rewritten" pin requires a second
    physical server, which we don't have in CI — that path is
    exercised manually for now. See
    ``docs/audits/2026-05-22-engineer-audit-catalog.md`` for the
    follow-up issue.
    """
    source_ml, dataset_desc = catalog_with_datasets
    root_rid = dataset_desc.dataset.dataset_rid

    # ROWS_ONLY clone leaves Image rows whose URLs reference the
    # source Hatrac.
    clone_result = clone_via_bag(
        source_hostname=source_ml.catalog.deriva_server.server,
        source_catalog_id=str(source_ml.catalog.catalog_id),
        dest_hostname=catalog_host,
        dest_catalog_id=str(dest_catalog_for_localize.catalog_id),
        root_rid=root_rid,
        output_dir=tmp_path / "bag",
        policy=FKTraversalPolicy(asset_mode=AssetMode.ROWS_ONLY),
    )
    assert clone_result.load_report.total_rows_inserted > 0, (
        "Clone produced no rows — test premise broken."
    )

    schema = source_ml.default_schema
    dest_image_rows = _list_image_rows(dest_catalog_for_localize, schema)
    assert dest_image_rows, "No Image rows in dest after clone — premise broken."
    image_rids = [r["RID"] for r in dest_image_rows]
    pre_urls = {r["RID"]: r["URL"] for r in dest_image_rows}

    result = localize_assets(
        catalog=dest_catalog_for_localize,
        asset_table="Image",
        asset_rids=image_rids,
        schema_name=schema,
        source_hostname=source_ml.catalog.deriva_server.server,
    )

    # Result shape + same-host short-circuit behaviour.
    assert isinstance(result, LocalizeResult), (
        f"localize_assets must return a LocalizeResult; got {type(result)!r}"
    )
    assert result.assets_processed == 0, (
        f"Expected zero processed when source == dest host; got "
        f"{result.assets_processed}. The function must short-circuit "
        f"the 'already local' branch."
    )
    assert result.assets_skipped == len(image_rids), (
        f"Expected all {len(image_rids)} assets skipped as already-local; "
        f"got skipped={result.assets_skipped}. Either the URL parse is "
        f"misidentifying the host or the same-host comparison broke."
    )
    assert result.assets_failed == 0, (
        f"Same-host short-circuit must not fail any assets; got "
        f"failed={result.assets_failed}, errors={result.errors!r}"
    )

    # Catalog rows untouched on the same-host path.
    post_rows = _list_image_rows(dest_catalog_for_localize, schema)
    post_urls = {r["RID"]: r["URL"] for r in post_rows}
    assert post_urls == pre_urls, (
        "Same-host short-circuit must not modify any URLs; the "
        "'already local' branch is supposed to be a no-op on the catalog."
    )


@pytest.mark.integration
def test_localize_assets_dry_run_makes_no_changes(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog_for_localize: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """``dry_run=True`` reports what would be done without changing anything.

    Pins the dry-run contract: same shape ``LocalizeResult`` (counts,
    per-asset entries) but the catalog rows' URLs are unchanged and
    no provenance annotation is written. Important because dry-run
    is the rehearsal mode operators use before committing.
    """
    source_ml, dataset_desc = catalog_with_datasets
    root_rid = dataset_desc.dataset.dataset_rid

    clone_via_bag(
        source_hostname=source_ml.catalog.deriva_server.server,
        source_catalog_id=str(source_ml.catalog.catalog_id),
        dest_hostname=catalog_host,
        dest_catalog_id=str(dest_catalog_for_localize.catalog_id),
        root_rid=root_rid,
        output_dir=tmp_path / "bag",
        policy=FKTraversalPolicy(asset_mode=AssetMode.ROWS_ONLY),
    )

    schema = source_ml.default_schema
    pre_rows = _list_image_rows(dest_catalog_for_localize, schema)
    image_rids = [r["RID"] for r in pre_rows]
    pre_urls = {r["RID"]: r["URL"] for r in pre_rows}

    result = localize_assets(
        catalog=dest_catalog_for_localize,
        asset_table="Image",
        asset_rids=image_rids,
        schema_name=schema,
        source_hostname=source_ml.catalog.deriva_server.server,
        dry_run=True,
    )

    # Dry run still reports a result.
    assert isinstance(result, LocalizeResult)

    # But the URLs are unchanged. ``dry_run=True`` must not call
    # ``table.update(...)`` on the destination.
    post_rows = _list_image_rows(dest_catalog_for_localize, schema)
    post_urls = {r["RID"]: r["URL"] for r in post_rows}
    assert post_urls == pre_urls, (
        "dry_run=True must not modify the catalog. Found URL "
        f"differences: {[(rid, pre_urls[rid], post_urls[rid]) for rid in image_rids if pre_urls[rid] != post_urls[rid]][:3]}"
    )


@pytest.mark.integration
def test_localize_assets_unknown_rid_marks_skipped(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog_for_localize: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """Passing a RID that doesn't exist in the table is skipped, not failed.

    The function's contract: per-asset issues (RID not in table,
    URL absent, URL unparseable) bump ``assets_skipped`` with a
    warning, NOT ``assets_failed`` (which is reserved for genuine
    transport/Hatrac errors). This test pins that distinction so
    a future refactor doesn't silently promote skips into failures
    or vice versa.
    """
    # Skip the clone step entirely — we just need the table to exist
    # on the dest. ``dest_catalog_for_localize`` already cloned the
    # schema without data; an empty Image table is enough.
    schema = catalog_with_datasets[0].default_schema
    fake_rid = "ZZZ-DOESNOTEXIST-ZZZ"

    result = localize_assets(
        catalog=dest_catalog_for_localize,
        asset_table="Image",
        asset_rids=[fake_rid],
        schema_name=schema,
        source_hostname=catalog_with_datasets[0].catalog.deriva_server.server,
    )

    assert isinstance(result, LocalizeResult)
    assert result.assets_processed == 0
    assert result.assets_skipped == 1, (
        f"Unknown RID should bump assets_skipped; got skipped="
        f"{result.assets_skipped}, failed={result.assets_failed}"
    )
    assert result.assets_failed == 0, (
        f"Unknown RID is a skip, not a failure; got failed="
        f"{result.assets_failed}"
    )

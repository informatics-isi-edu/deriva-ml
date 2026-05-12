"""Live-catalog integration tests for :func:`clone_via_bag`.

These tests stand up a real source catalog (populated by the demo
fixtures), build a bag, load it into a freshly-created destination
catalog, and verify that the destination's row counts match the
source. They cover the bag pipeline end-to-end — the network round
trips, the export-engine driver, the bag's on-disk artifact, the
loader's FK-safe insert ordering, and (where applicable) the policy
parameters that route asset bytes.

Each test creates and destroys its own destination catalog so that
runs are isolated. Source catalogs come from the session-scoped
``catalog_manager`` fixture.

These tests are slow (multiple minutes each) and require ``DERIVA_HOST``.
Select them with ``-m integration`` and expect to run them rarely.

**Currently xfailed**: each test is marked
``pytest.mark.xfail(strict=False)`` pending the
:class:`~deriva.bag.catalog_loader.BagCatalogLoader` conflict-policy
rewrite tracked by `deriva-py#214
<https://github.com/informatics-isi-edu/deriva-py/issues/214>`_ (per
`deriva-py ADR-0001
<https://github.com/informatics-isi-edu/deriva-py/blob/deriva-ml/docs/adr/0001-bag-catalog-loader-conflict-and-system-content.md>`_).
The loader currently fails on RID conflicts when the destination
catalog has pre-populated system vocabulary (which every catalog
created via ``create_ml_catalog`` does). When the rewrite lands and
the loader does vocabulary-by-name reconciliation, these tests will
flip to PASS automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deriva.bag.anchors import RIDAnchor
from deriva.bag.traversal import (
    AssetMode,
    DanglingFKStrategy,
    FKTraversalPolicy,
)

from deriva_ml.catalog.clone_via_bag import (
    CloneViaBagResult,
    clone_via_bag,
)
from deriva_ml.schema import create_ml_catalog

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml import DerivaML
    from deriva_ml.demo_catalog import DatasetDescription


# ----------------------------------------------------------------------
# Shared markers
# ----------------------------------------------------------------------


#: Each end-to-end test is xfailed pending the
#: :class:`BagCatalogLoader` conflict-policy rewrite tracked by
#: deriva-py#214 (per deriva-py ADR-0001). ``strict=False`` so the
#: tests flip to ``XPASS`` and surface a green diff the moment the
#: upstream rewrite lands. The reason is in module docstring.
_AWAITING_LOADER_REWRITE = pytest.mark.xfail(
    reason="awaits BagCatalogLoader rewrite per deriva-py ADR-0001 (issue #214)",
    strict=False,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def dest_catalog(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    catalog_host: str,
) -> "ErmrestCatalog":
    """Freshly-created destination catalog with the source's schema cloned in.

    Matches :class:`BagCatalogLoader`'s ADR-0001 precondition: the
    destination is schema-ready. Uses ``clone_catalog(copy_data=False)``
    so every dynamically-built table (feature association tables,
    SubjectHealth/ImageQuality vocabs, etc.) lands at the destination.
    """
    source_ml, _ = catalog_with_datasets
    dest = create_ml_catalog(catalog_host, project_name="clone-via-bag-dest")
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


def _row_count(catalog: "ErmrestCatalog", schema: str, table: str) -> int:
    """Return the row count of ``schema.table`` via path builder."""
    pb = catalog.getPathBuilder()
    return len(list(pb.schemas[schema].tables[table].path.entities().fetch()))


def _count_by_table(catalog: "ErmrestCatalog", schema: str, tables: list[str]) -> dict[str, int]:
    """Return ``{table: row_count}`` for each table in ``tables``."""
    return {t: _row_count(catalog, schema, t) for t in tables}


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.integration
@_AWAITING_LOADER_REWRITE
def test_clone_via_bag_end_to_end_default_policy(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """Default policy clones a Dataset-anchored slice end-to-end.

    Builds a bag rooted at the top-level Dataset RID from a source
    catalog populated by the demo fixture, then loads it into a
    fresh destination catalog. The destination must contain the
    same number of rows in the reachable tables as the source.
    """
    source_ml, dataset_desc = catalog_with_datasets
    root_rid = dataset_desc.dataset.dataset_rid

    # Snapshot source row counts for tables we expect the walk to
    # reach from the Dataset anchor.
    domain_tables = ["Subject", "Image", "Observation"]
    src_domain = _count_by_table(source_ml.catalog, source_ml.default_schema, domain_tables)
    src_dataset = _row_count(source_ml.catalog, "deriva-ml", "Dataset")
    assert src_dataset > 0, "demo fixture should have created datasets"
    assert src_domain["Subject"] > 0
    assert src_domain["Image"] > 0

    output_dir = tmp_path / "bag"

    result = clone_via_bag(
        source_hostname=source_ml.catalog.deriva_server.server,
        source_catalog_id=str(source_ml.catalog.catalog_id),
        dest_hostname=catalog_host,
        dest_catalog_id=str(dest_catalog.catalog_id),
        root_rid=root_rid,
        output_dir=output_dir,
    )

    # Smoke checks on the result object.
    assert isinstance(result, CloneViaBagResult)
    assert result.source_catalog_id == str(source_ml.catalog.catalog_id)
    assert result.dest_catalog_id == str(dest_catalog.catalog_id)
    assert result.bag_path.exists()
    assert result.bag_path.is_dir()
    assert result.load_report.total_rows_inserted > 0

    # Destination row counts match what the source had for the
    # reachable tables.
    dst_domain = _count_by_table(dest_catalog, source_ml.default_schema, domain_tables)
    assert dst_domain == src_domain
    assert _row_count(dest_catalog, "deriva-ml", "Dataset") == src_dataset


@pytest.mark.integration
@_AWAITING_LOADER_REWRITE
def test_clone_via_bag_rows_only_skips_asset_uploads(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """``AssetMode.ROWS_ONLY`` inserts rows but uploads zero asset bytes.

    The ROWS_ONLY policy is the cheap path: the bag still records
    the Image rows but the loader doesn't push their byte contents
    into the destination Hatrac. We verify both: row counts match,
    and the load report shows ``assets_uploaded == 0``.
    """
    source_ml, dataset_desc = catalog_with_datasets
    root_rid = dataset_desc.dataset.dataset_rid

    policy = FKTraversalPolicy(
        asset_mode=AssetMode.ROWS_ONLY,
        dangling_fk_strategy=DanglingFKStrategy.FAIL,
    )

    result = clone_via_bag(
        source_hostname=source_ml.catalog.deriva_server.server,
        source_catalog_id=str(source_ml.catalog.catalog_id),
        dest_hostname=catalog_host,
        dest_catalog_id=str(dest_catalog.catalog_id),
        root_rid=root_rid,
        output_dir=tmp_path / "bag",
        policy=policy,
    )

    # No asset uploads happened.
    total_assets_uploaded = sum(s.assets_uploaded for s in result.load_report.table_stats.values())
    assert total_assets_uploaded == 0, f"ROWS_ONLY must not upload asset bytes, got {total_assets_uploaded} uploads"

    # Image rows still landed.
    src_images = _row_count(source_ml.catalog, source_ml.default_schema, "Image")
    dst_images = _row_count(dest_catalog, source_ml.default_schema, "Image")
    assert dst_images == src_images


@pytest.mark.integration
@_AWAITING_LOADER_REWRITE
def test_clone_via_bag_rid_anchor_scopes_walk(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """A single-RID anchor walks only the reachable slice, not the full catalog.

    Anchored at one Subject RID, the destination should land **one**
    Subject (not the full source population). Verifies the bag walker
    is honoring anchor scope rather than dumping the whole table.
    """
    source_ml, _ = catalog_with_datasets

    # Pick one Subject RID to anchor at.
    pb = source_ml.catalog.getPathBuilder()
    subjects = list(pb.schemas[source_ml.default_schema].tables["Subject"].path.entities().fetch())
    src_total_subjects = len(subjects)
    assert src_total_subjects > 1, (
        "demo fixture should have multiple subjects so the scoped walk has something to exclude"
    )
    one_subject_rid = subjects[0]["RID"]

    result = clone_via_bag(
        source_hostname=source_ml.catalog.deriva_server.server,
        source_catalog_id=str(source_ml.catalog.catalog_id),
        dest_hostname=catalog_host,
        dest_catalog_id=str(dest_catalog.catalog_id),
        anchors=[RIDAnchor(table="Subject", rids=[one_subject_rid])],
        output_dir=tmp_path / "bag",
    )

    assert result.load_report.total_rows_inserted > 0

    dst_subjects = _row_count(dest_catalog, source_ml.default_schema, "Subject")
    assert dst_subjects == 1, (
        f"expected exactly the anchored Subject, got {dst_subjects} (source had {src_total_subjects})"
    )

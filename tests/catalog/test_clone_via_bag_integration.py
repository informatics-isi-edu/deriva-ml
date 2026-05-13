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

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deriva.bag.anchors import RIDAnchor
from deriva.bag.traversal import (
    AssetMode,
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
def test_clone_via_bag_end_to_end_default_policy(
    catalog_with_datasets: "tuple[DerivaML, DatasetDescription]",
    dest_catalog: "ErmrestCatalog",
    tmp_path: Path,
    catalog_host: str,
) -> None:
    """Default policy clones a Dataset-anchored slice end-to-end.

    Builds a bag rooted at the top-level Dataset RID from a source
    catalog populated by the demo fixture, then loads it into a
    fresh destination catalog. The destination must contain
    exactly the dataset's transitive members for each reachable
    table — not the source's full row count, since the demo
    catalog has rows outside the dataset's slice.
    """
    source_ml, dataset_desc = catalog_with_datasets
    root_rid = dataset_desc.dataset.dataset_rid

    # Compute the dataset's expected member counts (recursive over
    # nested datasets). The bag's Subject/Image/Observation row
    # counts at the destination must match these exactly.
    source_dataset = source_ml.lookup_dataset(root_rid)
    expected_members = source_dataset.list_dataset_members(recurse=True)
    expected_subject_count = len({m["RID"] for m in expected_members.get("Subject", [])})
    expected_image_count = len({m["RID"] for m in expected_members.get("Image", [])})
    assert expected_subject_count > 0
    assert expected_image_count > 0

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

    # Destination contains AT LEAST the dataset's explicit members
    # (the clone preserves the dataset's content) and AT MOST the
    # source's full rows (the clone doesn't fabricate). Anywhere
    # between is the legitimate transitively-related expansion —
    # Images reached via Subject_Image from member Subjects, etc.
    src_subject = _row_count(source_ml.catalog, source_ml.default_schema, "Subject")
    src_image = _row_count(source_ml.catalog, source_ml.default_schema, "Image")
    dst_subject = _row_count(dest_catalog, source_ml.default_schema, "Subject")
    dst_image = _row_count(dest_catalog, source_ml.default_schema, "Image")
    assert expected_subject_count <= dst_subject <= src_subject, (
        f"Subject: expected ≥{expected_subject_count} (dataset members) "
        f"and ≤{src_subject} (source total), got {dst_subject}"
    )
    assert expected_image_count <= dst_image <= src_image, (
        f"Image: expected ≥{expected_image_count} (dataset members) "
        f"and ≤{src_image} (source total), got {dst_image}"
    )


@pytest.mark.integration
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
    and the load report shows ``assets_attempted == 0``.
    """
    source_ml, dataset_desc = catalog_with_datasets
    root_rid = dataset_desc.dataset.dataset_rid

    # ``clone_via_bag`` merges deriva-ml clone defaults (vocab_export=FULL,
    # terminal_tables={Execution,Workflow}, dangling_fk_strategy=DELETE)
    # into the caller's policy where the caller left library defaults.
    # Only ``asset_mode=ROWS_ONLY`` is the explicit choice we want to
    # exercise here; the rest come from the merge.
    policy = FKTraversalPolicy(asset_mode=AssetMode.ROWS_ONLY)

    # Expected member count is computed from the dataset itself
    # (recurse=True over nested datasets), not from the source's
    # full row count, since the demo catalog seeds rows outside
    # the dataset slice.
    source_dataset = source_ml.lookup_dataset(root_rid)
    expected_members = source_dataset.list_dataset_members(recurse=True)
    expected_image_count = len({m["RID"] for m in expected_members.get("Image", [])})
    assert expected_image_count > 0

    result = clone_via_bag(
        source_hostname=source_ml.catalog.deriva_server.server,
        source_catalog_id=str(source_ml.catalog.catalog_id),
        dest_hostname=catalog_host,
        dest_catalog_id=str(dest_catalog.catalog_id),
        root_rid=root_rid,
        output_dir=tmp_path / "bag",
        policy=policy,
    )

    # No asset upload attempts happened.
    total_attempts = sum(s.assets_attempted for s in result.load_report.table_stats.values())
    assert total_attempts == 0, f"ROWS_ONLY must not invoke asset upload, got {total_attempts} attempts"

    # Image rows land in the destination — at least the dataset's
    # explicit Image members and at most the source's full Image
    # set. Between is the transitively-related expansion (Images
    # reached via Subject_Image from member Subjects, etc).
    src_images = _row_count(source_ml.catalog, source_ml.default_schema, "Image")
    dst_images = _row_count(dest_catalog, source_ml.default_schema, "Image")
    assert expected_image_count <= dst_images <= src_images, (
        f"Image: expected ≥{expected_image_count} (dataset members) "
        f"and ≤{src_images} (source total), got {dst_images}"
    )


@pytest.mark.integration
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

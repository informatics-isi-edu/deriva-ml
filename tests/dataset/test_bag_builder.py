"""Tests for :class:`deriva_ml.dataset.bag_builder.DatasetBagBuilder`.

The headline test is the **bag-content equivalence harness**:
the new :class:`DatasetBagBuilder` (driving
:class:`CatalogBagBuilder`) must produce bags whose *contents*
match the legacy :class:`CatalogGraph`-driven bags for the same
``(catalog, dataset)`` input. Spec internals (output_path
strings, query_processor ordering, bag metadata) are *not*
required to match — per decision D1 in
``docs/design/dataset-bag-cutover-2026-05.md``, we care about
the bag's externally observable contents (rows + assets), not
its on-disk spec representation.

Equivalence is checked at two layers:

1. **Per-table row sets** — every CSV in the bag, joined into
   the bag's SQLAlchemy ORM, must contain the same RID set on
   both sides.
2. **Asset RID + MD5 sets** — every asset table must reference
   the same RIDs with the same MD5 checksums on both sides
   (bytes themselves are not transferred during equivalence
   testing; the fetch.txt manifest is sufficient).

The harness is **load-bearing for the cutover** — it gates the
deletion of ``CatalogGraph`` — and **disposable after the
cutover** (will be deleted in the same commit that deletes
``CatalogGraph``).

The thinner unit tests below verify the bag-pipeline-shaped
helpers (:meth:`DatasetBagBuilder.anchors_for`,
:meth:`DatasetBagBuilder.build_policy`) — they exercise the
dataset-anchor + policy logic without driving a full export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from deriva.bag.anchors import RIDAnchor
from deriva.bag.traversal import FKTraversalPolicy, VocabExport
from deriva_ml.dataset.bag_builder import DatasetBagBuilder


# ---------------------------------------------------------------------------
# Bag-content equivalence harness — gated on catalog_with_datasets fixture
# ---------------------------------------------------------------------------


def _build_bag_via_catalog_graph(
    ml: Any,
    dataset: Any,
    out_dir: Path,
) -> Path:
    """Build a bag via the legacy ``CatalogGraph`` spec generator.

    Uses :class:`deriva.transfer.download.deriva_download.GenericDownloader`
    on the spec ``CatalogGraph.generate_dataset_download_spec`` produces.
    Returns the path to the resulting bag directory.
    """
    from deriva.transfer.download.deriva_download import GenericDownloader

    from deriva_ml.dataset.catalog_graph import CatalogGraph

    spec = CatalogGraph(ml_instance=ml).generate_dataset_download_spec(dataset)
    # The spec uses {RID} templates; fill them in.
    spec = _bind_rid_template(spec, dataset.dataset_rid)

    deriva_server = ml.catalog.deriva_server
    downloader = GenericDownloader(
        server={
            "host": deriva_server.server,
            "protocol": deriva_server.scheme,
            "catalog_id": str(ml.catalog.catalog_id),
        },
        config=spec,
        output_dir=str(out_dir),
        credentials=ml.catalog._credentials,
    )
    downloader.download()
    # Find the resulting bag — the spec named it Dataset_{RID}.
    bag_name = spec["bag"]["bag_name"]
    bag_path = out_dir / bag_name
    if not bag_path.exists():
        # GenericDownloader sometimes leaves bag under another name;
        # fall back to a single-subdirectory lookup.
        candidates = [p for p in out_dir.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            bag_path = candidates[0]
    return bag_path


def _build_bag_via_dataset_bag_builder(
    ml: Any,
    dataset: Any,
    out_dir: Path,
) -> Path:
    """Build a bag via the new ``DatasetBagBuilder`` spec generator.

    Mirrors :func:`_build_bag_via_catalog_graph` but routes through
    the new spec generator.
    """
    from deriva.transfer.download.deriva_download import GenericDownloader

    spec = DatasetBagBuilder(
        ml_instance=ml
    ).generate_dataset_download_spec(dataset)
    spec = _bind_rid_template(spec, dataset.dataset_rid)

    deriva_server = ml.catalog.deriva_server
    downloader = GenericDownloader(
        server={
            "host": deriva_server.server,
            "protocol": deriva_server.scheme,
            "catalog_id": str(ml.catalog.catalog_id),
        },
        config=spec,
        output_dir=str(out_dir),
        credentials=ml.catalog._credentials,
    )
    downloader.download()
    bag_name = spec["bag"]["bag_name"]
    bag_path = out_dir / bag_name
    if not bag_path.exists():
        candidates = [p for p in out_dir.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            bag_path = candidates[0]
    return bag_path


def _bind_rid_template(spec: dict, rid: str) -> dict:
    """Recursively replace ``{RID}`` placeholders with the concrete RID.

    The export-engine spec uses ``{RID}`` templates that Chaise
    fills at click-time. For programmatic test invocation, we
    substitute the value directly.
    """
    import json
    import re

    text = json.dumps(spec)
    text = re.sub(r"\{RID\}", rid, text)
    return json.loads(text)


def _bag_table_rid_sets(bag_path: Path) -> dict[str, set[str]]:
    """Return ``{table_name: {rid, ...}}`` for every CSV in a bag.

    Walks every ``*.csv`` under ``data/`` and reads its ``RID``
    column. The bag may have the same table appearing under
    multiple FK-path subdirectories; we union those into a single
    RID set per table.
    """
    import csv

    out: dict[str, set[str]] = {}
    data_dir = bag_path / "data"
    if not data_dir.exists():
        return out
    for csv_path in data_dir.rglob("*.csv"):
        table_name = csv_path.stem
        rids = out.setdefault(table_name, set())
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "RID" in row and row["RID"]:
                    rids.add(row["RID"])
    return out


def _bag_asset_md5_sets(bag_path: Path) -> dict[str, set[tuple[str, str]]]:
    """Return ``{asset_table: {(rid, md5), ...}}`` for every asset in a bag.

    Reads ``fetch.txt`` (and any inline asset CSVs) to enumerate
    the asset rows the bag references. Each entry is a
    ``(RID, MD5)`` pair; we compare these sets across the two
    bags so the comparison is independent of byte transfer.
    """
    out: dict[str, set[tuple[str, str]]] = {}
    # The asset CSVs (one per asset table, under data/) carry the
    # RID + MD5 we need. fetch.txt has filenames, not RIDs, so
    # the CSVs are the better source.
    data_dir = bag_path / "data"
    if not data_dir.exists():
        return out

    import csv

    for csv_path in data_dir.rglob("*.csv"):
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        if not rows:
            continue
        # Detect asset rows by the presence of MD5 + Filename columns.
        if "MD5" not in rows[0] or "Filename" not in rows[0]:
            continue
        table_name = csv_path.stem
        bucket = out.setdefault(table_name, set())
        for row in rows:
            rid = row.get("RID")
            md5 = row.get("MD5")
            if rid and md5:
                bucket.add((rid, md5))
    return out


class TestBagEquivalence:
    """The new bag must contain the same rows + assets as the legacy bag.

    Equivalence is the **load-bearing gate** for the cutover.
    These tests must pass before ``CatalogGraph`` is deleted. The
    harness builds two bags side by side, opens both, and
    compares their externally observable contents.

    Per decision D1 in the cutover design doc, spec internals
    (output_path strings, processor ordering, bag metadata) are
    *not* checked — bag *contents* are.
    """

    def test_bag_row_sets_equivalent(
        self, catalog_with_datasets, tmp_path: Path
    ) -> None:
        """Same RID set per table across both bag-construction paths."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset in the fixture.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)

        legacy_dir = tmp_path / "legacy"
        new_dir = tmp_path / "new"
        legacy_dir.mkdir()
        new_dir.mkdir()

        legacy_bag = _build_bag_via_catalog_graph(ml, dataset, legacy_dir)
        new_bag = _build_bag_via_dataset_bag_builder(ml, dataset, new_dir)

        legacy_rids = _bag_table_rid_sets(legacy_bag)
        new_rids = _bag_table_rid_sets(new_bag)

        # Same tables present, same RIDs per table.
        legacy_tables = set(legacy_rids.keys())
        new_tables = set(new_rids.keys())

        missing_in_new = legacy_tables - new_tables
        extra_in_new = new_tables - legacy_tables
        assert not missing_in_new, (
            f"Tables present in CatalogGraph's bag but missing from "
            f"DatasetBagBuilder's bag: {sorted(missing_in_new)}"
        )
        assert not extra_in_new, (
            f"Tables present in DatasetBagBuilder's bag but absent "
            f"from CatalogGraph's bag: {sorted(extra_in_new)}"
        )

        for table in sorted(legacy_tables):
            assert legacy_rids[table] == new_rids[table], (
                f"Row-set mismatch for table {table}: "
                f"legacy has {len(legacy_rids[table])} RIDs, "
                f"new has {len(new_rids[table])}; "
                f"diff = {legacy_rids[table] ^ new_rids[table]}"
            )

    def test_bag_asset_md5_sets_equivalent(
        self, catalog_with_datasets, tmp_path: Path
    ) -> None:
        """Same ``(RID, MD5)`` set per asset table across both paths."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset in the fixture.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)

        legacy_dir = tmp_path / "legacy"
        new_dir = tmp_path / "new"
        legacy_dir.mkdir()
        new_dir.mkdir()

        legacy_bag = _build_bag_via_catalog_graph(ml, dataset, legacy_dir)
        new_bag = _build_bag_via_dataset_bag_builder(ml, dataset, new_dir)

        legacy_assets = _bag_asset_md5_sets(legacy_bag)
        new_assets = _bag_asset_md5_sets(new_bag)

        # Asset tables match (a table is an "asset table" when it
        # has MD5 + Filename columns; both bags should agree).
        assert set(legacy_assets.keys()) == set(new_assets.keys()), (
            f"Asset-table set mismatch: legacy={sorted(legacy_assets)}, "
            f"new={sorted(new_assets)}"
        )
        for table in sorted(legacy_assets):
            assert legacy_assets[table] == new_assets[table], (
                f"Asset (RID, MD5) mismatch for {table}: "
                f"diff = {legacy_assets[table] ^ new_assets[table]}"
            )


# ---------------------------------------------------------------------------
# Spec smoke tests — confirm the new spec generator is callable
# ---------------------------------------------------------------------------


class TestSpecSmoke:
    """Light tests confirming the new spec generator runs end-to-end.

    Not equivalence tests — those live in :class:`TestBagEquivalence`.
    These exercise the spec/annotation methods enough to catch
    construction-level errors (missing imports, attribute access)
    that the harness would otherwise only surface as a download
    failure.
    """

    def test_spec_runs_without_error(
        self, catalog_with_datasets
    ) -> None:
        """``generate_dataset_download_spec`` produces a dict with the right keys."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)

        spec = DatasetBagBuilder(
            ml_instance=ml
        ).generate_dataset_download_spec(dataset)
        assert "env" in spec
        assert "bag" in spec
        assert "catalog" in spec
        assert spec["bag"]["bag_name"] == "Dataset_{RID}"
        assert spec["env"]["RID"] == "{RID}"
        assert "query_processors" in spec["catalog"]

    def test_annotations_run_without_error(
        self, catalog_with_datasets
    ) -> None:
        """``generate_dataset_download_annotations`` produces a dict with the right keys."""
        ml, _ = catalog_with_datasets
        ann = DatasetBagBuilder(
            ml_instance=ml
        ).generate_dataset_download_annotations()
        from deriva.core.utils.core_utils import tag as deriva_tags

        assert deriva_tags.export_fragment_definitions in ann
        assert deriva_tags.visible_foreign_keys in ann
        assert deriva_tags.export_2019 in ann

    def test_aggregate_queries_runs_without_error(
        self, catalog_with_datasets
    ) -> None:
        """``aggregate_queries`` produces a non-empty dict for a real dataset."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        result = DatasetBagBuilder(ml_instance=ml).aggregate_queries(dataset)
        # Should produce at least one entry for the Dataset table.
        assert "Dataset" in result
        for table_name, entries in result.items():
            for dp, pb_table, is_asset in entries:
                assert dp is not None
                assert pb_table is not None
                assert isinstance(is_asset, bool)


# ---------------------------------------------------------------------------
# Bag-pipeline-shaped helpers
# ---------------------------------------------------------------------------


class TestAnchorsAndPolicy:
    """The ``anchors_for`` + ``build_policy`` helpers expose the bag-pipeline form.

    These methods are the ADR-0006 surface — a future
    ``Dataset.download_dataset_bag`` rewrite that calls
    :class:`CatalogBagBuilder` directly will pull anchors + policy
    from here.
    """

    def test_anchors_for_dataset_no_children(
        self, catalog_with_datasets
    ) -> None:
        """A dataset with no nested children yields a single anchor."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        # Pick the first dataset that has no children; if all
        # datasets in the fixture have children, skip.
        candidate = None
        for ds_row in datasets:
            ds = ml.lookup_dataset(ds_row.dataset_rid)
            if not list(ds.list_dataset_children()):
                candidate = ds
                break
        if candidate is None:
            pytest.skip(
                "Need a dataset with no children to test the single-"
                "anchor path; fixture has only nested datasets."
            )

        builder = DatasetBagBuilder(ml_instance=ml)
        anchors = builder.anchors_for(candidate)
        assert len(anchors) == 1
        anchor = anchors[0]
        assert isinstance(anchor, RIDAnchor)
        assert anchor.table == "Dataset"
        assert anchor.rids == [candidate.dataset_rid]

    def test_build_policy_default(self, catalog_with_datasets) -> None:
        """Default policy enables full-vocab export."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        builder = DatasetBagBuilder(ml_instance=ml)
        policy = builder.build_policy(dataset)
        assert isinstance(policy, FKTraversalPolicy)
        assert policy.vocab_export is VocabExport.FULL

    def test_build_policy_respects_vocab_export_override(
        self, catalog_with_datasets
    ) -> None:
        """``vocab_export=REFERENCED_ONLY`` flows through to the policy."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        builder = DatasetBagBuilder(ml_instance=ml)
        policy = builder.build_policy(
            dataset, vocab_export=VocabExport.REFERENCED_ONLY
        )
        assert policy.vocab_export is VocabExport.REFERENCED_ONLY

    def test_build_policy_includes_user_exclude_tables(
        self, catalog_with_datasets
    ) -> None:
        """User-supplied ``exclude_tables`` make it into the policy."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)

        # Pick a real table from the catalog so the exclusion is
        # actually applied (the schema lookup needs to resolve it).
        # Dataset is always present; using it is harmless because
        # the user-exclude logic only flags it, not removes Dataset
        # from the walk (the anchor still pins it).
        builder = DatasetBagBuilder(
            ml_instance=ml, exclude_tables={"Dataset"}
        )
        policy = builder.build_policy(dataset)
        # We don't assert the exact set — Dataset_Version etc. may
        # also land in policy.exclude_tables via the empty-
        # association filter. We just assert the user's input made
        # it in.
        assert any(
            table_name == "Dataset"
            for _schema, table_name in policy.exclude_tables
        )

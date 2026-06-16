"""Tests for Format-B (rid-set) bag generation."""

import os
import zipfile
from collections import Counter

import pytest

from deriva_ml.dataset.bag_builder import _rid_sets_from_reachability


def test_rid_sets_from_reachability_tuple_keys_and_drops_vocab():
    # reached: (schema, table) -> [fk_path, ...]  (paths irrelevant here)
    reached = {
        ("eye-ai", "Image"): [()],
        ("eye-ai", "Subject"): [()],
        ("deriva-ml", "Asset_Role"): [()],  # vocab
    }
    rids_by_table = {
        "Image": {"r1", "r2"},
        "Subject": {"s1"},
        "Asset_Role": {"Input", "Output"},  # bare names; vocab
    }
    vocab_tables = {("deriva-ml", "Asset_Role")}
    result = _rid_sets_from_reachability(reached, rids_by_table, vocab_tables)
    # Tuple-keyed, vocab dropped, RID lists sorted for determinism.
    assert result == {
        ("eye-ai", "Image"): ["r1", "r2"],
        ("eye-ai", "Subject"): ["s1"],
    }


def test_rid_sets_from_reachability_missing_table_is_empty_list():
    """A reached non-vocab table with no RID-set entry maps to []."""
    reached = {("S", "T"): [()]}
    result = _rid_sets_from_reachability(reached, {}, set())
    assert result == {("S", "T"): []}


def test_compute_rid_sets_method_exists():
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    assert hasattr(DatasetBagBuilder, "_compute_rid_sets")


def test_compute_rid_sets_carries_from_model_and_reachability():
    """The shared helper owns the from_model fix and the reachability call."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    src = inspect.getsource(DatasetBagBuilder._compute_rid_sets)
    assert "from_model" in src
    assert "compute_reachability" in src
    assert "_rid_sets_from_reachability" in src


def test_estimate_delegates_to_compute_rid_sets():
    """estimate_bag_size delegates the reachability assembly (no longer inlines
    its own from_model fetch closure)."""
    import inspect

    from deriva_ml.dataset.dataset import Dataset

    src = inspect.getsource(Dataset.estimate_bag_size)
    assert "_compute_rid_sets" in src
    # The from_model closure now lives in the shared helper, not the estimate.
    assert "datapath.from_model" not in src


def test_catalog_bag_builder_accepts_rid_sets():
    """_catalog_bag_builder forwards an opt-in rid_sets to CatalogBagBuilder."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    sig = inspect.signature(DatasetBagBuilder._catalog_bag_builder)
    assert "rid_sets" in sig.parameters
    assert sig.parameters["rid_sets"].default is None


def test_build_bag_uses_rid_sets():
    """build_bag computes rid_sets and passes them to the CatalogBagBuilder."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    src = inspect.getsource(DatasetBagBuilder.build_bag)
    assert "_compute_rid_sets" in src
    assert "rid_sets=" in src


@pytest.mark.skipif(
    os.environ.get("DERIVA_HOST") in (None, ""),
    reason="needs a live catalog",
)
def test_format_b_bag_one_csv_per_table_matches_estimate(catalog_with_datasets, tmp_path):
    """A built bag has one clean CSV per table (Format B); loading it
    reproduces the estimate's per-table row counts, RID-distinct."""
    import csv as csvmod

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    ml, _desc = catalog_with_datasets

    # Pick a nested dataset (multi-path tables -- the interesting case).
    datasets = list(ml.find_datasets())
    nested = next((d for d in datasets if d.list_dataset_children()), None)
    if nested is None:
        pytest.skip("fixture has no nested dataset to exercise multi-path union")
    version = nested.current_version

    # Estimate = the oracle for per-table counts.
    est = nested.estimate_bag_size(version)
    est_counts = {t: d["row_count"] for t, d in est["tables"].items() if d["row_count"] > 0}

    # Build a Format-B bag.
    snap = nested._version_snapshot_catalog(version)
    builder = DatasetBagBuilder(ml_instance=snap)

    # The rid-set keys are exactly the NON-vocab reached tables: vocab tables
    # are deliberately dropped from rid_sets and exported FULL via the full
    # query (per the FKTraversalPolicy vocab_export=FULL rule). So a table's
    # bag CSV equals its reachable-subset estimate ONLY for non-vocab tables;
    # vocab CSVs hold the whole vocabulary (e.g. Asset_Role = Input + Output)
    # and legitimately exceed the reachable-subset count. Use the production
    # code's own vocab determination -- the rid_sets keys -- so this exemption
    # tracks the implementation rather than re-deriving it.
    rid_set_tables = {key[1] for key in builder._compute_rid_sets(nested).rid_sets}

    zip_path = builder.build_bag(nested, output_dir=tmp_path)
    assert zip_path.exists()

    # Extract and inspect data/ CSVs.
    extract = tmp_path / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract)

    csvs = list(extract.rglob("data/**/*.csv"))
    assert csvs, "bag contains no data CSVs"

    # Format B: exactly ONE CSV per table (no per-path fragmentation). The bag
    # tree is data/<schema>/<table>.csv, so key by (schema, table) -- a table is
    # "split across multiple CSVs" only if the SAME (schema, table) recurs. Keying
    # by bare p.stem would false-positive if two schemas held a same-named table.
    by_schema_table = Counter((p.parent.name, p.stem) for p in csvs)
    multi = {key: n for key, n in by_schema_table.items() if n > 1}
    assert not multi, f"Format B violated -- tables split across multiple CSVs: {multi}"

    # Every data CSV in the bag must be RID-distinct (Format B emits sorted,
    # de-duplicated rid-set rows; a duplicate RID signals a regression in the
    # rid-set emission). This applies to vocab CSVs too. Key by (schema, table) =
    # (p.parent.name, p.stem) so two same-named tables in different schemas stay
    # distinct rather than overwriting each other.
    rows_by_csv = {}
    for p in csvs:
        with p.open(encoding="utf-8") as fh:
            rows = list(csvmod.DictReader(fh))
        rids = [r["RID"] for r in rows if "RID" in r]
        assert len(set(rids)) == len(rids), f"{p.parent.name}/{p.stem}.csv has duplicate RIDs"
        rows_by_csv[(p.parent.name, p.stem)] = rows

    # For each NON-vocab data CSV, the bag's rid-set row count must match the
    # estimate exactly (the correctness gate: a dropped rid_sets would revert to
    # per-path multi-CSV emission caught above, or a count drift here). We iterate
    # over the real (schema, table) CSV files but compare to the estimate's count,
    # which is keyed by BARE table name -- that bare-name keying is the estimate's
    # contract. The two align on the demo catalog because table names are unique
    # across schemas; if a same-named table ever appeared in two schemas the
    # ambiguity would live in the estimate's keying, not in this check.
    asserted = 0
    for (_schema, table), rows in rows_by_csv.items():
        if table not in rid_set_tables:
            continue  # vocab table: exported FULL, not by reachable-subset rid-set
        if table not in est_counts:
            continue  # not estimated (e.g. zero-row table omitted from est_counts)
        assert len(rows) == est_counts[table], f"{table}: bag={len(rows)} estimate={est_counts[table]}"
        asserted += 1

    assert asserted, "no non-vocab table count was asserted -- the count gate did nothing"


@pytest.mark.skipif(
    os.environ.get("DERIVA_HOST") in (None, ""),
    reason="needs a live catalog",
)
def test_format_b_bag_fetch_txt_scoped_to_reachable_assets(catalog_with_datasets, tmp_path):
    """The bag's fetch.txt (asset byte manifest) is scoped to the dataset's
    reachable assets -- one entry per reachable asset RID, not the whole asset
    table.

    Consumer-side coverage for the asset-fetch over-fetch bug (deriva-py
    PR #275). Previously the csv processor was rid_set-scoped but the *fetch*
    processor queried the whole asset table, so fetch.txt -- and therefore the
    actual byte download -- pulled every asset in the table regardless of
    dataset membership (a ~6x over-download on eye-ai 2-277G). The estimate
    correctly counts only reachable assets; this asserts the bag's fetch
    manifest agrees with it.

    Validated as a true regression guard: against pre-#275 deriva-py this
    fails on the demo catalog (BoundingBox: fetch.txt 10 vs reachable 4); with
    the fix the fetch manifest is scoped to the 4 reachable rows.

    The existing one-CSV-per-table test only inspects ``data/**/*.csv`` -- the
    correctly-scoped half. This test inspects ``fetch.txt``, the half that was
    broken.
    """
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    ml, _desc = catalog_with_datasets

    datasets = list(ml.find_datasets())
    nested = next((d for d in datasets if d.list_dataset_children()), None)
    if nested is None:
        pytest.skip("fixture has no nested dataset")
    version = nested.current_version

    # Estimate is the oracle: per-asset-table reachable row counts.
    est = nested.estimate_bag_size(version)
    asset_est = {t: d["row_count"] for t, d in est["tables"].items() if d.get("is_asset") and d["row_count"] > 0}
    if not asset_est:
        pytest.skip("fixture's reachable slice has no asset rows to scope")

    snap = nested._version_snapshot_catalog(version)
    builder = DatasetBagBuilder(ml_instance=snap)
    zip_path = builder.build_bag(nested, output_dir=tmp_path)

    extract = tmp_path / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract)

    fetch_files = list(extract.rglob("fetch.txt"))
    assert fetch_files, "bag has no fetch.txt"

    # Bucket fetch entries by asset table. Entry filenames are templated
    # ``data/asset/{asset_rid}/{table_name}/...`` (the fetch processor's
    # output_path), so the path segment after ``asset/<rid>/`` is the table.
    per_table_fetched: dict[str, set[str]] = {}
    for fetch_txt in fetch_files:
        for line in fetch_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # fetch.txt is TSV: <url>\t<length>\t<filename>
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            filename = parts[2]
            segs = filename.split("/")
            if "asset" not in segs:
                continue
            i = segs.index("asset")
            # segs[i+1] = asset_rid, segs[i+2] = table_name
            if i + 2 >= len(segs):
                continue
            asset_rid, table_name = segs[i + 1], segs[i + 2]
            per_table_fetched.setdefault(table_name, set()).add(asset_rid)

    # For every asset table the estimate reached, the bag's fetch.txt must
    # contain exactly the reachable asset count -- not the full-table count.
    # (Pre-#275 the fetch manifest held every row in the table, so this count
    # would exceed the estimate.)
    asserted = 0
    for table, reachable in asset_est.items():
        fetched = len(per_table_fetched.get(table, set()))
        assert fetched == reachable, (
            f"{table}: fetch.txt has {fetched} asset entries but the dataset "
            f"reaches {reachable} -- fetch processor not scoped to the rid set"
        )
        asserted += 1
    assert asserted, "no asset table fetch-count was asserted"

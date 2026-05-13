"""Smoke-test the dataset-bag cutover against a live catalog.

Downloads one dataset two ways (legacy ``CatalogGraph`` and new
``DatasetBagBuilder``) and compares bag contents:

  - Per-table RID sets must match
  - Per-asset-table (RID, MD5) sets must match

Intended for ad-hoc verification against catalogs that the test
fixture doesn't cover — e.g., ``facebase.org``. Not part of the
pytest suite.

Run:
    uv run python scripts/cutover_smoke_check.py \\
        --host facebase.org \\
        --catalog-id 1 \\
        --dataset-rid <DATASET_RID> \\
        --out /tmp/smoke

The script exits non-zero on any equivalence mismatch and prints
the differing rows/assets. Bags are left on disk under ``--out``
for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def _bind_rid_template(spec: dict, rid: str) -> dict:
    text = json.dumps(spec)
    text = re.sub(r"\{RID\}", rid, text)
    return json.loads(text)


def _download_via_spec(
    ml: Any, spec: dict, rid: str, out_dir: Path
) -> Path:
    from deriva.transfer.download.deriva_download import GenericDownloader

    spec = _bind_rid_template(spec, rid)
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


def _bag_table_rid_sets(bag_path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    data_dir = bag_path / "data"
    if not data_dir.exists():
        return out
    for csv_path in data_dir.rglob("*.csv"):
        table = csv_path.stem
        rids = out.setdefault(table, set())
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "RID" in row and row["RID"]:
                    rids.add(row["RID"])
    return out


def _bag_asset_md5_sets(
    bag_path: Path,
) -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = {}
    data_dir = bag_path / "data"
    if not data_dir.exists():
        return out
    for csv_path in data_dir.rglob("*.csv"):
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        if not rows or "MD5" not in rows[0] or "Filename" not in rows[0]:
            continue
        table = csv_path.stem
        bucket = out.setdefault(table, set())
        for row in rows:
            rid = row.get("RID")
            md5 = row.get("MD5")
            if rid and md5:
                bucket.add((rid, md5))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Catalog hostname.")
    parser.add_argument("--catalog-id", required=True, help="Catalog ID.")
    parser.add_argument(
        "--dataset-rid", required=True, help="Dataset RID to download."
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory to write bags into. Will create subdirs legacy/ and new/.",
    )
    args = parser.parse_args()

    from deriva_ml import DerivaML
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder
    from deriva_ml.dataset.catalog_graph import CatalogGraph

    ml = DerivaML(hostname=args.host, catalog_id=args.catalog_id)
    dataset = ml.lookup_dataset(args.dataset_rid)

    legacy_dir = args.out / "legacy"
    new_dir = args.out / "new"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Building bag via CatalogGraph (legacy) in {legacy_dir} ...")
    legacy_spec = CatalogGraph(ml_instance=ml).generate_dataset_download_spec(
        dataset
    )
    legacy_bag = _download_via_spec(
        ml, legacy_spec, args.dataset_rid, legacy_dir
    )
    print(f"   bag: {legacy_bag}")

    print(f"== Building bag via DatasetBagBuilder (new) in {new_dir} ...")
    new_spec = DatasetBagBuilder(
        ml_instance=ml
    ).generate_dataset_download_spec(dataset)
    new_bag = _download_via_spec(ml, new_spec, args.dataset_rid, new_dir)
    print(f"   bag: {new_bag}")

    print()
    print("== Row-set comparison")
    legacy_rids = _bag_table_rid_sets(legacy_bag)
    new_rids = _bag_table_rid_sets(new_bag)
    legacy_tables = set(legacy_rids.keys())
    new_tables = set(new_rids.keys())

    mismatches = 0

    missing = legacy_tables - new_tables
    extra = new_tables - legacy_tables
    if missing:
        print(f"  ! Missing tables in new bag: {sorted(missing)}")
        mismatches += 1
    if extra:
        print(f"  ! Extra tables in new bag:   {sorted(extra)}")
        mismatches += 1

    for table in sorted(legacy_tables & new_tables):
        legacy = legacy_rids[table]
        new = new_rids[table]
        if legacy == new:
            print(f"  ok {table}: {len(legacy)} RIDs")
        else:
            diff = legacy ^ new
            print(
                f"  ! {table}: legacy={len(legacy)} new={len(new)} "
                f"diff={len(diff)}; first 10 differing RIDs: "
                f"{sorted(diff)[:10]}"
            )
            mismatches += 1

    print()
    print("== Asset (RID, MD5) comparison")
    legacy_assets = _bag_asset_md5_sets(legacy_bag)
    new_assets = _bag_asset_md5_sets(new_bag)
    for table in sorted(
        set(legacy_assets.keys()) | set(new_assets.keys())
    ):
        legacy = legacy_assets.get(table, set())
        new = new_assets.get(table, set())
        if legacy == new:
            print(f"  ok {table}: {len(legacy)} assets")
        else:
            diff = legacy ^ new
            print(
                f"  ! {table}: legacy={len(legacy)} new={len(new)} "
                f"diff={len(diff)}; first 5: {sorted(diff)[:5]}"
            )
            mismatches += 1

    print()
    if mismatches == 0:
        print("== EQUIVALENT — bags match on rows + assets.")
        return 0
    print(f"== {mismatches} mismatch(es). See above. Bags left on disk for inspection.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

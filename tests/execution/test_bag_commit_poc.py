"""Proof-of-concept: load an Image row via the bag pipeline.

End-to-end exercise of the deriva-py changes that compose
bag-based commit_execution:

- #227 — ``BagBuilder.add_asset(link=True)`` hardlink mode
- #228 — ``DanglingFKStrategy.PRESERVE``
- #229 — ``FKTraversalPolicy.preserve_provenance``
- #230 — ``_localize_asset_row`` embedded-asset fallback
- #231 — ``Filename`` preserved verbatim; bag-local path
  resolved on demand at upload time
- #232 — ``SchemaBuilder.make_table_name`` hyphen normalisation

The shape mirrors what ``Execution.commit_output_assets``
will do once rewritten: build a bag containing one Image row +
its bytes hardlinked from local flat storage, load it via
``BagCatalogLoader`` with ``PRESERVE`` policy + commit-mode
``preserve_provenance=False``, verify the row + bytes land at
the destination with the right Filename.

If this passes, the deriva-py pieces compose correctly and the
deriva-ml refactor is mechanical translation.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deriva.bag.builder import BagBuilder
from deriva.bag.catalog_loader import BagCatalogLoader
from deriva.bag.schema_io import ermrest_json_to_metadata
from deriva.bag.traversal import (
    AssetMode,
    DanglingFKStrategy,
    FKTraversalPolicy,
    VocabExport,
)

if TYPE_CHECKING:
    from deriva_ml import DerivaML


@pytest.mark.integration
def test_bag_commit_poc_image_round_trip(
    populated_catalog: "DerivaML",
    tmp_path: Path,
) -> None:
    """Build a bag with BagBuilder, load it via BagCatalogLoader, verify.

    Uses the actual destination catalog (populated by the
    fixture); the Subject and Observation FK targets already
    exist there, so ``PRESERVE`` handles them without bag-side
    validation. ``preserve_provenance=False`` lets ERMrest fill
    in ``RCT``/``RCB`` for the newly-minted Image row.
    """
    ml = populated_catalog
    pb = ml.pathBuilder()
    domain = pb.schemas[ml.default_schema]

    # FK targets pre-existing at the destination from the demo
    # population. PRESERVE policy lets the bag's Image row
    # reference them without bag-side parent-row validation.
    subjects = list(domain.Subject.path.entities().fetch())
    observations = list(domain.Observation.path.entities().fetch())
    assert subjects and observations
    subject_rid = subjects[0]["RID"]
    obs_rid = observations[0]["RID"]

    # The asset bytes the bag will carry.
    src = tmp_path / "poc-image.bin"
    src.write_bytes(b"proof-of-concept asset bytes\n")
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    length = src.stat().st_size

    # Lease a RID for the Image row up front (production path
    # uses ``manifest_lease.lease_manifest_pending_assets``).
    from deriva_ml.execution.rid_lease import (
        generate_lease_token,
        post_lease_batch,
    )

    token = generate_lease_token()
    leased_map = post_lease_batch(catalog=ml.catalog, tokens=[token])
    image_rid = leased_map[token]

    # The hatrac URL the loader's ``_upload_assets`` will PUT to.
    # Matches deriva-ml's upload template format
    # ``/hatrac/{table}/{md5}.{filename}``.
    hatrac_url = f"/hatrac/Image/{md5}.{src.name}"

    image_row = {
        "RID": image_rid,
        "URL": hatrac_url,
        "Filename": src.name,
        "Length": length,
        "MD5": md5,
        "Description": "POC asset",
        "Subject": subject_rid,
        "Observation": obs_rid,
        "Acquisition_Date": "2026-05-12",
        "Acquisition_Time": "2026-05-12T00:00:00",
    }

    # Build the bag via the supported API now that #232 makes
    # cross-schema-FK automap work with deriva-ml's hyphenated
    # schema names.
    bag_dir = tmp_path / "commit-bag"
    schema_doc = ml.catalog.getCatalogSchema()
    metadata = ermrest_json_to_metadata(
        schema_doc,
        schemas=["deriva-ml", ml.default_schema],
    )
    with BagBuilder(metadata=metadata, output_dir=bag_dir) as bb:
        bb.add_row("Image", image_row)
        bb.add_asset("Image", image_rid, src, link=True)
        bb.finalize(make_bdbag=True)

    # Bag's asset entry is a hardlink to the source — same inode.
    bag_image = bag_dir / "data" / "asset" / "Image" / image_rid / src.name
    assert bag_image.exists()
    assert bag_image.stat().st_ino == src.stat().st_ino, "POC requires hardlink mode (#227) so we know we're testing it"

    # Load the bag into the live catalog.
    policy = FKTraversalPolicy(
        asset_mode=AssetMode.UPLOAD_IF_MISSING,
        # Out-of-bag Subject/Observation FK targets exist at the
        # destination — trust the catalog's FK constraint to be
        # authoritative rather than the bag's row inventory.
        dangling_fk_strategy=DanglingFKStrategy.PRESERVE,
        vocab_export=VocabExport.REFERENCED_ONLY,
        # The Image row is newly minted; no RCT/RCB to preserve.
        preserve_provenance=False,
    )
    db_dir = tmp_path / "bag-db"
    loader = BagCatalogLoader(
        catalog=ml.catalog,
        bag=bag_dir,
        database_dir=db_dir,
        policy=policy,
    )
    report = asyncio.run(loader.arun())

    # Image row + asset upload both happened.
    image_stats = report.table_stats.get(f"{ml.default_schema}.Image")
    assert image_stats is not None, list(report.table_stats)
    assert image_stats.rows_inserted >= 1
    assert image_stats.assets_attempted >= 1, f"expected ≥1 asset upload attempt, got {image_stats.assets_attempted}"

    # Image row landed at the destination with the right values.
    landed = list(domain.Image.path.filter(domain.Image.RID == image_rid).entities().fetch())
    assert len(landed) == 1
    row = landed[0]
    assert row["RID"] == image_rid
    # Filename column is the bare source filename, not the
    # bag-local path (#231 guarantees this).
    assert row["Filename"] == src.name
    assert row["MD5"] == md5
    assert int(row["Length"]) == length
    assert row["Subject"] == subject_rid
    assert row["Observation"] == obs_rid

    # Asset bytes landed in hatrac. Hatrac echoes ``Content-MD5``
    # base64-encoded (RFC 1864); convert our hex digest to match.
    from deriva.core import HatracStore

    md5_b64 = base64.b64encode(bytes.fromhex(md5)).decode()
    hs = HatracStore("https", ml.host_name, ml.credential)
    assert hs.content_equals(hatrac_url, md5=md5_b64), f"hatrac at {hatrac_url} should have content matching MD5={md5}"

"""Bag-based ``commit_execution`` helpers.

The end-of-execution upload path constructs a bag from the
execution's pending asset+feature manifest, then loads the bag
into the destination catalog via :class:`BagCatalogLoader`.
Replaces the regex-driven ``GenericUploader``-based legacy path
that lived in ``Execution._upload_execution_dirs``.

The pipeline:

1. :func:`build_execution_bag` reads the
   :class:`ManifestStore` for pending assets and staged feature
   records, leases RIDs, validates NOT-NULL metadata, and
   constructs a transient bag containing the catalog rows to
   insert plus the asset bytes (hardlinked from flat storage,
   no copies).

2. :func:`load_execution_bag` hands the bag to
   :class:`BagCatalogLoader` with commit-mode policy
   (``dangling_fk_strategy=PRESERVE``,
   ``preserve_provenance=False``,
   ``asset_mode=UPLOAD_IF_MISSING``). The loader inserts rows
   in FK-safe order, PUTs the asset bytes to the destination
   Hatrac, and returns a report.

3. The caller in ``Execution.upload_execution_outputs``
   marshals the report into the legacy return shape
   (``dict[str, list[AssetFilePath]]``) so existing callers
   don't see the swap.

The bag is discarded after a successful load. The destination
catalog is the durable artifact.

Progress reporting:
    Both :func:`build_execution_bag` and :func:`load_execution_bag`
    accept an optional ``progress_callback``. The callback fires
    at known boundaries — once per asset during bag-build (with
    ``phase="Staging"``), once for the load start (``phase="Uploading"``),
    and once per asset upload completion (``phase="Uploaded"``).
    Byte-level streaming progress would require a new hook on
    :class:`BagCatalogLoader` (tracked as deriva-py follow-up);
    per-file event granularity is enough for the public callers
    that exist today.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deriva.bag.builder import BagBuilder
from deriva.bag.catalog_loader import BagCatalogLoader, LoadReport
from deriva.bag.schema_io import ermrest_json_to_metadata
from deriva.bag.traversal import (
    AssetMode,
    DanglingFKStrategy,
    FKTraversalPolicy,
    VocabExport,
)

from deriva_ml.asset.aux_classes import AssetFilePath
from deriva_ml.core.definitions import MLVocab
from deriva_ml.core.ermrest import UploadProgress
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.upload import asset_type_path, flat_asset_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from deriva_ml.asset.manifest import AssetManifest
    from deriva_ml.execution.execution import Execution
    from deriva_ml.local_db.manifest_store import ManifestStore

    ProgressCallback = Callable[[UploadProgress], None]


logger = logging.getLogger(__name__)


def build_execution_bag(
    execution: "Execution",
    bag_dir: Path,
    *,
    progress_callback: "ProgressCallback | None" = None,
) -> Path:
    """Construct a transient bag containing the execution's pending output.

    Reads pending asset entries and staged feature records from
    the execution's manifest, synthesizes the corresponding
    catalog rows (asset rows + ``{Asset}_Execution`` + ``{Asset}_Asset_Type``
    + feature rows), and hardlinks the asset bytes into the bag.

    Args:
        execution: The :class:`Execution` whose outputs are being
            committed. Used to read the manifest store, look up
            asset-table metadata, and find the working-dir
            location of the flat asset storage.
        bag_dir: Where to write the bag. Must not already exist
            as a populated bag. Caller is responsible for cleanup.
        progress_callback: Optional callback fired once per asset
            as it gets hardlinked into the bag. Phase=``Staging``.
            Per-asset MD5 + length are populated; ``percent_complete``
            tracks bag-staging progress (0-100 across all pending
            assets in this commit).

    Returns:
        Absolute path to the bag directory ready to hand to
        :func:`load_execution_bag`. The bag has the deriva-bag
        profile layout (``data/schema.json``,
        ``data/{schema}/*.csv``, ``data/asset/{table}/{rid}/{filename}``)
        and is bagit-validated.

    Raises:
        DerivaMLException: If any pending asset is missing a
            required (NOT-NULL) metadata column. The error
            aggregates every failure so the caller fixes them
            all at once rather than playing whack-a-mole.
    """
    # Lease RIDs for any pending entries that don't have one. The
    # legacy ``_upload_execution_dirs`` path delegated this to
    # ``manifest_lease.lease_manifest_pending_assets``; reuse the
    # same helper so the lease-store interaction is unchanged.
    from deriva_ml.execution.manifest_lease import (
        lease_manifest_pending_assets,
    )

    manifest = execution._get_manifest()
    lease_manifest_pending_assets(execution._ml_object.catalog, manifest)

    # Validate NOT-NULL metadata across every pending row. Raises
    # a single ``DerivaMLValidationError`` aggregating all
    # failures, matching the legacy path's contract.
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata

    _validate_pending_asset_metadata(execution._model, manifest)

    pending = manifest.pending_assets()

    # Pull the catalog's schema doc and parse to SQLAlchemy
    # MetaData. The reader stashes annotations/ACLs/typenames on
    # ``col.info`` so the bag's written schema.json round-trips
    # losslessly — required for ``Table.is_asset()`` to recognize
    # asset tables at load time. See deriva-py PRs #238, #240.
    schema_doc = execution._ml_object.catalog.get("/schema").json()
    schemas_to_carry = ["deriva-ml", execution._ml_object.default_schema]
    metadata = ermrest_json_to_metadata(
        schema_doc,
        schemas=schemas_to_carry,
    )

    # Make sure the Asset_Role term exists at the destination.
    # The legacy path called ``lookup_term`` to surface a clear
    # error before inserting; preserve that.
    execution._ml_object.lookup_term(MLVocab.asset_role, "Output")

    with BagBuilder(metadata=metadata, output_dir=bag_dir) as bb:
        # Group manifest entries by asset table so we can do one
        # ``add_rows`` per table (cheaper than per-row), and so
        # the association-row construction below works
        # table-by-table.
        by_table: dict[str, list[tuple[str, "AssetEntry"]]] = (
            defaultdict(list)
        )
        for key, entry in pending.items():
            parts = key.split("/", 1)
            if len(parts) != 2:
                logger.warning(
                    "skipping malformed manifest key %r", key
                )
                continue
            asset_table_name, filename = parts
            by_table[asset_table_name].append((filename, entry))

        # ``total_assets`` is the denominator for the staging
        # progress percentage. Computed up front so per-asset
        # callbacks can report a coherent 0-100 progression.
        total_assets = sum(len(es) for es in by_table.values())
        staged_so_far = 0
        for asset_table_name, entries in by_table.items():
            staged_so_far = _add_asset_rows_to_bag(
                bb=bb,
                execution=execution,
                asset_table_name=asset_table_name,
                entries=entries,
                progress_callback=progress_callback,
                staged_so_far=staged_so_far,
                total_assets=total_assets,
            )

        _add_staged_feature_rows_to_bag(
            bb=bb,
            execution=execution,
            manifest=manifest,
            pending=pending,
        )

        return bb.finalize(make_bdbag=True)


def _add_asset_rows_to_bag(
    *,
    bb: BagBuilder,
    execution: "Execution",
    asset_table_name: str,
    entries: list[tuple[str, "AssetEntry"]],
    progress_callback: "ProgressCallback | None" = None,
    staged_so_far: int = 0,
    total_assets: int = 0,
) -> int:
    """Add asset rows + ``*_Execution`` + ``*_Asset_Type`` rows for one table.

    For each pending entry:

    - Synthesize the asset row (RID, URL, Filename, Length, MD5,
      Description, metadata columns) and call ``bb.add_row``.
    - Hardlink the on-disk asset file into the bag via
      ``bb.add_asset(..., link=True)``. Hardlink mode keeps disk
      usage flat (one inode, two directory entries).
    - Add one ``{Asset}_Execution`` association row linking the
      asset to the execution with the ``Output`` role.
    - Add one ``{Asset}_Asset_Type`` association row per type
      the manifest entry carries. Asset-types come from the
      per-execution ``asset_type_path`` JSONL file the legacy
      path writes at ``asset_file_path()`` time.

    Returns:
        Updated ``staged_so_far`` counter (caller-supplied + the
        number of assets staged in this call), so the next
        ``_add_asset_rows_to_bag`` invocation can resume the
        global counter. Always non-negative.
    """
    model = execution._model
    ml = execution._ml_object
    asset_table = model.name_to_table(asset_table_name)
    schema_name = asset_table.schema.name

    metadata_cols = sorted(model.asset_metadata(asset_table_name))

    # ``hatrac_uri`` template the legacy path uses:
    # ``/hatrac/{table}/{md5}.{filename}``. Replicate here so
    # the bag's URL column matches what the loader will PUT to
    # at load time.
    asset_rows: list[dict[str, Any]] = []
    for filename, entry in entries:
        flat_dir = flat_asset_dir(
            execution._working_dir, execution.execution_rid, asset_table_name
        )
        src = flat_dir / filename
        if not src.exists():
            logger.warning(
                "Asset file not found, skipping: %s", src
            )
            continue

        md5 = _file_md5(src)
        length = src.stat().st_size
        hatrac_url = f"/hatrac/{asset_table_name}/{md5}.{filename}"

        row: dict[str, Any] = {
            "RID": entry.rid,
            "URL": hatrac_url,
            "Filename": filename,
            "Length": length,
            "MD5": md5,
            "Description": entry.description,
        }
        # Metadata columns from the manifest. Values are stored
        # in ``entry.metadata``; only emit columns the asset
        # table actually declares.
        for col_name in metadata_cols:
            if col_name in entry.metadata:
                row[col_name] = entry.metadata[col_name]
        asset_rows.append(row)

        # Hardlink the bytes into the bag's
        # ``data/asset/{table}/{rid}/{filename}`` slot. Loader's
        # ``_upload_assets`` PUTs them to hatrac at load time.
        bb.add_asset(asset_table_name, entry.rid, src, link=True)

        # Fire a per-asset progress event. Hardlinks are
        # effectively free so ``bytes_completed == length``
        # immediately; ``percent_complete`` tracks the running
        # total across all pending assets in this commit.
        staged_so_far += 1
        if progress_callback is not None and total_assets > 0:
            progress_callback(
                UploadProgress(
                    file_path=str(src),
                    file_name=filename,
                    bytes_completed=length,
                    bytes_total=length,
                    percent_complete=100.0 * staged_so_far / total_assets,
                    phase="Staging",
                    message=(
                        f"Staged {asset_table_name}/{filename} into bag "
                        f"({staged_so_far}/{total_assets})"
                    ),
                )
            )

    if asset_rows:
        bb.add_rows(asset_table_name, asset_rows)

    # {Asset}_Execution association rows. The legacy path does
    # this via ``_update_asset_execution_table`` after upload;
    # the bag path carries the rows up front and the loader's
    # FK-topo-sort inserts them after the asset rows.
    #
    # Association rows need a leased RID up front because the
    # loader inserts under ``nondefaults=RID`` (commit semantics
    # preserves the caller-supplied RID, blocking the server's
    # default). Sending without an RID lands a NULL and fails
    # the table's RID NOT-NULL unique constraint.
    from deriva_ml.execution.rid_lease import (
        generate_lease_token,
        post_lease_batch,
    )

    exec_assoc, asset_fk, execution_fk = model.find_association(
        asset_table_name, "Execution"
    )
    exec_tokens = [
        generate_lease_token() for _filename, _e in entries
    ]
    exec_lease_map = post_lease_batch(
        catalog=ml.catalog, tokens=exec_tokens
    )
    assoc_rows = [
        {
            "RID": exec_lease_map[exec_tokens[i]],
            asset_fk: entry.rid,
            execution_fk: execution.execution_rid,
            "Asset_Role": "Output",
        }
        for i, (_filename, entry) in enumerate(entries)
    ]
    if assoc_rows:
        bb.add_rows(exec_assoc.name, assoc_rows)

    # {Asset}_Asset_Type association rows. Asset types live in a
    # per-execution JSONL file populated by ``asset_file_path``;
    # read it once and emit one row per (asset, type) pair.
    type_assoc, _, _ = model.find_association(
        asset_table_name, "Asset_Type"
    )
    type_map = _read_asset_type_map(execution, asset_table)

    # Compute every (entry, asset_type) pair first so we can lease
    # the right number of RIDs in one batch.
    type_pairs: list[tuple[Any, str]] = []
    for _filename, entry in entries:
        for asset_type in type_map.get(_filename, []):
            type_pairs.append((entry, asset_type))

    type_rows: list[dict[str, Any]] = []
    if type_pairs:
        type_tokens = [generate_lease_token() for _ in type_pairs]
        type_lease_map = post_lease_batch(
            catalog=ml.catalog, tokens=type_tokens
        )
        for i, (entry, asset_type) in enumerate(type_pairs):
            type_rows.append(
                {
                    "RID": type_lease_map[type_tokens[i]],
                    asset_table_name: entry.rid,
                    "Asset_Type": asset_type,
                }
            )
    if type_rows:
        bb.add_rows(type_assoc.name, type_rows)

    return staged_so_far


def _add_staged_feature_rows_to_bag(
    *,
    bb: BagBuilder,
    execution: "Execution",
    manifest: "AssetManifest",
    pending: dict[str, Any],
) -> None:
    """Add staged feature records to the bag, rewriting asset RIDs.

    Pending feature records reference assets by local filename
    (the bag-build-time placeholder the staging API uses). The
    catalog needs them to reference the **leased** asset RID
    instead. Since RIDs are pre-leased before bag-build, the
    rewrite happens here and the loader inserts the rows
    verbatim.
    """
    ml = execution._ml_object
    pending_features = ml._manifest_store.list_pending_feature_records(
        execution.execution_rid
    ) if hasattr(ml, "_manifest_store") else (
        execution._manifest_store.list_pending_feature_records(
            execution.execution_rid
        )
    )
    if not pending_features:
        return

    # Build a lookup from manifest-key → leased RID. Asset
    # references in feature rows are stored as local filename
    # paths; the legacy ``_flush_staged_features`` does the same
    # rewrite by walking the uploaded_files map (which carries
    # the same key-to-rid mapping).
    by_filename: dict[tuple[str, str], str] = {}
    for key, entry in pending.items():
        parts = key.split("/", 1)
        if len(parts) != 2:
            continue
        asset_table, filename = parts
        by_filename[(asset_table, filename)] = entry.rid

    # Group by feature table for batch ``add_rows``.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pending_features:
        qualified = row.feature_table  # "schema.Table"
        try:
            payload = json.loads(row.record_json)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Skipping feature record %s: malformed JSON (%s)",
                row.stage_id, e,
            )
            continue
        # Strip RCT — server-assigned. Matches the legacy
        # _flush_staged_features behavior.
        payload.pop("RCT", None)

        # Asset-column rewriting. Read the feature's spec to
        # find which columns reference assets, then translate
        # local filenames to the pre-leased RIDs.
        try:
            feat = ml.lookup_feature(row.target_table, row.feature_name)
        except DerivaMLException as e:
            logger.error(
                "lookup_feature failed for %s: %s; skipping group",
                qualified, e,
            )
            continue
        for col in feat.asset_columns:
            val = payload.get(col.name)
            if val is None:
                continue
            # The staged value is a local file path; pull the
            # filename and look up which leased RID corresponds
            # to it. The legacy path matches in two ways
            # (structured-path key vs flat-path); the bag-build
            # path only needs the flat-path form since assets
            # always go through ``asset_file_path``.
            pv = Path(val)
            flat_key = (pv.parent.name, pv.name)
            if flat_key in by_filename:
                payload[col.name] = by_filename[flat_key]
            # else: leave the value alone — it may already be a
            # catalog RID (e.g., for assets uploaded in a prior
            # commit).

        groups[qualified].append(payload)

    # Feature rows need leased RIDs for the same reason
    # association rows do: the loader inserts under
    # ``nondefaults=RID`` (commit semantics), so missing/empty
    # RID lands as NULL and the table's unique constraint fires.
    # Lease one RID per row, write it into the payload, then
    # ``add_rows``.
    from deriva_ml.execution.rid_lease import (
        generate_lease_token,
        post_lease_batch,
    )

    for qualified, payloads in groups.items():
        # Strip any pre-existing RID values (the staged JSON may
        # carry empty strings from CSV → SQLite round-trip). We
        # supply the leased RID afresh.
        for p in payloads:
            p.pop("RID", None)
        tokens = [generate_lease_token() for _ in payloads]
        lease_map = post_lease_batch(catalog=ml.catalog, tokens=tokens)
        for i, p in enumerate(payloads):
            p["RID"] = lease_map[tokens[i]]
        # ``qualified`` is "schema.Table"; BagBuilder accepts the
        # qualified form natively, but the bag's per-schema CSV
        # layout uses the bare table name. ``add_rows`` resolves.
        bb.add_rows(qualified, payloads)


def load_execution_bag(
    execution: "Execution",
    bag_dir: Path,
    *,
    database_dir: Path | None = None,
    progress_callback: "ProgressCallback | None" = None,
) -> LoadReport:
    """Hand the bag to :class:`BagCatalogLoader` with commit-mode policy.

    The loader inserts rows in FK-safe order, PUTs asset bytes
    to the destination Hatrac, and returns a report the caller
    can use to mark manifest entries uploaded.

    Args:
        execution: The execution whose outputs are being
            committed. Provides the destination catalog handle.
        bag_dir: Path returned by :func:`build_execution_bag`.
        database_dir: Where the bag's SQLite mirror should live.
            Defaults to ``bag_dir / ".bag-db"``; callers that want
            inspection-friendly placement can override.
        progress_callback: Optional callback. Fired once with
            phase=``Uploading`` before the loader runs and once
            per asset table afterward summarising uploaded /
            deduped counts (phase=``Uploaded``). Byte-level
            streaming would require a hook on
            :class:`BagCatalogLoader` itself; tracked as a
            deriva-py follow-up.

    Returns:
        The :class:`LoadReport` from the loader. Caller marshals
        this into the manifest-update batches and the legacy
        return shape.
    """
    if database_dir is None:
        database_dir = bag_dir / ".bag-db"
    if progress_callback is not None:
        progress_callback(
            UploadProgress(
                phase="Uploading",
                message="Loading bag into destination catalog",
            )
        )
    policy = FKTraversalPolicy(
        # Output rows reference Subject / Observation / Workflow
        # parents that already exist at the destination but were
        # never carried into the bag. PRESERVE skips the bag-side
        # check and lets ERMrest's FK constraint be authoritative.
        dangling_fk_strategy=DanglingFKStrategy.PRESERVE,
        # Commit semantics: rows are newly minted. RCT/RCB get
        # server-side defaults; only RID is preserved across the
        # bag → catalog transfer.
        preserve_provenance=False,
        # Asset bytes go to Hatrac via ``UPLOAD_IF_MISSING``: HEAD
        # the destination first, skip if the MD5 matches, PUT
        # otherwise. Idempotent on re-run.
        asset_mode=AssetMode.UPLOAD_IF_MISSING,
        # The bag carries only the rows the execution generated;
        # vocab terms it references already exist at the
        # destination. No vocab-export pressure here.
        vocab_export=VocabExport.REFERENCED_ONLY,
    )
    loader = BagCatalogLoader(
        catalog=execution._ml_object.catalog,
        bag=bag_dir,
        database_dir=database_dir,
        policy=policy,
    )
    # ``asyncio.run`` would raise when called from inside a
    # running event loop (e.g. a Jupyter notebook kernel). Detect
    # the case and use the existing loop via ``nest_asyncio``.
    # Same pattern as ``dataset.dataset.Dataset._aggregate_sizes``.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        report = loop.run_until_complete(loader.arun())
    else:
        report = asyncio.run(loader.arun())

    # Per-table completion summaries. The loader's ``LoadReport``
    # carries ``assets_uploaded`` and ``assets_deduped`` per
    # table; surface those as one progress event per asset
    # table, so callers see the load's per-table effect even
    # without byte-streaming.
    if progress_callback is not None:
        for table_name, stats in report.table_stats.items():
            if stats.assets_uploaded == 0 and stats.assets_deduped == 0:
                continue
            progress_callback(
                UploadProgress(
                    file_name=table_name,
                    bytes_completed=0,
                    bytes_total=0,
                    percent_complete=100.0,
                    phase="Uploaded",
                    message=(
                        f"{table_name}: "
                        f"{stats.assets_uploaded} uploaded, "
                        f"{stats.assets_deduped} deduped"
                    ),
                )
            )

    return report


def report_to_asset_map(
    *,
    execution: "Execution",
    report: LoadReport,
    manifest: "AssetManifest",
    keys: list[str] | None = None,
) -> dict[str, list[AssetFilePath]]:
    """Translate a :class:`LoadReport` back to the legacy return shape.

    Existing callers of ``upload_execution_outputs`` expect a
    ``dict[str, list[AssetFilePath]]`` keyed by
    ``"{schema}/{table}"`` containing **only the assets uploaded
    on this call**. Additive uploads (kernel completes, then
    runner registers more) need to distinguish the per-call set
    from the full manifest history.

    Args:
        execution: For schema-prefix construction.
        report: The loader's report. Only inspected for which
            tables actually had rows go through — we don't
            rebuild the asset list from it (the manifest is
            authoritative).
        manifest: Reads the leased RID + asset-type list for each
            entry.
        keys: If supplied, restrict the result to manifest entries
            with these keys. The bag-commit caller passes the
            ``pending`` snapshot's keys (the rows that went into
            *this* bag), preserving the legacy contract that the
            return value covers only the just-uploaded subset.
            When ``None``, every uploaded entry in the manifest is
            included (rarely useful in production but convenient
            for inspection).

    Returns:
        ``{"{schema}/{table}": [AssetFilePath, ...]}`` with one
        entry per asset that went through the bag pipeline.
    """
    model = execution._model
    asset_map: dict[str, list[AssetFilePath]] = defaultdict(list)
    if keys is not None:
        key_set: set[str] | None = set(keys)
    else:
        key_set = None
    for key, entry in manifest.assets.items():
        if key_set is not None and key not in key_set:
            continue
        if entry.status != "uploaded":
            continue
        parts = key.split("/", 1)
        if len(parts) != 2:
            continue
        asset_table_name, filename = parts
        try:
            asset_table = model.name_to_table(asset_table_name)
        except Exception:
            continue
        qualified = f"{asset_table.schema.name}/{asset_table_name}"
        # Filter asset_metadata down to columns the table actually
        # has — matches the legacy AssetFilePath shape.
        meta_cols = model.asset_metadata(asset_table_name)
        asset_map[qualified].append(
            AssetFilePath(
                asset_path=str(
                    flat_asset_dir(
                        execution._working_dir,
                        execution.execution_rid,
                        asset_table_name,
                    )
                    / filename
                ),
                asset_table=qualified,
                file_name=filename,
                asset_metadata={
                    k: v for k, v in entry.metadata.items() if k in meta_cols
                },
                asset_types=list(entry.asset_types or []),
                asset_rid=entry.rid,
            )
        )
    return dict(asset_map)


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream the file at ``path`` and return its lowercase hex MD5.

    Matches deriva-py's :func:`deriva.bag.builder._file_md5`; not
    imported because that function is private. 1 MB chunks keep
    memory bounded for big assets without per-read syscall
    overhead on small ones.
    """
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def _read_asset_type_map(
    execution: "Execution", asset_table: Any
) -> dict[str, list[str]]:
    """Read the per-execution asset-type JSONL into a ``{filename: [types]}`` map.

    The legacy path writes one JSON dict per line at
    ``asset_type_path(working_dir, execution_rid, asset_table)``,
    with each dict mapping ``{filename: [types]}``. Some
    executions write multiple lines (each line covering one
    asset_file_path call); merge them.

    Returns an empty dict if no file exists — assets with no
    declared types just don't get ``Asset_Type`` association
    rows.
    """
    path = Path(
        asset_type_path(
            execution._working_dir,
            execution.execution_rid,
            asset_table,
        )
    )
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.update(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    "skipping malformed asset_type line in %s: %s",
                    path, e,
                )
    return out

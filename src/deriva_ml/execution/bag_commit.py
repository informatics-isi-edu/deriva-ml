"""Bag-based ``commit_execution`` helpers.

The end-of-execution upload path constructs a bag from the
execution's pending asset+feature manifest, then loads the bag
into the destination catalog via :class:`BagCatalogLoader`.
Replaces the regex-driven ``GenericUploader``-based legacy path
that previously lived in ``Execution._upload_execution_dirs``
(deleted alongside this module's adoption).

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

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deriva.bag.builder import BagBuilder, hatrac_url_for
from deriva.bag.catalog_loader import BagCatalogLoader, LoadReport
from deriva.bag.schema_io import ermrest_json_to_metadata
from deriva.bag.traversal import (
    AssetMode,
    DanglingFKStrategy,
    FKTraversalPolicy,
    VocabExport,
)
from deriva.core.utils.hash_utils import compute_file_hashes

from deriva_ml.asset.aux_classes import AssetFilePath
from deriva_ml.core.constants import INTENTIONAL_FK_CYCLES
from deriva_ml.core.definitions import MLVocab
from deriva_ml.core.ermrest import UploadProgress
from deriva_ml.core.exceptions import DerivaMLException, NoAssociationException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.upload_layout import asset_type_path, flat_asset_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from deriva_ml.asset.manifest import AssetEntry, AssetManifest
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.rid_lease import LeaseAggregator

    ProgressCallback = Callable[[UploadProgress], None]


logger = get_logger(__name__)

__all__ = [
    "build_execution_bag",
    "load_execution_bag",
    "report_to_asset_map",
]


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
    # Lease RIDs for any pending entries that don't have one.
    # ``manifest_lease.lease_manifest_pending_assets`` is the
    # canonical entry point — both the (now-removed) legacy upload
    # path and this bag path call it so the lease-store contract
    # is unchanged.
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
    # Use the catalog's cached schema document; deriva-py manages
    # freshness via ETag revalidation and schema-mutation invalidation.
    schema_doc = execution._ml_object.catalog.getCatalogSchema()
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
        by_table: dict[str, list[tuple[str, "AssetEntry"]]] = defaultdict(list)
        for key, entry in pending.items():
            parts = key.split("/", 1)
            if len(parts) != 2:
                logger.warning("skipping malformed manifest key %r", key)
                continue
            asset_table_name, filename = parts
            by_table[asset_table_name].append((filename, entry))

        # URL-based row dedup happens at load time via the
        # ``match_by_columns`` policy that :func:`load_execution_bag`
        # configures (deriva-py PR #245). The bag-build side adds
        # every pending row as if it were new; the loader does the
        # pre-flight catalog query, remaps source RIDs to existing
        # RIDs, and rewrites child FK references through
        # ``_rewrite_fks``. No pre-flight needed here.

        # ``total_assets`` is the denominator for the staging
        # progress percentage. Computed up front so per-asset
        # callbacks can report a coherent 0-100 progression.
        total_assets = sum(len(es) for es in by_table.values())
        staged_so_far = 0

        # All RID leases for the commit go through one aggregator
        # so the whole bag costs **one** ``post_lease_batch`` POST,
        # not one per asset-table + one per feature-group. Audit
        # P1 Ex-batch — a 3-table 5-feature-group commit drops
        # from 11 round trips to 1.
        from deriva_ml.execution.rid_lease import LeaseAggregator

        lease_agg = LeaseAggregator()
        deferred_emits: list[tuple[str, list[dict[str, Any]]]] = []

        for asset_table_name, entries in by_table.items():
            staged_so_far = _add_asset_rows_to_bag(
                bb=bb,
                execution=execution,
                asset_table_name=asset_table_name,
                entries=entries,
                progress_callback=progress_callback,
                staged_so_far=staged_so_far,
                total_assets=total_assets,
                lease_agg=lease_agg,
                deferred_emits=deferred_emits,
            )

        _add_staged_feature_rows_to_bag(
            bb=bb,
            execution=execution,
            manifest=manifest,
            pending=pending,
            lease_agg=lease_agg,
            deferred_emits=deferred_emits,
        )

        # One POST resolves every reserved token to its leased
        # RID. The deferred emits below substitute the leased
        # RID into each row's ``RID`` slot (rows carried a
        # placeholder token there) and call ``bb.add_rows``.
        lease_agg.flush(catalog=execution._ml_object.catalog)
        for table_name, token_rows in deferred_emits:
            for row in token_rows:
                # Each row's "RID" field was set to a token at
                # reserve-time. Replace with the leased RID
                # post-flush.
                row["RID"] = lease_agg.resolve(row["RID"])
            bb.add_rows(table_name, token_rows)

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
    lease_agg: "LeaseAggregator",
    deferred_emits: list[tuple[str, list[dict[str, Any]]]],
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

    No URL-based or asset-type-pair dedup is done here. The
    bag is built as if every row is new; the loader handles
    dedup at load time via :attr:`FKTraversalPolicy.match_by_columns`,
    configured in :func:`load_execution_bag` based on the bag's
    asset tables. When a URL or ``(asset_rid, Asset_Type)``
    composite already exists at the destination, the loader
    remaps the source RID to the existing destination RID and
    rewrites any child FK references through ``_rewrite_fks``.
    See deriva-py PR #245.

    Returns:
        Updated ``staged_so_far`` counter (caller-supplied + the
        number of assets staged in this call), so the next
        ``_add_asset_rows_to_bag`` invocation can resume the
        global counter. Always non-negative.
    """
    model = execution._model
    asset_table = model.name_to_table(asset_table_name)

    # See ``DerivaModel.asset_metadata_sorted`` — both this call
    # site and ``asset_table_upload_spec`` must produce metadata
    # columns in the same alphabetic order, since the upload regex
    # captures them positionally.
    metadata_cols = model.asset_metadata_sorted(asset_table_name)

    asset_rows: list[dict[str, Any]] = []
    for filename, entry in entries:
        flat_dir = flat_asset_dir(execution._working_dir, execution.execution_rid, asset_table_name)
        src = flat_dir / filename
        if not src.exists():
            logger.warning("Asset file not found, skipping: %s", src)
            continue

        md5 = compute_file_hashes(src, hashes=frozenset(["md5"]))["md5"][0]
        length = src.stat().st_size
        # Canonical hatrac URL convention — deriva-py PR #243's
        # ``hatrac_url_for`` keeps the format in one place.
        hatrac_url = hatrac_url_for(asset_table_name, md5, filename)

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
                    message=(f"Staged {asset_table_name}/{filename} into bag ({staged_so_far}/{total_assets})"),
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
    #
    # Pre-Ex-batch this function made two ``post_lease_batch``
    # POSTs (one for exec_assoc rows, one for type rows). Now it
    # reserves tokens against the shared ``lease_agg``; the
    # driver flushes once at end-of-commit and resolves each
    # row's placeholder ``RID`` token to its leased RID.

    exec_assoc, asset_fk, execution_fk = model.find_association(asset_table_name, "Execution")
    exec_tokens = lease_agg.reserve(len(entries))
    assoc_rows = [
        {
            "RID": exec_tokens[i],  # placeholder — driver resolves post-flush
            asset_fk: entry.rid,
            execution_fk: execution.execution_rid,
            "Asset_Role": "Output",
        }
        for i, (_filename, entry) in enumerate(entries)
    ]
    if assoc_rows:
        deferred_emits.append((exec_assoc.name, assoc_rows))

    # {Asset}_Asset_Type association rows. Two sources of tags:
    #
    # 1. User-supplied content types from each ``asset_file_path``
    #    call. Live in a per-execution JSONL file populated by
    #    that method; read once via ``_read_asset_type_map``.
    # 2. The directional ``Output_File`` tag — added by deriva-ml
    #    to every asset uploaded as an output of this execution.
    #    Symmetric with the ``Input_File`` tag added by
    #    ``update_asset_execution_table`` for assets consumed via
    #    ``download_asset``.
    #
    # **The directional-tag contract** (see the "How execution-
    # asset roles work" section of the execution user guide and
    # the docstring on
    # ``asset_upload.update_asset_execution_table``):
    #
    #   Every asset associated with an execution carries either
    #   an ``Input_File`` or ``Output_File`` directional Asset_Type
    #   tag, AND its ``{Asset}_Execution`` row has the matching
    #   ``Asset_Role`` ("Input" or "Output").
    #
    # The role is framework-supplied — callers don't pass
    # ``Output_File`` in their ``asset_types=`` argument to
    # ``asset_file_path``. The dedup below means an explicit
    # pass-through is harmless.
    #
    # Pre-fix the bag-commit path silently omitted
    # ``Output_File`` (catalog ended up with
    # ``["Model_File"]`` instead of
    # ``["Model_File", "Output_File"]``); this is the inline
    # equivalent of ``update_asset_execution_table``'s Output
    # branch.
    #
    # The loader's ``match_by_columns`` policy (configured in
    # :func:`load_execution_bag`) handles ``(asset_rid,
    # Asset_Type)`` dedup at load time — no pre-flight needed.
    type_assoc, _, _ = model.find_association(asset_table_name, "Asset_Type")
    type_map = _read_asset_type_map(execution, asset_table)

    # Lazy import to avoid a top-level circular with
    # ``core.definitions``. ``ExecAssetType.output_file.value`` is
    # the canonical string ("Output_File") that the loader will
    # write into ``Asset_Type``.
    from deriva_ml.core.definitions import ExecAssetType

    output_file_tag = ExecAssetType.output_file.value

    # Compute every (entry, asset_type) pair first so we can lease
    # the right number of RIDs in one batch. Auto-add Output_File
    # to each entry's tag list (deduped so a caller who explicitly
    # passed ``ExecAssetType.output_file`` doesn't produce a
    # duplicate row).
    type_pairs: list[tuple[Any, str]] = []
    for _filename, entry in entries:
        # Build the per-entry tag list — user-supplied + directional.
        # Preserve order so a downstream consumer that cares about
        # user-tag ordering sees Output_File appended last when it
        # wasn't explicit. The deduplication mirrors
        # ``update_asset_execution_table``'s Output branch behaviour
        # (see ``asset_upload.update_asset_execution_table``).
        tags = list(type_map.get(_filename, []))
        if output_file_tag not in tags:
            tags.append(output_file_tag)
        for asset_type in tags:
            type_pairs.append((entry, asset_type))

    if type_pairs:
        type_tokens = lease_agg.reserve(len(type_pairs))
        type_rows: list[dict[str, Any]] = []
        for i, (entry, asset_type) in enumerate(type_pairs):
            type_rows.append(
                {
                    "RID": type_tokens[i],  # placeholder
                    asset_table_name: entry.rid,
                    "Asset_Type": asset_type,
                }
            )
        deferred_emits.append((type_assoc.name, type_rows))

    return staged_so_far


def _add_staged_feature_rows_to_bag(
    *,
    bb: BagBuilder,
    execution: "Execution",
    manifest: "AssetManifest",
    pending: dict[str, Any],
    lease_agg: "LeaseAggregator",
    deferred_emits: list[tuple[str, list[dict[str, Any]]]],
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
    pending_features = (
        ml._manifest_store.list_pending_feature_records(execution.execution_rid)
        if hasattr(ml, "_manifest_store")
        else (execution._manifest_store.list_pending_feature_records(execution.execution_rid))
    )
    if not pending_features:
        return

    # Build a lookup from manifest-key → leased RID. Asset
    # references in feature rows are stored as local filename
    # paths; the rewrite walks the pending-asset map to substitute
    # the pre-leased RID before the bag insert.
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
                row.stage_id,
                e,
            )
            continue
        # Strip RCT — server-assigned; ermrest rejects inserts
        # that pre-set it.
        payload.pop("RCT", None)

        # Asset-column rewriting. Read the feature's spec to
        # find which columns reference assets, then translate
        # local filenames to the pre-leased RIDs.
        try:
            feat = ml.lookup_feature(row.target_table, row.feature_name)
        except DerivaMLException as e:
            logger.error(
                "lookup_feature failed for %s: %s; skipping group",
                qualified,
                e,
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
    # Reserve tokens against the shared aggregator and defer
    # the ``bb.add_rows`` until after the driver's single
    # ``flush()``. Audit P1 Ex-batch — pre-fix this loop made
    # one POST per qualified feature-table group; now zero POSTs
    # (the driver's single flush covers everything).
    for qualified, payloads in groups.items():
        # Strip any pre-existing RID values (the staged JSON may
        # carry empty strings from CSV → SQLite round-trip). We
        # supply the leased RID afresh.
        for p in payloads:
            p.pop("RID", None)
        tokens = lease_agg.reserve(len(payloads))
        for i, p in enumerate(payloads):
            p["RID"] = tokens[i]  # placeholder — driver resolves post-flush
        # ``qualified`` is "schema.Table"; BagBuilder accepts the
        # qualified form natively, but the bag's per-schema CSV
        # layout uses the bare table name. ``add_rows`` resolves.
        deferred_emits.append((qualified, payloads))


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

    # Build the ``match_by_columns`` map (deriva-py PR #245). Two
    # kinds of tables need caller-supplied unique-key reconcile:
    #
    # 1. Each **asset table** dedups by ``URL``. Asset table URLs
    #    are content-hashed (``/hatrac/{table}/{md5}.{filename}``);
    #    re-uploading the same file content (typically across
    #    executions of the same project, e.g. ``uv.lock``) would
    #    otherwise 409 on the table's ``URL`` unique key.
    # 2. Each **``{Asset}_Asset_Type``** association table dedups
    #    by its composite ``(asset_fk_column, Asset_Type)`` key.
    #    When an asset row was itself deduped to an existing
    #    destination row, the previous execution that registered
    #    that asset likely also registered the same asset-type;
    #    inserting the pair again would fail the association
    #    table's unique key. The loader's ``match_by_columns``
    #    redirect handles both: source RIDs in the bag get remapped
    #    to the destination's existing RIDs and child FK references
    #    are rewritten via ``_rewrite_fks``.
    match_by_columns: dict[tuple[str, str], list[str]] = {}
    model = execution._model
    for schema in model.schemas:
        for table in model.schemas[schema].tables.values():
            if not model.is_asset(table):
                continue
            match_by_columns[(schema, table.name)] = ["URL"]
            try:
                type_assoc, _, _ = model.find_association(table.name, "Asset_Type")
            except NoAssociationException:
                # No asset-type association for this asset table — skip.
                continue
            match_by_columns[(type_assoc.schema.name, type_assoc.name)] = [
                table.name,
                "Asset_Type",
            ]

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
        # See the ``match_by_columns`` construction above — handles
        # URL-key dedup on asset tables and ``(asset, type)`` dedup
        # on the ``{Asset}_Asset_Type`` association tables.
        match_by_columns=match_by_columns,
        # Silence the WARNING-level "Breaking cycle in FK dependencies"
        # log spam for the Dataset ↔ Dataset_Version cycle. See
        # core/constants.py:INTENTIONAL_FK_CYCLES — the loader still
        # breaks the cycle to sort tables; it just stops emitting a
        # WARNING for every read pass.
        intentional_cycles=set(INTENTIONAL_FK_CYCLES),
    )
    loader = BagCatalogLoader(
        catalog=execution._ml_object.catalog,
        bag=bag_dir,
        database_dir=database_dir,
        policy=policy,
    )
    # ``BagCatalogLoader.run`` now handles the notebook-loop
    # fallback itself (deriva-py #241): outside a running loop it
    # uses ``asyncio.run``, inside one it falls back to
    # ``nest_asyncio`` + ``loop.run_until_complete``. No
    # deriva-ml-side helper needed.
    report = loader.run()

    # Per-table completion summaries. The loader's ``LoadReport``
    # carries ``assets_attempted`` per table (the number of
    # _hatracUpload invocations — the bag-loader doesn't surface
    # dedup-vs-transfer from the underlying ``put_loc`` call, so
    # we don't either). Surface one progress event per asset
    # table with a nonzero attempt count so callers see the
    # load's per-table effect even without byte-streaming.
    if progress_callback is not None:
        for table_name, stats in report.table_stats.items():
            if stats.assets_attempted == 0:
                continue
            progress_callback(
                UploadProgress(
                    file_name=table_name,
                    bytes_completed=0,
                    bytes_total=0,
                    percent_complete=100.0,
                    phase="Uploaded",
                    message=f"{table_name}: {stats.assets_attempted} upload(s) attempted",
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
                asset_metadata={k: v for k, v in entry.metadata.items() if k in meta_cols},
                asset_types=list(entry.asset_types or []),
                asset_rid=entry.rid,
            )
        )
    return dict(asset_map)


def _read_asset_type_map(execution: "Execution", asset_table: Any) -> dict[str, list[str]]:
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
                    path,
                    e,
                )
    return out

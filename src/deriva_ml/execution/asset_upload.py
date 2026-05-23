"""Asset-staging + upload-orchestration helpers for ``Execution``.

Pre-extraction (audit P1 Ex-god) these helpers lived on the
:class:`~deriva_ml.execution.execution.Execution` god-class
alongside lifecycle (``__enter__``/``__exit__``,
``execution_start``/``execution_stop``), dataset attach
(``download_dataset_bag``, ``create_dataset``), feature staging
(``add_features``), and a dozen other concerns. The audit
asked for the asset-staging + upload pipeline to live in a
sibling module to ``bag_commit.py`` so the file structure
mirrors the conceptual structure.

**First sweep** (PR #216): the helpers with the lightest
coupling to ``Execution`` lifecycle state — the ones that
were nearly free functions already
(``get_metadata_description``, ``set_asset_descriptions``,
``save_runtime_environment``, ``upload_hydra_config_assets``,
``clean_folder_contents``).

**Second sweep** (this PR): the catalog-writing orchestrators —
``bag_commit_upload`` (transient bag build + load) and
``update_asset_execution_table`` (per-execution + per-type
association rows). These have heavier ``self`` coupling
(read ``execution_rid``, ``_model``, ``_ml_object``,
``_working_dir``, ``_manifest_store``) but no state-machine
touching — that stays on :class:`Execution`.

Still on :class:`Execution` (public API + state-machine
entanglement): ``upload_execution_outputs``,
``download_asset``, ``asset_file_path``, ``metrics_file``.
Future sweeps may continue the extraction once the
state-machine and manifest-store types stabilize further.

Pairs with ``bag_commit.py`` (the bag-build / bag-load
pipeline). Together the two modules carry the bulk of the
asset-upload work that ``Execution`` used to hold inline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from deriva.core import format_exception

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.environment import get_execution_environment

if TYPE_CHECKING:
    from deriva_ml.asset.aux_classes import AssetFilePath
    from deriva_ml.core.ermrest import UploadProgress
    from deriva_ml.execution.execution import Execution


# Public symbols re-exported via ``Execution`` thin delegates.
__all__ = [
    "bag_commit_upload",
    "clean_folder_contents",
    "get_metadata_description",
    "save_runtime_environment",
    "set_asset_descriptions",
    "update_asset_execution_table",
    "upload_hydra_config_assets",
]


def get_metadata_description(
    file_name: str, *, metadata_descriptions: dict[str, str], env_snapshot_description: str
) -> str | None:
    """Resolve a description for an execution-metadata file.

    Handles three filename conventions:

    1. Direct filenames in the canonical map (``configuration.json``,
       ``uv.lock``, etc.).
    2. Hydra-renamed files of the form
       ``hydra-{timestamp}-{original_name}`` — extracts the
       original name and looks it up.
    3. Runtime-environment snapshots
       (``environment_snapshot_{timestamp}.txt``).

    Pre-extraction this was a ``@staticmethod`` on
    :class:`Execution`. Moved here so callers can resolve
    descriptions without depending on the class. Keeps the
    description-resolution logic in one place even though only
    one downstream consumer (currently :func:`set_asset_descriptions`)
    needs it.

    Args:
        file_name: The filename as it appears in the catalog.
        metadata_descriptions: Map of canonical filename →
            human-readable description. Passed in (rather than
            imported) so the helper stays decoupled from the
            module-level constant on ``Execution``.
        env_snapshot_description: The description for runtime-
            environment snapshots. Same rationale as
            ``metadata_descriptions``.

    Returns:
        A human-readable description, or ``None`` if the file
        is unrecognized.

    Example:
        >>> from deriva_ml.execution.asset_upload import get_metadata_description
        >>> descs = {"uv.lock": "Dependency lockfile"}
        >>> get_metadata_description(
        ...     "uv.lock", metadata_descriptions=descs,
        ...     env_snapshot_description="Runtime env",
        ... )
        'Dependency lockfile'
        >>> get_metadata_description(
        ...     "hydra-20260522-uv.lock",
        ...     metadata_descriptions=descs,
        ...     env_snapshot_description="Runtime env",
        ... )
        'Dependency lockfile'
        >>> get_metadata_description(
        ...     "environment_snapshot_20260522_120000.txt",
        ...     metadata_descriptions=descs,
        ...     env_snapshot_description="Runtime env",
        ... )
        'Runtime env'
        >>> get_metadata_description(
        ...     "unknown_file.txt",
        ...     metadata_descriptions=descs,
        ...     env_snapshot_description="Runtime env",
        ... ) is None
        True
    """
    if file_name in metadata_descriptions:
        return metadata_descriptions[file_name]

    # Hydra renamed files: "hydra-{timestamp}-{original_name}"
    if file_name.startswith("hydra-"):
        for original, desc in metadata_descriptions.items():
            if file_name.endswith(f"-{original}"):
                return desc

    if file_name.startswith("environment_snapshot_"):
        return env_snapshot_description

    return None


def set_asset_descriptions(
    execution: "Execution",
    uploaded_assets: dict[str, list["AssetFilePath"]],
    *,
    metadata_descriptions: dict[str, str],
    env_snapshot_description: str,
) -> None:
    """Set ``Description`` on uploaded asset rows from manifest + canonical metadata.

    Two-source description resolution:

    1. **Manifest description** — wins. If
       ``asset_file_path(..., description=...)`` was called for
       this asset, the manifest carries the user's description.
    2. **Canonical metadata description** — fall-back for
       ``Execution_Metadata`` files only (``configuration.json``,
       ``uv.lock``, hydra configs, env snapshots). Looked up via
       :func:`get_metadata_description`.

    Pre-extraction this was an ``Execution`` method; the
    extraction makes the manifest read + catalog update
    visible as a pure data transform.

    Args:
        execution: The execution whose manifest + ml_object we
            consult. Reads ``_get_manifest()`` and
            ``_ml_object.pathBuilder()`` only.
        uploaded_assets: Dict mapping ``"{schema}/{table}"`` to
            list of uploaded :class:`AssetFilePath` records.
        metadata_descriptions: Threaded through to
            :func:`get_metadata_description`.
        env_snapshot_description: Same.

    Notes:
        Snapshots ``manifest.assets`` once before the inner
        loop. Pre-extraction the inline implementation re-read
        the manifest on every iteration; for a 10K-asset upload
        on a localhost SSD that meant ~10K SQL round-trips
        (~19 minutes). The single-read pattern is preserved
        here.
    """
    manifest = execution._get_manifest()
    pb = execution._ml_object.pathBuilder()

    # Snapshot the manifest's asset entries ONCE. ``manifest.assets``
    # is a property that calls ``ManifestStore.list_assets`` (a SQL
    # SELECT) on every access.
    manifest_assets = manifest.assets

    # Group updates by schema/table for batch efficiency.
    table_updates: dict[str, list[dict[str, str]]] = {}

    for table_key, assets in uploaded_assets.items():
        for asset in assets:
            # Determine description: check manifest first, then fall back
            # to hardcoded metadata descriptions for Execution_Metadata files.
            table_name = table_key.split("/")[1] if "/" in table_key else table_key
            manifest_key = f"{table_name}/{asset.file_name}"
            entry = manifest_assets.get(manifest_key)

            description = None
            if entry and entry.description:
                description = entry.description
            elif table_name == "Execution_Metadata":
                description = get_metadata_description(
                    asset.file_name,
                    metadata_descriptions=metadata_descriptions,
                    env_snapshot_description=env_snapshot_description,
                )

            if description and asset.asset_rid:
                table_updates.setdefault(table_key, []).append(
                    {"RID": asset.asset_rid, "Description": description}
                )

    for table_key, updates in table_updates.items():
        schema, table_name = (
            table_key.split("/", 1) if "/" in table_key else ("", table_key)
        )
        pb.schemas[schema].tables[table_name].update(updates)


def save_runtime_environment(
    execution: "Execution",
    *,
    runtime_env_asset_type: str,
    env_snapshot_description: str,
) -> None:
    """Capture and stage the runtime environment as an Execution_Metadata asset.

    Calls :func:`get_execution_environment` for the dict
    snapshot (Python version, loaded modules, OS info, etc.)
    and writes it as JSON to the
    ``environment_snapshot_{timestamp}.txt`` file. The
    timestamp keys distinct executions; subsequent calls in
    the same execution stage subsequent snapshots.

    Pre-extraction this was an ``Execution`` method.

    Args:
        execution: The bound :class:`Execution`. Reads
            ``asset_file_path(...)`` only.
        runtime_env_asset_type: The
            ``ExecMetadataType.runtime_env.value`` string —
            passed in to keep the helper decoupled from the
            ``ExecMetadataType`` enum import.
        env_snapshot_description: The description string for
            the staged metadata asset.
    """
    runtime_env_path = execution.asset_file_path(
        asset_name="Execution_Metadata",
        file_name=f"environment_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        asset_types=runtime_env_asset_type,
        description=env_snapshot_description,
    )
    with Path(runtime_env_path).open("w") as fp:
        json.dump(get_execution_environment(), fp)


def upload_hydra_config_assets(
    execution: "Execution",
    *,
    hydra_config_asset_type: str,
    metadata_descriptions: dict[str, str],
    env_snapshot_description: str,
    execution_metadata_asset_name: Any,
) -> None:
    """Register Hydra static config YAMLs as Execution_Metadata assets.

    Walks ``<hydra_run_dir>/hydra-config/`` — the subdirectory
    Hydra designates for the resolved configuration snapshots
    (``config.yaml``, ``hydra.yaml``, ``overrides.yaml``).
    These are immutable per-run artifacts written once at
    hydra-launch and the actual provenance for "what
    configuration did this run use".

    **Deliberately does NOT walk the parent ``<hydra_run_dir>``
    itself.** Hydra installs a job-logging ``FileHandler`` at
    ``<hydra_run_dir>/<job_name>.log`` (e.g., ``notebook.log``)
    that keeps appending log lines for the entire process —
    including every ``logger.info`` from this very upload pass.
    Registering a live, growing file as an immutable asset
    breaks the asset contract ("the bytes you registered are
    the bytes that get uploaded") and produces a known race in
    deriva-py's uploader: the file's MD5 changes between
    hash-time and a subsequent idempotency pre-check, the
    pre-check by ``MD5+Filename`` misses the row that was
    already inserted, and a duplicate-RID INSERT fires (HTTP
    409). The job log is the *runner's* asset, not the
    kernel's — see ``deriva_ml/run_notebook.py`` for the
    runner-side registration that closes the FileHandler
    before staging the log.

    Pre-extraction this was an ``Execution`` method.

    Args:
        execution: The bound :class:`Execution`. Reads
            ``_ml_object.hydra_runtime_output_dir`` and
            ``asset_file_path(...)``.
        hydra_config_asset_type: The
            ``ExecMetadataType.hydra_config.value`` string.
        metadata_descriptions: Threaded into
            :func:`get_metadata_description`.
        env_snapshot_description: Same.
        execution_metadata_asset_name: ``MLAsset.execution_metadata``
            — passed in as a value, not imported, so the helper
            stays decoupled from the ``MLAsset`` enum import.
    """
    hydra_runtime_output_dir = execution._ml_object.hydra_runtime_output_dir
    if not hydra_runtime_output_dir:
        return
    config_dir = hydra_runtime_output_dir / "hydra-config"
    if not config_dir.exists():
        return  # tolerate older Hydra layouts that don't separate config from log
    timestamp = hydra_runtime_output_dir.parts[-1]
    for hydra_asset in config_dir.iterdir():
        if hydra_asset.is_dir():
            continue
        # Register file for upload (side effect); result intentionally unused.
        # Use Hydra_Config type for Hydra YAML configuration files.
        execution.asset_file_path(
            asset_name=execution_metadata_asset_name,
            file_name=hydra_asset,
            rename_file=f"hydra-{timestamp}-{hydra_asset.name}",
            asset_types=hydra_config_asset_type,
            description=get_metadata_description(
                hydra_asset.name,
                metadata_descriptions=metadata_descriptions,
                env_snapshot_description=env_snapshot_description,
            ),
        )


def clean_folder_contents(
    folder_path: Path, remove_folder: bool = True, *, logger: Any = None
) -> None:
    """Clean up a folder's contents and optionally the folder itself.

    Removes all files and subdirectories within ``folder_path``.
    Uses bounded retry with delay for Windows compatibility
    where files may be temporarily locked.

    Pre-extraction this was an ``Execution`` method but didn't
    actually need ``self`` — pure filesystem op. The extraction
    makes that obvious.

    Args:
        folder_path: Path to the folder to clean.
        remove_folder: If ``True`` (default), also remove
            ``folder_path`` itself after cleaning its contents.
        logger: Optional logger for retry-failure warnings.
            Defaults to the module logger.

    Constants:
        Retries each removal up to 3 times with a 1-second
        delay between attempts; the values are inlined because
        they're never tuned at the call site (and over-
        parameterizing would just add an arg that's always
        passed the same way).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    def remove_with_retry(path: Path, is_dir: bool = False) -> bool:
        for attempt in range(MAX_RETRIES):
            try:
                if is_dir:
                    shutil.rmtree(path)
                else:
                    Path(path).unlink()
                return True
            except (OSError, PermissionError) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.warning(f"Failed to remove {path}: {e}")
                    return False
                time.sleep(RETRY_DELAY)
        return False

    try:
        # First remove all contents.
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    remove_with_retry(Path(entry.path), is_dir=True)
                else:
                    remove_with_retry(Path(entry.path))

        # Then remove the folder itself if requested.
        if remove_folder:
            remove_with_retry(folder_path, is_dir=True)

    except OSError as e:
        logger.warning(f"Failed to clean folder {folder_path}: {e}")


# ---------------------------------------------------------------------------
# Catalog-writing orchestrators (audit P1 Ex-god, second sweep)
# ---------------------------------------------------------------------------


def update_asset_execution_table(
    execution: "Execution",
    uploaded_assets: dict[str, list["AssetFilePath"]],
    *,
    asset_role: str,
    asset_role_vocab_term: str,
    input_file_tag: str,
    output_file_tag: str,
    asset_type_path_fn: Callable[..., Path],
) -> None:
    """Link assets to an execution and auto-tag them by role.

    Writes two kinds of association rows for each asset:

    1. ``{Asset}_Execution`` — links the asset RID to the
       execution with the given ``Asset_Role`` (``"Input"`` or
       ``"Output"``). Consumers query it via
       ``execution.list_assets(asset_role="Input")``.

    2. ``{Asset}_Asset_Type`` — auto-tags each asset's content
       classification:

       - For ``"Output"``: every user-supplied type from the
         ``asset_file_path`` calls **plus** ``Output_File``
         (added automatically if not already in the list).
         So a model file uploaded with ``ExecAssetType.model_file``
         ends up tagged ``["Model_File", "Output_File"]``.
       - For ``"Input"``: just ``Input_File`` is added. The
         asset's existing content types (from when it was
         originally created) are preserved.

    Both inserts use ``on_conflict_skip=True`` so re-running is
    idempotent — an asset that already has the
    ``Input_File``/``Output_File`` tag from a prior
    execution-link is unchanged.

    Pre-extraction this was an ``Execution`` method. The
    audit (P1) noted that **only the Input branch is exercised
    in production** — the Output flow now lives in
    :func:`bag_commit._add_asset_rows_to_bag`. The Output
    branch is preserved here for now because it's pinned by
    the asset-role-auto-tag tests; dropping it requires
    rewriting those tests against the bag-commit path. Tracked
    as a follow-up.

    Args:
        execution: The bound :class:`Execution`. Reads
            ``_dry_run``, ``_ml_object``, ``_model``,
            ``execution_rid``, and ``_working_dir`` only.
        uploaded_assets: ``{schema/table_name: [AssetFilePath, ...]}``.
            Each ``AssetFilePath`` carries the asset RID and
            (for outputs) the user-supplied content types.
        asset_role: ``"Input"`` or ``"Output"`` from the
            ``Asset_Role`` vocabulary.
        asset_role_vocab_term: The ``MLVocab.asset_role`` enum
            value (e.g., ``"Asset_Role"``). Threaded through to
            ``lookup_term`` so the helper stays decoupled from
            ``MLVocab`` imports.
        input_file_tag: ``ExecAssetType.input_file.value``.
        output_file_tag: ``ExecAssetType.output_file.value``.
        asset_type_path_fn: The ``asset_type_path`` function
            from ``core.upload_layout`` — passed in to keep
            the helper free of that import.
    """
    if execution._dry_run:
        # Don't do any updates if we are doing a dry run.
        return

    execution._ml_object.lookup_term(asset_role_vocab_term, asset_role)

    # Direction-tag for the {Asset}_Asset_Type write below. The
    # direction is multi-valued alongside content types — a file
    # uploaded as ExecAssetType.model_file ends up tagged with
    # both Model_File AND Output_File. The directional tag makes
    # "give me every asset that's ever served as input" queryable
    # through Asset_Type alone, regardless of which execution
    # it was input to.
    direction_tag = input_file_tag if asset_role == "Input" else output_file_tag

    pb = execution._ml_object.pathBuilder()
    for asset_table, asset_list in uploaded_assets.items():
        # Peel off the schema from the asset table.
        asset_table_name = asset_table.split("/")[1]
        asset_exe, asset_fk, execution_fk = execution._model.find_association(
            asset_table_name, "Execution"
        )
        asset_exe_path = pb.schemas[asset_exe.schema.name].tables[asset_exe.name]

        asset_exe_path.insert(
            [
                {
                    asset_fk: asset_path.asset_rid,
                    execution_fk: execution.execution_rid,
                    "Asset_Role": asset_role,
                }
                for asset_path in asset_list
            ],
            on_conflict_skip=True,
        )

        # Resolve the {Asset}_Asset_Type association once per
        # asset_table loop iteration — we need it for both the
        # Output (user-supplied + Output_File) and Input
        # (Input_File only) branches below.
        asset_asset_type, _, _ = execution._model.find_association(
            asset_table_name, "Asset_Type"
        )
        type_path = pb.schemas[asset_asset_type.schema.name].tables[
            asset_asset_type.name
        ]

        if asset_role == "Input":
            # Input branch: auto-tag each downloaded asset with
            # Input_File. We don't touch the asset's existing
            # content types — the asset was created by someone
            # else (likely a prior execution's output); whatever
            # types it carries should stay. ``on_conflict_skip``
            # makes re-downloads idempotent.
            type_path.insert(
                [
                    {asset_table_name: asset_path.asset_rid, "Asset_Type": direction_tag}
                    for asset_path in asset_list
                ],
                on_conflict_skip=True,
            )
            continue

        # Output branch: read the user-supplied per-file type
        # map produced during asset_file_path() calls, then
        # auto-add Output_File to every asset's type list (if
        # not already present). The user can still explicitly
        # pass ExecAssetType.output_file; we just don't require
        # it anymore.
        asset_type_map: dict[str, list[str]] = {}
        with Path(
            asset_type_path_fn(
                execution._working_dir,
                execution.execution_rid,
                execution._model.name_to_table(asset_table_name),
            )
        ).open("r") as asset_type_file:
            for line in asset_type_file:
                asset_type_map.update(json.loads(line.strip()))

        # Ensure the directional Output_File tag is in every
        # asset's type list. Use a list (preserving order) +
        # membership check rather than a set, so user-specified
        # tag ordering is preserved for any downstream consumer
        # that cares.
        for asset_path in asset_list:
            types = asset_type_map[asset_path.file_name]
            if direction_tag not in types:
                types.append(direction_tag)
            asset_path.asset_types = types

        type_path.insert(
            [
                {asset_table_name: asset.asset_rid, "Asset_Type": t}
                for asset in asset_list
                for t in asset_type_map[asset.file_name]
            ],
            on_conflict_skip=True,
        )


def bag_commit_upload(
    execution: "Execution",
    *,
    progress_callback: "Callable[[UploadProgress], None] | None" = None,
) -> dict[str, list["AssetFilePath"]]:
    """Upload pending execution outputs via the bag pipeline.

    Builds a transient bag from the execution's pending
    manifest entries and staged feature records, then loads it
    into the destination catalog. The bag's asset bytes are
    hardlinked from flat storage (zero disk copies); rows are
    inserted by :class:`BagCatalogLoader` in FK-safe order;
    bytes get PUT to Hatrac.

    After a successful load, marks every pending manifest
    entry as ``uploaded`` (matches the legacy contract that
    the manifest reflects the destination's state after a
    commit). The transient bag dir is left in place at
    ``working_dir/upload/{execution_rid}/`` so post-mortem
    inspection is possible if needed; users may delete the
    ``upload/{rid}/`` directory by hand once they no longer
    need it. The caller's ``clean_folder`` flag controls
    cleanup of the separate ``execution_root`` only (see
    :meth:`Execution.upload_execution_outputs`).

    Pre-extraction this was the ``_bag_commit_upload`` method
    on ``Execution``. The helper now takes ``execution`` as
    an explicit first arg so tests can mock the small subset
    of fields it touches (``_working_dir``, ``execution_rid``,
    ``_logger``, ``_get_manifest()``, ``_manifest_store``).

    Args:
        execution: The bound :class:`Execution`.
        progress_callback: Optional callback for bag-build +
            bag-load progress updates.

    Returns:
        ``{"{schema}/{table}": [AssetFilePath, ...]}`` for the
        asset rows that landed at the destination.

    Raises:
        DerivaMLException: If the bag-load fails. Pending
            feature records are marked failed with the loader's
            error so the user can retry from a known state.
    """
    from deriva_ml.execution.bag_commit import (
        build_execution_bag,
        load_execution_bag,
        report_to_asset_map,
    )

    # Issue #178: per-execution upload staging lives under a
    # dedicated ``upload/`` parent — not at the cache root, where
    # it used to sit beside ``cache/`` and ``schema-cache.json``
    # and was easy to mistake for cache state.
    bag_dir = Path(execution._working_dir) / "upload" / execution.execution_rid
    if bag_dir.exists():
        # An earlier (failed) attempt may have left a partial
        # bag on disk. Wipe it so the build starts clean —
        # bdbag's update path expects scaffolding it controls.
        shutil.rmtree(bag_dir)

    execution._logger.info("Building commit bag at %s", bag_dir)
    # ``build_execution_bag`` leases pending RIDs as its first
    # step. Capture the post-lease ``pending`` snapshot *after*
    # the build so ``entry.rid`` reflects the leased value (the
    # destination catalog now knows about these RIDs).
    bag_dir = build_execution_bag(
        execution,
        bag_dir,
        progress_callback=progress_callback,
    )
    manifest = execution._get_manifest()
    pending = manifest.pending_assets()

    execution._logger.info("Loading commit bag into destination catalog")
    try:
        report = load_execution_bag(
            execution,
            bag_dir,
            progress_callback=progress_callback,
        )
    except Exception as e:
        error = format_exception(e)
        execution._logger.error("BagCatalogLoader failed: %s", error)
        # Mark every pending feature record as failed with the
        # loader's error so the user can retry from a known
        # state.
        try:
            pending_features = execution._manifest_store.list_pending_feature_records(
                execution.execution_rid
            )
            if pending_features:
                execution._manifest_store.mark_feature_records_failed(
                    [(r.stage_id, f"bag-load failed: {error}") for r in pending_features]
                )
        except Exception as mark_err:  # noqa: BLE001 — never mask the original failure
            execution._logger.warning(
                "Could not mark pending features as failed: %s",
                mark_err,
            )
        raise DerivaMLException(
            f"Failed to upload execution outputs via bag pipeline: {error}"
        )

    # Mark every leased manifest entry as uploaded — its RID
    # was set during the bag build step's lease. Batch the
    # SQLite update; one transaction beats N for a typical
    # execution.
    manifest_updates = [(key, entry.rid) for key, entry in pending.items()]
    manifest.mark_uploaded_batch(manifest_updates)

    # Mark staged feature records as uploaded too. The bag
    # carried them; if the load succeeded, they're at the
    # destination.
    pending_features = execution._manifest_store.list_pending_feature_records(
        execution.execution_rid
    )
    if pending_features:
        execution._manifest_store.mark_feature_records_uploaded(
            [r.stage_id for r in pending_features]
        )

    manifest = execution._get_manifest()  # re-read with post-mark statuses
    # Restrict the return to assets that went through *this*
    # commit call — additive uploads (kernel commits, then
    # runner registers more) need the call-scoped subset, not
    # the full manifest history.
    asset_map = report_to_asset_map(
        execution=execution,
        report=report,
        manifest=manifest,
        keys=list(pending.keys()),
    )
    execution._logger.info(
        "Commit bag loaded: %d rows inserted, %d asset upload(s) attempted",
        report.total_rows_inserted,
        sum(s.assets_attempted for s in report.table_stats.values()),
    )
    return asset_map

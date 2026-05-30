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

**Second sweep**: the catalog-writing orchestrators —
``bag_commit_upload`` (transient bag build + load) and
``update_asset_execution_table`` (per-execution + per-type
association rows).

**Third sweep** (this PR): the public-API surface —
``asset_file_path`` (manifest registration + flat staging
+ asset_type JSONL), ``metrics_file`` (thin sugar over
``asset_file_path``), ``download_asset`` (hatrac fetch +
cache + execution-link), and ``commit_output_assets``
(state-machine bracketed bag-commit driver). These are
public-API methods on :class:`Execution`; the body moves
here but the class still exposes thin delegate methods so
the public surface is unchanged.

Pairs with ``bag_commit.py`` (the bag-build / bag-load
pipeline). Together the two modules carry the bulk of the
asset-upload work that ``Execution`` used to hold inline.
After the third sweep ``execution.py`` is the lifecycle
class — state-machine transitions, dataset attach, feature
staging, nested-execution hierarchy — and ``asset_upload.py``
owns the asset surface.
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
    from deriva.core.ermrest_model import Table

    from deriva_ml.asset.aux_classes import AssetFilePath, AssetRecord
    from deriva_ml.core.definitions import RID
    from deriva_ml.core.ermrest import UploadProgress
    from deriva_ml.execution.execution import Execution


# Public symbols re-exported via ``Execution`` thin delegates.
__all__ = [
    "asset_file_path",
    "bag_commit_upload",
    "clean_folder_contents",
    "download_asset",
    "get_metadata_description",
    "metrics_file",
    "save_runtime_environment",
    "set_asset_descriptions",
    "commit_output_assets",
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

    Example:
        >>> set_asset_descriptions(  # doctest: +SKIP
        ...     execution,
        ...     uploaded_assets,
        ...     metadata_descriptions={"uv.lock": "Locked dependencies"},
        ...     env_snapshot_description="Runtime environment snapshot",
        ... )
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
                table_updates.setdefault(table_key, []).append({"RID": asset.asset_rid, "Description": description})

    for table_key, updates in table_updates.items():
        schema, table_name = table_key.split("/", 1) if "/" in table_key else ("", table_key)
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

    Example:
        >>> from deriva_ml.core.enums import ExecMetadataType  # doctest: +SKIP
        >>> save_runtime_environment(  # doctest: +SKIP
        ...     execution,
        ...     runtime_env_asset_type=ExecMetadataType.runtime_env.value,
        ...     env_snapshot_description="Runtime environment snapshot",
        ... )
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

    Example:
        >>> from deriva_ml.core.enums import ExecMetadataType, MLAsset  # doctest: +SKIP
        >>> upload_hydra_config_assets(  # doctest: +SKIP
        ...     execution,
        ...     hydra_config_asset_type=ExecMetadataType.hydra_config.value,
        ...     metadata_descriptions={},
        ...     env_snapshot_description="Runtime environment snapshot",
        ...     execution_metadata_asset_name=MLAsset.execution_metadata,
        ... )
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


def clean_folder_contents(folder_path: Path, remove_folder: bool = True, *, logger: Any = None) -> None:
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

    Example:
        >>> from pathlib import Path  # doctest: +SKIP
        >>> clean_folder_contents(Path("/tmp/staging"), remove_folder=False)  # doctest: +SKIP
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

    Implements the **directional-tag contract** for assets
    associated with an execution: every asset gets one of
    ``Input_File``/``Output_File`` based on which side of the
    execution it sits on. See the "How execution-asset roles
    work" section of the execution user guide for the
    public-API view.

    Writes two kinds of association rows for each asset:

    1. ``{Asset}_Execution`` — links the asset RID to the
       execution with the given ``Asset_Role`` (``"Input"`` or
       ``"Output"``). Consumers query it via
       ``execution.list_assets(asset_role="Input")``.

    2. ``{Asset}_Asset_Type`` — adds the directional tag
       alongside whatever content tags the asset already
       carries:

       - For ``"Output"``: every user-supplied type from the
         ``asset_file_path`` calls **plus** ``Output_File``
         (added automatically if not already in the list).
         So a model file uploaded with ``ExecAssetType.model_file``
         ends up tagged ``["Model_File", "Output_File"]``.
       - For ``"Input"``: ``Input_File`` is added. The
         asset's existing content types (from when it was
         originally created) are preserved — we don't
         overwrite them.

    Both inserts use ``on_conflict_skip=True`` so re-running is
    idempotent — an asset that already has the
    ``Input_File``/``Output_File`` tag from a prior
    execution-link is unchanged.

    **Where each role gets written in production:**

    - **Inputs**: ``download_asset`` calls this helper with
      ``asset_role="Input"``. Both rows (Execution +
      directional Asset_Type) come from here.
    - **Outputs**: the bag-commit upload path
      (``bag_commit._add_asset_rows_to_bag``) writes the
      equivalent rows inline as part of its bag-build sweep —
      ``Asset_Role="Output"`` on the Execution row and
      ``Output_File`` auto-added to each asset's Asset_Type
      list. The Output branch HERE handles non-bag callers
      and is the reference implementation that the bag-commit
      inline writes mirror.

    The 2026-05-22 audit suggested dropping the Output branch
    as "dead in production." That suggestion was rejected:
    ``Asset_Role`` Input/Output is real public-API behaviour,
    and the Output branch here is the documented reference
    for the symmetric contract.

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

    Example:
        >>> from deriva_ml.core.definitions import MLVocab, ExecAssetType  # doctest: +SKIP
        >>> from deriva_ml.core.upload_layout import asset_type_path  # doctest: +SKIP
        >>> update_asset_execution_table(  # doctest: +SKIP
        ...     execution,
        ...     uploaded_assets,
        ...     asset_role="Input",
        ...     asset_role_vocab_term=MLVocab.asset_role,
        ...     input_file_tag=ExecAssetType.input_file.value,
        ...     output_file_tag=ExecAssetType.output_file.value,
        ...     asset_type_path_fn=asset_type_path,
        ... )
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
        asset_exe, asset_fk, execution_fk = execution._model.find_association(asset_table_name, "Execution")
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
        asset_asset_type, _, _ = execution._model.find_association(asset_table_name, "Asset_Type")
        type_path = pb.schemas[asset_asset_type.schema.name].tables[asset_asset_type.name]

        if asset_role == "Input":
            # Input branch: auto-tag each downloaded asset with
            # Input_File. We don't touch the asset's existing
            # content types — the asset was created by someone
            # else (likely a prior execution's output); whatever
            # types it carries should stay. ``on_conflict_skip``
            # makes re-downloads idempotent.
            type_path.insert(
                [{asset_table_name: asset_path.asset_rid, "Asset_Type": direction_tag} for asset_path in asset_list],
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
    :meth:`Execution.commit_output_assets`).

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
            pending_features = execution._manifest_store.list_pending_feature_records(execution.execution_rid)
            if pending_features:
                execution._manifest_store.mark_feature_records_failed(
                    [(r.stage_id, f"bag-load failed: {error}") for r in pending_features]
                )
        except Exception as mark_err:  # noqa: BLE001 — never mask the original failure
            execution._logger.warning(
                "Could not mark pending features as failed: %s",
                mark_err,
            )
        raise DerivaMLException(f"Failed to upload execution outputs via bag pipeline: {error}")

    # Mark every leased manifest entry as uploaded — its RID
    # was set during the bag build step's lease. Batch the
    # SQLite update; one transaction beats N for a typical
    # execution.
    manifest_updates = [(key, entry.rid) for key, entry in pending.items()]
    manifest.mark_uploaded_batch(manifest_updates)

    # Mark staged feature records as uploaded too. The bag
    # carried them; if the load succeeded, they're at the
    # destination.
    pending_features = execution._manifest_store.list_pending_feature_records(execution.execution_rid)
    if pending_features:
        execution._manifest_store.mark_feature_records_uploaded([r.stage_id for r in pending_features])

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


# ---------------------------------------------------------------------------
# Public-API surface (audit P1 Ex-god, third sweep)
# ---------------------------------------------------------------------------


def asset_file_path(
    execution: "Execution",
    asset_name: str,
    file_name: "str | Path",
    *,
    asset_types: "list[str] | str | None" = None,
    copy_file: bool = False,
    rename_file: str | None = None,
    metadata: "AssetRecord | dict[str, Any] | None" = None,
    description: str | None = None,
    asset_type_vocab_term: str,
    flat_asset_dir_fn: Callable[..., Path],
    asset_type_path_fn: Callable[..., Path],
    legacy_kwargs: dict[str, Any] | None = None,
) -> "AssetFilePath":
    """Register a file for upload and return a path to write to.

    Three modes depending on whether ``file_name`` refers to an existing file:

    1. **New file** — ``file_name`` doesn't exist; returns a path to write to.
    2. **Symlink** — ``file_name`` exists, ``copy_file=False``; symlinks into staging.
    3. **Copy** — ``file_name`` exists, ``copy_file=True``; copies into staging.

    Files land in a flat per-table directory
    (``assets/{AssetTable}/``). Metadata is tracked in a
    persistent JSON manifest for crash safety. Metadata can be
    set at registration time via the ``metadata`` parameter (an
    ``AssetRecord`` or dict) or incrementally after via the
    returned ``AssetFilePath``'s ``metadata`` property.

    Pre-extraction this was an :class:`Execution` method. The
    extracted helper reads
    ``execution._model.is_asset/name_to_table``,
    ``execution._ml_object.lookup_term``,
    ``execution._working_dir``, ``execution.execution_rid``,
    and ``execution._get_manifest()`` — no state-machine
    interaction. The enum-shaped args
    (``asset_type_vocab_term``, ``flat_asset_dir_fn``,
    ``asset_type_path_fn``) are threaded in so the helper
    stays decoupled from ``MLVocab`` and
    ``core.upload_layout`` imports.

    Args:
        execution: The bound :class:`Execution`.
        asset_name: Name of the asset table. Must be a valid
            asset table.
        file_name: Name of file to be uploaded, or path to an
            existing file.
        asset_types: Asset type terms from Asset_Type
            vocabulary. Defaults to ``asset_name`` if neither
            this nor ``legacy_kwargs["Asset_Type"]`` is set.
        copy_file: Whether to copy the file rather than
            symlinking.
        rename_file: If provided, rename the file during
            staging.
        metadata: An ``AssetRecord`` (uses ``model_dump``) or
            dict of metadata column values.
        description: Optional description for the asset record.
        asset_type_vocab_term: The ``MLVocab.asset_type`` enum
            value, threaded in to avoid the ``MLVocab`` import.
        flat_asset_dir_fn: The ``flat_asset_dir`` callable from
            ``core.upload_layout``.
        asset_type_path_fn: The ``asset_type_path`` callable
            from ``core.upload_layout``.
        legacy_kwargs: Extra ``**kwargs`` passed through from
            the delegate. Two roles: legacy ``Asset_Type``
            fallback (consumed by the empty-asset_types
            branch) and additional metadata column values
            (merged into ``metadata``).

    Returns:
        :class:`AssetFilePath` bound to the manifest for
        write-through metadata updates.

    Raises:
        DerivaMLException: If the asset table doesn't exist.
        DerivaMLValidationError: If ``asset_types`` contains
            invalid vocabulary terms.
    """
    from deriva_ml.asset.aux_classes import AssetFilePath
    from deriva_ml.asset.manifest import AssetEntry

    legacy_kwargs = legacy_kwargs or {}

    if not execution._model.is_asset(asset_name):
        raise DerivaMLException(f"Table {asset_name} is not an asset")

    asset_table = execution._model.name_to_table(asset_name)
    schema = asset_table.schema.name

    # Validate and normalize asset types. Use ``is None`` rather
    # than ``or`` so an explicit ``asset_types=[]`` (the user
    # saying "no content tags") is honored as-is — the previous
    # ``or`` chain collapsed an empty list to ``asset_name`` and
    # then ``lookup_term`` would fail on the table name. The
    # legacy ``Asset_Type`` kwarg keeps its fallback role.
    if asset_types is None:
        asset_types = legacy_kwargs.pop("Asset_Type", None)
        if asset_types is None:
            asset_types = asset_name
    asset_types = [asset_types] if isinstance(asset_types, str) else asset_types
    for t in asset_types:
        execution._ml_object.lookup_term(asset_type_vocab_term, t)

    # Resolve metadata from AssetRecord, dict, or kwargs.
    metadata_dict: dict[str, Any] = {}
    if metadata is not None:
        if hasattr(metadata, "model_dump"):
            metadata_dict = {k: v for k, v in metadata.model_dump().items() if v is not None}
        else:
            metadata_dict = dict(metadata)
    # Merge any legacy_kwargs that aren't standard parameters.
    metadata_dict.update(legacy_kwargs)

    # Determine file name and path.
    file_name = Path(file_name)
    if file_name.name == "_implementations.log":
        file_name = file_name.with_name("-implementations.log")

    if not file_name.is_absolute():
        file_name = file_name.resolve()

    target_name = Path(rename_file) if file_name.exists() and rename_file else file_name

    # Store file in flat per-table directory.
    flat_dir = flat_asset_dir_fn(execution._working_dir, execution.execution_rid, asset_name)
    flat_path = flat_dir / target_name.name

    if file_name.exists():
        if copy_file:
            flat_path.write_bytes(file_name.read_bytes())
        else:
            try:
                flat_path.symlink_to(file_name)
            except (OSError, PermissionError):
                flat_path.write_bytes(file_name.read_bytes())

    # Register in manifest (write-through + fsync).
    manifest = execution._get_manifest()
    manifest_key = f"{asset_name}/{target_name.name}"
    manifest.add_asset(
        manifest_key,
        AssetEntry(
            asset_table=asset_name,
            schema=schema,
            asset_types=asset_types,
            metadata=metadata_dict,
            description=description,
        ),
    )

    # Also write legacy asset-type JSONL for backward compatibility with upload.
    with Path(asset_type_path_fn(execution._working_dir, execution.execution_rid, asset_table)).open("a") as f:
        f.write(json.dumps({target_name.name: asset_types}) + "\n")

    result = AssetFilePath(
        asset_path=flat_path,
        asset_table=asset_name,
        file_name=target_name.name,
        asset_metadata=metadata_dict,
        asset_types=asset_types,
    )
    result._bind_manifest(manifest, manifest_key)
    return result


def metrics_file(
    execution: "Execution",
    filename: str,
    *,
    execution_metadata_asset_name: Any,
    metrics_file_asset_type: str,
) -> "AssetFilePath":
    """Return a path for writing training-metric records.

    Thin sugar over :func:`asset_file_path` that stamps the
    file with ``asset_types=Metrics_File`` so the catalog's
    ``Execution_Metadata.Type`` honestly describes the file's
    purpose. Repeated calls inside the same execution return
    the same ``AssetFilePath`` (the manifest registers the
    file once), so append-style writes across an epoch loop
    are safe.

    Pre-extraction this was a 5-line method on
    :class:`Execution`. Moved here for symmetry with
    :func:`asset_file_path` so the asset-staging API surface
    lives in one place.

    Args:
        execution: The bound :class:`Execution`.
        filename: Metrics filename inside Execution_Metadata.
        execution_metadata_asset_name: ``MLAsset.execution_metadata``.
        metrics_file_asset_type: ``ExecMetadataType.metrics_file.value``.

    Returns:
        :class:`AssetFilePath` for the metrics file.
    """
    return execution.asset_file_path(
        execution_metadata_asset_name,
        filename,
        asset_types=metrics_file_asset_type,
    )


def download_asset(
    execution: "Execution",
    asset_rid: "RID",
    dest_dir: Path,
    *,
    update_catalog: bool = True,
    use_cache: bool = False,
    _asset_table: "Table | None" = None,
    asset_type_vocab_term: str,
    check_overwrite_safe_fn: Callable[..., None],
) -> "AssetFilePath":
    """Download an asset from Hatrac and place it in a local directory.

    Writes to ``dest_dir / asset_record["Filename"]``. Overwrites
    any existing file at that path; a WARNING is logged when the
    existing content is byte-different from the asset's expected
    MD5 (via ``check_overwrite_safe_fn``). Idempotent
    re-downloads (existing file's md5 matches catalog's recorded
    MD5) log nothing.

    Cache behaviour: with ``use_cache=True`` and a matching MD5,
    the cached file is symlinked into ``dest_dir``. On a cache
    miss the asset is downloaded to ``dest_dir``, then moved to
    the cache directory and symlinked back so future
    ``use_cache=True`` calls see a hit.

    Pre-extraction this was an :class:`Execution` method.

    Args:
        execution: The bound :class:`Execution`.
        asset_rid: RID of the asset.
        dest_dir: Destination directory for the asset.
        update_catalog: Whether to update the catalog execution
            information after downloading (writes Input role
            association rows via
            :func:`update_asset_execution_table`).
        use_cache: When ``True``, check ``cache_dir/assets/{rid}_{md5}/``
            for a previously downloaded copy before fetching.
        _asset_table: Internal — pre-resolved Table object for
            this RID (skip the per-asset ``resolve_rid``
            round-trip).
        asset_type_vocab_term: The ``MLVocab.asset_type`` enum
            value, threaded in to avoid the ``MLVocab`` import.
        check_overwrite_safe_fn: The ``_check_overwrite_safe``
            callable, threaded in to keep the helper free of
            the module-private import.

    Returns:
        :class:`AssetFilePath` with the path to the downloaded
        (or cached) asset file.

    Raises:
        DerivaMLException: If ``asset_rid`` does not refer to an
            asset table.
    """
    # Local import — ``HatracStore`` carries network deps; keep
    # it lazy so unit tests that mock the helper don't pay the
    # cost.
    from deriva.core.hatrac_store import HatracStore

    from deriva_ml.asset.aux_classes import AssetFilePath

    asset_table = _asset_table if _asset_table is not None else execution._ml_object.resolve_rid(asset_rid).table
    if not execution._model.is_asset(asset_table):
        raise DerivaMLException(f"RID {asset_rid}  is not for an asset table.")

    asset_record = execution._ml_object._retrieve_rid(asset_rid)
    asset_metadata = {k: v for k, v in asset_record.items() if k in execution._model.asset_metadata(asset_table)}
    asset_url = asset_record["URL"]
    asset_filename = dest_dir / asset_record["Filename"]
    expected_md5 = asset_record.get("MD5")

    # Check cache before downloading.
    cache_hit = False
    if use_cache:
        md5 = expected_md5
        if md5:
            asset_cache_dir = execution._ml_object.cache_dir / "assets"
            asset_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = f"{asset_rid}_{md5}"
            cached_file = asset_cache_dir / cache_key / asset_record["Filename"]
            if cached_file.exists():
                # Cache hit — symlink from cache to destination. The
                # cached file's bytes ARE the asset's expected bytes
                # by construction (cache_key includes md5), so reuse
                # ``expected_md5`` as the "about to be written" md5
                # in the overwrite check.
                check_overwrite_safe_fn(asset_filename, expected_md5, asset_rid)
                execution._logger.info(f"Using cached asset {asset_rid} (MD5: {md5})")
                if asset_filename.exists() or asset_filename.is_symlink():
                    asset_filename.unlink()
                asset_filename.symlink_to(cached_file)
                cache_hit = True

    if not cache_hit:
        check_overwrite_safe_fn(asset_filename, expected_md5, asset_rid)
        hs = HatracStore("https", execution._ml_object.host_name, execution._ml_object.credential)
        hs.get_obj(path=asset_url, destfilename=asset_filename.as_posix())

        # Store in cache for future use.
        if use_cache:
            md5 = asset_record.get("MD5")
            if md5:
                asset_cache_dir = execution._ml_object.cache_dir / "assets"
                asset_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_key = f"{asset_rid}_{md5}"
                cache_entry_dir = asset_cache_dir / cache_key
                cache_entry_dir.mkdir(parents=True, exist_ok=True)
                cached_file = cache_entry_dir / asset_record["Filename"]
                # Move file to cache, then symlink back.
                shutil.move(str(asset_filename), str(cached_file))
                asset_filename.symlink_to(cached_file)
                execution._logger.info(f"Cached asset {asset_rid} (MD5: {md5})")

    asset_type_table, _col_l, _col_r = execution._model.find_association(asset_table, asset_type_vocab_term)
    type_path = execution._ml_object.pathBuilder().schemas[asset_type_table.schema.name].tables[asset_type_table.name]
    asset_types = [
        asset_type[asset_type_vocab_term]
        for asset_type in type_path.filter(type_path.columns[asset_table.name] == asset_rid)
        .attributes(type_path.Asset_Type)
        .fetch()
    ]

    asset_path = AssetFilePath(
        file_name=asset_filename,
        asset_rid=asset_rid,
        asset_path=asset_filename,
        asset_metadata=asset_metadata,
        asset_table=asset_table.name,
        asset_types=asset_types,
    )

    if update_catalog:
        # Input role assignment — preserved by design (see
        # ``update_asset_execution_table`` docstring on why the
        # Output branch is also kept).
        execution._update_asset_execution_table(
            {f"{asset_table.schema.name}/{asset_table.name}": [asset_path]},
            asset_role="Input",
        )
    return asset_path


def commit_output_assets(
    execution: "Execution",
    *,
    clean_folder: bool | None = None,
    progress_callback: "Callable[[UploadProgress], None] | None" = None,
    pending_upload_status: Any,
    uploaded_status: Any,
    failed_status: Any,
    running_status: Any,
    stopped_status: Any,
    format_duration_fn: Callable[..., str],
) -> dict[str, list["AssetFilePath"]]:
    """Upload all registered output assets via the bag pipeline and bracket the state-machine.

    Drives the upload workflow:

    1. Returns ``{}`` for dry-run executions.
    2. Auto-transitions ``Running → Stopped`` for callers that
       bypassed the context manager.
    3. Handles the ``Uploaded`` short-circuit (no pending work
       → return the full ``uploaded_assets`` view).
    4. Brackets the bag-commit + description-write phase with
       ``Pending_Upload → Uploaded`` (success) or
       ``Pending_Upload → Failed`` (exception) transitions.
       Upload_Duration is written to SQLite **before** each
       terminal transition so the state-machine PUT carries
       the measurement.
    5. Cleans the execution root on success when ``clean_folder``
       is truthy.

    Pre-extraction this was the public ``Execution`` method.
    Extracted helper takes ``execution`` plus the
    ``ExecutionStatus`` enum values + the ``_format_duration``
    callable as explicit args so the helper stays decoupled
    from the enum/utility imports.

    Args:
        execution: The bound :class:`Execution`.
        clean_folder: Whether to delete execution-root folders
            after upload. ``None`` uses
            ``execution._ml_object.clean_execution_dir``.
        progress_callback: Forwarded to ``bag_commit_upload``.
        pending_upload_status, uploaded_status, failed_status,
            running_status, stopped_status: The corresponding
            ``ExecutionStatus`` enum values.
        format_duration_fn: The ``_format_duration`` callable.

    Returns:
        ``{"{schema}/{table}": [AssetFilePath, ...]}`` per-call
        subset of uploaded assets (matching the legacy contract).
        ``{}`` for dry-run.

    Raises:
        Exception: Whatever the bag-commit raises; the state
            machine transitions to ``Failed`` first.
    """
    from datetime import timezone

    if execution._dry_run:
        return {}

    # Use DerivaML instance setting if not explicitly provided.
    if clean_folder is None:
        clean_folder = getattr(execution._ml_object, "clean_execution_dir", True)

    # Auto-stop a still-Running execution. Notebook code paths
    # and other imperative callers that don't use a ``with`` block
    # reach this method with status=Running because no
    # ``__exit__`` ever fired.
    if execution.status is running_status:
        execution.execution_stop()

    # Additive upload path: the execution already finished a prior
    # upload batch (status=Uploaded) and the caller has registered
    # additional assets that need to ship. If there's nothing
    # pending, treat the call as a no-op and return early.
    if execution.status is uploaded_status:
        manifest = execution._get_manifest()
        if not manifest.pending_assets():
            # Short-circuit: no new work. Return the full manifest
            # view of what's been uploaded for this execution.
            return execution.uploaded_assets
        execution.update_status(pending_upload_status)

    # Transition to Pending_Upload BEFORE starting the upload work
    # so that an exception during _bag_commit_upload can legally
    # transition to Failed (Stopped → Failed is not a legal
    # transition, but Pending_Upload → Failed is).
    if execution.status is stopped_status:
        execution.update_status(pending_upload_status)

    # Bracket the upload work with timestamps so Upload_Duration
    # gets populated regardless of whether the call succeeds or
    # raises. The write goes to SQLite *before* the terminal
    # status transition so the state-machine PUT that follows
    # carries the measurement.
    _upload_start = datetime.now(timezone.utc)
    try:
        # ``_bag_commit_upload`` returns the per-call subset of
        # uploaded assets. External callers depend on this
        # per-call shape; the ``uploaded_assets`` property is the
        # full-manifest view.
        uploaded = execution._bag_commit_upload(progress_callback=progress_callback)
        execution._set_asset_descriptions(uploaded)
        # Record Upload_Duration just before the terminal transition
        # so the catalog PUT for Pending_Upload → Uploaded carries
        # the new value.
        execution._ml_object.workspace.execution_state_store().update_execution(
            execution.execution_rid,
            upload_duration=format_duration_fn(_upload_start, datetime.now(timezone.utc)),
        )
        # Successful end of upload: Pending_Upload → Uploaded.
        if execution.status is pending_upload_status:
            execution.update_status(uploaded_status)
        if clean_folder:
            execution._clean_folder_contents(execution._execution_root)
        return uploaded
    except Exception as e:
        # Capture partial upload duration before transitioning to
        # Failed — same write-then-transition order so the failure
        # PUT carries the partial measurement.
        execution._ml_object.workspace.execution_state_store().update_execution(
            execution.execution_rid,
            upload_duration=format_duration_fn(_upload_start, datetime.now(timezone.utc)),
        )
        error = format_exception(e)
        execution.update_status(failed_status, error=error)
        raise e

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

This module hosts the helpers that have the **lightest
coupling** to ``Execution`` lifecycle state — the ones that
were nearly free functions already and only used a handful of
fields off ``self``. The corresponding ``Execution`` methods
remain as thin delegates so the public API is unchanged.

Heavier coupling — ``upload_execution_outputs``,
``_bag_commit_upload``, ``download_asset``, ``asset_file_path``,
``metrics_file``, ``_update_asset_execution_table`` — stays on
:class:`Execution` for now. Those touch the state-machine
transitions (``update_status``, ``execution_stop``) and the
public API surface; a clean extraction of those wants the
state-machine and manifest-store types to themselves stabilize
further first.

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
from typing import TYPE_CHECKING, Any

from deriva_ml.execution.environment import get_execution_environment

if TYPE_CHECKING:
    from deriva_ml.asset.aux_classes import AssetFilePath
    from deriva_ml.execution.execution import Execution


# Public symbols re-exported via ``Execution`` thin delegates.
__all__ = [
    "clean_folder_contents",
    "get_metadata_description",
    "save_runtime_environment",
    "set_asset_descriptions",
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

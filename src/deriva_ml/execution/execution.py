"""Execution management for DerivaML.

This module provides functionality for managing and tracking executions in DerivaML. An execution
represents a computational or manual process that operates on datasets and produces outputs.
The module includes:

- Execution class: Core class for managing execution state and context
- Asset management: Track input and output files
- Status tracking: Monitor and update execution progress
- Dataset handling: Download and materialize required datasets
- Provenance tracking: Record relationships between inputs, processes, and outputs

The Execution class serves as the primary interface for managing the lifecycle of a computational
or manual process within DerivaML.

Typical usage example:
    >>> config = ExecutionConfiguration(workflow="analysis_workflow", description="Data analysis")  # doctest: +SKIP
    >>> with ml.create_execution(config) as execution:  # doctest: +SKIP
    ...     execution.download_dataset_bag(dataset_spec)  # doctest: +SKIP
    ...     # Run analysis
    ...     path = execution.asset_file_path("Model", "model.pt")  # doctest: +SKIP
    ...     # Write model to path...
    ...
    >>> # IMPORTANT: Commit AFTER the context manager exits
    >>> execution.commit_output_assets()  # doctest: +SKIP

The context manager handles start/stop timing automatically. The commit_output_assets()
call must happen AFTER exiting the context manager to ensure proper status tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List

if TYPE_CHECKING:
    from deriva_ml.asset.asset import Asset
    from deriva_ml.execution.pending_summary import PendingSummary
    from deriva_ml.local_db.manifest_store import ManifestStore
from pydantic import validate_call

from deriva_ml.asset.aux_classes import AssetFilePath
from deriva_ml.asset.manifest import AssetManifest
from deriva_ml.core.base import DerivaML
from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.definitions import (
    DRY_RUN_RID,
    RID,
    ExecAssetType,
    ExecMetadataType,
    FileSpec,
    MLAsset,
    MLVocab,
    UploadProgress,
)
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.upload_layout import (
    asset_root,
    asset_type_path,
    execution_root,
    flat_asset_dir,
    table_path,
)
from deriva_ml.core.validation import VALIDATION_CONFIG
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.dataset.dataset import Dataset
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.execution_record import ExecutionRecord
from deriva_ml.execution.state_machine import transition
from deriva_ml.execution.state_store import ExecutionStatus
from deriva_ml.execution.upload_report import UploadReport
from deriva_ml.execution.workflow import Workflow
from deriva_ml.feature import FeatureRecord
from deriva_ml.model.deriva_ml_bag_view import DerivaMLBagView

logger = get_logger(__name__)

# Keep pycharm from complaining about undefined references in docstrings.
execution: Execution
ml: DerivaML
dataset_spec: DatasetSpec


# Descriptions for execution metadata files, keyed by original filename.
_METADATA_DESCRIPTIONS: dict[str, str] = {
    "config.yaml": ("Resolved Hydra configuration: complete merged config values used for this execution"),
    "overrides.yaml": ("Hydra overrides: command-line and sweep parameters that modified the default configuration"),
    "hydra.yaml": ("Hydra runtime config: job name, config group choices, sweep parameters, and runtime metadata"),
    "configuration.json": ("DerivaML execution configuration: datasets, assets, workflow, and config group choices"),
    "uv.lock": ("Python dependency lockfile: pinned versions of all packages in the execution environment"),
}

_ENV_SNAPSHOT_DESCRIPTION = "Runtime environment snapshot: installed packages, OS, platform, and Python configuration"


def _format_duration(start: "datetime | None", end: "datetime | None") -> str:
    """Format a wall-clock interval as ``"<H>H <M>min <S>sec"``.

    Used by every per-phase duration measurement (algorithm, download,
    upload, and the failed-path algorithm computation in ``__exit__``).
    Centralizes the rounding and string format so all four sites stay
    in sync.

    Args:
        start: Phase start timestamp. If None or tz-naive, falls back
            to UTC interpretation. None yields ``"0H 0min 0.0sec"``.
        end: Phase end timestamp. None yields ``"0H 0min 0.0sec"``.

    Returns:
        Pre-formatted string suitable for the catalog ``*_Duration``
        columns and the SQLite ``*_duration`` columns.
    """
    from datetime import timezone

    if start is None or end is None:
        return "0H 0min 0.0sec"
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    delta = end - start
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec"


def _check_overwrite_safe(
    asset_filename: Path,
    expected_md5: str | None,
    asset_rid: RID,
) -> None:
    """Log a WARNING when ``download_asset`` is about to overwrite different bytes.

    Issue #181 protects callers building custom download flows on top of
    ``Execution.download_asset`` from the silent-overwrite footgun: when two
    downloads share ``dest_dir`` and ``Filename`` but resolve to different
    bytes, the second one wins and the first one vanishes without a trace.
    PR #179 RID-keyed the platform-default ``dest_dir`` so the in-platform
    call site is collision-free; this helper guards ad-hoc callers.

    The check is *advisory* — overwrites are still permitted. The helper
    just surfaces the collision at fault time. Three branches:

    1. ``asset_filename`` does not exist: nothing to overwrite, return.
    2. ``asset_filename`` exists and ``expected_md5`` matches its on-disk
       md5: idempotent re-download (cache repopulation, retry, or genuinely
       identical content), return silently.
    3. Otherwise: log WARNING naming the asset RID, the colliding path,
       and both md5s. If ``expected_md5`` is None (catalog doesn't record
       MD5, or the value is missing from ``asset_record``), the conservative
       choice is to warn anyway — without the catalog MD5 we cannot prove
       the overwrite is idempotent.

    Args:
        asset_filename: The on-disk path ``download_asset`` is about to
            write to (``dest_dir / asset_record["Filename"]``).
        expected_md5: The MD5 of the bytes about to be written, typically
            ``asset_record.get("MD5")``. Pass None when unknown.
        asset_rid: RID of the asset being downloaded — included in the
            WARNING for diagnostics.

    Returns:
        None. Side-effects only (a single WARNING log line when the
        overwrite is non-idempotent or unverifiable).

    Example:
        >>> from pathlib import Path
        >>> _check_overwrite_safe(Path("/nonexistent/file"), "abc123", "RID1")
        >>> # No log, file does not exist.
    """
    if not asset_filename.exists() and not asset_filename.is_symlink():
        return

    on_disk_md5: str | None = None
    try:
        # Follow symlinks so the cache-hit branch (symlink to a cache
        # entry) compares against the cache entry's bytes, which is what
        # the next download will overwrite via unlink + symlink.
        target = asset_filename.resolve() if asset_filename.is_symlink() else asset_filename
        if target.is_file():
            hasher = hashlib.md5()
            with target.open("rb") as fp:
                for chunk in iter(lambda: fp.read(8192), b""):
                    hasher.update(chunk)
            on_disk_md5 = hasher.hexdigest()
    except OSError:
        # If we cannot read the file (permissions, broken symlink, etc.)
        # err on the side of warning — the overwrite is unverifiable.
        on_disk_md5 = None

    if expected_md5 is not None and on_disk_md5 is not None and expected_md5 == on_disk_md5:
        return

    logger.warning(
        "download_asset: overwriting existing file at %s with asset %s "
        "(existing md5=%s, expected md5=%s). "
        "Pass a unique dest_dir per asset to avoid collisions; the canonical "
        "pattern is dest_dir = working_dir / 'downloads' / asset_rid.",
        asset_filename,
        asset_rid,
        on_disk_md5 if on_disk_md5 is not None else "<unreadable>",
        expected_md5 if expected_md5 is not None else "<unknown>",
    )


class Execution:
    """Manages the lifecycle and context of a DerivaML execution.

    An Execution represents a computational or manual process within DerivaML. It provides:
    - Dataset materialization and access
    - Asset management (inputs and outputs)
    - Status tracking and updates
    - Provenance recording
    - Result upload and cataloging

    The class handles downloading required datasets and assets, tracking execution state,
    and managing the upload of results. Every dataset and file generated is associated
    with an execution record for provenance tracking.

    Attributes:
        dataset_rids (list[RID]): RIDs of datasets used in the execution.
        datasets (DatasetCollection): Collection wrapping the materialized
            ``DatasetBag`` objects (iterable, indexable; not a plain list).
        configuration (ExecutionConfiguration): Execution settings and parameters.
        workflow_rid (RID): RID of the associated workflow.
        status (ExecutionStatus): Current execution status (read-through from SQLite).
        asset_paths (dict[str, list[AssetFilePath]]): Mapping of asset-table name to
            the list of ``AssetFilePath`` objects for input assets downloaded by
            ``_initialize_execution``. Each downloaded asset lands at
            ``<working_dir>/<exec_rid>/downloaded-assets/<asset_table>/<asset_rid>/<Filename>``
            — keyed by asset RID, so two assets that share the same ``Filename`` do
            not collide on disk. Read files via ``AssetFilePath.file_name``; do not
            hand-construct paths from the asset table or filename.
        start_time (datetime | None): When execution started.
        stop_time (datetime | None): When execution completed.

    Example:
        The context manager handles start/stop timing. Upload must be called AFTER
        the context manager exits::

            >>> config = ExecutionConfiguration(  # doctest: +SKIP
            ...     workflow="analysis",  # doctest: +SKIP
            ...     description="Process samples",  # doctest: +SKIP
            ... )  # doctest: +SKIP
            >>> with ml.create_execution(config) as execution:  # doctest: +SKIP
            ...     bag = execution.download_dataset_bag(dataset_spec)  # doctest: +SKIP
            ...     # Run analysis using bag.path
            ...     output_path = execution.asset_file_path("Model", "model.pt")  # doctest: +SKIP
            ...     # Write results to output_path
            ...
            >>> # IMPORTANT: Call commit AFTER exiting the context manager
            >>> execution.commit_output_assets()  # doctest: +SKIP
    """

    @validate_call(config=VALIDATION_CONFIG)
    def __init__(
        self,
        configuration: ExecutionConfiguration,
        ml_object: DerivaML,
        workflow: Workflow | None = None,
        reload: RID | None = None,
        dry_run: bool = False,
    ):
        """Initializes an Execution instance.

        Creates a new execution or reloads an existing one. Initializes the execution
        environment, downloads required datasets, and sets up asset tracking.

        Args:
            configuration: Settings and parameters for the execution.
            ml_object: DerivaML instance managing the execution.
            workflow: Optional Workflow object. If not specified, the workflow is taken from
                the ExecutionConfiguration object. Must be a Workflow object, not a RID.
            reload: Optional RID of existing execution to reload.
            dry_run: If True, don't create catalog records or upload results.

        Raises:
            DerivaMLException: If initialization fails, configuration is invalid,
                or workflow is not a Workflow object.

        Example:
            Create an execution with a workflow::

                >>> workflow = ml.lookup_workflow("2-ABC1")  # doctest: +SKIP
                >>> config = ExecutionConfiguration(  # doctest: +SKIP
                ...     workflow=workflow,  # doctest: +SKIP
                ...     description="Process data"  # doctest: +SKIP
                ... )  # doctest: +SKIP
                >>> execution = Execution(config, ml)  # doctest: +SKIP

            Or pass workflow separately::

                >>> workflow = ml.lookup_workflow_by_url(  # doctest: +SKIP
                ...     "https://github.com/org/repo/blob/abc123/analysis.py"  # doctest: +SKIP
                ... )  # doctest: +SKIP
                >>> config = ExecutionConfiguration(description="Run analysis")  # doctest: +SKIP
                >>> execution = Execution(config, ml, workflow=workflow)  # doctest: +SKIP
        """

        self.asset_paths: dict[str, list[AssetFilePath]] = {}
        self.configuration = configuration
        self._ml_object = ml_object
        self._model = ml_object.model
        self._logger = ml_object._logger
        # NOTE(E1/E3): self._status / self.start_time / self.stop_time
        # intentionally removed — execution status and lifecycle
        # timestamps now live in SQLite (see `status`, `start_time`,
        # `stop_time` properties below). Every read hits the workspace
        # registry; no in-memory copy is kept. ``uploaded_assets``
        # similarly reads from the asset manifest on every access
        # (see the ``uploaded_assets`` @property below) — no instance
        # field needed.
        self.configuration.argv = sys.argv
        self._execution_record: ExecutionRecord | None = None  # Lazily created after RID is assigned

        self.dataset_rids: List[RID] = []
        self._datasets_list: list[DatasetBag] = []

        self._working_dir = self._ml_object.working_dir
        self._cache_dir = self._ml_object.cache_dir
        if self._working_dir is None:
            raise DerivaMLException(
                "DerivaML working_dir is not set. "
                "Ensure the DerivaML instance was initialized with a valid working_dir."
            )
        self._dry_run = dry_run

        # Make sure we have a valid Workflow object.
        if workflow:
            self.configuration.workflow = workflow

        if self.configuration.workflow is None:
            raise DerivaMLException("Workflow must be specified either in configuration or as a parameter")

        if not isinstance(self.configuration.workflow, Workflow):
            raise DerivaMLException(
                f"Workflow must be a Workflow object, not {type(self.configuration.workflow).__name__}. "
                "Use ml.lookup_workflow(rid) or ml.lookup_workflow_by_url(url) to get a Workflow object."
            )

        # Validate workflow type(s) and register in catalog
        for wt in self.configuration.workflow.workflow_type:
            self._ml_object.lookup_term(MLVocab.workflow_type, wt)
        self.workflow_rid = (
            self._ml_object._add_workflow(self.configuration.workflow) if not self._dry_run else DRY_RUN_RID
        )

        # Validate the datasets and assets to be valid.
        for d in self.configuration.datasets:
            if self._ml_object.resolve_rid(d.rid).table.name != "Dataset":
                raise DerivaMLException("Dataset specified in execution configuration is not a dataset")

        for a in self.configuration.assets:
            if not self._model.is_asset(self._ml_object.resolve_rid(a.rid).table.name):
                raise DerivaMLException("Asset specified in execution configuration is not an asset table")

        schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
        if reload:
            self.execution_rid = reload
            if self.execution_rid == DRY_RUN_RID:
                self._dry_run = True
        elif self._dry_run:
            self.execution_rid = DRY_RUN_RID
        else:
            self.execution_rid = schema_path.Execution.insert(
                [
                    {
                        "Description": self.configuration.description,
                        "Workflow": self.workflow_rid,
                        "Status": str(ExecutionStatus.Created),
                    }
                ]
            )[0]["RID"]

        # Ex-init2 (audit): the catalog row exists at this point.
        # Everything below — environment file writes, dataset
        # materialization, asset downloads, SQLite registry insert —
        # can fail and leave an orphaned catalog Execution row. Wrap
        # the post-insert work in a try/except that rolls back the
        # catalog row before re-raising. ``reload`` and ``dry_run``
        # paths skip the rollback because they never inserted a row
        # in the first place.
        _catalog_row_owned_by_us = not reload and not self._dry_run and self.execution_rid != DRY_RUN_RID
        try:
            self._post_catalog_init_init(reload, schema_path)
        except Exception:
            if _catalog_row_owned_by_us:
                # Best-effort orphan cleanup; never mask the
                # original failure with a delete-side error.
                try:
                    schema_path.Execution.filter(schema_path.Execution.RID == self.execution_rid).delete()
                    logger.warning(
                        "create_execution %s: post-insert work failed; rolled back orphaned catalog Execution row.",
                        self.execution_rid,
                    )
                except Exception as cleanup_exc:
                    logger.error(
                        "create_execution %s: post-insert work failed AND "
                        "orphan-rollback also failed (%s). Manual cleanup "
                        "required: ``deriva-ml`` Execution row at this RID "
                        "has no workspace SQLite sibling.",
                        self.execution_rid,
                        cleanup_exc,
                    )
            raise

    def _post_catalog_init_init(self, reload, schema_path) -> None:
        """The post-catalog-insert section of __init__.

        Extracted (audit Ex-init2) so the wrapping try/except in
        ``__init__`` can roll back the orphaned catalog row on
        any failure here. Includes the
        ``DERIVA_ML_SAVE_EXECUTION_RID`` write, workspace setup,
        ExecutionRecord construction, dataset+asset download
        (via ``_initialize_execution``), and the SQLite registry
        insert.
        """
        if rid_path := os.environ.get("DERIVA_ML_SAVE_EXECUTION_RID", None):
            # Put execution_rid into the provided file path so we can find it later.
            # Also include hydra_runtime_output_dir so an outer runner harness
            # (run_notebook.py) can locate the Hydra job log it owns and register
            # it post-kernel — see "additive upload" in the state-machine docs.
            hydra_dir = getattr(self._ml_object, "hydra_runtime_output_dir", None)
            with Path(rid_path).open("w") as f:
                json.dump(
                    {
                        "hostname": self._ml_object.host_name,
                        "catalog_id": self._ml_object.catalog_id,
                        "workflow_rid": self.workflow_rid,
                        "execution_rid": self.execution_rid,
                        "hydra_runtime_output_dir": str(hydra_dir) if hydra_dir else None,
                    },
                    f,
                )

        # Create a directory for execution rid so we can recover the state in case of a crash.
        execution_root(prefix=self._ml_object.working_dir, exec_rid=self.execution_rid)

        # Create the ExecutionRecord to handle catalog state operations
        if not self._dry_run:
            self._execution_record = ExecutionRecord(
                execution_rid=self.execution_rid,
                workflow=self.configuration.workflow,
                status=ExecutionStatus.Created,
                description=self.configuration.description,
                _ml_instance=self._ml_object,
                _logger=self._logger,
            )

        # Bracket _initialize_execution with timestamps so the
        # download/materialize phase contributes a Download_Duration to
        # the catalog row. The SQLite registry row doesn't exist yet at
        # this point (insert_execution runs below), so we stash the
        # formatted string on self and write it through after the row
        # is created. See docs/bugs/2026-05-19-execution-phase-durations-design.md.
        from datetime import timezone

        _download_start = datetime.now(timezone.utc)
        self._initialize_execution(reload)
        _download_end = datetime.now(timezone.utc)
        self._download_duration_str = _format_duration(_download_start, _download_end)

        # Guard SQLite registry insertion: skip when (a) this is a dry-run
        # (we never want to persist dry-run state) or (b) we are resuming an
        # existing execution (reload is not None), in which case the registry
        # entry was written by the original run and should not be overwritten.
        # Writing twice would corrupt the start-time and initial-status fields.
        #
        # Pre Ex-init2 fix this site wrapped its own try/except that just
        # logged + re-raised, leaving an orphaned catalog row. The outer
        # try/except in ``__init__`` now owns orphan rollback for any
        # post-catalog-insert failure (this one, dataset materialization,
        # asset download, environment writes), so the inner wrapper is gone.
        if not self._dry_run and reload is None:
            store = self._ml_object.workspace.execution_state_store()
            now = datetime.now(timezone.utc)

            # Serialize the ExecutionConfiguration. Pydantic v2 dumps to
            # a plain dict; model_dump_json then serializes. Includes
            # the RID + version info so a reconstructed configuration
            # from a resume_execution call is faithful.
            config_json = self.configuration.model_dump_json()

            store.insert_execution(
                rid=self.execution_rid,
                workflow_rid=self.workflow_rid,
                description=self.configuration.description,
                config_json=config_json,
                status=ExecutionStatus.Created,
                mode=self._ml_object._mode,
                working_dir_rel=f"execution/{self.execution_rid}",
                created_at=now,
                last_activity=now,
            )
            # Persist the just-measured download duration through
            # update_execution. Done as a follow-up write rather
            # than baked into insert_execution so the insert
            # signature stays focused on identity / config fields.
            store.update_execution(
                self.execution_rid,
                download_duration=self._download_duration_str,
            )

    @classmethod
    def from_registry(cls, *, ml_object, execution_rid: str) -> "Execution":
        """Bind an Execution to an existing SQLite registry row.

        Distinct from ``create_execution`` — does NOT contact the
        catalog and does NOT POST a new row. The bound ``Execution``
        instance reads its lifecycle fields (``status``, ``error``,
        ``start_time``, ``stop_time``) from SQLite via the
        read-through property machinery.

        Called by :meth:`DerivaML.resume_execution`.

        Args:
            ml_object: The DerivaML instance this Execution is bound to.
            execution_rid: The pre-existing Execution RID.

        Returns:
            A minimally-initialized Execution with just enough state for
            execution_rid lookup.
        """
        # Minimal construction: skip __init__'s catalog interactions.
        # The read-through properties handle lifecycle field access.
        instance = cls.__new__(cls)
        instance._ml_object = ml_object
        instance._model = ml_object.model
        instance._logger = getattr(ml_object, "_logger", None)
        instance.execution_rid = execution_rid
        instance._dry_run = False
        # Fields the existing class expects to exist:
        instance._datasets_list = []
        instance.dataset_rids = []
        instance.asset_paths = {}
        instance.configuration = None  # Group E loads from config_json
        instance._working_dir = ml_object.working_dir
        instance._cache_dir = ml_object.cache_dir
        # NOTE(E1/E3): self._status / self.start_time / self.stop_time /
        # self.uploaded_assets intentionally not set — they are all
        # read-through properties backed by SQLite / the asset manifest.
        instance._execution_record = None
        # Pull ``workflow_rid`` from the registry row. The SQLite
        # row carries it (see ``state_store.executions.workflow_rid``);
        # leaving it None silently breaks downstream code that
        # relies on ``execution.workflow_rid`` (notably
        # ``add_workflow_executions`` linkage during nested-execution
        # attach). Best-effort: if the registry lookup fails or
        # returns None, leave it None — but we try.
        instance.workflow_rid = None
        try:
            row = ml_object.workspace.execution_state_store().get_execution(execution_rid)
            if row is not None:
                instance.workflow_rid = row.get("workflow_rid")
        except Exception:
            # The registry read can fail in odd offline-mode tests
            # or partially-initialized fixtures; preserve the
            # legacy ``None`` rather than refuse to construct.
            pass
        return instance

    def _save_runtime_environment(self):
        # Delegates to ``asset_upload.save_runtime_environment``
        # — extracted in the Ex-god first sweep so the helper
        # has a single home alongside the other asset-staging
        # helpers in ``execution/asset_upload.py``.
        from deriva_ml.execution.asset_upload import save_runtime_environment

        save_runtime_environment(
            self,
            runtime_env_asset_type=ExecMetadataType.runtime_env.value,
            env_snapshot_description=_ENV_SNAPSHOT_DESCRIPTION,
        )

    def _upload_hydra_config_assets(self):
        """Register Hydra static config YAMLs as Execution_Metadata assets.

        Delegates to ``asset_upload.upload_hydra_config_assets``;
        see that helper for the design notes on why this only
        walks ``<hydra_run_dir>/hydra-config/`` and not the
        parent directory.
        """
        from deriva_ml.execution.asset_upload import upload_hydra_config_assets

        upload_hydra_config_assets(
            self,
            hydra_config_asset_type=ExecMetadataType.hydra_config.value,
            metadata_descriptions=_METADATA_DESCRIPTIONS,
            env_snapshot_description=_ENV_SNAPSHOT_DESCRIPTION,
            execution_metadata_asset_name=MLAsset.execution_metadata,
        )

    @staticmethod
    def _get_metadata_description(file_name: str) -> str | None:
        """Resolve a description for an execution metadata file.

        Thin delegate to
        ``asset_upload.get_metadata_description``; the module-level
        constants are passed through so the helper stays
        decoupled from this class's imports.
        """
        from deriva_ml.execution.asset_upload import get_metadata_description

        return get_metadata_description(
            file_name,
            metadata_descriptions=_METADATA_DESCRIPTIONS,
            env_snapshot_description=_ENV_SNAPSHOT_DESCRIPTION,
        )

    def _set_asset_descriptions(self, uploaded_assets: dict[str, list[AssetFilePath]]) -> None:
        """Set Description on asset records after upload.

        Delegates to ``asset_upload.set_asset_descriptions``;
        see that helper for the manifest-snapshot performance
        note and the two-source description-resolution rule.
        """
        from deriva_ml.execution.asset_upload import set_asset_descriptions

        set_asset_descriptions(
            self,
            uploaded_assets,
            metadata_descriptions=_METADATA_DESCRIPTIONS,
            env_snapshot_description=_ENV_SNAPSHOT_DESCRIPTION,
        )

    def _materialize_input_datasets(self, reload: RID | None) -> None:
        """Materialize each input dataset and link it to the execution.

        For every ``DatasetSpec`` in :attr:`configuration.datasets`:

        1. Download the bag (via :meth:`download_dataset_bag`).
        2. Append it to :attr:`_datasets_list` and track the RID
           in :attr:`dataset_rids`.

        When this is a live run (not ``reload`` or ``dry_run``)
        and at least one dataset is present, insert the
        ``Dataset_Execution`` association rows in one batch.

        Args:
            reload: ``None`` for a fresh execution; an existing
                RID when resuming.
        """
        for dataset in self.configuration.datasets:
            self._logger.info(f"Materialize bag {dataset.rid}... ")
            self._datasets_list.append(self.download_dataset_bag(dataset))
            self.dataset_rids.append(dataset.rid)

        schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
        if self.dataset_rids and not (reload or self._dry_run):
            # Config-declared datasets are *inputs* (consume edges). Record the
            # consumed version on each input edge via the
            # ``Dataset_Execution.Dataset_Version`` FK, resolved from the
            # ``DatasetSpec.version`` (authorship-canonical model, Task 4).
            schema_path.Dataset_Execution.insert(
                [
                    {
                        "Dataset": dataset.rid,
                        "Execution": self.execution_rid,
                        "Dataset_Version": self._ml_object._version_rid(dataset.rid, dataset.version),
                    }
                    for dataset in self.configuration.datasets
                ]
            )

    def _download_input_assets(self, reload: RID | None) -> None:
        """Resolve every input asset RID and download to its per-asset slot.

        Each asset lands at
        ``<working_dir>/<exec_rid>/downloaded-assets/<asset_table>/<asset_rid>/<Filename>``
        so two assets with the same ``Filename`` from the same
        table don't collide on disk (common for prediction CSVs
        emitted by parallel multirun children).

        Args:
            reload: ``None`` for a fresh execution; an existing
                RID when resuming. The ``update_catalog`` flag on
                :meth:`download_asset` is suppressed when
                resuming or dry-running.
        """
        self._logger.info("Downloading assets ...")
        self.asset_paths = {}
        # Batch-resolve every asset RID up front (one query per
        # candidate table) instead of one resolve_rid round-trip per
        # asset. Skip cleanly when there are no asset specs so an
        # empty configuration doesn't fire a no-op resolve_rids call.
        asset_specs = list(self.configuration.assets)
        if asset_specs:
            asset_rids = [spec.rid for spec in asset_specs]
            rid_results = self._ml_object.resolve_rids(asset_rids)
        else:
            rid_results = {}
        for asset_spec in asset_specs:
            asset_rid = asset_spec.rid
            use_cache = asset_spec.cache
            rid_info = rid_results.get(asset_rid)
            if rid_info is None:
                # resolve_rids would have raised on missing RIDs; this
                # branch only triggers if asset_specs was non-empty but
                # rid_info wasn't returned (defensive — should not
                # happen).
                rid_info_table = self._ml_object.resolve_rid(asset_rid).table
            else:
                rid_info_table = rid_info.table
            asset_table = rid_info_table.name
            dest_dir = (
                execution_root(self._ml_object.working_dir, self.execution_rid)
                / "downloaded-assets"
                / asset_table
                / asset_rid
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            self.asset_paths.setdefault(asset_table, []).append(
                self.download_asset(
                    asset_rid=asset_rid,
                    dest_dir=dest_dir,
                    update_catalog=not (reload or self._dry_run),
                    use_cache=use_cache,
                    _asset_table=rid_info_table,
                )
            )

    def _register_init_metadata(self) -> None:
        """Stage configuration / uv.lock / Hydra config / runtime env for upload.

        Writes the in-memory configuration to
        ``configuration.json``, attaches ``uv.lock`` when the
        workflow's git root carries one, registers Hydra config
        assets (when running under hydra-zen), and snapshots the
        runtime environment (Python version, package list, etc.)
        for provenance.

        These artifacts are staged into the manifest only; the
        actual upload is :meth:`_upload_init_assets`.

        Skipped in ``dry_run`` and ``reload`` modes — the caller
        of :meth:`_initialize_execution` guards on this.
        """
        # Save DerivaML configuration with Deriva_Config type.
        cfile = self.asset_file_path(
            asset_name=MLAsset.execution_metadata,
            file_name="configuration.json",
            asset_types=ExecMetadataType.deriva_config.value,
            description=_METADATA_DESCRIPTIONS["configuration.json"],
        )
        with Path(cfile).open("w", encoding="utf-8") as config_file:
            json.dump(self.configuration.model_dump(mode="json"), config_file)

        # Only try to copy uv.lock if git_root is available (local workflow).
        if self.configuration.workflow.git_root:
            lock_file = Path(self.configuration.workflow.git_root) / "uv.lock"
        else:
            lock_file = None
        if lock_file and lock_file.exists():
            _ = self.asset_file_path(
                asset_name=MLAsset.execution_metadata,
                file_name=lock_file,
                asset_types=ExecMetadataType.execution_config.value,
                description=_METADATA_DESCRIPTIONS["uv.lock"],
            )

        self._upload_hydra_config_assets()
        self._save_runtime_environment()

    def _upload_init_assets(self) -> None:
        """Commit the staged init-time assets so they land in the catalog.

        Pairs with :meth:`_register_init_metadata`: that method
        stages files, this one commits them. Splitting keeps the
        catalog-write boundary visible. Same skip-in-dry-run/reload
        guard at the caller.
        """
        uploaded = self._bag_commit_upload()
        self._set_asset_descriptions(uploaded)

    def _initialize_execution(self, reload: RID | None = None) -> None:
        """Initialize the execution environment.

        Sets up the working directory, downloads required datasets and assets,
        and saves initial configuration metadata. Each input asset is placed at
        ``<working_dir>/<exec_rid>/downloaded-assets/<asset_table>/<asset_rid>/<Filename>``
        so two assets that share an asset table and ``Filename`` do not collide
        on disk. After initialization, ``self.asset_paths`` maps asset-table
        name to the list of ``AssetFilePath`` objects produced; use
        ``AssetFilePath.file_name`` as the canonical read path.

        Post Ex-init extraction this method is a thin dispatcher
        over four private helpers:

        1. :meth:`_materialize_input_datasets` — bag downloads +
           ``Dataset_Execution`` insert.
        2. :meth:`_download_input_assets` — per-asset RID resolution
           and download.
        3. :meth:`_register_init_metadata` — config JSON + uv.lock
           + Hydra config + runtime env staging. Skipped in
           ``dry_run`` and ``reload`` modes.
        4. :meth:`_upload_init_assets` — commit the staged
           init-time assets. Same skip-in-dry-run/reload guard.

        Args:
            reload: Optional RID of a previously initialized execution to reload.

        Raises:
            DerivaMLException: If initialization fails.
        """
        self._materialize_input_datasets(reload)
        self._download_input_assets(reload)

        if not reload and not self._dry_run:
            self._register_init_metadata()
            # Now upload the files so we have the info in case the execution fails.
            self._upload_init_assets()

        # NOTE(E3): `start_time` is no longer an instance attribute —
        # the authoritative value is written to SQLite by __enter__ via
        # state_machine.transition. `_initialize_execution` predates the
        # read-through design and is part of the legacy path retained
        # for `execution_start`/`execution_stop` compatibility.
        self._logger.info("Initialize status finished.")

    def _get_registry_row(self) -> dict:
        """Read this execution's row from the workspace SQLite registry.

        Shared helper for the four read-through properties (``status``,
        ``error``, ``start_time``, ``stop_time``). No caching — a
        mutation from another process (e.g., ``deriva-ml upload``
        running in a shell) is visible on the next read.

        Returns:
            The row dict from ``ExecutionStateStore.get_execution``.

        Raises:
            DerivaMLStateInconsistency: If the executions row for this
                rid is missing (gc'd, never created, or dry-run).
        """
        from deriva_ml.core.exceptions import DerivaMLStateInconsistency

        store = self._ml_object.workspace.execution_state_store()
        row = store.get_execution(self.execution_rid)
        if row is None:
            raise DerivaMLStateInconsistency(
                f"Execution {self.execution_rid} no longer in workspace registry. "
                f"It may have been garbage-collected or the workspace was "
                f"recreated. Use ml.list_executions() to see current state."
            )
        return row

    @staticmethod
    def _coerce_utc(value: "datetime | None") -> "datetime | None":
        """Coerce a SQLite-returned datetime to UTC-aware.

        SQLite may return naive datetimes even though we store them
        timezone-aware. Re-attach the UTC tzinfo when missing.
        """
        from datetime import timezone

        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    @property
    def status(self) -> "ExecutionStatus":
        """Current execution status, read from SQLite on every access.

        No caching — a mutation from another process (e.g., ``deriva-ml
        upload`` running in a shell) is visible on the next read.

        Returns:
            The ExecutionStatus value from the workspace registry.

        Raises:
            DerivaMLStateInconsistency: If the executions row for this
                rid is missing (gc'd or never created).

        Example:
            >>> exe = ml.resume_execution("5-ABC")  # doctest: +SKIP
            >>> exe.status  # doctest: +SKIP
            <ExecutionStatus.Stopped>
        """
        return ExecutionStatus(self._get_registry_row()["status"])

    @property
    def error(self) -> str | None:
        """Last error message for this execution, read from SQLite.

        Parallels ``status`` — no caching, every read hits the workspace
        registry. Populated by ``__exit__`` when an exception occurs
        (see spec §2.8 / §2.12).

        Returns:
            The error message string, or None if no error has been
            recorded for this execution.

        Raises:
            DerivaMLStateInconsistency: If the executions row for this
                rid is missing (gc'd or never created).

        Example:
            >>> with exe.execute():  # doctest: +SKIP
            ...     raise RuntimeError("boom")  # doctest: +SKIP
            >>> exe.error  # doctest: +SKIP
            'RuntimeError: boom'
        """
        return self._get_registry_row()["error"]

    @property
    def start_time(self) -> "datetime | None":
        """Start time from SQLite, or None if not yet started.

        Parallels ``status`` / ``error`` — no caching, every read hits
        the workspace registry. Populated by ``__enter__`` when the
        execution transitions to ``running``. Returned values are
        coerced to UTC-aware datetimes (SQLite may return naive values
        even though we store tz-aware).

        Returns:
            Timezone-aware (UTC) datetime when the execution's
            ``__enter__`` ran, or None before.

        Raises:
            DerivaMLStateInconsistency: If the executions row for this
                rid is missing (gc'd or never created, e.g. dry-run).

        Example:
            >>> exe = ml.resume_execution("EXE-A")  # doctest: +SKIP
            >>> if exe.start_time is not None:  # doctest: +SKIP
            ...     print(f"started at {exe.start_time}")  # doctest: +SKIP
        """
        return self._coerce_utc(self._get_registry_row()["start_time"])

    @property
    def stop_time(self) -> "datetime | None":
        """Stop time from SQLite, or None if not yet stopped/failed.

        Parallels ``status`` / ``error`` — no caching, every read hits
        the workspace registry. Populated by ``__exit__`` on either
        clean stop or exception (see spec §2.8 / §2.12). Returned
        values are coerced to UTC-aware datetimes.

        Returns:
            Timezone-aware (UTC) datetime when the execution's
            ``__exit__`` ran, or None if still running.

        Raises:
            DerivaMLStateInconsistency: If the executions row for this
                rid is missing.

        Example:
            >>> exe = ml.resume_execution("EXE-A")  # doctest: +SKIP
            >>> if exe.stop_time is not None:  # doctest: +SKIP
            ...     print(f"stopped at {exe.stop_time}")  # doctest: +SKIP
        """
        return self._coerce_utc(self._get_registry_row()["stop_time"])

    @property
    def uploaded_assets(self) -> dict[str, list[AssetFilePath]]:
        """Assets this execution has uploaded, read from the asset manifest.

        Reads the manifest on every access — no in-memory cache. The
        returned dict carries every entry whose status is
        ``uploaded`` across the manifest's lifetime, regardless of
        which ``commit_output_assets()`` call produced it. Each
        value is a list of :class:`AssetFilePath` objects giving the
        leased ``asset_rid`` and ``file_name`` for the entry.

        Returns:
            Map of ``"{schema}/{table}"`` → list of
            :class:`AssetFilePath`. Empty dict for dry-run executions
            and for executions that haven't uploaded anything yet.
            Never ``None``.

        Note:
            Until the Phase 3 cleanup landed (audit §A.8) this was an
            instance attribute holding the **most-recent call's**
            return value. The manifest is now the source of truth;
            the property returns the full manifest's uploaded
            entries. Callers that need the per-call subset should
            use the return value of ``commit_output_assets()``
            directly.

        Example:
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     pass  # doctest: +SKIP
            >>> report = exe.commit_output_assets()  # doctest: +SKIP
            >>> # After commit, the property reflects the manifest:
            >>> sum(len(v) for v in exe.uploaded_assets.values()) >= report.total_uploaded  # doctest: +SKIP
            True
        """
        if self._dry_run:
            return {}
        from deriva_ml.execution.bag_commit import report_to_asset_map

        manifest = self._get_manifest()
        # ``report`` is unused when ``keys=None`` — pass a sentinel
        # that ``report_to_asset_map`` won't dereference.
        return report_to_asset_map(
            execution=self,
            report=None,  # type: ignore[arg-type]
            manifest=manifest,
            keys=None,
        )

    @property
    def datasets(self) -> "DatasetCollection":
        """Input datasets as a RID-keyed mapping + iterable.

        Replaces the previous ``list[DatasetBag]`` exposure (hard
        cutover per spec R5.1). The returned collection behaves like a
        ``Mapping[str, DatasetBag]`` (keyed by ``dataset_rid``) and is
        also iterable, yielding ``DatasetBag`` values in insertion
        order.

        Returns:
            A ``DatasetCollection`` wrapping the materialized
            ``DatasetBag`` objects that were downloaded during
            execution initialization.

        Example:
            >>> # RID lookup (primary access pattern)
            >>> bag = exe.datasets["1-XYZ"]  # doctest: +SKIP
            >>> # Iterate bags in insertion order
            >>> for bag in exe.datasets:  # doctest: +SKIP
            ...     print(bag.dataset_rid)  # doctest: +SKIP
            >>> # Introspect which RIDs are present
            >>> rids = list(exe.datasets.keys())  # doctest: +SKIP
            >>> # Count
            >>> n = len(exe.datasets)  # doctest: +SKIP

        Migration note:
            Callers that previously indexed by position
            (``exe.datasets[0]``) must switch to either
            ``list(exe.datasets)[0]`` or the RID-keyed lookup
            ``exe.datasets[rid]``.
        """
        from deriva_ml.execution.dataset_collection import DatasetCollection

        return DatasetCollection(self._datasets_list)

    @property
    def execution_record(self) -> ExecutionRecord | None:
        """Get the ExecutionRecord for catalog operations.

        Returns:
            ExecutionRecord if not in dry_run mode, None otherwise.
        """
        return self._execution_record

    @property
    def working_dir(self) -> Path:
        """Return the working directory for the execution."""
        return self._execution_root

    @property
    def _execution_root(self) -> Path:
        """Get the root directory for this execution's files.

        Returns:
            Path to the execution-specific directory.
        """
        return execution_root(self._working_dir, self.execution_rid)

    @property
    def _asset_root(self) -> Path:
        """Get the root directory for asset files.

        Returns:
            Path to the asset directory within the execution.
        """
        return asset_root(self._working_dir, self.execution_rid)

    @property
    def database_catalog(self) -> DerivaMLBagView | None:
        """Get a catalog-like interface for downloaded datasets.

        Returns a DerivaMLBagView that implements the DerivaMLCatalog
        protocol, allowing the same code to work with both live catalogs
        and downloaded bags.

        This is useful for writing code that can operate on either a live
        catalog (via DerivaML) or on downloaded bags (via DerivaMLBagView).

        Returns:
            DerivaMLBagView wrapping the primary downloaded dataset's model,
            or None if no datasets have been downloaded.

        Example:
            >>> with ml.create_execution(config) as exe:  # doctest: +SKIP
            ...     if exe.database_catalog:  # doctest: +SKIP
            ...         db = exe.database_catalog  # doctest: +SKIP
            ...         # Use same interface as DerivaML
            ...         dataset = db.lookup_dataset("4HM")  # doctest: +SKIP
            ...         term = db.lookup_term("Diagnosis", "cancer")  # doctest: +SKIP
            ...     else:
            ...         # No datasets downloaded, use live catalog
            ...         pass
        """
        if not self._datasets_list:
            return None
        # Use the first dataset's model as the primary
        return DerivaMLBagView(self._datasets_list[0].model)

    @property
    def catalog(self) -> "DerivaML":
        """Get the live catalog (DerivaML) instance for this execution.

        This provides access to the live catalog for operations that require
        catalog connectivity, such as looking up datasets or other read operations.

        Returns:
            DerivaML: The live catalog instance.

        Example:
            >>> with ml.create_execution(config) as exe:  # doctest: +SKIP
            ...     # Use live catalog for lookups
            ...     existing_dataset = exe.catalog.lookup_dataset("1-ABC")  # doctest: +SKIP
        """
        return self._ml_object

    def add_features(self, features: list[FeatureRecord]) -> int:
        """Stage feature records for batch insertion on execution completion.

        Writes the records to the execution's SQLite ``execution_state__feature_records`` table
        with status ``Pending``. The records are not sent to ermrest immediately
        — they are flushed in a single batch, **after asset upload**, when the
        execution completes successfully. This integrates with the SQLite
        execution-state design so crash-resume works for feature writes without
        extra plumbing.

        Records with ``Execution`` unset are auto-filled with this execution's
        RID. All records in a single call must share one feature definition;
        mixing features raises ``DerivaMLValidationError`` and nothing is staged.

        **Provenance requirement.** This is the only way to write feature values
        — ``DerivaML.add_features`` is retired (see the retired-API error shims).
        For "admin fixup" cases, create a short-lived execution with an
        appropriate ``Workflow_Type`` (e.g. ``Manual_Correction``) and call
        ``exe.add_features`` inside it. The three-extra-lines give you a real
        audit trail, which is the point.

        Args:
            features: List of FeatureRecord instances to stage. All must share
                the same feature definition. Create instances via
                ``Feature.feature_record_class()``.

        Returns:
            Number of records staged.

        Raises:
            ValueError: features list is empty.
            DerivaMLValidationError: Records do not share a single feature
                definition.
            DerivaMLDataError: SQLite staging write failed.

        Example:
            >>> feature = ml.lookup_feature("Image", "Glaucoma")  # doctest: +SKIP
            >>> RecordClass = feature.feature_record_class()  # doctest: +SKIP
            >>> records = [  # doctest: +SKIP
            ...     RecordClass(Image="IMG-1", Glaucoma="Normal"),  # doctest: +SKIP
            ...     RecordClass(Image="IMG-2", Glaucoma="Severe"),  # doctest: +SKIP
            ... ]  # doctest: +SKIP
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     exe.add_features(records)     # staged, not yet in ermrest
            ...     # ... more work ...
            >>> # on __exit__: staged records flushed to ermrest after assets
        """
        from deriva_ml.core.exceptions import DerivaMLValidationError

        if not features:
            raise ValueError("features list must not be empty")

        # All records must share one feature definition
        feature_defs = {type(f).feature for f in features if type(f).feature is not None}
        if len(feature_defs) > 1:
            raise DerivaMLValidationError(
                f"add_features called with records from {len(feature_defs)} different "
                f"feature definitions; all records must share one feature."
            )

        # Auto-fill Execution RID on records that don't have it
        for f in features:
            if f.Execution is None:
                f.Execution = self.execution_rid

        # Stage to SQLite — durability boundary is the write-through here.
        feat_class = type(features[0])
        feat = feat_class.feature
        schema_name = feat.feature_table.schema.name
        table_name = feat.feature_table.name
        qualified = f"{schema_name}.{table_name}"
        feature_name = feat.feature_name
        target_table_name = feat.target_table.name

        # Stage every record in a single SQLite transaction. The
        # legacy per-record loop wrapped each ``stage_feature_record``
        # call in its own ``engine.begin()`` block (one WAL fsync per
        # record); for a multi-thousand-record ``add_features`` call
        # that's N serialized fsyncs. The bulk path collapses to one.
        # See ``ManifestStore.stage_feature_records``.
        records_json = [f.model_dump_json() for f in features]
        self._manifest_store.stage_feature_records(
            execution_rid=self.execution_rid,
            feature_table=qualified,
            feature_name=feature_name,
            target_table=target_table_name,
            records_json=records_json,
        )
        return len(features)

    @validate_call(config=VALIDATION_CONFIG)
    def download_dataset_bag(self, dataset: DatasetSpec) -> DatasetBag:
        """Downloads and materializes a dataset for use in the execution.

        Downloads the specified dataset as a BDBag and materializes it in the execution's
        working directory. The dataset version is determined by the DatasetSpec.

        Args:
            dataset: Specification of the dataset to download, including version and
                materialization options.

        Returns:
            DatasetBag: Object exposing, among others:
                - ``path``: local filesystem path to the materialized bag
                - ``dataset_rid``: the dataset's Resource Identifier
                - ``current_version``: the resolved dataset version

        Raises:
            DerivaMLException: If download or materialization fails.

        Example:
            >>> spec = DatasetSpec(rid="1-abc123", version="1.2.0")  # doctest: +SKIP
            >>> bag = execution.download_dataset_bag(spec)  # doctest: +SKIP
            >>> print(f"Downloaded {bag.dataset_rid} to {bag.path}")  # doctest: +SKIP
        """
        return self._ml_object.download_dataset_bag(dataset)

    def update_status(
        self,
        target: "ExecutionStatus",
        *,
        error: str | None = None,
    ) -> None:
        """Transition this execution to a new status.

        Thin wrapper around state_machine.transition() that validates
        against ALLOWED_TRANSITIONS, writes the SQLite registry, and syncs
        to the catalog when online.

        Args:
            target: Target ExecutionStatus enum member.
            error: For Failed/Aborted transitions, a human-readable error
                message written to the `error` column. On a non-terminal
                transition, error is ignored and a warning is logged.

        Raises:
            InvalidTransitionError: If the (current, target) pair is not in
                ALLOWED_TRANSITIONS.
            DerivaMLStateInconsistency: If state_machine's catalog sync
                detects divergence.

        Example:
            >>> exe.update_status(ExecutionStatus.Running)  # doctest: +SKIP
            >>> exe.update_status(ExecutionStatus.Failed, error="Network timeout")  # doctest: +SKIP
        """
        store = self._ml_object.workspace.execution_state_store()
        row = store.get_execution(self.execution_rid)
        if row is None:
            raise DerivaMLException(f"Execution {self.execution_rid} not in workspace registry")
        current = ExecutionStatus(row["status"])

        extra_fields: dict = {}
        # Terminal = Failed, Aborted.
        if target in (ExecutionStatus.Failed, ExecutionStatus.Aborted):
            if error is not None:
                extra_fields["error"] = error
        elif error is not None:
            logger.warning(
                "error= ignored on non-terminal transition to %s: %s",
                target.value,
                error,
            )

        transition(
            store=store,
            catalog=self._ml_object.catalog if self._ml_object._mode is ConnectionMode.online else None,
            execution_rid=self.execution_rid,
            current=current,
            target=target,
            mode=self._ml_object._mode,
            extra_fields=extra_fields,
        )

    def execution_start(self) -> None:
        """Marks the execution as started.

        Records the start time in SQLite and transitions the execution's
        status from ``Created`` to ``Running`` via the state machine
        (the same path the context-manager ``__enter__`` uses). This
        should be called before beginning the main execution work — its
        non-context-manager counterpart for code paths that can't use
        ``with ml.create_execution(...) as exe:`` (e.g., the multirun
        parent execution managed by an ``atexit`` handler).

        Pairs with ``execution_stop()`` which transitions Running → Stopped.

        Raises:
            InvalidTransitionError: If the execution is not currently in
                ``ExecutionStatus.Created``.

        Example:
            >>> execution.execution_start()  # doctest: +SKIP
            >>> try:  # doctest: +SKIP
            ...     # Run analysis
            ...     execution.execution_stop()  # doctest: +SKIP
            ... except Exception:
            ...     execution.update_status(ExecutionStatus.Failed, error="Analysis error")  # doctest: +SKIP
        """
        from datetime import timezone

        self._logger.info("Start execution...")

        # Dry-run executions don't have a SQLite registry row — there's
        # nothing to transition and the status read-through returns a
        # sentinel. Skip the state-machine call entirely.
        if self._dry_run:
            return

        # Transition Created → Running through the state machine, writing
        # start_time atomically with the status change so a crash between
        # the two can't leave the row inconsistent. Mirrors __enter__ —
        # the only difference is that this path is invoked imperatively
        # by callers that don't (or can't) use the context manager.
        current = self.status
        transition(
            store=self._ml_object.workspace.execution_state_store(),
            catalog=(self._ml_object.catalog if self._ml_object._mode is ConnectionMode.online else None),
            execution_rid=self.execution_rid,
            current=current,
            target=ExecutionStatus.Running,
            mode=self._ml_object._mode,
            extra_fields={"start_time": datetime.now(timezone.utc)},
        )

    def execution_stop(self) -> None:
        """Marks the execution as stopped (algorithm finished successfully).

        Computes the wall-clock duration against ``start_time`` and
        transitions Running → Stopped through the state machine,
        writing ``stop_time`` and ``duration`` atomically with the
        status change. Online callers see a single catalog PUT that
        carries both Status and Duration (the historical second
        catalog write is gone — audit §4.5).

        This should be called after all execution work is finished;
        upload of outputs is a separate phase that moves status from
        Stopped → Pending_Upload → Uploaded.

        Example:
            >>> try:  # doctest: +SKIP
            ...     # Run analysis
            ...     execution.execution_stop()  # doctest: +SKIP
            ... except Exception:
            ...     execution.update_status(ExecutionStatus.Failed, error="Analysis error")  # doctest: +SKIP
        """
        from datetime import timezone

        now = datetime.now(timezone.utc)

        # Compute algorithm duration against start_time. Falls back to
        # "0H 0min 0.0sec" if start_time is missing (shouldn't happen
        # in practice — Running → Stopped requires __enter__ /
        # execution_start to have set start_time).
        duration_str = _format_duration(self.start_time, now)

        if self._dry_run:
            return

        # Single atomic transition: SQLite write (status + stop_time +
        # duration) followed by online catalog PUT (Status + Duration
        # via _catalog_body_for_execution). Replaces the historical
        # two-write design that left Duration vulnerable to a partial
        # failure between writes (audit §4.5).
        current = self.status
        transition(
            store=self._ml_object.workspace.execution_state_store(),
            catalog=(self._ml_object.catalog if self._ml_object._mode is ConnectionMode.online else None),
            execution_rid=self.execution_rid,
            current=current,
            target=ExecutionStatus.Stopped,
            mode=self._ml_object._mode,
            extra_fields={"stop_time": now, "duration": duration_str},
        )

    @validate_call(config=VALIDATION_CONFIG)
    def download_asset(
        self,
        asset_rid: RID,
        dest_dir: Path,
        update_catalog: bool = True,
        use_cache: bool = False,
        _asset_table: Any = None,
    ) -> AssetFilePath:
        """Download an asset from a URL and place it in a local directory.

        The file is written to ``dest_dir / asset_record["Filename"]``.
        Overwrites any existing file at that path, with a WARNING logged
        when the existing content is byte-different from the asset's
        expected MD5 (issue #181). Idempotent re-downloads — the existing
        file's md5 already matches the catalog's recorded MD5 — log
        nothing. Callers that need silent overwrites can accept the
        WARNING; callers that need genuine isolation should pass a unique
        ``dest_dir`` per asset. The canonical pattern is
        ``dest_dir = working_dir / 'downloads' / asset_rid``, which is
        what ``_initialize_execution`` uses internally so the
        platform-default download path is collision-free by construction.

        **Directional tagging.** Assets downloaded via this method are
        recorded as **inputs** of this execution (when
        ``update_catalog=True``, the default). deriva-ml auto-adds the
        ``Input_File`` Asset_Type tag and writes ``Asset_Role="Input"``
        on the ``{Asset}_Execution`` row. The asset's pre-existing
        content tags (e.g., ``Model_File`` if it was a prior
        execution's output) are preserved — the directional tag is
        additive, not a replacement. This is symmetric with the
        ``Output_File`` tag added when assets are uploaded via
        :meth:`asset_file_path` + :meth:`commit_output_assets`.
        See the "How execution-asset roles work" section of the
        execution user guide for the full contract.

        Args:
            asset_rid: RID of the asset.
            dest_dir: Destination directory for the asset. When ``Filename``
                may collide across concurrent ``download_asset`` calls,
                pass a unique directory per asset to avoid the WARNING and
                the silent overwrite it guards against.
            update_catalog: Whether to write the ``{Asset}_Execution``
                row (``Asset_Role="Input"``) and the ``Input_File``
                Asset_Type tag. Default ``True`` — the input-role
                contract requires it. Pass ``False`` only for ad-hoc
                downloads outside an execution-tracking context.
            use_cache: If True, check the cache directory for a previously downloaded copy
                with a matching MD5 checksum before downloading. Cached copies are stored
                in ``cache_dir/assets/{rid}_{md5}/`` and symlinked into the destination.
            _asset_table: Internal — pre-resolved Table object for this RID. When supplied
                (typically from ``_initialize_execution``'s batched ``resolve_rids`` call)
                skips the per-asset ``resolve_rid`` round-trip. Underscore-prefixed because
                callers should not rely on this; pass through the public ``resolve_rid``
                path otherwise.

        Returns:
            An AssetFilePath with the path to the downloaded (or cached) asset file.
            ``AssetFilePath.file_name`` is the canonical access path — read from it
            rather than hand-constructing a path from the asset table or filename.
        """
        from deriva_ml.execution.asset_upload import download_asset as _download_asset

        return _download_asset(
            self,
            asset_rid,
            dest_dir,
            update_catalog=update_catalog,
            use_cache=use_cache,
            _asset_table=_asset_table,
            asset_type_vocab_term=MLVocab.asset_type,
            check_overwrite_safe_fn=_check_overwrite_safe,
        )

    @validate_call(config=VALIDATION_CONFIG)
    def commit_output_assets(
        self,
        clean_folder: bool | None = None,
        progress_callback: Callable[[UploadProgress], None] | None = None,
    ) -> UploadReport:
        """Commit this execution's output assets to the catalog.

        Single per-execution upload entry point (ADR-0009). Reads the
        asset manifest, uploads each file to the catalog's Hatrac
        object store, and inserts ``{Asset}_Execution`` association
        records linking each uploaded asset to this execution with the
        ``Output`` role. Brackets the work with the
        ``Pending_Upload → Uploaded`` (success) or
        ``Pending_Upload → Failed`` (exception) state-machine
        transition, records ``Upload_Duration`` in the SQLite registry,
        writes asset descriptions to the catalog, and optionally cleans
        the execution working folder.

        Call this method **after** exiting the execution context
        manager, not inside it. The context manager sets execution
        status to ``Stopped`` on exit; this method transitions
        ``Stopped → Pending_Upload → Uploaded`` (or ``Failed``).

        Idempotent: re-running after a successful upload (status
        ``Uploaded``, no pending assets) is a no-op that returns an
        empty report. Re-running after a partial failure resumes from
        the last known-good state — ``BagCatalogLoader``'s
        ``match_by_columns`` dedup makes row inserts idempotent at the
        catalog.

        The method raises on failure. Failure isolation is the batch
        caller's job (:meth:`DerivaML.commit_pending_executions`), not
        the per-execution call's.

        **Directional tagging.** Every asset committed by this call
        gets ``Asset_Role="Output"`` on its ``{Asset}_Execution`` row
        and the ``Output_File`` Asset_Type tag (auto-added by
        deriva-ml, in addition to any content tags the caller passed
        via :meth:`asset_file_path(..., asset_types=...)`). This is
        symmetric with the ``Input_File`` tag added by
        :meth:`download_asset`. See the "How execution-asset roles
        work" section of the execution user guide for the full
        contract.

        Args:
            clean_folder: Whether to delete output folders after
                upload. If None (default), uses the DerivaML instance's
                ``clean_execution_dir`` setting. Pass True/False to
                override for this specific execution.
            progress_callback: Optional callback function to receive
                upload progress updates. Called with UploadProgress
                objects containing file name, bytes uploaded, total
                bytes, percent complete, phase, and status message.

        Returns:
            UploadReport with ``execution_rids=[self.execution_rid]``
            and per-(schema, table) upload counts. ``total_uploaded``
            is the sum of asset rows committed across all asset tables
            this execution touched. For dry-run executions, returns an
            empty report.

        Raises:
            DerivaMLUploadError: If any file upload fails. Partial
                uploads are recorded in the manifest so the upload can
                be resumed.
            DerivaMLReadOnlyError: If the catalog connection is
                read-only.

        Example:
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     path = exe.asset_file_path("Model", "model.pt")  # doctest: +SKIP
            >>> report = exe.commit_output_assets()  # doctest: +SKIP
            >>> print(report.total_uploaded, "assets committed")  # doctest: +SKIP
        """
        from deriva_ml.execution.asset_upload import (
            commit_output_assets as _commit_output_assets,
        )

        result = _commit_output_assets(
            self,
            clean_folder=clean_folder,
            progress_callback=progress_callback,
            pending_upload_status=ExecutionStatus.Pending_Upload,
            uploaded_status=ExecutionStatus.Uploaded,
            failed_status=ExecutionStatus.Failed,
            running_status=ExecutionStatus.Running,
            stopped_status=ExecutionStatus.Stopped,
            format_duration_fn=_format_duration,
        )

        # Build UploadReport from the free function's per-table dict.
        # Each value is a list of AssetFilePath; the count is the
        # number of asset rows committed for that table.
        total_uploaded = sum(len(v) for v in result.values())
        per_table = {fqn: {"uploaded": len(v), "failed": 0} for fqn, v in result.items()}
        return UploadReport(
            execution_rids=[self.execution_rid],
            total_uploaded=total_uploaded,
            total_failed=0,
            per_table=per_table,
            errors=[],
        )

    def _bag_commit_upload(
        self,
        *,
        progress_callback: Callable[[UploadProgress], None] | None = None,
    ) -> dict[str, list[AssetFilePath]]:
        """Upload pending execution outputs via the bag pipeline.

        Thin delegate to ``asset_upload.bag_commit_upload``;
        see that helper for the bag-build → bag-load → manifest-mark
        flow.
        """
        from deriva_ml.execution.asset_upload import bag_commit_upload

        return bag_commit_upload(self, progress_callback=progress_callback)

    def _clean_folder_contents(self, folder_path: Path, remove_folder: bool = True):
        """Clean up folder contents and optionally the folder itself.

        Thin delegate to ``asset_upload.clean_folder_contents``.
        The helper doesn't need ``self`` at all — it's a pure
        filesystem op that lived on ``Execution`` for legacy
        reasons. Preserved here as an instance method so the
        existing in-class call sites
        (``commit_output_assets``, ``__exit__``,
        ``execution_stop``) still read naturally.
        """
        from deriva_ml.execution.asset_upload import clean_folder_contents

        clean_folder_contents(
            folder_path,
            remove_folder=remove_folder,
            logger=self._logger,
        )

    def _update_asset_execution_table(
        self,
        uploaded_assets: dict[str, list[AssetFilePath]],
        asset_role: str = "Output",
    ) -> None:
        """Link assets to this execution and auto-tag them by role.

        Thin delegate to ``asset_upload.update_asset_execution_table``;
        see that helper for the per-branch (Input/Output)
        logic.

        **Audit ledger note (Output branch is NOT dead):**
        The 2026-05-22 audit recommended dropping the Output
        branch as "dead in production." That recommendation
        is rejected — ``Asset_Role`` Input vs Output is real
        public-API behaviour (``execution.list_assets(asset_role=...)``).
        A prior pass eliminated this in error; do not repeat
        that mistake. The bag-commit Output flow at
        ``bag_commit._add_asset_rows_to_bag`` writes the same
        rows for bag-pipeline assets; this branch handles
        non-bag callers.
        """
        from deriva_ml.execution.asset_upload import update_asset_execution_table

        update_asset_execution_table(
            self,
            uploaded_assets,
            asset_role=asset_role,
            asset_role_vocab_term=MLVocab.asset_role,
            input_file_tag=ExecAssetType.input_file.value,
            output_file_tag=ExecAssetType.output_file.value,
            asset_type_path_fn=asset_type_path,
        )

    @validate_call(config=VALIDATION_CONFIG)
    def asset_file_path(
        self,
        asset_name: str,
        file_name: str | Path,
        asset_types: list[str] | str | None = None,
        copy_file=False,
        rename_file: str | None = None,
        metadata=None,
        description: str | None = None,
        **kwargs,
    ) -> AssetFilePath:
        """Register a file for upload and return a path to write to.

        This routine has three modes depending on whether file_name refers to an existing file:
        1. **New file**: file_name doesn't exist — returns a path to write to.
        2. **Symlink**: file_name exists, copy_file=False — symlinks into staging.
        3. **Copy**: file_name exists, copy_file=True — copies into staging.

        Files are stored in a flat per-table directory (``assets/{AssetTable}/``).
        Metadata is tracked in a persistent JSON manifest for crash safety.
        Metadata can be set at registration time via the ``metadata`` parameter
        (an AssetRecord or dict) or incrementally after via the returned
        AssetFilePath's ``metadata`` property.

        Thin delegate to ``asset_upload.asset_file_path``;
        see that helper for the per-mode logic and the
        ``asset_types is None`` vs explicit-empty-list
        normalization.

        **Directional tagging.** Files registered via this method are
        uploaded as **outputs** of this execution. After
        :meth:`commit_output_assets` runs, deriva-ml auto-adds
        the ``Output_File`` Asset_Type tag to every uploaded asset
        (alongside any content tags you passed in ``asset_types``).
        You don't pass ``Output_File`` yourself — it's framework-
        supplied, deduplicated if explicit, and symmetric with the
        ``Input_File`` tag added by :meth:`download_asset`. The
        ``{Asset}_Execution`` row gets ``Asset_Role="Output"``.
        See the "How execution-asset roles work" section of the
        execution user guide for the full contract.

        Args:
            asset_name: Name of the asset table. Must be a valid asset table.
            file_name: Name of file to be uploaded, or path to an existing file.
            asset_types: Content-classification tags from the Asset_Type
                vocabulary (e.g., ``["Model_File"]``,
                ``["Segmentation_Mask"]``). The directional
                ``Output_File`` tag is added automatically — do not
                pass it explicitly. Defaults to ``asset_name``.
            copy_file: Whether to copy the file rather than creating a symbolic link.
            rename_file: If provided, rename the file during staging.
            metadata: An AssetRecord instance or dict of metadata column values.
            description: Optional description for the asset record.
            **kwargs: Additional metadata values (legacy support, merged with metadata).

        Returns:
            AssetFilePath bound to the manifest for write-through metadata updates.

        Raises:
            DerivaMLException: If the asset table doesn't exist.
            DerivaMLValidationError: If asset_types contains invalid terms.
        """
        from deriva_ml.execution.asset_upload import asset_file_path as _asset_file_path

        return _asset_file_path(
            self,
            asset_name,
            file_name,
            asset_types=asset_types,
            copy_file=copy_file,
            rename_file=rename_file,
            metadata=metadata,
            description=description,
            asset_type_vocab_term=MLVocab.asset_type,
            flat_asset_dir_fn=flat_asset_dir,
            asset_type_path_fn=asset_type_path,
            legacy_kwargs=kwargs,
        )

    def metrics_file(self, filename: str = "metrics.jsonl") -> AssetFilePath:
        """Return a path for writing training-metric records.

        Thin sugar over ``asset_file_path(MLAsset.execution_metadata, ...)``
        that stamps the file with ``asset_types=Metrics_File`` so the
        catalog's ``Execution_Metadata.Type`` column honestly describes
        the file's purpose. The file registers with the execution's asset
        manifest on first call and uploads as part of
        ``commit_output_assets()``.

        The file itself is plain text; callers decide the format. The
        default filename ``metrics.jsonl`` suggests one JSON record per
        line — the simplest shape that lets a downstream reader page
        through the file without loading the whole thing into memory —
        but CSV, YAML, or a single JSON object also work as long as the
        readback code knows the format.

        Repeated calls inside the same execution return the **same**
        AssetFilePath (registered once in the manifest), so append-style
        writes across an epoch loop are safe::

            with ml.create_execution(cfg) as exe:
                for epoch in range(num_epochs):
                    train_loss = train_one_epoch(...)
                    val_loss = evaluate(...)
                    with exe.metrics_file().open("a") as f:
                        json.dump(
                            {"epoch": epoch, "train_loss": train_loss,
                             "val_loss": val_loss},
                            f,
                        )
                        f.write("\\n")
            exe.commit_output_assets()

        Args:
            filename: Name of the metrics file inside the execution's
                Execution_Metadata staging area. Defaults to
                ``"metrics.jsonl"``. Override to distinguish multiple
                metric streams (e.g. ``"train_metrics.jsonl"`` +
                ``"eval_metrics.jsonl"``) — each distinct filename
                becomes a separate Execution_Metadata asset on upload.

        Returns:
            AssetFilePath for the metrics file. Use its ``.open(...)``
            method to read, write, or append; the manifest tracks the
            file for upload regardless of how you write to it.

        Raises:
            DerivaMLException: If the Execution_Metadata table is not
                present in the catalog schema (should never happen in a
                correctly-initialized catalog).
            DerivaMLValidationError: If the ``Metrics_File`` term is
                missing from the Asset_Type vocabulary (i.e. an old
                catalog predating this feature; run ``create_ml_schema``
                or ``initialize_ml_schema`` once to seed the term).

        Example:
            >>> from deriva_ml.core.enums import MLAsset, ExecMetadataType  # doctest: +SKIP
            >>> MLAsset.execution_metadata.value  # doctest: +SKIP
            'Execution_Metadata'
            >>> ExecMetadataType.metrics_file.value  # doctest: +SKIP
            'Metrics_File'

            >>> # Catalog-dependent end-to-end flow:
            >>> import json  # doctest: +SKIP
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     with exe.metrics_file().open("a") as f:  # doctest: +SKIP
            ...         json.dump({"epoch": 0, "val_loss": 0.23}, f)  # doctest: +SKIP
            ...         f.write("\\n")  # doctest: +SKIP
            >>> exe.commit_output_assets()  # doctest: +SKIP
        """
        from deriva_ml.execution.asset_upload import metrics_file as _metrics_file

        return _metrics_file(
            self,
            filename,
            execution_metadata_asset_name=MLAsset.execution_metadata,
            metrics_file_asset_type=ExecMetadataType.metrics_file.value,
        )

    def _get_manifest(self) -> AssetManifest:
        """Get or create the asset manifest for this execution."""
        if not hasattr(self, "_manifest") or self._manifest is None:
            ws = self._ml_object.workspace
            self._manifest = AssetManifest(ws.manifest_store(), self.execution_rid)
        return self._manifest

    @property
    def _manifest_store(self) -> "ManifestStore":
        """Return the ManifestStore for this workspace.

        Used by ``add_features`` to stage feature records to SQLite; the
        bag-commit path reads them back via
        ``bag_commit._add_staged_feature_rows_to_bag`` for the catalog insert.
        """
        return self._ml_object.workspace.manifest_store()

    def table_path(self, table: str) -> Path:
        """Return a local file path to a CSV to add values to a table on upload.

        Args:
            table: Name of table to be uploaded.

        Returns:
            Pathlib path to the file in which to place table values.

        Raises:
            DerivaMLException: If ``table`` is not found in any domain schema.

        Example:
            >>> path = exe.table_path("Measurement")  # doctest: +SKIP
            >>> with path.open("w") as f:  # doctest: +SKIP
            ...     writer = csv.DictWriter(f, fieldnames=["Subject", "Score"])  # doctest: +SKIP
            ...     writer.writerow({"Subject": "IMG-1", "Score": 0.95})  # doctest: +SKIP
        """
        # Find which domain schema contains this table
        table_schema = None
        for domain_schema in self._ml_object.domain_schemas:
            if domain_schema in self._model.schemas:
                if table in self._model.schemas[domain_schema].tables:
                    table_schema = domain_schema
                    break

        if table_schema is None:
            raise DerivaMLException("Table '{}' not found in any domain schema".format(table))

        return table_path(self._working_dir, schema=table_schema, table=table)

    def execute(self) -> Execution:
        """Return self so this Execution can be used as a context manager.

        Per spec §2.8, the lifecycle transitions (created → running →
        stopped/failed) live on ``__enter__`` / ``__exit__``. ``execute()``
        itself is a no-op that simply returns ``self`` so usage reads
        naturally as ``with exe.execute() as e: ...``.

        Returns:
            This Execution instance, which is itself a context manager.

        Example:
            >>> with exe.execute() as e:  # doctest: +SKIP
            ...     # e.status is ExecutionStatus.Running
            ...     pass
            >>> # e.status is ExecutionStatus.Stopped (or failed on exception)
        """
        return self

    def list_input_datasets(self) -> list[Dataset]:
        """List all datasets that were inputs to this execution.

        Excludes any dataset this execution itself *produced* — the
        ``Dataset_Execution`` association table has no role column to
        distinguish inputs from outputs, so we infer authorship from
        each dataset's ``Dataset_Version.Execution`` link.

        Returns:
            List of Dataset objects that were used as inputs.

        Example:
            >>> for ds in execution.list_input_datasets():  # doctest: +SKIP
            ...     print(f"Input: {ds.dataset_rid} - {ds.description}")  # doctest: +SKIP
        """
        if self._execution_record is not None:
            return self._execution_record.list_input_datasets()

        # Fallback for dry-run mode (no execution record bound).
        # Delegates to the shared helper so the dry-run path
        # and the canonical ``ExecutionRecord.list_input_datasets``
        # path stay in lockstep — pre-fix they re-implemented the
        # same producer-filter walk in two places.
        from deriva_ml.execution._helpers import list_input_datasets as _list_input_datasets

        return _list_input_datasets(
            ml_instance=self._ml_object,
            execution_rid=self.execution_rid,
        )

    def list_assets(self, asset_role: str | None = None) -> list["Asset"]:
        """List all assets that were inputs or outputs of this execution.

        Args:
            asset_role: Optional filter: "Input" or "Output". If None, returns all.

        Returns:
            List of Asset objects associated with this execution.

        Example:
            >>> inputs = execution.list_assets(asset_role="Input")  # doctest: +SKIP
            >>> outputs = execution.list_assets(asset_role="Output")  # doctest: +SKIP
        """
        if self._execution_record is not None:
            return self._execution_record.list_assets(asset_role=asset_role)

        # Fallback for dry-run mode (no execution record bound).
        # Delegates to the shared helper so the dry-run path
        # and the canonical ``ExecutionRecord.list_assets`` path
        # stay in lockstep — pre-fix the dry-run path only
        # walked ``Execution_Asset_Execution`` while the
        # canonical path walked every ``*_Execution`` association
        # across all schemas. That mismatch was invisible until a
        # dry-run scenario exercised assets in a domain-schema
        # table other than ``Execution_Asset``; now both paths
        # do the full walk.
        from deriva_ml.execution._helpers import list_assets as _list_assets

        return _list_assets(
            ml_instance=self._ml_object,
            execution_rid=self.execution_rid,
            asset_role=asset_role,
        )

    @validate_call(config=VALIDATION_CONFIG)
    def create_dataset(
        self,
        dataset_types: str | list[str] | None = None,
        version: DatasetVersion | str | None = None,
        description: str = "",
    ) -> Dataset:
        """Create a new dataset tracked to this execution.

        Creates a ``Dataset`` catalog record linked to this execution as its
        provenance. The dataset is immediately usable for adding members and
        incrementing versions.

        Args:
            dataset_types: One or more dataset type vocabulary term names to apply.
                Must be pre-registered via ``add_dataset_type``. Pass ``None``
                or an empty list to create an untyped dataset.
            description: Human-readable description of the dataset. Stored in
                the catalog ``Dataset.Description`` column.
            version: Dataset version. Defaults to 0.1.0.

        Returns:
            A ``Dataset`` instance bound to the newly created catalog record.

        Raises:
            DerivaMLInvalidTerm: If any name in ``dataset_types`` is not a
                registered ``Dataset_Type`` vocabulary term.
            DerivaMLExecutionError: If the execution context is no longer active.

        Example:
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     ds = exe.create_dataset(  # doctest: +SKIP
            ...         dataset_types=["training"],  # doctest: +SKIP
            ...         description="Training images v1",  # doctest: +SKIP
            ...     )  # doctest: +SKIP
        """
        return Dataset.create_dataset(
            ml_instance=self._ml_object,
            execution_rid=self.execution_rid,
            dataset_types=dataset_types,
            version=version,
            description=description,
        )

    def add_input_dataset(self, dataset_rid: RID, version: DatasetVersion | str | None = None) -> None:
        """Record an existing dataset as an *input* consumed by this execution.

        Writes a single ``Dataset_Execution`` association row linking
        ``dataset_rid`` to this execution. Because this execution did
        not *author* ``dataset_rid`` (its ``Dataset_Version.Execution``
        link points at whichever execution created it), the
        input/output inference used by :meth:`list_input_datasets`
        classifies it as an **input** — a consume edge — not an output.

        This is the counterpart of :meth:`create_dataset` (which records
        an *output*). Use it when an execution reads a dataset it did not
        produce and you want that consumption recorded as walkable
        provenance (so :meth:`list_input_datasets` and lineage walks can
        reach the consumed dataset), *without* the bag download that
        declaring the dataset in ``ExecutionConfiguration.datasets``
        would trigger. The most common caller is
        :func:`deriva_ml.dataset.split.split_dataset`, which records its
        source dataset as an input of the splitting execution.

        When ``version`` is supplied, the consumed version is recorded on
        the input edge via the ``Dataset_Execution.Dataset_Version`` FK
        (resolved through :meth:`_version_rid`). When ``version`` is
        ``None`` the ``Dataset_Version`` column is left NULL — fully
        backward-compatible with callers that pass only a RID.

        The link is idempotent: if ``dataset_rid`` is already associated
        with this execution, no duplicate row is inserted. In dry-run
        mode this is a no-op (the dry-run contract forbids catalog
        mutations).

        Args:
            dataset_rid: RID of the already-existing dataset that this
                execution consumed as an input.
            version: Optional consumed dataset version. May be a
                :class:`DatasetVersion` or a version string. When given,
                recorded on the input edge's ``Dataset_Version`` FK.

        Example:
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     exe.add_input_dataset("1-ABC0")  # doctest: +SKIP
            ...     # "1-ABC0" now appears in exe.list_input_datasets()
        """
        if self._dry_run:
            return
        schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
        dataset_exec = schema_path.Dataset_Execution
        already_linked = {
            row["Dataset"]
            for row in dataset_exec.filter(dataset_exec.Execution == self.execution_rid).entities().fetch()
        }
        if dataset_rid in already_linked:
            return
        version_rid = self._ml_object._version_rid(dataset_rid, version) if version is not None else None
        dataset_exec.insert(
            [{"Dataset": dataset_rid, "Execution": self.execution_rid, "Dataset_Version": version_rid}]
        )

    @validate_call(config=VALIDATION_CONFIG)
    def add_files(
        self,
        files: Iterable[FileSpec],
        dataset_types: str | list[str] | None = None,
        description: str = "",
    ) -> "Dataset":
        """Adds files to the catalog with their metadata.

        Registers files in the catalog along with their metadata (MD5, length, URL) and associates them with
        specified file types.

        Args:
            files: File specifications containing MD5 checksum, length, and URL.
            dataset_types: One or more dataset type terms from File_Type vocabulary.
            description: Description of the files.

        Returns:
            RID: Dataset  that identifies newly added files. Will be nested to mirror original directory structure
            of the files.

        Raises:
            DerivaMLInvalidTerm: If file_types are invalid or execution_rid is not an execution record.

        Examples:
            Add a single file type:
                >>> files = [FileSpec(url="path/to/file.txt", md5="abc123", length=1000)]  # doctest: +SKIP
                >>> rids = exe.add_files(files, dataset_types="text")  # doctest: +SKIP

            Add multiple file types:
                >>> rids = exe.add_files(  # doctest: +SKIP
                ...     files=[FileSpec(url="image.png", md5="def456", length=2000)],  # doctest: +SKIP
                ...     dataset_types=["image", "png"],  # doctest: +SKIP
                ... )  # doctest: +SKIP
        """
        return self._ml_object.add_files(
            files=files,
            execution_rid=self.execution_rid,
            dataset_types=dataset_types,
            description=description,
        )

    # =========================================================================
    # Execution Nesting Methods
    # =========================================================================

    def add_nested_execution(
        self,
        nested_execution: "Execution | ExecutionRecord | RID",
        sequence: int | None = None,
    ) -> None:
        """Add a nested (child) execution to this execution.

        Creates a parent-child relationship between this execution and another.
        This is useful for grouping related executions, such as parameter sweeps
        or pipeline stages.

        Args:
            nested_execution: The child execution to add (Execution, ExecutionRecord, or RID).
            sequence: Optional ordering index (0, 1, 2...). Use None for parallel executions.

        Raises:
            DerivaMLException: If the association cannot be created.

        Example:
            >>> parent_exec = ml.create_execution(parent_config)  # doctest: +SKIP
            >>> child_exec = ml.create_execution(child_config)  # doctest: +SKIP
            >>> parent_exec.add_nested_execution(child_exec, sequence=0)  # doctest: +SKIP
        """
        if self._dry_run:
            return

        # Get the RID from the nested execution
        if isinstance(nested_execution, Execution):
            nested_rid = nested_execution.execution_rid
        elif isinstance(nested_execution, ExecutionRecord):
            nested_rid = nested_execution.execution_rid
        else:
            nested_rid = nested_execution

        # Delegate to ExecutionRecord if available
        if self._execution_record is not None:
            self._execution_record.add_nested_execution(nested_rid, sequence=sequence)
        else:
            # Fallback for cases without execution record
            from deriva_ml.execution._helpers import insert_nested_execution_link

            insert_nested_execution_link(
                ml_instance=self._ml_object,
                parent_rid=self.execution_rid,
                child_rid=nested_rid,
                sequence=sequence,
            )

    def is_nested(self) -> bool:
        """Check if this execution is nested within another execution.

        Hierarchy queries live on :class:`ExecutionRecord` only (per spec
        R2.1). This shortcut delegates to
        ``self._execution_record.is_nested()`` when available.

        Returns:
            True if this execution has at least one parent execution.

        Raises:
            DerivaMLException: If this Execution has no bound
                ExecutionRecord (e.g. a dry-run execution).
        """
        if self._execution_record is None:
            raise DerivaMLException(
                "is_nested requires a bound ExecutionRecord. Hierarchy queries are not available in dry-run mode."
            )
        return self._execution_record.is_nested()

    def is_parent(self) -> bool:
        """Check if this execution has nested child executions.

        Hierarchy queries live on :class:`ExecutionRecord` only (per spec
        R2.1). This shortcut delegates to
        ``self._execution_record.is_parent()`` when available.

        Returns:
            True if this execution has at least one nested execution.

        Raises:
            DerivaMLException: If this Execution has no bound
                ExecutionRecord (e.g. a dry-run execution).
        """
        if self._execution_record is None:
            raise DerivaMLException(
                "is_parent requires a bound ExecutionRecord. Hierarchy queries are not available in dry-run mode."
            )
        return self._execution_record.is_parent()

    def __str__(self):
        items = [
            f"caching_dir: {self._cache_dir}",
            f"_working_dir: {self._working_dir}",
            f"execution_rid: {self.execution_rid}",
            f"workflow_rid: {self.workflow_rid}",
            f"asset_paths: {self.asset_paths}",
            f"configuration: {self.configuration}",
        ]
        return "\n".join(items)

    def __repr__(self) -> str:
        """One-line summary including status and pending counts.

        Pending counts read SQLite — no caching. Example output::

            <Execution EXE-A status=stopped pending=15rows/2files>

        Omits the pending suffix when there are no pending rows or
        files. Always guards against exceptions — ``repr`` MUST NOT
        raise, so reads that would raise (e.g., the registry row is
        missing) degrade to ``<Execution EXE-A>``.

        Returns:
            Compact repr string suitable for logs and interactive use.
        """
        try:
            store = self._ml_object.workspace.execution_state_store()
            row = store.get_execution(self.execution_rid)
            if row is None:
                return f"<Execution {self.execution_rid} status=? (not in registry)>"
            counts = store.count_pending_by_kind(execution_rid=self.execution_rid)
            pending_part = ""
            if counts["pending_rows"] or counts["pending_files"]:
                pending_part = f" pending={counts['pending_rows']}rows/{counts['pending_files']}files"
            return f"<Execution {self.execution_rid} status={row['status']}{pending_part}>"
        except Exception:  # repr must not raise
            return f"<Execution {self.execution_rid}>"

    def __enter__(self) -> "Execution":
        """Begin the execution: status created → running.

        Context-manager wrapper around :meth:`execution_start`. The
        actual transition (Created → Running, with ``start_time``
        atomically set in the same SQLite write and an online-mode
        catalog sync) lives in ``execution_start``; this method just
        delegates so the imperative and context-manager paths share
        one implementation.

        Dry-run executions skip the transition entirely — there is no
        SQLite registry row for a dry-run (sentinel RID), so there is
        nothing to transition. ``start_time`` reads back as None for
        dry-runs.

        Returns:
            This Execution instance.

        Raises:
            InvalidTransitionError: If the execution is not currently
                in ``ExecutionStatus.Created``.

        Example:
            >>> with exe.execute() as e:  # doctest: +SKIP
            ...     e.status  # doctest: +SKIP
            <ExecutionStatus.Running>
        """
        self.execution_start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> bool:
        """End the execution: status running → stopped (clean) or failed.

        On clean exit, transitions to ``stopped``. On exception, transitions
        to ``failed`` and stores the exception message in the ``error``
        column. Per spec §2.12 / R6.3, returns False to propagate any
        exception (unlike the legacy ``__exit__`` which returned True to
        suppress). After the transition, emits an INFO log summarizing
        pending rows/files if any are still staged (the full
        ``PendingSummary`` render lands in Group G; this is the
        placeholder.)

        Args:
           exc_type: Exception type (or None on clean exit).
           exc_value: Exception value (or None on clean exit).
           exc_tb: Exception traceback (or None on clean exit).

        Returns:
           False — always. Any exception propagates to the caller.
        """
        from datetime import datetime, timezone

        if self._dry_run:
            # No SQLite row for dry-run executions; stop_time read-through
            # returns None. Log any exception before returning.
            if exc_value is not None:
                logging.error(
                    "Dry-run execution failed: %s: %s",
                    exc_type.__name__,
                    exc_value,
                )
            return False

        current = self.status
        now = datetime.now(timezone.utc)

        # Tolerate executions that have already advanced past Running by
        # the time __exit__ fires. The canonical pattern for the
        # context manager is "exit at Running → Stopped"; but it is
        # legal (and fixture code in demo_catalog.py does this) to
        # call commit_output_assets() inside the with block, which
        # advances Running → Stopped → Pending_Upload → Uploaded
        # before __exit__ is invoked. Forcing a Stopped/Failed
        # transition from a terminal state would crash the caller's
        # successful path on the way out. Leave terminal states alone
        # — the work is already done.
        if current in {
            ExecutionStatus.Stopped,
            ExecutionStatus.Pending_Upload,
            ExecutionStatus.Uploaded,
            ExecutionStatus.Failed,
            ExecutionStatus.Aborted,
        }:
            return False

        if exc_value is None:
            # Clean exit: delegate Running → Stopped to execution_stop() so
            # the duration computation + single-atomic-transition contract
            # (audit §4.5) is honored. The inline transition this replaced
            # wrote stop_time but not duration, leaving the catalog
            # Execution_Duration column null for every with-block exit —
            # see docs/bugs/2026-05-19-execution-exit-omits-duration.md.
            self.execution_stop()
        else:
            # Failed exit: write stop_time, error, AND duration so the
            # Execution_Duration column reflects how long the run got
            # before it crashed. Useful diagnostic when comparing failed
            # vs successful runs of the same workflow.
            duration_str = _format_duration(self.start_time, now)
            transition(
                store=self._ml_object.workspace.execution_state_store(),
                catalog=(self._ml_object.catalog if self._ml_object._mode is ConnectionMode.online else None),
                execution_rid=self.execution_rid,
                current=current,
                target=ExecutionStatus.Failed,
                mode=self._ml_object._mode,
                extra_fields={
                    "stop_time": now,
                    "duration": duration_str,
                    "error": f"{exc_type.__name__}: {exc_value}",
                },
            )

        # Emit the pending-summary INFO log per §2.12 / R6.3. Full
        # PendingSummary object lands in Group G; this is a placeholder.
        store = self._ml_object.workspace.execution_state_store()
        counts = store.count_pending_by_kind(execution_rid=self.execution_rid)
        if counts["pending_rows"] or counts["pending_files"]:
            logger.info(
                "[Execution %s] exited with pending: %d rows, %d files. Call exe.commit_output_assets() to flush.",
                self.execution_rid,
                counts["pending_rows"],
                counts["pending_files"],
            )

        if exc_value is not None:
            logging.error(
                "Execution %s failed: %s: %s",
                self.execution_rid,
                exc_type.__name__,
                exc_value,
            )

        # Propagate any exception.
        return False

    def abort(self) -> None:
        """Mark this execution as aborted.

        Legal from any non-terminal status (``created``, ``running``,
        ``stopped``, ``failed``). Pending rows are NOT discarded —
        the user can inspect them and decide whether to recover via
        ``resume_execution`` or discard via ``gc_executions``.

        Dry-run executions have no SQLite registry row, so abort() is
        a no-op for them.

        Raises:
            InvalidTransitionError: If the current status doesn't allow
                abort (e.g., status='uploaded' — terminal).

        Example:
            >>> exe = ml.resume_execution("EXE-A")  # doctest: +SKIP
            >>> exe.abort()  # doctest: +SKIP
            >>> exe.status  # doctest: +SKIP
            <ExecutionStatus.Aborted>
        """
        if self._dry_run:
            return

        transition(
            store=self._ml_object.workspace.execution_state_store(),
            catalog=(self._ml_object.catalog if self._ml_object._mode is ConnectionMode.online else None),
            execution_rid=self.execution_rid,
            current=self.status,
            target=ExecutionStatus.Aborted,
            mode=self._ml_object._mode,
        )

    def pending_summary(self) -> "PendingSummary":
        """Return a snapshot of pending upload state for this execution.

        Read-only; does not affect state. Safe to call from anywhere at
        any time, including from a separate process holding the same
        workspace.

        Returns:
            PendingSummary with per-table row and asset counts and
            diagnostic messages from any failed rows.

        Example:
            >>> summary = exe.pending_summary()  # doctest: +SKIP
            >>> if summary.has_pending:  # doctest: +SKIP
            ...     print(summary.render())  # doctest: +SKIP
        """
        from deriva_ml.execution.pending_summary import (
            PendingAssetCount,
            PendingRowCount,
            PendingSummary,
        )

        store = self._ml_object.workspace.execution_state_store()
        data = store.pending_summary_rows(execution_rid=self.execution_rid)
        return PendingSummary(
            execution_rid=self.execution_rid,
            rows=[PendingRowCount(**r) for r in data["rows"]],
            assets=[PendingAssetCount(**a) for a in data["assets"]],
            diagnostics=data["diagnostics"],
        )

"""Execution management mixin for DerivaML.

This module provides the ExecutionMixin class which handles
execution operations including creating, resuming, and updating
execution status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
from deriva_ml.execution.state_machine import (
    flush_pending_sync,
    reconcile_with_catalog,
)
from deriva_ml.execution.state_store import ExecutionStatus
from deriva_ml.execution.upload_engine import run_upload_engine

if TYPE_CHECKING:
    from deriva_ml.asset.aux_classes import AssetSpec
    from deriva_ml.dataset.aux_classes import DatasetSpec
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.execution_record import ExecutionRecord
    from deriva_ml.execution.pending_summary import WorkspacePendingSummary
    from deriva_ml.execution.upload_engine import UploadReport
    from deriva_ml.execution.upload_job import UploadJob
    from deriva_ml.execution.workflow import Workflow
    from deriva_ml.experiment.experiment import Experiment
    from deriva_ml.model.catalog import DerivaModel


class ExecutionMixin:
    """Mixin providing execution management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - working_dir: Path - working directory path
        - pathBuilder(): method returning catalog path builder
        - _retrieve_rid(): method for retrieving RID data (from RidResolutionMixin)

    Methods:
        create_execution: Create a new execution environment
        resume_execution: Re-hydrate a previous execution from the registry
        list_executions / find_executions: Query the registry / catalog for executions
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    working_dir: Path
    pathBuilder: Callable[[], Any]
    _retrieve_rid: Callable[[RID], dict[str, Any]]
    _execution: "Execution"

    def create_execution(
        self,
        configuration: "ExecutionConfiguration | None" = None,
        *,
        datasets: "list[DatasetSpec | str] | None" = None,
        assets: "list[AssetSpec | str] | None" = None,
        workflow: "Workflow | RID | str | None" = None,
        description: "str | None" = None,
        dry_run: bool = False,
    ) -> "Execution":
        """Create an execution environment.

        Initializes a local compute environment for executing an ML or
        analytic routine. Accepts either a pre-built
        :class:`ExecutionConfiguration` (the config-object form) or
        individual keyword arguments that the method assembles into an
        ``ExecutionConfiguration`` (the kwargs form). Mixing the two
        forms is rejected with ``TypeError`` — pick one.

        Creating executions requires online mode because the Execution
        RID is server-assigned.

        Side effects:

        1. Downloads datasets specified in the configuration to the
           cache directory. If no version is specified, creates a new
           minor version for the dataset.
        2. Downloads any execution assets to the working directory.
        3. Creates an execution record in the catalog (unless
           ``dry_run=True``).

        Args:
            configuration: A pre-built ExecutionConfiguration. If this
                is provided, all of the kwargs below (except
                ``dry_run``) must be None.
            datasets: Kwargs form only. List of :class:`DatasetSpec`
                or ``"RID@version"`` shorthand strings; strings are
                coerced via :meth:`DatasetSpec.from_shorthand`.
            assets: Kwargs form only. List of :class:`AssetSpec` or
                bare RID strings; strings are coerced to
                ``AssetSpec(rid=...)``.
            workflow: A :class:`Workflow` object, a Workflow RID, or a
                workflow URL string. Strings that look like URLs are
                resolved via :meth:`lookup_workflow_by_url`. Accepted
                in both forms — in the config-object form it
                supplements / overrides ``configuration.workflow``; in
                the kwargs form it is the workflow for the execution.
            description: Kwargs form only. Human-readable description
                of the execution.
            dry_run: If True, skip creating catalog records and
                uploading results.

        Returns:
            An :class:`Execution` object for managing the execution
            lifecycle.

        Raises:
            TypeError: If ``configuration`` is given alongside
                ``datasets``, ``assets``, or ``description`` kwargs.
            DerivaMLOfflineError: If the current connection mode is
                :attr:`ConnectionMode.offline`.

        Example:
            Config-object form::

                >>> config = ExecutionConfiguration(  # doctest: +SKIP
                ...     workflow=workflow,
                ...     description="Process samples",
                ...     datasets=[DatasetSpec(rid="4HM", version="1.0.0")],
                ... )
                >>> with ml.create_execution(config) as execution:
                ...     # Run analysis
                ...     pass
                >>> execution.upload_execution_outputs()

            Kwargs form (equivalent)::

                >>> with ml.create_execution(  # doctest: +SKIP
                ...     datasets=["4HM@1.0.0"],
                ...     workflow=workflow,
                ...     description="Process samples",
                ... ) as execution:
                ...     # Run analysis
                ...     pass
        """
        # Import here to avoid circular dependency
        from deriva_ml.asset.aux_classes import AssetSpec
        from deriva_ml.core.exceptions import DerivaMLOfflineError
        from deriva_ml.dataset.aux_classes import DatasetSpec
        from deriva_ml.execution.execution import Execution
        from deriva_ml.execution.execution_configuration import ExecutionConfiguration
        from deriva_ml.execution.workflow import Workflow as WorkflowClass

        # Offline guard first — the error should be about the mode,
        # not a downstream validation issue.
        if self._mode is ConnectionMode.offline:
            raise DerivaMLOfflineError(
                "create_execution requires online mode — the Execution "
                "RID is server-assigned. Switch to "
                "ConnectionMode.online to create, then you can run "
                "offline scripts via resume_execution."
            )

        # Reject mixed forms. Note: ``workflow`` and ``dry_run`` are
        # accepted in both forms (workflow had legacy config-form
        # support; dry_run is universal). Only the kwargs-only fields
        # gate the mixing check.
        kwargs_form_only = any(
            x is not None for x in (datasets, assets, description)
        )
        if configuration is not None and kwargs_form_only:
            raise TypeError(
                "create_execution: cannot mix configuration= with "
                "datasets/assets/description kwargs; pass exactly one "
                "form."
            )

        # Resolve a string workflow to a Workflow object (used by both forms).
        resolved_workflow = workflow
        if isinstance(resolved_workflow, str):
            resolved_workflow = self.lookup_workflow_by_url(resolved_workflow)

        if configuration is None:
            # Kwargs form: assemble an ExecutionConfiguration.
            ds_specs: list[DatasetSpec] = []
            for d in datasets or []:
                if isinstance(d, str):
                    ds_specs.append(DatasetSpec.from_shorthand(d))
                else:
                    ds_specs.append(d)
            as_specs: list[AssetSpec] = []
            for a in assets or []:
                if isinstance(a, str):
                    as_specs.append(AssetSpec(rid=a))
                else:
                    as_specs.append(a)

            configuration = ExecutionConfiguration(
                datasets=ds_specs,
                assets=as_specs,
                workflow=resolved_workflow
                if isinstance(resolved_workflow, WorkflowClass)
                else None,
                description=description or "",
            )
            # If workflow is a RID (not a Workflow or string URL),
            # pass it through the legacy workflow= parameter so
            # Execution.__init__ can raise its own clear error
            # (our job is just assembly, not re-validation).
            workflow_for_execution = (
                resolved_workflow
                if not isinstance(resolved_workflow, WorkflowClass)
                else None
            )
        else:
            # Config-object form: preserve legacy behaviour.
            workflow_for_execution = resolved_workflow

        # Create and store an execution instance
        self._execution = Execution(
            configuration,
            self,
            workflow=workflow_for_execution,
            dry_run=dry_run,
        )  # type: ignore[arg-type]
        return self._execution

    def lookup_execution(self, execution_rid: RID) -> "ExecutionRecord":
        """Look up a single execution by RID in the live catalog.

        Queries the ERMrest catalog for the Execution row with the given
        RID and returns an ``ExecutionRecord`` — a live, catalog-bound
        value whose mutable properties (``status``, ``description``)
        write through to the catalog on assignment. Online mode only.

        For enumerating executions from the local SQLite registry without
        touching the catalog, see ``list_executions()``. For catalog-side
        filter queries returning live records, see ``find_executions()``.

        Args:
            execution_rid: Resource Identifier (RID) of the execution.

        Returns:
            A live ``ExecutionRecord`` bound to the catalog. Property
            setters (``record.status = ...``) write through.

        Raises:
            DerivaMLException: If execution_rid is not valid or doesn't
                refer to an Execution record.

        Example:
            >>> record = ml.lookup_execution("1-abc123")  # doctest: +SKIP
            >>> record.status = ExecutionStatus.Uploaded   # writes to catalog
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution_record import ExecutionRecord

        # Get execution record from catalog and verify it's an Execution
        resolved = self.resolve_rid(execution_rid)
        if resolved.table.name != "Execution":
            raise DerivaMLException(
                f"RID '{execution_rid}' refers to a {resolved.table.name}, not an Execution"
            )

        execution_data = self._retrieve_rid(execution_rid)

        # Parse timestamps if present
        start_time = None
        stop_time = None
        if execution_data.get("Start"):
            from datetime import datetime
            try:
                start_time = datetime.fromisoformat(execution_data["Start"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        if execution_data.get("Stop"):
            from datetime import datetime
            try:
                stop_time = datetime.fromisoformat(execution_data["Stop"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Look up the workflow if present
        workflow_rid = execution_data.get("Workflow")
        workflow = self.lookup_workflow(workflow_rid) if workflow_rid else None

        # Create ExecutionRecord bound to this catalog
        record = ExecutionRecord(
            execution_rid=execution_rid,
            workflow=workflow,
            status=ExecutionStatus(execution_data.get("Status") or "Created"),
            description=execution_data.get("Description"),
            start_time=start_time,
            stop_time=stop_time,
            duration=execution_data.get("Duration"),
            _ml_instance=self,
            _logger=getattr(self, "_logger", None),
        )

        return record

    def list_executions(
        self,
        *,
        status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
        workflow_rid: str | None = None,
        mode: "ConnectionMode | None" = None,
        since: datetime | None = None,
    ) -> list[ExecutionSnapshot]:
        """Enumerate locally-known executions from the SQLite registry.

        Reads from the workspace SQLite registry — **no server contact**.
        Works in both online and offline mode. Each returned
        ``ExecutionSnapshot`` is a frozen Pydantic value object captured
        at query time; it cannot mutate the catalog. Pending-row counts
        are included in the same pass.

        For live catalog queries that return mutable
        :class:`~deriva_ml.execution.execution_record.ExecutionRecord`
        objects bound to the catalog, see ``find_executions()`` and
        ``lookup_execution()``.

        Args:
            status: Single ExecutionStatus or list to filter; None = all.
            workflow_rid: Match only executions tagged with this Workflow
                RID; None = all.
            mode: ConnectionMode the execution was last active under;
                None = all.
            since: Return only executions with last_activity >= this
                timestamp (timezone-aware). None = no time filter.

        Returns:
            List of ``ExecutionSnapshot`` Pydantic models — one per matching
            row in the registry. Empty list if nothing matches.

        Example:
            >>> from deriva_ml.execution.state_store import ExecutionStatus  # doctest: +SKIP
            >>> failed = ml.list_executions(status=ExecutionStatus.Failed)
            >>> for snap in failed:
            ...     print(snap.rid, snap.error)
        """
        store = self.workspace.execution_state_store()
        rows = store.list_executions(
            status=status, workflow_rid=workflow_rid,
            mode=mode, since=since,
        )
        return [
            ExecutionSnapshot.from_row(
                row, **store.count_pending_by_kind(execution_rid=row["rid"])
            )
            for row in rows
        ]

    def pending_summary(self) -> "WorkspacePendingSummary":
        """Workspace-wide pending-upload summary.

        Queries every known-local execution and returns a
        WorkspacePendingSummary aggregating per-execution snapshots.
        Useful for standalone uploader processes that want to know
        what's pending across runs.

        Returns:
            WorkspacePendingSummary with one PendingSummary per execution
            that has at least one registry row.

        Example:
            >>> print(ml.pending_summary().render())  # doctest: +SKIP
        """
        from deriva_ml.execution.pending_summary import WorkspacePendingSummary

        summaries = []
        for rec in self.list_executions():
            summaries.append(rec.pending_summary(ml=self))
        return WorkspacePendingSummary(per_execution=summaries)

    def find_incomplete_executions(self) -> list[ExecutionSnapshot]:
        """Sugar over :meth:`list_executions` for everything not terminally done.

        Reads from the workspace SQLite registry — no server contact.
        Returns executions in status in (Created, Running, Stopped, Failed,
        Pending_Upload) — the set of things a user would want to either
        resume, retry, or clean up. Excludes Uploaded (terminal success)
        and Aborted (terminal cleanup).

        For live catalog queries returning mutable
        :class:`~deriva_ml.execution.execution_record.ExecutionRecord`
        objects, see ``find_executions(status=...)``.

        Returns:
            List of ``ExecutionSnapshot`` Pydantic models for each incomplete
            execution known to the local registry.

        Example:
            >>> for snap in ml.find_incomplete_executions():  # doctest: +SKIP
            ...     print(snap.rid, snap.status, snap.pending_rows)
        """
        return self.list_executions(
            status=[
                ExecutionStatus.Created,
                ExecutionStatus.Running,
                ExecutionStatus.Stopped,
                ExecutionStatus.Failed,
                ExecutionStatus.Pending_Upload,
            ],
        )

    def resume_execution(self, execution_rid: RID) -> "Execution":
        """Re-hydrate an Execution from the workspace SQLite registry.

        Works in both online and offline modes. The execution's recorded
        mode is independent of the current DerivaML instance's mode — a
        user can create an execution online, run it offline, then upload
        online, all via the same RID.

        Before returning, runs just-in-time state reconciliation
        (spec §2.2): if online and sync_pending=True, flushes SQLite to
        the catalog; then checks for catalog/SQLite disagreement and
        applies the disagreement rules.

        Args:
            execution_rid: Server-assigned Execution RID returned by a
                prior create_execution call.

        Returns:
            An Execution object bound to this DerivaML instance, with
            lifecycle fields as SQLite read-through properties (see
            spec §2.3).

        Raises:
            DerivaMLException: If no matching executions row exists in
                the workspace registry.
            DerivaMLStateInconsistency: If just-in-time reconciliation
                surfaces a disagreement outside the six documented cases
                (see state_machine.reconcile_with_catalog).

        Example:
            >>> ml = DerivaML(hostname="example.org", catalog_id="42")  # doctest: +SKIP
            >>> exe = ml.resume_execution("5-ABC")
            >>> exe.status
            <ExecutionStatus.Stopped>
            >>> exe.upload_outputs()
        """
        from deriva_ml.execution.execution import Execution

        store = self.workspace.execution_state_store()
        row = store.get_execution(execution_rid)
        if row is None:
            raise DerivaMLException(
                f"Execution {execution_rid} is not in the workspace registry. "
                f"Either it was never created on this workspace, or it was "
                f"garbage-collected. Use ml.list_executions() to see what's "
                f"available locally."
            )

        # Just-in-time reconciliation. Online only — offline mode has no
        # catalog to compare against.
        if self._mode is ConnectionMode.online:
            # Order matters: flush first (push our newer state) before
            # reconcile (which would otherwise see stale catalog state
            # as a disagreement). See spec §4.6 step 3.
            if row["sync_pending"]:
                flush_pending_sync(
                    store=store, catalog=self.catalog,
                    execution_rid=execution_rid,
                )
            reconcile_with_catalog(
                store=store, catalog=self.catalog,
                execution_rid=execution_rid,
            )

            # Scoped reconciliation. Cheaper than the workspace-wide sweep
            # we did at DerivaML init, and guarantees this specific
            # execution's pending rows are consistent before we hand out
            # the Execution object.
            from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases

            try:
                reconcile_pending_leases(
                    store=store, catalog=self.catalog,
                    execution_rid=execution_rid,
                )
            except Exception as exc:
                logging.getLogger("deriva_ml").warning(
                    "per-execution lease reconciliation failed for %s (%s); continuing",
                    execution_rid, exc,
                )

        # Construct Execution bound to this DerivaML — it reads lifecycle
        # fields from SQLite via read-through properties (Group E).
        return Execution._from_registry(
            ml_object=self, execution_rid=execution_rid,
        )

    def gc_executions(
        self,
        *,
        older_than: "timedelta | None" = None,
        status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
        delete_working_dir: bool = False,
    ) -> int:
        """Garbage-collect execution registry rows matching the filters.

        By default only removes registry state (SQLite rows and their
        pending_rows / directory_rules). Pass delete_working_dir=True to
        also ``rm -rf`` the on-disk execution root under the workspace.

        Does NOT touch the catalog. Executions uploaded to the catalog
        remain there regardless of local gc.

        Args:
            older_than: If set, only gc executions whose last_activity is
                older than this timedelta.
            status: Filter by status (single or list); None = any status.
                Typical: pass ExecutionStatus.Uploaded to clean up after
                successful uploads.
            delete_working_dir: If True, remove the per-execution working
                directory from disk. Defaults to False (registry-only).

        Returns:
            The number of executions removed.

        Example:
            >>> from datetime import timedelta  # doctest: +SKIP
            >>> from deriva_ml.execution.state_store import ExecutionStatus
            >>> n = ml.gc_executions(
            ...     status=ExecutionStatus.Uploaded,
            ...     older_than=timedelta(days=30),
            ...     delete_working_dir=True,
            ... )
            >>> print(f"cleaned {n} old executions")
        """
        import shutil
        from datetime import datetime, timezone
        from pathlib import Path

        store = self.workspace.execution_state_store()

        # Pull the filtered row list from SQLite, then narrow by
        # last_activity if older_than was provided.
        rows = store.list_executions(status=status)
        if older_than is not None:
            cutoff = datetime.now(timezone.utc) - older_than
            # SQLite's DateTime(timezone=True) stores as ISO text and
            # returns naive datetimes; coerce both sides to naive UTC
            # to avoid offset-aware/naive comparison errors.

            def _is_older(last_activity: datetime) -> bool:
                la = last_activity
                if la.tzinfo is None:
                    la = la.replace(tzinfo=timezone.utc)
                return la < cutoff

            rows = [r for r in rows if _is_older(r["last_activity"])]

        for row in rows:
            if delete_working_dir:
                wd = Path(self.working_dir) / row["working_dir_rel"]
                if wd.exists():
                    shutil.rmtree(wd)
            store.delete_execution(row["rid"])

        return len(rows)

    def find_executions(
        self,
        workflow: "Workflow | RID | None" = None,
        workflow_type: str | None = None,
        status: ExecutionStatus | None = None,
    ) -> Iterable["ExecutionRecord"]:
        """Search the live catalog for executions matching the given filters.

        Queries the ERMrest catalog (online only) and yields live,
        catalog-bound ``ExecutionRecord`` objects for each match. Each
        returned record's mutable properties (``status``, ``description``)
        write through to the catalog on assignment.

        For enumerating locally-known executions from the SQLite registry
        without touching the catalog (works in offline mode), see
        ``list_executions()`` and ``find_incomplete_executions()``.

        Args:
            workflow: Optional Workflow or RID to filter by.
            workflow_type: Optional workflow type name (e.g., "python_script").
            status: Optional ExecutionStatus to filter by.

        Returns:
            Iterable of live ``ExecutionRecord`` objects.

        Example:
            >>> for record in ml.find_executions(status=ExecutionStatus.Uploaded):  # doctest: +SKIP
            ...     print(record.execution_rid, record.status)
        """
        # Import for type checking
        from deriva_ml.execution.workflow import Workflow as WorkflowClass

        # Get datapath to the Execution table
        pb = self.pathBuilder()
        execution_path = pb.schemas[self.ml_schema].Execution

        # Apply filters
        filtered_path = execution_path

        # Filter by specific workflow
        if workflow:
            workflow_rid = workflow.rid if isinstance(workflow, WorkflowClass) else workflow
            filtered_path = filtered_path.filter(execution_path.Workflow == workflow_rid)

        # Filter by workflow type - find workflows with matching type, then filter executions
        matching_workflow_rids: set[str] | None = None
        if workflow_type:
            wt_assoc = pb.schemas[self.ml_schema].Workflow_Workflow_Type
            matching_workflow_rids = {
                row["Workflow"]
                for row in wt_assoc.filter(wt_assoc.Workflow_Type == workflow_type).entities().fetch()
            }
            if not matching_workflow_rids:
                return  # No workflows match this type, so no executions

        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        # Create ExecutionRecord objects
        for exec_record in filtered_path.entities().fetch():
            # If filtering by workflow type, check the execution's workflow is in the matching set
            if matching_workflow_rids is not None and exec_record.get("Workflow") not in matching_workflow_rids:
                continue
            yield self.lookup_execution(exec_record["RID"])

    def lookup_experiment(self, execution_rid: RID) -> "Experiment":
        """Look up an experiment by execution RID.

        Creates an Experiment object for analyzing completed executions.
        Provides convenient access to execution metadata, configuration choices,
        model parameters, inputs, and outputs.

        Args:
            execution_rid: Resource Identifier (RID) of the execution.

        Returns:
            Experiment: An experiment object for the given execution RID.

        Example:
            >>> exp = ml.lookup_experiment("47BE")  # doctest: +SKIP
            >>> print(exp.name)  # e.g., "cifar10_quick"
            >>> print(exp.config_choices)  # Hydra config names used
            >>> print(exp.model_config)  # Model hyperparameters
        """
        from deriva_ml.experiment import Experiment

        return Experiment(self, execution_rid)  # type: ignore[arg-type]

    def find_experiments(
        self,
        workflow_rid: RID | None = None,
        status: ExecutionStatus | None = None,
    ) -> Iterable["Experiment"]:
        """List all experiments (executions with Hydra configuration) in the catalog.

        Creates Experiment objects for analyzing completed ML model runs.
        Only returns executions that have Hydra configuration metadata
        (i.e., a config.yaml file in Execution_Metadata assets).

        Args:
            workflow_rid: Optional workflow RID to filter by.
            status: Optional status to filter by (e.g., ExecutionStatus.Uploaded).

        Returns:
            Iterable of Experiment objects for executions with Hydra config.

        Example:
            >>> experiments = list(ml.find_experiments(status=ExecutionStatus.Uploaded))  # doctest: +SKIP
            >>> for exp in experiments:
            ...     print(f"{exp.name}: {exp.config_choices}")
        """
        import re

        from deriva_ml.experiment import Experiment

        # Get datapath to tables
        pb = self.pathBuilder()
        execution_path = pb.schemas[self.ml_schema].Execution
        metadata_path = pb.schemas[self.ml_schema].Execution_Metadata
        meta_exec_path = pb.schemas[self.ml_schema].Execution_Metadata_Execution

        # Find executions that have metadata assets with config.yaml files
        # Query the association table to find executions with hydra config metadata
        exec_rids_with_config = set()

        # Get all metadata records and filter for config.yaml files in Python
        # (ERMrest regex support varies by deployment)
        config_pattern = re.compile(r".*-config\.yaml$")
        config_metadata_rids = set()
        for meta in metadata_path.entities().fetch():
            filename = meta.get("Filename", "")
            if filename and config_pattern.match(filename):
                config_metadata_rids.add(meta["RID"])

        if config_metadata_rids:
            # Query the association table to find which executions have these metadata
            for assoc_record in meta_exec_path.entities().fetch():
                if assoc_record.get("Execution_Metadata") in config_metadata_rids:
                    exec_rids_with_config.add(assoc_record["Execution"])

        # Apply additional filters and yield Experiment objects
        filtered_path = execution_path
        if workflow_rid:
            filtered_path = filtered_path.filter(execution_path.Workflow == workflow_rid)
        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        for exec_record in filtered_path.entities().fetch():
            if exec_record["RID"] in exec_rids_with_config:
                yield Experiment(self, exec_record["RID"])  # type: ignore[arg-type]

    def upload_pending(
        self,
        *,
        execution_rids: "list[RID] | None" = None,
        retry_failed: bool = False,
    ) -> "UploadReport":
        """Blocking upload of pending state for selected executions.

        Flushes all pending rows (catalog inserts, asset uploads) for the
        named executions to the live catalog. Blocks until complete. For a
        non-blocking version that returns a job handle, use
        :meth:`_start_upload`. Online mode only.

        Args:
            execution_rids: List of RIDs, or None to drain every execution
                that has pending work.
            retry_failed: Include rows in status='failed'.

        Returns:
            UploadReport with totals + per-table counts + error lines.

        Example:
            >>> report = ml.upload_pending()  # doctest: +SKIP
            >>> print(f"{report.total_uploaded} uploaded, "
            ...       f"{report.total_failed} failed")
        """
        return run_upload_engine(
            ml=self,
            execution_rids=execution_rids,
            retry_failed=retry_failed,
        )

    def _start_upload(
        self,
        *,
        execution_rids: "list[RID] | None" = None,
        retry_failed: bool = False,
    ) -> "UploadJob":
        """Non-blocking upload — returns an UploadJob to poll / wait.

        Spawns a daemon thread in the current process. If the process
        exits, the thread dies. For survive-process uploads, run
        ``deriva-ml upload`` from a shell (see CLI, Group H).

        Args: identical to upload_pending.

        Returns:
            An UploadJob; call job.wait() to block, job.progress() to
            poll, job.cancel() to stop.

        Example:
            >>> job = ml._start_upload()  # doctest: +SKIP
            >>> while job.status == "running":
            ...     time.sleep(5)
            ...     print(job.progress())
            >>> report = job.wait()
        """
        from deriva_ml.execution.upload_job import UploadJob
        return UploadJob(
            ml=self,
            execution_rids=execution_rids,
            retry_failed=retry_failed,
        )

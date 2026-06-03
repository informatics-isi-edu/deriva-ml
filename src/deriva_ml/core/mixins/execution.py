"""Execution management mixin for DerivaML.

This module provides the ExecutionMixin class which handles
execution operations including creating, resuming, and updating
execution status.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException, NoAssociationException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.sort import SortSpec, resolve_sort
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
from deriva_ml.execution.state_machine import (
    flush_pending_sync,
    reconcile_with_catalog,
)
from deriva_ml.execution.state_store import ExecutionStatus

logger = get_logger(__name__)

if TYPE_CHECKING:
    from deriva_ml.asset.aux_classes import AssetSpec
    from deriva_ml.dataset.aux_classes import DatasetSpec
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.execution_record import ExecutionRecord
    from deriva_ml.execution.lineage import (
        LineageNode,
        LineageResult,
        RootDescriptor,
        WorkflowSummary,
    )
    from deriva_ml.execution.pending_summary import WorkspacePendingSummary
    from deriva_ml.execution.upload_report import UploadReport
    from deriva_ml.execution.workflow import Workflow
    from deriva_ml.experiment.experiment import Experiment
    from deriva_ml.model.catalog import DerivaModel


__all__ = ["ExecutionMixin"]


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
                >>> with ml.create_execution(config) as execution:  # doctest: +SKIP
                ...     # Run analysis
                ...     pass
                >>> execution.commit_output_assets()  # doctest: +SKIP

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
        kwargs_form_only = any(x is not None for x in (datasets, assets, description))
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
                workflow=resolved_workflow if isinstance(resolved_workflow, WorkflowClass) else None,
                description=description or "",
            )
            # If workflow is a RID (not a Workflow or string URL),
            # pass it through the legacy workflow= parameter so
            # Execution.__init__ can raise its own clear error
            # (our job is just assembly, not re-validation).
            workflow_for_execution = resolved_workflow if not isinstance(resolved_workflow, WorkflowClass) else None
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
            >>> record.status = ExecutionStatus.Uploaded   # writes to catalog  # doctest: +SKIP
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution_record import ExecutionRecord

        # Get execution record from catalog and verify it's an Execution
        resolved = self.resolve_rid(execution_rid)
        if resolved.table.name != "Execution":
            raise DerivaMLException(f"RID '{execution_rid}' refers to a {resolved.table.name}, not an Execution")

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

        # Create ExecutionRecord bound to this catalog. Reads the three
        # per-phase duration columns added 2026-05-19; old catalogs that
        # predate the schema bump report None for all three (forward-only
        # migration — see docs/bugs/2026-05-19-execution-phase-durations-design.md).
        record = ExecutionRecord(
            execution_rid=execution_rid,
            workflow=workflow,
            status=ExecutionStatus(execution_data.get("Status") or "Created"),
            description=execution_data.get("Description"),
            start_time=start_time,
            stop_time=stop_time,
            duration=execution_data.get("Execution_Duration"),
            download_duration=execution_data.get("Download_Duration"),
            upload_duration=execution_data.get("Upload_Duration"),
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
            >>> failed = ml.list_executions(status=ExecutionStatus.Failed)  # doctest: +SKIP
            >>> for snap in failed:  # doctest: +SKIP
            ...     print(snap.rid, snap.error)
        """
        store = self.workspace.execution_state_store()
        rows = store.list_executions(
            status=status,
            workflow_rid=workflow_rid,
            mode=mode,
            since=since,
        )
        return [
            ExecutionSnapshot.from_row(row, **store.count_pending_by_kind(execution_rid=row["rid"])) for row in rows
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
            >>> exe = ml.resume_execution("5-ABC")  # doctest: +SKIP
            >>> exe.status  # doctest: +SKIP
            <ExecutionStatus.Stopped>
            >>> exe.commit_output_assets()  # doctest: +SKIP
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
                    store=store,
                    catalog=self.catalog,
                    execution_rid=execution_rid,
                )
            reconcile_with_catalog(
                store=store,
                catalog=self.catalog,
                execution_rid=execution_rid,
            )

        # Construct Execution bound to this DerivaML — it reads lifecycle
        # fields from SQLite via read-through properties (Group E).
        return Execution.from_registry(
            ml_object=self,
            execution_rid=execution_rid,
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
            >>> from deriva_ml.execution.state_store import ExecutionStatus  # doctest: +SKIP
            >>> n = ml.gc_executions(  # doctest: +SKIP
            ...     status=ExecutionStatus.Uploaded,
            ...     older_than=timedelta(days=30),
            ...     delete_working_dir=True,
            ... )
            >>> print(f"cleaned {n} old executions")  # doctest: +SKIP
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
        sort: SortSpec = None,
        dataset: "RID | DatasetSpec | None" = None,
        dataset_role: str = "any",
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
            sort: Optional sort spec.
                - ``None`` (default): backend-determined order (no sort
                  clause applied; cheapest path).
                - ``True``: newest-first by record creation time
                  (``RCT desc``). Recommended for "show me the most
                  recent executions" queries.
                - Callable ``(path) -> sort_keys``: receives the
                  Execution table path and returns one or more
                  path-builder sort keys (e.g. ``path.RCT.desc``,
                  or ``[path.Status, path.RCT.desc]``).
            dataset: Optional dataset filter. A dataset RID (``str``)
                or a :class:`DatasetSpec`. When a ``DatasetSpec`` is
                given, its ``version`` pins the filter to executions
                that touched that specific dataset version. Only
                executions with an edge to this dataset (per
                ``dataset_role``) are yielded.
            dataset_role: Which dataset edge to match when ``dataset``
                is given. One of:

                - ``"input"``: executions that *consumed* the dataset
                  (``Dataset_Execution`` rows).
                - ``"output"``: executions that *produced* the dataset
                  (``Dataset_Version.Execution`` authorship).
                - ``"any"`` (default): union of input and output.

                Authorship-canonical model: output edges live only in
                ``Dataset_Version.Execution``; input edges live in
                ``Dataset_Execution``. Raises ``ValueError`` if a
                non-``"any"`` role is given without a ``dataset``.

        Returns:
            Iterable of live ``ExecutionRecord`` objects.

        Example:
            >>> for record in ml.find_executions(status=ExecutionStatus.Uploaded):  # doctest: +SKIP
            ...     print(record.execution_rid, record.status)

            Newest-first (most common):

            >>> for record in ml.find_executions(sort=True):  # doctest: +SKIP
            ...     pass

            Custom sort -- group by status, then newest within group:

            >>> for record in ml.find_executions(  # doctest: +SKIP
            ...     sort=lambda path: [path.Status, path.RCT.desc],
            ... ):
            ...     pass
        """
        # Import for type checking
        from deriva_ml.dataset.aux_classes import DatasetSpec
        from deriva_ml.execution.workflow import Workflow as WorkflowClass

        # Get datapath to the Execution table
        pb = self.pathBuilder()
        execution_path = pb.schemas[self.ml_schema].Execution

        # Build the allowed-execution-RID set from the dataset filter.
        # None means "no dataset filter applied" (don't intersect).
        allowed_exec_rids: set[str] | None = None
        if dataset is None:
            if dataset_role != "any":
                raise ValueError("dataset_role requires a dataset argument")
        else:
            if isinstance(dataset, DatasetSpec):
                ds_rid, ds_version = dataset.rid, str(dataset.version)
            else:
                ds_rid, ds_version = dataset, None

            # Resolve a version pin once to a Dataset_Version RID so the
            # input-edge filter is a direct RID comparison (no per-row
            # _version_label query).
            pinned_version_rid = self._version_rid(ds_rid, ds_version) if ds_version is not None else None

            input_rids: set[str] = set()
            if dataset_role in ("input", "any"):
                ds_exec = pb.schemas[self.ml_schema].Dataset_Execution
                for row in ds_exec.filter(ds_exec.Dataset == ds_rid).entities().fetch():
                    if ds_version is not None and row.get("Dataset_Version") != pinned_version_rid:
                        continue
                    if row.get("Execution"):
                        input_rids.add(row["Execution"])

            output_rids: set[str] = set()
            if dataset_role in ("output", "any"):
                vp = pb.schemas[self.ml_schema].tables["Dataset_Version"]
                for row in vp.filter(vp.Dataset == ds_rid).entities().fetch():
                    if not row.get("Execution"):
                        continue
                    if ds_version is not None and (row.get("Version") or "") != ds_version:
                        continue
                    output_rids.add(row["Execution"])

            if dataset_role == "input":
                allowed_exec_rids = input_rids
            elif dataset_role == "output":
                allowed_exec_rids = output_rids
            else:  # any
                allowed_exec_rids = input_rids | output_rids

        # Apply filters
        filtered_path = execution_path

        # Filter by specific workflow
        if workflow:
            workflow_rid = workflow.workflow_rid if isinstance(workflow, WorkflowClass) else workflow
            filtered_path = filtered_path.filter(execution_path.Workflow == workflow_rid)

        # Filter by workflow type - find workflows with matching type, then filter executions
        matching_workflow_rids: set[str] | None = None
        if workflow_type:
            wt_assoc = pb.schemas[self.ml_schema].Workflow_Workflow_Type
            matching_workflow_rids = {
                row["Workflow"] for row in wt_assoc.filter(wt_assoc.Workflow_Type == workflow_type).entities().fetch()
            }
            if not matching_workflow_rids:
                return  # No workflows match this type, so no executions

        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        # Resolve sort spec against this method's default (newest-first
        # by record creation time). resolve_sort returns None when the
        # caller explicitly opted out of sorting (sort=None), in which
        # case we don't call .sort() at all -- backend default order.
        entity_set = filtered_path.entities()
        sort_keys = resolve_sort(sort, lambda p: p.RCT.desc, execution_path)
        if sort_keys is not None:
            entity_set = entity_set.sort(*sort_keys)

        # Create ExecutionRecord objects
        for exec_record in entity_set.fetch():
            # If filtering by workflow type, check the execution's workflow is in the matching set
            if matching_workflow_rids is not None and exec_record.get("Workflow") not in matching_workflow_rids:
                continue
            # If filtering by dataset, skip executions outside the allowed set
            if allowed_exec_rids is not None and exec_record.get("RID") not in allowed_exec_rids:
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
            >>> print(exp.name)  # e.g., "cifar10_quick"  # doctest: +SKIP
            >>> print(exp.config_choices)  # Hydra config names used  # doctest: +SKIP
            >>> print(exp.model_config)  # Model hyperparameters  # doctest: +SKIP
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
            >>> for exp in experiments:  # doctest: +SKIP
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

    def commit_pending_executions(
        self,
        *,
        execution_rids: "list[RID] | None" = None,
        clean_folder: bool = False,
    ) -> "UploadReport":
        """Batch-commit pending output assets for one or more executions.

        ADR-0009's batch upload entry point. For each requested
        execution, resumes it from the workspace registry and calls
        :meth:`Execution.commit_output_assets`, which brackets the
        bag-commit with the full lifecycle (``Pending_Upload →
        Uploaded`` transition, ``Upload_Duration`` recording, asset
        description writes, optional working-folder cleanup).

        Failure isolation is per-execution: an exception while
        committing execution A does not skip execution B; both
        outcomes appear in the returned :class:`UploadReport`. The
        blocking call returns even when one or more executions
        failed; callers check ``report.total_failed`` and
        ``report.errors`` for diagnosis.

        This is the engine behind the ``deriva-ml-upload`` CLI.
        Online mode only.

        Args:
            execution_rids: List of RIDs, or ``None`` to drain every
                execution that has pending work in the workspace
                registry. An empty list is treated as "drain nothing"
                and returns an empty report.
            clean_folder: Forwarded to
                :meth:`Execution.commit_output_assets`. When ``True``,
                each execution's working folder is removed after a
                successful commit. Default ``False`` preserves on-disk
                state for inspection.

        Returns:
            UploadReport aggregating per-execution outcomes. Successful
            executions contribute their per-(schema, table) counts to
            ``per_table`` and their asset-row count to
            ``total_uploaded``. Failed executions contribute one entry
            to ``total_failed`` and one human-readable line to
            ``errors`` prefixed by ``"execution {rid}: "``.

        Example:
            >>> report = ml.commit_pending_executions()  # doctest: +SKIP
            >>> print(f"{report.total_uploaded} uploaded, "  # doctest: +SKIP
            ...       f"{report.total_failed} failed")
        """
        from deriva_ml.execution.upload_report import UploadReport

        # Enumerate executions to drain. Caller-supplied list wins; an
        # empty caller-supplied list is treated as "drain nothing." A
        # ``None`` caller-supplied list means "drain every execution
        # in the workspace registry."
        store = self.workspace.execution_state_store()
        if execution_rids is None:
            rids = [row["rid"] for row in store.list_executions()]
        else:
            rids = list(execution_rids)

        total_uploaded = 0
        total_failed = 0
        per_table: dict[str, dict[str, int]] = {}
        errors: list[str] = []

        for rid in rids:
            try:
                execution = self.resume_execution(rid)  # type: ignore[attr-defined]
                exec_report = execution.commit_output_assets(clean_folder=clean_folder)
            except Exception as e:  # noqa: BLE001 — surface every failure into the report
                # Failure isolation: continue past this execution so
                # sibling executions still get a chance to drain.
                total_failed += 1
                errors.append(f"execution {rid}: {e}")
                continue

            # Aggregate the per-execution UploadReport into the batch
            # totals. The per-table dict gets summed across executions
            # so the caller sees one rolled-up view per asset table.
            total_uploaded += exec_report.total_uploaded
            for fqn, counts in exec_report.per_table.items():
                bucket = per_table.setdefault(fqn, {"uploaded": 0, "failed": 0})
                bucket["uploaded"] += counts.get("uploaded", 0)
                bucket["failed"] += counts.get("failed", 0)

        return UploadReport(
            execution_rids=rids,
            total_uploaded=total_uploaded,
            total_failed=total_failed,
            per_table=per_table,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Lineage walk
    # ------------------------------------------------------------------

    def lookup_lineage(
        self,
        rid: RID,
        *,
        depth: int | None = None,
        max_executions: int = 500,
    ) -> "LineageResult":
        """Walk the data-flow provenance chain behind an artifact.

        Given a Dataset, Asset, Feature value, or Execution RID,
        returns a tree of producing executions and their consumed
        inputs back to the natural root of every branch. Replaces
        what would otherwise be 5-15 client round-trips through
        typed read methods with one call.

        The walk follows **data-flow parents only**: for each
        execution node, the parents are the producing executions of
        its consumed datasets and assets (asset_role="Input"). This
        method explicitly does NOT walk ``Execution_Execution``
        (orchestration links) — that's a different question
        (``ExecutionRecord.list_execution_parents`` /
        ``list_execution_children``). See
        ``docs/adr/0001-lineage-walks-data-flow-not-orchestration.md``
        for the rationale.

        For Dataset roots, the producer is taken from the **current**
        version's ``Dataset_Version.Execution`` row. Walking a
        historical version is a future enhancement.

        Args:
            rid: RID of any Dataset, Asset, Feature value, or
                Execution. Workflow RIDs are not lineage-shaped and
                raise :class:`DerivaMLException`.
            depth: Number of parent levels to walk from the immediate
                producing execution. ``None`` (default) walks to the
                root. ``0`` returns only the immediate producing
                execution node. ``N>0`` walks ``N`` levels up.
            max_executions: Defensive cap on distinct executions the
                walk will expand. Default 500. If exceeded,
                ``walked_complete`` is set to False and the partial
                tree is returned.

        Returns:
            A :class:`~deriva_ml.execution.lineage.LineageResult`
            with the producing-execution tree plus transparency
            flags (``walked_complete``, ``cycle_detected``,
            ``depth_capped``, ``executions_visited``).

        Raises:
            DerivaMLException: If ``rid`` does not exist, points at a
                Workflow row, or points at a row whose table cannot
                be classified as Dataset / Asset / Feature value /
                Execution.

        Example:
            Trace an output asset back to its training dataset::

                >>> result = ml.lookup_lineage("3JSE")  # doctest: +SKIP
                >>> assert result.walked_complete  # doctest: +SKIP
                >>> for ds in result.lineage.consumed_datasets:  # doctest: +SKIP
                ...     print(ds.rid, ds.version)

            Just the immediate producer (one round-trip)::

                >>> result = ml.lookup_lineage(  # doctest: +SKIP
                ...     "3JSE", depth=0,
                ... )
                >>> producer = result.lineage.execution  # doctest: +SKIP

            For the orchestration view (which execution called
            which), use ``record.list_execution_parents()`` /
            ``list_execution_children()`` on an
            :class:`~deriva_ml.execution.execution_record.ExecutionRecord`.
        """
        from deriva_ml.execution.lineage import LineageResult

        # 1. Classify the root RID with a single resolve_rid call.
        root_descriptor, producer_rid = self._classify_rid(rid)

        if producer_rid is None:
            # No producer — return a valid result with an empty walk.
            return LineageResult(root=root_descriptor)

        # 2. Walk iteratively from the producing execution.
        visited_global: set[RID] = set()
        in_progress: set[RID] = set()
        flags = {"cycle_detected": False, "depth_capped": False, "walked_complete": True}

        lineage_root_node = self._walk_node(
            execution_rid=producer_rid,
            depth_remaining=depth,
            max_executions=max_executions,
            visited_global=visited_global,
            in_progress=in_progress,
            flags=flags,
        )

        # The producing-execution summary on the root descriptor matches
        # the top-most execution node we just expanded.
        if lineage_root_node is not None:
            root_descriptor = root_descriptor.model_copy(update={"producing_execution": lineage_root_node.execution})

        return LineageResult(
            root=root_descriptor,
            lineage=lineage_root_node,
            executions_visited=len(visited_global),
            walked_complete=flags["walked_complete"],
            cycle_detected=flags["cycle_detected"],
            depth_capped=flags["depth_capped"],
        )

    # -- private helpers -------------------------------------------------

    def _classify_rid(self, rid: RID) -> "tuple[RootDescriptor, RID | None]":
        """Classify ``rid`` and return ``(root_descriptor, producer_rid)``.

        ``producer_rid`` is the immediate producing-execution RID, or
        None if the artifact has no recorded producer. For an
        Execution root, it's the execution itself.

        Raises ``DerivaMLException`` for Workflow RIDs and for
        anything else not in the supported shape table.
        """
        from deriva_ml.execution.lineage import RootDescriptor

        resolved = self.resolve_rid(rid)
        table = resolved.table
        table_name = table.name

        if table_name == "Workflow":
            raise DerivaMLException(
                f"RID '{rid}' refers to a Workflow, which is not "
                f"lineage-shaped. Workflows describe what to run, "
                f"not a produced artifact."
            )

        if table_name == "Execution":
            row = self._retrieve_rid(rid)
            return (
                RootDescriptor(
                    rid=rid,
                    type="Execution",
                    description=row.get("Description"),
                    producing_execution=None,  # filled in by caller
                ),
                rid,
            )

        if table_name == "Dataset":
            row = self._retrieve_rid(rid)
            producer_rid = self._producer_of_dataset(rid)
            return (
                RootDescriptor(
                    rid=rid,
                    type="Dataset",
                    description=row.get("Description"),
                ),
                producer_rid,
            )

        if self.model.is_asset(table):
            row = self._retrieve_rid(rid)
            producer_rid = self._producer_of_asset(rid, table)
            return (
                RootDescriptor(
                    rid=rid,
                    type="Asset",
                    description=row.get("Description"),
                ),
                producer_rid,
            )

        # Feature value: a row in a feature association table has
        # both a "Feature_Name" column and an "Execution" column.
        col_names = {c.name for c in table.columns}
        if "Feature_Name" in col_names and "Execution" in col_names:
            row = self._retrieve_rid(rid)
            producer_rid = row.get("Execution")
            return (
                RootDescriptor(
                    rid=rid,
                    type="Feature",
                    description=None,
                ),
                producer_rid,
            )

        raise DerivaMLException(
            f"RID '{rid}' refers to a {table_name} row, which is not "
            f"lineage-shaped. lookup_lineage accepts Dataset, Asset, "
            f"Feature-value, or Execution RIDs."
        )

    def _producer_of_dataset(self, dataset_rid: RID) -> RID | None:
        """Return the Execution RID that produced the current version of ``dataset_rid``.

        Returns None if the dataset has no Dataset_Version rows yet
        or no version row carries an Execution link.
        """
        pb = self.pathBuilder()
        version_path = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        rows = list(version_path.filter(version_path.Dataset == dataset_rid).entities().fetch())
        if not rows:
            return None

        # Pick the row with the highest semver-style Version. The catalog
        # stores Version as text (e.g. "0.1.0"); sort lexically as a
        # tuple of ints so "1.10.0" beats "1.2.0".
        def _key(row: dict[str, Any]) -> tuple[int, ...]:
            v = row.get("Version") or "0.0.0"
            try:
                return tuple(int(p) for p in v.split("."))
            except ValueError:
                return (0,)

        latest = max(rows, key=_key)
        return latest.get("Execution")

    def _version_rid(self, dataset_rid: RID, version: Any) -> RID | None:
        """RID of the ``Dataset_Version`` row for (``dataset_rid``, ``version``), or None.

        Maps a (dataset, version) pair to the RID of its ``Dataset_Version``
        row so the consumed version can be recorded on the
        ``Dataset_Execution.Dataset_Version`` FK (the input edge). ``version``
        may be a :class:`DatasetVersion`, a version string, or anything whose
        ``str(...)`` matches the catalog's stored ``Version`` text (e.g.
        ``"1.0.0"``). Returns None when no matching version row exists.
        """
        pb = self.pathBuilder()
        vp = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        want = str(version)
        for row in vp.filter(vp.Dataset == dataset_rid).entities().fetch():
            if (row.get("Version") or "") == want:
                return row["RID"]
        return None

    def _version_label(self, version_rid) -> str | None:
        """Map a ``Dataset_Version`` RID to its ``Version`` string, or None.

        Inverse of :meth:`_version_rid`. Used by :meth:`find_executions`
        to resolve the ``Dataset_Execution.Dataset_Version`` FK on an
        input edge back to its semantic-version label for comparison
        against a caller's version pin. Returns None when ``version_rid``
        is falsy or no matching version row exists.
        """
        if not version_rid:
            return None
        pb = self.pathBuilder()
        vp = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        rows = list(vp.filter(vp.RID == version_rid).entities().fetch())
        return rows[0].get("Version") if rows else None

    def _producer_of_asset(self, asset_rid: RID, asset_table: Any) -> RID | None:
        """Return the Execution RID that produced ``asset_rid`` (asset_role="Output").

        Returns None if the asset has no Output association in any
        ``<AssetTable>_Execution`` row.
        """
        try:
            assoc_table, asset_fk, _exec_fk = self.model.find_association(asset_table, "Execution")
        except NoAssociationException:
            # Asset table has no <AssetTable>_Execution tracking — legitimate
            # case for catalogs that don't track execution provenance per asset.
            return None

        pb = self.pathBuilder()
        assoc_path = pb.schemas[assoc_table.schema.name].tables[assoc_table.name]
        rows = list(
            assoc_path.filter(assoc_path.columns[asset_fk] == asset_rid)
            .filter(assoc_path.Asset_Role == "Output")
            .entities()
            .fetch()
        )
        if not rows:
            return None
        # If multiple Output associations exist (rare), the first one
        # is fine — they all point at executions that wrote this asset.
        return rows[0].get("Execution")

    def _walk_node(
        self,
        *,
        execution_rid: RID,
        depth_remaining: int | None,
        max_executions: int,
        visited_global: set[RID],
        in_progress: set[RID],
        flags: dict[str, bool],
    ) -> "LineageNode | None":
        """Expand one execution node and recurse on its data-flow parents.

        Mutates ``visited_global``, ``in_progress``, and ``flags``.
        Returns None only if the execution couldn't be looked up
        (defensive).
        """
        from deriva_ml.execution.lineage import (
            AssetSummary,
            DatasetSummary,
            ExecutionSummary,
            LineageNode,
            WorkflowSummary,
        )

        # Cycle on the active path: do not expand, set flag, return a
        # leaf-style marker.
        if execution_rid in in_progress:
            flags["cycle_detected"] = True
            return LineageNode(
                execution=ExecutionSummary(
                    rid=execution_rid,
                    description=None,
                    workflow=None,
                    status="Unknown",
                ),
                already_shown=True,
            )

        # Diamond DAG: this execution was already expanded somewhere
        # else in the tree. Mark and don't recurse.
        if execution_rid in visited_global:
            return LineageNode(
                execution=ExecutionSummary(
                    rid=execution_rid,
                    description=None,
                    workflow=None,
                    status="Unknown",
                ),
                already_shown=True,
            )

        # Defensive cap on total expansions.
        if len(visited_global) >= max_executions:
            flags["walked_complete"] = False
            return None

        # Look up the execution and its inputs.
        try:
            record = self.lookup_execution(execution_rid)
        except DerivaMLException:
            # An input pointed at an Execution that no longer exists;
            # treat as missing rather than failing the whole walk.
            return None

        visited_global.add(execution_rid)
        in_progress.add(execution_rid)

        try:
            wf_summary: "WorkflowSummary | None" = None
            if record.workflow is not None and record.workflow.workflow_rid is not None:
                wf_summary = WorkflowSummary(
                    rid=record.workflow.workflow_rid,
                    name=record.workflow.name,
                )

            execution_summary = ExecutionSummary(
                rid=execution_rid,
                description=record.description,
                workflow=wf_summary,
                status=record.status.value if record.status else "Unknown",
            )

            # Consumed inputs.
            consumed_datasets: list[DatasetSummary] = []
            parent_rids: set[RID] = set()
            for ds in record.list_input_datasets():
                ds_version = None
                try:
                    ds_version = str(ds.current_version)
                except Exception:
                    pass
                consumed_datasets.append(
                    DatasetSummary(
                        rid=ds.dataset_rid,
                        description=ds.description or None,
                        version=ds_version,
                    )
                )
                producer = self._producer_of_dataset(ds.dataset_rid)
                if producer:
                    parent_rids.add(producer)

            consumed_assets: list[AssetSummary] = []
            for asset in record.list_assets(asset_role="Input"):
                consumed_assets.append(
                    AssetSummary(
                        rid=asset.asset_rid,
                        filename=asset.filename or None,
                        asset_table=asset.asset_table,
                    )
                )
                try:
                    asset_table_obj = self.model.name_to_table(asset.asset_table)
                    producer = self._producer_of_asset(asset.asset_rid, asset_table_obj)
                    if producer:
                        parent_rids.add(producer)
                except Exception:
                    # If we can't resolve the producer of one asset,
                    # keep walking the rest of the inputs.
                    pass

            # Recurse on parents.
            parents: list[LineageNode] = []
            if depth_remaining is None or depth_remaining > 0:
                next_depth = None if depth_remaining is None else depth_remaining - 1
                for pr in parent_rids:
                    child = self._walk_node(
                        execution_rid=pr,
                        depth_remaining=next_depth,
                        max_executions=max_executions,
                        visited_global=visited_global,
                        in_progress=in_progress,
                        flags=flags,
                    )
                    if child is not None:
                        parents.append(child)
            elif parent_rids:
                # We had parents but depth said stop. Mark depth_capped.
                flags["depth_capped"] = True

            return LineageNode(
                execution=execution_summary,
                consumed_datasets=consumed_datasets,
                consumed_assets=consumed_assets,
                parents=parents,
            )
        finally:
            in_progress.discard(execution_rid)

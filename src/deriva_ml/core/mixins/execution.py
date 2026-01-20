"""Execution management mixin for DerivaML.

This module provides the ExecutionMixin class which handles
execution operations including creating, restoring, and updating
execution status.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from deriva_ml.core.definitions import RID
from deriva_ml.core.enums import Status
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.upload import asset_file_path, execution_rids
from deriva_ml.execution.execution_configuration import ExecutionConfiguration

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.execution_record import ExecutionRecord
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
        - retrieve_rid(): method for retrieving RID data (from RidResolutionMixin)

    Methods:
        create_execution: Create a new execution environment
        restore_execution: Restore a previous execution
        _update_status: Update execution status in catalog
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    working_dir: Path
    status: str
    pathBuilder: Callable[[], Any]
    retrieve_rid: Callable[[RID], dict[str, Any]]
    _execution: "Execution"

    def _update_status(self, new_status: Status, status_detail: str, execution_rid: RID) -> None:
        """Update the status of an execution in the catalog.

        Args:
            new_status: New status.
            status_detail: Details of the status.
            execution_rid: Resource Identifier (RID) of the execution.
        """
        self.status = new_status.value
        self.pathBuilder().schemas[self.ml_schema].Execution.update(
            [
                {
                    "RID": execution_rid,
                    "Status": self.status,
                    "Status_Detail": status_detail,
                }
            ]
        )

    def create_execution(
        self, configuration: ExecutionConfiguration, workflow: "Workflow | RID | None" = None, dry_run: bool = False
    ) -> "Execution":
        """Create an execution environment.

        Initializes a local compute environment for executing an ML or analytic routine.
        This has several side effects:

        1. Downloads datasets specified in the configuration to the cache directory.
           If no version is specified, creates a new minor version for the dataset.
        2. Downloads any execution assets to the working directory.
        3. Creates an execution record in the catalog (unless dry_run=True).

        Args:
            configuration: ExecutionConfiguration specifying execution parameters.
            workflow: Optional Workflow object or RID if not present in configuration.
            dry_run: If True, skip creating catalog records and uploading results.

        Returns:
            Execution: An execution object for managing the execution lifecycle.

        Example:
            >>> config = ExecutionConfiguration(
            ...     workflow=workflow,
            ...     description="Process samples",
            ...     datasets=[DatasetSpec(rid="4HM")],
            ... )
            >>> with ml.create_execution(config) as execution:
            ...     # Run analysis
            ...     pass
            >>> execution.upload_execution_outputs()
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution import Execution

        # Create and store an execution instance
        self._execution = Execution(configuration, self, workflow=workflow, dry_run=dry_run)  # type: ignore[arg-type]
        return self._execution

    def lookup_execution(self, execution_rid: RID) -> "ExecutionRecord":
        """Look up an execution by RID and return an ExecutionRecord.

        Creates an ExecutionRecord object for querying and modifying execution
        metadata. The ExecutionRecord provides access to the catalog record
        state and allows updating mutable properties like status and description.

        For running computations with datasets and assets, use ``restore_execution()``
        or ``create_execution()`` which return full Execution objects.

        Args:
            execution_rid: Resource Identifier (RID) of the execution.

        Returns:
            ExecutionRecord: An execution record object bound to the catalog.

        Raises:
            DerivaMLException: If execution_rid is not valid or doesn't refer
                to an Execution record.

        Example:
            Look up an execution and query its state::

                >>> record = ml.lookup_execution("1-abc123")
                >>> print(f"Status: {record.status}")
                >>> print(f"Description: {record.description}")

            Update mutable properties::

                >>> record.status = Status.completed
                >>> record.description = "Analysis finished"

            Query relationships::

                >>> children = list(record.list_nested_executions())
                >>> parents = list(record.list_parent_executions())
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution_record import ExecutionRecord

        # Get execution record from catalog and verify it's an Execution
        resolved = self.resolve_rid(execution_rid)
        if resolved.table.name != "Execution":
            raise DerivaMLException(
                f"RID '{execution_rid}' refers to a {resolved.table.name}, not an Execution"
            )

        execution_data = self.retrieve_rid(execution_rid)

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
            status=Status(execution_data.get("Status", "Created")),
            description=execution_data.get("Description"),
            start_time=start_time,
            stop_time=stop_time,
            duration=execution_data.get("Duration"),
            _ml_instance=self,
            _logger=getattr(self, "_logger", None),
        )

        return record

    def restore_execution(self, execution_rid: RID | None = None) -> "Execution":
        """Restores a previous execution.

        Given an execution RID, retrieves the execution configuration and restores the local compute environment.
        This routine has a number of side effects.

        1. The datasets specified in the configuration are downloaded and placed in the cache-dir. If a version is
        not specified in the configuration, then a new minor version number is created for the dataset and downloaded.

        2. If any execution assets are provided in the configuration, they are downloaded and placed
        in the working directory.

        Args:
            execution_rid: Resource Identifier (RID) of the execution to restore.

        Returns:
            Execution: An execution object representing the restored execution environment.

        Raises:
            DerivaMLException: If execution_rid is not valid or execution cannot be restored.

        Example:
            >>> execution = ml.restore_execution("1-abc123")
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution import Execution

        # If no RID provided, try to find single execution in working directory
        if not execution_rid:
            e_rids = execution_rids(self.working_dir)
            if len(e_rids) != 1:
                raise DerivaMLException(f"Multiple execution RIDs were found {e_rids}.")
            execution_rid = e_rids[0]

        # Try to load configuration from a file
        cfile = asset_file_path(
            prefix=self.working_dir,
            exec_rid=execution_rid,
            file_name="configuration.json",
            asset_table=self.model.name_to_table("Execution_Metadata"),
            metadata={},
        )

        # Load configuration from a file or create from an execution record
        if cfile.exists():
            configuration = ExecutionConfiguration.load_configuration(cfile)
        else:
            execution = self.retrieve_rid(execution_rid)
            # Look up the workflow object from the RID
            workflow_rid = execution.get("Workflow")
            workflow = self.lookup_workflow(workflow_rid) if workflow_rid else None
            configuration = ExecutionConfiguration(
                workflow=workflow,
                description=execution["Description"],
            )

        # Create and return an execution instance
        return Execution(configuration, self, reload=execution_rid)  # type: ignore[arg-type]

    def find_executions(
        self,
        workflow: "Workflow | RID | None" = None,
        workflow_type: str | None = None,
        status: Status | None = None,
    ) -> Iterable["ExecutionRecord"]:
        """List all executions in the catalog.

        Returns ExecutionRecord objects for each execution. These provide access
        to execution metadata and allow updating mutable properties.

        Args:
            workflow: Optional Workflow object or RID to filter by.
            workflow_type: Optional workflow type name to filter by (e.g., "python_script").
                This filters by the Workflow_Type vocabulary term.
            status: Optional status to filter by (e.g., Status.completed).

        Returns:
            Iterable of ExecutionRecord objects.

        Example:
            List all executions::

                >>> for record in ml.find_executions():
                ...     print(f"{record.execution_rid}: {record.status}")

            Filter by status::

                >>> completed = list(ml.find_executions(status=Status.completed))

            Filter by specific workflow::

                >>> workflow = ml.lookup_workflow("2-ABC1")
                >>> for record in ml.find_executions(workflow=workflow):
                ...     print(f"{record.execution_rid}: {record.description}")

            Filter by workflow type (all notebooks)::

                >>> notebooks = list(ml.find_executions(workflow_type="python_notebook"))
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

        # Filter by workflow type - need to join with Workflow table
        if workflow_type:
            workflow_path = pb.schemas[self.ml_schema].Workflow
            # Link to workflows with matching type
            filtered_path = (
                filtered_path
                .link(workflow_path, on=(execution_path.Workflow == workflow_path.RID))
                .filter(workflow_path.Workflow_Type == workflow_type)
            )

        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        # Create ExecutionRecord objects
        for exec_record in filtered_path.entities().fetch():
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
            >>> exp = ml.lookup_experiment("47BE")
            >>> print(exp.name)  # e.g., "cifar10_quick"
            >>> print(exp.config_choices)  # Hydra config names used
            >>> print(exp.model_config)  # Model hyperparameters
        """
        from deriva_ml.experiment import Experiment

        return Experiment(self, execution_rid)  # type: ignore[arg-type]

    def find_experiments(
        self,
        workflow_rid: RID | None = None,
        status: Status | None = None,
    ) -> Iterable["Experiment"]:
        """List all experiments (executions with Hydra configuration) in the catalog.

        Creates Experiment objects for analyzing completed ML model runs.
        Only returns executions that have Hydra configuration metadata
        (i.e., a config.yaml file in Execution_Metadata assets).

        Args:
            workflow_rid: Optional workflow RID to filter by.
            status: Optional status to filter by (e.g., Status.Completed).

        Returns:
            Iterable of Experiment objects for executions with Hydra config.

        Example:
            >>> experiments = list(ml.find_experiments(status=Status.Completed))
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

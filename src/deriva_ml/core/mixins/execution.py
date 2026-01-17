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

    def lookup_execution(self, execution_rid: RID) -> "Execution":
        """Look up an execution by RID without restoring/downloading datasets.

        Creates a lightweight Execution object for querying execution metadata
        and relationships (e.g., nested executions) without initializing the
        full execution environment.

        Args:
            execution_rid: Resource Identifier (RID) of the execution.

        Returns:
            Execution: An execution object for the given RID.

        Raises:
            DerivaMLException: If execution_rid is not valid.

        Example:
            >>> execution = ml.lookup_execution("1-abc123")
            >>> children = execution.list_nested_executions()
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution import Execution

        # Get execution record from catalog
        execution_record = self.retrieve_rid(execution_rid)

        # Create minimal configuration
        configuration = ExecutionConfiguration(
            workflow=execution_record.get("Workflow"),
            description=execution_record.get("Description", ""),
        )

        # Create execution object without initialization (reload mode)
        exec_obj = Execution.__new__(Execution)
        exec_obj._ml_object = self  # type: ignore[arg-type]
        exec_obj._model = self.model
        exec_obj._logger = self._logger  # type: ignore[attr-defined]
        exec_obj.configuration = configuration
        exec_obj.execution_rid = execution_rid
        exec_obj.workflow_rid = execution_record.get("Workflow")
        exec_obj.status = Status(execution_record.get("Status", "Created"))
        exec_obj._dry_run = False
        exec_obj.dataset_rids = []
        exec_obj.datasets = []
        exec_obj.asset_paths = {}
        exec_obj.start_time = None
        exec_obj.stop_time = None
        exec_obj.uploaded_assets = None
        exec_obj._working_dir = self.working_dir
        exec_obj._cache_dir = self.cache_dir  # type: ignore[attr-defined]

        return exec_obj

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
            configuration = ExecutionConfiguration(
                workflow=execution["Workflow"],
                description=execution["Description"],
            )

        # Create and return an execution instance
        return Execution(configuration, self, reload=execution_rid)  # type: ignore[arg-type]

    def find_executions(
        self,
        workflow_rid: RID | None = None,
        status: Status | None = None,
    ) -> Iterable["Execution"]:
        """List all executions in the catalog.

        Args:
            workflow_rid: Optional workflow RID to filter by.
            status: Optional status to filter by (e.g., Status.Completed).

        Returns:
            Iterable of Execution objects.

        Example:
            >>> executions = list(ml.find_executions())
            >>> for exec in executions:
            ...     print(f"{exec.execution_rid}: {exec.status}")
            >>> # Filter by workflow
            >>> completed = list(ml.find_executions(status=Status.Completed))
        """
        # Get datapath to the Execution table
        pb = self.pathBuilder()
        execution_path = pb.schemas[self.ml_schema].Execution

        # Apply filters
        filtered_path = execution_path
        if workflow_rid:
            filtered_path = filtered_path.filter(execution_path.Workflow == workflow_rid)
        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        # Create Execution objects
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
        from deriva_ml.experiment import Experiment

        # Get datapath to Execution table with a link to Execution_Metadata
        # to find executions that have hydra config files
        pb = self.pathBuilder()
        execution_path = pb.schemas[self.ml_schema].Execution
        metadata_path = pb.schemas[self.ml_schema].Execution_Metadata

        # Find executions that have metadata assets with config.yaml files
        # Join Execution -> Execution_Metadata and filter for config.yaml filenames
        execution_metadata_link = metadata_path.link(execution_path)
        filtered_metadata = execution_metadata_link.filter(
            metadata_path.Filename.regexp(".*-config\\.yaml$")
        )

        # Get distinct execution RIDs that have config files
        exec_rids_with_config = set()
        for record in filtered_metadata.entities().fetch():
            exec_rids_with_config.add(record["Execution"])

        # Apply additional filters and yield Experiment objects
        filtered_path = execution_path
        if workflow_rid:
            filtered_path = filtered_path.filter(execution_path.Workflow == workflow_rid)
        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        for exec_record in filtered_path.entities().fetch():
            if exec_record["RID"] in exec_rids_with_config:
                yield Experiment(self, exec_record["RID"])  # type: ignore[arg-type]

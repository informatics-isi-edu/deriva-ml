"""Execution management mixin for DerivaML.

This module provides the ExecutionMixin class which handles
execution operations including creating, restoring, and updating
execution status.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from deriva_ml.core.definitions import RID
from deriva_ml.core.enums import Status
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.upload import asset_file_path, execution_rids
from deriva_ml.execution.execution_configuration import ExecutionConfiguration

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.workflow import Workflow
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
        """Creates an execution environment.

        Given an execution configuration, initialize the local compute environment to prepare for executing an
        ML or analytic routine.  This routine has a number of side effects.

        1. The datasets specified in the configuration are downloaded and placed in the cache-dir. If a version is
        not specified in the configuration, then a new minor version number is created for the dataset and downloaded.

        2. If any execution assets are provided in the configuration, they are downloaded
        and placed in the working directory.

        Args:
            configuration: ExecutionConfiguration object specifying the execution parameters.
            workflow: Workflow object representing the workflow to execute if not present in the ExecutionConfiguration.
            dry_run: Do not create an execution record or upload results.

        Returns:
            An execution object.
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

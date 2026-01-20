"""ExecutionRecord - Represents a catalog record for an execution.

This module provides the ExecutionRecord class which represents the state of an
execution record in the Deriva catalog. It provides getters and setters for
mutable properties that automatically sync changes to the catalog.

The ExecutionRecord is separate from the Execution class which manages the
execution lifecycle (start, stop, asset uploads, etc.). This separation allows
for lightweight lookups of execution records without initializing the full
execution environment.

Example:
    Look up an execution record and update its description::

        >>> record = ml.lookup_execution("2-ABC1")
        >>> print(record.status)
        Status.running
        >>> record.description = "Updated analysis description"
        >>> # The change is immediately written to the catalog

    Query nested executions::

        >>> children = record.list_nested_executions()
        >>> for child in children:
        ...     print(f"{child.execution_rid}: {child.status}")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from deriva_ml.core.definitions import RID, Status
from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.asset.asset import Asset
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.execution.workflow import Workflow
    from deriva_ml.interfaces import DerivaMLCatalog


class ExecutionRecord(BaseModel):
    """Represents a catalog record for an execution.

    An ExecutionRecord provides access to the persistent state of an execution
    stored in the Deriva catalog. When bound to a writable catalog, its mutable
    properties (status, description) can be set and changes are automatically
    synced to the catalog.

    This class is separate from the Execution class which manages the execution
    lifecycle. Use ExecutionRecord for lightweight queries and updates to
    execution metadata. Use Execution for running computations with datasets
    and assets.

    Attributes:
        execution_rid (RID): Resource Identifier of the execution record.
        workflow (Workflow | None): The associated workflow object, bound to catalog.
        status (Status): Current execution status (Created, Running, Completed, Failed).
            Setting this property updates the catalog.
        description (str | None): Description of the execution. Setting this
            property updates the catalog.
        start_time (datetime | None): When the execution started (read-only).
        stop_time (datetime | None): When the execution completed (read-only).
        duration (str | None): Duration string from catalog (read-only).

    Example:
        Look up an execution and query its state::

            >>> record = ml.lookup_execution("2-ABC1")
            >>> print(f"Status: {record.status}")
            >>> print(f"Workflow: {record.workflow.name}")
            >>> print(f"Started: {record.start_time}")

        Update mutable properties::

            >>> record.status = Status.completed
            >>> record.description = "Analysis completed successfully"

        Query relationships::

            >>> # Get child executions
            >>> children = record.list_nested_executions()
            >>> # Get parent executions
            >>> parents = record.list_parent_executions()
            >>> # Get input datasets
            >>> datasets = record.list_input_datasets()

        Attempting to update on a read-only catalog raises an error::

            >>> snapshot = ml.catalog_snapshot("2023-01-15T10:30:00")
            >>> record = snapshot.lookup_execution("2-ABC1")
            >>> record.status = Status.completed  # Raises DerivaMLException
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    execution_rid: RID
    _workflow: "Workflow | None" = PrivateAttr(default=None)
    _status: Status = PrivateAttr(default=Status.created)
    _description: str | None = PrivateAttr(default=None)
    start_time: datetime | None = None
    stop_time: datetime | None = None
    duration: str | None = None

    _ml_instance: "DerivaMLCatalog | None" = PrivateAttr(default=None)
    _logger: logging.Logger = PrivateAttr(default=None)

    def __init__(
        self,
        execution_rid: RID,
        workflow: "Workflow | None" = None,
        status: Status = Status.created,
        description: str | None = None,
        start_time: datetime | None = None,
        stop_time: datetime | None = None,
        duration: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an ExecutionRecord.

        Args:
            execution_rid: Resource Identifier of the execution.
            workflow: The associated Workflow object (bound to catalog).
            status: Current execution status.
            description: Description of the execution.
            start_time: When the execution started.
            stop_time: When the execution completed.
            duration: Duration string.
            **kwargs: Additional arguments (including _ml_instance for internal use).
        """
        super().__init__(
            execution_rid=execution_rid,
            start_time=start_time,
            stop_time=stop_time,
            duration=duration,
        )
        self._workflow = workflow
        self._status = status
        self._description = description
        # Handle _ml_instance passed as keyword arg
        if "_ml_instance" in kwargs:
            self._ml_instance = kwargs["_ml_instance"]
        if "_logger" in kwargs:
            self._logger = kwargs["_logger"]

    @property
    def workflow(self) -> "Workflow | None":
        """Get the associated workflow.

        Returns:
            The Workflow object, or None if no workflow is associated.
        """
        return self._workflow

    @property
    def workflow_rid(self) -> RID | None:
        """Get the RID of the associated workflow.

        Returns:
            The workflow RID, or None if no workflow is associated.
        """
        return self._workflow.rid if self._workflow else None

    @property
    def status(self) -> Status:
        """Get the current execution status.

        Returns:
            Status: The current status (Created, Running, Completed, Failed, etc.).
        """
        return self._status

    @status.setter
    def status(self, value: Status) -> None:
        """Set the execution status.

        When bound to a writable catalog, this updates the catalog record.

        Args:
            value: The new status value.

        Raises:
            DerivaMLException: If the catalog is read-only (snapshot).
        """
        if self._ml_instance is not None:
            self._update_status_in_catalog(value)
        self._status = value

    @property
    def description(self) -> str | None:
        """Get the execution description.

        Returns:
            The description string, or None if not set.
        """
        return self._description

    @description.setter
    def description(self, value: str | None) -> None:
        """Set the execution description.

        When bound to a writable catalog, this updates the catalog record.

        Args:
            value: The new description value.

        Raises:
            DerivaMLException: If the catalog is read-only (snapshot).
        """
        if self._ml_instance is not None:
            self._update_description_in_catalog(value)
        self._description = value

    def _check_writable_catalog(self, operation: str) -> None:
        """Check that the catalog is writable and execution is registered.

        Args:
            operation: Description of the operation being attempted.

        Raises:
            DerivaMLException: If the execution is not registered (no RID),
                or if the catalog is read-only (a snapshot).
        """
        import importlib
        _deriva_core = importlib.import_module("deriva.core")
        ErmrestSnapshot = _deriva_core.ErmrestSnapshot

        if self.execution_rid is None:
            raise DerivaMLException(
                f"Cannot {operation}: Execution is not registered in the catalog (no RID)"
            )

        if self._ml_instance is None:
            raise DerivaMLException(
                f"Cannot {operation}: ExecutionRecord is not bound to a catalog"
            )

        if isinstance(self._ml_instance.catalog, ErmrestSnapshot):
            raise DerivaMLException(
                f"Cannot {operation} on a read-only catalog snapshot. "
                "Use a writable catalog connection instead."
            )

    def _update_status_in_catalog(self, new_status: Status, status_detail: str = "") -> None:
        """Update the status field in the catalog.

        Args:
            new_status: The new status value.
            status_detail: Optional detail message for the status.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.
        """
        self._check_writable_catalog("update status")

        pb = self._ml_instance.pathBuilder()
        execution_path = pb.schemas[self._ml_instance.ml_schema].Execution
        update_data = {"RID": self.execution_rid, "Status": new_status.value}
        if status_detail:
            update_data["Status_Detail"] = status_detail
        execution_path.update([update_data])

    def _update_description_in_catalog(self, new_description: str | None) -> None:
        """Update the description field in the catalog.

        Args:
            new_description: The new description value.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.
        """
        self._check_writable_catalog("update description")

        pb = self._ml_instance.pathBuilder()
        execution_path = pb.schemas[self._ml_instance.ml_schema].Execution
        execution_path.update([{"RID": self.execution_rid, "Description": new_description}])

    def update_status(self, status: Status, status_detail: str = "") -> None:
        """Update execution status with an optional detail message.

        This method updates both the Status and Status_Detail columns in the
        catalog. Use this when you want to include a detail message, otherwise
        you can simply assign to the status property.

        Args:
            status: The new status value.
            status_detail: Optional detail message describing the status.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.

        Example:
            >>> record.update_status(Status.failed, "Network timeout during data transfer")
        """
        if self._ml_instance is not None:
            self._update_status_in_catalog(status, status_detail)
        self._status = status

    def is_nested(self) -> bool:
        """Check if this execution has any parent executions.

        Returns:
            True if this execution is nested under another execution.

        Example:
            >>> if record.is_nested():
            ...     print("This is a child execution")
        """
        return len(list(self.list_parent_executions())) > 0

    def is_parent(self) -> bool:
        """Check if this execution has any child executions.

        Returns:
            True if this execution has nested child executions.

        Example:
            >>> if record.is_parent():
            ...     print("This execution has children")
        """
        return len(list(self.list_nested_executions())) > 0

    def list_nested_executions(
        self, recurse: bool = False, _visited: set[RID] | None = None
    ) -> Iterable["ExecutionRecord"]:
        """List child executions nested under this execution.

        Args:
            recurse: If True, recursively list all descendants.
            _visited: Internal parameter to track visited nodes and prevent cycles.

        Returns:
            Iterable of ExecutionRecord objects for child executions.

        Raises:
            DerivaMLException: If not bound to a catalog.

        Example:
            >>> for child in record.list_nested_executions():
            ...     print(f"Child: {child.execution_rid}")
            >>> # Get all descendants
            >>> for desc in record.list_nested_executions(recurse=True):
            ...     print(f"Descendant: {desc.execution_rid}")
        """
        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        # Track visited nodes to prevent infinite loops
        if _visited is None:
            _visited = set()
        if self.execution_rid in _visited:
            return
        _visited.add(self.execution_rid)

        pb = self._ml_instance.pathBuilder()
        ml_schema = self._ml_instance.ml_schema
        exec_exec_path = pb.schemas[ml_schema].Execution_Execution
        execution_path = pb.schemas[ml_schema].Execution

        # Query for child executions (Execution column = parent, Nested_Execution = child)
        records = list(
            exec_exec_path
            .filter(exec_exec_path.Execution == self.execution_rid)
            .link(execution_path, on=(exec_exec_path.Nested_Execution == execution_path.RID))
            .entities()
            .fetch()
        )

        for record in records:
            # Look up the workflow if present
            workflow_rid = record.get("Workflow")
            workflow = self._ml_instance.lookup_workflow(workflow_rid) if workflow_rid else None

            child = ExecutionRecord(
                execution_rid=record["RID"],
                workflow=workflow,
                status=Status(record.get("Status", "Created")),
                description=record.get("Description"),
                _ml_instance=self._ml_instance,
                _logger=self._logger,
            )
            yield child
            if recurse:
                yield from child.list_nested_executions(recurse=True, _visited=_visited)

    def list_parent_executions(
        self, recurse: bool = False, _visited: set[RID] | None = None
    ) -> Iterable["ExecutionRecord"]:
        """List parent executions that this execution is nested under.

        Args:
            recurse: If True, recursively list all ancestors.
            _visited: Internal parameter to track visited nodes and prevent cycles.

        Returns:
            Iterable of ExecutionRecord objects for parent executions.

        Raises:
            DerivaMLException: If not bound to a catalog.

        Example:
            >>> for parent in record.list_parent_executions():
            ...     print(f"Parent: {parent.execution_rid}")
        """
        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        # Track visited nodes to prevent infinite loops
        if _visited is None:
            _visited = set()
        if self.execution_rid in _visited:
            return
        _visited.add(self.execution_rid)

        pb = self._ml_instance.pathBuilder()
        ml_schema = self._ml_instance.ml_schema
        exec_exec_path = pb.schemas[ml_schema].Execution_Execution
        execution_path = pb.schemas[ml_schema].Execution

        # Query for parent executions (Execution column = parent, Nested_Execution = child)
        records = list(
            exec_exec_path
            .filter(exec_exec_path.Nested_Execution == self.execution_rid)
            .link(execution_path, on=(exec_exec_path.Execution == execution_path.RID))
            .entities()
            .fetch()
        )

        for record in records:
            # Look up the workflow if present
            workflow_rid = record.get("Workflow")
            workflow = self._ml_instance.lookup_workflow(workflow_rid) if workflow_rid else None

            parent = ExecutionRecord(
                execution_rid=record["RID"],
                workflow=workflow,
                status=Status(record.get("Status", "Created")),
                description=record.get("Description"),
                _ml_instance=self._ml_instance,
                _logger=self._logger,
            )
            yield parent
            if recurse:
                yield from parent.list_parent_executions(recurse=True, _visited=_visited)

    def add_nested_execution(self, child: "ExecutionRecord | RID", sequence: int | None = None) -> None:
        """Add a child execution nested under this execution.

        Args:
            child: The child ExecutionRecord or its RID.
            sequence: Optional sequence number for ordering children.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.

        Example:
            >>> parent_record.add_nested_execution(child_record)
            >>> # Or by RID
            >>> parent_record.add_nested_execution("3-XYZ9", sequence=1)
        """
        self._check_writable_catalog("add nested execution")

        child_rid = child.execution_rid if isinstance(child, ExecutionRecord) else child

        pb = self._ml_instance.pathBuilder()
        exec_exec_path = pb.schemas[self._ml_instance.ml_schema].Execution_Execution

        record = {
            "Execution": self.execution_rid,
            "Nested_Execution": child_rid,
        }
        if sequence is not None:
            record["Sequence"] = sequence

        exec_exec_path.insert([record])

    def list_input_datasets(self) -> list["Dataset"]:
        """List datasets that were input to this execution.

        Returns:
            List of Dataset objects that were used as inputs to this execution.

        Raises:
            DerivaMLException: If not bound to a catalog.

        Example:
            >>> for ds in record.list_input_datasets():
            ...     print(f"Dataset: {ds.dataset_rid} version {ds.current_version}")
        """
        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        pb = self._ml_instance.pathBuilder()
        dataset_exec_path = pb.schemas[self._ml_instance.ml_schema].Dataset_Execution

        records = list(
            dataset_exec_path
            .filter(dataset_exec_path.Execution == self.execution_rid)
            .entities()
            .fetch()
        )

        # Look up each dataset and return Dataset objects
        datasets = []
        for record in records:
            dataset_rid = record.get("Dataset")
            if dataset_rid:
                datasets.append(self._ml_instance.lookup_dataset(dataset_rid))
        return datasets

    def list_input_assets(self, asset_role: str | None = None) -> list["Asset"]:
        """List assets associated with this execution.

        Args:
            asset_role: Optional filter for asset role ('Input' or 'Output').
                If None, returns all assets associated with this execution.

        Returns:
            List of Asset objects associated with this execution.

        Raises:
            DerivaMLException: If not bound to a catalog.

        Example:
            >>> # Get all input assets
            >>> for asset in record.list_input_assets(asset_role="Input"):
            ...     print(f"Input Asset: {asset.asset_rid} - {asset.filename}")
            >>> # Get all output assets
            >>> for asset in record.list_input_assets(asset_role="Output"):
            ...     print(f"Output Asset: {asset.asset_rid}")
        """
        from deriva_ml.asset.asset import Asset

        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        # Find all *_Execution association tables and query them
        # Search both the domain schema and the ML schema
        assets: list[Asset] = []
        schemas_to_search = [self._ml_instance.domain_schema, self._ml_instance.ml_schema]

        for schema_name in schemas_to_search:
            for table in self._ml_instance.model.model.schemas[schema_name].tables.values():
                if table.name.endswith("_Execution") and table.name != "Dataset_Execution":
                    # Extract asset table name from association table name
                    # e.g., "Image_Execution" -> "Image", "Execution_Asset_Execution" -> "Execution_Asset"
                    asset_table_name = table.name.replace("_Execution", "")

                    pb = self._ml_instance.pathBuilder()
                    table_path = pb.schemas[schema_name].tables[table.name]
                    try:
                        query = table_path.filter(table_path.Execution == self.execution_rid)
                        if asset_role:
                            query = query.filter(table_path.Asset_Role == asset_role)
                        records = list(query.entities().fetch())

                        # Look up each asset and convert to Asset object
                        for record in records:
                            asset_rid = record.get(asset_table_name)
                            if asset_rid:
                                try:
                                    assets.append(self._ml_instance.lookup_asset(asset_rid))
                                except Exception:
                                    pass  # Asset might not exist or be inaccessible
                    except Exception:
                        # Table might not have expected columns
                        pass
        return assets

    def __str__(self) -> str:
        """Return string representation of the execution record."""
        lines = [
            f"ExecutionRecord(rid={self.execution_rid})",
            f"  workflow_rid: {self.workflow_rid}",
            f"  status: {self.status.value}",
            f"  description: {self.description}",
        ]
        if self.start_time:
            lines.append(f"  start_time: {self.start_time}")
        if self.stop_time:
            lines.append(f"  stop_time: {self.stop_time}")
        if self.duration:
            lines.append(f"  duration: {self.duration}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return repr of the execution record."""
        return f"ExecutionRecord(execution_rid={self.execution_rid!r}, status={self.status!r})"

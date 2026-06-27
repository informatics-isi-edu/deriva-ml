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

        >>> record = ml.lookup_execution("2-ABC1")  # doctest: +SKIP
        >>> print(record.status)  # doctest: +SKIP
        ExecutionStatus.Running
        >>> record.description = "Updated analysis description"  # doctest: +SKIP
        >>> # The change is immediately written to the catalog

    Query nested executions::

        >>> children = record.list_execution_children()  # doctest: +SKIP
        >>> for child in children:  # doctest: +SKIP
        ...     print(f"{child.execution_rid}: {child.status}")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.validation import VALIDATION_CONFIG
from deriva_ml.execution.state_store import ExecutionStatus

logger = get_logger(__name__)
if TYPE_CHECKING:
    from deriva_ml.asset.asset import Asset
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.execution.workflow import Workflow
    from deriva_ml.interfaces import DerivaMLCatalog


__all__ = ["ExecutionRecord"]


class ExecutionRecord(BaseModel):
    """Represents a catalog record for an execution.

    A live, catalog-bound record. Property setters (``record.status = ...``,
    ``record.description = ...``) write through to the catalog on
    assignment; requires online mode for mutations. Returned by
    :meth:`~deriva_ml.DerivaML.lookup_execution` and
    :meth:`~deriva_ml.DerivaML.find_executions`.

    For a frozen snapshot value object that reads from the local
    SQLite registry and works offline, see
    :class:`~deriva_ml.execution.execution_snapshot.ExecutionSnapshot`
    returned by :meth:`~deriva_ml.DerivaML.list_executions` and
    :meth:`~deriva_ml.DerivaML.find_incomplete_executions`.

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
        status (ExecutionStatus): Current execution status (Created, Running,
            Stopped, Failed, Pending_Upload, Uploaded, Aborted). Setting this
            property updates the catalog.
        description (str | None): Description of the execution. Setting this
            property updates the catalog.
        start_time (datetime | None): When the execution started (read-only).
        stop_time (datetime | None): When the execution completed (read-only).
        duration (str | None): Algorithm-phase duration string from the
            catalog ``Execution_Duration`` column (read-only). Renamed
            from "Duration" 2026-05-19; old catalogs that predate the
            schema bump report None.
        download_duration (str | None): Init/download-phase duration
            string from the catalog ``Download_Duration`` column
            (read-only). None for old catalogs.
        upload_duration (str | None): Upload-phase duration string from
            the catalog ``Upload_Duration`` column (read-only). None for
            old catalogs.

    Example:
        Look up an execution and query its state::

            >>> record = ml.lookup_execution("2-ABC1")  # doctest: +SKIP
            >>> print(f"Status: {record.status}")  # doctest: +SKIP
            >>> print(f"Workflow: {record.workflow.name}")  # doctest: +SKIP
            >>> print(f"Started: {record.start_time}")  # doctest: +SKIP

        Update mutable properties::

            >>> record.status = ExecutionStatus.Uploaded  # doctest: +SKIP
            >>> record.description = "Analysis completed successfully"  # doctest: +SKIP

        Query relationships::

            >>> # Get child executions
            >>> children = record.list_execution_children()  # doctest: +SKIP
            >>> # Get parent executions
            >>> parents = record.list_execution_parents()  # doctest: +SKIP
            >>> # Get input datasets
            >>> datasets = record.list_input_datasets()  # doctest: +SKIP

        Attempting to update on a read-only catalog raises an error::

            >>> snapshot = ml.catalog_snapshot("2023-01-15T10:30:00")  # doctest: +SKIP
            >>> record = snapshot.lookup_execution("2-ABC1")  # doctest: +SKIP
            >>> record.status = ExecutionStatus.Uploaded  # Raises DerivaMLException  # doctest: +SKIP
    """

    model_config = VALIDATION_CONFIG

    execution_rid: RID
    _workflow: "Workflow | None" = PrivateAttr(default=None)
    _status: ExecutionStatus = PrivateAttr(default=ExecutionStatus.Created)
    _description: str | None = PrivateAttr(default=None)
    start_time: datetime | None = None
    stop_time: datetime | None = None
    duration: str | None = None
    download_duration: str | None = None
    upload_duration: str | None = None

    _ml_instance: "DerivaMLCatalog | None" = PrivateAttr(default=None)
    _logger: logging.Logger = PrivateAttr(default_factory=lambda: get_logger(__name__))

    def __init__(
        self,
        execution_rid: RID,
        workflow: "Workflow | None" = None,
        status: ExecutionStatus = ExecutionStatus.Created,
        description: str | None = None,
        start_time: datetime | None = None,
        stop_time: datetime | None = None,
        duration: str | None = None,
        download_duration: str | None = None,
        upload_duration: str | None = None,
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
            duration: Algorithm-phase duration string (Execution_Duration).
            download_duration: Init/download-phase duration string.
            upload_duration: Upload-phase duration string.
            **kwargs: Additional arguments (including _ml_instance for internal use).
        """
        super().__init__(
            execution_rid=execution_rid,
            start_time=start_time,
            stop_time=stop_time,
            duration=duration,
            download_duration=download_duration,
            upload_duration=upload_duration,
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
        return self._workflow.workflow_rid if self._workflow else None

    @property
    def status(self) -> ExecutionStatus:
        """Get the current execution status.

        Returns:
            ExecutionStatus: The current status (Created, Running, Stopped,
                Failed, Pending_Upload, Uploaded, Aborted).
        """
        return self._status

    @status.setter
    def status(self, value: ExecutionStatus) -> None:
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

        Delegates to the shared free helper in
        :mod:`deriva_ml.execution._helpers` — same contract used
        by :class:`Workflow`.

        Args:
            operation: Description of the operation being attempted.

        Raises:
            DerivaMLException: If the execution is not registered (no RID),
                or if the catalog is read-only (a snapshot).
        """
        from deriva_ml.execution._helpers import check_writable_catalog

        check_writable_catalog(
            rid=self.execution_rid,
            ml_instance=self._ml_instance,
            entity_label="Execution",
            operation=operation,
        )

    def _update_status_in_catalog(self, new_status: ExecutionStatus, status_detail: str = "") -> None:
        """Update the status field in the catalog.

        Args:
            new_status: The new status value.
            status_detail: Optional detail message for the status.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.
        """
        from deriva_ml.execution._helpers import update_field_in_catalog

        self._check_writable_catalog("update status")
        updates: dict[str, Any] = {"Status": new_status.value}
        if status_detail:
            updates["Status_Detail"] = status_detail
        update_field_in_catalog(
            rid=self.execution_rid,
            ml_instance=self._ml_instance,
            table_name="Execution",
            updates=updates,
        )

    def _update_description_in_catalog(self, new_description: str | None) -> None:
        """Update the description field in the catalog.

        Args:
            new_description: The new description value.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.
        """
        from deriva_ml.execution._helpers import update_field_in_catalog

        self._check_writable_catalog("update description")
        update_field_in_catalog(
            rid=self.execution_rid,
            ml_instance=self._ml_instance,
            table_name="Execution",
            updates={"Description": new_description},
        )

    def update_status(self, status: ExecutionStatus, status_detail: str = "") -> None:
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
            >>> record.update_status(ExecutionStatus.Failed, "Network timeout during data transfer")  # doctest: +SKIP
        """
        if self._ml_instance is not None:
            self._update_status_in_catalog(status, status_detail)
        self._status = status

    def is_nested(self) -> bool:
        """Check if this execution has any parent executions.

        Returns:
            True if this execution is nested under another execution.

        Example:
            >>> if record.is_nested():  # doctest: +SKIP
            ...     print("This is a child execution")
        """
        return len(list(self.list_execution_parents())) > 0

    def is_parent(self) -> bool:
        """Check if this execution has any child executions.

        Returns:
            True if this execution has nested child executions.

        Example:
            >>> if record.is_parent():  # doctest: +SKIP
            ...     print("This execution has children")
        """
        return len(list(self.list_execution_children())) > 0

    def list_execution_children(
        self, recurse: bool = False, _visited: set[RID] | None = None
    ) -> Iterable["ExecutionRecord"]:
        """List child executions nested under this execution.

        Mirrors the dataset-hierarchy template (``list_dataset_children``)
        per spec §2.9/§2.17 — renamed from ``list_nested_executions`` in
        the R5.1 hard cutover; there is no backward-compat alias.

        Args:
            recurse: If True, recursively list all descendants, yielding
                each descendant exactly once even when the hierarchy
                contains a cycle (the ``_visited`` set guards against
                infinite recursion).
            _visited: Internal parameter to track visited nodes and
                prevent cycles. Callers normally leave this at its
                default of ``None``.

        Yields:
            ExecutionRecord: Child executions. When ``recurse`` is
            False, only direct children; when True, all descendants in
            depth-first order.

        Raises:
            DerivaMLException: If this record is not bound to a catalog
                (e.g. a bare ``ExecutionRecord`` built without
                ``_ml_instance``).

        Example:
            Direct children only::

                >>> for child in record.list_execution_children():  # doctest: +SKIP
                ...     print(f"Child: {child.execution_rid}")

            All descendants::

                >>> for desc in record.list_execution_children(recurse=True):  # doctest: +SKIP
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

        from deriva_ml.execution._helpers import fetch_nested_execution_rows

        records = fetch_nested_execution_rows(
            ml_instance=self._ml_instance,
            execution_rid=self.execution_rid,
            direction="children",
        )

        for record in records:
            # Look up the workflow if present
            workflow_rid = record.get("Workflow")
            workflow = self._ml_instance.lookup_workflow(workflow_rid) if workflow_rid else None

            child = ExecutionRecord(
                execution_rid=record["RID"],
                workflow=workflow,
                status=ExecutionStatus(record.get("Status") or "Created"),
                description=record.get("Description"),
                # Phase-duration columns (PR-2, 2026-05-19): forward
                # the catalog values so consumers like the MCP
                # `deriva_ml_list_execution_children` see the same
                # timing data that the singular `lookup_execution`
                # path exposes. Old rows pre-PR-2 simply lack the
                # columns and fall to None.
                duration=record.get("Execution_Duration"),
                download_duration=record.get("Download_Duration"),
                upload_duration=record.get("Upload_Duration"),
                _ml_instance=self._ml_instance,
                _logger=self._logger,
            )
            yield child
            if recurse:
                yield from child.list_execution_children(recurse=True, _visited=_visited)

    def list_execution_parents(
        self, recurse: bool = False, _visited: set[RID] | None = None
    ) -> Iterable["ExecutionRecord"]:
        """List parent executions that this execution is nested under.

        Mirrors the dataset-hierarchy template (``list_dataset_parents``)
        per spec §2.9/§2.17 — renamed from ``list_parent_executions`` in
        the R5.1 hard cutover; there is no backward-compat alias.

        Args:
            recurse: If True, recursively list all ancestors, yielding
                each ancestor exactly once even when the hierarchy
                contains a cycle (the ``_visited`` set guards against
                infinite recursion).
            _visited: Internal parameter to track visited nodes and
                prevent cycles. Callers normally leave this at its
                default of ``None``.

        Yields:
            ExecutionRecord: Parent executions. When ``recurse`` is
            False, only direct parents; when True, all ancestors in
            depth-first order.

        Raises:
            DerivaMLException: If this record is not bound to a catalog
                (e.g. a bare ``ExecutionRecord`` built without
                ``_ml_instance``).

        Example:
            Direct parents only::

                >>> for parent in record.list_execution_parents():  # doctest: +SKIP
                ...     print(f"Parent: {parent.execution_rid}")

            All ancestors::

                >>> for anc in record.list_execution_parents(recurse=True):  # doctest: +SKIP
                ...     print(f"Ancestor: {anc.execution_rid}")
        """
        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        # Track visited nodes to prevent infinite loops
        if _visited is None:
            _visited = set()
        if self.execution_rid in _visited:
            return
        _visited.add(self.execution_rid)

        from deriva_ml.execution._helpers import fetch_nested_execution_rows

        records = fetch_nested_execution_rows(
            ml_instance=self._ml_instance,
            execution_rid=self.execution_rid,
            direction="parents",
        )

        for record in records:
            # Look up the workflow if present
            workflow_rid = record.get("Workflow")
            workflow = self._ml_instance.lookup_workflow(workflow_rid) if workflow_rid else None

            parent = ExecutionRecord(
                execution_rid=record["RID"],
                workflow=workflow,
                status=ExecutionStatus(record.get("Status") or "Created"),
                description=record.get("Description"),
                # Phase-duration columns (PR-2, 2026-05-19): forward
                # the catalog values so consumers like the MCP
                # `deriva_ml_list_execution_parents` see the same
                # timing data that the singular `lookup_execution`
                # path exposes. Old rows pre-PR-2 simply lack the
                # columns and fall to None.
                duration=record.get("Execution_Duration"),
                download_duration=record.get("Download_Duration"),
                upload_duration=record.get("Upload_Duration"),
                _ml_instance=self._ml_instance,
                _logger=self._logger,
            )
            yield parent
            if recurse:
                yield from parent.list_execution_parents(recurse=True, _visited=_visited)

    def add_nested_execution(self, child: "ExecutionRecord | RID", sequence: int | None = None) -> None:
        """Add a child execution nested under this execution.

        Args:
            child: The child ExecutionRecord or its RID.
            sequence: Optional sequence number for ordering children.

        Raises:
            DerivaMLException: If the catalog is read-only or not connected.

        Example:
            >>> parent_record.add_nested_execution(child_record)  # doctest: +SKIP
            >>> # Or by RID
            >>> parent_record.add_nested_execution("3-XYZ9", sequence=1)  # doctest: +SKIP
        """
        from deriva_ml.execution._helpers import insert_nested_execution_link

        self._check_writable_catalog("add nested execution")
        child_rid = child.execution_rid if isinstance(child, ExecutionRecord) else child
        insert_nested_execution_link(
            ml_instance=self._ml_instance,
            parent_rid=self.execution_rid,
            child_rid=child_rid,
            sequence=sequence,
        )

    def list_input_datasets(self) -> list["Dataset"]:
        """List datasets that were input to this execution.

        Returns every dataset linked to this execution via a
        ``Dataset_Execution`` row. Under the authorship-canonical model,
        ``Dataset_Execution`` is **input-only** — a dataset this execution
        *produced* lives on ``Dataset_Version.Execution``, never here — so every
        ``Dataset_Execution`` row is an input and no producer subtraction is
        performed. (One execution may be both producer and input-consumer of the
        same dataset — e.g. ``add_files`` records a source dataset as both its
        output and its declared input — and that dataset is correctly returned
        here as an input.)

        Returns:
            List of Dataset objects that were used as inputs to this execution.

        Raises:
            DerivaMLException: If not bound to a catalog.

        Example:
            >>> for ds in record.list_input_datasets():  # doctest: +SKIP
            ...     print(f"Dataset: {ds.dataset_rid} version {ds.current_version}")
        """
        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        from deriva_ml.execution._helpers import list_input_datasets as _list_input_datasets

        return _list_input_datasets(
            ml_instance=self._ml_instance,
            execution_rid=self.execution_rid,
        )

    def list_assets(self, asset_role: str | None = None) -> list["Asset"]:
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
            >>> for asset in record.list_assets(asset_role="Input"):  # doctest: +SKIP
            ...     print(f"Input Asset: {asset.asset_rid} - {asset.filename}")
            >>> # Get all output assets
            >>> for asset in record.list_assets(asset_role="Output"):  # doctest: +SKIP
            ...     print(f"Output Asset: {asset.asset_rid}")
        """

        if self._ml_instance is None:
            raise DerivaMLException("ExecutionRecord is not bound to a catalog")

        from deriva_ml.execution._helpers import list_assets as _list_assets

        return _list_assets(
            ml_instance=self._ml_instance,
            execution_rid=self.execution_rid,
            asset_role=asset_role,
            logger=logger,
        )

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
            lines.append(f"  execution_duration: {self.duration}")
        if self.download_duration:
            lines.append(f"  download_duration: {self.download_duration}")
        if self.upload_duration:
            lines.append(f"  upload_duration: {self.upload_duration}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return repr of the execution record."""
        return f"ExecutionRecord(execution_rid={self.execution_rid!r}, status={self.status!r})"


class MultirunStatusSummary(BaseModel):
    """Status counts across all executions of one workflow.

    Produced by :meth:`DerivaML.multirun_status_summary` -- the
    one-query answer to "is the sweep done?". Null catalog ``Status``
    values are counted under ``"Created"``, matching
    :meth:`lookup_execution`'s read contract.

    Attributes:
        workflow_rid: RID of the summarized workflow.
        counts: Mapping of status name (e.g. ``"Uploaded"``,
            ``"Running"``, ``"Failed"``) to execution count.
        total: Total executions of the workflow (the sum of
            ``counts`` values).
    """

    model_config = ConfigDict(extra="forbid")

    workflow_rid: str
    counts: dict[str, int]
    total: int

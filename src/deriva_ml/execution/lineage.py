"""Pydantic models for the lineage walk returned by ``lookup_lineage``.

Each :class:`LineageResult` describes the data-flow provenance chain
behind a single artifact (Dataset, Asset, Feature value, or
Execution). The walk follows producing-execution edges through
consumed inputs (datasets and assets) and explicitly does NOT walk
``Execution_Execution`` orchestration links — see
``docs/adr/0001-lineage-walks-data-flow-not-orchestration.md``.

The models live in their own module so they can cross a boundary:
the deriva-ml-mcp Round 6 follow-up serializes them with
``.model_dump()`` from a tool wrapper, and downstream agents
(notebook, skill, web app) consume the JSON.

Example:
    Inspect the producer of the immediate node::

        >>> result = ml.lookup_lineage("3-XYZ", depth=0)  # doctest: +SKIP
        >>> producer = result.lineage.execution
        >>> print(producer.rid, producer.workflow.name if producer.workflow else None)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deriva_ml.core.definitions import RID


class WorkflowSummary(BaseModel):
    """Compact view of a Workflow row.

    Only the fields a lineage consumer typically needs at a glance.
    Drill into the full record with ``ml.lookup_workflow(rid)``.

    Attributes:
        rid: Workflow RID.
        name: Human-readable workflow name (None if the row has no
            name set).
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    name: str | None = None


class ExecutionSummary(BaseModel):
    """Compact view of an Execution row.

    Surfaces just enough to identify the execution and decide whether
    to drill in. Use ``ml.lookup_execution(rid)`` for the live
    ``ExecutionRecord``.

    Attributes:
        rid: Execution RID.
        description: Execution description (may be None or empty).
        workflow: Compact workflow descriptor (None if the execution
            has no workflow link).
        status: Catalog status string (e.g. ``"Uploaded"``).
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    description: str | None = None
    workflow: WorkflowSummary | None = None
    status: str


class DatasetSummary(BaseModel):
    """Compact view of a consumed Dataset.

    Attributes:
        rid: Dataset RID.
        description: Dataset description (may be None or empty).
        version: Current version at the time the lineage was walked
            (e.g. ``"0.1.0"``). None if the dataset has no version
            history yet.
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    description: str | None = None
    version: str | None = None


class AssetSummary(BaseModel):
    """Compact view of a consumed Asset.

    Attributes:
        rid: Asset RID.
        filename: Original filename (may be empty if the asset row
            has no filename column populated).
        asset_table: Name of the asset table the row lives in
            (e.g. ``"Image"``, ``"Execution_Asset"``).
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    filename: str | None = None
    asset_table: str


class LineageNode(BaseModel):
    """One execution node in the lineage tree.

    Each node represents an execution that produced something further
    down the chain. ``parents`` holds the next layer up — the
    producing executions of this execution's consumed inputs.

    Attributes:
        execution: Compact execution descriptor for this node.
        consumed_datasets: Datasets this execution consumed as input.
        consumed_assets: Assets this execution consumed as input
            (asset_role="Input" in the ``<AssetTable>_Execution``
            association).
        parents: Producing executions of the consumed inputs.
            Deduplicated by execution RID.
        already_shown: True if this execution was already expanded
            elsewhere in the tree (diamond DAG marker). When True,
            ``parents`` is left empty to avoid re-walking; consumers
            should look up the original node by ``execution.rid``.
    """

    model_config = ConfigDict(extra="forbid")

    execution: ExecutionSummary
    consumed_datasets: list[DatasetSummary] = Field(default_factory=list)
    consumed_assets: list[AssetSummary] = Field(default_factory=list)
    parents: list["LineageNode"] = Field(default_factory=list)
    already_shown: bool = False


class RootDescriptor(BaseModel):
    """Describes the artifact the lineage walk started from.

    Attributes:
        rid: RID of the root artifact passed to ``lookup_lineage``.
        type: One of ``"Dataset"``, ``"Asset"``, ``"Feature"``, or
            ``"Execution"``. Determines how
            ``producing_execution`` was resolved.
        description: Description of the root, when available
            (Dataset.description, Asset.description,
            Execution.description). None for Feature values, which
            don't have a free-text description column at this layer.
        producing_execution: The execution that produced this
            artifact, or None if the artifact has no recorded
            producer (manually inserted data, etc.). For an
            Execution root, this is the execution itself.
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    type: Literal["Dataset", "Asset", "Feature", "Execution"]
    description: str | None = None
    producing_execution: ExecutionSummary | None = None


class LineageResult(BaseModel):
    """Result returned by :meth:`DerivaML.lookup_lineage`.

    Top-level transparency fields tell the caller whether the walk
    completed cleanly. ``walked_complete=False`` means the walk hit
    one of the defensive caps (``max_executions``) before reaching
    the root.

    Attributes:
        root: Descriptor of the artifact the walk started from.
        lineage: Tree of producing executions, rooted at the
            immediate producer of ``root``. None when the root has
            no recorded producer.
        executions_visited: Number of distinct executions the walk
            expanded. Includes the root execution when present.
        walked_complete: True if the walk ran to the natural root of
            every branch. False if ``max_executions`` was hit or a
            depth cap stopped the expansion.
        cycle_detected: True if a true cycle was detected (the same
            execution appearing on its own active recursion path).
            Diamond DAGs (the same execution reached via two
            independent paths) are NOT cycles; they're handled by
            the ``already_shown`` flag on :class:`LineageNode`.
        depth_capped: True if a positive ``depth`` argument
            prevented expansion of at least one branch.

    Example:
        Walk lineage of an output asset and pretty-print the chain::

            >>> result = ml.lookup_lineage("3JSE")  # doctest: +SKIP
            >>> assert result.walked_complete
            >>> print(f"visited {result.executions_visited} executions")
    """

    model_config = ConfigDict(extra="forbid")

    root: RootDescriptor
    lineage: LineageNode | None = None
    executions_visited: int = 0
    walked_complete: bool = True
    cycle_detected: bool = False
    depth_capped: bool = False


# Pydantic v2 forward-ref resolution for the recursive ``parents`` field.
LineageNode.model_rebuild()

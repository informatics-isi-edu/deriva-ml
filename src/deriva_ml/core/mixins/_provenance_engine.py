"""Shared provenance walk engine behind ``lookup_lineage`` / ``lookup_provenance``.

The engine owns the *mechanics* of walking data-flow provenance — cycle
detection, diamond-DAG deduplication, the defensive expansion cap, depth
bounding, and the per-execution input resolution — while a
:class:`WalkVisitor` owns what is *built* from the walk.

Two visitors are planned:

- :class:`TreeBuilder` (this module) rebuilds the historical
  :class:`~deriva_ml.execution.lineage.LineageNode` tree, byte-identical to
  the pre-extraction ``ExecutionMixin._walk_node``. Its closure hooks are
  no-ops.
- A closure builder (later tasks) consumes the same events plus the
  dataset-arc legs to assemble a flat provenance closure.

The engine calls every catalog seam on ``self._ml`` (the owning
``ExecutionMixin`` instance), so test harnesses that subclass or stub the
mixin keep working unchanged.

Example:
    >>> from deriva_ml.core.mixins._provenance_engine import TreeBuilder
    >>> builder = TreeBuilder()
    >>> builder.on_gap("no_workflow", "1-ABCD", "no workflow recorded") is None
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.provenance import ArcInputType, ArcKind, GapKind

if TYPE_CHECKING:
    from deriva_ml.execution.lineage import ExecutionSummary, LineageNode, VersionAttribution
    from deriva_ml.execution.provenance import DatasetVersionFacts, ParentLink
    from deriva_ml.feature import FeatureProducerRecord

__all__ = ["InputRef", "WalkVisitor", "WalkEngine", "TreeBuilder"]

N = TypeVar("N")


@dataclass(frozen=True)
class InputRef:
    """One concrete input consumed by an execution node.

    Attributes:
        kind: Whether the input is a dataset or an asset.
        rid: RID of the consumed dataset or asset.
        version: For datasets, the version that was actually consumed
            (``Dataset_Execution.Dataset_Version``); ``None`` when the edge
            carried no version pin. Always ``None`` for assets.
        summary: The display object for the input — a
            :class:`~deriva_ml.execution.lineage.DatasetSummary` or
            :class:`~deriva_ml.execution.lineage.AssetSummary`.
        producer_rids: Every recorded producing execution, in fetch order.
            The tree path uses the first entry (identical to the historical
            single-producer choice); closure consumers see the full tuple so
            a multiple-producer gap can be reported.

    Example:
        >>> from deriva_ml.execution.provenance import ArcInputType
        >>> ref = InputRef(kind=ArcInputType.asset, rid="1-ABCD")
        >>> ref.producer_rids
        ()
    """

    kind: ArcInputType
    rid: str
    version: str | None = None
    summary: Any = None
    producer_rids: tuple[str, ...] = ()


class WalkVisitor(Protocol[N]):
    """Callbacks a provenance walk emits, in walk order.

    Tree-construction methods (``make_*`` / ``attach_parents``) build the
    caller's node type ``N``. The remaining hooks are the closure event
    surface; a visitor that only needs a tree implements them as no-ops.
    """

    # -- tree construction ------------------------------------------------

    def make_node(self, rid: str, *, summary: "ExecutionSummary", inputs: list[InputRef], depth: int) -> N:
        """Build a node for a fully expanded execution."""
        ...

    def make_cycle_node(self, rid: str, *, depth: int) -> N:
        """Build the marker node for an execution already on the active path."""
        ...

    def make_duplicate_node(self, rid: str, *, depth: int) -> N:
        """Build the marker node for an execution already expanded elsewhere."""
        ...

    def attach_parents(self, node: N, parents: list[N]) -> None:
        """Attach the recursively built parent nodes to ``node``."""
        ...

    # -- seeding ----------------------------------------------------------

    def on_seed_candidate(self, rid: str, *, accepted: bool) -> None:
        """Observe one seed-candidate attempt and whether it expanded."""
        ...

    # -- closure hooks ----------------------------------------------------

    def on_execution(self, rid: str, *, summary: "ExecutionSummary", depth: int) -> None:
        """Observe an execution entering the closure."""
        ...

    def on_consumption(self, *, consumer_rid: str, input_ref: InputRef, depth: int) -> None:
        """Observe one consumption edge."""
        ...

    def on_version_author(
        self, *, dataset_rid: str, version: str, attribution: "VersionAttribution", depth: int
    ) -> None:
        """Observe the author of one walked dataset version."""
        ...

    def on_binding_record(self, *, dataset_rid: str, version: str, record: "FeatureProducerRecord", depth: int) -> None:
        """Observe one feature-binding record on a walked dataset version."""
        ...

    def on_parent_link(self, *, dataset_rid: str, link: "ParentLink") -> None:
        """Observe one snapshot-resolved parent hop of a dataset version."""
        ...

    def on_dataset_facts(self, *, dataset_rid: str, facts: "DatasetVersionFacts") -> None:
        """Observe the assembled facts for one dataset version."""
        ...

    def on_gap(self, kind: "GapKind", subject_rid: str, detail: str) -> None:
        """Observe an honest gap the walk could not resolve."""
        ...


@dataclass
class TreeBuilder:
    """Visitor that rebuilds the historical ``LineageNode`` tree.

    Byte-identical to the pre-extraction ``ExecutionMixin._walk_node``: the
    same node shape, the same cycle/duplicate markers, the same
    ``consumed_datasets`` / ``consumed_assets`` ordering. Every closure hook
    is a no-op — the tree carries no closure state.

    Example:
        >>> builder = TreeBuilder()
        >>> node = builder.make_cycle_node("1-ABCD", depth=2)
        >>> node.already_shown
        True
    """

    def _marker_node(self, rid: str) -> "LineageNode":
        """Build the shared cycle/duplicate marker node for ``rid``."""
        from deriva_ml.execution.lineage import ExecutionSummary, LineageNode

        return LineageNode(
            execution=ExecutionSummary(
                rid=rid,
                description=None,
                workflow=None,
                status="Unknown",
            ),
            already_shown=True,
        )

    # -- tree construction ------------------------------------------------

    def make_node(self, rid: str, *, summary: "ExecutionSummary", inputs: list[InputRef], depth: int) -> "LineageNode":
        """Build a ``LineageNode`` from an execution summary and its inputs."""
        from deriva_ml.execution.lineage import LineageNode

        return LineageNode(
            execution=summary,
            consumed_datasets=[i.summary for i in inputs if i.kind == ArcInputType.dataset],
            consumed_assets=[i.summary for i in inputs if i.kind == ArcInputType.asset],
        )

    def make_cycle_node(self, rid: str, *, depth: int) -> "LineageNode":
        """Build the marker node for an execution already on the active path."""
        return self._marker_node(rid)

    def make_duplicate_node(self, rid: str, *, depth: int) -> "LineageNode":
        """Build the marker node for an execution already expanded elsewhere."""
        return self._marker_node(rid)

    def attach_parents(self, node: "LineageNode", parents: list["LineageNode"]) -> None:
        """Attach ``parents`` to ``node`` in place."""
        node.parents = parents

    # -- seeding ----------------------------------------------------------

    def on_seed_candidate(self, rid: str, *, accepted: bool) -> None:
        """No-op: the tree does not record seed attempts."""

    # -- closure hooks (all no-ops for the tree) --------------------------

    def on_execution(self, rid: str, *, summary: "ExecutionSummary", depth: int) -> None:
        """No-op."""

    def on_consumption(self, *, consumer_rid: str, input_ref: InputRef, depth: int) -> None:
        """No-op."""

    def on_version_author(
        self, *, dataset_rid: str, version: str, attribution: "VersionAttribution", depth: int
    ) -> None:
        """No-op."""

    def on_binding_record(self, *, dataset_rid: str, version: str, record: "FeatureProducerRecord", depth: int) -> None:
        """No-op."""

    def on_parent_link(self, *, dataset_rid: str, link: "ParentLink") -> None:
        """No-op."""

    def on_dataset_facts(self, *, dataset_rid: str, facts: "DatasetVersionFacts") -> None:
        """No-op."""

    def on_gap(self, kind: "GapKind", subject_rid: str, detail: str) -> None:
        """No-op."""


@dataclass
class WalkEngine(Generic[N]):
    """Arc-gated data-flow provenance walker shared by lineage and closure.

    The engine expands executions depth-first, following data-flow parents
    only (producing executions of consumed datasets and input assets), and
    emits events to a :class:`WalkVisitor`. Which additional legs run is
    governed by ``arcs``: with no dataset-oriented arcs enabled (the lineage
    case) :meth:`expand_dataset` is a total no-op, so a lineage walk pays
    nothing for closure machinery.

    Args:
        ml: The owning ``ExecutionMixin`` instance; every catalog seam is
            called on it.
        visitor: Receives tree-construction and closure events.
        arcs: Which :class:`~deriva_ml.execution.provenance.ArcKind` legs are
            enabled for this walk.
        max_executions: Defensive cap on distinct executions expanded.
        dataset_budget: Optional cap on distinct dataset versions expanded;
            ``None`` means unbounded.

    Attributes:
        flags: ``cycle_detected`` / ``depth_capped`` / ``walked_complete``.
        executions_visited: Count of distinct executions expanded.
        datasets_visited: Count of distinct dataset versions expanded.
        cap_hit: True once a cap stopped an expansion.

    Example:
        >>> from deriva_ml.execution.provenance import ArcKind
        >>> engine = WalkEngine(
        ...     ml=None, visitor=TreeBuilder(), arcs=frozenset(), max_executions=500
        ... )
        >>> engine.flags["walked_complete"]
        True
    """

    ml: Any
    visitor: "WalkVisitor[N]"
    arcs: frozenset[ArcKind] = frozenset()
    max_executions: int = 500
    dataset_budget: int | None = None

    flags: dict[str, bool] = field(init=False)
    visited_global: set[RID] = field(init=False)
    in_progress: set[RID] = field(init=False)
    datasets_expanded: set[tuple[str, str | None]] = field(init=False)
    cap_hit: bool = field(init=False, default=False)
    _queue: list[tuple[RID, int]] = field(init=False)

    def __init__(
        self,
        ml: Any,
        visitor: "WalkVisitor[N]",
        *,
        arcs: frozenset[ArcKind] = frozenset(),
        max_executions: int = 500,
        dataset_budget: int | None = None,
    ) -> None:
        """Initialize the engine; see the class docstring for arguments."""
        self.ml = ml
        self.visitor = visitor
        self.arcs = arcs
        self.max_executions = max_executions
        self.dataset_budget = dataset_budget

        self.flags = {"cycle_detected": False, "depth_capped": False, "walked_complete": True}
        self.visited_global = set()
        self.in_progress = set()
        self.datasets_expanded = set()
        self.cap_hit = False
        self._queue = []

    # -- observable counters ---------------------------------------------

    @property
    def executions_visited(self) -> int:
        """Number of distinct executions the walk expanded."""
        return len(self.visited_global)

    @property
    def datasets_visited(self) -> int:
        """Number of distinct ``(dataset, version)`` pairs the walk expanded."""
        return len(self.datasets_expanded)

    # -- dataset arcs -----------------------------------------------------

    @property
    def _dataset_arcs_enabled(self) -> bool:
        """True when any dataset-oriented arc leg is enabled."""
        return bool(self.arcs & {ArcKind.version_authorship, ArcKind.member_binding})

    def expand_dataset(self, dataset_rid: str, version: str | None, *, depth: int) -> None:
        """Expand the dataset-side arcs for one consumed dataset version.

        Arc-gated and memoized per ``(dataset_rid, version)``. A walk with no
        dataset arcs enabled (the lineage case) returns immediately, so the
        lineage path pays nothing for closure machinery. ``version=None``
        (an unpinned edge) reports an ``unpinned_input`` gap and does no
        snapshot-dependent work.

        Args:
            dataset_rid: RID of the consumed dataset.
            version: The version actually consumed, or ``None`` when the edge
                carried no version pin.
            depth: Walk depth of the consuming execution.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder(), arcs=frozenset())
            >>> engine.expand_dataset("1-ABCD", "1.0.0", depth=0) is None
            True
            >>> engine.datasets_visited
            0
        """
        if not self._dataset_arcs_enabled:
            return

        key = (dataset_rid, version)
        if key in self.datasets_expanded:
            return

        if version is None:
            # Unpinned edge: honest gap, and no snapshot-dependent work is
            # possible because there is no version to pin a snapshot to.
            self.datasets_expanded.add(key)
            self.visitor.on_gap(
                GapKind.unpinned_input,
                dataset_rid,
                "consumption edge recorded no dataset version",
            )
            return

        if self.dataset_budget is not None and len(self.datasets_expanded) >= self.dataset_budget:
            self.cap_hit = True
            self.flags["walked_complete"] = False
            return

        self.datasets_expanded.add(key)
        # Arc legs (version authorship, member bindings, ancestry) are filled
        # in by the closure tasks; the gating skeleton lives here.

    # -- execution expansion ----------------------------------------------

    def expand_execution(
        self,
        rid: RID,
        *,
        depth_remaining: int | None,
        depth: int = 0,
        extra_parent_rids: set[RID] | None = None,
    ) -> N | None:
        """Expand one execution node and recurse on its data-flow parents.

        Mutates the engine's ``visited_global`` / ``in_progress`` / ``flags``
        state. Returns ``None`` only when the execution could not be looked
        up (defensive) or the expansion cap was already reached.

        Args:
            rid: Execution RID to expand.
            depth_remaining: Remaining parent levels to walk; ``None`` walks
                to the root, ``0`` stops after this node.
            depth: Depth of this node below the walk root, for visitor events.
            extra_parent_rids: Additional parent executions to merge in
                before recursion (root-seeded sibling producers).

        Returns:
            The visitor-built node, or ``None``.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder(), max_executions=0)
            >>> engine.expand_execution("1-ABCD", depth_remaining=None) is None
            True
        """
        from deriva_ml.execution.lineage import (
            AssetSummary,
            DatasetSummary,
            ExecutionSummary,
            WorkflowSummary,
        )

        ml = self.ml

        # Cycle on the active path: do not expand, set flag, return a
        # leaf-style marker.
        if rid in self.in_progress:
            self.flags["cycle_detected"] = True
            return self.visitor.make_cycle_node(rid, depth=depth)

        # Diamond DAG: this execution was already expanded somewhere
        # else in the tree. Mark and don't recurse.
        if rid in self.visited_global:
            return self.visitor.make_duplicate_node(rid, depth=depth)

        # Defensive cap on total expansions.
        if len(self.visited_global) >= self.max_executions:
            self.flags["walked_complete"] = False
            self.cap_hit = True
            return None

        # Look up the execution and its inputs. Deliberately per-node (not
        # batched): the walk discovers parents one node at a time, so there
        # is no batch to fetch until the node is expanded.
        try:
            record = ml.lookup_execution(rid)
        except DerivaMLException:
            # An input pointed at an Execution that no longer exists;
            # treat as missing rather than failing the whole walk.
            return None

        self.visited_global.add(rid)
        self.in_progress.add(rid)

        try:
            wf_summary: "WorkflowSummary | None" = None
            if record.workflow is not None and record.workflow.workflow_rid is not None:
                wf_summary = WorkflowSummary(
                    rid=record.workflow.workflow_rid,
                    name=record.workflow.name,
                    url=getattr(record.workflow, "url", None),
                    version=getattr(record.workflow, "version", None),
                    checksum=getattr(record.workflow, "checksum", None),
                )

            execution_summary = ExecutionSummary(
                rid=rid,
                description=record.description,
                workflow=wf_summary,
                status=record.status.value if record.status else "Unknown",
            )
            self.visitor.on_execution(rid, summary=execution_summary, depth=depth)

            # Consumed inputs. Walk the version that was ACTUALLY consumed
            # (Dataset_Execution.Dataset_Version), not the dataset's current
            # state, so lineage reflects the inputs as they were at consumption.
            inputs: list[InputRef] = []
            parent_rids: set[RID] = set()
            for ds, consumed_version in ml._input_dataset_pairs(rid):
                version_str = consumed_version
                if version_str is None:
                    try:
                        version_str = str(ds.current_version)
                    except Exception:
                        version_str = None
                producer = ml._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                input_ref = InputRef(
                    kind=ArcInputType.dataset,
                    rid=ds.dataset_rid,
                    version=consumed_version,
                    summary=DatasetSummary(
                        rid=ds.dataset_rid,
                        description=ds.description or None,
                        version=version_str,
                    ),
                    producer_rids=(producer,) if producer else (),
                )
                inputs.append(input_ref)
                self.visitor.on_consumption(consumer_rid=rid, input_ref=input_ref, depth=depth)
                # Never the execution we are currently expanding: if it produced
                # the consumed version of a dataset it also consumed, listing it
                # as its own parent re-enters `in_progress` and flags a false
                # cycle (same reason the member-producers below subtract it).
                if producer and producer != rid:
                    parent_rids.add(producer)
                # Member-producers of the CONSUMED version. Never the execution
                # we are currently expanding: an execution that both consumed
                # this dataset and produced some of its members must not become
                # its own parent (the mid-walk analogue of the root path's
                # version-producer subtraction).
                member_producers = ml._producers_of_dataset_members(ds.dataset_rid, version=consumed_version)
                parent_rids |= member_producers - {rid}
                self.expand_dataset(ds.dataset_rid, consumed_version, depth=depth)

            for asset in record.list_assets(asset_role="Input"):
                producer_rids: tuple[str, ...] = ()
                try:
                    asset_table_obj = ml.model.name_to_table(asset.asset_table)
                    producer_rids = tuple(ml._producers_of_asset(asset.asset_rid, asset_table_obj))
                    producer = producer_rids[0] if producer_rids else None
                    if producer:
                        parent_rids.add(producer)
                except Exception:
                    # If we can't resolve the producer of one asset,
                    # keep walking the rest of the inputs.
                    pass
                input_ref = InputRef(
                    kind=ArcInputType.asset,
                    rid=asset.asset_rid,
                    summary=AssetSummary(
                        rid=asset.asset_rid,
                        filename=asset.filename or None,
                        asset_table=asset.asset_table,
                    ),
                    producer_rids=producer_rids,
                )
                inputs.append(input_ref)
                self.visitor.on_consumption(consumer_rid=rid, input_ref=input_ref, depth=depth)

            # Root-seeded member-producers (and any other externally supplied
            # parents) are merged in before recursion so they get full
            # visited/cycle/depth handling.
            if extra_parent_rids:
                parent_rids |= extra_parent_rids

            node = self.visitor.make_node(rid, summary=execution_summary, inputs=inputs, depth=depth)

            # Recurse on parents.
            parents: list[N] = []
            if depth_remaining is None or depth_remaining > 0:
                next_depth = None if depth_remaining is None else depth_remaining - 1
                for pr in parent_rids:
                    child = self.expand_execution(pr, depth_remaining=next_depth, depth=depth + 1)
                    if child is not None:
                        parents.append(child)
            elif parent_rids:
                # We had parents but depth said stop. Mark depth_capped.
                self.flags["depth_capped"] = True

            self.visitor.attach_parents(node, parents)
            return node
        finally:
            self.in_progress.discard(rid)

    # -- seeding ----------------------------------------------------------

    def run_seed_candidates(self, candidates: list[RID], *, depth_remaining: int | None) -> N | None:
        """Try each seed candidate in order; return the first that expands.

        Each attempt's ``extra_parent_rids`` is every OTHER discovered
        candidate (already-tried seeds and the current seed excluded) so a
        later expansion still surfaces sibling producers as parents.

        Args:
            candidates: Seed execution RIDs, highest priority first.
            depth_remaining: Remaining parent levels to walk.

        Returns:
            The first successfully expanded node, or ``None`` when no
            candidate expanded.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.run_seed_candidates([], depth_remaining=None) is None
            True
        """
        all_candidates = set(candidates)
        tried: set[RID] = set()
        for seed in candidates:
            tried.add(seed)
            node = self.expand_execution(
                seed,
                depth_remaining=depth_remaining,
                depth=0,
                extra_parent_rids=(all_candidates - tried) or None,
            )
            self.visitor.on_seed_candidate(seed, accepted=node is not None)
            if node is not None:
                return node
        return None

    # -- closure queue ----------------------------------------------------

    def enqueue_execution(self, rid: RID, *, depth: int) -> None:
        """Queue an execution discovered by a closure arc for later expansion.

        Args:
            rid: Execution RID to expand later.
            depth: Depth to record for the queued node.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.enqueue_execution("1-ABCD", depth=1)
        """
        if rid in self.visited_global:
            return
        self._queue.append((rid, depth))

    def drain(self, *, depth_remaining: int | None) -> None:
        """Expand every queued execution, under the engine's caps.

        Args:
            depth_remaining: Remaining parent levels each queued expansion may
                walk.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.drain(depth_remaining=0)
        """
        while self._queue:
            rid, depth = self._queue.pop(0)
            if rid in self.visited_global:
                continue
            self.expand_execution(rid, depth_remaining=depth_remaining, depth=depth)

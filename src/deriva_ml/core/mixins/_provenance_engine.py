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
from deriva_ml.execution.provenance import (
    AncestryState,
    ArcInputType,
    ArcKind,
    DatasetVersionFacts,
    GapKind,
    ProvenanceArc,
    ProvenanceAsset,
    ProvenanceDataset,
    ProvenanceExecution,
    ProvenanceGap,
)

if TYPE_CHECKING:
    from deriva_ml.execution.lineage import ExecutionSummary, LineageNode, VersionAttribution
    from deriva_ml.execution.provenance import ParentLink
    from deriva_ml.feature import FeatureProducerRecord

__all__ = ["InputRef", "WalkVisitor", "WalkEngine", "TreeBuilder", "ClosureBuilder"]

# Version key recorded for a dataset whose consumption edge carried no version
# pin. Never a real version label (labels are PEP-440-ish strings), so it can
# never collide with a walked version in ``ProvenanceDataset.versions``.
UNPINNED_VERSION_KEY = "<unpinned>"

# Arc legs whose facts must be read AT the walked version's catalog snapshot.
# Enabling any of them makes ``expand_dataset`` resolve the strict snapshot
# (once, shared); enabling none of them means no snapshot is ever resolved.
_SNAPSHOT_DEPENDENT_ARCS = frozenset({ArcKind.version_authorship, ArcKind.member_binding})

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

    def on_dataset_walked(self, *, dataset_rid: str, version: str) -> None:
        """Observe that one ``(dataset, version)`` entered the walk.

        Fires before any arc leg runs, so a visitor can register the version
        as a member even when every leg then reports a gap. Must be
        idempotent and must not discard facts an earlier leg recorded.
        """
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

    def on_dataset_walked(self, *, dataset_rid: str, version: str) -> None:
        """No-op."""

    def on_dataset_facts(self, *, dataset_rid: str, facts: "DatasetVersionFacts") -> None:
        """No-op."""

    def on_gap(self, kind: "GapKind", subject_rid: str, detail: str) -> None:
        """No-op."""


class ClosureBuilder:
    """Visitor that accumulates a flat provenance closure.

    Implements :class:`WalkVisitor` with the node handle ``N = str`` — the
    execution RID — because a closure is a set of facts keyed by RID, not a
    tree: there is nothing to attach parents to, so ``make_node`` simply
    returns the RID and ``attach_parents`` is a no-op.

    Everything the builder accumulates is deduplicated and merged rather
    than appended blindly:

    - **Arcs** dedupe on :meth:`~deriva_ml.execution.provenance.ProvenanceArc.identity`
      (``kind, consumed_by, input_rid, input_version``), keeping the
      MINIMUM discovery depth and the union of ``evidence``.
    - **Gaps** dedupe on ``(kind, subject_rid, detail)`` — the same hole
      rediscovered on a second path is one hole.
    - **Assets** merge producer lists (the first sighting's fetched order
      wins) and accumulate consumers.
    - **Dataset facts** are stored per ``(dataset_rid, version)`` and never
      conflated across versions.

    Example:
        >>> from deriva_ml.execution.provenance import ArcKind
        >>> builder = ClosureBuilder()
        >>> builder.record_arc("1-ABCD", kind=ArcKind.root, depth=0)
        >>> [a.depth for a in builder.arcs_for("1-ABCD")]
        [0]
        >>> builder.record_arc("1-ABCD", kind=ArcKind.root, depth=2)
        >>> [a.depth for a in builder.arcs_for("1-ABCD")]
        [0]
    """

    def __init__(self) -> None:
        """Initialize an empty closure accumulator."""
        self.executions: dict[str, "ExecutionSummary"] = {}
        self.arcs: dict[str, dict[tuple, ProvenanceArc]] = {}
        self.datasets: dict[str, dict[str, DatasetVersionFacts]] = {}
        self.dataset_descriptions: dict[str, str | None] = {}
        self.assets: dict[str, ProvenanceAsset] = {}
        self.gaps: list[ProvenanceGap] = []
        self._gap_keys: set[tuple] = set()

    # -- accumulation primitives ------------------------------------------

    def record_arc(
        self,
        execution_rid: str,
        *,
        kind: ArcKind,
        depth: int,
        consumed_by: str | None = None,
        input_rid: str | None = None,
        input_type: ArcInputType | None = None,
        input_version: str | None = None,
        evidence: "list[FeatureProducerRecord] | None" = None,
    ) -> None:
        """Record one arc onto ``execution_rid``, merging on arc identity.

        A rediscovery of the same identity keeps the minimum ``depth`` and
        unions ``evidence`` instead of appending a second arc.

        Args:
            execution_rid: The execution the arc attaches to.
            kind: Why this execution is in the closure.
            depth: Hops from the root at which the arc was discovered.
            consumed_by: Consuming execution RID (consumption arcs).
            input_rid: The concrete input RID.
            input_type: Whether ``input_rid`` is a dataset or an asset.
            input_version: The pinned dataset version consumed, if any.
            evidence: ``member_binding`` justification records.

        Example:
            >>> from deriva_ml.execution.provenance import ArcKind
            >>> builder = ClosureBuilder()
            >>> builder.record_arc("1-ABCD", kind=ArcKind.root, depth=0)
            >>> len(builder.arcs_for("1-ABCD"))
            1
        """
        arc = ProvenanceArc(
            kind=kind,
            consumed_by=consumed_by,
            input_rid=input_rid,
            input_type=input_type,
            input_version=input_version,
            evidence=list(evidence or []),
            depth=depth,
        )
        bucket = self.arcs.setdefault(execution_rid, {})
        existing = bucket.get(arc.identity())
        if existing is None:
            bucket[arc.identity()] = arc
            return
        if arc.depth < existing.depth:
            existing.depth = arc.depth
        for record in arc.evidence:
            if record not in existing.evidence:
                existing.evidence.append(record)

    def arcs_for(self, execution_rid: str) -> list[ProvenanceArc]:
        """Return the deterministic arc list recorded for ``execution_rid``.

        Sorted by ``(kind, input_rid or "", consumed_by or "")`` per the
        design's determinism rule. Each arc's ``evidence`` is itself sorted
        by ``(feature_name, element_type, execution_rid or "")``.

        Args:
            execution_rid: The execution whose arcs to read back.

        Returns:
            The sorted arcs; empty when the execution has none.

        Example:
            >>> ClosureBuilder().arcs_for("1-ABCD")
            []
        """
        arcs = sorted(
            self.arcs.get(execution_rid, {}).values(),
            key=lambda a: (str(a.kind), a.input_rid or "", a.consumed_by or ""),
        )
        for arc in arcs:
            arc.evidence = sorted(
                arc.evidence,
                key=lambda r: (r.feature_name, r.element_type, r.execution_rid or ""),
            )
        return arcs

    # -- tree construction (degenerate: a closure has no tree) ------------

    def make_node(self, rid: str, *, summary: "ExecutionSummary", inputs: list[InputRef], depth: int) -> str:
        """Return the execution RID; the closure keys facts, not nodes."""
        return rid

    def make_cycle_node(self, rid: str, *, depth: int) -> str:
        """Return the execution RID for a cycle marker."""
        return rid

    def make_duplicate_node(self, rid: str, *, depth: int) -> str:
        """Return the execution RID for a diamond-duplicate marker."""
        return rid

    def attach_parents(self, node: str, parents: list[str]) -> None:
        """No-op: closure membership is flat, so there is nothing to attach."""

    # -- seeding ----------------------------------------------------------

    def on_seed_candidate(self, rid: str, *, accepted: bool) -> None:
        """No-op: seed-candidate iteration is a tree-shaping concern."""

    # -- closure hooks ----------------------------------------------------

    def on_execution(self, rid: str, *, summary: "ExecutionSummary", depth: int) -> None:
        """Record an execution's summary as a closure member."""
        self.executions.setdefault(rid, summary)

    def on_consumption(self, *, consumer_rid: str, input_ref: InputRef, depth: int) -> None:
        """Record one consumption edge: its arcs, and its input's facts.

        Every recorded producer of the input gets a ``consumption`` arc
        naming the concrete input, so a multi-producer asset attributes the
        consumption to all of them (spec §6.2) rather than to a first match.

        ``depth`` is the CONSUMER's depth; the arc records the PRODUCER's
        depth (``depth + 1``), so an arc's depth always answers "how many
        hops from the root is the execution this arc attaches to" — the same
        question ``ArcKind.root``'s depth 0 answers.
        """
        if input_ref.kind == ArcInputType.asset:
            self.register_asset(input_ref, consumer_rid)
        else:
            self.dataset_descriptions.setdefault(
                input_ref.rid,
                getattr(input_ref.summary, "description", None),
            )
        for producer in input_ref.producer_rids:
            if producer == consumer_rid:
                # An execution that both produced and consumed the same input
                # is not its own provenance ancestor.
                continue
            self.record_arc(
                producer,
                kind=ArcKind.consumption,
                depth=depth + 1,
                consumed_by=consumer_rid,
                input_rid=input_ref.rid,
                input_type=input_ref.kind,
                input_version=input_ref.version,
            )

    def register_asset(self, input_ref: InputRef, consumer_rid: str | None = None) -> None:
        """Merge one asset into the closure's asset map.

        Shared by the consumed-asset path (which passes the consuming
        execution) and the Asset-root path (which passes ``None``, because a
        root asset is a closure member by being the root, not by being
        consumed).

        Args:
            input_ref: The asset's reference, carrying its summary and ALL
                recorded producers in fetched order.
            consumer_rid: The consuming execution, or None for a root asset.

        Example:
            >>> from deriva_ml.execution.provenance import ArcInputType
            >>> builder = ClosureBuilder()
            >>> builder.register_asset(InputRef(kind=ArcInputType.asset, rid="1-ABCD"))
            >>> builder.assets["1-ABCD"].consumed_by
            []
        """
        from deriva_ml.execution.lineage import AssetSummary

        summary = input_ref.summary
        if not isinstance(summary, AssetSummary):
            summary = AssetSummary(rid=input_ref.rid, filename=None, asset_table="")
        entry = self.assets.get(input_ref.rid)
        if entry is None:
            entry = ProvenanceAsset(asset=summary, producers=[], consumed_by=[])
            self.assets[input_ref.rid] = entry
        # Fetched order of the FIRST sighting is authoritative; later
        # sightings only contribute producers the first one did not see.
        for producer in input_ref.producer_rids:
            if producer not in entry.producers:
                entry.producers.append(producer)
        if consumer_rid is not None and consumer_rid not in entry.consumed_by:
            entry.consumed_by.append(consumer_rid)

    def on_version_author(
        self, *, dataset_rid: str, version: str, attribution: "VersionAttribution", depth: int
    ) -> None:
        """Record one walked version's author (Task 11 fills the arc leg)."""
        facts = self.dataset_facts(dataset_rid, version)
        if attribution not in facts.version_authors:
            facts.version_authors.append(attribution)

    def on_binding_record(self, *, dataset_rid: str, version: str, record: "FeatureProducerRecord", depth: int) -> None:
        """No-op: the closure builder records binding arcs directly (see
        ``WalkEngine._expand_member_bindings``), so this hook has nothing
        additional to accumulate."""

    def on_parent_link(self, *, dataset_rid: str, link: "ParentLink") -> None:
        """Record one snapshot-resolved parent hop (Task 13 fills the leg)."""
        facts = self.dataset_facts(dataset_rid, link.child_version)
        if link not in facts.parents:
            facts.parents.append(link)

    def on_dataset_walked(self, *, dataset_rid: str, version: str) -> None:
        """Register a walked version as a closure member, non-destructively."""
        self.dataset_facts(dataset_rid, version)

    def on_dataset_facts(self, *, dataset_rid: str, facts: DatasetVersionFacts) -> None:
        """Store the assembled facts for one dataset version."""
        self.datasets.setdefault(dataset_rid, {})[facts.version] = facts

    def dataset_facts(self, dataset_rid: str, version: str) -> DatasetVersionFacts:
        """Return (creating if needed) the facts record for one version.

        Args:
            dataset_rid: The dataset the facts belong to.
            version: The version label the facts describe.

        Returns:
            The mutable :class:`DatasetVersionFacts` for that version.

        Example:
            >>> builder = ClosureBuilder()
            >>> builder.dataset_facts("1-ABCD", "1.0.0").ancestry_state
            <AncestryState.not_walked: 'not_walked'>
        """
        versions = self.datasets.setdefault(dataset_rid, {})
        facts = versions.get(version)
        if facts is None:
            facts = DatasetVersionFacts(
                version=version,
                parents=[],
                ancestry_state=AncestryState.not_walked,
                is_source=None,
                origin_recorded=None,
                version_authors=[],
            )
            versions[version] = facts
        return facts

    def on_gap(self, kind: "GapKind", subject_rid: str, detail: str) -> None:
        """Record an honest gap, deduped on ``(kind, subject_rid, detail)``."""
        key = (kind, subject_rid, detail)
        if key in self._gap_keys:
            return
        self._gap_keys.add(key)
        self.gaps.append(ProvenanceGap(kind=kind, subject_rid=subject_rid, detail=detail))

    # -- finalize ---------------------------------------------------------

    def build_executions(self) -> dict[str, ProvenanceExecution]:
        """Assemble the closure's execution map, deterministically ordered.

        Only executions that were actually expanded (i.e. have a resolved
        summary) become members: an arc recorded against a RID that never
        resolved is represented by its ``unresolved_rid`` gap, not by a
        half-empty member.

        Returns:
            RID-keyed :class:`ProvenanceExecution` records, key-sorted.

        Example:
            >>> ClosureBuilder().build_executions()
            {}
        """
        return {
            rid: ProvenanceExecution(execution=self.executions[rid], arcs=self.arcs_for(rid))
            for rid in sorted(self.executions)
        }

    def build_datasets(self) -> dict[str, ProvenanceDataset]:
        """Assemble the closure's dataset map, deterministically ordered.

        Returns:
            RID-keyed :class:`ProvenanceDataset` records, key-sorted, each
            with version-sorted facts whose ``parents`` are sorted by
            ``parent_rid``.

        Example:
            >>> ClosureBuilder().build_datasets()
            {}
        """
        out: dict[str, ProvenanceDataset] = {}
        for rid in sorted(self.datasets):
            versions: dict[str, DatasetVersionFacts] = {}
            for version in sorted(self.datasets[rid]):
                facts = self.datasets[rid][version]
                facts.parents = sorted(facts.parents, key=lambda p: p.parent_rid)
                versions[version] = facts
            out[rid] = ProvenanceDataset(
                rid=rid,
                description=self.dataset_descriptions.get(rid),
                versions=versions,
            )
        return out

    def build_assets(self) -> dict[str, ProvenanceAsset]:
        """Assemble the closure's asset map, deterministically ordered.

        During accumulation, ``producers`` keeps fetched order (RIDs carry
        no ordering semantics on their own, and fetched order is the only
        honest record of what the catalog returned — see
        :meth:`register_asset`). At finalize, both ``producers`` and
        ``consumed_by`` are sorted for output determinism: the fetched-order
        requirement only governs accumulation and gap logic (e.g. which
        sighting's order "wins" when an asset is seen more than once), which
        has already happened by the time this runs.

        Returns:
            RID-keyed :class:`ProvenanceAsset` records, key-sorted.

        Example:
            >>> ClosureBuilder().build_assets()
            {}
        """
        out: dict[str, ProvenanceAsset] = {}
        for rid in sorted(self.assets):
            entry = self.assets[rid]
            entry.producers = sorted(entry.producers)
            entry.consumed_by = sorted(entry.consumed_by)
            out[rid] = entry
        return out

    def build_gaps(self) -> list[ProvenanceGap]:
        """Return the recorded gaps in a deterministic order.

        Returns:
            Gaps sorted by ``(kind, subject_rid, detail)``.

        Example:
            >>> ClosureBuilder().build_gaps()
            []
        """
        return sorted(self.gaps, key=lambda g: (str(g.kind), g.subject_rid, g.detail))

    def finalize(
        self,
    ) -> tuple[
        dict[str, ProvenanceExecution],
        dict[str, ProvenanceDataset],
        dict[str, ProvenanceAsset],
        list[ProvenanceGap],
    ]:
        """Assemble every closure collection, fully sorted, in one call.

        This is the single determinism seam (spec §4): every sorted
        collection in the closure — ``executions`` / ``datasets`` / ``assets``
        key order, ``ProvenanceDataset.versions`` (by version label),
        ``gaps``, each execution's ``arcs``, each arc's ``evidence``, each
        dataset version's ``parents``, and each asset's ``producers`` /
        ``consumed_by`` — is sorted here, so ``model_dump()`` output is byte-
        identical across runs regardless of the order the walk discovered
        things in. Call it exactly once, after the walk (and any post-walk
        gap sweep, e.g. dangling-arc detection) has finished mutating the
        builder — the individual ``build_*`` methods it delegates to are
        idempotent, but gaps recorded after this call would not appear in
        the returned ``gaps`` list.

        Returns:
            ``(executions, datasets, assets, gaps)`` — the four sorted
            collections ``ProvenanceClosure`` is constructed from.

        Example:
            >>> ClosureBuilder().finalize()
            ({}, {}, {}, [])
        """
        return (
            self.build_executions(),
            self.build_datasets(),
            self.build_assets(),
            self.build_gaps(),
        )


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
        closure_mode: When True the engine reports what the tree path
            silently swallows — sentinel terminations, RIDs that fail to
            resolve mid-walk, unpinned dataset inputs, and workflow-less
            executions all become visitor gaps, and every recorded asset
            producer (not just the first) becomes a parent. Lineage leaves
            this False so its observable contract stays byte-identical.

    Attributes:
        flags: ``cycle_detected`` / ``depth_capped`` / ``walked_complete``.
        executions_visited: Count of distinct executions expanded.
        datasets_visited: Count of distinct dataset versions expanded.
        cap_hit: True once a cap stopped an expansion.
        truncated: Execution RIDs the execution cap refused to expand. These
            are resolvable — they were reached and then dropped for budget,
            not because anything about them was broken — so a consumer must
            not report them as unresolved. ``cap_hit`` is their explanation.

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
    closure_mode: bool = False

    flags: dict[str, bool] = field(init=False)
    visited_global: set[RID] = field(init=False)
    in_progress: set[RID] = field(init=False)
    datasets_expanded: set[tuple[str, str | None]] = field(init=False)
    cap_hit: bool = field(init=False, default=False)
    truncated: set[RID] = field(init=False)
    _queue: list[tuple[RID, int]] = field(init=False)
    _unpinned_quarantined: set[tuple[str, str | None]] = field(init=False)
    _ancestry_path: set[str] = field(init=False)

    def __init__(
        self,
        ml: Any,
        visitor: "WalkVisitor[N]",
        *,
        arcs: frozenset[ArcKind] = frozenset(),
        max_executions: int = 500,
        dataset_budget: int | None = None,
        closure_mode: bool = False,
    ) -> None:
        """Initialize the engine; see the class docstring for arguments."""
        self.ml = ml
        self.visitor = visitor
        self.arcs = arcs
        self.max_executions = max_executions
        self.dataset_budget = dataset_budget
        self.closure_mode = closure_mode

        self.flags = {"cycle_detected": False, "depth_capped": False, "walked_complete": True}
        self.visited_global = set()
        self.in_progress = set()
        self.datasets_expanded = set()
        self.cap_hit = False
        self.truncated = set()
        self._queue = []
        self._unpinned_quarantined = set()
        # Dataset RIDs on the ancestry branch currently being walked. NOT the
        # memo: a diamond ancestry (two children sharing an ancestor) is legal
        # and must expand the shared ancestor once, so only a revisit while
        # still ON the path is a cycle.
        self._ancestry_path = set()
        self._sentinel_resolved = False
        self._sentinel_rid: RID | None = None

    # -- sentinel ---------------------------------------------------------

    def sentinel_rid(self) -> RID | None:
        """RID of the unknown-provenance Execution sentinel, memoized per walk.

        Resolved lazily and cached (including a ``None`` answer) so a walk
        that touches many candidate producers pays at most one lookup.

        Returns:
            The sentinel Execution's RID, or None when the catalog has none.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._sentinel_resolved
            False
        """
        if not self._sentinel_resolved:
            self._sentinel_rid = self.ml._sentinel_execution_rid_or_none()
            self._sentinel_resolved = True
        return self._sentinel_rid

    def is_sentinel(self, rid: RID) -> bool:
        """True when ``rid`` is the unknown-provenance Execution sentinel.

        Always False outside closure mode: lineage filters the sentinel in
        the mixin's candidate list, and the engine must not change the tree
        path's behavior.

        Args:
            rid: Execution RID to classify.

        Returns:
            Whether ``rid`` is the sentinel.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder(), closure_mode=False)
            >>> engine.is_sentinel("1-ABCD")
            False
        """
        if not self.closure_mode:
            return False
        return rid is not None and rid == self.sentinel_rid()

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
        return bool(self.arcs & _SNAPSHOT_DEPENDENT_ARCS)

    def expand_dataset(self, dataset_rid: str, version: str | None, *, depth: int) -> None:
        """Expand the dataset-side arcs for one consumed dataset version.

        Arc-gated and memoized per ``(dataset_rid, version)``. A walk that is
        neither in closure mode nor has dataset arcs enabled (the lineage
        case) returns immediately, so the lineage path pays nothing for
        closure machinery.

        ``version=None`` (an unpinned edge) reports an ``unpinned_input``
        gap plus ``not_walked`` facts and does **no** snapshot-dependent
        work — the quarantine of spec §6.1. An unpinned edge is a
        *quarantine*, not an expansion: it is memoized under its own key so
        the gap fires once, but it deliberately does **not** consume the
        dataset budget, because nothing was walked.

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
        if not (self.closure_mode or self._dataset_arcs_enabled):
            return

        key = (dataset_rid, version)

        if version is None:
            # Unpinned edge: honest gap, and no snapshot-dependent work is
            # possible because there is no version to pin a snapshot to.
            # Tracked in its own memo set so the quarantine never counts
            # toward datasets_visited or the dataset budget.
            if key in self._unpinned_quarantined:
                return
            self._unpinned_quarantined.add(key)
            self.visitor.on_gap(
                GapKind.unpinned_input,
                dataset_rid,
                "consumption edge recorded no dataset version; no snapshot-dependent provenance was walked for it",
            )
            self.visitor.on_dataset_walked(dataset_rid=dataset_rid, version=UNPINNED_VERSION_KEY)
            return

        if key in self.datasets_expanded:
            return

        if self.dataset_budget is not None and len(self.datasets_expanded) >= self.dataset_budget:
            # ``cap_hit`` (and therefore ``traversal_complete``) is what
            # reports a partial closure. ``walked_complete`` is deliberately
            # NOT touched: it is the LINEAGE result's flag, describing the
            # execution walk, and lineage never runs the dataset legs at all.
            # Conflating the two would make a dataset-budget stop
            # indistinguishable from an execution-cap stop in the lineage
            # result, and would let ``traversal_complete`` be (wrongly)
            # derived from ``walked_complete``.
            self.cap_hit = True
            return

        self.datasets_expanded.add(key)
        # A walked (dataset, version) is a closure member in its own right.
        # Its facts start "not walked" and are filled in by the dataset arc
        # legs (authorship, bindings, ancestry).
        self.visitor.on_dataset_walked(dataset_rid=dataset_rid, version=version)

        # Every snapshot-dependent leg (authorship here; bindings and ancestry
        # in the sibling legs) shares ONE strict snapshot resolution: resolved
        # here, once per walked (dataset, version), and handed down. A walk
        # that cannot pin the snapshot does no snapshot-dependent work at all
        # — reading live state would let post-snaptime facts leak into a
        # pinned closure, which is exactly what the strict resolver exists to
        # prevent. Resolved lazily: a walk with no snapshot-dependent leg
        # enabled must not pay for (or fail on) a snapshot it never reads.
        if not (self.arcs & _SNAPSHOT_DEPENDENT_ARCS):
            return
        snapshot_catalog = self._strict_snapshot_or_gap(dataset_rid, version)
        if snapshot_catalog is None:
            return

        if ArcKind.version_authorship in self.arcs:
            self._expand_version_authorship(dataset_rid, version, snapshot_catalog, depth=depth)

        if ArcKind.member_binding in self.arcs:
            self._expand_member_bindings(dataset_rid, version, depth=depth)

        # Ancestry (§6.5) has no ArcKind of its own — parents contribute to
        # the closure through their OWN authorship/binding arcs, discovered by
        # the recursive `expand_dataset` below, so there is nothing for an
        # "ancestry" arc kind to attach to. It is gated on ``closure_mode``
        # instead: ``ancestry_state`` / ``is_source`` / ``parents`` live on
        # ``DatasetVersionFacts``, which only a closure visitor keeps. The leg
        # runs INSIDE the snapshot-dependent block, so it inherits the same
        # quarantines the sibling legs have: a lineage walk (no dataset arcs,
        # no closure mode) never reaches here, and neither does an unpinned
        # edge or a version whose strict snapshot did not resolve.
        if self.closure_mode:
            self._expand_ancestry(dataset_rid, version, depth=depth)

    def _strict_snapshot_or_gap(self, dataset_rid: str, version: str) -> Any:
        """Resolve the strict version snapshot, or report a chain break.

        Args:
            dataset_rid: The walked dataset.
            version: The walked version.

        Returns:
            The snapshot-bound catalog, or ``None`` when it could not be
            resolved (a ``snapshot_chain_break`` gap has then been emitted).

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine._strict_snapshot_or_gap)
            True
        """
        from deriva_ml.core.exceptions import SnapshotUnavailable

        try:
            return self.ml.lookup_dataset(dataset_rid).strict_version_snapshot_catalog(version)
        except (SnapshotUnavailable, DerivaMLException) as exc:
            self.visitor.on_gap(
                GapKind.snapshot_chain_break,
                dataset_rid,
                f"version {version} has no resolvable catalog snapshot, so authorship read skipped "
                f"and no snapshot-dependent provenance was walked for it: {exc}",
            )
            return None

    def _expand_version_authorship(self, dataset_rid: str, version: str, snapshot_catalog: Any, *, depth: int) -> None:
        """Attribute the walked version and every version recorded before it.

        Spec §6.3: for a walked ``D@vN``, the authors of versions **up to and
        including** ``vN`` in the RCT-primary total order enter the closure;
        later versions' authors do not, because a pinned closure must not
        report authors who acted after the snaptime it is pinned to. The rows
        are read AT the snapshot, so even the historical rows are the ones
        the catalog recorded then, not their live rewrites.

        Args:
            dataset_rid: The walked dataset.
            version: The walked version — the inclusive upper bound.
            snapshot_catalog: Catalog bound to that version's snaptime.
            depth: Walk depth of the consuming execution.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine._expand_version_authorship)
            True
        """
        from deriva_ml.core.mixins.execution import _version_row_sort_key
        from deriva_ml.execution.lineage import VersionAttribution

        rows = sorted(
            self.ml._dataset_version_rows_at(dataset_rid, version, snapshot_catalog),
            key=_version_row_sort_key,
        )

        # Truncate AFTER the walked version's row. Comparing by label (not by
        # sort key) keeps the bound anchored to the row the caller actually
        # pinned, even if two rows normalize to the same key.
        cutoff = next((index for index, row in enumerate(rows) if row.get("Version") == version), None)
        if cutoff is None:
            self.visitor.on_gap(
                GapKind.version_unresolvable,
                dataset_rid,
                f"walked version {version} has no Dataset_Version row at its own snapshot; "
                "no authorship could be bounded",
            )
            return
        bounded = rows[: cutoff + 1]

        # Resolve every author's summary in ONE batched fetch for the whole
        # leg. `VersionAttribution.execution` is contractually None only when
        # no author is recorded or the author could not be resolved, so a
        # resolved closure member that carried None would be misread as
        # unresolvable. Absence from the mapping IS the "could not resolve"
        # answer, and is preserved as None.
        summaries = self.ml._execution_summaries([row.get("Execution") for row in bounded])

        facts = None
        for index, row in enumerate(bounded):
            author_rid = row.get("Execution")
            self.visitor.on_version_author(
                dataset_rid=dataset_rid,
                version=version,
                attribution=VersionAttribution(
                    version=row.get("Version") or "",
                    execution_rid=author_rid,
                    execution=summaries.get(author_rid) if author_rid else None,
                    description=row.get("Description"),
                ),
                depth=depth,
            )
            if index == 0:
                # The ORIGIN row: `origin_recorded` answers "is a real,
                # non-sentinel execution recorded as having created this
                # dataset", read at this snapshot.
                facts = self._facts_for(dataset_rid, version)
                origin_recorded = bool(author_rid) and not self.is_sentinel(author_rid)
                if facts is not None:
                    facts.origin_recorded = origin_recorded
                if not origin_recorded:
                    # A missing or sentinel-attributed origin row is its own
                    # gap kind — distinct from `no_version_author` /
                    # `sentinel_origin` below, which flag the ROW's author.
                    # `origin_unrecorded` flags that the DATASET's origin
                    # specifically (the first row at this snapshot) carries
                    # no real provenance, which a consumer scanning gaps by
                    # kind needs to find without inspecting every row gap.
                    self.visitor.on_gap(
                        GapKind.origin_unrecorded,
                        dataset_rid,
                        f"{dataset_rid} has no real (non-sentinel) origin recorded "
                        f"at version {row.get('Version')} as observed at this snapshot",
                    )

            if not author_rid:
                self.visitor.on_gap(
                    GapKind.no_version_author,
                    dataset_rid,
                    f"Dataset_Version row for {row.get('Version')} records no authoring Execution",
                )
                continue
            if self.is_sentinel(author_rid):
                # Recorded, but recorded as unknown: an honest gap, never a
                # closure member (sentinel discipline, §6.1).
                self.visitor.on_gap(
                    GapKind.sentinel_origin,
                    author_rid,
                    f"version {row.get('Version')} of {dataset_rid} is authored by the "
                    "unknown-provenance Execution sentinel",
                )
                continue

            self._record_arc(
                author_rid,
                kind=ArcKind.version_authorship,
                depth=depth + 1,
                input_rid=dataset_rid,
                input_type=ArcInputType.dataset,
                input_version=row.get("Version"),
            )
            self.enqueue_or_truncate(author_rid, depth=depth + 1)

        if facts is not None:
            self.visitor.on_dataset_facts(dataset_rid=dataset_rid, facts=facts)

    def _expand_member_bindings(self, dataset_rid: str, version: str, *, depth: int) -> None:
        """Attribute the executions that bound feature values onto a walked
        dataset version's members (spec §6.4).

        Runs the internal diagnostic-returning binding scan
        (``ml._find_feature_producers_impl``) exactly once per walked
        ``(dataset_rid, version)`` — the caller (``expand_dataset``) already
        memoizes on that key, so this method is never invoked twice for the
        same pair. Each record naming an execution becomes a
        ``member_binding`` arc carrying the record as evidence; a record with
        no execution is a ``null_binding_execution`` gap; each diagnostic
        collected during the scan is a ``binding_scan_failed`` gap. Neither
        kind of hole suppresses the other records' arcs (degrade-with-honesty
        — a partial scan still reports what it found).

        Args:
            dataset_rid: The walked dataset.
            version: The walked version the scan is scoped to.
            depth: Walk depth of the consuming execution (or 0 for the root).

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine._expand_member_bindings)
            True
        """
        records, diagnostics = self.ml._find_feature_producers_impl(dataset_rid, version=version)

        for record in records:
            self.visitor.on_binding_record(dataset_rid=dataset_rid, version=version, record=record, depth=depth)

            author_rid = record.execution_rid
            if not author_rid:
                self.visitor.on_gap(
                    GapKind.null_binding_execution,
                    dataset_rid,
                    f"feature '{record.feature_name}' on element '{record.element_type}' has bound values "
                    "with no recorded Execution",
                )
                continue
            if self.is_sentinel(author_rid):
                # Recorded, but recorded as unknown: an honest gap, never a
                # closure member (sentinel discipline, §6.1) — same rule the
                # authorship leg applies to a sentinel-authored origin.
                self.visitor.on_gap(
                    GapKind.sentinel_origin,
                    author_rid,
                    f"feature '{record.feature_name}' on element '{record.element_type}' of "
                    f"{dataset_rid}@{version} is bound by the unknown-provenance Execution sentinel",
                )
                continue

            self._record_arc(
                author_rid,
                kind=ArcKind.member_binding,
                depth=depth + 1,
                input_rid=dataset_rid,
                input_type=ArcInputType.dataset,
                input_version=version,
                evidence=[record],
            )
            self.enqueue_or_truncate(author_rid, depth=depth + 1)

        for diagnostic in diagnostics:
            self.visitor.on_gap(
                GapKind.binding_scan_failed,
                diagnostic.subject,
                f"{diagnostic.kind}: {diagnostic.detail}",
            )

    def _expand_ancestry(self, dataset_rid: str, version: str, *, depth: int) -> None:
        """Walk the snapshot-strict parent chain of one walked version (§6.5).

        Reads ``strict_parents_at(version)`` — the snapshot-anchored
        ``Dataset_Dataset`` hop, which resolves the strict snapshot itself
        (the real ``Dataset`` method's shape; the engine's earlier resolution
        gates whether the leg runs at all, it does not substitute for it).
        Each returned row becomes a
        :class:`~deriva_ml.execution.provenance.ParentLink` recorded on this
        version's facts, and each parent whose ``parent_version_then``
        resolved is recursed through :meth:`expand_dataset` — which runs ALL
        of the parent's legs, so an ancestor's version authors and member
        binders enter the closure exactly as a consumed dataset's would.

        Three holes are reported as ``snapshot_chain_break`` gaps:

        - the parents read itself raising (no resolvable snapshot at this
          hop): ``ancestry_state=chain_break``, ``is_source=None``, branch
          stops;
        - a row whose ``parent_version_then`` is ``None``: the link is still
          recorded (the hop is a real schema fact) but that branch cannot be
          walked, because there is no version to pin a snapshot to;
        - a parent already on the ACTIVE ancestry path — a cycle. Note this
          is the active path, not the memo: a diamond (two children sharing
          one ancestor) is legal ancestry and must expand the shared ancestor
          once, not report a cycle.

        Args:
            dataset_rid: The walked dataset.
            version: The walked version whose snaptime anchors the hop.
            depth: Walk depth of the consuming execution.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine._expand_ancestry)
            True
        """
        from deriva_ml.core.exceptions import SnapshotUnavailable

        facts = self._facts_for(dataset_rid, version)

        try:
            rows = self.ml.lookup_dataset(dataset_rid).strict_parents_at(version)
        except (SnapshotUnavailable, DerivaMLException) as exc:
            self.visitor.on_gap(
                GapKind.snapshot_chain_break,
                dataset_rid,
                f"parents of version {version} could not be read at its snapshot, "
                f"so the ancestry chain stops here: {exc}",
            )
            if facts is not None:
                # `is_source` is only meaningful once ancestry resolved; a
                # broken chain means "unknown", which is spelled None.
                facts.ancestry_state = AncestryState.chain_break
                facts.is_source = None
                self.visitor.on_dataset_facts(dataset_rid=dataset_rid, facts=facts)
            return

        # This dataset is on the active path for exactly as long as its own
        # ancestors are being walked below it — that window is what makes a
        # parent naming it back a cycle, while a sibling branch reaching it
        # after the window closed is an ordinary diamond.
        already_on_path = dataset_rid in self._ancestry_path
        self._ancestry_path.add(dataset_rid)
        try:
            self._walk_parent_rows(dataset_rid, version, rows, depth=depth)
        finally:
            if not already_on_path:
                self._ancestry_path.discard(dataset_rid)

        if facts is not None:
            facts.ancestry_state = AncestryState.resolved
            facts.is_source = not rows
            self.visitor.on_dataset_facts(dataset_rid=dataset_rid, facts=facts)

    def _walk_parent_rows(self, dataset_rid: str, version: str, rows: list, *, depth: int) -> None:
        """Record and recurse one walked version's parent rows.

        Split out of :meth:`_expand_ancestry` so the active-path bookkeeping
        reads as one ``try``/``finally`` around the whole descent rather than
        being interleaved with the per-row logic.

        Args:
            dataset_rid: The walked (child) dataset.
            version: The child version whose snaptime anchored the read.
            rows: ``strict_parents_at``'s rows, each with ``parent_rid`` and
                ``parent_version_then``.
            depth: Walk depth of the consuming execution.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._walk_parent_rows("1-ABCD", "1.0.0", [], depth=0) is None
            True
        """
        from deriva_ml.execution.provenance import ParentLink

        for row in rows:
            parent_rid = row.get("parent_rid")
            parent_version_then = row.get("parent_version_then")
            self.visitor.on_parent_link(
                dataset_rid=dataset_rid,
                link=ParentLink(
                    parent_rid=parent_rid,
                    child_version=version,
                    parent_version_then=parent_version_then,
                ),
            )

            if parent_version_then is None:
                # The hop is recorded (it is a real schema fact) but cannot be
                # walked: without a version there is no snapshot to pin to,
                # and reading the parent live would leak post-snaptime state.
                self.visitor.on_gap(
                    GapKind.snapshot_chain_break,
                    parent_rid,
                    f"parent of {dataset_rid}@{version} has no resolvable version at that snaptime, "
                    "so its own ancestry could not be walked",
                )
                continue

            if parent_rid in self._ancestry_path:
                # A cycle in the ancestry graph. Detected on the ACTIVE PATH,
                # never on the memo: revisiting an already-expanded ancestor
                # off-path is an ordinary diamond, which must expand once.
                self.visitor.on_gap(
                    GapKind.snapshot_chain_break,
                    parent_rid,
                    f"ancestry cycle: {parent_rid} is already on the ancestry path being walked "
                    f"from {dataset_rid}@{version}, so the chain stops here",
                )
                continue

            self.expand_dataset(parent_rid, parent_version_then, depth=depth + 1)

    def _facts_for(self, dataset_rid: str, version: str) -> "DatasetVersionFacts | None":
        """Return the visitor's mutable facts record, when it keeps one.

        A tree-building visitor has no facts store; the closure builder does.

        Args:
            dataset_rid: The walked dataset.
            version: The walked version.

        Returns:
            The facts record, or ``None`` for a visitor that keeps none.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._facts_for("1-ABCD", "1.0.0") is None
            True
        """
        getter = getattr(self.visitor, "dataset_facts", None)
        return getter(dataset_rid, version) if getter is not None else None

    def _record_arc(self, execution_rid: RID, **kwargs: Any) -> None:
        """Record one arc on the visitor, when the visitor accumulates arcs.

        The tree visitor keeps no arcs, so this is a no-op there — the same
        shape as :meth:`_facts_for`.

        Args:
            execution_rid: The execution the arc attaches to.
            **kwargs: Forwarded to
                :meth:`ClosureBuilder.record_arc` verbatim.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._record_arc("1-ABCD", kind=ArcKind.root, depth=0) is None
            True
        """
        recorder = getattr(self.visitor, "record_arc", None)
        if recorder is not None:
            recorder(execution_rid, **kwargs)

    def enqueue_or_truncate(self, rid: RID, *, depth: int) -> None:
        """Queue ``rid``, or record it as budget-truncated when it cannot be.

        An execution the engine declines to enqueue because the execution
        budget is already exhausted is still perfectly resolvable — it was
        reached and dropped for budget. Recording it in ``truncated`` is what
        stops a later dangling-arc sweep from calling it ``unresolved_rid``.

        A RID that is already expanded or already QUEUED costs no additional
        budget: it holds one slot, not one per sighting. Re-offering it is a
        dedup (keeping the minimum depth), never a budget event — a shared
        version author, one migration execution authoring two walked
        datasets' versions, is the ordinary case, and charging it twice would
        report a fully-expanded execution as truncated and a complete closure
        as capped.

        Args:
            rid: Execution RID discovered through a non-tree arc.
            depth: Depth to record for the queued node.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder(), max_executions=0)
            >>> engine.enqueue_or_truncate("1-ABCD", depth=1)
            >>> "1-ABCD" in engine.truncated
            True
        """
        if rid in self.visited_global:
            return
        if any(queued_rid == rid for queued_rid, _ in self._queue):
            # Already holding a slot: dedup / lower the recorded depth.
            self.enqueue_execution(rid, depth=depth)
            return
        if len(self.visited_global) + len(self._queue) >= self.max_executions:
            self.flags["walked_complete"] = False
            self.cap_hit = True
            self.truncated.add(rid)
            return
        self.enqueue_execution(rid, depth=depth)

    # -- asset facts ------------------------------------------------------

    def report_asset_producers(
        self,
        asset_rid: str,
        producer_rids: tuple[str, ...],
        *,
        resolution_failed: bool,
    ) -> None:
        """Emit the closure's producer-multiplicity gaps for one input asset.

        Spec §6.2: zero recorded producers is a ``no_asset_producer`` gap;
        more than one is malformed under the current writer contract, so all
        are reported as producers **and** a ``multiple_asset_producers`` gap
        is raised. A producer lookup that raised is reported as
        ``unresolved_rid`` rather than silently reading as "zero producers".

        Args:
            asset_rid: The consumed asset's RID.
            producer_rids: Every recorded producer, in fetched order.
            resolution_failed: True when producer resolution raised.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder(), closure_mode=True)
            >>> engine.report_asset_producers("1-ABCD", (), resolution_failed=False) is None
            True
        """
        if resolution_failed:
            self.visitor.on_gap(
                GapKind.unresolved_rid,
                asset_rid,
                "producer resolution failed for this input asset",
            )
            return
        if not producer_rids:
            self.visitor.on_gap(
                GapKind.no_asset_producer,
                asset_rid,
                "no Execution has an Output association with this asset",
            )
        elif len(producer_rids) > 1:
            self.visitor.on_gap(
                GapKind.multiple_asset_producers,
                asset_rid,
                f"{len(producer_rids)} executions record an Output association "
                "with this asset; exactly one is expected",
            )

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

        # Sentinel discipline (closure only): the unknown-provenance sentinel
        # is never a closure member and is never expanded — encountering it
        # terminates that branch with an honest gap naming what it stopped.
        if self.is_sentinel(rid):
            self.visitor.on_gap(
                GapKind.sentinel_origin,
                rid,
                "provenance terminates at the unknown-provenance Execution sentinel",
            )
            return None

        # Cycle on the active path: do not expand, set flag, return a
        # leaf-style marker.
        if rid in self.in_progress:
            self.flags["cycle_detected"] = True
            return self.visitor.make_cycle_node(rid, depth=depth)

        # Diamond DAG: this execution was already expanded somewhere
        # else in the tree. Mark and don't recurse.
        if rid in self.visited_global:
            return self.visitor.make_duplicate_node(rid, depth=depth)

        # Defensive cap on total expansions. Record WHICH execution was
        # dropped: it is perfectly resolvable, so a consumer must be able to
        # tell "truncated for budget" from "could not be resolved".
        if len(self.visited_global) >= self.max_executions:
            self.flags["walked_complete"] = False
            self.cap_hit = True
            self.truncated.add(rid)
            return None

        # Look up the execution and its inputs. Deliberately per-node (not
        # batched): the walk discovers parents one node at a time, so there
        # is no batch to fetch until the node is expanded.
        try:
            record = ml.lookup_execution(rid)
        except DerivaMLException as exc:
            # An input pointed at an Execution that no longer exists;
            # treat as missing rather than failing the whole walk. The tree
            # path stays silent (byte-compat); the closure reports the hole.
            if self.closure_mode:
                self.visitor.on_gap(
                    GapKind.unresolved_rid,
                    rid,
                    f"recorded execution RID could not be resolved: {exc}",
                )
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

            if wf_summary is None and self.closure_mode:
                # Spec §6.6: the closure identifies the workflow record and
                # its checksum; an execution with no workflow link is a hole
                # in that identification, not a walk failure.
                self.visitor.on_gap(
                    GapKind.no_workflow,
                    rid,
                    "execution has no Workflow record; code identity is unrecorded",
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
                member_producers = member_producers - {rid}
                parent_rids |= member_producers
                if ArcKind.member_production in self.arcs:
                    # Ruling 7: a mid-walk member producer is in the closure
                    # because it produced a MEMBER of this consumed
                    # dataset@version — the asset analogue of member_binding,
                    # and a reason no other arc kind records. Without it these
                    # executions would sit in the closure with empty arcs,
                    # breaking the model's "arcs record every reason" promise.
                    # (Root-path member-fallback seeds are NOT this case: they
                    # are the seed, and keep ``ArcKind.root``.)
                    for member_producer in sorted(member_producers):
                        self._record_arc(
                            member_producer,
                            kind=ArcKind.member_production,
                            depth=depth + 1,
                            input_rid=ds.dataset_rid,
                            input_type=ArcInputType.dataset,
                            input_version=consumed_version,
                        )
                self.expand_dataset(ds.dataset_rid, consumed_version, depth=depth)

            for asset in record.list_assets(asset_role="Input"):
                producer_rids: tuple[str, ...] = ()
                resolution_failed = False
                try:
                    asset_table_obj = ml.model.name_to_table(asset.asset_table)
                    producer_rids = tuple(ml._producers_of_asset(asset.asset_rid, asset_table_obj))
                    producer = producer_rids[0] if producer_rids else None
                    if producer:
                        parent_rids.add(producer)
                    if self.closure_mode:
                        # Spec §6.2: the closure follows ALL Output rows,
                        # never first-match. The tree keeps its
                        # single-producer display choice (`producer` above)
                        # untouched.
                        parent_rids |= set(producer_rids) - {rid}
                except Exception:
                    # If we can't resolve the producer of one asset,
                    # keep walking the rest of the inputs.
                    resolution_failed = True
                if self.closure_mode:
                    self.report_asset_producers(
                        asset.asset_rid,
                        producer_rids,
                        resolution_failed=resolution_failed,
                    )
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

            # Recurse on parents. Closure mode sorts so recorded arc depths
            # are deterministic across runs (set iteration order is
            # PYTHONHASHSEED-dependent); lineage keeps raw set order because
            # its golden captures pin the historical behavior.
            parents: list[N] = []
            ordered_parents = sorted(parent_rids) if self.closure_mode else parent_rids
            if depth_remaining is None or depth_remaining > 0:
                next_depth = None if depth_remaining is None else depth_remaining - 1
                for pr in ordered_parents:
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

        Already-expanded and already-queued RIDs are dropped, so a diamond
        that rediscovers the same execution through several arcs enqueues it
        once. The queue keeps the SHALLOWEST pending depth for a RID, so a
        later, shallower discovery is not recorded at the deeper depth.

        Args:
            rid: Execution RID to expand later.
            depth: Depth to record for the queued node.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.enqueue_execution("1-ABCD", depth=1)
            >>> engine.enqueue_execution("1-ABCD", depth=1)
            >>> len(engine._queue)
            1
        """
        if rid in self.visited_global:
            return
        for index, (queued_rid, queued_depth) in enumerate(self._queue):
            if queued_rid == rid:
                if depth < queued_depth:
                    self._queue[index] = (rid, depth)
                return
        self._queue.append((rid, depth))

    def drain(self, *, depth_remaining: int | None) -> None:
        """Expand every queued execution, under the engine's caps.

        Expanding a queued execution can itself enqueue more work (its own
        consumption parents feed back through the visitor), so the loop runs
        until the queue is genuinely empty. Once the execution cap is hit the
        drain stops and leaves the remaining queue unexpanded — ``cap_hit``
        is what tells the caller the closure is partial.

        Args:
            depth_remaining: Remaining parent levels each queued expansion may
                walk.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.drain(depth_remaining=0)
        """
        while self._queue:
            if len(self.visited_global) >= self.max_executions:
                # Budget exhausted with real work still queued: the
                # traversal is provably incomplete, and everything still
                # queued was truncated rather than unresolvable.
                self.flags["walked_complete"] = False
                self.cap_hit = True
                self.truncated |= {queued_rid for queued_rid, _ in self._queue}
                return
            rid, depth = self._queue.pop(0)
            if rid in self.visited_global:
                continue
            self.expand_execution(rid, depth_remaining=depth_remaining, depth=depth)

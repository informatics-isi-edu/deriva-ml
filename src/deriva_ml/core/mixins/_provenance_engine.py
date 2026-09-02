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
    >>> builder.on_gap("no_workflow", f"1-{1:04X}", "no workflow recorded") is None
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.provenance import (
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

# Worker pool size for a round's binding scans (#391 C3). Each scan is an
# independent, snapshot-bound, HTTP-bound read, so the bound is about being
# a polite catalog client rather than about CPU: a closure round rarely has
# more than a handful of datasets, and a small fixed pool keeps the burst
# predictable for the server.
_BINDING_SCAN_WORKERS = 8

# Concurrency for a frontier's execution prefetch (#391b). Deliberately an
# internal module constant with an env override rather than a public
# ``lookup_provenance`` parameter: it tunes *how* the walk talks to the
# catalog, never *what* the closure contains, and adding it to the public
# signature would imply the answer can depend on it (it cannot — see
# ``test_worker_count_one_equals_default``). Set
# ``DERIVA_ML_PROVENANCE_WORKERS=1`` to force fully sequential prefetch,
# which is also the cheapest way to bisect a suspected concurrency
# problem in the field.
_EXPANSION_WORKERS_DEFAULT = 8
_EXPANSION_WORKERS_ENV = "DERIVA_ML_PROVENANCE_WORKERS"


def _expansion_workers() -> int:
    """Resolve the frontier prefetch concurrency for this process.

    Reads :data:`_EXPANSION_WORKERS_ENV`, falling back to
    :data:`_EXPANSION_WORKERS_DEFAULT`. A value that is not a positive
    integer is ignored rather than raising: a malformed tuning knob must
    never break a provenance query.

    Returns:
        The number of concurrent execution prefetches to allow, at least 1.

    Example:
        >>> _expansion_workers() >= 1
        True
    """
    import os

    raw = os.environ.get(_EXPANSION_WORKERS_ENV)
    if raw is None:
        return _EXPANSION_WORKERS_DEFAULT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _EXPANSION_WORKERS_DEFAULT
    return value if value >= 1 else _EXPANSION_WORKERS_DEFAULT


@dataclass(frozen=True)
class ExecutionReadout:
    """Everything one execution's expansion needs, read off the catalog.

    The read side of an expansion — ``lookup_execution``, its input-dataset
    pairs and each pair's producer / member-producer scans, its input assets
    and each asset's producers — is pure I/O against a snapshot of catalog
    state. Bundling it into one immutable record is what lets a frontier of
    executions be fetched CONCURRENTLY while every piece of engine and
    builder mutation stays on the main thread, applied in sorted-RID order.

    A readout is produced by :meth:`WalkEngine.read_execution` (which never
    touches engine or visitor state) and consumed by
    :meth:`WalkEngine.expand_execution` (which does all the mutating).

    Attributes:
        rid: The execution the readout describes.
        record: The ``ExecutionRecord``, or ``None`` when lookup failed.
        lookup_error: The exception ``lookup_execution`` raised, if any.
        dataset_inputs: One entry per input-dataset edge, in fetch order:
            ``(dataset, consumed_version, producer_rid, member_producers,
            member_scan_error)``. ``member_scan_error`` is the exception a
            member-producer scan raised, deferred so the gap is emitted on
            the main thread by the same wrapper that always emitted it.
        asset_inputs: One entry per input asset, in fetch order:
            ``(asset, producer_rids, resolution_failed)``.

    Example:
        >>> readout = ExecutionReadout(rid=f"1-{1:04X}", record=None)
        >>> readout.dataset_inputs
        ()
    """

    rid: RID
    record: Any = None
    lookup_error: BaseException | None = None
    dataset_inputs: tuple[tuple[Any, str | None, str | None, set, BaseException | None], ...] = ()
    asset_inputs: tuple[tuple[Any, tuple[str, ...], bool], ...] = ()


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
        >>> ref = InputRef(kind=ArcInputType.asset, rid=f"1-{1:04X}")
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
        >>> node = builder.make_cycle_node(f"1-{1:04X}", depth=2)
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
        >>> builder.record_arc(f"1-{1:04X}", kind=ArcKind.root, depth=0)
        >>> [a.depth for a in builder.arcs_for(f"1-{1:04X}")]
        [0]
        >>> builder.record_arc(f"1-{1:04X}", kind=ArcKind.root, depth=2)
        >>> [a.depth for a in builder.arcs_for(f"1-{1:04X}")]
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
            >>> builder.record_arc(f"1-{1:04X}", kind=ArcKind.root, depth=0)
            >>> len(builder.arcs_for(f"1-{1:04X}"))
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

    def drop_binding_arcs_for(self, dataset_rid: str) -> int:
        """Remove every ``member_binding`` arc recorded against ``dataset_rid``.

        Ruling 9 reports binding evidence **as of the maximum walked
        snaptime**. When a later round walks a higher pin of an
        already-scanned dataset, the dataset is rescanned at the new maximum
        and the previous, lower as-of view must be REPLACED — not merged.
        Monotonicity guarantees the newer view is a superset, so keeping both
        would duplicate every surviving record and leave arcs carrying two
        different ``input_version`` labels for one dataset, which is exactly
        the "evidence as of one snaptime" contract broken.

        Only arcs whose ``input_rid`` is this dataset are dropped: an
        execution that also has consumption or authorship arcs keeps them,
        and an execution that had ONLY the dropped arc is re-recorded by the
        rescan if it still binds (and correctly disappears if it does not).

        Args:
            dataset_rid: The rescanned dataset.

        Returns:
            How many arcs were dropped.

        Example:
            >>> ClosureBuilder().drop_binding_arcs_for(f"1-{1:04X}")
            0
        """
        dropped = 0
        for bucket in self.arcs.values():
            stale = [
                identity
                for identity, arc in bucket.items()
                if arc.kind == ArcKind.member_binding and arc.input_rid == dataset_rid
            ]
            for identity in stale:
                del bucket[identity]
                dropped += 1
        return dropped

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
            >>> ClosureBuilder().arcs_for(f"1-{1:04X}")
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
            >>> builder.register_asset(InputRef(kind=ArcInputType.asset, rid=f"1-{1:04X}"))
            >>> builder.assets[f"1-{1:04X}"].consumed_by
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
        ``WalkEngine._apply_binding_scan``), so this hook has nothing
        additional to accumulate."""

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
            >>> builder.dataset_facts(f"1-{1:04X}", "1.0.0").version
            '1.0.0'
        """
        versions = self.datasets.setdefault(dataset_rid, {})
        facts = versions.get(version)
        if facts is None:
            facts = DatasetVersionFacts(
                version=version,
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
            with version-sorted facts.

        Example:
            >>> ClosureBuilder().build_datasets()
            {}
        """
        out: dict[str, ProvenanceDataset] = {}
        for rid in sorted(self.datasets):
            versions: dict[str, DatasetVersionFacts] = {}
            for version in sorted(self.datasets[rid]):
                versions[version] = self.datasets[rid][version]
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
        ``gaps``, each execution's ``arcs``, each arc's ``evidence``, and
        each asset's ``producers`` / ``consumed_by`` — is sorted here, so
        ``model_dump()`` output is byte-
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
    _pending_binding_pins: dict[str, dict[str, int]] = field(init=False)
    _binding_scanned: dict[str, str] = field(init=False)
    _readouts: dict[RID, ExecutionReadout] = field(init=False)
    _worker_handle_pool: "list[Any] | None" = field(init=False, default=None)

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
        # Ruling 9: binding scans are DEFERRED, not run inline. Each walked
        # pin registers here as ``dataset_rid -> {version: shallowest depth}``;
        # ``run_pending_binding_scans`` then scans each dataset ONCE at its
        # maximum walked snaptime.
        self._pending_binding_pins = {}
        # ``dataset_rid -> the version its binding scan was run AT``. Not a
        # bare "scanned" set: a later round can walk a HIGHER pin of an
        # already-scanned dataset, and ruling 9 promises evidence as of the
        # MAXIMUM walked snaptime — so the dataset must be rescanned when
        # its max pin advances past what was scanned.
        self._binding_scanned = {}
        # Frontier prefetch cache (#391b): ``rid -> ExecutionReadout`` for
        # executions whose read side has already been fetched (concurrently)
        # but not yet applied. ``expand_execution`` consumes and removes an
        # entry; a miss falls back to reading inline, so every code path
        # works with an empty cache.
        self._readouts = {}
        # Per-worker catalog handles, built lazily on the first frontier
        # wide enough to be worth parallelizing. ``None`` means "not built
        # yet"; an empty list means "tried, and this ml cannot supply them"
        # — the difference is what stops a failed attempt from being retried
        # on every frontier.
        self._worker_handle_pool = None
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
            >>> engine.is_sentinel(f"1-{1:04X}")
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
        gap and does **no** snapshot-dependent work — the quarantine of
        spec §6.1. An unpinned edge is a
        *quarantine*, not an expansion: it is memoized under its own key so
        the gap fires once, but it deliberately does **not** consume the
        dataset budget, because nothing was walked.

        ``Dataset_Dataset`` containment is deliberately NOT walked (ruling 8,
        #389): provenance is execution-mediated, so a parent dataset reaches
        the closure through the consumption arc of the execution that authored
        the child version — never through the structural containment edge.

        The authorship leg runs HERE, per walked pin (it is bounded *by* the
        pin, so each pin's answer differs). The member-binding leg does not:
        ruling 9 (#391) makes binding evidence monotone across a dataset's
        versions, so this only REGISTERS the pin and
        :meth:`run_pending_binding_scans` later scans the dataset once, at
        its maximum walked snaptime.

        Args:
            dataset_rid: RID of the consumed dataset.
            version: The version actually consumed, or ``None`` when the edge
                carried no version pin.
            depth: Walk depth of the consuming execution.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder(), arcs=frozenset())
            >>> engine.expand_dataset(f"1-{1:04X}", "1.0.0", depth=0) is None
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
        # Its facts start empty and are filled in by the dataset arc legs
        # (authorship, bindings).
        self.visitor.on_dataset_walked(dataset_rid=dataset_rid, version=version)

        # Both snapshot-dependent legs (authorship and member bindings) share
        # ONE strict snapshot resolution: resolved here, once per walked
        # (dataset, version), and handed down. A walk that cannot pin the
        # snapshot does no snapshot-dependent work at all — reading live state
        # would let post-snaptime facts leak into a pinned closure, which is
        # exactly what the strict resolver exists to prevent. Resolved lazily:
        # a walk with no snapshot-dependent leg enabled must not pay for (or
        # fail on) a snapshot it never reads.
        if not (self.arcs & _SNAPSHOT_DEPENDENT_ARCS):
            return
        snapshot_catalog = self._strict_snapshot_or_gap(dataset_rid, version)
        if snapshot_catalog is None:
            return

        if ArcKind.version_authorship in self.arcs:
            self._expand_version_authorship(dataset_rid, version, snapshot_catalog, depth=depth)

        if ArcKind.member_binding in self.arcs:
            # Ruling 9: DEFERRED, never inline. Binding evidence is monotone
            # across a dataset's versions, so only the maximum walked pin's
            # scan is run — and which pin that is cannot be known until the
            # walk has finished discovering pins. Register and move on;
            # ``run_pending_binding_scans`` resolves the maximum and scans.
            pins = self._pending_binding_pins.setdefault(dataset_rid, {})
            if version not in pins or depth < pins[version]:
                pins[version] = depth

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

    # -- deferred binding scans (ruling 9) ---------------------------------

    def max_walked_pin(self, dataset_rid: str, versions: "set[str] | frozenset[str]") -> str:
        """Pick the maximum-snaptime version among a dataset's walked pins.

        Ruling 9 (#391) makes ONE binding scan per dataset — at the newest
        walked pin — exact for the requested closure, because new dataset
        versions only ADD feature values. "Newest" is decided by the
        catalog's own version-history order, not by parsing the label and
        not by assuming anything about snaptime string formats:
        ``_version_row_sort_key`` (RCT-primary, #367) is the total order the
        rest of the library already uses for "which version was recorded
        later", and under the monotone contract a later-recorded version's
        snaptime is later. A walked pin the live history has no row for (a
        row rewritten or removed since) falls back to its raw label and is
        grouped AFTER every row-backed pin, so a pin the history does not
        know about never silently outranks one it does.

        Args:
            dataset_rid: The dataset whose walked pins to compare.
            versions: The walked version labels to choose among.

        Returns:
            The label of the maximum-snaptime walked pin.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.max_walked_pin(f"1-{1:04X}", {"1.0.0"})
            '1.0.0'
        """
        from deriva_ml.core.mixins.execution import _version_row_sort_key

        labels = sorted(versions)
        if len(labels) == 1:
            # One pin: no history read is needed to know which is newest.
            return labels[0]

        keys: dict[str, tuple] = {}
        try:
            for row in self.ml._dataset_version_rows(dataset_rid):
                label = row.get("Version")
                if label in versions and label not in keys:
                    keys[label] = _version_row_sort_key(row)
        except Exception:  # noqa: BLE001 — a history read failure must not
            # break the walk; the label fallback below still yields a total,
            # deterministic order.
            keys = {}

        def order(label: str) -> tuple:
            row_key = keys.get(label)
            # (row-backed flag, row key, label): a pin the history knows
            # about is compared by its recorded order; an unknown pin is
            # grouped separately (flag 1, i.e. after) and broken by label.
            return (0, row_key, "") if row_key is not None else (1, (), label)

        return max(labels, key=order)

    def _pin_advanced(self, dataset_rid: str, scanned: str, candidate: str) -> bool:
        """True when ``candidate`` is a strictly NEWER pin than ``scanned``.

        Decided by the same total order :meth:`max_walked_pin` uses, so
        "which pin is newer" has exactly one definition in the engine.
        Implemented by asking that method to choose between the two: if it
        picks ``candidate`` over ``scanned``, the pin advanced.

        Args:
            dataset_rid: The dataset whose pins to compare.
            scanned: The version its binding scan was last run at.
            candidate: The maximum currently-walked pin.

        Returns:
            Whether a rescan is required.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._pin_advanced(f"1-{1:04X}", "1.0.0", "1.0.0")
            False
        """
        if scanned == candidate:
            return False
        return self.max_walked_pin(dataset_rid, {scanned, candidate}) == candidate

    def _run_leased(self, batch: list, worker: Any) -> list:
        """Run ``worker(item, handle)`` over ``batch``, one leased handle each.

        The shared execution path for every concurrent READ the walk makes.
        Binding scans used to run on their own ``ThreadPoolExecutor`` calling
        seams on ``self.ml``, which is the same unsynchronized
        ``requests.Session`` / ``_snapshot_cache`` hazard the expansion lease
        eliminated; routing both legs through this one runner closes that and
        retires the deferred "two concurrency frameworks" item at the same
        time.

        Handles are leased from the SAME pool the expansion prefetch uses, so
        the pool's depth remains the single global concurrency bound: an
        expansion frontier and a scan round can never together exceed it,
        because both draw from one queue.

        Falls back to running the batch sequentially, in order, whenever no
        handle pool can be built — which is also what
        ``DERIVA_ML_PROVENANCE_WORKERS=1`` produces, so that knob now
        serializes scans as well as expansion and is a true
        sequential-equivalence control for BOTH legs.

        Args:
            batch: Work items, already in deterministic (sorted) order.
            worker: ``(item, handle) -> result``; must be read-only.

        Returns:
            Results in the SAME order as ``batch``, never completion order.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._run_leased([], lambda item, handle: item)
            []
        """
        if not batch:
            return []
        handles = self._worker_handles()
        if len(batch) == 1 or not handles:
            return [worker(item, self.ml) for item in batch]

        import asyncio

        async def drive() -> list:
            loop = asyncio.get_running_loop()
            leases: asyncio.Queue = asyncio.Queue()
            for handle in handles:
                leases.put_nowait(handle)

            async def one(item: Any) -> Any:
                handle = await leases.get()
                try:
                    return await loop.run_in_executor(None, lambda: worker(item, handle))
                finally:
                    # Released even on failure — a leaked lease would shrink
                    # the pool for the rest of the walk and deadlock it.
                    leases.put_nowait(handle)

            return list(await asyncio.gather(*(one(item) for item in batch)))

        from deriva_ml.core.async_helpers import run_async

        return run_async(drive())

    def run_pending_binding_scans(self) -> bool:
        """Run one ROUND of deferred binding scans; True if any scan ran.

        Ruling 9 (#391): binding evidence is monotone across a dataset's
        versions, so the scan runs once per DATASET — at the maximum walked
        snaptime — rather than once per walked ``(dataset, version)`` pin.
        Older walked pins get no binding arcs of their own; what the newest
        scan reports is the closure's as-of view of that dataset's bindings.

        A round's scans are **independent snapshot-bound reads** — each
        resolves its own snapshot handle and touches no shared state — so
        they run concurrently on a bounded worker pool (#391 C3). The
        catalog reads dominate a scan's wall time and are almost entirely
        sequential HTTP otherwise.

        Determinism is preserved by construction: workers only *read*, and
        every result is applied to the visitor **single-threaded, in sorted
        dataset order**, after the whole batch has completed. The
        ``ClosureBuilder`` is never touched from a worker, so which scan
        finished first cannot influence the closure. A scan that raises is
        an honest ``binding_scan_failed`` gap for that dataset alone — it
        never aborts the round or loses the other datasets' results.

        Because a scan can discover executions that walk NEW datasets and
        pins, the caller runs this in rounds (drain, scan, repeat) until it
        returns False.

        Returns:
            True when at least one dataset was scanned this round, meaning
            more work may now be queued.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.run_pending_binding_scans()
            False
        """
        batch = []
        for dataset_rid in sorted(self._pending_binding_pins):
            pins = self._pending_binding_pins[dataset_rid]
            if not pins:
                continue
            version = self.max_walked_pin(dataset_rid, set(pins))
            scanned = self._binding_scanned.get(dataset_rid)
            if scanned is not None and not self._pin_advanced(dataset_rid, scanned, version):
                # Already scanned at this pin (or at a newer one): the
                # existing as-of view already covers what is walked.
                continue
            self._binding_scanned[dataset_rid] = version
            batch.append((dataset_rid, version, pins[version]))

        if not batch:
            return False

        def scan(item: tuple[str, str, int], handle: Any) -> tuple[Any, list[Any], Exception | None]:
            """Read-only worker: never touches the visitor or engine state.

            Runs against a LEASED handle, never ``self.ml``: the scan calls
            the same session-bound seams the expansion reads do, so sharing
            one ``DerivaML`` across concurrent scans is the identical
            unsynchronized-``requests.Session`` / ``_snapshot_cache`` hazard
            the expansion lease exists to prevent (#391b review).
            """
            dataset_rid, version, _depth = item
            try:
                records, diagnostics = handle._find_feature_producers_impl(dataset_rid, version=version)
                return records, list(diagnostics), None
            except Exception as exc:  # noqa: BLE001 — one dataset's failure
                # must not abort the round; it becomes that dataset's gap.
                return [], [], exc

        results = self._run_leased(batch, scan)

        # SINGLE-THREADED apply, in sorted dataset order (``batch`` is built
        # sorted and the runner preserves input order), so recorded depths
        # and every ordering derived from them are independent of both
        # discovery order and worker completion order.
        for (dataset_rid, version, depth), (records, diagnostics, error) in zip(batch, results, strict=True):
            if error is not None:
                self.visitor.on_gap(
                    GapKind.binding_scan_failed,
                    dataset_rid,
                    f"binding scan of {dataset_rid}@{version} failed, so its member bindings are unrecorded: {error}",
                )
                continue
            self._apply_binding_scan(dataset_rid, version, records, diagnostics, depth=depth)
        return True

    def _apply_binding_scan(
        self,
        dataset_rid: str,
        version: str,
        records: "list[FeatureProducerRecord]",
        diagnostics: list[Any],
        *,
        depth: int,
    ) -> None:
        """Attribute the executions that bound feature values onto a walked
        dataset's members (spec §6.4), as of ``version``'s snapshot.

        Each record naming an execution becomes a ``member_binding`` arc
        carrying the record as evidence and the SCANNED version label; a
        record with no execution is a ``null_binding_execution`` gap; each
        diagnostic collected during the scan is a ``binding_scan_failed``
        gap. Neither kind of hole suppresses the other records' arcs
        (degrade-with-honesty — a partial scan still reports what it found).

        Kept separate from the scan call itself so the (read-only, possibly
        concurrent) scan and the (strictly single-threaded) mutation of the
        visitor never interleave.

        Args:
            dataset_rid: The scanned dataset.
            version: The maximum walked pin the scan was scoped to.
            records: The scan's binding records.
            diagnostics: The scan's degrade diagnostics.
            depth: Shallowest walk depth at which that pin was reached.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._apply_binding_scan(f"1-{1:04X}", "1.0.0", [], [], depth=0) is None
            True
        """
        # RESCAN at an advanced pin REPLACES the previous as-of view. Ruling
        # 9 reports bindings as of ONE snaptime — the maximum walked — so a
        # lower pin's arcs must not survive alongside the newer scan's.
        # Monotonicity makes the new view a superset, so nothing real is
        # lost; what is dropped is a duplicate carrying a stale
        # ``input_version``.
        dropper = getattr(self.visitor, "drop_binding_arcs_for", None)
        if dropper is not None:
            dropper(dataset_rid)

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
            >>> engine._facts_for(f"1-{1:04X}", "1.0.0") is None
            True
        """
        getter = getattr(self.visitor, "dataset_facts", None)
        return getter(dataset_rid, version) if getter is not None else None

    def _on_consumption(self, *, consumer_rid: RID, input_ref: InputRef, depth: int) -> None:
        """Emit one consumption edge, honoring the ``consumption`` arc gate.

        The edge is always *traversed* — the producer of a consumed input
        is how the walk reaches its parents at all, and gating traversal
        would not narrow the closure so much as break it. What the gate
        governs is whether the edge is RECORDED as a
        :attr:`ArcKind.consumption` arc. A caller who excludes it is asking
        for the root plus the other requested legs, and gets exactly that:
        the executions are still reached, but no consumption arc claims
        them (#391 C4).

        Lineage (``arcs=frozenset()``) is unaffected: ``TreeBuilder``'s
        hook is a no-op either way, so its observable contract is
        untouched.

        Args:
            consumer_rid: The consuming execution.
            input_ref: The concrete input consumed.
            depth: Walk depth of the consumer.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> ref = InputRef(kind=ArcInputType.asset, rid=f"1-{1:04X}")
            >>> engine._on_consumption(consumer_rid=f"2-{1:04X}", input_ref=ref, depth=0) is None
            True
        """
        if ArcKind.consumption not in self.arcs and self.closure_mode:
            # Still register the input as a closure member (a consumed
            # dataset/asset is a member by being consumed, which is not the
            # arc's claim), but record no consumption arc for it.
            if input_ref.kind == ArcInputType.asset:
                register = getattr(self.visitor, "register_asset", None)
                if register is not None:
                    register(input_ref, consumer_rid)
            else:
                # Datasets need the SAME treatment as assets here. Only the
                # asset half was handled, so a consumption-gated closure
                # reported its consumed datasets with ``description=None``
                # — the description is a property of the dataset being a
                # member, not of the consumption ARC, so gating the arc must
                # not blank it. Mirrors ``ClosureBuilder.on_consumption``'s
                # dataset branch exactly.
                descriptions = getattr(self.visitor, "dataset_descriptions", None)
                if descriptions is not None:
                    descriptions.setdefault(
                        input_ref.rid,
                        getattr(input_ref.summary, "description", None),
                    )
            return
        self.visitor.on_consumption(consumer_rid=consumer_rid, input_ref=input_ref, depth=depth)

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
            >>> engine._record_arc(f"1-{1:04X}", kind=ArcKind.root, depth=0) is None
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
            >>> engine.enqueue_or_truncate(f"1-{1:04X}", depth=1)
            >>> f"1-{1:04X}" in engine.truncated
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

    def member_producers_or_gap(self, dataset_rid: str, version: str | None) -> set[RID]:
        """Scan a dataset version's member producers, degrading to a gap.

        ``_producers_of_dataset_members`` reads through the NON-strict
        ``_version_snapshot_catalog``, so a recorded-but-unreadable snapshot
        (garbage-collected, or a schema shape the snaptime cannot serve) can
        raise a raw error out of the scan — before ``expand_dataset``'s
        :meth:`_strict_snapshot_or_gap` ever gets to emit
        ``snapshot_chain_break``. That escapes as a crash from a method whose
        contract is "holes become gaps, never exceptions", so the scan is
        wrapped here instead: both ``SnapshotUnavailable`` and the raw errors
        the non-strict path raises for a broken snapshot degrade to a
        ``snapshot_chain_break`` gap naming the member scan, and the walk
        continues with an empty producer set.

        The catch is scoped to the scan call alone — it never spans the
        caller's own logic, so a bug elsewhere still surfaces as a crash.

        Args:
            dataset_rid: The dataset whose members to scan.
            version: The version pin to scope the scan to, or None for live.

        Returns:
            The distinct member-producing execution RIDs; empty when the scan
            could not be performed (a gap has then been emitted).

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine.member_producers_or_gap)
            True
        """
        from deriva_ml.core.exceptions import SnapshotUnavailable

        try:
            return self.ml._producers_of_dataset_members(dataset_rid, version=version)
        except (SnapshotUnavailable, DerivaMLException, AttributeError, KeyError, ValueError) as exc:
            self.visitor.on_gap(
                GapKind.snapshot_chain_break,
                dataset_rid,
                f"member-producer scan of version {version} could not be performed at its "
                f"catalog snapshot, so member production for it is unrecorded: {exc}",
            )
            return set()

    def member_producers_from(
        self,
        dataset_rid: str,
        version: str | None,
        producers: set[RID],
        error: BaseException | None,
    ) -> set[RID]:
        """Apply :meth:`member_producers_or_gap`'s degrade to an ALREADY-READ scan.

        The scan itself now happens on the read side (:meth:`read_execution`,
        possibly on a worker), which defers whatever it raised instead of
        reporting it. This applies exactly the classification
        :meth:`member_producers_or_gap` applies — the same exception types
        degrade to the same ``snapshot_chain_break`` gap with the same
        wording, and anything else propagates — on the main thread, so gap
        text and gap dedup are unchanged from the inline era.

        Args:
            dataset_rid: The dataset whose members were scanned.
            version: The version pin the scan was scoped to.
            producers: The producer set the scan returned (empty on failure).
            error: The exception the scan raised, or ``None``.

        Returns:
            The producer set, or an empty set when the scan degraded.

        Raises:
            BaseException: ``error`` itself, when it is not one of the kinds
                a broken snapshot is expected to raise.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.member_producers_from(f"1-{1:04X}", "1.0.0", set(), None)
            set()
        """
        from deriva_ml.core.exceptions import SnapshotUnavailable

        if error is None:
            return producers
        if isinstance(error, (SnapshotUnavailable, DerivaMLException, AttributeError, KeyError, ValueError)):
            self.visitor.on_gap(
                GapKind.snapshot_chain_break,
                dataset_rid,
                f"member-producer scan of version {version} could not be performed at its "
                f"catalog snapshot, so member production for it is unrecorded: {error}",
            )
            return set()
        raise error

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
            >>> engine.report_asset_producers(f"1-{1:04X}", (), resolution_failed=False) is None
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

    # -- execution read side (concurrency-safe) ----------------------------

    def read_execution(self, rid: RID) -> ExecutionReadout:
        """Read everything one execution's expansion needs. **Read-only.**

        Performs the whole HTTP-bound half of an expansion —
        ``lookup_execution``, the input-dataset pairs and each pair's
        ``_producer_of_dataset`` / ``_producers_of_dataset_members`` scans,
        the input assets and each asset's ``_producers_of_asset`` — and
        returns the answers as an immutable :class:`ExecutionReadout`.

        **This method must never touch engine or visitor state**, because it
        is what a frontier's prefetch runs concurrently (#391b). Every
        failure it encounters is *captured* into the readout rather than
        reported: gaps are emitted by :meth:`expand_execution` on the main
        thread, at the exact point they were emitted before, so gap identity,
        order, and dedup are unchanged.

        Note that ``member_producers`` is captured RAW here — the ``- {rid}``
        self-subtraction and the closure/lineage wrapper choice both stay in
        :meth:`expand_execution`, since those are semantics, not I/O.

        Args:
            rid: The execution to read.

        Returns:
            The readout; ``lookup_error`` is set (and every input list empty)
            when the execution itself did not resolve.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine.read_execution)
            True
        """
        ml = self.ml
        try:
            record = ml.lookup_execution(rid)
        except DerivaMLException as exc:
            return ExecutionReadout(rid=rid, lookup_error=exc)

        dataset_inputs: list[tuple[Any, str | None, str | None, set, BaseException | None]] = []
        for ds, consumed_version in ml._input_dataset_pairs(rid):
            producer = ml._producer_of_dataset(ds.dataset_rid, version=consumed_version)
            member_producers: set = set()
            member_error: BaseException | None = None
            try:
                member_producers = ml._producers_of_dataset_members(ds.dataset_rid, version=consumed_version)
            except Exception as exc:  # noqa: BLE001 — deferred verbatim to the
                # main thread, where ``member_producers_or_gap``'s existing
                # catch decides whether it degrades to a gap or re-raises.
                member_error = exc
            dataset_inputs.append((ds, consumed_version, producer, member_producers, member_error))

        asset_inputs: list[tuple[Any, tuple[str, ...], bool]] = []
        for asset in record.list_assets(asset_role="Input"):
            producer_rids: tuple[str, ...] = ()
            resolution_failed = False
            try:
                asset_table_obj = ml.model.name_to_table(asset.asset_table)
                producer_rids = tuple(ml._producers_of_asset(asset.asset_rid, asset_table_obj))
            except Exception:  # noqa: BLE001 — same degrade the inline path
                # applied: one unresolvable asset producer never stops the
                # rest of the inputs.
                resolution_failed = True
            asset_inputs.append((asset, producer_rids, resolution_failed))

        return ExecutionReadout(
            rid=rid,
            record=record,
            dataset_inputs=tuple(dataset_inputs),
            asset_inputs=tuple(asset_inputs),
        )

    def _take_readout(self, rid: RID) -> ExecutionReadout:
        """Return ``rid``'s prefetched readout, or read it inline on a miss.

        The cache is a pure optimization: an entry is consumed exactly once,
        and an empty cache degrades to the historical inline read. That is
        what keeps the sequential and prefetched paths byte-identical (and
        what makes ``DERIVA_ML_PROVENANCE_WORKERS=1`` meaningful).

        Args:
            rid: The execution being expanded.

        Returns:
            Its readout.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine._take_readout)
            True
        """
        cached = self._readouts.pop(rid, None)
        return cached if cached is not None else self.read_execution(rid)

    def _prefetchable(self, rid: RID) -> bool:
        """True when ``rid`` may be prefetched for the coming frontier.

        Mirrors the guards :meth:`expand_execution` applies BEFORE it does
        any I/O — sentinel, in-progress cycle, already-visited diamond — so
        the prefetch never issues a request for an execution the expansion
        would have refused to read. Anything the guards would let through is
        prefetchable; the budget check is applied by the caller, which is
        where the remaining-slot arithmetic lives.

        Args:
            rid: A queued execution RID.

        Returns:
            Whether prefetching ``rid`` is both safe and useful.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._prefetchable(f"1-{1:04X}")
            True
        """
        if rid in self._readouts:
            return False
        if rid in self.visited_global or rid in self.in_progress:
            return False
        return not self.is_sentinel(rid)

    async def _prefetch_frontier_async(self, rids: list[RID]) -> None:
        """Fetch ``rids``' read sides concurrently into :attr:`_readouts`.

        Follows the deriva-py asyncio pattern (``asyncio.gather`` under an
        ``asyncio.Semaphore``, as in
        ``deriva.core.asyncio.clone``'s concurrent table copy): a bounded
        number of tasks are in flight at once, and the whole frontier is
        awaited as one gather.

        The per-execution work itself is the EXISTING synchronous mixin
        seams, offloaded with ``loop.run_in_executor``. Keeping the seams
        synchronous is deliberate: the ``_FakeML`` harness, every seam
        contract, and every subclass override keep working untouched, and
        the only thing that changed is who calls them and when.

        **Thread safety: handles are LEASED, never routed.** ``requests.Session``
        is not thread-safe, and neither is the ``_snapshot_cache`` dict a
        ``DerivaML`` keeps, so two concurrent reads must never touch the same
        handle. An earlier version picked a handle by ``hash(rid) % len(pool)``,
        which is a *routing* scheme and not exclusion: with N concurrent tasks
        over N handles, collisions are near-certain (birthday paradox), and a
        collision silently shares an unsynchronized session.

        Handles are therefore taken from an :class:`asyncio.Queue` — acquired
        before the ``run_in_executor`` call and released in a ``finally`` — so
        possession is exclusive for the whole duration of the read. The queue
        doubles as the concurrency bound: its depth IS the number of tasks that
        can be in flight, which makes "pool size >= concurrency" an enforced
        invariant rather than a comment. No separate semaphore is needed, and
        none is used, because a second bound could drift out of step with the
        pool and reintroduce sharing.

        Results are stashed in :attr:`_readouts` and applied later,
        single-threaded, in the queue's own deterministic order; no worker
        touches the engine's sets, the budget, or the visitor.

        A readout whose read raised something the seams do not normally
        raise is simply NOT cached, so that execution falls back to the
        inline read on the main thread and fails (or degrades) exactly where
        it always did.

        Args:
            rids: The frontier's execution RIDs, already budget-limited.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> callable(engine._prefetch_frontier_async)
            True
        """
        import asyncio

        loop = asyncio.get_running_loop()
        handles = self._worker_handles()
        if not handles:
            return

        # The lease pool. Its depth is the ONLY concurrency bound: a task
        # cannot start until it owns a handle, and it owns that handle
        # exclusively until it releases it.
        leases: asyncio.Queue = asyncio.Queue()
        for handle in handles:
            leases.put_nowait(handle)

        async def one(rid: RID) -> tuple[RID, ExecutionReadout | None]:
            handle = await leases.get()
            try:
                worker = self._worker_view(handle)
                return rid, await loop.run_in_executor(None, lambda: worker.read_execution(rid))
            except Exception:  # noqa: BLE001 — an unexpected read failure
                # is not cached; the main thread re-reads inline and the
                # error surfaces from its historical call site.
                return rid, None
            finally:
                # Released even on failure — a leaked lease would shrink the
                # pool for the rest of the walk and eventually deadlock it.
                leases.put_nowait(handle)

        for rid, readout in await asyncio.gather(*(one(rid) for rid in rids)):
            if readout is not None:
                self._readouts[rid] = readout

    def _worker_view(self, handle: Any) -> "WalkEngine[N]":
        """Wrap one LEASED catalog handle in a read-only engine view.

        The caller has already acquired ``handle`` exclusively from the lease
        queue in :meth:`_prefetch_frontier_async`; this only wraps it so
        :meth:`read_execution` (read-only by construction) can be called
        against it. The view holds no walk state of its own — the engine's
        sets, budget and visitor are never reachable from it.

        Deliberately takes the handle rather than choosing one: handle
        selection is the lease queue's job, and any selection logic here
        would be a second, unsynchronized allocator.

        Args:
            handle: The leased catalog handle.

        Returns:
            A read-only engine view bound to that handle.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._worker_view(None).ml is None
            True
        """
        return WalkEngine(
            handle,
            self.visitor,
            arcs=self.arcs,
            max_executions=self.max_executions,
            dataset_budget=self.dataset_budget,
            closure_mode=self.closure_mode,
        )

    def _worker_handles(self) -> list[Any]:
        """Build (once) the pool of per-worker catalog handles.

        One extra ``DerivaML`` connection per worker, constructed up front
        from ``self.ml``'s own connection parameters and its already-parsed
        schema (so no worker re-fetches ``/schema``). Each handle owns its
        own ``requests.Session`` AND its own ``_snapshot_cache`` — both
        unsynchronized, which is exactly why a handle is leased to one
        reader at a time rather than shared.

        **The pool's size IS the concurrency bound.** Every in-flight read
        holds one handle from :meth:`_prefetch_frontier_async`'s lease queue,
        so at most ``len(pool)`` reads run at once and "pool size >=
        concurrency" holds by construction. Nothing else caps concurrency;
        adding a second bound would let the two drift apart and reintroduce
        handle sharing.

        Returns an empty list — meaning "do not parallelize" — whenever
        handles cannot be built: a stubbed ``ml`` in tests, an offline
        instance, a worker count of 1, or any construction failure. The
        caller then reads inline on the main thread. Failing closed here
        costs only speed.

        Returns:
            The handle pool, possibly empty.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine._worker_handles()
            []
        """
        if self._worker_handle_pool is not None:
            return self._worker_handle_pool

        pool: list[Any] = []
        workers = _expansion_workers()
        ml = self.ml
        factory = getattr(ml, "_provenance_worker_handle", None)
        if workers > 1 and callable(factory):
            for _ in range(workers):
                try:
                    handle = factory()
                except Exception:  # noqa: BLE001 — a handle we cannot build
                    # simply means less concurrency, never a failed walk.
                    handle = None
                if handle is None:
                    pool = []
                    break
                pool.append(handle)
        self._worker_handle_pool = pool
        return pool

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
            >>> engine.expand_execution(f"1-{1:04X}", depth_remaining=None) is None
            True
        """
        from deriva_ml.execution.lineage import (
            AssetSummary,
            DatasetSummary,
            ExecutionSummary,
            WorkflowSummary,
        )

        # NOTE: no ``ml = self.ml`` binding here any more. Every catalog read
        # this method used to make inline now lives in ``read_execution``
        # (#391b), which is what makes this method safe to run purely on the
        # main thread while the reads happen concurrently elsewhere.

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

        # Read side of the expansion. Prefetched concurrently for a closure
        # frontier (#391b) and read inline otherwise; either way the answers
        # arrive as one immutable ``ExecutionReadout`` and every decision
        # made from them below happens here, on the main thread.
        readout = self._take_readout(rid)
        record = readout.record
        if readout.lookup_error is not None:
            # An input pointed at an Execution that no longer exists;
            # treat as missing rather than failing the whole walk. The tree
            # path stays silent (byte-compat); the closure reports the hole.
            if self.closure_mode:
                self.visitor.on_gap(
                    GapKind.unresolved_rid,
                    rid,
                    f"recorded execution RID could not be resolved: {readout.lookup_error}",
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
            for ds, consumed_version, producer, raw_member_producers, member_error in readout.dataset_inputs:
                version_str = consumed_version
                if version_str is None:
                    try:
                        version_str = str(ds.current_version)
                    except Exception:
                        version_str = None
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
                self._on_consumption(consumer_rid=rid, input_ref=input_ref, depth=depth)
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
                # Closure mode routes the scan through the gap-safe wrapper: an
                # unreadable snapshot must become a ``snapshot_chain_break``,
                # not an exception escaping mid-walk. Lineage keeps the bare
                # call so its observable behavior stays byte-identical.
                if self.closure_mode:
                    member_producers = self.member_producers_from(
                        ds.dataset_rid, consumed_version, raw_member_producers, member_error
                    )
                elif member_error is not None:
                    # Lineage never swallowed this: raise it from the same
                    # place the bare call used to raise from.
                    raise member_error
                else:
                    member_producers = raw_member_producers
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

            for asset, producer_rids, resolution_failed in readout.asset_inputs:
                if not resolution_failed:
                    producer = producer_rids[0] if producer_rids else None
                    if producer:
                        parent_rids.add(producer)
                    if self.closure_mode:
                        # Spec §6.2: the closure follows ALL Output rows,
                        # never first-match. The tree keeps its
                        # single-producer display choice (`producer` above)
                        # untouched.
                        parent_rids |= set(producer_rids) - {rid}
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
                self._on_consumption(consumer_rid=rid, input_ref=input_ref, depth=depth)

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
                # #391b: this node's sorted parent set IS a frontier — every
                # parent's read side is independent of every other's — so
                # fetch them CONCURRENTLY before recursing. The recursion
                # below is unchanged and still strictly sequential, in the
                # same sorted order, so which parent is applied when (and
                # therefore every recorded depth) is untouched; the reads
                # have simply already happened.
                if self.closure_mode:
                    self.prefetch_executions(ordered_parents)
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
            >>> engine.enqueue_execution(f"1-{1:04X}", depth=1)
            >>> engine.enqueue_execution(f"1-{1:04X}", depth=1)
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

    def frontier_rids(self, candidates: "list[RID] | None" = None) -> list[RID]:
        """The frontier's prefetchable RIDs, in order, budget-limited.

        A **frontier** is a set of executions whose read sides are mutually
        independent, so they can be fetched together. The walk has two:

        - one node's SORTED PARENT SET, fetched before the recursion over it;
        - the closure QUEUE's prefix, fetched before a drain round.

        Both are filtered and bounded by the same two rules, and both matter:

        - **Never past the cap.** At most ``max_executions - len(visited)``
          entries are taken, so a frontier can never fetch an execution the
          walk would then refuse for budget. Under a cap the *same* prefix
          the sequential walk would have expanded is the prefix that gets
          fetched, and everything beyond it is left for the cap logic to
          move into ``truncated`` exactly as before. The cap is therefore
          honest in the strict sense: never exceeded *within* a batch, not
          merely reported afterwards.
        - **Walk order, not completion order.** The frontier is a PREFIX of
          the very order the walk will apply things in, so which executions
          happen to be in flight together cannot change which execution is
          applied when.

        Executions the expansion would refuse to read anyway (sentinel,
        cycle, diamond, already prefetched) are filtered by
        :meth:`_prefetchable`.

        Args:
            candidates: The frontier's RIDs in walk order. ``None`` uses the
                closure queue's own order.

        Returns:
            The RIDs to prefetch, in walk order; empty when nothing is
            eligible (which correctly disables the prefetch entirely).

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.frontier_rids()
            []
        """
        if candidates is None:
            candidates = [rid for rid, _depth in self._queue]
        # Every already-prefetched-but-not-yet-applied readout is an
        # execution that will consume a budget slot when it is applied, so
        # it is charged against the remaining budget HERE. Without that,
        # nested frontiers (a parent set prefetched inside an expansion that
        # was itself prefetched) would each measure the budget as if the
        # others did not exist, and the walk would read further ahead than
        # the sequential walk ever would — which is exactly the over-fetch
        # the cap is supposed to forbid.
        remaining = self.max_executions - len(self.visited_global) - len(self._readouts)
        if remaining <= 0:
            return []
        frontier: list[RID] = []
        seen: set[RID] = set()
        for rid in candidates:
            if len(frontier) >= remaining:
                break
            if rid in seen or not self._prefetchable(rid):
                continue
            seen.add(rid)
            frontier.append(rid)
        return frontier

    def prefetch_frontier(self) -> int:
        """Concurrently prefetch the QUEUE's next frontier. Sync entry point.

        Returns:
            How many executions were prefetched.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.prefetch_frontier()
            0
        """
        return self.prefetch_executions(None)

    def prefetch_executions(self, candidates: "list[RID] | None") -> int:
        """Concurrently prefetch one frontier's read side. Sync entry point.

        Bridges the async driver (:meth:`_prefetch_frontier_async`) back to
        this synchronous walk with
        :func:`~deriva_ml.core.async_helpers.run_async`, which is
        notebook-safe: inside a running event loop (Jupyter, papermill) it
        re-enters that loop via ``nest_asyncio`` instead of raising
        ``asyncio.run() cannot be called from a running event loop``. That
        is why ``lookup_provenance`` stays an ordinary sync method.

        A frontier of one is fetched inline, with no loop and no executor:
        there is nothing to overlap, and paying for a loop per singleton
        would make the common shallow walk slower, not faster.

        Args:
            candidates: The frontier in walk order, or ``None`` for the
                closure queue's own frontier.

        Returns:
            How many executions were prefetched.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.prefetch_executions([])
            0
        """
        frontier = self.frontier_rids(candidates)
        if len(frontier) < 2 or _expansion_workers() < 2 or not self._worker_handles():
            # Nothing to overlap (or no safe per-worker handles): let
            # ``_take_readout`` read inline, exactly as the pre-#391b walk
            # did. This is also the path a worker count of 1 takes, which
            # is what makes it a true sequential-equivalence control.
            return 0

        from deriva_ml.core.async_helpers import run_async

        run_async(self._prefetch_frontier_async(frontier))
        return len(frontier)

    def release_unapplied_readouts(self) -> int:
        """Drop prefetched readouts this walk will never apply.

        A readout sitting in :attr:`_readouts` is charged against the
        remaining execution budget by :meth:`frontier_rids`, because applying
        it will consume a slot. When the walk stops for budget, those
        readouts are never applied — so the charge must be released, or every
        subsequent round measures a remaining budget smaller than the real
        one.

        Safe to call at any termination point: a discarded readout is only a
        cache entry, and :meth:`_take_readout` re-reads inline on a miss.

        Returns:
            How many readouts were released.

        Example:
            >>> engine = WalkEngine(ml=None, visitor=TreeBuilder())
            >>> engine.release_unapplied_readouts()
            0
        """
        released = len(self._readouts)
        self._readouts.clear()
        return released

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
            if self.closure_mode:
                # #391b: prefetch the next FRONTIER concurrently before
                # expanding it. Purely a read-ahead — the loop below still
                # pops and applies one execution at a time, in queue order,
                # so nothing about what gets expanded (or in what order)
                # depends on the prefetch having happened.
                self.prefetch_frontier()
            if len(self.visited_global) >= self.max_executions:
                # Budget exhausted with real work still queued: the
                # traversal is provably incomplete, and everything still
                # queued was truncated rather than unresolvable.
                self.flags["walked_complete"] = False
                self.cap_hit = True
                self.truncated |= {queued_rid for queued_rid, _ in self._queue}
                # Readouts prefetched for work this walk will now never
                # apply are DISCARDED, releasing the budget slots they were
                # charged for in ``frontier_rids``. Without this, a walk that
                # terminates on the cap leaves the charge standing forever,
                # so any later round would compute a smaller remaining budget
                # than it actually has and under-fetch (or stop fetching).
                self.release_unapplied_readouts()
                return
            rid, depth = self._queue.pop(0)
            if rid in self.visited_global:
                continue
            self.expand_execution(rid, depth_remaining=depth_remaining, depth=depth)

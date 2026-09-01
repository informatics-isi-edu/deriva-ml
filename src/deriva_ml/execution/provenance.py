"""Pydantic models and StrEnum vocabularies for ``lookup_provenance``.

``lookup_provenance`` computes the full multi-hop closure of executions,
datasets, and assets reachable from a root artifact — as distinct from
``lookup_lineage`` (``execution/lineage.py``), which walks a single
producing-execution chain. The models here compose with the shared
summaries defined in ``lineage.py`` (``ExecutionSummary``,
``DatasetSummary``, ``AssetSummary``, ``VersionAttribution``,
``RootDescriptor``) rather than duplicating them.

See ``docs/superpowers/specs/2026-08-31-lookup-provenance-design.md``
§4 ("Result model") for the field-by-field design rationale — this
module implements those models verbatim.

Idiom notes (from that spec, §4):
    - Result models are Pydantic ``BaseModel`` (serialized across a
      boundary, consistent with the all-Pydantic lineage models).
    - Closed vocabularies deriva-ml itself controls are ``StrEnum``, so
      consumers can dispatch on them with autocomplete and catch typos
      at authoring time. Catalog-sourced open vocabularies (execution
      status, table names) deliberately stay ``str``.

Note on ``RootType``: this enum is now defined in ``lineage.py``
and re-exported here to keep the import flow acyclic (``lineage.py``
no longer imports from ``provenance.py``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deriva_ml.core.definitions import RID
from deriva_ml.execution.lineage import (
    AssetSummary,
    ExecutionSummary,
    RootDescriptor,
    RootType,
    VersionAttribution,
)
from deriva_ml.feature import FeatureProducerRecord

__all__ = [
    "RootType",
    "ArcKind",
    "ArcInputType",
    "AncestryState",
    "GapKind",
    "ParentLink",
    "ProvenanceArc",
    "ProvenanceExecution",
    "DatasetVersionFacts",
    "ProvenanceDataset",
    "ProvenanceAsset",
    "ProvenanceGap",
    "ProvenanceClosure",
]


class ArcKind(StrEnum):
    """Why an execution is a member of a provenance closure.

    Attributes:
        root: The root artifact itself, or its direct producer.
        consumption: The execution produced an input that some closure
            member consumed.
        version_authorship: The execution authored a version (at or
            before the walked version) of a walked dataset.
        member_binding: The execution bound feature values onto a
            walked dataset version's members.

    Example:
        >>> from deriva_ml.execution.provenance import ArcKind
        >>> ArcKind.consumption == "consumption"
        True
    """

    root = "root"
    consumption = "consumption"
    version_authorship = "version_authorship"
    member_binding = "member_binding"


class ArcInputType(StrEnum):
    """Kind of concrete input an arc's ``input_rid`` refers to.

    Example:
        >>> from deriva_ml.execution.provenance import ArcInputType
        >>> ArcInputType.asset == "asset"
        True
    """

    dataset = "dataset"
    asset = "asset"


class AncestryState(StrEnum):
    """How completely a dataset version's parent ancestry was resolved.

    Attributes:
        resolved: The version's parents were fully walked; ``is_source``
            on the owning :class:`DatasetVersionFacts` is meaningful.
        chain_break: Walking the parent chain hit a gap (e.g. an
            unresolvable snapshot); ``is_source`` is unknown.
        not_walked: The version's ancestry was never examined (e.g. it
            fell outside the traversal bound).

    Example:
        >>> from deriva_ml.execution.provenance import AncestryState
        >>> AncestryState.not_walked == "not_walked"
        True
    """

    resolved = "resolved"
    chain_break = "chain_break"
    not_walked = "not_walked"


class GapKind(StrEnum):
    """First-class honest-gap categories surfaced by a provenance walk.

    Each :class:`ProvenanceGap` names one of these kinds. Gaps are
    orthogonal to arcs: an execution's presence (or absence) in the
    closure is governed by arcs, while gaps document places the walk
    could not fully resolve provenance.

    Example:
        >>> from deriva_ml.execution.provenance import GapKind
        >>> len(GapKind)
        12
        >>> GapKind.sentinel_origin == "sentinel_origin"
        True
    """

    sentinel_origin = "sentinel_origin"
    origin_unrecorded = "origin_unrecorded"
    null_binding_execution = "null_binding_execution"
    no_workflow = "no_workflow"
    snapshot_chain_break = "snapshot_chain_break"
    unpinned_input = "unpinned_input"
    version_unresolvable = "version_unresolvable"
    no_version_author = "no_version_author"
    no_asset_producer = "no_asset_producer"
    multiple_asset_producers = "multiple_asset_producers"
    unresolved_rid = "unresolved_rid"
    binding_scan_failed = "binding_scan_failed"


class ParentLink(BaseModel):
    """One snapshot-resolved parent hop of a dataset version.

    Attributes:
        parent_rid: RID of the parent dataset.
        child_version: The child version whose snaptime resolved this
            hop (parent ancestry is always resolved AT a specific child
            snapshot, never conflated across versions).
        parent_version_then: The parent's then-current version at that
            snaptime, or None when that itself is a gap (the parent's
            version could not be resolved at the child's snaptime).

    Example:
        >>> from deriva_ml.execution.provenance import ParentLink
        >>> link = ParentLink(
        ...     parent_rid=f"1-{1:04X}",
        ...     child_version="1.0.0",
        ...     parent_version_then="0.3.0",
        ... )
        >>> link.parent_version_then
        '0.3.0'
    """

    model_config = ConfigDict(extra="forbid")

    parent_rid: RID
    child_version: str
    parent_version_then: str | None = None


class ProvenanceArc(BaseModel):
    """One schema-recorded reason an execution is in the closure.

    Arcs form a set, not a ranked list — there is no "strongest arc"
    (per the design's ruling 4). Identity for dedup purposes is
    :meth:`identity`, which deliberately excludes ``depth`` (rediscovery
    at a shallower depth updates ``depth`` to the minimum rather than
    appending a duplicate arc) and ``input_type``/``evidence`` (derivable
    from / merged on top of the identity fields).

    Attributes:
        kind: Why this arc exists — see :class:`ArcKind`.
        consumed_by: The consuming execution RID (``consumption`` arcs).
        input_rid: The concrete input RID — a dataset or asset.
        input_type: Whether ``input_rid`` is a dataset or an asset.
        input_version: For dataset inputs, the pinned version consumed
            (None means unpinned — see the corresponding gap).
        evidence: ``member_binding`` arcs only — the
            :class:`~deriva_ml.feature.FeatureProducerRecord` rows that
            justify this arc, sorted by
            ``(feature_name, element_type, execution_rid)`` and merged
            (deduped by record equality) when an arc is rediscovered.
        depth: Minimum number of hops from the root at which this arc
            was found.

    Example:
        >>> from deriva_ml.execution.provenance import ArcInputType, ArcKind, ProvenanceArc
        >>> arc = ProvenanceArc(
        ...     kind=ArcKind.consumption,
        ...     consumed_by=f"1-{1:04X}",
        ...     input_rid=f"1-{2:04X}",
        ...     input_type=ArcInputType.dataset,
        ...     input_version="1.0.0",
        ...     depth=1,
        ... )
        >>> arc.identity() == (ArcKind.consumption, f"1-{1:04X}", f"1-{2:04X}", "1.0.0")
        True
        >>> rediscovered = arc.model_copy(update={"depth": 0})
        >>> arc.identity() == rediscovered.identity()
        True
    """

    model_config = ConfigDict(extra="forbid")

    kind: ArcKind
    consumed_by: str | None = None
    input_rid: str | None = None
    input_type: ArcInputType | None = None
    input_version: str | None = None
    evidence: list[FeatureProducerRecord] = Field(default_factory=list)
    depth: int

    def identity(self) -> tuple[ArcKind, str | None, str | None, str | None]:
        """Return the dedup-identity tuple for this arc.

        Two arcs with equal identity tuples represent the same
        discovered fact and are merged (keeping the minimum ``depth``
        and the union of ``evidence``) rather than both appearing in
        :attr:`ProvenanceExecution.arcs`.

        Returns:
            The tuple ``(kind, consumed_by, input_rid, input_version)``.
            ``depth``, ``input_type``, and ``evidence`` are deliberately
            excluded.

        Example:
            >>> from deriva_ml.execution.provenance import ArcKind, ProvenanceArc
            >>> arc = ProvenanceArc(kind=ArcKind.root, depth=0)
            >>> arc.identity()
            (<ArcKind.root: 'root'>, None, None, None)
        """
        return (self.kind, self.consumed_by, self.input_rid, self.input_version)


class ProvenanceExecution(BaseModel):
    """One execution's membership record within a provenance closure.

    Attributes:
        execution: Compact summary of the execution (including workflow
            identity).
        arcs: Every distinct reason this execution is in the closure.
            An unordered typology (per the design's ruling 4); sorted
            for determinism by ``(kind, input_rid or "", consumed_by or "")``.

    Example:
        >>> from deriva_ml.execution.provenance import ArcKind, ProvenanceArc, ProvenanceExecution
        >>> from deriva_ml.execution.lineage import ExecutionSummary
        >>> pe = ProvenanceExecution(
        ...     execution=ExecutionSummary(rid=f"1-{1:04X}", status="Uploaded"),
        ...     arcs=[ProvenanceArc(kind=ArcKind.root, depth=0)],
        ... )
        >>> pe.arcs[0].kind
        <ArcKind.root: 'root'>
    """

    model_config = ConfigDict(extra="forbid")

    execution: ExecutionSummary
    arcs: list[ProvenanceArc] = Field(default_factory=list)


class DatasetVersionFacts(BaseModel):
    """Facts observed AT one dataset version's snapshot.

    Never merged across versions — the same dataset at two versions can
    have different parents, different ancestry resolution, and
    different recorded authors, so each version's facts are kept in
    its own record (see :attr:`ProvenanceDataset.versions`).

    Attributes:
        version: The version label these facts describe.
        parents: Snapshot-resolved parent links (per the design's
            ruling 6), sorted by ``parent_rid``.
        ancestry_state: How completely this version's ancestry was
            resolved — see :class:`AncestryState`.
        is_source: Whether this version has no parents (True), has
            parents (False), or is unknown (None). Meaningful (True or
            False) only when ``ancestry_state == AncestryState.resolved``;
            must be None otherwise, enforced by a validator.
        origin_recorded: Tri-state flag for whether a real (non-sentinel)
            origin execution is recorded, as observed at this snapshot.
        version_authors: Version-attribution entries bounded at this
            version (per the design §6.3).

    Example:
        >>> from deriva_ml.execution.provenance import AncestryState, DatasetVersionFacts
        >>> facts = DatasetVersionFacts(
        ...     version="1.0.0",
        ...     parents=[],
        ...     ancestry_state=AncestryState.resolved,
        ...     is_source=True,
        ...     origin_recorded=True,
        ...     version_authors=[],
        ... )
        >>> facts.is_source
        True
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    parents: list[ParentLink] = Field(default_factory=list)
    ancestry_state: AncestryState
    is_source: bool | None = None
    origin_recorded: bool | None = None
    version_authors: list[VersionAttribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def _is_source_requires_resolved_ancestry(self) -> DatasetVersionFacts:
        """Reject a non-None ``is_source`` unless ancestry was resolved.

        ``is_source`` is only meaningful once the version's parent
        ancestry has actually been walked; any other ``ancestry_state``
        means "unknown," which must be spelled ``None``, not a guessed
        True/False.
        """
        if self.ancestry_state != AncestryState.resolved and self.is_source is not None:
            raise ValueError(
                "is_source must be None unless ancestry_state is "
                f"'resolved' (got ancestry_state={self.ancestry_state!r}, "
                f"is_source={self.is_source!r})"
            )
        return self


class ProvenanceDataset(BaseModel):
    """One dataset's membership record within a provenance closure.

    Attributes:
        rid: Dataset RID.
        description: Live display metadata (current catalog state,
            explicitly labeled as such — not pinned to any walked
            version).
        versions: Per-version facts, keyed by version label. Facts are
            never conflated across versions — see
            :class:`DatasetVersionFacts`.

    Example:
        >>> from deriva_ml.execution.provenance import (
        ...     AncestryState, DatasetVersionFacts, ProvenanceDataset,
        ... )
        >>> ds = ProvenanceDataset(
        ...     rid=f"1-{1:04X}",
        ...     description="Training split",
        ...     versions={
        ...         "1.0.0": DatasetVersionFacts(
        ...             version="1.0.0",
        ...             parents=[],
        ...             ancestry_state=AncestryState.resolved,
        ...             is_source=True,
        ...             origin_recorded=True,
        ...             version_authors=[],
        ...         )
        ...     },
        ... )
        >>> ds.versions["1.0.0"].is_source
        True
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    description: str | None = None
    versions: dict[str, DatasetVersionFacts] = Field(default_factory=dict)


class ProvenanceAsset(BaseModel):
    """One asset's membership record within a provenance closure.

    Attributes:
        asset: Compact summary of the asset.
        producers: ALL producing execution RIDs. Empty means a gap (no
            recorded producer); more than one is also a gap (ambiguous
            producer) — see :class:`GapKind`.
        consumed_by: Closure executions that consumed this asset.

    Example:
        >>> from deriva_ml.execution.provenance import ProvenanceAsset
        >>> from deriva_ml.execution.lineage import AssetSummary
        >>> pa = ProvenanceAsset(
        ...     asset=AssetSummary(rid=f"1-{1:04X}", filename="weights.pt", asset_table="Execution_Asset"),
        ...     producers=[f"1-{2:04X}"],
        ...     consumed_by=[f"1-{3:04X}"],
        ... )
        >>> pa.producers
        ['1-0002']
    """

    model_config = ConfigDict(extra="forbid")

    asset: AssetSummary
    producers: list[str] = Field(default_factory=list)
    consumed_by: list[str] = Field(default_factory=list)


class ProvenanceGap(BaseModel):
    """A first-class, honest gap encountered during a provenance walk.

    Gaps are orthogonal to arcs (per the design's ruling 2): they
    document places the walk could not fully resolve provenance,
    independent of which executions ended up in the closure.

    Attributes:
        kind: The gap category — see :class:`GapKind`.
        subject_rid: RID of the entity the gap concerns.
        detail: Human-readable explanation of what could not be
            resolved and why.

    Example:
        >>> from deriva_ml.execution.provenance import GapKind, ProvenanceGap
        >>> gap = ProvenanceGap(
        ...     kind=GapKind.no_asset_producer,
        ...     subject_rid=f"1-{1:04X}",
        ...     detail="No Execution_Asset row with role=Output found for this asset.",
        ... )
        >>> gap.kind
        <GapKind.no_asset_producer: 'no_asset_producer'>
    """

    model_config = ConfigDict(extra="forbid")

    kind: GapKind
    subject_rid: str
    detail: str


class ProvenanceClosure(BaseModel):
    """Full result returned by ``lookup_provenance``.

    Attributes:
        root: Descriptor of the root artifact (reused from
            ``lineage.py``); ``root.version`` is the resolved pin
            the walk started from.
        executions: Closure member executions, keyed by execution RID.
        datasets: Closure member datasets, keyed by dataset RID; facts
            are recorded per-version inside each entry.
        assets: Closure member assets, keyed by asset RID.
        gaps: Every honest gap encountered during the walk.
        executions_visited: Count of distinct executions visited.
        datasets_visited: Count of distinct datasets visited.
        traversal_complete: False iff any traversal bound was hit.
            NOT a claim of gap-freedom — gap-freedom is ``not gaps``,
            deliberately tracked separately.
        cap_hit: True if a defensive cap (e.g. max executions/datasets)
            stopped the walk before it would have finished naturally.

    Example:
        >>> from deriva_ml.execution.provenance import ProvenanceClosure
        >>> from deriva_ml.execution.lineage import RootDescriptor
        >>> closure = ProvenanceClosure(
        ...     root=RootDescriptor(rid=f"1-{1:04X}", type="Dataset", version="1.0.0"),
        ...     executions={},
        ...     datasets={},
        ...     assets={},
        ...     gaps=[],
        ...     executions_visited=0,
        ...     datasets_visited=0,
        ...     traversal_complete=True,
        ...     cap_hit=False,
        ... )
        >>> closure.traversal_complete
        True
        >>> import json
        >>> isinstance(json.dumps(closure.model_dump(mode="json")), str)
        True
    """

    model_config = ConfigDict(extra="forbid")

    root: RootDescriptor
    executions: dict[str, ProvenanceExecution] = Field(default_factory=dict)
    datasets: dict[str, ProvenanceDataset] = Field(default_factory=dict)
    assets: dict[str, ProvenanceAsset] = Field(default_factory=dict)
    gaps: list[ProvenanceGap] = Field(default_factory=list)
    executions_visited: int = 0
    datasets_visited: int = 0
    traversal_complete: bool = True
    cap_hit: bool = False

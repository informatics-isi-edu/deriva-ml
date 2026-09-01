"""Unit tests for the provenance-closure result models.

These tests pin the model shapes defined in
``deriva_ml.execution.provenance`` — the StrEnum vocabularies, the
``ProvenanceArc`` identity/dedup contract, and JSON-dumpability of the
top-level ``ProvenanceClosure`` envelope — ahead of the walk engine
that will populate them (later tasks).

No live catalog is needed: every model here is constructed from
literals and generated RIDs.
"""

from __future__ import annotations


def _rid(n: int, prefix: str = "1") -> str:
    """Generate a deterministic, catalog-shaped RID for a test.

    Mirrors the ``_rid`` helper in ``test_lineage_goldens.py`` — RIDs
    are opaque and must never be hand-written literals (see repo
    CLAUDE.md "RIDs are opaque: equality only").
    """
    return f"{prefix}-{n:04X}"


def test_enums_are_strenums_with_string_equality():
    from deriva_ml.execution.provenance import (
        AncestryState,
        ArcInputType,
        ArcKind,
        GapKind,
        RootType,
    )

    # StrEnum members compare equal to their plain string values.
    assert RootType.dataset == "Dataset"
    assert RootType.asset == "Asset"
    assert RootType.feature == "Feature"
    assert RootType.execution == "Execution"

    assert ArcKind.root == "root"
    assert ArcKind.consumption == "consumption"
    assert ArcKind.version_authorship == "version_authorship"
    assert ArcKind.member_binding == "member_binding"

    assert ArcInputType.dataset == "dataset"
    assert ArcInputType.asset == "asset"

    assert AncestryState.resolved == "resolved"
    assert AncestryState.chain_break == "chain_break"
    assert AncestryState.not_walked == "not_walked"

    assert len(GapKind) == 12


def test_arc_identity_is_spec_tuple():
    from deriva_ml.execution.provenance import ArcInputType, ArcKind, ProvenanceArc

    a = ProvenanceArc(
        kind=ArcKind.consumption,
        consumed_by=_rid(1),
        input_rid=_rid(2),
        input_type=ArcInputType.dataset,
        input_version="1.0.0",
        depth=1,
    )
    assert a.identity() == (ArcKind.consumption, _rid(1), _rid(2), "1.0.0")

    b = a.model_copy(update={"depth": 5, "input_type": None})
    assert a.identity() == b.identity()  # depth and input_type excluded

    assert a.model_copy(update={"input_version": "2.0.0"}).identity() != a.identity()


def test_is_source_requires_resolved_ancestry():
    from deriva_ml.execution.provenance import AncestryState, DatasetVersionFacts

    resolved = DatasetVersionFacts(
        version="1.0.0",
        parents=[],
        ancestry_state=AncestryState.resolved,
        is_source=True,
        origin_recorded=True,
        version_authors=[],
    )
    assert resolved.is_source is True

    not_walked = DatasetVersionFacts(
        version="1.0.0",
        parents=[],
        ancestry_state=AncestryState.not_walked,
        is_source=None,
        origin_recorded=None,
        version_authors=[],
    )
    assert not_walked.is_source is None

    # is_source True/False is only meaningful when ancestry_state is
    # "resolved" — anything else must carry None, enforced by a
    # model validator.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DatasetVersionFacts(
            version="1.0.0",
            parents=[],
            ancestry_state=AncestryState.chain_break,
            is_source=True,
            origin_recorded=None,
            version_authors=[],
        )


def test_closure_dumps_to_plain_json():
    import json

    from deriva_ml.execution.lineage import RootDescriptor
    from deriva_ml.execution.provenance import (
        ProvenanceClosure,
    )

    root = RootDescriptor(rid=_rid(1), type="Dataset", description="root ds", version="1.0.0")
    closure = ProvenanceClosure(
        root=root,
        executions={},
        datasets={},
        assets={},
        gaps=[],
        executions_visited=0,
        datasets_visited=0,
        traversal_complete=True,
        cap_hit=False,
    )

    dumped = closure.model_dump(mode="json")
    # Must be plain-JSON serializable (StrEnum members dump as str).
    encoded = json.dumps(dumped)
    assert isinstance(encoded, str)
    assert dumped["root"]["rid"] == _rid(1)
    assert dumped["traversal_complete"] is True
    assert dumped["cap_hit"] is False

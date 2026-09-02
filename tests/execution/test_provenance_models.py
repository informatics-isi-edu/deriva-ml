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
    assert ArcKind.member_production == "member_production"
    assert len(ArcKind) == 5

    assert ArcInputType.dataset == "dataset"
    assert ArcInputType.asset == "asset"

    # ``snapshot_chain_break`` survives ruling 8 — it still fires for the
    # authorship/binding snapshot legs and the member-scan degrade.
    assert GapKind.snapshot_chain_break == "snapshot_chain_break"
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


def test_ancestry_fields_are_gone_from_the_model():
    """Ruling 8 (#389): containment is not provenance, so the ancestry
    fields and their vocabulary no longer exist on the result model."""
    import pytest
    from pydantic import ValidationError

    import deriva_ml.execution.provenance as prov
    from deriva_ml.execution.provenance import DatasetVersionFacts

    # The vocabulary and the link model are gone from the module and from
    # the package's public export surface.
    for name in ("AncestryState", "ParentLink"):
        assert not hasattr(prov, name), f"{name} should have been removed"
        assert name not in prov.__all__

    import deriva_ml.execution as execution_pkg

    for name in ("AncestryState", "ParentLink"):
        assert not hasattr(execution_pkg, name), f"{name} still exported from deriva_ml.execution"

    # The facts record no longer carries them, and (extra="forbid") rejects
    # anyone still passing them.
    facts = DatasetVersionFacts(version="1.0.0", origin_recorded=True, version_authors=[])
    for name in ("parents", "ancestry_state", "is_source"):
        assert not hasattr(facts, name), f"DatasetVersionFacts.{name} should have been removed"
        assert name not in DatasetVersionFacts.model_fields

    with pytest.raises(ValidationError):
        DatasetVersionFacts(version="1.0.0", parents=[], origin_recorded=True)
    with pytest.raises(ValidationError):
        DatasetVersionFacts(version="1.0.0", is_source=True, origin_recorded=True)
    with pytest.raises(ValidationError):
        DatasetVersionFacts(version="1.0.0", ancestry_state="resolved", origin_recorded=True)


def test_dataset_version_facts_dump_is_byte_stable():
    """Determinism: the trimmed facts record still dumps identically on
    repeated calls, with no ancestry keys in the envelope."""
    import json

    from deriva_ml.execution.provenance import DatasetVersionFacts

    facts = DatasetVersionFacts(version="1.0.0", origin_recorded=False, version_authors=[])
    first = json.dumps(facts.model_dump(mode="json"), sort_keys=True)
    second = json.dumps(facts.model_dump(mode="json"), sort_keys=True)
    assert first == second
    assert set(json.loads(first)) == {"version", "origin_recorded", "version_authors"}


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

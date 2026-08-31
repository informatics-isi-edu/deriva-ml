"""Model + export pins for the #367 lineage additions."""

from __future__ import annotations


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def test_version_attribution_importable_from_package():
    from deriva_ml.execution import VersionAttribution  # noqa: F401


def test_version_attribution_in_all():
    import deriva_ml.execution as pkg

    assert "VersionAttribution" in pkg.__all__


def test_root_descriptor_new_fields_default():
    from deriva_ml.execution.lineage import RootDescriptor

    root = RootDescriptor(rid=_rid(1), type="Asset", description=None)
    assert root.origin_recorded is None
    assert root.version_history == []


def test_version_attribution_carries_raw_rid_separately():
    from deriva_ml.execution import VersionAttribution

    entry = VersionAttribution(version="0.1.0", execution_rid=_rid(2), execution=None, description=None)
    assert entry.execution_rid == _rid(2)
    assert entry.execution is None  # recorded but unresolved


def test_model_dump_is_additive():
    """Serialized root carries the new keys alongside every old one."""
    from deriva_ml.execution.lineage import RootDescriptor

    d = RootDescriptor(rid=_rid(1), type="Dataset", description=None).model_dump()
    for key in ("rid", "type", "description", "producing_execution", "origin_recorded", "version_history"):
        assert key in d


def test_workflow_summary_carries_code_identity():
    """WorkflowSummary exposes url and version (issue #372)."""
    from deriva_ml.execution.lineage import WorkflowSummary

    wf = WorkflowSummary(
        rid=_rid(9),
        name="training",
        url="https://github.com/org/repo",
        version="1.2.3",
    )
    assert wf.url == "https://github.com/org/repo"
    assert wf.version == "1.2.3"
    # Defaults stay None so existing construction sites are unaffected.
    bare = WorkflowSummary(rid=_rid(9), name=None)
    assert bare.url is None and bare.version is None

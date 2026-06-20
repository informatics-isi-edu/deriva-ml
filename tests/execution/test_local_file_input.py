"""Provenance Contract — local-file input asset (LocalFile) tests.

Covers section E of the test plan (docs/reference/provenance-contract-test-plan.md):
declaring a local file as an input asset, the bare-string-is-always-a-RID
safety boundary, and the `validate_assets` shape-routing that is both the
hydra seam and the security chokepoint.

The pure-Python (unit) tests here run without a catalog. Tests for not-yet-
built behavior (the `LocalFile` spec, path routing) are `xfail(strict=True)`
until the implementation lands, per the TDD acceptance-test approach.
"""

from __future__ import annotations

import pytest

from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.execution.execution_configuration import ExecutionConfiguration


# ─────────────────────────────────────────────────────────────────────────
# E5/E6 safety boundary — a bare string in assets= is ALWAYS a RID.
# (Characterization: the existing validator already does this; pin it so the
#  no-type-sniffing safety guarantee can't regress.)
# ─────────────────────────────────────────────────────────────────────────


def test_E6_bare_string_routes_to_assetspec_rid():
    """A bare string in assets= resolves as a RID (AssetSpec), never a path.

    This is the abuse-surface boundary: the validator does not type-sniff a
    string to decide RID-vs-path. A bare string is always a RID.
    """
    config = ExecutionConfiguration(assets=["1-ABCD"])
    assert len(config.assets) == 1
    spec = config.assets[0]
    assert isinstance(spec, AssetSpec)
    assert spec.rid == "1-ABCD"


def test_E6_rid_keyed_dict_routes_to_assetspec():
    """A {'rid': ...} dict (e.g. from a config) routes to AssetSpec."""
    config = ExecutionConfiguration(assets=[{"rid": "1-ABCD", "cache": True}])
    spec = config.assets[0]
    assert isinstance(spec, AssetSpec)
    assert spec.rid == "1-ABCD"
    assert spec.cache is True


# ─────────────────────────────────────────────────────────────────────────
# E6 routing — a path-keyed entry routes to LocalFile (NOT AssetSpec).
# xfail until LocalFile + validate_assets shape-routing land.
# ─────────────────────────────────────────────────────────────────────────


def test_E6_path_keyed_dict_routes_to_localfile():
    """A {'path': ...} entry (from a hydra .yaml) routes to a LocalFile spec,
    not an AssetSpec — the path is an explicit local-file input declaration."""
    from deriva_ml.asset.aux_classes import LocalFile

    config = ExecutionConfiguration(assets=[{"path": "/data/labels.csv"}])
    spec = config.assets[0]
    assert isinstance(spec, LocalFile)
    assert str(spec.path) == "/data/labels.csv"


def test_E5_localfile_wrapper_is_the_only_path_entry_point():
    """A local file is declared via an explicit LocalFile; a bare string is
    NEVER sniffed into a path — it is treated only as a RID."""
    import pytest
    from pydantic import ValidationError

    from deriva_ml.asset.aux_classes import LocalFile

    config = ExecutionConfiguration(assets=[LocalFile(path="/data/labels.csv")])
    spec = config.assets[0]
    assert isinstance(spec, LocalFile)
    assert str(spec.path) == "/data/labels.csv"

    # A bare path-looking string is NOT routed to a LocalFile. It is treated
    # only as a RID — and since it is not a valid RID, it is rejected outright
    # (never silently read as a filesystem path). The wrapper is the ONLY way
    # to declare a path.
    with pytest.raises(ValidationError):
        ExecutionConfiguration(assets=["/data/labels.csv"])


# ─────────────────────────────────────────────────────────────────────────
# E4 — live: a LocalFile input is registered as a File and linked as Input,
# and is NOT downloaded (it is a reference, not a Hatrac asset).
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_E4_localfile_input_registered_and_linked_as_input(test_ml, tmp_path):
    """E4 — A LocalFile in assets= is registered as a File row and linked to
    the execution as an Input (File_Execution Asset_Role="Input"), by context.

    The file is referenced, not uploaded; its role comes from being declared
    in assets=, never from a flag.
    """
    from deriva_ml import MLVocab as vc
    from deriva_ml.asset.aux_classes import LocalFile

    ml = test_ml
    ml.add_term(vc.workflow_type, "LocalFile Test Workflow", description="local-file input test")
    workflow = ml.create_workflow(name="localfile_test", workflow_type="LocalFile Test Workflow")

    # A real local file on disk.
    src = tmp_path / "labels.csv"
    src.write_text("RID_Subject,glc_dx\n1-AAAA,0\n")

    config = ExecutionConfiguration(
        description="Consumes a local file",
        workflow=workflow,
        assets=[LocalFile(path=str(src))],
    )

    exe = ml.create_execution(config)

    # A File_Execution row with Asset_Role="Input" must exist for this run.
    pb = ml.pathBuilder()
    fe = pb.schemas[ml.ml_schema].File_Execution
    rows = [r for r in fe.entities().fetch() if r["Execution"] == exe.execution_rid]
    assert rows, "expected a File_Execution row for the LocalFile input"
    assert {r["Asset_Role"] for r in rows} == {"Input"}, (
        "a LocalFile declared in assets= must be linked as Input (role from context)"
    )

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
    """E4 — A LocalFile in assets= is registered as a File dataset and recorded
    as a Dataset_Execution input edge for the execution.

    The file is referenced, not uploaded; provenance is recorded via a
    Dataset_Execution row (one edge per dataset) rather than a per-file
    File_Execution Input row.  The execution is input-complete via that edge.
    """
    from deriva_ml import MLVocab as vc
    from deriva_ml.asset.aux_classes import LocalFile
    from deriva_ml.execution.provenance_enforcement import _execution_has_input

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
    exec_rid = exe.execution_rid

    # The LocalFile input is registered via add_files, which records ONE
    # Dataset_Execution input edge (the file's dataset) rather than a per-file
    # File_Execution Input row.  The execution must be input-complete via that edge.
    pb = ml.pathBuilder()

    # No per-file File_Execution Input row from the LocalFile registration.
    fe = pb.schemas[ml.ml_schema].File_Execution
    fe_inputs = [r for r in fe.filter(fe.Execution == exec_rid).entities().fetch() if r.get("Asset_Role") == "Input"]
    assert fe_inputs == [], "LocalFile input must not create a per-file File_Execution Input row"

    # A Dataset_Execution input edge exists (the LocalFile's dataset).
    de = pb.schemas[ml.ml_schema].Dataset_Execution
    de_rows = list(de.filter(de.Execution == exec_rid).entities().fetch())
    assert de_rows, "LocalFile input must be recorded as a Dataset_Execution input edge"

    # The execution is input-complete (no unknown-provenance sentinel needed).
    assert _execution_has_input(ml_instance=ml, execution_rid=exec_rid) is True


# ─────────────────────────────────────────────────────────────────────────
# Public export surface — LocalFile/LocalFileConfig are usable from the same
# place as AssetSpec (deriva_ml.execution), so consumers (e.g. data-curation's
# hydra-zen configs) have a stable public import, not an internal module path.
# ─────────────────────────────────────────────────────────────────────────


def test_localfile_is_exported_from_execution_package():
    """``LocalFile``/``LocalFileConfig`` re-export from ``deriva_ml.execution``.

    They live canonically in ``deriva_ml.asset.aux_classes`` but, like
    ``AssetSpec``, are used in ``ExecutionConfiguration.assets`` — so they must
    be reachable from the execution surface that consumers import from. Pins the
    public path so it can't silently regress to an internal-only import.
    """
    from deriva_ml.asset.aux_classes import LocalFile as _CanonicalLocalFile
    from deriva_ml.execution import LocalFile, LocalFileConfig

    # Same object as the canonical definition (a true re-export, not a shadow).
    assert LocalFile is _CanonicalLocalFile
    # Constructs as documented (keyword form; bare-string shorthand is only for
    # inside an assets= list, where the validator coerces it).
    assert LocalFile(path="/data/labels.csv").path == "/data/labels.csv"
    assert LocalFileConfig is not None

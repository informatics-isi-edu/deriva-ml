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


@pytest.mark.xfail(
    reason="LocalFile + validate_assets shape-routing not implemented. A "
    "{'path': ...} entry must route to a LocalFile spec; today the validator "
    "only knows rid-keyed/bare-string entries and would mishandle a path key. "
    "Flip to a real assertion when LocalFile lands.",
    strict=True,
)
def test_E6_path_keyed_dict_routes_to_localfile():
    """A {'path': ...} entry (from a hydra .yaml) routes to a LocalFile spec,
    not an AssetSpec — the path is an explicit local-file input declaration."""
    from deriva_ml.asset.aux_classes import LocalFile  # noqa: F401 — not yet implemented

    config = ExecutionConfiguration(assets=[{"path": "/data/labels.csv"}])
    spec = config.assets[0]
    assert isinstance(spec, LocalFile)
    assert str(spec.path) == "/data/labels.csv"


@pytest.mark.xfail(
    reason="LocalFile not implemented. A bare path string must NOT be sniffed "
    "into a path — it stays a RID (safety). A local file requires the explicit "
    "LocalFile wrapper. This pins that the wrapper is the only path entry point.",
    strict=True,
)
def test_E5_localfile_wrapper_is_the_only_path_entry_point():
    """A local file is declared via LocalFile('/path'); a bare path string is
    still treated as a RID, never sniffed into a filesystem read."""
    from deriva_ml.asset.aux_classes import LocalFile

    config = ExecutionConfiguration(assets=[LocalFile("/data/labels.csv")])
    spec = config.assets[0]
    assert isinstance(spec, LocalFile)

    # A bare path-looking string is NOT a LocalFile — it is a RID.
    config2 = ExecutionConfiguration(assets=["/data/labels.csv"])
    assert isinstance(config2.assets[0], AssetSpec)
    assert config2.assets[0].rid == "/data/labels.csv"

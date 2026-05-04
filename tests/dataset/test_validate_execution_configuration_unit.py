"""Unit tests for ``DerivaML.validate_execution_configuration``.

These tests mock the catalog-touching primitives and exercise the
composite validator: per-dataset/asset/workflow result fan-out,
delegation into ``validate_dataset_specs``, cross-spec issue
detection (duplicate RIDs, version conflicts, role conflicts),
empty-config edge cases, and round-trip serialization.

Live-catalog smoke tests live in
``tests/dataset/test_validate_specs_live.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.dataset import DatasetMixin
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.validation import (
    ExecutionConfigurationValidationReport,
)
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.workflow import Workflow

# ---------------------------------------------------------------------------
# Helpers — shared with test_validate_dataset_spec_unit (kept inline here so
# the file is self-contained and the per-test setup is obvious).
# ---------------------------------------------------------------------------


@dataclass
class _StubColumn:
    name: str


class _StubColumnEq(_StubColumn):
    def __eq__(self, other: Any) -> tuple[str, Any]:  # type: ignore[override]
        return (self.name, other)


@dataclass
class _StubTable:
    name: str
    columns: list[_StubColumn] = field(default_factory=list)


class _StubEntities:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def fetch(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _StubDatapath:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def entities(self) -> _StubEntities:
        return _StubEntities(self._rows)


@dataclass
class _StubResolved:
    table: _StubTable
    datapath: _StubDatapath


class _StubVersionPath:
    def __init__(self, version_rows_by_dataset: dict[str, list[str]]):
        self._versions = version_rows_by_dataset
        self._current_filter: str | None = None
        self.Dataset = _StubColumnEq("Dataset")

    def filter(self, expr: tuple[str, str]) -> "_StubVersionPath":
        _col, rid = expr
        self._current_filter = rid
        return self

    def entities(self) -> _StubEntities:
        rid = self._current_filter
        rows = [{"Version": v} for v in self._versions.get(rid, [])]
        return _StubEntities(rows)


class _StubSchema:
    def __init__(self, version_path: _StubVersionPath):
        self.tables = {"Dataset_Version": version_path}


class _StubPathBuilder:
    def __init__(self, ml_schema: str, version_path: _StubVersionPath):
        self.schemas = {ml_schema: _StubSchema(version_path)}


class _FakeML(DatasetMixin):
    """DatasetMixin host for composite-validator unit tests."""

    ml_schema = "deriva-ml"

    def __init__(self) -> None:
        self._rids: dict[str, tuple[str, dict[str, Any]]] = {}
        self._versions: dict[str, list[str]] = {}
        self._asset_table_names: set[str] = set()
        self.model = MagicMock()
        self.model.is_asset = lambda table: table.name in self._asset_table_names
        self._version_path = _StubVersionPath(self._versions)
        self._pb = _StubPathBuilder(self.ml_schema, self._version_path)

    # -- scripting helpers -------------------------------------------------

    def add_dataset(self, rid: str, description: str | None = None, versions: list[str] | None = None) -> None:
        self._rids[rid] = (
            "Dataset",
            {"Description": description, "Deleted": False, "RID": rid},
        )
        self._versions[rid] = list(versions or [])

    def add_asset(self, rid: str, asset_table: str, filename: str = "") -> None:
        self._rids[rid] = (asset_table, {"Filename": filename, "RID": rid})
        self._asset_table_names.add(asset_table)

    def add_workflow(self, rid: str, name: str = "") -> None:
        self._rids[rid] = ("Workflow", {"Name": name, "RID": rid})

    def add_other(self, rid: str, table_name: str) -> None:
        self._rids[rid] = (table_name, {"RID": rid})

    # -- DatasetMixin protocol --------------------------------------------

    def resolve_rid(self, rid: str) -> _StubResolved:  # type: ignore[override]
        if rid not in self._rids:
            raise DerivaMLException(f"Invalid RID {rid}")
        table_name, row = self._rids[rid]
        return _StubResolved(
            table=_StubTable(name=table_name, columns=[_StubColumn("RID")]),
            datapath=_StubDatapath(rows=[row]),
        )

    def pathBuilder(self) -> _StubPathBuilder:  # type: ignore[override]
        return self._pb


def _make_workflow(rid: str = "2-WFAA") -> Workflow:
    """Construct a Workflow we can attach to a config without catalog binding."""
    return Workflow(
        name="test wf",
        url="https://example.org/wf.py",
        workflow_type="python_script",
        rid=rid,
        checksum="abc123",
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_all_valid_composite():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="alpha", versions=["1.0.0"])
    ml.add_asset("3JSE", asset_table="Image", filename="scan.jpg")
    ml.add_workflow("2-WFAA", name="trainer")

    config = ExecutionConfiguration(
        workflow=_make_workflow("2-WFAA"),
        datasets=[DatasetSpec(rid="1-DSAA", version="1.0.0")],
        assets=[AssetSpec(rid="3JSE")],
    )
    report = ml.validate_execution_configuration(config)

    assert isinstance(report, ExecutionConfigurationValidationReport)
    assert report.all_valid is True
    assert report.dataset_results[0].valid is True
    assert report.asset_results[0].valid is True
    assert report.asset_results[0].asset_table == "Image"
    assert report.asset_results[0].filename == "scan.jpg"
    assert report.workflow_result is not None
    assert report.workflow_result.valid is True
    assert report.workflow_result.workflow_name == "trainer"
    assert report.cross_spec_issues == []


def test_dataset_failure_surfaces():
    ml = _FakeML()
    config = ExecutionConfiguration(
        datasets=[DatasetSpec(rid="9-NOPE", version="1.0.0")],
    )
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is False
    assert report.dataset_results[0].reasons == ["rid_not_found"]


def test_asset_failure_surfaces_not_an_asset():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])  # not an asset
    config = ExecutionConfiguration(
        assets=[AssetSpec(rid="1-DSAA")],
    )
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is False
    asset_r = report.asset_results[0]
    assert asset_r.valid is False
    assert asset_r.reasons == ["not_an_asset"]
    assert asset_r.actual_table == "Dataset"


def test_asset_failure_surfaces_rid_not_found():
    ml = _FakeML()
    config = ExecutionConfiguration(assets=[AssetSpec(rid="9-NOPE")])
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is False
    assert report.asset_results[0].reasons == ["rid_not_found"]


def test_workflow_failure_not_a_workflow():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])  # treat as if user pointed at a dataset
    # Build a Workflow whose RID points at the dataset row.
    config = ExecutionConfiguration(workflow=_make_workflow("1-DSAA"))
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is False
    assert report.workflow_result is not None
    assert report.workflow_result.reasons == ["not_a_workflow"]
    assert report.workflow_result.actual_table == "Dataset"


def test_workflow_failure_rid_not_found():
    ml = _FakeML()
    config = ExecutionConfiguration(workflow=_make_workflow("9-NOPE"))
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is False
    assert report.workflow_result is not None
    assert report.workflow_result.reasons == ["rid_not_found"]


def test_duplicate_dataset_rid_same_version():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    config = ExecutionConfiguration(
        datasets=[
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
        ],
    )
    report = ml.validate_execution_configuration(config)
    # Per-spec results are still both valid; the cross-spec issue is the flag.
    assert all(r.valid for r in report.dataset_results)
    issues = [i for i in report.cross_spec_issues if i.issue == "duplicate_rid"]
    assert len(issues) == 1
    assert issues[0].rids == ["1-DSAA"]
    assert report.all_valid is False


def test_version_conflict_same_rid_two_versions():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0", "0.5.0"])
    config = ExecutionConfiguration(
        datasets=[
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
            DatasetSpec(rid="1-DSAA", version="0.5.0"),
        ],
    )
    report = ml.validate_execution_configuration(config)
    issues = [i for i in report.cross_spec_issues if i.issue == "version_conflict"]
    assert len(issues) == 1
    assert issues[0].rids == ["1-DSAA"]
    assert report.all_valid is False


def test_role_conflict_input_and_output():
    ml = _FakeML()
    ml.add_asset("3JSE", asset_table="Image", filename="x.jpg")
    config = ExecutionConfiguration(
        assets=[
            AssetSpec(rid="3JSE", asset_role="Input"),
            AssetSpec(rid="3JSE", asset_role="Output"),
        ],
    )
    report = ml.validate_execution_configuration(config)
    issues = [i for i in report.cross_spec_issues if i.issue == "role_conflict"]
    assert len(issues) == 1
    assert report.all_valid is False


def test_empty_datasets_and_assets_no_workflow_is_valid():
    ml = _FakeML()
    config = ExecutionConfiguration()
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is True
    assert report.dataset_results == []
    assert report.asset_results == []
    assert report.workflow_result is None
    assert report.cross_spec_issues == []


def test_empty_data_with_valid_workflow_is_valid():
    ml = _FakeML()
    ml.add_workflow("2-WFAA", name="trainer")
    config = ExecutionConfiguration(workflow=_make_workflow("2-WFAA"))
    report = ml.validate_execution_configuration(config)
    assert report.all_valid is True
    assert report.workflow_result is not None
    assert report.workflow_result.workflow_name == "trainer"


def test_composite_delegates_to_singular(monkeypatch):
    """The composite must call validate_dataset_specs for the dataset half so
    that there is no duplicate logic. We spy on the singular method to confirm."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])

    spy_calls: list[list[Any]] = []
    real_method = ml.validate_dataset_specs

    def spy(specs):
        spy_calls.append(list(specs))
        return real_method(specs=specs)

    monkeypatch.setattr(ml, "validate_dataset_specs", spy)

    config = ExecutionConfiguration(
        datasets=[DatasetSpec(rid="1-DSAA", version="1.0.0")],
    )
    ml.validate_execution_configuration(config)

    assert len(spy_calls) == 1
    assert spy_calls[0][0].rid == "1-DSAA"


def test_round_trip_serialization():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    ml.add_workflow("2-WFAA", name="trainer")

    config = ExecutionConfiguration(
        workflow=_make_workflow("2-WFAA"),
        datasets=[DatasetSpec(rid="1-DSAA", version="1.0.0")],
        assets=[AssetSpec(rid="9-NOPE")],  # forces a failure for content
    )
    report = ml.validate_execution_configuration(config)

    blob = report.model_dump_json()
    restored = ExecutionConfigurationValidationReport.model_validate_json(blob)
    assert restored.all_valid == report.all_valid
    assert restored.workflow_result is not None
    assert restored.workflow_result.workflow_name == "trainer"
    assert restored.asset_results[0].valid is False

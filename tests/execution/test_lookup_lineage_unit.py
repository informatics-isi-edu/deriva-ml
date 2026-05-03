"""Unit tests for ``DerivaML.lookup_lineage``.

These tests mock the catalog-touching primitives (``resolve_rid``,
``_retrieve_rid``, ``lookup_execution``, ``_producer_of_dataset``,
``_producer_of_asset``) and exercise the walk shape, RID-type
detection, depth/cycle/cap behavior, and Pydantic round-tripping.

A live-catalog smoke test lives in
``tests/execution/test_lookup_lineage_live.py`` and is gated on
``DERIVA_HOST``.

Note on test RIDs: the ``RID`` Pydantic type validates against
ERMrest's RID pattern (``[A-Z\\d]{1,4}`` segments separated by
hyphens), so test RIDs are written in that form (e.g. ``"1-DSAA"``)
rather than human-readable shorthand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.execution.lineage import LineageNode, LineageResult
from deriva_ml.execution.state_store import ExecutionStatus

# ---------------------------------------------------------------------------
# Helpers — minimal stand-ins for the catalog primitives.
# ---------------------------------------------------------------------------


@dataclass
class _StubColumn:
    name: str


@dataclass
class _StubTable:
    name: str
    columns: list[_StubColumn]


@dataclass
class _StubResolved:
    table: _StubTable


@dataclass
class _StubWorkflow:
    rid: str
    name: str


@dataclass
class _StubAsset:
    asset_rid: str
    filename: str
    asset_table: str


class _StubDataset:
    """Minimal Dataset stand-in for list_input_datasets()."""

    def __init__(
        self,
        dataset_rid: str,
        description: str = "",
        version: str = "0.1.0",
    ) -> None:
        self.dataset_rid = dataset_rid
        self.description = description
        self._version = version

    @property
    def current_version(self) -> str:
        return self._version


class _StubExecutionRecord:
    """Stand-in for ExecutionRecord with the methods _walk_node calls."""

    def __init__(
        self,
        execution_rid: str,
        description: str | None = None,
        workflow: _StubWorkflow | None = None,
        status: ExecutionStatus = ExecutionStatus.Uploaded,
        input_datasets: list[_StubDataset] | None = None,
        input_assets: list[_StubAsset] | None = None,
    ) -> None:
        self.execution_rid = execution_rid
        self.description = description
        self.workflow = workflow
        self.status = status
        self._input_datasets = input_datasets or []
        self._input_assets = input_assets or []

    def list_input_datasets(self) -> list[_StubDataset]:
        return list(self._input_datasets)

    def list_assets(self, asset_role: str | None = None) -> list[_StubAsset]:
        if asset_role and asset_role != "Input":
            return []
        return list(self._input_assets)


class _FakeML(ExecutionMixin):
    """Bare ExecutionMixin host that scripts the primitives the lineage
    walk depends on. Keeps the surface minimal so individual tests can
    set up just the slice they care about.
    """

    def __init__(self) -> None:
        # Map RID -> (table_name, optional column-name set, optional row dict).
        self._rids: dict[str, tuple[str, set[str], dict[str, Any]]] = {}
        # Map RID -> _StubExecutionRecord (for lookup_execution).
        self._executions: dict[str, _StubExecutionRecord] = {}
        # Map dataset_rid -> producing-execution RID (or None).
        self._dataset_producers: dict[str, str | None] = {}
        # Map asset_rid -> producing-execution RID (or None).
        self._asset_producers: dict[str, str | None] = {}
        # Tracks which RIDs the model considers "asset" tables.
        self._asset_table_names: set[str] = set()
        # Mock model.
        self.model = MagicMock()
        self.model.is_asset = lambda table: table.name in self._asset_table_names
        self.model.name_to_table = lambda name: _StubTable(name=name, columns=[_StubColumn("RID")])

    # -- scripting helpers -------------------------------------------------

    def add_dataset(self, rid: str, description: str = "", producer: str | None = None) -> None:
        self._rids[rid] = ("Dataset", set(), {"Description": description})
        self._dataset_producers[rid] = producer

    def add_asset(
        self,
        rid: str,
        asset_table: str,
        filename: str = "",
        description: str = "",
        producer: str | None = None,
    ) -> None:
        self._rids[rid] = (asset_table, set(), {"Description": description, "Filename": filename})
        self._asset_table_names.add(asset_table)
        self._asset_producers[rid] = producer

    def add_workflow(self, rid: str) -> None:
        self._rids[rid] = ("Workflow", set(), {})

    def add_feature_value(self, rid: str, producer_execution: str | None) -> None:
        self._rids[rid] = (
            "Image_Some_Feature",
            {"Feature_Name", "Execution"},
            {"Execution": producer_execution},
        )

    def add_execution(
        self,
        rid: str,
        *,
        description: str | None = None,
        workflow: _StubWorkflow | None = None,
        status: ExecutionStatus = ExecutionStatus.Uploaded,
        input_datasets: list[_StubDataset] | None = None,
        input_assets: list[_StubAsset] | None = None,
    ) -> _StubExecutionRecord:
        self._rids[rid] = ("Execution", set(), {"Description": description})
        rec = _StubExecutionRecord(
            execution_rid=rid,
            description=description,
            workflow=workflow,
            status=status,
            input_datasets=input_datasets,
            input_assets=input_assets,
        )
        self._executions[rid] = rec
        return rec

    # -- ExecutionMixin protocol -------------------------------------------

    def resolve_rid(self, rid: str) -> _StubResolved:
        if rid not in self._rids:
            raise DerivaMLException(f"Invalid RID {rid}")
        table_name, extra_cols, _ = self._rids[rid]
        cols = [_StubColumn("RID")] + [_StubColumn(c) for c in extra_cols]
        return _StubResolved(table=_StubTable(name=table_name, columns=cols))

    def _retrieve_rid(self, rid: str) -> dict[str, Any]:
        if rid not in self._rids:
            raise DerivaMLException(f"Invalid RID {rid}")
        return self._rids[rid][2]

    def lookup_execution(self, rid: str) -> _StubExecutionRecord:
        if rid not in self._executions:
            raise DerivaMLException(f"No such execution {rid}")
        return self._executions[rid]

    def _producer_of_dataset(self, dataset_rid: str) -> str | None:  # type: ignore[override]
        return self._dataset_producers.get(dataset_rid)

    def _producer_of_asset(self, asset_rid: str, asset_table: Any) -> str | None:  # type: ignore[override]
        return self._asset_producers.get(asset_rid)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_lineage_workflow_rid_raises():
    ml = _FakeML()
    ml.add_workflow("3-WFAA")
    with pytest.raises(DerivaMLException, match="Workflow"):
        ml.lookup_lineage("3-WFAA")


def test_lineage_unknown_rid_raises():
    ml = _FakeML()
    with pytest.raises(DerivaMLException, match="Invalid RID"):
        ml.lookup_lineage("NOPE")


def test_lineage_dataset_with_no_producer_returns_empty_walk():
    ml = _FakeML()
    ml.add_dataset("1-DSAB", description="orphan dataset", producer=None)

    result = ml.lookup_lineage("1-DSAB")

    assert isinstance(result, LineageResult)
    assert result.root.type == "Dataset"
    assert result.root.rid == "1-DSAB"
    assert result.root.producing_execution is None
    assert result.lineage is None
    assert result.executions_visited == 0
    assert result.walked_complete is True


def test_lineage_dataset_one_level_chain():
    """Dataset DS-1 produced by EXE-A which consumed Dataset DS-0 (no producer)."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="root data", producer=None)
    ml.add_execution(
        "2-EXAA",
        description="train",
        workflow=_StubWorkflow(rid="3-WFAB", name="trainer"),
        input_datasets=[_StubDataset("1-DSAA", "root data", "0.1.0")],
    )
    ml.add_dataset("1-DSAB", description="trained set", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == "2-EXAA"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"
    assert result.lineage.execution.workflow is not None
    assert result.lineage.execution.workflow.name == "trainer"
    assert len(result.lineage.consumed_datasets) == 1
    assert result.lineage.consumed_datasets[0].rid == "1-DSAA"
    assert result.lineage.parents == []  # DS-0 has no producer
    assert result.executions_visited == 1
    assert result.walked_complete is True


def test_lineage_two_level_chain():
    """DS-2 <- EXE-B <- DS-1 <- EXE-A <- DS-0."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution(
        "2-EXAA",
        input_datasets=[_StubDataset("1-DSAA")],
    )
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution(
        "2-EXAB",
        input_datasets=[_StubDataset("1-DSAB")],
    )
    ml.add_dataset("1-DSAC", producer="2-EXAB")

    result = ml.lookup_lineage("1-DSAC")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAB"
    assert len(result.lineage.parents) == 1
    parent = result.lineage.parents[0]
    assert parent.execution.rid == "2-EXAA"
    assert parent.consumed_datasets[0].rid == "1-DSAA"
    assert parent.parents == []
    assert result.executions_visited == 2


def test_lineage_depth_zero_returns_only_immediate_producer():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution("2-EXAB", input_datasets=[_StubDataset("1-DSAB")])
    ml.add_dataset("1-DSAC", producer="2-EXAB")

    result = ml.lookup_lineage("1-DSAC", depth=0)

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAB"
    assert result.lineage.parents == []
    assert result.depth_capped is True
    assert result.executions_visited == 1


def test_lineage_depth_one_walks_one_layer():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution("2-EXAB", input_datasets=[_StubDataset("1-DSAB")])
    ml.add_dataset("1-DSAC", producer="2-EXAB")

    result = ml.lookup_lineage("1-DSAC", depth=1)

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAB"
    assert len(result.lineage.parents) == 1
    grandparent = result.lineage.parents[0]
    assert grandparent.execution.rid == "2-EXAA"
    # depth=1 means: walk one level. EXE-A's parents would be next layer
    # but only if EXE-A had data-flow parents. DS-0 has no producer, so
    # nothing more to walk. Hence depth_capped stays False.
    assert grandparent.parents == []
    assert result.depth_capped is False


def test_lineage_diamond_dag_marks_already_shown():
    """EXE-C consumed two datasets (DS-A, DS-B) both produced by EXE-X."""
    ml = _FakeML()
    ml.add_dataset("1-DSRT", producer=None)
    ml.add_execution("2-EXBX", input_datasets=[_StubDataset("1-DSRT")])
    ml.add_dataset("1-DSBA", producer="2-EXBX")
    ml.add_dataset("1-DSBB", producer="2-EXBX")
    ml.add_execution(
        "2-EXAC",
        input_datasets=[_StubDataset("1-DSBA"), _StubDataset("1-DSBB")],
    )
    ml.add_dataset("1-DSOT", producer="2-EXAC")

    result = ml.lookup_lineage("1-DSOT")

    # Only one EXE-X expansion (the second branch is collapsed).
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAC"
    assert len(result.lineage.parents) == 1, "diamond should dedupe to one parent edge by RID"
    # EXE-X visited once.
    assert result.executions_visited == 2  # EXE-C, EXE-X
    assert result.cycle_detected is False


def test_lineage_max_executions_cap():
    """Build a 4-deep chain and cap at 2 executions."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")
    ml.add_execution("2-EXAB", input_datasets=[_StubDataset("1-DSAB")])
    ml.add_dataset("1-DSAC", producer="2-EXAB")
    ml.add_execution("2-EXAC", input_datasets=[_StubDataset("1-DSAC")])
    ml.add_dataset("1-DSAD", producer="2-EXAC")
    ml.add_execution("2-EXAD", input_datasets=[_StubDataset("1-DSAD")])
    ml.add_dataset("1-DSAE", producer="2-EXAD")

    result = ml.lookup_lineage("1-DSAE", max_executions=2)

    assert result.walked_complete is False
    assert result.executions_visited == 2


def test_lineage_execution_rid_is_self_root():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution(
        "2-EXAA",
        description="root execution",
        input_datasets=[_StubDataset("1-DSAA")],
    )

    result = ml.lookup_lineage("2-EXAA")

    assert result.root.type == "Execution"
    assert result.root.rid == "2-EXAA"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"
    assert result.executions_visited == 1


def test_lineage_asset_root_walks_via_producer():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_asset("4-ASAA", asset_table="Image", filename="cat.png", producer="2-EXAA")

    result = ml.lookup_lineage("4-ASAA")

    assert result.root.type == "Asset"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"


def test_lineage_feature_value_root_walks_via_execution_column():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_feature_value("5-FVAA", producer_execution="2-EXAA")

    result = ml.lookup_lineage("5-FVAA")

    assert result.root.type == "Feature"
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXAA"


def test_lineage_consumes_both_datasets_and_assets():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_asset("4-ASIN", asset_table="Image", filename="in.png", producer=None)
    ml.add_execution(
        "2-EXAA",
        input_datasets=[_StubDataset("1-DSAA", description="d", version="0.1.0")],
        input_assets=[_StubAsset("4-ASIN", "in.png", "Image")],
    )
    ml.add_dataset("1-DSAB", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    assert result.lineage is not None
    assert len(result.lineage.consumed_datasets) == 1
    assert len(result.lineage.consumed_assets) == 1
    assert result.lineage.consumed_assets[0].asset_table == "Image"
    assert result.lineage.consumed_assets[0].filename == "in.png"


def test_lineage_result_round_trips_via_pydantic():
    """Models serialize and reload cleanly."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    ml.add_execution("2-EXAA", input_datasets=[_StubDataset("1-DSAA")])
    ml.add_dataset("1-DSAB", producer="2-EXAA")

    result = ml.lookup_lineage("1-DSAB")

    dumped = result.model_dump()
    reloaded = LineageResult.model_validate(dumped)
    assert reloaded == result


def test_lineage_node_recursive_validation_works():
    """Smoke test the recursive parents field on LineageNode."""
    leaf = LineageNode(
        execution={"rid": "2-EXLF", "status": "Uploaded"},
    )
    parent = LineageNode(
        execution={"rid": "2-EXPA", "status": "Uploaded"},
        parents=[leaf],
    )
    assert parent.parents[0].execution.rid == "2-EXLF"

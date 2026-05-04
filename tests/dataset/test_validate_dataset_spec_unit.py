"""Unit tests for ``DerivaML.validate_dataset_specs``.

These tests mock the catalog-touching primitives (``resolve_rid``,
``pathBuilder``) and exercise the per-spec validation shape, the
failure-reason vocabulary, the dataset-deleted warning, the input
shorthand coercion, the duplicate-RID caching, the round-trip
serialization, and the ``available_versions`` cap.

A live-catalog smoke test lives in
``tests/dataset/test_validate_specs_live.py`` and is gated on
``DERIVA_HOST``.

Note on test RIDs: the ``RID`` Pydantic type validates against
ERMrest's RID pattern (``[A-Z\\d]{1,4}`` segments separated by
hyphens), so test RIDs are written in that form (e.g. ``"1-DSAA"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.dataset import DatasetMixin
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.validation import (
    DatasetSpecResult,
    DatasetSpecValidationReport,
)

# ---------------------------------------------------------------------------
# Helpers — minimal stand-ins for the catalog primitives.
# ---------------------------------------------------------------------------


@dataclass
class _StubColumn:
    name: str


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
    """Stand-in for ``pb.schemas[ml_schema].tables['Dataset_Version']``.

    ``filter()`` accepts an opaque expression and returns self; ``entities()``
    and ``fetch()`` echo the rows for the most-recently-filtered RID.
    """

    def __init__(self, version_rows_by_dataset: dict[str, list[str]]):
        self._versions = version_rows_by_dataset
        self._current_filter: str | None = None
        self.Dataset = _StubColumn("Dataset")  # used by `version_path.Dataset == rid`

    def filter(self, expr: tuple[str, str]) -> "_StubVersionPath":  # type: ignore[override]
        # expr is the (column_name, rid) tuple our stub equality op produced.
        _col, rid = expr
        self._current_filter = rid
        return self

    def entities(self) -> _StubEntities:
        rid = self._current_filter
        rows = [{"Version": v} for v in self._versions.get(rid, [])]
        return _StubEntities(rows)


class _StubColumnEq(_StubColumn):
    """Column subclass that produces a tuple from ``==`` for filter()."""

    def __eq__(self, other: Any) -> tuple[str, Any]:  # type: ignore[override]
        return (self.name, other)


class _StubVersionPathTyped(_StubVersionPath):
    """Override ``Dataset`` attribute so ``version_path.Dataset == rid`` returns
    a tuple our ``filter`` can read."""

    def __init__(self, version_rows_by_dataset: dict[str, list[str]]):
        super().__init__(version_rows_by_dataset)
        self.Dataset = _StubColumnEq("Dataset")


class _StubTablesDict(dict):
    """``pb.schemas[ml_schema].tables[name]`` stand-in."""


class _StubSchema:
    def __init__(self, version_path: _StubVersionPathTyped):
        self.tables = _StubTablesDict({"Dataset_Version": version_path})


class _StubSchemasDict(dict):
    """``pb.schemas[name]`` stand-in."""


class _StubPathBuilder:
    def __init__(self, ml_schema: str, version_path: _StubVersionPathTyped):
        self.schemas = _StubSchemasDict({ml_schema: _StubSchema(version_path)})


class _FakeML(DatasetMixin):
    """Bare DatasetMixin host that scripts the primitives the validation
    methods depend on. Mirrors the ``_FakeML`` pattern used by the lookup_lineage
    unit tests in ``tests/execution/test_lookup_lineage_unit.py``.
    """

    ml_schema = "deriva-ml"

    def __init__(self) -> None:
        # Map RID -> (table_name, optional row dict).
        self._rids: dict[str, tuple[str, dict[str, Any]]] = {}
        # Map dataset RID -> list of version strings, newest-first.
        self._versions: dict[str, list[str]] = {}
        # Mock model.
        self.model = MagicMock()
        self.model.is_asset = lambda table: False  # only used for asset paths

        # Build the path builder once, refreshed lazily as datasets are added.
        self._version_path = _StubVersionPathTyped(self._versions)
        self._pb = _StubPathBuilder(self.ml_schema, self._version_path)

    # -- scripting helpers -------------------------------------------------

    def add_dataset(
        self,
        rid: str,
        description: str | None = None,
        deleted: bool = False,
        versions: list[str] | None = None,
    ) -> None:
        self._rids[rid] = (
            "Dataset",
            {"Description": description, "Deleted": deleted, "RID": rid},
        )
        self._versions[rid] = list(versions or [])

    def add_non_dataset(self, rid: str, table_name: str) -> None:
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


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_empty_specs_list_is_valid_and_empty():
    ml = _FakeML()
    report = ml.validate_dataset_specs(specs=[])
    assert isinstance(report, DatasetSpecValidationReport)
    assert report.all_valid is True
    assert report.results == []


def test_all_valid_case():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="alpha", versions=["1.0.0", "0.5.0"])
    ml.add_dataset("1-DSAB", description="beta", versions=["0.1.0"])

    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
            DatasetSpec(rid="1-DSAB", version="0.1.0"),
        ]
    )

    assert report.all_valid is True
    assert len(report.results) == 2
    for r, expected_name in zip(report.results, ["alpha", "beta"]):
        assert r.valid is True
        assert r.dataset_name == expected_name
        assert r.reasons == []
        assert r.warnings == []


def test_mixed_valid_and_invalid():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    ml.add_non_dataset("2-WFAA", table_name="Workflow")

    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
            DatasetSpec(rid="2-WFAA", version="1.0.0"),
            DatasetSpec(rid="9-NOPE", version="1.0.0"),
        ]
    )

    assert report.all_valid is False
    assert report.results[0].valid is True
    assert report.results[1].valid is False
    assert "not_a_dataset" in report.results[1].reasons
    assert report.results[1].actual_table == "Workflow"
    assert report.results[2].valid is False
    assert "rid_not_found" in report.results[2].reasons


def test_rid_not_found_short_circuits_other_checks():
    ml = _FakeML()
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="9-NOPE", version="0.1.0"),
        ]
    )
    assert report.all_valid is False
    r = report.results[0]
    assert r.reasons == ["rid_not_found"]
    assert r.actual_table is None
    assert r.available_versions is None


def test_not_a_dataset_includes_actual_table():
    ml = _FakeML()
    ml.add_non_dataset("3JSE", table_name="Image")
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="3JSE", version="0.1.0"),
        ]
    )
    r = report.results[0]
    assert r.valid is False
    assert r.reasons == ["not_a_dataset"]
    assert r.actual_table == "Image"


def test_version_not_found_includes_available_versions():
    ml = _FakeML()
    ml.add_dataset(
        "1-DSAA",
        description="alpha",
        versions=["0.4.0", "0.3.0", "0.2.0", "0.1.0"],
    )
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="9.9.9"),
        ]
    )
    r = report.results[0]
    assert r.valid is False
    assert r.reasons == ["version_not_found"]
    # Newest-first.
    assert r.available_versions == ["0.4.0", "0.3.0", "0.2.0", "0.1.0"]


def test_duplicate_specs_validated_independently():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
        ]
    )
    assert report.all_valid is True
    assert len(report.results) == 2
    # Both should be valid; the per-RID cache means we only resolve once.


def test_dataset_deleted_is_warning_not_failure():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="zombie", deleted=True, versions=["0.1.0"])
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="0.1.0"),
        ]
    )
    r = report.results[0]
    assert r.valid is True
    assert r.warnings == ["dataset_deleted"]
    assert report.all_valid is True


def test_shorthand_string_input_coerced():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    report = ml.validate_dataset_specs(specs=["1-DSAA@1.0.0"])
    assert report.all_valid is True
    assert report.results[0].spec.rid == "1-DSAA"
    assert str(report.results[0].spec.version) == "1.0.0"


def test_dict_input_coerced():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    report = ml.validate_dataset_specs(specs=[{"rid": "1-DSAA", "version": "1.0.0"}])
    assert report.all_valid is True
    assert report.results[0].spec.rid == "1-DSAA"


def test_round_trip_serialization():
    ml = _FakeML()
    ml.add_dataset("1-DSAA", description="alpha", versions=["1.0.0"])
    ml.add_dataset("1-DSAB", versions=["0.5.0"])
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="1.0.0"),
            DatasetSpec(rid="1-DSAB", version="9.9.9"),
        ]
    )

    blob = report.model_dump_json()
    restored = DatasetSpecValidationReport.model_validate_json(blob)
    assert restored.all_valid == report.all_valid
    assert len(restored.results) == 2
    assert restored.results[0].valid is True
    assert restored.results[1].valid is False
    assert "version_not_found" in restored.results[1].reasons


def test_available_versions_cap_at_20():
    ml = _FakeML()
    # 25 versions, all distinct; newest first.
    versions = [f"0.{n}.0" for n in range(25, 0, -1)]
    ml.add_dataset("1-DSAA", versions=versions)
    report = ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="1-DSAA", version="9.9.9"),
        ]
    )
    r = report.results[0]
    assert r.valid is False
    assert r.available_versions is not None
    assert len(r.available_versions) == 20
    assert r.available_versions[0] == "0.25.0"
    # Cap is the 20 newest, so the 21st-newest is the cutoff.
    assert "0.5.0" not in r.available_versions


def test_per_spec_results_carry_input_spec_back():
    """The result.spec field must echo the coerced input so callers can
    match results to inputs (especially when the input was a shorthand
    string)."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", versions=["1.0.0"])
    report = ml.validate_dataset_specs(specs=["1-DSAA@1.0.0"])
    assert isinstance(report.results[0], DatasetSpecResult)
    assert report.results[0].spec.rid == "1-DSAA"


@pytest.mark.parametrize(
    "bad_input",
    [
        # Pydantic accepts anything that coerces; bare invalid types fall through
        # to DatasetSpec.model_validate which raises ValidationError.
        12345,
        ["not", "a", "spec"],
    ],
)
def test_uncoercible_input_raises(bad_input):
    from pydantic import ValidationError

    ml = _FakeML()
    with pytest.raises((ValidationError, ValueError, TypeError)):
        ml.validate_dataset_specs(specs=[bad_input])

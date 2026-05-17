"""Tests for the Input_File / Output_File auto-tag behavior on
``Execution._update_asset_execution_table``.

When an execution consumes an existing asset (download_asset) or
produces a new asset (upload_execution_outputs), the system writes
two kinds of association rows:

1. ``{Asset}_Execution`` — the per-execution-link direction tag
   (``Asset_Role="Input"`` or ``"Output"``). This is how
   ``execution.list_assets(asset_role="Input")`` finds inputs.

2. ``{Asset}_Asset_Type`` — the per-asset content classification
   (``Model_File``, ``Hydra_Config``, etc.). Multi-valued.

The fix wired into ``_update_asset_execution_table`` is that the
directional axis is ALSO recorded as an ``Asset_Type`` entry —
``Input_File`` for inputs, ``Output_File`` for outputs. This makes
"give me every asset that has ever served as an input" queryable
through ``Asset_Type`` alone (multi-valued, alongside whatever
content types the asset carries from its original creation).

These tests pin both branches at the unit level using fake
collaborators — no live catalog. The tests verify *what gets
inserted* into the two association tables; they don't exercise
ERMrest itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from deriva_ml.asset.aux_classes import AssetFilePath

# ---------------------------------------------------------------------------
# Fake collaborators — mimic the deriva-py / deriva-ml surfaces that
# ``_update_asset_execution_table`` reaches into.
# ---------------------------------------------------------------------------


@dataclass
class _FakeSchema:
    """Stand-in for a deriva-py Schema (only the .name field is read)."""

    name: str


@dataclass
class _FakeTable:
    """Stand-in for a deriva-py Table (only .name and .schema are read)."""

    name: str
    schema: _FakeSchema


@dataclass
class _RecordedInsert:
    """One captured call to ``table_path.insert(...)``."""

    table_name: str
    rows: list[dict[str, Any]]
    on_conflict_skip: bool


class _FakePathTable:
    """Stand-in for a datapath table accessor with an ``insert`` method.

    Records every insert against the parent _FakePathBuilder so
    tests can assert on the writes.
    """

    def __init__(self, parent: "_FakePathBuilder", table_name: str):
        self._parent = parent
        self._table_name = table_name

    def insert(self, rows, *, on_conflict_skip: bool = False) -> None:
        self._parent.inserts.append(
            _RecordedInsert(
                table_name=self._table_name,
                rows=list(rows),
                on_conflict_skip=on_conflict_skip,
            )
        )


class _AutoFakeTables:
    """Dict-like that auto-creates _FakePathTable on first access.

    Production code does ``pb.schemas[name].tables[table_name].insert(...)``
    so this needs to support ``[]`` subscript that lazily creates.
    """

    def __init__(self, parent: "_FakePathBuilder"):
        self._parent = parent
        self._cache: dict[str, _FakePathTable] = {}

    def __getitem__(self, table_name: str) -> _FakePathTable:
        if table_name not in self._cache:
            self._cache[table_name] = _FakePathTable(self._parent, table_name)
        return self._cache[table_name]

    def __contains__(self, table_name: str) -> bool:
        return True  # every table name is valid in the fake


class _FakePathSchema:
    """``pb.schemas[schema_name]`` returns one of these."""

    def __init__(self, parent: "_FakePathBuilder"):
        self._parent = parent
        # tables supports dict-style auto-creation: every
        # ``pb.schemas[s].tables[t]`` lookup synthesizes a
        # _FakePathTable that records inserts back to the parent.
        self.tables = _AutoFakeTables(parent)


class _FakePathBuilder:
    """``self._ml_object.pathBuilder()`` returns one of these.

    Records every insert across all tables; tests inspect
    ``.inserts``.
    """

    def __init__(self):
        self.inserts: list[_RecordedInsert] = []
        self.schemas: dict[str, _FakePathSchema] = {}

    def _ensure_schema(self, schema_name: str) -> _FakePathSchema:
        if schema_name not in self.schemas:
            self.schemas[schema_name] = _FakePathSchema(self)
        return self.schemas[schema_name]

    def __getitem__(self, schema_name: str) -> _FakePathSchema:
        # Production code uses ``pb.schemas[name]`` so we need the
        # ``schemas`` attribute to be subscriptable. We expose
        # __getitem__ on the _FakePathBuilder itself for symmetry
        # but the schemas dict is the actual entry point.
        return self._ensure_schema(schema_name)


class _SubscriptableSchemas:
    """Drop-in for ``pb.schemas`` that supports both subscription
    and attribute access against a backing _FakePathBuilder.

    Wrapped this way because production code uses
    ``pb.schemas[name].tables[name]``.
    """

    def __init__(self, pb: _FakePathBuilder):
        self._pb = pb

    def __getitem__(self, schema_name: str) -> _FakePathSchema:
        return self._pb._ensure_schema(schema_name)


class _FakePathBuilderWithSchemas:
    """The actual ``pb`` returned by ``ml.pathBuilder()``.

    Mirrors deriva-py's path builder shape closely enough for the
    auto-tag code path: ``pb.schemas[schema_name].tables[table_name].insert(...)``.
    """

    def __init__(self):
        self._inner = _FakePathBuilder()
        self.schemas = _SubscriptableSchemas(self._inner)

    @property
    def inserts(self) -> list[_RecordedInsert]:
        return self._inner.inserts


class _FakeModel:
    """Stand-in for DerivaModel exposing only ``find_association`` +
    ``name_to_table``.

    ``find_association(asset, partner)`` returns
    ``(association_table, asset_fk_col, partner_fk_col)``. For our
    purposes the function reads ``.name`` and ``.schema.name`` on
    the returned table; the FK column names are returned but not
    consequential for the test assertions.
    """

    def __init__(self, *, ml_schema: str = "deriva-ml"):
        self._ml_schema = ml_schema

    def name_to_table(self, name: str) -> _FakeTable:
        return _FakeTable(name=name, schema=_FakeSchema(name=self._ml_schema))

    def find_association(self, asset_table_name: str, partner_name: str):
        assoc_name = f"{asset_table_name}_{partner_name}"
        assoc_table = _FakeTable(name=assoc_name, schema=_FakeSchema(name=self._ml_schema))
        # FK column names are conventionally the partner names.
        asset_fk = asset_table_name
        partner_fk = partner_name
        return assoc_table, asset_fk, partner_fk


class _FakeMLObject:
    """Stand-in for ``execution._ml_object`` (the DerivaML instance)."""

    def __init__(self, pb: _FakePathBuilderWithSchemas):
        self._pb = pb
        self.lookup_calls: list[tuple[str, str]] = []

    def pathBuilder(self):
        return self._pb

    def lookup_term(self, vocab, term: str) -> None:
        # Production code calls this to validate the asset_role term
        # exists in the vocabulary. Record the call for assertion;
        # don't raise.
        self.lookup_calls.append((str(vocab), term))


@dataclass
class _FakeExecution:
    """Minimal shape that ``_update_asset_execution_table`` reads.

    Carries the four attributes the method touches: ``_dry_run``,
    ``_ml_object``, ``_model``, ``_working_dir``, ``execution_rid``.
    """

    _model: _FakeModel
    _ml_object: _FakeMLObject
    _working_dir: Path
    execution_rid: str
    _dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_execution_fixture(
    tmp_path: Path,
    *,
    dry_run: bool = False,
) -> tuple[_FakeExecution, _FakePathBuilderWithSchemas]:
    """Construct a fake execution + path-builder pair for the test."""
    pb = _FakePathBuilderWithSchemas()
    ml_object = _FakeMLObject(pb)
    model = _FakeModel()
    execution = _FakeExecution(
        _model=model,
        _ml_object=ml_object,
        _working_dir=tmp_path,
        execution_rid="1-EXEC",
        _dry_run=dry_run,
    )
    return execution, pb


def _fake_asset_type_path(
    working_dir: Path,
    execution_rid: str,
    asset_table_name: str,
    ml_schema: str,
) -> Path:
    """Return the same path ``asset_type_path`` would build.

    The production helper is ``@validate_call``-decorated and
    requires a real ``deriva.core.ermrest_model.Table`` — too
    heavy for a unit test. We mirror its path-construction logic:
    ``working_dir/deriva-ml/execution/<rid>/asset-type/<schema>/<table>.jsonl``.
    """
    return (
        working_dir
        / "deriva-ml"
        / "execution"
        / execution_rid
        / "asset-type"
        / ml_schema
        / f"{asset_table_name}.jsonl"
    )


def _write_asset_type_jsonl(
    working_dir: Path,
    execution_rid: str,
    asset_table_name: str,
    ml_schema: str,
    type_map: dict[str, list[str]],
) -> None:
    """Write the JSONL file the Output branch reads."""
    path = _fake_asset_type_path(working_dir, execution_rid, asset_table_name, ml_schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for filename, types in type_map.items():
            f.write(json.dumps({filename: types}) + "\n")


@pytest.fixture
def stub_asset_type_path(monkeypatch):
    """Replace ``asset_type_path`` with a fake-Table-friendly version.

    The production code calls
    ``asset_type_path(working_dir, exec_rid, table)`` inside the
    Output branch, passing a Table object obtained via
    ``self._model.name_to_table(...)``. Our fake returns
    _FakeTable, which the real (``@validate_call``-decorated)
    helper rejects with a Pydantic ValidationError.

    Substitute a duck-typed version that reads ``.name`` and
    ``.schema.name`` like the real helper but skips the type
    check.
    """

    def _fake(prefix, exec_rid, asset_table):
        return _fake_asset_type_path(
            Path(prefix),
            exec_rid,
            asset_table.name,
            asset_table.schema.name,
        )

    monkeypatch.setattr(
        "deriva_ml.execution.execution.asset_type_path",
        _fake,
    )


def _make_asset_file_path(
    *,
    file_name: str,
    asset_rid: str,
    asset_table: str,
) -> AssetFilePath:
    """Build a minimal AssetFilePath for the auto-tag test.

    Production matches this shape: ``file_name`` is a string
    (basename), ``asset_path`` is the on-disk Path, ``asset_types``
    starts as an empty list — auto-tagging will populate it.
    """
    return AssetFilePath(
        file_name=file_name,  # str basename, not Path — matches production
        asset_rid=asset_rid,
        asset_path=Path(file_name),
        asset_metadata={},
        asset_table=asset_table,
        asset_types=[],
    )


def _inserts_for(pb: _FakePathBuilderWithSchemas, table_name: str) -> list[_RecordedInsert]:
    """Return only the recorded inserts that targeted ``table_name``."""
    return [r for r in pb.inserts if r.table_name == table_name]


# ---------------------------------------------------------------------------
# Test cases — Output branch
# ---------------------------------------------------------------------------


class TestOutputAutoTag:
    """Auto-tag with Output_File when role is Output."""

    def test_output_file_added_when_user_omits_it(self, tmp_path, stub_asset_type_path) -> None:
        """User passed only ``Model_File`` — Output_File is auto-added."""
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        asset = _make_asset_file_path(
            file_name="model.pt", asset_rid="2-AAA", asset_table="Execution_Asset"
        )
        _write_asset_type_jsonl(
            tmp_path,
            "1-EXEC",
            "Execution_Asset",
            ml_schema="deriva-ml",
            type_map={"model.pt": ["Model_File"]},
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Execution_Asset": [asset]},
            asset_role="Output",
        )

        # The {Asset}_Asset_Type insert should carry both Model_File
        # (user-supplied) AND Output_File (auto-added).
        type_inserts = _inserts_for(pb, "Execution_Asset_Asset_Type")
        assert type_inserts, "expected an Execution_Asset_Asset_Type insert"
        all_types = [r["Asset_Type"] for inst in type_inserts for r in inst.rows]
        assert "Model_File" in all_types
        assert "Output_File" in all_types

    def test_output_file_not_duplicated_when_user_supplies_it(self, tmp_path, stub_asset_type_path) -> None:
        """User already passed ``Output_File`` — it appears exactly once.

        Idempotent: a user who explicitly tags ``ExecAssetType.output_file``
        in their ``asset_file_path()`` call doesn't end up with the
        tag inserted twice.
        """
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        asset = _make_asset_file_path(
            file_name="out.csv", asset_rid="2-BBB", asset_table="Execution_Asset"
        )
        _write_asset_type_jsonl(
            tmp_path,
            "1-EXEC",
            "Execution_Asset",
            ml_schema="deriva-ml",
            type_map={"out.csv": ["Output_File"]},
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Execution_Asset": [asset]},
            asset_role="Output",
        )

        # The Output_File tag appears in the insert rows exactly once.
        type_inserts = _inserts_for(pb, "Execution_Asset_Asset_Type")
        output_file_rows = [
            r for inst in type_inserts for r in inst.rows if r["Asset_Type"] == "Output_File"
        ]
        assert len(output_file_rows) == 1, (
            "Output_File should be present exactly once, even when the "
            "user explicitly supplied it. Got rows: %s" % output_file_rows
        )

    def test_output_file_written_to_asset_file_path_asset_types(self, tmp_path, stub_asset_type_path) -> None:
        """``asset_path.asset_types`` is updated to include Output_File.

        Downstream consumers (e.g., MCP responses) read
        ``AssetFilePath.asset_types`` to learn what types the
        uploaded asset has. After auto-tagging, that field must
        also reflect Output_File.
        """
        from deriva_ml.execution.execution import Execution

        execution, _ = _build_execution_fixture(tmp_path)
        asset = _make_asset_file_path(
            file_name="model.pt", asset_rid="2-CCC", asset_table="Execution_Asset"
        )
        _write_asset_type_jsonl(
            tmp_path,
            "1-EXEC",
            "Execution_Asset",
            ml_schema="deriva-ml",
            type_map={"model.pt": ["Model_File"]},
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Execution_Asset": [asset]},
            asset_role="Output",
        )

        assert "Output_File" in asset.asset_types
        assert "Model_File" in asset.asset_types

    def test_execution_link_uses_output_role(self, tmp_path, stub_asset_type_path) -> None:
        """The ``{Asset}_Execution`` row is written with ``Asset_Role="Output"``."""
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        asset = _make_asset_file_path(
            file_name="x.csv", asset_rid="2-DDD", asset_table="Execution_Asset"
        )
        _write_asset_type_jsonl(
            tmp_path,
            "1-EXEC",
            "Execution_Asset",
            ml_schema="deriva-ml",
            type_map={"x.csv": ["Model_File"]},
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Execution_Asset": [asset]},
            asset_role="Output",
        )

        exec_inserts = _inserts_for(pb, "Execution_Asset_Execution")
        assert exec_inserts
        roles = [r["Asset_Role"] for inst in exec_inserts for r in inst.rows]
        assert roles == ["Output"]


# ---------------------------------------------------------------------------
# Test cases — Input branch
# ---------------------------------------------------------------------------


class TestInputAutoTag:
    """Auto-tag with Input_File when role is Input."""

    def test_input_file_inserted_for_input_role(self, tmp_path) -> None:
        """A downloaded input asset gets Input_File on {Asset}_Asset_Type.

        Closes the regression where the input branch returned before
        writing any ``Asset_Type`` entries. After this fix, every
        asset consumed as an execution input is tagged with
        ``Input_File`` alongside its existing content types (which
        are preserved unchanged on the asset itself).
        """
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        # No JSONL file needed for the Input branch — the input path
        # doesn't read user-supplied types because the asset already
        # exists in the catalog with whatever types it has.
        asset = _make_asset_file_path(
            file_name="input.csv", asset_rid="3-AAA", asset_table="Image"
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Image": [asset]},
            asset_role="Input",
        )

        # The Input branch writes Input_File to {Asset}_Asset_Type.
        type_inserts = _inserts_for(pb, "Image_Asset_Type")
        assert type_inserts, (
            "Input branch must write to {Asset}_Asset_Type with the "
            "Input_File tag — without this, asset role direction "
            "isn't queryable through Asset_Type."
        )
        all_types = [r["Asset_Type"] for inst in type_inserts for r in inst.rows]
        assert "Input_File" in all_types

    def test_input_branch_uses_on_conflict_skip(self, tmp_path) -> None:
        """Re-downloading the same asset doesn't duplicate the Input_File row.

        The {Asset}_Asset_Type insert is wrapped in
        ``on_conflict_skip=True`` so repeated execution-input
        registrations of the same asset are safe.
        """
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        asset = _make_asset_file_path(
            file_name="img.png", asset_rid="3-BBB", asset_table="Image"
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Image": [asset]},
            asset_role="Input",
        )

        type_inserts = _inserts_for(pb, "Image_Asset_Type")
        assert type_inserts
        assert all(inst.on_conflict_skip for inst in type_inserts), (
            "Input-branch Asset_Type insert must use on_conflict_skip=True "
            "so re-downloads are idempotent."
        )

    def test_input_branch_links_all_tables(self, tmp_path) -> None:
        """When inputs span multiple asset tables, ALL get linked.

        Regression guard for a quiet bug: the prior implementation
        used ``return`` inside the for loop in the Input branch,
        which stopped iteration after the first table. The fix
        switched to ``continue``, so every asset_table gets both
        the {Asset}_Execution link AND the Input_File tag.
        """
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        image_asset = _make_asset_file_path(
            file_name="img.png", asset_rid="3-CCC", asset_table="Image"
        )
        model_asset = _make_asset_file_path(
            file_name="model.pt", asset_rid="3-DDD", asset_table="Model"
        )

        Execution._update_asset_execution_table(
            execution,
            {
                "deriva-ml/Image": [image_asset],
                "deriva-ml/Model": [model_asset],
            },
            asset_role="Input",
        )

        # Both tables must have their _Execution AND _Asset_Type rows.
        image_exec = _inserts_for(pb, "Image_Execution")
        image_type = _inserts_for(pb, "Image_Asset_Type")
        model_exec = _inserts_for(pb, "Model_Execution")
        model_type = _inserts_for(pb, "Model_Asset_Type")
        assert image_exec, "Image_Execution link missing"
        assert image_type, "Image_Asset_Type Input_File tag missing"
        assert model_exec, "Model_Execution link missing — Input branch stopped after first table"
        assert model_type, "Model_Asset_Type Input_File tag missing — Input branch stopped after first table"

    def test_execution_link_uses_input_role(self, tmp_path) -> None:
        """The ``{Asset}_Execution`` row is written with ``Asset_Role="Input"``."""
        from deriva_ml.execution.execution import Execution

        execution, pb = _build_execution_fixture(tmp_path)
        asset = _make_asset_file_path(
            file_name="x.png", asset_rid="3-EEE", asset_table="Image"
        )

        Execution._update_asset_execution_table(
            execution,
            {"deriva-ml/Image": [asset]},
            asset_role="Input",
        )

        exec_inserts = _inserts_for(pb, "Image_Execution")
        assert exec_inserts
        roles = [r["Asset_Role"] for inst in exec_inserts for r in inst.rows]
        assert roles == ["Input"]


# ---------------------------------------------------------------------------
# Test cases — orthogonality
# ---------------------------------------------------------------------------


class TestRoleAxisIsMultiValued:
    """The directional tag is additive to other content tags."""

    def test_asset_can_carry_both_input_and_output_tags(self, tmp_path, stub_asset_type_path) -> None:
        """An asset that's Output of one execution and Input to another
        ends up with both Output_File AND Input_File in its type list.

        This is the intended multi-valued shape — Asset_Type is
        additive across an asset's full lifecycle, with each tag
        truthfully reflecting one of its roles.
        """
        from deriva_ml.execution.execution import Execution

        # First execution: asset is an Output.
        execution1, pb1 = _build_execution_fixture(tmp_path)
        asset_as_output = _make_asset_file_path(
            file_name="artifact.bin", asset_rid="4-AAA", asset_table="Execution_Asset"
        )
        _write_asset_type_jsonl(
            tmp_path,
            "1-EXEC",
            "Execution_Asset",
            ml_schema="deriva-ml",
            type_map={"artifact.bin": ["Model_File"]},
        )
        Execution._update_asset_execution_table(
            execution1,
            {"deriva-ml/Execution_Asset": [asset_as_output]},
            asset_role="Output",
        )

        # Second execution: the same asset RID is now an Input.
        execution2, pb2 = _build_execution_fixture(tmp_path)
        asset_as_input = _make_asset_file_path(
            file_name="artifact.bin", asset_rid="4-AAA", asset_table="Execution_Asset"
        )
        Execution._update_asset_execution_table(
            execution2,
            {"deriva-ml/Execution_Asset": [asset_as_input]},
            asset_role="Input",
        )

        # Across the two executions, the asset's Asset_Type writes
        # should include both Output_File (from exec 1) and
        # Input_File (from exec 2).
        type_writes_1 = _inserts_for(pb1, "Execution_Asset_Asset_Type")
        type_writes_2 = _inserts_for(pb2, "Execution_Asset_Asset_Type")
        types_1 = [r["Asset_Type"] for inst in type_writes_1 for r in inst.rows]
        types_2 = [r["Asset_Type"] for inst in type_writes_2 for r in inst.rows]
        assert "Output_File" in types_1
        assert "Input_File" in types_2
        # And the asset's original content type is preserved across both.
        assert "Model_File" in types_1


# ---------------------------------------------------------------------------
# Test case — dry-run respects the no-op contract
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path) -> None:
    """``_dry_run=True`` short-circuits before any catalog write."""
    from deriva_ml.execution.execution import Execution

    execution, pb = _build_execution_fixture(tmp_path, dry_run=True)
    asset = _make_asset_file_path(
        file_name="x.csv", asset_rid="5-AAA", asset_table="Execution_Asset"
    )

    Execution._update_asset_execution_table(
        execution,
        {"deriva-ml/Execution_Asset": [asset]},
        asset_role="Output",
    )

    assert pb.inserts == [], (
        "dry_run=True must skip every catalog insert. Recorded inserts: %s" % pb.inserts
    )

"""Tests for the schema-doc validator (spec §6)."""

from __future__ import annotations


def test_schema_model_fields():
    from deriva_ml.tools.validate_schema_doc import (
        SchemaModel,
        TableModel,
    )
    m = SchemaModel(tables=[])
    assert m.tables == []

    t = TableModel(
        name="Dataset",
        kind="table",
        columns=[],
        foreign_keys=[],
        terms=[],
        associates=[],
        metadata=[],
    )
    assert t.name == "Dataset"
    assert t.kind == "table"


def test_column_model_fields():
    from deriva_ml.tools.validate_schema_doc import ColumnModel
    c = ColumnModel(name="Status", type="text")
    assert c.name == "Status"
    assert c.type == "text"


def test_fk_model_fields():
    from deriva_ml.tools.validate_schema_doc import ForeignKeyModel
    fk = ForeignKeyModel(
        columns=["Workflow"],
        referenced_schema="deriva-ml",
        referenced_table="Workflow",
        referenced_columns=["RID"],
    )
    assert fk.columns == ["Workflow"]
    assert fk.referenced_table == "Workflow"


def test_vocabulary_term_model():
    from deriva_ml.tools.validate_schema_doc import VocabularyTermModel
    term = VocabularyTermModel(name="Running")
    assert term.name == "Running"


def test_mismatch_kinds_defined():
    """Enum of mismatch kinds is present."""
    from deriva_ml.tools.validate_schema_doc import MismatchKind
    assert MismatchKind.MISSING_TABLE.value == "missing_table"
    assert MismatchKind.COLUMN_MISMATCH.value == "column_mismatch"
    assert MismatchKind.FK_MISMATCH.value == "fk_mismatch"
    assert MismatchKind.VOCAB_TERMS_MISMATCH.value == "vocab_terms_mismatch"


def test_extract_yaml_blocks_basic(tmp_path):
    """Extracts YAML-fenced blocks and parses each into a dict."""
    from deriva_ml.tools.validate_schema_doc import _extract_yaml_blocks

    doc = tmp_path / "schema.md"
    doc.write_text(
        "# Intro\n"
        "\n"
        "## Dataset\n"
        "\n"
        "```yaml\n"
        "table: Dataset\n"
        "kind: table\n"
        "columns:\n"
        "  - name: Name\n"
        "    type: text\n"
        "```\n"
        "\n"
        "More prose.\n"
        "\n"
        "```yaml\n"
        "table: Workflow\n"
        "kind: table\n"
        "```\n"
    )

    blocks = _extract_yaml_blocks(doc)
    assert len(blocks) == 2
    assert blocks[0]["table"] == "Dataset"
    assert blocks[0]["columns"][0]["name"] == "Name"
    assert blocks[1]["table"] == "Workflow"


def test_extract_yaml_blocks_ignores_non_yaml_fences(tmp_path):
    """Code blocks fenced as other languages are not parsed as YAML."""
    from deriva_ml.tools.validate_schema_doc import _extract_yaml_blocks

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```python\n"
        "x = 1\n"
        "```\n"
        "\n"
        "```yaml\n"
        "table: Dataset\n"
        "kind: table\n"
        "```\n"
    )

    blocks = _extract_yaml_blocks(doc)
    assert len(blocks) == 1
    assert blocks[0]["table"] == "Dataset"


def test_load_from_doc_plain_table(tmp_path):
    """A plain table becomes a TableModel with kind='table'."""
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Dataset\n"
        "kind: table\n"
        "description: A data collection.\n"
        "columns:\n"
        "  - name: Name\n"
        "    type: text\n"
        "  - name: Description\n"
        "    type: markdown\n"
        "foreign_keys: []\n"
        "```\n"
    )
    model = load_from_doc(doc)
    assert len(model.tables) == 1
    t = model.tables[0]
    assert t.name == "Dataset"
    assert t.kind == "table"
    assert len(t.columns) == 2
    assert t.columns[0].name == "Name"
    assert t.columns[0].type == "text"


def test_load_from_doc_vocabulary(tmp_path):
    """A vocabulary table gets terms populated."""
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Asset_Type\n"
        "kind: vocabulary\n"
        "terms:\n"
        "  - name: Execution_Config\n"
        "  - name: Runtime_Env\n"
        "```\n"
    )
    model = load_from_doc(doc)
    t = model.tables[0]
    assert t.kind == "vocabulary"
    assert len(t.terms) == 2
    assert [term.name for term in t.terms] == ["Execution_Config", "Runtime_Env"]


def test_load_from_doc_association(tmp_path):
    """An association table gets associates populated."""
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Dataset_Execution\n"
        "kind: association\n"
        "associates:\n"
        "  - table: Dataset\n"
        "  - table: Execution\n"
        "```\n"
    )
    model = load_from_doc(doc)
    t = model.tables[0]
    assert t.kind == "association"
    assert [a.table for a in t.associates] == ["Dataset", "Execution"]


def test_load_from_doc_foreign_keys(tmp_path):
    """FKs parse correctly."""
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Execution\n"
        "kind: table\n"
        "columns:\n"
        "  - name: Workflow\n"
        "    type: text\n"
        "foreign_keys:\n"
        "  - columns: [Workflow]\n"
        "    referenced_schema: deriva-ml\n"
        "    referenced_table: Workflow\n"
        "    referenced_columns: [RID]\n"
        "```\n"
    )
    model = load_from_doc(doc)
    fk = model.tables[0].foreign_keys[0]
    assert fk.columns == ["Workflow"]
    assert fk.referenced_table == "Workflow"
    assert fk.referenced_columns == ["RID"]


def test_load_from_doc_invalid_kind_raises(tmp_path):
    """Unknown 'kind:' value raises SchemaDocError."""
    import pytest
    from deriva_ml.tools.validate_schema_doc import SchemaDocError, load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Bogus\n"
        "kind: not-a-real-kind\n"
        "```\n"
    )
    with pytest.raises(SchemaDocError, match="unknown kind"):
        load_from_doc(doc)


def test_load_from_doc_missing_required_key_raises(tmp_path):
    """Missing required key 'kind' raises SchemaDocError."""
    import pytest
    from deriva_ml.tools.validate_schema_doc import SchemaDocError, load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Bogus\n"
        "```\n"
    )
    with pytest.raises(SchemaDocError, match="missing required key"):
        load_from_doc(doc)


def test_load_from_doc_accepts_descriptions(tmp_path):
    """Descriptions on tables and columns are tolerated (doc-side only)."""
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: Dataset\n"
        "kind: table\n"
        "description: A collection of records.\n"
        "columns:\n"
        "  - name: Name\n"
        "    type: text\n"
        "    description: Human-readable label.\n"
        "```\n"
    )
    model = load_from_doc(doc)
    # Description text is NOT preserved in the model (doc-side only).
    t = model.tables[0]
    assert t.name == "Dataset"
    assert t.columns[0].name == "Name"
    # ColumnModel has no description field.
    assert not hasattr(t.columns[0], "description")


def test_resolve_enum_ref_mltable():
    """Resolver returns the enum value for MLTable.execution."""
    from deriva_ml.tools.validate_schema_doc import _resolve_enum_ref

    assert _resolve_enum_ref("MLTable", "execution") == "Execution"
    assert _resolve_enum_ref("MLTable", "dataset") == "Dataset"


def test_resolve_enum_ref_mlvocab():
    from deriva_ml.tools.validate_schema_doc import _resolve_enum_ref

    assert _resolve_enum_ref("MLVocab", "workflow_type") == "Workflow_Type"
    assert _resolve_enum_ref("MLVocab", "asset_type") == "Asset_Type"


def test_resolve_enum_ref_unknown_raises():
    import pytest
    from deriva_ml.tools.validate_schema_doc import SchemaCodeError, _resolve_enum_ref

    with pytest.raises(SchemaCodeError, match="unknown enum"):
        _resolve_enum_ref("UnknownEnum", "whatever")


def test_resolve_enum_ref_unknown_member_raises():
    import pytest
    from deriva_ml.tools.validate_schema_doc import SchemaCodeError, _resolve_enum_ref

    with pytest.raises(SchemaCodeError, match="has no member"):
        _resolve_enum_ref("MLTable", "no_such_member")

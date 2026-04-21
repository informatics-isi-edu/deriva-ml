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

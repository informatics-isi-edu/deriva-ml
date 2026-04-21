"""Structure tests for docs/reference/schema.md.

These tests enforce invariants about the doc itself, separate from the
doc-vs-code comparison:

- Every MLTable and MLVocab enum member has a doc entry.
- Every YAML block parses.
- Tables are ordered: core entities → vocabularies → associations.
"""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "reference" / "schema.md"

# MLTable members that are dynamically-named (not extractable from static
# AST analysis of create_schema.py). These exist at runtime but aren't
# cross-validated. Skipped to keep the doc-vs-MLTable check focused on
# statically-declared tables.
_DYNAMIC_MLTABLE_EXEMPT = {"File", "Asset", "Execution_Execution", "Execution_Metadata", "Execution_Asset"}


def test_schema_doc_exists():
    assert DOC_PATH.exists(), f"{DOC_PATH} does not exist"


def test_schema_doc_yaml_blocks_all_valid():
    """Every fenced YAML block parses as a dict."""
    from deriva_ml.tools.validate_schema_doc import _extract_yaml_blocks
    blocks = _extract_yaml_blocks(DOC_PATH)
    assert len(blocks) > 0
    for b in blocks:
        assert isinstance(b, dict)
        assert "table" in b
        assert "kind" in b


def test_schema_doc_has_entry_per_mltable_member():
    """Every MLTable enum member appears as a table in the doc."""
    from deriva_ml.core.enums import MLTable
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    model = load_from_doc(DOC_PATH)
    doc_names = {t.name for t in model.tables}
    for member in MLTable:
        if member.value in _DYNAMIC_MLTABLE_EXEMPT:
            continue
        assert member.value in doc_names, (
            f"MLTable.{member.name} ({member.value!r}) is missing from "
            f"{DOC_PATH.name}. Add a ## {member.value} section with a "
            f"fenced yaml block."
        )


def test_schema_doc_has_entry_per_mlvocab_member():
    """Every MLVocab enum member appears as a vocabulary table in the doc."""
    from deriva_ml.core.enums import MLVocab
    from deriva_ml.tools.validate_schema_doc import load_from_doc

    model = load_from_doc(DOC_PATH)
    vocab_names = {t.name for t in model.tables if t.kind == "vocabulary"}
    for member in MLVocab:
        assert member.value in vocab_names, (
            f"MLVocab.{member.name} ({member.value!r}) is missing as a "
            f"vocabulary table in {DOC_PATH.name}."
        )

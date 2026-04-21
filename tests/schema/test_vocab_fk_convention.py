"""Live-catalog audit: every FK targeting a vocabulary table references Name.

Creates a fresh catalog via create_ml_catalog, introspects the deriva-ml schema,
and asserts that every FK whose target table is a vocabulary (has a unique Name
key) references the Name column — never RID or another column.

This test captures the library-wide convention: 'FKs to vocabulary tables
reference Name, not RID.' Requires DERIVA_HOST.
"""

from __future__ import annotations

import pytest


def _is_vocabulary_table(table) -> bool:
    """A vocabulary table has a unique Name key (per VocabularyTableDef)."""
    cols = {c.name for c in table.column_definitions}
    if not {"Name", "ID", "URI"}.issubset(cols):
        return False
    for k in table.keys:
        kcols = {c.name for c in k.unique_columns}
        if kcols == {"Name"}:
            return True
    return False


@pytest.mark.integration
def test_all_vocab_fks_reference_name():
    """Create a fresh catalog and verify every vocab-targeting FK is on Name."""
    from deriva_ml.schema.create_schema import create_ml_catalog

    catalog = create_ml_catalog(
        hostname="localhost", project_name="s1b_audit_test",
    )
    try:
        model = catalog.getCatalogModel()
        schema = model.schemas["deriva-ml"]

        vocab_tables = {
            t.name for t in schema.tables.values() if _is_vocabulary_table(t)
        }
        assert vocab_tables, "No vocabulary tables found in deriva-ml schema"

        violations: list[str] = []
        for t in schema.tables.values():
            for fk in t.foreign_keys:
                tgt_cols = [c.name for c in fk.referenced_columns]
                tgt_table = fk.referenced_columns[0].table.name
                if tgt_table in vocab_tables and tgt_cols != ["Name"]:
                    src_cols = [c.name for c in fk.foreign_key_columns]
                    violations.append(
                        f"{t.name}({','.join(src_cols)}) → "
                        f"{tgt_table}({','.join(tgt_cols)}) "
                        f"(expected target Name)"
                    )

        assert not violations, (
            f"Vocabulary-FK convention violated:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
    finally:
        catalog.delete_ermrest_catalog(really=True)

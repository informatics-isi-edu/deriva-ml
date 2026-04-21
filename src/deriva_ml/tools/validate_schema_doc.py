"""Validator: assert docs/reference/schema.md and create_schema.py agree.

Path-1 architecture (per spec §2): both files are maintained by developers.
The doc describes intended schema; the code defines it at runtime. This
validator compares the two.

Runs in CI via the deriva-ml-validate-schema entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True)
class ColumnModel:
    """Column definition (name + ERMrest type)."""
    name: str
    type: str


@dataclass(frozen=True)
class ForeignKeyModel:
    """FK from a set of columns to a referenced table's columns."""
    columns: list[str]
    referenced_schema: str
    referenced_table: str
    referenced_columns: list[str]


@dataclass(frozen=True)
class VocabularyTermModel:
    """One seeded term in a vocabulary table."""
    name: str


@dataclass(frozen=True)
class AssociationEndpointModel:
    """One endpoint of an association table."""
    table: str
    role: str | None = None


@dataclass(frozen=True)
class TableModel:
    """One table in the schema — plain, vocabulary, or association."""
    name: str
    kind: str  # "table", "vocabulary", "association"
    columns: list[ColumnModel] = field(default_factory=list)
    foreign_keys: list[ForeignKeyModel] = field(default_factory=list)
    terms: list[VocabularyTermModel] = field(default_factory=list)
    associates: list[AssociationEndpointModel] = field(default_factory=list)
    metadata: list[ColumnModel] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaModel:
    """The full schema — one list of tables."""
    tables: list[TableModel]


class MismatchKind(StrEnum):
    """Categories of doc↔code mismatch."""
    MISSING_TABLE = "missing_table"
    EXTRA_TABLE = "extra_table"
    COLUMN_MISMATCH = "column_mismatch"
    FK_MISMATCH = "fk_mismatch"
    VOCAB_TERMS_MISMATCH = "vocab_terms_mismatch"
    ASSOCIATION_MISMATCH = "association_mismatch"

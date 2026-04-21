"""Validator: assert docs/reference/schema.md and create_schema.py agree.

Path-1 architecture (per spec §2): both files are maintained by developers.
The doc describes intended schema; the code defines it at runtime. This
validator compares the two.

Runs in CI via the deriva-ml-validate-schema entry point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml


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


class SchemaDocError(Exception):
    """Raised when the schema doc can't be parsed or is malformed."""


_YAML_FENCE_RE = re.compile(
    r"^```yaml\s*$(.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


def _extract_yaml_blocks(path: Path) -> list[dict]:
    """Extract YAML-fenced code blocks from a Markdown file.

    Args:
        path: Path to the Markdown file.

    Returns:
        List of dicts, one per ```yaml block, in file order.

    Raises:
        SchemaDocError: If any block fails to parse as YAML.
    """
    text = path.read_text()
    blocks: list[dict] = []
    for match in _YAML_FENCE_RE.finditer(text):
        body = match.group(1)
        line_number = text[: match.start()].count("\n") + 1
        try:
            parsed = yaml.safe_load(body)
        except yaml.YAMLError as exc:
            raise SchemaDocError(
                f"YAML parse error in {path}:{line_number}: {exc}"
            ) from exc
        if parsed is None:
            continue
        if not isinstance(parsed, dict):
            raise SchemaDocError(
                f"{path}:{line_number}: expected a mapping, got {type(parsed).__name__}"
            )
        blocks.append(parsed)
    return blocks


_VALID_KINDS = frozenset({"table", "vocabulary", "association"})


def load_from_doc(path: Path) -> SchemaModel:
    """Parse a schema doc Markdown file into a SchemaModel.

    Args:
        path: Path to docs/reference/schema.md or equivalent.

    Returns:
        SchemaModel with one TableModel per ```yaml block.

    Raises:
        SchemaDocError: If any block is malformed (unknown kind, missing
            required keys, parse error).
    """
    blocks = _extract_yaml_blocks(path)
    tables: list[TableModel] = []
    for block in blocks:
        if "table" not in block:
            raise SchemaDocError(
                f"{path}: block missing required key 'table': {block}"
            )
        if "kind" not in block:
            raise SchemaDocError(
                f"{path}: block missing required key 'kind' for table "
                f"{block['table']!r}"
            )
        kind = block["kind"]
        if kind not in _VALID_KINDS:
            raise SchemaDocError(
                f"{path}: table {block['table']!r} has unknown kind {kind!r}; "
                f"expected one of {sorted(_VALID_KINDS)}"
            )
        columns = [
            ColumnModel(name=c["name"], type=c["type"])
            for c in block.get("columns", [])
        ]
        foreign_keys = [
            ForeignKeyModel(
                columns=fk["columns"],
                referenced_schema=fk["referenced_schema"],
                referenced_table=fk["referenced_table"],
                referenced_columns=fk["referenced_columns"],
            )
            for fk in block.get("foreign_keys", [])
        ]
        terms = [
            VocabularyTermModel(name=t["name"])
            for t in block.get("terms", [])
        ]
        associates = [
            AssociationEndpointModel(table=a["table"], role=a.get("role"))
            for a in block.get("associates", [])
        ]
        metadata = [
            ColumnModel(name=m["name"], type=m["type"])
            for m in block.get("metadata", [])
        ]
        tables.append(TableModel(
            name=block["table"],
            kind=kind,
            columns=columns,
            foreign_keys=foreign_keys,
            terms=terms,
            associates=associates,
            metadata=metadata,
        ))
    return SchemaModel(tables=tables)

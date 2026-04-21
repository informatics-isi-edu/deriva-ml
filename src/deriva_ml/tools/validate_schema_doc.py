"""Validator: assert docs/reference/schema.md and create_schema.py agree.

Path-1 architecture (per spec §2): both files are maintained by developers.
The doc describes intended schema; the code defines it at runtime. This
validator compares the two.

Runs in CI via the deriva-ml-validate-schema entry point.
"""

from __future__ import annotations

import ast
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


class SchemaCodeError(Exception):
    """Raised when create_schema.py contains a pattern the validator can't extract."""


# Enum lookup for resolving MLTable.xxx / MLVocab.xxx references.
# Populated lazily via _resolve_enum_ref to avoid circular-import risk.
_ENUM_LOOKUP: dict[str, dict[str, str]] = {}


def _load_enum_lookup() -> dict[str, dict[str, str]]:
    """Build {enum_name: {member_name: value}} for MLTable and MLVocab."""
    global _ENUM_LOOKUP
    if _ENUM_LOOKUP:
        return _ENUM_LOOKUP
    from deriva_ml.core.enums import MLTable, MLVocab

    _ENUM_LOOKUP = {
        "MLTable": {m.name: m.value for m in MLTable},
        "MLVocab": {m.name: m.value for m in MLVocab},
    }
    return _ENUM_LOOKUP


def _resolve_enum_ref(enum_name: str, member: str) -> str:
    """Resolve an enum-member reference like MLTable.execution → 'Execution'."""
    lookup = _load_enum_lookup()
    if enum_name not in lookup:
        raise SchemaCodeError(
            f"unknown enum {enum_name!r}; expected one of {sorted(lookup)}"
        )
    members = lookup[enum_name]
    if member not in members:
        raise SchemaCodeError(
            f"{enum_name} has no member {member!r}; known: {sorted(members)}"
        )
    return members[member]


def _extract_ast_str(node: ast.AST) -> str:
    """Extract a string from an ast.Constant or return the ast.unparse repr."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Attribute access like BuiltinType.text → "text"
    if isinstance(node, ast.Attribute):
        return node.attr
    raise SchemaCodeError(
        f"expected string constant or attribute, got {ast.dump(node)}"
    )


def _extract_ast_name_or_enum(node: ast.AST) -> str:
    """Extract a table-name string from a literal, enum ref, or attr access."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        # Expect: MLTable.execution or MLVocab.workflow_type
        if isinstance(node.value, ast.Name):
            return _resolve_enum_ref(node.value.id, node.attr)
    raise SchemaCodeError(
        f"expected string literal or enum reference, got {ast.dump(node)}"
    )


def _extract_ast_str_list(node: ast.AST) -> list[str]:
    """Extract a list[str] from an ast.List node."""
    if not isinstance(node, ast.List):
        raise SchemaCodeError(
            f"expected list literal, got {ast.dump(node)}"
        )
    return [_extract_ast_str(elt) for elt in node.elts]


def _extract_column(node: ast.Call) -> ColumnModel:
    """Extract a ColumnDef(...) call into a ColumnModel."""
    # Positional or keyword args: ColumnDef("Name", BuiltinType.text)
    name: str | None = None
    type_: str | None = None
    if node.args:
        name = _extract_ast_str(node.args[0])
        if len(node.args) > 1:
            type_ = _extract_ast_str(node.args[1])
    for kw in node.keywords:
        if kw.arg == "name":
            name = _extract_ast_str(kw.value)
        elif kw.arg == "type":
            type_ = _extract_ast_str(kw.value)
    if name is None or type_ is None:
        raise SchemaCodeError(
            f"ColumnDef missing name or type: {ast.dump(node)}"
        )
    return ColumnModel(name=name, type=type_)


def _extract_fk(node: ast.Call) -> ForeignKeyModel:
    """Extract a ForeignKeyDef(...) call into a ForeignKeyModel."""
    columns: list[str] = []
    referenced_schema: str = ""
    referenced_table: str = ""
    referenced_columns: list[str] = []
    for kw in node.keywords:
        if kw.arg == "columns":
            columns = _extract_ast_str_list(kw.value)
        elif kw.arg == "referenced_schema":
            referenced_schema = _extract_ast_str(kw.value)
        elif kw.arg == "referenced_table":
            referenced_table = _extract_ast_str(kw.value)
        elif kw.arg == "referenced_columns":
            referenced_columns = _extract_ast_str_list(kw.value)
    return ForeignKeyModel(
        columns=columns,
        referenced_schema=referenced_schema,
        referenced_table=referenced_table,
        referenced_columns=referenced_columns,
    )


def _extract_tabledef(node: ast.Call) -> TableModel:
    """Extract a TableDef(name=..., columns=[...], foreign_keys=[...]) call."""
    name: str = ""
    columns: list[ColumnModel] = []
    foreign_keys: list[ForeignKeyModel] = []
    for kw in node.keywords:
        if kw.arg == "name":
            name = _extract_ast_name_or_enum(kw.value)
        elif kw.arg == "columns":
            if isinstance(kw.value, ast.List):
                columns = [
                    _extract_column(elt)
                    for elt in kw.value.elts
                    if isinstance(elt, ast.Call) and _callable_name(elt) == "ColumnDef"
                ]
        elif kw.arg == "foreign_keys":
            if isinstance(kw.value, ast.List):
                foreign_keys = [
                    _extract_fk(elt)
                    for elt in kw.value.elts
                    if isinstance(elt, ast.Call) and _callable_name(elt) == "ForeignKeyDef"
                ]
    return TableModel(
        name=name,
        kind="table",
        columns=columns,
        foreign_keys=foreign_keys,
    )


def _callable_name(node: ast.Call) -> str:
    """Return the callable's name: 'TableDef', 'Foo.bar', etc."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        cur: ast.AST = node.func
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.insert(0, cur.id)
        return ".".join(parts)
    return ""


def _extract_vocabulary(node: ast.Call) -> TableModel:
    """Extract a VocabularyTableDef(name=..., curie_template=...) call."""
    name: str = ""
    for kw in node.keywords:
        if kw.arg == "name":
            name = _extract_ast_name_or_enum(kw.value)
    return TableModel(name=name, kind="vocabulary")


def _extract_ensure_terms(node: ast.Call) -> tuple[str, list[VocabularyTermModel]]:
    """Extract (vocab_name, terms) from an _ensure_terms(vocab, [...]) call."""
    if len(node.args) < 2:
        raise SchemaCodeError(
            f"_ensure_terms expects 2 positional args, got {len(node.args)}"
        )
    vocab_name = _extract_ast_name_or_enum(node.args[0])
    terms_list = node.args[1]
    if not isinstance(terms_list, ast.List):
        raise SchemaCodeError(
            f"_ensure_terms second arg must be a list literal, got "
            f"{ast.dump(terms_list)}"
        )
    terms: list[VocabularyTermModel] = []
    for elt in terms_list.elts:
        if not isinstance(elt, ast.Dict):
            raise SchemaCodeError(
                f"_ensure_terms term must be a dict literal, got "
                f"{ast.dump(elt)}"
            )
        name_val: str | None = None
        for k, v in zip(elt.keys, elt.values, strict=True):
            if (
                isinstance(k, ast.Constant)
                and k.value == "Name"
                and isinstance(v, ast.Constant)
                and isinstance(v.value, str)
            ):
                name_val = v.value
        if name_val is None:
            raise SchemaCodeError(
                f"_ensure_terms term missing 'Name': {ast.dump(elt)}"
            )
        terms.append(VocabularyTermModel(name=name_val))
    return vocab_name, terms


def load_from_code(path: Path) -> SchemaModel:
    """Parse create_schema.py AST into a SchemaModel.

    Walks the tree for:
      - TableDef(...)              → plain table
      - VocabularyTableDef(...)    → vocabulary table (terms populated below)
      - Table.define_association(...) → association table (C4)
      - _ensure_terms(vocab, [...]) → seed terms for a vocab table

    After walking, vocab tables have their terms populated by matching
    _ensure_terms vocab_name to TableModel.name.

    Args:
        path: Path to create_schema.py or a fixture.

    Returns:
        SchemaModel with all discovered TableModels, vocab terms filled in.

    Raises:
        SchemaCodeError: If a call pattern can't be statically extracted.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    tables: list[TableModel] = []
    # Map from vocab table name → list of VocabularyTermModel to apply later.
    terms_by_vocab: dict[str, list[VocabularyTermModel]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callable_name(node)
        if name == "TableDef":
            tables.append(_extract_tabledef(node))
        elif name == "VocabularyTableDef":
            tables.append(_extract_vocabulary(node))
        elif name == "_ensure_terms":
            vocab_name, terms = _extract_ensure_terms(node)
            terms_by_vocab.setdefault(vocab_name, []).extend(terms)

    # Apply accumulated terms to matching vocab tables.
    tables_with_terms: list[TableModel] = []
    for t in tables:
        if t.kind == "vocabulary" and t.name in terms_by_vocab:
            tables_with_terms.append(TableModel(
                name=t.name,
                kind=t.kind,
                columns=t.columns,
                foreign_keys=t.foreign_keys,
                terms=terms_by_vocab[t.name],
                associates=t.associates,
                metadata=t.metadata,
            ))
        else:
            tables_with_terms.append(t)
    return SchemaModel(tables=tables_with_terms)

"""Validator: assert docs/reference/schema.md and create_schema.py agree.

Path-1 architecture (per spec §2): both files are maintained by developers.
The doc describes intended schema; the code defines it at runtime. This
validator compares the two.

Runs in CI via the deriva-ml-validate-schema entry point.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
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


# Sentinel for AST values the validator can't resolve statically — parameter
# names, attribute chains like `schema.name`, etc. When either side of a diff
# has this sentinel, comparison treats it as "any match." Used chiefly for
# referenced_schema in FKs where create_schema.py passes `sname` / `schema.name`
# as a parameter rather than a literal.
DYNAMIC_VALUE = "<dynamic>"


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


_KNOWN_TYPE_HOLDERS = frozenset({"BuiltinType"})


def _extract_ast_str_tolerant(node: ast.AST) -> str:
    """Like _extract_ast_str, but returns DYNAMIC_VALUE for unknown patterns.

    Used for fields whose code-side value is commonly a parameter or
    attribute chain (e.g., FK referenced_schema = sname or schema.name).
    The diff comparator treats DYNAMIC_VALUE as matching any counterpart.

    Resolution order for ast.Attribute nodes:
      - MLTable / MLVocab enum reference → resolved via _resolve_enum_ref.
      - BuiltinType.text → ".text" (type-like attribute).
      - Anything else (schema.name, self.catalog) → DYNAMIC_VALUE.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            # Enum-backed attr: MLVocab.dataset_type → "Dataset_Type".
            try:
                return _resolve_enum_ref(node.value.id, node.attr)
            except SchemaCodeError:
                pass
            # Known type holders: BuiltinType.text → "text".
            if node.value.id in _KNOWN_TYPE_HOLDERS:
                return node.attr
        # Dotted access on a local value (schema.name, self.catalog, …)
        # can't be resolved statically.
        return DYNAMIC_VALUE
    # ast.Name (parameter reference), ast.Call, etc.: treat as dynamic.
    return DYNAMIC_VALUE


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
            referenced_schema = _extract_ast_str_tolerant(kw.value)
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


def _extract_association(node: ast.Call) -> TableModel | None:
    """Extract Table.define_association(associates, metadata=[...]).

    `associates` may be passed positionally (first positional arg) or as a
    keyword. Endpoint-name references that are parameters (e.g., `asset_name`
    in `create_asset_table`) are dynamic — if either endpoint can't be
    resolved statically, the helper returns None and the caller skips it.
    """
    associates: list[AssociationEndpointModel] = []
    metadata: list[ColumnModel] = []
    assoc_list_node: ast.AST | None = None

    # associates as positional (first arg).
    if node.args:
        assoc_list_node = node.args[0]

    # Or associates= kwarg overrides.
    for kw in node.keywords:
        if kw.arg == "associates":
            assoc_list_node = kw.value
        elif kw.arg == "metadata":
            if isinstance(kw.value, ast.List):
                for m_elt in kw.value.elts:
                    if isinstance(m_elt, ast.Call) and _callable_name(m_elt) == "ColumnDef":
                        metadata.append(_extract_column(m_elt))

    if not isinstance(assoc_list_node, ast.List):
        # Can't resolve associates list → skip this table rather than fail.
        return None

    for elt in assoc_list_node.elts:
        if not isinstance(elt, ast.Tuple) or len(elt.elts) < 1:
            return None  # unexpected shape — skip silently
        table_name = _extract_ast_str_tolerant(elt.elts[0])
        if table_name == DYNAMIC_VALUE:
            # Endpoint name is a parameter like `asset_name` — skip this
            # association since we can't derive a table name.
            return None
        associates.append(AssociationEndpointModel(table=table_name))

    if not associates:
        return None

    derived_name = "_".join(a.table for a in associates)
    return TableModel(
        name=derived_name,
        kind="association",
        associates=associates,
        metadata=metadata,
    )


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
        elif name == "Table.define_association":
            assoc = _extract_association(node)
            if assoc is not None:
                tables.append(assoc)

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


@dataclass(frozen=True)
class Mismatch:
    """One discrepancy between expected and actual schemas."""
    kind: "MismatchKind"
    table: str | None
    detail: str


def diff_schemas(*, expected: SchemaModel, actual: SchemaModel) -> list[Mismatch]:
    """Compare two schemas; return list of Mismatches.

    Args:
        expected: The "expected" schema — typically the doc side.
        actual: The "actual" schema — typically the code side.

    Returns:
        Empty list if schemas match. Otherwise one Mismatch per discrepancy.
    """
    mismatches: list[Mismatch] = []
    expected_tables = {t.name: t for t in expected.tables}
    actual_tables = {t.name: t for t in actual.tables}

    # Missing from actual (in doc but not in code).
    for name in sorted(expected_tables.keys() - actual_tables.keys()):
        mismatches.append(Mismatch(
            kind=MismatchKind.MISSING_TABLE,
            table=name,
            detail=f"table {name!r} is in the doc but not in the code",
        ))
    # Extra in actual (in code but not in doc).
    for name in sorted(actual_tables.keys() - expected_tables.keys()):
        mismatches.append(Mismatch(
            kind=MismatchKind.EXTRA_TABLE,
            table=name,
            detail=f"table {name!r} is in the code but not in the doc",
        ))
    # Tables in both: D2-D4 extend this loop.
    for name in sorted(expected_tables.keys() & actual_tables.keys()):
        _compare_tables(mismatches, expected_tables[name], actual_tables[name])
    return mismatches


def _compare_tables(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    """Compare two same-named TableModels; append any differences."""
    _compare_columns(mismatches, expected, actual)
    _compare_fks(mismatches, expected, actual)
    if expected.kind == "vocabulary" or actual.kind == "vocabulary":
        _compare_terms(mismatches, expected, actual)
    if expected.kind == "association" or actual.kind == "association":
        _compare_associates(mismatches, expected, actual)


def _compare_columns(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    """Column-by-column diff."""
    expected_cols = {c.name: c for c in expected.columns}
    actual_cols = {c.name: c for c in actual.columns}
    for col_name in sorted(expected_cols.keys() - actual_cols.keys()):
        mismatches.append(Mismatch(
            kind=MismatchKind.COLUMN_MISMATCH,
            table=expected.name,
            detail=f"column {col_name!r} in doc but not in code",
        ))
    for col_name in sorted(actual_cols.keys() - expected_cols.keys()):
        mismatches.append(Mismatch(
            kind=MismatchKind.COLUMN_MISMATCH,
            table=expected.name,
            detail=f"column {col_name!r} in code but not in doc",
        ))
    for col_name in sorted(expected_cols.keys() & actual_cols.keys()):
        exp_c = expected_cols[col_name]
        act_c = actual_cols[col_name]
        if exp_c.type != act_c.type:
            mismatches.append(Mismatch(
                kind=MismatchKind.COLUMN_MISMATCH,
                table=expected.name,
                detail=(
                    f"column {col_name!r}: doc type {exp_c.type!r} "
                    f"vs code type {act_c.type!r}"
                ),
            ))


def _compare_fks(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    """FK comparison — by tuple (columns, referenced_table, referenced_columns).

    FKs are matched by their source columns tuple. If a doc FK and a code
    FK share columns, their referenced_* must match.
    """
    def key(fk: ForeignKeyModel) -> tuple[str, ...]:
        return tuple(fk.columns)

    expected_fks = {key(fk): fk for fk in expected.foreign_keys}
    actual_fks = {key(fk): fk for fk in actual.foreign_keys}

    for k in sorted(expected_fks.keys() - actual_fks.keys()):
        mismatches.append(Mismatch(
            kind=MismatchKind.FK_MISMATCH,
            table=expected.name,
            detail=f"FK on {list(k)} in doc but not in code",
        ))
    for k in sorted(actual_fks.keys() - expected_fks.keys()):
        mismatches.append(Mismatch(
            kind=MismatchKind.FK_MISMATCH,
            table=expected.name,
            detail=f"FK on {list(k)} in code but not in doc",
        ))
    for k in sorted(expected_fks.keys() & actual_fks.keys()):
        exp_fk = expected_fks[k]
        act_fk = actual_fks[k]
        # DYNAMIC_VALUE on either side of referenced_schema matches anything —
        # create_schema.py frequently passes `sname` / `schema.name` which the
        # AST extractor can't resolve statically.
        schemas_match = (
            exp_fk.referenced_schema == DYNAMIC_VALUE
            or act_fk.referenced_schema == DYNAMIC_VALUE
            or exp_fk.referenced_schema == act_fk.referenced_schema
        )
        if not schemas_match or (exp_fk.referenced_table, tuple(exp_fk.referenced_columns)) != (
            act_fk.referenced_table, tuple(act_fk.referenced_columns)
        ):
            mismatches.append(Mismatch(
                kind=MismatchKind.FK_MISMATCH,
                table=expected.name,
                detail=(
                    f"FK on {list(k)} differs: "
                    f"doc → {exp_fk.referenced_schema}.{exp_fk.referenced_table}{exp_fk.referenced_columns}; "
                    f"code → {act_fk.referenced_schema}.{act_fk.referenced_table}{act_fk.referenced_columns}"
                ),
            ))


def _compare_terms(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    """Compare vocabulary seeded terms (by Name)."""
    expected_terms = {t.name for t in expected.terms}
    actual_terms = {t.name for t in actual.terms}
    if expected_terms != actual_terms:
        doc_only = sorted(expected_terms - actual_terms)
        code_only = sorted(actual_terms - expected_terms)
        mismatches.append(Mismatch(
            kind=MismatchKind.VOCAB_TERMS_MISMATCH,
            table=expected.name,
            detail=f"doc-only: {doc_only}; code-only: {code_only}",
        ))


def _compare_associates(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    """Compare association-table endpoints (by table name)."""
    expected_assoc = [a.table for a in expected.associates]
    actual_assoc = [a.table for a in actual.associates]
    if expected_assoc != actual_assoc:
        mismatches.append(Mismatch(
            kind=MismatchKind.ASSOCIATION_MISMATCH,
            table=expected.name,
            detail=f"associates doc {expected_assoc} vs code {actual_assoc}",
        ))


def _format_mismatches(mismatches: list[Mismatch]) -> str:
    """Render mismatches in the §5.5 format."""
    if not mismatches:
        return "deriva-ml-validate-schema: schema.md and create_schema.py agree.\n"

    buckets: dict[MismatchKind, list[Mismatch]] = defaultdict(list)
    for m in mismatches:
        buckets[m.kind].append(m)

    lines = [
        "deriva-ml-validate-schema: schema.md and create_schema.py disagree.",
        "",
    ]

    def add_section(title: str, kind: MismatchKind) -> None:
        lines.append(f"{title}:")
        if kind in buckets:
            for m in buckets[kind]:
                lines.append(f"  - {m.detail}")
        else:
            lines.append("  (none)")
        lines.append("")

    add_section("MISSING FROM CODE", MismatchKind.MISSING_TABLE)
    add_section("EXTRA IN CODE", MismatchKind.EXTRA_TABLE)
    add_section("COLUMN MISMATCH", MismatchKind.COLUMN_MISMATCH)
    add_section("FOREIGN KEY MISMATCH", MismatchKind.FK_MISMATCH)
    add_section("VOCABULARY TERMS MISMATCH", MismatchKind.VOCAB_TERMS_MISMATCH)
    add_section("ASSOCIATION MISMATCH", MismatchKind.ASSOCIATION_MISMATCH)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deriva-ml-validate-schema",
        description=(
            "Validate that docs/reference/schema.md and "
            "src/deriva_ml/schema/create_schema.py agree on tables, "
            "columns, foreign keys, and vocabulary seeded terms."
        ),
    )
    p.add_argument(
        "--doc",
        default="docs/reference/schema.md",
        help="Path to the schema doc (default: docs/reference/schema.md).",
    )
    p.add_argument(
        "--code",
        default="src/deriva_ml/schema/create_schema.py",
        help="Path to create_schema.py (default: src/deriva_ml/schema/create_schema.py).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: compare doc vs code.

    Returns:
        0 if schemas agree.
        1 if they differ.
        2 if parsing failed on either side.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        expected = load_from_doc(Path(args.doc))
    except (SchemaDocError, FileNotFoundError) as exc:
        print(f"error loading doc: {exc}", file=sys.stderr)
        return 2

    try:
        actual = load_from_code(Path(args.code))
    except (SchemaCodeError, FileNotFoundError) as exc:
        print(f"error loading code: {exc}", file=sys.stderr)
        return 2

    mismatches = diff_schemas(expected=expected, actual=actual)
    print(_format_mismatches(mismatches))
    return 1 if mismatches else 0


def to_doc_markdown(model: SchemaModel) -> str:
    """Render a SchemaModel to the docs/reference/schema.md format.

    Used for bootstrap (generate an initial doc from the code) and for
    emergency regeneration. The output is round-trip identical with
    load_from_doc.

    Args:
        model: SchemaModel instance.

    Returns:
        Multi-section Markdown string.
    """
    sections: list[str] = []
    for t in model.tables:
        header = f"## {t.name}\n"
        yaml_body: dict = {
            "table": t.name,
            "kind": t.kind,
        }
        if t.columns:
            yaml_body["columns"] = [
                {"name": c.name, "type": c.type} for c in t.columns
            ]
        if t.foreign_keys:
            yaml_body["foreign_keys"] = [
                {
                    "columns": list(fk.columns),
                    "referenced_schema": fk.referenced_schema,
                    "referenced_table": fk.referenced_table,
                    "referenced_columns": list(fk.referenced_columns),
                }
                for fk in t.foreign_keys
            ]
        elif t.kind == "table":
            yaml_body["foreign_keys"] = []
        if t.terms:
            yaml_body["terms"] = [{"name": term.name} for term in t.terms]
        if t.associates:
            yaml_body["associates"] = [
                {"table": a.table} for a in t.associates
            ]
        if t.metadata:
            yaml_body["metadata"] = [
                {"name": m.name, "type": m.type} for m in t.metadata
            ]
        yaml_text = yaml.safe_dump(
            yaml_body, sort_keys=False, default_flow_style=False,
        )
        block = "```yaml\n" + yaml_text + "```\n"
        sections.append(header + "\n" + block)
    return "\n".join(sections)


if __name__ == "__main__":
    sys.exit(main())

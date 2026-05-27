# deriva-ml Schema Doc Source-of-Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `docs/reference/schema.md` as the authoritative description of the deriva-ml schema, with a CI validator that asserts the doc and `src/deriva_ml/schema/create_schema.py` agree on tables, columns, FKs, and vocabulary seeded terms.

**Architecture:** New dev-tool module `src/deriva_ml/tools/validate_schema_doc.py` exposes `SchemaModel`, two loaders (`load_from_doc` for Markdown + YAML, `load_from_code` for AST-parsed Python), and a `diff_schemas` comparator. CLI entry point `deriva-ml-validate-schema` runs the code↔doc comparison. GitHub Actions workflow runs the CLI on every PR. Bootstrap: generate initial doc from the existing `create_schema.py`, then hand-add prose and ordering.

**Tech Stack:** Python 3.12+, `ast` stdlib, PyYAML (already a direct dep), pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-04-21-schema-doc-source-of-truth-design.md`

---

## Conventions referenced throughout this plan

- **Worktree root:** `/Users/carl/github/deriva-ml/.claude/worktrees/phase2-status-enum/`. All paths below are relative to it.
- **Environment:** `export PATH="/Users/carl/.local/bin:$PATH"` if `uv` not found. `DERIVA_ML_ALLOW_DIRTY=true` for tests. The validator itself needs NO catalog — pure static analysis + YAML parsing.
- **Module layout:** tool lives at `src/deriva_ml/tools/validate_schema_doc.py` (a new sub-package of `deriva_ml`). Tests live at `tests/tools/test_*.py`. This matches how `src/deriva_ml/cli/upload.py` is structured (dev tool shipped in the wheel).
- **Enum shortcuts:** `MLTable` and `MLVocab` are `StrEnum`s in `src/deriva_ml/core/enums.py`. They are safe to import at validator runtime — no catalog, no network calls. `load_from_code` imports them rather than AST-parsing their definitions.
- **`create_schema.py` AST patterns:** the validator recognizes these call patterns in `src/deriva_ml/schema/create_schema.py`:
  - `TableDef(name=..., columns=[...], foreign_keys=[...])` → table
  - `VocabularyTableDef(name=..., curie_template=...)` → vocab table
  - `Table.define_association(associates=[...], metadata=[...])` → association table
  - `ColumnDef(name, type, ...)` → column inside a TableDef
  - `ForeignKeyDef(columns=[...], referenced_schema=..., referenced_table=..., referenced_columns=[...])` → FK inside a TableDef
  - `_ensure_terms(vocab_name, [{"Name": ..., ...}, ...])` → seeded terms list for a vocab
  - Call sites may reference `MLTable.xxx` / `MLVocab.xxx`; the validator resolves these via imported enum lookup.

---

## Task Group overview

Seven task groups, ordered so the tool builds bottom-up then the doc/CI integrate with it.

| Group | Scope | Tasks |
|---|---|---|
| **A** | `SchemaModel` dataclass hierarchy + empty scaffold | 2 tasks |
| **B** | `load_from_doc` (Markdown + YAML parser) | 3 tasks |
| **C** | `load_from_code` (AST-based code parser) | 4 tasks |
| **D** | `diff_schemas` comparator | 4 tasks |
| **E** | CLI entry point + output format | 3 tasks |
| **F** | Bootstrap: generate initial `docs/reference/schema.md` + hand-edit + commit | 2 tasks |
| **G** | CI workflow + `CHANGELOG` + final review | 3 tasks |

Total: ~21 bite-sized tasks.

---

## Task Group A — `SchemaModel` dataclass hierarchy

Build the in-memory representation that both loaders produce and the comparator consumes. Pure data; no I/O.

### Task A1: Create the `tools` sub-package

**Files:**
- Create: `src/deriva_ml/tools/__init__.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_validate_schema_doc.py` (initially empty; tests accumulate through the plan)

- [ ] **Step 1: Create the package init files**

```bash
mkdir -p src/deriva_ml/tools tests/tools
touch src/deriva_ml/tools/__init__.py
touch tests/tools/__init__.py
```

Create `tests/tools/test_validate_schema_doc.py`:

```python
"""Tests for the schema-doc validator (spec §6)."""

from __future__ import annotations
```

- [ ] **Step 2: Verify package is importable**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "import deriva_ml.tools"
```

Expected: no output (successful import).

- [ ] **Step 3: Commit**

```bash
git add src/deriva_ml/tools/__init__.py tests/tools/__init__.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
chore(tools): scaffold deriva_ml.tools sub-package + tests/tools

Empty sub-package for dev-tooling modules (schema validator, future
ones). Matches src/deriva_ml/cli/ pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A2: Add `SchemaModel` + sub-dataclasses

**Files:**
- Create: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
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
```

- [ ] **Step 2: Run to verify failures**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: 5 FAIL with `ModuleNotFoundError: No module named 'deriva_ml.tools.validate_schema_doc'`.

- [ ] **Step 3: Implement the dataclasses**

Create `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
```

Note: `TableModel` cannot be frozen if it contains mutable `list` defaults via `field(default_factory=list)`. Python's `@dataclass(frozen=True)` actually DOES allow `field(default_factory=list)` — the instance's attributes are immutable references, but the lists themselves are mutable. This matches the intended semantics (compare-by-value; don't let anyone reassign `.tables` but DO allow building the list incrementally during parsing).

Wait — `frozen=True` disallows attribute assignment AFTER construction, but construction itself populates via the field default factories. That's fine. Keep `frozen=True` on all of them.

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): SchemaModel dataclass hierarchy for schema-doc validator

Data shape both loaders (doc, code) produce and diff_schemas consumes.
Frozen dataclasses with list fields (defaults via field(default_factory)).

Includes MismatchKind StrEnum for diff outputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group B — `load_from_doc`

Parse the Markdown file. Find fenced `yaml` code blocks. Deserialize each into a `TableModel`.

### Task B1: Markdown YAML-block extraction

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
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
```

- [ ] **Step 2: Run to verify failures**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k extract_yaml_blocks
```

Expected: 2 FAIL with ImportError on `_extract_yaml_blocks`.

- [ ] **Step 3: Implement `_extract_yaml_blocks`**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
import re
from pathlib import Path

import yaml

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


class SchemaDocError(Exception):
    """Raised when the schema doc can't be parsed or is malformed."""
```

Add `import yaml` to the module-level imports at the top of the file (above the `from dataclasses` line).

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k extract_yaml_blocks
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): _extract_yaml_blocks for Markdown with YAML fences

Reads the doc, extracts fenced ```yaml code blocks, parses each via
yaml.safe_load. Non-YAML fences (e.g. python) are skipped. Malformed
YAML raises SchemaDocError with the Markdown line number.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B2: `load_from_doc` — dict → `SchemaModel`

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
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
```

- [ ] **Step 2: Run to verify failures**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k load_from_doc
```

Expected: 6 FAIL with ImportError on `load_from_doc`.

- [ ] **Step 3: Implement `load_from_doc`**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): load_from_doc — parse Markdown YAML blocks → SchemaModel

Handles plain tables, vocabularies, and association tables. Validates
required keys ('table', 'kind') and rejects unknown kinds. Raises
SchemaDocError with path context on malformed blocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B3: Handle descriptions in doc (narrative-only)

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

Per spec §2, descriptions are doc-side narrative and NOT validated. But `load_from_doc` should accept them without error — the parser currently doesn't look at the `description:` key, so it's already tolerated. We just need a test that asserts this.

- [ ] **Step 1: Write test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
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
```

- [ ] **Step 2: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py::test_load_from_doc_accepts_descriptions -v
```

Expected: PASS (nothing new to implement — the existing parser already ignores the `description:` key).

- [ ] **Step 3: Commit (test addition only)**

```bash
git add tests/tools/test_validate_schema_doc.py
git commit -m "test(tools): confirm load_from_doc tolerates descriptions as narrative"
```

---

## Task Group C — `load_from_code`

AST-parse `create_schema.py` without executing it. Walk the call tree for `TableDef`, `VocabularyTableDef`, `Table.define_association`, `ColumnDef`, `ForeignKeyDef`, `_ensure_terms`. Resolve `MLTable.xxx` / `MLVocab.xxx` references via imported enum lookup.

### Task C1: Skeleton `load_from_code` with enum-ref resolver

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
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
```

- [ ] **Step 2: Run to verify failures**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k resolve_enum_ref
```

Expected: 4 FAIL with ImportError.

- [ ] **Step 3: Implement resolver + exception**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k resolve_enum_ref
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): _resolve_enum_ref for MLTable/MLVocab references in AST

The validator's load_from_code needs to resolve MLTable.execution →
'Execution' when walking the AST. Builds a lookup on first call via
runtime enum import (safe — StrEnum, no catalog required).

SchemaCodeError is raised for unknown enums or members.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C2: AST-parse `ColumnDef` / `ForeignKeyDef` / `TableDef`

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_load_from_code_simple_tabledef(tmp_path):
    """A fixture module with one TableDef call produces a TableModel."""
    from deriva_ml.tools.validate_schema_doc import load_from_code

    fixture = tmp_path / "fixture_schema.py"
    fixture.write_text(
        'from deriva.core.typed import TableDef, ColumnDef, ForeignKeyDef, BuiltinType\n'
        'from deriva_ml.core.definitions import MLTable\n'
        '\n'
        'def create_execution_table(schema):\n'
        '    schema.create_table(TableDef(\n'
        '        name=MLTable.execution,\n'
        '        columns=[\n'
        '            ColumnDef("Workflow", BuiltinType.text),\n'
        '            ColumnDef("Description", BuiltinType.markdown),\n'
        '            ColumnDef("Status", BuiltinType.text),\n'
        '        ],\n'
        '        foreign_keys=[\n'
        '            ForeignKeyDef(\n'
        '                columns=["Workflow"],\n'
        '                referenced_schema="deriva-ml",\n'
        '                referenced_table="Workflow",\n'
        '                referenced_columns=["RID"],\n'
        '            )\n'
        '        ],\n'
        '    ))\n'
    )
    model = load_from_code(fixture)
    assert len(model.tables) == 1
    t = model.tables[0]
    assert t.name == "Execution"
    assert t.kind == "table"
    assert [c.name for c in t.columns] == ["Workflow", "Description", "Status"]
    assert [c.type for c in t.columns] == ["text", "markdown", "text"]
    assert len(t.foreign_keys) == 1
    assert t.foreign_keys[0].columns == ["Workflow"]
    assert t.foreign_keys[0].referenced_table == "Workflow"
```

- [ ] **Step 2: Run to verify failure**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py::test_load_from_code_simple_tabledef -v
```

Expected: FAIL with ImportError on `load_from_code`.

- [ ] **Step 3: Implement AST walker**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
import ast


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


def load_from_code(path: Path) -> SchemaModel:
    """Parse create_schema.py AST into a SchemaModel.

    Walks the tree looking for TableDef, VocabularyTableDef,
    Table.define_association, and _ensure_terms calls.

    Args:
        path: Path to create_schema.py or a fixture module.

    Returns:
        SchemaModel with one TableModel per table definition found.

    Raises:
        SchemaCodeError: If a call pattern can't be statically extracted.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    tables: list[TableModel] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callable_name(node)
        if name == "TableDef":
            tables.append(_extract_tabledef(node))
        # Vocabulary / association / _ensure_terms handled in C3 / C4.
    return SchemaModel(tables=tables)
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py::test_load_from_code_simple_tabledef -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): load_from_code AST parser — TableDef + ColumnDef + ForeignKeyDef

Walks the AST of create_schema.py-like modules, extracts TableDef
calls and their column/FK lists. Resolves MLTable.xxx enum references
via _resolve_enum_ref.

Vocabulary/association/_ensure_terms handling comes in C3/C4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C3: AST-parse `VocabularyTableDef` + `_ensure_terms`

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_load_from_code_vocabulary_with_terms(tmp_path):
    """A VocabularyTableDef plus _ensure_terms yields a vocabulary TableModel."""
    from deriva_ml.tools.validate_schema_doc import load_from_code

    fixture = tmp_path / "fixture_vocab.py"
    fixture.write_text(
        'from deriva.core.typed import VocabularyTableDef\n'
        'from deriva_ml.core.definitions import MLVocab\n'
        '\n'
        'def create():\n'
        '    schema.create_table(\n'
        '        VocabularyTableDef(name=MLVocab.workflow_type, curie_template="x:{RID}")\n'
        '    )\n'
        '\n'
        'def seed():\n'
        '    _ensure_terms(MLVocab.workflow_type, [\n'
        '        {"Name": "Training", "Description": "Train a model"},\n'
        '        {"Name": "Testing", "Description": "Evaluate a model"},\n'
        '    ])\n'
    )
    model = load_from_code(fixture)
    assert len(model.tables) == 1
    t = model.tables[0]
    assert t.name == "Workflow_Type"
    assert t.kind == "vocabulary"
    assert [term.name for term in t.terms] == ["Training", "Testing"]
```

- [ ] **Step 2: Run to verify failure**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py::test_load_from_code_vocabulary_with_terms -v
```

Expected: FAIL — VocabularyTableDef not extracted yet.

- [ ] **Step 3: Extend `load_from_code` to handle vocabulary + terms**

In `src/deriva_ml/tools/validate_schema_doc.py`, add helpers and extend `load_from_code`:

```python
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
```

Rewrite `load_from_code` to include vocab + ensure_terms handling:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS (C1-C3 tests plus earlier A/B tests).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): load_from_code handles VocabularyTableDef + _ensure_terms

Extracts vocabulary tables and their seeded terms. Terms are collected
across the entire module and then grafted onto the matching vocab
TableModel by name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C4: AST-parse `Table.define_association`

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_load_from_code_association_table(tmp_path):
    """Table.define_association produces a TableModel with kind='association'."""
    from deriva_ml.tools.validate_schema_doc import load_from_code

    fixture = tmp_path / "fixture_assoc.py"
    fixture.write_text(
        'from deriva.core.ermrest_model import Table\n'
        'from deriva.core.typed import ColumnDef, BuiltinType\n'
        '\n'
        'def create():\n'
        '    schema.create_table(\n'
        '        Table.define_association(\n'
        '            associates=[\n'
        '                ("Execution", execution_table),\n'
        '                ("Nested_Execution", execution_table),\n'
        '            ],\n'
        '            metadata=[\n'
        '                ColumnDef(name="Sequence", type="int4", nullok=True),\n'
        '            ],\n'
        '        )\n'
        '    )\n'
    )
    model = load_from_code(fixture)
    assert len(model.tables) == 1
    t = model.tables[0]
    assert t.kind == "association"
    # Name isn't given directly; derived from the associates pair.
    # We expect "Execution_Nested_Execution" as the default-constructed name.
    assert t.name == "Execution_Nested_Execution"
    assert [a.table for a in t.associates] == ["Execution", "Nested_Execution"]
    assert [m.name for m in t.metadata] == ["Sequence"]
```

**Note**: `Table.define_association` is actually a factory method from deriva-py that returns a `TableDef` with a derived name. We can't know the derived name without executing code. Two options:

(a) Have the test fixture and the doc agree that association-table names are derived as `"{assoc1}_{assoc2}"` (the common convention used by define_association). Document this as a convention.

(b) Require an explicit `name=` kwarg on the `Table.define_association(...)` call in the doc comparison context. Since the underlying code doesn't pass `name=`, the validator derives the name from `associates`.

Going with (a) — simpler and matches deriva-py's actual behavior. Most cases pass an explicit pair like `("Dataset", dataset_table), ("Dataset_Type", dataset_type)` and the library builds `"Dataset_Dataset_Type"` from them.

- [ ] **Step 2: Run to verify failure**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py::test_load_from_code_association_table -v
```

Expected: FAIL — association not extracted yet.

- [ ] **Step 3: Extend `load_from_code` for associations**

In `src/deriva_ml/tools/validate_schema_doc.py`:

```python
def _extract_association(node: ast.Call) -> TableModel:
    """Extract Table.define_association(associates=[...], metadata=[...])."""
    associates: list[AssociationEndpointModel] = []
    metadata: list[ColumnModel] = []
    for kw in node.keywords:
        if kw.arg == "associates":
            if not isinstance(kw.value, ast.List):
                raise SchemaCodeError(
                    f"define_association associates must be a list, got "
                    f"{ast.dump(kw.value)}"
                )
            for elt in kw.value.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) < 1:
                    raise SchemaCodeError(
                        f"associates entry must be a tuple (name, table), got "
                        f"{ast.dump(elt)}"
                    )
                table_name = _extract_ast_name_or_enum(elt.elts[0])
                associates.append(AssociationEndpointModel(table=table_name))
        elif kw.arg == "metadata":
            if isinstance(kw.value, ast.List):
                for m_elt in kw.value.elts:
                    if isinstance(m_elt, ast.Call) and _callable_name(m_elt) == "ColumnDef":
                        metadata.append(_extract_column(m_elt))
    # Association name: derived from associates — '{first}_{second}'.
    derived_name = "_".join(a.table for a in associates)
    return TableModel(
        name=derived_name,
        kind="association",
        associates=associates,
        metadata=metadata,
    )
```

Add a branch in `load_from_code`'s walker:

```python
elif name == "Table.define_association":
    tables.append(_extract_association(node))
```

Place this branch alongside the TableDef / VocabularyTableDef branches.

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS (C1-C4 plus earlier).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): load_from_code handles Table.define_association

Extracts association-table definitions. The derived name follows
deriva-py's convention: '{first_associate}_{second_associate}'.
Metadata columns are extracted as ColumnModels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group D — `diff_schemas` comparator

Compare two `SchemaModel` instances; emit a list of typed mismatches. No I/O.

### Task D1: `Mismatch` dataclass + empty-diff case

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_mismatch_dataclass_fields():
    from deriva_ml.tools.validate_schema_doc import Mismatch, MismatchKind
    m = Mismatch(
        kind=MismatchKind.MISSING_TABLE,
        table="Execution_Status",
        detail="declared in doc but not in code",
    )
    assert m.kind == MismatchKind.MISSING_TABLE
    assert m.table == "Execution_Status"


def test_diff_identical_schemas_empty():
    from deriva_ml.tools.validate_schema_doc import (
        ColumnModel,
        SchemaModel,
        TableModel,
        diff_schemas,
    )
    s1 = SchemaModel(tables=[
        TableModel(
            name="Dataset",
            kind="table",
            columns=[ColumnModel(name="Name", type="text")],
        ),
    ])
    s2 = SchemaModel(tables=[
        TableModel(
            name="Dataset",
            kind="table",
            columns=[ColumnModel(name="Name", type="text")],
        ),
    ])
    assert diff_schemas(expected=s1, actual=s2) == []
```

- [ ] **Step 2: Run to verify failures**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k "Mismatch or identical"
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement Mismatch + empty diff**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
    # D2-D4 populate this. Skeleton only for D1.
    pass
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v -k "Mismatch or identical"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): diff_schemas skeleton — Mismatch dataclass + missing/extra table

Empty-diff case works: two identical SchemaModels return []. Missing
tables (doc-only) and extra tables (code-only) emit Mismatches.
Per-table column/FK/term comparison comes in D2-D4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D2: Compare columns (name + type)

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_diff_column_missing_in_actual():
    from deriva_ml.tools.validate_schema_doc import (
        ColumnModel, MismatchKind, SchemaModel, TableModel, diff_schemas,
    )
    expected = SchemaModel(tables=[TableModel(
        name="Dataset", kind="table",
        columns=[ColumnModel(name="Name", type="text"), ColumnModel(name="Extra", type="text")],
    )])
    actual = SchemaModel(tables=[TableModel(
        name="Dataset", kind="table",
        columns=[ColumnModel(name="Name", type="text")],
    )])
    mismatches = diff_schemas(expected=expected, actual=actual)
    assert len(mismatches) == 1
    assert mismatches[0].kind == MismatchKind.COLUMN_MISMATCH
    assert "Extra" in mismatches[0].detail


def test_diff_column_type_mismatch():
    from deriva_ml.tools.validate_schema_doc import (
        ColumnModel, MismatchKind, SchemaModel, TableModel, diff_schemas,
    )
    expected = SchemaModel(tables=[TableModel(
        name="Dataset", kind="table",
        columns=[ColumnModel(name="Name", type="text")],
    )])
    actual = SchemaModel(tables=[TableModel(
        name="Dataset", kind="table",
        columns=[ColumnModel(name="Name", type="markdown")],
    )])
    mismatches = diff_schemas(expected=expected, actual=actual)
    assert any(
        m.kind == MismatchKind.COLUMN_MISMATCH
        and "text" in m.detail and "markdown" in m.detail
        for m in mismatches
    )
```

- [ ] **Step 2: Run to verify failures**

Expected: FAIL — `_compare_tables` is a pass-through currently.

- [ ] **Step 3: Populate column comparison in `_compare_tables`**

Modify `_compare_tables` in `src/deriva_ml/tools/validate_schema_doc.py`:

```python
def _compare_tables(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    """Compare two same-named TableModels; append any differences."""
    _compare_columns(mismatches, expected, actual)
    # D3 adds _compare_fks; D4 adds _compare_terms / _compare_associates.


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
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): diff_schemas — column name + type comparison

_compare_columns yields COLUMN_MISMATCH Mismatches for columns
present in one model but not the other, or present in both with
differing types.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D3: Compare FKs

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_diff_fk_target_mismatch():
    from deriva_ml.tools.validate_schema_doc import (
        ColumnModel, ForeignKeyModel, MismatchKind, SchemaModel, TableModel, diff_schemas,
    )
    expected = SchemaModel(tables=[TableModel(
        name="Execution", kind="table",
        columns=[ColumnModel(name="Workflow", type="text")],
        foreign_keys=[ForeignKeyModel(
            columns=["Workflow"],
            referenced_schema="deriva-ml",
            referenced_table="Workflow",
            referenced_columns=["RID"],
        )],
    )])
    actual = SchemaModel(tables=[TableModel(
        name="Execution", kind="table",
        columns=[ColumnModel(name="Workflow", type="text")],
        foreign_keys=[ForeignKeyModel(
            columns=["Workflow"],
            referenced_schema="deriva-ml",
            referenced_table="SomeOtherTable",
            referenced_columns=["RID"],
        )],
    )])
    mismatches = diff_schemas(expected=expected, actual=actual)
    assert any(m.kind == MismatchKind.FK_MISMATCH for m in mismatches)
```

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — FK comparison not implemented.

- [ ] **Step 3: Add FK comparison**

In `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
        fk = expected_fks[k]
        mismatches.append(Mismatch(
            kind=MismatchKind.FK_MISMATCH,
            table=expected.name,
            detail=f"FK on {list(k)} in doc but not in code",
        ))
    for k in sorted(actual_fks.keys() - expected_fks.keys()):
        fk = actual_fks[k]
        mismatches.append(Mismatch(
            kind=MismatchKind.FK_MISMATCH,
            table=expected.name,
            detail=f"FK on {list(k)} in code but not in doc",
        ))
    for k in sorted(expected_fks.keys() & actual_fks.keys()):
        exp_fk = expected_fks[k]
        act_fk = actual_fks[k]
        if (exp_fk.referenced_schema, exp_fk.referenced_table, tuple(exp_fk.referenced_columns)) != (
            act_fk.referenced_schema, act_fk.referenced_table, tuple(act_fk.referenced_columns)
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
```

Extend `_compare_tables` to call `_compare_fks`:

```python
def _compare_tables(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    _compare_columns(mismatches, expected, actual)
    _compare_fks(mismatches, expected, actual)
    # D4 adds _compare_terms / _compare_associates.
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): diff_schemas — FK comparison by column tuple + target

_compare_fks matches FKs by their source-column tuple. Emits
FK_MISMATCH for presence differences and for shared-key differences
in referenced_schema/table/columns.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D4: Compare vocabulary terms + associations

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_diff_vocab_terms_differ():
    from deriva_ml.tools.validate_schema_doc import (
        MismatchKind, SchemaModel, TableModel, VocabularyTermModel, diff_schemas,
    )
    expected = SchemaModel(tables=[TableModel(
        name="Asset_Type",
        kind="vocabulary",
        terms=[
            VocabularyTermModel(name="Execution_Config"),
            VocabularyTermModel(name="Extra_DocOnly_Term"),
        ],
    )])
    actual = SchemaModel(tables=[TableModel(
        name="Asset_Type",
        kind="vocabulary",
        terms=[
            VocabularyTermModel(name="Execution_Config"),
            VocabularyTermModel(name="Extra_CodeOnly_Term"),
        ],
    )])
    mismatches = diff_schemas(expected=expected, actual=actual)
    assert any(
        m.kind == MismatchKind.VOCAB_TERMS_MISMATCH
        and "Extra_DocOnly_Term" in m.detail
        and "Extra_CodeOnly_Term" in m.detail
        for m in mismatches
    )


def test_diff_association_endpoints_differ():
    from deriva_ml.tools.validate_schema_doc import (
        AssociationEndpointModel, MismatchKind, SchemaModel, TableModel, diff_schemas,
    )
    expected = SchemaModel(tables=[TableModel(
        name="Dataset_Execution",
        kind="association",
        associates=[
            AssociationEndpointModel(table="Dataset"),
            AssociationEndpointModel(table="Execution"),
        ],
    )])
    actual = SchemaModel(tables=[TableModel(
        name="Dataset_Execution",
        kind="association",
        associates=[
            AssociationEndpointModel(table="Dataset"),
            AssociationEndpointModel(table="Workflow"),  # differs
        ],
    )])
    mismatches = diff_schemas(expected=expected, actual=actual)
    assert any(m.kind == MismatchKind.ASSOCIATION_MISMATCH for m in mismatches)
```

- [ ] **Step 2: Run to verify failures**

Expected: FAIL.

- [ ] **Step 3: Add vocab + associates comparison**

In `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
```

Extend `_compare_tables`:

```python
def _compare_tables(
    mismatches: list[Mismatch],
    expected: TableModel,
    actual: TableModel,
) -> None:
    _compare_columns(mismatches, expected, actual)
    _compare_fks(mismatches, expected, actual)
    if expected.kind == "vocabulary" or actual.kind == "vocabulary":
        _compare_terms(mismatches, expected, actual)
    if expected.kind == "association" or actual.kind == "association":
        _compare_associates(mismatches, expected, actual)
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): diff_schemas — vocabulary terms + association endpoints

_compare_terms emits VOCAB_TERMS_MISMATCH with doc-only and code-only
term-name sets. _compare_associates compares associate-table-name
sequences; emits ASSOCIATION_MISMATCH on any divergence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group E — CLI entry point + output format

### Task E1: CLI `main(argv)` function

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_cli_match_returns_0(tmp_path, capsys):
    """Identical schemas: exit 0, success message."""
    from deriva_ml.tools.validate_schema_doc import main

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: X\n"
        "kind: table\n"
        "columns: []\n"
        "foreign_keys: []\n"
        "```\n"
    )
    code = tmp_path / "code.py"
    code.write_text(
        'from deriva.core.typed import TableDef\n'
        'schema.create_table(TableDef(name="X", columns=[], foreign_keys=[]))\n'
    )

    exit_code = main(["--doc", str(doc), "--code", str(code)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "agree" in captured.out


def test_cli_mismatch_returns_1(tmp_path, capsys):
    """Divergent schemas: exit 1, mismatch lines."""
    from deriva_ml.tools.validate_schema_doc import main

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: X\n"
        "kind: table\n"
        "columns:\n"
        "  - name: A\n"
        "    type: text\n"
        "foreign_keys: []\n"
        "```\n"
    )
    code = tmp_path / "code.py"
    code.write_text(
        'from deriva.core.typed import TableDef, ColumnDef, BuiltinType\n'
        'schema.create_table(TableDef(\n'
        '    name="X",\n'
        '    columns=[ColumnDef("B", BuiltinType.text)],\n'
        '    foreign_keys=[],\n'
        '))\n'
    )

    exit_code = main(["--doc", str(doc), "--code", str(code)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "COLUMN MISMATCH" in captured.out or "column" in captured.out.lower()


def test_cli_parse_error_returns_2(tmp_path, capsys):
    """Malformed doc: exit 2."""
    from deriva_ml.tools.validate_schema_doc import main

    doc = tmp_path / "schema.md"
    doc.write_text(
        "```yaml\n"
        "table: X\n"
        "kind: invalid-kind\n"
        "```\n"
    )
    code = tmp_path / "code.py"
    code.write_text("")

    exit_code = main(["--doc", str(doc), "--code", str(code)])
    assert exit_code == 2
```

- [ ] **Step 2: Run to verify failures**

Expected: FAIL — `main` not defined.

- [ ] **Step 3: Implement `main` + formatted output**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
import argparse
import sys
from collections import defaultdict


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


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): main() CLI — exit codes 0/1/2, formatted mismatch output

main(argv) parses --doc and --code args (default to production paths),
runs load_from_doc + load_from_code + diff_schemas, prints either the
'agree' single-liner or the bucketed mismatch report. Exit codes
0 (match), 1 (mismatch), 2 (parse error).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E2: Register `deriva-ml-validate-schema` entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current scripts section**

```bash
grep -B1 -A15 "\[project\.scripts\]" pyproject.toml
```

- [ ] **Step 2: Add the new entry**

Append to the `[project.scripts]` section (matching existing `deriva-ml-*` naming):

```toml
deriva-ml-validate-schema = "deriva_ml.tools.validate_schema_doc:main"
```

Place it alphabetically with the others (after `deriva-ml-upload`).

- [ ] **Step 3: Verify the entry is usable via `uv run`**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python -m deriva_ml.tools.validate_schema_doc --help
```

Expected: argparse `--help` output listing `--doc` and `--code` flags.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
build(scripts): register deriva-ml-validate-schema entry point

The schema-doc validator is now runnable as a first-class CLI via the
project's [project.scripts] section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E3: `SchemaModel.to_doc_markdown()` helper for bootstrap

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/test_validate_schema_doc.py`

Per spec §8: ship a helper that renders a `SchemaModel` to the doc Markdown format. Used for the F1 bootstrap and as an "emergency regenerate from code" tool.

- [ ] **Step 1: Write failing test**

Append to `tests/tools/test_validate_schema_doc.py`:

```python
def test_to_doc_markdown_roundtrip(tmp_path):
    """A SchemaModel rendered to Markdown can be loaded back losslessly."""
    from deriva_ml.tools.validate_schema_doc import (
        AssociationEndpointModel, ColumnModel, ForeignKeyModel,
        SchemaModel, TableModel, VocabularyTermModel,
        load_from_doc, to_doc_markdown,
    )
    original = SchemaModel(tables=[
        TableModel(
            name="Dataset",
            kind="table",
            columns=[ColumnModel(name="Name", type="text")],
            foreign_keys=[],
        ),
        TableModel(
            name="Workflow_Type",
            kind="vocabulary",
            terms=[VocabularyTermModel(name="Training")],
        ),
        TableModel(
            name="Dataset_Dataset_Type",
            kind="association",
            associates=[
                AssociationEndpointModel(table="Dataset"),
                AssociationEndpointModel(table="Dataset_Type"),
            ],
        ),
    ])
    md = to_doc_markdown(original)
    doc = tmp_path / "schema.md"
    doc.write_text(md)
    roundtripped = load_from_doc(doc)

    assert len(roundtripped.tables) == 3
    assert [t.name for t in roundtripped.tables] == [
        "Dataset", "Workflow_Type", "Dataset_Dataset_Type",
    ]
    assert roundtripped.tables[0].columns[0].name == "Name"
    assert roundtripped.tables[1].terms[0].name == "Training"
    assert roundtripped.tables[2].associates[0].table == "Dataset"
```

- [ ] **Step 2: Run to verify failure**

Expected: FAIL — `to_doc_markdown` not defined.

- [ ] **Step 3: Implement `to_doc_markdown`**

Append to `src/deriva_ml/tools/validate_schema_doc.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/tools/validate_schema_doc.py tests/tools/test_validate_schema_doc.py
git commit -m "$(cat <<'EOF'
feat(tools): to_doc_markdown — render SchemaModel back to Markdown

Used by the F1 bootstrap task and as an emergency regeneration helper.
Roundtrip through load_from_doc is lossless.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group F — Bootstrap the doc

### Task F1: Generate initial `docs/reference/schema.md` from code

**Files:**
- Create: `docs/reference/schema.md`
- Create: `docs/reference/README.md`

- [ ] **Step 1: Generate the initial doc from the real `create_schema.py`**

```bash
mkdir -p docs/reference
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from pathlib import Path
from deriva_ml.tools.validate_schema_doc import load_from_code, to_doc_markdown

model = load_from_code(Path('src/deriva_ml/schema/create_schema.py'))
print(to_doc_markdown(model))
" > docs/reference/schema.md
```

This produces a pure-generated doc with no prose. The validator should run clean against it.

- [ ] **Step 2: Run validator against the generated doc**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-validate-schema
```

Expected: `deriva-ml-validate-schema: schema.md and create_schema.py agree.` Exit 0.

If mismatches appear, they fall into two buckets:

1. **Validator bugs** — a pattern in `create_schema.py` the validator misparses. Fix the validator; re-generate.
2. **Intentional code-side dynamic patterns** — e.g., f-strings in `curie_template` — these are out of scope per spec §2. Document in the doc's top-of-file narrative which code patterns are ignored.

Iterate Step 1 + Step 2 until clean.

- [ ] **Step 3: Hand-edit — add top-of-file prose**

Prepend to `docs/reference/schema.md`:

```markdown
# deriva-ml Schema Reference

This document is the authoritative description of the deriva-ml schema. It is kept in sync with `src/deriva_ml/schema/create_schema.py` by the `deriva-ml-validate-schema` CI check.

## Editing this doc

Schema changes are **doc-first** per Phase-2-Subsystem-0. To change the schema:

1. Edit the relevant `## <Table>` section below (or add a new one).
2. Edit `src/deriva_ml/schema/create_schema.py` to match.
3. Run `uv run deriva-ml-validate-schema` locally to verify agreement.
4. Commit both files together. CI re-runs the validator.

## What is and isn't validated

The validator enforces: **table names, columns (name + type), foreign keys, vocabulary seeded terms, and association-table endpoints.**

The validator does NOT enforce: descriptions (narrative on both sides), ERMrest annotations (dynamic Python in the code), `curie_template` values (dynamic f-strings), indexes, or display configs. These may differ between doc and code without CI failure.

See `docs/superpowers/specs/2026-04-21-schema-doc-source-of-truth-design.md` for the full rationale.

## Table of contents

(Ordered as: core entities, then vocabularies, then association tables.)

---
```

- [ ] **Step 4: Hand-edit — add per-table descriptions**

For each `## <Table>` section, add a short paragraph of description text between the header and the YAML block. Example:

~~~markdown
## Execution

Per-execution lifecycle row. Created once per workflow run; status transitions through the Phase-1 state machine (`src/deriva_ml/execution/state_machine.py`).

```yaml
table: Execution
kind: table
...
```
~~~

Do this for every table. Keep descriptions short (1-3 sentences). Avoid restating the YAML's structural content.

- [ ] **Step 5: Verify validator still passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-validate-schema
```

Expected: still clean (descriptions aren't validated).

- [ ] **Step 6: Write `docs/reference/README.md`**

Create `docs/reference/README.md`:

```markdown
# Reference docs

## schema.md

Authoritative description of the `deriva-ml` schema.

To change the schema:

1. Edit `schema.md` to describe the intended state.
2. Edit `src/deriva_ml/schema/create_schema.py` to match.
3. Run `uv run deriva-ml-validate-schema` locally.
4. Commit both files together.

See `schema.md` for what is and isn't validated.
```

- [ ] **Step 7: Commit**

```bash
git add docs/reference/
git commit -m "$(cat <<'EOF'
docs(reference): bootstrap schema.md from create_schema.py

Initial generation via to_doc_markdown + hand-added prose and
per-table descriptions. Validator runs clean.

Adds docs/reference/README.md with the doc-first workflow instructions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task F2: Doc-structure tests

**Files:**
- Create: `tests/tools/test_schema_doc_structure.py`

- [ ] **Step 1: Write tests**

Create `tests/tools/test_schema_doc_structure.py`:

```python
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
```

- [ ] **Step 2: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_schema_doc_structure.py -v
```

Expected: all PASS. If MLTable members are missing, F1's generated doc was incomplete — investigate (the generator may have skipped an AST pattern). Fix and regenerate.

- [ ] **Step 3: Write the integration test for repo-scale validation**

Create `tests/tools/test_validate_schema_doc_integration.py`:

```python
"""Integration tests — run the validator on the real repo files.

If this test fails, docs/reference/schema.md has drifted from
src/deriva_ml/schema/create_schema.py. Fix one or both.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_validator_runs_clean_on_current_repo():
    """The authoritative doc and code agree. This IS the CI gate."""
    from deriva_ml.tools.validate_schema_doc import (
        diff_schemas, load_from_code, load_from_doc,
    )

    doc_path = REPO_ROOT / "docs" / "reference" / "schema.md"
    code_path = REPO_ROOT / "src" / "deriva_ml" / "schema" / "create_schema.py"

    expected = load_from_doc(doc_path)
    actual = load_from_code(code_path)
    mismatches = diff_schemas(expected=expected, actual=actual)

    assert mismatches == [], (
        "schema.md and create_schema.py disagree:\n"
        + "\n".join(f"  - {m.kind.value}: {m.detail}" for m in mismatches)
    )


def test_cli_exits_zero_on_current_repo():
    """Invoking the CLI against the real paths returns exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.tools.validate_schema_doc"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "agree" in result.stdout
```

- [ ] **Step 4: Run all tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/ -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/test_schema_doc_structure.py tests/tools/test_validate_schema_doc_integration.py
git commit -m "$(cat <<'EOF'
test(tools): schema doc structure + integration tests

test_schema_doc_structure.py asserts every MLTable / MLVocab enum
member appears in the doc — catches 'added a table to the enum but
forgot the doc.'

test_validate_schema_doc_integration.py runs the full doc-vs-code
diff on the real repo files. This IS the CI gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group G — CI workflow + CHANGELOG + final review

### Task G1: Add GitHub Actions workflow

**Files:**
- Create: `.github/workflows/validate-schema.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/validate-schema.yml`:

```yaml
name: Validate deriva-ml schema doc

on:
  pull_request:
  push:
    branches: [main, master]

jobs:
  validate-schema:
    name: schema.md ↔ create_schema.py
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install uv
        uses: astral-sh/setup-uv@v6
      - name: Sync deps
        run: uv sync
      - name: Run validator
        run: uv run deriva-ml-validate-schema
```

- [ ] **Step 2: Verify workflow locally with act (if available) or by lint-only**

```bash
# If actionlint is available:
actionlint .github/workflows/validate-schema.yml 2>&1 || echo "actionlint not installed; skipping"
# Or just visually review.
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/validate-schema.yml
git commit -m "$(cat <<'EOF'
ci: add validate-schema workflow for every PR

Runs uv run deriva-ml-validate-schema on each pull request and on
pushes to main. Fails the build if docs/reference/schema.md and
src/deriva_ml/schema/create_schema.py disagree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task G2: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Insert new section**

Near the top of `CHANGELOG.md`, above any existing Phase-2 entries:

```markdown
## Unreleased — Phase 2 Subsystem 0: Schema-doc source of truth

### New

- **`docs/reference/schema.md`** — authoritative description of the `deriva-ml` schema (tables, columns, FKs, vocabulary seeded terms). Edit this file **first** when changing the schema, then update `src/deriva_ml/schema/create_schema.py` to match.
- **`deriva-ml-validate-schema`** CLI (`src/deriva_ml/tools/validate_schema_doc.py`) — asserts the doc and code agree on structure and seeded terms. Exit 0 on match, 1 on mismatch, 2 on parse error.
- **CI workflow** `.github/workflows/validate-schema.yml` — runs the validator on every PR and push to main.
- **`docs/reference/README.md`** — developer workflow instructions for doc-first schema changes.

### Changed

- Schema changes now **require editing two files together**: `docs/reference/schema.md` and `src/deriva_ml/schema/create_schema.py`. CI enforces they agree.

### Not yet supported (filed follow-ups)

- Direction 2: `deriva-ml-validate-schema --against=catalog` for live-catalog drift detection.
- Description validation (table/column prose on doc vs `comment=` on code).
- Annotation validation.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): Phase 2 Subsystem 0 — schema-doc source of truth

Announces the new doc-first workflow for schema changes, the
deriva-ml-validate-schema CLI, and the CI workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task G3: Final review + regression

**Files:** none (verification only)

- [ ] **Step 1: Full tool test run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/ -v
```

Expected: all PASS.

- [ ] **Step 2: Verify the CLI runs clean**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-validate-schema
echo "exit=$?"
```

Expected: "agree" message, `exit=0`.

- [ ] **Step 3: Regression across the rest of the suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/ tests/local_db/ -q 2>&1 | tail -5
```

Expected: no Phase-2-Subsystem-0-introduced regressions. (We haven't touched library code, only tools + docs.)

- [ ] **Step 4: Ruff on new files**

```bash
uv run ruff check src/deriva_ml/tools/ tests/tools/
```

Expected: clean.

- [ ] **Step 5: Dispatch a `superpowers:code-reviewer` subagent for final review**

Per `superpowers:subagent-driven-development`:

```
Subagent: superpowers:code-reviewer
Task: review the Phase 2 Subsystem 0 diff (commits A1 through G2) against
spec docs/superpowers/specs/2026-04-21-schema-doc-source-of-truth-design.md.

Check: spec coverage (§§1-11), Path-1 architecture correctness, validator
behavior (structure-only per Q6/Q8), CI workflow shape, doc completeness.

Report blockers, important issues, and nits. Implementer fixes blockers
and important issues before merge.
```

- [ ] **Step 6: Address reviewer findings**

Follow subagent-driven-development pattern: re-dispatch implementer for each fix, re-run reviewer until approved.

- [ ] **Step 7: Finish branch**

Once approved, invoke `superpowers:finishing-a-development-branch` to PR and merge.

---

*(End of Task Group G — Phase 2 Subsystem 0 complete.)*

---

## Deferred from spec

- **Spec §6.3 test #18** (`test_schema_doc_table_order` — warn-not-fail on section-ordering violations): not implemented in this plan. Pytest's warn-not-fail semantics are fiddly, and the §4.1 ordering (core → vocab → association) is a readability convention that doesn't need CI enforcement. If the ordering drifts in practice, add this test as a follow-up.

## Post-Subsystem-0

After Subsystem 0 merges to main, resume **Subsystem 1 (status-enum reconciliation)**. The existing spec (`2026-04-21-status-enum-reconciliation-design.md`) is already written; the WIP plan (`2026-04-21-status-enum-reconciliation.md`) needs the following edits to reflect the Path-1 workflow:

- Each task that modifies `src/deriva_ml/schema/create_schema.py` must also update `docs/reference/schema.md` in the same commit.
- Subsystem 1's Q6 (vocab table shape — Execution_Status as a direct FK) is resolvable once Subsystem 0 merges: the schema doc's yaml shape for FK vs. association-table patterns is established, so the decision about Execution_Status becomes "which pattern does the doc describe?"

Then: Subsystem 3 (upload engine), Subsystem 4 (hygiene), Subsystem 2 (feature-consistency, the real Phase 2).

# Association Index SQL Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-30-association-index-sql-generator-design.md`

**Goal:** Add a `deriva-ml-generate-association-indexes` CLI that scans a catalog for pure binary association tables and emits a `.sql` file with composite-pair indexes for both join orderings on the live table and the corresponding `_ermrest_history` history table.

**Architecture:** Read-only against the catalog over the ermrest HTTP API. Two modules: `_index_sql.py` is a pure SQL builder (unit-testable without a catalog); `generate_association_indexes.py` is the CLI/orchestration that pulls the model, builds RID lookups from the raw `/schema` doc, walks association tables, and emits the file. No Postgres connection — admin runs the file via pgAdmin or `psql -f`.

**Tech Stack:**
- Python ≥3.12, deriva-py (`ErmrestCatalog`, `Model`, `Table.is_association`).
- argparse for CLI.
- pytest + ruff for tests/lint.
- No new runtime dependencies.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `src/deriva_ml/tools/_index_sql.py` | Pure SQL string builders. No I/O, no deriva-py imports. Single-quote/double-quote helpers, `truncate_index_name`, `live_index_sql`, `history_index_sql`. |
| `src/deriva_ml/tools/generate_association_indexes.py` | CLI entry, catalog connection, RID lookup builder, association walk, file writer. The orchestration layer. |
| `tests/tools/test_index_sql.py` | Unit tests for `_index_sql` — no catalog needed, fast. |
| `tests/tools/test_generate_association_indexes.py` | Unit tests for the orchestration module's pure functions (RID lookup parsing, association walk against a fake `Model`). No catalog needed. |
| `tests/integration/test_generate_association_indexes_integration.py` | Integration test against `DERIVA_HOST` test catalog. Gated. |

**Modified files:**

| Path | Lines | Change |
|---|---|---|
| `pyproject.toml` | ~`[project.scripts]` block (around lines 45–60) | Add `deriva-ml-generate-association-indexes = "deriva_ml.tools.generate_association_indexes:main"`. |

**Module boundaries:**
- `_index_sql.py` is pure — no `deriva.core` imports. The unit tests for it have zero environmental requirements.
- `generate_association_indexes.py` imports `deriva.core` and calls `_index_sql` for output formatting. Its unit tests construct fake `Model`-shaped objects rather than connecting to a live catalog.
- Integration tests live under `tests/integration/` to match existing convention; they're skipped automatically when `DERIVA_HOST` is not set.

---

## Task 1: SQL identifier quoting helpers

**Files:**
- Create: `src/deriva_ml/tools/_index_sql.py`
- Test: `tests/tools/test_index_sql.py`

These two helpers underpin every SQL string the tool emits. Building them first means every later task can compose them.

- [ ] **Step 1.1: Write failing tests for `quote_ident` and `quote_literal`**

Create `tests/tools/test_index_sql.py`:

```python
"""Unit tests for _index_sql pure SQL builders."""

from __future__ import annotations


def test_quote_ident_simple_name():
    from deriva_ml.tools._index_sql import quote_ident
    assert quote_ident("Subject") == '"Subject"'


def test_quote_ident_with_hyphen():
    from deriva_ml.tools._index_sql import quote_ident
    assert quote_ident("eye-ai") == '"eye-ai"'


def test_quote_ident_doubles_embedded_quote():
    from deriva_ml.tools._index_sql import quote_ident
    assert quote_ident('weird"name') == '"weird""name"'


def test_quote_ident_preserves_case():
    from deriva_ml.tools._index_sql import quote_ident
    assert quote_ident("MixedCase") == '"MixedCase"'


def test_quote_literal_simple_text():
    from deriva_ml.tools._index_sql import quote_literal
    assert quote_literal("1-ABCD") == "'1-ABCD'"


def test_quote_literal_doubles_embedded_apostrophe():
    from deriva_ml.tools._index_sql import quote_literal
    assert quote_literal("o'reilly") == "'o''reilly'"
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deriva_ml.tools._index_sql'`.

- [ ] **Step 1.3: Implement the helpers**

Create `src/deriva_ml/tools/_index_sql.py`:

```python
"""Pure SQL string builders for association-index generation.

This module produces strings only. It does not import deriva-py, does
not perform I/O, and does not execute SQL. Every function is unit-
testable in isolation.

The companion module ``generate_association_indexes`` orchestrates
catalog access and writes the assembled output file.
"""

from __future__ import annotations


def quote_ident(name: str) -> str:
    """Quote a Postgres identifier (schema, table, column, index name).

    Wraps ``name`` in double quotes and doubles any embedded double
    quotes, per the standard SQL identifier-quoting rule.

    Args:
        name: The identifier to quote.

    Returns:
        The double-quoted identifier, safe for use in DDL.

    Example:
        >>> quote_ident("Subject")
        '"Subject"'
        >>> quote_ident("eye-ai")
        '"eye-ai"'
        >>> quote_ident('weird"name')
        '"weird""name"'
    """
    return '"' + name.replace('"', '""') + '"'


def quote_literal(text: str) -> str:
    """Quote a Postgres string literal.

    Wraps ``text`` in single quotes and doubles any embedded single
    quotes. Used inside ``rowdata->>'<value>'`` expressions where
    ``<value>`` is a column RID.

    Args:
        text: The literal value to quote.

    Returns:
        The single-quoted literal, safe for SQL embedding.

    Example:
        >>> quote_literal("1-ABCD")
        "'1-ABCD'"
        >>> quote_literal("o'reilly")
        "'o''reilly'"
    """
    return "'" + text.replace("'", "''") + "'"
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 1.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/_index_sql.py tests/tools/test_index_sql.py
git commit -m "feat(tools): add SQL identifier and literal quoting helpers

Foundation for the association-index SQL generator. Pure string
builders; no deriva-py or I/O dependency.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Index name truncation with hash suffix

**Files:**
- Modify: `src/deriva_ml/tools/_index_sql.py`
- Modify: `tests/tools/test_index_sql.py`

Postgres caps identifiers at 63 bytes. Truncation must be deterministic and collision-resistant when two long table names share a long prefix.

- [ ] **Step 2.1: Write failing tests**

Append to `tests/tools/test_index_sql.py`:

```python
def test_truncate_index_name_short_unchanged():
    from deriva_ml.tools._index_sql import truncate_index_name
    assert truncate_index_name("short_idx") == "short_idx"


def test_truncate_index_name_at_limit_unchanged():
    from deriva_ml.tools._index_sql import truncate_index_name
    name = "a" * 63
    assert truncate_index_name(name) == name


def test_truncate_index_name_over_limit_is_capped():
    from deriva_ml.tools._index_sql import truncate_index_name
    name = "a" * 100
    out = truncate_index_name(name)
    assert len(out) == 63


def test_truncate_index_name_distinct_inputs_produce_distinct_outputs():
    from deriva_ml.tools._index_sql import truncate_index_name
    base = "x" * 80
    a = truncate_index_name(base + "_alpha_assoc_fwd_idx")
    b = truncate_index_name(base + "_beta_assoc_fwd_idx")
    assert a != b
    assert len(a) == 63 and len(b) == 63


def test_truncate_index_name_deterministic():
    from deriva_ml.tools._index_sql import truncate_index_name
    name = "x" * 100
    assert truncate_index_name(name) == truncate_index_name(name)
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: FAIL on the new tests with `ImportError` for `truncate_index_name`.

- [ ] **Step 2.3: Implement `truncate_index_name`**

Add to `src/deriva_ml/tools/_index_sql.py` (below `quote_literal`):

```python
import hashlib

POSTGRES_IDENT_MAX = 63
_HASH_SUFFIX_LEN = 8


def truncate_index_name(name: str) -> str:
    """Cap an index name at Postgres's 63-byte identifier limit.

    Names that fit are returned unchanged. Longer names are truncated
    and a deterministic 8-character md5 suffix is appended so two
    distinct long names always produce distinct truncated names.

    Args:
        name: The intended index name.

    Returns:
        ``name`` if ``len(name) <= 63``; otherwise ``name[:54] +
        "_" + md5(name)[:8]``, length 63.

    Example:
        >>> truncate_index_name("short_idx")
        'short_idx'
        >>> len(truncate_index_name("a" * 100))
        63
    """
    if len(name) <= POSTGRES_IDENT_MAX:
        return name
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_LEN]
    head_len = POSTGRES_IDENT_MAX - _HASH_SUFFIX_LEN - 1
    return name[:head_len] + "_" + digest
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 2.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/_index_sql.py tests/tools/test_index_sql.py
git commit -m "feat(tools): add deterministic index-name truncation

Postgres caps identifiers at 63 bytes. Long names are truncated and
suffixed with the first 8 hex chars of an md5 hash so distinct inputs
always produce distinct outputs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Live-table CREATE INDEX builder

**Files:**
- Modify: `src/deriva_ml/tools/_index_sql.py`
- Modify: `tests/tools/test_index_sql.py`

`live_index_sql` emits one `CREATE INDEX CONCURRENTLY IF NOT EXISTS` for the user-facing table.

- [ ] **Step 3.1: Write failing tests**

Append to `tests/tools/test_index_sql.py`:

```python
def test_live_index_sql_single_column_pair():
    from deriva_ml.tools._index_sql import live_index_sql
    sql = live_index_sql(
        schema="eye-ai",
        table="Subject_Image",
        index_name="Subject_Image_assoc_fwd_idx",
        columns=["Subject_RID", "Image_RID"],
    )
    assert sql == (
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS '
        '"Subject_Image_assoc_fwd_idx"\n'
        '  ON "eye-ai"."Subject_Image" ("Subject_RID", "Image_RID");'
    )


def test_live_index_sql_composite_fk_columns():
    from deriva_ml.tools._index_sql import live_index_sql
    sql = live_index_sql(
        schema="myschema",
        table="A",
        index_name="A_assoc_fwd_idx",
        columns=["fk1_a", "fk1_b", "fk2_x", "fk2_y"],
    )
    assert (
        '"A_assoc_fwd_idx"\n  ON "myschema"."A" '
        '("fk1_a", "fk1_b", "fk2_x", "fk2_y");'
    ) in sql


def test_live_index_sql_quotes_funky_names():
    from deriva_ml.tools._index_sql import live_index_sql
    sql = live_index_sql(
        schema='weird"schema',
        table="Has Spaces",
        index_name="i",
        columns=["col"],
    )
    assert '"weird""schema"."Has Spaces"' in sql
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: FAIL on new tests with `ImportError` for `live_index_sql`.

- [ ] **Step 3.3: Implement `live_index_sql`**

Add to `src/deriva_ml/tools/_index_sql.py`:

```python
def live_index_sql(
    schema: str,
    table: str,
    index_name: str,
    columns: list[str],
) -> str:
    """Build a CREATE INDEX CONCURRENTLY statement for a live table.

    Uses ``IF NOT EXISTS`` so re-running the generated file is a safe
    no-op for indexes that already exist. Uses ``CONCURRENTLY`` so the
    DDL does not acquire an ``ACCESS EXCLUSIVE`` lock against the live
    table — required for catalogs handling user traffic.

    Args:
        schema: The Postgres schema name (typically the ERMrest
            schema name, e.g. ``"eye-ai"``).
        table: The table name within the schema.
        index_name: The desired index name. Caller is responsible for
            running it through ``truncate_index_name`` first.
        columns: Index column list in declaration order. For an
            association table with two simple FKs this is two names;
            for composite FKs it expands to all columns of FK1
            followed by all columns of FK2.

    Returns:
        A single SQL statement (no trailing newline). Identifiers are
        double-quoted.

    Example:
        >>> live_index_sql("s", "t", "t_idx", ["a", "b"])
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS "t_idx"\\n  ON "s"."t" ("a", "b");'
    """
    cols = ", ".join(quote_ident(c) for c in columns)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(index_name)}\n"
        f"  ON {quote_ident(schema)}.{quote_ident(table)} ({cols});"
    )
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 3.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/_index_sql.py tests/tools/test_index_sql.py
git commit -m "feat(tools): add live_index_sql composite-pair builder

Emits CREATE INDEX CONCURRENTLY IF NOT EXISTS for the user-facing
association table. CONCURRENTLY avoids ACCESS EXCLUSIVE locks on a
live catalog; IF NOT EXISTS makes re-runs safe.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: History-table expression-index builder

**Files:**
- Modify: `src/deriva_ml/tools/_index_sql.py`
- Modify: `tests/tools/test_index_sql.py`

History indexes are expression indexes on the `_ermrest_history."t<table_RID>"` table. The expression is `(rowdata->>'<column_RID>')` — column RIDs are JSONB keys, not column names. (Confirmed by reading ERMrest's `ermrest_schema.sql`.)

- [ ] **Step 4.1: Write failing tests**

Append to `tests/tools/test_index_sql.py`:

```python
def test_history_index_sql_two_columns():
    from deriva_ml.tools._index_sql import history_index_sql
    sql = history_index_sql(
        table_rid="1-ABCD",
        index_name="Subject_Image_hist_assoc_fwd_idx",
        column_rids=["1-EEEE", "1-FFFF"],
    )
    assert sql == (
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS '
        '"Subject_Image_hist_assoc_fwd_idx"\n'
        '  ON _ermrest_history."t1-ABCD"\n'
        "     ((rowdata->>'1-EEEE'), (rowdata->>'1-FFFF'));"
    )


def test_history_index_sql_composite_columns():
    from deriva_ml.tools._index_sql import history_index_sql
    sql = history_index_sql(
        table_rid="2-ZZZZ",
        index_name="i",
        column_rids=["a", "b", "c", "d"],
    )
    assert (
        "((rowdata->>'a'), (rowdata->>'b'), "
        "(rowdata->>'c'), (rowdata->>'d'))"
    ) in sql
    assert '"t2-ZZZZ"' in sql


def test_history_index_sql_escapes_apostrophe_in_rid():
    # Defensive — RIDs are urlb32-encoded and never contain apostrophes,
    # but quote_literal must still handle them correctly.
    from deriva_ml.tools._index_sql import history_index_sql
    sql = history_index_sql(
        table_rid="1-A",
        index_name="i",
        column_rids=["o'odd"],
    )
    assert "(rowdata->>'o''odd')" in sql
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: FAIL on new tests with `ImportError` for `history_index_sql`.

- [ ] **Step 4.3: Implement `history_index_sql`**

Add to `src/deriva_ml/tools/_index_sql.py`:

```python
def history_index_sql(
    table_rid: str,
    index_name: str,
    column_rids: list[str],
) -> str:
    """Build a CREATE INDEX CONCURRENTLY statement for the history table.

    The history table for a user table with RID ``X`` lives at
    ``_ermrest_history."tX"`` and stores each row version as JSONB in
    its ``rowdata`` column. The JSONB keys are column RIDs (text), not
    column names — see ERMrest's ``ermrest_schema.sql`` (the
    ``jsonb_object_agg`` step uses ``c."RID"::text`` as the key for
    user-table history). The index expression therefore wraps each
    column with ``(rowdata->>'<column_rid>')``.

    Args:
        table_rid: The owning user-table's RID. The history table is
            named ``t<table_rid>`` in schema ``_ermrest_history``.
        index_name: The desired index name. Caller is responsible for
            ``truncate_index_name``.
        column_rids: Column RIDs in the same order the live-side index
            uses. For composite FKs this expands to all RIDs from FK1
            followed by all RIDs from FK2 (or the reverse, for the
            reverse-direction index).

    Returns:
        A single SQL statement (no trailing newline).

    Example:
        >>> sql = history_index_sql("1-A", "i", ["1-X", "1-Y"])
        >>> "_ermrest_history" in sql and "t1-A" in sql
        True
    """
    exprs = ", ".join(f"(rowdata->>{quote_literal(rid)})" for rid in column_rids)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(index_name)}\n"
        f"  ON _ermrest_history.{quote_ident('t' + table_rid)}\n"
        f"     ({exprs});"
    )
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_index_sql.py -v`
Expected: PASS, 17 tests.

- [ ] **Step 4.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/_index_sql.py tests/tools/test_index_sql.py
git commit -m "feat(tools): add history_index_sql expression-index builder

Emits CREATE INDEX CONCURRENTLY for _ermrest_history.\"t<rid>\" with
(rowdata->>'<col_rid>') expressions. JSONB keys are column RIDs, not
names — matches ERMrest's history-trigger output shape.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: AssociationInfo dataclass and walk

**Files:**
- Create: `src/deriva_ml/tools/generate_association_indexes.py`
- Create: `tests/tools/test_generate_association_indexes.py`

The orchestration module starts here. `AssociationInfo` is the value object passed from "found an association" to "emit SQL for it." The walk function detects associations on a model.

- [ ] **Step 5.1: Write failing tests**

Create `tests/tools/test_generate_association_indexes.py`:

```python
"""Unit tests for the association-index generator orchestration."""

from __future__ import annotations

from types import SimpleNamespace


def _fake_column(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _fake_fk(cols: list[str], target_schema: str, target_table: str) -> SimpleNamespace:
    """Minimal stand-in for a deriva-py ForeignKey."""
    target = SimpleNamespace(name=target_table, schema=SimpleNamespace(name=target_schema))
    return SimpleNamespace(
        foreign_key_columns=[_fake_column(c) for c in cols],
        pk_table=target,
    )


def _fake_table(
    schema_name: str,
    name: str,
    is_assoc_fkeys: list[SimpleNamespace] | None,
) -> SimpleNamespace:
    """Minimal stand-in for a deriva-py Table."""
    fkeys = is_assoc_fkeys
    def is_association(*, return_fkeys=False, **_kw):
        if fkeys is None:
            return False
        return set(fkeys) if return_fkeys else len(fkeys)
    return SimpleNamespace(
        name=name,
        schema=SimpleNamespace(name=schema_name),
        is_association=is_association,
    )


def _fake_schema(name: str, tables: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        tables={t.name: t for t in tables},
    )


def _fake_model(schemas: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(schemas={s.name: s for s in schemas})


def test_walk_skips_system_schemas():
    from deriva_ml.tools.generate_association_indexes import _walk_associations
    fk_a = _fake_fk(["a"], "s", "T1")
    fk_b = _fake_fk(["b"], "s", "T2")
    bad = _fake_table("_ermrest", "would_be_assoc", [fk_a, fk_b])
    model = _fake_model([_fake_schema("_ermrest", [bad])])
    assert list(_walk_associations(model)) == []


def test_walk_skips_non_associations():
    from deriva_ml.tools.generate_association_indexes import _walk_associations
    plain = _fake_table("eye-ai", "Subject", None)
    model = _fake_model([_fake_schema("eye-ai", [plain])])
    assert list(_walk_associations(model)) == []


def test_walk_yields_one_association_with_fk_pair():
    from deriva_ml.tools.generate_association_indexes import _walk_associations, AssociationInfo
    fk1 = _fake_fk(["Subject_RID"], "eye-ai", "Subject")
    fk2 = _fake_fk(["Image_RID"], "eye-ai", "Image")
    assoc = _fake_table("eye-ai", "Subject_Image", [fk1, fk2])
    model = _fake_model([_fake_schema("eye-ai", [assoc])])
    out = list(_walk_associations(model))
    assert len(out) == 1
    info = out[0]
    assert isinstance(info, AssociationInfo)
    assert info.schema_name == "eye-ai"
    assert info.table_name == "Subject_Image"
    # FK ordering is sorted by referenced table name for determinism.
    assert info.fk1_columns == ["Image_RID"]  # "Image" < "Subject"
    assert info.fk1_target == ("eye-ai", "Image")
    assert info.fk2_columns == ["Subject_RID"]
    assert info.fk2_target == ("eye-ai", "Subject")


def test_walk_skips_default_system_schemas():
    from deriva_ml.tools.generate_association_indexes import _walk_associations
    fk1 = _fake_fk(["a"], "s", "T1")
    fk2 = _fake_fk(["b"], "s", "T2")
    for sys_schema in ("_ermrest", "_ermrest_history", "pg_catalog", "public", "www"):
        bad = _fake_table(sys_schema, "x", [fk1, fk2])
        model = _fake_model([_fake_schema(sys_schema, [bad])])
        assert list(_walk_associations(model)) == [], sys_schema
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deriva_ml.tools.generate_association_indexes'`.

- [ ] **Step 5.3: Implement `AssociationInfo` and `_walk_associations`**

Create `src/deriva_ml/tools/generate_association_indexes.py`:

```python
"""CLI: generate CREATE INDEX SQL for catalog association tables.

Scans an ERMrest catalog for pure binary association tables, then
emits a SQL file with composite-pair indexes for both join orderings
on each association's live table and corresponding expression indexes
on its ``_ermrest_history`` history table.

Read-only against the catalog. Does not execute DDL — the generated
file is intended to be reviewed and applied via pgAdmin or
``psql -f`` by the database administrator.

Entry point: ``deriva-ml-generate-association-indexes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

# Schemas we never index (system internals or non-data scratch space).
_SKIP_SCHEMAS = frozenset({"_ermrest", "_ermrest_history", "pg_catalog", "public", "www"})


@dataclass(frozen=True)
class AssociationInfo:
    """One association table found in the catalog.

    FKs are ordered deterministically by referenced ``(schema, table)``
    so that re-running the tool produces byte-identical output.

    Attributes:
        schema_name: Schema containing the association table.
        table_name: The association table's own name.
        fk1_columns: Column names on the association table for FK1
            (the FK referencing the lexicographically-first endpoint).
        fk1_target: ``(schema, table)`` referenced by FK1.
        fk2_columns: Column names on the association table for FK2.
        fk2_target: ``(schema, table)`` referenced by FK2.
    """

    schema_name: str
    table_name: str
    fk1_columns: list[str]
    fk1_target: tuple[str, str]
    fk2_columns: list[str]
    fk2_target: tuple[str, str]


def _walk_associations(model) -> Iterator[AssociationInfo]:
    """Yield one AssociationInfo per pure binary association table.

    Uses ``Table.is_association(return_fkeys=True)`` with all defaults
    (pure, unqualified, non-overlapping, binary) to detect candidates.
    Skips tables in system schemas (``_ermrest``, ``_ermrest_history``,
    ``pg_catalog``, ``public``, ``www``).

    Args:
        model: A deriva-py ``ermrest_model.Model`` (or any object
            exposing ``schemas[name].tables[name]`` with each table
            having ``schema.name``, ``name``, and ``is_association``).

    Yields:
        ``AssociationInfo`` per detected association, FKs sorted by
        referenced ``(schema, table)``.
    """
    for schema_name, schema in model.schemas.items():
        if schema_name in _SKIP_SCHEMAS:
            continue
        for table_name, table in schema.tables.items():
            fkeys = table.is_association(return_fkeys=True)
            if not fkeys:
                continue
            fkey_list = sorted(
                fkeys,
                key=lambda fk: (fk.pk_table.schema.name, fk.pk_table.name),
            )
            if len(fkey_list) != 2:
                # is_association defaults to binary, but be defensive.
                continue
            fk1, fk2 = fkey_list
            yield AssociationInfo(
                schema_name=schema_name,
                table_name=table_name,
                fk1_columns=[c.name for c in fk1.foreign_key_columns],
                fk1_target=(fk1.pk_table.schema.name, fk1.pk_table.name),
                fk2_columns=[c.name for c in fk2.foreign_key_columns],
                fk2_target=(fk2.pk_table.schema.name, fk2.pk_table.name),
            )
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_generate_association_indexes.py
git commit -m "feat(tools): add AssociationInfo and _walk_associations

Identifies pure binary association tables in a model, sorted by
referenced endpoint for deterministic SQL output. Skips ERMrest
system schemas.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: RID lookup builder from raw schema doc

**Files:**
- Modify: `src/deriva_ml/tools/generate_association_indexes.py`
- Modify: `tests/tools/test_generate_association_indexes.py`

`_ermrest.known_columns` is not exposed via the data API (verified — `/attribute/_ermrest:known_columns` returns "Schema _ermrest does not exist"). The raw `/schema` endpoint **does** carry `"RID"` on every table and column entry — that's our source.

- [ ] **Step 6.1: Write failing tests**

Append to `tests/tools/test_generate_association_indexes.py`:

```python
def test_parse_rid_lookup_extracts_table_and_column_rids():
    from deriva_ml.tools.generate_association_indexes import _parse_rid_lookup
    schema_doc = {
        "schemas": {
            "eye-ai": {
                "RID": "0-AAAA",
                "schema_name": "eye-ai",
                "tables": {
                    "Subject_Image": {
                        "RID": "1-ABCD",
                        "schema_name": "eye-ai",
                        "table_name": "Subject_Image",
                        "column_definitions": [
                            {"RID": "1-EEEE", "name": "Subject_RID"},
                            {"RID": "1-FFFF", "name": "Image_RID"},
                            {"RID": "1-GGGG", "name": "RID"},
                        ],
                    }
                },
            }
        }
    }
    table_rid, column_rid = _parse_rid_lookup(schema_doc)
    assert table_rid[("eye-ai", "Subject_Image")] == "1-ABCD"
    assert column_rid[("1-ABCD", "Subject_RID")] == "1-EEEE"
    assert column_rid[("1-ABCD", "Image_RID")] == "1-FFFF"


def test_parse_rid_lookup_handles_missing_rid_field_gracefully():
    # Old ERMrest may omit RID. Missing entries are simply absent from
    # the lookup; downstream code is expected to handle the miss.
    from deriva_ml.tools.generate_association_indexes import _parse_rid_lookup
    schema_doc = {
        "schemas": {
            "s": {
                "tables": {
                    "T": {
                        "schema_name": "s",
                        "table_name": "T",
                        "column_definitions": [{"name": "c"}],
                    }
                }
            }
        }
    }
    table_rid, column_rid = _parse_rid_lookup(schema_doc)
    assert ("s", "T") not in table_rid
    assert column_rid == {}
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: FAIL with `ImportError` for `_parse_rid_lookup`.

- [ ] **Step 6.3: Implement `_parse_rid_lookup`**

Add to `src/deriva_ml/tools/generate_association_indexes.py` (below the `_walk_associations` function):

```python
TableRidMap = dict[tuple[str, str], str]
ColumnRidMap = dict[tuple[str, str], str]


def _parse_rid_lookup(schema_doc: dict) -> tuple[TableRidMap, ColumnRidMap]:
    """Build (table_rid, column_rid) lookups from a raw /schema doc.

    The deriva-py ``Model`` object discards RIDs at parse time; the
    raw schema document at ``/ermrest/catalog/<id>/schema`` retains
    them as ``"RID"`` fields on each table and column entry. We need
    them to compose history-table index expressions
    (``_ermrest_history.t<table_rid>`` and
    ``rowdata->>'<column_rid>'``).

    Args:
        schema_doc: The dict returned by ``catalog.get("/schema").json()``.

    Returns:
        A pair ``(table_rid, column_rid)``:

        - ``table_rid: dict[(schema_name, table_name), str]``
        - ``column_rid: dict[(table_rid, column_name), str]``

        Entries with missing ``RID`` are omitted.

    Example:
        >>> doc = {"schemas": {"s": {"tables": {
        ...     "T": {"RID": "1-A", "column_definitions": [
        ...         {"RID": "1-B", "name": "c"}]}}}}}
        >>> t, c = _parse_rid_lookup(doc)
        >>> t[("s", "T")], c[("1-A", "c")]
        ('1-A', '1-B')
    """
    table_rid: TableRidMap = {}
    column_rid: ColumnRidMap = {}
    for sname, schema in schema_doc.get("schemas", {}).items():
        for tname, table in schema.get("tables", {}).items():
            t_rid = table.get("RID")
            if t_rid is None:
                continue
            table_rid[(sname, tname)] = t_rid
            for col in table.get("column_definitions", []):
                c_rid = col.get("RID")
                c_name = col.get("name")
                if c_rid is None or c_name is None:
                    continue
                column_rid[(t_rid, c_name)] = c_rid
    return table_rid, column_rid
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_generate_association_indexes.py
git commit -m "feat(tools): parse table and column RIDs from raw /schema doc

The _ermrest schema is not exposed via the data API, but the raw
/schema endpoint carries RIDs inline on every table and column entry.
deriva-py's Model strips them out, so we read the raw doc directly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: SQL emission for one association

**Files:**
- Modify: `src/deriva_ml/tools/generate_association_indexes.py`
- Modify: `tests/tools/test_generate_association_indexes.py`

Now we compose `_index_sql` builders + RID lookups + an `AssociationInfo` into the four `CREATE INDEX` statements (and header comment block) for one association.

- [ ] **Step 7.1: Write failing tests**

Append to `tests/tools/test_generate_association_indexes.py`:

```python
def test_render_association_section_emits_four_indexes_and_header():
    from deriva_ml.tools.generate_association_indexes import (
        AssociationInfo,
        _render_association_section,
    )
    info = AssociationInfo(
        schema_name="eye-ai",
        table_name="Subject_Image",
        fk1_columns=["Image_RID"],
        fk1_target=("eye-ai", "Image"),
        fk2_columns=["Subject_RID"],
        fk2_target=("eye-ai", "Subject"),
    )
    table_rid = {("eye-ai", "Subject_Image"): "1-ABCD"}
    column_rid = {
        ("1-ABCD", "Image_RID"): "1-FFFF",
        ("1-ABCD", "Subject_RID"): "1-EEEE",
    }
    section, skipped = _render_association_section(info, table_rid, column_rid)
    assert skipped is None
    # Header comment names both endpoints and the association table.
    assert "Subject_Image" in section
    assert "eye-ai.Image" in section and "eye-ai.Subject" in section
    # Four CREATE INDEX statements: 2 live + 2 history.
    assert section.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 4
    # Live indexes reference the live table.
    assert '"eye-ai"."Subject_Image"' in section
    # History indexes reference _ermrest_history."t<rid>".
    assert '_ermrest_history."t1-ABCD"' in section
    # Forward and reverse orderings present.
    assert '("Image_RID", "Subject_RID")' in section
    assert '("Subject_RID", "Image_RID")' in section
    assert "(rowdata->>'1-FFFF'), (rowdata->>'1-EEEE')" in section
    assert "(rowdata->>'1-EEEE'), (rowdata->>'1-FFFF')" in section


def test_render_association_section_returns_skip_when_table_rid_missing():
    # Live indexes still emit; history indexes skipped.
    from deriva_ml.tools.generate_association_indexes import (
        AssociationInfo,
        _render_association_section,
    )
    info = AssociationInfo(
        schema_name="s", table_name="A",
        fk1_columns=["a"], fk1_target=("s", "T1"),
        fk2_columns=["b"], fk2_target=("s", "T2"),
    )
    section, skipped = _render_association_section(info, {}, {})
    assert skipped is not None
    assert "table RID" in skipped.lower()
    # Two live indexes still emitted.
    assert section.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 2
    assert "_ermrest_history" not in section


def test_render_association_section_skips_history_when_column_rid_missing():
    from deriva_ml.tools.generate_association_indexes import (
        AssociationInfo,
        _render_association_section,
    )
    info = AssociationInfo(
        schema_name="s", table_name="A",
        fk1_columns=["a"], fk1_target=("s", "T1"),
        fk2_columns=["b"], fk2_target=("s", "T2"),
    )
    table_rid = {("s", "A"): "1-AA"}
    column_rid = {("1-AA", "a"): "1-BB"}  # column "b" missing
    section, skipped = _render_association_section(info, table_rid, column_rid)
    assert skipped is not None
    assert "column RID" in skipped.lower()
    # Live still emitted, history skipped.
    assert section.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 2
    assert "_ermrest_history" not in section
```

- [ ] **Step 7.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: FAIL with `ImportError` for `_render_association_section`.

- [ ] **Step 7.3: Implement `_render_association_section`**

Add to `src/deriva_ml/tools/generate_association_indexes.py`:

```python
from deriva_ml.tools._index_sql import (
    history_index_sql,
    live_index_sql,
    truncate_index_name,
)


def _render_association_section(
    info: AssociationInfo,
    table_rid: TableRidMap,
    column_rid: ColumnRidMap,
) -> tuple[str, str | None]:
    """Render the SQL block for one association.

    Always emits the two live-table composite-pair indexes (forward
    and reverse). Emits the two history-table expression indexes only
    if the association's table RID and every FK column's RID can be
    resolved via the lookups.

    Args:
        info: The association descriptor from ``_walk_associations``.
        table_rid: Lookup built by ``_parse_rid_lookup``.
        column_rid: Lookup built by ``_parse_rid_lookup``.

    Returns:
        A pair ``(sql_section, skip_reason)``:

        - ``sql_section`` is the multi-line text for this association,
          including a comment header and 2–4 CREATE INDEX statements.
        - ``skip_reason`` is ``None`` on full success, or a short
          human-readable message explaining why history-table indexes
          were skipped.

    Example:
        >>> info = AssociationInfo("s", "A", ["a"], ("s", "T1"),
        ...                        ["b"], ("s", "T2"))
        >>> section, _ = _render_association_section(
        ...     info, {("s", "A"): "1-AA"},
        ...     {("1-AA", "a"): "1-XA", ("1-AA", "b"): "1-XB"})
        >>> "CREATE INDEX" in section
        True
    """
    fwd_cols = info.fk1_columns + info.fk2_columns
    rev_cols = info.fk2_columns + info.fk1_columns

    header = (
        "-- ========================================================================\n"
        f"-- {info.table_name}  ({info.schema_name}.{info.table_name})\n"
    )
    t_rid = table_rid.get((info.schema_name, info.table_name))
    if t_rid is not None:
        header += f"--   Table RID: {t_rid}\n"
    header += (
        f"--   FK1: ({', '.join(info.fk1_columns)})  ->  "
        f"{info.fk1_target[0]}.{info.fk1_target[1]}\n"
        f"--   FK2: ({', '.join(info.fk2_columns)})  ->  "
        f"{info.fk2_target[0]}.{info.fk2_target[1]}\n"
        "-- ========================================================================\n"
    )

    statements = [
        live_index_sql(
            schema=info.schema_name,
            table=info.table_name,
            index_name=truncate_index_name(f"{info.table_name}_assoc_fwd_idx"),
            columns=fwd_cols,
        ),
        live_index_sql(
            schema=info.schema_name,
            table=info.table_name,
            index_name=truncate_index_name(f"{info.table_name}_assoc_rev_idx"),
            columns=rev_cols,
        ),
    ]

    skip_reason: str | None = None
    if t_rid is None:
        skip_reason = "history indexes skipped: table RID not found in /schema"
    else:
        fwd_rids: list[str] = []
        for c in fwd_cols:
            r = column_rid.get((t_rid, c))
            if r is None:
                skip_reason = (
                    f"history indexes skipped: column RID for "
                    f"{info.schema_name}.{info.table_name}.{c} not found"
                )
                break
            fwd_rids.append(r)
        if skip_reason is None:
            rev_rids = [column_rid[(t_rid, c)] for c in rev_cols]
            statements.append(
                history_index_sql(
                    table_rid=t_rid,
                    index_name=truncate_index_name(
                        f"{info.table_name}_hist_assoc_fwd_idx"
                    ),
                    column_rids=fwd_rids,
                )
            )
            statements.append(
                history_index_sql(
                    table_rid=t_rid,
                    index_name=truncate_index_name(
                        f"{info.table_name}_hist_assoc_rev_idx"
                    ),
                    column_rids=rev_rids,
                )
            )

    section = header + "\n" + "\n\n".join(statements) + "\n"
    return section, skip_reason
```

- [ ] **Step 7.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 7.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_generate_association_indexes.py
git commit -m "feat(tools): render full SQL section for one association

Two live-table composite-pair indexes always emitted; two history-
table expression indexes emitted only when table RID and all column
RIDs are resolvable. Live emission is independent of history so a
missing column RID does not lose live indexes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Top-level header and summary structures

**Files:**
- Modify: `src/deriva_ml/tools/generate_association_indexes.py`
- Modify: `tests/tools/test_generate_association_indexes.py`

The output file gets a header block (catalog URL, generation timestamp, counts) and the tool returns a summary dataclass for callers.

- [ ] **Step 8.1: Write failing tests**

Append to `tests/tools/test_generate_association_indexes.py`:

```python
def test_render_file_header_includes_catalog_and_timestamp():
    from deriva_ml.tools.generate_association_indexes import _render_file_header
    out = _render_file_header(
        hostname="www.eye-ai.org",
        catalog_id="eye-ai",
        generated_at="2026-04-30T14:23:00Z",
        schemas_scanned=3,
        associations_found=12,
        tool_version="1.2.3",
    )
    assert "https://www.eye-ai.org/ermrest/catalog/eye-ai" in out
    assert "2026-04-30T14:23:00Z" in out
    assert "1.2.3" in out
    assert "Schemas scanned:    3" in out
    assert "Associations found: 12" in out
    # Header is comment-only; no executable SQL.
    assert "CREATE" not in out


def test_generation_summary_fields():
    from deriva_ml.tools.generate_association_indexes import GenerationSummary
    s = GenerationSummary(
        output_path="/tmp/foo.sql",
        schemas_scanned=2,
        associations_found=5,
        history_indexes_skipped=1,
        skip_reasons=["one"],
    )
    assert s.output_path == "/tmp/foo.sql"
    assert s.schemas_scanned == 2
    assert s.associations_found == 5
    assert s.history_indexes_skipped == 1
    assert s.skip_reasons == ["one"]
```

- [ ] **Step 8.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: FAIL with `ImportError` for `_render_file_header` and `GenerationSummary`.

- [ ] **Step 8.3: Implement header renderer and summary dataclass**

Add to `src/deriva_ml/tools/generate_association_indexes.py`:

```python
@dataclass(frozen=True)
class GenerationSummary:
    """Summary of a generate() run.

    Attributes:
        output_path: Path to the written .sql file (string for easy
            consumption by downstream callers).
        schemas_scanned: Count of non-system schemas walked.
        associations_found: Count of pure binary associations detected.
        history_indexes_skipped: Count of associations whose history
            indexes could not be emitted (table RID or column RID
            missing). Live indexes for these are still emitted.
        skip_reasons: Per-skip human-readable reasons.
    """

    output_path: str
    schemas_scanned: int
    associations_found: int
    history_indexes_skipped: int
    skip_reasons: list[str]


def _render_file_header(
    *,
    hostname: str,
    catalog_id: str,
    generated_at: str,
    schemas_scanned: int,
    associations_found: int,
    tool_version: str,
) -> str:
    """Render the top-of-file comment block.

    Args:
        hostname: ERMrest server hostname.
        catalog_id: Catalog ID or alias.
        generated_at: ISO 8601 UTC timestamp string.
        schemas_scanned: Number of non-system schemas walked.
        associations_found: Number of associations emitted.
        tool_version: deriva-ml version string.

    Returns:
        A multi-line comment block (no executable SQL).
    """
    return (
        f"-- Generated by deriva-ml-generate-association-indexes v{tool_version}\n"
        f"-- Catalog:    https://{hostname}/ermrest/catalog/{catalog_id}\n"
        f"-- Generated:  {generated_at}\n"
        f"-- Schemas scanned:    {schemas_scanned}\n"
        f"-- Associations found: {associations_found}\n"
        "--\n"
        "-- Review before running. Apply via pgAdmin or `psql -f`.\n"
        "-- Each CREATE INDEX uses CONCURRENTLY so it does not block live\n"
        "-- traffic, and IF NOT EXISTS so re-runs are safe no-ops.\n"
        "-- Expression indexes on _ermrest_history.* tables key on column\n"
        "-- RIDs, not column names -- see ERMrest's ermrest_schema.sql\n"
        "-- for details.\n"
    )
```

- [ ] **Step 8.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 8.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_generate_association_indexes.py
git commit -m "feat(tools): add file header renderer and GenerationSummary

Header identifies catalog URL, generation timestamp, and tool
version. Summary dataclass returned by generate() lets notebook
callers consume the run result programmatically.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: `generate()` end-to-end orchestration

**Files:**
- Modify: `src/deriva_ml/tools/generate_association_indexes.py`
- Modify: `tests/tools/test_generate_association_indexes.py`

`generate(hostname, catalog_id, output_path)` ties everything together: connect, fetch model + schema doc, walk, emit, write file, return summary.

- [ ] **Step 9.1: Write failing test using monkeypatch**

Append to `tests/tools/test_generate_association_indexes.py`:

```python
def test_generate_writes_file_and_returns_summary(monkeypatch, tmp_path):
    """End-to-end with the catalog accessor stubbed out."""
    from deriva_ml.tools import generate_association_indexes as mod

    fk1 = _fake_fk(["Image_RID"], "eye-ai", "Image")
    fk2 = _fake_fk(["Subject_RID"], "eye-ai", "Subject")
    assoc = _fake_table("eye-ai", "Subject_Image", [fk1, fk2])
    image = _fake_table("eye-ai", "Image", None)
    subject = _fake_table("eye-ai", "Subject", None)
    model = _fake_model([_fake_schema("eye-ai", [assoc, image, subject])])

    schema_doc = {
        "schemas": {
            "eye-ai": {
                "RID": "0-AAAA",
                "tables": {
                    "Subject_Image": {
                        "RID": "1-ABCD",
                        "column_definitions": [
                            {"RID": "1-EEEE", "name": "Subject_RID"},
                            {"RID": "1-FFFF", "name": "Image_RID"},
                        ],
                    },
                },
            }
        }
    }

    def fake_open(hostname, catalog_id):
        assert hostname == "h"
        assert catalog_id == "c"
        return model, schema_doc

    monkeypatch.setattr(mod, "_open_catalog", fake_open)

    out = tmp_path / "test.sql"
    summary = mod.generate(hostname="h", catalog_id="c", output_path=str(out))

    assert summary.output_path == str(out)
    assert summary.associations_found == 1
    assert summary.history_indexes_skipped == 0
    assert summary.schemas_scanned == 1
    text = out.read_text(encoding="utf-8")
    # 4 CREATE INDEX statements (2 live + 2 history).
    assert text.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 4
    assert "https://h/ermrest/catalog/c" in text
    assert '_ermrest_history."t1-ABCD"' in text


def test_generate_records_skips_when_history_rids_missing(monkeypatch, tmp_path):
    from deriva_ml.tools import generate_association_indexes as mod

    fk1 = _fake_fk(["a"], "s", "T1")
    fk2 = _fake_fk(["b"], "s", "T2")
    assoc = _fake_table("s", "A", [fk1, fk2])
    t1 = _fake_table("s", "T1", None)
    t2 = _fake_table("s", "T2", None)
    model = _fake_model([_fake_schema("s", [assoc, t1, t2])])
    schema_doc = {"schemas": {}}  # no RIDs at all

    monkeypatch.setattr(mod, "_open_catalog", lambda h, c: (model, schema_doc))
    out = tmp_path / "test.sql"
    summary = mod.generate(hostname="h", catalog_id="c", output_path=str(out))
    assert summary.associations_found == 1
    assert summary.history_indexes_skipped == 1
    assert len(summary.skip_reasons) == 1
    text = out.read_text(encoding="utf-8")
    # Live indexes still emitted; history not.
    assert text.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 2
    assert "_ermrest_history" not in text


def test_generate_zero_associations_writes_header_only(monkeypatch, tmp_path):
    from deriva_ml.tools import generate_association_indexes as mod
    plain = _fake_table("s", "Plain", None)
    model = _fake_model([_fake_schema("s", [plain])])
    monkeypatch.setattr(mod, "_open_catalog", lambda h, c: (model, {"schemas": {}}))
    out = tmp_path / "test.sql"
    summary = mod.generate(hostname="h", catalog_id="c", output_path=str(out))
    assert summary.associations_found == 0
    text = out.read_text(encoding="utf-8")
    assert "CREATE" not in text
    assert "Associations found: 0" in text
```

- [ ] **Step 9.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_open_catalog'` (or `generate`).

- [ ] **Step 9.3: Implement `_open_catalog` and `generate`**

Add to `src/deriva_ml/tools/generate_association_indexes.py`:

```python
import importlib
from datetime import datetime, timezone
from pathlib import Path

# Lazy-imported deriva-py modules — keep top-level imports cheap so
# unit tests don't pay the cost.
def _open_catalog(hostname: str, catalog_id: str):
    """Connect to a catalog and return (model, raw_schema_doc).

    Split out so unit tests can monkeypatch it without touching the
    real network. Production callers always reach this implementation.

    Args:
        hostname: ERMrest server hostname (no scheme).
        catalog_id: Catalog ID or alias.

    Returns:
        ``(model, schema_doc)`` where ``model`` is a deriva-py
        ``ermrest_model.Model`` and ``schema_doc`` is the raw
        ``/schema`` JSON document.
    """
    deriva_core = importlib.import_module("deriva.core")
    server = deriva_core.DerivaServer(
        "https", hostname,
        credentials=deriva_core.get_credential(hostname),
    )
    catalog = server.connect_ermrest(catalog_id)
    model = catalog.getCatalogModel()
    schema_doc = catalog.get("/schema").json()
    return model, schema_doc


def _tool_version() -> str:
    """Return the deriva-ml package version, or ``'unknown'``."""
    try:
        from importlib.metadata import version
        return version("deriva-ml")
    except Exception:
        return "unknown"


def generate(
    *,
    hostname: str,
    catalog_id: str,
    output_path: str,
) -> GenerationSummary:
    """Generate an association-indexes SQL file for one catalog.

    Connects to the catalog, walks pure binary association tables,
    and writes a ``.sql`` file ready to apply via pgAdmin or
    ``psql -f``. Read-only against the catalog.

    Args:
        hostname: ERMrest server hostname (no scheme).
        catalog_id: Catalog ID or alias (e.g. ``"eye-ai"``).
        output_path: Filesystem path to write the SQL file to.

    Returns:
        A ``GenerationSummary`` describing what was emitted and any
        partial skips.

    Raises:
        OSError: If the output file cannot be written.

    Example:
        >>> from deriva_ml.tools.generate_association_indexes import generate
        >>> generate(hostname="www.eye-ai.org",  # doctest: +SKIP
        ...          catalog_id="eye-ai",
        ...          output_path="/tmp/eye-ai.sql")
    """
    model, schema_doc = _open_catalog(hostname, catalog_id)
    table_rid_map, column_rid_map = _parse_rid_lookup(schema_doc)

    associations = list(_walk_associations(model))
    schemas_scanned = sum(
        1 for s in model.schemas if s not in _SKIP_SCHEMAS
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sections: list[str] = []
    skip_reasons: list[str] = []
    for info in associations:
        section, reason = _render_association_section(
            info, table_rid_map, column_rid_map
        )
        sections.append(section)
        if reason is not None:
            skip_reasons.append(
                f"{info.schema_name}.{info.table_name}: {reason}"
            )

    header = _render_file_header(
        hostname=hostname,
        catalog_id=catalog_id,
        generated_at=generated_at,
        schemas_scanned=schemas_scanned,
        associations_found=len(associations),
        tool_version=_tool_version(),
    )

    body = "\n".join(sections)
    contents = header + ("\n" + body if body else "")

    Path(output_path).write_text(contents, encoding="utf-8", newline="\n")

    return GenerationSummary(
        output_path=output_path,
        schemas_scanned=schemas_scanned,
        associations_found=len(associations),
        history_indexes_skipped=len(skip_reasons),
        skip_reasons=skip_reasons,
    )
```

- [ ] **Step 9.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 9.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_generate_association_indexes.py
git commit -m "feat(tools): add generate() end-to-end orchestration

Connects to a catalog, parses RIDs from the raw /schema doc, walks
associations, and writes a single .sql file. _open_catalog is split
out so unit tests can monkeypatch it without touching the network.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: CLI entry point

**Files:**
- Modify: `src/deriva_ml/tools/generate_association_indexes.py`
- Modify: `tests/tools/test_generate_association_indexes.py`

The argparse entry point. Default output path includes hostname, catalog ID, and a UTC timestamp so re-runs don't overwrite each other.

- [ ] **Step 10.1: Write failing tests**

Append to `tests/tools/test_generate_association_indexes.py`:

```python
def test_default_output_path_shape():
    from deriva_ml.tools.generate_association_indexes import _default_output_path
    p = _default_output_path("www.eye-ai.org", "eye-ai", "20260430T142300Z")
    assert p == "association-indexes-www.eye-ai.org-eye-ai-20260430T142300Z.sql"


def test_main_invokes_generate_and_returns_zero(monkeypatch, tmp_path, capsys):
    from deriva_ml.tools import generate_association_indexes as mod

    captured: dict = {}
    fake_summary = mod.GenerationSummary(
        output_path=str(tmp_path / "x.sql"),
        schemas_scanned=2,
        associations_found=3,
        history_indexes_skipped=0,
        skip_reasons=[],
    )

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return fake_summary

    monkeypatch.setattr(mod, "generate", fake_generate)
    rc = mod.main(["--hostname", "h", "--catalog-id", "c",
                   "--output", str(tmp_path / "x.sql")])
    assert rc == 0
    assert captured["hostname"] == "h"
    assert captured["catalog_id"] == "c"
    assert captured["output_path"] == str(tmp_path / "x.sql")
    out = capsys.readouterr().out
    assert "associations_found=3" in out or "Associations found: 3" in out


def test_main_returns_nonzero_on_connection_error(monkeypatch):
    from deriva_ml.tools import generate_association_indexes as mod

    def boom(**_kwargs):
        raise RuntimeError("could not connect: 401")

    monkeypatch.setattr(mod, "generate", boom)
    rc = mod.main(["--hostname", "h", "--catalog-id", "c"])
    assert rc == 2
```

- [ ] **Step 10.2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: FAIL with `ImportError` for `_default_output_path` / `main`.

- [ ] **Step 10.3: Implement CLI entry point**

Add to `src/deriva_ml/tools/generate_association_indexes.py`:

```python
import argparse
import logging
import sys


def _default_output_path(hostname: str, catalog_id: str, ts: str) -> str:
    """Build the default output filename.

    Includes hostname, catalog ID, and a UTC timestamp so successive
    runs don't overwrite each other. Returned as a relative filename
    (cwd-relative).

    Args:
        hostname: ERMrest server hostname.
        catalog_id: Catalog ID or alias.
        ts: UTC timestamp string in ``YYYYMMDDTHHMMSSZ`` form.

    Returns:
        ``"association-indexes-<host>-<id>-<ts>.sql"``.
    """
    return f"association-indexes-{hostname}-{catalog_id}-{ts}.sql"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deriva-ml-generate-association-indexes",
        description=(
            "Scan a Deriva catalog for pure binary association tables "
            "and emit a SQL file with composite-pair indexes for both "
            "join orderings on each association's live table and "
            "expression indexes on its _ermrest_history table. "
            "Read-only against the catalog. Apply the generated SQL "
            "via pgAdmin or `psql -f`."
        ),
    )
    p.add_argument("--hostname", required=True,
                   help="ERMrest server hostname (no scheme).")
    p.add_argument("--catalog-id", required=True,
                   help="Catalog ID or alias.")
    p.add_argument("--output",
                   help=(
                       "Output SQL file path. Defaults to "
                       "association-indexes-<host>-<id>-<utc>.sql in cwd."
                   ))
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``2`` on connection / write failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    log = logging.getLogger("deriva_ml.tools.generate_association_indexes")

    output_path = args.output or _default_output_path(
        args.hostname,
        args.catalog_id,
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )

    try:
        summary = generate(
            hostname=args.hostname,
            catalog_id=args.catalog_id,
            output_path=output_path,
        )
    except Exception as exc:
        log.error("generation failed: %s", exc)
        return 2

    print(
        f"wrote {summary.output_path} "
        f"(schemas_scanned={summary.schemas_scanned}, "
        f"associations_found={summary.associations_found}, "
        f"history_indexes_skipped={summary.history_indexes_skipped})"
    )
    for reason in summary.skip_reasons:
        log.warning("skip: %s", reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 10.4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/tools/test_generate_association_indexes.py -v`
Expected: PASS, 17 tests.

- [ ] **Step 10.5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_generate_association_indexes.py
git commit -m "feat(tools): add CLI entry point for index SQL generator

argparse-driven; default output path includes hostname, catalog ID,
and UTC timestamp. Returns 0 on success, 2 on failure. Follows the
existing deriva-ml-* CLI pattern.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Register the CLI in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 11.1: Read the current `[project.scripts]` block**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n -A 20 'project.scripts' pyproject.toml`
Note the line range for the next step.

- [ ] **Step 11.2: Add the new script entry**

Use the Edit tool on `/Users/carl/GitHub/DerivaML/deriva-ml/pyproject.toml`. Find the line:

```
deriva-ml-validate-schema = "deriva_ml.tools.validate_schema_doc:main"
```

and append immediately below it (preserving alphabetical-by-suffix grouping with the other tools entries):

```
deriva-ml-generate-association-indexes = "deriva_ml.tools.generate_association_indexes:main"
```

- [ ] **Step 11.3: Sync and verify the entry point installs**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv sync && uv run deriva-ml-generate-association-indexes --help`
Expected: argparse usage block with `--hostname`, `--catalog-id`, `--output`, `--verbose`.

- [ ] **Step 11.4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add pyproject.toml uv.lock
git commit -m "chore(tools): register deriva-ml-generate-association-indexes script

Adds the CLI entry point to pyproject.toml so the command is available
after uv sync.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Lint and ruff format

**Files:**
- All new/modified `.py` files.

- [ ] **Step 12.1: Run ruff check**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/tools/_index_sql.py src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_index_sql.py tests/tools/test_generate_association_indexes.py`
Expected: no errors. If there are errors, fix them and re-run.

- [ ] **Step 12.2: Run ruff format**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff format src/deriva_ml/tools/_index_sql.py src/deriva_ml/tools/generate_association_indexes.py tests/tools/test_index_sql.py tests/tools/test_generate_association_indexes.py`
Expected: files left unchanged or reformatted.

- [ ] **Step 12.3: Re-run all tests after formatting**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_index_sql.py tests/tools/test_generate_association_indexes.py -v`
Expected: PASS (17 + the count from Task 1–10).

- [ ] **Step 12.4: Commit if format made changes**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git status
git add -p   # only stage the formatting changes if any
git commit -m "style(tools): ruff format on association-index generator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(If `git status` shows no changes, skip the commit and continue.)

---

## Task 13: Integration test against a live catalog

**Files:**
- Create: `tests/integration/test_generate_association_indexes_integration.py`

This task only runs against `DERIVA_HOST`. Skip the integration test if the env var is unset.

- [ ] **Step 13.1: Verify the integration directory exists and has a conftest pattern**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && ls tests/integration/`
If the directory does not exist, create it with `mkdir -p tests/integration && touch tests/integration/__init__.py`.

- [ ] **Step 13.2: Write the integration test**

Create `tests/integration/test_generate_association_indexes_integration.py`:

```python
"""Integration test: run generate() against a real catalog.

Skipped automatically when DERIVA_HOST is not set, matching the
project's existing test gating convention.
"""

from __future__ import annotations

import os
import re

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires DERIVA_HOST pointing at a live test catalog",
)


def test_generate_against_live_catalog(tmp_path, catalog_with_datasets):
    """Generate against the populated test catalog and inspect output.

    Uses the existing ``catalog_with_datasets`` session fixture so we
    have a known set of association tables to find. Asserts:
      - the file is non-empty,
      - all CREATE INDEX statements are well-formed,
      - re-running produces byte-identical output (modulo timestamp),
      - every emitted index name fits in 63 bytes.
    """
    from deriva_ml.tools.generate_association_indexes import generate

    hostname = os.environ["DERIVA_HOST"]
    catalog_id = catalog_with_datasets.catalog_id

    out1 = tmp_path / "run1.sql"
    out2 = tmp_path / "run2.sql"

    s1 = generate(hostname=hostname, catalog_id=catalog_id, output_path=str(out1))
    s2 = generate(hostname=hostname, catalog_id=catalog_id, output_path=str(out2))

    assert s1.associations_found == s2.associations_found
    assert s1.schemas_scanned == s2.schemas_scanned

    text1 = out1.read_text(encoding="utf-8")
    text2 = out2.read_text(encoding="utf-8")

    # The only line allowed to differ is the "Generated:" timestamp.
    norm1 = re.sub(r"^-- Generated:.*$", "-- Generated: <stripped>", text1, flags=re.M)
    norm2 = re.sub(r"^-- Generated:.*$", "-- Generated: <stripped>", text2, flags=re.M)
    assert norm1 == norm2, "non-timestamp output should be deterministic"

    # Each CREATE INDEX names an identifier <= 63 bytes.
    pattern = re.compile(
        r'CREATE INDEX CONCURRENTLY IF NOT EXISTS "([^"]+(?:""[^"]*)*)"'
    )
    names = pattern.findall(text1)
    assert names, "expected at least one CREATE INDEX statement"
    for name in names:
        unescaped = name.replace('""', '"')
        assert len(unescaped.encode("utf-8")) <= 63, (
            f"index name exceeds 63 bytes: {unescaped!r}"
        )

    # 4 statements per association on full success; minimum 2 per
    # association in the worst case (history skipped). At least 2x
    # associations_found, at most 4x.
    n_create = text1.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS")
    assert 2 * s1.associations_found <= n_create <= 4 * s1.associations_found
```

- [ ] **Step 13.3: Run the integration test (only if `DERIVA_HOST` is set)**

If you have a local catalog:
```
cd /Users/carl/GitHub/DerivaML/deriva-ml
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  tests/integration/test_generate_association_indexes_integration.py -v --timeout=600
```
Expected: PASS, or SKIP if `DERIVA_HOST` is unset.

If you do not have a local catalog, that is fine — the test is gated and will be exercised by CI / on a developer machine that has one. Verify the skip behavior:
```
cd /Users/carl/GitHub/DerivaML/deriva-ml
unset DERIVA_HOST
uv run pytest tests/integration/test_generate_association_indexes_integration.py -v
```
Expected: SKIPPED with the "requires DERIVA_HOST" reason.

- [ ] **Step 13.4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add tests/integration/test_generate_association_indexes_integration.py
# also add tests/integration/__init__.py if you created the dir
git status   # double-check
git commit -m "test(tools): integration test for association-index generator

Runs generate() twice against the populated test catalog and asserts
byte-identical output (modulo the timestamp line), well-formed
CREATE INDEX statements, and 63-byte identifier compliance. Gated
on DERIVA_HOST.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 14: Verify against eye-ai (manual smoke check)

**Files:** none (manual verification).

This is a one-time manual check, not automated. The eye-ai catalog is referenced explicitly in the spec as the example. Confirm the tool works against it before declaring the feature done.

- [ ] **Step 14.1: Run against eye-ai**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run deriva-ml-generate-association-indexes --hostname www.eye-ai.org --catalog-id eye-ai --output /tmp/eye-ai-indexes.sql`
Expected: success message of the form `wrote /tmp/eye-ai-indexes.sql (schemas_scanned=N, associations_found=M, history_indexes_skipped=K)` with M > 0.

- [ ] **Step 14.2: Sanity-check the output**

Run: `head -40 /tmp/eye-ai-indexes.sql && echo '---' && grep -c 'CREATE INDEX' /tmp/eye-ai-indexes.sql`
Expected:
- Header block names `https://www.eye-ai.org/ermrest/catalog/eye-ai`.
- At least one association section follows.
- Total `CREATE INDEX` count is between `2*associations_found` and `4*associations_found`.

- [ ] **Step 14.3: Validate the SQL parses (without executing)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -c "import sqlparse; t = open('/tmp/eye-ai-indexes.sql').read(); stmts = [s for s in sqlparse.split(t) if s.strip() and not s.strip().startswith('--')]; print(f'{len(stmts)} executable statements'); [sqlparse.parse(s) for s in stmts]"`
Expected: prints "<N> executable statements" and exits 0. (`sqlparse` is already a transitive dependency.)

If any of these fail, file a follow-up to fix and do not move on. If they pass, the feature is functional.

- [ ] **Step 14.4: No commit needed.**

---

## Self-review

Spec coverage check (matched §-by-§ to spec at `docs/superpowers/specs/2026-04-30-association-index-sql-generator-design.md`):

- §3 step 1 (connect): Task 9 (`_open_catalog`).
- §3 step 2 (pull model, skip system schemas): Task 5 (`_walk_associations`).
- §3 step 3 (find associations): Task 5.
- §3 step 4 (RID lookups via raw `/schema`): Task 6 (`_parse_rid_lookup`) + Task 9 (call site).
- §3 step 5 (emit SQL per association, two live + two history): Tasks 3, 4, 7.
- §3 step 6 (write output file with default name): Tasks 9, 10.
- §4 (output format header + per-association comment block + 4 statements): Tasks 7, 8.
- §4 (index naming + 63-byte cap): Task 2.
- §4 (composite-FK column expansion): Task 3 (column list passthrough), Task 7 (fwd/rev expansion).
- §5 (module layout, CLI entry, dataclass `AssociationInfo`, `GenerationSummary`): Tasks 5, 8, 10.
- §6 (data flow): all of Task 9.
- §7 (error handling: connect failure → exit 2; RID miss → skip with warning; zero associations → header-only): Tasks 9, 10.
- §8 (idempotence via `IF NOT EXISTS`): Tasks 3, 4.
- §9 (testing strategy: unit + gated integration): Tasks 1–10 (unit), Task 13 (integration), Task 14 (manual smoke).

No spec section lacks a task.

Placeholder scan: no TBDs, every code step shows the actual code, every test step shows the actual assertions, every commit step shows the actual command.

Type consistency: `AssociationInfo` fields used consistently across Tasks 5, 7, 9. `GenerationSummary` fields consistent across Tasks 8, 9, 10. `TableRidMap` / `ColumnRidMap` aliases defined in Task 6, used in Task 7's signature.

One ambiguity worth flagging for the implementer: `_walk_associations` test in Task 5 sorts FKs by `(referenced_schema, referenced_table)`, and my fake-FK helper sets `fk.pk_table.schema.name` accordingly. The real deriva-py `ForeignKey.pk_table.schema.name` exposes the same attribute path — verified by reading `ermrest_model.py`. If the integration test in Task 13 surfaces a mismatch, the fix is in `_walk_associations` only.

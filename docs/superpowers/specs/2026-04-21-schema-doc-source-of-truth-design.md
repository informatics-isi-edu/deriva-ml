# deriva-ml Schema Documentation — Source-of-Truth Design

**Date:** 2026-04-21
**Author:** Claude (with Carl)
**Status:** Approved, ready for implementation-plan
**Phase:** Phase 2, Subsystem 0 (of 5)
**Related spec:** `2026-04-21-status-enum-reconciliation-design.md` (Subsystem 1; blocked on this)

## 1. Motivation

The `deriva-ml` schema (tables, columns, FKs, vocabulary terms) is defined programmatically in `src/deriva_ml/schema/create_schema.py`. That file is Python, executable at catalog-creation time via `create_ml_schema` / `initialize_ml_schema`. It is the authoritative source for what a freshly-created catalog contains.

But there is no corresponding reviewable document describing the schema. The existing `docs/architecture.md` has a 5-row summary table and references `docs/assets/ERD.png` — an image that is 15+ months stale (last updated Jan 2025, well before Phase 1's SQLite-backed execution state and numerous other additions).

Subsystem 1 (status-enum reconciliation) adds `Execution_Status` as a new vocabulary table. Reviewing that change, and every future schema change, is harder than it should be without a current-state reference.

This spec establishes `docs/reference/schema.md` as the **authoritative document** for the `deriva-ml` schema structure and vocabulary seeding. A CI validator enforces that the document and `create_schema.py` agree.

## 2. Decisions (from brainstorming)

- **Path 1 (doc + code coexist, validator enforces agreement).** Both `docs/reference/schema.md` and `src/deriva_ml/schema/create_schema.py` are maintained by developers. The doc describes the intended schema; the code defines it at runtime. Neither is generated from the other. A validator asserts they match on structure and seeded terms.
- **Scope**: `deriva-ml` schema only. Domain schemas (Subject, Image, test-schema, etc.) are user-defined and out of scope.
- **Doc format**: single Markdown file `docs/reference/schema.md`, one section per table, with a fenced YAML block per section carrying the machine-readable structure.
- **Validator scope**: enforces tables, columns (name + type), FKs, vocabulary seeded terms. Descriptions (both table-level and column-level) are narrative in the doc but NOT cross-validated against code comments. Annotations, display configs, indexes — skipped.
- **When validation runs**: CI only. Local developers run `uv run deriva-ml-validate-schema` for faster feedback.
- **Direction 1 ships (code ↔ doc)**; Direction 2 (doc ↔ live catalog) is a filed follow-up.

## 3. Architecture

The validator has three loaders producing one in-memory `SchemaModel` and one comparator:

```
  docs/reference/schema.md                                 schema.md ←→ create_schema.py
      ├── Markdown sections per table                       [Direction 1, SHIPS NOW]
      └── Fenced YAML blocks per section
          │
          ▼ load_from_doc(path) ─┐
                                 │
                                 │  ─────►  diff_schemas(expected, actual)
                                 │              │
  src/deriva_ml/schema/create_schema.py          ▼
      ├── TableDef / VocabularyTableDef          list[Mismatch]
      ├── ColumnDef / ForeignKeyDef              │
      └── _ensure_terms(...)                     ▼
          │                                   Exit 0 or 1
          ▼ load_from_code(module) ─┘

  (follow-up: load_from_catalog(catalog) for Direction 2)
```

All loaders produce the same `SchemaModel` shape. The comparator knows nothing about where the data came from.

## 4. File layout

### 4.1 `docs/reference/schema.md`

Single Markdown file. Structure:

- **Top-of-file prose** (~50-100 lines): overview, how to edit the doc + code together, schema-level invariants.
- **Per-table sections**, ordered:
  1. Core entity tables (Dataset, Execution, Workflow, Feature_Name, ...).
  2. Vocabulary tables (Asset_Type, Dataset_Type, Workflow_Type, Execution_Status, ...).
  3. Association tables (Dataset_Execution, Nested_Execution, ...).

Each section:

~~~markdown
## Execution

Per-execution lifecycle row. Created once per workflow run; updated as state transitions per Phase-1 spec §2.2.

```yaml
table: Execution
kind: table
description: Per-execution lifecycle row.
columns:
  - name: Workflow
    type: text
  - name: Description
    type: markdown
  - name: Duration
    type: text
  - name: Status
    type: fk
  - name: Status_Detail
    type: text
foreign_keys:
  - columns: [Workflow]
    referenced_schema: deriva-ml
    referenced_table: Workflow
    referenced_columns: [RID]
  - columns: [Status]
    referenced_schema: deriva-ml
    referenced_table: Execution_Status
    referenced_columns: [Name]
```
~~~

Vocabulary tables also include a `terms:` list:

~~~yaml
table: Execution_Status
kind: vocabulary
description: Controlled vocabulary for Execution.Status lifecycle values.
terms:
  - name: Created
  - name: Running
  - name: Stopped
  - name: Failed
  - name: Pending_Upload
  - name: Uploaded
  - name: Aborted
~~~

Association tables have an `associates:` list:

~~~yaml
table: Nested_Execution
kind: association
description: Hierarchical execution nesting (parent → child).
associates:
  - table: Execution
    role: parent
  - table: Execution
    role: child
metadata:
  - name: Sequence
    type: int4
    nullok: true
~~~

### 4.2 `tools/validate_schema_doc.py`

Python module with:

- `SchemaModel` dataclass (tables list).
- `TableModel`, `ColumnModel`, `ForeignKeyModel`, `VocabularyTermModel`, `AssociationEndpointModel` sub-dataclasses.
- `load_from_doc(path: Path) -> SchemaModel` — Markdown parser + YAML deserializer.
- `load_from_code(module_path: Path) -> SchemaModel` — AST parser over `create_schema.py`.
- `diff_schemas(expected: SchemaModel, actual: SchemaModel) -> list[Mismatch]` — comparator.
- `Mismatch` dataclass with `kind`, `location`, `detail` fields.
- `main(argv)` — CLI entry point.

### 4.3 `pyproject.toml` entry point

```toml
[project.scripts]
deriva-ml-validate-schema = "deriva_ml_tools.validate_schema_doc:main"
```

(Or similar path — consistent with existing `deriva-ml-*` scripts.)

### 4.4 `.github/workflows/validate-schema.yml`

GitHub Actions workflow (or step in existing test workflow):

```yaml
name: Validate deriva-ml schema doc
on: [pull_request]
jobs:
  validate-schema:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Run validator
        run: uv run deriva-ml-validate-schema
```

### 4.5 `docs/reference/README.md`

Short contributor doc explaining the doc-first workflow:

```markdown
# Reference docs

`schema.md` is the authoritative description of the deriva-ml schema.
To modify the schema:

1. Edit `docs/reference/schema.md` to describe the intended state.
2. Edit `src/deriva_ml/schema/create_schema.py` to match.
3. Run `uv run deriva-ml-validate-schema` locally to verify agreement.
4. Commit both files together. CI re-runs the validator.
```

## 5. Validator implementation

### 5.1 `load_from_doc`

Parse the Markdown. Find fenced code blocks with language `yaml`. Deserialize each block into a dict via `yaml.safe_load`. Build `SchemaModel` entries based on `table:` and `kind:` keys.

Malformed YAML, missing required keys, unknown `kind:` → raise `SchemaDocError` with the Markdown line number. The CLI reports these as parse failures rather than mismatches.

### 5.2 `load_from_code`

AST-parse `create_schema.py` without executing it. Walk the tree looking for these call patterns:

- `schema.create_table(TableDef(name=..., columns=[...], foreign_keys=[...]))` → table.
- `schema.create_table(VocabularyTableDef(name=..., curie_template=...))` → vocab table.
- `schema.create_table(Table.define_association(associates=[...], metadata=[...]))` → association table.
- `_ensure_terms(vocab_name, [{"Name": ..., ...}, ...])` → seeded terms for a vocab.

Enum references (`MLTable.execution`, `MLVocab.workflow_type`) are resolved by AST-parsing `src/deriva_ml/core/definitions.py` and building a `{enum_member: value}` lookup.

Dynamic Python (f-strings in `curie_template`, conditional FKs, comprehensions building column lists) are intentionally out of scope. If a `ColumnDef(...)` or `ForeignKeyDef(...)` is inside a comprehension or conditional, the validator extracts it when the outer structure is still statically walkable; if not, it reports a `SchemaCodeError` with the code line number.

### 5.3 `diff_schemas`

Given expected (doc-side) and actual (code-side) `SchemaModel` instances:

- For every table name in either: if present in both, recursively compare; if only in one, emit `MissingTable` mismatch.
- For each table's columns: compare by name; mismatches on type emit `ColumnMismatch`.
- For each table's FKs: compare by `(columns, referenced_table, referenced_columns)` tuple. Missing or extra FK → `ForeignKeyMismatch`.
- For vocab tables: compare the set of `term.name` values. Symmetric difference → `VocabularyTermsMismatch`.
- For association tables: compare `associates` endpoints (by target table name) and `metadata` columns.
- Column/table descriptions skipped per §2.
- Annotations, indexes, display configs skipped per §2.

### 5.4 CLI

```
$ deriva-ml-validate-schema [--doc PATH] [--code PATH] [--definitions PATH]
```

Default paths: `docs/reference/schema.md`, `src/deriva_ml/schema/create_schema.py`, `src/deriva_ml/core/definitions.py` (resolved from the working directory).

Output on mismatch: structured text (see §5.5 for format). Exit code 1.
Output on match: single line `deriva-ml-validate-schema: schema.md and create_schema.py agree.` Exit code 0.
Output on parse error: error location + message. Exit code 2.

### 5.5 Mismatch output format

```
deriva-ml-validate-schema: schema.md and create_schema.py disagree.

MISSING FROM CODE:
  - table 'Execution_Status' declared in docs/reference/schema.md:142
    but not found in src/deriva_ml/schema/create_schema.py.

MISSING FROM DOC:
  (none)

COLUMN MISMATCH:
  - Execution.Status
    doc (schema.md:78):   type=fk
    code (create_schema.py:149): type=text

FOREIGN KEY MISMATCH:
  (none)

VOCABULARY TERMS MISMATCH:
  - Asset_Type (schema.md:203; create_schema.py:395):
    code-only: {Model_File, Notebook_Output}
    doc-only:  {Execution_Config}

Exit code: 1
```

## 6. Testing strategy

### 6.1 Unit tests (`tests/tools/test_validate_schema_doc.py`)

1. `test_load_from_doc_parses_yaml_blocks` — handcrafted fixture Markdown with two tables; verify `SchemaModel` fields populated.
2. `test_load_from_code_parses_simple_tabledef` — fixture Python module with a `TableDef`; verify AST extraction matches.
3. `test_load_from_code_parses_vocabulary_table` — fixture with `VocabularyTableDef`; verify.
4. `test_load_from_code_parses_ensure_terms` — fixture with `_ensure_terms` calls; verify seeded-terms extraction.
5. `test_load_from_code_resolves_mlvocab_enum_refs` — fixture that uses `MLVocab.workflow_type`; verify resolver maps to "Workflow_Type".
6. `test_diff_identical_schemas_is_empty` — two identical models → zero mismatches.
7. `test_diff_missing_table` — table in doc, missing in code → `MissingTable` mismatch.
8. `test_diff_column_type_mismatch` — same column, different types → `ColumnMismatch`.
9. `test_diff_fk_target_mismatch` — FK target differs → `ForeignKeyMismatch`.
10. `test_diff_vocab_terms_differ` — extra term in code → `VocabularyTermsMismatch` with correct symmetric-difference.
11. `test_diff_ignores_descriptions` — schemas identical except doc has descriptions → zero mismatches (per §2).
12. `test_diff_ignores_annotations` — schemas identical except code has annotations → zero mismatches.

### 6.2 Integration tests (`tests/tools/test_validate_schema_doc_integration.py`)

13. `test_validator_runs_clean_on_current_repo` — load actual `docs/reference/schema.md` + actual `src/deriva_ml/schema/create_schema.py`; assert zero mismatches. This test IS the CI gate in test form — if the two drift, this fails.
14. `test_cli_exit_codes` — subprocess invocation of `deriva-ml-validate-schema` on the real repo; assert exit 0.

### 6.3 Doc structure tests (`tests/tools/test_schema_doc_structure.py`)

15. `test_schema_doc_yaml_blocks_all_valid` — every fenced YAML block parses without error.
16. `test_schema_doc_has_entry_per_mltable_member` — every member of `MLTable` enum has a corresponding `table:` entry in the doc. Catches "added a table to `MLTable` but forgot the doc."
17. `test_schema_doc_has_entry_per_mlvocab_member` — parallel for `MLVocab`.
18. `test_schema_doc_table_order` — core → vocabulary → association. Warns (not fails) on ordering violations; enforces the readability convention.

### 6.4 CI

GitHub Actions workflow runs `uv run deriva-ml-validate-schema` on every PR. Failure surfaces in the PR check. No extra setup — the validator is pure-Python + PyYAML (transitive) + stdlib `ast`.

## 7. Error handling

- **Parse errors in the doc** (malformed YAML, unknown `kind:`, missing required key): validator reports `SchemaDocError` with Markdown line number. Exit 2.
- **Parse errors in the code** (call pattern inside a comprehension, enum member can't resolve): `SchemaCodeError` with code line number. Exit 2.
- **Mismatch**: structured output per §5.5. Exit 1.
- **Agreement**: single-line OK. Exit 0.

Parse errors are a developer-facing signal: "the tool couldn't understand this section; refactor it or extend the validator."

## 8. Bootstrap

The initial `docs/reference/schema.md` is hand-written to match the current `create_schema.py` exactly. Implementation plan task 1:

1. Run the validator's `load_from_code` on the current `create_schema.py` to produce a `SchemaModel`.
2. Render that `SchemaModel` back to the doc format as a starting point.
3. Hand-edit the result: add prose, add the top-of-file overview, add descriptions, re-order sections per §4.1.
4. Commit the doc. Validator passes with zero mismatches.

A one-shot "render SchemaModel to Markdown" helper (`SchemaModel.to_doc_markdown() -> str`) can ship as part of the tool — useful for bootstrap, handy for future "emergency regenerate" flows (reset the doc from code, re-add prose).

## 9. Filed follow-ups

These are deferred by design:

1. **Direction 2 — live catalog validation.** Add `load_from_catalog(catalog: ErmrestCatalog) -> SchemaModel` loader and CLI flags `--against=catalog --hostname=X --catalog=N`. Enables operational "does my deployed catalog match the doc?" diagnostics. Separate subsystem; not Subsystem 0.
2. **Descriptions validation.** Extend validator to enforce `comment=` ↔ `description:` matching (opt-in per-column). Requires bootstrap pass adding `comment=` to existing `ColumnDef` sites. Deferred per Q8-c decision.
3. **Annotation validation.** `curie_template`, display annotations, FK annotations. Heavy; likely never worth it.
4. **ERD image regeneration.** Replace stale `docs/assets/ERD.png`. Separate effort; could be driven from `docs/reference/schema.md` via Graphviz or similar.

## 10. Non-goals

- Domain schema documentation (Subject, Image, test-schema, etc.) — user-defined, not part of `create_schema.py`.
- Generating `create_schema.py` from the doc — explicitly rejected (Path 2); loses Python expressiveness.
- Replacing `docs/architecture.md` — different purpose (overview + concepts vs. structural reference). The two coexist.
- Schema migration tooling for existing catalogs — Subsystem 1's concern.
- Real-time schema drift monitoring — out of scope; Direction 2 is on-demand.

## 11. Delivery order (for the plan)

Rough sequence; the implementation plan will refine into bite-sized steps:

1. Create `tools/validate_schema_doc.py` skeleton: `SchemaModel` dataclass hierarchy, loader/diff/CLI stubs.
2. Implement `load_from_doc` with unit tests.
3. Implement `load_from_code` (AST-based) with unit tests, including `MLTable` / `MLVocab` resolver.
4. Implement `diff_schemas` with unit tests.
5. Implement CLI (`main`) and the mismatch output format.
6. Implement `SchemaModel.to_doc_markdown()` helper (bootstrap tool).
7. Use (6) to generate an initial `docs/reference/schema.md`. Hand-edit prose + ordering.
8. Register `deriva-ml-validate-schema` entry point in `pyproject.toml`.
9. Add `docs/reference/README.md` with the doc-first workflow instructions.
10. Add GitHub Actions workflow step.
11. Run integration test (`test_validator_runs_clean_on_current_repo`) — green gate.
12. CHANGELOG entry noting the new doc-first workflow for schema changes.

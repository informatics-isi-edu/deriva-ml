# Execution_Status Vocabulary Implementation Plan (S1b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Execution_Status` as a controlled vocabulary table with an FK from `Execution.Status` → `Execution_Status.Name`, seeded with the 7 canonical lifecycle terms, documented in the S0 schema doc, and protected by an audit test that enforces the library-wide "FKs to vocabulary tables reference Name, not RID" rule.

**Architecture:** Small schema-doc-first change. Single session. No helper wrappers — deriva-py's `Table.define_association` default heuristic already picks `Name` for vocabulary targets (audit of the current schema confirms 100% compliance with zero violations). Two code changes in `create_schema.py` (add vocab table + add FK to Execution), three doc changes in `docs/reference/schema.md` (add Conventions section, add `Execution_Status` section, update `Execution` section), one new seed call in `initialize_ml_schema`, one new test asserting the rule. The S0 validator catches any doc ↔ code drift.

**Tech Stack:** Python 3.12+, deriva-py (Table.define_association, VocabularyTableDef, ForeignKeyDef), pytest, PyYAML (already a dep).

**Parent specs:**
- Phase 1: `docs/superpowers/specs/2026-04-18-sqlite-execution-state-design.md`
- S1a (status-enum migration): `docs/superpowers/specs/2026-04-21-status-enum-reconciliation-design.md`
- S0 (schema-doc source of truth): `docs/superpowers/specs/2026-04-21-schema-doc-source-of-truth-design.md`

---

## Conventions referenced throughout this plan

- **Worktree root:** `/Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab/`. All paths below are relative to it.
- **Environment:** `export PATH="/Users/carl/.local/bin:$PATH"` if `uv` not found. `DERIVA_ML_ALLOW_DIRTY=true` for tests. Live catalog tests need `DERIVA_HOST=localhost`.
- **Doc-first rule (from S0):** every change to `src/deriva_ml/schema/create_schema.py` requires a matching change to `docs/reference/schema.md`. CI validator (`deriva-ml-validate-schema`) enforces.
- **Canonical ExecutionStatus values:** `Created`, `Running`, `Stopped`, `Failed`, `Pending_Upload`, `Uploaded`, `Aborted`. Exact match between the enum in `src/deriva_ml/execution/state_store.py` and the terms seeded in `Execution_Status`.
- **The vocabulary-FK convention:** "In the `deriva-ml` schema, every FK that references a vocabulary table references the vocabulary's `Name` column, not `RID`." Already universally followed by existing FKs (audit confirmed); this plan documents it and adds a test that will catch future violations.

---

## Task Group overview

Four task groups, ordered so each produces a working increment.

| Group | Scope | Tasks |
|---|---|---|
| **A** | Audit test (captures current compliance as an assertion) | 1 task |
| **B** | `Execution_Status` vocab table + FK + seed terms | 3 tasks |
| **C** | Schema-doc updates + Conventions section | 2 tasks |
| **D** | CHANGELOG + final review + finish | 2 tasks |

Total: ~8 tasks.

---

## Task Group A — Audit test

Captures the current "all vocab FKs target Name" state as a running test. If a future contributor adds a vocab FK targeting RID, this test fails.

### Task A1: Vocab-FK audit test

**Files:**
- Create: `tests/schema/test_vocab_fk_convention.py`
- Create: `tests/schema/__init__.py` (if not already present)

- [ ] **Step 1: Ensure `tests/schema/` exists**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && ls tests/schema 2>&1 | head
```

If `tests/schema/` doesn't exist, create it with an empty `__init__.py`:

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && mkdir -p tests/schema && touch tests/schema/__init__.py
```

(Note: several tests/ subdirs already have `__init__.py` files — match that pattern.)

- [ ] **Step 2: Write the failing test**

Create `tests/schema/test_vocab_fk_convention.py`:

```python
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
```

- [ ] **Step 3: Run the test**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_vocab_fk_convention.py -v --timeout=600
```

Expected: PASS. The audit we ran during planning showed zero violations, so the test should pass against the current codebase unmodified. If it fails, investigate before proceeding — something in the audit assumptions is off.

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && git add tests/schema/ && git commit -m "$(cat <<'EOF'
test(schema): vocab-FK-on-Name convention — live-catalog audit

Creates a fresh catalog via create_ml_catalog, introspects the deriva-ml
schema, and asserts every FK targeting a vocabulary table references
Name (not RID or another column).

Captures the current 100%-compliant state as a living assertion.
Catches future violations before they ship.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group B — Execution_Status vocab table + FK + seed

### Task B1: Add Execution_Status VocabularyTableDef to the schema

**Files:**
- Modify: `src/deriva_ml/core/enums.py` — add `execution_status` to `MLVocab` enum
- Modify: `src/deriva_ml/schema/create_schema.py` — add `Execution_Status` vocab table creation inside `create_ml_schema`

- [ ] **Step 1: Add `execution_status` member to MLVocab**

Find the `MLVocab` StrEnum in `src/deriva_ml/core/enums.py` (around line 80):

```python
class MLVocab(StrEnum):
    ...
    dataset_type = "Dataset_Type"
    workflow_type = "Workflow_Type"
    asset_type = "Asset_Type"
    asset_role = "Asset_Role"
    feature_name = "Feature_Name"
```

Add a new member (alphabetically after `asset_role` to match the existing convention for schema-canonical names):

```python
class MLVocab(StrEnum):
    ...
    dataset_type = "Dataset_Type"
    workflow_type = "Workflow_Type"
    asset_type = "Asset_Type"
    asset_role = "Asset_Role"
    execution_status = "Execution_Status"
    feature_name = "Feature_Name"
```

Update the class docstring if it enumerates the members explicitly (check first — at worktree-creation time the docstring may not do this).

- [ ] **Step 2: Verify the MLVocab change is reflected correctly**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from deriva_ml.core.enums import MLVocab
assert MLVocab.execution_status == 'Execution_Status'
print('OK: MLVocab.execution_status =', repr(MLVocab.execution_status.value))
"
```

Expected: `OK: MLVocab.execution_status = 'Execution_Status'`.

- [ ] **Step 3: Add the VocabularyTableDef in create_ml_schema**

Find `create_ml_schema` in `src/deriva_ml/schema/create_schema.py` (around line 274). Locate the block that creates existing vocabulary tables:

```python
    schema.create_table(
        VocabularyTableDef(name=MLVocab.feature_name, curie_template=f"{project_name}:{{RID}}")
    )
    asset_type_table = schema.create_table(
        VocabularyTableDef(name=MLVocab.asset_type, curie_template=f"{project_name}:{{RID}}")
    )
    asset_role_table = schema.create_table(
        VocabularyTableDef(name=MLVocab.asset_role, curie_template=f"{project_name}:{{RID}}")
    )
```

Add a new `Execution_Status` vocab table creation alongside the others. Since it will be referenced by an FK from `Execution.Status`, it must be created BEFORE `create_execution_table` is called. Place the creation immediately after `asset_role_table`:

```python
    schema.create_table(
        VocabularyTableDef(name=MLVocab.feature_name, curie_template=f"{project_name}:{{RID}}")
    )
    asset_type_table = schema.create_table(
        VocabularyTableDef(name=MLVocab.asset_type, curie_template=f"{project_name}:{{RID}}")
    )
    asset_role_table = schema.create_table(
        VocabularyTableDef(name=MLVocab.asset_role, curie_template=f"{project_name}:{{RID}}")
    )
    execution_status_table = schema.create_table(
        VocabularyTableDef(name=MLVocab.execution_status, curie_template=f"{project_name}:{{RID}}")
    )
```

We keep the handle in `execution_status_table` for reference even though the FK is specified via table-name string (not via the Table object), because future code may need the handle.

- [ ] **Step 4: Run the audit test from A1 to verify nothing regressed**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_vocab_fk_convention.py -v --timeout=600
```

Expected: PASS (new Execution_Status table is a vocab table but has NO incoming FKs yet — no violations possible; the A1 audit only checks *outgoing* FKs from non-vocab tables).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && git add src/deriva_ml/core/enums.py src/deriva_ml/schema/create_schema.py && git commit -m "$(cat <<'EOF'
feat(schema): add Execution_Status vocabulary table

New vocabulary table in the deriva-ml schema for controlled Execution
lifecycle status values. Standard VocabularyTableDef shape (Name+ID+URI
+Description+Synonyms). Created alongside the other 5 ML vocabulary
tables.

MLVocab.execution_status = "Execution_Status" added for canonical
access. FK from Execution.Status → Execution_Status.Name and seed
terms come in B2 and B3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B2: Add FK from Execution.Status to Execution_Status.Name

**Files:**
- Modify: `src/deriva_ml/schema/create_schema.py` — extend `create_execution_table`

- [ ] **Step 1: Extend the Execution TableDef with the new FK**

Find `create_execution_table` in `src/deriva_ml/schema/create_schema.py` (around line 131). The current state after S1a:

```python
    execution = schema.create_table(
        TableDef(
            name=MLTable.execution,
            columns=[
                ColumnDef("Workflow", BuiltinType.text),
                ColumnDef("Description", BuiltinType.markdown),
                ColumnDef("Duration", BuiltinType.text),
                ColumnDef("Status", BuiltinType.text),
                ColumnDef("Status_Detail", BuiltinType.text),
            ],
            foreign_keys=[
                ForeignKeyDef(
                    columns=["Workflow"],
                    referenced_schema=schema.name,
                    referenced_table="Workflow",
                    referenced_columns=["RID"],
                )
            ],
            annotations=annotation,
        )
    )
```

Add a second `ForeignKeyDef` inside `foreign_keys=[...]` pointing Status → Execution_Status.Name:

```python
    execution = schema.create_table(
        TableDef(
            name=MLTable.execution,
            columns=[
                ColumnDef("Workflow", BuiltinType.text),
                ColumnDef("Description", BuiltinType.markdown),
                ColumnDef("Duration", BuiltinType.text),
                ColumnDef("Status", BuiltinType.text),
                ColumnDef("Status_Detail", BuiltinType.text),
            ],
            foreign_keys=[
                ForeignKeyDef(
                    columns=["Workflow"],
                    referenced_schema=schema.name,
                    referenced_table="Workflow",
                    referenced_columns=["RID"],
                ),
                ForeignKeyDef(
                    columns=["Status"],
                    referenced_schema=schema.name,
                    referenced_table=MLVocab.execution_status,
                    referenced_columns=["Name"],
                ),
            ],
            annotations=annotation,
        )
    )
```

Notes on the choices:
- `referenced_table=MLVocab.execution_status` (the StrEnum value is `"Execution_Status"`). Prefer the enum reference for consistency with other vocab references in the file.
- `referenced_columns=["Name"]` is the library-wide convention (FK to vocab → Name, not RID).

- [ ] **Step 2: Verify schema creation still works end-to-end**

A fresh catalog must build cleanly. Run a targeted live check:

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run python <<'PYEOF'
"""Smoke: create a catalog, verify Execution.Status FK targets Execution_Status.Name."""
from deriva_ml.schema.create_schema import create_ml_catalog
catalog = create_ml_catalog(hostname="localhost", project_name="s1b_b2_smoke")
try:
    model = catalog.getCatalogModel()
    execution = model.schemas["deriva-ml"].tables["Execution"]
    fks_to_exec_status = [
        fk for fk in execution.foreign_keys
        if fk.referenced_columns[0].table.name == "Execution_Status"
    ]
    assert len(fks_to_exec_status) == 1, f"expected 1 FK to Execution_Status, found {len(fks_to_exec_status)}"
    fk = fks_to_exec_status[0]
    src_cols = [c.name for c in fk.foreign_key_columns]
    tgt_cols = [c.name for c in fk.referenced_columns]
    assert src_cols == ["Status"], f"bad source cols: {src_cols}"
    assert tgt_cols == ["Name"], f"bad target cols: {tgt_cols}"
    print("OK: Execution.Status FK correctly targets Execution_Status.Name")
finally:
    catalog.delete_ermrest_catalog(really=True)
PYEOF
```

Expected: `OK: Execution.Status FK correctly targets Execution_Status.Name`.

- [ ] **Step 3: Run the audit test — it MUST still pass**

The audit test from A1 must still pass because the new FK is itself compliant (targets Name). Running it is a safety check that we didn't break the rule with our own addition.

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_vocab_fk_convention.py -v --timeout=600
```

Expected: PASS.

- [ ] **Step 4: Run the Phase-1 + S1a test suites — no regression**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_status_migration.py tests/execution/test_update_status.py tests/test_migration_complete.py -q --timeout=300
```

Expected: all pass (the enum + update_status API are unchanged by S1b).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && git add src/deriva_ml/schema/create_schema.py && git commit -m "$(cat <<'EOF'
feat(schema): Execution.Status → Execution_Status.Name FK

Constrains Execution.Status to the controlled vocabulary. Per the
library-wide rule, the FK targets the vocabulary's Name column (not
RID), which lets existing ExecutionStatus(row["Status"]) parses keep
working unchanged.

After this change a fresh catalog rejects any Execution insert where
Status isn't one of the 7 canonical terms (seeded in B3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B3: Seed Execution_Status vocabulary with the 7 canonical terms

**Files:**
- Modify: `src/deriva_ml/schema/create_schema.py` — extend `initialize_ml_schema`'s `_ensure_terms` calls

- [ ] **Step 1: Verify current seed calls**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && grep -n "_ensure_terms\|MLVocab\." src/deriva_ml/schema/create_schema.py | head -20
```

Expected hits include:
- `_ensure_terms(MLVocab.asset_type, [...])`
- `_ensure_terms(MLVocab.asset_role, [...])`
- `_ensure_terms(MLVocab.dataset_type, [...])`
- `_ensure_terms(MLVocab.workflow_type, [...])`

Feature_Name is intentionally unseeded (extensible per-project).

- [ ] **Step 2: Add the Execution_Status seed call**

In `initialize_ml_schema`, after the existing `_ensure_terms` calls, add:

```python
    _ensure_terms(MLVocab.execution_status, [
        {"Name": "Created", "Description": "Execution row has been created; work has not started."},
        {"Name": "Running", "Description": "Execution algorithm is actively running."},
        {"Name": "Stopped", "Description": "Algorithm finished successfully; output assets not yet uploaded."},
        {"Name": "Failed", "Description": "Execution encountered an unrecoverable error."},
        {"Name": "Pending_Upload", "Description": "Algorithm succeeded; asset upload to the catalog is in progress."},
        {"Name": "Uploaded", "Description": "Execution ran to success and all outputs are persisted to the catalog."},
        {"Name": "Aborted", "Description": "Execution was canceled by the user before reaching a terminal state."},
    ])
```

- [ ] **Step 3: End-to-end smoke — a fresh Execution can be inserted with Status="Created"**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run python <<'PYEOF'
"""Smoke: after catalog creation, we can insert an Execution row with Status='Created'."""
from deriva_ml.schema.create_schema import create_ml_catalog
catalog = create_ml_catalog(hostname="localhost", project_name="s1b_b3_smoke")
try:
    pb = catalog.getPathBuilder()
    exec_table = pb.schemas["deriva-ml"].tables["Execution"]
    # Minimal Execution insert referencing the new Execution_Status.Name FK.
    result = exec_table.insert([
        {"Description": "b3 smoke", "Workflow": None, "Status": "Created"}
    ])
    rid = result[0]["RID"]
    print(f"OK: inserted Execution {rid} with Status='Created'")
    # And invalid status SHOULD be rejected.
    try:
        exec_table.insert([
            {"Description": "b3 bogus", "Workflow": None, "Status": "NotAValidStatus"}
        ])
        print("FAIL: invalid Status was accepted (expected FK rejection)")
    except Exception as exc:
        print(f"OK: invalid Status rejected ({type(exc).__name__})")
finally:
    catalog.delete_ermrest_catalog(really=True)
PYEOF
```

Expected output:
```
OK: inserted Execution <RID> with Status='Created'
OK: invalid Status rejected (<SomeException>)
```

If the invalid-Status insert succeeds, the FK isn't being enforced — investigate (ERMrest may need FK enforcement enabled, or the schema may need tweaks).

- [ ] **Step 4: Run the audit + ExecutionStatus tests**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_vocab_fk_convention.py tests/execution/test_status_migration.py tests/execution/test_update_status.py -q --timeout=600
```

Expected: all pass.

- [ ] **Step 5: Run the broader test suite to catch any indirect breakage**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q --timeout=600
```

Expected: 259+ pass (the S1a baseline plus anything new). No regressions.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && git add src/deriva_ml/schema/create_schema.py && git commit -m "$(cat <<'EOF'
feat(schema): seed Execution_Status vocabulary with 7 canonical terms

Seeds Created, Running, Stopped, Failed, Pending_Upload, Uploaded,
Aborted — exactly matching the ExecutionStatus StrEnum from S1a.

After seeding + the B2 FK, any Execution insert with a Status value
outside this canonical set is rejected by the catalog's FK constraint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group C — Schema-doc updates

The S0 validator requires `docs/reference/schema.md` to agree with `src/deriva_ml/schema/create_schema.py`. After Group B, running the validator will report mismatches. Group C resolves them.

### Task C1: Add Conventions section + Execution_Status doc entry; update Execution entry

**Files:**
- Modify: `docs/reference/schema.md`

- [ ] **Step 1: Run the validator to see what's missing**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-validate-schema 2>&1 | head -40
```

Expected output: something like

```
deriva-ml-validate-schema: schema.md and create_schema.py disagree.

MISSING FROM CODE:
  (none)

EXTRA IN CODE:
  - table 'Execution_Status' is in the code but not in the doc.

COLUMN MISMATCH:
  (none)

FOREIGN KEY MISMATCH:
  - FK on ['Status'] in code but not in doc
...
```

Exit code: 1.

- [ ] **Step 2: Add the Conventions section at the top of schema.md**

Near the top of `docs/reference/schema.md` — right after the "What is and isn't validated" section — add a new section:

```markdown
## Conventions

### Foreign keys to vocabulary tables reference `Name`, not `RID`

In the `deriva-ml` schema, every FK that references a vocabulary table
references the vocabulary's `Name` column — never `RID` or another column.
This applies to both explicit `ForeignKeyDef` call sites (e.g.
`Execution.Status` → `Execution_Status.Name`) and to FKs generated by
`Table.define_association` when one of the endpoints is a vocabulary
table (e.g. `Dataset_Dataset_Type(Dataset_Type)` → `Dataset_Type.Name`).

**Rationale.** A `Name`-valued FK stores a human-readable string, not an
opaque RID. Raw catalog exports are self-describing; `EnumClass(row["Col"])`
parses directly; migration from a plain-text column to an FK'd column
requires no data conversion.

**Enforcement.** `tests/schema/test_vocab_fk_convention.py` creates a fresh
catalog via `create_ml_catalog`, introspects every FK in the `deriva-ml`
schema, and fails if any vocabulary-targeting FK references a column
other than `Name`. Deriva-py's `Table.define_association` default
heuristic already picks `Name` for vocabulary endpoints; this convention
makes that implicit behavior explicit and enforces it for new explicit
FKs.
```

- [ ] **Step 3: Add the Execution_Status section**

In `docs/reference/schema.md`, find the **Vocabularies** group (other vocab sections: Dataset_Type, Workflow_Type, Feature_Name, Asset_Type, Asset_Role). Add a new section between Asset_Role and Asset_Type (alphabetical ordering isn't strict but lifecycle-ordering puts Execution_Status near Execution-related tables):

~~~markdown
## Execution_Status

Controlled vocabulary for the `Execution.Status` column — the lifecycle state of an `Execution` row. Seeded with the 7 canonical values matching the `ExecutionStatus` StrEnum in `src/deriva_ml/execution/state_store.py`. New terms should NOT be added without updating that enum in tandem.

```yaml
table: Execution_Status
kind: vocabulary
description: Controlled vocabulary for Execution lifecycle state.
terms:
  - name: Created
  - name: Running
  - name: Stopped
  - name: Failed
  - name: Pending_Upload
  - name: Uploaded
  - name: Aborted
```
~~~

Also update the top-of-doc Table of Contents to include `Execution_Status` under Vocabularies. Example (current doc likely has a ToC list):

```markdown
**Vocabularies**: [Dataset_Type](#dataset_type), [Workflow_Type](#workflow_type), [Feature_Name](#feature_name), [Asset_Type](#asset_type), [Asset_Role](#asset_role), [Execution_Status](#execution_status)
```

- [ ] **Step 4: Update the Execution section to include the new FK**

Find the existing `## Execution` section in `docs/reference/schema.md`. Its YAML block currently has one FK:

~~~yaml
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
  type: text
- name: Status_Detail
  type: text
foreign_keys:
- columns: [Workflow]
  referenced_schema: deriva-ml
  referenced_table: Workflow
  referenced_columns: [RID]
~~~

Add a second FK entry to `foreign_keys` for Status → Execution_Status.Name:

~~~yaml
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
  type: text
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
~~~

Also update the prose above the YAML block to mention the new FK briefly, e.g.:

```markdown
## Execution

Per-execution lifecycle row. The `Status` column is FK-constrained to the
`Execution_Status` vocabulary (7 canonical lifecycle values, seeded at
schema creation); invalid status writes are rejected by the catalog.
```

- [ ] **Step 5: Re-run the validator**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-validate-schema 2>&1 | tail -5
```

Expected: `deriva-ml-validate-schema: schema.md and create_schema.py agree.` Exit code 0.

If it reports mismatches, the YAML edits don't match the code. Common slips:
- Missing `referenced_columns: [Name]` → validator reports FK mismatch
- Term list order differs from the seed order → validator reports VOCAB_TERMS_MISMATCH (term order shouldn't matter to the validator, but if ordering matters, match the seed order literally)

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && git add docs/reference/schema.md && git commit -m "$(cat <<'EOF'
docs(reference): Execution_Status vocab + Conventions section

- New ## Conventions section documents the library-wide rule: FKs
  to vocabulary tables reference Name, not RID. Explains rationale
  (self-describing exports, direct enum parse, simpler migration)
  and points at the audit test for enforcement.
- New ## Execution_Status section describes the vocabulary (kind,
  description, 7 canonical terms).
- Updated ## Execution section to include the new FK
  (Status → Execution_Status.Name) in its foreign_keys list.
- Updated table of contents to include Execution_Status.

Validator reports doc and code agree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C2: Verify S0 structure tests still pass

**Files:** none (verification)

- [ ] **Step 1: Run the S0 structure-check test**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_schema_doc_structure.py -v
```

Expected: all pass. This test iterates over every member of `MLTable` and `MLVocab` and asserts a matching doc section exists. Because we added `MLVocab.execution_status` in B1 and a doc section in C1, the test should cleanly pick up the new entry.

- [ ] **Step 2: Run the S0 integration test (full doc-vs-code comparison)**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/test_validate_schema_doc_integration.py -v
```

Expected: all pass.

- [ ] **Step 3: No commit** (verification only).

---

## Task Group D — CHANGELOG + final review

### Task D1: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add S1b entry at the top of CHANGELOG.md**

Insert immediately after `# Changelog ...` intro, above the S1a entry:

```markdown
## Unreleased — Phase 2 Subsystem 1b: Execution_Status vocabulary

### New

- **`Execution_Status` vocabulary table** in the `deriva-ml` schema. Seeded with 7 canonical terms (Created, Running, Stopped, Failed, Pending_Upload, Uploaded, Aborted) matching the `ExecutionStatus` StrEnum from S1a.
- **FK on `Execution.Status` → `Execution_Status.Name`.** Catalog now rejects any `Execution` insert whose `Status` value isn't one of the seeded canonical terms.
- **`MLVocab.execution_status`** enum member for canonical access to the new vocabulary-table name.
- **`tests/schema/test_vocab_fk_convention.py`** — live-catalog audit test that asserts every FK targeting a vocabulary table in the `deriva-ml` schema references the `Name` column.
- **Conventions section in `docs/reference/schema.md`** documenting the vocabulary-FK-on-Name rule, its rationale, and the enforcing test.

### Audit notes

- Audit of the pre-S1b schema found **zero violations** of the Name-not-RID rule. Deriva-py's `Table.define_association` default `key_column_search_order = ['Name', 'name', 'ID', 'id']` already picks `Name` for vocabulary targets and falls back to `RID` only for entity targets. The new convention makes this implicit behavior explicit and test-enforced.

### Migration notes

- Existing deployed catalogs MAY need a one-time migration to add the `Execution_Status` vocabulary table + FK if they were created before S1b. A separate cleanup task (outside S1b scope) verifies compliance on any live deployments. New catalogs created via `create_ml_catalog` have everything set up.
- Historical `Execution` rows whose `Status` value was `"Initializing"` or `"Pending"` (pre-S1a) will violate the new FK constraint if the catalog is migrated to include the FK. Users should clean those rows via `gc_executions` or manual updates.

---
```

- [ ] **Step 2: Commit**

```bash
cd /Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1b-execution-status-vocab && git add CHANGELOG.md && git commit -m "$(cat <<'EOF'
docs(changelog): Phase 2 Subsystem 1b — Execution_Status vocabulary

Documents the new vocab table, the FK constraint on Execution.Status,
the Name-not-RID rule + audit test, and the zero-violation audit
finding on the pre-S1b schema.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D2: Final review + finish branch

**Files:** none (process task)

- [ ] **Step 1: Dispatch a `superpowers:code-reviewer` subagent**

Per `superpowers:subagent-driven-development`, review the full S1b delivery (commits from Task A1 through D1):

```
Subagent: superpowers:code-reviewer
Task: review Phase 2 Subsystem 1b — Execution_Status vocabulary + FK +
convention audit — against the S1b plan
(docs/superpowers/plans/2026-04-21-execution-status-vocab.md).

Check: every task commit corresponds to a plan task; vocab-FK convention
documented in schema.md; FK actually enforces at ERMrest layer (B3 smoke
should have confirmed); S0 validator agrees.

Report blocker / important / nit items. Implementer addresses blockers
and important issues before merge.
```

- [ ] **Step 2: Address reviewer findings**

Fix any blocker/important items via targeted edits. Re-run the full test suite plus the validator after fixes.

- [ ] **Step 3: Finish branch**

Once approved, invoke `superpowers:finishing-a-development-branch` to push, open PR, merge, and clean up.

---

*(End of S1b.)*

---

## Post-S1b

After S1b merges to `origin/main`, remaining Phase 2 work (per CHANGELOG tracking):

1. **One-time migration check** for the live deployed catalog: verify it has `Execution_Status` + the FK + the seeded terms; if not, run `initialize_ml_schema` against it (idempotent for the seed; the vocab table + FK require `create_execution_table` / model edits).
2. **Subsystem 3** — Upload engine completions: real `_invoke_deriva_py_uploader`, `parallel_files` / `bandwidth_limit_mbps` plumbing, `UploadJob` real cancellation.
3. **Subsystem 4** — Hygiene bundle: offline-mode `DerivaML.__init__` network-skip, distutils-shim test flake, docstring misc.
4. **Subsystem 2** — Feature-consistency follow-on ("the real Phase 2"). Requires its own brainstorming session.

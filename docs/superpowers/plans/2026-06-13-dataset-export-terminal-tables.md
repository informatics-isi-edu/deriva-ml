# Dataset-Export Terminal Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the dataset-bag export from walking the catalog's entire provenance DAG (an 18-minute hang on large nested datasets) by making `DatasetBagBuilder.build_policy` mark `Execution`/`Workflow` as terminal tables — the same protection `clone_via_bag` already has.

**Architecture:** Extract `{Execution, Workflow}` into a shared constant in `core/constants.py` so the two bag-producing paths can't diverge, then consume it in both `build_policy` and `clone_via_bag`. The walker enters terminal tables (keeping the provenance link) but does not follow their FKs outward, severing the explosion.

**Tech Stack:** Python 3.12+, deriva-py `FKTraversalPolicy`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-dataset-export-terminal-tables-design.md`
**Branch:** `fix/dataset-export-terminal-tables` (already created off `main`; spec commit is on it).

---

## Context for the implementer (read first)

- **CWD discipline:** chain `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>` in ONE Bash call every time. Tests: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest ...`.
- **The bug (verified):** `bag_builder.build_policy` returns an `FKTraversalPolicy` with **no `terminal_tables`**, so the `CatalogBagBuilder` walk enters `Execution` (a hub with 21 inbound FKs) and fans out across the whole catalog provenance graph. `clone_via_bag` already sets `terminal_tables={("deriva-ml","Execution"),("deriva-ml","Workflow")}`; `build_policy` must do the same.
- **Existing shared-constant pattern:** `INTENTIONAL_FK_CYCLES` lives in `src/deriva_ml/core/constants.py` and is imported by BOTH `bag_builder.py:64` and `clone_via_bag.py:76`. The new terminal-tables constant follows this exact pattern (same file, same dual-import).
- **`ML_SCHEMA = "deriva-ml"`** is defined at `core/constants.py:44` — use it to build the constant rather than hard-coding the string.
- **Current sites:**
  - `core/constants.py` — `INTENTIONAL_FK_CYCLES` at line 75, `__all__` starts line 80.
  - `bag_builder.py` — `build_policy` at line 756, its `return FKTraversalPolicy(...)` at line 804 (no `terminal_tables` arg today).
  - `clone_via_bag.py` — `default_terminal_tables` literal at line 360, used at lines 367 and 410.
- **Tests:** `tests/dataset/test_bag_builder.py` (`TestAnchorsAndPolicy`-style class, uses the live `catalog_with_datasets` fixture; `test_build_policy_default` at line 147 is the sibling to extend). `tests/catalog/test_clone_via_bag.py` already asserts terminal_tables merge (lines 304-305) — those must still pass after the constant extraction.

### File map

| File | Action | Responsibility |
|---|---|---|
| `src/deriva_ml/core/constants.py` | modify | define `PROVENANCE_TERMINAL_TABLES` + add to `__all__` |
| `src/deriva_ml/dataset/bag_builder.py` | modify | import it; pass `terminal_tables=` in `build_policy` |
| `src/deriva_ml/catalog/clone_via_bag.py` | modify | replace local `default_terminal_tables` literal with the import |
| `tests/dataset/test_bag_builder.py` | modify | assert `build_policy` sets the terminal tables + walk excludes Execution_Asset |
| `tests/core/test_constants.py` (or existing) | modify/create | pin the constant's value + dual-import no-divergence |

---

## Task 1: Shared constant in `core/constants.py`

**Files:**
- Modify: `src/deriva_ml/core/constants.py` (after `INTENTIONAL_FK_CYCLES`, ~line 77; `__all__` ~line 80)
- Test: `tests/core/test_constants.py`

- [ ] **Step 1: Write the failing test**

Check whether `tests/core/test_constants.py` exists. If it does, append; if not, create it with this content:

```python
"""Tests for shared FK-traversal constants."""

from __future__ import annotations


def test_provenance_terminal_tables_value():
    from deriva_ml.core.constants import ML_SCHEMA, PROVENANCE_TERMINAL_TABLES

    assert PROVENANCE_TERMINAL_TABLES == frozenset(
        {(ML_SCHEMA, "Execution"), (ML_SCHEMA, "Workflow")}
    )


def test_provenance_terminal_tables_exported():
    import deriva_ml.core.constants as c

    assert "PROVENANCE_TERMINAL_TABLES" in c.__all__
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_constants.py::test_provenance_terminal_tables_value -v 2>&1 | tail -5
```
Expected: FAIL — `ImportError: cannot import name 'PROVENANCE_TERMINAL_TABLES'`.

- [ ] **Step 3: Add the constant**

In `src/deriva_ml/core/constants.py`, immediately after the
`INTENTIONAL_FK_CYCLES` definition (the closing `})` near line 77),
add:

```python
# Provenance tables the dataset-bag walk ENTERS but does not traverse
# outward. ``Execution`` and ``Workflow`` describe *how* rows came to
# be, not *what they are*; one Execution aggregates state across every
# Subject/Image/Dataset its run touched and has many inbound FKs (every
# ``*_Execution`` association, plus self-loops via ``Execution_Execution``
# and back-edges to feature tables like ``Annotation``). Following those
# inbound FKs makes the walk fan out across the entire catalog provenance
# graph — an 18-minute hang on large nested datasets (eye-ai 2-277G).
# Marking them terminal keeps the provenance *link* in the bag while
# severing the fan-out. Shared by ``DatasetBagBuilder.build_policy`` and
# ``clone_via_bag`` so the two bag-producing paths cannot diverge.
PROVENANCE_TERMINAL_TABLES: frozenset[tuple[str, str]] = frozenset(
    {
        (ML_SCHEMA, "Execution"),
        (ML_SCHEMA, "Workflow"),
    }
)
```

Then add `"PROVENANCE_TERMINAL_TABLES",` to the `__all__` list (next to
`"INTENTIONAL_FK_CYCLES",`).

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_constants.py -v 2>&1 | tail -6
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/core/constants.py tests/core/test_constants.py && \
git commit -m "feat(constants): shared PROVENANCE_TERMINAL_TABLES {Execution, Workflow}"
```

---

## Task 2: Apply terminal tables in `build_policy` (the fix)

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (import line ~64; `build_policy` return ~804)
- Test: `tests/dataset/test_bag_builder.py`

- [ ] **Step 1: Write the failing test**

In `tests/dataset/test_bag_builder.py`, add this test to the class that
holds `test_build_policy_default` (the one using the
`catalog_with_datasets` fixture — around line 147):

```python
    def test_build_policy_marks_execution_workflow_terminal(
        self, catalog_with_datasets
    ) -> None:
        """Execution and Workflow are terminal so the walk doesn't fan
        out across the catalog provenance graph (the 2-277G 18-min hang)."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        builder = DatasetBagBuilder(ml_instance=ml)
        policy = builder.build_policy(dataset)
        assert ("deriva-ml", "Execution") in policy.terminal_tables
        assert ("deriva-ml", "Workflow") in policy.terminal_tables
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  "tests/dataset/test_bag_builder.py::TestAnchorsAndPolicy::test_build_policy_marks_execution_workflow_terminal" -v 2>&1 | tail -8
```
(If the class name differs, run the whole file and find the new test.)
Expected: FAIL — `terminal_tables` is empty (assertion error), because
`build_policy` doesn't set it yet. Requires the local catalog stack
(`catalog_with_datasets` is a live fixture).

- [ ] **Step 3: Implement the fix**

In `src/deriva_ml/dataset/bag_builder.py`:

(a) Extend the existing constants import at line 64:

```python
from deriva_ml.core.constants import INTENTIONAL_FK_CYCLES, PROVENANCE_TERMINAL_TABLES, RID
```

(b) In `build_policy`, change the `return FKTraversalPolicy(...)` (line
804) to pass `terminal_tables`:

```python
        return FKTraversalPolicy(
            exclude_tables=exclude_tables,
            vocab_export=vocab_export,
            # Provenance tables are entered but not traversed outward —
            # otherwise the walk fans out across the whole catalog
            # provenance graph (see core/constants.py
            # :PROVENANCE_TERMINAL_TABLES). Same protection clone_via_bag
            # applies.
            terminal_tables=set(PROVENANCE_TERMINAL_TABLES),
            # Silence WARNING-level "Breaking cycle in FK
            # dependencies" log spam for the known
            # Dataset ↔ Dataset_Version cycle. See
            # core/constants.py:INTENTIONAL_FK_CYCLES.
            intentional_cycles=set(INTENTIONAL_FK_CYCLES),
        )
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  tests/dataset/test_bag_builder.py -v --timeout=600 2>&1 | tail -12
```
Expected: all pass, including the new test. (Requires localhost catalog.)

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/dataset/bag_builder.py tests/dataset/test_bag_builder.py && \
git commit -m "fix(dataset): mark Execution/Workflow terminal in bag-export walk

The export walk entered Execution (a provenance hub with 21 inbound
FKs) and fanned out across the whole catalog provenance graph — an
18-minute hang on large nested datasets (eye-ai 2-277G/5-WEBG). Mark
the provenance tables terminal so the walk keeps the provenance link
but stops following outward. Mirrors clone_via_bag."
```

---

## Task 3: De-duplicate `clone_via_bag` onto the shared constant

**Files:**
- Modify: `src/deriva_ml/catalog/clone_via_bag.py` (import ~76; literal ~360, uses ~367/410)
- Test: `tests/catalog/test_clone_via_bag.py` (existing assertions must still pass)

- [ ] **Step 1: Write the no-divergence test**

Append to `tests/core/test_constants.py`:

```python
def test_clone_and_bag_builder_share_terminal_tables():
    """clone_via_bag and bag_builder must use the SAME terminal set —
    a divergence is what let the dataset-export path miss this guard."""
    from deriva_ml.catalog import clone_via_bag as cvb
    from deriva_ml.core.constants import PROVENANCE_TERMINAL_TABLES

    # clone_via_bag builds its default terminal set from the constant.
    assert cvb._DEFAULT_TERMINAL_TABLES == set(PROVENANCE_TERMINAL_TABLES)
```

(Note: this references `clone_via_bag._DEFAULT_TERMINAL_TABLES`, a
module-level name introduced in Step 3 below.)

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_constants.py::test_clone_and_bag_builder_share_terminal_tables -v 2>&1 | tail -5
```
Expected: FAIL — `module 'deriva_ml.catalog.clone_via_bag' has no attribute '_DEFAULT_TERMINAL_TABLES'`.

- [ ] **Step 3: Replace the local literal with the shared constant**

In `src/deriva_ml/catalog/clone_via_bag.py`:

(a) Extend the constants import at line 76:

```python
from deriva_ml.core.constants import INTENTIONAL_FK_CYCLES, PROVENANCE_TERMINAL_TABLES
```

(b) Read the current `default_terminal_tables` block (line 360). It is
a local variable inside a function. Replace the inline literal:

```python
    default_terminal_tables: set[tuple[str, str]] = {
        ("deriva-ml", "Execution"),
        ("deriva-ml", "Workflow"),
    }
```

with a reference to a module-level name built from the constant. Add
this near the top of the module (after imports, module scope):

```python
# Module-level alias so tests can assert clone_via_bag and bag_builder
# share one terminal set (core/constants.py:PROVENANCE_TERMINAL_TABLES).
_DEFAULT_TERMINAL_TABLES: set[tuple[str, str]] = set(PROVENANCE_TERMINAL_TABLES)
```

and replace the in-function literal with:

```python
    default_terminal_tables = set(_DEFAULT_TERMINAL_TABLES)
```

(keep the existing comment block above it). The two downstream uses at
lines 367 and 410 (`terminal_tables=default_terminal_tables` /
`merge_kwargs["terminal_tables"] = default_terminal_tables`) are
unchanged — they still reference the local `default_terminal_tables`.

- [ ] **Step 4: Run to verify it passes (incl. existing clone tests)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest \
  tests/core/test_constants.py tests/catalog/test_clone_via_bag.py -v 2>&1 | tail -12
```
Expected: all pass — the no-divergence test AND the pre-existing
`test_clone_via_bag.py` assertions (lines 304-305 checking
`("deriva-ml","Execution")`/`("deriva-ml","Workflow")` are still in the
merged policy). If any clone test fails, the literal→constant swap
changed a value: stop and reconcile (the values must be identical).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/catalog/clone_via_bag.py tests/core/test_constants.py && \
git commit -m "refactor(clone): clone_via_bag uses shared PROVENANCE_TERMINAL_TABLES

De-duplicates the {Execution, Workflow} terminal set onto the shared
constant so it can't diverge from bag_builder's copy (the divergence
is what let the dataset-export path miss this guard)."
```

---

## Task 4: Walk-scope regression test (the explosion arm is severed)

**Files:**
- Test: `tests/dataset/test_bag_builder.py`

This pins the *behavioral* fix (not just the policy field): with
Execution terminal, the reached-table set excludes the
`Execution_Asset` explosion arm while keeping `Execution` and the
member/feature tables. Uses the live `catalog_with_datasets` fixture
(the demo catalog has executions + assets, so the arm is reachable but
tiny).

- [ ] **Step 1: Write the test**

Add to the same test class in `tests/dataset/test_bag_builder.py`:

```python
    def test_walk_excludes_execution_asset_closure(
        self, catalog_with_datasets
    ) -> None:
        """With Execution terminal, aggregate_queries reaches Execution
        (provenance link) and member/feature tables, but NOT the
        Execution_Asset closure the inbound fan-out used to pull in."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        reached = set(DatasetBagBuilder(ml_instance=ml).aggregate_queries(dataset))

        # Provenance link kept:
        assert "Execution" in reached
        # Explosion arm severed (these are reached ONLY by following
        # Execution's inbound FKs outward, which terminal_tables blocks):
        assert "Execution_Asset" not in reached
        assert "Execution_Asset_Execution" not in reached
```

- [ ] **Step 2: Run to verify it passes (this is the guard, not a red→green pair)**

The fix from Task 2 already makes this pass — it's a behavioral
regression guard. Run it to confirm the fix produces the claimed walk
shape:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  tests/dataset/test_bag_builder.py -k "walk_excludes_execution_asset" -v --timeout=600 2>&1 | tail -8
```
Expected: PASS. If `Execution_Asset` IS still reached, the fix is
incomplete (some other inbound path reaches it) — stop and
re-investigate; do NOT weaken the assertion.

> Sanity note for the implementer: to prove this test has teeth, you
> may temporarily revert Task 2's `terminal_tables=` line, confirm the
> test FAILS (Execution_Asset reached), then re-apply. Don't commit the
> revert.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add tests/dataset/test_bag_builder.py && \
git commit -m "test(dataset): pin that the Execution_Asset closure is severed from the walk"
```

---

## Task 5: Lint, full bag/clone suite, PR

**Files:** none new.

- [ ] **Step 1: Lint + format the touched files**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
uv run ruff check src/deriva_ml/core/constants.py src/deriva_ml/dataset/bag_builder.py src/deriva_ml/catalog/clone_via_bag.py tests/core/test_constants.py tests/dataset/test_bag_builder.py && \
uv run ruff format src/deriva_ml/core/constants.py tests/core/test_constants.py 2>&1 | tail -2
```
Expected: ruff clean (pre-existing unrelated findings elsewhere are out
of scope — only the touched files must be clean).

- [ ] **Step 2: Run the affected suites together**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  tests/core/test_constants.py tests/dataset/test_bag_builder.py tests/catalog/test_clone_via_bag.py \
  -q --timeout=600 2>&1 | tail -6
```
Expected: all pass. (Catalog-dependent tests need the localhost stack;
if it's down, note it and run at least `tests/core/test_constants.py`
which is catalog-free.)

- [ ] **Step 3: Commit any format changes, push, open PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add -A src tests && (git diff --cached --quiet || git commit -m "style: ruff format") ; \
git push -u origin fix/dataset-export-terminal-tables && \
gh pr create --title "fix(dataset): mark Execution/Workflow terminal in bag-export walk" --body "$(cat <<'EOF'
## Problem

Downloading a dataset bag for a large nested dataset (eye-ai \`2-277G\`, a 50-dataset nested pool) never completes bag formation. The export's deepest query — \`Dataset → Dataset_Image → Image → Annotation → Execution → Execution_Asset_Execution → Execution_Asset\` — ran **18 minutes** before a ConnectionError on a single large child (\`5-WEBG\`, 9,511 images).

## Root cause (verified)

\`DatasetBagBuilder.build_policy\` set **no \`terminal_tables\`**, so the walk entered \`Execution\` — a provenance hub with **21 inbound FKs** (every \`*_Execution\` association, a self-loop via \`Execution_Execution\`, a back-edge to \`Annotation\`) — and fanned out across the whole catalog provenance DAG. Hop-by-hop on a 20-image dataset: \`Annotation\` (100) → \`Execution\` (1 distinct) → \`Execution_Asset_Execution\` (**9,066** — ×1,800 blow-up, every asset of the producing execution). \`clone_via_bag\` already guards this with \`terminal_tables={Execution, Workflow}\`; \`build_policy\` simply omitted it.

## Fix

- Extract \`{Execution, Workflow}\` into \`core/constants.py:PROVENANCE_TERMINAL_TABLES\` (shared so the two bag-producing paths can't diverge again — the divergence is what caused this).
- \`build_policy\` passes \`terminal_tables=\`; \`clone_via_bag\` consumes the same constant.

## Validation

In-process patch against \`5-WEBG\` (the 18-minute killer): \`aggregate_queries\` built in **1.9 s**; \`Execution_Asset\` / \`Execution_Asset_Execution\` no longer reached; \`Execution\` and \`Annotation\`/\`Dataset_Image\` still reached. Tests pin the policy field, the no-divergence invariant, and the severed-walk behavior.

## Not in scope

deriva-py query-staging/batching (considered and rejected — it would have efficiently fetched a catalog-sized *wrong* result; the problem was traversal scope, not query mechanics). Per-export-configurable terminal tables (YAGNI).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" 2>&1 | tail -2
```

---

## Self-review

- **Spec coverage:** §2 fix → Task 2; §3 shared-constant extraction → Tasks 1+3; §6 testing — policy-field test (Task 2), no-divergence test (Task 3), walk-scope test (Task 4), existing clone suite kept green (Task 3/5); §7 non-goals respected (no deriva-py changes, no configurability). All spec sections map to a task.
- **Placeholder scan:** none — every code block is complete; the one "if the class name differs" note is a real fallback, not a TBD.
- **Type/name consistency:** `PROVENANCE_TERMINAL_TABLES` (frozenset) is defined in Task 1 and consumed identically in Tasks 2/3; `clone_via_bag._DEFAULT_TERMINAL_TABLES` (set) is introduced in Task 3 Step 3 and asserted in Task 3 Step 1 — names match. `set(PROVENANCE_TERMINAL_TABLES)` is used everywhere a mutable set is needed (the policy field and clone's local var), consistent.
- **Branch-first:** the branch already exists with the spec commit; all task commits land on it; Task 5 pushes + opens the PR. Nothing on `main`.
- **Catalog dependency:** Tasks 2 and 4's behavioral tests need the localhost stack (live `catalog_with_datasets` fixture); Tasks 1 and 3's constant tests are catalog-free. If the stack is down, the constant + policy-field-shape work still proceeds; flag the deferred live tests in the PR.

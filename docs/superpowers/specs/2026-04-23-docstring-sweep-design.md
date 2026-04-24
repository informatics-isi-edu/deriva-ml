# Docstring Sweep Design — Sub-project 2 of the Documentation Pass

**Date:** 2026-04-23
**Status:** Draft — awaiting Carl's review
**Branch:** `feature/docstring-sweep`
**Related:** Sub-project 1 (user-guide rewrite, separate worktree)
**Source material:** `docs/superpowers/specs/2026-04-23-post-s2-findings.md` §1 (Reviewer #2) and §3–4 (Reviewer #4)

---

## 1. Problem Statement

The post-S2 audit (Reviewer #2) found docstring coverage at roughly **60–70%** against the project's stated bar. The weak areas are older modules that predate the spec-driven standard: `DatasetMixin`, `AnnotationMixin`, `Feature`, and `DatasetBag` each have methods with missing `Raises:`, `Example:`, or both. Several modules lack module-level docstrings entirely. Five non-trivial machinery sites have no "why" comment, making them opaque to contributors and to AI assistants reading the source.

Concurrently, Reviewer #4 identified 12 public-named symbols that should be private (`_prefixed`) plus 7 dead symbols that should be deleted. Private renames touch the same files and methods as docstring edits; doing them together in one sweep is cheaper than two separate passes and keeps the diff reviewable per module.

The result of this sub-project is a library where every public API has a complete docstring, every module has a meaningful module-level docstring, and the public namespace contains only intentional user-facing names.

---

## 2. Goals

1. Every public method in `src/deriva_ml/` satisfies the **docstring contract** defined in §4.
2. Every `.py` file in `src/deriva_ml/` has a **module-level docstring** (§5).
3. Five flagged **inline comment gaps** receive "why" comments (§6).
4. All 12 **leaked names** from Reviewer #4's confirmed list are `_prefixed` (§7).
5. All 7 **dead symbols** from Reviewer #4 are deleted (or explicitly deferred — see §8 open question).
6. Changes are delivered **module by module**, one commit per module, so each commit is independently reviewable and can be bisected.

---

## 3. Non-Goals

- **No behavior changes.** Docstrings, renames, and dead-code deletion only.
- **No changes to `src/deriva_ml/feature.py` S2-touched surface.** The selector classmethod suite (`Feature.for_table`, `Feature.for_dataset`, etc.) is already at spec bar from S2 work. This sweep adds only the missing `Feature.__init__` docstring and one inline comment.
- **No changes to `docs/`.** All prose improvements for human readers belong in sub-project 1 (user-guide rewrite). This sweep is reference material — docstrings consumed by IDE hover, `help()`, and MCP tool indexing.
- **No new tests beyond doctest verification.** Test file changes are limited to updating any test that references a renamed private symbol.
- **No DRY refactoring.** Reviewer #3's DRY findings are a separate hygiene task. The only structural changes here are renames and deletions.

---

## 4. Docstring Contract

Every public method, property, and class in `src/deriva_ml/` must conform to this shape:

```python
def method_name(self, param: Type) -> ReturnType:
    """One-line imperative summary (≤ 72 chars, no trailing period).

    Extended description explaining intent, non-obvious behavior, and
    important invariants. Omit this block only for trivially-named
    no-argument methods where the one-liner is exhaustive.

    Args:
        param: Type and semantic meaning. Document every parameter
            including ``self`` is implied. For optional parameters note
            the default behavior when omitted.

    Returns:
        Type and semantic meaning. Describe the shape for complex
        return types (e.g., dict keys, list element type).

    Raises:
        DerivaMLNotFoundError: When ``param`` does not resolve to a
            known catalog entity.
        DerivaMLValidationError: When the supplied value fails schema
            validation.

    Example:
        >>> ml = DerivaML(hostname="example.org", catalog_id="42")
        >>> result = ml.method_name(param)
    """
```

**Mandatory sections:** one-line summary, `Args:` (even if empty — omit only for zero-parameter methods), `Returns:` (omit only for `-> None` methods with no meaningful return), `Raises:` (omit only if the method genuinely raises nothing), `Example:` (at least one).

**Exemptions from `Example:`:**

- Abstract methods and protocol stubs (they document the contract, not an invocation).
- Pure `@property` getters where the getter is self-documenting and the class already has an example that exercises it.

**Module docstring shape:**

```python
"""Short noun phrase describing what this module provides.

Extended description covering the key classes/functions exported, the
design rationale, and any important usage constraints (e.g., "not
intended for direct import — access via DerivaML methods").
"""
```

---

## 5. Module-Level Docstrings

Modules confirmed to need new or expanded module docstrings:

| Module | Current state | Action |
|---|---|---|
| `catalog/localize.py` | One-liner only | Expand: explain three-stage clone flow, asset copy semantics |
| `feature.py` | Names Feature + FeatureRecord only | Add: document selector classmethod suite |

All other modules must be checked during the sweep. Any module with no docstring at all gets one.

---

## 6. Inline Comment Gaps

Five machinery sites identified by Reviewer #2 that need "why" comments:

| # | Location | Comment to add |
|---|---|---|
| 1 | `core/mixins/dataset.py:217–236` | Why workspace ORM is rebuilt after `add_dataset_element_type` (ORM eagerly built at init; new DDL not visible without rebuild) |
| 2 | `dataset/dataset.py:1580–1610` | Why the two-step INSERT + GET `snaptime` pattern (ERMrest does not return snaptime on INSERT) |
| 3 | `execution/execution.py:322–361` | Why `not self._dry_run and reload is None` gates SQLite registry insertion |
| 4 | `dataset/dataset_bag.py:276–315` | Why `union(*)` is load-bearing (SQLAlchemy UNION is DISTINCT by default; de-duplicates rows reachable via multiple FK paths) |
| 5 | `feature.py:448–463` | Why `assoc_fkeys` is subtracted before role classification |

---

## 7. Private-Naming Convention — The Rename List

All 12 items from Reviewer #4's confirmed-leak list. These are **breaking changes to the technically-public namespace**, but no external consumers import these names (they are internal helpers). Note that test files that call these through `DerivaML` mixins will need corresponding updates.

| # | Current name | New name | Module | Rationale |
|---|---|---|---|---|
| 1 | `domain_path()` | `_domain_path()` | `core/mixins/path_builder.py` | Low-level ERMrest handle; only used inside deriva-ml |
| 2 | `table_path()` | `_table_path()` | `core/mixins/path_builder.py` | Filesystem path helper for CSV bag export |
| 3 | `prefetch_dataset()` | `_prefetch_dataset()` | `core/mixins/dataset.py` | Deprecated shim; zero callers — also flagged for deletion (§8) |
| 4 | `list_foreign_keys()` | `_list_foreign_keys()` | `core/mixins/annotation.py` | Zero callers in src/, tests/, docs/ — also flagged for deletion (§8) |
| 5 | `is_system_schema()` | `_is_system_schema()` | `core/constants.py` | Schema introspection predicate; users will not call this |
| 6 | `get_domain_schemas()` | `_get_domain_schemas()` | `core/constants.py` | Same as above |
| 7 | `apply_logger_overrides()` | `_apply_logger_overrides()` | `core/logging_config.py` | Called once inside `DerivaML.__init__`; not user-facing |
| 8 | `compute_diff()` | `_compute_diff()` | `core/schema_diff.py` | Only used inside `base.py` pin/diff logic |
| 9 | `retrieve_rid()` | `_retrieve_rid()` | `core/mixins/rid_resolution.py` | Low-level; user-facing is `resolve_rid()` |
| 10 | `asset_record_class()` | `_asset_record_class()` | `asset/asset_record.py` | Internal factory; users access via mixin method |
| 11 | `cache_features()` | `_cache_features()` | `core/base.py` | No callers in tests; legacy workspace-cache shortcut |
| 12 | `add_workflow()` | `_add_workflow()` | `core/mixins/workflow.py` | `create_workflow()` is the user-facing factory |

**Possibly-but-not-certain (triage items for Carl):**

- `core/mixins/execution.py:745` — `start_upload()` — may be internal plumbing; reviewer unsure.
- `dataset/catalog_graph.py` — `CatalogGraph` class used only internally; consider `_CatalogGraph` or `_internal` submodule.
- `VocabularyTermHandle` (`ermrest.py:245`) — re-exported in `definitions.py __all__`, returned by `add_term` / `lookup_term`; reviewer flagged as borderline. Carl to decide whether it stays public.

---

## 8. Dead Code — Open Question

Reviewer #4 identified 7 dead symbols. **This sweep has two options:**

**Option A (recommended): Delete in-scope.** Dead-code deletion is mechanical, touches the same modules as the docstring sweep, and makes the diff smaller overall (no new docstring needed for a deleted method). Each deletion is one commit within the module task.

**Option B: Defer to a separate hygiene PR.** Keep this sub-project strictly documentation/naming; dead code goes to a follow-on task.

**Dead symbol list:**

| # | Location | Evidence |
|---|---|---|
| 1 | `core/mixins/dataset.py:485` — `prefetch_dataset()` | One-line `return self.cache_dataset(...)`; docstring says "Deprecated"; zero callers |
| 2 | `core/mixins/annotation.py:405` — `list_foreign_keys()` | Zero callers |
| 3 | `core/base.py:1364` — `add_page()` | Zero callers outside own docstring |
| 4 | `core/base.py:1097` — `user_list()` | Zero callers |
| 5 | `core/base.py:955` — `globus_login()` | Zero callers |
| 6 | `core/base.py:856` — `working_data` property | Deprecation stub; only a test asserts the warning fires |
| 7 | `tools/validate_schema_doc.py:127,500,568` — `load_from_doc()` / `load_from_code()` / `diff_schemas()` | Only referenced within same file's `main()` — these may be intentional CLI entry points; triage needed |

**Decision needed from Carl:** Option A or Option B? And for item 7 (`validate_schema_doc.py`): are those three functions meant to be importable helpers or purely internal to `main()`?

---

## 9. Testing Approach — Open Question

Docstring `Example:` blocks must actually run. Two options:

**Option A: `doctest` integration via pytest.**
Add `--doctest-modules` to the pytest invocation (or a `conftest.py` `collect_ignore` whitelist). Pro: zero new test files; examples stay co-located with the code they document. Con: examples that require a live catalog will need `# doctest: +SKIP` markers, which reduces coverage to catalog-free code paths only.

**Option B: Dedicated `tests/docstring_examples/` test file per module.**
Each module that gets swept gets a corresponding `test_<module>_examples.py` that imports the example from the docstring and runs it. Pro: can parametrize against a live catalog host; integrates with the existing fixture infrastructure. Con: doubles the per-module work and creates a second place to keep in sync with the docstring.

**Option C (minimal): Author verification only.**
The implementer runs each `Example:` block manually before committing. No automated test harness. Pro: zero overhead. Con: examples can bit-rot silently.

**Recommendation:** Option A with `# doctest: +SKIP` for catalog-dependent examples, plus a note in CLAUDE.md that new public methods must have a passing doctest before merge. This gives automated coverage for the pure-Python surface (enums, config classes, Pydantic models) without requiring a live catalog for CI.

**Decision needed from Carl:** Which option?

---

## 10. Per-Module Breakdown

The sweep is organized as one task per module (or tightly-coupled module pair). Each task produces one commit: docstring upgrades + module-level docstring (if missing) + private renames for that module's leaked names + dead-code deletions for that module (if Option A is chosen).

Modules with zero findings from either reviewer are swept for completeness (module docstring check, spot-check of any methods that might have been missed) but are low-effort.

### Priority 1 — Worst areas (Reviewer #2 "weak areas")

| Module | Docstring gaps | Inline comments | Renames | Dead code |
|---|---|---|---|---|
| `core/mixins/dataset.py` | Items 6, 7, 8 (delete/list-types/add-type) | Item 1 (ORM rebuild) | `prefetch_dataset` → `_prefetch_dataset` | `prefetch_dataset` (if Option A) |
| `core/mixins/annotation.py` | Item 10 (all `AnnotationMixin.*`) | — | `list_foreign_keys` → `_list_foreign_keys` | `list_foreign_keys` (if Option A) |
| `feature.py` | Items 11, 12 (`Feature.__init__`, `feature_record_class`) | Item 5 (assoc_fkeys) | — | — |
| `dataset/dataset_bag.py` | Item 13 (`list_dataset_members`) | Item 4 (union semantics) | — | — |

### Priority 2 — Named method gaps (Reviewer #2 table)

| Module | Docstring gaps | Inline comments | Renames | Dead code |
|---|---|---|---|---|
| `execution/execution.py` | Items 1, 14 (`create_dataset`, `upload_execution_outputs`) | Item 3 (dry_run guard) | — | — |
| `dataset/dataset.py` | Items 2, 3, 4, 5 (parents/children/add-members/download) | Item 2 (snaptime pattern) | — | — |
| `core/mixins/asset.py` | Item 9 (`create_asset`) | — | — | — |

### Priority 3 — Private renames (Reviewer #4, no docstring gaps)

| Module | Docstring gaps | Renames | Dead code |
|---|---|---|---|
| `core/mixins/path_builder.py` | Spot check | `domain_path` → `_domain_path`, `table_path` → `_table_path` | — |
| `core/constants.py` | Spot check | `is_system_schema` → `_is_system_schema`, `get_domain_schemas` → `_get_domain_schemas` | — |
| `core/logging_config.py` | Module docstring (expand) | `apply_logger_overrides` → `_apply_logger_overrides` | — |
| `core/schema_diff.py` | Spot check | `compute_diff` → `_compute_diff` | — |
| `core/mixins/rid_resolution.py` | Spot check | `retrieve_rid` → `_retrieve_rid` | — |
| `asset/asset_record.py` | Spot check | `asset_record_class` → `_asset_record_class` | — |
| `core/base.py` | Spot check | `cache_features` → `_cache_features` | Items 3–6 if Option A |
| `core/mixins/workflow.py` | Spot check | `add_workflow` → `_add_workflow` | — |

### Priority 4 — Module-level docstring sweep (all other modules)

Remaining ~50 modules get a sweep pass: add/expand module docstring, verify all public methods have at minimum a one-line summary. No Reviewer #2 or #4 findings in these modules; expect low effort.

Key modules in this tier (not exhaustive — implementer to walk full tree):

- `catalog/localize.py` (module docstring expansion noted by Reviewer #2)
- `core/mixins/execution.py` (also: triage `start_upload()` visibility — §7)
- `dataset/catalog_graph.py` (also: triage `CatalogGraph` naming — §7)
- `execution/execution_configuration.py`
- `execution/workflow.py`
- `model/catalog.py`, `model/annotations.py`, `model/schema_builder.py`
- `schema/create_schema.py`, `schema/check_schema.py`
- `local_db/workspace.py`, `local_db/denormalize.py`
- `interfaces.py`
- `core/exceptions.py`

### Summary counts

| Priority tier | Modules | Docstring items | Inline comments | Renames | Dead-code items |
|---|---|---|---|---|---|
| 1 — worst areas | 4 | 7 | 3 | 2 | 2 |
| 2 — named gaps | 3 | 8 | 2 | 0 | 0 |
| 3 — rename-only | 8 | ~16 (spot check) | 0 | 10 | 4 |
| 4 — docstring sweep | ~50 | varies | 0 | 2 (triage) | 1 (triage) |
| **Total (confirmed)** | **~65** | **~30** | **5** | **12** | **7** |

---

## 11. Approach

- **One commit per module.** Commit message format: `docs(docstrings): <module-name> — docstrings + renames`.
- **Test file updates.** After all renames are committed, a single follow-on commit updates any test file that references the old public name.
- **No squash.** Module-by-module history makes it easy to identify which commit introduced a regression if a doctest starts failing.
- **Review cadence.** Priority 1 and 2 commits go for review before Priority 3 and 4 begin, to get early signal on whether the docstring contract shape is right.

---

## 12. Open Questions for Carl

1. **Dead code (§8):** Option A (delete in this PR) or Option B (defer)?

2. **Testing approach (§9):** Option A (`--doctest-modules` + `# doctest: +SKIP`), Option B (dedicated test files), or Option C (manual verification)?

3. **`start_upload()` visibility:** Is `core/mixins/execution.py:745 start_upload()` an internal helper (→ rename to `_start_upload`) or intentionally callable by advanced users?

4. **`CatalogGraph` naming:** Rename the class to `_CatalogGraph` (marks it internal) or move it to an `_internal` submodule, or leave it public?

5. **`VocabularyTermHandle` public API:** Keep it in `__all__` as a public type? Or move it to `_internal`?

6. **`validate_schema_doc.py` dead functions:** Are `load_from_doc()`, `load_from_code()`, `diff_schemas()` intended as importable helpers for external tooling, or purely internal to the `main()` CLI entry point?

7. **`working_data` deprecation stub (`core/base.py:856`):** Delete the property and its test, or keep the stub with a proper `DeprecationWarning` docstring until a major version bump?

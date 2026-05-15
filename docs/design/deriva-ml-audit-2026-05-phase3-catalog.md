# deriva-ml audit 2026-05 — Phase 3: catalog/

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/catalog/`
(1 665 LoC across 5 Python files: `clone.py` 359, `clone_via_bag.py`
457, `localize.py` 464, `provenance.py` 319, `__init__.py` 66) and the
corresponding test surface
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/catalog/` (754 LoC, 2 test
files, ~10 tests) at the tip of `main` (HEAD `e8d20a5b`).
Cross-workspace references were grepped against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

The Phase 1 audit flagged `catalog/` as "recently rewritten" and
deferred deep review: `clone.py` had shrunk from ~1 900 LoC to 412
(now 359), `clone_via_bag` was clean, and the worst legacy items
(`CloneReport`, `CloneIssue`, the three-stage clone) were deleted in
ADR-0006's wake. That call was correct — the active code path
(`clone_via_bag` → `CatalogBagBuilder` → `BagCatalogLoader`) is in
good shape. The dragging tail here is **everything that exists
*around* that path**: a back-compat wrapper (`create_ml_workspace`)
with no in-repo callers, a localization module with no callers at
all, a provenance dataclass family that's never populated, and a
package re-export tree that routes through the back-compat shim
instead of the canonical module.

## Executive summary

Overall posture: **the clone path is healthy and load-bearing; the
surface around it is mostly archaeology.** `clone_via_bag` is the
canonical implementation, well-tested at the unit level, and has
three live-catalog integration smoke tests. The four other public
surfaces — `create_ml_workspace`, `localize_assets`, the
`CatalogProvenance`/`CloneDetails` dataclass pair, and the legacy
`OrphanStrategy` alias — have zero callers in the deriva-ml `src/`
tree and zero callers in any active sibling project. The single
external consumer (`deriva-mcp/src/deriva_mcp/tools/catalog.py`) is
**already broken at runtime** against the current
`create_ml_workspace` — it imports a non-existent `AssetCopyMode`
and reads attributes (`result.hostname`, `result.report.summary.*`)
that `CloneViaBagResult` does not provide. Per CLAUDE.md
("MCP cut-over (Phase 4)"), `deriva-mcp` is legacy and slated for
replacement by `deriva-mcp-core + deriva-ml-mcp`; the new MCP path
does not call any catalog/ symbol.

Top themes ranked by impact:

1. **`localize_assets` has zero callers across the entire workspace
   and zero tests.** `catalog/localize.py:54` is a 350-LoC public
   function exported from `catalog/__init__.py:38` and documented in
   the module docstring of `catalog/__init__.py`. A workspace-wide
   grep for `localize_assets` returns hits only inside `localize.py`
   itself and the package `__init__.py`. No `src/`, no `tests/`, no
   sibling project. The dataclass `LocalizeResult` is in the same
   state. This is the largest single LoC liability in `catalog/`.
   Either land an integration test (it's a real feature that
   ML-developer users would want after a clone with
   `asset_mode=ROWS_ONLY`) or delete it.

2. **`create_ml_workspace` is a back-compat shim whose only consumer
   is broken.** `catalog/clone.py:93` exists per its module
   docstring to keep the pre-bag-pipeline call shape working. The
   one consumer (`deriva-mcp/src/deriva_mcp/tools/catalog.py:557`)
   imports a name that doesn't exist (`AssetCopyMode`) and expects
   the old `CloneCatalogResult` shape, not `CloneViaBagResult`. The
   shim's `_warn_about_legacy_params` walks 11 dead parameter names
   to emit warnings; the `_coerce_asset_mode` / `_coerce_orphan_strategy`
   helpers translate string spellings nobody actually passes. None
   of this is exercised by deriva-ml's own tests. CLAUDE.md's
   "No backwards-compat shims" rule applies cleanly: `clone.py` is
   the textbook example.

3. **`CloneDetails` is a 20-field dataclass that nothing ever
   populates.** `provenance.py:62-157` defines a structural record
   for clone provenance with explicit `to_dict` / `from_dict`
   round-trip support and 20 fields covering legacy clone knobs
   (`truncate_oversized`, `prune_hidden_fkeys`, `orphan_rows_nullified`,
   `fkeys_pruned`, ...). A workspace-wide grep for `CloneDetails(`
   (constructor calls) returns **zero** hits — no production code,
   no test code, no sibling project. The bag pipeline records its
   provenance in the bag's `metadata/` directory, not on the
   catalog annotation; the dataclass is fossil scaffolding. The
   `CatalogProvenance.is_clone` property (line 219-222) always
   returns False as a consequence.

4. **Internal imports route through `clone.py` instead of
   `provenance.py`.** `core/base.py:70` and `core/base.py:1037`,
   plus all five `__init__.py` lazy-import branches (lines 86, 90,
   94, 98, 102), import `CatalogProvenance` / `get_catalog_provenance`
   / etc. from `deriva_ml.catalog.clone`. The actual implementation
   lives in `deriva_ml.catalog.provenance`; `clone.py:66-72` re-exports
   it with a `# noqa: F401` marker for back-compat. Since the
   re-export path is internal, there is no compatibility concern —
   these imports should point at `provenance.py` directly, and the
   re-export should be deleted.

5. **`clone_via_bag`'s policy-merge silently overrides explicit
   `DanglingFKStrategy.FAIL`.** `clone_via_bag.py:372-375` reads
   "if the caller's policy has `dangling_fk_strategy == FAIL`,
   replace it with `DELETE` because that's the library default."
   The same logic at line 368-369 overrides explicit
   `VocabExport.REFERENCED_ONLY` with `FULL`. The merge can't
   distinguish "left at default" from "set to this value
   deliberately" — Pydantic doesn't surface `model_fields_set` on
   model_copy. A DBA-persona user who picks FAIL intentionally for
   strict-mode safety has no way to express it through
   `clone_via_bag`. The unit test pins this surprising behavior
   (`test_clone_via_bag_merges_defaults_into_partial_policy`)
   rather than catching it.

Worst-offending modules within `catalog/`:

1. **`localize.py`** (464 LoC). Zero callers, zero tests, public
   function, top-level import in package `__init__.py`. Either it
   needs an integration test that ML-developer users would
   recognize as serving them, or it needs to be deleted. The body
   of the function has its own quality issues (variable
   shadowing at line 252, unreachable branch in `_extract_hatrac_path`
   at line 446, `or "..."` patterns at lines 178/256/257 that
   confuse empty-string vs missing-column).

2. **`clone.py`** (359 LoC). Pure shim. Only callers are deriva-mcp
   (legacy, broken) and itself. The wrapper carries 11 dead
   parameter names, two coercion helpers for legacy string
   spellings, and a deprecation-warning emitter that walks the
   same 11 names. None of it is tested.

3. **`provenance.py`** (319 LoC). Healthy core (the
   `CatalogProvenance` annotation read/write does what it says).
   But `CloneDetails` (95 LoC counting dataclass + to_dict +
   from_dict) is uninstantiated dead code; `set_catalog_provenance`
   doesn't accept a `clone_details` argument; `is_clone` always
   returns False. The whole "clone-specific provenance" half of
   the module is unwired.

4. **`clone_via_bag.py`** (457 LoC). The healthy one. The policy
   merge (lines 339-377) is the only design wart — it silently
   rewrites caller-explicit values.

---

## Subsystem inventory

| File | LoC | Posture |
|---|---:|---|
| `__init__.py` | 66 | Public surface; `__all__` includes everything. Module docstring describes the bag-pipeline reorganization. **Mostly healthy** but exports `localize_assets`, `LocalizeResult`, `create_ml_workspace`, `OrphanStrategy` that nothing in-repo uses. |
| `clone.py` | 359 | Back-compat wrapper over `clone_via_bag`. `create_ml_workspace`, `_coerce_asset_mode`, `_coerce_orphan_strategy`, `_warn_about_legacy_params`. Re-exports provenance API. **No internal callers.** |
| `clone_via_bag.py` | 457 | The canonical clone path. `clone_via_bag()`, plus helpers `_materialize_bag_dir`, `_materialize_bag_assets`, `_expand_nested_dataset_anchors`, `_collect_nested_dataset_rids`. **Load-bearing and healthy.** |
| `localize.py` | 464 | `localize_assets()` + `LocalizeResult` + 5 private helpers. **No callers anywhere; no tests.** |
| `provenance.py` | 319 | `CatalogProvenance`, `CloneDetails`, `CatalogCreationMethod`, `set_catalog_provenance`, `get_catalog_provenance`, `CATALOG_PROVENANCE_URL`. `CloneDetails` is fossil. |

Cross-module dependencies inside `catalog/`:

- `clone.py:194` lazy-imports `clone_via_bag.clone_via_bag` — the
  shim's whole purpose.
- `clone.py:66-72` re-exports the provenance API from
  `provenance.py` for back-compat.
- `clone_via_bag.py` has no internal imports from other `catalog/`
  modules. Independent of `provenance.py` (a fresh clone does NOT
  write a provenance annotation, even though
  `CatalogCreationMethod.CLONE` exists for exactly this).
- `localize.py` has no internal imports from other `catalog/`
  modules. Standalone.

External (deriva-ml internal) call sites:

- `src/deriva_ml/__init__.py:85-104` — lazy-loads
  `CatalogProvenance`, `CatalogCreationMethod`, `CloneDetails`,
  `get_catalog_provenance`, `set_catalog_provenance` (all from
  `clone.py` re-export, not from `provenance.py` directly).
- `src/deriva_ml/core/base.py:70` — type-only import of
  `CatalogProvenance` (under `TYPE_CHECKING`).
- `src/deriva_ml/core/base.py:1037` — calls
  `get_catalog_provenance` inside the `DerivaML.catalog_provenance`
  property. Imports from `clone.py`, not `provenance.py`.

That's the entirety of internal usage. `create_ml_workspace`,
`clone_via_bag`, `localize_assets`, `OrphanStrategy`, and
`set_catalog_provenance` have zero in-repo callers outside the
catalog/ package itself.

---

## Cross-workspace usage check

Verification discipline per the audit prompt: every symbol whose
deletion or privatization is proposed below was grepped across
`deriva-mcp`, `deriva-mcp-core`, `deriva-ml-mcp`,
`deriva-ml-model-template`, `deriva-skills`, and `deriva-ml-skills`.

| Symbol | External callers | Notes |
|---|---|---|
| `create_ml_workspace` | `deriva-mcp/src/deriva_mcp/tools/catalog.py:557, 569`, `deriva-mcp/src/deriva_mcp/tools/background_tasks.py:122, 156`, `deriva-mcp/tests/test_catalog.py` (7 mock-target references) | **Broken at runtime** — `deriva-mcp` also imports `AssetCopyMode` (which does not exist in `deriva_ml.catalog`; grep confirms) and reads attributes `result.hostname`, `result.catalog_id`, `result.source_snapshot`, `result.datasets_reinitialized`, `result.ml_schema_added`, `result.report.summary.*`, `result.truncated_values` that `CloneViaBagResult` does not provide. The `deriva-mcp/tests/test_catalog.py` references are mock-targets (`patch("deriva_ml.catalog.create_ml_workspace", ...)`) — the legacy MCP's own tests stub out the function, so they pass without ever invoking it. |
| `OrphanStrategy` | `deriva-mcp/src/deriva_mcp/tools/catalog.py:557, 561`, `deriva-mcp/src/deriva_mcp/tools/background_tasks.py:122, 136` | Same broken legacy MCP. |
| `AssetCopyMode` | `deriva-mcp/src/deriva_mcp/tools/catalog.py:557`, `deriva-mcp/src/deriva_mcp/tools/background_tasks.py:122` | **Does not exist in `deriva_ml.catalog`.** `grep -rn 'AssetCopyMode' deriva-ml/` returns zero hits. The legacy MCP import is broken. |
| `clone_via_bag` | None | No external callers; zero hits in any sibling project. |
| `CloneViaBagResult` | None | Same. |
| `localize_assets` | None | Zero hits anywhere in workspace. |
| `LocalizeResult` | None | Same. |
| `set_catalog_provenance` | `deriva-ml-model-template/src/scripts/_cifar10_schema.py:44, 120`, `deriva-ml-model-template/docs/superpowers/plans/2026-05-13-e2e-platform-test.md:1242` (doc example) | **Live external consumer.** The model-template uses it at catalog-creation time. |
| `get_catalog_provenance` | None outside `deriva-ml/` | `DerivaML.catalog_provenance` is the indirect surface; no sibling project queries provenance directly. |
| `CatalogProvenance` | None outside `deriva-ml/` | Same. |
| `CloneDetails` | **Never instantiated anywhere in the workspace.** `grep -rn 'CloneDetails('` returns zero hits across all six sibling projects + deriva-ml itself. |
| `CatalogCreationMethod` | None outside `deriva-ml/` | Indirectly visible through `set_catalog_provenance(creation_method=...)` but the model-template caller doesn't pass it explicitly. |
| `CATALOG_PROVENANCE_URL` | None outside `deriva-ml/` | Only used inside `provenance.py`. |

**Conclusions from the table:**

- `localize_assets` / `LocalizeResult` / `CloneDetails` have **zero
  callers anywhere in the workspace** (not even doc examples).
  Deletion or test-or-die is the call.
- `create_ml_workspace` / `OrphanStrategy` are kept alive only for a
  legacy MCP server that is already broken against them and is
  scheduled for replacement per the workspace cut-over plan. Once
  `deriva-mcp` retires, this whole surface deletes cleanly. Until
  then, the wrapper exists and ships warnings nobody sees.
- `set_catalog_provenance` is the one externally-load-bearing
  symbol. **Don't break it.** It's used by the model-template's
  catalog bootstrap script — keep the signature stable, keep the
  three-arg call form (`name=`, `description=`, optionally
  `workflow_url=` / `workflow_version=`) working.

---

## Lens 1 — Dead code

### 1.1 `localize_assets` has zero callers and zero tests

`catalog/localize.py:54-330` defines a 277-LoC public function that
downloads assets from remote Hatrac and uploads them to local
Hatrac. The module docstring (`localize.py:1-15`) describes a
post-clone workflow: clone with `asset_mode=REFERENCES`, then run
`localize_assets` to copy bytes locally. The package docstring
mentions it (`catalog/__init__.py:36-39` imports + `__all__:64-65`
exports), but nothing else in the workspace ever calls it.

Verification:

```
$ grep -rn 'localize_assets\|LocalizeResult' \
       deriva-ml/src/ deriva-ml/tests/ \
       deriva-mcp/ deriva-mcp-core/ deriva-ml-mcp/ \
       deriva-ml-model-template/ deriva-skills/ \
       deriva-ml-skills/ 2>/dev/null
deriva-ml/src/deriva_ml/catalog/__init__.py:36-39  (package re-export)
deriva-ml/src/deriva_ml/catalog/__init__.py:64-65  (__all__)
deriva-ml/src/deriva_ml/catalog/localize.py        (the file itself)
```

The function is plausible — an ML-developer user who clones a
catalog with `AssetMode.ROWS_ONLY` (the cheap-clone mode) would
want a follow-up step to copy the bytes locally. But (a) no
integration test pins the behavior, (b) the existing comment in
the function's docstring still refers to the deleted
`asset_mode="refs"` spelling, and (c) the bag pipeline now offers
`AssetMode.UPLOAD_IF_MISSING` which copies during clone instead
of after. The follow-up workflow this module serves may not be
necessary anymore.

Additionally, the function body has implementation-quality issues
(see §1.2, §3.2, §3.4) — without callers exercising it, those
have gone unnoticed.

**Fix:** three options, in order of preference:

(a) **Delete the module.** `localize_assets`, `LocalizeResult`,
the 5 helper functions, the `catalog/__init__.py` imports +
`__all__` entries. **LoC: −464.** **Risk: low** — no callers
break.

(b) **Add an integration test** that clones with `ROWS_ONLY`
and then runs `localize_assets` end-to-end against the demo
catalog. If the test passes, the module is live; if it doesn't,
delete the module. The test would belong in
`tests/catalog/test_localize_integration.py` with the
`@pytest.mark.integration` marker that
`test_clone_via_bag_integration.py` uses.

(c) **Rename to underscore-private and move inside
`clone_via_bag.py`** — turn it into a localize-after-clone
helper that the canonical clone function exposes. This makes
sense only if someone is going to use it; otherwise (a).

**Recommended:** (b) first to test market fit; (a) if no test
materializes. **Severity: high** — by LoC, this is the single
biggest dead-code candidate in the audit.

### 1.2 `_extract_hatrac_path` has an unreachable branch

`localize.py:438-449`:

```python
def _extract_hatrac_path(url: str) -> str | None:
    parsed = urlparse(url)
    path = parsed.path

    if "/hatrac/" in path:
        idx = path.find("/hatrac/")
        return path[idx:]

    if path.startswith("/hatrac/"):
        return path

    return None
```

The second `if path.startswith("/hatrac/")` is unreachable: any
path that starts with `/hatrac/` necessarily contains `/hatrac/`,
so the first branch always wins. The dead branch returns the
same value the first branch would have anyway.

**Fix:** delete lines 446-447. **Risk: trivial. LoC: −3.**
**Severity: low** but symbolic — it suggests the function was
written without tests.

### 1.3 `CloneDetails` is never instantiated

`provenance.py:62-157` defines a 95-LoC dataclass with 20 fields
covering legacy clone knobs: `orphan_strategy`, `truncate_oversized`,
`prune_hidden_fkeys`, `schema_only`, `asset_mode`, `exclude_schemas`,
`exclude_objects`, `add_ml_schema`, `copy_annotations`,
`copy_policy`, `reinitialize_dataset_versions`, `rows_copied`,
`rows_skipped`, `skipped_rids`, `truncated_count`,
`orphan_rows_removed`, `orphan_rows_nullified`, `fkeys_pruned`,
`source_hostname`, `source_catalog_id`, `source_snapshot`,
`source_schema_url`. With `to_dict` and `from_dict` round-trip
helpers.

A workspace-wide grep for `CloneDetails(` (the constructor call,
not the type reference) returns **zero** hits — nothing
populates it. The fields exist solely to be deserialized back
out of `from_dict` if someone ever wrote them, which they never
did. `CatalogProvenance.is_clone` (line 219-222) returns
`self.creation_method == CLONE and self.clone_details is not None`
— since the second clause is always False in practice, the
property always returns False.

The bag pipeline records clone provenance in the bag's
`metadata/` directory (per `clone_via_bag.py`'s `producer=`
parameter on `CatalogBagBuilder`). The dataclass is fossil
scaffolding from the legacy three-stage clone era.

**Fix:** delete `CloneDetails`, remove the
`clone_details` field from `CatalogProvenance`, remove the
`is_clone` property, drop the relevant `to_dict`/`from_dict`
branches, drop the `CloneDetails` symbol from `__all__` and
package re-export and `deriva_ml/__init__.py` lazy loader.
**LoC: −110.** **Risk: low** — zero callers. **Severity: medium**
(largest single dead-block in `provenance.py`).

A more conservative option: keep `CloneDetails` but
add a `clone_details: CloneDetails | None = None` parameter to
`set_catalog_provenance` so the field can actually be
populated. The catalog provenance API would then describe what
the bag pipeline records, and `is_clone` would mean something.
This is the option (b)-style answer — useful only if someone
intends to wire it.

### 1.4 `_warn_about_legacy_params` walks 11 dead parameter names

`clone.py:314-348` accepts kwargs of 11 known legacy names and
emits a `DeprecationWarning` when any is set to a non-default
value. Master audit Lens 5.3 (`deriva-ml-audit-2026-05.md:765`)
already flagged this: "walks 11 named kwargs and emits a
[warning]." The constants are duplicated between the function's
signature (`clone.py:93-119`) and the defaults dict
(`clone.py:323-335`). The `_LEGACY_ONLY_PARAMS` constant the
master audit referenced has since been removed, but the parallel
defaults dict pattern persists.

The whole purpose of these warnings is to nudge callers off the
legacy parameter set. With `deriva-mcp` already broken on
`AssetCopyMode` (the import fails before any warning fires) and
the new MCP not calling this surface at all, **the warnings have
no audience**.

**Fix:** subsumed by §1.5 (delete the whole shim). If the shim
must stay during the legacy-MCP retirement window, replace the
11-name walk with a single `**kwargs` capture + a one-line
"these kwargs are ignored" warning. **Severity: low** — code
quality only; semantic correctness is fine.

### 1.5 `create_ml_workspace` is a back-compat shim with no working callers

`clone.py:93-267` is a 175-LoC wrapper around `clone_via_bag`.
Per its own module docstring (lines 1-50), it exists to keep the
legacy pre-bag-pipeline call shape working. Verification of
external callers (see §"Cross-workspace usage check"):

- `deriva-mcp/src/deriva_mcp/tools/catalog.py:557`,
  `deriva-mcp/src/deriva_mcp/tools/background_tasks.py:122` —
  both import `AssetCopyMode, OrphanStrategy, create_ml_workspace`.
  **`AssetCopyMode` does not exist in `deriva_ml.catalog`.**
  Workspace-wide grep returns zero hits. The import statement
  itself raises `ImportError`.
- Both call sites then read `result.hostname`,
  `result.catalog_id`, `result.source_snapshot`,
  `result.datasets_reinitialized`, `result.ml_schema_added`,
  `result.report.summary.*`, `result.truncated_values` — none
  of which `CloneViaBagResult` provides (`CloneViaBagResult` has
  only `source_catalog_id`, `dest_catalog_id`, `bag_path`,
  `load_report`).
- `deriva-mcp/tests/test_catalog.py` mocks the function name
  (`patch("deriva_ml.catalog.create_ml_workspace", ...)`) — the
  tests stub the function out, so they pass without ever calling
  the real implementation. The tests don't tell us anything
  about real-world callers.
- New MCP (`deriva-mcp-core` + `deriva-ml-mcp`): zero hits.
- Model template: zero hits.
- Skills plugins: zero hits.

In other words: **the shim is preserved for a caller that doesn't
work and is being deleted.** Per CLAUDE.md ("No backwards-compat
shims — if something is unused, delete it"), this is the textbook
case.

**Fix:** delete `clone.py` in its entirety after the legacy MCP
retires. Move the back-compat provenance re-exports
(`CatalogProvenance`, `CatalogCreationMethod`, `CloneDetails`,
`get_catalog_provenance`, `set_catalog_provenance`) directly into
`catalog/__init__.py` from `provenance.py` (already the case for
the package re-export; the `clone.py:66-72` re-export becomes
unneeded once `__init__.py` and `core/base.py` import directly
from `provenance.py`). **LoC: −359.** **Risk: medium** —
breaks any out-of-tree caller that imports
`deriva_ml.catalog.clone`. Per the cross-workspace check above,
there is no such caller that isn't already broken. **Severity:
medium** — symbolically high (largest cleanup candidate).

If the legacy MCP retirement is months away, the conservative
option is to keep `clone.py` but mark it `_clone.py` and rewrite
the public symbols' re-export through `__init__.py`. This buys
nothing — the broken caller stays broken — and adds a rename
churn cost. (a) is cleaner.

### 1.6 `OrphanStrategy` is an alias of `DanglingFKStrategy` with no in-repo callers

`clone.py:85`:

```python
OrphanStrategy = _DanglingFKStrategy
```

Used only by `deriva-mcp` (which is already broken — see §1.5)
and by `create_ml_workspace`'s default parameter (`clone.py:111`).
Inside deriva-ml, every test and live code path uses
`DanglingFKStrategy` directly (see `test_clone_via_bag.py:22`,
`test_clone_via_bag.py:202`, `test_bag_commit_poc.py:39` etc.).

**Fix:** subsumed by §1.5. Delete with `clone.py`. **Severity:
low.**

### 1.7 `set_catalog_provenance` cannot record clone details

`provenance.py:225-280` accepts `name`, `description`, `workflow_url`,
`workflow_version`, and `creation_method`. Notably absent:
`clone_details`. So even when a caller passes
`creation_method=CatalogCreationMethod.CLONE`, the resulting
`CatalogProvenance.clone_details` is `None`. The
`CatalogProvenance.is_clone` property at `provenance.py:219-222`
returns False as a consequence:

```python
@property
def is_clone(self) -> bool:
    return self.creation_method == CatalogCreationMethod.CLONE \
        and self.clone_details is not None
```

This is internally inconsistent: the enum says "this catalog is a
clone," but the property says "no it isn't." Pair this with §1.3
(nothing populates `CloneDetails`) and the whole clone-provenance
half of the module is unwired.

**Fix:** either (a) add `clone_details: CloneDetails | None = None`
to `set_catalog_provenance` so the field can be populated, then
wire `clone_via_bag` to call `set_catalog_provenance(... creation_method=CLONE, clone_details=CloneDetails(...))` at the end of a successful clone, or (b) delete the whole
clone-provenance leg per §1.3. **Risk: low.** **Severity:
medium** — bug, not just dead code; `is_clone` is a public
property that promises something the API can't deliver.

### 1.8 Internal re-exports route through `clone.py` instead of `provenance.py`

`src/deriva_ml/__init__.py:86-104` lazy-loads 5 names from
`deriva_ml.catalog.clone`:

```python
elif name == "CatalogProvenance":
    from deriva_ml.catalog.clone import CatalogProvenance
    return CatalogProvenance
elif name == "CatalogCreationMethod":
    from deriva_ml.catalog.clone import CatalogCreationMethod
    return CatalogCreationMethod
elif name == "CloneDetails":
    from deriva_ml.catalog.clone import CloneDetails
    return CloneDetails
elif name == "get_catalog_provenance":
    from deriva_ml.catalog.clone import get_catalog_provenance
    return get_catalog_provenance
elif name == "set_catalog_provenance":
    from deriva_ml.catalog.clone import set_catalog_provenance
    return set_catalog_provenance
```

`core/base.py:70, 1037` does the same. The actual definitions
live in `provenance.py`; `clone.py:66-72` re-exports them with
`# noqa: F401`. Since these are internal imports, there is no
external compatibility concern — and routing through the
back-compat shim is the opposite of what a clean module
boundary looks like.

**Fix:** change every internal import from
`deriva_ml.catalog.clone` to `deriva_ml.catalog.provenance`. Drop
the re-export at `clone.py:66-72`. **Risk: trivial. LoC: ±0**
(literally a path swap). **Severity: low** — code quality, not
correctness.

### 1.9 `__init__.py` package docstring lies about `OrphanStrategy`'s home

`catalog/__init__.py:16-17`:

> :data:`~deriva_ml.catalog.clone.OrphanStrategy` is an alias of
> :class:`deriva.bag.traversal.DanglingFKStrategy`.

`DanglingFKStrategy` is the canonical type, exported from the
upstream `deriva.bag.traversal` package. `OrphanStrategy` is the
back-compat alias maintained for the legacy MCP. New code should
import `DanglingFKStrategy` directly (per the test suite's own
convention). The docstring doesn't tell readers this — it implies
`OrphanStrategy` is the preferred deriva-ml spelling.

**Fix:** rewrite the docstring entry to say "use
`deriva.bag.traversal.DanglingFKStrategy` directly; `OrphanStrategy`
is a legacy alias retained for back-compat." Or, with §1.5
landed, just delete the entry. **Severity: low.**

---

## Lens 2 — Privatization

### 2.1 `_coerce_asset_mode`, `_coerce_orphan_strategy`, `_warn_about_legacy_params` are already private — fine

These are correctly underscore-prefixed in `clone.py` (lines 275,
298, 314). No fix.

### 2.2 `_extract_hatrac_path`, `_ensure_hatrac_namespace`,
`_fetch_asset_records`, `_find_asset_table_path`,
`_get_catalog_info` in `localize.py` are correctly private — fine

All five helpers are underscore-prefixed. No fix. (If the whole
module deletes per §1.1, moot.)

### 2.3 `_materialize_bag_dir`, `_materialize_bag_assets`,
`_expand_nested_dataset_anchors`, `_collect_nested_dataset_rids`
in `clone_via_bag.py` are correctly private — fine

All four helpers are underscore-prefixed. The test file imports
two of them
(`from deriva_ml.catalog.clone_via_bag import _expand_nested_dataset_anchors`
at `test_clone_via_bag.py:439, 465, 482`) — accessing private
symbols from tests is the documented pattern in the deriva-ml
test suite. No fix.

### 2.4 `_write_provenance_annotation` is correctly private — fine

`provenance.py:297`. No fix.

### 2.5 `OrphanStrategy` is public-named but should be private until deleted

Until §1.5/§1.6 ship, the public name `OrphanStrategy` invites
external callers to import it. The single existing external
import (deriva-mcp, already broken) doesn't justify keeping it
publicly visible.

**Fix:** subsumed by §1.5. If §1.5 is deferred, rename to
`_OrphanStrategy` and drop from `__all__`. **Severity: low.**

---

## Lens 3 — Test coverage

### 3.1 No orphan-strategy integration tests

`tests/catalog/test_clone_via_bag_integration.py` covers three
end-to-end scenarios: default policy (which becomes DELETE via
the merge), `AssetMode.ROWS_ONLY`, and a single-RID anchor scope.
None of the three exercises an orphan path with rows that would
genuinely dangle. The integration suite cannot tell whether
`DanglingFKStrategy.FAIL` aborts a clone with dangling FKs, or
whether `DanglingFKStrategy.NULLIFY` actually nullifies rather
than deletes them.

Unit tests cover the *pass-through* of the orphan-strategy
parameter (`test_clone_via_bag.py:202` constructs a policy with
`NULLIFY` and asserts the loader receives the same instance) but
do not assert on real loader behavior — they mock the loader.

Per the audit prompt's specific call-out: orphan strategies
(FAIL/DELETE/NULLIFY) are a load-bearing correctness feature, and
the audit should flag this as a coverage gap. **It is a gap.**

**Fix:** add three integration tests:

1. `test_clone_via_bag_fail_strategy_aborts_on_orphan` — set up
   the source with a row whose FK target lies outside the slice,
   anchor the clone such that the orphan would be reached, and
   assert `clone_via_bag` raises `OrphanError` (or whatever the
   loader raises with FAIL).
2. `test_clone_via_bag_delete_strategy_prunes_orphan` — same
   setup, anchor the same slice, expect the clone to succeed
   with the orphan-bearing row deleted. Assert the row is
   absent at the destination.
3. `test_clone_via_bag_nullify_strategy_nulls_fk` — same setup,
   expect the row to be present at the destination with the FK
   column null. Assert column is null.

Each test requires the demo fixture to produce a slice with a
known dangling FK, which the current `catalog_with_datasets`
fixture probably doesn't. May need a new
`catalog_with_orphan_fk` fixture.

**Effort: medium-high.** **Severity: high** for `DBA` /
`Testing engineer` personas — the orphan strategies are exactly
the kind of feature whose subtle behavior differences need
end-to-end pinning, and we don't have it.

### 3.2 `localize_assets` has zero tests

Subsumed by §1.1. The 277-LoC function has no unit tests, no
integration tests, no doctest coverage. Anything that exists in
the workspace exercises it via "is this file syntactically
valid" alone.

### 3.3 `create_ml_workspace` has zero tests

The shim function (`clone.py:93`), the legacy-parameter warning
emitter (`clone.py:314`), the asset-mode coercion (`clone.py:275`),
and the orphan-strategy coercion (`clone.py:298`) have zero unit
tests in deriva-ml's own suite. The legacy MCP's
`tests/test_catalog.py` mocks the function out (`patch(...)`).

This is fine if the shim deletes per §1.5. If it stays, it needs
at least:

- A test that `_coerce_asset_mode("refs")` returns
  `AssetMode.ROWS_ONLY`.
- A test that `_coerce_asset_mode(AssetMode.UPLOAD_FORCE)` returns
  the same enum member.
- A test that `_coerce_asset_mode("bogus")` raises `TypeError`.
- A test that `_warn_about_legacy_params(truncate_oversized=True)`
  emits a `DeprecationWarning`.
- A test that `create_ml_workspace(... dest_catalog_id=None)`
  raises `ValueError`.

**Effort: low** (mechanical). **Severity: low** if §1.5 ships;
**medium** if it doesn't.

### 3.4 `CatalogProvenance.from_dict` / `to_dict` round-trip has zero tests

`provenance.py:180-217` defines round-trip serialization for the
annotation payload. No tests anywhere assert the round-trip works
or that defaults are preserved. The model-template's caller
(`_cifar10_schema.py:120`) writes a provenance but never reads
one back; deriva-ml's own `DerivaML.catalog_provenance` property
reads but the property itself has no test.

**Fix:** add unit tests in a new file
`tests/catalog/test_provenance.py`:

- `test_catalog_provenance_round_trip` — construct a
  `CatalogProvenance` with every field set, `to_dict()`, then
  `from_dict()`, and assert equality.
- `test_catalog_provenance_unknown_method_falls_back` — pass a
  dict with `creation_method="bogus"`, assert
  `CatalogCreationMethod.UNKNOWN` results.
- `test_set_catalog_provenance_writes_annotation` — mock
  `catalog.put`, call `set_catalog_provenance`, assert the
  expected JSON payload.
- `test_get_catalog_provenance_handles_missing_annotation` —
  mock `catalog.getCatalogModel().annotations.get` returning
  None, assert `get_catalog_provenance` returns None.

**Effort: low.** **Severity: medium** — `CatalogProvenance` is
ostensibly a stable public API for an external consumer
(model-template); the round-trip semantics shouldn't be
verified only by inspection.

### 3.5 Two unit tests rely on private-symbol imports — fine, but flag

`test_clone_via_bag.py:439, 465, 482` each begin with:

```python
from deriva_ml.catalog.clone_via_bag import (
    _expand_nested_dataset_anchors,
)
```

This is the documented test convention. It does however mean the
private helper is part of an implicit test contract — renaming
or refactoring `_expand_nested_dataset_anchors` will break the
tests. Not a fix; a note for future refactor.

---

## Lens 4 — Docs sync

### 4.1 `__init__.py` docstring claims feature parity that isn't accurate

`catalog/__init__.py:10-15` describes `create_ml_workspace` as
"the legacy spelling, reimplemented on top of `clone_via_bag`"
and says legacy-only parameters "emit a deprecation warning when
set away from default." Accurate as far as it goes — but the
docstring doesn't mention that the function's return type is now
`CloneViaBagResult`, completely different from what legacy
callers expect. A reader who trusted the docstring would assume
they could swap call sites with no further changes.

Master audit `5.6` (`deriva-ml-audit-2026-05.md:782` neighborhood)
made the same observation about `clone_via_bag` itself. Same
finding here, doubled.

**Fix:** add a sentence: "Return type is
`CloneViaBagResult`, not the legacy `CloneCatalogResult` —
callers that read `result.report.summary.*` will not work
unchanged. See the function docstring for the new attribute
set." **Severity: low.**

### 4.2 `localize_assets` docstring still references deleted `asset_mode="refs"` spelling

`localize.py:67-72`:

> This is useful after cloning a catalog with `asset_mode="refs"`
> where the asset URLs still point to the source server...

`AssetMode.REFERENCES` was renamed to `AssetMode.ROWS_ONLY` in
the bag pipeline; the string spelling `"refs"` is now a legacy
coercion target in `_coerce_asset_mode` (`clone.py:287`). The
docstring instructs users to use a value that the canonical clone
path doesn't accept anymore.

**Fix:** rewrite to: "useful after cloning with
`AssetMode.ROWS_ONLY` where the bag's asset URLs still point to
the source." Subsumed by §1.1 if the module deletes. **Severity:
low.**

### 4.3 `provenance.py` docstring describes a feature that doesn't work

`provenance.py:65-69` describes `CloneDetails`:

> Populated only when `CatalogProvenance.creation_method` is
> `CatalogCreationMethod.CLONE`.

The "populated only when ..." formulation implies the field is
*ever* populated. Per §1.3 / §1.7, it isn't. Reader is misled.

**Fix:** subsumed by §1.3 or §1.7. **Severity: low.**

### 4.4 Docstring examples missing `# doctest: +SKIP`

`core/base.py:1028-1036` (the `DerivaML.catalog_provenance`
property example):

```python
Example:
    >>> ml = DerivaML('localhost', '45')
    >>> prov = ml.catalog_provenance
    >>> if prov:
    ...     print(f"Created: {prov.created_at} by {prov.created_by}")
    ...     print(f"Method: {prov.creation_method.value}")
    ...     if prov.is_clone:
    ...         print(f"Cloned from: {prov.clone_details.source_hostname}")
```

No `# doctest: +SKIP` markers. Per CLAUDE.md's "Docstring
Examples (Doctest)" section: "Catalog-dependent examples must
carry `# doctest: +SKIP` on the first interactive line." This
example creates a `DerivaML` instance — catalog-dependent. The
doctest collection at `pytest --doctest-modules` time will try
to execute it.

The same risk exists for `localize.py:104-133` — the
`Example:` block has three sub-examples; the first
(`>>> ml = DerivaML("localhost", "42")`) is `# doctest: +SKIP`'d
implicitly because the line before it has the marker, but
`>>> result = localize_assets(ml, asset_table="Image", ...)`
in the next paragraph (line 108) does *not* — it's a follow-on
line in a paragraph whose first line has the SKIP, which means
pytest-doctest will still try to run it.

**Fix:** audit every catalog-dependent example block; add
`# doctest: +SKIP` to the first interactive line of every
paragraph that touches a live catalog. **Risk: trivial.**
**Severity: low.**

### 4.5 `clone_via_bag` Example uses default `Dataset` table assumption — fine

`clone_via_bag.py:293-306` example shows `root_rid="1-ABCD"` and
the convenience `Dataset` anchor mapping. The example has
`# doctest: +SKIP` correctly applied. **No fix.**

---

## Lens 5 — deriva-py API conventions

### 5.1 `_collect_nested_dataset_rids` uses datapath `.in_()` — fine

`clone_via_bag.py:170` uses `dd.filter(dd.Dataset.in_(sorted(frontier))).entities().fetch()`.
Master audit Lens 2.1 flagged the predecessor of this code for
rolling its own ERMrest URL; **the rewrite is now datapath-native**.
Good shape. **No fix.**

### 5.2 `_fetch_asset_records` uses datapath `.in_()` — fine

`localize.py:416` uses
`table_path.path.filter(table_path.RID.in_(list(rids))).entities().fetch()`.
Master audit Lens 2.2 flagged the predecessor (which built an OR
chain); **the rewrite is now datapath-native**. Good. The
`DataPathException` fallback at line 418 is overcautious — the
`.in_()` operator has been in deriva-py for years — but
defensive. **No fix.**

### 5.3 `provenance.py` uses raw `catalog.put` and `catalog.get` for annotations

`provenance.py:251` — `catalog.get("/authn/session")` — necessary,
no datapath equivalent.

`provenance.py:259` — `catalog.get("/")` — necessary, no datapath
equivalent.

`provenance.py:303-306`:

```python
catalog.put(
    f"/annotation/{urlquote(CATALOG_PROVENANCE_URL)}",
    json=provenance.to_dict(),
)
```

The deriva-py model-API alternative is:

```python
model = catalog.getCatalogModel()
model.annotations[CATALOG_PROVENANCE_URL] = provenance.to_dict()
model.apply()
```

Both work. The model-API form is one network round trip (fetch
schema, modify, apply), the raw-URL form is one round trip too.
The model-API form is what `model/annotations.py` uses elsewhere
in the codebase — `apply_catalog_annotations` (`core/base.py:1043`)
goes through the model API.

Inconsistency: the **read** path (`get_catalog_provenance` at
`provenance.py:288-289`) goes through the model API
(`catalog.getCatalogModel().annotations.get(...)`), but the
**write** path goes raw. Not wrong, just asymmetric.

**Fix:** swap the write to the model API form to match the read.
Optional; doesn't affect correctness. **Risk: low.**
**Severity: low.**

### 5.4 `_get_catalog_info` uses `hasattr` to discriminate types — fine, but brittle

`localize.py:333-360` uses `hasattr(catalog, "catalog") and
hasattr(catalog, "host_name")` to discriminate `DerivaML` from
`ErmrestCatalog`. The protocol-checking idiom isn't great
(a future `DerivaML` refactor that renames `host_name` to
`hostname` silently breaks this); `isinstance` would be cleaner.

**Fix:** replace with `if isinstance(catalog, DerivaML):` and a
TYPE_CHECKING import. **Risk: low.** **LoC: ±0.** **Severity:
low.** Subsumed by §1.1 if `localize.py` deletes.

---

## Lens 6 — Inconsistencies / dataclass-vs-Pydantic

### 6.1 Provenance types should be Pydantic per CLAUDE.md

CLAUDE.md "Class idiom choice — Pydantic vs `@dataclass`" rules:

> Use Pydantic BaseModel when ANY of these apply: ... The class
> may be serialized or cross a boundary (JSON I/O, logs, cache,
> API, bag metadata). Users should reach for one API
> (.model_dump()) rather than juggling dataclasses.asdict()
> depending on type.

`CatalogProvenance` and `CloneDetails` (`provenance.py:62, 160`)
are:

- Returned to users (boundary type — `DerivaML.catalog_provenance`
  property).
- JSON-serialized (the entire point — they're an annotation
  payload).
- Carry explicit `to_dict` / `from_dict` round-trip code that
  duplicates what Pydantic does for free.

Per CLAUDE.md's "When in doubt, pick Pydantic," these are
unambiguous Pydantic cases. The explicit `to_dict` methods are
40 LoC of manual serialization that Pydantic's `.model_dump()`
would replace.

The same applies to `CloneViaBagResult` (`clone_via_bag.py:217`)
and `LocalizeResult` (`localize.py:35`) — both are boundary-type
return values; both should be Pydantic.

**Fix:** convert all four dataclasses to Pydantic `BaseModel`s.
Replace `to_dict`/`from_dict` with `model_dump`/`model_validate`.
**Risk: medium** (external API change; callers that did
`asdict(provenance)` would need to use `model_dump()`). Since
the model-template caller uses `set_catalog_provenance(...)` and
doesn't directly call `asdict`, the actual breaking surface is
near zero. **LoC: −80** (the `to_dict`/`from_dict` methods
disappear). **Severity: medium** — explicit CLAUDE.md
convention violation, but the cost-of-change is real.

### 6.2 `clone_via_bag` policy-merge silently overrides explicit FAIL

`clone_via_bag.py:367-377` merges deriva-ml clone defaults into a
caller-supplied policy when the caller "left them at library
defaults." The merge predicate:

```python
if policy.dangling_fk_strategy == DanglingFKStrategy.FAIL:
    # The library default. Caller almost certainly didn't
    # think about it; replace with the clone default.
    merge_kwargs["dangling_fk_strategy"] = DanglingFKStrategy.DELETE
```

The same logic applies at line 368-369 for
`VocabExport.REFERENCED_ONLY` → `VocabExport.FULL`.

This is the classic "you can't distinguish default from explicit"
trap. A caller who:

- Genuinely wants FAIL (strict-mode clone where any orphan should
  abort the clone for safety) gets silently switched to DELETE.
- Genuinely wants REFERENCED_ONLY (minimal vocab footprint) gets
  silently switched to FULL.

The unit test
`test_clone_via_bag_merges_defaults_into_partial_policy`
(`test_clone_via_bag.py:244-302`) pins this behavior:

```python
user_policy = FKTraversalPolicy(asset_mode=AssetMode.ROWS_ONLY)
# ...
assert merged.dangling_fk_strategy == DanglingFKStrategy.DELETE
```

This test verifies the merge happens. It does not verify the
merge respects an explicit FAIL choice — because the merge
doesn't. A different test
`test_clone_via_bag_passes_policy_through`
(`test_clone_via_bag.py:189-241`) takes a fully-specified policy
with `dangling_fk_strategy=NULLIFY` and asserts pass-through —
but only because NULLIFY isn't the library default. There is no
test asserting that explicit `dangling_fk_strategy=FAIL`
survives — because it doesn't.

Pydantic exposes `model_fields_set` on a model instance — the
set of field names that were explicitly passed at construction
time. The merge could use this to distinguish:

```python
if "dangling_fk_strategy" not in policy.model_fields_set:
    merge_kwargs["dangling_fk_strategy"] = DanglingFKStrategy.DELETE
```

This requires `FKTraversalPolicy` to be Pydantic (it is — see
`test_clone_via_bag.py:23`).

**Fix:** switch the three merge predicates from "field equals
library default" to "field name not in `model_fields_set`."
Update the existing test to reflect the new semantics; add a new
test asserting that explicit FAIL survives. **Risk: low**
(localized to `clone_via_bag.py:367-377`). **LoC: ±0.**
**Severity: high** for a DBA persona — silent semantic override
on a safety-critical flag.

### 6.3 `localize.py` line 252 shadows the outer `source_hostname` parameter

`localize.py:54-66` declares `source_hostname` as a function
parameter. Line 252 inside the per-asset loop:

```python
source_hostname = asset_info["source_hostname"]
```

reassigns the local name to the per-asset value. The intent is
clear (each asset can have a different source host), but using
the same identifier as the function parameter makes the
asset-level meaning collide with the function-level meaning.
Code reading this needs to know which `source_hostname` is in
scope at line 264-267:

```python
source_cred = get_credential(source_hostname)
remote_hatrac_cache[source_hostname] = HatracStore(...)
```

— the per-asset one. Confusing.

**Fix:** rename the inner variable to `asset_source_host` (also
used at line 186 — `asset_source_hostname`). **LoC: ±0.**
**Severity: low.** Subsumed by §1.1 if the module deletes.

### 6.4 `localize.py` `or` patterns confuse "empty" with "missing"

`localize.py:178, 256, 257` use the same idiom:

```python
current_url = record.get("URL") or record.get("url")
filename = record.get("Filename") or record.get("filename")
md5 = record.get("MD5") or record.get("md5")
```

This is the pattern master audit Lens 5.7 flagged in the broader
audit (`deriva-ml-audit-2026-05.md:789-792`). The semantic
problem: an empty-string value (an actual catalog row with
`URL=""`) collapses to the second branch identically to a
truly-missing column. Either:

- The catalog will never have empty strings (in which case the
  pattern works; the failure mode is invisible).
- The catalog can have empty strings (in which case the code
  silently picks the wrong field).

The right test is `record.get("URL") if "URL" in record else
record.get("url")`. Cleaner: detect the column name once
(line 162-166 already does this) and use the detected name.
Lines 178 and 256-257 then become `record.get(url_column)` —
but the detection logic at 162-166 only runs once on the first
record, so subsequent records use the same `url_column`. Good
— except line 178 still uses `or`. Line 178 fires inside the
loop where the detection has already happened; using
`record.get(url_column)` would work and respect the detection.

**Fix:** use the detected column name uniformly. For `filename`
and `md5`, apply the same detection-once pattern. **Risk: low.**
**LoC: ±0.** **Severity: low.** Subsumed by §1.1 if the module
deletes.

### 6.5 `localize.py` `chunk_size or default_chunk_size` ignores explicit 0

`localize.py:283`:

```python
actual_chunk_size = chunk_size or default_chunk_size
```

If the caller passes `chunk_size=0` (a plausible "no chunking"
spelling), it silently falls back to 50 MB. The function
signature comment says "Optional chunk size in bytes for large
file uploads. If None, uses default chunking behavior." — the
intended sentinel is None, not 0. The current code treats both
the same.

**Fix:** `actual_chunk_size = default_chunk_size if chunk_size is
None else chunk_size`. **Severity: low.** Subsumed by §1.1.

### 6.6 `_extract_hatrac_path` does not validate the URL is HTTP(S)

`localize.py:438`: `parsed = urlparse(url)` runs without checking
`parsed.scheme`. A `file:///hatrac/...` URL would parse and the
function would return a misleading "hatrac path." This is
unlikely in practice (catalog URLs are HTTPS) but the function
is public-ish.

**Fix:** add `if parsed.scheme not in ("http", "https", ""):
return None`. **Severity: low.** Subsumed by §1.1.

---

## Lens 7 — Maintainability / naming

### 7.1 `"ErmrestCatalog"` forward-ref string with no underlying import

`clone_via_bag.py:117, 155`:

```python
def _expand_nested_dataset_anchors(
    anchors: list[Anchor], source_catalog: "ErmrestCatalog"
) -> list[Anchor]:
```

The annotation is a string forward-ref, but `ErmrestCatalog` is
never imported anywhere in the module — not in `TYPE_CHECKING`,
not in the module body. With `from __future__ import annotations`
(which the module has, line 53), the annotation is never
evaluated at runtime, so this works. But:

- Type-checkers (mypy, pyright) cannot resolve the symbol.
- IDE go-to-definition fails.
- A future contributor who tries to evaluate the annotation
  (e.g., for runtime validation) gets a `NameError`.

**Fix:** add an `if TYPE_CHECKING: from deriva.core import
ErmrestCatalog` block. **Risk: trivial. LoC: +3.** **Severity:
low.** Compare `localize.py:29-30` which does exactly this for
the `DerivaML` forward-ref.

### 7.2 `_collect_nested_dataset_rids` catches bare `Exception`

`clone_via_bag.py:171-177`:

```python
try:
    rows = list(dd.filter(dd.Dataset.in_(sorted(frontier))).entities().fetch())
except Exception as e:
    logger.warning(
        "Could not expand nested datasets from %s: %s; proceeding with the seed RID set only",
        sorted(frontier),
        e,
    )
    return collected
```

`Exception` swallows everything: network errors, programmer
errors (typo in column name), interrupts. The function silently
falls back to "use only the seed set" — which then produces a
bag missing nested datasets, which then loads silently with
dangling FKs (mitigated only by the DELETE default per §6.2).
The error gets logged at warning level, but a user inspecting the
clone result has no way to know nested-dataset expansion silently
failed.

**Fix:** narrow to expected exception types (`DataPathException`,
`requests.HTTPError`); re-raise programmer errors. **Risk:
low-medium.** **Severity: low** — the failure path is rare and
the warning log catches it.

### 7.3 `localize.py` uses f-string interpolation in `logger.info`

`localize.py:156, 173, 180, 193, 195, 200, 207, 222, 225, 230,
259, 296, 312, 316, 319, 324, 326, 329` — every log call uses
f-strings. The deriva-ml convention (per master audit Lens 6 and
`core/logging_config.py`) is `logger.info("msg %s", value)` so
the formatting only happens if the log level admits the message.

Counter-example in the same audit scope: `clone_via_bag.py:172,
213, 406, 437` use `logger.info("msg %s", value)` correctly.

`provenance.py:256, 263, 293, 307, 309` are split (some f-strings,
some not).

**Fix:** convert f-string logger calls to lazy-format. **Risk:
trivial.** **LoC: ±0.** **Severity: low** — code quality only.
Subsumed by §1.1 for `localize.py`.

### 7.4 `CloneViaBagResult` is a `@dataclass`; `LocalizeResult` is a `@dataclass` — should be Pydantic

Subsumed by §6.1.

### 7.5 Module docstring sizes

`localize.py:1-15` is 14 lines. `provenance.py:1-25` is 24 lines.
`clone_via_bag.py:1-51` is 50 lines (the longest, with a feature
parity table). `clone.py:1-51` is 50 lines. All reasonable —
no fix.

---

## Persona check

**Senior engineer:**
- §1.7 (`is_clone` always returns False) is a public-API bug, not
  just dead code. Important.
- §6.2 (policy merge silently overrides explicit FAIL) is the most
  significant correctness concern in the module. The merge logic
  needs `model_fields_set`, not value comparison.
- §1.5 (`create_ml_workspace` shim with no working callers) is
  the clearest deletion candidate, but blocked behind the legacy
  MCP retirement timeline.

**Testing engineer:**
- §3.1 (no orphan-strategy integration tests) is the biggest
  coverage gap. The three strategies (FAIL/DELETE/NULLIFY) need
  end-to-end pinning.
- §3.4 (no `CatalogProvenance` round-trip tests) is a smaller
  but cheap gap to close.
- §3.2 (no `localize_assets` tests) — gate the module's
  existence on whether anyone is willing to write a test.

**Technical writer:**
- §4.1 (docstring claim of feature parity is misleading) and §4.2
  (stale `asset_mode="refs"` reference) are user-facing.
- §1.3 / §1.7 / §4.3 — the docstring of `CloneDetails` and
  `is_clone` describes behavior the code doesn't deliver.

**ML-developer user:**
- The clone-then-localize-assets workflow that `localize_assets`
  exists to serve isn't tested and isn't documented anywhere
  outside the module itself. A user wanting this workflow has
  to read the source. Either fix that or delete it.
- `create_ml_workspace`'s docstring promises legacy compatibility
  but the return type makes the promise empty. A user migrating
  from pre-bag-pipeline code gets surprised at the first
  `result.report.summary.errors` access.
- `clone_via_bag`'s policy merge surprise (§6.2): a user who
  passes a policy expecting their choices to be respected gets
  silent overrides on three fields. Documentation should
  explicitly call out the merge defaults *and* the user should
  be able to opt out.

**DBA:**
- §3.1 (no orphan-strategy integration tests) and §6.2 (silent
  FAIL → DELETE override) compound: the strict-mode clone is
  both untested and unreachable through the documented API.
- §1.7 — clone provenance recording is half-built; cloned
  catalogs cannot be reliably distinguished from
  freshly-created ones by reading their annotation.

---

## Ranked actions

Numbered by recommended landing order, balancing impact, risk,
and whether one item subsumes another.

| # | Action | Effort | Risk | LoC | Severity |
|---|---|---|---|---|---|
| 1 | **§6.2 — Switch policy merge to `model_fields_set`.** Replace the "value == library default" predicates with "field name not in `model_fields_set`." Update the existing test to reflect; add a new test asserting explicit FAIL survives. | low | low | ±0 | high |
| 2 | **§1.8 — Route internal provenance imports through `provenance.py` directly.** `src/deriva_ml/__init__.py` (5 paths), `core/base.py:70, 1037` — change the import sources, drop the `clone.py:66-72` re-export. | low | trivial | −10 | low |
| 3 | **§3.4 — Add `tests/catalog/test_provenance.py`.** Round-trip test, unknown-method-fallback test, set/get test with mocked catalog. | low | trivial | +80 (tests) | medium |
| 4 | **§3.1 — Add orphan-strategy integration tests.** Three tests covering FAIL/DELETE/NULLIFY against a deliberately-constructed dangling-FK fixture. | medium-high | low | +200 (tests + fixture) | high |
| 5 | **§1.7 — Decide on the `CloneDetails` future.** Either wire it into `set_catalog_provenance` (param) + `clone_via_bag` (call site at end of clone) OR delete the whole leg per §1.3. The "do nothing" answer is what we have now and is the worst option. | medium | medium | depends | medium |
| 6 | **§1.1 — Decide on `localize_assets`' future.** Add a single integration test (§1.1.b) or delete the module (§1.1.a). | medium | low | −464 (delete) or +60 (test) | high |
| 7 | **§6.1 — Convert `CatalogProvenance`, `CloneDetails`, `CloneViaBagResult`, `LocalizeResult` to Pydantic.** Drops the manual `to_dict`/`from_dict` boilerplate. | medium | medium | −80 | medium |
| 8 | **§5.3 — Switch `_write_provenance_annotation` to model-API form.** Symmetric with the read path. | low | low | ±0 | low |
| 9 | **§4.4 — Audit doctest examples for `# doctest: +SKIP`.** `core/base.py:1028`, `localize.py:104-133`. | low | trivial | ±0 | low |
| 10 | **§1.5 + §1.6 + §1.4 + §1.9 — Delete `clone.py` (or rename to `_clone.py`).** Wait for the legacy MCP retirement signal. Drop `create_ml_workspace`, `_coerce_*`, `_warn_about_legacy_params`, `OrphanStrategy`, the legacy re-exports. | low | medium (timing) | −359 | medium |
| 11 | **§1.2 — Delete unreachable branch in `_extract_hatrac_path`.** Subsumed by §1.1 if delete; otherwise standalone. | trivial | trivial | −3 | low |
| 12 | **§6.3 / §6.4 / §6.5 / §6.6 / §7.3 — `localize.py` quality fixes.** Variable shadowing, `or` ambiguity, chunk_size sentinel, URL scheme check, lazy-format logging. Subsumed by §1.1.a if delete. | low | low | ±0 | low |
| 13 | **§7.1 — Add `ErmrestCatalog` TYPE_CHECKING import in `clone_via_bag.py`.** | trivial | trivial | +3 | low |
| 14 | **§7.2 — Narrow bare `except Exception` in `_collect_nested_dataset_rids`.** | low | low | ±0 | low |

Items 1, 2, 3 are quick wins for the next cleanup PR. Item 4 wants
its own PR because the fixture is non-trivial. Items 5 and 6 are
product decisions (wire or delete) — they should go to the user
as ⏸ NEED INPUT questions before a follow-up audit-action PR.
Items 7-14 are smaller follow-ups.

Combined cleanup ceiling if every "delete" branch lands:
**−460 to −920 LoC** (depending on §1.1 outcome and §1.5 timing),
plus **+280 LoC of tests.** Net is comfortably negative.

---

## Worst-offender modules

1. **`localize.py`** (464 LoC) — zero callers, zero tests,
   exported as public, doc-stale, several quality issues
   inside. Cleanest delete-or-test target in the audit.

2. **`clone.py`** (359 LoC) — pure back-compat shim, no
   working external caller, no internal callers, no tests.
   Delete on the legacy-MCP retirement signal.

3. **`provenance.py`** (319 LoC) — healthy `CatalogProvenance`
   half (the load-bearing read/write API the model template
   uses); fossil `CloneDetails` half (95 LoC dead
   scaffolding) plus a broken `is_clone` property.

4. **`clone_via_bag.py`** (457 LoC) — the load-bearing module.
   One design wart (§6.2 silent merge override) and one
   forward-ref import gap (§7.1). Otherwise in good shape.

5. **`__init__.py`** (66 LoC) — fine in shape, but exports
   four symbols (`localize_assets`, `LocalizeResult`,
   `create_ml_workspace`, `OrphanStrategy`) that nothing in
   the workspace uses live. The docstring sells the package's
   surface as larger than it functionally is.

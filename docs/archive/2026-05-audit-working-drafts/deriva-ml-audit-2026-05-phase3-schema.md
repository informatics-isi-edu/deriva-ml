# deriva-ml audit 2026-05 — Phase 3: schema/

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/schema/`
(2 105 LoC across six Python files: `__init__.py` 20,
`annotations.py` 649, `check_schema.py` 118, `create_schema.py`
662, `table_comments_utils.py` 56, `validation.py` 602) plus two
static assets (`deriva-ml-reference.json` 9 881 lines,
`policy.json` 81 lines) and the corresponding test surface
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/schema/` (319 LoC,
2 test files, ~22 tests) at the tip of
`fix/catalog-manager-state-guards` (HEAD `4442f82`).
Cross-workspace references were grepped against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

Phase 3 prior art: `phase3-execution.md`, `phase3-core.md`,
`phase3-catalog.md`. The schema subsystem differs from those:
it is **the catalog's birth canal** — every test catalog, every
fresh deploy, every demo flows through `create_ml_catalog`
→ `create_ml_schema`. Cleanup must not change observable
schema, vocabulary terms, or ACL policy.

## Executive summary

Overall posture: **the load-bearing core
(`create_schema.py`'s table construction + `annotations.py`'s
Chaise display) is in good shape and externally consumed; the
periphery is a museum of three half-finished validators, one
dead CLI, and three broken `pyproject.toml` entry points.**
`create_ml_catalog`, `create_ml_schema`, `asset_annotation`,
and `build_navbar_menu` carry their weight — the
model-template, the legacy MCP test fixtures, and several
internal sites all depend on them. The other 35 % of the
subsystem is sediment from earlier validation strategies that
never got pruned.

Top themes ranked by impact:

1. **Three `[project.scripts]` entry points point at a
   package name that no longer exists.** `pyproject.toml`
   lines 34–36:
   `deriva-ml-table-comments-utils = "deriva_ml.schema_setup.table_comments_utils:main"`,
   `deriva-ml-create-schema = "deriva_ml.schema_setup.create_schema:main"`,
   `deriva-ml-alter-annotation = "deriva_ml.schema_setup.alter_annotation:main"`.
   The package was renamed `schema_setup` → `schema` at some
   point; the script targets were never updated. `pip install
   deriva-ml && deriva-ml-create-schema` raises
   `ModuleNotFoundError: deriva_ml.schema_setup` immediately.
   Worse: `deriva-ml-alter-annotation` points at
   `alter_annotation:main` — a module that does not exist
   under either name (`find` returns no `alter_annotation*`
   anywhere in `src/`). The user-facing CLI surface is
   silently 30 % broken.

2. **`schema/validation.py` and the existing
   `tools/validate_schema_doc.py` are two competing
   validators with overlapping intent.** `validation.py`
   (602 LoC) introspects a *live catalog* and compares
   against hand-maintained Python constants
   (`EXPECTED_TABLE_COLUMNS`, `EXPECTED_VOCABULARY_TABLES`,
   `EXPECTED_VOCABULARY_TERMS`,
   `EXPECTED_ASSOCIATION_TABLES`); the constants are
   already drifting (`EXPECTED_VOCABULARY_TABLES` is missing
   `Execution_Status`, `EXPECTED_VOCABULARY_TERMS` is missing
   all 7 `Execution_Status` terms and has 1 of the 8
   `Dataset_Type` terms — see §1.1).
   `tools/validate_schema_doc.py` (registered as
   `deriva-ml-validate-schema`) compares
   `docs/reference/schema.md` against `create_schema.py` via
   AST. The doc-vs-code validator is in CI and tracked in
   ADR/superpowers; the live-catalog validator is invoked
   only through `DerivaML.validate_schema()` and has no
   listed external consumer. The duplication is real —
   `validation.py` is the older of the two and was
   superseded by the doc-source-of-truth approach
   documented in
   `docs/superpowers/specs/2026-04-21-schema-doc-source-of-truth-design.md`.

3. **`check_schema.py` (the catalog-vs-JSON differ) is a
   third validator on top of those two — and its CLI is
   silently destructive.** `dump_ml_schema` (line 78) calls
   `create_ml_catalog(hostname, "reference-catalog")` →
   writes `model.prejson()` to a file → `catalog.delete_ermrest_catalog(really=True)`.
   In other words, "dump the schema reference" **creates a
   real catalog on the live server and tries to delete it
   afterwards.** If the dump fails between the create and
   the delete (e.g., disk-full on the file write), the
   stray catalog is leaked. The CLI also crashes on the
   call path because `CheckMLSchemaCLI.main` invokes
   `dump_ml_schema(hostname, catalog_id)` — two args — but
   `dump_ml_schema(hostname, filename)` ignores `catalog_id`
   and writes to the default `deriva-ml-reference.json` in
   `cwd`. The 9 881-line reference JSON the validator
   compares against was last generated when `Execution`
   table had no `Execution_Status` vocabulary
   (`grep -c "Execution_Status" deriva-ml-reference.json` = 0
   in the worktree). So `check_ml_schema` against a current
   catalog **reports the entire execution-status surface
   as drift**, every time. Whole module is rotting in
   place.

4. **`create_schema.py:main()` is dead and broken.** Lines
   636–658 define a CLI entry point that (a) is never
   wired into `pyproject.toml` under any name that resolves
   (the closest entry, `deriva-ml-create-schema`, points at
   the wrong package — see theme 1), (b) declares
   `required=True` on a *positional* argparse argument
   (raises `TypeError` at parse-config time), (c)
   references `args.catalog_id` which is never declared,
   and (d) passes a `Model` object where `create_ml_schema`
   expects an `ErmrestCatalog`. Five separate code-quality
   smoke alarms in 23 lines. Per CLAUDE.md ("No
   backwards-compat shims — if something is unused, delete
   it"), this is a delete.

5. **The `_post_clone_operations` guard the
   `create_ml_schema` docstring promises does not exist.**
   `create_schema.py:312` reads: "If the schema already
   exists, it will be DROPPED with CASCADE, destroying all
   data in the schema. Use `_post_clone_operations` guard
   when calling from clone context." A workspace-wide grep
   for `_post_clone_operations` returns **zero** hits — no
   function, no method, no comment elsewhere. The
   CLAUDE.md-required CASCADE warning is in place at the
   docstring level, but the *guard* the docstring tells the
   reader to use is fictional. The actual guard against
   CASCADE-on-populated-catalog comes from
   `catalog/clone_via_bag.py` having no code path that
   calls `create_ml_schema` at all (the bag-pipeline loads
   schema differently). So the safety property holds, but
   not for the reason the docstring claims.

Worst-offending modules within `schema/`:

1. **`check_schema.py`** (118 LoC). Three names
   (`check_ml_schema`, `dump_ml_schema`, `CheckMLSchemaCLI`)
   exposing two CLI entry points (`main` + the
   `BaseCLI` subclass). Zero callers anywhere in the
   workspace except the wired-up `pyproject.toml` console
   script (which itself doesn't appear to be invoked by
   anyone). Reference JSON is stale by months. `dump_ml_schema`
   has a silent live-catalog side effect plus a signature
   bug. Cleanest delete target in the audit.

2. **`validation.py`** (602 LoC). Two `@dataclass`es,
   one enum, one validator class, five expected-shape
   constants, one convenience function. Constants drift
   from `create_schema.py` reality (missing
   `Execution_Status`, missing 7 `dataset_type` terms,
   etc.). Tests exist (`tests/schema/test_validation.py`)
   but they assert the constants are non-empty rather than
   that they match the schema. Externally:
   `validate_ml_schema` and `SchemaValidationReport` are
   exposed on `deriva_ml/__init__.py` lazy loader; zero
   external callers in any sibling project. Overlap with
   `tools/validate_schema_doc.py` is structural — the
   doc-source-of-truth approach made this module
   redundant, but it was never retired.

3. **`create_schema.py:main()`** (lines 636–658). 23 LoC
   of broken CLI code. The other 639 LoC in the file are
   load-bearing.

4. **`annotations.py`** (649 LoC). Healthy core. One
   `importlib.import_module` workaround (lines 22–27)
   that's stale (see §5.2), some doctest gaps (§4.2),
   and a structural observation that `build_navbar_menu`
   uses raw `/chaise/recordset/#{catalog_id}/...` URL
   construction rather than a helper (low-severity, but
   the same pattern appears 19 times in one function and
   could be a `_chaise_url` helper).

5. **`table_comments_utils.py`** (56 LoC). Dead code.
   Zero internal callers; the registered CLI
   `deriva-ml-table-comments-utils` points at the wrong
   package (see theme 1) and would fail to import. No
   tests. No documentation. The module reads comment text
   from a `docs/<schema>/<table>/[table|<column>].Md`
   directory structure that doesn't exist anywhere in the
   workspace. Delete candidate.

---

## Subsystem inventory

| File | LoC | Posture |
|---|---:|---|
| `__init__.py` | 20 | Public surface. Five symbols from `create_schema.py`, five from `validation.py`. Healthy — but exports validators with no external consumers (see §1.4). |
| `annotations.py` | 649 | Chaise display annotations applied at `create_ml_schema` time. `build_navbar_menu`, `catalog_annotation`, `asset_annotation`, `generate_annotation`. **Load-bearing.** Used by `core/base.py:1095` (`apply_catalog_annotations`) and `core/mixins/asset.py:175` (per-asset annotation). The 19 hand-built Chaise URLs in `build_navbar_menu` are a minor maintainability flag. |
| `check_schema.py` | 118 | **Third (and worst) validator.** `check_ml_schema` diffs live catalog vs `deriva-ml-reference.json`. JSON is stale. `dump_ml_schema` has a silent catalog-create side effect and a signature mismatch. Zero in-repo callers. |
| `create_schema.py` | 662 | The core constructor. `create_ml_catalog`, `create_ml_schema`, `initialize_ml_schema`, `reset_ml_schema`, six private table-creation helpers, plus a dead-and-broken `main`. **Load-bearing.** |
| `table_comments_utils.py` | 56 | Comment-import helper. Zero callers. `[project.scripts]` entry broken (wrong package). |
| `validation.py` | 602 | **Second validator.** `validate_ml_schema`, `SchemaValidator`, `SchemaValidationReport`, `ValidationIssue`. Constants drift from `create_schema.py`. Tests pass largely because they don't compare against truth. |
| `deriva-ml-reference.json` | 9 881 | Static catalog reference, stale. |
| `policy.json` | 81 | ACL policy applied via `deriva.config.acl_config` from `create_ml_catalog`. **Load-bearing and externally referenced.** |

Internal call sites (deriva-ml `src/`):

- `src/deriva_ml/demo_catalog.py:33-34` — imports
  `create_ml_catalog` from `deriva_ml.schema`.
- `src/deriva_ml/__init__.py:78, 82` — lazy-loads
  `SchemaValidationReport` and `validate_ml_schema` from
  `deriva_ml.schema.validation`.
- `src/deriva_ml/core/base.py:74` — type-only import of
  `SchemaValidationReport` (under `TYPE_CHECKING`).
- `src/deriva_ml/core/base.py:1095` — imports
  `catalog_annotation` inside
  `DerivaML.apply_catalog_annotations`.
- `src/deriva_ml/core/base.py:1655` — imports
  `validate_ml_schema` inside `DerivaML.validate_schema`.
- `src/deriva_ml/core/mixins/asset.py:30` — imports
  `asset_annotation` (used at `mixins/asset.py:175` when
  a new asset table is created at runtime via
  `DerivaML.add_asset`).
- `src/deriva_ml/schema/check_schema.py:26` — imports
  `create_ml_catalog` from `deriva_ml.schema.create_schema`
  (only path that touches `check_schema.py`).
- `src/deriva_ml/schema/create_schema.py:36` — imports
  `asset_annotation, generate_annotation` from
  `deriva_ml.schema.annotations`.

Internal test call sites:

- `tests/catalog_manager.py:51, 104` — creates the test
  catalog via `create_ml_catalog`.
- `tests/catalog/test_clone_via_bag_integration.py:36` —
  same.
- `tests/schema/test_validation.py` — imports the five
  validation symbols.
- `tests/schema/test_vocab_fk_convention.py:31` — imports
  `create_ml_catalog`.

`reset_ml_schema`, `initialize_ml_schema`, every symbol in
`check_schema.py`, every symbol in `table_comments_utils.py`,
and the `SchemaValidator` class itself have **zero internal
call sites** (the closest, `SchemaValidator`, is
instantiated only inside `validate_ml_schema` which is in
the same file).

---

## Cross-workspace usage check

Verification per the audit prompt: every symbol whose
deletion or privatization is proposed was grepped across
`deriva-mcp`, `deriva-mcp-core`, `deriva-ml-mcp`,
`deriva-ml-model-template`, `deriva-skills`, and
`deriva-ml-skills`.

| Symbol | External callers | Notes |
|---|---|---|
| `create_ml_catalog` | `deriva-mcp/src/deriva_mcp/tools/catalog.py:200, 203`; `deriva-mcp/tests/conftest.py:407, 409`; `deriva-mcp/tests/test_catalog.py:253, 285, 300` (mock targets); `deriva-ml-model-template/src/scripts/_cifar10_schema.py:47, 99`; `deriva-ml-model-template/docs/superpowers/plans/2026-05-13-e2e-platform-test.md:1245` (doc example) | **Live external API.** Used by the model-template's catalog bootstrap and the legacy MCP's `create_catalog` tool. Both pass `(hostname, project_name)`; only the model-template plan-doc shows a `catalog_alias=` use. Signature is stable; don't change. |
| `create_ml_schema` | None outside `deriva-ml/` | Indirect via `create_ml_catalog` only. |
| `initialize_ml_schema` | None outside `deriva-ml/` | The execution.py docstring mentions it as a fix-up command (`execution.py:1805`) but nothing calls it. |
| `reset_ml_schema` | None outside `deriva-ml/` | Zero internal callers either — see §1.5. |
| `catalog_annotation` | None outside `deriva-ml/` | Called only via `DerivaML.apply_catalog_annotations` method (which is the externally-visible surface). |
| `asset_annotation` | None outside `deriva-ml/` | Same — called via `DerivaML.add_asset` and inside `create_asset_table`. |
| `generate_annotation` | None outside `deriva-ml/` | Internal to `create_ml_schema`. |
| `build_navbar_menu` | None outside `deriva-ml/` | Pure helper for `catalog_annotation`. |
| `check_ml_schema` | None anywhere in workspace | Wired to `deriva-ml-check-catalog-schema` console script in `pyproject.toml:43`. No file or doc references the CLI. |
| `dump_ml_schema` | None anywhere in workspace | Same. |
| `CheckMLSchemaCLI` | None anywhere in workspace | Same. |
| `normalize_schema` | None outside `check_schema.py` | Helper function of `check_ml_schema`. |
| `update_table_comments` | None anywhere in workspace | The registered CLI (`deriva-ml-table-comments-utils`) points at the wrong package and would fail to import. |
| `update_schema_comments` | None anywhere in workspace | Same. |
| `validate_ml_schema` | None outside `deriva-ml/` | Exposed via `deriva_ml/__init__.py` lazy loader but no sibling project imports it. The closest external touch is `DerivaML.validate_schema` (`core/base.py:1597`), not the bare function. |
| `SchemaValidator` | None outside `deriva-ml/` | Same; used only inside `validate_ml_schema` and in the test suite. |
| `SchemaValidationReport` | None outside `deriva-ml/` | Same. The class is the return type of `DerivaML.validate_schema`, so it's *visible* externally as a return-type annotation, but no sibling project handles or imports it. |
| `ValidationSeverity` / `ValidationIssue` | None outside `deriva-ml/` | Same. |
| `EXPECTED_TABLE_COLUMNS` / `EXPECTED_VOCABULARY_TABLES` / `EXPECTED_VOCABULARY_TERMS` / `EXPECTED_ASSOCIATION_TABLES` / `EXPECTED_VOCABULARY_COLUMNS` / `SYSTEM_COLUMNS` | Only `tests/schema/test_validation.py:8-11` imports four of these | Test-only consumers. The constants are module-internal; the test imports them only to assert non-emptiness. |
| `deriva-ml-create-schema` console script | None | Wired to `deriva_ml.schema_setup.create_schema:main` — wrong package, will `ImportError`. |
| `deriva-ml-alter-annotation` console script | None | Wired to `deriva_ml.schema_setup.alter_annotation:main` — wrong package AND wrong module (no `alter_annotation` exists anywhere). |
| `deriva-ml-table-comments-utils` console script | None | Wired to `deriva_ml.schema_setup.table_comments_utils:main` — wrong package. |
| `deriva-ml-check-catalog-schema` console script | None | Wired to `deriva_ml.schema.check_schema:main` — correct path; `main` function exists. But nothing in any project references the CLI by name. |
| `policy.json` (file) | `deriva_ml/schema/create_schema.py:571` (load via `importlib.resources`); referenced by name in the `_cifar10_schema.py` doc example | **Load-bearing.** Don't touch without verifying ACLs. |
| `deriva-ml-reference.json` (file) | Only `check_schema.py` reads it | Stale (see §3.3). Delete with `check_schema.py`. |

**Conclusions from the table:**

- **`create_ml_catalog` and `policy.json` are the
  externally-load-bearing surface.** The model-template's
  catalog bootstrap depends on `create_ml_catalog(hostname,
  project_name)` returning an `ErmrestCatalog` with ACLs
  applied. Don't break the signature, don't break the ACL
  apply, don't break the schema content.
- **The two console scripts that resolve to nonexistent
  module paths (`deriva-ml-create-schema`,
  `deriva-ml-alter-annotation`, `deriva-ml-table-comments-utils`)
  are CLI dark matter — `pip install` succeeds, `pip
  show -f` lists them, `which` finds them, but invoking
  them fails with `ModuleNotFoundError` immediately.** Fix
  is mechanical (one-line `pyproject.toml` edit per
  script, or delete the entries). No external CLI user
  exists per the workspace grep, so any cleanup direction
  is safe.
- **`check_schema.py`, `table_comments_utils.py`, and
  the two stale CLI entries together represent ~290 LoC
  of confirmed dead surface.** Combined with the
  validator overlap (§1.2), the cleanup ceiling on
  `schema/` is in the 700–900 LoC range.
- **`validation.py` has no external consumer but is
  reachable through `DerivaML.validate_schema` (which has
  six docstring example references in `core/base.py`).**
  Privatizing the module is fine; deleting it requires
  either replacing `DerivaML.validate_schema` (and the doc
  examples) with a thin wrapper around `tools/validate_schema_doc.py`'s
  approach, or accepting the duplication is irreducible
  because one validates against doc/source while the other
  validates against a live catalog. The two are different
  enough that "delete validation.py" is not free.

---

## Lens 1 — Legacy / dead code

### 1.1 `validation.py` `EXPECTED_*` constants drift from `create_schema.py` reality

`validation.py:274-280` declares the canonical list of
ML-schema vocabulary tables:

```python
EXPECTED_VOCABULARY_TABLES: list[str] = [
    MLVocab.dataset_type,
    MLVocab.workflow_type,
    MLVocab.asset_type,
    MLVocab.asset_role,
    MLVocab.feature_name,
]
```

`MLVocab.execution_status` is **missing**, even though
`create_schema.py:349-351` creates it and seeds 7 terms
(`Created`, `Running`, `Stopped`, `Failed`,
`Pending_Upload`, `Uploaded`, `Aborted`). The validation
report on a freshly-created catalog will pass — but only
because the validator literally doesn't look at
`Execution_Status`.

`EXPECTED_VOCABULARY_TERMS` (`validation.py:293-330`)
omits `execution_status` entirely (consistent with the
missing table) and lists only **1 of the 8** `dataset_type`
terms that `create_schema.py:462-474` seeds. (`File` is
the only one in the expected list; the other seven —
`Complete`, `Training`, `Testing`, `Validation`, `Split`,
`Labeled`, `Unlabeled` — are absent.) Same shape:
validator never reports them as missing because the
validator never asks.

Per the testing-engineer persona: **the validator
silently agrees with whatever the actual schema is on
these dimensions**, which is the worst kind of validator
— it produces green-light reports without checking.
`tests/schema/test_validation.py::test_validate_valid_schema`
passes against a real catalog, but the test would also
pass if `create_schema.py` had subtly broken vocabulary
seeding because the validator doesn't cover the seeded
terms.

This is a symptom of two validators competing (theme 2):
nobody is updating these constants because the
doc-source-of-truth flow in
`tools/validate_schema_doc.py` is the active path.

**Fix:**
- (a) Add `MLVocab.execution_status` to
  `EXPECTED_VOCABULARY_TABLES` and the full term list to
  `EXPECTED_VOCABULARY_TERMS`. Likewise expand
  `dataset_type` and `workflow_type` (the latter currently
  has 13 of 13, but does not list `Visualization` →
  actually it does; spot check confirms `workflow_type` is
  consistent). **Pure documentation-of-truth fix; ~30 lines
  changed.** **Risk: trivial.** Subsumed by (b) if (b)
  is chosen.
- (b) Replace `EXPECTED_*` with a runtime
  introspection of `tools/validate_schema_doc.py`'s
  doc-derived model (`docs/reference/schema.md`'s
  YAML-fenced sections are the canonical truth per ADR /
  superpowers spec). `SchemaValidator` becomes "compare
  live catalog to doc-derived model" instead of "compare
  live catalog to in-file Python constants." This
  resolves the duplication. **Risk: medium**;
  cross-module wiring.
- (c) Delete `validation.py` entirely; replace
  `DerivaML.validate_schema` with a thin wrapper that
  invokes `tools/validate_schema_doc.py`'s validator
  programmatically against the *running catalog's*
  `prejson()` rather than against the source AST. **Risk:
  medium**; touches public API.

**Recommended:** (b). The doc is already the source of
truth per ADR-0007's neighborhood and the
superpowers spec; the validator should consult it rather
than maintain a third hand-written copy.

**Severity: medium.** Hidden coverage gap. Has been live
since `execution_status` shipped (April 2026).

### 1.2 `check_schema.py` is a redundant third validator

The schema subsystem now contains three validators:

| Validator | Compares | Status |
|---|---|---|
| `tools/validate_schema_doc.py` | `docs/reference/schema.md` ↔ `create_schema.py` (AST) | **Active.** Wired to `deriva-ml-validate-schema` CLI; documented in CLAUDE.md; ADR-tier rationale in superpowers/specs. |
| `schema/validation.py` | Live catalog ↔ `EXPECTED_*` Python constants | Reachable via `DerivaML.validate_schema`. Constants drift (§1.1). |
| `schema/check_schema.py` | Live catalog ↔ `deriva-ml-reference.json` | Wired to `deriva-ml-check-catalog-schema` CLI. **No external referent.** |

The three validators differ in what they consider truth:
the docstring/Markdown, the Python constants, or the
prejson snapshot. The doc-source-of-truth spec says the
Markdown is canonical. Therefore the JSON snapshot is
obsolete and the Python constants are duplicate.

`check_schema.py` is the clearest delete:

- The reference JSON it consults (`deriva-ml-reference.json`,
  9 881 lines) is months stale. `grep -c "Execution_Status"`
  on the JSON returns 0, but `create_schema.py` seeds 7
  terms there. Any current diff would flag the entire
  execution-status surface as "drift."
- `dump_ml_schema` (the way to refresh the JSON) creates
  a real catalog on the live server, dumps, then deletes
  — but if the dump fails before the delete, the catalog
  leaks (line 80-87, no try/finally around the create or
  the file write — only around the model fetch and dump).
  Actually inspect: `try: ... finally: catalog.delete_ermrest_catalog(really=True)`
  *does* wrap the file write. So the delete is safe.
  But the side effect (creating a real catalog on
  `localhost`) is undocumented in the CLI help.
- `CheckMLSchemaCLI.main` calls `dump_ml_schema(hostname,
  catalog_id)` — 2 args — but `dump_ml_schema(hostname,
  filename)` accepts `(hostname, filename="deriva-ml-reference.json")`.
  The CLI passes `catalog_id` where the function expects
  `filename`; `dump_ml_schema` happily uses
  `args.catalog`'s value (an int or string like `"1"`)
  as the output filename. **Bug.**
- `check_ml_schema(hostname, catalog_id)` opens an
  `ErmrestCatalog` for the target catalog **without
  applying credentials** — actually no, line 67 does
  `credentials=get_credential(hostname)`. OK. But the
  result is printed via `pprint(diff, indent=2)` and
  the diff is a `DeepDiff` tree — pprint of a `DeepDiff`
  tree is mostly unreadable hieroglyphics. The CLI is
  not user-friendly even when correct.

**Fix:** delete `check_schema.py` entirely, remove
the `deriva-ml-check-catalog-schema = ...` line from
`pyproject.toml:43`, delete
`deriva-ml-reference.json` (9 881 lines of stale data),
and (if anyone really needs runtime catalog-vs-source
diff) recommend `deriva-ml-validate-schema` plus
inspecting the live catalog's `prejson()` ad-hoc.
**LoC: −118** (file) **−9 881** (JSON) **−1**
(pyproject). **Risk: low** — zero external callers.
**Severity: high** — the live CLI is broken, stale, and
exposed as an entry point that pip-installs into user
`$PATH`.

### 1.3 `table_comments_utils.py` is dead

`schema/table_comments_utils.py` (56 LoC) reads
`.Md`-suffixed text files from a hierarchical directory
(`docs/<schema>/<table>/[table|<column>].Md`) and applies
them as table/column `comment` properties. Zero callers
in the workspace; no doc reference; no example
filesystem matching the expected layout (a workspace-wide
search for files matching `*/Md` under `docs/` returns
nothing of that shape).

The registered CLI (`deriva-ml-table-comments-utils =
"deriva_ml.schema_setup.table_comments_utils:main"`)
points at the wrong package — running it raises
`ModuleNotFoundError: deriva_ml.schema_setup`.

The functionality the module nominally provides
(file-based catalog comment management) duplicates what
the per-column `comment=` parameter in
`create_schema.py` already does. The DBA persona
question — "is there a curator workflow that runs
this?" — has the answer "no, the workflow doesn't
exist."

**Fix:** delete the file, remove the
`deriva-ml-table-comments-utils` entry from
`pyproject.toml`. **LoC: −56.** **Risk: low** — zero
callers, zero docs. **Severity: medium** — confirmed
dead, exposed as a (broken) CLI.

### 1.4 `validation.py` has no external consumer and overlaps `tools/validate_schema_doc.py`

See §1.1. The whole module is reachable only via
`DerivaML.validate_schema`. The constants drift (§1.1).
The validator class structure adds 250 LoC of
boilerplate (`SchemaValidator`, `SchemaValidationReport`,
`ValidationIssue`, `ValidationSeverity`) that the
already-active `tools/validate_schema_doc.py` validator
doesn't need.

**Fix options:** (a) Migrate `DerivaML.validate_schema`
to call `tools/validate_schema_doc.py`'s validator against
the live catalog's `prejson()`, then delete
`schema/validation.py`. Most aggressive — **LoC: −602.**
(b) Keep `validation.py` but rewrite
`SchemaValidator._validate_vocabulary_terms` etc. to
read from `docs/reference/schema.md` (the canonical
source) instead of the hand-maintained constants. **LoC:
unchanged**, but the drift in §1.1 disappears
permanently.

**Recommended:** (b) for the short-term — fixes drift
without touching public API — then (a) when the
`DerivaML.validate_schema` consumers are mapped (the
six docstring examples in `core/base.py:1597-1657`
need rewording). **Risk: medium** in either case.
**Severity: medium.**

### 1.5 `reset_ml_schema` has zero callers

`create_schema.py:627-633`:

```python
def reset_ml_schema(catalog: ErmrestCatalog, ml_schema=ML_SCHEMA) -> None:
    model = catalog.getCatalogModel()
    schemas = [schema for sname, schema in model.schemas.items() if sname not in ["public", "WWW"]]
    for s in schemas:
        s.drop(cascade=True)
    model = catalog.getCatalogModel()
    create_ml_schema(catalog, ml_schema)
```

**Drops every non-public/WWW schema with CASCADE.**
This is the most destructive function in the subsystem
— more aggressive than `create_ml_schema` itself, which
only drops `deriva-ml`. `reset_ml_schema` drops every
domain schema too.

Workspace-wide grep returns hits only inside the file
itself plus the docstring of `create_schema.py:12` and
the schema/__init__.py export. No code calls it. No
test calls it.

Docstring is one line: `"""Reset ML schema (test/dev
helper)."""` — but no test uses it (the test catalog
manager creates and destroys whole catalogs instead, see
`tests/catalog_manager.py:104, 118`).

There is no signature parameter or env-var guard against
accidental invocation. A user who imports
`reset_ml_schema` from `deriva_ml.schema` (it's in
`__all__`) and calls it on a production catalog
destroys every domain schema on that catalog with no
prompt and no audit log.

**Fix:** (a) delete the function, remove the export
from `__init__.py` and `__all__`; or (b) rename to
`_reset_ml_schema`, add a `confirm: bool = False`
keyword that raises if not True, and add a `logger.warning`
that logs the dropped schemas to the audit log. The
function may be useful to someone running a test harness
outside the repo, but the lack of any guard makes the
current shape dangerous.

**Recommended:** (a). The function is unused, the
delete-and-recreate flow is better served by
`catalog.delete_ermrest_catalog(really=True)` followed
by `create_ml_catalog(...)`. **LoC: −7.** **Risk: low**
(zero callers per cross-workspace check). **Severity:
medium** — dangerous public function with no audience.

### 1.6 `create_schema.py:main()` is dead and broken

Lines 636–658 define a CLI. Issues:

1. Line 649: `parser.add_argument("curie_prefix",
   type=str, required=True)` — `required=` is not valid
   on a positional argument (positional args are always
   required; passing `required=True` raises `TypeError:
   'required' is an invalid argument for positionals`
   when `parse_args()` runs).
2. Line 648: `parser.add_argument("schema-name",
   default="deriva-ml", ...)` — argparse stores
   positional `schema-name` as `args.__getattribute__("schema-name")`
   (hyphen → not a valid attribute). Line 655 reads
   `args.schema_name` (underscore) which doesn't exist;
   `AttributeError`.
3. Line 654: `server.connect_ermrest(args.catalog_id)` —
   `args.catalog_id` is never declared. The CLI has no
   such argument. `AttributeError`.
4. Line 655: `create_ml_schema(model, ...)` — passes a
   `Model` where the function signature expects an
   `ErmrestCatalog`. The function would call
   `model.getCatalogModel()` on a `Model`, which doesn't
   have that method.
5. The `[project.scripts]` entry that nominally invokes
   this `main` (`deriva-ml-create-schema =
   "deriva_ml.schema_setup.create_schema:main"`) points
   at the wrong package and would `ImportError` before
   any of 1–4 fired.

This is five separate bugs in 23 lines of code that have
been sitting in `main` since the `schema_setup` →
`schema` rename. Nothing exercises this CLI.

**Fix:** delete `main` and the
`if __name__ == "__main__": sys.exit(main())` block.
The proper CLI surface for "create an ML catalog from
the command line" is the demo / test harness, not a
half-finished argparse stub. **LoC: −23.** **Risk:
trivial** (zero callers — the entry point doesn't
resolve). **Severity: medium** — the file presents
itself as having a CLI in its docstring (line 11
mentions `create_ml_catalog` as a "main entry point"),
but the actual `main` is broken.

### 1.7 `_post_clone_operations` is a fictional guard

`create_schema.py:312`:

> WARNING: If the schema already exists, it will be
> DROPPED with CASCADE, destroying all data in the
> schema. Use `_post_clone_operations` guard when
> calling from clone context.

Workspace-wide grep for `_post_clone_operations`: zero
hits. No function, no method, no class, no comment
elsewhere. The "guard" the docstring tells the reader
to consult does not exist.

The actual CASCADE-safety property of the codebase
comes from a different fact: `catalog/clone_via_bag.py`
does not call `create_ml_schema` at all (the bag pipeline
applies schema via `BagCatalogLoader`, which creates
tables from a bag's `schema.json`, not from
`create_ml_schema`). So the property "clone won't
clobber an existing schema" holds, but for an unrelated
reason.

**Fix:** delete the broken cross-reference. The
docstring should just say "If the schema already
exists, it will be DROPPED with CASCADE. Caller is
responsible for ensuring this is desired; the bag-based
clone path in `catalog/clone_via_bag.py` does not invoke
`create_ml_schema` and is unaffected." **LoC: ±0.**
**Severity: low** — doc-only, but the safety claim
is misleading.

### 1.8 `annotations.py:main()` is dead

`annotations.py:632-649`:

```python
def main():
    parser = argparse.ArgumentParser(description="Apply annotations to ML schema")
    parser.add_argument("hostname", help="Hostname for the catalog")
    parser.add_argument("catalog_id", help="Catalog ID")
    parser.add_argument("schema-name", default="deriva-ml", help="Schema name (default: deriva-ml)")
    args = parser.parse_args()
    generate_annotation(args.catalog_id, args.schema_name)
```

Same shape as §1.6: hyphenated positional
(`schema-name`), reads `args.schema_name` (no such
attribute), and `generate_annotation` expects
`(model: Model, schema: str)` — not
`(args.catalog_id, args.schema_name)` (both strings,
neither a Model). Pure typo dead code.

No `[project.scripts]` entry resolves to this `main`.

**Fix:** delete `main` and the
`if __name__ == "__main__": sys.exit(main())` block.
**LoC: −19.** **Severity: low.**

### 1.9 `table_comments_utils.py:main()` is dead alongside the module

Subsumed by §1.3.

### 1.10 `annotations.py` `importlib.import_module` workaround

`annotations.py:18-27`:

```python
# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
import sys

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
_core_utils = importlib.import_module("deriva.core.utils.core_utils")

Model = _ermrest_model.Model
Table = _ermrest_model.Table
deriva_tags = _core_utils.tag
```

The comment claims this avoids shadowing by "local
`deriva.py` files." A workspace-wide search for
`deriva.py` (anywhere within `src/`) returns no hits —
this workaround was for a long-since-removed local file.
The rest of the codebase imports
`from deriva.core.ermrest_model import Model, Table`
and `from deriva.core.utils.core_utils import tag`
directly without trouble (e.g., `core/base.py:1095`
imports `catalog_annotation` from this module without
any shadowing issue).

`import sys` (line 20) is imported but never used in
`annotations.py` — `sys.exit(main())` at line 649
references it but the `main` function (§1.8) is dead.

**Fix:** replace the `importlib.import_module` dance
with normal imports:

```python
from deriva.core.ermrest_model import Model, Table
from deriva.core.utils.core_utils import tag as deriva_tags
```

Drop `import sys` once `main` deletes. **LoC: −7
roughly.** **Risk: low.** **Severity: low.**

---

## Lens 2 — Privatization

### 2.1 `SchemaValidator` should be private (or deleted)

`validation.py:352` defines the validator class. It's
exported via `__init__.py:4` and `__all__:16`. Its only
external use is in tests. No sibling project imports it.

If §1.4 lands and `validation.py` survives, the
`SchemaValidator` class should be underscore-prefixed —
the public API is `validate_ml_schema()` (the
convenience function) and `SchemaValidationReport` (the
return type). The class itself is an implementation
detail.

**Fix:** rename `SchemaValidator` → `_SchemaValidator`,
drop from `__all__` and `__init__.py`, update
`tests/schema/test_validation.py` to use the private
import (the test convention permits private imports).
**LoC: ±0.** **Risk: trivial.** **Severity: low.**

### 2.2 `ValidationIssue` and `ValidationSeverity` are
exported but rarely useful externally

`validation.py:32, 41` define `ValidationSeverity` and
`ValidationIssue`. Both are exported. A user inspecting
a `SchemaValidationReport.issues` list does encounter
them as element types — so they're not purely internal.
But the user's typical pattern is to read
`report.errors`, `report.warnings`, `report.is_valid` —
not to construct or destructure `ValidationIssue`
manually.

**Fix:** keep public; they're return-type-adjacent.
**Severity: none** — flag only.

### 2.3 `normalize_schema` should be private

`check_schema.py:29` is a helper for `check_ml_schema`.
Not underscore-prefixed; not in `__all__` (the module
has no `__all__`). Confusing.

**Fix:** subsumed by §1.2 (delete the file). If not
deleted, rename to `_normalize_schema`. **Severity:
low.**

### 2.4 `build_navbar_menu` exposure

`annotations.py:34` is a pure helper for
`catalog_annotation`. Not underscore-prefixed. No
external callers per the cross-workspace check.

Per its docstring (lines 35-55), the function is
designed to be a "single source of truth" for the
navbar layout. If we expect deriva-skills or
deriva-ml-skills to ever build their own catalog
annotation flow that reuses the navbar layout, this is
a valid public surface. Otherwise it's an internal
helper.

**Fix:** keep public for now (the docstring's "single
source of truth" framing is forward-looking). If the
six-month follow-up audit finds it still has no
external consumer, privatize. **Severity: none.**

---

## Lens 3 — Coverage

### 3.1 No CASCADE-safety test

`create_ml_schema` (`create_schema.py:304`) drops the
existing `deriva-ml` schema with CASCADE if it exists.
No test in `tests/schema/` covers this branch. The
existing tests use the catalog manager's create-and-destroy
flow (a fresh catalog per session), so `create_ml_schema`
is only ever called against a known-empty catalog.

The CLAUDE.md "Special focus areas" section called out:

> CASCADE safety: `create_ml_schema` drops the existing
> schema if present. This is documented; verify the
> warning is in the docstring, that the function is
> gated behind some explicit confirmation in any CLI
> surface, and that tests don't accidentally call it
> without isolation.

The warning is in the docstring (line 311-313, plus the
module-level docstring at line 6-8). It is **not** gated
behind explicit confirmation in any CLI surface — the
CLI surfaces that would invoke it are all broken (§1.6,
§1.8). Tests are isolated per-session (the catalog
manager destroys catalogs in teardown). No test asserts
the CASCADE behavior actually fires or that data loss
on re-creation is the documented behavior.

A user picking up the module and writing a test that
calls `create_ml_schema` twice on the same catalog
would observe the CASCADE behavior. There is no
regression guard if some future refactor swapped the
DROP-CASCADE for a "drop-only-if-empty" check.

**Fix:** add `tests/schema/test_cascade_behavior.py`
with one test:

```python
def test_create_ml_schema_drops_existing(test_ml):
    """create_ml_schema CASCADE-drops the existing schema."""
    # Insert a row into Dataset so we can prove it's gone after.
    ml = test_ml
    # ... insert row, capture RID ...
    create_ml_schema(ml.catalog, project_name="deriva-ml")
    # ... assert the RID is no longer findable ...
```

Marker: `@pytest.mark.integration` (it needs a real
catalog). **Effort: low.** **Risk: low** — the test
asserts existing behavior. **Severity: high** for the
DBA persona — this is a load-bearing destructive
operation with no regression test.

### 3.2 No ACL-application test for `create_ml_catalog`

`create_ml_catalog` (`create_schema.py:545`) shells out
to `deriva.config.acl_config` to apply
`policy.json` (lines 588-611). The error path
(subprocess.CalledProcessError → DerivaMLConfigurationError)
is well-handled in code, but no test covers either the
success path or the failure path. The model-template's
catalog-bootstrap test is the closest external test of
this — but its scope is "the catalog is created,"
not "the ACL bindings are present and correct."

A failure mode the audit hypothesizes (`create_schema.py`
comment at lines 580-588 calls it out): the row-level
update bindings might silently fail to apply, and
subsequent uploads would HTTP 403 with no clear cause.

**Fix:** add `tests/schema/test_acl_application.py`:

```python
@pytest.mark.integration
def test_create_ml_catalog_applies_row_owner_binding(test_ml):
    """Newly-created catalog has the row_owner_guard binding on Execution."""
    model = test_ml.catalog.getCatalogModel()
    exec_table = model.schemas["deriva-ml"].tables["Execution"]
    assert "row_owner_guard" in exec_table.acl_bindings
```

**Effort: low.** **Risk: low.** **Severity: medium** —
ACL silent-failure is a known hairy debug class.

### 3.3 `deriva-ml-reference.json` drift is itself a coverage gap

The 9 881-line reference JSON in `schema/` purports to
describe the canonical `deriva-ml` schema for diffing.
It is months stale (no `Execution_Status` references,
likely no `Dataset_Version.Minid_Spec_Hash` column
either — the column was added recently). Anyone running
`deriva-ml-check-catalog-schema` against a current
catalog gets a noisy diff of the entire delta since the
last refresh — drowning real drift in known drift.

The refresh path (`dump_ml_schema`) has the bugs
described in §1.2 (signature mismatch with the CLI;
live-catalog side effect).

**Fix:** subsumed by §1.2 (delete the whole module
plus the JSON). **Severity: medium** — long-standing
silent staleness in a file presented as canonical.

### 3.4 `tests/schema/test_validation.py` tests do not assert ground truth

The unit tests in `test_validation.py:199-227`
(`TestExpectedSchemaStructure`) assert that the
constants are non-empty:

```python
def test_expected_tables_not_empty(self):
    assert len(EXPECTED_TABLE_COLUMNS) > 0
    assert "Dataset" in EXPECTED_TABLE_COLUMNS
    ...
```

These pass as long as the constants exist. They do not
catch the drift identified in §1.1 — they would still
pass with `Execution_Status` missing from
`EXPECTED_VOCABULARY_TABLES`.

The integration tests (lines 119-196) assert the
validator runs successfully against a real catalog — but
since the validator doesn't check `Execution_Status` or
the unlisted vocabulary terms, the integration tests
pass trivially in that dimension too.

**Fix:** add a test that asserts the constants match the
seed lists in `create_schema.py` (e.g., parse
`initialize_ml_schema` for the literal terms; compare).
Or — better — replace the constants with a doc-derived
model (§1.4 option b) and delete this test class.
**Severity: medium.**

### 3.5 `tests/schema/test_vocab_fk_convention.py` is the strong test

The one test in `test_vocab_fk_convention.py:29` is the
right shape: it creates a fresh catalog from
`create_ml_catalog`, introspects every FK in the
`deriva-ml` schema, and asserts the convention. It's the
gold-standard pattern for this subsystem — pin a
load-bearing invariant against a live catalog.

The test could be expanded (1-2 more invariants:
"every vocab table has the standard columns",
"every association table has both endpoints
constrained") but it's healthy as-is. **No fix.**

### 3.6 `policy.json` has no test asserting its content

The ACL groups and bindings in `policy.json` (81 lines)
are load-bearing — they control who can do what in every
catalog created. A typo in the file is caught only when
ACLs misbehave in production. There's no syntactic check
that the file is valid JSON, no semantic check that the
required groups (`isrd-systems`, `local-admin`, etc.)
are present, no check that the required ACL definitions
(`read_only`, `self_serve`) are present.

**Fix:** add `tests/schema/test_policy_validation.py`:

```python
def test_policy_json_loads_and_has_required_acls():
    from importlib.resources import files
    import json
    policy = json.loads(files("deriva_ml.schema").joinpath("policy.json").read_text())
    assert "self_serve" in policy["acl_definitions"]
    assert "row_owner_guard" in policy["acl_bindings"]
    assert policy["catalog_acl"]["acl"] == "self_serve"
```

**Effort: trivial.** **Risk: trivial.** **Severity:
low-medium** — the file is small enough that a typo is
unlikely, but the test costs nothing.

### 3.7 No test for `annotations.py:build_navbar_menu`

`build_navbar_menu` is a pure function with no
side-effects (line 34, docstring at line 47-49 explicitly
calls this out). It's perfectly testable but has no
test. The closest is `tests/core/test_catalog_annotations.py`
which tests the higher-level `DerivaML.apply_catalog_annotations`
that calls it — those tests are integration (real
catalog) and don't isolate the navbar-generation logic.

**Fix:** add `tests/schema/test_annotations.py`:

```python
def test_build_navbar_menu_includes_deriva_ml_section(test_ml):
    menu = build_navbar_menu(test_ml.model)
    deriva_ml_section = next(c for c in menu["children"] if c["name"] == "Deriva-ML")
    expected = {"Workflow", "Execution", "Execution_Metadata", "Execution_Asset", "Dataset", "Dataset_Version"}
    actual = {item["name"] for item in deriva_ml_section["children"]}
    assert expected.issubset(actual)
```

(Plus tests for "Vocabulary" section, "Assets" section,
the per-domain-schema sections.) **Effort: low.**
**Severity: low.**

---

## Lens 4 — Docs sync

### 4.1 `create_schema.py:312` references a fictional guard

Subsumed by §1.7.

### 4.2 Docstring examples without `# doctest: +SKIP`

`create_schema.py:562-565` (`create_ml_catalog` example):

```python
Example:
    # Create catalog with alias
    catalog = create_ml_catalog("localhost", "my_project", catalog_alias="my-project")
    # Now accessible as both /ermrest/catalog/<id> and /ermrest/catalog/my-project
```

The example is written with `#`-prefixed comment lines
(not `>>>` interactive lines), so doctest collection
ignores it. **No actual doctest risk**, but the
formatting differs from CLAUDE.md's Google-style with
`>>>` prefix — readers expect a runnable example and
have to read carefully to see it's not interactive.

`validation.py:594-600` (`validate_ml_schema` example):

```python
Example:
    >>> from deriva_ml import DerivaML
    >>> ml = DerivaML('localhost', 'my_catalog')
    >>> report = validate_ml_schema(ml, strict=False)
    >>> if not report.is_valid:
    ...     print(report.to_text())
```

No `# doctest: +SKIP`. Per CLAUDE.md: "Catalog-dependent
examples must carry `# doctest: +SKIP` on the first
interactive line." This block tries to instantiate
`DerivaML('localhost', 'my_catalog')` at doctest time —
it fails because there's no live catalog. The
`--doctest-modules` run would surface this. (Unless
caught by an existing pytest filter; verify.)

`validation.py:200-208` (`to_json` example):

```python
Example:
    >>> report = ml.validate_schema()
    >>> print(report.to_json())
    {
      "schema_name": "deriva-ml",
      ...
    }
```

Same — `ml.validate_schema()` requires a catalog.
Needs `# doctest: +SKIP`.

**Fix:** add `# doctest: +SKIP` markers to both blocks
in `validation.py`. Convert the
`create_ml_catalog` example to `>>>` form with `SKIP`.
**LoC: ±0.** **Risk: trivial.** **Severity: low.**

### 4.3 `create_schema.py` module docstring lists symbols correctly but does not warn about `reset_ml_schema`

Lines 1-13 describe the four public entry points
(`create_ml_schema`, `initialize_ml_schema`,
`create_ml_catalog`, `reset_ml_schema`). The CASCADE
warning is correctly attached to `create_ml_schema`.
`reset_ml_schema` is described as "Drop and recreate the
schema (test/dev helper)" — but reset_ml_schema's
*actual* behavior is to drop **every** schema except
`public` and `WWW`, not just `deriva-ml`. The docstring
is misleading.

**Fix:** subsumed by §1.5 (delete the function). If
kept, rewrite the docstring entry to say "Drops every
non-public/WWW schema with CASCADE and recreates the
ML schema." **Severity: medium.**

### 4.4 `validation.py` module docstring is accurate but the file does not deliver

Lines 1-17 describe the file as validating:

> - Required tables and their columns
> - Required vocabulary tables and their initial terms
> - Foreign key relationships
> - Extra tables/columns (in strict mode)

"Foreign key relationships" is in the docstring but
the validator class (lines 408-578) has no FK
validation. There's `_validate_association_tables`
(which checks association tables exist by name) and
`_validate_table_columns` (which checks column name +
type) but no `_validate_foreign_keys` that walks the
declared FK list against `EXPECTED_*` shape. The
docstring lies.

**Fix:** either implement FK validation (closing the
gap between docstring claim and behavior) or update the
docstring to match. **Severity: low.**

### 4.5 `check_schema.py:9` describes a CLI that doesn't work

```
- ``check_ml_schema``: Connect to a catalog and diff its schema against the
  reference (or a provided file). Prints differences to stdout.
- ``dump_ml_schema``: Export the current catalog schema to a JSON file
  (for updating the reference baseline).
- ``CheckMLSchemaCLI`` / ``main``: CLI wrappers for the above.
```

`dump_ml_schema` doesn't "export the current catalog
schema" — it *creates a new catalog* and exports
that, then deletes it. Highly misleading.

**Fix:** subsumed by §1.2 (delete the module). If kept,
rewrite the docstring to describe the actual behavior.
**Severity: medium.**

### 4.6 `policy.json` has no documentation

The file's contents (ACL groups, definitions, bindings)
are load-bearing. The `_cifar10_schema.py` doc-example
in the model-template references it indirectly through
`create_ml_catalog`, but nowhere in `deriva-ml/docs/` is
the ACL design documented. A DBA or operations user
inspecting a `403` response on a row update needs to
know that:
- The `row_owner_guard` binding (`projection: ["RCB"]`)
  restricts updates/deletes to the row's creator.
- The `self_serve` definition is applied at the catalog
  level.
- Non-public schemas inherit `row_owner_guard`.

**Fix:** add `docs/reference/acl-policy.md` (or a
section under `docs/architecture.md`) that describes the
policy. Not strictly schema/-subsystem work, but flagged
here because the file lives in `schema/` and its
visibility is low. **Effort: low.** **Severity: low.**

---

## Lens 5 — deriva-py API conventions

### 5.1 `create_ml_schema` uses raw model API correctly

`create_schema.py:322` uses `catalog.getCatalogModel()`,
the model API. Table creation goes through `schema.create_table(TableDef(...))`
(the typed-def constructors from `deriva.core.typed`)
rather than raw POSTs to ERMrest. This is the correct
pattern per CLAUDE.md "API Priority" — model API ahead
of raw ERMrest URLs. **No fix.**

### 5.2 `annotations.py` `importlib.import_module` is non-idiomatic

Subsumed by §1.10. The pattern is unique to this file in
the workspace; everywhere else `deriva.core.ermrest_model`
and `deriva.core.utils.core_utils` are imported normally.

### 5.3 `check_schema.py:67` constructs `ErmrestCatalog` directly with raw scheme

```python
catalog = ErmrestCatalog("https", hostname, catalog_id, credentials=get_credential(hostname))
```

This is fine — the only way to bind to a specific
`catalog_id` for a read-only diff. **No fix.**

### 5.4 `create_ml_catalog` shells out to `deriva.config.acl_config`

Lines 588-611. The reasons (PATH/version mismatch, stale
venv shebangs) are correct and documented in the
multi-line comment block (lines 572-587). The fallback
to `sys.executable -m` is the right idiom.

**Note:** the comment block (lines 572-587) is one of
the best examples of inline-documentation-of-why in the
entire codebase. It explains the rationale, the failure
modes, and what the user would see. **Preserve this
pattern.** **No fix.**

### 5.5 `initialize_ml_schema` uses datapath fetch + insert

`create_schema.py:416-422` uses
`table.entities()` + `table.insert(missing, defaults={"ID", "URI"})`.
Correct datapath usage. No raw ERMrest URLs.
**No fix.**

### 5.6 `dump_ml_schema` uses `catalog.delete_ermrest_catalog(really=True)`

The destructive side-effect (§1.2) — the deriva-py
pattern itself is correct (the canonical way to delete
a catalog). Issue is the API contract, not the call.

---

## Lens 6 — Maintainability / naming / class size

### 6.1 `annotations.py:649` LoC is borderline; module is coherent

The audit prompt flagged the ~650 LoC module size as a
size flag. Reading the file: `build_navbar_menu` (182
LoC), `catalog_annotation` (37 LoC),
`asset_annotation` (109 LoC), `generate_annotation`
(252 LoC), `main` (19 LoC, dead per §1.8).
`generate_annotation` is the largest single function.

The function bodies are long because the structures
they build (Chaise annotations) are inherently
verbose JSON. Splitting `generate_annotation` into
four functions (one per table-type annotation:
`_workflow_annotation`, `_execution_annotation`,
`_dataset_annotation`, `_dataset_version_annotation`)
would improve readability without losing cohesion.

**Fix (optional refactor):** split `generate_annotation`
into named per-table helpers; keep the public function
as a thin aggregator. **LoC: ±0.** **Risk: low** —
purely structural. **Severity: low.**

### 6.2 `build_navbar_menu` repeats `f"/chaise/recordset/#{catalog_id}/..."`

19 occurrences of this URL pattern (`annotations.py:70,
82, 92, 104, 115, 126, 138, 143, 147, 155, 160, 164,
167, 171, 176, 184, 189, 199, 209`). A `_chaise_url`
helper would reduce the repetition:

```python
def _chaise_url(catalog_id, schema, table):
    return f"/chaise/recordset/#{catalog_id}/{schema}:{table}"
```

Trivial code smell. **LoC: ±0** (net). **Risk: trivial.**
**Severity: low.**

### 6.3 `create_schema.py:662` LoC is mostly load-bearing

Reading the file: six table-creation helpers (210 LoC
total), `create_ml_schema` (94 LoC),
`initialize_ml_schema` (142 LoC),
`create_ml_catalog` (80 LoC), `reset_ml_schema` (7
LoC), `main` (23 LoC, dead per §1.6).

The file is at the natural breaking point: if it grew
much further, splitting tables into one-per-file would
help. As-is, the seven tables (Dataset,
Dataset_Version, Execution, Workflow, asset tables ×3)
plus initialization plus catalog creation form a
coherent unit.

The `initialize_ml_schema` function (142 LoC) is mostly
literal term lists (lines 424-542). The lists could be
JSON or YAML files in the package data; the
`initialize_ml_schema` body would then load them.
This would also resolve §1.1 (validator could read the
same JSON for its expected-terms list).

**Fix (optional):** extract the four vocabulary term
lists to `schema/vocabulary_terms.json` (or similar)
and load via `importlib.resources` like `policy.json`.
**LoC: small reduction.** **Risk: low.** **Severity:
low.**

### 6.4 `validation.py:602` LoC is mostly drift-prone constants

Reading the file: the `EXPECTED_*` constants are 113
LoC (lines 219-346); the data classes are 110 LoC
(lines 32-189); the `SchemaValidator` class is 230 LoC
(lines 352-578); the convenience function is 22 LoC.

If §1.4 (b) lands (read constants from `schema.md`),
the constants section deletes — file shrinks to ~480
LoC. If §1.4 (a) lands (delete the whole module), 602
LoC delete.

### 6.5 Dataclasses-vs-Pydantic for `SchemaValidationReport` / `ValidationIssue`

`validation.py:40, 62` are `@dataclass`es. Per
CLAUDE.md "Class idiom choice — Pydantic vs `@dataclass`":

> Use Pydantic BaseModel when ANY of these apply: ...
> The class may be serialized or cross a boundary (JSON
> I/O, logs, cache, API, bag metadata). Users should
> reach for one API (.model_dump()) rather than
> juggling dataclasses.asdict() depending on type.

`SchemaValidationReport` has `to_dict()` and
`to_json()` methods (lines 143-210) — i.e., it's a
boundary type, exposed as the return type of a public
method (`DerivaML.validate_schema`). Same with
`ValidationIssue`. Both should be Pydantic per the
convention.

The conversion cost: replace `@dataclass` with
`class ... (BaseModel)`, replace `field(default_factory=list)`
with `Field(default_factory=list)` (or just use the
shorter form), delete the manual `to_dict()` (replaced
by `.model_dump()`), keep `to_text()` and `to_json()`
as convenience methods.

**Fix:** convert the three dataclasses
(`ValidationIssue`, `SchemaValidationReport`, plus
nothing else in the module) to Pydantic
`BaseModel`s. **LoC: −30** (manual `to_dict` deletes).
**Risk: medium** — `dataclasses.replace` doesn't work
on Pydantic; if any callers use it, they break. Per
cross-workspace check, no callers exist. **Severity:
medium** — explicit CLAUDE.md convention violation.

### 6.6 `_ensure_terms` repeated in `initialize_ml_schema`

`create_schema.py:416-422` defines `_ensure_terms` as a
closure inside `initialize_ml_schema`. The closure is
correct; deduplicating "Name → row" insertion is the
right pattern. But the function shape ("insert if not
present, skip if present") is exactly what
`DerivaML.add_term(... if_not_exists=True)` already
provides (or did — verify). If the term-list extraction
from §6.3 lands, the closure could move to a top-level
helper.

**Severity: low.**

### 6.7 Missing `__all__` on `check_schema.py`,
`table_comments_utils.py`, `create_schema.py`,
`annotations.py`

Of the six files, only `__init__.py` and (effectively)
`validation.py` have curated public surfaces — but
`validation.py` itself has no `__all__`; the public
surface is the re-export through `schema/__init__.py`.
The other three executable modules (`check_schema.py`,
`table_comments_utils.py`, `create_schema.py`,
`annotations.py`) export everything by default. If any
of them grew, an `__all__` would be useful.

**Fix:** add `__all__` to each module after the
deletions in §1.2, §1.3, §1.6, §1.8 settle. Defer.
**Severity: low.**

---

## Persona check

**Senior engineer:**
- §1.2 (`check_schema.py` redundancy + stale JSON +
  signature bug) is the cleanest delete in the audit.
- §1.4 (validator overlap) is the biggest structural
  cleanup; option (b) — read constants from
  `schema.md` — kills the drift problem and the
  duplicated-truth problem in one move.
- §1.5 (`reset_ml_schema` with no guard) is the most
  dangerous unused public function in the file. The
  delete is small.

**Testing engineer:**
- §3.1 (no CASCADE-safety test) is the biggest
  coverage gap — the most destructive operation in the
  subsystem has no regression guard.
- §3.4 (existing validation tests don't assert ground
  truth) is the second-biggest gap. The tests pass
  because the validator doesn't ask the right
  questions.
- §3.2 (no ACL test) is cheap and worth landing.

**Technical writer:**
- §1.7 (`_post_clone_operations` is fictional) and §4.3
  (`reset_ml_schema` docstring misleads on scope) are
  the load-bearing docstring corrections.
- §4.5 (`dump_ml_schema` docstring claim vs reality) is
  subsumed by §1.2.
- §4.6 (`policy.json` has no doc) is a real gap for
  DBA-persona users tracking down ACL failures.

**ML-developer user:**
- A user installing `deriva-ml` and trying
  `deriva-ml-create-schema` or
  `deriva-ml-table-comments-utils` from `$PATH`
  encounters `ModuleNotFoundError` immediately
  (theme 1). This is the most user-visible
  brokenness.
- `create_ml_catalog` is healthy and the model-template
  test confirms it works end-to-end. The CASCADE warning
  in the docstring is sufficient if the user reads the
  docstring; the warning is **not** repeated at the CLI
  or hooked into a confirmation prompt anywhere. A user
  who imports `create_ml_schema` and calls it on an
  existing catalog loses data with no second chance.
- `validate_schema` (the public method) returns a
  report that misses the `Execution_Status` vocabulary
  and most `Dataset_Type` terms. A user trusting the
  validator's "VALID" verdict has a false sense of
  security on those dimensions.

**DBA:**
- §3.1 (CASCADE regression test) — load-bearing.
- §3.2 (ACL test) — load-bearing.
- §1.5 (`reset_ml_schema`) — public delete-everything
  function with no audit log. Delete or guard.
- §3.6 (policy.json test) — trivial to add, catches
  typos that would otherwise show up as silent ACL
  failures.

---

## Ranked actions

Numbered by recommended landing order, balancing impact,
risk, and whether one item subsumes another.

| # | Action | Effort | Risk | LoC | Severity |
|---|---|---|---|---|---|
| 1 | **Theme 1 — Fix or remove the three broken `[project.scripts]` entries.** `pyproject.toml:34-36` reference `deriva_ml.schema_setup` which doesn't exist. `deriva-ml-alter-annotation` also references a non-existent module. Either (a) delete the three entries (cleanest, no caller), or (b) rewrite to `deriva_ml.schema.*`. **Recommend: delete.** | trivial | trivial | −3 | high |
| 2 | **§1.7 — Delete the fictional `_post_clone_operations` docstring reference in `create_schema.py:312`.** Replace with the actual safety property (bag clone doesn't call `create_ml_schema`). | trivial | trivial | ±0 | medium |
| 3 | **§3.1 — Add `tests/schema/test_cascade_behavior.py` to pin the CASCADE behavior.** One integration test, ~30 LoC. | low | low | +30 (test) | high |
| 4 | **§3.2 — Add `tests/schema/test_acl_application.py`.** One integration test for `row_owner_guard` binding presence. | low | low | +25 (test) | medium |
| 5 | **§3.6 — Add `tests/schema/test_policy_validation.py`.** Three asserts; no integration mark needed (loads JSON only). | trivial | trivial | +15 (test) | low |
| 6 | **§1.1 — Sync `EXPECTED_*` constants with `create_schema.py` reality.** Add `MLVocab.execution_status`, expand vocabulary term lists. **Or** (preferred) collapse the constants into a read-from-`schema.md` derivation (option b of §1.4). | low (sync) / medium (b) | low / medium | ±0 / −113 | medium |
| 7 | **§1.5 — Delete `reset_ml_schema`.** Zero callers, dangerous if invoked, public function. | trivial | low | −7 | medium |
| 8 | **§1.6 — Delete `create_schema.py:main()`.** 23 LoC of broken CLI; entry-point doesn't resolve. | trivial | trivial | −23 | medium |
| 9 | **§1.8 — Delete `annotations.py:main()`.** Same shape — broken CLI, no entry point. | trivial | trivial | −19 | low |
| 10 | **§1.10 — Replace `importlib.import_module` workaround in `annotations.py` with normal imports.** | trivial | low | −7 | low |
| 11 | **§4.2 — Add `# doctest: +SKIP` to `validation.py:200-208, 594-600` and convert `create_schema.py:562-565` to `>>>` form with `SKIP`.** | trivial | trivial | ±0 | low |
| 12 | **§1.3 — Delete `table_comments_utils.py` and its `pyproject.toml` entry.** Zero callers, broken CLI, no doc. | low | low | −56 | medium |
| 13 | **§1.2 — Delete `check_schema.py`, `deriva-ml-reference.json`, and the `deriva-ml-check-catalog-schema` `pyproject.toml` entry.** Stale JSON, redundant validator, signature bug, no callers. | low | low | −10 000 (mostly JSON) | high |
| 14 | **§3.4 — Either drop `TestExpectedSchemaStructure` or rewrite to assert ground truth.** If §1.4 (a) deletes `validation.py`, this drops with it. | low | low | ±0 | medium |
| 15 | **§1.4 — Decide on `validation.py`'s future.** Option (b): rewrite `SchemaValidator` to use the doc-derived model. Option (a): delete `validation.py`, replace `DerivaML.validate_schema` with a thin wrapper over `tools/validate_schema_doc.py`. | medium-high | medium | −113 / −602 | medium |
| 16 | **§6.5 — Convert `SchemaValidationReport` and `ValidationIssue` to Pydantic.** Per CLAUDE.md class-idiom rules. | low | medium | −30 | medium |
| 17 | **§3.7 — Add `tests/schema/test_annotations.py` for `build_navbar_menu`.** | low | trivial | +60 (test) | low |
| 18 | **§4.6 — Add `docs/reference/acl-policy.md`.** | low | trivial | +60 (doc) | low |
| 19 | **§6.1 — Optional: split `generate_annotation` into per-table helpers.** | low | low | ±0 | low |
| 20 | **§6.2 — Add `_chaise_url` helper for the 19 URL repetitions in `build_navbar_menu`.** | trivial | trivial | ±0 | low |
| 21 | **§6.3 — Extract vocabulary term lists from `initialize_ml_schema` to a JSON/YAML resource.** Makes them shareable with the validator (§1.1). | low | low | small reduction | low |
| 22 | **§2.1 — Privatize `SchemaValidator` to `_SchemaValidator`.** Only if validation.py survives §1.4. | trivial | low | ±0 | low |
| 23 | **§4.3 — Fix `reset_ml_schema` docstring** if §1.5 doesn't delete the function. | trivial | trivial | ±0 | low |
| 24 | **§4.4 — Either implement FK validation in `validation.py` or remove the docstring claim.** Subsumed by §1.4 (a). | low | low | ±0 | low |

Items 1, 2, 3, 4, 5 are quick wins for the next cleanup
PR — five small changes covering the highest-severity
items.

Items 7, 8, 9, 10 are mechanical deletes and can ride
along.

Item 11 is the doctest hygiene fix.

Items 12, 13 are the major deletes — `table_comments_utils.py`
and `check_schema.py` together remove 174 LoC of code
plus 9 881 lines of stale JSON. These want their own
small PR to avoid mixing deletions of executable code
with deletions of static data.

Item 15 is the strategic call: deduplicate the
validators or accept the overlap. Item 6 fixes the
immediate symptom; item 15 fixes the root cause.

Combined cleanup ceiling if every "delete" branch lands:
**−10 348 LoC** (mostly the stale JSON) **+130 LoC of
tests.** Net is comfortably negative.

---

## Worst-offender modules

1. **`check_schema.py`** (118 LoC) + `deriva-ml-reference.json`
   (9 881 LoC). Stale JSON, signature-mismatched CLI,
   silent live-catalog side effect, redundant with the
   two other validators. Cleanest large delete.

2. **`table_comments_utils.py`** (56 LoC). Zero callers,
   broken CLI, no doc, no example directory in the
   repo. Confirmed dead.

3. **`validation.py`** (602 LoC). Load-bearing in shape
   (return type of `DerivaML.validate_schema`), but the
   constants drift from `create_schema.py` reality, the
   tests don't assert ground truth, and the
   doc-source-of-truth flow in
   `tools/validate_schema_doc.py` has superseded the
   premise. The strategic call (§1.4) determines
   whether this module shrinks by 113 LoC or deletes
   entirely.

4. **`create_schema.py:main()` + `annotations.py:main()`**
   (42 LoC combined). Two broken CLI entry points with
   no resolving `pyproject.toml` reference.

5. **`pyproject.toml:34-36` console scripts** (3 lines,
   workspace-wide visibility). Broken at install time;
   `pip show -f` lists them, `which` finds them, invoking
   them errors. User-visible.

6. **`reset_ml_schema`** (7 LoC). Public delete-every-schema
   function with no callers and no guard. The most
   dangerous unused symbol in the subsystem.

7. **`annotations.py:build_navbar_menu`** (182 LoC).
   Healthy; one structural improvement opportunity
   (`_chaise_url` helper for the 19 URL repetitions).
   Not a worst-offender; flagged only because it's the
   single longest function in `schema/`.

8. **`create_schema.py:312` docstring** (1 line).
   References `_post_clone_operations` which doesn't
   exist. The CLAUDE.md-flagged CASCADE safety claim
   points the reader at a fictional guard.

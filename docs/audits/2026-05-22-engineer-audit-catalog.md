# Engineer audit — `catalog/` subsystem (v1.37.1, 2026-05-22)

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/catalog/`
(5 files, 1 952 LoC: `__init__.py` 74, `clone.py` 355,
`clone_via_bag.py` 569, `localize.py` 643, `provenance.py` 311) and
the test surface
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/catalog/` (6 files,
1 546 LoC: 1 integration test module under `@pytest.mark.integration`,
4 unit-test modules). Cross-workspace grep against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-skills,deriva-ml-skills,deriva-ml-model-template}/`.

Scope ground rules (per audit brief):

- Test coverage of cloning (`clone.py` and `clone_via_bag.py`), the
  bag-pipeline path, `localize_assets`, three-stage clone behaviour
  the legacy surface still claims to provide, orphan strategies, and
  dataset-version reinitialization.
- Code duplication (orphan handling, FK helpers, dataset-version
  recreation, alias creation).
- Logic clarity (async-loop hoisting, orphan-strategy plumbing,
  snapshot-id propagation).
- Docstring completeness/accuracy (public functions documented,
  examples, three-stage approach explained).
- Legacy/historical concerns are out of scope (per brief).

## Executive summary

The active clone path (`clone_via_bag` → `CatalogBagBuilder` →
`BagCatalogLoader`) is in good shape: well-documented, unit-tested
for argument shape and policy merging, and exercised by three
`-m integration` smoke tests. The wrappers and adjacent modules are
where the risk lives:

- **`localize_assets` is 350 LoC of asset-bytes movement with no
  integration coverage and no per-asset error-recovery testing.**
  Dry-run, helper purity (`_extract_hatrac_path`, `LocalizeResult`
  defaults / round-trip) are tested. The actual fetch / push /
  catalog-update / provenance-write loop has zero coverage. P0.
- **`create_ml_workspace` claims a "three-stage approach" in its
  module docstring** (`__init__.py`:7-12 of the legacy `clone.py`
  doc and `CLAUDE.md`) **but the implementation is now a single
  pipeline call.** The docstring still says "schema without FKs →
  async data copy → FK application with orphan handling"; the body
  is one `clone_via_bag(...)` invocation. P1 (false advertising).
- **`_record_clone_provenance` and `_record_localize_provenance`
  carry similar dataclass-update logic with slightly different
  fallback rules.** Phase 1 writes a fresh `CloneDetails`; phase 2
  reads-modifies-or-writes one. Neither has a test that exercises
  the "phase 2 runs without phase 1" branch end-to-end. P2.
- **The policy-merge `model_fields_set` predicate covers 4 of the
  12 policy fields** (`vocab_export`, `terminal_tables`,
  `dangling_fk_strategy`, `intentional_cycles`). The other 8 fields
  (`schemas`, `exclude_schemas`, `exclude_tables`, `max_depth`,
  `asset_mode`, `content_on_conflict`, `match_by_columns`,
  `preserve_provenance`) get no clone-required defaults. Whether
  that's intentional or accidental is undocumented. P2.
- **The "dataset-version reinitialization" feature documented in
  `create_ml_workspace`'s legacy parameter list** (and listed under
  `CloneDetails.reinitialize_dataset_versions`) **is now a complete
  no-op.** The legacy parameter is warned-and-ignored; the bag
  pipeline does not re-seed dataset versions. No code path
  re-initializes; nothing in tests reflects this. P1 (silent loss
  of a documented feature).

Severity mix targets: P0 — must address before release; P1 — should
address; P2 — track for the next cycle; P3 — nit / cosmetic.

---

## 1. Test coverage

### 1.1 `localize_assets` has no integration test (P0)

- File: `tests/catalog/test_localize.py` (122 LoC)
- Covered: `LocalizeResult` defaults + round-trip;
  `_extract_hatrac_path` parsing variants.
- Uncovered (the bulk of `localize.py`, 644 LoC):
  - `localize_assets` happy path (fetch → push → URL update).
  - `dry_run=True` branch and the early-return when
    `not assets_to_localize`.
  - Per-asset error handling — the inner `except Exception` at
    `localize.py:367-371` swallows everything into
    `result.errors`. No test asserts that one failure doesn't
    abort the batch.
  - The batch-update-then-fallback-to-individual-update logic at
    `localize.py:374-391`. The fallback re-tries each row on
    `table.update()` failure; nothing pins that "one bad row
    doesn't lose the whole batch."
  - `_record_localize_provenance` ("phase 2 with no phase 1" vs
    "phase 2 over existing phase 1") branches.
  - `_find_asset_table_path`: schema-known vs. schema-search
    branches; "not found" returning the documented
    `(None, None)` tuple **but the caller at
    `localize.py:179-181` unpacks as `table_path, found_schema`
    and only checks `table_path is None`**. Confirmed correct,
    but no test guards the contract.
  - URL-column auto-detection at `localize.py:198-201` (`URL` vs
    `url`). Two production catalogs may legitimately differ;
    no test pins either branch.
  - The chunked-upload threshold logic (`file_size >
    100 * 1024 * 1024`, `default_chunk_size = 50 MB`).
    `localize.py:339-353` has three interacting predicates and
    zero tests.

Action: add at least the happy-path, dry-run, and one-failed-asset
tests with mocked `HatracStore` + datapath update. The
`_make_fake_catalog` helper in `test_provenance.py:147-158` is a
model for the right level of mocking.

### 1.2 `clone_via_bag` lacks tests for `_materialize_bag_dir`, `_materialize_bag_assets`, `_record_clone_provenance` (P1)

- File: `tests/catalog/test_clone_via_bag.py` (666 LoC).
- The wrapper-logic surface (`root_rid` → `RIDAnchor` mapping,
  policy merge, credential lookup, nested-dataset expansion) is
  thoroughly tested.
- Uncovered helpers in `clone_via_bag.py`:
  - `_materialize_bag_dir` (lines 89-128): zip-fallback path,
    "neither directory nor zip exists" error path, and the
    "extracted directory ambiguous" branch
    (`len(extracted) != 1`) all have zero coverage.
  - `_materialize_bag_assets` (lines 212-236): the
    "skip when fetch.txt absent or empty" guard and the
    `bdb.materialize(...)` call have zero coverage.
  - `_record_clone_provenance` (lines 512-566): the orphan-stat
    aggregation loop (`getattr(stats, "rows_skipped_orphan", 0)`),
    the str-Enum coercion
    (`str(policy.dangling_fk_strategy).split(".")[-1].lower()`),
    and the broad `except Exception` guard are not tested.

Action: small unit tests with a `MagicMock` `LoadReport` and a
mock catalog suffice — the `set_catalog_provenance` boundary is
already mockable.

### 1.3 Orphan strategies (`FAIL`/`DELETE`/`NULLIFY`) have no
behavioural test (P1)

- The integration tests (`test_clone_via_bag_end_to_end_default_policy`,
  `test_clone_via_bag_rows_only_skips_asset_uploads`,
  `test_clone_via_bag_rid_anchor_scopes_walk`) all use the default
  `DanglingFKStrategy.DELETE`. There is no test (unit or
  integration) that:
  - Verifies `FAIL` actually fails-fast when an orphan is reached.
  - Verifies `NULLIFY` produces a row with the FK column set to
    NULL on the destination.
  - Verifies the aggregated orphan-stat counters
    (`orphan_rows_removed`, `orphan_rows_nullified`) on the
    written `CloneDetails` annotation reflect what happened.
- `test_clone_via_bag_preserves_explicit_fail_strategy` (line 368)
  pins **policy plumbing** (the merge doesn't override `FAIL`) but
  not behaviour — it asserts on the policy passed to the mock,
  not on a real load.

Action: at least one integration test that constructs a slice with
a known dangling FK (or mocks `BagCatalogLoader` to return a
`LoadReport` with non-zero orphan counters) and asserts the
provenance annotation reflects the result.

### 1.4 Dataset-version reinitialization is documented but not
implemented (P1)

- `CloneDetails.reinitialize_dataset_versions` exists as a field
  (`provenance.py:117`) and `create_ml_workspace` accepts the
  parameter (`clone.py:111`).
- The legacy parameter is silently ignored: `clone.py:209` passes
  it to `_warn_about_legacy_params`, which emits a deprecation
  warning **only if the caller passed a non-default value**. The
  default is `True`, so a caller relying on the documented
  behaviour gets a silent no-op.
- No test verifies that a clone re-seeds `Dataset_Version` rows.
  Bag-loaded `Dataset_Version` rows carry whatever values the
  source had — including the source's catalog snapshot ID — which
  may or may not be the right thing for the destination catalog.

Action: either delete the field from `CloneDetails` and the
parameter from `create_ml_workspace` (per workspace CLAUDE.md
"No backwards-compat shims" rule) **or** wire up the actual
re-seeding step and add an integration test. Picking the first
option is consistent with the rest of the legacy-parameter cleanup.

### 1.5 Bag-pipeline clone path has integration tests but no
asset-upload coverage (P2)

- `test_clone_via_bag_rows_only_skips_asset_uploads`
  (`test_clone_via_bag_integration.py:165`) asserts
  `total_attempts == 0` for `AssetMode.ROWS_ONLY`. There is no
  symmetric test for `AssetMode.UPLOAD_IF_MISSING` or
  `UPLOAD_FORCE` — i.e., the asset bytes actually reaching the
  destination Hatrac. The `_materialize_bag_assets` step is
  load-bearing for those modes and silently invoked.

Action: add an integration test that clones with
`UPLOAD_IF_MISSING`, then verifies one asset's bytes exist on
the destination Hatrac (download → MD5 compare, or
`head_obj` 200 OK).

### 1.6 No test pins the `is_clone` predicate against `creation_method` enum form (P3)

- `provenance.py:172-176` compares
  `self.creation_method == CatalogCreationMethod.CLONE.value`
  (against the string form) because the docstring claims
  `use_enum_values=True`. But `VALIDATION_CONFIG`
  (`core/validation.py`) does **not** set `use_enum_values=True`;
  the model stores enum members. The current `==` works because
  `CatalogCreationMethod` inherits from `str` so
  `member == "clone"` is True. Tests at `test_provenance.py:72-95`
  pass, but the predicate's `.value` access is misleading.
- The docstring comment at `provenance.py:172-174` is wrong about
  why the comparison works.

Action: simplify to
`self.creation_method == CatalogCreationMethod.CLONE` and update
the comment. Lowest priority.

### 1.7 Nested-dataset expansion tests use `MagicMock` plumbing, not real Datapath shape (P3)

- `tests/catalog/test_clone_via_bag.py:562-609`
  (`_make_mock_catalog_with_dataset_dataset_rows`) is 50 LoC of
  bespoke mock setup that mimics the real datapath call chain
  (`pb.schemas[...].tables[...].filter(...).entities().fetch()`).
  This works but is brittle to refactors: changing the helper's
  query shape silently breaks every test that uses the mock.
- Action: extract a fixture, or document the call shape inline.
  No urgent change required.

---

## 2. Code duplication

### 2.1 Two `set_catalog_provenance` / `CloneDetails` write paths with subtle differences (P2)

- `clone_via_bag.py:512-566` (`_record_clone_provenance`):
  - Builds a fresh `CloneDetails` from policy + report.
  - Sets `orphan_strategy=str(policy.dangling_fk_strategy).split(".")[-1].lower()`
    and `asset_mode=str(policy.asset_mode).split(".")[-1].lower()`.
  - Always calls `set_catalog_provenance(creation_method=CLONE,
    clone_details=…)` — does **not** preserve the existing
    `name`/`description`/`workflow_url`/`workflow_version`.
- `localize.py:554-643` (`_record_localize_provenance`):
  - Reads existing provenance first.
  - If phase-1 details exist, `model_copy(update=…)` over them.
  - If not, builds a fresh `CloneDetails` with phase-2-only fields.
  - Explicitly preserves
    `name`/`description`/`workflow_url`/`workflow_version` from
    the existing annotation.

The two paths handle the same provenance annotation; the second
preserves descriptive fields the first wipes. A user who runs
`set_catalog_provenance(name=…)` before `clone_via_bag` will lose
the name when phase 1 writes; localize will then preserve
whatever phase 1 wrote (i.e., still no name).

Action: factor a helper `_merge_clone_details(existing,
**updates)` so both phases use the same read-modify-write story
and `name`/`description` survive both phases.

### 2.2 Lower-case + dot-split enum stringification appears twice and is unnecessary (P3)

- `clone_via_bag.py:550-551`:
  ```python
  orphan_strategy=str(policy.dangling_fk_strategy).split(".")[-1].lower(),
  asset_mode=str(policy.asset_mode).split(".")[-1].lower(),
  ```
- Both enums (`DanglingFKStrategy`, `AssetMode`) inherit from
  `StrEnum` (verified at audit time:
  `DanglingFKStrategy.__mro__ = (..., StrEnum, str, ReprEnum, Enum, object)`).
  `str(DanglingFKStrategy.DELETE) == "delete"`; no dot to split.
- The two-step `split(".")[-1].lower()` is dead-defensive code
  inherited from a pre-`StrEnum` shape (`"DanglingFKStrategy.DELETE"
  → "DELETE" → "delete"`). It works coincidentally on `StrEnum`
  because `str(...)` returns the value already.

Action: replace with `policy.dangling_fk_strategy.value` /
`policy.asset_mode.value`. Less code, less misleading.

### 2.3 `_coerce_asset_mode` / `_coerce_orphan_strategy` are parallel
string-mapping helpers with no remaining callers (P2)

- `clone.py:276-296` and `clone.py:299-312` translate legacy
  string spellings (`"refs"`, `"FULL"`, `"none"`, etc.) into the
  new enum members. They are called only from
  `create_ml_workspace` lines 243-244.
- The only external caller of `create_ml_workspace` is
  `deriva-mcp/src/deriva_mcp/tools/{catalog,background_tasks}.py`,
  which is already broken on a non-string-related import
  (`AssetCopyMode` doesn't exist; documented in the prior phase-3
  audit). When deriva-mcp's cut-over completes (per workspace
  CLAUDE.md), these coercers have no callers.
- Per CLAUDE.md "No backwards-compat shims": these are textbook
  backward-compat helpers.

Action: tag for deletion in the next legacy-cleanup pass once
deriva-mcp is retired. Not a v1.37.1 blocker.

### 2.4 Two `from deriva_ml.catalog.provenance import …` import lists (P3)

- `clone_via_bag.py:71-75` imports
  `CatalogCreationMethod, CloneDetails, set_catalog_provenance`.
- `localize.py:592-597` (inside `_record_localize_provenance`)
  imports
  `CatalogCreationMethod, CloneDetails, get_catalog_provenance,
  set_catalog_provenance`.
- The lazy import in `localize.py` is justified with "avoids a
  circular reference" — but `provenance.py` does not import
  anything from `localize.py` or `clone_via_bag.py`, so there is
  no actual cycle. The lazy form costs a runtime import on every
  call and obscures the dependency for static analysers.

Action: hoist to module-level imports in `localize.py`. The
"defensive against future cycles" comment can stay if you prefer,
but the lazy form is unnecessary today.

### 2.5 `_find_asset_table_path` and "search all schemas" pattern duplicate model traversal in `localize.py` (P3)

- `localize.py:438-468` searches schemas one by one with a try /
  `KeyError` loop. Functionally identical to a single
  `for sname in pb.schemas: …` walk; the schema-known branch is
  a separate try/except that returns early.
- `DerivaML` already exposes `model_table(...)` /
  `resolve_table(...)` helpers (per CLAUDE.md API-priority note);
  using those would route through one cached lookup rather than
  iterating `pb.schemas`.

Action: refactor to a single lookup via the deriva-ml helper.
Defer to next refactor wave.

---

## 3. Logic clarity

### 3.1 Policy merge predicate covers 4 of 12 fields (P2)

- `clone_via_bag.py:405-419` merges `vocab_export`,
  `terminal_tables`, `dangling_fk_strategy`, `intentional_cycles`
  with the `model_fields_set` predicate.
- `FKTraversalPolicy` exposes 12 fields:
  `schemas, exclude_schemas, exclude_tables, terminal_tables,
   max_depth, vocab_export, asset_mode, dangling_fk_strategy,
   content_on_conflict, match_by_columns, preserve_provenance,
   intentional_cycles`.
- For 8 fields, the caller's default-library-default is taken
  silently — with no documentation that the merge is selective.
  A caller who supplies `policy=FKTraversalPolicy()` (everything
  defaulted) gets clone-friendly `vocab_export=FULL` and
  `dangling_fk_strategy=DELETE` injected but library-default
  `asset_mode=UPLOAD_IF_MISSING`. The asymmetry is surprising.

Action: either (a) expand the merge to all clone-relevant fields,
or (b) document the selective merge in the
`clone_via_bag` docstring and the policy-merge inline comment
(lines 386-404). Doing nothing is the worst option.

### 3.2 `output_dir` defaults to a path under `Path.cwd()` with no overwrite semantics (P2)

- `clone_via_bag.py:421-423`: when `output_dir=None`, the bag is
  written under `Path.cwd() / f"clone-{src}-to-{dst}"`.
- If the directory already exists from a previous run,
  `CatalogBagBuilder` behaviour is undefined here (the docstring
  says nothing). The clone may silently merge into a stale bag
  or fail with an opaque error from deeper in `bdbag`.
- No test exercises the "stale output_dir" case;
  `test_clone_via_bag_default_output_dir` only verifies the path
  derivation.

Action: document the precondition (output_dir must be empty / not
exist) in the docstring; consider raising a clear error when a
non-empty directory is passed; add a unit test that confirms the
error path.

### 3.3 Nested-dataset BFS exception swallowing hides real failures (P2)

- `clone_via_bag.py:189-199` catches `DataPathException`,
  `KeyError`, `ConnectionError` and proceeds with the seed RID
  set only.
- The rationale ("catalog doesn't have nested-dataset
  infrastructure") is right for `KeyError` on
  `pb.schemas["deriva-ml"]`. It's questionable for
  `DataPathException` (could be a transient query error, a
  permission issue, or a schema typo — none of which should be
  silently ignored) and `ConnectionError` (a transient transport
  failure mid-BFS may produce a partial expansion that *looks*
  complete to the rest of the pipeline).
- The exception is logged at WARNING level, but the caller has
  no programmatic signal that the expansion was truncated.
  Downstream, the bag walker will likely produce dangling-FK
  warnings; the connection between the two is non-obvious.

Action: narrow to `KeyError` only for the "no nested-dataset
infrastructure" case; let `DataPathException` / `ConnectionError`
propagate. If the partial-expansion fallback must stay, return a
sentinel or include a counter on `CloneViaBagResult` for
"nested-expansion-truncated".

### 3.4 `_record_clone_provenance`'s "best-effort" swallow is too wide (P2)

- `clone_via_bag.py:561-566`:
  ```python
  except Exception as e:
      logger.warning("clone_via_bag: failed to write provenance: %s", e)
  ```
- The except catches anything from the orphan-stat aggregation
  loop (which calls `getattr` with defaults — safe) through the
  `CloneDetails(...)` construction (Pydantic validation; should
  not fail with correct input) and the
  `set_catalog_provenance(...)` call (which is itself
  best-effort).
- Two of those failure modes are noise-suppression we want; the
  third (CloneDetails validation) is a real bug we'd want to
  surface as an error in dev.

Action: narrow the except to the network-touching call only;
let validation errors propagate.

### 3.5 `localize_assets` records the function-parameter `source_hostname` as `asset_source_hostname` even in mixed-source slices (P1)

- `localize.py:402` calls
  `_record_localize_provenance(..., asset_source_hostname=source_hostname)`.
- `source_hostname` is the **fallback for relative URLs**, not
  the actual source of each asset. In a mixed-source slice
  (where different assets came from different upstream hosts)
  the annotation will read a single hostname that doesn't
  describe the localization correctly.
- Inside the loop, `assets_to_localize[*]["source_hostname"]`
  carries the actual per-asset source. That information is
  discarded by the time we write the annotation.

Action: aggregate the actual per-asset source hostnames into a
list (or `set`) on `LocalizeResult`, and write either the single
hostname (when it's a single-source slice) or `None` + a
log warning ("mixed-source slice; see audit log for details") on
the annotation. Tag P1 because the provenance annotation is
documented as durable observability of the phase-2 leg.

### 3.6 `chunk_size if use_chunked else 0` is a confusing default (P3)

- `localize.py:353` passes
  `chunk_size=actual_chunk_size if use_chunked else 0` to
  `local_hatrac.put_loc(...)`.
- Passing `0` for `chunk_size` when `chunked=False` is dead
  code — `HatracStore.put_loc` ignores `chunk_size` when
  `chunked=False` per deriva-py's contract. Reading the line, a
  reviewer naturally asks "what does chunk_size=0 mean?"
- The "treat `chunk_size=0` from the caller as None" rule
  (commented at lines 333-338) is also subtle: a user passing
  `chunk_size=0` explicitly *might* mean "force non-chunked
  upload", but is interpreted as "use default chunking".

Action: pass `chunk_size=actual_chunk_size` unconditionally;
let the `chunked` flag control behaviour. Update the docstring
to clarify that `chunk_size=0` is the same as `chunk_size=None`.

### 3.7 `_record_localize_provenance` reads `existing.clone_details is not None` but writes when it is `None` (P3)

- `localize.py:601-628`: branch A reads existing
  `clone_details`, copies fields; branch B writes a fresh
  `CloneDetails` with empty `source_hostname=""` /
  `source_catalog_id=""`.
- Branch B's `source_hostname=""` is a structurally-invalid
  `CloneDetails` value masquerading as valid (the type allows
  empty strings, but the field is documented as the source
  hostname — empty means "unknown"). A future reader who
  filters provenance by source hostname will hit empty strings
  and have to special-case them.

Action: change branch B to use
`asset_source_hostname or "unknown"` (or a similar sentinel),
and document the "phase 2 without phase 1" semantics on
`CloneDetails`.

### 3.8 `is_clone` property comment is incorrect (P3)

- `provenance.py:172-174`:
  ```python
  # use_enum_values=True stores the value, not the enum member,
  # so compare against the string form.
  return (
      self.creation_method == CatalogCreationMethod.CLONE.value
      and self.clone_details is not None
  )
  ```
- `VALIDATION_CONFIG` (`core/validation.py`) does **not** set
  `use_enum_values=True`. The comparison works because
  `CatalogCreationMethod` is a `str`-inheriting `Enum`, so
  `enum_member == "clone"` returns True. The comment is
  misleading and the `.value` access is unnecessary.

Action: simplify to `self.creation_method ==
CatalogCreationMethod.CLONE`, fix the comment.

### 3.9 `_extract_hatrac_path` accepts URLs with `/hatrac/` anywhere in the path (P2)

- `localize.py:528-534`: any URL containing `/hatrac/` in its
  path component returns the suffix from `/hatrac/`. This is
  tested at `test_extract_hatrac_path_hatrac_in_middle_of_path`
  (line 116) — i.e., the behaviour is **deliberate** but the
  test name's "real-world prefixed URLs (e.g., reverse-proxy
  paths)" rationale is thin: an actual reverse-proxy
  configuration that strips a path prefix would not require the
  client to send the un-stripped form.
- More concerning: a deliberately-crafted URL like
  `https://evil.example.org/some/hatrac/foo.bin` would parse to
  a valid hatrac path and (since the source hostname is taken
  from the URL netloc) get fetched from the attacker's server.
  The threat is low (Hatrac is behind auth and the caller
  supplies the URL anyway), but the loose parse contributes.

Action: tighten to `path.startswith("/hatrac/")` only; or
document the relaxed semantics as a deliberate quirk.

### 3.10 Provenance `is_clone` doesn't verify `clone_details` is well-formed (P3)

- `provenance.py:172-176`: the predicate only checks
  `clone_details is not None`. A `CloneDetails(source_hostname="",
  source_catalog_id="")` would return True.
- See finding 3.7 — the empty-fallback CloneDetails is the
  scenario where this matters.

Action: strengthen to require non-empty
`source_hostname` and `source_catalog_id`, or document the
weakness.

---

## 4. Docstring completeness & accuracy

### 4.1 `clone.py` docstring claims a three-stage pipeline that no longer exists (P1)

- `clone.py:8-9`: "schema without FKs → async data copy → FK
  application with orphan handling". The module-level docstring
  truthfully says "The whole legacy implementation has been
  removed in favor of `clone_via_bag`."
- However, the `CLAUDE.md` for `deriva-ml` (Catalog Cloning
  section, ~line 270) still reads: "Three-stage approach:
  schema without FKs → async data copy → FK application with
  orphan handling".
- A new contributor reading the `CLAUDE.md` description will
  expect to find the three-stage logic in `clone.py` and
  discover it doesn't exist.

Action: rewrite the `CLAUDE.md` "Catalog Cloning" subsection to
describe the bag pipeline. Keep the historical note in
`clone.py`'s module docstring; remove the parallel-but-stale
description elsewhere.

### 4.2 `create_ml_workspace` lists "Three-stage approach" properties as if they're still load-bearing (P1)

- `clone.py:117-188`: the docstring mentions
  `reinitialize_dataset_versions` and a bunch of legacy knobs
  as parameters with semantics — without flagging that the
  values are accepted-but-ignored. Yes, "Legacy parameters
  (accepted, no-op, logged as warnings):" is at line 161, but
  it's mid-docstring and easy to miss.
- A user who reads the parameter table top-down will see
  `reinitialize_dataset_versions: True` and assume the function
  re-initializes versions. They will discover the no-op only
  after running the clone and finding the destination's
  `Dataset_Version` rows wrong.

Action: move the "Legacy parameters" block to the top of the
docstring (right under the one-line description), and mark each
no-op parameter inline in the `Args:` block.

### 4.3 `clone_via_bag` "Feature parity tracking" table mentions
features that no longer have a "legacy parameter" counterpart (P3)

- `clone_via_bag.py:27-50`: the table lists
  `prune_hidden_fkeys`, `truncate_oversized`, `table_concurrency`,
  `copy_annotations` as "Legacy-only" — these are the parameters
  that `create_ml_workspace` accepts and ignores.
- For someone reading `clone_via_bag` cold, the table is
  confusing: it documents parameters of a *different* function
  (the legacy `create_ml_workspace`). It belongs in `clone.py`'s
  docstring, not here.

Action: move the table into `clone.py:create_ml_workspace`'s
docstring (or delete it; the table is also already half-stale).

### 4.4 `localize_assets` docstring omits the "no provenance write
on dry-run" behaviour (P2)

- `localize.py:68-169`: the `Args:` block mentions
  `dry_run: ... Provenance is not updated in dry-run.` This is
  good. But the `Returns:` section does not explain that the
  returned `LocalizeResult` in dry-run mode has
  `assets_processed` populated as "would-process" count, not as
  actual transfers.
- A test author reading the docstring will reach for
  `result.assets_processed > 0` as a success indicator and find
  it true even in dry-run mode.

Action: clarify in `Returns:`: "In dry-run mode,
`assets_processed` reflects the count of assets that *would
have been* localized; `localized_assets` is empty."

### 4.5 `_record_clone_provenance` and `_record_localize_provenance`
docstrings don't cross-reference each other (P3)

- The two functions form a two-phase write protocol on the same
  annotation. The phase-1 docstring
  (`clone_via_bag.py:520-535`) mentions the phase-2 update
  ("later updated by `localize_assets`") but says nothing about
  preserving caller-set fields (`name`/`description`).
- The phase-2 docstring (`localize.py:560-587`) does explain
  the preservation behaviour but doesn't note the asymmetry
  with phase-1 (which doesn't preserve them).
- See finding 2.1 — the asymmetry should be fixed; either way,
  the docstrings should note the contract.

Action: align the docstrings, ideally by aligning the behaviour
first.

### 4.6 `CloneDetails.orphan_strategy` field type and value semantics undocumented (P2)

- `provenance.py:107`: `orphan_strategy: str = "fail"`. The
  string values are constrained to the lowercase enum-value
  forms (`"fail"`, `"delete"`, `"nullify"`) but neither the
  field comment nor the surrounding model docstring says so.
- A future reader writing
  `CloneDetails(source_hostname=…, orphan_strategy="DELETE")`
  will get a structurally-valid object that fails to compare
  against
  `CatalogCreationMethod.CLONE.value`-style downstream code.

Action: switch the type to a literal-string union or to
`DanglingFKStrategy` (a `StrEnum` so it serializes correctly).
The wire format stays compatible.

### 4.7 `CatalogProvenance.created_at` is typed `str` (ISO8601) without explanation (P3)

- `provenance.py:158`: `created_at: str`. The value is set to
  `datetime.now(timezone.utc).isoformat()` in
  `set_catalog_provenance` (line 240); a reader has to follow
  the writer to learn the format.

Action: type as `str` is fine (catalog annotation transport is
JSON), but the field's docstring/Field description should say
"ISO 8601 UTC".

### 4.8 `OrphanStrategy` alias's docstring claims "value names match" but doesn't show the equivalence (P3)

- `clone.py:78-82`: "Value names (`FAIL`, `DELETE`, `NULLIFY`)
  match between the two; `OrphanStrategy.FAIL is
  DanglingFKStrategy.FAIL`."
- There is no test asserting that identity. A future change to
  `deriva.bag.traversal.DanglingFKStrategy` (renaming a member,
  changing the value form, adding a new member) would silently
  break the alias claim. The closest assertion is
  `test_intentional_fk_cycles_includes_dataset_dataset_version`
  which doesn't touch the alias.

Action: add a one-liner test
`assert OrphanStrategy.FAIL is DanglingFKStrategy.FAIL` (and the
two others). Or, when deriva-mcp retires, delete the alias.

### 4.9 `clone_via_bag` Example doctest assertion has unrealistic value (P3)

- `clone_via_bag.py:330`:
  ```text
  >>> result.load_report.total_rows_inserted  # doctest: +SKIP
  12345
  ```
- The number 12345 is a placeholder. SKIPped, so it never runs;
  but a reader unfamiliar with the codebase will wonder if it's
  meaningful.

Action: replace with a comment "# some positive number" or
remove the assertion line.

### 4.10 `_get_catalog_info` uses fragile `hasattr` duck-typing (P3)

- `localize.py:419-424`:
  ```python
  if hasattr(catalog, "catalog") and hasattr(catalog, "host_name"):
      ...
  ```
- The duck-typing has no test, and a future class that happens
  to expose both attributes (e.g., a wrapper / mock) would route
  through the wrong branch. The interface
  `DerivaMLCatalogReader` exists for exactly this kind of
  polymorphism (`CLAUDE.md` "Protocol Hierarchy").

Action: switch to `isinstance(catalog, DerivaML)` (the runtime
type) or add a protocol check; tag the function as accepting
the union explicitly.

---

## 5. Cross-reference notes

### 5.1 Workspace consumers of `catalog/` exports

In-tree (`src/deriva_ml/`):

- `core/base.py:1068`: `get_catalog_provenance` (read).
- `__init__.py:78-94`: re-exports
  `CatalogProvenance, CatalogCreationMethod, CloneDetails,
  get_catalog_provenance, set_catalog_provenance`.
- No in-tree caller of `clone_via_bag`, `create_ml_workspace`,
  `localize_assets`, `OrphanStrategy`, or `CloneViaBagResult` /
  `LocalizeResult`.

Out-of-tree:

- `deriva-ml-model-template/src/scripts/_cifar10_schema.py:44, 119`
  uses `set_catalog_provenance`.
- `deriva-mcp/src/deriva_mcp/tools/{catalog,background_tasks}.py`
  imports `AssetCopyMode` (which doesn't exist), `OrphanStrategy`,
  `create_ml_workspace`. **Already broken at runtime.** Per
  workspace CLAUDE.md, deriva-mcp is being retired.

The provenance API is the only catalog-package symbol with a
working external consumer. Everything else in the `__all__` block
(`OrphanStrategy`, `create_ml_workspace`, `clone_via_bag`,
`CloneViaBagResult`, `localize_assets`, `LocalizeResult`) is
internal-only at v1.37.1.

### 5.2 `core/constants.INTENTIONAL_FK_CYCLES` is correctly wired everywhere (positive finding)

- `tests/catalog/test_intentional_fk_cycles.py` is a thorough
  cross-module wiring test (180 LoC, 6 tests).
- Both `clone_via_bag.py:255` (legacy bag-builder) and
  `clone_via_bag.py:383, 417` (clone_via_bag) wire it in.
- Both branches of the policy-merge path also wire it (the
  auto-build and the merge-into-caller-policy branches).
- This is the cleanest part of the subsystem.

### 5.3 Bag pipeline correctly preserves explicit FAIL strategy
(positive finding)

- `test_clone_via_bag_preserves_explicit_fail_strategy` (line
  368) regression-tests audit §6.2 in the previous audit cycle:
  a DBA passing `dangling_fk_strategy=FAIL` keeps the value
  despite FAIL also being the library default. The
  `model_fields_set` predicate is the right tool.
- See finding 3.1 — this guarantee should extend to other
  policy fields or be documented as deliberately scoped.

---

## 6. Totals

- Total findings: **32**.
- By severity:
  - P0 — **1** (1.1 — `localize_assets` integration coverage).
  - P1 — **6** (1.2, 1.3, 1.4, 3.5, 4.1, 4.2).
  - P2 — **12** (1.5, 2.1, 2.3, 3.1, 3.2, 3.3, 3.4, 3.9, 4.4,
    4.6).
  - P3 — **13** (1.6, 1.7, 2.2, 2.4, 2.5, 3.6, 3.7, 3.8, 3.10,
    4.3, 4.5, 4.7, 4.8, 4.9, 4.10).

(Section 5 is observational notes, not findings.)

## 7. Release decision

Recommended action for v1.37.1:

1. **Land at least one integration test for `localize_assets`
   happy-path** (finding 1.1). The function ships in the public
   `__init__.py` `__all__` and has zero coverage of its actual
   work. Without this, "localize works" is folklore.
2. **Either delete `reinitialize_dataset_versions` from
   `CloneDetails` and `create_ml_workspace` or implement it**
   (finding 1.4). Currently it's a documented feature that is a
   silent no-op.
3. **Update `CLAUDE.md`'s "Catalog Cloning" section** to
   describe the bag pipeline rather than the deleted
   three-stage clone (finding 4.1).
4. Treat all P2 findings as v1.38 work; P3 as opportunistic
   cleanup.

The subsystem is **release-able** at P0=1 — the single P0 is a
test-coverage gap on a feature that is already documented and
shipped. Adding the test now (or marking the function clearly as
"experimental, no integration coverage") closes the audit.

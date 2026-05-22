# deriva-ml pre-release engineer audit (v1.37.1) — `asset/`

Reviewed the asset subsystem under
`/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/asset/` (six
files, 986 LoC) and the corresponding test surface
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/asset/` (four files,
1 325 LoC) at the v1.37.1 release point. Cross-references against the
rest of `src/deriva_ml/` (notably `execution/execution.py`,
`execution/bag_commit.py`, `core/upload_layout.py`,
`local_db/manifest_store.py`, and `core/mixins/asset.py`) confirm the
in-tree consumers of the asset surface. Severity scale: P0 (block
release), P1 (must fix this release), P2 (next release), P3 (cleanup
when convenient).

## Summary

Overall posture: **the manifest-first storage layer is the strongest
part of the subsystem**. `manifest.py` is small (270 LoC), single-
purpose, well-documented, and well-tested (the `test_manifest.py`
suite covers crash recovery, point-query semantics, the batch
wrappers, and the JSON serializer in 642 LoC). The lower-level
`local_db/manifest_store.py` carries the SQLite contracts cleanly. The
NULL-sentinel processor is also tight: 43 LoC of code paired with 113
LoC of focused unit tests.

The weaker areas are:

1. **`asset.py` (the `Asset` live-catalog wrapper)** — three near-
   identical `find_association("Asset_Type") → pathBuilder → filter`
   blocks across `_load_asset_types`, `add_asset_type`, and
   `remove_asset_type`. The class also has zero coverage for
   `find_features`, `download`, and the `update_catalog` parameter on
   `download`. The retired `list_feature_values` stub raises
   `DerivaMLException` rather than `AttributeError`, which is the
   wrong typed exception under the codebase's exception hierarchy.

2. **`aux_classes.py` `AssetSpec` vs `AssetSpecConfig` divergence** —
   the hydra-zen mirror omits `asset_role`. `AssetSpec` accepts it and
   `ExecutionConfiguration` honors it, but no hydra-zen config can
   express it. This is either a real gap or `asset_role` should be
   dropped from `AssetSpec`; the inconsistency itself is the smell.

3. **Cross-module duplication of the alphabetic-sort invariant** —
   `core/upload_layout.py:214` sorts `metadata_columns` for the regex
   directory order, `execution/bag_commit.py:264` separately sorts the
   same call site for the bag-build row emit, and neither has a unit
   test pinning the invariant. If a future refactor changes either
   call's sort key (or drops it), the failure mode is silent
   corruption of NULL-sentinel column ordering at upload time.

4. **No integration test that an asset-manifest entry survives a
   process kill mid-upload and resumes correctly with the same RID.**
   The unit-level "crash recovery" test (`test_resume_after_crash`)
   uses two `AssetManifest` instances over the same store; it does
   not cover the full execution → bag-commit → catalog round-trip
   that the `MANIFEST_VERSION = 2` design promises. This is the most
   important missing test for a release that ships the
   write-through-fsync claim.

5. **Public-API/private-API smell** — `_asset_record_class` starts
   with an underscore but is re-exported from
   `deriva_ml.asset.__init__` and consumed by
   `core/mixins/asset.py:450` (`asset_record_class`). Either un-
   underscore it or move the import to a `_internal` submodule;
   exporting a leading-underscore name from `__init__` is the
   contract violation `# A private helper used by mixin code` was
   meant to prevent.

Counts: **34 findings** (1 P0, 7 P1, 17 P2, 9 P3) across the five
modules, plus 5 cross-module coverage gaps and 4 duplication
candidates.

---

## `asset/asset.py` — Asset live-catalog wrapper

### A1. `list_feature_values` raises `DerivaMLException`, not `AttributeError` [P1]

`asset.py:224-241`. The retired method raises
`DerivaMLException("Asset.list_feature_values() has been retired.
Use ml.feature_values(asset_table, feature_name) instead.")`. The
project's stated convention for retired-method stubs is to raise
`AttributeError` (so `hasattr()` and IDE tooling treat it as absent)
or to delete the method entirely. The feature-module retirement
audit (`docs/design/deriva-ml-audit-2026-05-phase3-feature.md`) shows
this same pattern was deliberately converted *away* from
`DerivaMLException` for retired container methods. Either align with
that — raise `AttributeError` — or delete the stub now that
`feature_values` has been the documented entry point through one
release cycle. Tests pin the current behavior at
`test_asset.py` (none — no regression test exists for the retired
stub; if it changes silently, no one notices).

### A2. Three duplicated `find_association("Asset_Type")` blocks [P1]

`asset.py:147-164` (`_load_asset_types`), `asset.py:243-263`
(`add_asset_type`), `asset.py:265-285` (`remove_asset_type`) all do:

```python
asset_table_obj = self._ml_instance.model.name_to_table(self.asset_table)
type_assoc_table, asset_fk, _ = self._ml_instance.model.find_association(asset_table_obj, "Asset_Type")
pb = self._ml_instance.pathBuilder()
type_path = pb.schemas[type_assoc_table.schema.name].tables[type_assoc_table.name]
```

A private helper `self._asset_type_path()` returning `(type_path,
asset_fk)` would eliminate ~18 LoC and pin the column-name convention
in one place. The current shape also re-resolves the table every
call, where caching `asset_table_obj` and `(type_assoc_table,
asset_fk)` on the `Asset` instance is the more obvious move (these
are derived from `self.asset_table`, which is immutable for the
lifetime of the wrapper).

### A3. `Asset.download` has no test [P1]

`asset.py:287-310`. The method is documented as a download primitive
(used by `lookup_asset(...).download(Path)`). It writes to
`dest_dir / self.filename` via `hatrac.get_obj`. The `update_catalog`
parameter is declared but **never read inside the method body** —
the docstring claims it "tracks this asset as an input to the
current execution", but no execution-context tracking happens here.
This is a docstring-vs-implementation drift, not just a missing test:
either implement the tracking (which probably belongs on
`Execution.download_asset`, where it already exists) or drop the
parameter from the signature.

### A4. `Asset.find_features` has no test [P2]

`asset.py:205-222`. Calls through to `ml.find_features(asset_table)`,
which is itself tested in `tests/feature/`, but no test pins the
Asset-side wrapper. A 4-line smoke test confirming a feature defined
on an asset table is returned by `asset.find_features()` is enough.

### A5. `Asset.execution_rid` makes a lazy catalog call from a property [P2]

`asset.py:166-178`. Property-access doing a catalog round-trip is the
kind of footgun that bites in N+1 loops (e.g. iterating
`ml.find_assets()` and asking each for `.execution_rid`). The lazy
load mirrors `asset_types`, but `asset_types` is at least bounded by
a single association fetch; `execution_rid` calls
`list_executions(asset_role="Output")` which goes through
`list_asset_executions → lookup_execution` per call. Document the
N+1 cost or make it explicit (`get_creating_execution()` method
instead of a property).

### A6. `Asset.__init__` declares `asset_types` as `list[str] | None` then stores `[]` [P3]

`asset.py:93,126`. The `or []` idiom is fine, but the property at
line 137-145 checks `if not self._asset_types` and triggers a load
— meaning a caller who legitimately constructs an asset with
**zero** types gets a per-property-read catalog hit. Use a sentinel
(`_UNSET = object()`) or a `_types_loaded: bool` flag to distinguish
"never loaded" from "loaded and empty".

### A7. `get_metadata` returns `{}` for missing records instead of raising [P2]

`asset.py:312-328`. If `asset_rid` no longer exists at the catalog
(deleted, snapshot mismatch), the method silently returns `{}`. The
sibling `lookup_asset` raises `DerivaMLException` for the same case.
Empty dict vs. exception is an asymmetry callers can't easily
distinguish from "row exists but has no metadata columns" (which is
itself impossible — every catalog row has at least RID/RCB/RMB
columns, but a downstream consumer can't know that). Raise.

### A8. `get_chaise_url` hardcodes `https://` and `/chaise/record/#` [P3]

`asset.py:330-345`. No test, no construction via deriva-py's URL
helpers. If a deployment ever uses `http://` (test catalogs do) or
a Chaise path prefix, the URL is wrong. At minimum: a `protocol`
fallback or a call to `self._ml_instance.catalog.host_proto` if such
an attribute exists.

### A9. Class-docstring example block uses `# doctest: +SKIP` on the class-level docstring [P3]

`asset.py:13-25` (module docstring) and `asset.py:72-80` (class
docstring) both annotate every `>>>` line with `# doctest: +SKIP`.
This works but clutters the rendered docs. The convention used
elsewhere (see `feature.py` audit) is to lift the SKIP into a top-
of-block directive when every line is skipped. Cosmetic.

---

## `asset/asset_record.py` — Dynamic Pydantic records

### B1. `_asset_record_class` exported from `__init__` despite leading underscore [P1]

`asset/__init__.py:13`. The naming says "private helper", the export
says "public API". `core/mixins/asset.py:450` calls it as
`_asset_record_class`. Either:

- Rename to `asset_record_class` (the mixin already wraps it under
  that name; the underscore prefix is the inconsistency), or
- Drop from `__all__` and the `__init__` re-export and let mixin
  code import it from `deriva_ml.asset.asset_record` directly.

The first is preferable — it's already documented in the module-
level docstring at `asset_record.py:1-12`.

### B2. `_map_column_type` swallows unknown types as `str` [P2]

`asset_record.py:39-55`. The catch-all `case _: return str` means a
column with an unrecognized ERMrest type (e.g., a future
`geography` or a typo'd typename in the model JSON) becomes a `str`
Pydantic field. The unit test at
`test_manifest.py:466-475` only covers the documented mappings;
there is no negative test confirming that an unknown type either
raises or warns. Add a log warning at minimum.

### B3. Date / timestamp / timestamptz all map to `str` [P2]

`asset_record.py:50-51`. Pydantic supports `date` and `datetime`
natively with parsing. Choosing `str` is a documented decision but
it means metadata round-trips lose type information — a caller who
constructs `ImageAsset(Acquisition_Date="2026-01-15")` gets back the
same string with no validation that it parses. At minimum, document
*why* str was chosen (likely: avoid JSON-roundtrip headaches in the
manifest store).

### B4. No test for `_asset_record_class` end-to-end [P1]

`tests/asset/test_manifest.py:466-511` tests the base class and
mapping in isolation but never calls `_asset_record_class(model,
"Image")` against a real catalog or a mocked model. The mixin entry
point `ml.asset_record_class("Image")` has zero test coverage
(`grep -r "asset_record_class" tests/` returns nothing). A
fixture-driven test that creates an asset table with three metadata
columns (text, int, optional float) and confirms the generated
class accepts/rejects values appropriately is the missing
regression net.

### B5. `_asset_record_class` does not deduplicate when called twice [P3]

`asset_record.py:92-100`. `create_model("ImageAssetRecord", ...)`
returns a fresh class every call. Two calls for the same table name
yield two distinct classes — `isinstance` checks across them fail.
Document this or cache by `(model_id, asset_table_name)`.

### B6. `Config.arbitrary_types_allowed = True` is unnecessary [P3]

`asset_record.py:34-36`. Every type produced by `_map_column_type`
is a Python builtin (`str`/`int`/`float`/`bool`/`Any`). The flag is
load-bearing only if a future column type maps to a non-Pydantic-
native type. Drop until needed.

---

## `asset/aux_classes.py` — AssetFilePath, AssetSpec, AssetSpecConfig

### C1. `AssetSpecConfig` lacks `asset_role` field [P0]

`aux_classes.py:172` (AssetSpec) vs `aux_classes.py:198-202`
(AssetSpecConfig). `AssetSpec` has `asset_role: str = "Input"` and
`tests/dataset/test_validate_execution_configuration_unit.py:281`
exercises both Input and Output roles. The hydra-zen `@hydrated_dataclass(AssetSpec)`
wrapper declares only `rid` and `cache` —
no `asset_role`. A user writing a hydra-zen config has no way to
specify `asset_role="Output"` for an asset. The
`@hydrated_dataclass(AssetSpec)` would normally pick up the fields
from the wrapped class, but the explicit `rid: str` and `cache:
bool` shadow it. This is a P0 because it's a silent functional
gap shipping in a release that documents the hydra-zen surface as
the recommended config interface.

Repro: try to set `asset_role="Output"` on a hydra-zen asset
config. There is no way.

### C2. `AssetSpec` validates `asset_role` as a free-form string [P1]

`aux_classes.py:173`. `asset_role: str = "Input"` — no enum
validation, no membership check against the `Asset_Role` vocabulary.
Typo `"Imput"` or `"input"` (lowercase) flows through to
`ExecutionConfiguration` and either no-ops silently or fails at the
catalog. Either Literal-type the field (`Literal["Input",
"Output"]`) or validate against `MLVocab.asset_role` via a
`@field_validator`.

### C3. `AssetFilePath.with_segments` returns plain `Path`, but `parent`/`parents` are silently downgraded [P2]

`aux_classes.py:136-144`. The docstring explains *why* — `Path`
subclass methods call `type(self)(*args)` and `AssetFilePath`
requires extra constructor args. The fix is correct, but the
behavioral consequence isn't tested: after `afp.parent`, callers
can't access `afp.asset_table` or `afp.asset_rid` on the parent.
`test_path_parent_and_resolve` (`test_manifest.py:340-356`) checks
`isinstance(parent, Path)` but doesn't assert that the asset
metadata is **gone**. Add the assertion so the behavior is pinned —
or, better, replace the `Path` subclassing with composition (an
`AssetFilePath` that *has* a `Path` and proxies file ops). The
subclass design is a long-standing footgun the codebase has now
worked around three times.

### C4. `_bind_manifest` is named with a leading underscore but called from `execution.py:2013` (cross-package) [P2]

`aux_classes.py:80-86`. `execution.asset_file_path` is the documented
public entry point; it constructs an `AssetFilePath` and binds it
via `result._bind_manifest(manifest, manifest_key)`. The underscore
says "private", the call site says "package-internal API". Either:

- Rename to `bind_manifest` (it's an internal protocol between
  Execution and AssetFilePath; the underscore is misleading), or
- Use `__init__` keyword arguments instead of post-construction
  binding (cleaner, but more disruptive).

### C5. `AssetFilePath.metadata` setter mutates `asset_metadata` and the manifest, but `set_asset_types` only updates the manifest if bound [P2]

`aux_classes.py:102-122` vs `124-134`. `metadata.setter` writes to
both `self.asset_metadata` (always) and the manifest (if bound).
`set_asset_types` writes to `self.asset_types` (always) and the
manifest (if bound). Symmetric. Good.

But: `metadata` *getter* reads from the manifest if bound, falling
back to `asset_metadata`. There is no symmetric getter for
`asset_types` — callers reach into `self.asset_types` directly, and
if the manifest was updated by another process or via
`update_asset_types` from elsewhere, the in-memory copy is stale.
Either add a `types` property that reads from the manifest, or
document that `asset_types` is the only field with stale-read risk.

### C6. `AssetSpec` docstring example doesn't show the `asset_role` field [P3]

`aux_classes.py:167-170`. The two examples shown are bare-RID and
`cache=True`. Add `AssetSpec(rid="3JSE", asset_role="Output")` for
the documented capability — or, if C1 leads to dropping
`asset_role` from `AssetSpec`, this becomes moot.

### C7. `_check_bare_rid` validator returns `None` for `None` input [P2]

`aux_classes.py:178-182`. `return {"rid": data} if isinstance(data,
str) else data` — if `data` is `None`, returns `None`, which then
fails the rest of Pydantic validation with an unclear error.
Acceptable, but a one-line guard (`if data is None: raise
ValueError("AssetSpec requires a RID")`) would surface a better
message.

### C8. `AssetFilePath.asset_name` backward-compat alias has no deprecation path [P3]

`aux_classes.py:147-150`. The "backward compatibility alias" returns
`self.asset_table`. Either it's deprecated (then add a
`DeprecationWarning`) or it's a permanent alias (then drop the
"backward compatibility" framing). The CLAUDE.md "No backwards-
compat shims" rule applies here — if nothing in `src/` reads
`asset_name`, delete.

Grep confirms: `grep -rn "\.asset_name" src/ tests/` returns only
the test at `test_manifest.py:376` (which exists *because* the alias
exists). Drop both.

---

## `asset/manifest.py` — Persistent manifest

### D1. `MANIFEST_VERSION = 2` is never checked on load [P1]

`manifest.py:74`. The class-level constant is documented as "bumped:
storage layer changed" — but no code reads it. `to_json()` emits it
into the dict; no loader validates it. If a v3 ever lands, there is
no migration trigger. Either:

- Remove the version field (it's vestigial), or
- Wire a check in `ManifestStore.ensure_schema` or
  `AssetManifest.__init__` that loads the version from the store
  metadata and raises on mismatch.

The migration test at
`tests/local_db/test_manifest_migration.py` confirms migration
*from* the old JSON format exists, but no version-bump test exists
for the SQLite layer itself.

### D2. `_validate_pending_asset_metadata_iter` sorts entries inside the loop [P2]

`manifest.py:215`. `for key, _schema, asset_table, metadata in
sorted(entries):` — `sorted()` materializes the entire iterable.
For a small manifest this is fine; for an execution with thousands
of pending assets, this is an unnecessary O(n log n) walk *and*
a list materialization where the function only needs a stable
iteration order for the error message. Either:

- Defer sorting until after `missing_by_key` is built (sort the
  keys of that dict instead of the input iterator), or
- Document the upper bound on `entries` size.

### D3. `_validate_pending_asset_metadata_iter` takes `schema` but ignores it [P3]

`manifest.py:202-203,215`. The tuple shape is `(key, schema,
asset_table, metadata)` but `_schema` is discarded. If schema is
never needed, drop it from the tuple shape and the call site at
`manifest.py:267`. Less surface, less to break.

### D4. `_json_default` has no test for `Decimal` or `UUID` [P3]

`manifest.py:23-35`. Handles `datetime`, `date`, `Path` — all
plausible. But ERMrest can return `numeric` columns as Python
`Decimal` (deriva-py's typed-row API), and asset RIDs are sometimes
passed around as `uuid.UUID` in tests. The serializer raises
`TypeError` for both. Either add cases or document that callers
must coerce.

### D5. `AssetEntry.status` field accepts arbitrary strings [P2]

`manifest.py:50`. `status: str = "pending"  # pending | uploaded |
failed`. The comment documents three values; the type allows any
string. A typo (`"upload"`, `"complete"`) writes silently and
breaks the `pending_assets()`/`uploaded_assets()` filters. Use a
`Literal["pending", "uploaded", "failed"]` or an `Enum`.

### D6. `AssetEntry.from_dict` silently drops unknown fields [P2]

`manifest.py:58-60`. `cls(**{k: v for k, v in data.items() if k in
cls.__dataclass_fields__})`. The intent is forward-compatibility
(reading a future manifest version with extra fields), but the
silent drop also masks bugs (a typo'd field name in caller code).
Log a warning when unknown fields are dropped.

### D7. `mark_uploaded` docstring claims the bulk delegation but mark_uploaded passes raw key without existence check [P2]

`manifest.py:129-136`. The docstring says "Delegates to the bulk
path with a one-element list; the store layer's single-row method
retains the existence check that raises ``KeyError`` for missing
keys." Inspecting `local_db/manifest_store.py:320-337` confirms
this: `mark_asset_uploaded` calls `_require_asset` then
`mark_assets_uploaded([(key, rid)])`. So the manifest-layer wrapper
*itself* doesn't enforce — it relies on the store. Fine, but the
two-layer split makes the contract harder to reason about. A
comment in `manifest.py:129` pointing at the store's
`_require_asset` would help.

### D8. `to_json` returns the legacy format but `_json_default` is referenced only in the docstring [P3]

`manifest.py:187-198`. The docstring says "Serialize with
``json.dumps(manifest.to_json(), default=_json_default)``" but the
function doesn't do that itself — every caller must remember the
incantation. Wrap into a `to_json_str() -> str` convenience that
encapsulates the serializer call.

### D9. `pending_assets()` / `uploaded_assets()` go through `list_assets()` in the store [P2]

`local_db/manifest_store.py:260-280`. Both filters delegate to
`list_assets()` (full SELECT) and filter client-side. SQLite has an
index `ix_assets_exec_status` (line 146-151) but it's unused by
these filters — they always pull every row. For an execution with
thousands of assets at upload time, this materializes the entire
set just to throw most away. Push the status filter into the SQL.

This is a `local_db/` concern but the public API contract
`AssetManifest.pending_assets()` is in `asset/manifest.py:114` —
fix at the store layer, no public API change.

### D10. `manifest.assets` is a property that does a full SELECT per access [P2]

`manifest.py:84-86`. `self._store.list_assets(self._execution_rid)`
runs on every `.assets` access. Tests like
`test_asset.py:90,93,94,96` access `.assets["Image/scan.jpg"]` —
each access is a fresh round-trip. The fix doc-comment at
`aux_classes.py:88-95` is the precedent (point-query getter); apply
the same logic here: cache or document.

Particularly painful: `to_json()` at line 197 calls `self.assets`
inside a comprehension; the comprehension reads `assets` once, but
this is non-obvious to a future caller who might add a second
access in the same function.

---

## `asset/null_sentinel_processor.py` — NULL-sentinel translator

### E1. `process()` imports `NULL_SENTINEL` lazily on every call [P3]

`null_sentinel_processor.py:38-39`. `from
deriva_ml.core.upload_layout import NULL_SENTINEL` inside the loop
hot path. The import is cached by Python after the first call, but
the symbol could just as well be module-scope. Move to the top.

### E2. Sentinel-collision risk documented but not asserted in tests [P2]

`null_sentinel_processor.py:25-28`. The docstring warns: "if a user's
legitimate metadata value equals the sentinel string, it will be
corrupted to NULL." No test exists that exercises this edge case
(asserting either: (a) we corrupt, accept it, or (b) we detect and
raise). Add a regression test pinning the documented behavior so a
future change to "detect and raise" doesn't accidentally regress
case (a).

### E3. `__init__` reads `metadata` from kwargs but `BaseProcessor` might not forward it [P2]

`null_sentinel_processor.py:31-36`. The comment "BaseProcessor stores
kwargs on self.kwargs; the metadata dict is passed via
kwargs["metadata"] by deriva-py's _execute_processors" documents a
deriva-py implementation detail. The unit tests bypass deriva-py
entirely (constructing the processor directly with `metadata=...`).
A smoke test that confirms the processor actually fires *via the
deriva-py pipeline* with a real `asset_table_upload_spec` is
missing — `test_asset_table_upload_spec_includes_null_sentinel_processor_when_metadata_present`
proves the wiring exists, not that it executes.

### E4. `process()` iterates with `list(self.metadata.items())` to mutate during iteration [P3]

`null_sentinel_processor.py:41-43`. The `list(...)` is correct
(avoiding dict-mutation-during-iteration), but a comment explaining
why the list copy is needed would prevent a future "clean up the
unused list" PR.

---

## Cross-module coverage gaps

### X1. No end-to-end resume test with a real catalog kill [P1]

Tests cover `test_resume_after_crash` at the `AssetManifest` layer
(`test_manifest.py:223-240`) using two manifest instances over the
same store. There is no test that exercises a full execution
lifecycle: stage assets → kill the process (e.g., `os.kill` in a
subprocess fixture) → resume the execution → confirm pending assets
finish uploading and uploaded assets are skipped. Given that
write-through+fsync is the headline claim of the v1 manifest design,
the absence of this test is the single biggest gap.

### X2. No test for the alphabetic-sort invariant linking `upload_layout.py:214` and `bag_commit.py:264` [P1]

Both call `sorted(model.asset_metadata(...))`. The comment at
`upload_layout.py:219-220` ("sorted to ensure deterministic
directory order matching the regex") states the invariant. Nothing
asserts that the two sites stay in sync. A unit test in
`tests/asset/` constructing an asset table with three metadata
columns out-of-order and confirming both
`asset_table_upload_spec`'s regex and the bag-build path emit them
in the same order would pin it.

### X3. No test that `find_assets(asset_type="X")` is correctly filtered [P2]

`tests/asset/test_asset.py:123-137` (`test_find_assets_by_type`)
asserts `len(assets) >= 1` and `all("Model_File" in a.asset_types
for a in assets)`. The first assertion is too weak (passes if
*all* assets carry Model_File); the second is silently true if no
assets are returned. Combine with a negative test: create an asset
without Model_File and confirm it's excluded.

### X4. No test for `Asset.add_asset_type` rejecting an unknown vocab term [P2]

`tests/asset/test_asset.py:163-184` (`test_add_asset_type`) adds a
new vocabulary term first (`vc.asset_type, "New_Type"`), then adds
it. No negative test for the case where the term doesn't exist —
which is the more common user error. The current code at
`asset.py:243-263` doesn't validate the term; the catalog FK
constraint fires. The test should pin the failure mode (catalog
error class + message shape).

### X5. No test for `Input_File`/`Output_File` auto-tagging at the asset layer [P2]

The directional auto-tagging logic lives in
`execution/execution.py:1820-1899`. The asset-side post-condition
(an asset uploaded as Output ends up with both its content type
*and* `Output_File` in `asset.asset_types`) is not asserted in
`tests/asset/`. `test_asset.py:90` checks `"Model_File" in
asset.asset_types` after upload but not the `"Output_File"` peer.

---

## Cross-module duplication candidates

### Y1. `find_association(asset_table, "Asset_Type")` appears 5+ times [P2]

`asset/asset.py:152`, `:253`, `:275`; `core/mixins/asset.py:220`,
`:343`; `execution/execution.py:1845`; `execution/bag_commit.py:359`.
Each site re-resolves the association table and `asset_fk`. A
single helper on `DerivaModel` (`asset_type_association(table) →
(assoc_table, fk_col)`) would centralize the pattern and let a
future schema change (e.g., a renamed FK column) update one place.

### Y2. `find_association(asset_table, "Execution")` appears 4+ times [P2]

`core/mixins/asset.py:285`, `execution/execution.py:1826`,
`execution/bag_commit.py:337`. Same pattern as Y1, same fix.

### Y3. Two parallel "alphabetic-sort metadata columns" sites [P1]

`core/upload_layout.py:214` (upload spec regex order) and
`execution/bag_commit.py:264` (bag-build row emit). Both rely on
`sorted(model.asset_metadata(table))`. A helper
`model.asset_metadata_sorted(table)` (or a `@cached_property` on
`Table`) would make the invariant explicit and let the two sites
share a single source of sort order.

### Y4. Asset-manifest construction in `Execution.asset_file_path` overlaps with `AssetFilePath.__init__` [P3]

`execution/execution.py:1989-2014` constructs an `AssetEntry`,
adds it to the manifest, constructs an `AssetFilePath` with the
*same* values, then calls `_bind_manifest`. The two objects (the
entry in the manifest and the in-memory `AssetFilePath`) carry
redundant state. A factory method `AssetFilePath.create_and_register
(manifest, ...)` would collapse the two-step into one.

---

## Recommended release-blockers

For v1.37.1, the following are the minimum cuts before tag:

- **C1** (`AssetSpecConfig` missing `asset_role`) — silent functional
  gap in a documented hydra-zen surface.
- **B1** (`_asset_record_class` leading-underscore re-exported) —
  contract violation, cheap to fix (rename + `__all__` update).
- **A1** (`list_feature_values` raises wrong exception type) — wrong
  exception class in a stable API, cheap to fix.
- **D1** (`MANIFEST_VERSION` never checked) — dead code or missing
  migration trigger; pick one for the release.
- **A2** (`asset.py` 3× duplication) — not strictly a release
  blocker, but the cleanup ships with the same patch that lands A1
  and a private helper makes A3's docstring fix a one-line touch.

The P0 (C1) is genuinely user-visible; the P1s are pre-release
hygiene that compounds badly if it ships into v1.37.x and a v1.38
brings additional asset surface changes.

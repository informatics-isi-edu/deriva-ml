# deriva-ml audit 2026-05 — Phase 2: dataset/

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/dataset/`
(~6 800 LoC across 13 Python files) at the tip of
`fix/catalog-manager-state-guards` (HEAD `4442f82`). External-API
checks were run against `/Users/carl/GitHub/deriva-py/deriva/bag/`
(the ADR-0006 producer/consumer module). The Phase 1 cleanup sprint
has landed: `CatalogGraph` is gone, `DatasetBagBuilder` now drives
`CatalogBagBuilder` for real, `bag_cache.py` is a `BagCacheIndex`
wrapper, `icecream` imports are gone, `LoggerMixin` is gone, the
`model/` shim layer is gone. What's left is mostly the part the
Phase 1 audit didn't reach into: the two god modules (`dataset.py`
3 579, `dataset_bag.py` 2 385) and the legacy cache/download path
inside `dataset.py` that didn't move when `bag_cache.py` did.

## Executive summary

Overall posture: **the load-bearing migration is done; the surface
that consumed the old machinery is still on the old shape.**
`DatasetBagBuilder` correctly drives `CatalogBagBuilder` per the
cutover doc. But `Dataset.download_dataset_bag` and its three
helpers (`_get_dataset_minid`, `_download_dataset_minid`,
`_create_dataset_bag_client`) still implement the legacy
`{rid}_{checksum}/Dataset_{rid}` cache layout, write
`validated_check.txt` markers, and re-implement the export-engine's
query-processor loop in pure Python — *while* the bag pipeline can
now answer all three concerns. `BagCache` already speaks the new
content-addressed index but only `Dataset.bag_info` reads it; nothing
writes through it.

Top themes ranked by impact:

1. **Two cache layouts coexist.** `bag_cache.py:143` reads via
   `BagCacheIndex`. `dataset.py:2874, 2898, 3365` writes via the
   legacy `{rid}_{spec_hash[:16]}_{snapshot}/` directory layout
   and the `validated_check.txt` marker. The two are stitched by
   `BagCache.cache_status` (`bag_cache.py:145-186`) which falls
   back to the legacy glob — a wallpaper-over-the-crack that the
   bag-cutover doc explicitly called for finishing. Until the
   write-side flips, every download writes a directory the index
   doesn't know about.

2. **`Dataset._create_dataset_bag_client` re-implements the export
   engine.** `dataset.py:3024-3255` walks the spec returned by
   `DatasetBagBuilder.generate_dataset_download_spec`, branches on
   `processor=` strings (`env`, `json`, `csv`, `fetch`), and writes
   CSVs + a remote-file manifest by hand. The bag pipeline's
   `CatalogBagBuilder.build()` (`deriva-py/deriva/bag/catalog_builder.py:167`)
   does exactly this work and writes a profile-conformant bag. The
   client-export path predates ADR-0006 and was kept "for paged
   query support and snapshot-catalog timeout handling"; both are
   now in `CatalogBagBuilder._run_export` (`catalog_builder.py:1160`).
   This is the biggest single LoC win available in `dataset/`.

3. **`Dataset` and `DatasetBag` reimplement the same operations
   differently.** `list_dataset_members`, `list_dataset_children`,
   `list_dataset_parents`, `list_executions`, `get_denormalized_*`,
   `find_features`, `feature_values`, `lookup_feature`,
   `list_dataset_element_types` — every one of these has a live-
   catalog body in `dataset.py` and a SQLite body in `dataset_bag.py`.
   The element-type loop alone is reimplemented three times (`dataset.py:1535`,
   `dataset_bag.py:501`, `_dataset_table_view` at `dataset_bag.py:359`).
   Half the divergences are necessary (live vs. SQLite); the other
   half are accidental and the tests pin both shapes.

4. **`upload.py` is in the wrong package.** Nothing in
   `dataset/` imports it; every consumer
   (`execution/execution.py`, `execution/bag_commit.py`,
   `execution/upload_engine.py`, `asset/null_sentinel_processor.py`,
   `schema/annotations.py`, `core/base.py`,
   `core/mixins/path_builder.py`) is outside `dataset/`. It is a
   642-line upload-staging-layout module that lives here for
   historical reasons.

5. **`BagCache` carries a one-shot migrator with no caller.**
   `bag_cache.py:285-384` exposes `migrate_legacy_cache` to
   walk `{cache_dir}/{rid}_*` directories and record them in
   `BagCacheIndex`. No production code path invokes it. Its sole
   reason to exist is to let users with pre-migration caches keep
   them — but the legacy-write path in `Dataset` is *also* still
   alive (theme 1), so users keep producing legacy directories
   without ever migrating them.

Worst-offending modules within `dataset/`:

1. **`dataset.py`** (3 579 LoC). `_create_dataset_bag_client`
   (231 LoC), `_get_dataset_minid` (197 LoC), and
   `_download_dataset_minid` (94 LoC) are the dead-weight. The
   `Dataset` class itself has 58 public/private methods and
   carries three concerns: catalog-side dataset CRUD, version
   management, and bag-download orchestration.

2. **`dataset_bag.py`** (2 385 LoC). Internal asset-restructuring
   helpers (`_get_asset_dataset_mapping`,
   `_load_feature_values_cache`, `_resolve_grouping_value`,
   `_detect_asset_table`, `_validate_dataset_types`,
   `_build_dataset_type_path_map`) are private to
   `restructure_assets` but parallel internally what
   `target_resolution._resolve_targets` solves for
   `as_torch_dataset` / `as_tf_dataset`. Two label-resolution
   paths through the same bag.

3. **`bag_cache.py`** (387 LoC). Healthy direction; the legacy
   fallback branches and the migrator are dead-on-arrival until
   the write-side cutover (theme 1).

---

## Subsystem inventory

| File | LoC | Posture |
|---|---:|---|
| `__init__.py` | 35 | Public-package surface. Clean. |
| `aux_classes.py` | 417 | `DatasetVersion`, `DatasetHistory`, `DatasetMinid`, `DatasetSpec`, `DatasetSpecConfig`. Pydantic. **Healthy.** |
| `bag_builder.py` | 730 | `DatasetBagBuilder`. The wrapper that drives `deriva.bag.catalog_builder.CatalogBagBuilder`. **Healthy.** |
| `bag_cache.py` | 387 | `BagCache` + `CacheStatus` + `migrate_legacy_cache`. Half-cutover (see Lens 4.1). |
| `bag_feature_cache.py` | 180 | `BagFeatureCache` (per-feature SQLite denorm cache). Single consumer: `DatasetBag.feature_values`. **Healthy.** |
| `dataset.py` | 3 579 | God module. Three responsibilities (CRUD, versions, bag download) in one class. |
| `dataset_bag.py` | 2 385 | God module. `DatasetBag` + `restructure_assets` helpers. Half overlap with `dataset.py`. |
| `split.py` | 1 276 | `split_dataset` + CLI. Independent. Carries its own logging-config code (`split.py:75, 1172-1177`) — Phase-1 §4.5 contagion. |
| `target_resolution.py` | 137 | `_resolve_targets` for adapters/restructure. **Healthy.** |
| `tf_adapter.py` | 204 | `build_tf_dataset`. **Healthy.** |
| `torch_adapter.py` | 179 | `build_torch_dataset`. **Healthy.** |
| `upload.py` | 642 | Upload-staging directory layout + spec builder. **Mis-placed** (zero in-package consumers). |
| `validation.py` | 225 | Pydantic models for spec-validation reports. **Healthy.** |

Cross-module dependencies worth naming:

- `bag_builder.py` consumes only `deriva.bag.*` + `core/`. It does
  *not* depend on `Dataset` — `DatasetLike` is the protocol it
  takes. Clean.
- `dataset.py` imports `DatasetBag` from `dataset_bag.py` (return
  type of `download_dataset_bag`) *and* `DatasetBagBuilder` from
  `bag_builder.py` (for the spec, aggregate-queries, and drift
  paths). Single highest-fanout module.
- `dataset_bag.py` consumes `target_resolution`, `bag_feature_cache`,
  `tf_adapter`, `torch_adapter` via lazy imports. Healthy structure;
  the body is just too large.
- `bag_cache.py` is only imported by `dataset.py:2678` (one call
  site — `bag_info`). Despite the module's role, the live download
  path doesn't touch it.

---

## Lens 1 — Dead code

### 1.1 `BagCache.migrate_legacy_cache` has no callers

`bag_cache.py:285-384` — 100-line migrator for pre-deriva.bag
caches. `grep -r "migrate_legacy_cache" src/ tests/` returns zero
hits; it isn't exposed on `DerivaML`, no CLI ships it, no test
exercises it. The function itself is correct but the use case it
exists for is unreachable from the current API surface.

**Two options:**

(a) Expose it as a one-shot CLI (`deriva-ml-migrate-cache`) and
document it in the migration guide.

(b) Delete it. Legacy caches will be re-downloaded on next access
and the deterministic-cache key (`{rid}_{spec_hash[:16]}_{snapshot}`)
matches the same content, so users pay one re-download per dataset
and never need the migrator.

**Recommended:** (b), because the *write-side* legacy layout
hasn't been retired either (see §4.1) — adding a one-time migrator
on top of a still-writing legacy path produces dirty caches
indefinitely.

**Risk: low. LoC: −100.** **Severity: medium** (the function is the
only piece of `bag_cache.py` that signals "migration in progress";
deleting it forces the issue).

### 1.2 `BagCache.index` property is exposed but not used

`bag_cache.py:115-123` exposes the underlying `BagCacheIndex` as a
public property for "callers that need richer index queries." Grep
across `src/` and `tests/` shows zero callers. The docstring
suggests `find_bags_for_rid`, `list_bags`, `total_size_bytes` as
intended use cases — none are invoked.

**Fix:** delete the property until a caller needs it.
**Risk: low. LoC: −15.** **Severity: low.**

### 1.3 `BagCache.cache_status` returns `versions_cached` that nobody reads

`bag_cache.py:153-172` builds and returns a `versions_cached`
list ("supplied for back-compat with the pre-migration API").
`grep -rn '\["versions_cached"\]' src/ tests/` shows zero
consumers. `Dataset.bag_info` (the single caller) returns
`cache.cache_status(...)` merged with size info but no caller of
`bag_info` reads the list either.

**Fix:** drop `versions_cached` from the return dict. Or move it
to a separate `BagCache.list_versions(rid)` method that nobody is
currently asked to call.
**Risk: low. LoC: −5.** **Severity: low.**

### 1.4 `Dataset.prefetch` is a back-compat alias on a method that has no callers needing the alias

`dataset.py:2725-2727` — `prefetch(*args, **kwargs)` is a single-
line wrapper around `cache(*args, **kwargs)` documented as
"Deprecated: Use cache() instead." `grep -rn '\.prefetch(' src/`
returns one hit (the alias itself). `grep -rn '\.prefetch(' tests/`
returns zero hits. No external project under the workspace uses
it either.

**Fix:** delete the alias. **Risk: low.** **LoC: −5.**
**Severity: low.**

### 1.5 `_default_dir_name_from_target` does an import-inside-import dance

`dataset_bag.py:67-114` imports `DerivaMLException` at module load
*and* re-imports it inside the function body (line 92). The
function-body import is dead. Same pattern at `dataset_bag.py:2096`
inside `restructure_assets`.

**Fix:** remove the redundant import. **Risk: zero. LoC: −2.**
**Severity: low.**

### 1.6 `DatasetBag.list_dataset_members._visited` etc. are marked underscore-private but appear in the public signature

`dataset_bag.py:445-543`, `829-888`, `890-941` — the recursion-
guard parameter `_visited: set[RID] | None = None` is part of every
external signature, including in `Dataset` itself (`dataset.py:1493,
2111, 2177`). Pydantic `validate_call` won't catch user values
because the same default is used; static type checkers see it as a
public kwarg. The leading `_` is a fig leaf.

**Fix:** either prefix with the standard "private" call wrappers
(`_list_dataset_members_internal(visited=...)`) or document
`_visited` as an internal sentinel callers must not pass.
**LoC: ±0** for documentation; **+~20 / −~20** for splitting.
**Severity: low.**

### 1.7 `DatasetBag.fetch_table_features` / `list_feature_values` are retired but still defined

`dataset_bag.py:772-811` defines two methods that exist only to
raise `DerivaMLException("retired")`. Each carries an 18-line
docstring explaining what to use instead. Their presence makes the
public surface noisier than a `__getattr__` would.

**Fix:** delete both methods and let attribute access raise
`AttributeError`. Or replace with a `__getattr__` that maps the
two stale names to a single `DerivaMLException`.
**Risk: low** (the deprecation messages are still relevant; users
hitting them get a clear pointer either way). **LoC: −45.**
**Severity: low.**

### 1.8 `Dataset._list_dataset_parents_current` / `_list_dataset_children_current` are private but exist for one caller

`dataset.py:2232-2257` — two near-duplicate methods that exist
because `_build_dataset_graph_1` (`dataset.py:859`) needs a
version-agnostic walk. Each is ~12 lines, both duplicate the
relevant branch of `list_dataset_parents` / `list_dataset_children`
with `version=None`.

**Fix:** replace with `list_dataset_parents(version=None)` /
`list_dataset_children(version=None)` calls. The version=None
case already exists in the public methods. **LoC: −25.**
**Severity: low.**

### 1.9 `DatasetBag.__init__` argument `dataset_types` is unused-as-input

`dataset_bag.py:170-217` — the constructor accepts
`dataset_types: str | list[str] | None` and stores it. No caller
in `src/` or `tests/` ever passes this argument (verified by
`grep -rn 'DatasetBag(' src/ tests/`). Every call path goes
through `DerivaMLBagView.lookup_dataset(rid)` which doesn't
forward the kwarg.

**Fix:** drop the parameter and the normalisation block at
`dataset_bag.py:208-214`. Promote `dataset_types` to a computed
property that reads from the SQLite `Dataset_Dataset_Type` table
on demand (matching the live `Dataset.dataset_types` property at
`dataset.py:230`).
**Risk: medium** (one tests asserts `bag.dataset_types == [...]`;
spot-check first). **LoC: −15 + +10.** **Severity: medium** — the
two classes have parallel `dataset_types` surfaces that disagree
on storage (list field on bag, live query on dataset).

### 1.10 Stale `CatalogGraph` references in test docstrings

The class is deleted but three test files still mention it:

- `tests/dataset/test_bag_builder.py:17-22` — docstring describes
  "the bag-content equivalence harness that gated the cutover from
  ``CatalogGraph``."
- `tests/dataset/test_bag_builder.py:42` — `TestSpecSmoke` class
  doc says "Not equivalence tests — those live in
  `TestBagEquivalence`." `TestBagEquivalence` was deleted with
  `CatalogGraph`.
- `tests/dataset/test_composite_fk_denormalize.py:257` — code
  comment "This tests the FK path traversal in CatalogGraph."
- `tests/dataset/test_estimate_bag_size.py:60` — "CatalogGraph._aggregate_queries".

**Fix:** scrub the stale references. **LoC: −5 doc-only.**
**Severity: low.**

---

## Lens 2 — Deriva-py / deriva.bag interface usage

### 2.1 `Dataset._create_dataset_bag_client` re-implements `CatalogBagBuilder._run_export`

`dataset.py:3024-3255` is the biggest single hand-rolled
mirror-of-deriva-py left in `dataset/`. It:

1. Takes the spec dict from `DatasetBagBuilder.generate_dataset_download_spec`.
2. Walks `spec["catalog"]["query_processors"]`.
3. Switches on `processor in {"env", "json", "csv", "fetch"}`.
4. For `csv`: calls `catalog.get_as_file(... paged=True ...)`.
5. For `fetch`: builds a remote-file manifest from JSON.
6. Calls `bdb.make_bag(remote_file_manifest=...)`.

The deriva-py equivalent is already what `CatalogBagBuilder.build()`
(`deriva-py/deriva/bag/catalog_builder.py:167-185`) drives via
`_run_export` (`catalog_builder.py:1160-1196`). The deriva-py path:

- Reads the same `query_processors` list (it built it).
- Uses the canonical `DerivaExport` driver inside deriva-py
  (`deriva/transfer/download/deriva_export.py`).
- Handles the same paged-query, env-template, and remote-file-
  manifest concerns the dataset.py code does.
- Writes a profile-conformant bag with the `BagIt-Profile-Identifier`
  declared.

The historical motivation for the deriva-ml copy was "the
deriva-py export driver doesn't tolerate snapshot-catalog
timeouts" — `dataset.py:3151-3163` even has the "tolerate
individual query failures" branch. After ADR-0006 that
tolerance is `FKTraversalPolicy.exclude_tables` (callers can
pre-prune) and `CatalogBagBuilder.build`'s own retry loop.

**Recommended action:** delete `_create_dataset_bag_client`. Make
`_create_dataset_minid(use_minid=False, ...)` call
`CatalogBagBuilder(...).build()` and archive the result to the
same `client_export` directory shape. Confirmed:
`CatalogBagBuilder.build` returns the bag directory path
(`catalog_builder.py:170`); the existing tier-3 wiring in
`_get_dataset_minid` already expects a `file://` URL that can be
constructed from that path.

**Risk: medium-high** (live-catalog smoke needed; the
`DatasetMinid.RID = "{rid}@{snap}"` invariant + the
`spec_hash`/`snapshot` cache key need to round-trip cleanly).
**LoC: −230.** **Severity: high.**

### 2.2 `Dataset._download_dataset_minid` and `_bag_is_fully_materialized` parallel `bag_cache.py:_is_fully_materialized`

`dataset.py:3475-3503` is a 28-line method that:

1. Calls `bdb.validate_bag_structure`.
2. Reads `fetch.txt`.
3. Returns True iff every line's third field references an
   existing file.

`bag_cache.py:230-262` (`BagCache._is_fully_materialized`) does
*exactly* this. Two implementations of the same predicate, one
on `Dataset` (with `self._logger.debug` — but actually broken at
line 3491: it references `self._logger` from inside a `@staticmethod`,
so it would `NameError` on a validation failure today) and one
on `BagCache` (working).

**The static method is broken.** `dataset.py:3475` declares
`@staticmethod`; `dataset.py:3491` calls `self._logger.debug(...)`.
The only reason this isn't a live bug is that the `try`/`except`
swallows the `bdb.validate_bag_structure` exception path almost
always when the bag is valid. Trip the validation, and you
crash with `NameError` instead of returning `False`.

**Fix:** delete `Dataset._bag_is_fully_materialized` and call
`BagCache._is_fully_materialized` (or expose it as a module-level
helper in `bag_cache.py`). Two callsites
(`dataset.py:3489, 3550`) update to use the public helper.
**Risk: low. LoC: −30 (delete + fix the crash).** **Severity: medium**
— there's a latent bug in the staticmethod's error path.

### 2.3 `_create_dataset_bag_client` builds the remote-file manifest by hand

`dataset.py:3231-3243` writes the JSON-stream remote-file manifest
line-by-line:

```python
entry = {"url": url, "length": int(length) if length else 0, "filename": filename}
if md5: entry["md5"] = md5
f.write(json.dumps(entry) + "\n")
```

The repo's own CLAUDE.md ("BDBag Remote File Manifest") documents
the format and points at `deriva_download.py` in deriva-py as
the canonical implementation. The same routine is now also
exposed via `deriva.bag.profile` helpers (verified at
`deriva-py/deriva/bag/profile.py`).

**Fix:** subsumed by §2.1 — once the client export uses
`CatalogBagBuilder.build`, the manifest writer is gone with the
rest of `_create_dataset_bag_client`.

### 2.4 `bag_builder.py:_catalog_bag_builder` stashes a `TemporaryDirectory` on the instance to satisfy a constructor requirement

`bag_builder.py:319-332`:

```python
tmp = TemporaryDirectory()
output_dir = Path(tmp.name)
builder = CatalogBagBuilder(catalog=..., anchors=..., output_dir=output_dir, policy=...)
builder._datasetbag_output_tmp = tmp  # type: ignore[attr-defined]
```

The comment at 315-318 explains: "`CatalogBagBuilder` requires an
`output_dir` even when nothing will be written; `aggregate_queries`
and the annotation path don't run `build`." Two of the three
public methods on `DatasetBagBuilder` *never* call `build`, so
the constructor's required `output_dir` is dead weight for those
paths.

**This is a deriva-py-side defect**, not a deriva-ml one — the
audit of deriva-py would file it as "CatalogBagBuilder.output_dir
should be optional (or lazy) when only read-only methods are
called." Until deriva-py changes, the `_datasetbag_output_tmp`
attribute-poke is the right workaround. **No deriva-ml fix.**
**Severity: low.** Flag as a deriva-py follow-up.

### 2.5 `Dataset.estimate_bag_size._extract_path` parses a URI back into a path

`dataset.py:2507-2517`:

```python
def _extract_path(uri: str) -> str:
    for marker in ("/aggregate/", "/entity/", "/attribute/"):
        idx = uri.find(marker)
        if idx >= 0:
            return uri[idx:]
    raise ValueError(f"Cannot extract catalog path from URI: {uri}")
```

This is the same anti-pattern Phase 1 §2.5 flagged on
`CatalogGraph.source_path` (which is now deleted). `estimate_bag_size`
needs the path-only string because `AsyncErmrestCatalog.get_async`
takes a path, not a full URI. The datapath's `.uri` attribute is the
canonical way to get the full URI; deriva-py exposes `.path` on the
same object (`deriva-py/deriva/core/datapath.py:DataPath`) — using
it avoids the substring-marker scan.

**Fix:** replace the search-for-marker logic with
`dp.path` (or `dp.uri.split('?', 1)[0].rsplit('/ermrest/catalog/' + cat_id, 1)[1]`
if `.path` doesn't yet exist on the public API — confirmed it does:
`deriva-py/deriva/core/datapath.py` exposes `_path` privately,
should be promoted). **Risk: low.** **LoC: −10.** **Severity: low.**

### 2.6 `estimate_bag_size` uses `AsyncErmrestCatalog` directly instead of the bag pipeline

`dataset.py:2486-2568` constructs an `AsyncErmrestCatalog` /
`AsyncErmrestSnapshot`, builds datapaths through
`DatasetBagBuilder.aggregate_queries`, then evaluates each
datapath against the async catalog by extracting path strings
and calling `catalog.get_async(query_path)`.

The work this is doing — "evaluate every datapath the bag walk
discovered, with custom filters / aggregations, against a live
snapshot catalog" — is exactly what
`CatalogBagBuilder.iter_table_datapaths` returns
(`catalog_builder.py:210`). The datapaths it returns are already
bound to the live catalog. The async dance is an optimisation
that doesn't have a sibling on the bag pipeline today, but the
cost in code size (~80 LoC of async glue + the URI parser from
§2.5) is high for the gain.

**Recommendation:** flag as Phase 3. Either lift the async-query
optimisation upstream to `CatalogBagBuilder` (and let
deriva-ml just call `iter_table_datapaths` and iterate), or
document `estimate_bag_size` as the one place we deliberately
bypass the bag-builder for parallelism.
**LoC: net ±0 today; potentially −80 if upstream changes.**
**Severity: low.**

### 2.7 `bag_builder.py:_dataset_visible_fkeys` could use `pure=False, max_arity=3` consistency

`bag_builder.py:535` calls `dataset_table.find_associations(max_arity=3, pure=False)`.
The other `find_associations()` callsites in `dataset.py:1535, 1959, 2056`
call with default args. The two argument shapes return different
sets of associations (`pure=False` includes those with annotation
columns), so the bag annotation enumerates a superset of what
`add_dataset_members` validates against. Phase 1 §2.7 noted the
same divergence on the `is_association` side; the
`find_associations` callsites have the same drift.

**Recommendation:** decide the canonical args for the
"dataset's element-type association enumeration" and use it in
all four callsites. **LoC: ±0 (single-line change × 4).**
**Severity: low.**

---

## Lens 4 — Inconsistencies / duplication

### 4.1 Two cache layouts coexist; the legacy one is still being written

**The hazard:** `dataset.py` is the only `dataset/` module that
*writes* a cached bag. It writes to the pre-ADR-0006 layout
(`dataset.py:2874, 2898, 3365, 3387, 3454`):

```python
bag_dir = self._ml_instance.cache_dir / f"{minid.dataset_rid}_{minid.checksum}"
```

and uses `validated_check.txt` markers (`dataset.py:3545-3578`)
for materialization status.

`bag_cache.py` reads via `BagCacheIndex`
(`deriva-py/deriva/bag/cache_index.py:102`) — the index expects
bag content under `{cache_dir}/bags/{checksum}/` and records
`(checksum, anchor table, anchor RID)` triples in
`index.sqlite`. The reader at `bag_cache.py:143-186` has a
fallback branch that globs for the legacy `{rid}_*` directories
because — per the cutover doc — the writer cutover was
deferred.

**Effect today:**

- Every `download_dataset_bag` call writes to the legacy layout.
- `BagCache.cache_status` finds those via the legacy glob, not via
  the index. The index stays empty for downloaded datasets.
- `BagCacheIndex.find_bags_for_rid` (the canonical "is this
  cached?" API) returns no results for downloaded bags.
- `migrate_legacy_cache` (§1.1) exists to backfill the index from
  the legacy dirs, but it's never called.

**Recommendation:** make `download_dataset_bag` write through the
index. Three changes:

1. Replace the `{rid}_{checksum}` directory construction in
   `_download_dataset_minid` and `_get_dataset_minid` with
   `BagCache.index.bag_dir_for(checksum)` (already returns
   `bags/{checksum}/`).
2. Replace the `validated_check.txt` marker with the materialised
   status that `BagCacheIndex` already records.
3. Once #1 and #2 land, delete the `cache_status` legacy-fallback
   branch at `bag_cache.py:174-186` and the migrator.

**Risk: high** (most-touched in `dataset/`; the cache layout
change ripples into every test in `test_dataset_caching.py` and
`test_deterministic_cache.py`). **LoC: ~−150 (delete fallback +
migrator + validated_check + the legacy directory branch in
`_download_dataset_minid`).** **Severity: high.**

### 4.2 `Dataset.list_dataset_members` and `DatasetBag.list_dataset_members` reimplement the element-type loop

`dataset.py:1532-1577` (catalog-side, via datapath):

```python
for assoc_table in self._dataset_table.find_associations():
    other_fkey = assoc_table.other_fkeys.pop()
    target_table = other_fkey.pk_table
    member_table = assoc_table.table
    if not self._ml_instance.model.is_domain_schema(...) and not (
        target_table == self._dataset_table or target_table.name == "File"
    ):
        continue
    ...
```

`dataset_bag.py:498-543` (bag-side, via SQLAlchemy):

```python
for element_table in self.model.list_dataset_element_types():
    element_class = self.model.get_orm_class_for_table(element_table)
    assoc_class, dataset_rel, element_rel = self.model.get_association_class(...)
    element_table = inspect(element_class).mapped_table
    if not self.model.is_domain_schema(element_table.schema) and element_table.name not in ["Dataset", "File"]:
        continue
    ...
```

The filter shape is identical (domain-schema OR Dataset/File);
the body differs because one's a datapath fetch and the other's
a SQLAlchemy join. The two are correct but easy to drift —
the live version dropped the magic "File" name as a string
once (in `dataset.py:1542` `target_table.name == "File"`) but
the bag version still hardcodes the list (`["Dataset", "File"]`).
Sibling concerns:

- `add_dataset_members` (`dataset.py:1955-2009`) builds the same
  `association_map = {target_table.name: assoc_table.name}` again
  (`dataset.py:1959-1960`).
- `delete_dataset_members` (`dataset.py:2055-2057`) again.

That's four implementations of "which assoc table covers which
target table for the Dataset anchor."

**Fix:** lift to a single helper on `DerivaModel` or
`Dataset._build_association_map(self) -> dict[str, str]` and call
it from the four sites. **Risk: low** (mechanical). **LoC: −40.**
**Severity: medium** — every drift between these four maps is a
silent membership bug.

### 4.3 `Dataset.list_dataset_members` vs. `DatasetBag._dataset_table_view`

`DatasetBag._dataset_table_view` (`dataset_bag.py:359-404`)
computes "all rows in a table that belong to this dataset by
walking every FK path" and UNIONs the per-path queries.
`Dataset.list_dataset_members` (`dataset.py:1489-1578`) computes
"all rows in element tables directly associated with this dataset
via the association table." Different definitions — the bag's
view is broader (transitive FK reachability) and the live
catalog's view is narrower (direct membership only).

This divergence is **intentional** for `list_dataset_members` —
the contract is "direct members" — but `DatasetBag` then *also*
exposes `list_dataset_members` (`dataset_bag.py:445-543`) which
*matches* the live-catalog narrow definition, while
`_dataset_table_view` is used internally by `_get_reachable_assets`
to do the broader walk for `restructure_assets`. So inside
`DatasetBag` we have two different "which rows belong here"
definitions — and only one is documented at the call site.

**Recommendation:** name them. Promote `_get_reachable_assets`
to a public `list_reachable(...)` method with a clear docstring
distinguishing it from `list_dataset_members`. The internal
helpers stay; only the naming and a docstring change.
**Risk: low. LoC: ±0.** **Severity: low** — but the two-meaning
collision will trip a user every release if not labeled.

### 4.4 Label resolution: two paths in `dataset_bag.py`

`dataset_bag.py:1267-1441` (`_load_feature_values_cache`)
implements label / feature-value resolution for
`restructure_assets`. It supports column targets, feature-name
targets, dotted `Feature.column` syntax, vocab enforcement,
selector callables — 175 LoC.

`target_resolution.py:_resolve_targets` does the same logical job
(walks `bag.feature_values(target_name)`, groups by RID,
applies missing-policy) in 137 LoC and is used by
`as_torch_dataset` / `as_tf_dataset`. `restructure_assets` calls
`_resolve_targets` *for the feature path* (`dataset_bag.py:2184-2189`)
*and* `_load_feature_values_cache` *for vocabulary enforcement*
(`dataset_bag.py:2211-2212`).

The duplication is a transition artifact. `_resolve_targets` is
the post-spec implementation; `_load_feature_values_cache`
predates it and persists because (a) `restructure_assets`'s
column-vs-feature classification was harder to fold in, and (b)
the cache-style return type differs from what `_resolve_targets`
emits.

**Recommendation:** extend `_resolve_targets` with an
`enforce_vocabulary` mode and a `value_selector` parameter,
then delete `_load_feature_values_cache`. The classification
between "column" and "feature" stays in
`restructure_assets`; only the value-extraction loop merges.
**Risk: medium** (multiple callers, behavior is subtle). **LoC: −125.**
**Severity: medium** — biggest unforced duplicate in `dataset_bag.py`.

### 4.5 Snapshot resolution duplicated three ways

The same "given a dataset_rid and a version, get the catalog
snapshot" computation appears at:

- `dataset.py:2781-2801` (`_version_snapshot_catalog`) — returns
  a catalog bound to the snapshot.
- `dataset.py:2803-2833` (`_version_snapshot_catalog_id`) — returns
  the `"{cat_id}@{snap}"` string.
- `dataset.py:3318-3326` — inlined again in `_get_dataset_minid`
  for the version-record lookup.

`_version_snapshot_catalog_id` walks `dataset_history` to find
the version row; `_get_dataset_minid` walks `dataset_history`
again to find the same row. Two history fetches per download.

**Fix:** rewrite `_get_dataset_minid` to call
`_version_snapshot_catalog_id` instead of re-implementing the
walk. **LoC: −20.** **Severity: low.**

### 4.6 `dataset.py` and `dataset_bag.py` reimplement `list_dataset_children` / `list_dataset_parents` similarly but not identically

Compare `dataset.py:2173-2230` (catalog) and `dataset_bag.py:829-941`
(bag). The recursion guard (`_visited`) is identical but stored
on the parameter; the cycle-breaking logic is identical; the
ordering policy (depth-first via `copy()` of the working list)
is identical. Only the data-fetch differs.

This is the "live versus bag" version of the same class of split
called out in Phase 1's "two protocols, one operation" pattern.
**Phase-3 scope:** investigate whether the two could share a
common base that takes a `_fetch_neighbors(rid) -> list[RID]`
callable. Not actionable in a quick-cleanup PR.

### 4.7 `bag_cache.py:CacheStatus` carries an aliased member that doesn't behave like an alias

`bag_cache.py:47-76`:

```python
class CacheStatus(StrEnum):
    not_cached = "not_cached"
    cached_metadata_only = "cached_metadata_only"
    cached_materialized = "cached_materialized"
    cached_holey = "cached_holey"
    cached_incomplete = "cached_holey"   # alias
```

`CacheStatus.cached_incomplete` is the same enum member as
`CacheStatus.cached_holey` (StrEnum aliasing rules), so
`==` works, but `repr(CacheStatus.cached_incomplete)` shows
`<CacheStatus.cached_holey: 'cached_holey'>`. The "alias" docstring
implies a backward-compatibility helper; the actual behavior is
identical to deleting `cached_incomplete` and trusting users to
have migrated.

Phase 1 §4.3 already flagged this; the change in `bag_cache.py:76`
is the actual one to make. **LoC: −2 (delete the alias line).
Severity: low.**

---

## Lens 5 — Simplification opportunities

### 5.1 Delete `_create_dataset_bag_client` (subsumes §2.1, §2.3)

Net: **−230 LoC**, biggest single LoC win in `dataset/`. Replace
the body of `_create_dataset_minid(use_minid=False, ...)` with a
`CatalogBagBuilder(catalog=..., anchors=anchors_for(...), output_dir=...).build()`
call. Map the result to a `file://` URL.

The path-builder construction (`DerivaServer(... session_config)`
from `dataset.py:3056-3061`) for snapshot-catalog timeouts is
already inside `CatalogBagBuilder.__init__` (it accepts a
prebuilt `catalog` handle — pass the snapshot one in).

Two pieces of work to verify:

- `bag_builder.py`'s `_minid_post_processors` (lines 340-364)
  is the post-processor injection point for client export. With
  `_create_dataset_bag_client` gone, the post-processor list
  becomes unconditionally part of the spec; `use_minid=False`
  means "don't add the cloud-upload processor" — the spec
  body stays the same.
- The deterministic cache key (`{rid}_{spec_hash[:16]}_{snapshot}`)
  becomes the `BagCacheIndex` lookup key after §4.1 lands.

**Severity: high** (action 1 below).

### 5.2 Unify the cache layout (§4.1)

Net: ~**−150 LoC** of fallbacks + migrator + validated_check.
This is the most invasive of the recommendations; it requires
a test-suite pass to confirm the cache-status semantics are
unchanged. Order of operations: do §5.1 first (the legacy writer
goes), then §5.2 (the legacy reader fallback can go).

### 5.3 Fold `_load_feature_values_cache` into `_resolve_targets` (§4.4)

Net: **−125 LoC**. The function it replaces is private and has
two callers; refactor confined to `dataset_bag.py` +
`target_resolution.py`.

### 5.4 Build a single `_association_map` helper (§4.2)

Net: **−40 LoC**, low risk.

### 5.5 Replace `_bag_is_fully_materialized` with the working `BagCache._is_fully_materialized` (§2.2)

Net: **−30 LoC** and fixes the `self._logger`-in-staticmethod
crash.

### 5.6 Delete `_list_dataset_*_current` duplicates (§1.8)

Net: **−25 LoC**.

### 5.7 Move `upload.py` to `execution/upload_layout.py` (or similar)

Mechanical move — every consumer is in `execution/`,
`asset/`, `schema/`, `core/`. No `dataset/` module imports it.
The current location is a Phase-1-era organizational accident.

**Risk: low** (mechanical rename + import update). **LoC: ±0.**
**Severity: medium** — the package directory is misleading. New
maintainers reading `dataset/` will not expect to find the
upload-staging spec there.

### 5.8 Inline `Dataset.prefetch` away (§1.4)

Trivial.

### 5.9 Use `dp.path` instead of substring marker (§2.5)

Trivial. Pending deriva-py promotion of `_path` → `path`.

---

## Lens 6 — Maintainability

### 6.1 `dataset.py` carries three responsibilities

`Dataset` class has 58 methods (counting helpers). They cluster
into:

- **Dataset CRUD** (lines 213-541, 1864-2257):
  `create_dataset`, `dataset_types`,
  `add_dataset_type`/`remove_dataset_type`/`add_dataset_types`,
  `list_members`, `list_dataset_members`/`*_children`/`*_parents`,
  `add_dataset_members`/`delete_dataset_members`, `list_executions`.
- **Version management** (lines 675-1488):
  `dataset_history`, `current_version`, `release`,
  `is_dirty`, `release_diff`, `compare_versions`, `mark_dev`,
  `_create_or_advance_dev_row`, `_build_dataset_graph*`,
  `_increment_dataset_version`, `_release_diff_bounds`,
  `_resolve_version_to_rmt`, `_iter_drift_counts`,
  `_insert_dataset_versions`.
- **Bag download orchestration** (lines 2355-3579):
  `download_dataset_bag`, `estimate_bag_size`, `bag_info`,
  `cache`, `prefetch`, `_create_dataset_minid`,
  `_create_dataset_bag_client`, `_get_dataset_minid`,
  `_download_dataset_minid`, `_materialize_dataset_bag`,
  `_fetch_minid_metadata`, `_version_snapshot_catalog*`,
  `_bag_is_fully_materialized`, `_estimate_csv_bytes`,
  `_human_readable_size`.

The bag-download cluster is ~1 200 LoC. With §5.1 it shrinks
significantly. Phase 3 scope: consider splitting `dataset.py`
into `dataset_core.py` (CRUD+versions) + `dataset_download.py`
(bag-download orchestration). The current `dataset_bag.py` is a
different concern (read-side bag access), not the place for the
download orchestration code.

**Severity: medium** (size; mixed concerns).

### 6.2 `dataset_bag.py` carries three responsibilities too

- **Bag access** (lines 117-1051): `DatasetBag` SQLAlchemy-backed
  catalog mirror.
- **Asset reachability** (lines 1160-1481):
  `_build_dataset_type_path_map`, `_get_asset_dataset_mapping`,
  `_get_reachable_assets`, `_load_feature_values_cache`,
  `_resolve_grouping_value`, `_detect_asset_table`,
  `_validate_dataset_types`. ~320 LoC.
- **Restructure / framework adapters** (lines 1483-2381):
  `_validate_dataset_types`, `as_torch_dataset`, `as_tf_dataset`,
  `restructure_assets`. ~900 LoC, half of which is one method's
  docstring.

The asset-reachability cluster's only consumer is
`restructure_assets`. Phase 3 scope: extract to
`restructure.py` adjacent to `target_resolution.py`.

**Severity: medium.**

### 6.3 Docstring quality is **good**, with a few drifted comments

Sampled the public methods on `Dataset` and `DatasetBag` — the
Google-format `Args` / `Returns` / `Raises` / `Example` skeleton is
filled in correctly almost everywhere. Three exceptions:

- `dataset.py:1939-1944` — `check_dataset_cycle` docstring is the
  template-default placeholder (`Args:` and `Returns:` are empty
  bullet headers with no content). Same pattern as Phase 1 §6.3
  flagged in `model/catalog.py`.
- `dataset.py:1499` — `list_dataset_members` says "returns a
  dictionary mapping member types to lists of member records"
  without specifying that the keys are *element table names*. The
  bag version of the same docstring (`dataset_bag.py:445-486`) is
  precise.
- `dataset_bag.py:264-269` — the `path` property's
  `Example::` block contains code that calls
  `ml.download_dataset_bag(spec)` — but `ml` has no
  `download_dataset_bag` method (it's on `Dataset`, accessed via
  `dataset.download_dataset_bag(...)`). Subtle but wrong.

**LoC: ±0 (docs).** **Severity: low.**

### 6.4 Naming: `bag_path`, `bag_dir`, `cached_bag_path`, `cache_dir`, `cache_suffix`, `cache_dir_name`

In a single 100-line block (`dataset.py:3318-3455`) we have:

- `cache_dir` (instance prop on the catalog),
- `cache_dir_name` (the legacy directory's name string),
- `cached_dir` (the path Object for the legacy directory),
- `cached_bag_path` (the bag-dir inside the cache dir),
- `cache_suffix` (the `spec_hash[:16]_snapshot` suffix string),
- `bag_dir` (the same as `cached_dir` but constructed differently
  three lines later in `_download_dataset_minid`),
- `bag_path` (returned from `_materialize_dataset_bag`).

Eight names, four concepts. Phase 1 §6.5 already noted the
problem at the inter-module level; intra-`dataset.py` it's worse.

**Severity: low** (cosmetic but a documented friction point).

### 6.5 `Dataset.dataset_types` re-fetches on every access

`dataset.py:230-253` — the property runs a fresh catalog
fetch every time. The docstring is explicit about this ("ensures
consistency when multiple Dataset instances reference the same
dataset"), but a single notebook loop that calls
`dataset.dataset_types` four times pays four round-trips.

Not a bug — the contract is correct — but inconsistent with the
analogous `DatasetBag.dataset_types` which is a frozen attribute
(`dataset_bag.py:210-214`). The "live versus immutable snapshot"
boundary is right, but no caller of `dataset.dataset_types` has
chosen "live read"-semantics on purpose.

**Recommendation:** add an internal cache invalidated by
`add_dataset_type`/`remove_dataset_type`. **LoC: +20.**
**Severity: low.**

### 6.6 `__all__` is missing on every non-`__init__` module

Spot-checked `dataset.py`, `dataset_bag.py`, `bag_builder.py`,
`bag_cache.py`, `aux_classes.py`, `validation.py`. None define
`__all__`. `dataset/__init__.py` does, but it re-exports a curated
subset. Anyone reaching into a module path directly gets every
import and every helper.

`bag_builder.py:730` is the exception (`__all__ = ["DatasetBagBuilder"]`).
`bag_cache.py:387` does too.

**Recommendation:** add `__all__` to `dataset.py`, `dataset_bag.py`,
`aux_classes.py`, and `validation.py`. **LoC: +20.**
**Severity: low.**

### 6.7 `_visited` parameter is "private" but public

See §1.6. A maintainability gripe more than a bug.

### 6.8 `Dataset.create_dataset` is a staticmethod that uses `cls` in its docstring example

`dataset.py:255-333` — declared `@staticmethod` (line 255 is
inside the class). The docstring at lines 270-280 shows
`dataset = Dataset.create_dataset(...)` — fine — but the type
annotation returns `Dataset` (the class), not `Self`. Since it's
a staticmethod, `Self` wouldn't help, but the method is morally
a class constructor and should be `@classmethod` returning `Self`.

**Severity: low.** Phase-3 polish.

### 6.9 `split.py:_logger` initialization at line 1172

The CLI entry point initializes logging by hand instead of using
`configure_logging`:

```python
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
```

Phase 1 §4.5 was about adopting `get_logger`; this CLI never
adopted `configure_logging`. **LoC: −10.** **Severity: low.**

---

## Ranked actions (1–N)

Ranked by `(impact × confidence) / cost`. Items 1–3 are the high-
value cleanups; 4–7 are mechanical wins; 8–10 are Phase-3 scope.

| # | Action | Risk | LoC | Files | Rationale |
|---|---|---|---:|---|---|
| 1 | **§2.1 + §5.1** Delete `Dataset._create_dataset_bag_client`; route `_create_dataset_minid(use_minid=False)` through `CatalogBagBuilder.build()` | medium-high | −230 | `dataset.py` | Biggest single LoC win; eliminates a hand-rolled mirror of `_run_export`. Required precursor to §2 (cache layout cutover). |
| 2 | **§4.1 + §5.2** Cache-layout cutover: write through `BagCacheIndex`, delete `validated_check.txt` markers, drop the legacy-glob fallback in `BagCache.cache_status` | high | −150 | `dataset.py`, `bag_cache.py` | Half-cutover today. After action 1, this is the last legacy write path in `dataset/`. |
| 3 | **§4.4 + §5.3** Fold `_load_feature_values_cache` into `_resolve_targets` with `enforce_vocabulary` + `value_selector` | medium | −125 | `dataset_bag.py`, `target_resolution.py` | Removes the largest unforced duplicate in `dataset_bag.py`. |
| 4 | **§2.2 + §5.5** Delete `Dataset._bag_is_fully_materialized` (fixes the staticmethod `self._logger` bug); call `BagCache._is_fully_materialized` instead | low | −30 | `dataset.py`, `bag_cache.py` | Latent crash on validation-failure path; net code reduction. |
| 5 | **§5.7** Move `upload.py` out of `dataset/` (e.g., `execution/upload_layout.py`) | low | ±0 | `upload.py` + ~8 import sites | Mechanical; the directory currently lies about its contents. |
| 6 | **§4.2 + §5.4** Lift `association_map` construction to a single helper called from `list_dataset_members`, `add_dataset_members`, `delete_dataset_members`, and `DatasetBag.list_dataset_members` | low | −40 | `dataset.py`, `dataset_bag.py` | Four parallel implementations of "what assoc table covers what element type" today. |
| 7 | **§1.1 + §1.2 + §1.3 + §1.4 + §1.5 + §1.7 + §1.8 + §1.9 + §1.10 + §4.5 + §4.7 + §5.6 + §5.8 + §5.9 + §6.3 + §6.6 + §6.9** Pure dead-code / duplicate-helper / stale-doc sweep | low | ~−250 | ~10 files | Mechanical cleanup; each item small but they accumulate. |
| 8 | **§2.5 + §2.6** Replace URI substring parser with `dp.path` and consider lifting the async-query loop to `CatalogBagBuilder.iter_table_datapaths` | low-medium | −10 to −90 | `dataset.py`, deriva-py | Half is a deriva-py promotion; the rest is a Phase-3 question. |
| 9 | **§4.6** Phase 3 — share `list_dataset_children` / `list_dataset_parents` body between `Dataset` and `DatasetBag` via a `_fetch_neighbors` callable | medium | −60 | `dataset.py`, `dataset_bag.py` | Real cleanup needs the "live versus bag" base class question answered. |
| 10 | **§6.1 + §6.2** Phase 3 — split `dataset.py` into CRUD/versions/download and `dataset_bag.py` into access/restructure | medium-high | ±0 net (LoC moves) | several | Out of audit scope per the brief; flagged as motivation. |

Items 1 + 2 ship together (the cache cutover doesn't make sense
without the writer flip). Items 4–7 are good for a single cleanup
PR. Item 3 wants its own PR with restructure-test focus. Items
8–10 are Phase-3.

---

## Follow-up scope (Phase 3 candidates)

### 3.A `dataset.py` structural split (action 10)

The bag-download orchestration cluster
(`download_dataset_bag` → `_get_dataset_minid` →
`_create_dataset_minid` → `_create_dataset_bag_client` /
`_download_dataset_minid` / `_materialize_dataset_bag`) is a
self-contained subsystem with one external entry point
(`download_dataset_bag`) and one external dependency
(`DatasetBagBuilder`). After action 1 it's ~600 LoC instead of
1 200 — small enough to live next to `bag_builder.py` as
`bag_pipeline.py` or `download.py`.

### 3.B `dataset_bag.py` structural split (action 10)

The restructure / asset-reachability cluster
(`_build_dataset_type_path_map`, `_get_asset_dataset_mapping`,
`_get_reachable_assets`, `_load_feature_values_cache`,
`_resolve_grouping_value`, `_detect_asset_table`,
`_validate_dataset_types`, `restructure_assets` itself) is
~1 050 LoC. Currently it's pinned to `DatasetBag` because it
calls `self.list_dataset_members`, `self._catalog`,
`self.engine` — but those are all available on a
`DatasetLike` (the protocol the bag implements). Lift to
`restructure.py` taking a `DatasetLike`.

### 3.C `Dataset` / `DatasetBag` interface harmonization

The two classes claim to implement `DatasetLike`, but the
member-walking, parent-walking, and child-walking methods have
diverged enough that the protocol is more aspirational than
load-bearing. Phase 1 §4.7 flagged a similar terminology drift
between `_DEFAULT_SKIP_TABLES` and `terminal_tables`; the same
applies here. A real harmonization probably means making the
protocol enforce method-signature parity (kwargs, return shapes)
in a CI check, not just docstring agreement.

### 3.D `Dataset.estimate_bag_size` performance vs. bag-pipeline alignment

The async-query loop in `estimate_bag_size` (action 8) is the only
piece of `dataset/` that bypasses `CatalogBagBuilder` for
performance. Decision point: lift the optimisation upstream to
`CatalogBagBuilder` (giving every consumer free parallelism), or
formalise `estimate_bag_size` as an opt-out path. The current
"opportunistic copy-paste" is the worst of both worlds.

**Update (Phase 3 follow-up):** decision recorded in
[ADR-0008](../adr/0008-estimate-bag-size-bypasses-bag-pipeline.md):
the bypass is **formalised as a deliberate opt-out**, not lifted
upstream. The estimator continues to share the *walker* via
`DatasetBagBuilder.aggregate_queries` and bypasses
`CatalogBagBuilder` only for *execution* (the async parallel
queries). The docstring on `Dataset.estimate_bag_size` now points
at the ADR. The "lift upstream" alternative remains open as future
work; ADR-0008 documents what it would entail and why we haven't
done it yet.

### 3.E `BagCacheIndex` semantics for multi-anchor bags

The cutover doc (D1) noted that pre-cutover MINIDs may become
unresolvable. After action 2, every cached bag is keyed by
checksum and resolvable by any anchor RID. Users who run
`download_dataset_bag(rid=A)` then `download_dataset_bag(rid=B)`
on overlapping content will be storing one bag, not two. Test
the round-trip story explicitly before merging the cutover.

# deriva-ml audit 2026-05 — cross-cutting (Phase 1)

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/`
(~53.6 KLoC across 122 Python files in 8 subsystems) at the tip of
`fix/catalog-manager-state-guards` (HEAD `4442f82`). External-consumer
checks were run against `/Users/carl/GitHub/deriva-py/deriva/` —
ADR-0006's bag pipeline has landed upstream and several deriva-ml
subsystems have been (partially) ported to it. Findings below are
concrete: file:line citation, what's wrong, what to do, estimated
risk, estimated LoC delta. Items not raised here either survived
scrutiny or are minor enough to not warrant a maintainer's attention
right now.

## Executive summary

Overall posture: **healthy core, dragging tail**. The recent bag
migration (ADR-0006) is a load-bearing improvement — `catalog/clone.py`
shrank from ~1900 to 412 lines, the new `clone_via_bag` is clean, and
the `model/` subsystem is now mostly thin shims over `deriva.bag.*`.
But that migration left a layer of half-eviscerated compatibility
surface (six near-empty shim modules, two parallel `DerivaMLDatabase`
names, two `AssetCopyMode` enums, three deprecated parameter buckets
on `create_ml_workspace`) and three places where the new
`CatalogBagBuilder` is reachable but the legacy `CatalogGraph`
implementation is still doing the actual work (`DatasetBagBuilder`,
`Dataset.download_dataset_bag`, `estimate_bag_size`).

Top themes ranked by impact:

1. **`DatasetBagBuilder` is named after ADR-0006 but is a façade
   around `CatalogGraph`.** `src/deriva_ml/dataset/bag_builder.py:109`
   instantiates `CatalogGraph` and every public method delegates to it.
   The bag-pipeline `anchors_for` / `build_policy` helpers exist
   alongside but **no production caller uses them** — every call site
   (`dataset.py:1302, 2495, 2976, 3353`, `mixins/dataset.py:290`)
   reaches for `generate_dataset_download_spec` which still routes
   through the legacy `CatalogGraph._collect_paths`. ADR-0006's
   "dataset bag cutover" is *unfinished*, not just deferred.

2. **Legacy shim layer is wider than necessary** (6 modules under
   `model/`, 1 under `local_db/`, plus duplicate names in `__init__.py`
   lazy loaders). Each shim is ~20–40 lines of "re-export from
   `deriva.bag.x`" plus a docstring promising removal. Only two of
   them have external (test) callers; the other five are entirely
   internal and can collapse to direct imports.

3. **9 modules carry an unused `icecream` import + fallback pattern
   (~6 lines each, ~50 LoC of dead infrastructure).** No `ic(...)` call
   sites exist anywhere in `src/`. The 4 callers that *do* configure
   `ic.configureOutput` write to a stub that ignores its argument.

4. **Logger initialization is split between three idioms** (44 callsites
   of `logging.getLogger`, half `__name__` and half `"deriva_ml"`, plus
   a `LoggerMixin` that's exported but never used). `core/logging_config.py`
   defines a `get_logger()` helper for exactly this consolidation —
   nothing imports it except the configurator itself.

5. **Three Chaise-navbar builders** that solve the same problem
   (`core/base.py:apply_catalog_annotations` 1218–1335,
   `schema/annotations.py:catalog_annotation` 33–212, plus a partial
   third copy in `schema/create_schema.py`'s init path). The vocabulary
   + asset + domain-schema enumeration loops are byte-for-byte
   duplicated.

Worst-offending subsystems for Phase 2 deep audit:

1. **`dataset/`** (10.9 KLoC across 14 files). Two 2000+-line god
   modules (`dataset.py` 3590, `dataset_bag.py` 2392), the
   stubbed-out `DatasetBagBuilder`, plus parallel data-shaping paths
   between `CatalogGraph` and `_prepare_wide_table` that both walk the
   same FK graph differently.

2. **`model/`** (4.3 KLoC). Mostly transitional: `catalog.py` is
   2085 lines and carries two distinct association-table predicates
   (`is_association` vs `_is_association_table`) plus the
   denormalization planner that probably belongs adjacent to
   `local_db/denormalizer.py`. The six shim files in this directory
   are the right candidate for "delete and re-import" cleanup.

3. **`execution/`** (11.3 KLoC, 24 files). Largest subsystem. Three
   distinct module families that should probably be three packages
   (lifecycle: `execution.py`, `runner.py`, `state_machine.py`;
   I/O: `bag_commit.py`, `upload_engine.py`, `rid_lease.py`;
   config: `execution_configuration.py`, `multirun_config.py`,
   `base_config.py`). Deferred for Phase 2 because the surface is
   large and most call sites are in `execution.py` itself —
   intra-subsystem audit, not the cross-cutting kind.

---

## Subsystem inventory

| Subsystem    | Files | LoC    | Posture                                                                                                                                                                                                                |
|--------------|------:|-------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `asset/`     | 6     | 1 000  | **Healthy.** Manifest-first design well-implemented (`asset/manifest.py`). `AssetSpec` / `AssetSpecConfig` clean split. One minor gripe: `asset_record.py` is 102 lines but only used by `AssetFilePath.metadata` setter. |
| `catalog/`   | 5     | 1 755  | **Recently rewritten.** `clone.py` is now a 412-line back-compat wrapper over `clone_via_bag.py` (473 LoC) — the legacy 1900-line implementation is gone. `localize.py` (477 LoC) is independent and predates ADR-0006.   |
| `core/`      | 29    | 11 143 | **The hub.** `base.py` is 1934 LoC (`DerivaML` class + 11 mixins). `mixins/execution.py` 1211, `mixins/annotation.py` 1000, `mixins/dataset.py` 959 — three of these together are the bulk of the `DerivaML` public API.   |
| `dataset/`   | 14    | 10 856 | **Worst-offender candidate.** `dataset.py` (3590) and `dataset_bag.py` (2392) carry parallel implementations of every read operation. `DatasetBagBuilder` (364 LoC) is a façade over `CatalogGraph` (799 LoC).            |
| `execution/` | 24    | 11 260 | **Largest, mostly load-bearing.** Bag-commit path (`bag_commit.py` 744) is new and clean. Lifecycle path (`execution.py` 2519, `runner.py` 705, `state_machine.py` 617) is dense but coherent.                            |
| `experiment/`| 2     | 425    | **Small isolated module.** Only consumer in src is `mixins/execution.py:43`.                                                                                                                                           |
| `local_db/`  | 11    | 4 841  | **Healthy.** `denormalizer.py` (1282 LoC) is the largest concern; the SQLite-helper shim is the only legacy carryover.                                                                                                  |
| `model/`     | 11    | 4 260  | **In transition.** `catalog.py` 2085 + `annotations.py` 1277 + `database.py` 290; six other files are 14–40 line shims over `deriva.bag.*`.                                                                              |
| `schema/`    | 6     | 2 024  | **Two utilities + one large generator.** `validation.py` (617) and `create_schema.py` (607) are independent. `annotations.py` (606) duplicates `core/base.py:apply_catalog_annotations`.                                  |
| `tools/`     | 2     | 887    | `validate_schema_doc.py` — CLI-only doc-validation tool. Not load-bearing.                                                                                                                                              |
| `cli/`       | 2     | 129    | Two thin CLI entry points. No issues.                                                                                                                                                                                  |

Cross-subsystem dependencies worth naming:

- `dataset/dataset.py` imports from `catalog/` (`localize_assets`),
  `core/` (mixins, definitions), `execution/` (typing only),
  `local_db/` (denormalizer), and `model/`. It is the single
  highest-fanout module.
- `model/catalog.py:DerivaModel` is the schema-handle used by every
  mixin via `self.model`. Its `_schema_to_paths` is the FK-graph
  primitive consumed by `dataset/catalog_graph.py:_collect_paths`,
  the denormalization planner in `model/catalog.py` itself, and the
  Chaise-navbar builders in `core/base.py` and `schema/annotations.py`.
- `dataset/bag_cache.py` and `dataset/bag_builder.py` both import
  from `deriva.bag.*` directly; everything else in `dataset/` still
  goes through `dataset/catalog_graph.py`.

---

## Lens 1 — Dead code

### 1.1 `icecream`-debug fallback pattern is dead-code in 9 modules

Nine modules import `icecream` with a no-op fallback (`ic = lambda *a:
...`) and four of them also call `ic.configureOutput(...)` inside the
import-success branch. **No `ic(...)` call sites exist anywhere in
`src/deriva_ml/**`** — verified by `grep -rn ' ic(' src/`. The fallback
silently swallows whatever an `ic(...)` call would have produced.

Affected sites:

- `core/catalog.py:48-51` (the import + fallback)
- `dataset/dataset.py:75-80`
- `dataset/dataset_bag.py:67-72`
- `dataset/catalog_graph.py:31-39`
- `dataset/upload.py:74-79`
- `dataset/aux_classes.py:29-35`
- `execution/execution.py:94-97`
- `model/catalog.py:48-51`
- `schema/create_schema.py:41-44`
- `demo_catalog.py:39-44`

**Fix:** delete the import + fallback + `configureOutput` calls.
Bring back a single project-wide `# breakpoint: import icecream as ic`
in `conftest.py` if developers want it. **Risk: trivial. LoC: −55.**
**Severity: low** (cosmetic; runtime is unaffected).

### 1.2 `LoggerMixin` is exported but never used

`core/logging_config.py:195-213` defines `LoggerMixin` and exports it
from `core/__init__.py:36, 67`. Grep across `src/` shows zero
consumers. The intended consumers (`DatabaseModel`, `Asset`,
`Dataset`, `Execution`, etc.) all do `self._logger =
logging.getLogger("deriva_ml")` inline instead.

**Fix:** delete `LoggerMixin` and its `core/__init__.py` export.
**Risk: low** (no callers in deriva-ml; spot-check the MCP plugins
turned up no external consumers either). **LoC: −20.** **Severity: low.**

### 1.3 `DerivaMLNoExecutionContext` is defined but never raised

`core/exceptions.py:174-193` declares this exception with a
nine-line docstring + example. Grep across `src/` shows it imported
nowhere and raised nowhere. The docstring refers to "`ml.table()`
handles" — that surface doesn't exist (no `DerivaML.table()` method).

**Fix:** delete `DerivaMLNoExecutionContext`. **Risk: low** (zero
callers). **LoC: −20.** **Severity: low.**

### 1.4 `AssetRID` is deprecated, exported, and replaced everywhere

`execution/execution_configuration.py:147-171` defines `AssetRID`
with a "deprecated; use `AssetSpec`" notice. It is exported from
`execution/__init__.py:28, 67`. The only `isinstance(v, AssetRID)`
check is inside `ExecutionConfiguration.validate_assets` itself
(`execution_configuration.py:111`), which converts it to `AssetSpec`.
No production caller constructs an `AssetRID`.

**Fix:** delete `AssetRID` and the `isinstance(v, AssetRID)` branch
of `validate_assets`. **Risk: low-medium** (external callers passing
a serialized `AssetRID` would still be handled by the `dict` or
`str` branches). **LoC: −30.** **Severity: low.**

### 1.5 `DerivaML.working_data` deprecated property

`core/base.py:865-893` — emits a `DeprecationWarning` and returns
`self.workspace`. No internal callers. The companion property
`workspace` has been the canonical name for several releases now.

**Fix:** delete the property. **Risk: low** (`DeprecationWarning`
shouldn't be load-bearing for any caller). **LoC: −30.** **Severity: low.**

### 1.6 Six shim modules under `model/` have only test consumers (and not always)

Each is a 14–40 line module that re-exports from `deriva.bag.*`:

| Shim file                                  | LoC | External (non-shim) consumers          |
|--------------------------------------------|-----|-----------------------------------------|
| `model/schema_builder.py`                  | 39  | `local_db/schema.py:26` (in `src/`)    |
| `model/data_loader.py`                     | 24  | tests only (no `src/` callers)         |
| `model/data_sources.py`                    | 33  | tests only                              |
| `model/fk_orderer.py`                      | 21  | tests only                              |
| `model/handles.py`                         | 14  | none — its symbols are re-exported     |
| `model/deriva_ml_database.py`              | 30  | tests only                              |
| `local_db/sqlite_helpers.py`               | 45  | tests only                              |

`model/__init__.py:54-57, 60` then imports from these shims and
re-exports the same symbols a second time. The `DatabaseModel` lazy-load
in `model/__init__.py:118-126` exposes `DerivaMLDatabase` as an alias
for `DerivaMLBagView`, *and* `model/deriva_ml_database.py:28` re-binds
the same alias. Two parallel "back-compat" surfaces for the same name.

**Fix (staged):**

1. **Now** — update `local_db/schema.py:26` to `from deriva.bag.schema
   import SchemaBuilder, SchemaORM` directly, then delete
   `model/schema_builder.py` and remove the `model/__init__.py`
   import.
2. **Now** — update the four test files that still import via
   `model/data_loader.py`, `model/data_sources.py`,
   `model/fk_orderer.py`, `model/handles.py` to import from
   `deriva.bag.*` directly; delete those four shims and the
   corresponding `model/__init__.py` entries.
3. **One release cycle** — collapse `model/deriva_ml_database.py`'s
   alias into `model/__init__.py` (single lazy-import path); drop
   `local_db/sqlite_helpers.py` once the two test imports flip.

**Risk: low-medium** (per-shim; the deletion is mechanical once test
imports are updated). **LoC: −200 across all seven shims plus their
`__init__` plumbing.** **Severity: medium** — the shim layer obscures
the actual deriva-py boundary and adds two re-export paths for every
symbol it touches.

### 1.7 Legacy aliases on `DatabaseModel`

`model/database.py:265-287` defines three back-compat methods:

- `_get_table_contents(self, table)` — generator that delegates to
  the inherited `get_table_contents`. Called from 5 sites in
  `dataset_bag.py` and `deriva_ml_bag_view.py`.
- `get_orm_association_class(...)` — alias for inherited
  `get_association_class`. Called from `dataset_bag.py:509` only.
- `delete_database()` — alias for `dispose()` with a docstring noting
  the "delete" name was misleading. Zero callers.

**Findings:**

- `delete_database()` (`database.py:279-287`) has no callers in
  `src/`. Delete. **LoC: −10.**
- `_get_table_contents` is consumed by 5 sites outside the
  defining class — defeats the underscore convention. Either rename
  the callsites to `get_table_contents` (already public on
  `BagDatabase`) or rename the alias to drop the underscore.
  **LoC: −5 (delete alias, update 5 callsites).**
- `get_orm_association_class` is a one-call alias. Update the call
  in `dataset_bag.py:509` and delete the alias. **LoC: −5.**

**Risk: low** (mechanical). **Severity: low** — but together with §1.6
this is a noticeable accumulation of "back-compat I can never delete."

### 1.8 `AssetCopyMode` enum-style class is a 25-line wrapper for three string constants

`catalog/clone.py:92-117` defines a class that "exists for back-compat"
and isn't allowed to be instantiated. Its members are aliases for
`AssetMode.ROWS_ONLY` / `AssetMode.UPLOAD_IF_MISSING`, used as the
default for `create_ml_workspace`'s `asset_mode` parameter
(`clone.py:153`). `_coerce_asset_mode` (`clone.py:324-346`) accepts
both the legacy and the new spellings.

`AssetCopyMode` is exported (`catalog/__init__.py:28, 61`,
`clone.py:404`) but no internal caller references it after the
parameter coercion. The simplification is: **make `create_ml_workspace`'s
`asset_mode` default `AssetMode.ROWS_ONLY` directly, delete
`AssetCopyMode`, delete `_coerce_asset_mode`**, keep only string
coercion at the boundary. **Risk: low-medium** (one
DeprecationWarning per stray external caller using `.REFERENCES` or
`.FULL` — they get caught by the string branch). **LoC: −60.**
**Severity: low.**

### 1.9 `_LEGACY_ONLY_PARAMS` set defined but never iterated

`catalog/clone.py:125-137` defines `_LEGACY_ONLY_PARAMS` (11 names),
but the runtime check that warns about non-default values lives in
`_warn_about_legacy_params` (`clone.py:366-400`) and has its own
`defaults = {...}` dict of 11 names. Two parallel declarations of
the same eleven-element set.

**Fix:** delete `_LEGACY_ONLY_PARAMS` (`clone.py:125-137`) or use it
inside `_warn_about_legacy_params`. **LoC: −15.** **Severity: low.**

### 1.10 Outdated comment / docstring drift

- `catalog/localize.py:413` — "datapath doesn't have an 'in' operator"
  — incorrect since deriva-py PR #242 added `_ColumnWrapper.in_`
  (`deriva-py/deriva/core/datapath.py:1269`). The `_fetch_asset_records`
  fallback builds an OR chain (`localize.py:417-423`) when one
  `.in_(...)` call would do.
- `catalog/clone_via_bag.py:178` — raw URL `/entity/deriva-ml:Dataset_Dataset/Dataset=any(...)`
  built by string concatenation in `_collect_nested_dataset_rids`.
  Same anti-pattern as bag-audit §2.1. See Lens 2.
- `model/database.py:281-286` — `delete_database` docstring claims
  back-compat with pre-migration callers, but those callers no longer
  exist (see §1.7).

**Severity: low. LoC: ±0 (doc-only) for the comment; see Lens 2 for
the URL.**

---

## Lens 2 — deriva-py interface usage

### 2.1 `clone_via_bag._collect_nested_dataset_rids` rolls its own ERMrest URL

`catalog/clone_via_bag.py:161-201` issues a BFS over `Dataset_Dataset`
by constructing the URL with `urlquote` + f-string and calling
`source_catalog.get(path)`. The body:

```python
rid_list = ",".join(urlquote(r) for r in sorted(frontier))
path = f"/entity/deriva-ml:Dataset_Dataset/Dataset=any({rid_list})"
response = source_catalog.get(path)
```

The datapath equivalent (verified in
`deriva-py/deriva/core/datapath.py:1269-1299` — `_ColumnWrapper.in_`):

```python
pb = source_catalog.getPathBuilder()
dd = pb.schemas["deriva-ml"].tables["Dataset_Dataset"]
rows = dd.filter(dd.Dataset.in_(sorted(frontier))).entities().fetch()
```

This is the same fix the bag-package audit recommended for
`_validate_anchors` (`bag-package-audit-2026-05.md` §2.1). **Risk: low-medium**
(needs a live-catalog smoke test; the unit path is straightforward).
**LoC: −10.** **Severity: medium** — every reimplemented raw URL is
a place where a hyphen-in-RID or a new ERMrest filter syntax will
silently break.

### 2.2 `catalog/localize.py:_fetch_asset_records` builds an OR chain instead of using `.in_()`

`catalog/localize.py:413-428` constructs `filter_expr = (col == r1) |
(col == r2) | ...` because the comment at line 413 claims "datapath
doesn't have an 'in' operator." The fallback handles
`DataPathException` by issuing one fetch per RID — fine for the
single-RID fallback, but the primary path should use `.in_()`. Same
fix as §2.1.

**Risk: low. LoC: −8.** **Severity: low.**

### 2.3 `execution/rid_lease.py:_validate_pending_asset_leases` builds an ERMrest filter clause by hand

`execution/rid_lease.py:124-130` does:

```python
filter_clause = ";".join(f"RID={rid}" for rid in chunk)
path = f"/entity/public:ERMrest_RID_Lease/{filter_clause}"
response = catalog.get(path)
```

The `;` separator means OR in ERMrest filter syntax — exactly what
`column.in_(values)` produces. The function chunks by
`PENDING_ROWS_LEASE_CHUNK` (defined in
`execution/lease_orchestrator.py:211` which also builds the same
filter clause). Two callers of the same idiom.

**Fix:** consolidate into a single helper `lease_rids_present(catalog,
rids) -> set[str]` using `pb.schemas["public"].tables["ERMrest_RID_Lease"]`
and `.in_(rids)`. **Risk: low-medium** (live-catalog test needed). **LoC: −15.**
**Severity: low.**

### 2.4 `execution/state_machine.py:reconcile_one_execution` uses raw ERMrest GET

`execution/state_machine.py:488-491`:

```python
response = catalog.get(
    f"/entity/deriva-ml:Execution/RID={execution_rid}"
)
rows = response.json()
```

Standard "fetch a single row by RID" pattern. The deriva-py path is
`pb.schemas["deriva-ml"].tables["Execution"].filter(t.RID ==
execution_rid).entities().fetch()` — same wire request, but the URL
construction lives in deriva-py where escaping and snapshot handling
already exist. The neighboring `create_catalog_execution`
(`state_machine.py:611`) does a raw `catalog.post("/entity/deriva-ml:Execution",
json=body)` — also fine, but inconsistent with the rest of the
codebase's preference for the datapath API per the
deriva-ml `CLAUDE.md` "API priority" note.

**Recommendation:** track as a follow-up cleanup. Both callsites work;
the bigger issue is that `reconcile_one_execution` is the only place
in the state machine that needs to query an arbitrary RID, and the
datapath path-builder is already constructed for `update_catalog`
calls in the same file. **LoC: −5.** **Severity: low.**

### 2.5 `dataset/catalog_graph.py:source_path` parses a datapath URI back into a path string

`dataset/catalog_graph.py:651-663` builds a datapath, then peels the
ERMrest URI back apart by hand:

```python
def source_path(path: tuple[Table, ...]) -> str:
    dp, _ = self._build_linked_datapath(path, pb)
    uri = dp.uri
    entity_prefix = "/entity/"
    idx = uri.index(entity_prefix)
    entity_path = uri[idx + len(entity_prefix):]
    parts = entity_path.split("/", 1)
    if len(parts) == 1:
        return f"{parts[0]}/RID={{RID}}"
    return f"{parts[0]}/RID={{RID}}/{parts[1]}"
```

This is the conversion layer between datapath and the export engine's
expected query path. The "parse a URI back into the path" step is
fragile (it would break if datapath ever emitted ``?snaptime=...`` or
a different query-string format). deriva-py exposes a private
`_path` attribute on the datapath that's the string form before URI
assembly; using it would be slightly cleaner.

**Risk: low** (current code works). **LoC: ±0.** **Severity: low** —
flag for Phase 2 review of `CatalogGraph` (which is itself slated
for replacement by `CatalogBagBuilder`).

### 2.6 Asset upload spec uses raw URL templates instead of pathBuilder

`dataset/upload.py:277` defines `record_query_template:
"/entity/{target_table}/MD5={md5}&Filename={file_name}"` and
`hatrac_options` / `hatrac_templates` carry literal path-template
strings. This is fine — the upload spec is consumed by deriva-py's
`GenericUploader`, which expects raw URL templates. No fix needed,
but worth a comment explaining that these strings are spec-DSL not
URL construction, since `localize.py` and `clone_via_bag.py` (the
files that *should* use pathBuilder) have similar-looking idioms
that are wrong.

**Recommendation:** add a one-line comment at `dataset/upload.py:277`
distinguishing the upload-spec template from runtime URL construction.
**LoC: +2.** **Severity: low.**

### 2.7 `DerivaModel` carries two association-table predicates

`model/catalog.py:492-513` exposes `is_association(table, ...)` —
delegates to deriva-py `Table.is_association(...)` (verified in
`deriva-py/deriva/core/ermrest_model.py:2206`). Default arity is
2..2, default pure-ness is `True`.

`model/catalog.py:718-772` defines `_is_association_table` — a custom
predicate that ignores `pure`, filters out `ERMrest_Client` /
`ERMrest_Group` FKs, and returns True if **exactly** two domain FKs
remain. The docstring at 738-742 explains it's "stricter in one
direction, looser in another" than the deriva-py method.

There's also `deriva.bag.schema.SchemaBuilder.is_association_table`
upstream (`deriva-py/deriva/bag/_orm_helpers.py:33`,
`deriva-py/deriva/bag/schema.py:291`) — a *third* predicate, with
yet a different arity-rule semantic, intended for ORM building.

Three predicates with overlapping but non-identical semantics:

| Predicate | Counts system FKs? | Requires pure? | Used by                                                |
|---|---|---|---|
| `Table.is_association` (deriva-py) | yes | configurable (default yes) | `model/catalog.py:is_association` (8 callsites) |
| `DerivaModel._is_association_table` | no | no (topology only) | `model/catalog.py` denormalization planner (5 callsites) |
| `SchemaBuilder.is_association_table` (deriva-py bag) | n/a (SQL-level) | n/a | `BagDatabase` ORM auto-build (deriva-ml inherits it) |

The denormalization planner needs the "topology not purity" variant
because real Deriva linkage tables routinely carry annotation
columns (`Role`, `Ordinal`, `Comment`) without losing their M:N
identity. That's a legitimate need.

**Recommendation:** rename `_is_association_table` to
`is_topological_association` and document it as a public method that
exists alongside `is_association` (the deriva-py default). Phase 2
deep-audit should decide whether the denormalization planner's
predicate can be folded into deriva-py's `is_association(pure=False,
return_fkeys=True)` — likely yes, with a custom filter on the
returned fkey set.

**Risk: low** (rename + doc only). **LoC: +10.** **Severity: low.**

### 2.8 `DerivaModel.is_dataset_rid` uses `catalog.resolve_rid` correctly

`model/catalog.py:678-690` calls `self.model.catalog.resolve_rid(rid,
self.model)` — the canonical deriva-py API
(`deriva-py/deriva/core/ermrest_catalog.py`). This is exactly the
right pattern. Mentioning it because the rest of this lens has been
"watch out for raw URL builders" — this is the counterexample of how
the integration *should* look.

### 2.9 `catalog/localize.py:_extract_hatrac_path` rolls a string parser

`catalog/localize.py:442-462` uses `urlparse` + `path.find("/hatrac/")`
to extract the hatrac path from a URL. Self-contained, no dep on
deriva-py beyond `urlparse`. Could use
`deriva.core.hatrac_store.HatracStore` helpers but those are
geared toward GET/PUT, not URL parsing. **No action.**

---

## Lens 4 — Inconsistencies / duplication added by incremental change

### 4.1 Three Chaise-navbar builders

The exact same navbar tree (User Info / Deriva-ML / WWW / domain
schemas / Vocabulary / Assets / Features / Catalog Registry /
Documentation) is built in:

- **`core/base.py:1218-1334`** — `DerivaML.apply_catalog_annotations()`,
  117 lines, instance method.
- **`schema/annotations.py:33-212`** — `catalog_annotation(model:
  DerivaModel)`, ~180 lines, module function.
- **`schema/create_schema.py`** (partial copy in the schema-init path)
  — referenced from `schema/__init__.py`.

Each implementation:

1. Iterates `domain_schemas` in sorted order to build per-schema menus.
2. Iterates both ML and domain schemas to enumerate vocabularies.
3. Iterates both ML and domain schemas to enumerate asset tables.
4. Skips associations and vocabularies in the domain menu.

The filter predicates at `core/base.py:1162`,
`schema/annotations.py:135`, and `core/base.py:1177, 1191` use
`is_association(tname, pure=False, max_arity=3)` — identical kwargs.

**Recommendation:** lift the navbar builder to a single module
function (`schema/annotations.py:build_catalog_navbar(model) -> dict`)
and have `DerivaML.apply_catalog_annotations` and `create_ml_schema`
both call it. The wrapping `apply()` step stays where it is.
**Risk: medium** (touches three callsites with subtly different
state). **LoC: −150 to −180.** **Severity: high** — three copies of
the same logic means three places to update when the navbar layout
changes; it has already drifted (the WWW menu in
`schema/annotations.py:111` lists the same two entries as
`base.py:1281`, but the icon rendering rules differ).

### 4.2 `CatalogGraph` and `DatasetBagBuilder` are parallel implementations

`dataset/catalog_graph.py:CatalogGraph` (799 LoC, production) and
`dataset/bag_builder.py:DatasetBagBuilder` (364 LoC, also production
but slated to replace `CatalogGraph` per ADR-0006).

The current state, as of HEAD:

- `DatasetBagBuilder.__init__` (`bag_builder.py:83-114`) **instantiates a
  `CatalogGraph`** and stores it as `self._catalog_graph`.
- `generate_dataset_download_spec`, `generate_dataset_download_annotations`,
  `aggregate_queries` all delegate to `self._catalog_graph.<same
  name>`.
- The bag-pipeline-shaped helpers (`anchors_for`, `build_policy`,
  `_exclude_empty_associations`, `_iter_descendant_rids`) — 165
  lines — exist on `DatasetBagBuilder` but **no production caller
  invokes them**. The comment at `bag_builder.py:99-108` explicitly
  flags this as a deferred cutover.

The five production callers (`dataset.py:1302`, `2495`, `2976`,
`3353`, `mixins/dataset.py:290`) all reach for
`generate_dataset_download_spec` / `generate_dataset_download_annotations`,
so the bag-pipeline path is unreachable from anywhere except tests.

**Severity: high.** ADR-0006 declared this cutover; it isn't done.
The intermediate state — bag-pipeline helpers present and untested
against production loads, legacy `CatalogGraph` still doing the
actual work — is the worst of both worlds.

**Recommended action:** schedule the cutover as a discrete piece of
work. Until it lands, the parallel surface is a maintenance hazard
because any change to `CatalogGraph._collect_paths` semantics has to
be back-ported to `_exclude_empty_associations` and vice versa.

### 4.3 Two duplicate cache-status name maps

`dataset/bag_cache.py:CacheStatus` defines both `cached_holey` and
`cached_incomplete` as enum members that share the same string value
(`"cached_holey"`). The class comment at `bag_cache.py:72-76` explains
this is back-compat for the old name. But the consuming docstrings in
the function that returns the dict use the **old name** —
`mixins/dataset.py:387` and `dataset/dataset.py:2682` both say
``"cache_status": one of "not_cached", "cached_metadata_only",
"cached_materialized", "cached_incomplete"``.

Two consequences:

- The docstring is wrong about the *current* return value
  (`bag_cache.py:213` and `231` return `CacheStatus.cached_holey`,
  whose `.value` is `"cached_holey"`, not `"cached_incomplete"`).
- The aliasing in the enum means `CacheStatus("cached_holey") is
  CacheStatus.cached_incomplete` — which works for `==` comparisons
  but is surprising for `is` comparisons.

**Fix:** update the two docstrings to say `"cached_holey"` instead
of `"cached_incomplete"`, and consider removing the alias entry
after a release cycle. **Risk: zero (doc-only). LoC: +2 / −0** for
the doc fix; **−5** if the alias is removed. **Severity: low.**

### 4.4 Two `validate_call` configuration patterns

`core/validation.py:49-58` defines `VALIDATION_CONFIG` and
`STRICT_VALIDATION_CONFIG` exactly so callers can write
`@validate_call(config=VALIDATION_CONFIG)`. The module's docstring at
line 14 explicitly demonstrates this pattern.

In practice, **44 callsites across `src/`** write
`@validate_call(config=ConfigDict(arbitrary_types_allowed=True))`
inline. **Zero callsites** import `VALIDATION_CONFIG`.

Sample sites (confirmed by grep):

- `core/mixins/annotation.py:79`
- `core/mixins/dataset.py:230`
- `core/mixins/feature.py:154` (and 6 others in the same file)
- `model/catalog.py:569`
- `asset/asset.py:116` (in `__init__`)
- `dataset/dataset.py:186, 580` (and many others)
- `execution/execution.py:165, 322` (etc.)

**Fix:** project-wide search-and-replace of the inline `ConfigDict(...)`
with the `VALIDATION_CONFIG` import. **Risk: low** (semantically
identical). **LoC: roughly neutral but consolidates the configuration
point.** **Severity: medium** — if the `ConfigDict` defaults ever need
to change (e.g., to forbid extra fields globally), the current state
requires 44 edits.

### 4.5 Three `logging.getLogger` initialization patterns

Documented in §1.2 (LoggerMixin) and the executive summary. Summary
of patterns in use:

| Pattern | Sample callsite | Count |
|---|---|---|
| `logger = logging.getLogger(__name__)` | `model/catalog.py:120` | ~22 |
| `logger = logging.getLogger("deriva_ml")` | `validation.py:40`, `bag_cache.py:44` | ~12 |
| `self._logger = logging.getLogger("deriva_ml")` (in `__init__`) | `model/database.py:94`, `asset/asset.py:116` | ~5 |
| Inline `logging.getLogger("deriva_ml").warning(...)` | `core/base.py:377, 452, 589, 636, 832, 839` | ~6 |

The mix means messages from the same conceptual subsystem can hit
the root logger under different names depending on which path was
taken. The `get_logger()` helper in `core/logging_config.py:81-95`
exists explicitly to consolidate; nothing uses it.

**Recommendation:** project-wide adoption of `from
deriva_ml.core.logging_config import get_logger; logger =
get_logger(__name__)`. Risk: trivial (logger hierarchies are
forgiving). LoC: roughly neutral. **Severity: medium** — log
configuration is the kind of cross-cutting concern that benefits
most from one canonical entry point.

### 4.6 Two `find_associations()` filter idioms for "elements of a dataset"

The "list of tables that can be a dataset element type" computation
appears three times with slightly different forms:

- `model/catalog.py:692-716` — `list_dataset_element_types()` on
  `DerivaModel`: filters by `is_domain_or_dataset_table(t)` (domain
  schema, or `Dataset` itself).
- `core/mixins/dataset.py:203-228` — `list_dataset_element_types()`
  on `DatasetMixin`: same filter, almost word-for-word identical.
  The mixin version delegates the filter logic and then re-applies
  it. The DRY thing is to call `self.model.list_dataset_element_types()`
  — which is *almost* what the mixin does (it duplicates the loop).
- `dataset/catalog_graph.py:450` — `dataset_association_tables = [a.table
  for a in self._dataset_table.find_associations()]` — a *different*
  filter (every association, no schema check).

The third site is doing something genuinely different — it wants
`Dataset_X` associations regardless of which schema X lives in. The
first two are duplicate.

**Fix:** delete the mixin's local implementation
(`core/mixins/dataset.py:203-228`) and have it call
`self.model.list_dataset_element_types()`. **Risk: low** (the methods
are byte-equivalent). **LoC: −20.** **Severity: low.**

### 4.7 `_DEFAULT_SKIP_TABLES` and walker terminal-table sets diverge

`model/catalog.py:1895` defines `_DEFAULT_SKIP_TABLES =
frozenset({"Dataset_Dataset", "Execution"})` for the
`_schema_to_paths` traversal.

`catalog/clone_via_bag.py:349-352` defines
`default_terminal_tables = {("deriva-ml", "Execution"), ("deriva-ml",
"Workflow")}` for the bag-pipeline traversal.

`dataset/catalog_graph.py` doesn't carry an explicit list — it
relies on `_schema_to_paths`'s default plus a separate filter to
skip vocab-ending paths (`catalog_graph.py:497`).

So three traversers, three different "what tables are special"
policies:

- `_schema_to_paths` skips `Dataset_Dataset` + `Execution`.
- `clone_via_bag` treats `Execution` + `Workflow` as terminal.
- `CatalogGraph` runs a separate vocab-export pass and prunes
  paths ending at vocab tables.

Each policy is defensible in isolation but the divergence means
the same FK graph produces three different "this table is special"
classifications. ADR-0006's `FKTraversalPolicy.terminal_tables`
field is the right place to centralize this for the bag pipeline;
`_schema_to_paths` predates the policy concept.

**Recommendation:** Phase 2 — survey the three policies, decide
whether they can be unified through `FKTraversalPolicy` once
`DatasetBagBuilder` finishes its cutover. **Risk: medium** (these
policies have catalog-shape semantics; merging them changes the
bag walk). **LoC: ±0 in Phase 1.** **Severity: medium.**

---

## Lens 5 — Simplification opportunities

### 5.1 Collapse the `model/` shim layer

See §1.6. Net: -200 LoC by deleting the seven shims (`schema_builder.py`,
`data_loader.py`, `data_sources.py`, `fk_orderer.py`, `handles.py`,
`deriva_ml_database.py`, `local_db/sqlite_helpers.py`) after
flipping 5 internal-source imports and 5 test-file imports.
**Risk: low (mechanical).** **Severity: medium.**

### 5.2 `DatasetBagBuilder` should either become `CatalogGraph`'s replacement or be deleted

Currently 364 LoC for a class that:

- Constructs a `CatalogGraph` (`bag_builder.py:109`).
- Delegates three methods to it (`bag_builder.py:120-192`).
- Defines `anchors_for`, `build_policy`,
  `_exclude_empty_associations`, `_iter_descendant_rids` (lines
  198-361) that nothing in `src/` invokes.

Two paths forward:

(a) **Finish the cutover.** Rewire the five callers in
   `dataset/dataset.py` and `core/mixins/dataset.py` to use the
   bag-pipeline helpers — this is the work ADR-0006 commits to.
   `CatalogGraph` becomes deletable once equivalence is verified.
   ~−800 LoC net (lose `CatalogGraph`'s 799, keep
   `DatasetBagBuilder`'s 165 useful lines, lose 200 of delegation
   façade). **Risk: high** (live-catalog equivalence tests required).

(b) **Delete the bag-pipeline-shaped helpers** (`anchors_for`,
   `build_policy`, etc.) until the cutover is actually scheduled.
   Reduces `DatasetBagBuilder` to a 200-line delegation wrapper
   that still works for production callers. **Risk: low.** **LoC:
   −165.**

The current intermediate state — 165 lines of unreachable code
that purport to implement ADR-0006 — is strictly worse than either
option. **Recommended:** option (b) as Phase 1, with the cutover
tracked separately. **Severity: high** (active hazard).

### 5.3 Unify the three Chaise-navbar implementations

See §4.1. **Risk: medium. LoC: −150 to −180.** **Severity: high.**

### 5.4 `_warn_about_legacy_params` could use a Pydantic deprecation hook

`catalog/clone.py:366-400` walks 11 named kwargs and emits a
`DeprecationWarning` per non-default value. The function is called
exactly once. The simpler form is a 3-line helper or a single
`for name, default in DEFAULTS.items()` loop reusing the function's
existing `defaults` dict. Currently both `_LEGACY_ONLY_PARAMS`
(line 125) and `defaults` (line 375) declare the same 11 names.

**Fix:** delete `_LEGACY_ONLY_PARAMS` and have
`_warn_about_legacy_params` iterate `defaults` directly. **LoC: −15.**
**Severity: low.**

### 5.5 `_get_table_contents` alias proliferation

`model/database.py:265-273` and 5 callsites in `model/deriva_ml_bag_view.py`
+ 1 in `dataset/dataset_bag.py`. The `_` prefix is misleading —
this method is part of the public read-side surface. Either rename
to `get_table_contents` and inherit cleanly from `BagDatabase` (which
already exposes it at `bag/database.py:692`), or document the
override. **Risk: low (mechanical rename). LoC: −10.** **Severity: low.**

### 5.6 `DatabaseModel.delete_database` is a one-line alias with no callers

See §1.7. **Risk: low. LoC: −10.** **Severity: low.**

### 5.7 Asset-localization `or "..."` pattern is uniform but fragile

`catalog/localize.py:178, 254-255` each pick a column by trying both
`URL` and `url`, `Filename` and `filename`, `MD5` and `md5`:

```python
current_url = record.get("URL") or record.get("url")
filename = record.get("Filename") or record.get("filename")
md5 = record.get("MD5") or record.get("md5")
```

This works because empty strings are falsy and lowercase columns
hold non-empty filenames. But it shadows the case where `URL` is
intentionally `None` (a placeholder asset row). A cleaner version
calls `_detect_url_column` once (the function already exists at
line 163) and uses the detected name for all three.

**Risk: low. LoC: −5.** **Severity: low.**

### 5.8 Schema cache + `_apply_logger_overrides` is the only consumer of `DEFAULT_LOGGER_OVERRIDES`

`core/base.py:347` calls `_apply_logger_overrides(DEFAULT_LOGGER_OVERRIDES)`
once at `__init__` time. The function lives in `core/logging_config.py:181-192`,
marked as internal ("not part of the public API"). Either fold it into
`configure_logging` or make it a top-level helper. **Risk: trivial. LoC: ±0.**
**Severity: low.**

---

## Lens 6 — Maintainability

### 6.1 `model/catalog.py` is a 2085-line god class with two distinct responsibilities

`DerivaModel` (`model/catalog.py:135-2085`) carries:

1. **Schema-introspection surface** (lines 260-823): `is_system_schema`,
   `is_domain_schema`, `is_vocabulary`, `is_asset`, `is_association`,
   `find_features`, `lookup_feature`, `asset_metadata`,
   `_is_association_table`, `_fk_neighbors`,
   `_downstream_fk_sources`, `_outbound_reachable`, `_find_sinks`.
   ~565 lines. Used by every mixin and most subsystems.
2. **Denormalization planner** (lines 825-1616): `_build_join_tree`,
   `_determine_row_per`, `_enumerate_paths`, `_find_path_ambiguities`,
   `_prepare_wide_table`. ~790 lines. Used only by
   `local_db/denormalizer.py` and `mixins/dataset.py:457`.
3. **Schema mutation** (lines 1618-2085): `_schema_to_paths`,
   `create_table`, `_define_association`. ~470 lines. Used by all
   subsystems but for different reasons (`_schema_to_paths` is FK
   traversal; `create_table` is mutation).

This works but the file is the second-largest in the codebase and
mixes three distinct roles. The denormalization-planner methods all
share an `_` prefix while the public-API methods don't — confusingly,
`_prepare_wide_table` is *the* entry point used by external callers
(`mixins/dataset.py:457`).

**Recommendation:** Phase 2 — split `DerivaModel` into a base
class + denormalization-planner mixin, or extract the planner into
`local_db/denormalize_planner.py` (where it would sit next to the
executor). **Risk: medium-high** (a popular class; lots of `self.`
references to chase). **LoC: ±0 net, but ~800 lines move.**
**Severity: medium.**

### 6.2 `dataset/dataset.py` is 3590 LoC

The largest single file in the codebase. `Dataset` class is at line
126; the file continues to line 3590 with method after method on the
same class. A handful of standalone helpers (`_hash_spec` at line
107) live inline.

By line count, the natural splits:

- Lines 126–1500 or so: read-side operations
  (`list_dataset_members`, `list_dataset_parents`, `dataset_history`,
  `current_version`, etc.).
- Lines 1500–2800: bag-export pipeline (`download_dataset_bag`,
  `estimate_bag_size`, `_create_dataset_minid`,
  `_create_dataset_bag_client`).
- Lines 2800–3590: write-side operations + version management.

**Recommendation:** Phase 2 — survey the cross-method dependencies
and decide whether to split. Not blocking. **Severity: medium.**

### 6.3 Docstring quality is **uneven**

Sampled sites with notable defects:

- **`model/catalog.py:492-513`** — `is_association`'s docstring has
  empty "Args:" and "Returns:" sections (placeholders never filled
  in) and a malformed parameter list. Compare with the well-formed
  `_is_association_table` at line 718-772, which is a model of how
  this should read.
- **`model/catalog.py:617-635`** — `lookup_feature` docstring has
  the same placeholder shape.
- **`schema/annotations.py:33-44`** — `catalog_annotation` has a
  3-line docstring; the analogous `apply_catalog_annotations` in
  `core/base.py:1098-1142` is 50 lines and detailed. Same operation,
  vastly different documentation.

**Severity: medium.** The docstring standard in deriva-ml's
`CLAUDE.md` is Google-style with `Args:` / `Returns:` / `Raises:` /
`Example:`. Several methods drop the `Example:` block (excused for
catalog-touching ones) but the `Args:` placeholders are bugs.

### 6.4 The `Workflow` module's IPython / Jupyter import wrappers

`execution/workflow.py:34-67` carries three nested try/except blocks
that each import an optional dep and provide a fallback stub. The
fallback for `list_running_servers` (`workflow.py:49-50`) defines
the stub *twice* — once inside the `except` branch (line 48-50) and
once at module level after the branch (line 52-53). The second
definition is dead.

```python
except ImportError:
    def list_running_servers():
        return []

    def get_servers() -> list[Any]:
        return list_running_servers()
```

**Fix:** delete the duplicate inner definition. **Risk: zero. LoC: −2.**
**Severity: low.**

### 6.5 Naming inconsistency: `bag_path`, `bag_dir`, `output_dir`, `cache_path`

| Concept | Names used | Inconsistency |
|---|---|---|
| Cached bag directory | `bag_path` (`bag_cache.py:181`), `cache_path` (`bag_cache.py:158`), `bag_dir` (`bag_cache.py:166`) | Three names in the same module |
| Output destination | `output_dir` (`clone_via_bag.py:286-287`), `bag_path` (`clone_via_bag.py:96-114`) | Same `Path` value, two names depending on caller |
| Local SQLite | `dbase_path` (`database.py:91`), `database_dir` (`database.py:118`), `cache_dir` (`bag_cache.py:101`) | Three names for related paths |

This isn't load-bearing but it makes the storage-layout discussion
ambiguous. **Severity: low.**

### 6.6 `__all__` consistency in shim modules

The seven shim modules (§1.6) each define an `__all__`, but
`model/__init__.py:62-109` then re-exports a much larger set including
symbols (e.g. `DatabaseModel`, `DerivaMLBagView`) that come from
different submodules. The `model/__init__.py:64-67`'s `__all__`
contains `"DerivaMLDatabase"` — a legacy alias resolved by
`__getattr__` — but the actual class binding happens in
`model/deriva_ml_database.py:28` not in `__init__`. New maintainers
reading `__all__` first won't see where the alias is created.

**Recommendation:** consolidate the lazy-import in `model/__init__.py`
and delete the orphan binding in `model/deriva_ml_database.py:28`.
**LoC: −5.** **Severity: low.**

### 6.7 Re-exports through `core/definitions.py`

`core/definitions.py` is a 181-line re-export hub for symbols from
`constants.py`, `enums.py`, `ermrest.py`, `exceptions.py`,
`filespec.py`. The module's docstring claims it's "the recommended
import location" — but it imports 18 names from `ermrest.py` (a
file that itself imports from `deriva.core.typed`), so the chain is
`deriva.core.typed → core/ermrest.py → core/definitions.py →
caller`. Three hops to reach `ColumnDef`.

For internal callers this is fine. For external (MCP / skill) callers
who land on `from deriva_ml.core.definitions import ColumnDef` it
works but obscures the upstream source.

**Recommendation:** add a one-line "actual source" pointer in
`definitions.py` per re-exported group. **Severity: low.**

### 6.8 Doctest coverage on selectors / factories is good; on catalog methods is correctly skipped

Spot-check: `feature.py:51-71` (`FeatureRecord` class) has a
runnable doctest. `dataset/aux_classes.py:DatasetSpec.from_shorthand`
(lines 379-389) — runnable. `core/exceptions.py:323-348` —
runnable. The pattern from deriva-ml's `CLAUDE.md` is followed:
runnable for pure-Python, `# doctest: +SKIP` for catalog-touching.

**No issue.**

### 6.9 `Workflow.workflow_type` accepts `str | list[str]`, returns `list[str]`

`execution/workflow.py:85-89` documents this as a public API choice
("Accepts a single string or a list of strings. Internally normalized
to a list"). This is fine and well-documented. Mentioning it as the
counter-example: when a method legitimately needs an asymmetric
type signature, the docstring carries the contract — same standard
the rest of the codebase should aim for in `model/catalog.py`
(§6.3).

---

## Ranked actions (1–N)

Ranked by `(impact × confidence) / cost`. Items 1–4 are mechanical
quick wins; 5–7 want a small test plan; 8–9 are larger refactors
worth scheduling.

| # | Action | Risk | LoC | Files | Rationale |
|---|---|---|---|---|---|
| 1 | **§1.1** Delete the icecream import + fallback pattern from all 9 modules | trivial | −55 | 9 files | Pure dead-code removal; no `ic()` callers exist |
| 2 | **§1.2 + §1.3 + §1.5** Delete `LoggerMixin`, `DerivaMLNoExecutionContext`, `DerivaML.working_data` | low | −70 | 3 files | Three independent dead symbols; mechanical |
| 3 | **§1.4 + §1.8 + §5.4** Delete `AssetRID`, `AssetCopyMode`, `_LEGACY_ONLY_PARAMS` (collapse on top of `AssetMode`, `_warn_about_legacy_params`'s own dict) | low-medium | −100 | 3 files | Three different back-compat sinks that no internal caller uses |
| 4 | **§4.3** Fix `cache_status` docstrings to say `cached_holey` (drop `cached_incomplete`) | trivial | ±0 | 2 files | Doc-bug; current code returns `cached_holey` |
| 5 | **§1.6 + §5.1** Collapse the 7-module shim layer to direct `deriva.bag.*` imports | low-medium | −200 | 10 files (7 shims + 3 callers) | Removes a re-export layer the bag-pipeline migration is already done with |
| 6 | **§4.6 + §5.5 + §1.7** Consolidate `list_dataset_element_types` duplication, rename `_get_table_contents`, delete `delete_database` + `get_orm_association_class` aliases | low | −45 | 4 files | Removes "fix it in two places" hazards in the model layer |
| 7 | **§2.1 + §2.2 + §2.3** Migrate three raw-URL builders to `pathBuilder` + `.in_()` (clone_via_bag nested-dataset expansion, localize asset record fetch, rid-lease validate) | low-medium | −30 | 3 files | Brings the codebase in line with the API-priority rule from `CLAUDE.md` and removes three "datapath doesn't have `.in_`" comments that became wrong upstream |
| 8 | **§4.1 + §5.3** Unify the three Chaise-navbar builders into a single function in `schema/annotations.py` | medium | −150 to −180 | 3 files | Highest-LoC simplification in the audit; eliminates a known-drifted parallel implementation |
| 9 | **§5.2** Decide `DatasetBagBuilder` cutover: either finish ADR-0006 wiring or remove the unreachable bag-pipeline helpers | high (option a) / low (option b) | option b: −165 | `bag_builder.py` + tests | Active hazard. Option (b) recovers correctness now; option (a) finishes the ADR. Either is better than the current intermediate state |
| 10 | **§4.4** Project-wide replace inline `@validate_call(config=ConfigDict(arbitrary_types_allowed=True))` with `@validate_call(config=VALIDATION_CONFIG)` | low | ±0 net | ~30 files | Consolidates configuration point for the Pydantic validator |
| 11 | **§4.5** Adopt `from deriva_ml.core.logging_config import get_logger; logger = get_logger(__name__)` across the 44 callsites | low | ±0 | ~25 files | Final piece of the logging-consolidation begun by `logging_config.py` |
| 12 | **§4.7** Phase 2 — survey `_DEFAULT_SKIP_TABLES` / `terminal_tables` / vocab-end pruning policies and unify under `FKTraversalPolicy` | medium | TBD | 3 files | Depends on action 9 landing first |
| 13 | **§6.1** Phase 2 — split `DerivaModel`'s denormalization planner (lines 825-1616) into a separate module adjacent to `local_db/denormalizer.py` | medium-high | ±0 net (800 LoC moves) | 2 files | Largest single file in `model/`; mixed responsibilities |

Items 1–4 are good to land in one cleanup PR. Items 5–7 want a
brief test pass each. Items 8–9 are the highest-value
simplifications and should be scheduled. Items 10–11 are good
follow-ups for a code-mod tool. Items 12–13 are Phase 2 scope.

---

## Worst-offending subsystems for Phase 2 deep audit

### Worst-offender #1: `dataset/`

Two 2000+-line modules (`dataset.py` 3590, `dataset_bag.py` 2392),
plus a 799-line `CatalogGraph` that's slated for replacement but
hasn't been replaced, plus a 364-line `DatasetBagBuilder` façade.
Phase 2 should:

- Establish whether `DatasetBag` and `Dataset` can share more
  implementation (currently they reimplement `list_dataset_members`,
  `list_dataset_parents`, etc. with similar but not identical code).
- Resolve the `DatasetBagBuilder` cutover one way or the other
  (Lens 5 action 9).
- Audit the bag-export → MINID flow (`_create_dataset_minid`,
  `_create_dataset_bag_client`) for opportunities to use
  `BagBuilder` directly.

### Worst-offender #2: `model/`

The shim layer (Lens 1, action 5) belongs in Phase 1. Phase 2 should
audit:

- `model/catalog.py` god-class structure (§6.1). 2085 LoC across
  three roles is too much for one class.
- `model/annotations.py` (1277 LoC) — Chaise annotation builders.
  Has its own duplicated navbar code (§4.1). Phase 2 should
  reconcile with `schema/annotations.py`.
- The two association predicates (`is_association` vs
  `_is_association_table`) — §2.7. Decide whether the
  topological-only variant can be folded into deriva-py's
  `is_association(return_fkeys=True)`.

### Worst-offender #3 (optional): `execution/`

11.3 KLoC across 24 files is the largest subsystem but most of the
LoC is load-bearing — execution lifecycle, RID leasing, state
machine, upload engine, bag commit. Phase 2 would benefit from
package-level structural review:

- Three module clusters (lifecycle, I/O, config) are intermixed in
  one flat directory.
- `execution.py` is 2519 LoC — second-largest file in the codebase.
- The `bag_commit.py` (744 LoC) is the new bag-pipeline integration;
  its interaction with the older `upload_engine.py` (865 LoC) is
  worth a focused audit.

Recommendation: scope Phase 2 to `dataset/` + `model/` if a single
follow-up audit. `execution/` is large enough to deserve its own
session.

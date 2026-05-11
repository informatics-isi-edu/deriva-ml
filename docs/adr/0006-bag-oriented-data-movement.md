# ADR-0006: Bag-oriented data movement — bag is the unifying artifact

Date: 2026-05-10
Status: Accepted

## Context

DerivaML moves catalog content across three boundaries today, each
implemented by its own machinery:

1. **Catalog → bag** (dataset download). `Dataset.download_dataset_bag`
   constructs an export spec via `CatalogGraph` and drives the
   deriva-py export engine to write a BDBag. Anchored on the `Dataset`
   table; FK walk shaped specifically by dataset element types,
   features, and nested-dataset semantics.
2. **Catalog → catalog** (clone). `catalog/clone.py` implements a
   bespoke three-stage pipeline: schema-without-FKs → async row copy
   → FK application with orphan handling. ~1900 lines. Does not pass
   through a bag — moves rows directly between ERMrest catalogs.
3. **Files → catalog** (model template / `load_cifar10.py`). Hand-rolled
   image-upload loop that calls deriva-py's upload machinery directly,
   bypassing both bags and clone. Schema setup is one path
   (`setup_domain_model`); data load is another (loops calling
   `add_dataset_members`, `asset_file_path`, etc.).

The three paths share a problem domain — write rows that satisfy FK
constraints, handle assets, preserve provenance — but no
implementation. Each grew its own FK-traversal logic, its own asset
handling, its own error semantics. `CatalogGraph` walks paths and
filters by dataset membership; `clone._discover_reachable_tables`
walks a flat reachable set with schema/table blocklists; `load_cifar10`
ignores the question entirely because it has no source schema to walk.

The redesign goal is to **make the BDBag the unifying artifact**.
Every data-movement operation factors through a bag:

- "Catalog → bag" stays (it already does).
- "Catalog → catalog" becomes "catalog → bag → catalog."
- "Files → catalog" becomes "files → bag → catalog."

A bag is on-disk, citable (via MINID), inspectable (via Bag
SQLAlchemy),
and round-trippable. Centralizing on it gives one set of FK
semantics, one asset story, and one provenance trail.

## Decision

### Three bag-producing use cases converge on one pipeline

Three distinct producer scenarios — all today implemented by
separate, divergent code paths — converge under the redesign on
a single bag-production pipeline plus a single bag-consumption
pipeline. The bag in the middle is the canonical interchange
artifact.

| # | Use case | Input | `DataSource` | Producer class |
| --- | --- | --- | --- | --- |
| 1 | Programmatic build (cifar, Kaggle, TF dataset) | DataFrames / dict iterables in caller's memory | `DataFrameDataSource`, `IterableDataSource` | `BagBuilder` |
| 2 | End-of-execution upload | Local SQLite from `local_db` | `LocalDBDataSource` | `BagBuilder` |
| 3 | Catalog slice / clone / dataset bag | Live ERMrest catalog | n/a (deriva-py export engine) | `CatalogBagBuilder` |

All three produce a bag conforming to the deriva-bag profile.
All three carry a `data/schema.json` describing every table in
the bag plus every FK relationship between them — byte-identical
format regardless of which producer wrote it (cases 1 and 2 use
`schema_io.metadata_to_ermrest_json(metadata)`; case 3 routes
the live ERMrest model through the same conversion path).

All three are consumed identically:

- **Inspect** — `BagDatabase.open(path)` returns SQLAlchemy
  `MetaData` + row access + asset access.
- **Load into a catalog** — `BagCatalogLoader(catalog, bag, policy).run()`
  pushes the bag's contents to any ERMrest destination, using
  one FK-ordered insert + Hatrac-dedupe upload path.
- **Cite** — every bag has a MINID (the producer mints it as
  part of finalization, regardless of use case).

This unification is the load-bearing claim of the redesign. The
end-of-execution upload (use case 2) becomes "build a bag from
the local DB, then load it" — same two steps as catalog
re-import (use case 3 → load) and same two steps as
programmatic publication (use case 1 → load). Today these are
three separate code paths (a hand-rolled upload loop, the export
engine, and bespoke clone logic); after the redesign they share
the bag pipeline.

A consequence worth naming: the end-of-execution bag becomes a
real durable artifact of every execution. Today's
`commit_execution` hands data to the uploader and forgets it; the
new path leaves the bag on disk, citable via MINID, inspectable
via `BagDatabase`, and re-loadable if the catalog push fails
mid-flight.

### Six classes, one profile, one policy

All bag machinery lives in **`deriva.bag`** (a new submodule of
deriva-py, on the `deriva-ml` branch). Six classes, two shared
types, and one profile JSON document:

| Class | Side | Role |
| --- | --- | --- |
| `BagDatabase` | consumer | Open a bag → Bag SQLAlchemy layer (engine + metadata + ORM `Base`) + asset access. |
| `BagBuilder` | producer (constructive) | Write a bag from in-memory data. No source catalog. |
| `CatalogBagBuilder` | producer (walking) | Catalog → bag, via the deriva-py export engine. |
| `BagCatalogLoader` | round-trip | Bag → catalog, walking the Bag SQLAlchemy layer. |
| `SchemaBuilder` + `SchemaORM` | shared primitive | ERMrest `Model` → SQLAlchemy ORM. Used by `BagDatabase`, deriva-ml `local_db`, and `BagCatalogLoader`. |
| `DataLoader` + `DataSource` / `BagDataSource` / `CatalogDataSource` | shared primitive | Generic "fill an ORM from a source." |

Shared types:

- **`Anchor`** — discriminated union (`RIDAnchor`, `TableAnchor`,
  `PathAnchor`) describing where `CatalogBagBuilder` starts walking.
  "Anchor" rather than "seed" to avoid the random-number-seed
  reading in ML contexts.
- **`FKTraversalPolicy`** — config for the FK graph walk:
  schema/table allow/block lists, depth bound, `vocab_export`,
  `asset_mode`, `dangling_fk_strategy`. Walks are always bidirectional
  and vocab tables are always inbound-only (hardcoded walker
  rules; not policy fields). Shared between `CatalogBagBuilder`
  (catalog → bag) and `BagCatalogLoader` (bag → catalog).

Profile artifact:

- **`deriva-bag-profile.json`** — the BagIt Profile JSON document
  the producers declare and the consumers may validate. Hosted in
  the deriva-py repo under `deriva/bag/profiles/`. Referenced by
  identifier URL from each bag's `bag-info.txt`.

### The deriva-bag profile is the contract

Bags written by `BagBuilder` or `CatalogBagBuilder` follow a
single layout — the *deriva-bag profile*. The profile is
registered as a formal **BagIt Profile** (a JSON document
validatable by `bdb.validate_bag_profile`), declared in
`bag-info.txt` via the `BagIt-Profile-Identifier` field, and
hosted as part of the `deriva.bag` submodule
(`deriva/bag/profiles/deriva-bag-profile.json`). The profile
constrains:

- `data/schema.json` — ERMrest model description (subset of source
  catalog, or projection of the SQLAlchemy `MetaData` supplied to
  `BagBuilder`).
- `data/<schema>/<table>.csv` — one CSV per table, header row +
  rows in ERMrest CSV format.
- `data/asset/<asset_table>/<RID>/<filename>` — asset bytes (when
  embedded) or referenced via `fetch.txt`.
- `metadata/` — optional provenance: the `Anchor` + `FKTraversalPolicy`
  used by `CatalogBagBuilder`, build timestamps, source catalog
  handle.

`BagDatabase` reads this profile. `BagCatalogLoader` consumes it.
Both may call `bdb.validate_bag_profile` before reading to detect
non-conforming bags early. The profile is what makes round-trip
closed: `catalog → bag → catalog` produces an equivalent
destination catalog (modulo orphan-policy outcomes).

#### Back-compat for bags produced before profile registration

Dataset bags produced today (by the pre-redesign export path) do
not declare a `BagIt-Profile-Identifier`. The migration treats an
absent identifier as **implicit conformance**: `BagDatabase` and
`BagCatalogLoader` validate only when the identifier is present,
and otherwise assume the bag follows the profile's layout
conventions. New bags produced by either builder always declare
the identifier explicitly.

### Asset mode and the load-side reconciliation

`FKTraversalPolicy.asset_mode` has four values, all of which
delegate the per-asset work to deriva-py's existing upload recipe
(`DerivaUpload._uploadAsset`) rather than reimplementing it.
That recipe already handles three things we'd otherwise rebuild:

- **Hatrac byte dedupe.** Before transferring bytes, `put_loc`
  issues a HEAD against the destination object. If MD5 (or SHA256)
  matches, no bytes are uploaded. The catalog row is still
  reconciled afterward.
- **Catalog row reconciliation.** Even when bytes are already
  in Hatrac, the asset row is compared column-by-column against
  the bag's row and updated when the bag carries additional or
  changed metadata.
- **Pre-allocated RID handling and transfer-state resumption.**
  Existing in `_createFileRecordWithRid` and the transfer-state
  machinery — the loader inherits both.

The three modes:

| Mode | Asset rows | Asset bytes |
| --- | --- | --- |
| `ROWS_ONLY` | inserted/reconciled with source URL | not transferred |
| `UPLOAD_IF_MISSING` *(default)* | inserted/reconciled | uploaded via Hatrac dedupe (no-op when destination already has matching MD5) |
| `UPLOAD_FORCE` | inserted/reconciled | re-uploaded regardless (`force=True`) |

The "don't include this asset table at all" case belongs to
`exclude_tables` (or `exclude_schemas`), not to `asset_mode`.
Keeping scope and behavior in separate policy fields avoids the
"I included X in traversal but set mode to SKIP — which wins?"
ambiguity.

`UPLOAD_IF_MISSING` is the default because it's the right behavior
for round-trip clone, re-import, partial-bag-update, and any "load
this bag idempotently" scenario. Re-running the loader against the
same destination should be a no-op for bytes that are already
there, but should still pick up any metadata changes in the bag's
rows. Hatrac's HEAD-and-compare gives us that automatically.

`UPLOAD_FORCE` is reserved for the rare case where the operator
knows the source bytes are authoritative and wants to overwrite
the destination's. It bypasses dedupe and re-uploads every asset.

The bag-state × mode compatibility matrix, enforced at policy
validation time before any rows are inserted:

| Bag state | Mode | Behavior |
| --- | --- | --- |
| materialized | `ROWS_ONLY` | rows reconciled with source URLs; bag bytes ignored |
| materialized | `UPLOAD_IF_MISSING` | rows reconciled; bytes uploaded via dedupe |
| materialized | `UPLOAD_FORCE` | rows reconciled; bytes re-uploaded |
| holey | `ROWS_ONLY` | rows reconciled with source URLs; no bytes needed |
| holey | `UPLOAD_IF_MISSING` | **rejected** — no local bytes; user must materialize first or switch to `ROWS_ONLY` |
| holey | `UPLOAD_FORCE` | **rejected** — same reason |

The two rejected combinations error early with a clear message
pointing at `bdb.materialize(bag)` or `asset_mode=AssetMode.ROWS_ONLY`. The
rejection is deliberate: silently materializing a multi-gigabyte
bag at load time would hide the cost from the caller. Forcing the
explicit step keeps the network and disk traffic visible.

### Orphan strategy and where it lives

`FKTraversalPolicy` carries a `dangling_fk_strategy` field with three
values: `FAIL` (default), `DELETE`, `NULLIFY`. Orphans arise when
the bag carries a row whose FK reference points at a parent that
is not in the bag — a consequence of *incoherent row-level
policies* in source catalogs (a row is visible to the cloning user
but its parent is not). Without orphan handling, the destination
either rejects the row at FK-application time or is left with
broken references.

Detection and application split:

- **Detection happens at bag-read time** in `BagDatabase`. The
  SQLAlchemy ORM can compute dangling FK references cheaply with
  no catalog round-trips. The list is exposed (e.g.,
  `bag.orphans`) so callers can inspect a bag before deciding
  which strategy to apply.
- **Application happens at load time** in `BagCatalogLoader`,
  which consumes the orphan list and acts according to
  `dangling_fk_strategy`: refuse to start (`FAIL`), drop orphan rows
  (`DELETE`), or null out the FK column (`NULLIFY`, only when the
  column is nullable).
- **`CatalogBagBuilder` is orphan-agnostic.** It writes the walk's
  output as-is. Orphans are a property of the bag's contents, not
  the build process. A user who wants the *builder* to chase
  parents and prevent orphans entirely should adjust the
  `Anchor` + `FKTraversalPolicy` scope (e.g., extend the walk), not
  reach for an orphan strategy.

The default `FAIL` is deliberate: orphans usually indicate either
a scope mismatch or a real data-integrity problem at the source.
Users who know their catalog has policy incoherence opt in to
`DELETE` or `NULLIFY` explicitly.

### Anchor resolution semantics

`Anchor` instances (`RIDAnchor`, `TableAnchor`, `PathAnchor`) are
resolved by `CatalogBagBuilder` into a starting set of
RIDs-per-table before the FK walk begins. Three edge cases get
defined behavior:

- **Overlap is silently deduped.** Anchors that name overlapping
  rows are unioned; the FK walk's visited-set takes care of the
  rest. No warning. Anchors describe intent; overlap doesn't
  change the bag's contents.
- **`RIDAnchor` with missing RIDs fails fast.** Each `RIDAnchor`
  table is validated against the source catalog with a single
  `?RID=in(...)` query; absent RIDs raise an error naming the
  missing ones before any walk work begins. Conservative default;
  catches typos and stale configs cleanly.
- **`TableAnchor` is silently empty-tolerant.** A table with no
  rows in scope is a legitimate "no rows from this table" outcome.
- **`PathAnchor` warns but proceeds when empty.** An empty
  datapath result is ambiguous between "the filter is correct"
  and "the filter is wrong" — log the empty result, let the build
  continue.

The defaults follow the same logic as `dangling_fk_strategy=DanglingFKStrategy.FAIL`:
strict on enumeration mistakes (where user intent is precise),
permissive on filter-based emptiness (where intent is fuzzier).

### Table-kind handling lives in the generic walker

The walker inside `CatalogBagBuilder` (and the corresponding
inverse walk inside `BagCatalogLoader`) recognizes three
structural table kinds and applies a hardcoded rule for each.
The kinds are detected via methods that already exist on the
deriva-py `Table` class — `is_vocabulary()`, `is_asset()`, and
the `find_associations()` arity rules used by
`BagDatabase.is_association_table`. None of the rules require
deriva-ml-specific knowledge.

| Kind | Detection | Rule |
| --- | --- | --- |
| Vocabulary | `Table.is_vocabulary()` | Follow FKs *into* the vocab table (pulling in referenced terms) but not *out of* it. Hardcoded; no policy field. Optionally run a separate full-export pass after the main walk when `vocab_export=VocabExport.FULL`. |
| Asset | `Table.is_asset()` | Walked like any other table; the walker emits asset layout (`data/asset/<table>/<RID>/<filename>` + `fetch.txt`). Loader's `asset_mode` decides byte handling. |
| Association | arity rules | Tag-along: included automatically when both endpoints are in the bag; doesn't count toward `max_depth`. |

The vocab rule is what fixes the loop-back explosion: without
it, a walk that reaches a vocab table (e.g., Species) would
follow inbound FKs back to every other table that uses that
vocab (every Subject, every Specimen, etc.), pulling in
unrelated rows the user didn't ask for. The walker enters
vocab tables (pulling the referenced terms) but does not exit
them. This is hardcoded; the policy has no field controlling
vocab-traversal direction because every real producer wants the
same behavior.

The full-vocabulary pass (`vocab_export=VocabExport.FULL`) is what today's
`CatalogGraph._export_vocabulary` does for dataset bags —
emitting every term, not just the ones referenced by member
rows. Folding it into the generic walker means
`DatasetBagBuilder` doesn't run a separate vocab-export step:
it sets `vocab_export=VocabExport.FULL` on the policy and the engine
handles both passes.

Putting these rules in the walker is what keeps domain wrappers
thin. Clone, slice-by-RID, and dataset-bag all need consistent
vocab / asset / association handling; if the rules lived in
`DatasetBagBuilder`, clone and slice would either replicate them
or quietly diverge. Centralizing them in `CatalogBagBuilder`
gives every producer case the same structural-table semantics
for free.

### Deriva-ml domain wrapper: `DatasetBagBuilder`

`CatalogBagBuilder` in `deriva.bag` is deriva-ml-agnostic. The
dataset-download path today (`CatalogGraph` in deriva-ml) carries
four dataset-specific concerns that do not belong in the generic
builder:

1. **Association filtering by member element types** — include
   `Dataset_X` association tables only when the dataset has members
   of element type `X`. Skips empty associations and prunes paths.
2. **Vocabulary tables exported in full** — vocabularies are
   exported as standalone queries, not joined through the
   element-type graph. Paths ending in vocabulary tables are
   pruned from the main walk.
3. **Feature tables per element type** — for each member element
   type, the dataset's feature tables must be included even though
   the generic walk wouldn't reach them.
4. **Nested datasets** — a dataset's child datasets get their own
   walks, recursively, up to a configurable `nesting_depth`.

These four concerns become `DatasetBagBuilder`
(`src/deriva_ml/dataset/bag_builder.py`), a thin wrapper that:

- Constructs the `Anchor` list from the dataset's row plus its
  nested children.
- Constructs an `FKTraversalPolicy` with `exclude_tables`
  covering concern 1 (associations without members) and
  `vocab_export=VocabExport.FULL` for concern 2 (the bag wants the complete
  vocabulary, not just terms referenced by member rows). Feature
  tables (concern 3) are reached naturally by FK-following from
  member element rows; no separate "force-include" mechanism is
  needed.
- Hands the result to a generic `CatalogBagBuilder`. Vocabulary
  handling — inbound-only traversal *into* vocab tables to avoid
  loop-back, plus the optional full-vocabulary pass — is part of
  the generic engine, not a `DatasetBagBuilder`-specific
  post-processing step.

`Dataset.download_dataset_bag` is rewired to construct a
`DatasetBagBuilder` internally and return a `DatasetMinid` from
its output. The public signature does not change.

`FKTraversalPolicy` deliberately omits a `force_include` field:
"always include table X" reduces either to `TableAnchor("X")`
(every row) or to "let the walk reach X via FK-following" (rows
reachable from another anchor). Either way, the generic policy
does not need a third mechanism. Domain-specific row-shape
concerns (which features matter for which element types, how
vocabularies are exported) belong in the domain wrapper, not in
the generic policy.

### Bag SQLAlchemy strategy — align with `local_db`

The Bag SQLAlchemy layer (the per-bag SQLite database `BagDatabase`
creates) must match the strategy already used by the `local_db`
subsystem. Today they diverge in three places worth reconciling
inside the deriva-py PR:

| Concern | Today's `BagDatabase` | Today's `local_db` | Decision |
|---|---|---|---|
| SQLite pragmas | default (no WAL, no busy_timeout, FK off) | WAL + synchronous=NORMAL + foreign_keys=ON + busy_timeout=5000 (`create_wal_engine`) | Use `create_wal_engine` everywhere. |
| Read-only access | not supported | URI form `mode=ro&uri=true` | Support read-only on the bag SQLite too (loader, MCP reader). |
| Schema versioning | none | `schema_meta` table + `SchemaVersionError` | Add to bag SQLite (`BAG_SCHEMA_VERSION` constant, bumped when the bag-SQLite layout changes). |

Three concrete actions in the deriva-py PR:

1. **Move `sqlite_helpers.py` upstream.** It's a generic SQLite
   engine factory (WAL + pragmas + read-only URI + `schema_meta`)
   used by both bag-SQLite and `local_db`. It lives in
   `deriva.bag.sqlite_helpers` (or `deriva.bag.engine`) post-move.
   The deriva-ml migration PR replaces `local_db/sqlite_helpers.py`
   with a re-export.
2. **Update `BagDatabase` to use `create_wal_engine`.** One-line
   refactor inside `BagDatabase.__init__` — replaces the bare
   `create_engine(f"sqlite:///{...}")` call. Picks up WAL,
   busy_timeout=5000, foreign_keys=ON, and the read-only URI mode
   for free.
3. **Add `ensure_schema_meta` to `BagDatabase.__init__`.** A
   `BAG_SCHEMA_VERSION = 1` constant in `deriva.bag` gets recorded
   on first build. Future versions bump it; mismatched on-disk
   state raises `SchemaVersionError` rather than failing with a
   confusing column-missing error.

**The on-disk layout is content-addressed by the BDBag checksum.**
Today's layout (`{database_dir}/{rid}_{checksum}/main.db`)
assumes one bag = one anchoring RID. That assumption fails for
multi-RID anchors (`RIDAnchor("Subject", ["S1", "S2", ...])`),
table-wide anchors (`TableAnchor("Subject")`), `PathAnchor`, and
catalog-wide clones — none of which has a single load-bearing
RID. The new layout:

```
{cache_dir}/
    index.sqlite                          # per host/catalog index
    bags/
        {checksum}/
            bag/                          # bag directory
            db/main.db                    # Bag SQLAlchemy main file
            db/{schema}.db                # attached schemas
```

Identical-content bags (same closure built under different
anchors) share one cache entry automatically.

The **cache index** (`index.sqlite`, one per host/catalog) carries
two tables:

- `bags(checksum, profile_id, built_at, anchor_summary_json,
  size_bytes)` — one row per cached bag.
- `bag_anchor_rids(checksum, table, rid)` — reverse index from
  anchor RIDs to bag checksums. Replaces today's `{rid}_*`
  directory-glob affordance: callers ask
  "`SELECT checksum FROM bag_anchor_rids WHERE table='Dataset'
  AND rid=?`" instead of scanning filenames.

The index itself uses `create_wal_engine` and `schema_meta` —
same engine policy as the bags it indexes.

The reverse index is what makes the "list cached versions of
dataset X" workflow continue to work in a world where (a) a bag
can be anchored by many RIDs and (b) the same bag can result
from different anchor configurations. The on-disk filename is no
longer load-bearing; the index is.

`BagCatalogLoader` opens the bag SQLite read-only (it only reads
rows to push to the destination catalog). `BagBuilder` opens it
read-write (it's writing rows). `DatasetBag` opens it read-only,
which makes multi-worker PyTorch DataLoader readers safe.

### Producer split: constructive vs. walking

`BagBuilder` and `CatalogBagBuilder` are deliberately separate.
They share the *output contract* (the profile) and a substantial
chunk of *schema-handling implementation* (`schema_io`), but
each owns its own row-writing path:

- **`CatalogBagBuilder` drives the deriva-py export engine.** It
  generates an export spec (query processors + bag settings) from
  `Anchor` + `FKTraversalPolicy` and hands it to the existing export
  engine. The engine writes the bag — paged streaming, async
  fetching, MD5 manifest, S3 upload, MINID post-processors are all
  preserved. `CatalogBagBuilder` does not reimplement them.
- **`BagBuilder` writes bytes directly.** No export engine, no
  catalog. Takes either deriva-py `typed.SchemaDef`s or a
  SQLAlchemy `MetaData` (or both); normalizes to `MetaData`
  internally; `add_rows`
  accepts a pandas DataFrame, a list of dicts, or any iterable of
  dicts; rows flow through `DataLoader` with a CSV sink. Same
  `DataLoader + DataSource` machinery the rest of the system
  already uses, with a write-out sink instead of an into-SQLite
  sink.

**SQLAlchemy `MetaData` is the canonical *internal* schema
vocabulary; producers accept multiple input forms.** Internally,
every consumer (`BagDatabase`, `BagCatalogLoader`, `local_db`)
and the catalog walker all speak `MetaData`. Externally,
`BagBuilder` accepts whichever schema vocabulary fits the
caller's context:

- **deriva-py `typed.SchemaDef`** — for callers who are already
  using `typed` to push schemas to a catalog. The same
  `SchemaDef` object can be used for both the catalog push and
  the bag construction; no second declaration.
- **SQLAlchemy `MetaData`** — for callers bridging from
  non-deriva tooling or who already have a `MetaData` from
  another source.

Both are accepted; `BagBuilder` normalizes to `MetaData` at
construction time.

The other layers all converge on `MetaData` too:
- `BagDatabase` returns a `MetaData` on bag read.
- `local_db` already manipulates `MetaData` internally.
- `CatalogBagBuilder` projects an ERMrest `Model` to a `MetaData`
  slice (`ermrest_model_to_metadata`) before driving the export
  engine — so even the live-catalog path internally speaks
  `MetaData` for the schema half of its work.

A new module **`deriva.bag.schema_io`** owns the conversions:
- `typed_schema_def_to_metadata(schema_def, metadata)` — adds
  typed defs into a `MetaData`. Used by `BagBuilder` to accept
  `typed` input.
- `metadata_to_typed_schema_defs(metadata) -> list[SchemaDef]` —
  inverse. Useful when `BagCatalogLoader` needs to create
  missing tables in the destination catalog from the bag's
  schema.
- `metadata_to_ermrest_json(metadata) -> dict` — used by
  `BagBuilder` to write `schema.json`. Also used by
  `CatalogBagBuilder` so that file's format is identical between
  the two producers.
- `ermrest_json_to_metadata(schema_dict) -> MetaData` — used by
  `BagDatabase` to read `schema.json`.
- `ermrest_model_to_metadata(model) -> MetaData` — used by
  `CatalogBagBuilder` to derive the bag's `MetaData` from a live
  catalog's `Model`.
- The ERMrest ↔ SQLAlchemy type-mapping tables (one direction
  each), single source of truth.

`MetaData` is the hub of the conversion graph: every other
representation has a round-trip path through it.

Code shared by both producers — `schema_io` (schema-side),
archive-as-zip, `metadata/` provenance writer, profile path
constants, asset layout conventions — lives in `deriva.bag.profile`
(asset/archive bits) and `deriva.bag.schema_io` (schema bits).
The shared surface is no longer the ~20-line "common profile
constants" originally estimated; it's the entire schema
vocabulary.

Where the producers still diverge is the **row-writing path**:
`CatalogBagBuilder` uses the deriva-py export engine for paged
async ERMrest reads; `BagBuilder` uses `DataLoader + CSVSink`
for local DataFrame/iterable inputs. Losing async/paged
streaming on the catalog-walking side would be a regression we
don't want, so the row-writing path stays bifurcated. The
schema-writing path unifies.

### Consumer side already exists upstream

`BagDatabase` already lives in deriva-py's `deriva-ml` branch as
`deriva/core/bag_database.py`. The B3 migration moves it (and the
primitives it shares with deriva-ml's `local_db`) into
`deriva.bag.*` and deletes the parallel implementations in
deriva-ml's `model/` directory.

### deriva-ml domain layer is unchanged in shape

After the migration:

- `DatabaseModel(BagDatabase)` — adds `bag_rids`, `dataset_version`,
  `rid_lookup`, `_get_dataset_execution`. ~80 lines, down from ~400.
- `DerivaMLBagView` (renamed from `DerivaMLDatabase`) — domain
  layer over `DatabaseModel`: vocabulary, features, element types,
  factory for `DatasetBag`.
- `DatasetBag` — per-dataset view over `DerivaMLBagView`.
  Implements `DatasetLike`; unchanged interface for callers.

### Clone is rewritten as `CatalogBagBuilder + BagCatalogLoader`

After migration, `catalog/clone.py` collapses from ~1900 lines to
a thin wrapper: build an `FKTraversalPolicy` and a single
`TableAnchor("*")` / equivalent full-catalog anchor, run
`CatalogBagBuilder` to a temp bag, run `BagCatalogLoader` from
that bag to the destination. The orphan strategies (FAIL, DELETE,
NULLIFY) become a field on `FKTraversalPolicy` consumed by the
loader's FK-application step.

### `load_cifar10.py` is rewritten as `BagBuilder + BagCatalogLoader`

The hand-rolled image upload loop becomes:

```python
from deriva.core.typed import (
    SchemaDef, TableDef, ColumnDef, KeyDef, ForeignKeyDef, BuiltinType,
)

# One schema definition; used both for the catalog push and the bag build.
cifar_schema = SchemaDef("cifar", tables=[
    TableDef("Label",
        columns=[ColumnDef("Name", BuiltinType.text, nullok=False)],
        keys=[KeyDef(["Name"], primary=True)],
    ),
    TableDef("Image",
        columns=[
            ColumnDef("RID", BuiltinType.ermrest_rid),
            ColumnDef("Filename", BuiltinType.text, nullok=False),
            ColumnDef("Label", BuiltinType.text, nullok=False),
        ],
        keys=[KeyDef(["RID"], primary=True)],
        foreign_keys=[
            ForeignKeyDef(["Label"], "cifar", "Label", ["Name"]),
        ],
    ),
])

# Push schema to the catalog using the same defs.
model.create_schema_from_def(cifar_schema)

images_df = pd.DataFrame({...})
labels_df = pd.DataFrame({"Name": [...]})

with BagBuilder(schema_defs=[cifar_schema], output_dir=output_dir) as bb:
    bb.add_rows("Label", labels_df)
    bb.add_rows("Image", images_df)
    for rid, path in image_files.items():
        bb.add_asset("Image", rid, path)
    bag = bb.finalize()

BagCatalogLoader(catalog, bag, traversal).run()
```

The schema-setup half of `load_cifar10` stays (catalog-side
`setup_domain_model`); the data-load half collapses to the snippet
above.

## Consequences

### Positive

- **One set of FK semantics.** `FKTraversalPolicy` is the single policy
  object; today's three independent FK-walks become one.
- **One asset story.** `asset_mode`
  (`ROWS_ONLY`/`UPLOAD_IF_MISSING`/`UPLOAD_FORCE`)
  applies uniformly across `BagBuilder`, `CatalogBagBuilder`, and
  `BagCatalogLoader`.
- **Round-trip is closed.** `catalog → bag → catalog` is a defined
  operation with a stable artifact in the middle — debuggable,
  inspectable, citable.
- **Clone shrinks dramatically.** ~1900 lines → a thin wrapper.
- **`load_cifar10.py` shrinks.** Hand-rolled upload loop replaced
  by `BagBuilder` + `BagCatalogLoader`.
- **One implementation of Bag-SQLAlchemy-from-bag.** Today's parallel
  `BagDatabase` (upstream) + `SchemaBuilder` + `DataLoader` +
  `BagDataSource` (deriva-ml) collapse into one tree under
  `deriva.bag`.
- **Producer-side machinery is co-located with consumer-side**.
  The bag is the contract; the code that produces it and the code
  that consumes it live next to each other.
- **One canonical internal schema vocabulary, multiple accepted
  inputs.** SQLAlchemy `MetaData` is the internal hub for
  `BagDatabase` (output), `BagBuilder` (normalized input),
  `BagCatalogLoader`, `local_db`, and `CatalogBagBuilder` (via
  `ermrest_model_to_metadata`). `BagBuilder` accepts either
  deriva-py `typed.SchemaDef` (so users already pushing schemas
  to a catalog can use the same defs for bag construction —
  *one declaration, two uses*) or a SQLAlchemy `MetaData` (for
  callers bridging from other tooling). All conversions live in
  one module (`schema_io`); the type-mapping tables have one
  home.

### Negative

- **One large deriva-py PR.** B3 unifies five primitives (`BagDatabase`,
  `SchemaBuilder`, `SchemaORM`, `DataLoader`, `DataSource` family)
  plus the new producer classes in a single deriva-py PR. Larger
  blast radius than a narrower migration would have.
- **`local_db` is touched indirectly.** Its `SchemaBuilder` import
  moves from `deriva_ml.model` to `deriva.bag`. Behavior unchanged.
- **Mid-flight catalog operations.** While the deriva-py PR is in
  flight (before deriva-ml migrates), deriva-ml's `main` still
  uses local `SchemaBuilder` etc. The migration PR in deriva-ml
  pins the new deriva-py commit and switches imports atomically.
- **Some `CatalogGraph` complexity moves to `CatalogBagBuilder`.**
  Dataset-specific association filtering (filter associations by
  member element types, prune paths ending in vocabularies, walk
  nested datasets) does not belong in the generic
  `CatalogBagBuilder`. It becomes a deriva-ml-domain wrapper
  (`DatasetBagBuilder`) that
  refines the generic anchor + traversal before handing them to the
  upstream class.

### Neutral / open

- **MINID minting is unchanged.** Today's `_create_dataset_minid` /
  S3 post-processor flow is part of the export-engine spec
  `CatalogBagBuilder` generates. Untouched.
- **Asset upload mode for `BagBuilder`-produced bags.** The
  builder writes complete or fetch-referenced bags directly; no
  `asset_mode` enum needed at builder time. `asset_mode` only
  matters when *moving* assets between catalogs (i.e., on the
  loader side).
- **`CatalogDataSource` has no live users in deriva-ml.** It is
  moving upstream nonetheless because it is the natural mirror of
  `BagDataSource` and `BagCatalogLoader`'s reverse-direction
  sibling.

## Alternatives considered

- **Keep three independent code paths; ship producer-side
  improvements piecemeal.** Rejected: every new feature pays the
  three-times tax. The current divergence between `CatalogGraph`
  and `clone._discover_reachable_tables` is exactly the kind of
  drift that compounds.
- **Make `BagBuilder` the universal bag-writing primitive;
  `CatalogBagBuilder` feeds it row by row.** Rejected: would
  bypass the deriva-py export engine's paged streaming, async
  queries, MD5 manifesting, S3 upload, and MINID post-processors.
  Re-implementing those inside the producer is a regression.
- **Promote `BagDatabase` upstream but keep the producer side in
  deriva-ml.** Rejected: splits the bag contract across two repos.
  The producer would import the profile from deriva-py while
  living in deriva-ml — a coupling point that buys nothing.
- **Build the new module in deriva-ml first, lift to deriva-py
  later.** Rejected for this redesign: the consumer half
  (`BagDatabase`) already lives upstream. Building the producer
  half in a different repo would re-introduce the split that
  motivates the redesign.

## Delivery sequence

Delivery is two PRs, sequenced:

1. **deriva-py PR against `deriva-ml` branch.** New
   `deriva/bag/` submodule with:
   - `sqlite_helpers.py` (or `engine.py`) — SQLite engine factory
     with WAL + `synchronous=NORMAL` + `foreign_keys=ON` +
     `busy_timeout=5000`, read-only URI support, and
     `ensure_schema_meta` for forward-compat schema versioning.
     Lifted from deriva-ml's `local_db/sqlite_helpers.py`.
   - `profile.py` — profile helpers (path constants, archive-as-zip,
     `metadata/` writer, `BagIt-Profile-Identifier` constant,
     validation wrapper around `bdb.validate_bag_profile`,
     `BAG_SCHEMA_VERSION` constant).
   - `profiles/deriva-bag-profile.json` — the profile JSON document
     (BagIt Profile spec). Hosted under the deriva-py repo; the
     identifier URL points at the raw file on the `deriva-ml`
     branch (e.g.,
     `https://raw.githubusercontent.com/informatics-isi-edu/deriva-py/deriva-ml/deriva/bag/profiles/deriva-bag-profile.json`)
     until it stabilizes and a versioned URL is minted.
   - `database.py` — `BagDatabase` (refactored from
     `deriva/core/bag_database.py`; switched to `create_wal_engine`
     and `ensure_schema_meta`).
   - `schema.py` — `SchemaBuilder` / `SchemaORM`.
   - `schema_io.py` — schema-interchange functions:
     `typed_schema_def_to_metadata`,
     `metadata_to_typed_schema_defs`,
     `metadata_to_ermrest_json`, `ermrest_json_to_metadata`,
     `ermrest_model_to_metadata`, and the ERMrest ↔ SQLAlchemy
     type-mapping tables. Single source of truth for schema
     interchange between deriva-py `typed` defs, ERMrest JSON,
     SQLAlchemy `MetaData`, and ERMrest `Model`.
   - `loader.py` — `DataLoader` + `ForeignKeyOrderer` + `CSVSink`
     (write-out sink used by `BagBuilder`; complements the
     existing into-SQLite sink used by `BagDatabase` on read).
   - `sources.py` — `DataSource` + `BagDataSource` +
     `CatalogDataSource` + `DataFrameDataSource` +
     `IterableDataSource` + `LocalDBDataSource`. The last three
     are inputs to `BagBuilder` for the three producer cases
     (in-memory pandas, in-memory dict iterables, local SQLite
     from `local_db`).
   - `builder.py` — `BagBuilder`.
   - `catalog_builder.py` — `CatalogBagBuilder`.
   - `catalog_loader.py` — `BagCatalogLoader`.
   - `anchors.py` — `Anchor` union.
   - `traversal.py` — `FKTraversalPolicy`.
   - `cache_index.py` — `BagCacheIndex` (SQLite-backed cache
     index: `bags` and `bag_anchor_rids` tables, content-addressed
     lookup, reverse-index queries by anchor RID). Replaces the
     `{rid}_*` directory-glob affordance.

   Same PR deletes `deriva/core/bag_database.py`. Designed for
   clean cherry-pick into the deriva-py `2.0` branch later.

2. **deriva-ml migration PR.** Pin deriva-py to the commit on
   `deriva-ml` that contains the new module. Replace local
   `model/schema_builder.py`, `model/data_loader.py`,
   `model/data_sources.py`, `model/fk_orderer.py` with imports
   from `deriva.bag`. Replace `local_db/sqlite_helpers.py` with a
   re-export of `deriva.bag.sqlite_helpers` (or remove and update
   `local_db` callers to import upstream directly). Rebuild
   `DatabaseModel` as a `BagDatabase` subclass. Rename
   `DerivaMLDatabase` → `DerivaMLBagView`. Update
   `local_db/schema.py` imports. Replace `catalog/clone.py` body
   with the `CatalogBagBuilder + BagCatalogLoader` wrapper.
   Rewrite `load_cifar10.py` to use `BagBuilder +
   BagCatalogLoader`. Add `DatasetBagBuilder` wrapper that layers
   dataset-specific filtering on `CatalogBagBuilder`. Update
   `BagCache`'s `CacheStatus` enum to use `cached_holey` instead
   of `cached_incomplete` (matching the bag-terminology decisions
   in CONTEXT.md); convert `CacheStatus(str, Enum)` to
   `CacheStatus(StrEnum)` for consistency with the project's
   established enum style. Rewrite `BagCache` to use the new
   content-addressed cache index (`index.sqlite`) instead of the
   `{rid}_*` directory glob. Include a one-shot migration helper
   that scans any existing `{cache_dir}/{rid}_{checksum}/...`
   directories, populates `index.sqlite`, and moves them into the
   new `bags/{checksum}/` layout — cached bags do not need to be
   re-downloaded.
   Rewrite `Execution.commit_execution` to use the bag pipeline:
   `BagBuilder(metadata=local_db_metadata,
   source=LocalDBDataSource(local_db))` to build a bag from the
   working SQLite, then `BagCatalogLoader(catalog, bag,
   policy).run()` to push it. Replaces today's bespoke
   `upload_directory()` + ad-hoc row-insert path with the same
   two-step flow used by cifar load and catalog clone. The bag is
   left on disk after the upload as a durable execution artifact
   (citable, inspectable, re-loadable on failure).

A subsequent ADR (0007) may slice the deriva-py PR into sub-PRs
if it grows unreviewable; for now the assumption is one PR.

### Commit-stack inside the deriva-py PR

The deriva-py PR is **one PR with multiple logically coherent
commits** rather than one squash. Each commit leaves the tree in
a working state, so individual commits can be cherry-picked into
the `2.0` branch later. Proposed boundaries:

| # | Commit | Tree state |
| --- | --- | --- |
| 1 | Add `deriva/bag/__init__.py`, `sqlite_helpers.py`, `profile.py`, `profiles/deriva-bag-profile.json` | Empty package + engine factory + profile artifact |
| 2 | Move `deriva/core/bag_database.py` → `deriva/bag/database.py`; switch to `create_wal_engine` + `ensure_schema_meta` | `BagDatabase` lives at new path with WAL; old path still works via re-export shim |
| 3 | Add `deriva/bag/schema.py`, `schema_io.py`, `loader.py` (with `CSVSink`), `sources.py` (with `DataFrameDataSource`, `IterableDataSource`, `LocalDBDataSource`) — primitives lifted from deriva-ml plus the new schema-interchange, write-out sink, and producer-side data sources | Primitives + schema-interchange + write-out sink + producer sources upstream; deriva-ml will switch imports in migration PR |
| 4 | Add `deriva/bag/anchors.py`, `traversal.py` | Shared types ready |
| 5 | Add `deriva/bag/cache_index.py` (`BagCacheIndex`) | Content-addressed index ready |
| 6 | Add `deriva/bag/builder.py` (`BagBuilder`) | Constructive producer ready |
| 7 | Add `deriva/bag/catalog_builder.py` (`CatalogBagBuilder`) | Walking producer ready |
| 8 | Add `deriva/bag/catalog_loader.py` (`BagCatalogLoader`) | Loader ready |

The shim at `deriva/core/bag_database.py` (re-exporting from the
new path) stays in place through the deriva-py PR and gets
removed in a follow-up commit on the `deriva-ml` branch after
the deriva-ml migration PR has landed. That removal is **not**
part of this PR — keeps the deriva-py PR self-contained and
non-breaking for any deriva-ml code that still imports the old
path during migration.

Move-only commits (commits 2 and 3) stay pure renames where
possible: the diff is short, easy to audit, and obviously
behavior-preserving. Behavior changes (WAL adoption, schema_meta
addition) are isolated into the same commit as the move but
called out in the commit message.

## References

- ADR-0001: Lineage walks data flow, not orchestration.
- ADR-0003: Dataset dev-versioning model.
- CONTEXT.md "Bag producer/consumer model" section.
- `src/deriva_ml/dataset/catalog_graph.py` — the dataset-side
  walker `CatalogBagBuilder` generalizes.
- `src/deriva_ml/catalog/clone.py` — the catalog-side walker
  `CatalogBagBuilder` + `BagCatalogLoader` replace.
- `src/deriva_ml/model/*` — the consumer-side primitives moving
  upstream.
- `deriva-ml-model-template/src/scripts/load_cifar10.py` — the
  constructive-producer case driving `BagBuilder`'s design.

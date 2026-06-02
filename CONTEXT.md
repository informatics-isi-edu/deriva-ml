# DerivaML Domain Context

DerivaML is a reproducible-ML layer over Deriva catalogs. This file
captures the vocabulary specific to that layer — terms whose meaning
is project-specific and not derivable from the code or from generic
Deriva concepts.

## Language

### Versioning

**Released version**:
A `Dataset_Version` row with a stamped catalog `Snapshot`, an
immutable PEP 440 label of the form `MAJOR.MINOR.PATCH`. The
canonical reference for an execution that consumed the dataset.
_Avoid_: "real version", "pinned version", "frozen version".

**Dev version**:
A `Dataset_Version` row with `Snapshot=NULL` and a PEP 440 label of
the form `<last_release>.post1.devN`. Represents drift since the
last release; never used as the recorded version of an execution's
consumed dataset. Mutable: `.devN` advances and `Description` is
replaced on each mutation. A dev label resolves *only when it
matches the dataset's current dev row's `Version`* — the row is the
same row whose label keeps changing, so `.dev2` is addressable
only at the moment it is current. Stale or post-release dev labels
error.
_Avoid_: "pre-release version", "draft version", "dirty version" (as
a label — "dirty" is a state of the dataset, not a kind of version).

**Dev period**:
The span from a dataset's first mutation after a release to the next
`release()` call. During a dev period, the dataset has exactly one
dev row; it is updated, not duplicated, on each mutation.

**Drift**:
A change to any catalog row reachable from a dataset's `Dataset` row
via the FK paths that `CatalogGraph` enumerates for export. Drift
happens whether or not the dataset row itself was touched — adding a
feature value to a member is drift.
_Avoid_: "modification", "update" (too generic).

**Dirty**:
A property of a dataset, not of a version: "this dataset has drift
since its last released version." Detected via
`Dataset.is_dirty()` by walking `CatalogGraph`'s FK paths with an
`RMT > T_release` filter. The drift walk *is* the bag walk plus an
`RMT` filter.
_Avoid_: "modified", "stale", "out-of-date".

**Mark dev**:
The user action `Dataset.mark_dev(description)` that flips a
released dataset to a dev version. Used when drift was caused by an
operation that didn't auto-flip (e.g., a feature value added by a
separate execution, or a deletion not visible to `is_dirty`).

**Release**:
The user action `Dataset.release(component, description)` that
promotes a dev row to a released row in place: `Version` is rewritten
to the new released label, `Snapshot` is stamped, `Description` is
replaced with release notes. The only operation that produces a
released version.
_Avoid_: "publish", "tag", "freeze" (overloaded with other Deriva
concepts).

### Datasets — types and partitions

Dataset_Type vocabulary terms have three orthogonal axes. A single
dataset carries one or more tags from any combination of them, and
the axes mean different things to readers of the catalog.

**Role-axis Dataset_Type**:
A tag that says what this dataset is *for* in its immediate
context. Includes `Training`, `Testing`, `Validation`, and `Split`
(the parent of a split hierarchy). Role-axis types are **not
inherited from a parent dataset** and are **not propagated to
child datasets**. `split_dataset` assigns the partition roles
(Training / Testing / Validation) to its children based on the
partition's position in the split, regardless of what the source
dataset's role was.
_Avoid_: "kind" (too generic), "purpose" (a dataset's purpose is
in its description, not its type tag).

**Content-axis Dataset_Type**:
A tag that says what *kind of data* the dataset contains. Includes
`Labeled`, `Unlabeled`, and domain-specific types like `Complete`,
`CIFAR_10`, `Eye_Image_Fundus`. Content-axis types describe
properties of the data itself and **may propagate** when the
partitioning operation preserves them (a stratified sample of a
Labeled dataset is still Labeled). Whether they propagate is a
caller decision, expressed via `training_types=`, `testing_types=`,
`validation_types=` arguments to `split_dataset` and the
`dataset_types=` argument to `subsample`.
_Avoid_: "data kind", "content kind" (vague).

**Origin-axis Dataset_Type**:
A tag that says how this dataset *came to exist*. Includes `Split`
(parent dataset of a split hierarchy), `Split_Partition` (a child
of a Split — Training, Testing, or Validation), and `Subsample`
(a dataset produced by `subsample()`). Origin tags are
**never inherited**. They are always set by the producing
operation, never copied from another dataset.
_Avoid_: "lineage tag" (lineage lives in execution edges, not in
type tags; the origin tag is a denormalized signal, not the
truth).

**Split**:
A parent dataset produced by `split_dataset` that contains the
Training, Testing, and optional Validation children as
`Dataset_Dataset` members. Carries `Dataset_Type=["Split"]` only
(not `Split_Partition` — it is the container, not a partition).
_Avoid_: "split parent" (Split *is* the parent — the modifier is
redundant), "split dataset" as a phrase (ambiguous with the
function name `split_dataset`).

**Split_Partition**:
The origin-axis tag attached to every child of a Split — the
Training, Testing, and Validation children of `split_dataset`'s
output. The discriminator that distinguishes a corpus-role
`Training` dataset (a dataset tagged `Training` because it is a
training corpus) from a partition-role `Training` dataset (a
dataset tagged `Training` because it is the training partition of
a split). Hand-built Split hierarchies should also tag children
with `Split_Partition` to remain discoverable through the same
filters.
_Avoid_: "split child", "split result", "split member" (Split has
`Dataset_Dataset` members, but those members are tagged
`Split_Partition`, not "Split_Member").

**Subsample**:
The origin-axis tag attached to the single dataset output of the
`subsample()` operation. Distinguishes a subsampled dataset from a
hand-curated dataset of the same role and content. The source
relationship lives in the execution graph (the source is an input
of the subsample's producing execution), not in `Dataset_Dataset`
edges — there is no parent/child hierarchy between a subsample and
its source. Same provenance shape as `split_dataset` outputs.
_Avoid_: "subsampled dataset" (the `Subsample` *is* the dataset),
"sample" (too generic, conflates with the `subsample()` operation
itself).

**Partition unit**:
The granularity at which `split_dataset` and `subsample`
distribute rows across partitions. Controlled by the `partition_by`
parameter and constrained by `element_table` plus `row_per`. Two
values: `"element"` dedupes the denormalized dataframe to one row
per `element_table` RID before partitioning and asserts
element-RID disjointness after; `"row"` partitions denormalized
rows directly and may legitimately place the same element RID in
multiple partitions.
_Avoid_: "granularity", "split unit" (vague).

**Source dataset**:
The dataset passed as `source_dataset_rid` to `split_dataset` or
`subsample`. The split/subsample operation does NOT create a
`Dataset_Dataset` edge from the source to its outputs; instead,
the source is recorded as an *input of the producing execution*
via the existing `Dataset_Execution` association. The derivation
relationship is therefore reached via
`output.producing_execution.list_input_datasets()`, not via
`output.list_dataset_parents()`. This matches ADR-0001 (lineage
walks data flow, not orchestration).
_Avoid_: "parent dataset" (the source is NOT a Dataset_Dataset
parent of the output), "input dataset" outside of execution
context (sources become "inputs" only in the producing
execution's frame).

### Bags

**Deriva-bag profile**:
The structural contract `deriva.bag` imposes on a BDBag carrying
catalog content. Slots into BDBag's existing **BagIt Profile**
mechanism (a JSON document validatable by `bdb.validate_bag_profile`
and declared via the `BagIt-Profile-Identifier` field in
`bag-info.txt`). The profile constrains:

- `data/schema.json` — ERMrest model description.
- `data/<schema>/<table>.csv` — table data, one CSV per table.
- `data/asset/<asset_table>/<RID>/<filename>` — asset bytes (when
  embedded) or referenced via `fetch.txt`.
- Optional `metadata/` for provenance (`Anchor`,
  `FKTraversalPolicy`, build timestamps).

The profile JSON is authored and hosted by deriva-py (under
`deriva/bag/profiles/`); the identifier URL is fixed at the time the
profile is registered. `BagBuilder` and `CatalogBagBuilder` both
write the identifier into `bag-info.txt`; `BagDatabase` and
`BagCatalogLoader` may validate before reading.
_Avoid_: "bag profile" without the "deriva-bag" qualifier (collides
with the BDBag-spec sense of "BagIt Profile"); "bag format", "bag
spec" (too generic).

**Bag directory**:
An unpacked bag — a filesystem tree conforming to the BDBag
specification plus the deriva-bag profile. `bagit.txt`, `manifest-*.txt`,
`data/`, optional `fetch.txt`. The unit `BagReader` operates on.
_Avoid_: "bag folder", "expanded bag".

**Bag archive**:
A bag directory packaged as a single file (zip, by convention). What
`download_dataset_bag` writes to the cache and what an upload reads.
_Avoid_: "bag zip", "bag tarball" (we standardize on zip).

**Fetch.txt**:
The BDBag manifest listing remote files (URL, length, path, hash)
that are *referenced but not yet present* in the bag. Always
written via `bdb.make_bag(remote_file_manifest=...)`, never by hand
(see `deriva_ml/dataset/dataset.py` and the "BDBag Remote File
Manifest" note in CLAUDE.md).

**Holey bag** / **Materialized bag**:
A bag with a non-empty `fetch.txt` whose referenced files are not
present on disk is *holey*; once `bdb.materialize()` has fetched
them, it is *materialized*. The BDBag specification calls these
"incomplete" and "complete"; we use the project terms instead.
_Avoid_: "incomplete bag", "complete bag" (the BDBag-spec terms —
we use "holey" / "materialized" in conversation and code so the
state is unambiguous), "partial bag", "lazy bag", "hydrated bag".

**Materialize** (verb):
Fetch the files listed in `fetch.txt` so a holey bag becomes
materialized. Always used as a verb on a bag directory.
_Avoid_: "hydrate", "download" (ambiguous — "download" is the
producer-side operation that produces the archive in the first
place).

**MINID** / **MINID handle**:
A Minimal Viable Identifier — a citable URL that resolves to a bag
archive's location and checksum. A `DatasetMinid` wraps a MINID plus
the parsed metadata needed to locate the archive in the bag cache.
For released versions, a MINID is the canonical reference; for dev
versions, MINIDs are not issued (the catalog can drift between
calls), and `use_minid=True` is rejected with a clear error.
_Avoid_: using "MINID" loosely for "any bag reference"; reserve it
for the actual minted identifier.

**Bag cache**:
The local on-disk store of bag archives and their materialized
directories, keyed by MINID for released versions and *not used* for
dev versions. Lives under the working directory configured for the
`DerivaML` instance.

**Cache state**:
The status of a particular bag in the cache. One of:
- *not_cached* — no archive on disk.
- *cached_metadata_only* — `DatasetMinid` known, archive not
  fetched.
- *cached_holey* — archive on disk, expanded directory present,
  `fetch.txt` references not yet materialized.
- *cached_materialized* — archive on disk, expanded directory
  present, all referenced files present.

**Bag SQLAlchemy**:
The SQLAlchemy layer derived from a bag directory by loading
`schema.json` into a `SchemaORM` and populating tables from
`data/<schema>/<table>.csv`. Physically it is one main SQLite file
plus one attached SQLite file per ERMrest schema (so a bag with
schemas `deriva-ml` and `domain` becomes `main.db` + `deriva-ml.db`
+ `domain.db`); SQLAlchemy hides the multi-file layout behind a
single engine, metadata, and ORM `Base`. The user-facing API for
reading bag contents — the file-system layout (both the bag's CSVs
and the underlying `.db` files) is an implementation detail behind
this SQLAlchemy layer.

Built using the same SQLite engine policy as the `local_db`
subsystem: WAL journal mode, `synchronous=NORMAL`,
`foreign_keys=ON`, `busy_timeout=5000`, and read-only access via
the SQLite `mode=ro&uri=true` URI form. The shared factory is
`deriva.bag.sqlite_helpers.create_wal_engine` (lifted from
deriva-ml's `local_db/sqlite_helpers.py`). A `schema_meta` table
with a `BAG_SCHEMA_VERSION` constant guards against version drift
between the on-disk layout and the running code.

On-disk layout is **content-addressed** by the BDBag checksum,
not by any anchoring RID. Bags can be rooted in zero, one, or
many RIDs (`RIDAnchor` lists, `TableAnchor("*")`, `PathAnchor`,
catalog-wide clones), so a single-RID layout
(`{rid}_{checksum}/...`) is no longer adequate. The layout:

```
{cache_dir}/
    index.sqlite                          # cache index (per host/catalog)
    bags/
        {checksum}/
            bag/                          # the bag directory
            db/main.db                    # Bag SQLAlchemy main file
            db/{schema}.db                # one per attached ERMrest schema
```

Identical-content bags (same closure under different anchor
configurations) share one cache entry automatically.

The **cache index** (`index.sqlite`, one per host/catalog under
`cache_dir`) carries:
- `bags(checksum, profile_id, built_at, anchor_summary_json, size_bytes)`
  — one row per cached bag.
- `bag_anchor_rids(checksum, table, rid)` — flat reverse index so
  callers can ask "what bags have I built that include
  `(Subject, S1)` as an anchor?" in one query. Required because
  the directory glob `{rid}_*` that today's `BagCache` relies on
  no longer works.

The index itself uses `create_wal_engine` and the same
`schema_meta` versioning as everything else in `deriva.bag`.
_Avoid_: "Bag SQLite" (technically inaccurate — SQLite alone has
no schema concept, so the layer is SQLAlchemy-with-multiple-
SQLite-files, not "a SQLite database"); "bag DB", "local catalog"
(the second is misleading; it is not an ERMrest catalog).

### Bag producer/consumer model

The bag is the unifying artifact: producers write bags from
catalogs or in-memory data; consumers read bags as SQLAlchemy. The
*deriva-bag profile* (the file-system layout the producer writes
and the consumer reads, registered as a BagIt Profile) is the
contract that holds the two sides apart.

All bag machinery lives in **`deriva.bag`** (the deriva-py submodule
on the `deriva-ml` branch). Producers and consumers there are
deriva-ml-agnostic; deriva-ml adds a domain layer on top
(`DatabaseModel`, `DerivaMLBagView`, `DatasetBag`).

The profile (defined under "Language → Deriva-bag profile" above)
is what makes round-trip closed: `catalog → bag → catalog`
produces an equivalent destination catalog (modulo orphan-policy
outcomes). It is registered as a BagIt Profile so any
BDBag-compliant tool can validate conformance.

### Producer side

**`BagBuilder`** (`deriva.bag.builder`):
The *constructive* producer. Caller supplies a schema description
(either deriva-py `typed.SchemaDef`s or a SQLAlchemy `MetaData`,
or both) and a `DataSource` (or calls
`add_row`/`add_rows`/`add_asset` imperatively). No source
catalog. Serves two distinct input shapes:

- **Programmatic build** (cifar / Kaggle / TF-dataset cases) —
  caller has data in memory as DataFrames or dict iterables.
  Uses `DataFrameDataSource` or `IterableDataSource`.
- **End-of-execution upload** — `Execution.commit_execution`
  builds a bag from the working `local_db` SQLite using
  `LocalDBDataSource`. The same bag is then handed to
  `BagCatalogLoader` to push to the catalog. Replaces today's
  bespoke `upload_directory()` flow; the bag becomes a durable
  artifact of every execution.

Methods: `add_row`, `add_rows` (accepts a pandas DataFrame, a
list of dicts, or any iterable of dicts), `add_asset`,
`add_assets`, `add_asset_reference`, `finalize`. Writes a
materialized bag from local data (or a holey bag if
`add_asset_reference` is used).

Two schema-input vocabularies are accepted because deriva-py
users already have one (`typed.SchemaDef` — the same defs they
use to push schemas to the catalog) and SQLAlchemy users have
another (`MetaData` — the same object `BagDatabase` returns and
that `local_db` manipulates). Accepting both means a deriva-py
user can declare the schema once and use it for both the catalog
push and the bag construction; a non-deriva-py user can bring an
existing `MetaData` from another tool.

Internally `BagBuilder` normalizes to `MetaData` because that is
the vocabulary the consumer side (`BagDatabase`,
`BagCatalogLoader`, `local_db`) speaks. `CatalogBagBuilder` also
internally projects its source catalog's ERMrest `Model` to a
`MetaData` slice via
`deriva.bag.schema_io.ermrest_model_to_metadata`. Conversion
functions live in `deriva.bag.schema_io`:
- `typed_schema_def_to_metadata` / `metadata_to_typed_schema_defs`
- `metadata_to_ermrest_json` / `ermrest_json_to_metadata`
- `ermrest_model_to_metadata`

The hub is `MetaData`; every other representation has a
round-trip path through it.

**`CatalogBagBuilder`** (`deriva.bag.catalog_builder`):
The *catalog-walking* producer. Generates an export spec from a
`Anchor` list and an `FKTraversalPolicy`, then hands the spec to
deriva-py's existing export engine. The engine writes the bag —
we do not reimplement paged streaming, async fetching, MD5
manifest generation, S3 upload, or MINID post-processors. The
catalog builder is purely a spec generator + driver.

**`BagCatalogLoader`** (`deriva.bag.catalog_loader`):
The *bag → catalog* writer. Walks the Bag SQLAlchemy layer (via
`BagDatabase`) and inserts rows into a destination ERMrest catalog
in FK-safe order, with asset upload mode controlled by
`FKTraversalPolicy.asset_mode`. Replaces clone's stage-2 data-copy +
stage-3 FK-application logic; clone becomes
`CatalogBagBuilder` → `BagCatalogLoader`.

**`Anchor`** (`deriva.bag.anchors`):
Discriminated union — `RIDAnchor`, `TableAnchor`, `PathAnchor` —
that describes "where to start walking" for `CatalogBagBuilder`.
Serializable into the bag's `metadata/` for provenance.

Resolution semantics:
- **Overlap is silently deduped.** Multiple anchors that name
  overlapping rows (e.g., `RIDAnchor("Subject", ["S1"])` +
  `TableAnchor("Subject")`) get unioned; the FK walk's visited-set
  handles the rest. No warning. Anchors describe intent ("start
  from here"); overlap doesn't change what gets built.
- **`RIDAnchor` fails fast on missing RIDs.** Each `RIDAnchor` is
  resolved against the source catalog at validation time (one
  `?RID=in(...)` query per anchor table). If any RID is absent,
  raise with the list of missing RIDs before the walk starts.
  Almost always a user error (typo, stale config, deleted row);
  silently dropping unfound RIDs produces a bag the user didn't
  intend.
- **`TableAnchor("X")` is silently fine when empty.** A table with
  no rows in scope is a legitimate "no rows from this table"
  outcome.
- **`PathAnchor` warns but proceeds when empty.** A datapath that
  resolves to no rows might mean the user's filter excluded
  everything (legitimate) or that the filter is wrong (likely
  error). Log the empty result; let the build continue.

_Avoid_: "seed" (collides with random-number seed in ML
conversations), "root" (overloaded with `root_rid` / `root_table`
already used in `clone.py`).

**`FKTraversalPolicy`** (`deriva.bag.traversal`):
Foreign-key graph traversal policy. Seven fields, all with
sensible defaults so `FKTraversalPolicy()` (no arguments) is a
working value for every producer case identified so far:

```python
from enum import StrEnum

class VocabExport(StrEnum):
    REFERENCED_ONLY = "referenced_only"
    FULL = "full"

class AssetMode(StrEnum):
    ROWS_ONLY = "rows_only"
    UPLOAD_IF_MISSING = "upload_if_missing"
    UPLOAD_FORCE = "upload_force"

class DanglingFKStrategy(StrEnum):
    FAIL = "fail"
    DELETE = "delete"
    NULLIFY = "nullify"

class FKTraversalPolicy(BaseModel):
    # Scope
    schemas: set[str] | None = None
    exclude_schemas: set[str] = {"public", "_acl_admin", "WWW"}
    exclude_tables: set[tuple[str, str]] = set()
    # Bounds
    max_depth: int | None = None
    # Vocabulary
    vocab_export: VocabExport = VocabExport.REFERENCED_ONLY
    # Asset handling (loader-side)
    asset_mode: AssetMode = AssetMode.UPLOAD_IF_MISSING
    # Orphan handling (loader-side)
    dangling_fk_strategy: DanglingFKStrategy = DanglingFKStrategy.FAIL
```

All closed-set fields are `StrEnum`s, matching the convention
already established in `deriva_ml.core.enums`,
`deriva_ml.execution.state_store`, and other recent modules.
String values preserve the readable JSON for the bag's
`metadata/` provenance; enum members give static analysis, IDE
completion, single-source-of-truth values, and clear errors on
deserialization from a stale `metadata/` file. Python ≥3.11
makes `StrEnum` available; deriva-ml's floor is ≥3.12 so it's
always present.

The single contract shared between catalog builder (walk catalog
→ bag) and catalog loader (walk bag → catalog). Replaces clone's
old `OrphanStrategy`/`AssetCopyMode` enums + `CatalogGraph`'s
exclude-tables knob with one policy object.

**Walks are always bidirectional.** The policy intentionally
omits a `direction` field. Every real producer case (dataset
bag, catalog-wide clone, RID-anchored slice, path-anchored
slice) wants both inbound and outbound FK following: outbound
pulls a row's referenced parents; inbound pulls its children.
A single-direction walk is theoretically expressible but no
caller has been identified that wants it. If a use case surfaces
later, add a `direction` field then.

**Vocabulary traversal is a hardcoded walker rule**, not a
policy field. The bidirectional default would loop back through
vocab tables to every row that references them — Subject →
Species (outbound) → every other Subject that uses Species
(inbound) → explosion. The walker prevents this by following
FKs *into* vocab tables (pulling in referenced terms) but not
*out of* them. The rule applies in every producer case;
exposing it as a policy field would be a field with one sensible
value. If a future case needs different behavior, add the field
then.

`vocab_export=VocabExport.FULL` does carry real decision content
— it adds a separate post-walk pass that copies *every* term
(including unused ones), for cases like dataset bags where
downstream consumers need the complete vocabulary. That field
stays.

The policy is deliberately minimal. Four things it does *not*
have:
- *No `force_include`* — "always include this table" duplicates
  either `TableAnchor` (every row) or "let the walk reach it via
  FK-following" (rows reachable from another anchor). Domain
  wrappers that need extra tables construct appropriate
  `Anchor`s, not a separate field.
- *No `direction`* — every real producer wants bidirectional
  walking; the field would have one value.
- *No `vocab_direction`* — vocab tables are always inbound-only;
  the hardcoded walker rule covers the same ground.
- *No row-level filters* — filtering happens at anchor resolution
  (e.g., `PathAnchor` with a datapath predicate). The traversal
  policy operates on the graph at the table level.
- *No domain-specific knobs* — concerns like "skip empty
  associations for dataset members" live in domain wrappers
  (`DatasetBagBuilder`), not in the generic policy.

**Table-kind handling in the generic walker**:
The walker recognizes three structural table kinds via methods
that already exist on ERMrest `Table` (no deriva-ml-specific
knowledge). Each kind gets a hardcoded rule, applied uniformly
regardless of whether the walk anchors at a Dataset row, a
Subject row, or every row in every table.

- *Vocabulary tables* (`Table.is_vocabulary()`). The walker
  follows FKs *into* vocab tables (pulling in the referenced
  terms) but does not follow FKs back out — preventing the
  loop-back to every other content table that uses that vocab.
  This is hardcoded; there is no `vocab_direction` policy field.
  After the main walk, if
  `vocab_export=VocabExport.FULL`, run a second pass that copies
  every row from each reachable vocab table — used by dataset
  bags so downstream consumers see the complete vocabulary.
- *Asset tables* (`Table.is_asset()`). Walked like any other
  table; recognized so the walker emits the asset directory
  layout (`data/asset/<table>/<RID>/<filename>`) and the
  `fetch.txt` entries. The loader's `asset_mode` decides what to
  do with them at write time.
- *Association tables* (`Table.find_associations()` arity rules
  — same logic as `BagDatabase.is_association_table`). Tag-along
  rows: included automatically when both endpoints are in the
  bag, and they don't count toward `max_depth`.

These rules live in the generic walker because all three table
kinds need consistent treatment in every producer case (dataset
bag, clone, slice). Putting them in the walker is what lets
`DatasetBagBuilder` stay a thin wrapper rather than a place
where domain-specific structural handling re-emerges.

**`asset_mode`** (field on `FKTraversalPolicy`):
Three-valued enum controlling how `BagCatalogLoader` handles
asset tables that are in scope for traversal. All modes delegate
per-asset work to deriva-py's existing `DerivaUpload._uploadAsset`
recipe (which already does Hatrac byte-dedupe via HEAD + MD5,
catalog row reconciliation, and pre-allocated RID handling):
- *`ROWS_ONLY`* — asset rows inserted/reconciled with the bag's
  URL column unchanged. No bytes transferred. Destination depends
  on the source's Hatrac being reachable.
- *`UPLOAD_IF_MISSING`* (default) — rows inserted/reconciled; for
  each asset, Hatrac is HEAD-checked and bytes are uploaded only
  when the destination's MD5 differs. Row metadata is reconciled
  column-by-column regardless of whether bytes moved. This is the
  right semantic for round-trip clone, re-import, and any
  idempotent re-run.
- *`UPLOAD_FORCE`* — rows inserted/reconciled; bytes re-uploaded
  unconditionally (`force=True` to Hatrac). Bypasses dedupe.
  Reserved for "source bytes are authoritative; overwrite" scenarios.

The "don't include this asset table at all" case is handled by
`exclude_tables` (or by `schemas`/`exclude_schemas`) — not by an
`asset_mode` value. Keeping scope and behavior in separate fields
avoids the "I included X but set mode to SKIP — which wins?"
ambiguity.

Two `(bag_state, mode)` combinations are rejected at policy
validation time: *holey* bag + `UPLOAD_IF_MISSING` and *holey* +
`UPLOAD_FORCE`. Both require local bytes the holey bag does not
have. The rejection message points at `bdb.materialize(bag)` or
at `ROWS_ONLY`.
_Avoid_: "NONE", "REFERENCES", "FULL" (the old clone.py names —
too binary; they miss the dedupe semantics that
`UPLOAD_IF_MISSING` makes explicit); "SKIP" (overlaps with
`exclude_tables`).

**`dangling_fk_strategy`** (field on `FKTraversalPolicy`):
How `BagCatalogLoader` resolves *orphan rows* — rows in the bag
whose FK references point at parents not included in the bag.
Three values:
- *`FAIL`* (default) — abort the load if any orphans are detected.
  Conservative; the caller adjusts scope or fixes source data.
- *`DELETE`* — drop orphan rows entirely. The destination ends up
  with fewer rows than the bag.
- *`NULLIFY`* — set the dangling FK column to `NULL` (only legal
  when the column is nullable). The row stays; only the reference
  disappears.

Orphans arise because of *incoherent row-level policies* in
source catalogs: a row is visible to the cloning user but its
FK-referenced parent is not. Without orphan handling, the
destination either rejects the row outright (FK violation) or is
left with broken FKs.

Detection happens at *bag-read time* in `BagDatabase`: the
SQLAlchemy ORM can cheaply compute the set of dangling FK
references with no catalog round-trips. The list is exposed via
e.g. `bag.orphans`. Application happens at *load time* in
`BagCatalogLoader`, which consumes the list and acts according to
`dangling_fk_strategy`. `CatalogBagBuilder` itself is orphan-agnostic
— it writes the walk's output as-is.
_Avoid_: applying the strategy at build time (changes the bag's
contents based on a destination concern); adding `INCLUDE_PARENT`
as a strategy value (that would be a build-policy choice, not an
orphan-resolution choice).

**`DatasetBagBuilder`** (deriva-ml,
`src/deriva_ml/dataset/bag_builder.py`):
Deriva-ml-domain wrapper around `CatalogBagBuilder`. Layers
dataset-specific scope refinements onto the generic
`Anchor` + `FKTraversalPolicy` mechanism:

- Builds the anchor list from the dataset's row plus its nested
  children (recursively to a `nesting_depth`).
- Builds an `FKTraversalPolicy` with `exclude_tables` covering
  association tables for element types with no members and
  `vocab_export=VocabExport.FULL` (the dataset bag wants the complete
  vocabulary, not just terms referenced by member rows). Feature
  tables for member element types are reached naturally by
  FK-following from the element rows. Vocabulary handling
  (inbound-only direction, optional full export) is built into
  the generic policy — `DatasetBagBuilder` doesn't run its own
  vocabulary pass anymore.
- Exports vocabularies in a separate standalone pass after the
  main walk (matching today's `CatalogGraph._export_vocabulary`
  behavior).

After the redesign, `Dataset.download_dataset_bag` becomes a thin
wrapper that constructs a `DatasetBagBuilder`, calls `build()`,
caches the bag, and returns a `DatasetMinid`. The public
`download_dataset_bag` signature does not change.

### Consumer side

**`BagDatabase`** (`deriva.bag.database`):
The generic consumer — opens a bag directory and returns the Bag
SQLAlchemy layer (engine, metadata, ORM `Base`) plus asset-access
helpers. Knows nothing about
datasets, ML schemas, or executions. Already exists in deriva-py's
`deriva-ml` branch under `deriva.core.bag_database`; will move to
`deriva.bag.database` as part of the B3 migration.

**`DatabaseModel`** (deriva-ml, `model/database.py`):
deriva-ml extension of `BagDatabase`. Adds dataset-version tracking
(`bag_rids`, `dataset_version`, `rid_lookup`,
`_get_dataset_execution`). After B3 migration: ~80 lines of
deriva-ml-specific behavior instead of the current ~400 (the rest
is delegation that becomes inheritance).

**`DerivaMLBagView`** (deriva-ml, planned rename of
`DerivaMLDatabase`):
The deriva-ml-domain layer over a `DatabaseModel`. Knows about the
`deriva-ml` schema — datasets, vocabularies, features, element
types. Factory for `DatasetBag` objects when the bag carries
dataset-shaped content.

**`DatasetBag`** (deriva-ml):
A per-dataset view over a `DerivaMLBagView` — bound to a single
dataset RID inside the bag. Implements `DatasetLike` so user code
that consumes a `Dataset` (catalog) and a `DatasetBag` (bag) reads
the same way.

## Relationships

- A **Dataset** has at most one **Dev version** at any time, and any
  number of **Released versions**.
- A **Mutation** during a **Dev period** advances the dev row's
  `.devN` counter; it never inserts a new row.
- A **Release** transforms the dev row into a released row in place;
  the dev period ends.
- An **Execution** consumes **Released versions** only. Recording a
  dev version on an execution's input is a category error.
- A **Cite URL** for a **Released version** is snapshot-pinned. A
  **Cite URL** for a **Dev version** carries no `@snaptime` — it
  resolves to live state.

## Example dialogue

> **Dev:** When a feature value is added to an image that's a member
> of dataset D, does D's version change automatically?
>
> **Project lead:** No. D is now *dirty* — `D.is_dirty()` would
> return true — but its `current_version` is still the last
> released version. The user has to call `D.mark_dev("...")` to flip
> D to a dev version. We considered automatic detection-and-flip,
> but the user owns the decision: not every drift warrants release
> attention.

> **Dev:** If I run `D.add_dataset_members([...])` on a dataset at
> `0.4.0`, what's `current_version` after?
>
> **Project lead:** `0.4.0.post1.dev1`. The mutation auto-flipped D
> to a dev version. Run `add_dataset_members` again and you'll get
> `0.4.0.post1.dev2`. Call `D.release(VersionPart.minor, "...")` to
> promote to `0.5.0`.

> **Dev:** The dev label says `post1.dev3` after three mutations.
> Why not just `dev3`?
>
> **Project lead:** PEP 440 requires the `.post1` segment to make
> the label sort *after* `0.4.0`. Without it, `0.4.0.dev3` would
> sort *before* `0.4.0`, which is wrong — dev state is post-release,
> not pre-release. We borrowed the convention from setuptools-scm,
> which produces the same form between tagged releases.

> **Dev:** I noted the version was `0.4.0.post1.dev2` yesterday.
> Can I download the dataset at that exact label today?
>
> **Project lead:** Only if today's dev row still says `.dev2`.
> If anything mutated the dataset overnight, the dev row's
> `Version` is now `.dev3` (or later), and `.dev2` no longer
> resolves — the system errors out. The rule is: a dev label
> resolves to live state if and only if it matches the dataset's
> current dev row. Want a stable reference that survives further
> mutations? Call `release()` to cut a real released version
> (snapshot-pinned). Dev labels are notational, not citational.

## Flagged ambiguities

- **"Version"** was used to mean both the `Version` *column* on a
  `Dataset_Version` row (a string) and the `DatasetVersion`
  *object* (a parsed PEP 440 type). Resolved: the column is
  `Version` (text); the object is `DatasetVersion`. The label
  inside either is a *PEP 440 version*, never "a semver".
- **"Snapshot"** is a Deriva concept (catalog snapshot ID,
  timestamp-keyed) but appears in DerivaML in two places: the
  `Dataset_Version.Snapshot` column (nullable; populated only on
  released rows) and the catalog-cloning machinery
  (`_version_snapshot_catalog`). Same concept; the dev-versioning
  work makes the *nullability* on `Dataset_Version` load-bearing.

# FK Traversal Reference

> Verified against a populated demo catalog
> (`create_demo_catalog(create_features=True, create_datasets=True)`,
> deriva-ml v1.46.x, captured 2026-06-13). Engine rules tagged
> `[engine: deriva-py]` are enforced upstream in the `deriva.bag`
> package; rules tagged `[deriva-ml]` are deriva-ml's own
> configuration of that engine. The "Worked examples" section pastes
> verified excerpts from `docs/reference/.examples/`.

FK traversal is the mechanism that decides **which rows of which
tables** a dataset operation (bag export, size estimate, clone, drift
check) includes, by walking the catalog's foreign-key graph from a set
of *anchor* rows. This page is the formal reference for that mechanism.

The behavior is split across two layers:

- The **engine** — `FKTraversalPolicy` (the configuration object) and
  `SchemaPathWalker` (the graph walker) — lives in **deriva-py**, in
  the `deriva.bag` package. It is shared by every producer and the bag
  loader.
- **deriva-ml** *configures* that engine: it builds the policy for a
  dataset export (`DatasetBagBuilder.build_policy`) and for a catalog
  clone (`clone_via_bag`), choosing which tables are terminal, how
  vocabularies are exported, and how dangling FKs are resolved.

For how a *bag* drives this walk end-to-end, see
[Bag Export](bag-export.md).

## Quick answers

| Question | Rule |
|---|---|
| Does the walk follow FKs in both directions? | [T2](#t2) |
| Why did my bag pull in another execution's assets? | [T3](#t3) (terminal tables) |
| How do I stop the walk at a table? | [T3](#t3), [T5](#t5) |
| Are vocabularies always fully exported? | [T4](#t4), [T7](#t7) |
| What limits how deep the walk goes? | [T6](#t6) |
| Does `dangling_fk_strategy` change which tables are reached? | No — [T7](#t7) (load-phase) |
| Which policy fields actually shape the walk? | [T2](#t2), [T3](#t3), [T5](#t5), [T6](#t6) |

## Mental model

The pipeline runs in **two phases**, and `FKTraversalPolicy`'s fields
divide cleanly between them.

1. **Walk phase.** `SchemaPathWalker` discovers the reachable
   `(schema, table)` set and the FK paths to each, starting from an
   **anchor set**. Only the *walk-shaping* fields are honored here —
   they change **what is reached**.
2. **Load phase.** After the walk, the loader
   (`BagCatalogLoader`) materializes rows and resolves
   vocabulary/asset/orphan/conflict handling. The *behavior-on-output*
   fields are honored here — they change **how reached rows are
   handled**, never which tables are reached.

This split is stated in `SchemaPathWalker`'s own docstring, which lists
the walk-shaping fields it honors and notes that "behavior-on-output
fields … are ignored — those belong to the consumer"
(`deriva/bag/path_walker.py`, the `SchemaPathWalker.__init__`
docstring).

| Phase | Fields | What they change |
|---|---|---|
| **Walk** | `schemas`, `exclude_schemas`, `exclude_tables`, `terminal_tables`, `max_depth` | The set of reached tables |
| **Load** | `vocab_export`, `asset_mode`, `dangling_fk_strategy`, `content_on_conflict`, `match_by_columns`, `preserve_provenance`, `intentional_cycles` | How reached rows are written |

Confusing the two phases is the most common source of surprise —
"I set `dangling_fk_strategy=DELETE`, why is that table *still* in my
bag?" The answer is always [T7](#t7): the walk already reached it; the
load merely changed how its orphan rows were resolved.

## The policy: `FKTraversalPolicy`

`[engine: deriva-py]`
`deriva/bag/traversal.py::FKTraversalPolicy` (a Pydantic `BaseModel`).
Every field has a default, so `FKTraversalPolicy()` is itself a valid,
working policy. The fields, with their exact code defaults:

| Field | Phase | Default | Meaning | Layer |
|---|---|---|---|---|
| `schemas` | walk | `None` | Schema allow-list. `None` = every schema reachable from the anchors except `exclude_schemas`. | `[engine: deriva-py]` |
| `exclude_schemas` | walk | `{"public", "_acl_admin", "WWW"}` (`DEFAULT_EXCLUDE_SCHEMAS`) | Schemas skipped during the walk. | `[engine: deriva-py]` |
| `exclude_tables` | walk | `set()` (empty) | Specific `(schema, table)` tuples to skip. | `[engine: deriva-py]` |
| `terminal_tables` | walk | `set()` (empty) | `(schema, table)` tuples the walker enters but does not follow *inbound* FKs out of. See [T3](#t3). | `[engine: deriva-py]` |
| `max_depth` | walk | `None` | Maximum FK hops from the anchor set. `None` = unbounded. | `[engine: deriva-py]` |
| `vocab_export` | load | `VocabExport.REFERENCED_ONLY` | How vocab tables are exported (`REFERENCED_ONLY` / `FULL`). | `[engine: deriva-py]` |
| `asset_mode` | load | `AssetMode.UPLOAD_IF_MISSING` | How the loader handles asset bytes (`ROWS_ONLY` / `UPLOAD_IF_MISSING` / `UPLOAD_FORCE`). | `[engine: deriva-py]` |
| `dangling_fk_strategy` | load | `DanglingFKStrategy.FAIL` | How orphan rows are resolved (`FAIL` / `DELETE` / `NULLIFY` / `PRESERVE`). | `[engine: deriva-py]` |
| `content_on_conflict` | load | `ContentConflictStrategy.FAIL` | RID-collision handling on content tables (`FAIL` / `SKIP_BY_RID`). | `[engine: deriva-py]` |
| `match_by_columns` | load | `{}` (empty) | Per-table "reconcile by these columns" rule for non-vocabulary content-addressed tables. | `[engine: deriva-py]` |
| `preserve_provenance` | load | `True` | Whether to send the bag row's `RCT`/`RCB` audit columns verbatim. | `[engine: deriva-py]` |
| `intentional_cycles` | load | `set()` (empty) | FK cycles marked intentional; logged at DEBUG instead of WARNING when the loader breaks them. | `[engine: deriva-py]` |

> **Note.** The walker (`SchemaPathWalker`) honors only the five
> walk-phase fields. The remaining seven are consumed by the loader
> after the walk; passing them to the walker has no effect on the
> reached set. This is exactly the walk/load split of the [mental
> model](#mental-model).

### deriva-ml's chosen configurations

`[deriva-ml]` Both of deriva-ml's main consumers mark the provenance
hub terminal, sourcing the set from a single shared constant —
`src/deriva_ml/core/constants.py::PROVENANCE_TERMINAL_TABLES`, the
`frozenset({("deriva-ml", "Execution"), ("deriva-ml", "Workflow")})`:

- **Dataset bag export** —
  `src/deriva_ml/dataset/bag_builder.py::DatasetBagBuilder.build_policy`.
  Sets `terminal_tables=set(PROVENANCE_TERMINAL_TABLES)`
  (`bag_builder.py:812`), `vocab_export=VocabExport.FULL` (consumers
  need the complete controlled vocabulary), `exclude_tables` = the
  `Dataset_X` association tables whose element type has no members in
  *this* dataset, and `intentional_cycles` to silence the known
  `Dataset ↔ Dataset_Version` cycle warning.
- **Catalog clone** —
  `src/deriva_ml/catalog/clone_via_bag.py`. Marks the same set terminal
  via the module-level
  `_DEFAULT_TERMINAL_TABLES = set(PROVENANCE_TERMINAL_TABLES)`
  (`clone_via_bag.py:90`), plus `vocab_export=VocabExport.FULL` and
  `dangling_fk_strategy=DanglingFKStrategy.DELETE`. When the caller
  passes their own `FKTraversalPolicy`, these clone-required settings
  are merged in only for fields the caller left unset
  (`model_fields_set`-based merge), so an explicit caller choice always
  wins.

Both consumers therefore mark `{Execution, Workflow}` terminal by
default. The reached-set effect of that choice is the verified contrast
in the [Worked examples](#worked-examples).

## Formal rules

<a id="t1"></a>
**T1 — Anchors define the walk's origin.** The walk starts from an
*anchor set*: deriva-ml translates the operation's target (a dataset
RID, or a whole table) into anchors and hands them to the walker as the
BFS roots. A `RIDAnchor` pins specific RIDs (e.g. one dataset's RID); a
`TableAnchor` anchors a whole table. The walker visits the anchor
table(s) first, then expands outward.
`[deriva-ml]` supplies the anchors
(`clone_via_bag.py` builds `RIDAnchor(table="Dataset", rids=[root_rid])`;
`bag_builder.py::anchors_for` builds the dataset's anchors);
`[engine: deriva-py]` consumes them as the `roots` argument of
`SchemaPathWalker.walk_bfs` (`deriva/bag/path_walker.py`).

<a id="t2"></a>
**T2 — The walk is bidirectional.** From each table the walker follows
**both** directions: *outbound* FKs (the tables this one references,
`table.foreign_keys`, `path_walker.py:229`) **and** *inbound* FKs (the
tables that reference this one, `table.referenced_by`,
`path_walker.py:243`). There is no single-direction mode — bidirectional
is the only walk any real producer wants (`FKTraversalPolicy`'s module
docstring: "Walks are always bidirectional"). Multi-FK edges between the
same pair of tables are deduplicated to one walk edge per
`(source, target)`.
`[engine: deriva-py]` `deriva/bag/path_walker.py::SchemaPathWalker`.

<a id="t3"></a>
**T3 — Terminal tables are entered but their inbound FKs are not
followed.** A table listed in `terminal_tables` is still *reached* (its
rows are emitted, so other rows' FKs into it resolve at load time) and
its *outbound* FKs are still followed. What the walker skips is the
table's *inbound* FKs — the FKs other tables declare *at* the terminal
table (`path_walker.py:241`: `if is_terminal: continue` before the
`referenced_by` loop). That asymmetry is the whole point: inbound
traversal from a provenance hub is what aggregates cross-anchor state.
From a terminal `Execution` row, inbound would otherwise reach every
`*_Execution` association and from there fan out to every *other* anchor
scope that shares the `Execution`.
`[engine: deriva-py]` is the mechanism
(`SchemaPathWalker`, the `is_terminal` check at
`path_walker.py:226`/`241`); `[deriva-ml]` chooses the set — both
`bag_builder.build_policy` (dataset export) and `clone_via_bag` source
`{Execution, Workflow}` from
`core/constants.py::PROVENANCE_TERMINAL_TABLES`.

On the demo catalog this rule severs **12 tables** from dataset `5D0`'s
walk — the `Execution_Asset` / `Execution_Metadata` / `File` /
`Execution_Execution` provenance closure plus the `Workflow_*` tail.
With the hub terminal the walk reaches **30** tables; without it, **42**
(see [Worked examples](#worked-examples) for the exact severed set).
That is the *reached-set* effect. On a hub-heavy topology such as the
eye-ai catalog the same rule additionally prevents a *row-count*
explosion (the original bug): one execution's slice would otherwise
sweep in *other* executions' rows through the shared `Execution`. The
reached-set contrast is the verified, demonstrable effect; the
row-count blow-up is the larger-scale consequence on catalogs whose
feature tables point into `Execution`.

<a id="t4"></a>
**T4 — Vocabularies are always leaves.** Independent of
`terminal_tables`, the walker enters a vocabulary table but never
follows *any* FK out of it (`path_walker.py:223`: `if
table.is_vocabulary(): continue`, *before* the outbound/inbound loops).
This is a universal walker guard — the rule that prevents the classic
`Subject → Species → every-other-Subject` explosion (a vocab term is
referenced by countless rows; following its inbound FKs would reach them
all). It is hardcoded precisely because every producer wants the same
behavior (`FKTraversalPolicy` module docstring: "Vocabulary tables are
always traversed inbound-only … It's hardcoded because every producer
wants the same behavior").
`[engine: deriva-py]` `deriva/bag/path_walker.py`.

<a id="t5"></a>
**T5 — Schema/table pruning happens before the edge is walked.** When
the walker considers a candidate edge it applies, in order: the
`exclude_schemas` deny-check (`_is_excluded_schema`,
`path_walker.py:336`), the `exclude_tables` deny-check
(`path_walker.py:338`), then the `schemas` allow-list — if `schemas` is
not `None`, any schema not in it is dropped (`path_walker.py:341`). A
candidate that fails any check is never enqueued, so the table (and
everything reachable only through it) is pruned from the walk.
`exclude_schemas` defaults to `DEFAULT_EXCLUDE_SCHEMAS` = `{"public",
"_acl_admin", "WWW"}` — ERMrest's structural/administrative schemas.
`[engine: deriva-py]` `deriva/bag/path_walker.py::SchemaPathWalker._enqueue_candidate`.

<a id="t6"></a>
**T6 — `max_depth` bounds FK hops from the anchor.** Each queue entry
carries its depth; when a table is dequeued at `depth >= max_depth` the
walker records the table but does not expand its edges
(`path_walker.py:211`: `if max_depth is not None and depth >= max_depth:
continue`). `max_depth=None` (the default) means unbounded; `max_depth=0`
means "only the anchored rows themselves." Negative values are rejected
at validation time (`FKTraversalPolicy._max_depth_nonnegative`).
`[engine: deriva-py]` `deriva/bag/path_walker.py` /
`deriva/bag/traversal.py::FKTraversalPolicy`.

<a id="t7"></a>
**T7 — Load-phase fields don't change what's reached.** The walker
honors only the five walk-phase fields ([T2](#t2)–[T6](#t6)). The
remaining `FKTraversalPolicy` fields are consumed by the loader *after*
the walk and change only how reached rows are written. They are:

- **`vocab_export`** — `VocabExport.REFERENCED_ONLY` (default; only
  terms the walk pulled in land in the bag) or `VocabExport.FULL`
  (a separate post-walk pass copies *every* term in each reached vocab
  table). deriva-ml's dataset export and clone both choose `FULL`.
- **`asset_mode`** — `AssetMode.ROWS_ONLY` (insert rows, transfer no
  bytes), `AssetMode.UPLOAD_IF_MISSING` (default; upload bytes only
  when the destination MD5 differs), or `AssetMode.UPLOAD_FORCE`
  (re-upload unconditionally).
- **`dangling_fk_strategy`** — `DanglingFKStrategy.FAIL` (default;
  abort on orphan rows), `DELETE` (drop orphan rows), `NULLIFY` (set the
  dangling FK column `NULL`), or `PRESERVE` (skip the bag-side parent
  check, trust the destination's FK constraint). deriva-ml's clone
  chooses `DELETE`.
- **`content_on_conflict`** — `ContentConflictStrategy.FAIL` (default;
  abort on a content-table RID collision) or `SKIP_BY_RID` (skip rows
  whose RID already exists, for resumable loads).
- **`match_by_columns`** — per-table "reconcile by these columns"
  rule for non-vocabulary content-addressed tables; empty by default
  (all non-vocabulary tables stay on the standard content path).

Because these run after the walk, changing any of them moves no table
into or out of the reached set — only the bytes/rows written for the
already-reached tables change.
`[engine: deriva-py]` mechanism
(`deriva/bag/traversal.py` enums + `BagCatalogLoader`); `[deriva-ml]`
chooses the values per operation (export and clone set
`vocab_export=FULL`; clone sets `dangling_fk_strategy=DELETE`).

## Worked examples

### A RID-anchored walk's reached set

This is the **verified** reached-table set for a real dataset (RID
`5D0`) on the demo catalog, anchored on that one Dataset RID, with the
default terminal-tables configuration applied (the post-#297 reality —
`Execution` / `Workflow` terminal). Source:
`docs/reference/.examples/traversal.txt`, field `REACHED_WITH_FIX` —
**30 tables**:

```
Asset_Role, Asset_Type, BoundingBox, BoundingBox_Asset_Type,
BoundingBox_Execution, ClinicalRecord, ClinicalRecord_Observation,
Dataset, Dataset_Dataset_Type, Dataset_Execution, Dataset_Subject,
Dataset_Type, Dataset_Version, Execution,
Execution_Image_BoundingBox, Execution_Image_Quality,
Execution_Status, Execution_Subject_Health, Feature_Name, Image,
ImageQuality, Image_Asset_Type, Image_Dataset_Legacy,
Image_Execution, OCR_Report, Observation, Report, Subject,
SubjectHealth, Workflow
```

Reading the shape:

- **Members** — `Dataset`, `Dataset_Subject`, `Subject`, `Image`
  (via `Image_Dataset_Legacy`), plus the domain rows they reference
  (`Observation`, `ClinicalRecord`, `Report`, `OCR_Report`).
- **Features** — feature tables (`BoundingBox`, `ImageQuality`,
  `SubjectHealth`, and their `*_Execution` associations) reach the walk
  by FK-following from the member rows; no separate policy field is
  needed ([T2](#t2)).
- **Vocabularies** — `Asset_Role`, `Asset_Type`, `Dataset_Type`,
  `Feature_Name` are reached and entered, but the walk stops there
  ([T4](#t4)).
- **Provenance hub** — `Execution` and `Workflow` are present (their
  rows are emitted, so other rows' FKs into them resolve), but they're
  terminal, so the walk does not fan back out through them ([T3](#t3)).
  This is exactly why the execution-asset/metadata closure
  (`Execution_Asset`, `Execution_Metadata`, `File`, …) and the
  `Workflow_Type` tail are **absent** from this set — see the contrast
  below.

### Terminal tables in action (the provenance-hub rule)

Marking `{Execution, Workflow}` terminal ([T3](#t3)) is the post-#297
default for both dataset export and clone. On the demo catalog, dataset
`5D0`'s walk shows a **real, verified reached-set contrast** between the
two configurations (`docs/reference/.examples/traversal.txt`):

| Configuration | Field | Reached tables |
|---|---|---|
| **With terminal** (current default) | `REACHED_WITH_FIX` | **30** |
| **Without terminal** | `REACHED_NO_TERMINAL` | **42** |

`EXECUTION_ASSET_IN` is `False` with the terminal rule — `Execution_Asset`
is **not** reached when the hub is terminal. The **12 tables** that the
terminal rule severs (`EXTRA_WITHOUT_TERMINAL`) are exactly the
provenance closure reached by walking *inbound* from `Execution` /
`Workflow`:

```
Execution_Asset, Execution_Asset_Asset_Type, Execution_Asset_Execution,
Execution_Execution, Execution_Metadata, Execution_Metadata_Asset_Type,
Execution_Metadata_Execution, File, File_Asset_Type, File_Execution,
Workflow_Type, Workflow_Workflow_Type
```

These are the `Execution_Asset` / `Execution_Metadata` / `File`
asset-and-metadata tables (and their `*_Execution` / `*_Asset_Type`
associations), the `Execution_Execution` self-association, and the
`Workflow_Type` / `Workflow_Workflow_Type` vocabulary tail. With
`Execution` and `Workflow` terminal, the walk emits their rows but does
not follow their inbound FKs ([T3](#t3)), so this entire closure drops
out of the slice — 42 tables shrink to 30.

**The mechanism, and why it matters at scale.** Each severed table is
reached *only* by inbound traversal from the hub: e.g.
`Execution → [inbound, referenced_by] → Execution_Asset_Execution →
Execution_Asset`. That inbound edge is precisely what [T3](#t3) cuts.
On the demo catalog the effect is the reached-set contrast above. On a
*hub-heavy* topology such as the eye-ai catalog — where a feature table
(`Annotation`) carries an FK pointing **into** `Execution` — the same
inbound edge additionally triggers a **row-count explosion** (the
original bug PR #297 fixed): one execution's slice would sweep in
*other* executions' rows through the shared `Execution`, because every
`*_Execution` association fans out to every anchor scope that touched
that execution. The reached-set contrast shown here is the directly
demonstrable, verified effect; the row-count blow-up is the
larger-scale consequence the same rule prevents.

### `max_depth` truncation

The demo catalog's natural FK depth is shallow, so an unbounded walk and
a generous `max_depth` reach the same set — the demo catalog does not
exhibit meaningful `max_depth` truncation. *Illustrative:* setting
`max_depth=2` on the dataset-`5D0` walk above would record `Dataset`
(depth 0), its directly-referenced/​referencing tables such as
`Dataset_Subject` and `Dataset_Version` (depth 1) and *their* neighbours
`Subject`, `Execution` (depth 2), but would **not** expand past depth 2
to reach `Image`, `Observation`, or the feature tables — those sit 3+
hops from the anchor ([T6](#t6)).

## See also

- [Bag Export](bag-export.md) — how a dataset bag drives this walk
  (anchors, excluded-empty associations, the export spec).
- [Denormalization](denormalization.md) — how the walked FK paths
  become a single denormalized frame.
- ADR-0006 — bag-oriented data movement (`docs/adr/0006-*`), the
  decision record behind the shared `FKTraversalPolicy` contract.

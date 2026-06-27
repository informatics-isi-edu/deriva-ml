# Lineage member-asset traversal — design

**Date:** 2026-06-26
**Status:** Approved
**Component:** `deriva_ml.core.mixins.execution` (`lookup_lineage` and helpers)
**Relates to:** ADR-0001 (lineage walks data-flow, not orchestration) — this
change stays inside that doctrine.

## Problem

`lookup_lineage(rid)` walks data-flow provenance backward from an artifact
through producing executions and their consumed inputs. For a **dataset**, the
walk derives a single producer from `Dataset_Version.Execution` (the execution
that built/versioned the dataset) and recurses through that execution's consumed
inputs.

It never considers **what produced the dataset's member assets**. When a
dataset's members were created by a *different* execution than the one that
assembled the dataset, the members' upstream lineage is invisible.

### Concrete failure (the motivating case)

In the two-execution CIFAR ingest:

- An **upload execution** consumes a source `File` dataset as Input and produces
  per-image `Image` assets as Output.
- A separate **datasets-phase execution** assembles those `Image` assets into
  the training dataset hierarchy (`Complete`, `Training`, `Small_Testing`, …).

So for an image dataset:

- `_producer_of_dataset(image_dataset)` → the **datasets-phase** execution,
  which consumed no source. Its input walk is empty/unrelated.
- The source is only reachable via the dataset's **member `Image` assets**:
  `Image_Execution` (Asset_Role="Output") → upload execution → its Input = the
  source `File` dataset.

`lookup_lineage(image_dataset)` therefore never surfaces the source File
dataset, even though every edge exists in the catalog. (Verified live on
catalog 278 / the cifar-example `test_lineage_connected.py`: the raw
`Dataset_Execution` edge is present; only the *walk* fails to traverse it.)

This is recorded as `tk-018` in the cifar-example tacit-knowledge log.

## Goal

`lookup_lineage(<dataset>)` — at the root, and for any dataset reached as a
consumed input mid-walk — must also follow the dataset's member assets back to
their producing execution(s) and continue the walk from there, so the members'
upstream inputs (the source File dataset, in the CIFAR case) appear in the
lineage tree. The fix is **general**: it helps any dataset whose members were
execution-produced (predictions, ingested images, derived assets), not just
CIFAR.

## Doctrine alignment (ADR-0001)

ADR-0001 defines a **data-flow parent** of an execution as the producing
execution of a consumed input, where an asset's producer is derived from
`<AssetTable>_Execution` with `Asset_Role="Output"`. Member-asset producers are
exactly that kind of edge. This change **does not** introduce a new edge class
and **does not** traverse `Execution_Execution` orchestration links. It widens
the set of *data-flow* parents the dataset walk already conceptually owns. No
ADR change is required; a one-line note may be added to ADR-0001's
consequences.

## Design

### Component 1 — `_producers_of_dataset_members` (new private helper)

Signature (in `execution.py`, beside `_producer_of_dataset` /
`_producer_of_asset`):

```python
def _producers_of_dataset_members(
    self, dataset_rid: RID, version: Any | None = None
) -> set[RID]:
    """Distinct executions that produced the member assets of a dataset.

    Enumerates the dataset's member asset tables and, for each, collects the
    distinct producing executions (the asset's <Asset>_Execution association,
    Asset_Role="Output"). Deduplicated across all members and tables, so a
    dataset of 2000 images that share one producing execution yields a single
    RID. Nested-Dataset and non-asset member kinds are skipped (datasets'
    producers are handled by _producer_of_dataset; the recursion reaches them
    via the normal dataset-input path).

    Returns an empty set when the dataset has no member assets or none of them
    have a recorded Output producer.
    """
```

Behavior:

1. Call `self.lookup_dataset(dataset_rid).list_dataset_members(version=version)`
   to get `{member_type: [rows]}`.
2. For each member type that resolves to an **asset** table (use
   `self.model.is_asset(table)`; skip the nested-`Dataset` member kind and any
   non-asset member kind):
   - Resolve the asset→execution association with
     `self.model.find_association(asset_table, "Execution")` (the same call
     `_producer_of_asset` uses). If the asset table has no `<Asset>_Execution`
     association (`NoAssociationException`), skip that table.
   - Collect the **distinct** `Execution` RIDs for that table's members with
     `Asset_Role="Output"`. **Query strategy (performance, see below).**
3. Union the per-table distinct sets and return them.

### Performance — join through membership, do not pass a giant RID list

A dataset can have thousands of members. Filtering `<Asset>_Execution` by an
`IN (<thousands of RIDs>)` list risks an HTTP 414 (the same failure class as the
`resolve_rids` scale bug, tk-006/007/008 in the cifar log).

**Primary strategy:** build the query server-side by joining through the
dataset-membership association rather than materializing a client-side RID
list. Conceptually:

```
Dataset_<member>  (Dataset == dataset_rid)
   → <member asset table>
   → <member>_Execution  (Asset_Role == "Output")
   → project distinct Execution
```

This issues **O(number of member asset tables)** queries (typically 1–2), each
server-side, with no large client-side `IN` clause and no per-asset round-trip.
Use the catalog `pathBuilder()` link chaining already used by
`list_dataset_members` / `find_executions`.

**Documented fallback:** if a clean membership join is not expressible for a
given association shape, fall back to fetching member RIDs and filtering
`<Asset>_Execution` in **chunks** (reuse the existing chunk size used elsewhere
for RID-bounded queries, e.g. 500) to stay under URL limits — never a single
unbounded `IN`.

Either way the helper must be **O(member-asset-tables) queries, not
O(members)**, and that property is asserted by a test.

### Component 2 — `_walk_node` gains `extra_parent_rids`

Add an internal-only keyword parameter:

```python
def _walk_node(self, *, execution_rid, depth_remaining, max_executions,
               visited_global, in_progress, flags,
               extra_parent_rids: set[RID] | None = None) -> "LineageNode | None":
```

`extra_parent_rids` is merged into the node's computed `parent_rids` set
**before** the recursion loop. Default `None` preserves every existing call
path unchanged. This is the mechanism by which root-seeded member-producers
attach as parents of the root node and get walked (with full
`visited_global` / cycle / depth handling).

### Component 3 — root seeding (`lookup_lineage`, dataset roots only)

After classification, for a **Dataset** root (`_classify_rid` already returns
`producer_rid = _producer_of_dataset(rid)` as its second element):

```python
# producer_rid is the value _classify_rid returned for this dataset root.
member_producers = self._producers_of_dataset_members(rid)
seed = member_producers - ({producer_rid} if producer_rid else set())
```

The `- {producer_rid}` subtraction prevents listing the root execution as its
own parent in the common case where it also produced some members. The deeper
"same execution reached via two paths" case (a member-producer that recurs
elsewhere in the tree) is handled, as for any parent, by `visited_global` /
`already_shown`. Both guards are intentional and complementary, not redundant.

- **Both exist** (version-producer and member-producers): walk from
  `producer_rid` and pass `extra_parent_rids=seed` to the root `_walk_node`
  call. Root node = the version-producer (preserves the existing contract that
  `root.producing_execution` is the dataset's `Dataset_Version.Execution`);
  member-producers appear in its `parents`, and the recursion reaches the
  source from there.
- **Only member-producers** (`producer_rid is None`, `member_producers`
  non-empty): pick a deterministic representative (e.g. `min(member_producers)`)
  as the root `execution_rid`, pass the rest as `extra_parent_rids`. The result
  is non-empty (today it would be an empty `LineageResult`).
- **Neither**: empty `LineageResult` (unchanged).

Only the **Dataset** branch is affected. Asset, Feature-value, and Execution
roots are untouched.

### Component 4 — mid-walk seeding (`_walk_node`, consumed-dataset expansion)

In the existing `for ds in record.list_input_datasets()` loop, after adding
`_producer_of_dataset(ds.dataset_rid)` to `parent_rids`, also:

```python
parent_rids |= self._producers_of_dataset_members(ds.dataset_rid)
```

No other change. The existing `visited_global` dedup, cycle detection, and
`depth` capping apply to these parents exactly as to any other.

### No public-model change

`LineageNode`, `LineageResult`, `RootDescriptor`, and the summary models in
`execution/lineage.py` are unchanged. Member-producers are ordinary producing
executions and appear as ordinary `parents`. Consumers (deriva-ml-mcp tool
wrapper, notebook, skills) that already render the tree get the deeper lineage
for free; nothing in the serialized shape changes.

## Data flow (CIFAR example, after the fix)

```
lookup_lineage(image_dataset)
  root node = datasets-phase exec        (Dataset_Version.Execution)
    parents:
      └─ upload exec                     (member-producer of the Image members)
           consumed_datasets:
             └─ source File dataset      ← now reachable (tk-018 closed)
                  parents:
                    └─ source-registration exec  (produced the File dataset)
```

## Testing

### Offline unit (mocked catalog / fixtures)

- `_producers_of_dataset_members`:
  - 2000 members sharing one producing execution → **one** RID (dedup proven).
  - No member assets / no Output producers → empty set.
  - Mixed member asset types → union across tables.
  - Nested-Dataset members and non-asset members skipped.
- Root seeding:
  - Dataset whose version-producer ≠ member-producer → root node is the
    version-producer, member-producer present in `parents`; walking up reaches
    the member-producer's consumed inputs.
  - Dataset with **no** version-producer but with member-producers → non-empty
    result walked from the member-producers (not the old empty result).
  - Dataset where version-producer == member-producer (they coincide) → no
    duplicate parent (dedup via the `seed` subtraction + `visited_global`).
- Mid-walk seeding: an execution consuming a dataset whose members have a
  distinct producer → that producer appears among the node's parents.
- Regression: existing lineage tests for Asset / Feature / Execution roots and
  for datasets whose producers coincide remain green.
- Cycle/diamond/depth: a member-producer already in `visited_global` is marked
  `already_shown` (not re-expanded); a positive `depth` caps member-producer
  expansion and sets `depth_capped` like any other parent.

### Performance

- Assert `_producers_of_dataset_members` issues **O(member-asset-tables)**
  queries on a large-member fixture, not O(members) — e.g. by counting
  path-builder fetches or asserting completion without per-asset round-trips.

### Live (end-to-end, gated)

- After a two-execution CIFAR load, `lookup_lineage(<image dataset>)` yields a
  tree in which the **source File dataset** appears (as a `consumed_datasets`
  entry of the upload-exec node, reached via the member-producer parent). This
  is the end-to-end proof that tk-018 is closed. It complements the
  cifar-example's `test_lineage_connected.py` (which asserts the raw
  `Dataset_Execution` edge); this asserts the *walk* now finds it.

## Out of scope

- Walking historical dataset versions (the existing "current version only"
  limitation is unchanged; `version=` is plumbed through the helper for future
  use but the walk still resolves the current version).
- Any change to forward lineage (`find_executions_consuming`) — it already
  handles asset and dataset RIDs.
- Any public-model or `Execution_Execution`/orchestration change.

# Lineage member-producer membership-join — design

**Date:** 2026-06-27
**Status:** Approved
**Component:** `deriva_ml.core.mixins.execution`
(`_producers_of_dataset_members`, `_distinct_member_output_producers`)
**Relates to:** tk-018 (member-asset traversal), tk-022 (consumed-version
faithfulness). Fixes the live URL-length 404 recorded as cifar-example `tk-023`.

## Problem

`_producers_of_dataset_members(dataset_rid, version)` finds the executions that
produced a dataset's member assets. It currently:
1. enumerates the dataset's member RIDs via `list_dataset_members(version=...)`, then
2. for each member asset table, calls `_distinct_member_output_producers`, which
   filters `<member>_Execution` by `assoc.columns[asset_fk].in_(chunk)` over the
   member RIDs in chunks of `_MEMBER_PRODUCER_CHUNK = 500`.

ERMrest renders `.in_(...)` as a **URL-path** predicate
(`(Image=RID);(Image=RID);...`). A real dataset with 500 members produces ~6.5 KB
of RIDs in the URL path, exceeding Apache's default URL length limit (~8 KB). The
server returns a bare HTML **404**, surfaced as a `DataPathException`.

Confirmed live: `lookup_lineage('1MEP')` on catalog 278's `Small_Testing`
(500 members) 404s. The traversal **logic is correct** — with a smaller chunk it
walks to the source File dataset `M0J` — but the default chunk blows the URL.
Every prior test used 1-member datasets (offline mocks bypass the query; the two
live tests attach one Image), so the URL-length class was never exercised.

`_MEMBER_PRODUCER_CHUNK = 500` looks safe (it mirrors the resolve_rids chunk) but
resolve_rids chunks go in a POST body; `.in_()` goes in the URL path — different
limits. 500 is fine for a body, far too many for a path.

## Goal

Find a dataset's member-producing executions **without ever placing a
client-side member RID list in the URL** — eliminating the URL-length class
entirely — while preserving the tk-022 consumed-version faithfulness (the member
set must reflect the consumed version, not current membership).

## Design

Replace the enumerate-RIDs-then-`.in_()` approach with a **server-side
membership join**. The URL then carries only the single dataset RID.

### The join (proven viable live)

For a dataset and a member asset table, the producers are:

```
Dataset_<member>  (Dataset == dataset_rid)
   → <member asset table>
   → <member>_Execution  (Asset_Role == "Output")
   → distinct Execution
```

Proven against catalog 278: `Dataset_Image(Dataset='1MEP') → Image →
Image_Execution(Asset_Role="Output")` projected to distinct `Execution` returns
`{'QYE'}` (the upload execution) — with only `Dataset=1MEP` in the URL, no member
RIDs.

### Version faithfulness — build the join on the version-snapshot catalog

`list_dataset_members` resolves a version by building its path against a
**version-snapshot catalog**: `self._version_snapshot_catalog(version).pathBuilder()`
(`dataset.py`, the `_version_snapshot_catalog` method returns `self._ml_instance`
when `version` is None, or a snapshot-bound catalog otherwise). The membership
join MUST be built on the **same** snapshot pathBuilder so that, for a consumed
version, `Dataset_<member>(Dataset=rid)` returns the **consumed version's**
members — server-side, no RID list, AND version-faithful. This preserves tk-022.

A `Dataset` object is already reachable from the helper (`self.lookup_dataset(dataset_rid)`),
and `_version_snapshot_catalog` is a method on `Dataset`, so the helper can obtain
the right pathBuilder via `dataset._version_snapshot_catalog(version).pathBuilder()`.

### Component 1 — `_producers_of_dataset_members(dataset_rid, version=None)` rewrite

No longer enumerates member RIDs. Instead:

1. `dataset = self.lookup_dataset(dataset_rid)`.
2. `snapshot_pb = dataset._version_snapshot_catalog(version).pathBuilder()`.
3. Discover the dataset's **member asset tables** the same way
   `list_dataset_members` does: iterate `self._dataset_table.find_associations()`
   (the membership association tables, e.g. `Dataset_Image`, `Dataset_File`); for
   each, the membership table is `assoc.table` and the member target table is
   `assoc.other_fkeys`' `pk_table`. (Mirror `list_dataset_members`'s discovery
   loop — `dataset.py` around the `find_associations()` block — to get the
   membership-association table + target table pair.)
4. Skip target tables that are not asset tables (`self.model.is_asset(target)`)
   and the nested-`Dataset` membership (as the existing code skips non-asset/
   nested kinds).
5. For each (membership association, member asset table), delegate to the join
   helper and union the results.

The `self._dataset_table` reference: the existing code reaches member info via
`lookup_dataset(...).list_dataset_members(...)`. The rewrite needs the dataset
table + its associations; reuse however `list_dataset_members` obtains
`self._dataset_table` (it is the deriva-ml `Dataset` table). The plan pins the
exact accessor after reading the current `list_dataset_members` body.

### Component 2 — `_distinct_member_output_producers` rewritten as a join

New shape (keyed by the join inputs, not a RID list):

```python
def _distinct_member_output_producers(
    self, snapshot_pb, membership_assoc, member_table, dataset_rid: RID
) -> set[RID]:
    """Distinct Output-producing executions of a dataset's members of one
    asset table, via a server-side membership join (no client-side RID list).
    """
```

Behavior:
- Resolve the `<member>_Execution` association:
  `assoc_exec, member_fk, _exec_fk = self.model.find_association(member_table, "Execution")`;
  on `NoAssociationException`, return `set()`.
- Build the path on `snapshot_pb`:
  `membership_path.filter(Dataset == dataset_rid).link(member_path).link(member_exec_path).filter(Asset_Role == "Output")`
  using the schema/table names from `membership_assoc`, `member_table`, and
  `assoc_exec`. Link on the FK columns the model exposes (the same FKs
  `find_association` / `find_associations` return); do NOT hand-write RID
  predicates.
- Project distinct `Execution`: `.attributes(member_exec_path.Execution).fetch()`,
  collect non-null `Execution` values into a set.
- The URL carries only `Dataset == dataset_rid`. No `.in_()`, no chunking.

### Component 3 — delete `_MEMBER_PRODUCER_CHUNK`

Remove the `_MEMBER_PRODUCER_CHUNK = 500` constant from `execution.py` — it has no
remaining use. (`_VERSION_RID_CHUNK` in `_helpers.py` STAYS: the version fetch
references only a handful of distinct RIDs and is a different code path that does
not hit this URL-length problem.)

### No public-model / contract change

`lineage.py` unchanged. `_producers_of_dataset_members`'s public signature
`(dataset_rid, version=None) -> set[RID]` is unchanged. Only the internal helper's
signature and the query mechanism change.

## Testing

### Offline unit (`tests/execution/test_producers_of_dataset_members.py`)

The existing offline tests mock `list_dataset_members` + a `_distinct_member_output_producers`
seam. Rework them for the new structure:
- The `_FakeMembersML` harness must now provide the join inputs instead of member
  RIDs: stub `lookup_dataset(rid)` to return a fake `Dataset` whose
  `_version_snapshot_catalog(version)` returns a fake catalog with a `.pathBuilder()`,
  and whose dataset-table `find_associations()` yields the membership associations.
  Override the join helper `_distinct_member_output_producers` to return scripted
  producer sets per member table (so the enumerate/dedup/skip logic of
  `_producers_of_dataset_members` is exercised offline without a catalog).
- Keep the existing behavioral assertions (dedup across member tables, skip
  non-asset/nested members, empty when no producers), re-expressed for the new
  inputs.
- Add a **join-shape** unit that drives the REAL `_distinct_member_output_producers`
  against a mock snapshot pathBuilder and asserts: (a) the filter is
  `Dataset == dataset_rid` (NOT an `.in_()` over member RIDs — assert no `.in_`
  call on the member column), (b) the path links membership → member →
  member_Execution and filters `Asset_Role == "Output"`, (c) distinct Execution
  is returned. The assertion that matters: **no member-RID `.in_()` is ever
  built** (that's the bug class).

### Live regression (`tests/execution/test_lookup_lineage_live.py`, DERIVA_HOST-gated)

- **≥200-member dataset:** build a dataset with at least ~200 (target ~500)
  asset members produced by an upstream execution, then call
  `lookup_lineage(<that dataset>)` (or `_producers_of_dataset_members` directly)
  and assert it returns the producing execution WITHOUT a 404. A 1-member test
  cannot catch the URL-length class; the fixture MUST be large enough that the
  OLD code would have 404'd. Document the member count in the test.
- Confirm the existing live tests
  (`test_lookup_lineage_descends_into_member_asset_producers`,
  `test_lookup_lineage_reflects_consumed_version_not_latest`) still pass — the
  join must produce the same producers for their small datasets, and the
  consumed-version test confirms version faithfulness through the snapshot
  pathBuilder.

### No-regression

- `tests/execution/test_lookup_lineage_unit.py` (walk tests scripting member
  producers via the `_FakeML` override) stays green — the override returns the
  scripted set regardless of mechanism.

## Out of scope

- Changing `_VERSION_RID_CHUNK` / the version fetch in `_helpers.py` (different
  path, not URL-length-bound at realistic scale).
- Any change to `list_dataset_members`, the root path, or `_walk_node`.
- Reworking other `.in_()` call sites in the codebase (only the member-producer
  one is on the lineage hot path with hundreds of RIDs).

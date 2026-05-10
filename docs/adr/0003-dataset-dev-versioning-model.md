# ADR-0003: Dataset dev versioning — every mutation lands on dev, release is the only path to a released version

Date: 2026-05-09
Status: Accepted

## Context

DerivaML datasets carry a semantic version (`current_version`) that
points at a row in `Dataset_Version`. Today, every member-mutation
operation (`add_dataset_members`, `delete_dataset_members`) auto-bumps
to a new released `Dataset_Version` row with a stamped catalog
`Snapshot`. Three problems with that model:

1. **No representation of "modified since last release but not yet
   re-released."** Every mutation is its own release, so the dataset
   is always pointing at a frozen snapshot — there's no notational
   space for "the dataset has drifted, but I haven't decided what kind
   of release that warrants."
2. **No way to record indirect drift.** When a feature value is added
   to a member of a dataset, the dataset's `Dataset` row and member
   list are unchanged, but the bag's content has changed. Today's
   model has nowhere to put that fact.
3. **`Dataset.cite()` cannot distinguish frozen from live state.** A
   citation URL for "the current version" can only ever be a
   snapshot-pinned URL, even when the catalog has drifted underneath.

The shape of the fix: introduce a *dev* version state, separate from
released versions. The hard design questions are *which mutations
land on dev versus release*, *how dev rows are stored*, and *what
happens at release time*.

## Decision

**Every mutation lands on the dev version. Release is the only
operation that produces a released version.**

Concretely:

- A dataset's `current_version` is either a *released* PEP 440
  version (`0.4.0`) pinning a catalog `Snapshot`, or a *dev* PEP 440
  version (`0.4.0.post1.devN`) with `Snapshot=NULL`.
- `add_dataset_members`, `delete_dataset_members`, `mark_dev`, and
  any future drift-recording operation flip the dataset to a dev
  version. They never produce a released version.
- `release(bump, description, execution=None)` is the only
  operation that produces a released version. It promotes the dev
  row in place by setting `Version` to the released label, stamping
  `Snapshot`, replacing `Description` with release notes, and
  overwriting the dev row's `Execution` link with the supplied
  `execution` (or `NULL` if none). The released row's `Execution`
  link records "the execution that called `release()`", not "the
  most recent mutator" — mutator authorship during the dev period
  is recoverable from the audit trail (`RMT`, per-feature-value
  provenance) and doesn't need the version row to carry it.
- `release()` raises `DerivaMLValidationError` when called on a
  dataset with no dev row. Users wanting a no-op release call
  `mark_dev` first to declare a dev period, then `release` to
  promote it. The error message points at this resolution path.
- Dev rows are *lazy*: a dev row exists only while the dataset is
  in dev state. Releases do not preemptively create the next dev row.
- Dev rows are *mutable*: one row per (dataset, dev period). The
  `.devN` counter advances by `UPDATE`, not by `INSERT`. The
  `Description` column is **replaced** on each mutation with the
  most recently supplied description (not appended). Prior values
  are recoverable from the catalog's audit log if needed.
- **A dev label resolves if and only if it matches the dataset's
  current dev row's `Version`.** The dev row is mutable, so
  `Version="0.4.0.post1.dev2"` is observable only at the moment
  that's its current value; afterward, the same row's `Version`
  says `.dev3`, `.dev4`, etc. The rule is "dev labels resolve to
  the live present, and only when they describe it accurately" —
  not "dev labels never resolve". As a corollary: methods that
  produce a dev label (e.g., `mark_dev`) **return `None`, not
  `DatasetVersion`** — a returned dev label can't be passed to
  any version-accepting API later (the next mutation makes it
  unaddressable), so returning it would invite caller mistakes.
  Callers who want to display the new label read
  `current_version` after the call. `release()`, by contrast,
  returns `DatasetVersion` because a released label *is*
  addressable across time.
  APIs that accept a `version=` argument:
   - Treat `version=None` (or omitted) as "current version" —
     whichever the dataset has at request time, dev or released.
   - Accept any released label as today (snapshot-pinned).
   - Accept the current dev label (matches the dev row's current
     `Version`) and resolve it to the live catalog (no
     `@snaptime`).
   - Reject a dev label that does **not** match the current dev
     row's `Version` — raise a clear error: dev versions are
     mutable and historical or post-release `.devN` values are
     not addressable.
- The `.devN` counter is a generation number, not a handle to
  historical state. Its purpose is notational change-detection —
  two reads of `current_version` at different times can be told
  apart by their `.devN`. It is not a stable identifier across
  time.
- Bag downloads of the current dev version use live catalog
  state with no `@snaptime` pin. Two downloads of the same dev
  label may differ if the catalog drifted between them. The
  cite-URL form follows the same rule (no `@snaptime` for dev
  versions).
- The `.devN` counter advances **per call** that actually changes
  at least one row — `add_dataset_members`, `delete_dataset_members`,
  `mark_dev`, or any future drift-recording operation. A call that
  no-ops (e.g., `add_dataset_members([])`) does not advance the
  counter and does not create a dev row. The first effective
  mutation after a release creates the dev row at `.dev1`; there
  is no `.dev0`. Per-call granularity matches setuptools-scm's
  `.devN` semantics — one commit equals one increment regardless
  of how many files it touched.
- `create_dataset` initializes a new dataset at `0.1.0` released
  (no dev row). The "every mutation lands on dev" rule applies after
  creation, where it's load-bearing — at creation time there's no
  drift to record.
- **A dev version must never appear as the recorded version of an
  `Execution`'s consumed dataset.** Dev versions are notational, not
  citational. Executions consume released versions; live-state
  consumption is recorded as live, not as a moving dev label.

There is **no schema migration**.
`Dataset_Version.Snapshot` is already nullable in the existing
schema (verified in `create_schema.py` and `validation.py`); the
dev-versioning work just makes that nullability load-bearing
instead of incidental. No DDL change. No data migration (no dev
rows exist yet, so nothing to backfill). The relevant code change
in `create_schema.py` is to update the `Snapshot` column's comment
to reflect its new contract — `NULL` means "live state, dev row" —
and to update the schema-validator's expected-columns map
(`validation.py`) where any expectations need sharpening.

## Considered Options

### Option B (rejected): Computed dev versions, no persistence

`Dataset_Version` would be unchanged. `current_version` would return
a synthetic dev label after every released bump. **Rejected**
because there'd be nothing to attach a description, an execution
link, or accumulated drift information to. The "notational clarity"
that motivated the work would be lost — the version label would
exist in name only.

### Option C (rejected): Explicit dev-mode entry

Member-mutations would keep auto-bumping to a real release as today;
dev versions would only be reached via an explicit `start_dev()`
call. **Rejected** because mixed semantics (mutations bump to
release, but feature drift bumps to dev) are impossible to remember,
and the `start_dev`/`end_dev` API surface is the same shape as
`mark_dev` once we already need that — the explicit mode entry
collapses into something we already have.

### Option A.b/A.c (rejected): Insert new released row at release;
delete or archive the dev row

`release()` would `INSERT` a new released `Dataset_Version` row and
either delete the dev row or keep it as a "superseded" record.
**Rejected** because dev rows are already established as mutable
(advancing `.devN` is a row update). One more update at release —
to set `Version` to released and stamp `Snapshot` — is consistent
with that pattern. Insert-and-delete tells the same story with
extra steps; archive-superseded is the over-engineering the
notational-clarity goal was supposed to avoid.

## Consequences

- `increment_dataset_version` is renamed to `release(bump,
  description, execution=None)` and moves to `Dataset` as an
  instance method. The argument formerly known as `component` is
  renamed to `bump` to match the workspace's `bump-version` CLI
  vocabulary. The `execution` argument changes type from `RID |
  None` to `Execution | None` to match the rest of the new API
  surface (typed objects, not bare RIDs). This is a breaking
  change for callers of the previous public API and belongs in a
  2.0 release. No deprecated alias is provided — CLAUDE.md's "no
  backwards-compat shims" rule applies.
- A new `Dataset.is_dirty()` / `Dataset.release_diff()` /
  `Dataset.compare_versions(v_a, v_b)` trio detects catalog drift
  by walking the same FK paths used to generate the dataset bag
  (via `CatalogGraph`), filtering by an `RMT` time predicate. The
  drift walk *is* the bag walk plus an `RMT` filter. The three
  methods all flow through one internal
  `_diff_between(t_lower, t_upper)` helper; they differ only in
  what they pass to it:
   - `is_dirty()` — fast `bool` predicate, short-circuits on first
     non-zero count. Calls `_diff_between(t_last_release, None)`
     where `None` means "live state, no upper bound."
   - `release_diff()` — per-table change counts. **Implemented as
     a thin wrapper around** `compare_versions(self.last_released_version,
     self.current_version)`. When the dataset is in dev,
     `current_version` is the dev label, which resolves to live
     state. When the dataset is at its last release with no
     drift, both endpoints coincide and the result is `{}`. This
     is the right answer in both cases.
   - `compare_versions(v_a, v_b)` — per-table change counts
     between any two endpoints. Each argument may independently
     be a released label (resolves to that snapshot's timestamp)
     or the current dev label (resolves to live state). Stale or
     post-release dev labels error per the addressability rule.
     The predicate is `min(t_a, t_b) < RMT <= max(t_a, t_b)`;
     order is symmetric for the result set.

  All three live on `Dataset` only, not on `DatasetLike` — bags
  can never be dirty.
- Deletions of catalog rows referenced by a dataset are **not**
  detected by any of these methods (a deleted row is invisible to
  the bag walk too). Users who delete content rows must call
  `mark_dev` manually. Closing this gap would require querying
  the snapshots' RID sets directly and computing set differences
  — a separate, deferred operation with a different cost profile.
- Cite-URL routing falls out: released versions get
  snapshot-pinned URLs; dev versions get no-snapshot URLs. The
  check is the PEP 440 `is_devrelease` property — see ADR-0004 for
  why we use PEP 440.

## Read-side surface

`dataset_history()` and `current_version` are unchanged in shape but
gain new behaviors implied by dev rows being first-class
`Dataset_Version` entries:

- `dataset_history()` returns **all** `Dataset_Version` rows for
  the dataset, dev or released, with no filtering. Callers who want
  released-only filter by the PEP 440 typed property:
  `[h for h in ds.dataset_history() if not
  h.dataset_version.is_devrelease]`. Hiding the filter inside the
  method would make the API disagree with the catalog; we don't.
- `dataset_history()` results are **sorted by `dataset_version`
  ascending**. Reads forward in time; `[0.1.0, 0.2.0, ...,
  0.4.0.post1.dev3]`. Today's order is whatever the catalog
  returns — that's a fragile contract and the change is net
  positive.
- `current_version` keeps using `max(history)`. Under PEP 440
  ordering (ADR-0004), the dev version sorts after the last
  released version, so `max` correctly returns the dev version
  when there is one.
- `current_version`'s defensive fallback to `DatasetVersion(0, 1,
  0)` for empty history is **removed**. `create_dataset` always
  inserts a version row, so empty history is a catalog
  inconsistency — raise rather than silently invent a version.
  This matches CLAUDE.md's "no defensive code for impossible
  cases" rule.
- No new read methods. No `current_dev_version()`, no
  `released_history()` filter helper. Dev rows are not
  second-class; they're a different state of the same row type.
- Display methods (`to_markdown`, etc.) render PEP 440 version
  strings literally. `"0.4.0.post1.dev3"` is what the version
  *is*; introducing a separate display form would require keeping
  it in sync with the canonical form.

## Concurrency model

`release()` and dev-row mutations both modify the same row (the
dev row). Concurrent writers are reconciled at the database level
via ERMrest's row-level conditional updates, not by application
locking:

- Each writer reads the dev row's `RMT` (row-modified time) before
  acting.
- Each writer's `UPDATE` carries `WHERE RID=<dev_row_rid> AND
  RMT=<observed>`. If a competing writer landed first, the
  predicate matches zero rows; the update is a no-op and the
  caller raises a clear "concurrent modification" error pointing
  at re-reading the dataset.
- The framework does **not** auto-retry. A concurrent release
  between read and write may change the *meaning* of the caller's
  operation (they thought they were mutating the dev period after
  `0.4.0`; the dataset has since been released to `0.5.0` and any
  new mutation would land on a fresh dev period after that). The
  caller decides whether the new state is still what they want.
- A mutation that arrives on a *just-released* dataset is **not** a
  conflict — it's the normal lazy-dev-row case. The mutation
  observes a released row, creates a new dev row at
  `<just_released>.post1.dev1`, and proceeds.

The key reason for using ERMrest's row-level concurrency rather
than an application-level `Status` column or a `Dataset` row
generation counter: the database already solves this; introducing
a parallel locking scheme adds schema and bug surface for a corner
that the conditional-update primitive handles cleanly.

## Migration impact

This work is a 2.0 breaking change. There is no DDL change (the
schema's `Snapshot` column is already nullable), but the
*semantics* of several existing public methods change. The
behavior-change inventory:

| Existing call | Today | New |
| --- | --- | --- |
| `add_dataset_members` / `delete_dataset_members` | Bumps to a new released version | Lands on a dev version (creates `.dev1` if needed, advances `.devN` if existing) |
| `increment_dataset_version(component, description, execution_rid=None)` | Public mixin method on `DerivaML`, creates a released row | Renamed to `Dataset.release(bump, description, execution=None)`. Instance method on `Dataset`. Errors if no dev row. `execution_rid: RID` → `execution: Execution`. |
| `Dataset.current_version` | Returns latest released | Returns dev when present, else latest released |
| `Dataset.dataset_history()` | Released rows in unspecified order | All rows (dev + released) sorted ascending |

Downstream callers must:

1. **Find every call to `increment_dataset_version`** and rewrite
   to `Dataset.release()`. Mechanical rename plus signature
   update.
2. **Audit any logic that assumes "the version after a mutation
   is the released version"** — that's no longer true. To produce
   a released version, follow the mutation with an explicit
   `release()` call.
3. **Audit any `find_datasets`/`dataset_history` consumers** that
   filter on "released versions" — they'll need to add an explicit
   `is_devrelease` filter rather than relying on every version
   being released.

Out-of-repo blast radius (handled in dependent PRs, not this ADR):
`deriva-ml-mcp`'s tool that exposes `increment_dataset_version`,
the model-template workflows, and any user notebooks. The 2.0
changelog and migration guide are written at PR-ready time and
point users at this ADR.

## Schema-creation and validation touch points

Implementation must update these places to reflect the new
contract on `Dataset_Version`, even though the column types do not
change:

- **`src/deriva_ml/schema/create_schema.py`** — the `Snapshot`
  column's `comment` should make the new contract explicit:
  populated for released rows, `NULL` for dev rows. The comment
  today is "Catalog Snapshot ID for dataset", which is silent on
  the nullable case.
- **`src/deriva_ml/schema/create_schema.py`** — the `Version`
  column's `comment` ("Semantic version of dataset") and `default`
  ("0.1.0") should be updated. The label is no longer "semantic
  version" (that's semver-specific); it's a PEP 440 version string.
  Released rows carry `MAJOR.MINOR.PATCH`; dev rows carry
  `<last_release>.post1.devN`.
- **`src/deriva_ml/schema/validation.py`** —
  `EXPECTED_TABLE_COLUMNS[MLTable.dataset_version]["Snapshot"]` is
  already `("text", True)` (nullable). Confirm this is the expected
  contract; no change needed unless a stricter expectation is
  desired for released rows (which would require splitting the
  validator's view into "released-row expectation" vs
  "dev-row expectation" — out of scope for this ADR).

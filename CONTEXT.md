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

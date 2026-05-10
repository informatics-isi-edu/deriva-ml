# ADR-0005: Dev-versioning delivery — seven sequenced PRs

Date: 2026-05-09
Status: Accepted (with one deviation at delivery — see "Delivery outcome" below)

## Delivery outcome (added 2026-05-10)

The plan below described a **2.0 major-version bump**. At delivery
time, the maintainer chose to ship the work as a **1.34.0 minor bump**
instead. The breaking changes (renamed `release()` method, dev-flip
behavior for member mutations, changed `current_version` semantics,
etc.) are unchanged — what changed is the package version label
attached to them. References to "2.0" in the ADR body below describe
the plan at the time it was written. The release-notes and migration
guide use the actual shipped version (1.34.0).

## Context

The dev-versioning work (ADR-0003) and the PEP 440 rebase
(ADR-0004) together define a 2.0 breaking change for `deriva-ml`.
The diff is sizeable and touches multiple concerns — versioning
arithmetic, schema documentation, dev-row creation, mutation
behavior, the public method rename, drift detection, and the
release ceremony. Landing it as a single PR would be unreviewable
and would gate the version bump on tests passing for everything at
once. This ADR records the slicing.

## Decision

The work lands as **seven sequenced PRs**, each leaving the
codebase in a working state, with breaking changes confined to
PR 4 and PR 5. The version bump to 2.0 is deferred to PR 7.

| # | Title | Breaking? |
| --- | --- | --- |
| 1 | `DatasetVersion` rebases on PEP 440 (ADR-0004) | No |
| 2 | Schema column comments — `Snapshot` and `Version` contracts | No |
| 3 | `mark_dev` + `current_version`/`dataset_history` updates | No |
| 4 | Member mutations land on dev versions | **Yes** |
| 5 | Rename `increment_dataset_version` → `Dataset.release()` | **Yes** |
| 6 | `is_dirty` / `release_diff` / `compare_versions` drift detection | No |
| 7 | Migration guide, changelog, `bump-version major` to 2.0 | (release) |

### PR 1 — PEP 440 rebase

`DatasetVersion` switches its parent class from `semver.Version` to
`packaging.Version`. `bump_major`/`bump_minor`/`bump_patch` are
replaced by `next_release(bump: VersionPart)`. The `semver`
dependency is dropped. **Non-breaking** at the public API surface
— released-version strings parse and serialise identically.

### PR 2 — Schema column comments

`Snapshot.comment` spells out the dev-row contract: populated for
released rows, `NULL` for dev rows. `Version.comment` references
PEP 440 rather than "semantic versioning". **Non-breaking** — comments
only.

### PR 3 — `mark_dev` and read-side updates

Adds `Dataset.mark_dev(description, execution=None)`. Adds the
shared `_create_or_advance_dev_row` helper. Updates
`current_version` and `dataset_history()` per ADR-0003's Read-side
surface section: drop the empty-history fallback, sort history
ascending. **Non-breaking** — additive plus minor read-side
behavior changes that don't break correct callers.

### PR 4 — Member mutations land on dev

`add_dataset_members` and `delete_dataset_members` replace their
`increment_dataset_version` tail call with the
`_create_or_advance_dev_row` helper from PR 3. Tests for these
methods are rewritten — they currently assert the post-mutation
version is the next released version; they will now assert it is a
dev version. **Breaking.**

### PR 5 — Rename to `Dataset.release()`

`increment_dataset_version` (mixin method on `DerivaML`) is
removed. `Dataset.release(bump, description, execution=None)` is
added. The method errors when called on a dataset with no dev row.
In-repo callers are updated in this PR; out-of-repo callers
(deriva-ml-mcp, model-template, user notebooks) are filed as
follow-ups. **Breaking.**

### PR 6 — `is_dirty` / `release_diff` / `compare_versions`

The drift-detection methods. Reuse `CatalogGraph`'s path
enumeration; differ only in the `RMT` time predicate. All three
share one internal `_diff_between(t_lower, t_upper)` helper:

- `is_dirty()` — `bool`; predicate `RMT > T_last_release`;
  short-circuits on first non-zero count.
- `release_diff()` — per-table change counts; same predicate as
  `is_dirty`; walks full closure.
- `compare_versions(v_a, v_b)` — per-table change counts between
  two released versions; predicate spans both endpoints.

All three live on `Dataset` only, not on `DatasetLike` (bags are
never dirty by construction). **Non-breaking** — additive.

The shared helper means the marginal cost of including
`compare_versions` in this PR is small — the same walk is reused;
only the predicate changes. Deferring it would cost more in
context re-establishment than landing it now.

### PR 7 — User guide, migration guide, and 2.0 bump

Updates the user guide pages whose content is invalidated by the
new model:

- `docs/user-guide/datasets.md` — the "How to version a dataset"
  section. Rewrite the worked example using `release()`,
  `mark_dev()`, `is_dirty()`, `release_diff()`. Replace the
  "every `add_dataset_members` call increments the version" claim
  with the dev-version model. Refresh the surrounding prose so
  the chapter teaches the model, not the old API.
- `docs/user-guide/reproducibility.md` — the version-pinning story.
  Released versions stay reproducible (snapshot-pinned); the new
  thing to teach is that dev versions are *not* reproducible
  references and shouldn't be passed to executions.
- `docs/user-guide/migration.md` — extend with the 2.0 migration
  story: the rename, the behavior change for `add_dataset_members`,
  the addition of `mark_dev`/`is_dirty`/`release_diff`/`compare_versions`.
  Worked before/after examples for the common cases.
- `docs/api-reference/*.md` — these pages are mkdocstrings-driven
  and pick up changes from the docstrings; no direct edits beyond
  ensuring the source docstrings are right.

(`docs/concepts/datasets.md` was checked and does not currently
reference versioning, so no edit is needed there. Re-verify before
2.0 ships in case it has changed.)

Changelog entries reference the migration guide. `uv run bump-version
major` produces 2.0 once PRs 1–6 have all merged, the user-guide
edits are in, and `main` is green.

## Documentation as a deliverable

Every method whose behavior changes in PRs 3–6 ships **with** an
updated docstring in the same PR. No "I'll update the docs in a
follow-up." The docstring bar:

- **Updated** — rewritten to match the new contract, not patched
  on top of the old one.
- **Accurate** — matches the implementation and the ADR.
- **Understandable without ADR context** — written for a notebook
  user who hasn't read the design history. The ADR explains *why*;
  the docstring explains *what does this do*.

Each docstring follows the template the `release_diff` docstring
(in ADR-0003 / commit message of PR 6) sets:

1. One-line summary that names the method's purpose.
2. Paragraph explaining the meaning (what counts, what doesn't).
3. Paragraph naming the use case (when you'd reach for this).
4. Paragraph on mechanism (demoted, for readers who need it).
5. Limitations explicitly listed.
6. `Args` / `Returns` / `Raises` sections per the workspace's
   Google-style policy.
7. `Example` block — runnable, or `# doctest: +SKIP` for
   catalog-dependent examples.

Methods that need a fresh docstring under this bar:
`add_dataset_members`, `delete_dataset_members`,
`Dataset.current_version`, `Dataset.dataset_history`,
`Dataset.release` (new), `Dataset.mark_dev` (new),
`Dataset.is_dirty` (new), `Dataset.release_diff` (new),
`Dataset.compare_versions` (new). Each lands in the PR that adds
or changes the corresponding behavior — not in PR 7.

## Out-of-repo follow-ups

Filed as dependent issues, fired after the relevant PRs land:

- **deriva-ml-mcp** — rename the `deriva_ml_increment_dataset_version`
  tool to `deriva_ml_release`, update its signature and bundle
  resource shape. Blocked on PR 5.
- **Cite URL plumbing** — `cite_url` field on `ExecutionAssetRef`,
  `ExecutionInputDatasetRef`, and `DatasetDetail`, plus
  `Dataset.cite()`. Blocked on PR 5 + PR 6 (needs both `is_devrelease`
  and the released-version contract).
- **Deletion-aware diff** — closes the deletion-detection gap in
  `release_diff` and `compare_versions` by querying the relevant
  snapshots for RID sets and computing set differences. Different
  cost profile (proportional to reachable rows on each endpoint,
  not just to changes). Filed separately when the need is
  concrete.

## Rationale

- **Two breaking PRs (4 and 5) rather than one.** PR 4 changes
  what `add_dataset_members` does; PR 5 renames the release verb.
  Bundling them would make the breaking diff harder to review and
  harder to revert if either turned out to be wrong. Splitting
  also lets PR 4's tests be reviewed before PR 5's signature work
  layers on top.
- **The 2.0 bump is its own PR.** Landing the bump in the same PR
  as one of the breaking changes would couple the release ceremony
  to that PR's review cycle. PR 7 is small, mechanical, and lands
  only when `main` is green.
- **Non-breaking PRs flank the breaking ones.** PRs 1–3 prepare
  the ground; PRs 4–5 do the breaking work; PR 6 builds on it; PR
  7 closes it out. A reviewer reading the sequence sees the design
  unfold, not just the final state.

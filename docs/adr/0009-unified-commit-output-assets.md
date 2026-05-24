# ADR-0009: One method to commit execution outputs

Date: 2026-05-23
Status: Accepted

## Context

By mid-2026, deriva-ml had accumulated **three different methods**
named like upload entry points on the execution surface, each with
different semantics:

| Method | Class | Returns | Lifecycle work |
|---|---|---|---|
| `upload_execution_outputs(clean_folder, progress_callback)` | `Execution` | `dict[str, list[AssetFilePath]]` | Full: status transitions, descriptions, Upload_Duration, folder cleanup |
| `upload_outputs(retry_failed)` | `Execution` | `UploadReport` | None |
| `upload_outputs(ml=, retry_failed=)` | `ExecutionSnapshot` | `UploadReport` | None |
| `upload_pending(execution_rids, retry_failed)` | `DerivaML` | `UploadReport` | None |

All four eventually bottomed out at the same private method
`Execution._bag_commit_upload()`, which performs the actual bag-build
+ `BagCatalogLoader` + Hatrac PUT work. The lifecycle bracketing
(status transitions, asset description writes, Upload_Duration
recording, working-folder cleanup) lived only in the in-script path
(`upload_execution_outputs`); the other three skipped it entirely.

The names were not just confusing — they encoded **two real
behavioral bugs**:

1. **CLI-uploaded executions stayed `Stopped` forever.**
   `deriva-ml-upload --execution X` ran `upload_pending([X])` →
   `_bag_commit_upload()`. The bag-commit succeeded, the rows landed
   at the catalog, the manifest got marked uploaded — but the
   execution's status was never transitioned to `Uploaded`. Any
   code that filtered on `status=Uploaded` (e.g.
   `gc_executions(status=ExecutionStatus.Uploaded)`) silently missed
   every CLI-uploaded execution.

2. **`exe.upload_outputs()` silently skipped asset description
   writes.** Any caller following the method's own docstring example
   (`with exe.execute() as e: ...; exe.upload_outputs()`) got a
   successful-looking `UploadReport` while losing every asset
   description set via `asset_file_path(description=...)`,
   skipping the `Upload_Duration` provenance write, and leaving
   the execution stuck in `Stopped` status.

How we got here: `upload_execution_outputs` was the original method.
The `upload_outputs` / `upload_pending` / `start_upload` surface was
added later (`feat(upload): public API`, 2026-04-20) to support
**out-of-process upload via `deriva-ml-upload`** and **workspace
batch flush**. The new surface should have called the existing
lifecycle bracket; it didn't. The two methods coexisted, the
lifecycle drifted between them, and the bugs sat latent in code
paths that worked in tests but produced wrong end-state in
production.

A May 2026 audit (`docs/audits/2026-05-22-engineer-audit-execution.md`)
caught duplication around `_format_duration` and `_set_asset_descriptions`
but did not surface the upload-method duplication directly. The
sibling-repo audit at `deriva-ml-skills/docs/superpowers/specs/
2026-05-18-skill-update-assessment.md` §1.5 *did* flag the smell —
titled *"upload_execution_outputs(...) is legacy"* — and recommended
that skills standardize on `upload_outputs`. **That recommendation
was the wrong direction**: skills moving to `upload_outputs` would
have spread the lifecycle bug to every Claude Code agent following
those skills.

## Decision

There is **one method** that commits an execution's output assets
to the catalog. It takes an `Execution` and brings it from "work
done, status `Stopped`" to "everything visible at the catalog,
status `Uploaded`, `Upload_Duration` recorded, asset descriptions
written, working folder optionally cleaned."

### The unified surface

#### Per-execution

```python
class Execution:
    def commit_output_assets(
        self,
        clean_folder: bool | None = None,
        progress_callback: Callable[[UploadProgress], None] | None = None,
    ) -> "UploadReport":
        ...
```

The method is idempotent. Re-running after a previous success
(status=`Uploaded`, no pending) is a no-op that returns an empty
report. Re-running after a partial failure resumes from the last
known-good state — `BagCatalogLoader`'s `match_by_columns` dedup
makes row inserts idempotent at the catalog.

The method raises on failure. Failure isolation is the batch
caller's job (see below), not the per-execution call's.

#### Batch

```python
class DerivaML:
    def commit_pending_executions(
        self,
        *,
        execution_rids: list[RID] | None = None,
        clean_folder: bool = False,
    ) -> "UploadReport":
        for rid in (execution_rids or self._list_pending_rids()):
            try:
                ml.resume_execution(rid).commit_output_assets(
                    clean_folder=clean_folder,
                )
            except Exception as e:
                # aggregate into UploadReport.errors, continue
```

Per-execution failure isolation: a failure on execution A does
not skip execution B; both outcomes appear in the returned
`UploadReport`. The CLI (`deriva-ml-upload`) is a thin wrapper
around this method.

### What's removed

| Removed | Replaced by |
|---|---|
| `Execution.upload_execution_outputs(...)` | `Execution.commit_output_assets(...)` |
| `Execution.upload_outputs(...)` | `Execution.commit_output_assets(...)` |
| `ExecutionSnapshot.upload_outputs(ml=...)` | `ml.resume_execution(snap.rid).commit_output_assets()` |
| `DerivaML.upload_pending(...)` | `DerivaML.commit_pending_executions(...)` |
| `retry_failed=` kwarg (everywhere) | Removed — was already a documented no-op under the bag pipeline |

**No deprecation shims.** The breaking change ships as **v1.39.0**
— a minor bump that carries an API-incompatible change. Major-
version (`v2.0.0`) is intentionally deferred until the unified
surface has more bake time; landing the rename at v1.39 keeps the
v2.0 marker free for a future release we are more confident in.
Internal callers (10 source files) migrate in the same PR.
External Python consumers (`deriva-mcp`,
`deriva-ml-model-template`) migrate in coordinated companion PRs.

### Why no shims

Shims would have lengthened the period during which both names
were in use and the lifecycle behavior was inconsistent. The
goal of this ADR is to end the split-brain, not extend it. The
external blast radius (6 production call sites across 2 sibling
repos) is small enough to coordinate via PRs in a single
session.

### Why the name

- **`commit_`** matches the internal terminology already used in
  this code path: `bag_commit.py`, `_bag_commit_upload`,
  `bag_commit_upload`, the module docstring's "the commit
  pipeline." The public method finally aligns with the internal
  vocabulary.
- **`_output_assets`** names what gets shipped, and ties to the
  Asset_Role contract pinned in PR #220
  (`docs/user-guide/executions.md` §"How execution-asset roles
  work"): every committed asset gets `Asset_Role="Output"` on
  its `{Asset}_Execution` row plus the directional
  `Output_File` Asset_Type tag. The name `commit_output_assets`
  makes the role-contract self-documenting at the call site.
- The pairing is symmetric:
  - **Input side:** `Execution.download_asset()` brings in
    Input-role assets, tags them `Input_File`.
  - **Output side:** `Execution.commit_output_assets()` ships
    Output-role assets, tags them `Output_File`.

### Why `commit_pending_executions` for the batch

- Names the scope precisely (executions, not rows or files —
  the workspace registry might have many of each per
  execution).
- Pairs with single-execution `commit_output_assets` as the
  loop sibling.
- Honest about what it does: drain every execution in the
  registry that has pending work, or the explicit subset.

## Consequences

**Positive:**

- One method, one semantics. Inline scripts, post-resume calls,
  and the CLI all produce the same end state.
- The two latent bugs (CLI executions stuck `Stopped`,
  `upload_outputs` skipping descriptions) cannot exist under the
  new surface — there is no code path that does the work
  without the lifecycle bracket.
- New test surface (`tests/execution/test_commit_lifecycle.py`)
  asserts the three callers (inline, post-resume, CLI) all
  produce identical end state. Pins the unification in CI.
- Method name self-documents the Asset_Role contract.

**Negative:**

- Breaking change. Every external Python caller of the four
  removed methods must migrate at v1.39.0 upgrade. No shim.
- Internal-caller migration is in the same PR as the rename —
  the PR is larger than a minimum-viable behavior fix would have
  been.
- Sibling-repo PRs (`deriva-mcp`, `deriva-ml-model-template`)
  must merge within a coordinated window or risk a transient
  broken pin.

**Neutral:**

- `_bag_commit_upload` (the private implementation) is unchanged
  — only callers move.
- `Execution._set_asset_descriptions`, `_format_duration`, the
  `Pending_Upload → Uploaded` state-machine transitions are
  unchanged; they're now reached through one code path instead
  of two.
- `UploadReport.per_table` is now populated for successful
  drains (it was previously always `{}` for success per an
  explicit comment in `mixins/execution.py:844`).

## Alternatives considered

### Patch `upload_outputs` / `upload_pending` to do the lifecycle, keep both names

Mechanically possible — extract the lifecycle bracket into a
helper, call from both. Fixes both bugs without renaming
anything.

Rejected: leaves the public surface confusing. A user reading the
API still sees two methods that look like aliases but aren't, and
the next developer adding a third upload path repeats the same
mistake. The names also don't align with the underlying `commit`
vocabulary or the `Output` role contract — the chance to make
both visible at the call site is gone.

### Keep `upload_execution_outputs`, delete the others

Smaller surface change. Preserves the most common name (133 test
sites + 5 production callers).

Rejected: `upload_execution_outputs` says "execution" twice (it's
a method *on* `Execution`); the redundancy was an artifact of
when it lived on `DerivaML`. The breaking-rename window is the
right time to drop it.

### Auto-commit on `__exit__`

Make the `with exe.execute() as e:` block automatically commit
output assets on exit, so users never need to call
`commit_output_assets` directly.

Rejected: the user explicitly requested an explicit call. Reasons:
visible step, skippable for dry-runs or inspect-before-commit
flows, no magic when an exception in the `with` block would
otherwise commit a half-done execution.

### Deprecation shims for one minor cycle

Standard semver hygiene — ship with shims that warn, remove a
minor cycle later.

Rejected by the user: shims extend the period during which both
names are in use. The blast radius is small enough that a single
coordinated PR session resolves all known external consumers.
Shims would have kept the broken methods reachable in any pinned
v1.39+ environment.

### Ship as v2.0.0

Initial intent — semver-honest. A breaking API change is what
v2.0 numerals signal.

Rejected by the user **after** the initial v2.0.0 tag was pushed:
v2.0 should mark a release we are confident enough in to stand
behind for years; the unified surface needs bake time before it
earns that marker. The v2.0.0 tag was deleted from origin and
this commit was re-tagged as v1.39.0. The release notes call out
that v1.39.0 carries an API-incompatible change despite the minor
version number; consumers must read them before upgrading.

## References

- Release notes for v1.39.0 — migration table.
- `tests/execution/test_commit_lifecycle.py` — the contract test.
- `docs/user-guide/executions.md` §"Committing output assets" —
  user-facing documentation of the unified surface.
- ADR-0006 (`bag-oriented-data-movement`) — the underlying
  upload pipeline this method drives.
- PR #220 — the Asset_Role / `Output_File` directional-tag
  contract that this method's name now self-documents.

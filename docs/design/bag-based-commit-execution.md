# Bag-based `commit_execution` — design

## Goal

Replace the end-of-execution upload pipeline behind
`Execution.upload_execution_outputs` with the same two-step
bag flow used by clone-via-bag:

1. **Build** a bag from execution outputs via `BagBuilder`.
2. **Load** the bag into the destination catalog via
   `BagCatalogLoader.run()`.

Same machinery as use-case 1 (clone), reused for use-case 2
(commit). See ADR-0006 for the broader bag-oriented data
movement context.

## Why this is viable now

Earlier evaluation rejected this approach because
`BagBuilder.add_asset` does `shutil.copy2` on asset bytes —
2× disk during commit, real cost for ML workloads with
multi-GB output assets. The current pipeline uses `shutil.copy2`
only as a fallback; the primary mechanism is
`target.symlink_to(source.resolve())` (`execution.py:1231`).

Two upstream changes make the bag-based path cost-equivalent
to today's pipeline:

- **`BagBuilder.add_asset(link=True)`** — opt-in hardlink mode.
  Asset bytes get a second directory entry inside the bag
  pointing at the same inode; zero bytes duplicated. bagit
  sees a regular file (which it is — just sharing storage with
  the source), so the bagit security check passes. MD5
  manifest is correct because hashlib reads through to the
  shared inode's actual content.

  Hardlinks rather than symlinks: bagit's
  `_validate_bag_contents` rejects manifest entries whose
  resolved path escapes the bag root for security reasons.
  Symlinks pointing at external flat-asset storage fail this
  check. Hardlinks live inside the bag tree as ordinary file
  entries — same storage, different name. The link must be
  on the same filesystem as the source; the implementation
  falls back to copy with a warning on EXDEV.

- **`DanglingFKStrategy.PRESERVE`** — fourth dangling-FK
  strategy that skips the bag-side parent-row check entirely
  and trusts the destination catalog's FK constraint to be
  authoritative. Necessary because execution outputs reference
  parent rows (Subject, Observation, Workflow) that already
  exist at the destination but are out-of-bag by construction.

With these, the cost/benefit shifts:

| Aspect | Copy-based bag (rejected earlier) | Hardlink + PRESERVE (this proposal) |
|---|---|---|
| Disk overhead during commit | 2× | ~0 (one extra inode per asset) |
| I/O overhead | 2× (copy then upload) | 1× (upload only) |
| Code footprint reduction | ~1000 lines | ~1000 lines |
| Code reuse with clone | Yes | Yes |
| Durable bag artifact | Yes | No (intentional — bag is transient) |
| Cross-filesystem | Works | Falls back to copy + warning |
| Passes bagit validation | Yes | Yes (hardlinks look like regular files) |

The "durable bag artifact" tradeoff is real but not a v1
parity feature. Discarding the bag post-upload is correct
for commit semantics.

## Footprint estimate

| File / method | Action | Lines |
|---|---|---|
| `asset/null_sentinel_processor.py` | Delete | -41 |
| `execution/upload_engine.py` | Delete | -865 |
| `execution/upload_job.py` | Delete | -138 |
| `execution.py::_build_upload_staging` | Delete | -65 |
| `execution.py::_cleanup_upload_staging` | Delete | -10 |
| `execution.py::_upload_execution_dirs` | Replace (-120 / +30) | -90 |
| `execution.py::_update_asset_execution_table` | Refactor (-70 / +50) | -20 |
| `execution.py::_flush_staged_features` | Refactor (-50 / +30) | -20 |
| `execution.py::_build_execution_bag` (new) | Add | +60 |
| `execution.py::_load_execution_bag` (new) | Add | +30 |
| **Total** | | **~-1160 net** |

`_set_asset_descriptions` (~10 lines) is unchanged.

## What the new pipeline does

`upload_execution_outputs` becomes:

1. **Auto-stop / status-skip / `Pending_Upload` transition** —
   unchanged. The state machine doesn't care how we upload.
2. **`_build_execution_bag()`** —
   - Lease pre-allocated RIDs for any pending manifest entries
     that don't have one (existing
     `manifest_lease.lease_manifest_pending_assets` logic).
   - Validate NOT-NULL metadata
     (existing `_validate_pending_asset_metadata`).
   - Create a `BagBuilder` rooted at
     `working_dir/.commit-bag/`. Pass the destination catalog's
     model via `metadata=...` so BagBuilder knows the schema.
   - For each pending manifest entry:
     - `bag.add_asset(table=..., rid=..., source_path=flat_file,
       link=True)` — hardlinks the asset bytes into the bag.
     - `bag.add_row(table=asset_table_name, row=image_row)` —
       synthesizes the catalog row from manifest metadata + leased
       RID + filename + uploaded asset's URL placeholder (filled
       by `BagCatalogLoader._upload_assets` at load time).
   - Generate `{Asset}_Execution` association rows from the
     manifest's `(rid, execution_rid)` pairs. Add via
     `bag.add_rows(table="{Asset}_Execution", rows=[...])`.
   - Generate `{Asset}_Asset_Type` association rows from the
     manifest's asset_types lists. Add via `add_rows`.
   - Read staged feature records from `manifest_store.feature_staging`,
     rewrite asset-column local-filename values to the
     pre-leased RIDs (already known — no post-upload lookup
     needed), add via `add_rows`.
   - `bag.finalize(make_bdbag=False)` — writes bagit manifest
     (computes MD5 over the hardlink targets). No archive step.
3. **`_load_execution_bag()`** —
   - `BagCatalogLoader(catalog=self.ml.catalog, bag=bag_path,
     policy=FKTraversalPolicy(
       asset_mode=UPLOAD_IF_MISSING,
       dangling_fk_strategy=PRESERVE,
       vocab_export=REFERENCED_ONLY,  # we only insert refs to terms
     ))`.
   - `loader.run()` does FK-safe insert ordering, two-phase
     cycle handling, hatrac PUT via `_upload_assets`, vocab
     reconciliation by name (no-op for commit since we don't
     send vocab rows).
   - Read the load report → mark manifest entries as uploaded
     (`mark_uploaded_batch`).
4. **`_set_asset_descriptions`** — unchanged.
5. **`Pending_Upload → Uploaded`**. Clean folder. Discard bag dir.

## Upstream changes required

### `deriva-py`: `BagBuilder.add_asset(..., link: bool = False)`

Add a `link` parameter, default `False` (preserves existing
copy-based behavior).

```python
def add_asset(
    self,
    table: str,
    rid: str,
    source_path: Path,
    *,
    filename: str | None = None,
    link: bool = False,
) -> None:
    ...
    dest.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source_path, dest)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            # Cross-filesystem — fall back to copy with warning.
            logger.warning("...")
            shutil.copy2(source_path, dest)
    else:
        shutil.copy2(source_path, dest)
```

Trade-offs documented in the BagBuilder docstring:

- `link=False` (default): self-contained bag, portable,
  archivable. Pays disk + I/O for the copy.
- `link=True`: hardlink mode. Bag shares storage with the
  source files (one inode per asset, two directory entries).
  Bagit sees regular files (hardlinks look like files at every
  filesystem-level definition) and validation passes. MD5
  manifest matches source content exactly. Falls back to copy
  on cross-filesystem `OSError(EXDEV)`. Hardlinks survive
  source unlinking — the bag entry keeps the inode alive
  until it too is unlinked.

### `deriva-py`: `DanglingFKStrategy.PRESERVE`

Add a fourth strategy to the existing enum:

```python
class DanglingFKStrategy(StrEnum):
    FAIL = "fail"
    DELETE = "delete"
    NULLIFY = "nullify"
    PRESERVE = "preserve"  # new
```

Semantics: for FKs whose parent table is in `bag_schemas` but
the parent row isn't in the bag, **assume the destination
catalog already has the row**. Skip the bag-side validation
entirely. If the destination is wrong, ERMrest's FK constraint
fires at insert time and the load fails with the real HTTP 409
(no need to pre-emptively validate).

Implementation: short-circuit in `_apply_dangling_fk_strategy`:

```python
if self.policy.dangling_fk_strategy == DanglingFKStrategy.PRESERVE:
    return rows, 0, 0  # trust destination
```

Use case: end-of-execution commits. Image rows reference
Subject FKs that exist at the destination but were created
in an earlier execution — by construction, the new bag
doesn't carry Subject rows.

The existing FAIL/DELETE/NULLIFY semantics are unchanged.

## What stays

- `Execution` state machine, status transitions, dry-run guard.
- `asset_file_path()` API — unchanged. Callers still write to
  the flat `assets/{table}/` storage; the bag pipeline collects
  from there.
- `ManifestStore` schema and lifecycle. The bag-build step is
  a new consumer; nothing in the manifest changes.
- `_set_asset_descriptions` — runs after bag load with
  uploaded RIDs known.
- Return shape of `upload_execution_outputs`:
  `dict[str, list[AssetFilePath]]` keyed by `"{schema}/{table}"`.
  Built from the bag-load report + the local manifest. Callers
  (demo_catalog, runner, split.py) don't change.

## What goes away

- `null_sentinel_processor.py` — the NULL-via-regex hack is
  unnecessary because the bag loader inserts directly from CSVs
  with proper NULL handling via `_coerce_empty_to_null`.
- `upload_engine.py` (entirely) — `BagCatalogLoader.run()` does
  the same job: FK-safe insert order via `ForeignKeyOrderer`,
  two-phase deferred FK updates for cycles, deterministic
  retry semantics via the bag's idempotent `ON CONFLICT DO NOTHING`.
- `upload_job.py` (entirely) — the threaded job wrapper around
  `run_upload_engine`. `BagCatalogLoader` runs async natively
  and the existing `upload_execution_outputs` signature stays
  synchronous-with-progress-callback.
- `_build_upload_staging` + `_cleanup_upload_staging` — replaced
  by `BagBuilder`'s own directory management.

## Comparison: `ForeignKeyOrderer` vs `CatalogBagBuilder` walk

Both pieces of upstream machinery handle FK relationships, but
they serve different roles. Worth pinning down their behaviors
since the new commit pipeline depends on both:

| Aspect | `ForeignKeyOrderer` | `CatalogBagBuilder._compute_reached_tables` |
|---|---|---|
| Direction | Outbound (FKs the table declares) | Bidirectional (out + inbound) |
| Purpose | Decide insert order for a known set | Discover which tables to fetch |
| Output | Sorted list | Set + per-target FK-path list |
| Cycle handling | Drop edges, retry sort | Simple-path guard, `max_paths` cap |
| Terminal tables | N/A | Vocab + `policy.terminal_tables` |
| Self-references | Skipped (cycles-of-one) | Caught by simple-path test |

For commit_execution we hand `BagCatalogLoader` a bag whose
table set is **known** (Image rows, association rows, feature
rows, optionally Execution). The loader's internal
`ForeignKeyOrderer` decides insert order over that known set.
The walker isn't involved on the commit path — the bag content
is constructive, not discovered.

Both algorithms agree on the FK direction semantics
(`table.foreign_keys` = "this table depends on those"), so
they compose correctly: BagBuilder's `add_row` calls determine
the candidate set; ForeignKeyOrderer determines the order.

## Open questions

### Q1. Feature records — translation timing

Today: `_flush_staged_features` reads `feature_staging`,
rewrites asset-column values from local filenames to uploaded
RIDs (using `uploaded_files: dict[str, list[AssetFilePath]]`),
and batch-inserts via the datapath.

In the bag approach: feature records get the asset-column
rewrite at **bag-build time**. RIDs are pre-leased, so the
asset RIDs are known before upload. The bag loader inserts
them verbatim.

**Decision**: do the rewrite at bag-build time. Simpler.

### Q2. Additive upload (`Uploaded → Pending_Upload`)

Caller registers more assets after a first upload — today
we add only the new entries.

The bag pipeline handles this naturally: the second bag
contains only the new manifest entries (those still in
`pending` status). The loader is idempotent on rows it
has already inserted (`ON CONFLICT DO NOTHING`).

**Decision**: no special handling. Build a fresh bag per
commit call. Filter to `pending` entries at bag-build time.

### Q3. Crash recovery mid-upload

Bag-based pipeline: if the loader crashes mid-load, the bag
still exists on disk. Re-running `upload_execution_outputs`
rebuilds the bag (same content — pending entries didn't get
marked uploaded) and the loader retries from a clean state,
with HEAD-then-PUT (`UPLOAD_IF_MISSING`) skipping bytes
already in hatrac.

**Decision**: rely on bag's natural idempotency. Drop the
`upload_engine.py` SQLite-staged retry harness.

### Q4. Execution row in the bag? — RESOLVED

The Execution row is inserted into the destination catalog at
**execution-start time**, inside
`Execution._initialize_execution()` (execution.py:274):

```python
self.execution_rid = schema_path.Execution.insert([{
    "Description": self.configuration.description,
    "Workflow": self.workflow_rid,
    "Status": str(ExecutionStatus.Created),
}])[0]["RID"]
```

The `Dataset_Execution` rows (linking input datasets to the
execution) are inserted at the same phase (execution.py:555).

**Therefore at commit time the destination catalog already has:**

- The `Execution` row.
- Its `Workflow` FK target (also pre-existing — workflows are
  catalog-level long-lived objects).
- The `Dataset_Execution` rows for input datasets.

**The commit bag carries only:**

- Output asset rows (Image, BoundingBox, etc.).
- `{Asset}_Execution` association rows (Output role).
- `{Asset}_Asset_Type` association rows.
- Feature rows (asset RIDs pre-resolved via leased mapping).

Every FK in the bag rows points at either (a) a row also in the
bag, or (b) a row that already exists at the destination
(Execution, Workflow, Subject, Observation). `PRESERVE` strategy
handles case (b) by skipping bag-side validation.

### Q5. Bag location

Where on disk should the commit bag live?

Options:
- `working_dir / ".commit-bag" / execution_rid /` — co-located
  with execution state. Cleaned up by `_clean_folder_contents`.
- `tempfile.TemporaryDirectory()` — auto-cleanup on Python exit.

Option 1 lets a developer inspect the bag after a failed
commit. Option 2 is tidier. Option 1 is more debug-friendly.

**Recommendation**: option 1, gated by an env flag
`DERIVA_ML_KEEP_COMMIT_BAG=1` for post-commit retention;
otherwise auto-deleted on success.

## Delivery sequence

### PR-A (deriva-py): `BagBuilder.add_asset(link=True)`

- Add `link` parameter to `add_asset` and `add_assets`.
- Update docstring with the trade-off matrix.
- Unit test: hardlink mode creates a link, MD5 manifest is
  correct, the dest path matches the copy mode's path.

Small, mechanical, low risk.

### PR-B (deriva-py): `DanglingFKStrategy.PRESERVE`

- Add `PRESERVE` to the enum.
- Short-circuit in `_apply_dangling_fk_strategy`.
- Unit test: rows with FKs into in-schema but out-of-bag
  parents are kept verbatim under `PRESERVE`.

Small, mechanical, low risk.

### PR-C (deriva-ml): bag-based commit_execution, behind env flag

- Introduce `_build_execution_bag` and `_load_execution_bag`
  as private methods on `Execution`.
- Wire behind `DERIVA_ML_BAG_COMMIT=1`. Default remains the
  legacy path.
- Unit tests:
  - `_build_execution_bag` produces a bag whose row counts and
    asset directory match the manifest.
  - Feature records get asset-column rewritten correctly.
- Integration test (parametrized by env flag): a full
  `with ml.create_execution(): ... ; upload_execution_outputs()`
  cycle ends in the same catalog state under both paths.

The env flag is the safety net. Side-by-side validation in CI;
fall back transparently if the bag path regresses on a fixture
we haven't covered.

### PR-D (deriva-ml): flip default to bag path

- Default the env flag to bag-mode.
- Mark `_upload_execution_dirs` `@deprecated` (still
  accessible via the env flag for one release).
- Update CLAUDE.md to document the new pipeline.

### PR-E (deriva-ml): delete legacy upload pipeline

- Delete `null_sentinel_processor.py`,
  `execution/upload_engine.py`, `execution/upload_job.py`.
- Delete `_upload_execution_dirs`, `_build_upload_staging`,
  `_cleanup_upload_staging` from `execution.py`.
- Refactor `_update_asset_execution_table` and
  `_flush_staged_features` into the bag-building adapters they
  became.

PR-E lands after PR-D has been live for one release without
issues.

## Test coverage required

- Unit: `_build_execution_bag` builds a bag whose row counts
  match the manifest.
- Unit: `BagBuilder.add_asset(link=True)` produces a working
  hardlink, correct MD5, correct path.
- Unit: `DanglingFKStrategy.PRESERVE` lets through rows with
  in-schema but out-of-bag FKs.
- Integration: full execution cycle round-trip under both paths
  (legacy vs bag), end-state catalog comparison.
- Integration: feature records with asset-column references
  resolve to the correct asset RID at the destination.
- Integration: additive upload (two `upload_execution_outputs`
  calls on one Execution).
- Integration: crash mid-upload → re-run → catalog converges.

## Non-goals

- We do not redesign `asset_file_path` or the manifest store.
- We do not change the public `Execution` context-manager
  surface.
- We do not move local_db code to deriva-py (separate ADR-0006
  follow-up).
- We do not promise the commit bag is a durable artifact — it's
  transient by design.
- We do not change the upload's network configuration knobs
  (timeout, chunk_size) — they pass through to
  `BagCatalogLoader`'s underlying HatracStore.

# add_files Dataset-Input Edge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `add_files` record input provenance in O(1) — one `Dataset_Execution` input row (the registered source dataset) instead of N per-file `File_Execution` Input rows.

**Architecture:** Remove the per-file `File_Execution` `Asset_Role="Input"` insert from `add_files`. After the nested dataset tree is built, insert ONE `Dataset_Execution` row linking the root source dataset to the execution as an input (the same shape `Execution.add_input_dataset` / the `datasets=` path uses). The dataset's producer edge (`Dataset_Version.Execution`), membership (`Dataset_File`), and file rows are untouched. Provenance stays complete (`_execution_has_input` returns True via the dataset edge) with no enforcement change; the tk-022 self-parent guard already makes the dataset-is-its-own-producer self-loop lineage-safe.

**Tech Stack:** Python 3.13, `uv`, pytest, deriva-ml core (`deriva_ml.core.mixins.file.FileMixin`, `deriva_ml.execution.execution.Execution`).

## Global Constraints

- Work in `../deriva-ml` (`/Users/carl/GitHub/DerivaML/deriva-ml`) on branch `feature/add-files-dataset-input-edge` (already created; the spec is committed there).
- Use `uv` for everything: `uv run python -m pytest` (NOT `uv run pytest`), `uv run ruff check`, `uv run ruff format`.
- Always `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>` in every Bash call — the shell CWD is not persistent.
- Google-style docstrings on changed methods.
- NO public-model change: `src/deriva_ml/execution/lineage.py` is NOT modified. The `add_files` signatures (FileMixin + Execution wrapper) are UNCHANGED (no new parameter — this is a default behavior change).
- The change applies UNIFORMLY to every `add_files` call (incl. the `LocalFile` execution-input path at `execution.py:716`).
- **`Dataset_Execution` has NO `Asset_Role` column** — it is input-only by design (`execution.py:1903-1908`: "Dataset_Execution association table has no role column"; authorship is inferred from `Dataset_Version.Execution`). The new edge row is `{"Dataset": <root rid>, "Execution": execution_rid, "Dataset_Version": <version rid or None>}` — do NOT add an `Asset_Role` key.
- Keep UNCHANGED: the `File` row inserts, `File_Asset_Type` tags, the nested `Dataset` tree, `Directory_Dataset` rows, `Dataset_Version.Execution` (via `create_dataset(execution_rid=...)`), and the chunked-streaming behavior.
- Other `File_Execution` writers (`asset_upload.update_asset_execution_table`, `provenance_enforcement._link_file_sentinel_as_input`) and all readers are NOT touched — `File_Execution` remains a live table.
- Release as deriva-ml 1.53.0 (behavior change), done by the controller after merge — NOT part of these tasks.

## Reference: the canonical Dataset_Execution input-insert (pattern to mirror)

`Execution.add_input_dataset` (`execution.py:2013-2065`) writes a single input edge:
```python
schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
dataset_exec = schema_path.Dataset_Execution
# de-dupe: skip if this Dataset is already linked to this Execution
already_linked = {row["Dataset"] for row in
    dataset_exec.filter(dataset_exec.Execution == self.execution_rid).entities().fetch()}
if dataset_rid in already_linked:
    return
version_rid = self._ml_object._version_rid(dataset_rid, version) if version is not None else None
dataset_exec.insert([{"Dataset": dataset_rid, "Execution": self.execution_rid, "Dataset_Version": version_rid}])
```
The `datasets=` batch path (`execution.py:673`) uses the same row shape. `Dataset_Version` may be `None` (the FK is nullable — the existing code passes `None` when no version is given).

## Reference: the current `add_files` per-file insert (the thing to remove)

`core/mixins/file.py`:
- Line ~270: `file_execution_path = pb.schemas[self.ml_schema].File_Execution`
- Lines ~300-308: per-batch insert of `{"File": rid, "Execution": execution_rid, "Asset_Role": "Input"}` for every file in the batch.
- The root dataset is `node_dataset[ingest_root]`, returned at the end of `add_files` (`return node_dataset[ingest_root]`, ~line 382). `node_dataset[ingest_root].dataset_rid` is the root RID.

---

### Task 1: swap the per-file edge for one Dataset_Execution input edge

**Files:**
- Modify: `src/deriva_ml/core/mixins/file.py` (remove per-file `File_Execution` insert ~lines 270, 300-308; add one `Dataset_Execution` insert after the tree is built, before `return`; update docstring)
- Modify: `src/deriva_ml/execution/execution.py` (update the `Execution.add_files` wrapper docstring only — no code change)
- Modify: `tests/core/test_file.py` (rewrite `test_add_files_links_as_input` to the dataset-edge contract; confirm other add_files tests still pass)

**Interfaces:**
- Unchanged public: `add_files(files, execution_rid, dataset_types=None, description="", chunk_size=500, *, root_name=None) -> Dataset` (FileMixin) and `Execution.add_files(files, dataset_types=None, description="", chunk_size=500, *, root_name=None) -> Dataset`.

- [ ] **Step 1: Rewrite the failing unit test to the new contract**

In `tests/core/test_file.py`, replace `test_add_files_links_as_input` (currently asserts per-file `File_Execution` Input rows) with the dataset-edge contract. Read the existing test's setup (`file_table_setup` fixture, how it builds filespecs + runs `exe.add_files`) and keep that; change only the assertions:

```python
    def test_add_files_links_dataset_as_input(self, file_table_setup):
        """add_files records ONE Dataset_Execution input edge (the root source
        dataset), not per-file File_Execution Input rows. The execution is
        input-complete via the dataset edge; producer + membership intact."""
        ml_instance, filespecs, exe_or_ctx = file_table_setup  # adapt to the fixture's actual unpacking
        # Build/obtain the execution context exactly as the existing add_files
        # tests in this file do, then:
        with ml_instance.create_execution(...) as exe:   # mirror the existing add_files tests' execution setup
            root = exe.add_files(filespecs)

        pb = ml_instance.pathBuilder()

        # (a) ZERO File_Execution Input rows were written by add_files.
        fe = pb.schemas[ml_instance.ml_schema].File_Execution
        fe_input_rows = [
            r for r in fe.filter(fe.Execution == exe.execution_rid).entities().fetch()
            if r.get("Asset_Role") == "Input"
        ]
        assert fe_input_rows == [], (
            f"add_files must NOT write per-file File_Execution Input rows; got {len(fe_input_rows)}"
        )

        # (b) EXACTLY ONE Dataset_Execution input edge: the root dataset.
        de = pb.schemas[ml_instance.ml_schema].Dataset_Execution
        de_rows = list(de.filter(de.Execution == exe.execution_rid).entities().fetch())
        assert len(de_rows) == 1, f"expected exactly one Dataset_Execution input edge, got {len(de_rows)}"
        assert de_rows[0]["Dataset"] == root.dataset_rid

        # (c) Producer edge intact: the root dataset's current version is produced
        #     by this execution (Dataset_Version.Execution).
        assert ml_instance._producer_of_dataset(root.dataset_rid) == exe.execution_rid

        # (d) Membership intact: the root dataset still transitively contains the files.
        members = root.list_dataset_members(recurse=True)
        assert members.get("File"), "root dataset should contain File members"
```

Note: `file_table_setup` and the exact execution-creation idiom vary — copy them verbatim from a working add_files test in the SAME file (e.g. `test_add_files` at line 104). The assertion block (a)-(d) is the contract; adapt only the setup/unpacking to match the fixture.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py -k "links_dataset_as_input" -v`
Expected: FAIL — `add_files` currently writes N File_Execution Input rows (assertion (a) fails) and zero Dataset_Execution rows (assertion (b) fails).
(If `test_file.py` is live-gated and needs a catalog, run with the project's live env, e.g. `DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true`. Check the top of the file for the gate and mirror it.)

- [ ] **Step 3: Implement the edge swap in `add_files`**

In `src/deriva_ml/core/mixins/file.py`:

(a) DELETE the per-file `File_Execution` insert. Remove the block (~lines 300-308):
```python
            # Link each file to the execution as an INPUT. ...
            file_execution_path.insert(
                [
                    {"File": file_record["RID"], "Execution": execution_rid, "Asset_Role": "Input"}
                    for file_record in file_records
                ]
            )
```
and remove the now-unused `file_execution_path = pb.schemas[self.ml_schema].File_Execution` line (~270) IF it is not referenced elsewhere in the method (grep within the function; if used only by the deleted block, remove it).

(b) ADD one `Dataset_Execution` input edge after the membership-wiring loop, immediately before `return node_dataset[ingest_root]` (~line 382):
```python
        # Record ONE input edge: the root source dataset is this execution's
        # input (consumed by reference). This replaces the per-file
        # File_Execution Input rows — O(1) instead of O(N). The dataset is also
        # this execution's OUTPUT (Dataset_Version.Execution, written by
        # create_dataset above); the self-loop is lineage-safe (the lookup_lineage
        # self-parent guard skips producer == the expanding execution). Dataset_
        # Execution has no Asset_Role column — every row is an input edge.
        root_dataset = node_dataset[ingest_root]
        try:
            root_version_rid = self._version_rid(root_dataset.dataset_rid, root_dataset.current_version)
        except Exception:
            root_version_rid = None
        de_assoc = pb.schemas[self.ml_schema].Dataset_Execution
        de_assoc.insert(
            [
                {
                    "Dataset": root_dataset.dataset_rid,
                    "Execution": execution_rid,
                    "Dataset_Version": root_version_rid,
                }
            ],
            on_conflict_skip=True,
        )
        return root_dataset
```
Notes:
- `_version_rid` and `current_version` exist (used by `add_input_dataset` and `Dataset` respectively). If `current_version` raises (no version row yet) or `_version_rid` returns None, pass `Dataset_Version=None` — the FK is nullable and the existing input-edge code tolerates None.
- `on_conflict_skip=True` makes the insert idempotent (mirrors `add_input_dataset` / `_link_file_sentinel_as_input`).
- `pb` is already bound earlier in `add_files`; reuse it (don't re-call `self.pathBuilder()` if `pb` is in scope — verify the variable name in the method).

(c) UPDATE the `add_files` docstring (FileMixin, ~lines 176-200): replace the "links each file to the execution as an **input** (`File_Execution.Asset_Role="Input"`)" sentences with language stating the registered **source dataset** is recorded as the execution's input (one `Dataset_Execution` edge) — per-file consumption edges are intentionally not written (find consumers via the dataset). Keep the "references not Hatrac uploads; produced files use asset_file_path" framing.

- [ ] **Step 4: Update the `Execution.add_files` wrapper docstring**

In `src/deriva_ml/execution/execution.py` (`Execution.add_files`, ~line 2067), apply the same docstring correction (dataset-input edge, not per-file). No code change in the wrapper.

- [ ] **Step 5: Run the rewritten test + the full add_files test class**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py -v` (use the live env if the file is gated).
Expected: PASS — the new `test_add_files_links_dataset_as_input` passes, and the other add_files tests (`test_add_files`, `test_add_files_tags_datasets_as_directory`, `test_add_files_directory_datasets_record_path`, `test_add_files_returns_single_root_for_forest`, `test_add_files_chunked_streaming_matches_single_batch`) still pass (they assert tree/membership/Directory_Dataset shape, which is unchanged).

- [ ] **Step 6: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/file.py src/deriva_ml/execution/execution.py tests/core/test_file.py
uv run ruff format --check src/deriva_ml/core/mixins/file.py src/deriva_ml/execution/execution.py tests/core/test_file.py
git add src/deriva_ml/core/mixins/file.py src/deriva_ml/execution/execution.py tests/core/test_file.py
git commit -m "feat(add_files): record one Dataset_Execution input edge, not N per-file File_Execution rows (tk-024)"
```

---

### Task 2: update the LocalFile-input test + any other per-file-edge assertions

**Files:**
- Modify: `tests/execution/test_local_file_input.py` (the LocalFile input now links the dataset, not the file)
- Modify: `tests/core/test_asset_table.py` (IF it asserts add_files per-file File_Execution Input rows — check and update)

**Interfaces:**
- Consumes: the new `add_files` behavior from Task 1.

**Background:** `LocalFile` execution inputs go through `add_files` (`execution.py:716`), so a `LocalFile` input now becomes a small File dataset with one `Dataset_Execution` input edge instead of a `File_Execution` Input row. `test_local_file_input.py:120-126` currently asserts a `File_Execution` `Asset_Role="Input"` row exists — that assertion must move to the dataset edge. The execution must still be input-complete.

- [ ] **Step 1: Read the current LocalFile test + update the assertion**

Read `tests/execution/test_local_file_input.py` in full. The test (around line 96-126) runs an execution with a `LocalFile` in `assets=`/`configuration` and asserts a `File_Execution` Input row. Replace that assertion block with the dataset-edge contract while keeping the rest of the test (setup, execution run) intact:

```python
    # The LocalFile input is registered via add_files, which now records ONE
    # Dataset_Execution input edge (the file's dataset) rather than a per-file
    # File_Execution Input row. The execution must be input-complete via that edge.
    pb = ml.pathBuilder()

    # No per-file File_Execution Input row from the LocalFile registration.
    fe = pb.schemas[ml.ml_schema].File_Execution
    fe_inputs = [r for r in fe.filter(fe.Execution == <exec_rid>).entities().fetch()
                 if r.get("Asset_Role") == "Input"]
    assert fe_inputs == [], "LocalFile input must not create a per-file File_Execution Input row"

    # A Dataset_Execution input edge exists (the LocalFile's dataset).
    de = pb.schemas[ml.ml_schema].Dataset_Execution
    de_rows = list(de.filter(de.Execution == <exec_rid>).entities().fetch())
    assert de_rows, "LocalFile input must be recorded as a Dataset_Execution input edge"

    # The execution is input-complete (no unknown-provenance sentinel needed).
    from deriva_ml.execution.provenance_enforcement import _execution_has_input
    assert _execution_has_input(ml_instance=ml, execution_rid=<exec_rid>) is True
```
Replace `<exec_rid>` with however the test references the execution's RID (read the test to find the variable). Keep the test's docstring accurate (update the "linked as Input (File_Execution Asset_Role=Input)" wording to "recorded as a Dataset_Execution input edge").

- [ ] **Step 2: Check `test_asset_table.py` and any other File_Execution assertions**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "File_Execution" tests/core/test_asset_table.py`
- If it asserts add_files per-file Input rows, update those assertions to the dataset-edge contract (or remove if they were specifically testing the per-file behavior that no longer exists — but prefer re-expressing the provenance assertion at dataset granularity, do NOT silently drop coverage).
- If its `File_Execution` references are about Output/uploaded assets (NOT add_files Input edges), leave them — only the add_files Input path changed.

- [ ] **Step 3: Run the updated tests**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_local_file_input.py tests/core/test_asset_table.py -v` (live env if gated).
Expected: PASS.

- [ ] **Step 4: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check tests/execution/test_local_file_input.py tests/core/test_asset_table.py
uv run ruff format --check tests/execution/test_local_file_input.py tests/core/test_asset_table.py
git add tests/execution/test_local_file_input.py tests/core/test_asset_table.py
git commit -m "test(add_files): LocalFile + asset-table tests assert dataset-input edge (tk-024)"
```

---

### Task 3: live lineage regression — large add_files dataset shows no per-file asset bloat

**Files:**
- Modify: `tests/execution/test_lookup_lineage_live.py` (add one DERIVA_HOST-gated test)

**Interfaces:**
- Consumes: the new `add_files` behavior (Task 1), the `test_ml` fixture, `tests/factories.py`, the `DERIVA_HOST` gate.

**Background:** The point of the change is that `lookup_lineage` of an `add_files`-registered dataset no longer carries N per-file `consumed_assets` on the registration execution. This test registers a dataset with a meaningful number of files via `add_files`, walks its lineage, and asserts the registration execution has empty `consumed_assets`, the source dataset as its one consumed dataset, and no false cycle.

- [ ] **Step 1: Write the live test**

Add to `tests/execution/test_lookup_lineage_live.py` (after the existing tests). Read the existing live tests + `tests/factories.py` (and `FileSpec.create_filespecs`) for how to build filespecs and run `add_files` live; the assertion block is the contract:

```python
@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_add_files_lineage_has_no_per_file_asset_bloat(test_ml, tmp_path):
    """tk-024: lookup_lineage of an add_files-registered dataset shows the
    registration execution with ZERO consumed_assets (no per-file File Inputs)
    and the source dataset as its single consumed dataset; the self-loop
    (dataset is both produced and consumed) does not flag a cycle."""
    N_FILES = 60  # enough that the OLD code would attach >=60 consumed_assets
    # Create N_FILES local files in tmp_path, build FileSpecs, run an execution
    # that add_files them, capture the root dataset RID + the execution RID.
    # (Study the existing live tests + FileSpec.create_filespecs + factories.)
    #
    #   src_dataset_rid = the dataset add_files returned
    #   reg_exec_rid    = the execution that ran add_files
    ...

    result = test_ml.lookup_lineage(src_dataset_rid)

    assert result.cycle_detected is False
    assert result.lineage is not None
    # The registration execution is the producer of the source dataset.
    assert result.root.producing_execution is not None
    reg_node = result.lineage
    assert reg_node.execution.rid == reg_exec_rid

    # No per-file File Inputs on the registration execution.
    assert reg_node.consumed_assets == [], (
        f"expected zero consumed_assets, got {len(reg_node.consumed_assets)}"
    )
    # The source dataset is recorded as the one consumed dataset (the input edge).
    consumed = {d.rid for d in reg_node.consumed_datasets}
    assert src_dataset_rid in consumed, (
        f"source dataset {src_dataset_rid} not in consumed_datasets {consumed}"
    )
```

The implementer MUST replace `...` with real construction (≥~50 files so the old behavior would have bloated the node). Do NOT leave the ellipsis; do NOT weaken the `consumed_assets == []` or `cycle_detected is False` assertions. If a live host is unavailable, report it, confirm the test SKIPS cleanly and collects without error — do NOT fake a pass.

- [ ] **Step 2: Run the live test (gated)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/execution/test_lookup_lineage_live.py::test_add_files_lineage_has_no_per_file_asset_bloat -v`
Expected: PASS (the registration execution's consumed_assets is empty; the source dataset is its consumed dataset; no cycle). Building ~60 files + add_files is quick (no Hatrac upload — by reference). If no container, report + confirm skip.

- [ ] **Step 3: Confirm existing live tests still pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/execution/test_lookup_lineage_live.py -v`
Expected: all live tests pass (the prior lineage tests don't use add_files Input edges, so they're unaffected). If no container: confirm clean skip + collection.

- [ ] **Step 4: Confirm skip path (offline)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_live.py -v`
Expected: all live tests SKIP (no DERIVA_HOST), no collection errors.

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check tests/execution/test_lookup_lineage_live.py
uv run ruff format --check tests/execution/test_lookup_lineage_live.py
git add tests/execution/test_lookup_lineage_live.py
git commit -m "test(lineage): live add_files dataset has no per-file consumed_assets bloat (tk-024)"
```

---

## Final verification (after all tasks)

- [ ] No add_files per-file `File_Execution` Input writer remains: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "File_Execution" src/deriva_ml/core/mixins/file.py` — should show only docstring/removed references, no per-file Input insert. (The constant/lookup line should be gone if unused.)
- [ ] Other `File_Execution` writers untouched: `grep -rn "File_Execution" src/deriva_ml/execution/asset_upload.py src/deriva_ml/execution/provenance_enforcement.py` still present (Output/downloaded-Input/sentinel paths unchanged).
- [ ] Full core + execution test dirs: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/ tests/execution/ -q` — green (live tests skip without DERIVA_HOST; the 2 pre-existing `test_workflow_creation_*` failures are unrelated).
- [ ] Lint the touched surface: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src tests`.

# Directory_Dataset Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every directory dataset created by `add_files` a structured, queryable relative path (the source folder it represents), stored in a new `Directory_Dataset` satellite table, surfaced in Chaise, readable from Python and from a downloaded `DatasetBag`, replacing the prose `Description`-suffix hack.

**Architecture:** A new `Directory_Dataset` satellite table (one row per directory dataset: `Dataset` FK → `Dataset.RID`, plus `Path` = path relative to the ingest root). `add_files` writes one row per directory dataset it creates, using the relative path it already computes. The `Dataset.Description` reverts to the bare caller string (no `— relpath` suffix). A `Dataset.path` Python property reads the table (works live and from a bag). A Chaise `visible-columns` annotation surfaces `Path` inline as "Folder" on the Dataset record page.

**Tech Stack:** Python ≥3.12, deriva-py datapath API, ERMrest, BDBag, pytest, ruff, uv.

## Global Constraints

- Use `uv` for everything: `uv run pytest`, `uv run ruff check src tests`. Never invoke `pytest`/`ruff`/`python` directly.
- Tests need a live catalog at `DERIVA_HOST=localhost` and `DERIVA_ML_ALLOW_DIRTY=true`.
- RIDs are opaque, equality-only. Never hard-code a RID literal in a test; obtain every RID from a fixture-produced catalog row or a fresh `create_*`/`insert` call in the test.
- Google-style docstrings with a runnable-or-`# doctest: +SKIP` `Example:` block on every new public method.
- Path stored is **relative to the ingest root** (e.g. `a/x`), never absolute — no local-filesystem-path leak. The ingest root's own dataset stores `.`.
- All work lands on branch `feat/add-files-directory-descriptions` (the open PR #348). This plan SUPERSEDES the `Description`-suffix in that PR — remove it as part of Task 3.
- `bump-version` happens only after merge, on `main`, never in this plan.
- The `schema.md ↔ create_schema.py` CI check is blocking: any schema change requires the matching `docs/reference/schema.md` edit in the SAME task (see Task 1).

---

### Task 1: Create the `Directory_Dataset` satellite table in the schema

**Files:**
- Modify: `src/deriva_ml/schema/create_schema.py` (in `create_dataset_table`, after the `Dataset_Dataset` / `Dataset_Execution` associations are created, ~line 160-171 region; add a new table-creation block before the function returns `dataset_table`)
- Modify: `docs/reference/schema.md` (add a `## Directory_Dataset` entry + a Table-of-contents link)
- Test: `tests/schema/test_directory_dataset_table.py` (Create)

**Interfaces:**
- Produces: a `Directory_Dataset` table in the `deriva-ml` schema with columns `Dataset` (text), `Path` (text), an FK `Directory_Dataset.Dataset → Dataset.RID`, and a key on `Dataset` (one row per dataset max). Reachable for later tasks as `pb.schemas[ml_schema].tables["Directory_Dataset"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/schema/test_directory_dataset_table.py
"""The Directory_Dataset satellite table exists with the expected shape."""


class TestDirectoryDatasetTable:
    def test_directory_dataset_table_shape(self, test_ml):
        model = test_ml.model.model
        ml_schema = test_ml.ml_schema
        assert "Directory_Dataset" in model.schemas[ml_schema].tables, (
            "Directory_Dataset satellite table must exist in the deriva-ml schema"
        )
        table = model.schemas[ml_schema].tables["Directory_Dataset"]
        colnames = {c.name for c in table.columns}
        # System columns plus the two payload columns.
        assert {"Dataset", "Path"} <= colnames

        # FK Dataset -> Dataset.RID exists.
        fk_targets = {
            (fk.pk_table.name, tuple(c.name for c in fk.foreign_key_columns))
            for fk in table.foreign_keys
        }
        assert ("Dataset", ("Dataset",)) in fk_targets, (
            "Directory_Dataset.Dataset must be an FK to Dataset.RID"
        )

    def test_directory_dataset_one_row_per_dataset(self, test_ml):
        """A key on Dataset enforces at most one Directory_Dataset row per dataset."""
        table = test_ml.model.model.schemas[test_ml.ml_schema].tables["Directory_Dataset"]
        key_colsets = {tuple(sorted(c.name for c in k.unique_columns)) for k in table.keys}
        assert ("Dataset",) in key_colsets, "expected a uniqueness key on the Dataset column"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_directory_dataset_table.py -q`
Expected: FAIL — `Directory_Dataset` not in tables (KeyError / assertion).

- [ ] **Step 3: Add a shared table-def factory + create the table in `create_dataset_table`**

In `src/deriva_ml/schema/create_schema.py`, FIRST add a module-level factory (so Task 6's migration can reuse the EXACT same definition — DRY, no drift). Place it as a module-level function near `create_dataset_table`:

```python
def directory_dataset_table_def(schema_name: str) -> TableDef:
    """TableDef for the Directory_Dataset satellite table.

    Shared by ``create_dataset_table`` (fresh catalogs) and the
    ``add_directory_dataset_table`` migration (existing catalogs) so both
    produce an identical table.

    Args:
        schema_name: The deriva-ml schema name (for the FK's referenced_schema).

    Returns:
        TableDef: definition of the Directory_Dataset table.
    """
    return TableDef(
        name="Directory_Dataset",
        comment=(
            "Source folder a directory dataset (auto-created by add_files) "
            "represents, as a path relative to the ingest root. One row per "
            "directory dataset; absent for datasets not built from a "
            "directory tree."
        ),
        columns=[
            ColumnDef("Dataset", BuiltinType.text, comment="RID of the directory dataset."),
            ColumnDef(
                "Path",
                BuiltinType.text,
                comment=(
                    "Source directory this dataset represents, relative to "
                    "the ingest root. The ingest root stores '.'."
                ),
            ),
        ],
        keys=[KeyDef(columns=["Dataset"])],
        foreign_keys=[
            ForeignKeyDef(
                columns=["Dataset"],
                referenced_schema=schema_name,
                referenced_table="Dataset",
                referenced_columns=["RID"],
            ),
        ],
    )
```

THEN, inside `create_dataset_table`, after the `Dataset_Dataset` self-association is created and before `return dataset_table`, add:

```python
    # Directory_Dataset: satellite recording the source folder a directory
    # dataset (created by add_files) represents (see directory_dataset_table_def).
    schema.create_table(directory_dataset_table_def(schema.name))
```

(Confirm `TableDef`, `ColumnDef`, `BuiltinType`, `KeyDef`, `ForeignKeyDef` are already imported at the top of `create_schema.py` — they are used by `create_dataset_table`/`define_table_dataset_version` already. No new imports needed. Export `directory_dataset_table_def` if the module has an `__all__`.)

- [ ] **Step 4: Add the `schema.md` entry**

In `docs/reference/schema.md`, add to the "Associations"/table list in the Table of contents (line ~40) a link `[Directory_Dataset](#directory_dataset)`, and add a new section after the `## Dataset_Version` block:

```markdown
## Directory_Dataset

Source folder a directory dataset (auto-created by `add_files`) represents, as a path relative to the ingest root. One row per directory dataset; absent for datasets not built from a directory tree.

```yaml
table: Directory_Dataset
kind: table
columns:
- name: Dataset
  type: text
- name: Path
  type: text
foreign_keys:
- columns:
  - Dataset
  referenced_schema: deriva-ml
  referenced_table: Dataset
  referenced_columns:
  - RID
```
```

- [ ] **Step 5: Run the schema-doc validator**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m deriva_ml.tools.validate_schema_doc`
Expected: `deriva-ml-validate-schema: schema.md and create_schema.py agree.`

- [ ] **Step 6: Run the table-shape test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_directory_dataset_table.py -q`
Expected: PASS (2 passed). Note: `test_ml` builds a fresh catalog, so the new table is created by `create_schema`.

- [ ] **Step 7: Lint**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/schema/create_schema.py tests/schema/test_directory_dataset_table.py`
Expected: All checks passed (or only pre-existing errors unrelated to these files — confirm via `git stash` + `--no-cache` comparison if any appear).

- [ ] **Step 8: Commit**

```bash
git add src/deriva_ml/schema/create_schema.py docs/reference/schema.md tests/schema/test_directory_dataset_table.py
git commit -m "feat(schema): add Directory_Dataset satellite table (Dataset FK + relative Path)"
```

---

### Task 2: `add_files` writes a `Directory_Dataset` row per directory dataset; `Description` reverts to bare caller string

**Files:**
- Modify: `src/deriva_ml/core/mixins/file.py` (the dataset-building section of `add_files`, currently ~lines 198-256 — the `dir_description` helper and the node-creation loop)
- Test: `tests/core/test_file.py` (Modify — add a new test; update the existing `test_add_files_directory_datasets_describe_their_path` to assert on `Directory_Dataset.Path` instead of the Description suffix)

**Interfaces:**
- Consumes: the `Directory_Dataset` table from Task 1 (`pb.schemas[self.ml_schema].tables["Directory_Dataset"]`).
- Produces: for each directory dataset `add_files` creates, a `Directory_Dataset` row `{"Dataset": <dataset_rid>, "Path": <relpath>}` where `<relpath>` is `directory.relative_to(ingest_root).as_posix()` (the ingest root stores `"."`). `Dataset.Description` is the bare caller `description` for ALL nodes.

- [ ] **Step 1: Write the failing test**

Replace the body of the existing `test_add_files_directory_datasets_describe_their_path` (in `tests/core/test_file.py`) and add a path-table test:

```python
    def test_add_files_directory_datasets_record_path(self, file_table_setup):
        """Each directory dataset gets a Directory_Dataset row with its path
        relative to the ingest root; the ingest root stores '.'. Description is
        the bare caller string for every node (no path suffix)."""
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            file_dataset = exe.add_files(filespecs, description="Ingest run")

        # Description is the bare caller string everywhere now.
        assert file_dataset.description == "Ingest run"
        for child in file_dataset.list_dataset_children():
            assert child.description == "Ingest run"

        # Directory_Dataset.Path holds the relative folder for each dataset.
        pb = ml_instance.pathBuilder()
        rows = list(pb.schemas[ml_instance.ml_schema].tables["Directory_Dataset"].entities().fetch())
        path_by_dataset = {r["Dataset"]: r["Path"] for r in rows}

        assert path_by_dataset[file_dataset.dataset_rid] == "."
        child_paths = {path_by_dataset[c.dataset_rid] for c in file_dataset.list_dataset_children()}
        assert child_paths == {"d1", "d2"}
```

Also DELETE the old `test_add_files_directory_datasets_describe_their_path` method (its Description-suffix assertions are now wrong — superseded).

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_file.py::TestFile::test_add_files_directory_datasets_record_path -q`
Expected: FAIL — `Directory_Dataset` has no rows (add_files doesn't write them yet) → KeyError on `path_by_dataset[file_dataset.dataset_rid]`. (And the Description assertions would fail too, since current code appends the suffix.)

- [ ] **Step 3: Update the `add_files` dataset-build section**

In `src/deriva_ml/core/mixins/file.py`, replace the `dir_description` helper and the membership loop. Change `dir_description` back to bare description, and after the node datasets are created, insert `Directory_Dataset` rows. The relevant section becomes:

```python
        # The ingest root keeps the bare caller description; every node dataset
        # uses the same description. The folder each node represents is recorded
        # structurally in Directory_Dataset (below), not in the prose Description.
        node_dataset: dict[Path, "Dataset"] = {
            directory: Dataset.create_dataset(
                self,  # type: ignore[arg-type]
                dataset_types=dataset_types,
                execution_rid=execution_rid,
                description=description,
            )
            for directory in nodes
        }

        # Record each directory dataset's source folder as a path relative to the
        # ingest root (the root stores "."). Structured + queryable; consumers
        # never parse the Description.
        pb.schemas[self.ml_schema].tables["Directory_Dataset"].insert(
            [
                {
                    "Dataset": ds.dataset_rid,
                    "Path": "." if directory == ingest_root else directory.relative_to(ingest_root).as_posix(),
                }
                for directory, ds in node_dataset.items()
            ]
        )

        # Wire membership: each node's dataset gets its own files plus its
        # immediate child-directory datasets (the nodes whose parent is this node).
        for directory in sorted(nodes, key=lambda d: len(d.parts), reverse=True):
            members = list(dir_rid_map.get(directory, []))
            members += [
                child_ds.dataset_rid
                for child_dir, child_ds in node_dataset.items()
                if child_dir != directory and child_dir.parent == directory
            ]
            if members:
                node_dataset[directory].add_dataset_members(members=members, execution_rid=execution_rid)

        # The ingest root's dataset transitively contains every file.
        return node_dataset[ingest_root]
```

Delete the now-unused `dir_description` closure. `pb` is already bound earlier in `add_files` (the streaming-insert section); confirm it is in scope at this point (it is — `pb = self.pathBuilder()` is set before the batched loop). Update the `description` Args docstring entry of `add_files` to drop the "appends its path relative to the root" sentence and instead say: "Recorded verbatim on every directory dataset; the source folder each dataset represents is stored structurally in the `Directory_Dataset` table."

- [ ] **Step 4: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_file.py::TestFile::test_add_files_directory_datasets_record_path -q`
Expected: PASS.

- [ ] **Step 5: Run the full file test suite (no regressions)**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_file.py -q`
Expected: all pass (the single-root forest test, tagging test, chunked-streaming test, add_files structure test all still green).

- [ ] **Step 6: Lint**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/core/mixins/file.py tests/core/test_file.py`
Expected: All checks passed (the pre-existing E731 in test_file.py:15 is unrelated; confirm no NEW errors via stash + `--no-cache`).

- [ ] **Step 7: Commit**

```bash
git add src/deriva_ml/core/mixins/file.py tests/core/test_file.py
git commit -m "feat(files): record directory dataset folder in Directory_Dataset.Path; bare Description"
```

---

### Task 3: `Dataset.path` + `Dataset.is_directory` accessors (live + bag)

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` (add `path` and `is_directory` properties on the `Dataset` class; both read `Directory_Dataset` for this dataset's RID)
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (add the same `path` + `is_directory` properties on `DatasetBag`, reading the bag's `Directory_Dataset` table)
- Test: `tests/core/test_file.py` (Modify — add a live-catalog accessor test) and `tests/dataset/test_dataset_bag.py` (Modify — add a bag accessor test if that file exists; otherwise add to the nearest bag test module)

**Interfaces:**
- Consumes: `Directory_Dataset` rows from Task 2.
- Produces:
  - `Dataset.path -> str | None` and `DatasetBag.path -> str | None` — the relative source folder this (directory) dataset represents, or `None` if the dataset has no `Directory_Dataset` row.
  - `Dataset.is_directory -> bool` and `DatasetBag.is_directory -> bool` — `True` iff the dataset has a `Directory_Dataset` row (equivalently, `.path is not None`). This is the authoritative predicate; it does NOT consult the `Directory` Dataset_Type tag (which can diverge for pre-feature/hand-tagged datasets).

- [ ] **Step 1: Write the failing test (live)**

Add to `tests/core/test_file.py`:

```python
    def test_dataset_path_and_is_directory_accessor(self, file_table_setup):
        """Dataset.path returns the directory dataset's relative folder and
        is_directory is True for those datasets; both reflect the
        Directory_Dataset row."""
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            root = exe.add_files(filespecs, description="Ingest run")

        assert root.path == "."
        assert root.is_directory is True
        child_paths = {child.path for child in root.list_dataset_children()}
        assert child_paths == {"d1", "d2"}
        assert all(child.is_directory for child in root.list_dataset_children())

        # A non-directory dataset (created directly, not via add_files) has no
        # Directory_Dataset row: path is None and is_directory is False.
        with execution.execute() as exe:
            plain = exe.create_dataset(dataset_types="Complete", description="not a dir")
        assert plain.path is None
        assert plain.is_directory is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_file.py::TestFile::test_dataset_path_and_is_directory_accessor -q`
Expected: FAIL — `AttributeError: 'Dataset' object has no attribute 'path'`.

- [ ] **Step 3: Implement `Dataset.path` and `Dataset.is_directory` (live)**

In `src/deriva_ml/dataset/dataset.py`, add two properties on the `Dataset` class (near the `description` property, ~line 144):

```python
    @property
    def path(self) -> str | None:
        """Source folder this directory dataset represents, relative to the
        ingest root.

        Returns the path stored in ``Directory_Dataset`` for this dataset (the
        ingest root stores ``"."``), or ``None`` if the dataset has no
        ``Directory_Dataset`` row — i.e. it was not created from a directory
        tree by :meth:`add_files`.

        Returns:
            str | None: The relative source folder, or None.

        Example:
            >>> root = exe.add_files(specs, description="ingest")  # doctest: +SKIP
            >>> root.path  # doctest: +SKIP
            '.'
            >>> [c.path for c in root.list_dataset_children()]  # doctest: +SKIP
            ['d1', 'd2']
        """
        pb = self._ml_instance.pathBuilder()
        dd = pb.schemas[self._ml_instance.ml_schema].tables["Directory_Dataset"]
        rows = list(dd.filter(dd.Dataset == self.dataset_rid).attributes(dd.Path).fetch())
        return rows[0]["Path"] if rows else None

    @property
    def is_directory(self) -> bool:
        """Whether this dataset represents a source directory.

        ``True`` iff the dataset has a ``Directory_Dataset`` row (equivalently,
        :attr:`path` is not ``None``) — i.e. it was created by :meth:`add_files`
        to mirror a folder. This is the authoritative predicate; it deliberately
        does NOT consult the ``Directory`` ``Dataset_Type`` tag, which can
        diverge from the path row for pre-feature or hand-tagged datasets.

        Returns:
            bool: True if this is a directory dataset.

        Example:
            >>> root = exe.add_files(specs, description="ingest")  # doctest: +SKIP
            >>> root.is_directory  # doctest: +SKIP
            True
        """
        return self.path is not None
```

(Confirm the attribute used to reach the catalog is `self._ml_instance` — match whatever the neighbouring `description` property uses; if the class stores it as `self._catalog` or similar, use that name.)

- [ ] **Step 4: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_file.py::TestFile::test_dataset_path_and_is_directory_accessor -q`
Expected: PASS.

- [ ] **Step 5: Write the failing bag test**

Find the bag test module (`ls tests/dataset/ | grep bag`). Add a test that downloads a directory dataset as a bag and reads `.path` / `.is_directory`. Use the existing bag-download fixture pattern in that module. Skeleton (adapt fixture names to the module's conventions):

```python
    def test_dataset_bag_path_and_is_directory(self, <bag fixture>):
        """DatasetBag.path / .is_directory work offline from the materialized bag."""
        bag = <materialize a directory dataset built via add_files>
        assert bag.path == "."
        assert bag.is_directory is True
        child_paths = {child.path for child in bag.list_dataset_children()}
        assert child_paths == {"d1", "d2"}
```

If no existing fixture builds an add_files directory dataset and downloads it, this test requires building one inline (create files → add_files → download_dataset_bag → materialize). Keep it in the bag test module so it shares that module's catalog/bag fixtures.

- [ ] **Step 6: Run the bag test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/<bag test file>::<test> -q`
Expected: FAIL — `AttributeError: 'DatasetBag' object has no attribute 'path'`.

- [ ] **Step 7: Implement `DatasetBag.path` and `DatasetBag.is_directory`**

In `src/deriva_ml/dataset/dataset_bag.py`, add a `path` property mirroring the live one but reading the bag's local `Directory_Dataset` table (the bag backs queries with a local database — use the same query mechanism the bag uses for other member reads, e.g. its `list_dataset_members`/path-builder equivalent). Read the row where `Dataset == self.dataset_rid`, return `Path` or `None`. Add `is_directory` returning `self.path is not None`. Match the bag's existing query idiom (do NOT introduce a live catalog connection — the bag is offline).

- [ ] **Step 8: Run the bag test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/<bag test file>::<test> -q`
Expected: PASS. (This implicitly verifies Task 4's bag-export inclusion — if `Directory_Dataset` isn't in the bag, this fails, signalling Task 4 is needed first. If it fails for that reason, do Task 4, then return here.)

- [ ] **Step 9: Lint + doctest**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/dataset/dataset.py src/deriva_ml/dataset/dataset_bag.py && DERIVA_ML_ALLOW_DIRTY=true uv run pytest --doctest-modules src/deriva_ml/dataset/dataset.py -q`
Expected: lint clean; doctest passes (the `path` example is `+SKIP`).

- [ ] **Step 10: Commit**

```bash
git add src/deriva_ml/dataset/dataset.py src/deriva_ml/dataset/dataset_bag.py tests/
git commit -m "feat(dataset): Dataset.path / DatasetBag.path read Directory_Dataset folder"
```

---

### Task 4: Ensure `Directory_Dataset` exports into the BDBag

**Files:**
- Modify (if needed): `src/deriva_ml/dataset/bag_builder.py` and/or the FK-traversal policy in `src/deriva_ml/core/constants.py` (only if the walker does not already reach `Directory_Dataset`)
- Test: `tests/dataset/<bag test file>` (the bag test from Task 3 Step 5 is the proof — a separate explicit check is added here)

**Interfaces:**
- Consumes: the `Directory_Dataset` table (Task 1), populated by `add_files` (Task 2).
- Produces: `Directory_Dataset` rows for the exported dataset present in the materialized bag, so `DatasetBag.path` (Task 3) resolves offline.

- [ ] **Step 1: Write/confirm the failing test**

This is the bag test from Task 3 Step 5. If it already passes after Task 3, the walker reaches `Directory_Dataset` automatically (it's an inbound FK from `Dataset`, like `Dataset_Version`) and NO walker change is needed — record that and skip to Step 4. If it fails because the bag lacks `Directory_Dataset` rows, continue.

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/<bag test file>::test_dataset_bag_path_accessor -q`
Expected (if walker change needed): FAIL — bag has no `Directory_Dataset` rows / table.

- [ ] **Step 2: Add `Directory_Dataset` to the bag traversal**

Only if Step 1 shows it's missing. `Directory_Dataset` is an inbound FK target from `Dataset` (Dataset → has-many Directory_Dataset). The `DatasetBagBuilder` walks FK paths from the Dataset RID; an inbound association/satellite is normally included like `Dataset_Version`. If it's being pruned, the likely cause is the empty-association pruning or a terminal-table guard. Inspect `bag_builder.py` for where satellite tables like `Dataset_Version` are included and mirror that handling for `Directory_Dataset`. (Document the exact change found necessary — do not guess a policy edit without confirming the walker omits it.)

- [ ] **Step 3: Re-run the bag test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/<bag test file>::test_dataset_bag_path_accessor -q`
Expected: PASS.

- [ ] **Step 4: Commit (or note no-op)**

If a walker change was needed:
```bash
git add src/deriva_ml/dataset/bag_builder.py
git commit -m "feat(bag): include Directory_Dataset in dataset bag export"
```
If no change was needed (walker already reaches it), add a one-line note to the bag test's module docstring stating `Directory_Dataset` exports automatically as an inbound satellite, and commit that.

---

### Task 5: Chaise — surface `Path` as a "Folder" column on the Dataset record page

**Files:**
- Modify: `src/deriva_ml/schema/annotations.py` (the Dataset table's `visible-columns` annotation — add an inline source pulling `Directory_Dataset.Path`)
- Test: `tests/schema/test_annotations.py` (Modify — assert the Dataset annotation contains the Folder source) OR `tests/model/test_annotations.py` if that's where Dataset annotations are tested

**Interfaces:**
- Consumes: `Directory_Dataset` table + FK (Task 1).
- Produces: the Dataset table's `tag:isrd.isi.edu,2016:visible-columns` annotation includes, in the **`detailed` context only**, an inline pseudo-column titled "Folder" sourced from `Directory_Dataset.Path`, **with `show_null` false** so Chaise SUPPRESSES the field on non-directory datasets (null Path) and shows the path on directory datasets. This is how "show the folder only if the dataset is a directory" is achieved — Chaise has no row-conditional column visibility, but `show_null` hides a null-valued field.

- [ ] **Step 1: Locate the Dataset annotation builder**

Run: `grep -n "dataset_annotation\|def .*dataset.*annotation\|visible_columns" src/deriva_ml/schema/annotations.py | head`
Identify the function that builds the Dataset table's `visible-columns` (mirror the `asset_annotation` "Produced By" inline-from-related-table pattern at annotations.py:316-332).

- [ ] **Step 2: Write the failing test**

In the Dataset-annotation test module, add:

```python
    def test_dataset_annotation_includes_folder_column(self):
        # Build the Dataset annotation and assert a "Folder" source referencing
        # Directory_Dataset.Path is present in DETAILED visible-columns only,
        # with show_null false so it's hidden for non-directory datasets.
        ann = <call the Dataset annotation builder>
        vc = ann["tag:isrd.isi.edu,2016:visible-columns"]
        # Folder lives in detailed only, NOT in the compact (*) list.
        detailed = vc.get("detailed", [])
        compact = vc.get("*", [])
        folder_sources = [
            s for s in detailed
            if isinstance(s, dict) and s.get("markdown_name") == "Folder"
        ]
        assert folder_sources, "Dataset detailed visible-columns must include a 'Folder' source"
        assert not any(
            isinstance(s, dict) and s.get("markdown_name") == "Folder" for s in compact
        ), "Folder must NOT appear in the compact (*) context"
        folder = folder_sources[0]
        src_path = folder["source"]
        # The source walks inbound to Directory_Dataset and reads Path.
        assert any(
            isinstance(seg, dict) and "inbound" in seg and "Directory_Dataset" in seg["inbound"][1]
            for seg in src_path
        )
        assert src_path[-1] == "Path"
        # show_null false in detailed → Chaise hides the field when Path is null
        # (i.e. for non-directory datasets).
        assert folder.get("display", {}).get("show_null") is False
```

(Adapt `<call the Dataset annotation builder>` to the actual function name found in Step 1, and the exact tag constant via `deriva_tags.visible_columns` if that's how the module references it.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/<dataset annotation test> -q`
Expected: FAIL — no "Folder" source present.

- [ ] **Step 4: Add the Folder source to the Dataset DETAILED visible-columns (with show_null)**

In the Dataset annotation builder, add the source to the `detailed` context ONLY (NOT the compact `*` list — a folder path is detail-level, shouldn't clutter the recordset list). Mirror the FK-constraint naming the schema generates — `Directory_Dataset_Dataset_fkey` is the inbound FK from Directory_Dataset to Dataset:

```python
    folder_source = {
        "source": [
            {"inbound": [ml_schema, "Directory_Dataset_Dataset_fkey"]},
            "Path",
        ],
        "aggregate": "array_d",
        "markdown_name": "Folder",
        # show_null false → Chaise SUPPRESSES this field on rows whose Path is
        # null (non-directory datasets), so "Folder" appears only for directory
        # datasets. Chaise has no true row-conditional column visibility; this
        # null-suppression is the idiomatic way to get "show only if directory".
        "display": {"show_null": False},
    }
```

Insert `folder_source` into the Dataset `visible-columns` **`detailed`** array only. Verify the exact inbound FK constraint name the schema produces (run a quick introspection: build a test_ml catalog and print `[fk.name for fk in model.schemas[ml_schema].tables["Directory_Dataset"].foreign_keys]`), and use that exact name — Chaise sources reference FK constraints by name. `array_d` renders the single related Path as one value (same approach as "Produced By").

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest tests/<dataset annotation test> -q`
Expected: PASS.

- [ ] **Step 6: (Manual/optional) Verify in Chaise**

Build a fresh catalog, run `add_files` over a small tree, open a directory Dataset's record page in Chaise (`/chaise/record/#<cat>/deriva-ml:Dataset/RID=<rid>`), confirm a "Folder" field shows the relative path. (Document the check; not an automated test.)

- [ ] **Step 7: Lint + commit**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/schema/annotations.py tests/`
```bash
git add src/deriva_ml/schema/annotations.py tests/
git commit -m "feat(chaise): show Directory_Dataset.Path as 'Folder' on Dataset page"
```

---

### Task 6: Migration helper to add `Directory_Dataset` to existing catalogs (eye-ai dev + prod)

**Files:**
- Create: `src/deriva_ml/schema/add_directory_dataset_table.py` (idempotent migration: add the `Directory_Dataset` table + FK to an existing catalog if absent)
- Test: `tests/schema/test_directory_dataset_table.py` (Modify — add an idempotency test)

**Context (verified live 2026-06-24):** `dev.eye-ai.org` and `www.eye-ai.org` (both `eye-ai` catalog) currently LACK the `Directory_Dataset` table (it's brand new) and have **0** existing `Directory`-tagged datasets. So the migration is **table-create only — NO historical data backfill needed**. `create_schema` only builds tables at catalog-creation; existing catalogs need the new table added explicitly (same propagation gap as the File/Directory vocab backfill).

**Interfaces:**
- Consumes: the `Directory_Dataset` `TableDef` shape from Task 1 (must match exactly so a migrated catalog is identical to a freshly-created one).
- Produces: `add_directory_dataset_table(ml, *, apply=False) -> bool` — returns True if the table was added (or would be, in dry-run), False if it already exists. Idempotent; safe to re-run.

- [ ] **Step 1: Write the failing idempotency test**

Add to `tests/schema/test_directory_dataset_table.py`:

```python
    def test_add_directory_dataset_table_idempotent(self, test_ml):
        """The migration is a no-op on a catalog that already has the table
        (fresh test_ml catalogs already include it via create_schema)."""
        from deriva_ml.schema.add_directory_dataset_table import add_directory_dataset_table

        # test_ml already has the table (created by create_schema), so the
        # migration must report 'already present' and not error.
        added = add_directory_dataset_table(test_ml, apply=True)
        assert added is False, "table already exists → migration should be a no-op"
        # Table still present and well-formed.
        assert "Directory_Dataset" in test_ml.model.model.schemas[test_ml.ml_schema].tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_directory_dataset_table.py::TestDirectoryDatasetTable::test_add_directory_dataset_table_idempotent -q`
Expected: FAIL — `ModuleNotFoundError: add_directory_dataset_table`.

- [ ] **Step 3: Implement the migration helper**

Create `src/deriva_ml/schema/add_directory_dataset_table.py`:

```python
"""Idempotent migration: add the Directory_Dataset table to an existing catalog.

``create_schema`` only builds tables at catalog-creation time, so catalogs that
predate the Directory_Dataset feature need this one-time, additive migration. The
table shape MUST match create_schema's exactly (see create_dataset_table) so a
migrated catalog is indistinguishable from a freshly-created one.
"""

from __future__ import annotations

from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["add_directory_dataset_table"]


def add_directory_dataset_table(ml, *, apply: bool = False) -> bool:
    """Add the ``Directory_Dataset`` table to ``ml``'s catalog if absent.

    Args:
        ml: A DerivaML instance bound to the target catalog.
        apply: When False (default), report what WOULD happen without writing
            (dry-run). When True, create the table.

    Returns:
        bool: True if the table was created (or, in dry-run, would be); False if
            it already exists.

    Example:
        >>> from deriva_ml import DerivaML  # doctest: +SKIP
        >>> ml = DerivaML(hostname="dev.eye-ai.org", catalog_id="eye-ai")  # doctest: +SKIP
        >>> add_directory_dataset_table(ml, apply=False)  # dry-run  # doctest: +SKIP
        True
        >>> add_directory_dataset_table(ml, apply=True)   # create it  # doctest: +SKIP
        True
    """
    model = ml.model.model
    schema = model.schemas[ml.ml_schema]
    if "Directory_Dataset" in schema.tables:
        logger.info("Directory_Dataset already present on %s; nothing to do.", ml.ml_schema)
        return False

    if not apply:
        logger.info("[dry-run] would create Directory_Dataset on %s.", ml.ml_schema)
        return True

    # Reuse the EXACT definition create_schema uses for fresh catalogs (factory
    # added in Task 1) so a migrated catalog is identical to a created one.
    from deriva_ml.schema.create_schema import directory_dataset_table_def

    schema.create_table(directory_dataset_table_def(ml.ml_schema))
    return True
```

(Remove the unused `builtin_types` import from the skeleton in Step 3 — the factory carries the column types. Keep the `logger` import.)

- [ ] **Step 4: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/schema/test_directory_dataset_table.py -q`
Expected: PASS (all table-shape + idempotency tests).

- [ ] **Step 5: Lint + commit**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/schema/`
```bash
git add src/deriva_ml/schema/add_directory_dataset_table.py src/deriva_ml/schema/create_schema.py tests/schema/test_directory_dataset_table.py
git commit -m "feat(schema): idempotent migration to add Directory_Dataset to existing catalogs"
```

- [ ] **Step 6: Run the migration against eye-ai — DEV FIRST, dry-run then apply (REQUIRES the PR to be merged + released, or run from the feature branch)**

This is an OUT-OF-BAND operation against live catalogs, NOT part of CI. Run only after the code is available (merged/released, or executed from the checked-out feature branch). Dev first:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from deriva_ml import DerivaML
from deriva_ml.schema.add_directory_dataset_table import add_directory_dataset_table
ml = DerivaML(hostname='dev.eye-ai.org', catalog_id='eye-ai')
print('dry-run:', add_directory_dataset_table(ml, apply=False))
print('apply  :', add_directory_dataset_table(ml, apply=True))
print('present:', 'Directory_Dataset' in ml.model.model.schemas[ml.ml_schema].tables)
"
```
Expected: dry-run True, apply True, present True.

- [ ] **Step 7: Run the migration against eye-ai PROD (www) — only after dev verified**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from deriva_ml import DerivaML
from deriva_ml.schema.add_directory_dataset_table import add_directory_dataset_table
ml = DerivaML(hostname='www.eye-ai.org', catalog_id='eye-ai')
print('dry-run:', add_directory_dataset_table(ml, apply=False))
print('apply  :', add_directory_dataset_table(ml, apply=True))
print('present:', 'Directory_Dataset' in ml.model.model.schemas[ml.ml_schema].tables)
"
```
Expected: dry-run True, apply True, present True. (No data backfill — both catalogs have 0 existing directory datasets, verified 2026-06-24.)

NOTE: writing schema to `www.eye-ai.org` is a production change — it is additive (new empty table) and must be Carl-authorized at run time (this plan records the intent; the actual prod run is an explicit, separately-confirmed step).

---

### Task 7: Full regression + PR update

**Files:**
- None (verification + PR metadata only)

- [ ] **Step 1: Run the affected suites**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_file.py tests/schema/ tests/dataset/ -q --timeout=600
```
Expected: all pass. (Dataset/bag suites are the integration surface for the new table + export.)

- [ ] **Step 2: Schema validator + full lint**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m deriva_ml.tools.validate_schema_doc && uv run ruff check src tests
```
Expected: validator agrees; ruff clean (modulo pre-existing unrelated errors confirmed earlier).

- [ ] **Step 3: Push and update PR #348**

```bash
git push
```
Update the PR title to "feat(files): directory datasets as a single-root tree with structured folder paths" and the body to describe: single-root containment fix + `Directory_Dataset.Path` (structured folder, replaces the Description suffix) + Python/bag accessor + Chaise Folder column. Confirm CI (`schema.md ↔ create_schema.py`) is green and `mergeStateStatus: CLEAN`.

- [ ] **Step 4: Record outcome in tacit-knowledge**

Append a dated entry to `/Users/carl/GitHub/DerivaML/tacit-knowledge.md` marking the `Directory_Dataset` design IMPLEMENTED (table + add_files writer + bare Description + Python/bag accessor + Chaise Folder column), noting any walker change Task 4 needed (or that export was automatic), and that it superseded the Description-suffix.

---

## Self-Review

**Spec coverage:**
- Goal #1 (File row per file) — already implemented (streaming insert); unchanged by this plan. ✓
- Goal #2 (nested datasets mirroring structure, single root) — already implemented (Task: single-root fix, prior commit on the branch); this plan adds the *navigability* via structured Path. ✓
- "Consumer can determine the directory name" — Task 2 (write), Task 3 (Python live + bag accessor), Task 5 (Chaise). ✓
- `is_directory` predicate (user request, complements `.path`) — Task 3 (live + bag), defined as `.path is not None`. ✓
- Bag + live (user choice) — Task 3 (both accessors) + Task 4 (bag export). ✓
- Chaise display (user requirement) — Task 5. ✓
- Update eye-ai dev + prod (user requirement) — Task 6 (migration helper + dev-then-prod run; verified table-create-only, 0 datasets to backfill). ✓
- Supersede Description-suffix (user choice "fold into #348, drop the suffix") — Task 2 Step 3 reverts Description to bare; deletes the suffix test. ✓
- Relative path, no absolute leak (global constraint) — Task 2 stores `relative_to(ingest_root)`. ✓

**Placeholder scan:** Task 3 Step 5/7 and Task 4/5 reference "the bag test file" / "the Dataset annotation builder" by discovery (grep/ls) rather than a hard-coded path, because those module names must be confirmed against the tree at execution time; each step gives the exact discovery command and the adaptation rule. Acceptable — these are discovery steps, not vague TODOs. Task 6 Step 3 has a deliberate refactor instruction (extract a shared `_directory_dataset_table_def` factory so Task 1's create-path and the migration cannot drift) rather than duplicated DDL — DRY.

**Type consistency:** `path -> str | None` and `is_directory -> bool` consistent across Task 3 (live) and Task 3 (bag). `Directory_Dataset` columns `{Dataset, Path}` and FK `Directory_Dataset_Dataset_fkey` consistent across Tasks 1, 3, 4, 5, 6. `Path` value semantics (`"."` for root, `relative_to(ingest_root).as_posix()` otherwise) consistent across Task 2 (write) and Tasks 3/5 (read/display). The shared `_directory_dataset_table_def` factory (Task 6 Step 3) guarantees the create-path (Task 1) and migration-path (Task 6) produce identical tables.

---

## Amendment (2026-06-27): root Path stores its basename, not "."

The original design stored the ingest root's `Directory_Dataset.Path` as `"."`.
As of the 2026-06-27 root-path-name change, the root stores its directory
basename (e.g. `cifar10_source`) so the catalog "Folder" column is
self-describing. Root identification moved from `source_directory == "."` to the
structural `Dataset.is_source_root` / `DatasetBag.is_source_root` accessor, which
works on both old (`"."`) and new catalogs — no backfill required. See
`docs/superpowers/specs/2026-06-27-directory-dataset-root-path-name-design.md`.

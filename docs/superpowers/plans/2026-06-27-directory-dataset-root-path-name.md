# Directory Dataset Root Path Name — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The root node of an `add_files` nested File-dataset tree records its directory basename (e.g. `cifar10_source`) in `Directory_Dataset.Path` instead of the literal `"."`, and a new structural `is_source_root` accessor replaces every `source_directory == "."` root-identification reader across three repos.

**Architecture:** Add `_root_path_name()` beside `_root_description()` in deriva-ml's `core/mixins/file.py` (shared name precedence: `root_name or ingest_root.name or "root"`); change the one writer line so the root stores that name. Add an `is_source_root` property to both `Dataset` (live) and `DatasetBag` (offline) defined structurally as "is a directory dataset AND has no parent that is a directory dataset" — name-independent, so it works on both pre-change (`"."`) and new catalogs. Migrate the five `== "."` readers to `is_source_root`. No schema migration, no catalog backfill.

**Tech Stack:** Python 3.13, `uv`, pytest, deriva-py path/ORM APIs, hydra-zen (cifar-example only), setuptools_scm + `bump-version`.

## Global Constraints

- **deriva-ml ships first.** It defines `is_source_root`; cifar-example and deriva-skills consume it. Order: all deriva-ml tasks → release → cifar-example lock bump → deriva-skills.
- **`is_source_root` rule (verified live on catalog 328):** `self.is_directory and not any(p.is_directory for p in self.list_dataset_parents())`. Root `AT4` → True; `train`/`test` children → False; `split_dataset()` outputs (parents but no `Directory_Dataset` row) → False.
- **Root Path precedence = root Description precedence:** `root_name or ingest_root.name or "root"`. Never empty. Path and Description name the root identically.
- **Children Path unchanged:** still `directory.relative_to(ingest_root).as_posix()`.
- **No schema migration** (only the `Directory_Dataset.Path` column *comment* changes) and **no catalog backfill** (`is_source_root` is structural, works on old `"."` catalogs).
- **Use `uv` for everything**: `uv run python -m pytest …`, `uv run ruff …`. Never bare `pytest`/`python`/`ruff`.
- **Google-style docstrings** with runnable `Example:` on every new function/property.
- **Dirty-tree override for live tests:** prefix live-catalog test commands with `DERIVA_ML_ALLOW_DIRTY=true`.
- **deriva-ml is a shared multi-agent repo:** all deriva-ml work happens on branch `directory-dataset-root-path-name` (already created; the spec is committed there). Do not merge to main without review.

---

### Task 1: `_root_path_name` helper + writer change (deriva-ml)

**Files:**
- Modify: `deriva-ml/src/deriva_ml/core/mixins/file.py` (add `_root_path_name` after `_root_description` at ~line 144; change the writer at line 382)
- Test: `deriva-ml/tests/core/test_file.py` (update `test_add_files_directory_datasets_record_path` at line 145; `test_dataset_source_directory_and_is_directory_accessor` at line 264)

**Interfaces:**
- Consumes: `_root_description(ingest_root, root_name, description)` (existing, file.py:105); `add_files` `root_name` param (existing).
- Produces: `_root_path_name(ingest_root: Path, root_name: str | None) -> str` — returns `root_name or ingest_root.name or "root"`. Used only by the `add_files` writer.

- [ ] **Step 1: Update the writer test to the new contract (failing)**

In `tests/core/test_file.py`, `test_add_files_directory_datasets_record_path` (line 173), change the root assertion from `"."` to the ingest-root basename. The fixture's `test_dir` is named `"test_dir"` (file.py setup line 28), so:

```python
        # Root Directory_Dataset.Path is now the ingest-root basename, not ".".
        assert path_by_dataset[file_dataset.dataset_rid] == "test_dir"
        child_paths = {path_by_dataset[c.dataset_rid] for c in file_dataset.list_dataset_children()}
        assert child_paths == {"d1", "d2"}
```

Also update the docstring line 147 from `the ingest root stores '.'.` to `the ingest root stores its directory basename.`

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py::TestFileMixin::test_add_files_directory_datasets_record_path -v`
Expected: FAIL — `assert '.' == 'test_dir'` (writer still stores `"."`).

(If the test class name differs, discover it with `uv run python -m pytest tests/core/test_file.py --collect-only -q | grep record_path`.)

- [ ] **Step 3: Add `_root_path_name` helper**

In `src/deriva_ml/core/mixins/file.py`, immediately after `_root_description` (ends line 144), add:

```python
def _root_path_name(ingest_root: Path, root_name: str | None) -> str:
    """Folder name to store in ``Directory_Dataset.Path`` for the ingest root.

    Shares precedence with :func:`_root_description` so the root dataset's
    ``Path`` and ``Description`` name it identically. Unlike the description,
    there is no caller-``description`` fallback — the root's *folder* is its
    basename, not a prose blurb — but the empty-basename guard is the same.

    Precedence (first truthy value wins):
    1. ``root_name`` (explicit caller override)
    2. ``ingest_root.name`` (basename, if non-empty)
    3. ``"root"`` (sentinel — never returns an empty string)

    Args:
        ingest_root: The common-ancestor directory of all ingested files.
        root_name: Caller-supplied name for the root dataset, or ``None``.

    Returns:
        str: A non-empty folder name for the root's ``Directory_Dataset.Path``.

    Example:
        >>> _root_path_name(Path("/tmp/abc/cifar10_source"), None)
        'cifar10_source'
        >>> _root_path_name(Path("/tmp/abc/cifar10_source"), "CIFAR-10 source")
        'CIFAR-10 source'
        >>> _root_path_name(Path("/"), None)
        'root'
    """
    return root_name or ingest_root.name or "root"
```

- [ ] **Step 4: Change the writer to use it**

In `add_files` (file.py:378-386), change the `"Path"` value (line 382) from:

```python
                    "Path": "." if directory == ingest_root else directory.relative_to(ingest_root).as_posix(),
```

to:

```python
                    "Path": _root_path_name(ingest_root, root_name)
                    if directory == ingest_root
                    else directory.relative_to(ingest_root).as_posix(),
```

Also update the comment at line 376 from `(the root stores ".")` to `(the root stores its basename)`.

- [ ] **Step 5: Update the second accessor test to the new contract**

In `test_dataset_source_directory_and_is_directory_accessor` (line 278), change:

```python
        assert root.source_directory == "."
```
to:
```python
        assert root.source_directory == "test_dir"
```

(The `plain.source_directory is None` and child `{"d1","d2"}` assertions stay unchanged.)

- [ ] **Step 6: Run both tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py::TestFileMixin::test_add_files_directory_datasets_record_path tests/core/test_file.py::TestFileMixin::test_dataset_source_directory_and_is_directory_accessor -v`
Expected: PASS (2 passed). Also run the doctest on the new helper:
`uv run python -m pytest --doctest-modules src/deriva_ml/core/mixins/file.py -k _root_path_name -q` → expected PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/core/mixins/file.py tests/core/test_file.py
git commit -m "feat(add_files): root Directory_Dataset.Path stores basename, not '.'

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `is_source_root` accessor on `Dataset` (live) (deriva-ml)

**Files:**
- Modify: `deriva-ml/src/deriva_ml/dataset/dataset.py` (add property after `is_directory` at ~line 216; update `source_directory` docstring at line 171)
- Test: `deriva-ml/tests/core/test_file.py` (new live test)

**Interfaces:**
- Consumes: `Dataset.is_directory` (property, dataset.py:197); `Dataset.list_dataset_parents()` (dataset.py:2331) → `list[Dataset]`, each with `.is_directory`.
- Produces: `Dataset.is_source_root` (property) → `bool`. `True` iff this dataset is a directory dataset and none of its parents is a directory dataset.

- [ ] **Step 1: Write the failing live test**

In `tests/core/test_file.py`, add a method to the same test class (alongside `test_dataset_source_directory_and_is_directory_accessor`):

```python
    def test_dataset_is_source_root_accessor(self, file_table_setup):
        """is_source_root is True for the add_files tree root, False for its
        directory children and for non-directory datasets. Identification is
        structural (parent graph), not based on the Path string."""
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            root = exe.add_files(filespecs, description="Ingest run")
            plain = exe.create_dataset(dataset_types="Complete", description="not a dir")

        # The root is the source root.
        assert root.is_source_root is True
        # Directory children are NOT source roots (they have a directory parent).
        assert all(not child.is_source_root for child in root.list_dataset_children())
        # A non-directory dataset is never a source root.
        assert plain.is_source_root is False

        # Identity is structural, independent of the Path value: locate the root
        # among all CIFAR-tree datasets via is_source_root and confirm it's `root`.
        tree = [root] + list(root.list_dataset_children())
        roots = [d for d in tree if d.is_source_root]
        assert [d.dataset_rid for d in roots] == [root.dataset_rid]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py::TestFileMixin::test_dataset_is_source_root_accessor -v`
Expected: FAIL — `AttributeError: 'Dataset' object has no attribute 'is_source_root'`.

- [ ] **Step 3: Implement the property**

In `src/deriva_ml/dataset/dataset.py`, after the `is_directory` property (ends line 216), add:

```python
    @property
    def is_source_root(self) -> bool:
        """Whether this dataset is the ROOT of an ``add_files`` directory tree.

        ``True`` iff this dataset is a directory dataset (:attr:`is_directory`)
        and none of its parent datasets is itself a directory dataset. This is
        the structural, name-independent way to identify the tree root: it does
        NOT depend on :attr:`source_directory` holding any particular string, so
        it works on catalogs built before the root began recording its basename
        (where the root's path is the legacy ``"."``) as well as new ones.

        Datasets with no ``Directory_Dataset`` row (plain datasets, split-dataset
        outputs) return ``False`` even when they are the parent of other
        datasets — only directory datasets can be source roots.

        Returns:
            bool: True if this dataset is the root of an add_files tree.

        Example:
            >>> root = exe.add_files(specs, description="ingest")  # doctest: +SKIP
            >>> root.is_source_root  # doctest: +SKIP
            True
            >>> any(c.is_source_root for c in root.list_dataset_children())  # doctest: +SKIP
            False
        """
        if not self.is_directory:
            return False
        return not any(parent.is_directory for parent in self.list_dataset_parents())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py::TestFileMixin::test_dataset_is_source_root_accessor -v`
Expected: PASS.

- [ ] **Step 5: Update the `source_directory` docstring**

In `dataset.py`, the `source_directory` property docstring (line 176-179 and example 187-188): replace `(the\n        ingest root stores ``"."``)` with `(the ingest root stores its directory basename)`, and change the example output from `'.'` to a basename, e.g.:

```python
            >>> root.source_directory  # doctest: +SKIP
            'cifar10_source'
```

Add a final docstring line: `See :attr:`is_source_root` to identify the tree root structurally (independent of this string).`

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/dataset.py tests/core/test_file.py
git commit -m "feat(dataset): add structural is_source_root accessor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `is_source_root` accessor on `DatasetBag` (offline) + bag tests (deriva-ml)

**Files:**
- Modify: `deriva-ml/src/deriva_ml/dataset/dataset_bag.py` (add property after `is_directory` at line 283; update `source_directory` docstring at line 228)
- Test: `deriva-ml/tests/dataset/test_bag_api_coverage.py` (update root assertions at lines 519, 530; add a legacy-`"."` resolution test)

**Interfaces:**
- Consumes: `DatasetBag.is_directory` (dataset_bag.py:261); `DatasetBag.list_dataset_parents()` (dataset_bag.py:943) → `list[DatasetBag]`, each with `.is_directory`.
- Produces: `DatasetBag.is_source_root` (property) → `bool`, same semantics as the live one.

- [ ] **Step 1: Update the bag test root assertions + add an `is_source_root` assertion (failing)**

In `tests/dataset/test_bag_api_coverage.py`, the directory-bag test (the `_make_test_tree` root is named `"test_dir"`):

Change line 519:
```python
        assert root_dataset.source_directory == "test_dir", "live: root Dataset.source_directory must be the basename"
```
Change line 530-532:
```python
        assert bag.source_directory == "test_dir", (
            f"bag root .source_directory should be 'test_dir', got {bag.source_directory!r}"
        )
```
After line 533 (`assert bag.is_directory is True...`), add:
```python
        # Structural root identification works offline on the bag.
        assert root_dataset.is_source_root is True, "live root must be is_source_root"
        assert bag.is_source_root is True, "bag root must be is_source_root"
        assert not any(c.is_source_root for c in bag.list_dataset_children()), (
            "bag directory children must not be source roots"
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/dataset/test_bag_api_coverage.py -k "directory" -v`
Expected: FAIL — first on `source_directory == "test_dir"` (writer now stores basename so this *passes* after Task 1; the genuinely failing line is the new `is_source_root` assertion → `AttributeError`). If Task 1 is already committed the source_directory lines pass and only `is_source_root` fails; that is the expected failure to drive Step 3.

- [ ] **Step 3: Implement the bag property**

In `src/deriva_ml/dataset/dataset_bag.py`, after the `is_directory` property (ends line 283), add:

```python
    @property
    def is_source_root(self) -> bool:
        """Whether this bag is the ROOT of an ``add_files`` directory tree.

        ``True`` iff this dataset is a directory dataset (:attr:`is_directory`)
        and none of its parent datasets is itself a directory dataset. Structural
        and name-independent — it does NOT depend on :attr:`source_directory`
        holding any particular string, so it resolves the root on both legacy
        catalogs (root path ``"."``) and new ones (root path = basename).

        The bag is offline — this reads the bag's local SQLite mirror; no catalog
        connection is used.

        Returns:
            bool: True if this bag is the root of an add_files tree.

        Example:
            >>> root_bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> root_bag.is_source_root  # doctest: +SKIP
            True
        """
        if not self.is_directory:
            return False
        return not any(parent.is_directory for parent in self.list_dataset_parents())
```

- [ ] **Step 4: Run the bag tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/dataset/test_bag_api_coverage.py -k "directory" -v`
Expected: PASS.

- [ ] **Step 5: Add a legacy-`"."` resolution unit test**

This proves no backfill is needed: a dataset whose stored Path is the legacy `"."` is still found by `is_source_root`. Add to `tests/core/test_file.py` (offline, fabricate the legacy shape by directly writing the root's Directory_Dataset.Path back to "."):

```python
    def test_is_source_root_resolves_legacy_dot_path(self, file_table_setup):
        """A pre-change catalog stores the root Path as '.'; is_source_root must
        still identify it (identity is structural, not string-based)."""
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            root = exe.add_files(filespecs, description="Ingest run")

        # Simulate a legacy catalog: rewrite the root's Path back to ".".
        pb = ml_instance.pathBuilder()
        dd = pb.schemas[ml_instance.ml_schema].tables["Directory_Dataset"]
        dd.update([{"Dataset": root.dataset_rid, "Path": "."}])

        legacy_root = ml_instance.lookup_dataset(root.dataset_rid)
        assert legacy_root.source_directory == "."          # legacy shape restored
        assert legacy_root.is_source_root is True           # still found structurally
```

If `Directory_Dataset` lacks an RID key that `update` needs, use the table's update form already used elsewhere (grep `Directory_Dataset` writes); the key column is `Dataset` (`create_schema.py:73` `KeyDef(columns=["Dataset"])`), so `update([...])` keyed on `Dataset` is valid.

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py::TestFileMixin::test_is_source_root_resolves_legacy_dot_path -v`
Expected: PASS.

- [ ] **Step 6: Update the bag `source_directory` docstring**

In `dataset_bag.py`, `source_directory` docstring (lines 232-233, example 246-247): same edit as the live one — drop `(the\n        ingest root stores ``"."``)`, change example output `'.'` → `'cifar10_source'`, add the `See :attr:`is_source_root`` pointer.

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_bag_api_coverage.py tests/core/test_file.py
git commit -m "feat(bag): is_source_root accessor + legacy-path resolution test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Schema comment + remaining docs (deriva-ml)

**Files:**
- Modify: `deriva-ml/src/deriva_ml/schema/create_schema.py:67-70` (Path column comment)
- Modify: `deriva-ml/docs/superpowers/plans/2026-06-24-directory-dataset-path.md` (note the convention change)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the schema column comment**

In `create_schema.py`, the `Path` `ColumnDef` comment (lines 67-70):

```python
                comment=(
                    "Source directory this dataset represents, relative to "
                    "the ingest root. The ingest root stores its own directory "
                    "basename (the same name as its Description); children store "
                    "paths relative to it. Identify the tree root structurally "
                    "via Dataset.is_source_root, not by matching this string."
                ),
```

- [ ] **Step 2: Note the change in the prior design plan**

Append a dated note to `docs/superpowers/plans/2026-06-24-directory-dataset-path.md`:

```markdown

---

## Amendment (2026-06-27): root Path stores its basename, not "."

The original design stored the ingest root's `Directory_Dataset.Path` as `"."`.
As of the 2026-06-27 root-path-name change, the root stores its directory
basename (e.g. `cifar10_source`) so the catalog "Folder" column is
self-describing. Root identification moved from `source_directory == "."` to the
structural `Dataset.is_source_root` / `DatasetBag.is_source_root` accessor, which
works on both old (`"."`) and new catalogs — no backfill required. See
`docs/superpowers/specs/2026-06-27-directory-dataset-root-path-name-design.md`.
```

- [ ] **Step 3: Verify the schema comment doesn't break a schema test**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/ -k "schema and comment" -q || true`
Then run the full create-schema-touching tests:
`uv run python -m pytest tests/ -k "create_schema or annotation" -q`
Expected: PASS or no-tests-collected (the comment is free text; no test should assert its exact value — if one does, update it to the new text).

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/schema/create_schema.py docs/superpowers/plans/2026-06-24-directory-dataset-path.md
git commit -m "docs(schema): Directory_Dataset.Path root stores basename; point to is_source_root

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: deriva-ml full suite + release

**Files:** none (validation + release).

**Interfaces:** Produces a released deriva-ml version that the consumers pin to.

- [ ] **Step 1: Lint + format**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: clean. If format flags the touched files, run `uv run ruff format src tests` and re-commit.

- [ ] **Step 2: Run the directory/dataset + file test groups**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/core/test_file.py tests/dataset/test_bag_api_coverage.py -v`
Expected: all PASS. These are the live-catalog tests; if the harness needs a host, they self-skip without `DERIVA_HOST` — in that case run them against the local test catalog the suite normally uses (the repo's conftest provisions it). If any assert still references `"."`, grep `tests/ -rn '== "\."'` and fix.

- [ ] **Step 3: Full offline suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/ -q -m "not integration"`
Expected: PASS (no `"."`-root regressions elsewhere).

- [ ] **Step 4: Confirm the branch is clean and review-ready**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && git status --short && git log --oneline main..HEAD`
Expected: clean tree; commits for Tasks 1-4 + the spec. **STOP here for the whole-branch review** (subagent-driven-development's final review) before releasing — deriva-ml is shared; do not bump/merge without it.

- [ ] **Step 5: Version bump (after review approves)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && git checkout main && git merge --no-ff directory-dataset-root-path-name && uv run bump-version minor`
Expected: a new minor tag created + pushed (e.g. 1.53.0 → 1.54.0). Record the new version for Task 6.

---

### Task 6: cifar-example readers + lock bump

**Files:**
- Modify: `deriva-ml-cifar-example/src/scripts/load_cifar10.py:170-178`
- Modify: `deriva-ml-cifar-example/src/scripts/_cifar10_upload.py:318-320` (comment)
- Modify: `deriva-ml-cifar-example/tests/test_lineage_connected.py:211-221`
- Modify: `deriva-ml-cifar-example/uv.lock` / `pyproject.toml` (pin to the new deriva-ml)

**Interfaces:**
- Consumes: `Dataset.is_source_root` (from the released deriva-ml).

- [ ] **Step 1: Switch the loader's root filter**

In `src/scripts/load_cifar10.py`, replace lines 170-178:

```python
    # find_datasets(sort=True) returns datasets ordered by RCT desc (newest
    # first).  We then filter to CIFAR_Source roots: is_source_root identifies
    # the root of the nested add_files tree (not train/test children),
    # independent of the stored Directory_Dataset.Path value.
    all_datasets = list(ml.find_datasets(sort=True))
    candidates = [
        d
        for d in all_datasets
        if "CIFAR_Source" in d.dataset_types and d.is_source_root
    ]
```

- [ ] **Step 2: Switch the lineage test's root filter**

In `tests/test_lineage_connected.py`, replace lines 211-221:

```python
        source_root_rid: str | None = None
        for row in cifar_source_rows:
            ds = ml.lookup_dataset(row["Dataset"])
            if ds.is_source_root:
                source_root_rid = row["Dataset"]
                break

        assert source_root_rid is not None, (
            f"Could not find a CIFAR_Source dataset that is_source_root. "
            f"Candidates: {[r['Dataset'] for r in cifar_source_rows]}"
        )
```

- [ ] **Step 3: Update the upload-phase comment**

In `src/scripts/_cifar10_upload.py`, the comment at lines 319-320:

```python
    # The root dataset (the add_files tree root, identified via is_source_root)
    # holds labels.csv; partition children (source_directory "train" / "test")
    # hold the images.
```

(No code change here — `source_ds` is already the root passed in, and the child filter `source_directory in {"train","test"}` is unchanged.)

- [ ] **Step 4: Bump the deriva-ml pin**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example && uv lock --upgrade-package deriva-ml && uv sync`
Then confirm: `uv run python -c "import deriva_ml, importlib.metadata as m; print(m.version('deriva-ml'))"` shows the new version, and `uv run python -c "from deriva_ml import Dataset; print(hasattr(Dataset, 'is_source_root'))"` prints `True`.

- [ ] **Step 5: Config smoke tests + the source-dataset test**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example && uv run python -m pytest tests/test_find_source_dataset.py -v`
Expected: PASS. (This test stubs datasets; Task 7 updates its stub if it asserts on `source_directory == "."`.)

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
git add src/scripts/load_cifar10.py src/scripts/_cifar10_upload.py tests/test_lineage_connected.py uv.lock pyproject.toml
git commit -m "feat: identify add_files source root via is_source_root (deriva-ml bump)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: cifar-example `test_find_source_dataset` stub + live verification

**Files:**
- Modify: `deriva-ml-cifar-example/tests/test_find_source_dataset.py` (stub at line 78, docstrings at 20, 69-72)

**Interfaces:** Consumes the updated `_find_latest_source_dataset_rid` (Task 6) which now filters on `is_source_root`.

- [ ] **Step 1: Inspect the stub**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example && uv run python -m pytest tests/test_find_source_dataset.py -v` and read `tests/test_find_source_dataset.py` around lines 18-90. The stub `_make_dataset(rid, types, source_dir)` currently sets `source_directory`; the loader now reads `is_source_root`. The stub must expose `is_source_root` matching the intended root.

- [ ] **Step 2: Update the stub to provide `is_source_root`**

In `_make_dataset` (and the `"2-ROOT"`/child rows around line 78), set `is_source_root=True` for the root stub and `False` for children, e.g. if the stub is a `SimpleNamespace`/Mock:

```python
    def _make_dataset(rid, types, source_dir, is_root):
        return SimpleNamespace(
            dataset_rid=rid,
            dataset_types=types,
            source_directory=source_dir,
            is_source_root=is_root,
        )
    # ...
    _make_dataset("2-ROOT", ["CIFAR_Source"], "cifar10_source", is_root=True),
    _make_dataset("2-TRAIN", ["CIFAR_Source"], "train", is_root=False),
```

Update the test docstrings (lines 20, 69-72) to describe identification via `is_source_root` rather than `source_directory == "."`.

- [ ] **Step 3: Run the test**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example && uv run python -m pytest tests/test_find_source_dataset.py -v`
Expected: PASS.

- [ ] **Step 4: Live end-to-end verification**

Load a fresh catalog and confirm the root Folder reads its basename and the loader finds the root:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
DERIVA_ML_ALLOW_DIRTY=true uv run python src/scripts/load_cifar10.py \
  --hostname localhost --create-catalog rootname_verify --num-images 1100 --phase all
```

Then check the root Path:
```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from deriva_ml import DerivaML
ml = DerivaML(hostname='localhost', catalog_id='<NEW_ID>')
import deriva_ml
roots = [d for d in ml.find_datasets() if d.is_source_root]
print('source roots:', [(d.dataset_rid, d.source_directory) for d in roots])
"
```
Expected: exactly one source root whose `source_directory` is the loader's source basename (NOT `"."`).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
git add tests/test_find_source_dataset.py
git commit -m "test: find-source-dataset stub uses is_source_root

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: deriva-skills template readers

**Files:**
- Modify: `deriva-ml-skills/skills/setup-ml-catalog/scripts/loader_orchestrator_template.py:70-78`
- Modify: `deriva-ml-skills/skills/setup-ml-catalog/scripts/upload_phase_template.py:95-103`

**Interfaces:** Consumes `Dataset.is_source_root` (template code users run against the released deriva-ml).

- [ ] **Step 1: Switch the orchestrator template root filter**

In `loader_orchestrator_template.py`, replace the docstring tail + filter (lines 70-78):

```python
    When ``--phase upload`` runs in isolation, the RID isn't threaded from
    register — find it from the catalog: newest dataset typed FILE_DATASET_TYPE
    that is the add_files tree root (``is_source_root``).
    """
    candidates = [
        d
        for d in ml.find_datasets(sort=True)  # newest first
        if FILE_DATASET_TYPE in d.dataset_types and d.is_source_root
    ]
```

- [ ] **Step 2: Switch the upload-phase template root check**

In `upload_phase_template.py`, line 102 uses `partition == "."` where `partition` is a key iterated from a `partitions` set/dict. The root partition is whichever child set corresponds to the tree root. Replace the `"."` sentinel with the root's actual partition name. Read lines 88-110 to see how `partitions` and `children` are built (children come from `source_ds.list_dataset_children()` keyed by `source_directory`). The root itself is `source_ds`; change:

```python
            part_ds = source_ds if partition == "." else children.get(partition)
```
to identify the root partition by the source dataset's own `source_directory` (its basename) rather than `"."`:

```python
            # The root partition is the source dataset itself; children are keyed
            # by their source_directory (e.g. "train"/"test").
            part_ds = source_ds if partition == source_ds.source_directory else children.get(partition)
```

Verify the surrounding loop's `partitions` includes `source_ds.source_directory` for the root case; if `partitions` is built only from children, the root branch was dead in the template anyway — in that case simplify to `part_ds = children.get(partition)` and drop the root special-case. Choose based on what lines 88-101 actually construct (read them first; do not guess).

- [ ] **Step 3: Lint the templates (syntax only — these are template scripts)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-skills && uv run ruff check skills/setup-ml-catalog/scripts/loader_orchestrator_template.py skills/setup-ml-catalog/scripts/upload_phase_template.py 2>/dev/null || python -m py_compile skills/setup-ml-catalog/scripts/loader_orchestrator_template.py skills/setup-ml-catalog/scripts/upload_phase_template.py`
Expected: no syntax errors. (deriva-ml-skills may not have a uv project; `py_compile` is the fallback.)

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-skills
git add skills/setup-ml-catalog/scripts/loader_orchestrator_template.py skills/setup-ml-catalog/scripts/upload_phase_template.py
git commit -m "fix(templates): identify add_files source root via is_source_root, not '.'

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Structural `is_source_root` rule → Tasks 2 (live), 3 (bag). ✓
- Writer change to basename + precedence shared with Description → Task 1. ✓
- Empty-basename → `"root"` sentinel → Task 1 (`_root_path_name`). ✓
- Schema comment + docstrings + prior-plan note → Tasks 2, 3, 4. ✓
- 4 existing `== "."` tests → basename: test_file.py:173 (T1S1), :278 (T1S5); test_bag:519,530 (T3S1). ✓
- New tests: root True/children False/splits False/non-dir False (T2,T3), identity-not-name (T2S1), legacy-`"."` resolves (T3S5). ✓
- 5 readers migrated: load_cifar10:177 (T6S1), test_lineage_connected:214 (T6S2), _cifar10_upload:319 (T6S3 comment), loader_orchestrator_template:75 (T8S1), upload_phase_template:102 (T8S2). ✓
- No migration/backfill → asserted by T3S5 legacy test + spec "out of scope". ✓
- deriva-ml-first ordering + minor bump → Global Constraints + Task 5. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases". The one conditional instruction (T8S2: "read lines 88-101, choose based on what they construct") is bounded with an explicit fallback, not an open TODO — the template's `partitions` construction genuinely must be read before editing, and both branches are spelled out.

**3. Type consistency:** `is_source_root` is a `bool` property on both `Dataset` and `DatasetBag`, implemented identically (`is_directory and not any(p.is_directory for p in list_dataset_parents())`), consumed as `d.is_source_root` everywhere. `_root_path_name(ingest_root: Path, root_name: str | None) -> str` matches its sole call site in Task 1 Step 4. Test root basename is `"test_dir"` for both `test_file.py` and the bag test (`_make_test_tree` → `base/test_dir`), used consistently in T1S1, T1S5, T3S1.

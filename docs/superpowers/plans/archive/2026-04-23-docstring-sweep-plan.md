# Docstring Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every public method in `src/deriva_ml/` to the full docstring contract (one-line summary + extended description + Args/Returns/Raises/Example), eliminate 12 leaked public names via `_prefix` renames, and delete 7 dead symbols — all in per-module commits on the `feature/docstring-sweep` branch.

**Architecture:** Each task covers one module (or a tightly-coupled pair). The task flow is always: (1) apply docstring edits, (2) apply renames/deletions, (3) add inline comments, (4) run fast unit tests, (5) commit. Priority 1 (worst areas) and Priority 2 (named gaps) go first; reviewer sign-off is expected before Priority 3 and 4 begin. A single follow-on task updates test files that reference renamed symbols, and a final task wires up `--doctest-modules` and confirms docs build.

**Tech Stack:** Python ≥ 3.12, Pydantic v2, `uv run pytest`, `pytest --doctest-modules` with `# doctest: +SKIP` for catalog-dependent examples, `uv run ruff check / ruff format`, `uv run mkdocs build --strict`.

---

## File Structure

| Task | Files modified |
|---|---|
| Task 0 — prep | `src/deriva_ml/core/mixins/execution.py` (grep only, no edits yet) |
| Task 1 — doctest infra | `pyproject.toml`, `CLAUDE.md` |
| Task 2 — `core/mixins/dataset.py` | `src/deriva_ml/core/mixins/dataset.py` |
| Task 3 — `core/mixins/annotation.py` | `src/deriva_ml/core/mixins/annotation.py` |
| Task 4 — `feature.py` | `src/deriva_ml/feature.py` |
| Task 5 — `dataset/dataset_bag.py` | `src/deriva_ml/dataset/dataset_bag.py` |
| Task 6 — `execution/execution.py` | `src/deriva_ml/execution/execution.py` |
| Task 7 — `dataset/dataset.py` | `src/deriva_ml/dataset/dataset.py` |
| Task 8 — `core/mixins/asset.py` | `src/deriva_ml/core/mixins/asset.py` |
| Task 9 — rename group A | `src/deriva_ml/core/mixins/path_builder.py`, `src/deriva_ml/core/constants.py` |
| Task 10 — rename group B | `src/deriva_ml/core/logging_config.py`, `src/deriva_ml/core/schema_diff.py` |
| Task 11 — rename group C | `src/deriva_ml/core/mixins/rid_resolution.py`, `src/deriva_ml/asset/asset_record.py`, `src/deriva_ml/core/mixins/workflow.py` |
| Task 12 — `core/base.py` | `src/deriva_ml/core/base.py` |
| Task 13 — `tools/validate_schema_doc.py` | `src/deriva_ml/tools/validate_schema_doc.py`, `tests/tools/` |
| Task 14 — `core/mixins/execution.py` | `src/deriva_ml/core/mixins/execution.py` |
| Task 15 — Tier 4 module sweep | ~20 remaining modules (catalog, dataset, execution, model, schema, local_db sub-modules) |
| Task 16 — test file renames | `tests/**/*.py` (update references to renamed symbols) |
| Task 17 — final verification | run test suites + mkdocs build |

---

## Task 0: Preparatory Verification — `start_upload()` Visibility

**Files:**
- Read-only: `/Users/carl/GitHub/deriva-ml-model-template` (grep only)
- No edits in this task

- [ ] **Step 1: Grep the template repo for `start_upload` callers**

```bash
grep -r "start_upload" /Users/carl/GitHub/deriva-ml-model-template 2>/dev/null && echo "FOUND" || echo "NOT FOUND"
```

Expected: if `NOT FOUND` → `start_upload` will be renamed to `_start_upload` in Task 14.
If `FOUND` → keep it public, add a complete docstring, and document the external caller in Task 14's commit message.

- [ ] **Step 2: Record the decision**

Write the decision as a comment at the top of Task 14 (or note it in the commit message). No code changes yet.

---

## Task 1: Doctest Infrastructure Setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `--doctest-modules` to pytest configuration**

In `pyproject.toml`, find the `[tool.pytest.ini_options]` section and add `--doctest-modules` to `addopts`, plus the `doctest_optionflags` needed to handle whitespace variation:

```toml
[tool.pytest.ini_options]
addopts = "--doctest-modules"
doctest_optionflags = ["NORMALIZE_WHITESPACE", "ELLIPSIS"]
```

If there is no existing `addopts`, add it. If there is, append `--doctest-modules` to the existing value.

Note: `--doctest-modules` collects doctests from all `src/deriva_ml/` Python files. Examples that require a live catalog must be annotated `# doctest: +SKIP` or they will fail in CI.

- [ ] **Step 2: Verify fast unit tests still pass with doctest collection enabled**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS (any existing docstring examples that lack `# doctest: +SKIP` will surface here as failures; fix them before committing)

- [ ] **Step 3: Update CLAUDE.md with doctest guidance**

Find the `## Best Practices & Patterns` section in `CLAUDE.md` and add a new subsection after the last pattern:

```markdown
### Doctest Policy

New public methods must include an `Example:` block with a passing doctest or a `+SKIP` annotation before merge. The pattern for catalog-dependent examples:

```python
Example:
    >>> ml = DerivaML(hostname="example.org", catalog_id="42")  # doctest: +SKIP
    >>> result = ml.list_datasets()  # doctest: +SKIP
```

Examples that run without a catalog (no `+SKIP` needed):
- `ExecutionConfiguration` and `DatasetSpecConfig` construction
- Enum values and `DerivaMLConfig` field validation
- `FeatureRecord` creation with a stub container
- Selector factory classmethods that operate on a pre-built record
```

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add pyproject.toml CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(docstrings): configure --doctest-modules + CLAUDE.md doctest policy

Adds pytest --doctest-modules so Example: blocks are verified on every
test run. Catalog-dependent examples use # doctest: +SKIP. Pure-Python
examples (Pydantic construction, enum values, selector factories) run
for real. CLAUDE.md policy section documents the requirement for new
public methods.

Reviewer #2 gaps addressed: doctest infrastructure for all Example: blocks
EOF
)"
```

---

## Task 2: `core/mixins/dataset.py` — Priority 1

**Files:**
- Modify: `src/deriva_ml/core/mixins/dataset.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

The current `delete_dataset` docstring is missing an `Example:`. `list_dataset_element_types` and `add_dataset_element_type` lack `Raises:` and `Example:`. The ORM-rebuild block at lines 217–236 needs a "why" comment. `prefetch_dataset` (line ~485) is dead and must be deleted.

- [ ] **Step 1: Add `Example:` to `delete_dataset`**

Find `def delete_dataset` and add after the existing `Raises:` block:

```python
    Example:
        >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
        >>> ml.delete_dataset(ds, recurse=False)  # doctest: +SKIP
```

- [ ] **Step 2: Expand `list_dataset_element_types` docstring**

Replace the current stub with:

```python
    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the table types that can be added as dataset members.

        Returns every table that has an association with the Dataset table,
        restricted to domain-schema tables and the Dataset table itself.
        These are the types accepted by ``add_dataset_members()``.

        Returns:
            Iterable of ``Table`` objects representing valid member types.

        Raises:
            DerivaMLException: If the catalog schema cannot be read.

        Example:
            >>> types = ml.list_dataset_element_types()  # doctest: +SKIP
            >>> print([t.name for t in types])  # doctest: +SKIP
        """
```

- [ ] **Step 3: Expand `add_dataset_element_type` docstring and add inline comment**

Replace the existing docstring to add `Raises:` and `Example:`, and insert the inline comment explaining the ORM rebuild:

```python
    def add_dataset_element_type(self, element: str | Table) -> Table:
        """Make it possible to add objects from ``element`` table to a dataset.

        Creates a new association table linking Dataset to the given table,
        then updates catalog annotations so the new type is included in
        bag-export specs. If the workspace ORM was already built, it is
        rebuilt to pick up the new association table — the ORM is eagerly
        constructed at init time and does not see DDL changes applied after
        that point.

        Args:
            element: Name of the table (str) or Table object to register as
                a valid dataset element type.

        Returns:
            The Table object that was registered.

        Raises:
            DerivaMLException: If ``element`` is not a valid table name.
            DerivaMLTableTypeError: If the table is a system or ML table
                and cannot be a dataset element type.

        Example:
            >>> ml.add_dataset_element_type("Image")  # doctest: +SKIP
        """
```

Then, immediately before the `if getattr(self, "_workspace", None) is not None:` block (currently around line 218), add:

```python
        # Rebuild the workspace ORM so it can resolve the new association table.
        # The workspace ORM is built eagerly at init time from the schema snapshot;
        # DDL applied after that point (like this new association table) is not
        # visible until the ORM is rebuilt from a fresh model fetch.
```

- [ ] **Step 4: Delete `prefetch_dataset`**

Find the method `def prefetch_dataset(self, ...)` (around line 485). Delete the entire method definition including its docstring. It is a one-line `return self.cache_dataset(...)` wrapper with a "Deprecated" docstring and zero callers.

- [ ] **Step 5: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/mixins/dataset.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep core/mixins/dataset.py — 3 docstrings, 1 inline comment, 1 dead-code deletion

Reviewer #2 gaps addressed: delete_dataset Example:, list_dataset_element_types
(Args/Raises/Example), add_dataset_element_type (Raises/Example)
Reviewer #2 inline comment: add_dataset_element_type ORM-rebuild block explains
why workspace.rebuild_schema() is called after new DDL (ORM built eagerly at init)
Reviewer #4 dead code: deleted prefetch_dataset (deprecated shim, zero callers)
EOF
)"
```

---

## Task 3: `core/mixins/annotation.py` — Priority 1

**Files:**
- Modify: `src/deriva_ml/core/mixins/annotation.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

Every `AnnotationMixin` method has stub examples but no `Raises:` and no annotation-dict format guidance. `list_foreign_keys` is dead and must be deleted (it was also going to be renamed, but deletion takes precedence).

- [ ] **Step 1: Add `Raises:` and improve `Example:` for `get_table_annotations`**

```python
    def get_table_annotations(self, table: str | Table) -> dict[str, Any]:
        """Get all Chaise display-related annotations for a table.

        Returns the current values of display, visible-columns,
        visible-foreign-keys, and table-display annotations. Missing
        annotations are represented as ``None`` in the returned dict.

        Args:
            table: Table name (str) or ``Table`` object.

        Returns:
            Dict with keys ``table`` (str), ``schema`` (str),
            ``display`` (dict | None), ``visible_columns`` (dict | None),
            ``visible_foreign_keys`` (dict | None), ``table_display``
            (dict | None).

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> anns = ml.get_table_annotations("Image")  # doctest: +SKIP
            >>> anns["visible_columns"]  # doctest: +SKIP
        """
```

- [ ] **Step 2: Add `Raises:` and annotation-dict format guidance to `get_column_annotations`**

```python
    def get_column_annotations(self, table: str | Table, column_name: str) -> dict[str, Any]:
        """Get all Chaise display-related annotations for a column.

        Returns display and column-display annotations. Missing annotations
        are ``None``.

        Args:
            table: Table name (str) or ``Table`` object.
            column_name: Name of the column.

        Returns:
            Dict with keys ``table`` (str), ``schema`` (str),
            ``column`` (str), ``display`` (dict | None),
            ``column_display`` (dict | None).

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.
            DerivaMLException: If ``column_name`` is not a column of ``table``.

        Example:
            >>> anns = ml.get_column_annotations("Image", "Filename")  # doctest: +SKIP
            >>> anns["display"]  # doctest: +SKIP
        """
```

- [ ] **Step 3: Add `Raises:` + format guidance to `set_display_annotation`, `set_visible_columns`, `set_visible_foreign_keys`, `set_table_display`, `set_column_display`**

For each setter, add or replace the docstring to include:
- `Raises: DerivaMLTableTypeError: If the table is not found.`
- `Example:` block with `# doctest: +SKIP`

Template (adapt per method):

```python
    def set_display_annotation(self, table: str | Table, display: dict[str, Any]) -> None:
        """Set the Chaise display annotation on a table.

        The display annotation controls how the table is labeled in the
        Chaise web UI. The dict shape follows the Chaise display tag
        specification: ``{"name": "Human Readable Name"}``.

        Args:
            table: Table name (str) or ``Table`` object.
            display: Annotation dict. Must be a valid Chaise display
                annotation, e.g. ``{"name": "My Table"}``.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not found in the catalog model.

        Example:
            >>> ml.set_display_annotation("Image", {"name": "Scan Image"})  # doctest: +SKIP
        """
```

Apply the same pattern (with appropriate Args/Example content) to:
- `set_visible_columns(table, columns)` — note that `columns` is a list of column name strings or column filter dicts per the Chaise visible-columns spec
- `set_visible_foreign_keys(table, foreign_keys)` — `foreign_keys` is a list of FK specs per the Chaise visible-foreign-keys spec
- `set_table_display(table, table_display)` — `table_display` is a dict per the Chaise table-display tag
- `set_column_display(table, column_name, column_display)` — `column_display` is a dict per the Chaise column-display tag

- [ ] **Step 4: Add `Raises:` and `Example:` to `add_visible_column`, `remove_visible_column`, `reorder_visible_columns`**

Same pattern — each needs `Raises: DerivaMLTableTypeError` and a `# doctest: +SKIP` example.

- [ ] **Step 5: Add `Raises:` and `Example:` to `add_visible_foreign_key`, `remove_visible_foreign_key`, `reorder_visible_foreign_keys`**

Same pattern.

- [ ] **Step 6: Add `Raises:` and `Example:` to `apply_annotations`**

```python
    def apply_annotations(self) -> None:
        """Apply all staged annotation changes to the catalog.

        Pushes any in-memory annotation changes to the live catalog. Must
        be called after any sequence of ``set_*`` or ``add_*/remove_*``
        annotation calls to make changes visible in Chaise.

        Raises:
            DerivaMLException: If the catalog is read-only or the apply
                call fails.

        Example:
            >>> ml.set_display_annotation("Image", {"name": "Scan"})  # doctest: +SKIP
            >>> ml.apply_annotations()  # doctest: +SKIP
        """
```

- [ ] **Step 7: Delete `list_foreign_keys`**

Find `def list_foreign_keys(self, ...)` (around line 405). Delete the entire method. Zero callers in `src/`, `tests/`, or `docs/`.

- [ ] **Step 8: Remove `list_foreign_keys` from the class-level `Methods:` docstring in `AnnotationMixin`**

Find the line `list_foreign_keys: List all foreign keys related to a table` in the class docstring and delete it.

- [ ] **Step 9: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/mixins/annotation.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep core/mixins/annotation.py — 12 docstrings, 1 dead-code deletion

Reviewer #2 gaps addressed: all AnnotationMixin methods now have Raises:,
annotation-dict format guidance, and Example: blocks
Reviewer #4 dead code: deleted list_foreign_keys (zero callers)
Reviewer #4 renames: list_foreign_keys deletion supersedes rename
EOF
)"
```

---

## Task 4: `feature.py` — Priority 1

**Files:**
- Modify: `src/deriva_ml/feature.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

`Feature.__init__` has no docstring at all. `feature_record_class` lacks `Args:`, `Raises:`, and `Example:`. The `assoc_fkeys` subtraction block (lines 448–463) needs an inline "why" comment. The module-level docstring should document the selector classmethod suite.

- [ ] **Step 1: Expand the module-level docstring**

Replace the current module docstring with:

```python
"""Feature implementation for deriva-ml.

This module provides classes for defining and managing features in deriva-ml.
Features represent measurable properties or characteristics associated with
records in a target table (e.g., a diagnostic label on an Image row).

Exported classes:
    Feature: Encapsulates a feature's schema — target table, vocabulary columns,
        asset columns, and value columns. Obtained via ``DerivaML.create_feature``
        or ``DerivaML.lookup_feature``. Not constructed directly.
    FeatureRecord: Pydantic base class for dynamically generated feature record
        models. Subclasses are created by ``Feature.feature_record_class()``.

Selector classmethod suite (``FeatureRecord`` class methods):
    ``FeatureRecord.select_newest(records)`` — Returns the record with the most
        recent ``RCT`` (Row Creation Time). Useful when multiple annotators have
        labelled the same object.
    ``FeatureRecord.select_by_execution(records, execution_rid)`` — Returns the
        record produced by a specific execution run.

Typical usage:
    >>> feature = ml.lookup_feature("Image", "Diagnosis")  # doctest: +SKIP
    >>> DiagnosisRecord = feature.feature_record_class()  # doctest: +SKIP
    >>> record = DiagnosisRecord(Diagnosis="benign", Confidence=0.97)  # doctest: +SKIP
"""
```

- [ ] **Step 2: Add docstring to `Feature.__init__`**

Insert a docstring immediately after `def __init__(self, atable: FindAssociationResult, model: "DerivaModel") -> None:`:

```python
        """Initialize a Feature from an association table result.

        Classifies the feature table's FK columns into three disjoint sets:
        ``asset_columns`` (FK to an asset table), ``term_columns`` (FK to a
        vocabulary table), and ``value_columns`` (everything else). The
        association FKs linking back to the target table and to the feature
        name vocabulary are excluded before classification.

        Args:
            atable: Result from ``deriva.core.ermrest_model.FindAssociationResult``
                describing the feature association table. Provides the feature
                table, the self-FK back to the target, and the set of other FKs.
            model: ``DerivaModel`` instance used to classify FK targets as
                asset or vocabulary tables.

        Note:
            This constructor is not part of the public API. Obtain ``Feature``
            instances via ``DerivaML.create_feature`` or
            ``DerivaML.lookup_feature``.
        """
```

- [ ] **Step 3: Add inline comment to the `assoc_fkeys` block**

Find the line `assoc_fkeys = {atable.self_fkey} | atable.other_fkeys` (around line 448) and add a comment immediately before it:

```python
        # Exclude the two FKs that are structural parts of the association table
        # itself — the self-FK pointing back to the target table (e.g., Image)
        # and the other-FKs pointing to Feature_Name and Execution — before
        # classifying the remaining FKs as asset, term, or value columns. Without
        # this subtraction, those structural FKs would be misclassified as feature
        # columns and create spurious fields in the generated FeatureRecord class.
        assoc_fkeys = {atable.self_fkey} | atable.other_fkeys
```

- [ ] **Step 4: Expand `feature_record_class` docstring**

Replace the current docstring:

```python
    def feature_record_class(self) -> type[FeatureRecord]:
        """Create a dynamically generated Pydantic model class for this feature.

        Builds a ``FeatureRecord`` subclass with fields derived from the feature
        table's columns. Column types are mapped as follows:

        - Term columns (FK to vocabulary): ``str`` (vocabulary term name)
        - Asset columns (FK to asset table): ``str | Path`` (file path)
        - Value columns (direct data): typed per the database column type
          (``int``, ``float``, or ``str``)

        All fields are Optional with a default of ``None`` to allow partial
        construction when building records for insertion.

        Returns:
            A subclass of ``FeatureRecord`` whose fields match this feature's
            schema. The class's ``feature`` ClassVar is set to ``self``.

        Raises:
            DerivaMLException: If the feature table schema cannot be read.

        Example:
            >>> feature = ml.lookup_feature("Image", "Diagnosis")  # doctest: +SKIP
            >>> DiagnosisRecord = feature.feature_record_class()  # doctest: +SKIP
            >>> rec = DiagnosisRecord(Diagnosis="benign")  # doctest: +SKIP
        """
```

- [ ] **Step 5: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/feature.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep feature.py — 3 docstrings, 1 inline comment, module docstring expanded

Reviewer #2 gaps addressed: Feature.__init__ (no docstring at all),
feature_record_class (Args/Raises/Example), module docstring (selector suite)
Reviewer #2 inline comment: assoc_fkeys subtraction explains why structural
association FKs are excluded before role classification
EOF
)"
```

---

## Task 5: `dataset/dataset_bag.py` — Priority 1

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

`list_dataset_members` has a thin docstring with the `version` kwarg unexplained. The `_dataset_table_view` union block (lines 276–315) needs a "why" comment about SQLAlchemy UNION semantics.

- [ ] **Step 1: Expand `list_dataset_members` docstring**

Find `def list_dataset_members` and replace its docstring:

```python
    def list_dataset_members(
        self,
        recurse: bool = False,
        version: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return all members of this dataset bag grouped by table name.

        Queries the local SQLite replica of the downloaded bag. Each key
        in the returned dict is a table name (e.g. ``"Image"``); each value
        is a list of row dicts with the full set of columns for that table.

        Args:
            recurse: If ``True``, recursively include members from nested
                child datasets. Default is ``False``.
            version: Dataset version string (e.g. ``"1.2.0"``) to query.
                When ``None`` (default), uses the latest materialized version
                in the bag. This parameter exists for API symmetry with the
                live-catalog ``Dataset.list_dataset_members``; bag contents
                are immutable so changing ``version`` only filters which
                version's membership snapshot is read.

        Returns:
            Dict mapping table name to list of row dicts. Empty dict if
            no members are present.

        Raises:
            DerivaMLException: If the bag SQLite database cannot be read
                or the requested version is not present in the bag.

        Example:
            >>> bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> members = bag.list_dataset_members(recurse=True)  # doctest: +SKIP
            >>> images = members.get("Image", [])  # doctest: +SKIP
        """
```

- [ ] **Step 2: Add inline comment to `_dataset_table_view` union block**

Find the `union(*)` call inside `_dataset_table_view` (around lines 276–315) and add a comment immediately before the union call:

```python
        # Use UNION (not UNION ALL) to deduplicate rows that are reachable via
        # multiple FK paths. SQLAlchemy's union() is DISTINCT by default, which
        # is exactly what we want here: a dataset member table (e.g., Image) may
        # appear in the bag via two separate FK paths (e.g., directly in the
        # dataset AND via a nested child dataset), and without DISTINCT we would
        # count or return the same row twice. See §6 inline comment gap #4.
```

- [ ] **Step 3: Spot-check all other public methods in the file for missing sections**

Walk through `DatasetBag`'s public methods and confirm each has a one-line summary, `Args:`, `Returns:` (where applicable), `Raises:`, and `Example: # doctest: +SKIP`. Patch any gaps with the same contract template.

- [ ] **Step 4: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/dataset/dataset_bag.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep dataset/dataset_bag.py — 2+ docstrings, 1 inline comment

Reviewer #2 gaps addressed: list_dataset_members (version kwarg explained,
Raises/Example added)
Reviewer #2 inline comment: _dataset_table_view union(*) block explains why
DISTINCT union is load-bearing (SQLAlchemy UNION is DISTINCT; de-duplicates
rows reachable via multiple FK paths)
EOF
)"
```

---

## Task 6: `execution/execution.py` — Priority 2

**Files:**
- Modify: `src/deriva_ml/execution/execution.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

`Execution.create_dataset` (line ~2006) lacks `Raises:` and `Example:`. `Execution.upload_execution_outputs` (line ~1397) has an unexplained `Returns:` dict shape and no `Raises:`. `Execution.__init__` (lines 322–361) needs a "why" comment on the dry-run guard.

- [ ] **Step 1: Expand `create_dataset` docstring**

Find `def create_dataset` in `Execution` and replace its docstring:

```python
    def create_dataset(
        self,
        dataset_types: list[str] | None = None,
        description: str = "",
        members: list[RID] | None = None,
    ) -> "Dataset":
        """Create a new dataset tracked to this execution.

        Creates a ``Dataset`` catalog record linked to this execution as its
        provenance. The dataset is immediately usable for adding members and
        incrementing versions.

        Args:
            dataset_types: List of dataset type vocabulary term names to apply.
                Must be pre-registered via ``add_dataset_type``. Pass ``None``
                or an empty list to create an untyped dataset.
            description: Human-readable description of the dataset. Stored in
                the catalog ``Dataset.Description`` column.
            members: Optional list of RIDs to immediately add as dataset members
                after creation. Equivalent to calling ``dataset.add_dataset_members``
                after this call.

        Returns:
            A ``Dataset`` instance bound to the newly created catalog record.

        Raises:
            DerivaMLInvalidTerm: If any name in ``dataset_types`` is not a
                registered ``Dataset_Type`` vocabulary term.
            DerivaMLExecutionError: If the execution context is no longer active.

        Example:
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     ds = exe.create_dataset(  # doctest: +SKIP
            ...         dataset_types=["training"],  # doctest: +SKIP
            ...         description="Training images v1",  # doctest: +SKIP
            ...     )  # doctest: +SKIP
        """
```

- [ ] **Step 2: Expand `upload_execution_outputs` docstring**

Find `def upload_execution_outputs` and replace its docstring:

```python
    def upload_execution_outputs(
        self,
        timeout: int = 30,
        chunk_size: int = 10 * 1024 * 1024,
    ) -> dict[str, list[str]]:
        """Upload all registered output assets to Hatrac and record provenance.

        Reads the asset manifest, uploads each file to the catalog's Hatrac
        object store, and inserts ``{Asset}_Execution`` association records
        linking each uploaded asset to this execution with the ``Output`` role.

        Call this method **after** exiting the execution context manager, not
        inside it. The context manager sets execution status to ``Completed``
        on exit; uploading after that preserves the correct status ordering.

        Args:
            timeout: HTTP session timeout in seconds for each upload request.
                Default is 30.
            chunk_size: Hatrac chunk upload size in bytes. Default is 10 MiB.
                Increase for large files on high-bandwidth connections.

        Returns:
            Dict mapping asset table name to list of uploaded RIDs, e.g.
            ``{"Image": ["1-ABC", "1-DEF"], "Model": ["2-GHI"]}``.

        Raises:
            DerivaMLUploadError: If any file upload fails. Partial uploads are
                recorded in the manifest so the upload can be resumed.
            DerivaMLReadOnlyError: If the catalog connection is read-only.

        Example:
            >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
            ...     path = exe.asset_file_path("Model", "model.pt")  # doctest: +SKIP
            >>> uploaded = exe.upload_execution_outputs()  # doctest: +SKIP
        """
```

- [ ] **Step 3: Add inline comment to the dry-run guard in `__init__`**

Find the block around lines 322–361 in `Execution.__init__` containing `not self._dry_run and reload is None`. Add a comment immediately before that condition:

```python
        # Guard SQLite registry insertion: skip when (a) this is a dry-run
        # (we never want to persist dry-run state) or (b) we are resuming an
        # existing execution (reload is not None), in which case the registry
        # entry was written by the original run and should not be overwritten.
        # Writing twice would corrupt the start-time and initial-status fields.
```

- [ ] **Step 4: Spot-check all other public methods in `Execution` for missing sections**

Walk through `Execution`'s public methods and confirm each has the required docstring sections. Patch any gaps.

- [ ] **Step 5: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/execution/execution.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep execution/execution.py — 3 docstrings, 1 inline comment

Reviewer #2 gaps addressed: Execution.create_dataset (Raises/Example),
upload_execution_outputs (Returns dict shape, Raises)
Reviewer #2 inline comment: __init__ dry-run guard explains why
`not self._dry_run and reload is None` gates SQLite registry insertion
EOF
)"
```

---

## Task 7: `dataset/dataset.py` — Priority 2

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

Four methods lack `Raises:` and/or `Example:`. The `_insert_dataset_versions` two-step INSERT+GET pattern (lines 1580–1610) needs an inline "why" comment.

- [ ] **Step 1: Expand `list_dataset_parents` docstring**

Find `def list_dataset_parents` (around line 1404) and replace its docstring:

```python
    def list_dataset_parents(self) -> list["Dataset"]:
        """Return all datasets that directly contain this dataset as a member.

        Queries the catalog for ``Dataset_Dataset`` association records where
        this dataset appears as the child. Returns an empty list if this dataset
        is not nested inside any other dataset.

        Returns:
            List of ``Dataset`` instances that are direct parents. May be empty.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
            >>> parents = ds.list_dataset_parents()  # doctest: +SKIP
            >>> print([p.dataset_rid for p in parents])  # doctest: +SKIP
        """
```

- [ ] **Step 2: Expand `list_dataset_children` docstring**

Find `def list_dataset_children` (around line 1453) and replace its docstring:

```python
    def list_dataset_children(self) -> list["Dataset"]:
        """Return all datasets directly nested inside this dataset.

        Queries the catalog for ``Dataset_Dataset`` association records where
        this dataset appears as the parent. Returns an empty list if this
        dataset has no nested children.

        Returns:
            List of ``Dataset`` instances that are direct children. May be empty.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
            >>> children = ds.list_dataset_children()  # doctest: +SKIP
        """
```

- [ ] **Step 3: Add `Raises:` to `add_dataset_members`**

Find `def add_dataset_members` (around line 1194). Add after the existing `Args:` block:

```python
        Raises:
            DerivaMLNotFoundError: If any RID in ``members`` does not resolve
                to a catalog entity of the expected table type.
            DerivaMLTableTypeError: If a member RID belongs to a table type
                that is not registered as a dataset element type.
            DerivaMLValidationError: If ``members`` contains duplicates that
                are already present in the dataset.
```

Also add if missing:

```python
        Example:
            >>> ds.add_dataset_members(["1-ABC", "1-DEF"])  # doctest: +SKIP
```

- [ ] **Step 4: Expand `download_dataset_bag` docstring**

Find `def download_dataset_bag` (around line 1612) and replace its docstring to document the `DatasetBag` return shape and add `Raises:`:

```python
    def download_dataset_bag(self, ...) -> "DatasetBag":
        """Download this dataset to the local filesystem as a BDBag.

        Exports the dataset's tables and asset references into a BDBag
        directory structure. If the catalog has ``s3_bucket`` configured
        and ``use_minid=True``, the bag is also uploaded to S3 and
        registered with the MINID service.

        Returns:
            A ``DatasetBag`` instance wrapping the downloaded bag. Key attributes:
            ``bag.path`` (``Path``) — local directory containing the bag;
            ``bag.dataset_rid`` (str) — RID of the dataset;
            ``bag.version`` (str) — version string at time of download.

        Raises:
            DerivaMLDatasetNotFound: If this dataset's RID no longer exists
                in the catalog (e.g., was deleted after this object was created).
            DerivaMLException: If the bag export or materialization fails.

        Example:
            >>> spec = DatasetSpecConfig(rid="1-ABC", version="1.0.0")  # doctest: +SKIP
            >>> bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> members = bag.list_dataset_members()  # doctest: +SKIP
        """
```

- [ ] **Step 5: Add inline comment to `_insert_dataset_versions` two-step pattern**

Find the two-step INSERT + GET block around lines 1580–1610. Add a comment immediately before the GET:

```python
        # ERMrest does not return system-generated columns (including snaptime)
        # in the INSERT response — it only echoes back the columns you sent.
        # We need the snaptime to record the version's catalog snapshot for
        # point-in-time reads. Perform a separate GET immediately after the
        # INSERT to retrieve the server-assigned snaptime for this row.
```

- [ ] **Step 6: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/dataset/dataset.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep dataset/dataset.py — 4 docstrings, 1 inline comment

Reviewer #2 gaps addressed: list_dataset_parents (Raises/Example),
list_dataset_children (Raises/Example), add_dataset_members (Raises),
download_dataset_bag (Returns DatasetBag shape, Raises)
Reviewer #2 inline comment: _insert_dataset_versions explains two-step
INSERT + GET snaptime pattern (ERMrest does not return snaptime on INSERT)
EOF
)"
```

---

## Task 8: `core/mixins/asset.py` — Priority 2

**Files:**
- Modify: `src/deriva_ml/core/mixins/asset.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

`create_asset` is missing `Raises:` and `Example:`. Module-level docstring is minimal — expand it.

- [ ] **Step 1: Expand the module-level docstring**

Replace the current module docstring:

```python
"""Asset management mixin for DerivaML.

This module provides the ``AssetMixin`` class, which manages asset tables
in a Deriva catalog. An "asset" is a file-backed table whose rows track
uploaded files (images, documents, model weights, etc.) together with
provenance metadata.

The mixin provides:
    - ``create_asset``: Define a new asset table schema in the catalog.
    - ``list_assets``: Query the contents of an existing asset table.

Asset tables follow a fixed schema convention (Filename, URL, Length, MD5,
Description plus system columns) augmented with user-defined metadata columns.
Access controlled by ``AssetMixin`` is layered; the ``AssetMixin`` handles schema
management while upload/download is handled by ``ExecutionMixin`` and the
upload engine.
"""
```

- [ ] **Step 2: Expand `create_asset` docstring**

Replace the existing docstring:

```python
    def create_asset(
        self,
        asset_name: str,
        column_defs: Iterable[ColumnDefinition] | None = None,
        fkey_defs: Iterable[ColumnDefinition] | None = None,
        referenced_tables: Iterable[Table] | None = None,
        comment: str = "",
        schema: str | None = None,
        update_navbar: bool = True,
    ) -> Table:
        """Create a new asset table in the catalog.

        Defines a Chaise-compatible asset table (Filename, URL, Length, MD5,
        Description, plus system columns) with optional additional metadata
        columns and foreign-key references. Registers the asset type in the
        ``Asset_Type`` vocabulary and optionally updates the Chaise navigation
        bar.

        Args:
            asset_name: Name for the new asset table, e.g. ``"Image"`` or
                ``"ModelWeights"``.
            column_defs: Extra metadata columns beyond the standard asset
                columns. Each is a ``ColumnDefinition`` specifying name, type,
                nullability, and comment.
            fkey_defs: Foreign-key definitions from the asset table to other
                tables (e.g., linking images to a subject table).
            referenced_tables: Tables that the new asset table should reference
                via FKs. Convenience alternative to ``fkey_defs`` when only
                a reference to an existing table is needed.
            comment: Human-readable description of the asset table stored
                as the table comment in the catalog.
            schema: Schema in which to create the table. Defaults to
                ``self.default_schema``.
            update_navbar: If ``True`` (default), call
                ``apply_catalog_annotations()`` immediately to add the new
                table to the Chaise navigation bar. Set ``False`` when
                creating multiple asset tables in a batch; call
                ``apply_catalog_annotations()`` once at the end.

        Returns:
            The newly created ``Table`` object.

        Raises:
            DerivaMLException: If a table named ``asset_name`` already exists
                in the target schema.
            DerivaMLSchemaError: If ``schema`` is not a valid schema in this
                catalog.

        Example:
            >>> from deriva.core.typed import Column, builtin_types  # doctest: +SKIP
            >>> ml.create_asset(  # doctest: +SKIP
            ...     "ScanImage",  # doctest: +SKIP
            ...     comment="MRI scan images",  # doctest: +SKIP
            ... )  # doctest: +SKIP
        """
```

- [ ] **Step 3: Spot-check `list_assets` for missing sections**

Confirm `list_assets` has `Args:`, `Returns:`, `Raises:`, `Example:`. Add any missing sections.

- [ ] **Step 4: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/mixins/asset.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep core/mixins/asset.py — 2 docstrings, module docstring expanded

Reviewer #2 gaps addressed: create_asset (Raises/Example),
list_assets (spot-checked and patched if needed)
Module docstring: expanded to document asset table convention and mixin layering
EOF
)"
```

---

## Task 9: Rename Group A — `core/mixins/path_builder.py` + `core/constants.py`

**Files:**
- Modify: `src/deriva_ml/core/mixins/path_builder.py`
- Modify: `src/deriva_ml/core/constants.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

Four renames: `domain_path` → `_domain_path`, `table_path` → `_table_path`, `is_system_schema` → `_is_system_schema`, `get_domain_schemas` → `_get_domain_schemas`.

- [ ] **Step 1: Rename `domain_path` → `_domain_path` in `path_builder.py`**

Find `def domain_path(` and rename to `def _domain_path(`. Update the class-level `Methods:` docstring entry. Search the entire `src/` tree for any other callers and update them:

```bash
grep -rn "\.domain_path(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site found. Also update the class-level `Methods:` docstring to reference `_domain_path`.

- [ ] **Step 2: Rename `table_path` → `_table_path` in `path_builder.py`**

Find `def table_path(` and rename to `def _table_path(`. Search for callers:

```bash
grep -rn "\.table_path(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site.

- [ ] **Step 3: Add spot-check docstrings if missing in `path_builder.py`**

Verify `pathBuilder`, `get_table_as_dataframe`, `get_table_as_dict` have full docstrings. Add `Raises:` and `Example: # doctest: +SKIP` where missing.

- [ ] **Step 4: Rename `is_system_schema` → `_is_system_schema` in `core/constants.py`**

Find `def is_system_schema(` and rename. Search for all callers:

```bash
grep -rn "is_system_schema(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site. Also update the module-level docstring list that names `is_system_schema`.

- [ ] **Step 5: Rename `get_domain_schemas` → `_get_domain_schemas` in `core/constants.py`**

Find `def get_domain_schemas(` and rename. Search for callers:

```bash
grep -rn "get_domain_schemas(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site. Update the module docstring.

- [ ] **Step 6: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/mixins/path_builder.py src/deriva_ml/core/constants.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep path_builder.py + constants.py — spot-check docstrings, 4 renames

Reviewer #4 renames: domain_path→_domain_path, table_path→_table_path
  (low-level ERMrest/filesystem helpers; not user-facing)
Reviewer #4 renames: is_system_schema→_is_system_schema,
  get_domain_schemas→_get_domain_schemas (schema introspection predicates;
  users won't call these directly)
EOF
)"
```

---

## Task 10: Rename Group B — `core/logging_config.py` + `core/schema_diff.py`

**Files:**
- Modify: `src/deriva_ml/core/logging_config.py`
- Modify: `src/deriva_ml/core/schema_diff.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

Two renames: `apply_logger_overrides` → `_apply_logger_overrides`, `compute_diff` → `_compute_diff`. Module docstrings need expansion per spec §5.

- [ ] **Step 1: Expand `core/logging_config.py` module docstring**

The current module docstring is already detailed (mentions `get_logger`, `configure_logging`, `LoggerMixin`, `is_hydra_initialized`). Verify it also documents `apply_logger_overrides` (now `_apply_logger_overrides`) as an internal helper. If the current docstring lists it as part of the public API, remove it from the public surface description.

- [ ] **Step 2: Rename `apply_logger_overrides` → `_apply_logger_overrides`**

Find `def apply_logger_overrides(` and rename. Search for callers:

```bash
grep -rn "apply_logger_overrides(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site (expected: one call in `DerivaML.__init__`).

- [ ] **Step 3: Spot-check remaining public methods in `logging_config.py`**

Verify `get_logger`, `configure_logging`, `LoggerMixin` class, and `is_hydra_initialized` have full docstrings. Add missing `Raises:` / `Example:` as needed.

- [ ] **Step 4: Rename `compute_diff` → `_compute_diff` in `core/schema_diff.py`**

Find `def compute_diff(` and rename. Search for callers:

```bash
grep -rn "compute_diff(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site (expected: in `core/base.py` pin/diff logic).

- [ ] **Step 5: Spot-check `SchemaDiff` and helper dataclasses in `schema_diff.py`**

Verify `SchemaDiff`, `AddedTable`, `RemovedTable`, `AddedColumn`, `RemovedColumn`, `AddedFK`, `RemovedFK` all have at least a one-line class docstring. Add any missing docstrings.

- [ ] **Step 6: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/logging_config.py src/deriva_ml/core/schema_diff.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep logging_config.py + schema_diff.py — spot-check, 2 renames

Reviewer #4 renames: apply_logger_overrides→_apply_logger_overrides
  (called once in DerivaML.__init__; not user-facing)
Reviewer #4 renames: compute_diff→_compute_diff (only used inside base.py
  pin/diff logic; not user-facing)
EOF
)"
```

---

## Task 11: Rename Group C — `rid_resolution.py` + `asset_record.py` + `workflow.py`

**Files:**
- Modify: `src/deriva_ml/core/mixins/rid_resolution.py`
- Modify: `src/deriva_ml/asset/asset_record.py`
- Modify: `src/deriva_ml/core/mixins/workflow.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

Three renames: `retrieve_rid` → `_retrieve_rid`, `asset_record_class` → `_asset_record_class` (module-level function), `add_workflow` → `_add_workflow`.

- [ ] **Step 1: Rename `retrieve_rid` → `_retrieve_rid` in `rid_resolution.py`**

Find `def retrieve_rid(` and rename. Search for callers:

```bash
grep -rn "retrieve_rid(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site. Note: the user-facing API is `resolve_rid()` — that stays public. Add a spot-check docstring to `resolve_rid()` if it is missing `Raises:` or `Example:`.

- [ ] **Step 2: Rename module-level `asset_record_class` → `_asset_record_class` in `asset_record.py`**

Find `def asset_record_class(` (the module-level function, not a method). Rename it. Search for callers:

```bash
grep -rn "asset_record_class(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
```

Update every call site. The mixin's method that wraps it will now call `_asset_record_class(...)` internally.

- [ ] **Step 3: Spot-check `AssetRecord` and `_asset_record_class` docstrings**

`AssetRecord` has a docstring. Verify it includes `Example:`. `_asset_record_class` is now private — add or keep a one-line summary docstring (it's fine for private helpers to have a minimal docstring).

- [ ] **Step 4: Rename `add_workflow` → `_add_workflow` in `workflow.py`**

Find `def add_workflow(` and rename. Search for callers:

```bash
grep -rn "add_workflow(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
grep -rn "add_workflow(" /Users/carl/GitHub/deriva-ml/tests/ --include="*.py"
```

Update every call site. Note: `create_workflow()` remains public — no changes to it.

- [ ] **Step 5: Spot-check `WorkflowMixin` public methods**

Verify `find_workflows`, `lookup_workflow`, `find_workflow_by_url`, `create_workflow`, `list_workflow_executions` all have complete docstrings. Add missing `Raises:` / `Example:` as needed.

- [ ] **Step 6: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/mixins/rid_resolution.py \
        src/deriva_ml/asset/asset_record.py \
        src/deriva_ml/core/mixins/workflow.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep rid_resolution.py + asset_record.py + workflow.py — spot-check, 3 renames

Reviewer #4 renames: retrieve_rid→_retrieve_rid (user-facing is resolve_rid())
Reviewer #4 renames: asset_record_class→_asset_record_class (internal factory;
  users access via mixin method)
Reviewer #4 renames: add_workflow→_add_workflow (create_workflow() is the
  user-facing factory)
EOF
)"
```

---

## Task 12: `core/base.py` — Renames + Dead Code + `working_data` Docstring

**Files:**
- Modify: `src/deriva_ml/core/base.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

One rename: `cache_features` → `_cache_features`. Three dead-code deletions: `add_page`, `user_list`, `globus_login`. One docstring addition: `working_data` property needs a proper deprecation-documenting docstring. Spot-check remaining public methods.

- [ ] **Step 1: Rename `cache_features` → `_cache_features`**

Find `def cache_features(` and rename. Search for callers:

```bash
grep -rn "cache_features(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
grep -rn "cache_features(" /Users/carl/GitHub/deriva-ml/tests/ --include="*.py"
```

Update every call site.

- [ ] **Step 2: Delete `add_page`**

Find `def add_page(` (around line 1364) and delete the entire method definition including its docstring. Zero callers.

- [ ] **Step 3: Delete `user_list`**

Find `def user_list(` (around line 1097) and delete the entire method. Zero callers.

- [ ] **Step 4: Delete `globus_login`**

Find `def globus_login(` (around line 955) and delete the entire method. Zero callers.

- [ ] **Step 5: Add proper docstring to `working_data` property**

Find `def working_data(` (around line 856). Replace its docstring:

```python
    @property
    def working_data(self) -> Path:
        """Return the working data directory path.

        .. deprecated::
            ``working_data`` is deprecated and will be removed in the next
            major version. Use ``working_dir`` instead.

            ``working_dir`` is the canonical attribute; it is set during
            execution initialization and contains all output assets, metadata,
            and intermediate files for the current execution.

        Returns:
            Path to the working data directory (same as ``working_dir``).

        Raises:
            DeprecationWarning: Always emitted at access time.

        Example:
            >>> exe.working_dir  # use this instead  # doctest: +SKIP
        """
```

- [ ] **Step 6: Spot-check remaining public methods in `core/base.py`**

Walk the public surface of `DerivaML` (excluding mixin methods). Confirm each has a complete docstring. Focus on any methods not touched by other tasks. Patch gaps.

- [ ] **Step 7: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/base.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep core/base.py — 1 rename, 3 dead-code deletions, working_data docstring

Reviewer #4 renames: cache_features→_cache_features (legacy workspace-cache
  shortcut; no callers in tests)
Reviewer #4 dead code: deleted add_page (zero callers), user_list (zero callers),
  globus_login (zero callers)
Reviewer #4 dead code: working_data KEPT — deprecation stub still serves its
  purpose as a deprecation signal; given a proper docstring documenting
  DeprecationWarning and replacement (working_dir). Removal deferred to
  next major-version bump per spec decision #7.
EOF
)"
```

---

## Task 13: `tools/validate_schema_doc.py` — Rename Internal Helpers

**Files:**
- Modify: `src/deriva_ml/tools/validate_schema_doc.py`
- Modify: `tests/tools/` (update imports to underscored names)
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/ -q --timeout=60`

Three functions used only from `main()` and from `tests/tools/` get `_prefix` renames: `load_from_doc` → `_load_from_doc`, `load_from_code` → `_load_from_code`, `diff_schemas` → `_diff_schemas`.

- [ ] **Step 1: Rename `load_from_doc` → `_load_from_doc`**

Find `def load_from_doc(` and rename. Then find all call sites:

```bash
grep -rn "load_from_doc(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
grep -rn "load_from_doc(" /Users/carl/GitHub/deriva-ml/tests/ --include="*.py"
```

Update all call sites in both `validate_schema_doc.py`'s `main()` and `tests/tools/`.

- [ ] **Step 2: Rename `load_from_code` → `_load_from_code`**

Find `def load_from_code(` and rename. Update all call sites (same grep pattern as Step 1).

- [ ] **Step 3: Rename `diff_schemas` → `_diff_schemas`**

Find `def diff_schemas(` and rename. Update all call sites.

- [ ] **Step 4: Run tools tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/tools/ -q --timeout=60`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/tools/validate_schema_doc.py
git add tests/tools/
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep tools/validate_schema_doc.py — 3 renames

Reviewer #4 dead code (item 7): load_from_doc→_load_from_doc,
  load_from_code→_load_from_code, diff_schemas→_diff_schemas
  (CLI-tool internals; only called from main() and tests/tools/)
  Tests in tests/tools/ updated to use underscored names.
EOF
)"
```

---

## Task 14: `core/mixins/execution.py` — `start_upload` Decision + Docstring Sweep

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py`
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

Apply the `start_upload` decision from Task 0. Spot-check all public methods in `ExecutionMixin`.

- [ ] **Step 1: Apply `start_upload` decision**

Based on Task 0:
- If `start_upload` had **no external callers**: rename to `_start_upload`. Search for internal callers and update them:
  ```bash
  grep -rn "start_upload(" /Users/carl/GitHub/deriva-ml/src/ --include="*.py"
  ```
- If `start_upload` had **external callers** (found in template repo): keep it public, add a complete docstring following the contract shape.

Document the decision in the commit message.

- [ ] **Step 2: Spot-check all public methods in `ExecutionMixin`**

Walk all public methods: `create_execution`, `resume_execution`, `get_execution`, `update_execution_status`, `list_executions`. Confirm each has complete docstrings. Add any missing `Raises:` / `Example: # doctest: +SKIP` blocks.

- [ ] **Step 3: Run fast tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add src/deriva_ml/core/mixins/execution.py
git commit -m "$(cat <<'EOF'
docs(docstrings): sweep core/mixins/execution.py — spot-check, start_upload decision

Reviewer #4 renames: start_upload → [_start_upload | kept public] — see decision
  from Task 0 grep of deriva-ml-model-template.
  [OUTCOME: <replace with actual grep result>]
EOF
)"
```

---

## Task 15: Tier 4 — Module-Level Docstring Sweep (Remaining Modules)

**Files:** All `src/deriva_ml/` modules not already swept in Tasks 2–14. Key targets below.
**Test:** `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

These modules have no Reviewer #2 or #4 findings. Each gets: (1) add/expand module docstring if missing or minimal, (2) verify all public methods have at minimum a one-line summary, (3) add `Example: # doctest: +SKIP` where missing.

Group them into three sub-commits for reviewability:

**Sub-commit A — catalog + dataset support modules:**

- `src/deriva_ml/catalog/localize.py` — expand module docstring to explain three-stage clone flow and asset-copy semantics (§5 of spec)
- `src/deriva_ml/catalog/clone.py` — spot-check module docstring and public functions
- `src/deriva_ml/dataset/catalog_graph.py` — `CatalogGraph` stays public; add/expand class and method docstrings
- `src/deriva_ml/dataset/split.py` — spot-check
- `src/deriva_ml/dataset/bag_cache.py`, `dataset/bag_feature_cache.py` — spot-check

- [ ] **Step 1: Expand `catalog/localize.py` module docstring**

Replace the one-liner:

```python
"""Localize remote Hatrac assets to a local catalog server.

Copies assets referenced in a cloned catalog's asset tables from the source
Hatrac store to the local Hatrac instance. Intended for use after
``create_ml_workspace`` is called with ``asset_mode=REFERENCES``.

Three-stage flow:
1. Enumerate all rows in asset tables (tables with URL, Filename, MD5 columns).
2. For each row, download the file from the source Hatrac URL to a temp file,
   then upload it to the local Hatrac namespace.
3. Update the catalog row's URL to point to the new local Hatrac path.

The ``LocalizeResult`` dataclass summarizes counts of processed/skipped/failed
assets and provides the old-to-new URL mapping for auditing.
"""
```

- [ ] **Step 2: Spot-check `catalog/clone.py` + `dataset/catalog_graph.py`**

For each, confirm:
- Module docstring present and meaningful.
- `CatalogGraph`: class docstring covers FK-traversal purpose, `generate_dataset_download_spec`, `estimate_bag_size`, and the RID-union semantics.
- All public methods/functions have at minimum a one-line summary.

Patch as needed.

- [ ] **Step 3: Sub-commit A**

```bash
git add src/deriva_ml/catalog/ src/deriva_ml/dataset/catalog_graph.py \
        src/deriva_ml/dataset/split.py src/deriva_ml/dataset/bag_cache.py \
        src/deriva_ml/dataset/bag_feature_cache.py
git commit -m "docs(docstrings): Tier 4 sweep — catalog + dataset support modules"
```

**Sub-commit B — execution support + model + schema:**

- `src/deriva_ml/execution/execution_configuration.py`
- `src/deriva_ml/execution/workflow.py`
- `src/deriva_ml/execution/runner.py`, `execution/state_machine.py`, `execution/state_store.py`
- `src/deriva_ml/model/catalog.py`, `model/annotations.py`, `model/schema_builder.py`
- `src/deriva_ml/schema/create_schema.py`, `schema/check_schema.py`, `schema/annotations.py`

- [ ] **Step 4: Spot-check the execution support modules**

For each listed module: add module docstring if missing, verify public class/method docstrings. Patch gaps.

- [ ] **Step 5: Sub-commit B**

```bash
git add src/deriva_ml/execution/execution_configuration.py \
        src/deriva_ml/execution/workflow.py \
        src/deriva_ml/execution/runner.py \
        src/deriva_ml/execution/state_machine.py \
        src/deriva_ml/execution/state_store.py \
        src/deriva_ml/model/ \
        src/deriva_ml/schema/
git commit -m "docs(docstrings): Tier 4 sweep — execution support + model + schema modules"
```

**Sub-commit C — local_db + interfaces + exceptions + remaining:**

- `src/deriva_ml/local_db/workspace.py`, `local_db/denormalize.py`, `local_db/manifest_store.py`
- `src/deriva_ml/interfaces.py` — protocol docstrings
- `src/deriva_ml/core/exceptions.py` — exception class docstrings
- `src/deriva_ml/core/mixins/vocabulary.py`, `core/mixins/feature.py`, `core/mixins/file.py`
- `src/deriva_ml/core/ermrest.py`, `core/config.py`, `core/definitions.py`, `core/enums.py`
- `src/deriva_ml/core/validation.py`, `core/schema_cache.py`, `core/filespec.py`
- `src/deriva_ml/experiment/experiment.py`
- `src/deriva_ml/asset/asset.py`, `asset/manifest.py`, `asset/aux_classes.py`

- [ ] **Step 6: Spot-check the local_db + interfaces + remaining modules**

For each: add/expand module docstring if missing, verify all public symbols have at minimum a one-line summary. Patch gaps.

- [ ] **Step 7: Sub-commit C**

```bash
git add src/deriva_ml/local_db/ \
        src/deriva_ml/interfaces.py \
        src/deriva_ml/core/exceptions.py \
        src/deriva_ml/core/mixins/vocabulary.py \
        src/deriva_ml/core/mixins/feature.py \
        src/deriva_ml/core/mixins/file.py \
        src/deriva_ml/core/ermrest.py \
        src/deriva_ml/core/config.py \
        src/deriva_ml/core/definitions.py \
        src/deriva_ml/core/enums.py \
        src/deriva_ml/core/validation.py \
        src/deriva_ml/core/schema_cache.py \
        src/deriva_ml/core/filespec.py \
        src/deriva_ml/experiment/ \
        src/deriva_ml/asset/
git commit -m "docs(docstrings): Tier 4 sweep — local_db, interfaces, exceptions, remaining modules"
```

---

## Task 16: Test File Renames — Update All Renamed Symbol References

**Files:**
- Modify: any file under `tests/` that references the 12 renamed symbols
- Test: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q`

This task runs after all module-level renames are committed (Tasks 9–14). Find and update every test that uses the old public names.

- [ ] **Step 1: Find all test references to renamed symbols**

```bash
grep -rn "domain_path\|table_path\|is_system_schema\|get_domain_schemas\
\|apply_logger_overrides\|compute_diff\|retrieve_rid\|asset_record_class\
\|cache_features\|add_workflow\|start_upload\|prefetch_dataset\
\|list_foreign_keys" \
  /Users/carl/GitHub/deriva-ml/tests/ --include="*.py"
```

For each hit: update to the underscored name. Note: `prefetch_dataset` and `list_foreign_keys` were deleted (not renamed) — any test asserting those methods exist should be deleted.

- [ ] **Step 2: Run the full fast test suite**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml
git add tests/
git commit -m "$(cat <<'EOF'
test: update test references for renamed private symbols

All 12 renames from the docstring sweep (Tasks 9-14) reflected in tests.
Deleted test assertions for prefetch_dataset and list_foreign_keys (those
methods were deleted, not renamed).
EOF
)"
```

---

## Task 17: Final Verification

**Files:** none modified — verification only.

- [ ] **Step 1: Run fast unit tests with doctest collection**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q --timeout=60`
Expected: PASS (including all collected doctests that don't have `+SKIP`)

- [ ] **Step 2: Verify mkdocs build**

Run: `uv run mkdocs build --strict 2>&1 | tail -20`
Expected: no warnings or errors. If any docstring renders with broken markup (e.g., unbalanced backticks, malformed admonitions), fix inline and re-run.

- [ ] **Step 3: Run ruff on the swept modules**

```bash
uv run ruff check src/deriva_ml/ && uv run ruff format --check src/deriva_ml/
```

Expected: no errors. Fix any issues and amend the relevant module's last commit or add a fixup commit.

- [ ] **Step 4: Spot-check that no old public names leak into `__all__` or the public API surface**

```bash
grep -rn "domain_path\|table_path\|is_system_schema\|get_domain_schemas\
\|apply_logger_overrides\|compute_diff\|retrieve_rid\|asset_record_class\
\|cache_features\|add_workflow\|prefetch_dataset\|list_foreign_keys" \
  /Users/carl/GitHub/deriva-ml/src/ --include="*.py" \
  | grep -v "^.*:.*#" \
  | grep -v "_domain_path\|_table_path\|_is_system_schema\|_get_domain_schemas\
\|_apply_logger_overrides\|_compute_diff\|_retrieve_rid\|_asset_record_class\
\|_cache_features\|_add_workflow"
```

Expected: empty (or only comments / docstring text referencing the old names).

- [ ] **Step 5: Final commit if any fixups were needed**

```bash
git add <any fixup files>
git commit -m "docs(docstrings): fixups from final verification (ruff + mkdocs)"
```

---

## Summary Checklist

| Task | Module(s) | Priority | Docstrings | Renames | Dead code |
|---|---|---|---|---|---|
| 0 | (grep only) | prep | — | — | — |
| 1 | `pyproject.toml`, `CLAUDE.md` | infra | — | — | — |
| 2 | `core/mixins/dataset.py` | P1 | 3 | 1 (delete) | 1 |
| 3 | `core/mixins/annotation.py` | P1 | 12 | — | 1 |
| 4 | `feature.py` | P1 | 3 | — | — |
| 5 | `dataset/dataset_bag.py` | P1 | 2+ | — | — |
| 6 | `execution/execution.py` | P2 | 3 | — | — |
| 7 | `dataset/dataset.py` | P2 | 4 | — | — |
| 8 | `core/mixins/asset.py` | P2 | 2 | — | — |
| 9 | `path_builder.py` + `constants.py` | P3 | spot | 4 | — |
| 10 | `logging_config.py` + `schema_diff.py` | P3 | spot | 2 | — |
| 11 | `rid_resolution.py` + `asset_record.py` + `workflow.py` | P3 | spot | 3 | — |
| 12 | `core/base.py` | P3 | spot + `working_data` | 1 | 3 |
| 13 | `tools/validate_schema_doc.py` | P3 | — | 3 | — |
| 14 | `core/mixins/execution.py` | P3/4 | spot | 0–1 (pending §7a) | — |
| 15 | ~20 remaining modules | P4 | module + spot | — | — |
| 16 | `tests/` | follow-on | — | (reflected) | — |
| 17 | (verification) | final | — | — | — |

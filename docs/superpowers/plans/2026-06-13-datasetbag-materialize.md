# DatasetBag.materialize() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `DatasetBag.materialize()` method that fetches any not-yet-downloaded `fetch.txt` entries for an already-extracted bag, in place, with no catalog connection.

**Architecture:** Extract the path-only fetch tail of `materialize_dataset_bag` into a shared free function `materialize_bag_dir(bag_path, *, fetch_concurrency, logger)` in `bag_download.py`. Both the existing download path (`materialize_dataset_bag`, after its `download_dataset_minid` step) and the new offline method (`DatasetBag.materialize`) call it. The `DatasetBag` is a deliberately offline object — it imports only the path-only helper, never the `Dataset`-oriented function.

**Tech Stack:** Python ≥3.12, `bdbag` (`bdb.materialize`), pytest, `uv`, ruff.

---

## File Structure

- `src/deriva_ml/dataset/bag_download.py` — add `materialize_bag_dir` free function; refactor `materialize_dataset_bag` to delegate its fetch tail to it; export the new function in `__all__`.
- `src/deriva_ml/dataset/dataset_bag.py` — add `DatasetBag.materialize(self, *, fetch_concurrency=1) -> Self`.
- `tests/dataset/test_bag_materialize.py` — new unit tests (no catalog) for `materialize_bag_dir` and `DatasetBag.materialize`.
- `tests/dataset/test_bag_api_coverage.py` — add one integration test (live catalog) for the metadata-only → materialized flip.

---

## Task 1: Extract `materialize_bag_dir` free function (path-only fetch tail)

**Files:**
- Modify: `src/deriva_ml/dataset/bag_download.py` (imports near top; `materialize_dataset_bag` at 744–815; `__all__` at 818–824)

- [ ] **Step 1: Add a module logger import at the top of `bag_download.py`**

`bag_download.py` has no module logger today. Add the import alongside the other stdlib imports (after line 57, `from typing import ...`) and a module-level logger after the imports block. Add:

```python
import logging
```

and, immediately after the import block (before the first function/`TYPE_CHECKING` block), add:

```python
logger = logging.getLogger("deriva_ml.dataset.bag_download")
```

(Place the `logger = ...` line at module top level, near where other module constants would live — directly after the final `import`/`from` line group. If a `TYPE_CHECKING` block is the first thing after imports, put `logger = ...` just before it.)

- [ ] **Step 2: Add the `materialize_bag_dir` free function**

Insert this function **immediately before** `def materialize_dataset_bag(` (i.e., before line 744):

```python
def materialize_bag_dir(
    bag_path: Path,
    *,
    fetch_concurrency: int = 1,
    logger: "logging.Logger | None" = None,
) -> Path:
    """Fetch every ``fetch.txt`` entry for an already-extracted bag dir.

    Path-only — needs no catalog connection. ``fetch.txt`` carries
    absolute (Hatrac/S3) URLs, so materialization is a pure local
    operation over a bag that is already on disk. Idempotent: if the
    bag is already fully materialized, returns immediately without
    fetching.

    This is the shared fetch tail used by both
    :func:`materialize_dataset_bag` (after it has downloaded/extracted
    the bag) and :meth:`~deriva_ml.dataset.dataset_bag.DatasetBag.materialize`
    (which operates on a bag already on disk).

    Args:
        bag_path: Path to the extracted BDBag directory (parent of
            ``data/``).
        fetch_concurrency: Maximum number of concurrent file downloads.
        logger: Logger for progress messages. Defaults to this module's
            logger.

    Returns:
        ``Path`` to the bag directory (unchanged; only its contents grow).

    Raises:
        Exception: Propagates any error raised by ``bdb.materialize`` —
            e.g. a ``fetch.txt`` URL that is unreachable. The bag is
            left partially materialized in that case.
    """
    from deriva_ml.dataset.bag_cache import BagCache

    log = logger if logger is not None else globals()["logger"]
    bag_path = Path(bag_path)

    def fetch_progress_callback(current, total):
        log.info(f"Materializing bag: {current} of {total} file(s) downloaded.")
        return True

    def validation_progress_callback(current, total):
        log.info(f"Validating bag: {current} of {total} file(s) validated.")
        return True

    # If the bag already has every fetch.txt entry resolved, skip the
    # materialize call — there's nothing to download.
    if BagCache._is_fully_materialized(bag_path):
        log.info(f"Bag at {bag_path} already materialized.")
        return bag_path

    log.info(f"Materializing bag at {bag_path}")
    # Ensure parent directories exist for all fetch entries.
    fetch_file = bag_path / "fetch.txt"
    if fetch_file.exists():
        with fetch_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    rel_path = parts[2]
                    (bag_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
    bdb.materialize(
        bag_path.as_posix(),
        fetch_callback=fetch_progress_callback,
        validation_callback=validation_progress_callback,
        fetch_concurrency=fetch_concurrency,
    )
    return bag_path
```

- [ ] **Step 3: Refactor `materialize_dataset_bag` to delegate its fetch tail**

Replace the body of `materialize_dataset_bag` from line 776 (the first `def fetch_progress_callback`) through line 815 (`return Path(bag_path)`) with a thin body that downloads then delegates. The new body (keeping the existing signature and docstring at 744–775):

```python
    bag_path = download_dataset_minid(dataset, minid, use_minid)
    return materialize_bag_dir(
        bag_path,
        fetch_concurrency=fetch_concurrency,
        logger=dataset._logger,
    )
```

This removes the now-duplicated inline callbacks, the `_is_fully_materialized` short-circuit, and the parent-dir/`bdb.materialize` block (all moved into `materialize_bag_dir`). The "already materialized" log message text changes slightly (now keyed on path, not RID/version) — that is an internal log line, not asserted anywhere.

- [ ] **Step 4: Export `materialize_bag_dir` in `__all__`**

In the `__all__` list (was 818–824), add `"materialize_bag_dir",` in alphabetical position (between `"get_dataset_minid",` and `"materialize_dataset_bag",`):

```python
__all__ = [
    "create_dataset_minid",
    "download_dataset_minid",
    "fetch_minid_metadata",
    "get_dataset_minid",
    "materialize_bag_dir",
    "materialize_dataset_bag",
]
```

- [ ] **Step 5: Lint and confirm import works**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/dataset/bag_download.py && uv run python -c "from deriva_ml.dataset.bag_download import materialize_bag_dir, materialize_dataset_bag; print('ok')"
```
Expected: ruff passes (no errors), prints `ok`.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/bag_download.py && git commit -m "$(cat <<'EOF'
refactor(bag): extract path-only materialize_bag_dir from materialize_dataset_bag

Pulls the fetch.txt → bdb.materialize tail into a standalone free
function that needs no catalog connection, so it can be shared by both
the download path and the upcoming DatasetBag.materialize() method.
Behavior of materialize_dataset_bag is unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Unit test `materialize_bag_dir` (no catalog)

**Files:**
- Create: `tests/dataset/test_bag_materialize.py`

These tests build a minimal BDBag on disk with a `file://` fetch entry, so they need no catalog and no network beyond the local filesystem.

- [ ] **Step 1: Write the failing tests**

Create `tests/dataset/test_bag_materialize.py`:

```python
"""Unit tests for in-place bag materialization (no catalog required).

Builds a minimal BDBag on disk whose fetch.txt points at a local
``file://`` source, then exercises ``materialize_bag_dir`` and
``DatasetBag.materialize`` without any catalog connection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from bdbag import bdbag_api as bdb

from deriva_ml.dataset.bag_cache import BagCache
from deriva_ml.dataset.bag_download import materialize_bag_dir


def _make_holey_bag(tmp_path: Path) -> tuple[Path, Path]:
    """Create a valid BDBag with one un-fetched fetch.txt entry.

    Returns (bag_dir, expected_target_file). The referenced payload
    lives at a local ``file://`` URL so materialization needs no network.
    """
    # The remote payload the bag will fetch.
    payload = b"hello-materialize"
    src = tmp_path / "remote" / "payload.txt"
    src.parent.mkdir(parents=True)
    src.write_bytes(payload)

    # Build a bag around an (initially) empty data dir.
    bag_dir = tmp_path / "bag"
    (bag_dir / "data").mkdir(parents=True)
    bdb.make_bag(bag_dir.as_posix())

    # Add a remote entry via fetch.txt: <url>\t<length>\t<relpath>.
    rel = "data/assets/payload.txt"
    fetch_txt = bag_dir / "fetch.txt"
    fetch_txt.write_text(f"{src.as_uri()}\t{len(payload)}\t{rel}\n", encoding="utf-8")

    # Record the entry in the tag manifest so bag structure stays valid.
    md5 = hashlib.md5(payload).hexdigest()
    (bag_dir / "manifest-md5.txt").write_text(f"{md5}  {rel}\n", encoding="utf-8")

    return bag_dir, bag_dir / rel


def test_materialize_bag_dir_fetches_missing_entry(tmp_path: Path):
    """materialize_bag_dir downloads an un-fetched fetch.txt entry."""
    bag_dir, target = _make_holey_bag(tmp_path)
    assert not target.exists()
    assert not BagCache._is_fully_materialized(bag_dir)

    result = materialize_bag_dir(bag_dir)

    assert result == bag_dir
    assert target.exists()
    assert target.read_bytes() == b"hello-materialize"
    assert BagCache._is_fully_materialized(bag_dir)


def test_materialize_bag_dir_idempotent_noop(tmp_path: Path, monkeypatch):
    """On an already-complete bag, materialize_bag_dir does not fetch."""
    bag_dir, _ = _make_holey_bag(tmp_path)
    materialize_bag_dir(bag_dir)  # first call completes it
    assert BagCache._is_fully_materialized(bag_dir)

    # Second call must short-circuit before touching bdb.materialize.
    def _boom(*args, **kwargs):
        raise AssertionError("bdb.materialize should not be called on a complete bag")

    monkeypatch.setattr("deriva_ml.dataset.bag_download.bdb.materialize", _boom)
    result = materialize_bag_dir(bag_dir)
    assert result == bag_dir
```

- [ ] **Step 2: Run tests to verify they fail / behave**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_bag_materialize.py -q
```
Expected: both tests PASS (the implementation from Task 1 already exists). If `_make_holey_bag` doesn't produce a structure that `_is_fully_materialized` rejects (e.g. `validate_bag_structure` fails for an unrelated reason), the first test will surface it — fix the fixture, not the implementation.

> Note: this is a refactor-then-test ordering (the function exists from Task 1). The "failing first" discipline is preserved at the *feature* level by Task 3 (the `DatasetBag.materialize` method does not exist yet).

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/dataset/test_bag_materialize.py && git commit -m "$(cat <<'EOF'
test(bag): unit-test materialize_bag_dir with a local file:// fetch bag

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `DatasetBag.materialize()` method

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (add method after the `path` property, which ends at line 225)
- Modify: `tests/dataset/test_bag_materialize.py` (add a method-level unit test)

- [ ] **Step 1: Write the failing test for the method**

Append to `tests/dataset/test_bag_materialize.py`:

```python
class _StubModel:
    """Minimal stand-in for DatabaseModel exposing only ``bag_path``."""

    def __init__(self, bag_path: Path):
        self.bag_path = bag_path


def test_datasetbag_materialize_fetches_in_place(tmp_path: Path, monkeypatch):
    """DatasetBag.materialize() fetches missing files and returns self."""
    from deriva_ml.dataset.dataset_bag import DatasetBag

    bag_dir, target = _make_holey_bag(tmp_path)

    # Build a DatasetBag without running its catalog-touching __init__.
    bag = DatasetBag.__new__(DatasetBag)
    bag.model = _StubModel(bag_dir)

    assert not target.exists()
    result = bag.materialize()

    assert result is bag
    assert target.exists()
    assert target.read_bytes() == b"hello-materialize"
    assert BagCache._is_fully_materialized(bag_dir)
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_bag_materialize.py::test_datasetbag_materialize_fetches_in_place -q
```
Expected: FAIL with `AttributeError: 'DatasetBag' object has no attribute 'materialize'`.

- [ ] **Step 3: Implement `DatasetBag.materialize`**

In `src/deriva_ml/dataset/dataset_bag.py`, insert this method immediately after the `path` property (after line 225, before `def list_tables`):

```python
    def materialize(self, *, fetch_concurrency: int = 1) -> Self:
        """Fetch any not-yet-downloaded files for this bag, in place.

        A :class:`DatasetBag` may be downloaded metadata-only (via
        ``download_dataset_bag(..., materialize=False)``) or be left
        partially materialized (``cached_holey``) after an interrupted
        fetch. This method completes it: it reads the bag's ``fetch.txt``
        (which carries absolute Hatrac/S3 URLs) and downloads every
        referenced file into the bag directory. No catalog connection is
        used — materialization is a pure local operation over the bag
        already on disk.

        The bag's :attr:`path` is unchanged; only the directory contents
        grow. The SQLite mirror is unaffected (it is built from the CSV
        tables already present in a metadata-only bag).

        The call is idempotent: a fully-materialized bag returns
        immediately without fetching.

        Args:
            fetch_concurrency: Maximum number of concurrent file
                downloads.

        Returns:
            Self: this same bag (its assets are now present on disk),
            so the call can be chained, e.g.
            ``bag = ml.download_dataset_bag(spec).materialize()``.

        Raises:
            Exception: Propagates any error raised by the underlying
                ``bdbag`` fetch — e.g. a ``fetch.txt`` URL that is
                unreachable (source store down, or asset bytes never
                uploaded to a reachable store). The bag is left
                partially materialized in that case.

        Example:
            >>> spec = DatasetSpec(rid="1-abc123", materialize=False)  # doctest: +SKIP
            >>> bag = ml.download_dataset_bag(spec)  # metadata only      # doctest: +SKIP
            >>> bag.materialize()  # fetch the asset bytes in place       # doctest: +SKIP
            <deriva_ml.DatasetBag object ...>
        """
        from deriva_ml.core.logging_config import get_logger
        from deriva_ml.dataset.bag_download import materialize_bag_dir

        materialize_bag_dir(
            self.path,
            fetch_concurrency=fetch_concurrency,
            logger=get_logger(__name__),
        )
        return self
```

- [ ] **Step 4: Run the method test to verify it passes**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_bag_materialize.py -q
```
Expected: all three tests PASS.

- [ ] **Step 5: Lint**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_bag_materialize.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_bag_materialize.py && git commit -m "$(cat <<'EOF'
feat(bag): add DatasetBag.materialize() for in-place materialization

Lets a metadata-only or partially-materialized DatasetBag fetch its
remaining fetch.txt entries in place, with no catalog connection,
complementing download_dataset_bag(materialize=True). Returns self so
the call chains.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integration test — metadata-only → materialized flip (live catalog)

**Files:**
- Modify: `tests/dataset/test_bag_api_coverage.py` (add a test class at end of file, after line 413)

This mirrors the existing `TestNonMaterializedBagAccess` fixture pattern. The demo catalog may produce a bag with an empty `fetch.txt` (no remote assets); in that case `materialize=False` already yields a complete bag, so the test skips rather than asserting a flip that can't occur.

- [ ] **Step 1: Write the integration test**

Append to `tests/dataset/test_bag_api_coverage.py`:

```python
class TestBagMaterializeInPlace:
    """Tests for DatasetBag.materialize() against a live catalog."""

    def test_materialize_flips_metadata_only_to_materialized(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """A metadata-only bag becomes fully materialized in place."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Download metadata-only.
        bag = dataset.download_dataset_bag(version=version, use_minid=False, materialize=False)

        if BagCache._is_fully_materialized(bag.path):
            pytest.skip("Demo bag has no remote fetch.txt entries; nothing to materialize.")

        # Materialize in place; must return the same object.
        result = bag.materialize()
        assert result is bag
        assert BagCache._is_fully_materialized(bag.path)

        # Cache status flips to materialized.
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value

    def test_materialize_idempotent_on_materialized_bag(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """materialize() on an already-materialized bag is a safe no-op."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        bag = dataset.download_dataset_bag(version=version, use_minid=False)  # materialize=True default
        assert BagCache._is_fully_materialized(bag.path)

        result = bag.materialize()
        assert result is bag
        assert BagCache._is_fully_materialized(bag.path)
```

- [ ] **Step 2: Run the integration test**

Run (needs `DERIVA_HOST`):
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_bag_api_coverage.py::TestBagMaterializeInPlace -q --timeout=600
```
Expected: PASS (or SKIP on the first test if the demo bag has no remote assets; the second test always runs).

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/dataset/test_bag_api_coverage.py && git commit -m "$(cat <<'EOF'
test(bag): integration test for DatasetBag.materialize() in-place flip

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Full unit-suite sanity, PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run the fast unit suites to confirm no regression**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_bag_materialize.py tests/local_db/ tests/asset/ tests/model/ -q
```
Expected: all pass.

- [ ] **Step 2: Lint + format the whole change set**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff format src/deriva_ml/dataset/bag_download.py src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_bag_materialize.py tests/dataset/test_bag_api_coverage.py && uv run ruff check src tests
```
Expected: format makes no/minimal changes; check passes. Commit any format-only changes:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git diff --quiet || (git add -A && git commit -m "style: ruff format")
```

- [ ] **Step 3: Push and open the PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git push -u origin feature/datasetbag-materialize && gh pr create --title "feat(bag): DatasetBag.materialize() for in-place materialization" --body "$(cat <<'EOF'
## Summary
Adds `DatasetBag.materialize(*, fetch_concurrency=1)` — fetches any
not-yet-downloaded `fetch.txt` entries for an already-extracted bag, in
place, with no catalog connection. Complements the existing cache-aware
`download_dataset_bag(materialize=True)` path by giving an affordance
directly on a `DatasetBag` object (a metadata-only or `cached_holey`
bag you already hold).

## Design
- The fetch tail of `materialize_dataset_bag` (parent-dir prep +
  `bdb.materialize`, gated by `_is_fully_materialized`) is extracted into
  a shared, catalog-free free function `materialize_bag_dir(bag_path, *,
  fetch_concurrency, logger)`. Both the download path and the new method
  call it — one implementation, two callers.
- `DatasetBag` stays offline-only: it imports only the path-only helper,
  never the `Dataset`-oriented `materialize_dataset_bag`.
- Errors from `bdbag` fetch propagate (no new exception layer), matching
  the existing download path.

## Tests
- Unit (no catalog): `tests/dataset/test_bag_materialize.py` builds a
  minimal BDBag with a local `file://` fetch entry and verifies the fetch,
  idempotent no-op, and the method returning `self`.
- Integration (live catalog): `TestBagMaterializeInPlace` verifies the
  metadata-only → `cached_materialized` flip and idempotency (skips the
  flip assertion if the demo bag carries no remote assets).

Spec: `docs/superpowers/specs/2026-06-13-datasetbag-materialize-design.md`
Plan: `docs/superpowers/plans/2026-06-13-datasetbag-materialize.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report the PR URL to the user**

Paste the PR URL. Note the patch version bump happens **after** merge, on clean `main` (repo rule), not in this branch.

---

## Self-Review notes

- **Spec coverage:** §3.1 method → Task 3; §3.2 propagate error → docstring `Raises` + no try/except anywhere; §3.3 shared helper → Task 1; §3.4 logger arg → Task 1 Step 2 (`logger` param) + Task 3 (`get_logger(__name__)`); §4 `__all__` export → Task 1 Step 4; §5.1 unit → Tasks 2 & 3; §5.2 integration → Task 4.
- **Type consistency:** `materialize_bag_dir(bag_path, *, fetch_concurrency, logger)` signature is identical at definition (Task 1), free-function callers (`materialize_dataset_bag` Task 1 Step 3), and method caller (Task 3 Step 3). `DatasetBag.materialize(self, *, fetch_concurrency=1) -> Self` consistent between test (Task 3 Step 1) and impl (Task 3 Step 3).
- **No placeholders:** every code step shows full code; every run step shows the command and expected result.

# `DatasetBag.materialize()` — in-place bag materialization — Design

**Date:** 2026-06-13
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml`

## 1. Problem statement

There is a user-accessible path to materialize an *unmaterialized*
(metadata-only) bag today, but it is keyed off the **dataset spec +
cache**, not off a `DatasetBag` instance:

```python
bag = ml.download_dataset_bag(DatasetSpec(rid=..., version=..., materialize=True))
```

`download_dataset_bag(materialize=True)` is cache-aware — it skips the
export/metadata work already on disk and fetches only the missing asset
bytes (via `materialize_dataset_bag` → `bdb.materialize()`). What is
**missing** is a convenience affordance directly on a `DatasetBag`
object: given a `DatasetBag` you already hold (downloaded with
`materialize=False`, or a `cached_holey` bag), there is no
`bag.materialize()` to fill it in place. The grep for public
materialize/fetch methods on `dataset_bag.py` returns nothing.

This spec adds that method.

## 2. Key constraint (what makes this clean)

A `DatasetBag` is a deliberately **offline** object. It holds **no**
`DerivaML` instance and **no** catalog connection — only the on-disk
bag path (`bag.path`, via `self.model.bag_path`) and the SQLite mirror.
`DerivaMLBagView` and `DatabaseModel` likewise carry no back-reference
to a live catalog.

Verified that in-place materialization needs no catalog:

- `bdb.materialize(bag_path)` reads `fetch.txt` (which holds **absolute**
  Hatrac/S3 URLs) and fetches the referenced files into the bag dir.
  Path-only; no catalog.
- `BagCache._is_fully_materialized(bag_path)` is a pure path check
  (validate bag structure + every `fetch.txt` entry resolves locally).
- The SQLite mirror is built from the CSV tables, which are already
  present in a metadata-only bag, so materialization does not touch it.

Both of these already run as the **tail** of
`materialize_dataset_bag` (`bag_download.py:744`), *after* its download
step. Because a `DatasetBag` can only exist if its bag is already
extracted on disk, materializing it in place is exactly that tail —
no download, no catalog, no re-export, no snapshot re-pin.

## 3. Design

### 3.1 New method on `DatasetBag`

```python
def materialize(self, *, fetch_concurrency: int = 1) -> Self:
    """Fetch any not-yet-downloaded files for this bag, in place."""
```

Behavior:

1. If `_is_fully_materialized(self.path)` → log "already materialized"
   and return `self` (idempotent; no work, no network).
2. Otherwise: ensure parent dirs exist for every `fetch.txt` entry,
   then call `bdb.materialize(self.path, fetch_concurrency=...)` with
   the same fetch/validation progress callbacks the existing free
   function uses, then return `self`.
3. Returns `self` so the call chains
   (`bag = ml.download_dataset_bag(spec).materialize()`) and the
   *same* object's assets are now present on disk (its `path` is
   unchanged; only the directory contents grew).

`Self` is already imported in `dataset_bag.py`.

### 3.2 Failure mode

**Propagate the `bdb` fetch error.** If a `fetch.txt` URL is
unreachable (source Hatrac down, or asset bytes never uploaded to a
reachable store), `bdb.materialize` raises and the bag is left in the
partially-materialized (`cached_holey`) state it was already in. This
matches `materialize_dataset_bag`'s current behavior exactly — no new
error-translation layer, consistent with the existing download path.

### 3.3 Avoiding duplication

The "ensure parent dirs + `bdb.materialize`" block is currently inline
in `materialize_dataset_bag`. Extract that tail into a small free
function in `bag_download.py`:

```python
def materialize_bag_dir(
    bag_path: Path,
    *,
    fetch_concurrency: int = 1,
    logger: logging.Logger,
) -> Path:
    """Fetch every fetch.txt entry for an already-extracted bag dir.

    Path-only; no catalog. Idempotent: returns immediately if the bag
    is already fully materialized.
    """
```

Both call sites use it:

- `materialize_dataset_bag` — after `download_dataset_minid` returns the
  on-disk path, delegate the fetch tail to `materialize_bag_dir`
  (passing `dataset._logger`). The early
  `_is_fully_materialized` short-circuit moves into the helper.
- `DatasetBag.materialize()` — call `materialize_bag_dir(self.path,
  fetch_concurrency=..., logger=get_logger(__name__))`.

One implementation, two callers. `DatasetBag` does **not** import the
`Dataset`-oriented `materialize_dataset_bag` (which needs a live
catalog); it imports only the path-only helper.

### 3.4 Logging

`materialize_dataset_bag` logs via `dataset._logger`. The extracted
helper takes a `logger` argument so both callers supply their own:
the free function passes `dataset._logger`; `DatasetBag.materialize`
passes `get_logger(__name__)` (the standard
`deriva_ml.core.logging_config` pattern).

## 4. Deliverables

- `src/deriva_ml/dataset/bag_download.py`:
  - new free function `materialize_bag_dir(...)`;
  - `materialize_dataset_bag` refactored to delegate its fetch tail to
    it (behavior unchanged);
  - `materialize_bag_dir` added to `__all__`.
- `src/deriva_ml/dataset/dataset_bag.py`:
  - new `DatasetBag.materialize(self, *, fetch_concurrency=1) -> Self`
    with a Google-style docstring + runnable-where-possible `Example:`.
- Tests (below).
- Patch version bump after the PR merges to `main`.

## 5. Testing

### 5.1 Unit (no catalog)

In a new `tests/dataset/test_bag_materialize.py` (or alongside existing
bag tests):

- **Fetches a missing entry:** build a minimal bag dir with a valid
  BDBag structure and a `fetch.txt` whose single entry points at a
  local `file://` source and a not-yet-present target; call
  `bag.materialize()` (or `materialize_bag_dir` directly for the
  pure-path unit); assert the target file now exists and
  `_is_fully_materialized` is `True`.
- **Idempotent no-op:** on an already-complete bag, `materialize()`
  returns immediately (assert no fetch attempted — e.g. monkeypatch
  `bdb.materialize` to raise, prove it is not called) and returns the
  same object.

RIDs in any fixture come from a fixture-produced row, never a literal
(repo rule).

### 5.2 Integration (live catalog)

Guarded like the other catalog-dependent dataset tests:

- Download a small demo dataset with `materialize=False`; assert cache
  status is `cached_metadata_only`/`cached_holey`.
- Call `bag.materialize()`; assert `_is_fully_materialized(bag.path)`
  and that the cache status flips to `cached_materialized`.
- Assert the call returns the same `DatasetBag` object (`is`).

## 6. Non-goals

- No change to `download_dataset_bag` signatures or the `materialize=`
  flag — purely additive.
- No catalog re-export, snapshot re-pin, or version change — the method
  fetches exactly the URLs the bag's `fetch.txt` already carries.
- No new error-translation layer (§3.2).
- Not a `de-materialize` / prune-assets operation (cache deletion is the
  `manage-deriva-storage` surface).

## 7. Verification

- `uv run ruff check` / `format` clean.
- Unit tests pass with no catalog
  (`DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_bag_materialize.py -q`).
- Integration test passes against `DERIVA_HOST=localhost`.
- All changes via a feature-branch PR (repo rule); `bump-version patch`
  on clean `main` after merge.

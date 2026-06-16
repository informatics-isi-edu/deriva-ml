# `BagBuilder.add_asset` resolves symlink sources (Linux bag corruption fix)

**Date:** 2026-06-16
**Status:** Approved design — ready for implementation plan
**Scope:** deriva-py (fix + contract test) + deriva-ml (pin advance only)

## Problem

Asset uploads assemble files into a bag via symlinks, then hardlink them in.
This works on macOS and **corrupts the bag on Linux**.

`deriva_ml.execution.asset_upload.asset_file_path()` (default `copy_file=False`)
creates an **absolute symlink** in the flat staging tree pointing at the original
source file — which is frequently *outside* the bag (e.g. Hydra writes
`overrides.yaml` / `notebook.log` to `~/.deriva-ml/.../hydra/<timestamp>/`, and
`run_notebook.py:869` registers it with the default `copy_file=False`).

`deriva_ml.execution.bag_commit._add_asset_rows_to_bag` then passes that staging
path — **which is the symlink** — to deriva-py's
`BagBuilder.add_asset(src, link=True)`, which does `os.link(src, dest)`.

### Root cause (proven on both platforms)

`os.link(symlink, dest)` behaves differently by OS, despite both defaulting to
`follow_symlinks=True` and both having `os.link in os.supports_follow_symlinks`:

| Platform | `os.link(symlink, dest)` hardlinks the… | Bag result |
|---|---|---|
| **macOS** (Darwin) | **target** (real file) — BSD `link(2)` follows symlinks | bytes ✅ |
| **Linux** | **symlink itself** — POSIX `link(2)` does not follow | a symlink to an external absolute path ❌ |

Verified empirically: the identical `os.link(absolute_symlink, dest)` script
hardlinks the real file on this macOS host and the symlink inode under
`python:3.12-slim` (Linux Docker).

This violates `add_asset`'s own documented invariant
(`builder.py:397-402`: "Hardlinks live inside the bag as regular files, so
bagit's safety model is satisfied"). Two downstream symptoms result from the
*same* defect:

- **Symptom A (Linux, `os.link`):** the bag's `data/asset/.../file` is a hardlink
  to the symlink inode whose target is an external absolute path. When the bag is
  hashed, zipped, moved, or loaded elsewhere, the bytes are missing/broken.
- **Symptom B (any OS, bagit):** `bdbag.make_bag(update=True)` →
  `_load_manifests()` → bagit `_path_is_dangerous()` calls
  `os.path.realpath()` on the symlink, escapes the bag root, and the
  `commonprefix` check fails → **"unsafe path"** rejection.

Both are: *a symlink-to-an-external-absolute-path ended up in the bag payload.*

### Why it slipped through

- Symlinks are created **by default** (`copy_file=False`), so every macOS dev run
  silently works (BSD `os.link` follows).
- deriva-ml's asset/bag tests need a live catalog and don't run in CI; deriva-ml
  CI (`ubuntu-latest`) only runs schema-validation/release/docs — the symlink→bag
  path is never exercised on Linux in CI.
- deriva-py's existing `test_builder_add_asset_link_mode_hardlinks_source` links a
  **regular file** (passes everywhere) and never covers the symlink-source case.

## Goal

`BagBuilder.add_asset(src, link=True)` must embed the **real file's bytes as a
regular file** in the bag on every platform, even when `src` is a symlink —
restoring its documented invariant and fixing both symptoms at the source.

## Approach (decided during brainstorming)

**deriva-py-only fix + a deriva-ml pin advance. No deriva-ml code change, no
temporary bridge** (deploy immediately: the deriva-py fix lands and the pin
advances in one cycle, so there is no window where deriva-ml runs the broken
`add_asset`).

### The fix (deriva-py, `deriva/bag/builder.py`)

In `add_asset`, resolve the source before any link/copy so the hardlink always
targets the real inode:

```python
source_path = Path(source_path).resolve()
```

Placed right after the existing `source_path = Path(source_path)` (~L424),
**before** the `.is_file()` check. Verified-safe against the rest of the method:

- `.is_file()` (L425) already follows symlinks — unaffected; resolving first just
  makes the not-found error report the real path.
- `os.link(source_path, dest)` (L452) now links the real file on every platform —
  the fix.
- `shutil.copy2` copy + EXDEV-fallback paths (L465, L467) already follow symlinks
  — unaffected; resolving makes it explicit.
- Idempotency key `_embedded_assets[dest]["source_path"]` (L437, L471) records the
  **resolved** path; both compare sides see the resolved form, so re-adding the
  same logical source (symlink vs resolved) stays consistent rather than
  spuriously raising the "different source" `ValueError`.

`os` and `Path` are already imported in `builder.py`. One line.

### deriva-ml (pin advance only)

No code change. After the deriva-py PR merges on the `deriva-ml` branch:

1. `git -C /Users/carl/GitHub/deriva-py rev-parse origin/deriva-ml` → the merge SHA.
2. Update **both** pin locations in `deriva-ml/pyproject.toml` in lockstep — the
   `project.dependencies` URL `deriva @ git+...@<sha>` and the `[tool.uv.sources]`
   `rev = "<sha>"`.
3. `uv lock && uv sync`, run the deriva-ml suite, land via PR.

## Components

This is a single-function fix; the "components" are the two repos' deliverables.

- **deriva-py — the fix** (`deriva/bag/builder.py::add_asset`): one-line resolve.
- **deriva-py — the contract test** (`tests/deriva/bag/test_builder.py`): a new
  test pinning the symlink-source case (below). Lands in the **same PR** as the
  fix (never fix-without-test — that opens a regression window in deriva-py's main
  branch).
- **deriva-ml — the pin advance** (`pyproject.toml` + `uv.lock`): mechanical,
  separate PR after upstream merges.

## Test strategy

**The contract test belongs upstream in deriva-py**, alongside the fix that makes
the promise (per the deriva-ml CLAUDE.md cross-repo rule). It is a near-twin of
the existing `test_builder_add_asset_link_mode_hardlinks_source`, but links a
**symlink** source — the case that diverges by platform:

```python
def test_builder_add_asset_link_mode_resolves_symlink_source(tmp_path):
    """link=True must embed the REAL file as a regular file in the bag even when
    the source is a SYMLINK — not hardlink the symlink inode (which on Linux puts
    a symlink-to-an-external-path in the bag payload and breaks it)."""
    real = tmp_path / "real.bin"
    real.write_bytes(b"real bytes")
    link = tmp_path / "link.bin"
    link.symlink_to(real.resolve())  # absolute symlink, as asset_file_path makes

    out = tmp_path / "bag"
    bb = BagBuilder(metadata=_two_table_metadata(), output_dir=out)
    bb.add_asset("Image", "I1", link, link=True)
    bb.finalize(make_bdbag=False)

    dest = out / "data" / "asset" / "Image" / "I1" / "link.bin"
    assert dest.is_file()
    assert not dest.is_symlink(), "bag payload must not contain a symlink"
    assert dest.read_bytes() == b"real bytes"
```

The load-bearing assertion is **`not dest.is_symlink()`** — exactly what fails on
Linux today and passes after the resolve. (On macOS it already passes, which is
why the bug was invisible.)

**Enforcement caveat (verified):** the deriva-py checkout has **no `.github/`
workflows** — there is no Linux CI in this branch to automatically catch the
regression. Therefore the implementation MUST **run the new test on Linux via
Docker** (`python:3.12-slim`, the same probe used to prove the bug) as part of the
fix's verification, not rely on CI. If the test passes only on macOS it has not
demonstrated the fix. Optionally, the implementation may add a `make_bdbag=True`
assertion that bagit validation does not raise "unsafe path" (symptom B) — but a
regular file cannot trip `_path_is_dangerous`, so the `not is_symlink` assertion
already prevents symptom B; add the bagit-level assertion only if it earns its
weight.

**deriva-ml verification (not committed):** after the pin advances, re-run the
Docker-Linux `os.link`-on-symlink probe against the fixed deriva-py (now embeds
bytes), and run a real deriva-ml asset-commit on localhost to confirm the bag
contains regular files with no symlinks and no "unsafe path." These are
acceptance checks, not new committed deriva-ml tests (the fix isn't in deriva-ml).

## Sequencing

1. **deriva-py PR** off the `deriva-ml` branch: the resolve fix **+** the contract
   test in one PR. Run the deriva-py bag suite locally **and on Linux via Docker**.
   Human reviews/merges the upstream PR (I open it; I do not self-merge an upstream
   repo without approval) — so the flow pauses for the merge.
2. **Capture merge SHA** from `origin/deriva-ml` after merge.
3. **deriva-ml pin advance PR**: both pin locations → the SHA, `uv lock && uv sync`,
   deriva-ml suite green, land.
4. **Verify** (Docker-Linux probe + live localhost asset-commit) against the
   pinned, fixed deriva-py.
5. **Release**: `bump-version` on deriva-ml `main` after the pin PR merges (the pin
   advance is part of the bump — never bump with a stale pin). Level chosen at that
   point (this is a consumer-visible bug fix → likely patch).

## Out of scope

- No change to `asset_file_path`'s default `copy_file=False` — symlinked staging
  is a deliberate disk-saving choice; the fix is that `add_asset` resolves before
  linking, which makes symlinked staging safe. (Changing the default to copy would
  pay 1× disk/IO for every asset and is unnecessary once `add_asset` resolves.)
- No change to `restructure.py`'s symlink export (ImageFolder trees for third-party
  trainers) — it has a copy fallback and is not in the bag-upload path.
- No change to the download-side asset-cache symlinks (`asset_upload.py:1073/1093`)
  — separate flow, not bag payload.
- No deriva-ml code change and no temporary bridge.

## Risks

- **`.resolve()` on a broken/missing symlink:** if `src` is a dangling symlink,
  `.resolve()` returns a non-existent path and the existing `.is_file()` check
  raises `FileNotFoundError` — same failure class as today (a missing source),
  just reported with the resolved path. Acceptable; the contract test uses a valid
  symlink.
- **Idempotency-key change:** resolving changes the recorded `source_path`. A
  caller that re-adds the *same logical file* via a symlink then via the resolved
  path now sees them as equal (both resolved) — strictly better than today's
  spurious "different source" mismatch. No regression; the existing
  `test_builder_add_asset_idempotent_same_source` and
  `...duplicate_destination_different_source` tests must still pass.
- **No upstream Linux CI:** mitigated by the mandatory Docker-Linux test run during
  implementation (the regression won't be auto-caught later, but the fix is proven
  on Linux at land time, and the test is in place for whenever CI is added).

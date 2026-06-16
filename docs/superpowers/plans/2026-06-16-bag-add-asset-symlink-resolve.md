# BagBuilder.add_asset Symlink-Resolve Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `deriva.bag.builder.BagBuilder.add_asset(src, link=True)` embed the real file's bytes as a regular file in the bag even when `src` is a symlink, fixing Linux bag corruption (and bagit "unsafe path") — then advance deriva-ml's deriva-py pin to the fixed commit.

**Architecture:** One-line fix in deriva-py (`source_path = Path(source_path).resolve()` before linking) plus a contract test pinning the symlink-source case, in the **same** deriva-py PR. The bug is Linux-only (`os.link` follows symlinks on macOS/BSD but hardlinks the symlink inode on Linux), so the failing-test (RED) and the fixed-test (GREEN) are both verified on **Linux via Docker** (`python:3.12-slim`) since the deriva-py checkout has no Linux CI. deriva-ml gets no code change — only a two-location pin advance after the upstream PR merges.

**Tech Stack:** Python 3.12, `os.link`/`pathlib`, pytest, `uv`, Docker (`python:3.12-slim`) for the Linux proof. deriva-py checkout at `/Users/carl/GitHub/deriva-py` (on branch `deriva-ml`, at the SHA deriva-ml currently pins). deriva-ml at `/Users/carl/GitHub/DerivaML/deriva-ml`.

---

## File structure

- **deriva-py — Modify** `/Users/carl/GitHub/deriva-py/deriva/bag/builder.py` (the `add_asset` method, ~L424): one-line resolve.
- **deriva-py — Modify** `/Users/carl/GitHub/deriva-py/tests/deriva/bag/test_builder.py`: add one contract test for the symlink-source case (mirrors the existing `test_builder_add_asset_link_mode_hardlinks_source` at L224).
- **deriva-ml — Modify** `/Users/carl/GitHub/DerivaML/deriva-ml/pyproject.toml`: advance the deriva-py pin in BOTH locations (`project.dependencies` URL + `[tool.uv.sources]` rev). `uv.lock` updates via `uv lock`.

> **CWD discipline:** chain `cd` into every Bash call. deriva-py commands run in `/Users/carl/GitHub/deriva-py`; deriva-ml commands in `/Users/carl/GitHub/DerivaML/deriva-ml`. Both are `uv`-managed (`uv run pytest`, `uv lock`, `uv sync`).

> **Upstream-merge pause:** the deriva-py PR (Tasks 1–3) is on an **upstream** repo. Open it but do NOT self-merge — the human reviews/merges. Tasks 4–5 (pin advance) only run **after** the upstream PR has merged to the `deriva-ml` branch.

---

### Task 1: Failing contract test in deriva-py (RED on Linux)

Add a test that links a **symlink** source with `link=True` and asserts the bag contains a regular file with the real bytes — NOT a symlink. This fails on Linux today (the bug) and passes after Task 2.

**Files:**
- Modify: `/Users/carl/GitHub/deriva-py/tests/deriva/bag/test_builder.py`

- [ ] **Step 1: Read the existing twin test to mirror its style**

Run: `cd /Users/carl/GitHub/deriva-py && sed -n '224,250p' tests/deriva/bag/test_builder.py`
This is `test_builder_add_asset_link_mode_hardlinks_source` — it links a REGULAR file and asserts `not dest.is_symlink()`. The new test is its symlink-source twin.

- [ ] **Step 2: Write the failing test**

Add this function immediately after `test_builder_add_asset_link_mode_hardlinks_source` (after its last line, ~L248) in `tests/deriva/bag/test_builder.py`:

```python
def test_builder_add_asset_link_mode_resolves_symlink_source(
    tmp_path: Path,
) -> None:
    """link=True must embed the REAL file as a regular file in the bag even when
    the source is a SYMLINK.

    asset_file_path() in deriva-ml stages assets as absolute symlinks. os.link()
    follows the symlink on macOS/BSD (hardlinks the real file) but hardlinks the
    SYMLINK INODE on Linux — putting a symlink-to-an-external-path in the bag
    payload, which corrupts the bag and trips bagit's _path_is_dangerous(). The
    bag payload must contain the real file's bytes as a regular file on every
    platform.
    """
    real = tmp_path / "real.bin"
    real.write_bytes(b"real bytes")
    link = tmp_path / "link.bin"
    link.symlink_to(real.resolve())  # absolute symlink, as asset_file_path makes
    assert link.is_symlink()

    out = tmp_path / "bag"
    bb = BagBuilder(metadata=_two_table_metadata(), output_dir=out)
    bb.add_asset("Image", "I1", link, link=True)
    bb.finalize(make_bdbag=False)

    dest = out / "data" / "asset" / "Image" / "I1" / "link.bin"
    assert dest.is_file()
    # The load-bearing assertion: the bag payload must not be a symlink.
    # Fails on Linux pre-fix (hardlink to the symlink inode), passes after.
    assert not dest.is_symlink(), f"bag payload must not contain a symlink: {dest}"
    assert dest.read_bytes() == b"real bytes"
```

- [ ] **Step 3: Run on the host (macOS) — note it likely PASSES (bug is invisible on macOS)**

Run: `cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/bag/test_builder.py::test_builder_add_asset_link_mode_resolves_symlink_source -v 2>&1 | tail -8`
Expected on macOS: **PASS** (BSD `os.link` follows the symlink). This is EXPECTED and is exactly why the bug was invisible — do NOT treat a macOS pass as "test is wrong." The RED must be demonstrated on Linux (next step).

- [ ] **Step 4: Run on Linux via Docker — must FAIL (the RED that proves the bug)**

Run:
```bash
cd /Users/carl/GitHub/deriva-py && docker run --rm -v "$PWD":/work -w /work python:3.12-slim bash -c "pip -q install uv 2>/dev/null; uv run --frozen pytest tests/deriva/bag/test_builder.py::test_builder_add_asset_link_mode_resolves_symlink_source -v 2>&1 | tail -15"
```
Expected on Linux: **FAIL** at `assert not dest.is_symlink()` — the bag payload IS a symlink (the bug). If the test instead ERRORs on setup (uv/install/import problem) rather than failing on the assertion, fix the Docker invocation until it runs and fails on the assertion. If `uv run --frozen` has trouble in the container, fall back to: `pip -q install -e . pytest && python -m pytest tests/deriva/bag/test_builder.py::test_builder_add_asset_link_mode_resolves_symlink_source -v`. The goal: a clean assertion failure on Linux proving the bug.

> If Docker is unavailable in the environment, STOP and report — the Linux RED/GREEN proof is mandatory per the spec (no Linux CI exists upstream). Do not proceed on the macOS pass alone.

- [ ] **Step 5: Commit the failing test**

```bash
cd /Users/carl/GitHub/deriva-py
git add tests/deriva/bag/test_builder.py
git commit -m "test(bag): add_asset(link=True) must resolve symlink sources (RED on Linux)"
```

---

### Task 2: The fix in deriva-py `add_asset`

Resolve the source before linking so `os.link` always targets the real inode.

**Files:**
- Modify: `/Users/carl/GitHub/deriva-py/deriva/bag/builder.py` (`add_asset`, ~L424)

- [ ] **Step 1: Read the exact lines around the source-path handling**

Run: `cd /Users/carl/GitHub/deriva-py && sed -n '422,432p' deriva/bag/builder.py`
You should see `self._check_not_finalized()`, then `source_path = Path(source_path)`, then `if not source_path.is_file():`.

- [ ] **Step 2: Apply the one-line resolve**

In `deriva/bag/builder.py`, change:
```python
        self._check_not_finalized()
        source_path = Path(source_path)
        if not source_path.is_file():
```
to:
```python
        self._check_not_finalized()
        # Resolve symlinks so link=True hardlinks the REAL file, not the symlink
        # inode. os.link() follows symlinks on macOS/BSD but not on Linux, where
        # it would otherwise put a symlink-to-an-external-path in the bag payload
        # (corrupting the bag and tripping bagit's _path_is_dangerous safety
        # check). Resolving here keeps the documented invariant — "hardlinks live
        # inside the bag as regular files" — true on every platform.
        source_path = Path(source_path).resolve()
        if not source_path.is_file():
```

- [ ] **Step 3: Run the new test on the host (macOS) — PASS**

Run: `cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/bag/test_builder.py::test_builder_add_asset_link_mode_resolves_symlink_source -v 2>&1 | tail -6`
Expected: PASS (was already passing on macOS; still passes).

- [ ] **Step 4: Run the new test on Linux via Docker — now PASS (the GREEN proving the fix)**

Run:
```bash
cd /Users/carl/GitHub/deriva-py && docker run --rm -v "$PWD":/work -w /work python:3.12-slim bash -c "pip -q install uv 2>/dev/null; uv run --frozen pytest tests/deriva/bag/test_builder.py::test_builder_add_asset_link_mode_resolves_symlink_source -v 2>&1 | tail -10"
```
Expected on Linux: **PASS** — `dest` is now a regular file with the real bytes (was FAIL in Task 1 Step 4). This is the proof the fix works on the platform where it broke.

- [ ] **Step 5: Run the FULL deriva-py bag-builder suite (host) — no regressions**

Run: `cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/bag/test_builder.py -q 2>&1 | tail -8`
Expected: ALL pass (was 33 tests + your new one = 34). Pay special attention that these still pass — they exercise the idempotency/duplicate-source key that now records a resolved path:
- `test_builder_add_asset_idempotent_same_source` (real file, resolves stably → still idempotent)
- `test_builder_add_asset_duplicate_destination_different_source` (two distinct real files → still raises "already populated")
- `test_builder_add_asset_link_mode_passes_bagit_validation` (regular-file source → unaffected)
If any of these fail, STOP — resolving changed real-file behavior unexpectedly; report it.

- [ ] **Step 6: Run the broader deriva-py bag tests (host) to catch any other consumer**

Run: `cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/bag/ -q 2>&1 | tail -8`
Expected: all pass (or pre-existing skips unrelated to this change). `add_asset` is also used by `add_assets` (bulk) and the database/loader tests — confirm none regress.

- [ ] **Step 7: Commit the fix**

```bash
cd /Users/carl/GitHub/deriva-py
git add deriva/bag/builder.py
git commit -m "fix(bag): add_asset resolves symlink sources so link=True hardlinks the real file

os.link() follows symlinks on macOS/BSD but hardlinks the symlink inode on
Linux, putting a symlink-to-an-external-path in the bag payload (corrupting
the bag and tripping bagit's unsafe-path check). Resolve source_path before
linking so the documented 'hardlinks are regular files' invariant holds on
every platform. Pinned by test_builder_add_asset_link_mode_resolves_symlink_source."
```

---

### Task 3: Open the deriva-py PR (upstream — do NOT self-merge)

**Files:** none (git/PR only)

- [ ] **Step 1: Confirm branch + push**

Run: `cd /Users/carl/GitHub/deriva-py && git rev-parse --abbrev-ref HEAD && git log --oneline -3`
The fix + test commits must be on a branch off the `deriva-ml` branch (the branch deriva-ml pins). If you are on `deriva-ml` directly, create a feature branch first: `cd /Users/carl/GitHub/deriva-py && git checkout -b fix/add-asset-resolve-symlink && git log --oneline -2` (the two commits come with you). Then push:
`cd /Users/carl/GitHub/deriva-py && git push -u origin HEAD 2>&1 | tail -3`

- [ ] **Step 2: Open the PR against the `deriva-ml` branch**

```bash
cd /Users/carl/GitHub/deriva-py && gh pr create --base deriva-ml \
  --title "fix(bag): add_asset resolves symlink sources (Linux bag corruption)" \
  --body "## Summary
\`BagBuilder.add_asset(src, link=True)\` did \`os.link(src, dest)\`. When \`src\` is a symlink (deriva-ml stages assets as absolute symlinks via \`asset_file_path\`), \`os.link\` follows it on macOS/BSD (bag gets the real bytes) but hardlinks the **symlink inode** on Linux — putting a symlink-to-an-external-path in the bag payload. This corrupts the bag on Linux and trips bagit's \`_path_is_dangerous()\` 'unsafe path' check.

## Fix
Resolve \`source_path\` before linking so the hardlink targets the real file on every platform, restoring the method's documented invariant ('hardlinks live inside the bag as regular files').

## Test
\`test_builder_add_asset_link_mode_resolves_symlink_source\` links a symlink source and asserts the bag payload is a regular file with the real bytes (\`not dest.is_symlink()\`). Verified on Linux via Docker (\`python:3.12-slim\`): FAILS before the fix, PASSES after. (macOS passes both ways — which is why the bug was invisible.)

🤖 Generated with [Claude Code](https://claude.com/claude-code)" 2>&1 | tail -3
```

- [ ] **Step 2b: STOP — hand off for human merge**

Report the PR URL to the controller. The upstream PR must be reviewed/merged by a human. Do NOT proceed to Task 4 until the controller confirms the PR has merged to `origin/deriva-ml`.

---

### Task 4: Advance the deriva-py pin in deriva-ml

**Only after the deriva-py PR has merged to `origin/deriva-ml`.**

**Files:**
- Modify: `/Users/carl/GitHub/DerivaML/deriva-ml/pyproject.toml` (two pin locations)
- Modify: `/Users/carl/GitHub/DerivaML/deriva-ml/uv.lock` (via `uv lock`)

- [ ] **Step 1: Get the merged SHA**

Run: `cd /Users/carl/GitHub/deriva-py && git fetch -q origin && git rev-parse origin/deriva-ml`
Capture this SHA (call it `<NEWSHA>`). Confirm it is the merge of the fix: `cd /Users/carl/GitHub/deriva-py && git log --oneline origin/deriva-ml -3` should show the add_asset-resolve commit(s).

- [ ] **Step 2: Read the current pin (both locations must match each other)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -nE "deriva-py@|rev = " pyproject.toml | grep -i deriva`
Note the current SHA (`93add80...`) in both `project.dependencies` (the `deriva @ git+...@<sha>` URL, ~L19) and `[tool.uv.sources]` (`rev = "<sha>"`, ~L65).

- [ ] **Step 3: Replace the SHA in BOTH locations**

Edit `/Users/carl/GitHub/DerivaML/deriva-ml/pyproject.toml`:
- `project.dependencies`: `"deriva @ git+https://github.com/informatics-isi-edu/deriva-py@<NEWSHA>"`
- `[tool.uv.sources]`: `deriva = { git = "https://github.com/informatics-isi-edu/deriva-py", rev = "<NEWSHA>" }`
Both must be exactly `<NEWSHA>`. Verify they match: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -nE "deriva-py@|rev = " pyproject.toml | grep -i deriva` — both lines show `<NEWSHA>`.

- [ ] **Step 4: Re-lock and sync**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv lock 2>&1 | tail -4 && uv sync 2>&1 | tail -6`
Expected: `uv.lock` updates to the new deriva-py commit; `uv sync` installs it. Confirm the installed deriva-py is the new SHA: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -c "import deriva, subprocess" 2>&1; grep -A1 "name = \"deriva\"" uv.lock | grep -i "rev\|commit" | head` (or check the lock entry references `<NEWSHA>`).

- [ ] **Step 5: Confirm the fix is present in the installed deriva-py**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "resolve()" .venv/lib/python3.13/site-packages/deriva/bag/builder.py | head`
Expected: the `source_path = Path(source_path).resolve()` line is present in the installed `add_asset`. If absent, the sync didn't pick up the fix — re-check the SHA and re-run `uv sync`.

- [ ] **Step 6: Commit the pin advance**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add pyproject.toml uv.lock
git commit -m "chore(deps): advance deriva-py pin to add_asset symlink-resolve fix

Picks up the deriva-py fix where BagBuilder.add_asset resolves symlink sources
before link=True, fixing Linux bag corruption (os.link hardlinking the symlink
inode) and bagit 'unsafe path' rejection on symlinked staging assets."
```

---

### Task 5: Verify against the pinned fix + broad sweep + PR

**Files:** none (verification + PR)

- [ ] **Step 1: Re-confirm the full deriva-py bag suite on Linux against the FIXED code**

The Task 2 Step 4 Docker run already proved the new test green on Linux. Re-confirm the WHOLE bag-builder suite is green on Linux (catches any sibling regression the resolve introduced):
Run: `cd /Users/carl/GitHub/deriva-py && docker run --rm -v "$PWD":/work -w /work python:3.12-slim bash -c "pip -q install uv 2>/dev/null; uv run --frozen pytest tests/deriva/bag/test_builder.py -q 2>&1 | tail -5"`
Expected: full bag-builder suite green on Linux. No throwaway script needed — the committed test IS the acceptance.

- [ ] **Step 2: Confirm the deriva-ml asset/commit flow against the new pin (localhost)**

Run the existing asset + execution suites — they exercise `asset_file_path` (default `copy_file=False` → symlink staging) → `commit_output_assets` → bag end to end against the newly-pinned deriva-py:
Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/ -q 2>&1 | tail -8`
Then (if not too long): `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q 2>&1 | tail -8`
Expected: green. These run on macOS (where the bug was already invisible), so their purpose here is confirming the **pin advance didn't break the asset/commit flow** — not re-proving the Linux fix (Step 1 does that). Report the pass/skip/fail counts.

- [ ] **Step 3: Broad deriva-ml regression sweep**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/model/ tests/asset/ -q 2>&1 | tail -8`
Expected: green. Confirms the pin advance is clean across the unit surface.

- [ ] **Step 4: Push the pin-advance branch and open the deriva-ml PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git push -u origin fix/bag-add-asset-symlink-resolve 2>&1 | tail -3
gh pr create --title "fix(deps): advance deriva-py pin — add_asset resolves symlink sources (Linux bag fix)" \
  --body "<summary: the Linux os.link-on-symlink root cause, the deriva-py fix (PR link), the pin advance, and the verification (Docker-Linux test green + deriva-ml asset/execution suites green). Link the spec docs/superpowers/specs/2026-06-16-bag-add-asset-symlink-resolve-design.md.>" 2>&1 | tail -3
```
The spec + this plan are already committed on the branch (from brainstorming/planning); the PR bundles them with the pin advance.

- [ ] **Step 5: Release (after the deriva-ml PR merges to main)**

This step is the controller's, run on `main` after merge: `cd /Users/carl/GitHub/DerivaML/deriva-ml && git checkout main && git pull --ff-only && uv run bump-version patch`. The pin advance IS the release content. (Level: patch — a consumer-visible bug fix with no API change. Controller confirms the level.)

---

## Self-Review notes

- **Spec coverage:** the deriva-py fix → Task 2; the upstream contract test → Task 1 (RED) + Task 2 (GREEN), in the same PR (Task 3); the Linux Docker proof (mandatory, no upstream CI) → Task 1 Step 4 + Task 2 Step 4; idempotency/duplicate-source safety → Task 2 Step 5 (named tests must stay green); pin advance in both locations → Task 4; verification → Task 5; sequencing with the upstream-merge pause → Task 3 Step 2b gate before Task 4; release → Task 5 Step 5. Out-of-scope items (no asset_file_path default change, no restructure/cache change, no deriva-ml code, no bridge) are respected — no task touches them.
- **TDD subtlety handled:** the RED is macOS-invisible, so the plan demonstrates RED and GREEN on Linux via Docker explicitly, and treats a macOS pass as expected (not a wrong test).
- **No placeholders:** verification uses concrete commands (the committed test on Linux via Docker + the existing asset/execution suites) rather than bespoke throwaway scripts. The PR bodies are described, not templated with fake content, because the real PR/spec links aren't known until those steps run.
- **Cross-repo:** deriva-py commands are pinned to `/Users/carl/GitHub/deriva-py`; deriva-ml to `/Users/carl/GitHub/DerivaML/deriva-ml`. The upstream PR is opened but human-merged (Task 3 Step 2b gate).

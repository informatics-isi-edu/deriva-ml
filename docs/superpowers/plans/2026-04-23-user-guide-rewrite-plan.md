# User-Guide Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deriva-ml's fragmented Getting Started + Concepts + Running Workflows sections with a coherent 8-chapter User Guide, following the technical-documentation pattern of a narrative middle layer between Introduction and API Reference.

**Architecture:** Content-preserving rewrite. Walk every cut-list page, extract unique content into an inventory keyed by destination chapter, draft each chapter from the inventory, then perform a coordinated nav rewrite + deletions + link audit. Stub-redirect pages preserve bookmarked URLs for one release cycle.

**Tech Stack:** Markdown (mkdocs-material), mkdocs-jupyter for notebooks, `mkdocs build --strict` for link validation, git for version control. No code changes to src/.

**Spec:** `docs/superpowers/specs/2026-04-23-user-guide-rewrite-design.md`

**Non-goals (out of scope for this plan):**
- Docstring content improvements (sub-project 2, separate spec + plan).
- Any change to `src/deriva_ml/`.
- Any change to `docs/configuration/`, `docs/Notebooks/`, `docs/release-notes.md`, `docs/architecture.md`.
- New library features.
- Template-repo changes.

**Verification contract:**
- Every task that modifies `docs/` must end with `uv run mkdocs build --strict` passing.
- Every chapter's commit message cites the migration-map entries (source pages) it absorbs, so the content-preservation invariant is auditable.
- Final read-through (Task 15) is the acceptance gate.

**Progress style:** Each chapter is a single commit. 8 chapters + 5 supporting tasks = 13 commits minimum. Final read-through may add fix-up commits.

---

## File Structure

### New files to create

```
docs/user-guide/exploring.md
docs/user-guide/datasets.md
docs/user-guide/features.md
docs/user-guide/executions.md
docs/user-guide/offline.md
docs/user-guide/sharing.md
docs/user-guide/reproducibility.md
docs/user-guide/hydra-zen.md
docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md   # migration map
```

### Files to rewrite in place

- `docs/index.md` — Introduction: positioning + four-object model + template-repo pointer.

### Files to rename (content unchanged)

```
docs/code-docs/dataset.md                    → docs/api-reference/dataset.md
docs/code-docs/dataset_aux_classes.md        → docs/api-reference/dataset_aux_classes.md
docs/code-docs/dataset_bag.md                → docs/api-reference/dataset_bag.md
docs/code-docs/dataset_split.md              → docs/api-reference/dataset_split.md
docs/code-docs/deriva_definitions.md         → docs/api-reference/deriva_definitions.md
docs/code-docs/deriva_ml_base.md             → docs/api-reference/deriva_ml_base.md
docs/code-docs/deriva_model.md               → docs/api-reference/deriva_model.md
docs/code-docs/exceptions.md                 → docs/api-reference/exceptions.md
docs/code-docs/execution.md                  → docs/api-reference/execution.md
docs/code-docs/execution_configuration.md    → docs/api-reference/execution_configuration.md
docs/code-docs/feature.md                    → docs/api-reference/feature.md
docs/code-docs/upload.md                     → docs/api-reference/upload.md
docs/code-docs/workflow.md                   → docs/api-reference/workflow.md
```

### Files to delete (replaced by stub redirects in one-release deprecation cycle)

```
docs/getting-started/quick-start.md
docs/getting-started/install.md
docs/getting-started/project-setup.md
docs/concepts/overview.md
docs/concepts/identifiers.md
docs/concepts/datasets.md
docs/concepts/features.md
docs/concepts/file-assets.md
docs/concepts/annotations.md
docs/concepts/denormalization.md
docs/workflows/execution-lifecycle.md
docs/workflows/running-models.md
docs/workflows/git-and-versioning.md
docs/cli-reference.md
```

### File modified

- `mkdocs.yml` — nav section rewritten (one edit in Task 11).

---

## Task 1: Build the content inventory (migration map)

**Why:** Before drafting a single chapter we need to walk every cut-list page and catalogue what's there. This prevents content loss and makes each chapter's source material explicit.

**Files:**
- Create: `docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md`

- [ ] **Step 1: Read every cut-list page end to end**

Fourteen files. Open each in sequence and read start to finish (don't skim headings):

```
docs/getting-started/quick-start.md
docs/getting-started/install.md
docs/getting-started/project-setup.md
docs/concepts/overview.md
docs/concepts/identifiers.md
docs/concepts/datasets.md
docs/concepts/features.md
docs/concepts/file-assets.md
docs/concepts/annotations.md
docs/concepts/denormalization.md
docs/workflows/execution-lifecycle.md
docs/workflows/running-models.md
docs/workflows/git-and-versioning.md
docs/cli-reference.md
```

For each, note: all code examples, all diagrams (reference the asset path), all "Note/Warning/Tip" callouts, all edge-case paragraphs, all cross-references to other pages.

- [ ] **Step 2: Write the inventory doc**

Create `docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md` with this structure:

```markdown
# User-Guide Content Inventory

**Purpose:** migration map from cut-list pages to User Guide chapters.
Every source sentence has a destination (or a retirement rationale).
Used as the source-of-truth during chapter drafting.

## Destination: Introduction (docs/index.md rewrite)

### From docs/concepts/overview.md
- Four-object mental model (Catalog / Dataset / Execution / Feature)
- ERD image reference (docs/assets/ERD.png)
- Paragraph on domain schema vs ML schema

### From docs/getting-started/* (all three pages)
- RETIRED: replaced by single pointer to template repo. No content re-homed.

## Destination: Chapter 1 — Exploring a catalog (docs/user-guide/exploring.md)

### From docs/concepts/overview.md
- <bullet per piece of content>

### From docs/concepts/identifiers.md
- <bullet per piece of content>

### From docs/cli-reference.md
- <only the read/browse CLI commands, if any>

## Destination: Chapter 2 — Working with datasets (docs/user-guide/datasets.md)

<same shape>

[... through Chapter 8 ...]

## Retirements (content deliberately dropped with rationale)

### From docs/concepts/annotations.md
- <list the content and justify why it doesn't belong in any User Guide chapter>

### From docs/getting-started/install.md
- All content: covered by template repo.
```

Each destination section is a bulleted list of content items. Each item is short (one line per item). Retirements MUST include a rationale — "template repo owns it" or "obsolete in S2" or "belongs in API Reference."

- [ ] **Step 3: Verify every source sentence has a destination**

Spot-check: pick 3 source pages at random. For each, confirm every paragraph is represented somewhere in the inventory (destination OR retirement). Add missing entries.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md
git commit -m "docs(plan): user-guide content inventory — migration map for rewrite"
```

---

## Task 2: Chapter 1 — Exploring a catalog

**Files:**
- Create: `docs/user-guide/exploring.md`

- [ ] **Step 1: Read the inventory entries for Chapter 1**

Open `docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md` and read the Chapter 1 destination section. These are your raw materials.

- [ ] **Step 2: Write the chapter using the fixed shape**

Create `docs/user-guide/exploring.md`. Follow the spec's §Writing approach structure exactly:

1. **Opening paragraph** — one sentence stating what the chapter covers, one sentence listing what the reader will know at the end.

2. **Concept setup** (1-2 paragraphs) — a DerivaML instance is a Python handle on a remote Deriva catalog. Everything in the catalog has a Resource Identifier (RID). The rest of the chapter shows how to list and browse catalog contents.

3. **Task sections** in this order:
   - `## Connecting to a catalog` — one sentence on how, one code block, pointer to template repo for project setup.
   - `## Understanding RIDs` — opaque-ID story. Why they're random-looking strings, that they're stable, that they show up everywhere.
   - `## Listing tables and browsing the schema` — `ml.model` and related introspection.
   - `## Finding datasets, features, workflows, executions` — the four `find_*` methods with one example each.
   - `## Querying with pathBuilder` — when to reach for `ml.pathBuilder()` directly.
   - `## When to reach for pathBuilder vs. the high-level APIs` — comparison subsection.
   - `## Jumping to Chaise` — `get_chaise_url()` for GUI browsing.

4. **Common pitfalls** (if warranted — RID opaqueness is a likely candidate).

5. **See also** — one-line cross-references to Chapter 2 (datasets deep dive), Chapter 4 (executions), API Reference entries.

Use `python` language tag on all code blocks. All examples short (5-15 lines). Reuse the exact imports at the top of the first example (`from deriva_ml import DerivaML`, etc.).

- [ ] **Step 3: Verify the build**

Run:

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

Expected: SUCCESS with zero warnings. If broken links or bad syntax, fix.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/exploring.md
git commit -m "docs(user-guide): Chapter 1 — Exploring a catalog

Re-homes content from:
- docs/concepts/overview.md (four-object context, RIDs)
- docs/concepts/identifiers.md (all)
- docs/cli-reference.md (browse-related commands)

See docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md."
```

---

## Task 3: Chapter 2 — Working with datasets

**Files:**
- Create: `docs/user-guide/datasets.md`

- [ ] **Step 1: Read the inventory entries for Chapter 2**

Open the content inventory. The Chapter 2 destination will be the largest section — `docs/concepts/datasets.md` is 770 lines and is the primary source. Also note content from `docs/workflows/execution-lifecycle.md` about dataset versioning.

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/datasets.md` following the fixed shape:

1. **Opening paragraph** — one sentence, one sentence.

2. **Concept setup** — a Dataset is a versioned, named collection of RIDs. It doesn't copy data; it points at existing rows. Versions back onto catalog snapshots, so a named version always resolves the same rows.

3. **Task sections:**
   - `## Creating a dataset` — `ml.create_dataset(...)`. Dataset types. Adding element types.
   - `## Adding members to a dataset` — list-of-RIDs form AND dict form; when to use each.
   - `## Designing dataset types` — orthogonal tagging; when a new type is warranted.
   - `## Parent and child datasets` — hierarchy for train/val/test splits.
   - `## Versioning` — `current_version`, `increment_dataset_version`, semantic versioning, catalog snapshots. This is the reproducibility primitive.
   - `## Splitting a dataset` — `split_dataset()` with stratification, seed, dry-run.
   - `## Downloading as a bag` — `DatasetSpec`, `DatasetBag`, `materialize=True/False` flag. When to use each.

4. **Common pitfalls:**
   - Dataset types aren't mutually exclusive (orthogonal tagging).
   - `stratify_by_column` expects `TableName_ColumnName` format (opaque; document the format explicitly).
   - Version numbers don't auto-bump on member changes; you must call `increment_dataset_version`.

5. **See also** — Chapter 3 (features on dataset members), Chapter 5 (offline work with bags), Chapter 7 (reproducibility via version pinning).

Preserve every code example from `concepts/datasets.md` — adapt wording to the task-oriented structure but keep the code itself intact unless there's a clear reason to rewrite.

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

Expected: SUCCESS.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/datasets.md
git commit -m "docs(user-guide): Chapter 2 — Working with datasets

Re-homes content from:
- docs/concepts/datasets.md (primary — 770 lines)
- docs/workflows/execution-lifecycle.md (dataset versioning interplay)"
```

---

## Task 4: Chapter 3 — Defining and using features

**Files:**
- Create: `docs/user-guide/features.md`

- [ ] **Step 1: Read the inventory entries for Chapter 3**

Primary source: `docs/concepts/features.md` (460 lines). Also the new S2 spec (`docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md`) is reference material for API shape. Also `docs/concepts/denormalization.md` for the "when to use `Denormalizer` vs `feature_values`" section.

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/features.md` following the shape:

1. **Opening paragraph.**

2. **Concept setup** — a Feature is an association table linking a target table + execution + values. It's how you attach structured, provenance-linked annotations to existing data. Features differ from regular columns in that multiple executions can record different values for the same target row (multi-annotator scenarios).

3. **Task sections:**
   - `## When to use a feature vs. a column` — decision framework.
   - `## Creating a vocabulary` — prerequisite for term features.
   - `## Creating a feature` — `create_feature` with terms, assets, metadata columns.
   - `## Recording feature values` — `exe.add_features(records)` within an execution context.
   - `## Reading feature values` — three-method surface: `find_features`, `feature_values`, `lookup_feature`.
   - `## Using selectors to reduce multi-annotator data` — `select_newest`, `select_first`, `select_by_execution`, `select_majority_vote`, `select_by_workflow`. One code block per selector.
   - `## Asset features` — crop boxes, embeddings, segmentation masks. What makes them special.
   - `## Reading features from a downloaded bag` — API parity on `DatasetBag`.
   - `## When to reach for feature_values vs. Denormalizer` — comparison subsection (the spec calls this out explicitly).

4. **Common pitfalls:**
   - Workflow deduplication by checksum: re-running from the same script gives you the same workflow RID (document this since it's surprising).
   - Features that have asset columns need the asset upload to land before the feature flush — the library handles this; document the invariant for users debugging failures.
   - `feature_values` materializes all rows into memory before yielding (not true streaming — we documented this in the S2 docstring too, repeat it here).

5. **See also** — Chapter 4 (executions as the write context), Chapter 5 (offline feature reads).

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/features.md
git commit -m "docs(user-guide): Chapter 3 — Defining and using features

Re-homes content from:
- docs/concepts/features.md (primary)
- docs/concepts/denormalization.md (Denormalizer vs feature_values comparison)

Incorporates S2 API surface (feature_values iterator, select_by_workflow
factory, lookup_feature)."
```

---

## Task 5: Chapter 4 — Running an experiment

**Files:**
- Create: `docs/user-guide/executions.md`

- [ ] **Step 1: Read the inventory entries for Chapter 4**

Primary sources: `docs/workflows/execution-lifecycle.md`, `docs/workflows/running-models.md`, `docs/cli-reference.md`.

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/executions.md`:

1. **Opening paragraph.**

2. **Concept setup** — an Execution is a tracked run of a Workflow at a point in time. It captures inputs, outputs, status, and the environment the code ran in. Everything you produce during the run — asset files, feature values, metadata — is linked to the execution for provenance.

3. **Task sections:**
   - `## Describing an execution with ExecutionConfiguration` — datasets, assets, description. Show creating from scratch AND from a hydra-zen-produced config.
   - `## Running an execution` — the `with ml.create_execution(cfg) as exe:` context manager pattern. Status transitions (Created → Running → Stopped).
   - `## Writing asset files` — `exe.asset_file_path(asset_table, filename, asset_types=...)` and why you must NOT manually write to `working_dir / "Execution_Metadata"`.
   - `## Writing feature values` — `exe.add_features(records)`. Staged to SQLite, flushed on upload.
   - `## Uploading outputs` — `execution.upload_execution_outputs()`. Ordering: assets first, features second. `__exit__` does NOT auto-upload; you must call explicitly.
   - `## Execution status lifecycle` — diagram of the state machine (Running → Stopped → Pending_Upload → Uploaded, plus Failed branches).
   - `## Crash-resume` — what happens if the process dies mid-run. Staged rows persist.
   - `## CLI reference` — sub-section covering `deriva-ml-run`, `deriva-ml-run-notebook`, their flags. Short; the template repo has the full onramp.

4. **Common pitfalls:**
   - `__exit__` does NOT auto-upload.
   - Workflow dedup by checksum (again — this surfaces for users creating executions too).
   - Dirty-tree check (`DERIVA_ML_ALLOW_DIRTY` is an override, not the default).

5. **See also** — Chapter 3 (features as execution outputs), Chapter 7 (reproducibility).

Add a state-machine diagram. Use Mermaid syntax inside a ```mermaid fenced block — mkdocs-material renders these.

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/executions.md
git commit -m "docs(user-guide): Chapter 4 — Running an experiment

Re-homes content from:
- docs/workflows/execution-lifecycle.md (primary)
- docs/workflows/running-models.md (CLI subsection)
- docs/cli-reference.md (CLI subsection)

Incorporates S2 changes: __exit__ does not auto-upload (verified during
Task 7 review), feature flush order (assets then features), crash-resume
via execution_state__feature_records."
```

---

## Task 6: Chapter 5 — Working offline

**Files:**
- Create: `docs/user-guide/offline.md`

- [ ] **Step 1: Read the inventory entries for Chapter 5**

Primary sources: `docs/concepts/datasets.md` bag sections, S2 design doc's offline-to-online section.

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/offline.md`:

1. **Opening paragraph.**

2. **Concept setup** — a bag is a downloaded, self-contained, immutable snapshot of a dataset including all referenced asset files. Everything you do online via `Dataset` works on a bag via `DatasetBag`, with a small exception list. Bags let you develop on a laptop and sync results back when reconnected.

3. **Task sections:**
   - `## Downloading a dataset as a bag` — `DatasetSpec`, `download_dataset_bag`, `materialize=True/False`.
   - `## Reading from a bag` — `bag.find_features`, `bag.feature_values`, `bag.lookup_feature`, `bag.list_dataset_members`. API parity with live `Dataset`/`DerivaML`.
   - `## What bags can't do` — can't write; workflow-based queries that need the live catalog are limited (some cases work from bag SQLite).
   - `## Constructing feature records offline` — `bag.lookup_feature(...)` returns a `Feature` whose `feature_record_class()` works without a catalog connection. Build records from model predictions, keep them in memory.
   - `## Committing offline-built records when back online` — hand the records to `exe.add_features(records)` via a fresh execution. The execution provides provenance.
   - `## Restructuring assets for ML frameworks` — `bag.restructure_assets()` produces the `class/image.jpg` directory layout that `torchvision.datasets.ImageFolder` and `tf.keras.utils.image_dataset_from_directory` expect.

4. **Common pitfalls:**
   - `restructure_assets()` returns a `dict[Path, Path]`, not a `Path` (CLAUDE.md flag).
   - Bags are immutable: if the source dataset changes, re-download the bag.

5. **See also** — Chapter 2 (creating datasets), Chapter 3 (features API).

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/offline.md
git commit -m "docs(user-guide): Chapter 5 — Working offline

Re-homes content from:
- docs/concepts/datasets.md (bag sections)

Incorporates S2 offline-to-online write cycle (tested in
tests/feature/test_feature_values.py::test_offline_construct_records_online_stage)."
```

---

## Task 7: Chapter 6 — Sharing and collaboration

**Files:**
- Create: `docs/user-guide/sharing.md`

**Note:** This chapter is mostly new content. Reviewer #5 flagged `create_ml_workspace` as undocumented; sources will include reading `src/deriva_ml/catalog/clone.py` and `src/deriva_ml/catalog/localize.py` directly to understand behavior.

- [ ] **Step 1: Read the inventory entries for Chapter 6**

Primary sources: `docs/concepts/file-assets.md` partial coverage (bag-related bits), plus the CLAUDE.md section on catalog cloning, plus direct source read of:

- `src/deriva_ml/catalog/clone.py` — `create_ml_workspace` function and orphan-handling strategies.
- `src/deriva_ml/catalog/localize.py` — `localize_assets` function.

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/sharing.md`:

1. **Opening paragraph.**

2. **Concept setup** — three ways to share data: citable bag URLs (MINIDs), downloadable bag archives (BDBag format), and partial catalog clones (for multi-site collaboration). Each has different portability vs. authority tradeoffs.

3. **Task sections:**
   - `## Sharing a dataset with a MINID` — `use_minid=True` when downloading. What a MINID guarantees (persistent global identifier). How collaborators resolve it.
   - `## Sharing a bag archive` — the `.bag.zip` format, what's in it, and how to inspect without deriva-ml installed.
   - `## Cloning a subset of a catalog` — `create_ml_workspace`. Three-stage approach: schema without FKs, async data copy, FK application with orphan handling.
   - `## Orphan-handling strategies` — `FAIL`, `DELETE`, `NULLIFY`. When each is right.
   - `## Copying assets after a clone` — `localize_assets` for `asset_mode=REFERENCES` clones.
   - `## Access control` — brief, link to Deriva/Globus docs. We don't re-document that.

4. **Common pitfalls:**
   - `create_ml_schema` drops the existing schema with CASCADE (CLAUDE.md flag); never call on a populated catalog.
   - BDBag archives have their own spec; collaborators who want to inspect without deriva-ml need the `bdbag` CLI tool.

5. **See also** — Chapter 2 (datasets), Chapter 5 (bag-based workflows).

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/sharing.md
git commit -m "docs(user-guide): Chapter 6 — Sharing and collaboration

Re-homes content from:
- docs/concepts/file-assets.md (partial, bag-related)
- CLAUDE.md (catalog cloning section)
- source reading of src/deriva_ml/catalog/clone.py + localize.py

Documents create_ml_workspace and localize_assets, which were flagged as
undocumented by reviewer #5 in the S2 post-audit."
```

---

## Task 8: Chapter 7 — Reproducibility

**Files:**
- Create: `docs/user-guide/reproducibility.md`

- [ ] **Step 1: Read the inventory entries for Chapter 7**

Primary sources: `docs/workflows/git-and-versioning.md`, dataset-versioning sections from `docs/concepts/datasets.md`, Execution snapshot work (from `src/deriva_ml/execution/execution_snapshot.py` if it exists).

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/reproducibility.md`:

1. **Opening paragraph.**

2. **Concept setup** — reproducibility in deriva-ml means: given an execution RID, you can reconstruct exactly what code ran, on what data, in what environment. The library captures these automatically, but you need to cooperate (commit before running, pin versions, use a clean tree).

3. **Task sections:**
   - `## What is captured automatically` — workflow checksum (git commit), environment snapshot (`uv.lock`), Hydra config, Docker image digest.
   - `## Pinning a dataset version` — `DatasetSpec(rid, version="1.0.0")`; catalog snapshot semantics.
   - `## Workflow checksums and git commits` — what gets hashed; why workflow dedup works.
   - `## Docker image digest capture` — for containerized runs; the `DERIVA_MCP_IMAGE_DIGEST` env var.
   - `## Dirty-tree handling` — the warning, `DERIVA_ML_ALLOW_DIRTY` override, when to use each.
   - `## Re-running a past execution` — finding it via `find_executions`, reading its config back, constructing a new `ExecutionConfiguration` with the same inputs.

4. **Common pitfalls:**
   - `DERIVA_ML_ALLOW_DIRTY=true` silently pollutes provenance. Only use in tests.
   - Workflow dedup by checksum means repeat runs from the same script share one workflow RID; use that property intentionally.

5. **See also** — Chapter 2 (dataset versioning), Chapter 4 (execution lifecycle).

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/reproducibility.md
git commit -m "docs(user-guide): Chapter 7 — Reproducibility

Re-homes content from:
- docs/workflows/git-and-versioning.md (primary)
- docs/concepts/datasets.md (version pinning sections)"
```

---

## Task 9: Chapter 8 — Integrating with hydra-zen

**Files:**
- Create: `docs/user-guide/hydra-zen.md`

- [ ] **Step 1: Read the inventory entries for Chapter 8**

Primary sources: `docs/configuration/overview.md` (NOT rewritten — this section stays; Chapter 8 is the shallow intro to it). Also CLAUDE.md §Hydra-zen Configuration for the four config classes.

- [ ] **Step 2: Write the chapter**

Create `docs/user-guide/hydra-zen.md`:

1. **Opening paragraph.**

2. **Concept setup** — hydra-zen is a layer on top of hydra that lets you define configs as Python classes instead of YAML. deriva-ml uses it so all ML-experiment configuration is type-checked Python, versionable with the code, with CLI overrides for free.

3. **Task sections:**
   - `## When to reach for hydra-zen vs. straight Python` — decision framework. Simple one-off notebook work: straight Python. Reproducible project-structured runs: hydra-zen.
   - `## The four config classes` — `DerivaMLConfig`, `DatasetSpecConfig`, `AssetRIDConfig`, `ExecutionConfiguration`. One-line summary each + link to Configuration section.
   - `## Composing configs through the CLI` — how the `deriva-ml-run` CLI consumes configs; config groups; multi-run.
   - `## Project structure conventions` — pointer to template repo for the structure; don't re-document.

4. **See also** — Configuration section (deep dive), template repo (project setup), Chapter 4 (running an experiment).

Keep this chapter short (1500 words ceiling). It's deliberately shallow; the Configuration section is the authoritative deep dive.

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/hydra-zen.md
git commit -m "docs(user-guide): Chapter 8 — Integrating with hydra-zen

Shallow orientation chapter; the Configuration section remains the
deep-dive reference. Re-homes conceptual content from
docs/workflows/running-models.md and CLAUDE.md §Hydra-zen."
```

---

## Task 10: Rewrite Introduction (docs/index.md)

**Files:**
- Modify: `docs/index.md`

- [ ] **Step 1: Read the inventory entries for Introduction**

Inventory destination: Introduction. Primary source: `docs/concepts/overview.md` (four-object model), positioning content from the S2 post-audit (Reviewer #5's "positioning vs competition" analysis — see `docs/superpowers/specs/2026-04-23-post-s2-findings.md`).

- [ ] **Step 2: Rewrite index.md**

Replace the current `docs/index.md` (22 lines) with:

```markdown
# DerivaML

DerivaML is a Python library for reproducible machine learning workflows
backed by a Deriva catalog. It captures code provenance, input data
versions, configuration, and outputs so experiments can be reproduced,
cited, and shared.

## What deriva-ml does

Four core concepts organize the library:

- **Catalog** — the schema + data store. An ERMrest-backed Deriva catalog
  with domain tables (Subject, Image, Observation, etc.) and an ML schema
  (Dataset, Execution, Workflow, Feature_Name).
- **Dataset** — a versioned, named collection of RIDs. Backs onto catalog
  snapshots so a named version always resolves the same rows.
- **Execution** — a tracked run of a Workflow. Captures inputs, outputs,
  environment, and status with full provenance.
- **Feature** — structured, provenance-linked annotations on existing rows.
  The unit of record for labels, predictions, and derived metadata.

![ERD](assets/ERD.png)

## When to use deriva-ml

Strong fit:

- Research labs where data governance, audit trails, and multi-annotator
  ground truth are first-class requirements.
- Multi-site collaborations that need citable dataset identifiers and
  reproducible execution records.
- Biomedical imaging, clinical records, or similar domains with structured
  schemas and vocabulary-controlled annotations.

Weaker fit:

- Quick single-dataset experiments where a folder of files and git is
  enough. deriva-ml has a non-trivial setup cost; don't pay it without the
  governance need.
- Online feature-serving for low-latency inference. deriva-ml is
  research-oriented; Feast / Tecton are better for that.

See also: _Deriva-ML: A Continuous FAIRness Approach to Reproducible
Machine Learning Models_ (Li et al., 2024, IEEE e-Science).

## Starting a new project

To start a new deriva-ml project, use the
[deriva-ml-model-template repository](https://github.com/informatics-isi-edu/deriva-ml-model-template).
It provides:

- Hydra-zen configuration scaffolding
- CLI entry points (`deriva-ml-run`, `deriva-ml-run-notebook`)
- GitHub Actions for versioning and documentation deployment
- An example model (CIFAR-10) with config variants

These docs cover the deriva-ml library itself, for developers who already
have a project and want to understand the library's concepts and APIs.
Start with the [User Guide](user-guide/exploring.md) for a task-oriented
walkthrough, or jump to the [API Reference](api-reference/deriva_ml_base.md)
for per-method documentation.

## Further reading

The underlying FAIR-data principles are described in:

> Dempsey, William, Ian Foster, Scott Fraser, and Carl Kesselman.
> "Sharing begins at home: how continuous and ubiquitous FAIRness can
> enhance research productivity and data reuse."
> _Harvard Data Science Review_ 4, no. 3 (2022).
> [PDF](assets/sharing-at-home.pdf)

The deriva-ml architecture and design decisions are described in:

> Li, Zhiwei, Carl Kesselman, Mike D'Arcy, Michael Pazzani, and
> Benjamin Yizing Xu. "Deriva-ML: A Continuous FAIRness Approach to
> Reproducible Machine Learning Models." In _2024 IEEE 20th International
> Conference on e-Science (e-Science)_, pp. 1-10. IEEE, 2024.
> [PDF](assets/deriva-ml.pdf)
```

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

Expected: WARNINGS about broken links to `user-guide/exploring.md` and `api-reference/deriva_ml_base.md` are expected at this point — those paths don't exist yet at the final location (user-guide files exist, but api-reference is still `code-docs/` until Task 11). If the build fails outright due to a missing reference, fix by using relative paths from `docs/index.md` in a way that resolves against the current state of the tree.

Alternative: if `--strict` fails on this task, temporarily drop to `mkdocs build` (non-strict) for this one task; Task 12's link audit will catch any real broken links after the nav rewrite.

- [ ] **Step 4: Commit**

```bash
git add docs/index.md
git commit -m "docs(user-guide): rewrite Introduction with positioning and template-repo pointer

Replaces 22-line index.md with positioning-forward intro that names the
four-object mental model (Catalog/Dataset/Execution/Feature), states the
fit-for-purpose decision, and points new-project readers to the template
repo. Library-users see the User Guide and API Reference paths."
```

---

## Task 11: Rename code-docs to api-reference

**Files:**
- Rename: `docs/code-docs/*.md` → `docs/api-reference/*.md` (13 files)
- Modify: `mkdocs.yml` (nav section)

- [ ] **Step 1: Move the directory**

```bash
git mv docs/code-docs docs/api-reference
```

Expected: 13 files renamed. Verify:

```bash
ls docs/api-reference/
```

Expected: 13 `.md` files (dataset.md, dataset_aux_classes.md, dataset_bag.md, dataset_split.md, deriva_definitions.md, deriva_ml_base.md, deriva_model.md, exceptions.md, execution.md, execution_configuration.md, feature.md, upload.md, workflow.md).

- [ ] **Step 2: Update mkdocs.yml nav entry**

In `mkdocs.yml`, find the `Library Documentation:` section and rename to `API Reference:`, updating all child paths from `code-docs/` to `api-reference/`. Do NOT rewrite the rest of the nav yet — that's Task 12.

Example of the exact edit (before → after):

```yaml
# BEFORE:
- Library Documentation:
    - Core Classes:
        - DerivaML: code-docs/deriva_ml_base.md
        - DerivaModel: code-docs/deriva_model.md
    - Datasets:
        - Dataset: code-docs/dataset.md
        ...

# AFTER:
- API Reference:
    - Core Classes:
        - DerivaML: api-reference/deriva_ml_base.md
        - DerivaModel: api-reference/deriva_model.md
    - Datasets:
        - Dataset: api-reference/dataset.md
        ...
```

- [ ] **Step 3: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

Expected: SUCCESS. If there are warnings about broken internal links to `code-docs/` paths, they'll come from the cut-list pages that still exist — those get deleted in Task 12. For now, if `--strict` fails on this task due to those references, the build is caught up with reality after Task 12.

- [ ] **Step 4: Commit**

```bash
git add docs/api-reference/ docs/code-docs/ mkdocs.yml
git commit -m "docs(user-guide): rename code-docs/ to api-reference/

Matches the conventional docs terminology used elsewhere in the industry.
No content changes; only directory rename and nav update. Cut-list page
deletions (Task 12) will clean up any remaining code-docs/ references."
```

---

## Task 12: Nav rewrite + cut-list deletions + stub redirects

**Files:**
- Modify: `mkdocs.yml` (full nav rewrite)
- Delete: 14 cut-list pages (replaced by redirect stubs)
- Create: 14 stub redirect files at same paths

- [ ] **Step 1: Rewrite the mkdocs.yml nav**

In `mkdocs.yml`, replace the entire `nav:` section with:

```yaml
nav:
  - Introduction: index.md
  - User Guide:
      - Exploring a catalog: user-guide/exploring.md
      - Working with datasets: user-guide/datasets.md
      - Defining and using features: user-guide/features.md
      - Running an experiment: user-guide/executions.md
      - Working offline: user-guide/offline.md
      - Sharing and collaboration: user-guide/sharing.md
      - Reproducibility: user-guide/reproducibility.md
      - Integrating with hydra-zen: user-guide/hydra-zen.md
  - Configuration:
      - Overview: configuration/overview.md
      - Configuration Groups: configuration/groups.md
      - Experiments and Multiruns: configuration/experiments.md
      - Notebook Configuration: configuration/notebooks.md
  - Sample Notebooks:
      - Datasets: 'Notebooks/DerivaML Dataset.ipynb'
      - Execution: 'Notebooks/DerivaML Execution.ipynb'
      - Features: 'Notebooks/DerivaML Features.ipynb'
      - Vocabulary: 'Notebooks/DerivaML Vocabulary.ipynb'
  - API Reference:
      - Core Classes:
          - DerivaML: api-reference/deriva_ml_base.md
          - DerivaModel: api-reference/deriva_model.md
      - Datasets:
          - Dataset: api-reference/dataset.md
          - DatasetBag: api-reference/dataset_bag.md
          - Dataset Splitting: api-reference/dataset_split.md
          - Dataset Auxiliary Classes: api-reference/dataset_aux_classes.md
      - Execution:
          - Execution: api-reference/execution.md
          - ExecutionConfiguration: api-reference/execution_configuration.md
          - Workflow: api-reference/workflow.md
      - Features:
          - Feature: api-reference/feature.md
      - Utilities:
          - Definitions & Types: api-reference/deriva_definitions.md
          - Exceptions: api-reference/exceptions.md
          - Upload: api-reference/upload.md
  - Release Notes: release-notes.md
```

Note: `Getting Started`, `Concepts`, `Running Workflows`, and `CLI Reference` entries are all removed from the nav.

- [ ] **Step 2: Delete cut-list pages and replace with redirect stubs**

For each of the 14 cut-list pages, replace its content with a `meta refresh` HTML stub. This preserves bookmarked URLs for one release cycle.

The stub template (with `<new-url>` substituted per file):

```html
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="refresh" content="0; url=<new-url>">
<link rel="canonical" href="<new-url>">
<title>Page moved</title>
</head>
<body>
<p>This page has moved. If you are not redirected automatically, follow
<a href="<new-url>">this link to the new location</a>.</p>
</body>
</html>
```

The new-URL mapping per cut-list page:

```
docs/getting-started/quick-start.md           → https://github.com/informatics-isi-edu/deriva-ml-model-template
docs/getting-started/install.md               → https://github.com/informatics-isi-edu/deriva-ml-model-template
docs/getting-started/project-setup.md         → https://github.com/informatics-isi-edu/deriva-ml-model-template
docs/concepts/overview.md                     → ../index.md
docs/concepts/identifiers.md                  → ../user-guide/exploring.md
docs/concepts/datasets.md                     → ../user-guide/datasets.md
docs/concepts/features.md                     → ../user-guide/features.md
docs/concepts/file-assets.md                  → ../user-guide/executions.md
docs/concepts/annotations.md                  → ../user-guide/features.md
docs/concepts/denormalization.md              → ../user-guide/features.md
docs/workflows/execution-lifecycle.md         → ../user-guide/executions.md
docs/workflows/running-models.md              → ../user-guide/hydra-zen.md
docs/workflows/git-and-versioning.md          → ../user-guide/reproducibility.md
docs/cli-reference.md                         → user-guide/executions.md
```

Do this for all 14 files. Example for `docs/concepts/datasets.md`:

```html
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="refresh" content="0; url=../user-guide/datasets.md">
<link rel="canonical" href="../user-guide/datasets.md">
<title>Page moved</title>
</head>
<body>
<p>This page has moved. If you are not redirected automatically, follow
<a href="../user-guide/datasets.md">this link to the new location</a>.</p>
</body>
</html>
```

**Important:** These stubs are `.md` files but contain HTML content. mkdocs-material renders raw HTML inside markdown files.

- [ ] **Step 3: Link audit**

Grep the repo for every internal link to a deleted-or-moved page and fix it:

```bash
grep -rn "getting-started/\|concepts/\|workflows/\|cli-reference\|code-docs/" docs/ 2>&1 | grep -v stub | grep -v "superpowers/" | head -50
```

Fix every hit. Three categories:

1. **Links in User Guide chapters** — should use `user-guide/*.md` paths directly, not the old paths. Fix by rewriting.
2. **Links in API Reference** — already corrected by Task 11's rename.
3. **Links in Sample Notebooks** — may exist in notebook markdown cells; check each notebook.
4. **Links in Release Notes** — historical references are OK; a Release Notes entry saying "Added `concepts/features.md`" in v1.5.0 should stay historical.

Skip `docs/superpowers/` (design/plan docs have their own histories and aren't user-facing).

- [ ] **Step 4: Verify the build**

```bash
uv run mkdocs build --strict 2>&1 | tail -30
```

Expected: SUCCESS with zero warnings. If broken links remain, fix them and re-run.

- [ ] **Step 5: Commit**

```bash
git add mkdocs.yml docs/
git commit -m "docs(user-guide): rewrite nav, delete cut-list pages, add redirect stubs

Nav restructured to the three-layer architecture: Introduction / User
Guide / Configuration / Sample Notebooks / API Reference / Release Notes.
Getting Started, Concepts, Running Workflows, CLI Reference sections
removed; their content is re-homed into the 8 User Guide chapters
(migration audit in docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md).

14 cut-list pages replaced with meta-refresh redirect stubs to preserve
bookmarked URLs for one release cycle. Stubs can be removed in the
next minor release."
```

---

## Task 13: Full build verification

**Files:** none modified; verification only.

- [ ] **Step 1: Strict build succeeds with zero warnings**

```bash
uv run mkdocs build --strict 2>&1 | tail -30
```

Expected: `INFO    -  Documentation built in X.XX seconds` with no `WARNING` lines. If warnings remain, fix them.

- [ ] **Step 2: Verify all 8 chapter files exist and are non-empty**

```bash
for f in exploring datasets features executions offline sharing reproducibility hydra-zen; do
    path="docs/user-guide/$f.md"
    if [ -s "$path" ]; then echo "OK: $path"; else echo "MISSING: $path"; fi
done
```

Expected: 8 "OK" lines.

- [ ] **Step 3: Verify all 14 stubs exist and redirect**

```bash
for f in \
    docs/getting-started/quick-start.md \
    docs/getting-started/install.md \
    docs/getting-started/project-setup.md \
    docs/concepts/overview.md \
    docs/concepts/identifiers.md \
    docs/concepts/datasets.md \
    docs/concepts/features.md \
    docs/concepts/file-assets.md \
    docs/concepts/annotations.md \
    docs/concepts/denormalization.md \
    docs/workflows/execution-lifecycle.md \
    docs/workflows/running-models.md \
    docs/workflows/git-and-versioning.md \
    docs/cli-reference.md; do
    if grep -q "meta http-equiv=\"refresh\"" "$f" 2>/dev/null; then
        echo "STUB OK: $f"
    else
        echo "STUB MISSING: $f"
    fi
done
```

Expected: 14 "STUB OK" lines.

- [ ] **Step 4: Verify api-reference exists, code-docs doesn't**

```bash
ls docs/api-reference/ | wc -l
ls docs/code-docs/ 2>&1 | head -3
```

Expected: `api-reference/` has 13 files; `code-docs/` doesn't exist.

- [ ] **Step 5: Serve the site locally and spot-check**

```bash
uv run mkdocs serve 2>&1 | tail -10 &
sleep 5
curl -s http://localhost:8000/ | grep -qi "DerivaML" && echo "HOME OK"
curl -s http://localhost:8000/user-guide/exploring/ | grep -qi "Exploring" && echo "CH1 OK"
curl -s http://localhost:8000/api-reference/deriva_ml_base/ | grep -qi "DerivaML" && echo "API OK"
kill %1 2>/dev/null
```

Expected: 3 OK lines.

- [ ] **Step 6: Commit (if fixes applied during verification)**

If any steps 1-5 surfaced issues that required changes, commit them:

```bash
git add docs/ mkdocs.yml
git commit -m "docs(user-guide): build verification fixes"
```

If no fixes needed, skip this step.

---

## Task 14: Code-example verification (selective)

**Files:** none modified; verification only. This task ensures the code examples in chapters 2, 3, 4, and 5 actually work against a live catalog.

- [ ] **Step 1: Identify verified-tier examples**

For chapters 2 (datasets), 3 (features), 4 (executions), 5 (offline), pick the 3 most important examples per chapter. These are the ones that will actually execute against a live catalog. Other examples remain illustrative.

A "most important" example is one that demonstrates the primary use case (e.g., "create a feature + add values + read them back" for Chapter 3). Mark these in each chapter with an HTML comment at the top of the code block:

```markdown
<!-- verified: tests/docs/test_chapter3_example1.py -->
```python
# ... example code ...
```
```

- [ ] **Step 2: Create executable test files for verified examples**

For each verified example, create a file in `tests/docs/` that runs it against a `test_ml` fixture:

```
tests/docs/
├── __init__.py
├── conftest.py              # reuse test_ml fixture from tests/conftest.py
├── test_chapter2_dataset_example.py
├── test_chapter3_feature_example.py
├── test_chapter4_execution_example.py
└── test_chapter5_offline_example.py
```

Each test file extracts the example code from the chapter and runs it with minimal scaffolding:

```python
"""Verifies Chapter 3 example 1 from docs/user-guide/features.md."""
from deriva_ml.feature import FeatureRecord


def test_chapter3_example_reads_feature_values(test_ml_with_feature):
    """The example shows reading feature values with select_newest selector."""
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
        selector=FeatureRecord.select_newest,
    ))
    assert records, "Example should yield at least one record"
```

Use existing fixtures where possible (e.g., `test_ml_with_feature` from `tests/feature/conftest.py`).

- [ ] **Step 3: Run the doc tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/docs/ -v 2>&1 | tail -20
```

Expected: all tests pass. If a test fails, the chapter's example is broken — fix the example in the chapter (not the test) and re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/docs/ docs/user-guide/
git commit -m "docs(user-guide): add live-catalog verification for primary examples

Each verified example is marked in the chapter with an HTML comment
pointing to its test file. Illustrative examples remain unverified."
```

---

## Task 15: Cover-to-cover read-through

**Files:** may modify any chapter; verification + polish task.

- [ ] **Step 1: Read the full user guide in one sitting**

Starting with `docs/index.md`, then chapters 1 through 8 in order, then a spot-check of the Configuration section and the redirect stubs. Budget 45-60 minutes; this is the most valuable verification step. Read fully, not skimming.

Look for:

1. **Internal inconsistencies** — one chapter defines a term (e.g., "bag"), another redefines it differently.
2. **Repeated material** — each concept should have one primary home. Repetition across chapters suggests a chapter owns something it shouldn't, or a See-also link is missing.
3. **Missing transitions** — end of Chapter N doesn't set up Chapter N+1; reader has no sense of forward momentum.
4. **Forward references** — Chapter N refers to a concept introduced in Chapter N+2. Fix by either rewording the reference, or introducing the concept earlier.
5. **Tone drift** — one chapter is terse, another is chatty. The spec's §Writing approach sets the tone; realign.
6. **Broken links** — all internal links work. `mkdocs build --strict` caught these at build time, but links may still point at the wrong anchor (same file, wrong heading).

- [ ] **Step 2: Fix issues inline**

For each issue found, edit the chapter directly. Don't batch all edits into one commit; commit per issue category (e.g., "fix forward references", "align tone in Chapters 2-3").

- [ ] **Step 3: Re-verify build after each fix**

```bash
uv run mkdocs build --strict 2>&1 | tail -5
```

Expected: SUCCESS after each commit.

- [ ] **Step 4: Final commit**

After all read-through fixes are in:

```bash
git add -A
git commit -m "docs(user-guide): read-through polish

Cross-chapter consistency, tone alignment, and transition cleanup
after a full cover-to-cover read. No new content; readability fixes only."
```

---

## Self-review checklist (run after writing this plan)

### 1. Spec coverage

Walk the spec's §Chapter outlines. Every chapter in the spec has a task:

- ✓ Chapter 1 → Task 2
- ✓ Chapter 2 → Task 3
- ✓ Chapter 3 → Task 4
- ✓ Chapter 4 → Task 5
- ✓ Chapter 5 → Task 6
- ✓ Chapter 6 → Task 7
- ✓ Chapter 7 → Task 8
- ✓ Chapter 8 → Task 9
- ✓ Introduction → Task 10
- ✓ Nav rewrite → Task 12
- ✓ Cut-list deletions → Task 12
- ✓ API-reference rename → Task 11
- ✓ Content inventory → Task 1
- ✓ Migration plan phases (0-5) → Tasks 1, 2-9, 10, 11, 12, 13-15

All phases from the spec are mapped to tasks.

### 2. Placeholder scan

No "TBD", "TODO", "fill in" remain. Each task has concrete content — source materials named, commit message templates provided, verification commands given.

### 3. Type / name consistency

- Chapter filenames consistent throughout: `exploring.md`, `datasets.md`, `features.md`, `executions.md`, `offline.md`, `sharing.md`, `reproducibility.md`, `hydra-zen.md`.
- Directory names consistent: `docs/user-guide/`, `docs/api-reference/` (not `user_guide`, not `api_reference`).
- Commit-message prefix consistent: `docs(user-guide):` for User Guide work, `docs(plan):` for inventory.
- Stub-redirect format consistent across all 14 cut-list pages.

### 4. Scope check

- Plan covers one cohesive subsystem (user-guide rewrite).
- No scope creep into docstring work (sub-project 2) or code changes.
- Estimated effort (5-7 days) matches the spec's estimate.

### 5. Execution readiness

- All file paths are absolute and concrete.
- All commands are runnable as-written.
- Verification commands have expected output documented.
- TDD shape adapted to the docs-project context: `mkdocs build --strict` is the test; per-chapter builds are the incremental checkpoint; the full read-through is the acceptance test.

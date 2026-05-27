# User-Guide Rewrite — Design

**Status:** Draft
**Date:** 2026-04-23
**Scope:** Sub-project 1 of the two-part documentation pass. Rewrites the deriva-ml user-facing documentation as a conventional technical user guide. Sub-project 2 (docstring sweep) is a separate brainstorm → spec → plan cycle.

## Problem

The current `deriva-ml` documentation has three user-facing sections (`Getting Started`, `Concepts`, `Running Workflows`) that overlap, collectively mis-orient readers, and duplicate content that the `deriva-ml-model-template` repo already owns. Reviewer #5 (the ML-developer lifecycle reviewer in the S2 post-audit) flagged the onramp story specifically: the current quick-start leads with `deriva-ml-run` and hydra-zen configuration before the reader knows what a Dataset or Execution is, which doubles time-to-first-success.

With the template repo now responsible for zero-to-first-success (install, project scaffolding, CLI usage, hydra-zen), `deriva-ml`'s own docs are free to stop trying to be an onramp and become what they should be: a **user guide** — the narrative layer between a conceptual white-paper and the API reference, task-oriented and readable cover-to-cover.

## Goals

- **One authoritative narrative** for what the library does and how to use it. Resolve the overlap between today's `Concepts` and `Running Workflows` sections.
- **Task-oriented chapters** — organized by what the user wants to do (explore, build a dataset, record features, run an experiment), not by API surface.
- **No duplication with the template repo.** Onramp / install / project scaffolding / CLI is the template's concern; link to it instead of repeating it.
- **Complete content preservation.** Every piece of information in the existing deriva-ml docs either gets re-homed into the user guide or is explicitly retired with rationale.
- **Backward-compatible URLs for one release cycle** via HTML `meta refresh` stubs on deleted pages.

## Non-goals

- **Docstring sweep.** Sub-project 2, separate spec/plan. Different audience (future Claude + MCP), different shape (per-method), different review loop.
- **New library features.** If a chapter example would be cleaner with a code change, file a follow-on issue; don't write around it, don't change the library here.
- **API-reference content improvement.** Renaming `docs/code-docs/ → docs/api-reference/` is mechanical; substantive improvement comes from the docstring sweep.
- **Hydra-zen Configuration section rewrite.** Stays as the deep-dive reference; the user-guide chapter 8 is only a shallow introduction that points to it.
- **Template-repo docs.** The template's README and docs are the template's concern.
- **Positioning / branding changes to the project as a whole** beyond what the Introduction page requires.

## Architecture

### Three-layer information model

- **Conceptual layer (Introduction):** one page. Positions deriva-ml in the ML tooling landscape, explains the problem it solves, names the four core objects (Catalog, Dataset, Execution, Feature), and points new readers to the template repo for onramp.
- **Narrative layer (User Guide):** eight chapters. Task-oriented. The spine of the docs — meant to be read cover-to-cover or sampled by task.
- **Reference layer (API Reference):** auto-generated from docstrings, renamed from today's "Library Documentation." Content improvement is deferred to sub-project 2.

### New top-level nav

```
1. Introduction                 (was: index.md + Getting Started section)
2. User Guide                   (NEW — 8 chapters replacing Concepts + Workflows)
3. Configuration                (unchanged — hydra-zen deep dive)
4. Sample Notebooks             (unchanged)
5. API Reference                (renamed from Library Documentation)
6. Release Notes                (unchanged)
```

### Content-preservation invariant

Every sentence currently in `docs/` outside the untouched sections either:
(a) is re-homed into a User Guide chapter with the same or improved clarity,
(b) is folded into the Introduction,
(c) is explicitly marked obsolete in this spec's §Cut list with a rationale.

No information loss. A migration map (see §Migration plan) is built before any rewriting starts.

## Components

### New files to create

```
docs/
├── user-guide/
│   ├── exploring.md            # Chapter 1: Exploring a catalog
│   ├── datasets.md             # Chapter 2: Working with datasets
│   ├── features.md             # Chapter 3: Defining and using features
│   ├── executions.md           # Chapter 4: Running an experiment
│   ├── offline.md              # Chapter 5: Working offline
│   ├── sharing.md              # Chapter 6: Sharing and collaboration
│   ├── reproducibility.md      # Chapter 7: Reproducibility
│   └── hydra-zen.md            # Chapter 8: Integrating with hydra-zen
```

### Files to rewrite

- `docs/index.md` — Introduction, rewritten with positioning and a pointer to the template repo.

### Files to rename (content unchanged)

- `docs/code-docs/*.md` → `docs/api-reference/*.md` — eleven files; no content changes; `mkdocs.yml` nav updated to match.

### Files to delete (content re-homed)

Twelve pages. See §Cut list for the full table.

### Files unchanged

- `docs/configuration/*.md` (4 pages)
- `docs/Notebooks/*.ipynb` (4 notebooks)
- `docs/release-notes.md`
- `docs/architecture.md` (developer-facing, not user-facing)
- `docs/assets/*` (images, PDFs)

### Configuration change

- `mkdocs.yml` — nav section rewritten to match the new three-layer structure.

## Chapter outlines

Each chapter: 1500-3000 words, task-oriented. Every chapter follows the same internal shape (see §Writing approach).

### Chapter 1: Exploring a catalog

**Goal:** the reader knows how to discover what's in a DerivaML instance they've just connected to.

**Sections:**
- Connecting (`DerivaML(hostname, catalog_id)`) with a pointer to the template repo for project setup
- Understanding RIDs (the opaque-ID story + how they show up everywhere)
- Listing tables and browsing the schema
- `find_datasets`, `find_features`, `find_workflows`, `find_executions`
- `ml.pathBuilder()` for ad-hoc queries — when to reach for it vs. the high-level APIs (the "when to reach for X vs Y" pattern)
- Jumping to Chaise (`get_chaise_url`) for GUI browsing

**Source material:** current `concepts/overview.md`, `concepts/identifiers.md`, parts of `cli-reference.md`.

### Chapter 2: Working with datasets

**Goal:** the reader can create, populate, version, and split datasets.

**Sections:**
- What a Dataset is: a versioned, named collection of RIDs; it doesn't copy data, it points at existing rows
- Creating a dataset; adding members (list-of-RIDs form and dict form)
- Dataset types — choosing and designing them (orthogonal tagging)
- Parent/child relationships, nested datasets
- Version management: `current_version`, `increment_dataset_version`, catalog snapshot semantics
- Splitting: `split_dataset` with stratification, dry-run, seed control
- Downloading as a bag; `DatasetSpec` vs `DatasetBag`

**Source material:** current `concepts/datasets.md` (770 lines — major rewrite, preserve all examples), chunks of `workflows/execution-lifecycle.md` on versioning.

### Chapter 3: Defining and using features

**Goal:** the reader can add structured, provenance-linked annotations to their data.

**Sections:**
- What a Feature is (association table with target + execution + values)
- When to use a feature vs. a regular column
- Creating a vocabulary (prerequisite for term features)
- `create_feature` — terms, assets, metadata columns
- Three-method read surface: `find_features`, `feature_values`, `lookup_feature`
- Selectors: `select_newest`, `select_by_workflow`, `select_majority_vote`, custom selectors
- Multi-annotator scenarios (the canonical use case)
- Asset features (crop boxes, embeddings)
- Reading features from a downloaded bag (same API, offline)

**Source material:** current `concepts/features.md` (460 lines — rewrite, preserve worked examples), S2 spec, the new docstrings we wrote during S2.

### Chapter 4: Running an experiment

**Goal:** the reader can execute a training or analysis run with full provenance.

**Sections:**
- The Execution mental model (a workflow run instance)
- `ExecutionConfiguration`: datasets, assets, description
- The `with ml.create_execution(cfg)` context manager pattern
- Status lifecycle: Created → Running → Stopped → Pending_Upload → Uploaded (+ failure paths)
- Writing outputs: `exe.asset_file_path()` for files, `exe.add_features()` for records
- `upload_execution_outputs()` — what it does and the ordering guarantees (assets first, features second, asset-column RID rewriting between)
- Crash-resume and what the SQLite execution state does for you
- CLI reference as a subsection (`deriva-ml-run`, `deriva-ml-run-notebook`)

**Source material:** `workflows/execution-lifecycle.md`, `workflows/running-models.md`, `cli-reference.md`, S2 execution changes.

### Chapter 5: Working offline

**Goal:** the reader can do work on a laptop with a downloaded bag and sync back when online.

**Sections:**
- Bag vs live catalog — the mental model
- Downloading a bag; materializing assets
- `DatasetBag` API parity with `Dataset` — what works, what doesn't, and why
- The offline-to-online write cycle: construct `FeatureRecord` from bag metadata, hand to `exe.add_features()` when reconnected
- `restructure_assets()` for framework-friendly directory layouts (ImageFolder pattern)

**Source material:** `concepts/datasets.md` bag sections, S2 offline-cycle tests, new S2 docstrings.

### Chapter 6: Sharing and collaboration

**Goal:** the reader can share data with collaborators — whether they have catalog access or not.

**Sections:**
- MINIDs as citable identifiers — what they guarantee
- BDBag format — enough to understand what you're sharing
- `create_ml_workspace` — partial catalog clones for multi-site collaboration (currently undocumented; reviewer #5 flagged this as a major gap)
- `localize_assets` — copying Hatrac assets after a clone
- Orphan-handling strategies during cloning (FAIL / DELETE / NULLIFY)
- Access control — brief (delegated to Deriva/Globus; link to their docs)

**Source material:** `concepts/file-assets.md` (partial), `CLAUDE.md` section on catalog cloning, net new from source-code reading where needed.

### Chapter 7: Reproducibility

**Goal:** the reader understands what reproducibility deriva-ml guarantees, what it costs, and how to exercise it.

**Sections:**
- What reproducibility costs and buys
- Dataset version pinning via catalog snapshots
- Workflow checksums and git commit capture
- Docker image digest capture for containerized runs
- `configuration.json` and execution metadata — what's captured automatically
- Dirty-tree handling — warnings vs. hard errors, `DERIVA_ML_ALLOW_DIRTY`
- Patterns for re-running past executions (discovering a past execution, reading its config back, launching a new execution with the same inputs)

**Source material:** `workflows/git-and-versioning.md`, dataset-versioning sections of `concepts/datasets.md`, execution-snapshot work.

### Chapter 8: Integrating with hydra-zen

**Goal:** the reader understands how hydra-zen fits with deriva-ml and can choose when to use it.

**Sections:**
- What hydra-zen is and why deriva-ml uses it
- The four config classes (`DerivaMLConfig`, `DatasetSpecConfig`, `AssetRIDConfig`, `ExecutionConfiguration`)
- How configs thread through the `deriva-ml-run` CLI
- When to reach for hydra-zen vs. straight Python (the comparison pattern)
- Link to the Configuration section for the deep dive
- Link to the template repo for project-setup patterns

**Source material:** current `configuration/overview.md` (the Configuration section is preserved; chapter 8 is a shallow intro), CLAUDE.md hydra-zen sections, summarize-and-link approach.

## Writing approach

### Voice and audience

Written for a working ML developer who has a project scaffolded from the template and wants to understand the library they're using. Assume Python fluency, ML vocabulary (train/val/test, labels, features, assets), and basic familiarity with relational schemas. Do not assume prior Deriva knowledge — introduce Deriva-specific concepts (RIDs, catalogs, ERMrest, Hatrac, Chaise) on first use with a sentence of explanation each.

Tone: direct, technical, second person ("you"). No hype, no "easy and powerful," no emoji. Short paragraphs, short sentences. Diagrams only when they clarify something text can't.

### Chapter structure

Every chapter has the same shape:

1. **Opening paragraph** — what the chapter is for, what the reader will know at the end. No preamble about the topic's importance.
2. **Concept setup** (1-2 paragraphs) — the mental model the rest of the chapter builds on.
3. **Task sections** — each is a `## How to <verb> <noun>` heading with: one-paragraph motivation, runnable code example, explanation of what the code did, bulleted "Notes" list of gotchas.
4. **"When to reach for X vs Y" subsections** — comparison subsections where two APIs are legitimately both valid (e.g., `Denormalizer` vs `feature_values`, `ml.pathBuilder()` vs high-level APIs).
5. **Common pitfalls** (optional) — things that will actually trip people up. One per chapter at most.
6. **See also** — one-line cross-references to related chapters and API reference anchors.

### Runnable examples

Every code block either (a) runs against any DerivaML instance with no setup, or (b) has a prominent `# assumes: dataset_rid is a valid dataset in your catalog` comment at the top. Examples are short (5-15 lines) and self-contained. Long worked examples go into `docs/Notebooks/`.

Code style matches what users would actually write: imports at the top of each chapter's first example, then assumed available in subsequent examples within the same chapter.

### Formatting conventions

- Markdown with mkdocs-material Admonition syntax: `!!! note`, `!!! warning`, `!!! tip`. Used for pitfalls and important caveats only, not decoration.
- Code blocks use language-tagged fences: `python` by default, `bash` for shell, `json` for catalog payloads.
- Diagrams: PNG or Mermaid. Keep the existing ERD on the Introduction page. Add execution-state-machine diagram in Chapter 4. Add dataset-version → catalog-snapshot flow in Chapter 2 if prose is unclear. No more than 4-5 diagrams total across the guide.

### Cross-references

Inside the user guide: relative paths (`../user-guide/features.md#selectors`).
To API reference: `../api-reference/deriva_ml_base.md#DerivaML.feature_values`.
External links: full URLs.

## Data flow

### Building the migration map

Before writing any chapter:

1. Walk the 12 pages in the cut list and extract every code example, diagram, edge-case note, and "gotcha" box. Store keyed by destination chapter.
2. For each chapter, assemble the source material into an outline following the §Chapter outlines structure.
3. Flag any content that doesn't fit the new structure. Three outcomes: (a) belongs in a different chapter, (b) redundant with other content, (c) obsolete.
4. Produce a content inventory document and commit it alongside the chapter drafts. Reviewers can verify nothing was lost.

### Drafting a chapter

1. Open the migration map entry for the chapter.
2. Draft the opening paragraph + concept setup.
3. For each task section: write the motivation → write or adapt the code example → verify it runs (if verification-tier; see §Testing) → write the explanation → list gotchas.
4. Add "When to reach for X vs Y" subsections where the current code has two valid paths.
5. Add common-pitfalls admonition if warranted.
6. Add See-also cross-references.
7. Run `mkdocs build --strict` locally; fix broken links.
8. Commit the chapter as a single commit.

### Nav rewrite and deletions

After all 8 chapters are drafted and committed:

1. Rewrite `mkdocs.yml` nav.
2. Rename `docs/code-docs/` to `docs/api-reference/` in one commit; update nav.
3. Delete the 12 cut-list pages; replace each with a stub `meta refresh` redirect to its new home. Stub stays for one release cycle.
4. Run a full link audit (`grep -rn "concepts/\|workflows/\|getting-started/" docs/`) and fix every intra-doc reference.
5. Commit the nav change and deletions together.

## Error handling

This is a documentation project, not a code project, so "errors" are shape-of-the-content problems:

- **Content loss** — a sentence from a cut page isn't captured in the migration map. Mitigation: the migration map is a required pre-chapter artifact and is reviewable independently.
- **Link rot** — internal links pointing at deleted pages. Mitigation: `mkdocs build --strict` rejects broken intra-doc links; external link-checker catches external ones; stub-redirect pages cover user-facing bookmark URLs for one release.
- **Example drift** — code examples that stop working as the library evolves. Mitigation: each chapter declares which examples were executed against a live catalog. Out-of-date examples surface the next time someone reads the chapter.
- **Stale cross-references to API names** — API Reference anchors that break when docstrings get restructured in sub-project 2. Mitigation: sub-project 2's spec includes a link-audit step that updates user-guide cross-references.
- **Inconsistent voice / pattern across chapters** — one chapter has worked examples, another has only API sketches. Mitigation: the §Writing approach structure is mandatory per chapter; the final read-through (§Testing) catches drift.

## Testing

Five verification stages, ordered by cost:

### Tier 1 — per-chapter mechanical (blocking per commit)

- `uv run mkdocs build --strict` succeeds with zero warnings.
- All `!!! admonition` blocks use valid types.
- All code blocks have language tags.

### Tier 2 — cross-chapter integration (blocking before nav rewrite)

- Link audit: `grep -rn "(concepts|workflows|getting-started)/" docs/` after deletions returns nothing.
- `mkdocs-linkcheck` (or equivalent) passes — no broken external links.
- All 8 chapter files exist; all 12 cut-list files are either deleted or stub-redirects.

### Tier 3 — code verification (selective, blocking per chapter)

For the most important examples per chapter (3-5 per chapter), actually execute the code against a live catalog (`test_ml` fixture or equivalent) and confirm it works. Document which examples were verified in each chapter's migration-map entry. Pure illustrative examples (e.g., "assuming you had this data…") are marked as illustrative.

### Tier 4 — cover-to-cover read-through (blocking before PR merge)

Read the full user guide in one sitting. Catch:

- Internal inconsistencies (e.g., one chapter defines a term, another redefines it differently)
- Repeated material across chapters (each concept should have one primary home)
- Missing transitions between chapters
- "Forward references" — chapter N referring to a concept introduced in chapter N+2

### Tier 5 — external reader (non-blocking but recommended)

Ask a developer not on the project to read chapters 1-4 and report confusion points. Single-session, no pre-briefing. Their confusion reveals what the intended audience would stumble on.

## Cut list (migration plan)

| Source page | Disposition | Destination |
|---|---|---|
| `docs/getting-started/quick-start.md` | Delete | Pointer in Introduction to the template repo |
| `docs/getting-started/install.md` | Delete | Pointer in Introduction to the template repo |
| `docs/getting-started/project-setup.md` | Delete | Pointer in Introduction to the template repo |
| `docs/concepts/overview.md` | Delete | Content merged into Introduction (positioning + four-object model) and Chapter 1 (catalog exploration) |
| `docs/concepts/identifiers.md` | Delete | Merged into Chapter 1 (the "Understanding RIDs" section) |
| `docs/concepts/datasets.md` | Rewrite | Becomes Chapter 2; preserve all code examples |
| `docs/concepts/features.md` | Rewrite | Becomes Chapter 3; preserve worked examples |
| `docs/concepts/file-assets.md` | Delete | Split across Chapter 4 (assets as execution outputs) and Chapter 6 (assets in shared bags) |
| `docs/concepts/annotations.md` | Delete | Merged into Chapter 3 (feature-column annotations) if the content is user-facing; retire to developer docs if not |
| `docs/concepts/denormalization.md` | Delete | Content about `Denormalizer` merged into Chapter 3 ("When to reach for X vs Y" subsection) |
| `docs/workflows/execution-lifecycle.md` | Rewrite | Becomes Chapter 4 core |
| `docs/workflows/running-models.md` | Delete | Split across Chapter 8 (hydra-zen integration) and Chapter 4 (CLI reference subsection) |
| `docs/workflows/git-and-versioning.md` | Rewrite | Becomes Chapter 7 core |
| `docs/cli-reference.md` | Delete | Folded into Chapter 4 as a "CLI reference" subsection |

Stubs (`meta refresh` redirects) stay for one release cycle for all deleted pages. Rewrite-destination files replace their source files in place (same filename where applicable, like `concepts/datasets.md → user-guide/datasets.md`).

## Migration plan

Work is ordered so each phase is independently reviewable:

### Phase 0 — inventory and map

1. Walk every cut-list page; extract code examples, diagrams, and edge-case notes into a structured inventory (keyed by destination chapter).
2. Draft the inventory doc as `docs/superpowers/specs/2026-04-23-user-guide-content-inventory.md`.
3. Commit the inventory and this spec together.

### Phase 1 — chapter drafts (one commit per chapter)

1. **Chapter 1: Exploring a catalog** — simplest, establishes voice and pattern. First to draft.
2. **Chapter 2: Working with datasets** — largest source material; big rewrite.
3. **Chapter 3: Defining and using features** — S2 surface is well-understood.
4. **Chapter 4: Running an experiment** — depends on chapters 2 and 3 (references datasets and features).
5. **Chapter 5: Working offline** — depends on chapters 2, 3, 4.
6. **Chapter 6: Sharing and collaboration** — mostly new content (reviewer #5 flagged as undocumented).
7. **Chapter 7: Reproducibility** — cross-cuts; written after 2-4 so it can reference their content.
8. **Chapter 8: Integrating with hydra-zen** — last; just orients to the existing Configuration section.

Each chapter commit: chapter content + updated `mkdocs.yml` nav entry for that chapter.

### Phase 2 — Introduction rewrite

Rewrite `docs/index.md` with positioning, four-object mental model, template-repo pointer. One commit.

### Phase 3 — API reference rename

Rename `docs/code-docs/ → docs/api-reference/`. Update `mkdocs.yml`. No content changes. One commit.

### Phase 4 — nav rewrite and deletions

1. Rewrite `mkdocs.yml` top-level nav to match the §New top-level nav.
2. Delete the 12 cut-list pages; replace with `meta refresh` stubs.
3. Run the full link audit and fix every internal reference.
4. One commit.

### Phase 5 — read-through + external reader

1. Cover-to-cover read of all 8 chapters + Introduction.
2. Fix drift, missing transitions, forward references.
3. (Optional) external reader; fix their confusion points.
4. Final commit, then PR.

## Open dependencies

- Sub-project 2 (docstring sweep) is the companion. User-guide cross-references to API Reference anchors depend on docstring stability. Solution: write the user guide first with coarse anchor references (`api-reference/feature.md#FeatureRecord`), then update anchors as part of the docstring-sweep work.
- Template repo must clearly own the onramp story for the Introduction's pointer to be honest. Current state (verified: `/Users/carl/GitHub/deriva-ml-model-template`) is sufficient.

## Scale estimate

- **Content volume:** 15,000–25,000 words of new and rewritten content across 8 chapters + Introduction.
- **Effort:** roughly half a day per chapter to draft well. Full sub-project: 5–7 working days if done straight through, longer with review cycles.
- **Review load:** each chapter commit is a natural review unit; the final read-through is the biggest review checkpoint.

# FK Traversal / Bag Export / Denormalization Reference Manual — Design

**Date:** 2026-06-13
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml`

## 1. Problem statement

deriva-ml's FK-traversal, bag-export, and denormalization behavior is
load-bearing and subtle (the 2026-06-13 provenance-hub explosion bug —
PR #297 — turned on one missing policy field). The exact behavior is
documented nowhere as a reference: it's spread across docstrings, three
ADRs (0002/0006/0008), an archived bag-cutover design, and a
tutorial-style `user-guide/denormalization.md`. There is no document a
person (or the assistant) can consult to answer "exactly what does the
walk include / exclude, and why?" with code-grounded precision.

This spec defines a **reference manual** — formal, example-rich, and
readable by both humans and the LLM — for these operations.

## 2. Goals

- One self-contained, end-to-end reference for: **FK traversal**, **bag
  export**, **denormalization**, plus brief "same engine, different
  consumer" entries for **estimate_bag_size**, **clone_via_bag**, and
  **dataset drift (`is_dirty`)**.
- **Formal:** numbered rules stating exact behavior, each tagged with
  the **enforcing code location** and the **layer** (`[engine: deriva-py]`
  vs `[deriva-ml]`).
- **Example-rich:** worked examples on the demo catalog's small schema,
  every example **run against the live demo catalog at authoring time**
  with real output pasted in.
- **LLM-readable:** stable rule IDs, a "Quick answers" question→rule
  index per doc, explicit code paths the assistant can cite.

## 3. Scope decisions (from brainstorming)

- **Repo boundary:** document the full pipeline *in deriva-ml*, citing
  deriva-py as the authority for engine rules (every engine rule tagged
  `[engine: deriva-py]` with the upstream file/class).
- **Structure:** per-topic three-layer — (1) Mental model, (2) Formal
  rules (numbered, with code location), (3) Worked examples.
- **Example fidelity:** verify every example against the local demo
  catalog while authoring; each doc header notes "verified against demo
  catalog, deriva-ml v1.46.x".
- **Denorm overlap:** keep the existing tutorial
  `user-guide/denormalization.md`; the new `reference/denormalization.md`
  is the formal counterpart, cross-linked both ways.
- **Breadth:** core three + one-paragraph same-engine entries for
  estimate/clone/drift; NOT cache/MINID/download-tiers (covered
  elsewhere — `offline.md`, `manage-deriva-storage`).

## 3a. Discoverability requirement (MCP RAG + Claude Code)

The docs must be discoverable by both the deriva-ml MCP server's RAG
search and a Claude Code session. **Verified:** the
`deriva-ml-mcp-plugin` RAG indexer
(`resources/rag.py:register_rag_sources`) crawls the
`informatics-isi-edu/deriva-ml` repo at `main` with `path_prefix=""`
and `glob="**/*.md"` (deriva-mcp-core `GitHubCrawler`), tagging every
`.md` as `doc_type="ml-docs"`. Therefore:

- Placing the three docs under `docs/reference/` (inside the repo) is
  sufficient — they are automatically indexed as `ml-docs`, searchable
  via `rag_search(query=..., doc_type="ml-docs")`, with **no special
  registration**. (The crawler has no exclude-paths filter; everything
  under the repo root that ends in `.md` is indexed.)
- **Index timing:** the RAG source tracks `main`, so the docs become
  MCP-searchable only **after the PR merges to `main`** (and the next
  index pass runs) — not from the feature branch. State this in the PR.
- **Claude Code:** the docs are on disk in the repo, so a Claude Code
  session reads them directly. To make the assistant *aware* they exist,
  add a one-line pointer in the repo-root `deriva-ml/CLAUDE.md` (e.g.
  under a "Reference docs" note) directing it to
  `docs/reference/{fk-traversal,bag-export,denormalization}.md` for
  exact traversal/export/denorm behavior. This is the one CLAUDE.md edit
  in scope.

## 4. Deliverables

Three new files under `docs/reference/` (joining the existing
`schema.md`), plus mkdocs nav wiring and cross-links.

### 4.1 `docs/reference/fk-traversal.md` — the engine

The foundation the other two reference. Covers `FKTraversalPolicy` and
the walker.

**Mental model:** the pipeline is **two phases** — *walk* (the
`PathWalker` discovers which tables/rows are reachable from the anchor
set by following FKs) then *load* (the loader materializes rows and
resolves orphans/conflicts). This split is why some policy fields
affect *what is reached* and others affect *how reached rows are
handled* — a distinction the manual must state up front
(`path_walker.py` docstring: "behavior-on-walk" vs "behavior-on-output"
fields).

**Formal rules** (derived by reading code, numbered `T1, T2, …`), at
minimum:
- Anchors: the walk starts from an anchor set (`RIDAnchor` = specific
  RIDs; `TableAnchor` = whole table). `[deriva-ml]` supplies them.
- Bidirectional FKs: the walker follows **both** outbound
  (`table.foreign_keys`) and inbound (`table.referenced_by`) FKs.
  `[engine: deriva-py path_walker.PathWalker]`.
- Terminal tables: entered but neither outbound nor inbound FKs
  followed. The Execution/Workflow provenance-hub rule (PR #297) is
  documented here with its rationale and the
  `core/constants.py:PROVENANCE_TERMINAL_TABLES` link. `[engine]` for
  the mechanism, `[deriva-ml]` for the chosen set.
- Vocabularies: always treated as leaves (entered, not traversed out) —
  the universal walker guard, independent of `terminal_tables`.
- `exclude_schemas` / `exclude_tables` / `schemas` allow-list: pruning
  before the walk.
- `max_depth`: FK-hop ceiling from the anchor (default unbounded).
- The "behavior-on-output" fields (`vocab_export`, `asset_mode`,
  `dangling_fk_strategy`, `content_on_conflict`, `match_by_columns`):
  named here, each one-line, with a pointer to where the *loader* (not
  walker) applies it.

**Worked examples:** on the demo schema (Subject/Image/Dataset +
features + executions), show: a RID-anchored walk's reached-table set;
the effect of marking Execution terminal (reached set with vs without —
the PR #297 behavior, real numbers from the demo catalog); a
`max_depth` truncation.

### 4.2 `docs/reference/bag-export.md` — driving the engine

How deriva-ml turns a dataset into a bag.

**Mental model:** `DatasetBagBuilder` (deriva-ml) computes anchors +
policy, hands them to `CatalogBagBuilder` (deriva-py), which generates a
server-side **export spec** (one `query_processor` per FK path) that
the ERMrest export engine runs and paginates. deriva-ml decides *scope*;
deriva-py decides *fetch mechanics*.

**Formal rules** (`B1, B2, …`):
- Anchor construction: root dataset RID + **every recursive descendant
  dataset RID** (`anchors_for` / `_iter_descendant_rids`). Worked from
  the nesting structure. `[deriva-ml bag_builder]`.
- Empty-association pruning: `Dataset_X` is excluded unless the dataset
  (or any descendant) has ≥1 member of element-type X, or X is a
  vocabulary (`_exclude_empty_associations`). State precisely why a
  Subject-only dataset *with image-bearing children* still includes
  `Dataset_Image` (the descendant rule) — the exact subtlety from the
  2026-06-13 investigation. `[deriva-ml]`.
- Terminal tables applied: `{Execution, Workflow}` from the shared
  constant (links to T-rules). `[deriva-ml]`.
- Export-spec shape: one CSV `query_processor` per FK path, `paged_query`
  true; asset tables also get a `fetch` processor.
  `[engine: deriva-py catalog_builder._build_export_spec]`.
- What lands in the bag: members + their features/annotations + vocab +
  the provenance *link* rows, NOT the producing-executions' full asset
  closure (the consequence of the terminal rule).

**Worked examples:** the anchor set for a small nested dataset (real
RIDs/counts); reached tables for a real demo dataset; the export-spec
JSON for one path (real, from `get_export_spec()`).

**Same-engine siblings (one paragraph each):** `estimate_bag_size`
(shares the walker via `aggregate_queries`, bypasses the export engine —
cite ADR-0008); `clone_via_bag` (same policy incl. terminal tables,
different anchors/loader settings — cite ADR-0006); dataset drift
`is_dirty` (shares the walk to compare snapshots).

### 4.3 `docs/reference/denormalization.md` — the wide-table rules

Formal counterpart to the tutorial in `user-guide/`.

**Mental model:** "one row per leaf" star-schema flattening over a
`DatasetBag`'s already-downloaded tables — pure local computation, no
catalog, no FK *traversal-policy* (it's a different operation from bag
export; clarify this explicitly so the two aren't conflated).

**Formal rules** (`D1, D2, …`): the `row_per` / `via` / `include_tables`
semantics; column hoisting and duplication; feature-value handling and
the `selector` (newest/by-workflow/custom); `missing` / system-column
options; what `ignore_unrelated_anchors` does; the exact return shapes
of `get_denormalized_as_dataframe` / `_as_dict` /
`list_denormalized_columns` / `describe_denormalized`. All
`[deriva-ml dataset_bag.py]`.

**Worked examples:** a real denormalized frame from a demo bag (shape +
a few rows), a multi-table `via` example, a feature `selector` example.

### 4.4 Wiring

- `mkdocs.yml`: add a "Reference Manual" nav group containing the three
  docs (and keep `schema.md` discoverable).
- Cross-links: `user-guide/denormalization.md` ↔
  `reference/denormalization.md`; bag-export ↔ fk-traversal ↔ the ADRs;
  `user-guide/offline.md` (cache) → bag-export for "what's in the bag".
- Each doc header: a one-line "verified against demo catalog, deriva-ml
  vX.Y" provenance note + a "Quick answers" index.
- `deriva-ml/CLAUDE.md` (repo root): a one-line "Reference docs" pointer
  to the three files (§3a) so a Claude Code session knows they exist.

## 5. Authoring method (how the examples stay honest)

For each worked example: construct it against the **local demo
catalog** (`create_demo_catalog` with features + datasets, the same
fixture the test suite uses), run the real call, paste the real output.
Where an example shows a reached-table set or export spec, capture it
from `aggregate_queries` / `get_export_spec` directly. No hand-waved
numbers; if the demo catalog can't exhibit a behavior (e.g. deep
nesting), say so and mark that example illustrative.

## 6. Non-goals

- No code changes (pure documentation). If authoring uncovers a code
  bug or a docstring error, file/fix it separately, don't bundle.
- Not re-documenting cache, MINID, download tiers, asset upload (covered
  in `offline.md` / `manage-deriva-storage` / `executions.md`).
- Not a deriva-py reference — deriva-py rules are *cited*, not
  exhaustively re-specified; we document the behavior deriva-ml relies
  on and link to the upstream code.
- Not CI-enforced doctests for catalog-dependent examples (live-catalog
  dependency); the provenance note is the drift signal.

## 7. Testing / verification

- **Build check:** `uv run mkdocs build` succeeds with the new nav and
  no broken internal links (mkdocs warns on dead links).
- **Example verification:** the authoring method (§5) is the
  verification — every pasted output came from a real run, recorded in
  the PR description as "verified against demo catalog <id> on <date>".
- **Rule/code-location audit:** a final read-through confirms every
  numbered rule cites a real symbol/file that exists at the documented
  path (grep each cited path).

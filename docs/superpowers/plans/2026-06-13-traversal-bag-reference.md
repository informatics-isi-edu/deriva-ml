# FK Traversal / Bag Export / Denormalization Reference Manual — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE ON FORM:** This is a **documentation** project. Each doc-authoring task follows: (1) gather ground-truth facts (read code + run against the demo catalog), (2) write the section, (3) verify (cited symbols exist; pasted example outputs are real). "Tests" here = the verification step + `mkdocs build` (dead-link check) + a cited-symbol audit. Do NOT invent numbers — every example output is captured from a real run.

**Goal:** Author a formal, example-rich reference manual for FK traversal, bag export, and denormalization under `docs/reference/`, code-grounded and demo-catalog-verified, discoverable via the MCP RAG index (`ml-docs`) and Claude Code.

**Architecture:** Three new `docs/reference/*.md` files (mental model → numbered rules with `[layer]` tags + code locations → demo-catalog-verified worked examples), wired into mkdocs nav, cross-linked to the existing tutorial + ADRs, with a CLAUDE.md pointer. No source-code changes.

**Tech Stack:** Markdown, mkdocs (build/link check), the local demo catalog (`create_demo_catalog`), deriva-py `FKTraversalPolicy`/`PathWalker`/`CatalogBagBuilder` (cited), deriva-ml `DatasetBagBuilder`/`DatasetBag`.

**Spec:** `docs/superpowers/specs/2026-06-13-traversal-bag-reference-design.md`
**Branch:** `docs/traversal-bag-reference` (already created off `main`; spec commits are on it).

---

## Context for the implementer (read first)

- **CWD:** chain `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>` in ONE Bash call. Run Python with `DERIVA_ML_ALLOW_DIRTY=true uv run python ...`.
- **The demo catalog** is the example substrate. Build one with
  `create_demo_catalog(hostname="localhost", create_features=True, create_datasets=True, on_exit_delete=False)` (needs the localhost Docker stack up and a valid `localhost` bearer token — if `POST /ermrest/catalog` 401s, the token is stale; STOP and ask the user to re-auth). It clones deriva-ml from GitHub and populates Subject/Image/Dataset + features + nested datasets + executions/assets. Capture its catalog id; reuse across all example-gathering. Delete it (or leave it) at the end.
- **Layer tags:** every formal rule is tagged `[engine: deriva-py]` (enforced upstream — cite the file/class) or `[deriva-ml]` (deriva-ml's own logic). The split that matters most:
  - **Walk-phase fields** (the `PathWalker` honors): `schemas`, `exclude_schemas`, `exclude_tables`, `terminal_tables`, `max_depth`. `deriva/bag/path_walker.py`.
  - **Load-phase fields** (the loader honors): `vocab_export` (default `REFERENCED_ONLY`), `asset_mode` (default `UPLOAD_IF_MISSING`), `dangling_fk_strategy` (default `FAIL`... but deriva-ml's clone/export override to `DELETE`), `content_on_conflict` (default `FAIL`), `match_by_columns`. `deriva/bag/traversal.py` defines them; the loader applies them.
  - Vocabularies are ALWAYS leaves (walker guard), independent of `terminal_tables`.
- **Key code anchors to cite** (verify each path before citing — Task 5 audits this):
  - `deriva/bag/traversal.py::FKTraversalPolicy` (the 10 fields), `VocabExport`/`AssetMode`/`DanglingFKStrategy` enums.
  - `deriva/bag/path_walker.py::PathWalker` (outbound `table.foreign_keys` + inbound `table.referenced_by`; terminal check; vocab-leaf guard; `max_depth`).
  - `deriva/bag/catalog_builder.py::CatalogBagBuilder` (`_build_export_spec`, `_table_query_path`, `get_export_spec`).
  - `src/deriva_ml/dataset/bag_builder.py::DatasetBagBuilder` (`anchors_for`, `_iter_descendant_rids`, `_exclude_empty_associations`, `build_policy`, `aggregate_queries`).
  - `src/deriva_ml/core/constants.py::PROVENANCE_TERMINAL_TABLES`.
  - `src/deriva_ml/dataset/dataset_bag.py` (`get_denormalized_as_dataframe`, `get_denormalized_as_dict`, `list_denormalized_columns`, `describe_denormalized`, `restructure_assets`).
- **ADRs to cite (not restate):** 0006 (bag-oriented movement), 0008 (estimate bypasses the engine), 0002 (validate vs dry_run).
- **Existing docs:** `docs/user-guide/denormalization.md` (tutorial — cross-link, keep). `docs/reference/README.md` + `schema.md` (the reference dir; README lists each reference doc — update it).
- **mkdocs nav** (`mkdocs.yml`): `nav:` at line 40; the `User Guide` group ends ~line 53, `Configuration` starts line 54. Insert a new `Reference Manual` group.

### File map

| File | Action | Responsibility |
|---|---|---|
| `docs/reference/fk-traversal.md` | create | the engine: policy fields + walker rules + examples |
| `docs/reference/bag-export.md` | create | deriva-ml driving the engine: anchors, pruning, export spec, siblings |
| `docs/reference/denormalization.md` | create | formal wide-table rules over a DatasetBag |
| `docs/reference/README.md` | modify | list the three new docs |
| `mkdocs.yml` | modify | add "Reference Manual" nav group |
| `docs/user-guide/denormalization.md` | modify | cross-link to the formal reference |
| `CLAUDE.md` | modify | one-line "Reference docs" pointer |

---

## Task 0: Build the demo catalog + capture ground-truth example data

**Files:** none (produces `/tmp/refdoc-examples/` scratch with captured outputs).

This runs ONCE; all three docs draw their real numbers from here.

- [ ] **Step 1: Verify stack + build the catalog**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
docker ps --format '{{.Names}}' | grep -E 'deriva-postgres|deriva-webserver' && \
mkdir -p /tmp/refdoc-examples && \
DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY' 2>&1 | grep -vE "InsecureRequestWarning|warnings.warn" | tail -3
from pathlib import Path
from deriva_ml.demo_catalog import create_demo_catalog
cat = create_demo_catalog(hostname="localhost", create_features=True, create_datasets=True, on_exit_delete=False)
Path("/tmp/refdoc-examples/catalog_id.txt").write_text(str(cat.catalog_id))
print("CATALOG_ID", cat.catalog_id)
PY
```
Expected: `CATALOG_ID <n>`. If it 401s, STOP — report BLOCKED (stale localhost token; user must re-auth).

- [ ] **Step 2: Capture FK-traversal example data**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY' 2>&1 | grep -vE "InsecureRequestWarning|warnings.warn" > /tmp/refdoc-examples/traversal.txt
from deriva_ml import DerivaML
from deriva_ml.dataset.bag_builder import DatasetBagBuilder
cid = open("/tmp/refdoc-examples/catalog_id.txt").read().strip()
ml = DerivaML(hostname="localhost", catalog_id=cid)
ds = ml.lookup_dataset(list(ml.find_datasets())[0].dataset_rid)
b = DatasetBagBuilder(ml_instance=ml)
reached = sorted(b.aggregate_queries(ds).keys())
print("DATASET_RID", ds.dataset_rid)
print("REACHED_WITH_FIX", reached)
print("EXECUTION_ASSET_IN", "Execution_Asset" in reached)
# Reached set WITHOUT terminal tables (temporarily, to show the contrast):
import deriva_ml.dataset.bag_builder as bb
orig = bb.DatasetBagBuilder.build_policy
def no_term(self, dataset, *, vocab_export=None):
    from deriva.bag.traversal import FKTraversalPolicy, VocabExport
    from deriva_ml.core.constants import INTENTIONAL_FK_CYCLES
    return FKTraversalPolicy(exclude_tables=self._exclude_empty_associations(dataset),
        vocab_export=vocab_export or VocabExport.FULL, intentional_cycles=set(INTENTIONAL_FK_CYCLES))
bb.DatasetBagBuilder.build_policy = no_term
reached_no = sorted(DatasetBagBuilder(ml_instance=ml).aggregate_queries(ds).keys())
print("REACHED_NO_TERMINAL", reached_no)
print("EXTRA_WITHOUT_TERMINAL", sorted(set(reached_no) - set(reached)))
PY
cat /tmp/refdoc-examples/traversal.txt
```
Expected: prints the reached sets; `EXTRA_WITHOUT_TERMINAL` lists the Execution_Asset closure tables (the PR #297 contrast — real numbers for the example).

- [ ] **Step 3: Capture bag-export example data**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY' 2>&1 | grep -vE "InsecureRequestWarning|warnings.warn" > /tmp/refdoc-examples/export.txt
import json
from deriva_ml import DerivaML
from deriva_ml.dataset.bag_builder import DatasetBagBuilder
cid = open("/tmp/refdoc-examples/catalog_id.txt").read().strip()
ml = DerivaML(hostname="localhost", catalog_id=cid)
ds = ml.lookup_dataset(list(ml.find_datasets())[0].dataset_rid)
b = DatasetBagBuilder(ml_instance=ml)
print("ANCHORS", [type(a).__name__ + ":" + str(getattr(a, "rids", getattr(a, "table", "")))[:60] for a in b.anchors_for(ds)])
excl = b._exclude_empty_associations(ds)
print("EXCLUDED_EMPTY_ASSOCIATIONS", sorted(f"{s}.{t}" for s, t in excl))
# Members of the dataset (to explain the empty-association rule):
m = ds.list_dataset_members()
print("MEMBER_COUNTS", {k: len(v) for k, v in m.items()} if isinstance(m, dict) else m)
PY
cat /tmp/refdoc-examples/export.txt
```
Expected: anchors list, the excluded empty-associations set, member counts — the real basis for the bag-export examples.

- [ ] **Step 4: Capture denormalization example data**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY' 2>&1 | grep -vE "InsecureRequestWarning|warnings.warn" > /tmp/refdoc-examples/denorm.txt
from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetSpec
cid = open("/tmp/refdoc-examples/catalog_id.txt").read().strip()
ml = DerivaML(hostname="localhost", catalog_id=cid)
ds = ml.lookup_dataset(list(ml.find_datasets())[0].dataset_rid)
bag = ds.download_dataset_bag(DatasetSpec(rid=ds.dataset_rid, version=str(ds.current_version), materialize=False))
print("BAG_TABLES", sorted(bag.list_tables()))
cols = bag.list_denormalized_columns(["Subject"])
print("DENORM_COLUMNS_Subject", cols[:12])
df = bag.get_denormalized_as_dataframe(["Subject"])
print("DENORM_SHAPE", df.shape)
print("DENORM_HEAD", df.head(3).to_dict(orient="records"))
PY
cat /tmp/refdoc-examples/denorm.txt
```
Expected: bag tables, denormalized column list + frame shape + a few rows — the real denorm examples. (If `download_dataset_bag` is slow/needs minid config, fall back to whatever the demo bag fixture uses; capture whatever real output you get.)

- [ ] **Step 5: Commit the captured data as a reference artifact (so reviewers can check provenance)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
mkdir -p docs/reference/.examples && \
cp /tmp/refdoc-examples/*.txt docs/reference/.examples/ && \
git add docs/reference/.examples/ && \
git commit -m "docs(reference): capture demo-catalog ground-truth for reference examples

Raw outputs (traversal reached-sets, export anchors/exclusions, denorm
shapes) captured from a populated demo catalog; the reference docs paste
verified excerpts from these. Provenance for the 'verified against demo
catalog' note."
```

> If you prefer not to commit raw capture files, skip the `.examples/`
> commit and keep `/tmp/refdoc-examples/` for the authoring tasks only —
> but committing them makes the "verified" claim auditable. Default:
> commit them.

---

## Task 1: `docs/reference/fk-traversal.md` — the engine

**Files:** Create `docs/reference/fk-traversal.md`.

- [ ] **Step 1: Re-read the cited code to ground every rule**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
sed -n '185,265p' .venv/lib/python3.13/site-packages/deriva/bag/traversal.py && \
sed -n '60,260p' .venv/lib/python3.13/site-packages/deriva/bag/path_walker.py | grep -nE "foreign_keys|referenced_by|terminal|is_vocabulary|max_depth|exclude" | head
```
Confirm: the 10 FKTraversalPolicy fields + their defaults; the walker's outbound+inbound rule, terminal check, vocab-leaf guard, max_depth. These are the source of the rules you write.

- [ ] **Step 2: Write the document**

Create `docs/reference/fk-traversal.md` with this structure (fill the bracketed pieces from the code you just read + `/tmp/refdoc-examples/traversal.txt`):

```markdown
# FK Traversal Reference

> Verified against a populated demo catalog (`create_demo_catalog`,
> deriva-ml v1.46.x). Engine rules tagged `[engine: deriva-py]` are
> enforced upstream; `[deriva-ml]` rules are deriva-ml's own logic.

FK traversal is the mechanism that decides **which rows of which tables**
a dataset operation (bag export, size estimate, clone, drift check)
includes, by walking the catalog's foreign-key graph from a set of
anchor rows. This page is the formal reference for that mechanism. For
how a *bag* uses it, see [Bag Export](bag-export.md).

## Quick answers

| Question | Rule |
|---|---|
| Does the walk follow FKs in both directions? | [T2](#t2) |
| Why did my bag pull in unrelated executions' assets? | [T3](#t3) (terminal tables) |
| How do I stop the walk at a table? | [T3](#t3), [T5](#t5) |
| Are vocabularies fully exported? | [T4](#t4), [T7](#t7) |
| What limits how deep the walk goes? | [T6](#t6) |

## Mental model

The pipeline runs in **two phases**. The **walk** (`PathWalker`,
`[engine: deriva-py]`) discovers the reachable `(table)` set and the FK
paths to each, starting from an **anchor set**. The **load** then
materializes rows and resolves orphans/conflicts. Policy fields split
accordingly: *walk-phase* fields change **what is reached**; *load-phase*
fields change **how reached rows are handled**. Confusing the two is the
most common source of surprise.

## The policy: `FKTraversalPolicy`

[A table of all 10 fields: name | phase (walk/load) | default | one-line
meaning | layer tag. Pull the defaults exactly from the code:
vocab_export=REFERENCED_ONLY, asset_mode=UPLOAD_IF_MISSING,
dangling_fk_strategy=FAIL, content_on_conflict=FAIL, max_depth=None, etc.
`[engine: deriva-py] deriva/bag/traversal.py::FKTraversalPolicy`.]

## Formal rules

<a id="t1"></a>
**T1 — Anchors define the walk's origin.** [RIDAnchor = specific RIDs;
TableAnchor = whole table. The walk visits the anchor table(s) first.
`[deriva-ml]` supplies anchors; `[engine]` consumes them.]

<a id="t2"></a>
**T2 — The walk is bidirectional.** [The walker follows both outbound
FKs (`table.foreign_keys`) and inbound FKs (`table.referenced_by`).
`[engine: deriva-py] path_walker.PathWalker`.]

<a id="t3"></a>
**T3 — Terminal tables are entered but not exited.** [Neither outbound
nor inbound FKs followed. This is what stops a provenance hub
(`Execution`) from fanning out across the catalog graph. deriva-ml's
dataset export and clone set `{Execution, Workflow}` via
`core/constants.py::PROVENANCE_TERMINAL_TABLES`. `[engine]` mechanism,
`[deriva-ml]` chosen set.]

<a id="t4"></a>
**T4 — Vocabularies are always leaves.** [Entered, never traversed out,
regardless of `terminal_tables` — a universal walker guard
(`table.is_vocabulary()`). `[engine: deriva-py]`.]

<a id="t5"></a>
**T5 — Schema/table pruning happens before the walk.** [`schemas`
allow-list, `exclude_schemas` (default DEFAULT_EXCLUDE_SCHEMAS),
`exclude_tables`. `[engine]`.]

<a id="t6"></a>
**T6 — `max_depth` bounds FK hops from the anchor.** [Default None =
unbounded. `[engine]`.]

<a id="t7"></a>
**T7 — Load-phase fields don't change what's reached.** [vocab_export,
asset_mode, dangling_fk_strategy, content_on_conflict, match_by_columns
are applied by the loader after the walk; list each with one line + its
enum values (VocabExport FULL/REFERENCED_ONLY, etc.). `[engine]` for the
mechanism; deriva-ml chooses values per operation (e.g. export sets
vocab_export=FULL, dangling=DELETE).]

## Worked examples

### A RID-anchored walk's reached set
[Paste DATASET_RID + REACHED_WITH_FIX from traversal.txt; explain the
shape — members, features, vocab, Execution (terminal) present, no
Execution_Asset.]

### Terminal tables in action (the provenance-hub rule)
[Paste REACHED_NO_TERMINAL vs REACHED_WITH_FIX and EXTRA_WITHOUT_TERMINAL
— the real tables that appear ONLY when Execution isn't terminal. This
is the PR #297 behavior, with demo-catalog numbers. Link T3.]

[If the demo catalog can't exhibit max_depth truncation meaningfully,
state that and give an illustrative (clearly-marked) example instead.]

## See also
- [Bag Export](bag-export.md) — how a dataset bag drives this walk.
- ADR-0006 (bag-oriented data movement).
```

- [ ] **Step 3: Verify cited symbols exist + build**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
for sym in "class FKTraversalPolicy" "class VocabExport" "class PathWalker"; do grep -rq "$sym" .venv/lib/python3.13/site-packages/deriva/bag/ && echo "OK: $sym" || echo "MISSING: $sym"; done && \
grep -q "PROVENANCE_TERMINAL_TABLES" src/deriva_ml/core/constants.py && echo "OK: PROVENANCE_TERMINAL_TABLES" && \
uv run mkdocs build 2>&1 | grep -iE "warn|error|dead|not found" | grep -i "fk-traversal" || echo "mkdocs: no fk-traversal link warnings"
```
Expected: all `OK:`; no mkdocs warnings naming fk-traversal. (mkdocs nav wiring is Task 4 — a not-in-nav warning here is fine; broken *links* are not.)

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/reference/fk-traversal.md && \
git commit -m "docs(reference): FK traversal reference — policy fields, walk rules, examples"
```

---

## Task 2: `docs/reference/bag-export.md` — driving the engine

**Files:** Create `docs/reference/bag-export.md`.

- [ ] **Step 1: Re-read the cited code**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
sed -n '720,760p;818,900p' src/deriva_ml/dataset/bag_builder.py && \
grep -nE "def _build_export_spec|def _table_query_path|paged_query|query_processor" .venv/lib/python3.13/site-packages/deriva/bag/catalog_builder.py | head
```
Confirm: `anchors_for` (root + descendant RIDs), `_exclude_empty_associations` (the descendant-member rule), `build_policy` (terminal tables), and the export-spec shape (one query_processor per FK path, paged_query).

- [ ] **Step 2: Write the document**

Create `docs/reference/bag-export.md` (rules `B1…`, pull real numbers from `/tmp/refdoc-examples/export.txt`):

```markdown
# Bag Export Reference

> Verified against a populated demo catalog (deriva-ml v1.46.x).

Bag export turns a dataset into a downloadable BDBag. `DatasetBagBuilder`
(`[deriva-ml]`) computes the **anchors** and the **traversal policy**,
hands them to `CatalogBagBuilder` (`[engine: deriva-py]`), which emits a
server-side **export spec** the ERMrest export engine runs and paginates.
deriva-ml decides *scope*; deriva-py decides *fetch mechanics*. Read
[FK Traversal](fk-traversal.md) first for the walk rules this builds on.

## Quick answers

| Question | Rule |
|---|---|
| Why are there 50 dataset RIDs in the export query? | [B1](#b1) (nested anchors) |
| Why is `Dataset_Image` included for a Subject-only dataset? | [B2](#b2) (descendant rule) |
| What's actually in the bag? | [B5](#b5) |
| Why doesn't the bag contain executions' full asset sets? | [B3](#b3) |

## Mental model
[The scope-vs-mechanics split; the two builders; the server-side engine.]

## Formal rules

<a id="b1"></a>
**B1 — The anchor set is the root RID plus every recursive descendant
dataset RID.** [`anchors_for` + `_iter_descendant_rids`. A nested pool of
N datasets anchors at N+1 RIDs. `[deriva-ml]`. Paste ANCHORS from
export.txt.]

<a id="b2"></a>
**B2 — `Dataset_X` is included iff the dataset OR any descendant has a
member of element-type X (or X is a vocabulary).**
[`_exclude_empty_associations`. The subtlety: a dataset with zero direct
`Image` members but image-bearing *children* still includes
`Dataset_Image`, because the member scan covers descendants. Paste
MEMBER_COUNTS + EXCLUDED_EMPTY_ASSOCIATIONS. `[deriva-ml]`.]

<a id="b3"></a>
**B3 — Provenance tables are terminal.** [`{Execution, Workflow}` from
the shared constant; links to [T3](fk-traversal.md#t3). Consequence: the
bag carries the provenance *link* rows but not the producing-executions'
full asset closure. `[deriva-ml]`.]

<a id="b4"></a>
**B4 — The export spec is one `query_processor` per FK path
(`paged_query`), plus a `fetch` processor per asset table.**
[`[engine: deriva-py] catalog_builder._build_export_spec` /
`_table_query_path`. The server runs each path and streams results.]

<a id="b5"></a>
**B5 — What lands in the bag:** members + their features/annotations +
referenced vocabulary + provenance-link rows — NOT the execution
asset-closure (B3).

## Worked examples
### Anchor set for a nested dataset
[Paste ANCHORS. Explain root + descendants.]
### The empty-association rule
[Paste MEMBER_COUNTS + EXCLUDED_EMPTY_ASSOCIATIONS; walk through one
included and one excluded `Dataset_X`.]

## Same engine, different consumer
- **`estimate_bag_size`** — shares the *walk* (via `aggregate_queries`)
  but bypasses the export engine, running the queries directly for a
  size prediction. See ADR-0008. `[deriva-ml]`.
- **`clone_via_bag`** — same policy (incl. terminal tables), different
  anchors and loader settings (`dangling_fk_strategy=DELETE`,
  `vocab_export=FULL`). See ADR-0006. `[deriva-ml]`.
- **Dataset drift (`is_dirty`)** — shares the walk to compare the
  dataset's current snapshot against its recorded version. `[deriva-ml]`.

## See also
- [FK Traversal](fk-traversal.md), [Denormalization](denormalization.md)
- ADR-0006, ADR-0008, ADR-0002 (validate vs dry_run).
```

- [ ] **Step 3: Verify + build**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
grep -q "def anchors_for" src/deriva_ml/dataset/bag_builder.py && \
grep -q "def _exclude_empty_associations" src/deriva_ml/dataset/bag_builder.py && \
grep -q "def _build_export_spec" .venv/lib/python3.13/site-packages/deriva/bag/catalog_builder.py && \
echo "cited symbols OK" && \
uv run mkdocs build 2>&1 | grep -iE "error|dead link" | grep -i "bag-export" || echo "mkdocs: no bag-export link errors"
```
Expected: `cited symbols OK`; no broken-link errors for bag-export.

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/reference/bag-export.md && \
git commit -m "docs(reference): bag export reference — anchors, pruning, export spec, siblings"
```

---

## Task 3: `docs/reference/denormalization.md` — the wide-table rules

**Files:** Create `docs/reference/denormalization.md`.

- [ ] **Step 1: Re-read the cited code + the tutorial (don't duplicate it)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
sed -n '912,1075p' src/deriva_ml/dataset/dataset_bag.py | grep -nE "def get_denormalized|def list_denormalized|def describe_denormalized|row_per|via|selector|missing|system_columns|ignore_unrelated" | head && \
head -60 docs/user-guide/denormalization.md
```
Confirm the method signatures + the tutorial's "one row per leaf" framing (the reference states it formally; the tutorial teaches it).

- [ ] **Step 2: Write the document**

Create `docs/reference/denormalization.md` (rules `D1…`, real output from `/tmp/refdoc-examples/denorm.txt`):

```markdown
# Denormalization Reference

> Verified against a populated demo catalog (deriva-ml v1.46.x). For a
> gentle, example-led introduction see the
> [Denormalization tutorial](../user-guide/denormalization.md); this page
> is the formal rule reference.

Denormalization flattens a `DatasetBag`'s already-downloaded normalized
tables into a single **wide table** (one row per leaf observation), with
related-table columns hoisted alongside. It is **pure local computation**
over a downloaded bag — no catalog access, and **not** an FK
*traversal-policy* operation (distinct from [bag export](bag-export.md);
don't conflate them). All rules are `[deriva-ml]`
(`dataset_bag.py`).

## Quick answers

| Question | Rule |
|---|---|
| What determines the number of rows? | [D1](#d1) (row_per) |
| How are upstream columns added? | [D2](#d2) (hoisting) |
| Which feature value is used when there are several? | [D4](#d4) (selector) |
| What are the exact return types? | [D6](#d6) |

## Mental model
[One row per leaf / star-schema flattening; downstream table = row_per.]

## Formal rules
<a id="d1"></a>
**D1 — `row_per` selects the grain.** [Default = the furthest-downstream
requested table; one row per instance of it. `via` disambiguates the
join path.]
<a id="d2"></a>
**D2 — Upstream columns are hoisted and duplicated.** [Across rows
sharing the same upstream row.]
<a id="d3"></a>
**D3 — `include_tables` names the column sources.** [...]
<a id="d4"></a>
**D4 — Feature values use a `selector`.** [Default vs newest/by-workflow/
custom; the FeatureRecord callable signature.]
<a id="d5"></a>
**D5 — `missing` / `system_columns` / `ignore_unrelated_anchors`
behavior.** [Each precisely.]
<a id="d6"></a>
**D6 — Return shapes.** [`get_denormalized_as_dataframe` → pandas
DataFrame; `_as_dict` → list[dict]; `list_denormalized_columns` →
list[(table, column)]; `describe_denormalized` → its exact structure.]

## Worked examples
### Columns and shape of a denormalized frame
[Paste DENORM_COLUMNS_Subject, DENORM_SHAPE, DENORM_HEAD from denorm.txt.]
### Multi-table via / a feature selector
[A second example using real bag tables; if the demo bag lacks the shape,
mark illustrative.]

## See also
- [Denormalization tutorial](../user-guide/denormalization.md) (intro)
- [Bag Export](bag-export.md) (how the bag this reads was produced)
- `restructure_assets` (reorganizing bag files for ImageFolder trainers)
```

- [ ] **Step 3: Verify + build**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
for sym in get_denormalized_as_dataframe get_denormalized_as_dict list_denormalized_columns describe_denormalized; do grep -q "def $sym" src/deriva_ml/dataset/dataset_bag.py && echo "OK: $sym" || echo "MISSING: $sym"; done && \
uv run mkdocs build 2>&1 | grep -iE "error|dead link" | grep -i "denormalization" || echo "mkdocs: no denormalization link errors"
```
Expected: all `OK:`; no broken-link errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/reference/denormalization.md && \
git commit -m "docs(reference): denormalization reference — formal wide-table rules"
```

---

## Task 4: Wiring — nav, README, cross-links, CLAUDE.md

**Files:** Modify `mkdocs.yml`, `docs/reference/README.md`, `docs/user-guide/denormalization.md`, `CLAUDE.md`.

- [ ] **Step 1: Add the mkdocs nav group**

In `mkdocs.yml`, after the `User Guide` group (ends ~line 53, before
`- Configuration:` at line 54), insert:

```yaml
  - Reference Manual:
      - FK Traversal: reference/fk-traversal.md
      - Bag Export: reference/bag-export.md
      - Denormalization (formal): reference/denormalization.md
      - Schema: reference/schema.md
```
(Indent to match the existing two-space nav style. `schema.md` was
previously not in nav; including it here makes the reference dir
discoverable.)

- [ ] **Step 2: Update `docs/reference/README.md`**

Append entries describing the three new docs (mirror the existing
`## schema.md` style — a heading + one-paragraph description each):
fk-traversal.md, bag-export.md, denormalization.md.

- [ ] **Step 3: Cross-link the tutorial**

Near the top of `docs/user-guide/denormalization.md`, add a callout:

```markdown
> For the **formal rules** (exact `row_per` / `via` / selector behavior
> and return types), see the
> [Denormalization Reference](../reference/denormalization.md). This page
> is the example-led introduction.
```

- [ ] **Step 4: Add the CLAUDE.md pointer**

In `CLAUDE.md`, add a short subsection (near the other doc-pointer notes,
e.g. after the `model/annotations.py` public-API note around line 600):

```markdown
### Reference manual for traversal / export / denormalization

Exact behavior of FK traversal, bag export, and denormalization is
documented formally in `docs/reference/`:
- `docs/reference/fk-traversal.md` — FKTraversalPolicy fields + walk rules
- `docs/reference/bag-export.md` — anchors, empty-association pruning,
  terminal tables, export-spec shape
- `docs/reference/denormalization.md` — wide-table (`get_denormalized_*`)
  rules

These are RAG-indexed as `ml-docs` (searchable via
`rag_search(doc_type="ml-docs")`) once on `main`. Consult them when
answering questions about what a bag includes or how denormalization
shapes a frame.
```

- [ ] **Step 5: Full mkdocs build (clean) + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
uv run mkdocs build 2>&1 | tail -15
```
Expected: build succeeds; no `WARNING` about the three reference pages
being absent from nav, and no dead-link warnings. (Other pre-existing
mkdocs warnings unrelated to these files are out of scope.)

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add mkdocs.yml docs/reference/README.md docs/user-guide/denormalization.md CLAUDE.md && \
git commit -m "docs(reference): wire reference manual into nav + cross-links + CLAUDE.md"
```

---

## Task 5: Final audit — cited symbols, examples real, build clean; PR

**Files:** none new.

- [ ] **Step 1: Cited-symbol audit across all three docs**

For every `` `symbol` `` or code path referenced in the three docs,
confirm it exists. Run:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
echo "=== deriva-py symbols ===" && \
for s in FKTraversalPolicy PathWalker CatalogBagBuilder VocabExport AssetMode DanglingFKStrategy; do grep -rqw "$s" .venv/lib/python3.13/site-packages/deriva/bag/ && echo "OK $s" || echo "MISSING $s"; done && \
echo "=== deriva-ml symbols ===" && \
for s in anchors_for _iter_descendant_rids _exclude_empty_associations build_policy aggregate_queries get_denormalized_as_dataframe list_denormalized_columns describe_denormalized; do grep -rq "def $s" src/deriva_ml/ && echo "OK $s" || echo "MISSING $s"; done && \
grep -q PROVENANCE_TERMINAL_TABLES src/deriva_ml/core/constants.py && echo "OK PROVENANCE_TERMINAL_TABLES"
```
Expected: all `OK`. Any `MISSING` → the doc cites a wrong/renamed symbol; fix the doc before proceeding.

- [ ] **Step 2: Confirm examples trace to captured data**

Spot-check that the numbers pasted into each doc's "Worked examples"
match `/tmp/refdoc-examples/*.txt` (or `docs/reference/.examples/` if
committed). They must be byte-for-byte from a real run, not rounded or
invented. If any number can't be traced to a capture, re-run the capture
and fix the doc.

- [ ] **Step 3: Clean full build**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
uv run mkdocs build --strict 2>&1 | tail -20
```
`--strict` fails on broken links/nav. If it fails ONLY on pre-existing
issues in files this PR didn't touch, note them; if it fails on any of
the new reference pages, fix before PR.

- [ ] **Step 4: Push + open PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git push -u origin docs/traversal-bag-reference && \
gh pr create --title "docs(reference): FK-traversal / bag-export / denormalization reference manual" --body "$(cat <<'EOF'
Adds a formal, example-rich reference manual under `docs/reference/` for the FK-traversal, bag-export, and denormalization mechanics — the operations behind dataset bags and ML-frame production.

## What's here
- `reference/fk-traversal.md` — the engine: `FKTraversalPolicy`'s 10 fields (walk-phase vs load-phase), the bidirectional walker, **terminal tables** (the provenance-hub rule from #297), the vocab-leaf guard, `max_depth`.
- `reference/bag-export.md` — how `DatasetBagBuilder` drives the engine: nested-descendant anchors, the empty-association pruning rule (incl. the Subject-only-dataset-with-image-children subtlety), the server-side export spec, and one-paragraph entries for the same-engine siblings (estimate / clone / drift).
- `reference/denormalization.md` — formal wide-table rules (`row_per` / `via` / selector / return shapes), complementing the existing tutorial.

## Form
Each doc is **mental model → numbered rules (with `[engine: deriva-py]` / `[deriva-ml]` layer tags + code locations) → worked examples**. Every example was **run against a populated demo catalog** and the real output pasted in (provenance under `docs/reference/.examples/`); deriva-py rules are cited as the upstream authority, not re-specified.

## Discoverability
`docs/reference/*.md` is auto-indexed by the deriva-ml-mcp-plugin RAG crawler as `doc_type="ml-docs"` (repo-root, `**/*.md`, tracks `main`) — searchable via `rag_search` **after this merges to main**. A `CLAUDE.md` pointer makes a Claude Code session aware of them. No code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" 2>&1 | tail -2
```

- [ ] **Step 5: Catalog cleanup**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
cid=$(cat /tmp/refdoc-examples/catalog_id.txt) && \
echo "Demo catalog $cid was created for example capture. Ask the user whether to delete it (it's a localhost demo catalog); to delete: DerivaML/ErmrestCatalog delete_ermrest_catalog(really=True)."
```
Report the catalog id and let the user decide (do not auto-delete a catalog without confirmation).

---

## Self-review

- **Spec coverage:** §4.1 fk-traversal → Task 1; §4.2 bag-export (incl. siblings) → Task 2; §4.3 denormalization → Task 3; §4.4 wiring (nav/README/cross-link) → Task 4; §3a discoverability (RAG auto-index is verified; CLAUDE.md pointer) → Task 4 Step 4 + documented in the PR; §5 authoring method (demo-catalog capture) → Task 0; §7 verification (mkdocs build, cited-symbol audit, example provenance) → Tasks 1-3 per-doc + Task 5 final. All spec sections map to a task.
- **Placeholder scan:** the bracketed `[...]` pieces in the doc templates are authoring instructions ("fill from code/captured data"), not shipped placeholders — the implementer replaces them with real content sourced from the Task-0 captures and the cited code. The rule IDs (T/B/D-numbered) are the doc's stable-anchor convention. No `TBD`/`TODO` ship.
- **Consistency:** rule-ID schemes (T#/B#/D#) and anchor format (`<a id="t3"></a>` + `[T3](#t3)`) are uniform across the three docs and their cross-links; layer tags (`[engine: deriva-py]` / `[deriva-ml]`) used identically; the walk-phase/load-phase field split is stated the same way in fk-traversal and referenced from bag-export.
- **Branch-first:** branch exists with the spec commits; all task commits land on it; Task 5 pushes + opens the PR. Nothing on `main`.
- **Catalog dependency:** Task 0 needs the localhost stack + a valid token; if unavailable, the whole example-capture is blocked — the plan says STOP/report BLOCKED rather than fabricate numbers (the entire point of the doc is accuracy).

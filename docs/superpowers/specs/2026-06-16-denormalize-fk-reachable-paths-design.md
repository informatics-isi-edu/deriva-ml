# Denormalize planner: union FK-reachable paths (subject-partitioned feature reads)

**Date:** 2026-06-16
**Status:** Approved design — ready for implementation plan
**Author:** investigation + design this session

## Problem

`DatasetBag.feature_values` / `Dataset.feature_values` (and therefore
`as_tf_dataset` / `as_torch_dataset` label resolution, and every
`get_denormalized_*` consumer) return **zero feature rows** on
**subject-partitioned** datasets — datasets whose members are `Subject`s, with
the asset/element table (`Image`) reachable only via the FK chain
`Subject → Observation → Image`, never as a direct dataset member.

This is the **label-side** follow-up to the element-enumeration fix
(deriva-ml #316, the `reachable=` parameter on `as_tf_dataset` /
`as_torch_dataset` / `resolve_element_rids`). Element enumeration is now
FK-reachability-aware; the **feature-value join is not**, so
`as_tf_dataset(element_type="Image", targets={...}, missing="skip")` finds the
elements but resolves **0 labels**, drops all of them under `missing="skip"`,
and raises the empty-generator error.

### Verified root cause (not the reported framing)

The report framed two distinct problems ("no reachability option" + "returns 0
even where data is present"). Investigation on the live `6-CQZE` bag
(`dev.eye-ai.org`, training child `2-7KA2`) proved they are **one root cause**
in the denormalize planner, and it is **not** a keying bug:

| Layer | 6-CQZE parent | 2-7KA2 child |
|---|---|---|
| `Image_Diagnosis` SQLite rows | 591 | 591 |
| `Image` SQLite rows | 102 | 102 |
| `resolve_element_rids(reachable=True)` | 102 ✓ | 102 ✓ |
| `feature_values("Image","Image_Diagnosis")` | **0** ✗ | **0** ✗ |

`feature_values` → `Denormalizer` → planner `_prepare_wide_table`. Instrumenting
the planner on the real bag showed it **discovers** the correct path but
**chooses the wrong one**:

- Discovered by `_schema_to_paths()`:
  `Dataset → Subject_Dataset → Subject → Observation → Image → Image_Diagnosis`
  (FK-reachable — has data)
- **Chosen** by Phase-1b path dedup:
  `Dataset → Image_Dataset → Image → Image_Diagnosis`
  (direct membership — `Image_Dataset` has **0 rows** on a subject-partitioned
  dataset)

Raw SQL on the bag confirmed: the chosen membership join → **0 rows**; the
FK-reachable chain → **980 raw rows**, which dedups to **591 distinct
`Image_Diagnosis.RID`** (= the ground-truth count). `Subject_Dataset` has 25
rows; `Image_Dataset` / `Dataset_Image_Diagnosis` / `Dataset_Observation` are
all empty (this is what "subject-partitioned" means).

The culprit is `_prepare_wide_table`'s **Phase-1b path dedup**
(`src/deriva_ml/model/denormalize_planner.py` ~lines 1746–1783): when an element
is reachable via several routes it keys by `(element, endpoint)` and **prefers
the `Dataset_{Element}` membership association** (`_is_standard_assoc`),
**discarding the FK-reachable chain**. The prefix selection then takes
`paths[0][1]` as the single `Dataset → assoc → element` join prefix
(~lines 1808–1814). That assumption holds for directly-populated datasets and
silently produces empty results for subject-partitioned ones.

## Goal

Make the denormalize planner **union all reachable `Dataset → … → element`
paths** (membership *and* FK-reachable), **deduplicated RID-distinct on the
`row_per` leaf**, so every denormalize consumer returns the correct
FK-reachable rows on subject-partitioned datasets while returning **identical
row sets** on directly-populated ones (row-set identity, not SQL identity — the
demo already has two `Dataset → Image` routes; see the demo-schema note below).

## Chosen approach (decided during brainstorming)

- **Fix locus:** the **planner path selection** (root), not the feature-read
  wrapper. One fix corrects `feature_values`, `get_denormalized_*`, and
  `describe` consistently.
- **Trigger rule: Option 2 — always UNION** the membership path and the
  FK-reachable path(s), **RID-distinct on the `row_per` leaf** (the same
  "UNION not UNION ALL" semantics already shipped in
  `DatasetBag._dataset_table_view`). One rule for all datasets; constant,
  predictable behavior. Not the narrower "fall back only when membership is
  empty" — constant behavior was preferred and the blast radius is covered by
  the regression nets below.

### Why UNION-distinct is correct (verified)

The FK-reachable join **fans out** (an element reachable via multiple
`Subject → Observation → Image` routes appears multiple times): 980 raw rows.
Deduplicating **DISTINCT on the `row_per` leaf RID** (`Image_Diagnosis.RID`)
collapses to exactly **591** — matching both the total `Image_Diagnosis` rows in
the bag and the count whose `Image` is reachable. `DISTINCT Image.RID` = 102
(matches enumeration). The dedup grain is unambiguous: **the `row_per` leaf
table's RID**.

For a **directly-populated** dataset the membership path is non-empty and the FK
path adds no new leaf RIDs (or duplicates existing ones), so UNION-distinct
yields exactly today's **row set** — the property the demo regression suite
enforces.

**Important demo-schema note (verified):** the demo `Image` table already has
**two** `Dataset → Image` routes at the schema level —
`Dataset → Dataset_Image → Image` (membership) **and**
`Dataset → Dataset_Subject → Subject → Image` (FK-reachable, since
`Image.Subject → Subject`). The normal demo datasets (members = Subject + Image)
mask the bug because the membership route is populated. This means:

1. The always-union rule **does** change the demo's *generated SQL* (one
   statement → a UNION) even on directly-populated datasets, so the Net-1
   regression gate is **row-set identity, not SQL identity** — assert the
   returned record set is unchanged, not that the query string is unchanged.
2. The demo therefore **already exercises** the new multi-path code, making the
   committed gate genuinely representative.
3. The demo works today with two routes present, which means the planner does
   **not** currently classify the two `Dataset → Image` routes as a Rule-6
   ambiguity — validating Unit C's premise that `Dataset`-anchored reachability
   routes are a different path class from `row_per`↔include-table column
   ambiguity.

## Components

### Unit A — Path retention (`_prepare_wide_table`, Phase 1b/1c)

Today Phase-1b dedups by `(element, endpoint)` keeping one preferred
(membership) association route. Change: dedup only **truly identical** paths
(same table sequence); **retain distinct routes to the same element**
(membership *and* FK-reachable). `element_tables` carries **multiple paths per
element** instead of one. This is the only structural change; downstream
already loops over paths.

The `_is_standard_assoc` / `Dataset_{Element}` *preference* logic is removed as
the selection mechanism (it currently discards reachable paths). Path ordering
may still place the membership route first for stable output, but **no route is
dropped**.

### Unit B — UNION emission (`_denormalize_impl`)

`local_db/denormalize.py:420–460` already loops over the per-element paths and
`union(*sql_statements)` when there is more than one. It currently receives one
path per element; now it receives N. SQLAlchemy `union` is `UNION` (deduped),
not `UNION ALL`, so the fan-out collapses automatically.

**Correctness pin:** all paths to a given `row_per` leaf must project the
**identical column tuple** (same leaf table → same columns), so `UNION` dedups
on leaf-row identity. The plan must assert this invariant (a guard or a test),
because mismatched projections would silently defeat the dedup.

### Unit C — Ambiguity guard interaction (`_find_path_ambiguities`, Rule 6)

Rule 6 raises `DerivaMLDenormalizeAmbiguousPath` when multiple FK paths exist
between `row_per` and a requested **column-contributing** `include_table`.
Retaining multiple `Dataset → element` routes must **not** trip Rule 6 — that
would convert today's silent-empty bug into a loud-raise bug.

Two path classes must stay distinct:

- **Multiple `Dataset`-anchored membership/reachability routes to an element**
  → unioned (this fix). NOT a Rule-6 ambiguity.
- **Multiple paths between `row_per` and a column-contributing
  `include_table`** → still a genuine ambiguity; still raises; still resolved
  with `via=`.

The fix scopes the union to the **`Dataset`-anchored reachability paths** and
leaves Rule 6's `row_per`↔include-table column-ambiguity check untouched. The
plan must include a test that a column-ambiguous request still raises.

## Edge cases (each gets a test)

1. **Directly-populated** (demo `Image` is a direct member) → membership path
   non-empty, FK path adds nothing new → **identical to today** (regression
   gate).
2. **Subject-partitioned** (6-CQZE; demo subject-partitioned fixture) →
   membership empty, FK path supplies the rows → **0 → N**.
3. **Mixed** (some direct + some FK-reachable members) → UNION-distinct =
   correct superset (neither under-returns like today nor double-counts).
4. **No path at all** → unchanged: empty result / existing orphan handling.
5. **Fan-out** (980 → 591) → handled by UNION-distinct on the leaf RID.

## Testing & acceptance gates

Three independent nets; all are mandatory acceptance gates. This is what makes
the always-union blast radius safe.

### Net 1 — Directly-populated regression (catches "broke the normal case")

The existing planner suite runs on the **directly-populated demo catalog**.
Gate: `tests/local_db/test_planner_rules.py`, `test_paths.py`,
`test_denormalize_impl.py`, `test_denormalize_feature_records.py`,
`test_denormalizer.py`, `test_denormalize_selector.py` stay **green with no
expectation edits**. A changed demo result is a finding to explain, not a test
to update. New unit test: the planner emits **multiple paths** for an element
reachable two ways, and UNION-distinct yields the same row set as the
single-path result when membership is populated.

### Net 2 — Exactness oracle (catches "non-empty but wrong")

`estimate_bag_size` independently computes RID-distinct reachable counts per
table via the client-side reachability BFS. Gate:
`len(feature_values(T, F))` equals the estimate's reachable feature-table count.
Two independent code paths (planner UNION vs. BFS) must agree. Run on **both**
the directly-populated demo and a subject-partitioned dataset.

### Net 3 — Live equivalence (real-world shape)

- **Committed, demo-based (durable gate):** a **subject-partitioned demo
  fixture** — a dataset whose members are `Subject`s only, with `Image`
  FK-reachable via `Image.Subject` (the demo `Image` table has an `Image.Subject
  → Subject` FK; the demo features `Image/Quality` and `Image/BoundingBox` hang
  off `Image`). Assert: before the fix `feature_values("Image","Quality")`
  returns 0 (RED), after the fix it returns the FK-reachable count, and
  `as_tf_dataset(element_type="Image", targets={"Quality": …}, missing="skip")`
  yields one labeled `(image, label, rid)` per reachable image. The demo chain
  is one hop (`Subject → Image`) vs eye-ai's two hops (`Subject → Observation →
  Image`); both exercise the same "membership-empty, FK-reachable" planner
  path, so the demo fixture is a valid committed gate that needs no live
  eye-ai access.

- **Manual, live eye-ai (not committed — needs a token):**
  - `6-CQZE` child `2-7KA2`: `feature_values("Image","Image_Diagnosis")` → **591**
    (was 0); `as_tf_dataset(targets={"Image_Diagnosis": select_initial_diagnosis},
    missing="skip")` → **102** labeled triples.
  - `2-277G` v4.8.0 (prod): `feature_values("Image","Image_Diagnosis")` returns
    the `bag_info` count; `as_tf_dataset` yields ~28,546 labeled triples. Record
    the actual numbers in the PR.

### TDD order

Write the failing subject-partitioned demo `feature_values` test first
(RED: 0), implement the planner union (Units A–C), confirm GREEN, then run the
full planner suite as the Net-1 regression gate, then Net-2 oracle, then the
live Net-3 checks.

## Documentation

`docs/reference/denormalization.md` documents the path/scoping rules
(membership is "a filter on what's in scope"). Update it to state the
union-of-reachable-paths rule and the RID-distinct-on-`row_per`-leaf dedup, so
the reference matches the implemented semantics. Tag the rule `[deriva-ml]`.

## Out of scope

- No change to element **enumeration** (`resolve_element_rids` /
  `_dataset_table_view`) — that shipped in #316 and is correct.
- No new public API surface. `feature_values` keeps its signature; the report's
  suggested `reachable=` parameter on `feature_values` is **not** added — the
  planner fix makes the correct behavior the default, and a per-call opt-out
  would re-introduce the inconsistency the union is removing. (If a future
  caller genuinely needs membership-only scoping, that is a separate, justified
  request.)
- No change to Rule 6 column-ambiguity semantics.

## Risks

- **Largest risk:** a demo (directly-populated) result silently shifts under the
  always-union rule. Mitigation: Net-1 forbids expectation edits; any demo diff
  is investigated, not rubber-stamped.
- **Rule 6 false-positive:** retaining multiple element routes trips the
  ambiguity guard. Mitigation: Unit C scopes the union to `Dataset`-anchored
  routes and a test pins that genuine column-ambiguity still raises.
- **Projection-mismatch dedup defeat:** Unit B's identical-projection invariant
  is asserted by guard/test.

# Testing the Association-Index Generator — Design

**Date:** 2026-06-02
**Status:** Draft for review.
**Subproject:** `deriva-ml`
**Script under test:** `scripts/generate_association_indexes.py` (added in #271).

## 1. Goal

Validate, end-to-end against a real catalog, that
`scripts/generate_association_indexes.py` produces SQL which, when
applied, leaves **every pure binary association table with composite
indexes in both join directions on both the live table and its
`_ermrest_history` partner**.

The demo catalog is small by design, so **wall-clock timing is not a
meaningful signal** — a handful of rows are read fast with or without
an index, and any ms difference is noise. Instead of measuring
performance, the test answers the question that *is* meaningful at any
scale: **does the Postgres planner actually use each generated index
when running the query it was built for?** That is a yes/no fact about
plan shape and index-scan counters, not a timing measurement. The
two are different claims and the test makes the one it can stand
behind.

This is a **test/validation exercise**, not a change to the script or
the library. No production code is modified. The deliverable is a
documented test run (a report) plus any bugs found in the script.

## 2. Environment (already in place)

The full Deriva stack runs locally in Docker:

| Container          | Role                                              |
|--------------------|---------------------------------------------------|
| `deriva-postgres`  | postgres:16.13 backing the catalog (port 5432)    |
| `deriva-webserver` | serves ermrest/hatrac at `https://localhost`      |
| `deriva-mcp-test`  | the `dev-localhost` MCP server (same catalog)     |

Postgres is reachable directly with
`docker exec deriva-postgres psql -U ermrest -d <db>`. There is **no
local `psql` binary and no pgAdmin app installed** — so the "apply via
pgAdmin" step from the original request is performed with
`docker exec ... psql` instead, which is the identical DDL operation.
If a pgAdmin GUI is later required, that is a separate setup step out
of scope here.

The catalog DB is named `_ermrest_catalog_<catalog_id>` (e.g. the
existing catalog 1 is `_ermrest_catalog_1`). The test creates a new
catalog, so its DB name is derived from the new catalog id at runtime —
**never hard-coded.**

## 3. Approach

A five-phase pipeline. Phase 2 records the baseline plan (no index
available); Phases 4–5 apply the indexes and prove the planner uses
them.

Three distinct kinds of evidence, with different coverage:
- **Existence / contract verification** (Phase 5, `pg_indexes`) covers
  **every** association table the script reported — this is the
  pass/fail gate for the stated success criterion.
- **Usability evidence** (Phase 5, `EXPLAIN` with `enable_seqscan =
  off`) covers a **representative subset** (≥1 feature association, ≥1
  dataset-member association, each on live + history, both directions)
  — it proves the planner *chooses our named index* to satisfy the
  query when a seq scan is taken off the table. This is the
  "indexes are being used" check, and it holds regardless of row count.
- **Real-query touch counter** (Phase 5, `pg_stat_user_indexes`) — run
  the representative queries normally (no planner override) and read
  `idx_scan` before/after to confirm a genuine query incremented the
  index's scan counter. Belt-and-suspenders on top of the plan
  inspection.

### Phase 1 — Build the test catalog

Use the one-call demo-catalog entry point, fully populated so the
standard deriva-ml association tables exist with rows in both the live
and history tables:

```python
from deriva_ml.demo_catalog import create_demo_catalog
catalog = create_demo_catalog(
    hostname="localhost",
    create_features=True,    # creates feature association tables
    create_datasets=True,    # creates Dataset-member association tables
    on_exit_delete=False,    # keep it alive across the test run
)
catalog_id = catalog.catalog_id
```

`create_demo_catalog` clones deriva-ml from GitHub (network is
available), installs the `deriva-ml` schema + the `demo-schema` domain
schema, and runs an execution that populates subjects/images, features,
and datasets. The resulting catalog contains the association tables the
script targets: dataset-member associations, feature-value
associations, and `{Asset}_Execution` link tables.

Run with the dirty-tree override so the populate execution is allowed:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "<phase-1 snippet>"
```

The new catalog id and its Postgres DB name (`_ermrest_catalog_<id>`)
are captured and reused for every later phase. Because RIDs and catalog
ids are opaque, both are read back from the live objects — never
written as literals.

### Phase 2 — Record the baseline plan (no index yet)

Enumerate the association tables the script *will* target by running
the script's own discovery logic (import `_walk_associations` and
`_build_rid_lookup` from the script, or just run the script and read
its output header). For the representative subset (≥1 feature
association, ≥1 dataset-member association) capture, via
`docker exec deriva-postgres psql -d _ermrest_catalog_<id>`, the plan
for each query the index targets:

- **Live forward:** `EXPLAIN` of a query with an equality predicate on
  `(FK1, FK2)`.
- **Live reverse:** same on `(FK2, FK1)`.
- **History forward/reverse:** `EXPLAIN` of the equivalent
  `rowdata->>'<col_rid>'` predicates on
  `_ermrest_history."t<table_rid>"`.

This is the **baseline**: with no composite index present, every plan
should be a Seq Scan (possibly with a Filter). Recording it makes the
Phase-5 contrast unambiguous — the *same* query, *same* predicate, goes
from Seq Scan (now) to Index Scan on our named index (after apply).

The point of Phase 2 is **not** timing. `EXPLAIN` without `ANALYZE` is
enough to see the plan shape; we capture the node type, not the ms.
The exact predicate values (RIDs) are pulled from the populated
catalog, not invented.

### Phase 3 — Run the generator

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/generate_association_indexes.py \
    --hostname localhost \
    --catalog-id <id> \
    --output /tmp/assoc-indexes-<id>.sql \
    --verbose
```

Assertions on the output:
- Exit code 0; the stderr summary reports a non-zero
  "associations found" count.
- The `.sql` file contains, per association, the expected `CREATE INDEX
  CONCURRENTLY IF NOT EXISTS` statements — 2 live composite-pair
  (`_assoc_fwd_idx`, `_assoc_rev_idx`) and, when table+column RIDs
  resolved, 2 history expression (`_hist_assoc_fwd_idx`,
  `_hist_assoc_rev_idx`).
- Any warnings (RID-lookup misses) are surfaced and explained — a
  miss is a finding, not silently ignored.

### Phase 4 — Apply the SQL

`CREATE INDEX CONCURRENTLY` **cannot run inside a transaction block.**
`psql -f` wraps a file in an implicit transaction only with
`--single-transaction`; the default is autocommit per statement, which
is what we want. Apply with:

```bash
docker cp /tmp/assoc-indexes-<id>.sql deriva-postgres:/tmp/idx.sql
docker exec deriva-postgres psql -U ermrest -d _ermrest_catalog_<id> \
    -v ON_ERROR_STOP=1 -f /tmp/idx.sql
```

`ON_ERROR_STOP=1` makes any failing statement abort the run with a
non-zero exit so failures are not missed. `IF NOT EXISTS` keeps a
re-run a safe no-op (we run it twice to confirm idempotence — second
run produces zero new indexes and exits clean).

### Phase 5 — Verify the indexes exist and are used

**(a) Existence / contract verification (the core success criterion).**
For every association the script reported, query `pg_indexes` to assert
all four indexes exist:

```sql
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE indexname LIKE '%\_assoc\_fwd\_idx'
   OR indexname LIKE '%\_assoc\_rev\_idx'
   OR indexname LIKE '%\_hist\_assoc\_fwd\_idx'
   OR indexname LIKE '%\_hist\_assoc\_rev\_idx'
ORDER BY tablename, indexname;
```

Cross-check the count: `4 × (associations with resolved RIDs) +
2 × (associations missing history RIDs)` indexes present. Confirm both
directions exist on **both** the live table and the
`_ermrest_history."t<rid>"` table for each association — this is the
literal restatement of the success criterion in the request.

**(b) Usability — does the planner pick the index?** This is the
"indexes are being used" check that replaces performance measurement.
For each query in the representative subset:

1. Re-run the Phase-2 `EXPLAIN`. With a few rows the planner may still
   prefer a Seq Scan purely on cost — that is expected and is **not** a
   failure, because at this scale a seq scan genuinely is cheaper.
2. Then take the seq scan off the table and re-EXPLAIN:

   ```sql
   SET enable_seqscan = off;
   SET enable_bitmapscan = off;   -- so we see a plain Index Scan, not a fallback
   EXPLAIN <the same query>;
   ```

   **Pass:** the plan is an `Index Scan` / `Index Only Scan` whose
   `Index Cond` names *our* index (`..._assoc_fwd_idx` etc.) and
   matches both predicate columns in order. This proves the index is
   eligible for the query it was built for — true at any row count.
   **Fail (a real finding):** the plan is *still* a Seq Scan even with
   seqscan disabled, which means the index does not cover the query
   (wrong column order, wrong expression on the history side, etc.).

Record the disabled-seqscan plan for every direction × {live, history}
in the subset. A small results table maps each
(association, direction, table-kind) → index name → "Index Scan ✓ /
Seq Scan ✗".

**(c) Real-query touch counter (corroboration).** Reset stats, run the
representative queries *normally* (no planner overrides) against a
copy/scaled context if available, and read `idx_scan` from
`pg_stat_user_indexes` for the named indexes:

```sql
SELECT indexrelname, idx_scan
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%\_assoc\_%\_idx'
ORDER BY indexrelname;
```

A non-zero `idx_scan` after a real query confirms the index was
actually consulted, independent of the EXPLAIN override. (If at demo
scale every natural query stays on a seq scan, this counter may legitimately
read zero — in that case (b)'s forced-plan proof stands as the
usability evidence, and we say so plainly rather than implying a touch
that didn't happen.)

## 4. Deliverable

A short report (written to
`docs/audits/2026-06-02-association-index-test-report.md`) containing:
- catalog id used and association tables discovered,
- the generated `.sql` (or a link to it),
- the `pg_indexes` end-state table proving 4-per-association coverage
  on live + history, both directions (the pass/fail gate),
- the **index-usage table**: for each representative
  (association, direction, table-kind), the baseline plan (Seq Scan)
  next to the forced-plan result (Index Scan on our named index ✓ /
  still Seq Scan ✗),
- `pg_stat_user_indexes.idx_scan` readings where a natural query
  touched an index,
- idempotence-rerun result,
- any script bugs / warnings found.

## 5. Non-goals

- Modifying `generate_association_indexes.py` (unless the test exposes
  a bug — then a fix goes through the normal PR workflow, separately).
- Driving the apply through a pgAdmin GUI (psql-in-container is the
  equivalent; GUI is a separate setup if wanted).
- **Performance / timing measurement of any kind.** The demo catalog
  is too small for ms deltas to mean anything; the test deliberately
  makes the weaker-but-true claim ("the planner uses the index when a
  seq scan is disabled") instead of an unsupportable performance claim.
  No `EXPLAIN ANALYZE` timings are reported as evidence.
- Leaving the test catalog behind: it is deleted at the end unless the
  user wants to keep it for inspection.

## 6. Risks / open points

- **`create_demo_catalog` git-clone dependency.** It shells out to
  `git clone` of deriva-ml from GitHub. If the network blocks that, the
  fallback is the `CatalogManager`/`create_ml_catalog` +
  `create_domain_schema` path used by the test fixtures, which builds
  the same association tables without a clone.
- **Postgres role for DDL.** Indexes are created as the `ermrest` role
  (the catalog owner). `docker exec ... -U ermrest` runs as that role,
  which owns the `_ermrest_history` tables, so `CREATE INDEX` there is
  permitted.
- **History-table index on `rowdata->>` is an expression index** keyed
  by **column RID**, not column name. Verification reads `indexdef`
  from `pg_indexes` and confirms the expression references the correct
  RID — caught by comparing against the script's emitted SQL.

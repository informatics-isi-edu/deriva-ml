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
`_ermrest_history` partner**. Along the way, capture before/after
query-plan evidence that the composite indexes are actually usable by
the Postgres planner.

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

A five-phase pipeline. Phases 2 and 5 (before/after EXPLAIN) bracket
the apply.

Two distinct kinds of evidence, with different coverage:
- **Contract verification** (Phase 5, `pg_indexes`) covers **every**
  association table the script reported — this is the pass/fail gate.
- **Plan evidence** (Phase 2 + Phase 5 EXPLAIN) covers a
  **representative subset** (≥1 feature association, ≥1 dataset-member
  association) — this demonstrates usability, it is not exhaustive.

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

### Phase 2 — Capture "before" plans

Enumerate the association tables the script *will* target by running
the script's own discovery logic (import `_walk_associations` and
`_build_rid_lookup` from the script, or just run the script and read
its output header). For a representative subset (at minimum: one
feature association and one dataset-member association) capture, via
`docker exec deriva-postgres psql -d _ermrest_catalog_<id>`:

- **Live forward join:** `EXPLAIN (ANALYZE, BUFFERS)` of a query
  filtering/joining on `(FK1, FK2)`.
- **Live reverse join:** same on `(FK2, FK1)`.
- **History forward/reverse:** `EXPLAIN (ANALYZE, BUFFERS)` of the
  equivalent `rowdata->>'<col_rid>'` predicates on
  `_ermrest_history."t<table_rid>"`.

Expectation before indexing: sequential scans / filters (no composite
index available). The small row count means timings are sub-millisecond
— **the evidence is the plan node type (Seq Scan vs Index Scan) and
buffer counts, not wall-clock ms.** This is recorded honestly.

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

### Phase 5 — Verify end-state + capture "after" plans

**Contract verification (the core success criterion).** For every
association the script reported, query `pg_indexes` to assert all four
indexes exist:

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

**After-plans.** Re-run the Phase-2 EXPLAIN queries. Expect the plan to
switch to an Index Scan / Index Only Scan on the new composite index
(or at minimum the index appears as a candidate and buffer reads drop).
Record before vs after side by side.

A subtlety: with tiny tables the planner may *still* choose a seq scan
because it's cheaper than an index scan for a handful of rows. If so,
force index usage for the demonstration with
`SET enable_seqscan = off;` before the EXPLAIN, proving the index is
*usable* even when the cost model wouldn't pick it at this scale. This
is documented as the reason, not hidden.

## 4. Deliverable

A short report (written to
`docs/audits/2026-06-02-association-index-test-report.md`) containing:
- catalog id used and association tables discovered,
- the generated `.sql` (or a link to it),
- before/after EXPLAIN plans for the representative joins,
- the `pg_indexes` end-state table proving 4-per-association coverage
  on live + history, both directions,
- idempotence-rerun result,
- any script bugs / warnings found.

## 5. Non-goals

- Modifying `generate_association_indexes.py` (unless the test exposes
  a bug — then a fix goes through the normal PR workflow, separately).
- Driving the apply through a pgAdmin GUI (psql-in-container is the
  equivalent; GUI is a separate setup if wanted).
- Large-scale timing benchmarks. The demo catalog is small by design;
  plan-shape is the evidence. (A synthetic bulk-load for real ms
  deltas was considered and deferred — see decision in §3 Phase 2.)
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

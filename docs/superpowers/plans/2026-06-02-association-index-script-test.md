# Association-Index Generator Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE ON FORM:** This is a **runbook**, not a feature build. The "code" is a sequence of commands run against a live Dockerized catalog; outputs of one phase (catalog id, table names, RIDs, plans) feed the next. There is no library code to TDD. Each task therefore reads: *run command → capture output → assert gate → record in report*. Do not skip the assert/record steps — they are the deliverable.

**Goal:** Prove end-to-end that `scripts/generate_association_indexes.py`, when run against a populated catalog and applied via psql, leaves every pure binary association table with composite indexes in both directions on both the live and `_ermrest_history` tables, and that the Postgres planner will use those indexes.

**Architecture:** Build a fresh populated demo catalog in the local Docker stack → record baseline query plans (Seq Scan) → run the generator to emit `.sql` → apply with `docker exec ... psql` → verify index existence via `pg_indexes` and usability via `EXPLAIN` with `enable_seqscan=off`. No production code changes.

**Tech Stack:** Python (deriva-ml demo-catalog API), the script under test, `docker exec deriva-postgres psql` (postgres:16.13), bash.

**Spec:** `docs/superpowers/specs/2026-06-02-association-index-script-test-design.md`

---

## Conventions used throughout

- **Always chain `cd`** into the project root in one Bash call:
  `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>`.
- **Never hard-code a catalog id, DB name, table name, or RID.** Every
  one is read back from a command and stored in the fixtures file
  (Task 1) or a later capture file. The Postgres DB name is always
  `_ermrest_catalog_${CATALOG_ID}`, derived — never typed.
- **psql access:** `docker exec deriva-postgres psql -U ermrest -d _ermrest_catalog_${CATALOG_ID}`.
  Use `-tA` for clean machine-readable output, `-c "<sql>"` for one-liners.
- **Working scratch dir:** `/tmp/assoc-idx-test/` holds the fixtures
  file, generated SQL, and captured plan output. Created in Task 1.
- **The report** is appended to incrementally at
  `docs/audits/2026-06-02-association-index-test-report.md`. Each task
  that produces evidence writes its section immediately, so a crash
  mid-run still leaves a partial record.

---

## Task 1: Scratch dir + report skeleton

**Files:**
- Create: `/tmp/assoc-idx-test/` (scratch, not committed)
- Create: `docs/audits/2026-06-02-association-index-test-report.md`

- [ ] **Step 0: Create the working branch FIRST (nothing lands on `main`)**

Per repo policy every commit — including this report — goes through a
PR off a branch. Create it before any commit in this plan:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git checkout -b test/association-index-validation && echo "on $(git branch --show-current)"
```
Expected: `on test/association-index-validation`. All later commits in
Tasks 1–8 land here; Task 8 Step 4 only pushes + opens the PR.

- [ ] **Step 1: Create the scratch directory**

```bash
mkdir -p /tmp/assoc-idx-test && echo created
```

- [ ] **Step 2: Write the report skeleton**

Create `docs/audits/2026-06-02-association-index-test-report.md`:

```markdown
# Association-Index Generator — Test Report

**Date:** 2026-06-02
**Script under test:** `scripts/generate_association_indexes.py`
**Plan:** `docs/superpowers/plans/2026-06-02-association-index-script-test.md`
**Environment:** local Docker Deriva stack (`deriva-postgres` postgres:16.13, `deriva-webserver` at https://localhost).

## 1. Catalog under test

_(filled in Task 2)_

## 2. Associations discovered

_(filled in Task 4)_

## 3. Baseline plans (no index)

_(filled in Task 3)_

## 4. Generated SQL

_(filled in Task 4)_

## 5. Apply result + idempotence

_(filled in Task 5)_

## 6. Index existence (contract gate)

_(filled in Task 6)_

## 7. Index usability (planner uses the index)

_(filled in Task 7)_

## 8. Findings / bugs

_(filled in Task 8)_
```

- [ ] **Step 3: Commit the skeleton**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): report skeleton for index-generator validation"
```

---

## Task 2: Build the populated demo catalog

**Files:**
- Create: `/tmp/assoc-idx-test/fixtures.sh` (runtime values: catalog id, db name)

- [ ] **Step 1: Verify the stack is up**

```bash
docker ps --format '{{.Names}}' | grep -E 'deriva-postgres|deriva-webserver'
```
Expected: both names print. If either is missing, STOP — the stack
must be running (the catalog HTTP endpoint and Postgres are both
required).

- [ ] **Step 2: Create + populate the catalog, capture its id**

This writes the catalog id to the fixtures file. `create_demo_catalog`
clones deriva-ml from GitHub, installs the deriva-ml + demo schema, and
runs a populate execution; `create_features=True` and
`create_datasets=True` produce the feature and dataset-member
association tables. `on_exit_delete=False` keeps the catalog alive
after the Python process exits.

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY'
from pathlib import Path
from deriva_ml.demo_catalog import create_demo_catalog

catalog = create_demo_catalog(
    hostname="localhost",
    create_features=True,
    create_datasets=True,
    on_exit_delete=False,
)
cid = str(catalog.catalog_id)
Path("/tmp/assoc-idx-test/fixtures.sh").write_text(
    f'export CATALOG_ID="{cid}"\n'
    f'export PG="docker exec deriva-postgres psql -U ermrest -d _ermrest_catalog_{cid}"\n'
)
print("CATALOG_ID", cid)
PY
```
Expected: prints `CATALOG_ID <n>` (n is opaque — do not assume a
value). Build takes ~1–3 min (clone + populate). If the GitHub clone
fails (network), see the fallback note at the end of this task.

- [ ] **Step 3: Sanity-check Postgres sees the catalog DB**

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
docker exec deriva-postgres psql -U ermrest -lqt | cut -d'|' -f1 | grep -qw "_ermrest_catalog_${CATALOG_ID}" && \
echo "DB _ermrest_catalog_${CATALOG_ID} exists"
```
Expected: prints the "exists" line.

- [ ] **Step 4: Record catalog under test in the report**

Append the catalog id and the `https://localhost/ermrest/catalog/<id>`
URL to section 1 of the report. (Read the value from
`/tmp/assoc-idx-test/fixtures.sh`, write it into the markdown.)

- [ ] **Step 5: Commit the report update**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): record catalog under test"
```

**Fallback (only if Step 2's GitHub clone fails):** build the catalog
without the clone via the fixture path —
`create_ml_catalog("localhost", project_name="ml-test")` then
`create_domain_schema(catalog, "demo-schema")` then a `DerivaML`
populate execution as in `tests/catalog_manager.py`. Same association
tables result. Record in the report which path was used.

---

## Task 3: Capture baseline plans (no index yet)

This MUST run before the generator/apply so the "before" is a true
no-index baseline. We need real predicate values first.

**Files:**
- Create: `/tmp/assoc-idx-test/baseline-plans.txt`

- [ ] **Step 1: Pick one feature association and one dataset-member association**

List candidate association tables straight from Postgres (the live
deriva-ml association tables live in the `deriva-ml` schema; feature
associations and dataset-member tables are there). Print row counts so
we pick tables that actually have rows:

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
eval "$PG" -tA -c "
  SELECT n.nspname, c.relname, c.reltuples::bigint
  FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
  WHERE c.relkind='r' AND n.nspname='deriva-ml'
  ORDER BY c.relname;"
```
Expected: a list of `deriva-ml|<Table>|<rowcount>` lines. Choose:
- one **feature** association (name contains a feature, e.g. an
  `Image_<Feature>` link or a `..._Image` / `..._Subject` value table
  that `is_association` will flag), and
- one **dataset-member** association (e.g. `Dataset_Image`,
  `Dataset_Subject`, or `Dataset_Dataset`),

each with `reltuples > 0`. Record the two chosen `(schema, table)`
pairs. **Do not invent names** — copy them from this output.

> Note: the authoritative list of what the *script* targets comes in
> Task 4 (the generator's own discovery). Here we only need two
> populated association tables to baseline; Task 6 cross-checks the two
> we picked are in the generated set.

- [ ] **Step 2: Find the two FK columns + a real value pair for each chosen table**

For each chosen live table, get its FK column names and one existing
row's value pair (so the predicate matches real data):

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
TBL="<ChosenTable>" && \
eval "$PG" -tA -c "
  SELECT string_agg(a.attname, ',' ORDER BY a.attnum)
  FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
  JOIN pg_namespace n ON n.oid=c.relnamespace
  WHERE n.nspname='deriva-ml' AND c.relname='$TBL'
    AND a.attnum>0 AND NOT a.attisdropped
    AND a.attname LIKE '%RID';" && \
eval "$PG" -tA -c 'SELECT * FROM "deriva-ml"."'$TBL'" LIMIT 1;'
```
Expected: the FK column list (the two `*_RID` columns) and one sample
row. Record the two FK column names (call them `FK1`, `FK2`) and the
concrete value pair from the sample row. These are read from the
catalog — never literals.

- [ ] **Step 3: EXPLAIN the four targeted queries per chosen table (baseline)**

For each chosen live table, with its real `FK1=v1`, `FK2=v2`:

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
{
  echo "=== LIVE $TBL forward (FK1,FK2) ==="
  eval "$PG" -c 'EXPLAIN SELECT 1 FROM "deriva-ml"."'$TBL'" WHERE "FK1"='\''v1'\'' AND "FK2"='\''v2'\'';'
  echo "=== LIVE $TBL reverse (FK2,FK1) ==="
  eval "$PG" -c 'EXPLAIN SELECT 1 FROM "deriva-ml"."'$TBL'" WHERE "FK2"='\''v2'\'' AND "FK1"='\''v1'\'';'
} | tee -a /tmp/assoc-idx-test/baseline-plans.txt
```
Replace `FK1`/`FK2`/`v1`/`v2` with the real names/values from Step 2.
Expected: each plan is a **Seq Scan** (possibly with a Filter) — no
index named `*_assoc_*_idx` yet (none exist). If any baseline plan is
already an Index Scan on such a name, STOP: a prior run left indexes
behind; drop them or use a fresh catalog before continuing.

- [ ] **Step 4: EXPLAIN the history-side baseline**

Get the table RID and the two column RIDs (history rowdata is keyed by
column RID), then EXPLAIN the rowdata predicate. The script's
`_build_rid_lookup` reads these from `/schema`; here we read them from
Postgres `_ermrest.known_tables` / `known_columns`:

```bash
source /tmp/assoc-idx-test/fixtures.sh && TBL="<ChosenTable>" && \
TRID=$(eval "$PG" -tA -c "SELECT \"RID\" FROM _ermrest.known_tables WHERE table_name='$TBL' AND schema_name='deriva-ml';") && \
echo "table RID: $TRID" && \
eval "$PG" -tA -c "SELECT \"RID\", column_name FROM _ermrest.known_columns WHERE table_rid='$TRID' AND column_name LIKE '%RID';" && \
eval "$PG" -c 'EXPLAIN SELECT 1 FROM _ermrest_history."t'$TRID'" WHERE (rowdata->>'\''<col1_rid>'\'')='\''v1'\'' AND (rowdata->>'\''<col2_rid>'\'')='\''v2'\'';' | tee -a /tmp/assoc-idx-test/baseline-plans.txt
```
Expected: prints the table RID and the column-name→RID mapping, then a
**Seq Scan** plan on `t<RID>`. Substitute the real `<col1_rid>`,
`<col2_rid>`, `v1`, `v2`. Record the RID mapping — Task 7 reuses it.

> If `_ermrest.known_tables` is not queryable in this server build,
> fall back to reading the RIDs from the script's generated SQL header
> comments (Task 4 emits `Table RID:` and the column RIDs inline).

- [ ] **Step 5: Record baselines in the report**

Paste the `baseline-plans.txt` content into report section 3, noting
each plan is a Seq Scan. Commit:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): record baseline Seq Scan plans"
```

---

## Task 4: Run the generator

**Files:**
- Create: `/tmp/assoc-idx-test/indexes.sql`

- [ ] **Step 1: Run the script against the catalog**

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/generate_association_indexes.py \
    --hostname localhost \
    --catalog-id "${CATALOG_ID}" \
    --output /tmp/assoc-idx-test/indexes.sql \
    --verbose ; echo "exit=$?"
```
Expected: `exit=0`; stderr ends with `Wrote <N> association block(s)`
where N > 0. Any `Warnings:` lines (RID-lookup misses) are captured —
they are findings for Task 8, not failures.

- [ ] **Step 2: Assert the SQL shape**

```bash
echo "associations (blocks):" ; grep -c '^-- ===' /tmp/assoc-idx-test/indexes.sql ; \
echo "live fwd:" ; grep -c '_assoc_fwd_idx' /tmp/assoc-idx-test/indexes.sql ; \
echo "live rev:" ; grep -c '_assoc_rev_idx' /tmp/assoc-idx-test/indexes.sql ; \
echo "hist fwd:" ; grep -c '_hist_assoc_fwd_idx' /tmp/assoc-idx-test/indexes.sql ; \
echo "hist rev:" ; grep -c '_hist_assoc_rev_idx' /tmp/assoc-idx-test/indexes.sql ; \
echo "CONCURRENTLY:" ; grep -c 'CREATE INDEX CONCURRENTLY IF NOT EXISTS' /tmp/assoc-idx-test/indexes.sql
```
Expected: per association the script emits 2 live indexes always, and 2
history indexes when RIDs resolved. So `live fwd == live rev == blocks`,
and `hist fwd == hist rev == (blocks − history-skipped-warnings)`. Note
the exact counts — Task 6 reconciles them against `pg_indexes`.

- [ ] **Step 3: Confirm the two Task-3 tables are in the generated set**

```bash
grep -E '<ChosenTable1>|<ChosenTable2>' /tmp/assoc-idx-test/indexes.sql | grep '^-- '
```
Expected: both chosen table names appear as block headers. If a chosen
table is absent, it wasn't a pure binary association — pick a different
one for the usability check in Task 7 (any table that *is* in the
generated set and has rows).

- [ ] **Step 4: Record the generated SQL in the report**

Embed (or link) `indexes.sql` into report section 4 and list the
discovered associations (block headers) into section 2. Commit:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): record generated SQL and discovered associations"
```

---

## Task 5: Apply the SQL (and prove idempotence)

- [ ] **Step 1: Copy the SQL into the postgres container**

```bash
docker cp /tmp/assoc-idx-test/indexes.sql deriva-postgres:/tmp/indexes.sql && echo copied
```

- [ ] **Step 2: Apply it**

`CREATE INDEX CONCURRENTLY` cannot run inside a transaction block.
Default psql autocommits each statement, so do **not** pass
`--single-transaction`. `ON_ERROR_STOP=1` makes any failure abort
loudly.

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
docker exec deriva-postgres psql -U ermrest -d "_ermrest_catalog_${CATALOG_ID}" \
    -v ON_ERROR_STOP=1 -f /tmp/indexes.sql ; echo "exit=$?"
```
Expected: `exit=0`; output is a series of `CREATE INDEX` lines (first
run actually creates them).

- [ ] **Step 3: Re-apply to prove idempotence**

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
docker exec deriva-postgres psql -U ermrest -d "_ermrest_catalog_${CATALOG_ID}" \
    -v ON_ERROR_STOP=1 -f /tmp/indexes.sql 2>&1 | tee /tmp/assoc-idx-test/reapply.txt ; \
grep -c 'already exists, skipping' /tmp/assoc-idx-test/reapply.txt
```
Expected: `exit=0` again; every statement now reports
`NOTICE: relation "..." already exists, skipping` (the
`IF NOT EXISTS` no-op). The skip count equals the number of CREATE
INDEX statements in the file.

- [ ] **Step 4: Record apply + idempotence in the report**

Write the first-apply result and the all-skipped re-apply into report
section 5. Commit:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): record apply result and idempotent re-run"
```

---

## Task 6: Verify index existence (the contract gate)

This is the pass/fail gate for the stated success criterion: every
association has composite indexes in **both** directions on **both** the
live table and its `_ermrest_history` table.

- [ ] **Step 1: List all created indexes from `pg_indexes`**

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
eval "$PG" -c "
  SELECT schemaname, tablename, indexname, indexdef
  FROM pg_indexes
  WHERE indexname LIKE '%\_assoc\_fwd\_idx'
     OR indexname LIKE '%\_assoc\_rev\_idx'
     OR indexname LIKE '%\_hist\_assoc\_fwd\_idx'
     OR indexname LIKE '%\_hist\_assoc\_rev\_idx'
  ORDER BY tablename, indexname;" | tee /tmp/assoc-idx-test/pg_indexes.txt
```
Expected: rows for live (`schemaname='deriva-ml'` or the domain schema)
and history (`schemaname='_ermrest_history'`) indexes.

- [ ] **Step 2: Reconcile the counts against the generated SQL**

```bash
echo "live fwd in pg:" ; grep -c '_assoc_fwd_idx' /tmp/assoc-idx-test/pg_indexes.txt ; \
echo "live rev in pg:" ; grep -c '_assoc_rev_idx' /tmp/assoc-idx-test/pg_indexes.txt ; \
echo "hist fwd in pg:" ; grep -c '_hist_assoc_fwd_idx' /tmp/assoc-idx-test/pg_indexes.txt ; \
echo "hist rev in pg:" ; grep -c '_hist_assoc_rev_idx' /tmp/assoc-idx-test/pg_indexes.txt
```
Expected — each count equals the corresponding count from Task 4 Step 2:
- live fwd (pg) == live rev (pg) == number of association blocks,
- hist fwd (pg) == hist rev (pg) == blocks minus any history-skipped
  warnings.

**GATE:** if any count is short, the contract is NOT met — record which
association/direction/table-kind is missing in section 8 and stop
(this is a real script bug or apply failure, not a test artifact).

- [ ] **Step 3: Confirm both-directions-on-both-tables per association**

For each association block header in `indexes.sql`, confirm all four
(or two, if history was legitimately skipped) names are present in
`pg_indexes.txt`. A short cross-check:

```bash
for t in $(grep '^--   Table RID' -B2 /tmp/assoc-idx-test/indexes.sql | grep '^-- [A-Za-z]' | sed 's/^-- //; s/ .*//'); do
  fwd=$(grep -c "${t}_assoc_fwd_idx" /tmp/assoc-idx-test/pg_indexes.txt)
  rev=$(grep -c "${t}_assoc_rev_idx" /tmp/assoc-idx-test/pg_indexes.txt)
  hfwd=$(grep -c "${t}_hist_assoc_fwd_idx" /tmp/assoc-idx-test/pg_indexes.txt)
  hrev=$(grep -c "${t}_hist_assoc_rev_idx" /tmp/assoc-idx-test/pg_indexes.txt)
  echo "$t live(fwd=$fwd rev=$rev) hist(fwd=$hfwd rev=$hrev)"
done
```
Expected: every line shows `live(fwd=1 rev=1)` and `hist(fwd=1 rev=1)`
— or `hist(fwd=0 rev=0)` only for associations the script explicitly
warned it skipped (truncated names may not substring-match; for those
fall back to comparing the truncated name from `indexes.sql`).

- [ ] **Step 4: Record existence verification in the report**

Paste `pg_indexes.txt` and the per-association cross-check into section
6, with an explicit PASS/FAIL on the contract. Commit:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): verify index existence contract (both dirs, live+history)"
```

---

## Task 7: Verify the planner uses the indexes

The "indexes are being used" check — holds at any row count.

- [ ] **Step 1: Re-run the two live baseline EXPLAINs (natural plan)**

Re-run the exact Task-3 Step-3 EXPLAINs (same tables, same real values).
Expected: at demo scale the planner may *still* show Seq Scan purely on
cost — that is **acceptable, not a failure**. Record the natural plan.

- [ ] **Step 2: Force the planner off seq scans and re-EXPLAIN (live)**

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
eval "$PG" -c "
  SET enable_seqscan = off;
  SET enable_bitmapscan = off;
  EXPLAIN SELECT 1 FROM \"deriva-ml\".\"<ChosenTable>\" WHERE \"FK1\"='v1' AND \"FK2\"='v2';" \
  | tee -a /tmp/assoc-idx-test/usage-plans.txt
```
**PASS:** the plan is an `Index Scan` / `Index Only Scan` and the
`Index Cond` names `<ChosenTable>_assoc_fwd_idx` (or `_rev_idx` for the
reversed predicate). **FAIL (finding):** still a Seq Scan with seqscan
disabled → the index does not cover the query; record in section 8.
Run for both forward and reverse predicate orderings.

- [ ] **Step 3: Force-plan check on the history side**

Reuse the table RID + column RIDs from Task 3 Step 4:

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
eval "$PG" -c "
  SET enable_seqscan = off;
  SET enable_bitmapscan = off;
  EXPLAIN SELECT 1 FROM _ermrest_history.\"t<TRID>\"
    WHERE (rowdata->>'<col1_rid>')='v1' AND (rowdata->>'<col2_rid>')='v2';" \
  | tee -a /tmp/assoc-idx-test/usage-plans.txt
```
**PASS:** `Index Scan` whose `Index Cond` references the expression
index `t<TRID>_hist_assoc_fwd_idx` (or the truncated name). **FAIL:**
still Seq Scan → the history expression index doesn't match the query
(e.g. emitted expression differs from the predicate); record in
section 8. Run forward and reverse.

- [ ] **Step 4: Corroborate with `pg_stat_user_indexes` (best-effort)**

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
eval "$PG" -c "
  SELECT indexrelname, idx_scan
  FROM pg_stat_user_indexes
  WHERE indexrelname LIKE '%\_assoc\_%\_idx'
  ORDER BY indexrelname;"
```
Expected: indexes touched by the forced-plan EXPLAINs above (EXPLAIN
without ANALYZE does **not** execute, so to actually bump the counter,
re-run the Step-2/3 queries as `EXPLAIN ANALYZE` or plain `SELECT` once
with seqscan disabled). A non-zero `idx_scan` corroborates usage; if it
reads zero at demo scale, note that the forced-plan proof in Steps 2–3
stands as the usability evidence and say so plainly.

- [ ] **Step 5: Record usability in the report**

Build the index-usage table in section 7: rows of
`(association, direction, table-kind) → index name → Index Scan ✓ / Seq Scan ✗`,
plus the `idx_scan` readings. Commit:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): verify planner uses generated indexes (forced-plan proof)"
```

---

## Task 8: Findings, cleanup, and PR

- [ ] **Step 1: Write findings**

In section 8, list: any generator warnings, any FAIL gates from Task 6
(missing indexes) or Task 7 (index not picked even with seqscan off),
truncated-name collisions, or "all clean". If a real script bug was
found, note it and open a tracking issue / plan a separate fix PR — do
**not** fix the script in this test PR.

- [ ] **Step 2: Decide catalog disposition**

Ask the user whether to keep the test catalog for inspection or delete
it. To delete:

```bash
source /tmp/assoc-idx-test/fixtures.sh && \
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run python - <<PY
from deriva.core import ErmrestCatalog, get_credential
cat = ErmrestCatalog("https", "localhost", "${CATALOG_ID}", credentials=get_credential("localhost"))
cat.delete_ermrest_catalog(really=True)
print("deleted ${CATALOG_ID}")
PY
```
Record the disposition in the report.

- [ ] **Step 3: Final report commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/audits/2026-06-02-association-index-test-report.md && \
git commit -m "test(assoc-idx): findings and catalog disposition"
```

- [ ] **Step 4: Push the branch and open the PR**

The branch `test/association-index-validation` already exists (created
in Task 1 Step 0) with all the report commits on it. Just push + open:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git push -u origin test/association-index-validation && \
gh pr create --title "test: validate generate_association_indexes end-to-end" \
  --body "Runs scripts/generate_association_indexes.py against a populated demo catalog, applies the SQL via psql, and verifies (1) every association has composite indexes in both directions on both the live and _ermrest_history tables, and (2) the planner uses them (forced-plan proof). Report in docs/audits/2026-06-02-association-index-test-report.md."
```
Expected: PR URL printed.

---

## Self-review note

- **Branch-first ordering:** resolved — Task 1 Step 0 creates the
  branch before any commit; Tasks 1–8 commit there; Task 8 Step 4
  pushes + opens the PR. Nothing lands on `main`.
- **No hard-coded RIDs/ids:** every catalog id, table name, RID, and
  value is read from a command and substituted; placeholders like
  `<ChosenTable>`, `v1`, `<TRID>` are explicitly "fill from prior step
  output," consistent with the RID-opacity rule.
- **Spec coverage:** §3 Phase 1→Task 2, Phase 2→Task 3, Phase 3→Task 4,
  Phase 4→Task 5, Phase 5(a) existence→Task 6, 5(b) usability→Task 7,
  5(c) idx_scan→Task 7 Step 4; idempotence→Task 5 Step 3; deliverable
  report→all tasks; cleanup/disposition→Task 8.

# Association-Index Generator — Test Report

**Date:** 2026-06-02
**Script under test:** `scripts/generate_association_indexes.py`
**Plan:** `docs/superpowers/plans/2026-06-02-association-index-script-test.md`
**Environment:** local Docker Deriva stack (`deriva-postgres` postgres:16.13, `deriva-webserver` at https://localhost).

## 1. Catalog under test

- **Catalog id:** `3` (`https://localhost/ermrest/catalog/3`)
- **Postgres DB:** `_ermrest_catalog_3`
- **Built with:** `create_demo_catalog(hostname="localhost", create_features=True, create_datasets=True, on_exit_delete=False)`
- **Schemas:** `deriva-ml` (core ML) + `demo-schema` (domain).
- **Populated** — representative association tables and approximate row counts (post-`ANALYZE`):

  | Table | Kind | Rows |
  |-------|------|------|
  | `demo-schema.Execution_Subject_Health` | feature association | 12 |
  | `demo-schema.Execution_Image_Quality` | feature association | 12 |
  | `demo-schema.Execution_Image_BoundingBox` | feature association | 12 |
  | `demo-schema.Dataset_Subject` | dataset-member association | 8 |
  | `demo-schema.Dataset_Image` | dataset-member association | 4 |
  | `deriva-ml.Dataset_Dataset` | nested-dataset association | 6 |

- **Note (env):** an empty stray catalog `4`/`_ermrest_catalog_2` was
  created and partially cleaned during an auth probe before the real
  build; it contains no association tables and does not affect this
  test (the generator is run against catalog 3 explicitly). Flagged in
  §8.

> Auth note: catalog creation initially failed with HTTP 401; the
> `localhost` bearer token in `~/.deriva/credential.json` was refreshed
> by the user, after which `POST /ermrest/catalog` returned 201 and the
> build succeeded.

## 2. Associations discovered

The generator found **12 pure binary association tables** across the
`demo-schema` and `deriva-ml` schemas. All 12 resolved table + column
RIDs cleanly, so each got the full 4-index set (2 live + 2 history):

| # | Association | FK pair |
|---|-------------|---------|
| 1 | `demo-schema.BoundingBox_Asset_Type` | (Asset_Type, BoundingBox) |
| 2 | `demo-schema.Dataset_Image` | (Dataset, Image) |
| 3 | `demo-schema.Dataset_Subject` | (Dataset, Subject) |
| 4 | `demo-schema.Image_Asset_Type` | (Asset_Type, Image) |
| 5 | `deriva-ml.Dataset_Dataset` | (Dataset, Nested_Dataset) |
| 6 | `deriva-ml.Dataset_Dataset_Type` | (Dataset, Dataset_Type) |
| 7 | `deriva-ml.Dataset_Execution` | (Dataset, Execution) |
| 8 | `deriva-ml.Dataset_File` | (Dataset, File) |
| 9 | `deriva-ml.Execution_Asset_Asset_Type` | (Asset_Type, Execution_Asset) |
| 10 | `deriva-ml.Execution_Metadata_Asset_Type` | (Asset_Type, Execution_Metadata) |
| 11 | `deriva-ml.File_Asset_Type` | (Asset_Type, File) |
| 12 | `deriva-ml.Workflow_Workflow_Type` | (Workflow, Workflow_Type) |

(Feature-value tables like `Execution_Subject_Health` are absent —
correctly, as noted in §3, they are not pure binary associations.)

## 3. Baseline plans (no index)

**Representative pure-binary association chosen:**
`demo-schema.Dataset_Subject` — FK pair `(Dataset, Subject)`, table
RID `57G`, column RIDs `Dataset=57W`, `Subject=57Y`. History table
`_ermrest_history."t57G"` (8 rows). Real predicate values pulled from a
live row: `Dataset='5CT'`, `Subject='4C6'`.

> **Scope finding (carried to §8):** feature-value association tables
> like `demo-schema.Execution_Subject_Health` are **not** pure binary
> associations — they carry 4 user FKs (`Execution`, `Feature_Name`,
> `Subject`, `SubjectHealth`), so `Table.is_association()` (pure,
> binary) correctly rejects them and the script does **not** index
> them. The pure-binary associations the script targets are the
> dataset-member (`Dataset_Subject`, `Dataset_Image`), nested-dataset
> (`Dataset_Dataset`), and `{X}_Execution` / `{X}_Asset_Type` link
> tables. The second representative table for the usability check is
> taken from the script's own discovered set in §2/Task 4.

All four targeted queries are **Seq Scan** before indexing (the clean
"before" state):

```
-- LIVE Dataset_Subject forward (Dataset, Subject)
Seq Scan on "Dataset_Subject"  (cost=0.00..1.12 rows=1 width=4)
  Filter: (("Dataset" = '5CT') AND ("Subject" = '4C6'))

-- LIVE Dataset_Subject reverse (Subject, Dataset)
Seq Scan on "Dataset_Subject"  (cost=0.00..1.12 rows=1 width=4)
  Filter: (("Subject" = '4C6') AND ("Dataset" = '5CT'))

-- HIST t57G forward (rowdata->>'57W', rowdata->>'57Y')
Seq Scan on "t57G"  (cost=0.00..1.16 rows=1 width=4)
  Filter: (((rowdata ->> '57W') = '5CT') AND ((rowdata ->> '57Y') = '4C6'))

-- HIST t57G reverse (rowdata->>'57Y', rowdata->>'57W')
Seq Scan on "t57G"  (cost=0.00..1.16 rows=1 width=4)
  Filter: (((rowdata ->> '57Y') = '4C6') AND ((rowdata ->> '57W') = '5CT'))
```

## 4. Generated SQL

- **Command:** `uv run python scripts/generate_association_indexes.py --hostname localhost --catalog-id 3 --output /tmp/assoc-idx-test/indexes.sql --verbose`
- **Exit code:** 0. **Warnings:** none (only a harmless urllib3
  `InsecureRequestWarning` for the self-signed localhost cert — not a
  generator warning; no RID-lookup misses).
- **Shape:** 48 `CREATE INDEX CONCURRENTLY IF NOT EXISTS` statements =
  12 associations × 4 (live fwd, live rev, hist fwd, hist rev). Counts
  per direction: live fwd 12, live rev 12, hist fwd 12, hist rev 12.

**Cross-check — the `Dataset_Subject` block matches the RIDs this test
discovered independently from Postgres `_ermrest` metadata (table
`57G`, `Dataset=57W`, `Subject=57Y`):**

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS "Dataset_Subject_assoc_fwd_idx"
  ON "demo-schema"."Dataset_Subject" ("Dataset", "Subject");
CREATE INDEX CONCURRENTLY IF NOT EXISTS "Dataset_Subject_assoc_rev_idx"
  ON "demo-schema"."Dataset_Subject" ("Subject", "Dataset");
CREATE INDEX CONCURRENTLY IF NOT EXISTS "Dataset_Subject_hist_assoc_fwd_idx"
  ON _ermrest_history."t57G"
     (((rowdata->>'57W')), ((rowdata->>'57Y')));
CREATE INDEX CONCURRENTLY IF NOT EXISTS "Dataset_Subject_hist_assoc_rev_idx"
  ON _ermrest_history."t57G"
     (((rowdata->>'57Y')), ((rowdata->>'57W')));
```

Full file: `/tmp/assoc-idx-test/indexes.sql` (not committed — scratch).

## 5. Apply result + idempotence

Applied with
`docker exec deriva-postgres psql -U ermrest -d _ermrest_catalog_3 -v ON_ERROR_STOP=1 -f /tmp/indexes.sql`
(default psql autocommit — `CREATE INDEX CONCURRENTLY` is **not** wrapped
in a transaction, which it cannot be).

| Run | Exit | New indexes created | `already exists, skipping` | Errors |
|-----|------|---------------------|----------------------------|--------|
| First apply | 0 | 48 | 0 | 0 |
| Re-apply (idempotence) | 0 | 0 (all skipped) | 48 | 0 |

The re-run is a clean no-op: every `CREATE INDEX CONCURRENTLY IF NOT
EXISTS` reports `NOTICE: relation "…" already exists, skipping`. Safe
to re-run against a live catalog.

## 6. Index existence (contract gate) — ✅ PASS

`pg_indexes` confirms **48 indexes** = 12 associations × 4. Per-direction
totals: live fwd 12, live rev 12, hist fwd 12, hist rev 12. Schema
split: 24 in `_ermrest_history`, 24 live (16 `deriva-ml`, 8
`demo-schema`).

**Per-association cross-check — all 12 fully covered, both directions on
both live + history:**

```
BoundingBox_Asset_Type          live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Dataset_Image                   live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Dataset_Subject                 live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Image_Asset_Type                live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Dataset_Dataset                 live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Dataset_Dataset_Type            live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Dataset_Execution               live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Dataset_File                    live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Execution_Asset_Asset_Type      live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Execution_Metadata_Asset_Type   live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
File_Asset_Type                 live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
Workflow_Workflow_Type          live(fwd=1 rev=1) hist(fwd=1 rev=1)  OK
---
associations fully covered: 12 / 12   (incomplete: 0)
```

**This is the literal success criterion from the request:** every
association table has composite-key indexes in both directions on both
the live table and the history table. **Met.**

## 7. Index usability (planner uses the index) — ✅ PASS

Representative table: `demo-schema.Dataset_Subject` (live + history,
both directions). Method per the spec: natural plan, then forced plan
with `enable_seqscan=off` / `enable_bitmapscan=off`, then an
`idx_scan` counter check.

**Natural plan (no override):** still `Seq Scan` on both live and
history — **expected and not a failure**: at 8 rows a seq scan is
genuinely cheaper than an index lookup, so the cost model correctly
declines the index.

**Forced plan (`enable_seqscan=off`) — the usability proof:**

| Query | Table kind | Plan | Index used |
|-------|-----------|------|-----------|
| `Dataset='5CT' AND Subject='4C6'` | live | Index Only Scan | `Dataset_Subject_assoc_rev_idx` ✓ |
| `Subject='4C6' AND Dataset='5CT'` | live | Index Only Scan | `Dataset_Subject_assoc_rev_idx` ✓ |
| `rowdata->>'57W' AND rowdata->>'57Y'` | history | Index Scan | `Dataset_Subject_hist_assoc_rev_idx` ✓ |
| `rowdata->>'57Y' AND rowdata->>'57W'` | history | Index Scan | `Dataset_Subject_hist_assoc_rev_idx` ✓ |

Every query switches from Seq Scan to an **Index (Only) Scan on one of
our named composite indexes**, with the `Index Cond` matching both
predicate columns — including the JSONB `rowdata->>'<col_rid>'`
expressions on the history side, which is the non-trivial case (the
expression index must match the query's expression exactly to be
eligible). The history-side proof is the key result: the script's
RID-keyed expression indexes are real, correctly-formed, and
planner-usable.

> **Why `rev` for both orderings (not a defect):** a B-tree composite
> index supports equality predicates on its columns regardless of the
> order they appear in the `WHERE` clause, so both the `fwd` and `rev`
> indexes are eligible for either query; the planner picked `rev` for
> all four. Both directions' indexes exist (§6) and are usable; having
> both simply lets a *range/sort* on either leading column be served.

**`idx_scan` corroboration (independent of EXPLAIN):** counter
`0 → 1` after executing the live and history queries (via
`EXPLAIN ANALYZE`, which actually runs them):

```
Dataset_Subject_assoc_rev_idx        0 → 1
Dataset_Subject_hist_assoc_rev_idx   0 → 1
```
`EXPLAIN ANALYZE` reported `Index (Only) Scan ... actual time=… rows=1`
for both — the indexes were genuinely consulted, not merely eligible.

## 8. Findings / bugs

**Overall: the script works end-to-end.** Both gates pass — every pure
binary association ends up with composite indexes in both directions on
both the live and history tables (§6), and the planner uses them (§7).
No bugs found in `generate_association_indexes.py`.

**Findings (none block the script; recorded for completeness):**

1. **Feature-value association tables are intentionally not indexed.**
   `Execution_Subject_Health` (and the other `Execution_<X>_<Feature>`
   tables) carry 4 user FKs, so `Table.is_association()` (pure binary)
   rejects them. This is correct behavior, but worth knowing: if the
   motivation for the indexes is *feature-query* performance, those
   tables are out of scope for this tool and would need a separate
   indexing approach. The tool covers dataset-member, nested-dataset,
   and `{X}_Asset_Type` / `{X}_Execution` link tables.

2. **Planner picks the `rev` index for both orderings at this scale.**
   Not a defect — B-tree equality is order-insensitive across indexed
   columns, so both `fwd` and `rev` are eligible and the planner chose
   one. Both are present and usable (§6, §7). The two directions earn
   their keep on range/sort-on-leading-column queries, not equality.

3. **`pg_indexes`/`pg_class` substring counting needs care.**
   `_assoc_fwd_idx` is a substring of `_hist_assoc_fwd_idx`; naive
   `grep -c '_assoc_fwd_idx'` double-counts. This is a *test-script*
   gotcha (handled here by anchoring on whole index names), not a
   script issue.

4. **Environment / auth.** Catalog creation first failed HTTP 401
   (stale `localhost` bearer token); resolved by a user re-login, after
   which everything ran. An empty stray catalog (`_ermrest_catalog_2`)
   was created during the auth probe and could not be auto-deleted
   (permission guard on shared infra) — **left in place; harmless**
   (no association tables). Clean it up manually if desired:
   `curl -k -X DELETE -H "Authorization: Bearer <tok>" https://localhost/ermrest/catalog/2`.

5. **zsh vs the plan's `eval "$PG"` pattern.** The plan's
   `eval "$PG" -c "<sql>"` idiom breaks under zsh (glob/word-split on
   the parenthesised SQL). Switched to a `pg()` helper piping SQL via
   `docker exec -i ... psql` stdin heredocs. A future run of this
   runbook should use the heredoc form.

## 9. Catalog disposition

Both test catalogs deleted via the ermrest API (user-authorized):

- **Catalog 3** (the test catalog) — `DELETE /ermrest/catalog/3` → 204;
  subsequent `GET` → 404.
- **Catalog 2** (the auth-probe stray) — `DELETE /ermrest/catalog/2` →
  204; `GET` → 404.
- **Catalog 1** (pre-existing, untouched) — still `GET` → 200.

(The backing `_ermrest_catalog_2/3` Postgres databases linger until
ermrest's async reaper drops them; the catalogs are deregistered and
no longer API-accessible, which is the meaningful deletion.)

---

**Conclusion:** `scripts/generate_association_indexes.py` is validated
end-to-end. It discovers every pure binary association, emits correct
composite-pair indexes for the live table and matching JSONB expression
indexes for the history table in both directions, applies cleanly and
idempotently via psql, and the resulting indexes are planner-usable
(including the non-trivial history-side expression indexes). No bugs
found.

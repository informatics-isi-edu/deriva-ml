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

_(filled in Task 5)_

## 6. Index existence (contract gate)

_(filled in Task 6)_

## 7. Index usability (planner uses the index)

_(filled in Task 7)_

## 8. Findings / bugs

_(filled in Task 8)_

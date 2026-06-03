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

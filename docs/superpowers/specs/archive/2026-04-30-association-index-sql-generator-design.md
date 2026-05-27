# Association Index SQL Generator — Design

**Date:** 2026-04-30
**Status:** Draft for review.
**Subproject:** `deriva-ml` (CLI command in
`src/deriva_ml/tools/generate_association_indexes.py`).

## 1. Problem statement

Performance testing on production catalogs has shown that join queries
through pure binary association tables are noticeably faster when the
association table carries composite indexes covering both join
orderings, and when the corresponding history table (in
`_ermrest_history`) carries equivalent expression indexes on its JSONB
`rowdata` column. ERMrest creates a per-FK index on the live table by
default but does **not** create the composite-pair indexes for the two
join directions, and it never creates expression indexes on the
history table. Walking a real catalog (e.g. eye-ai) by hand to find
every association table and write the corresponding `CREATE INDEX`
statements is tedious and error-prone — column RIDs in the history
JSONB have to be looked up per-column (from the catalog's raw
`/schema` document, since the `_ermrest` schema is not exposed as a
queryable schema), and the table's own RID determines the
history-table name.

This tool automates the discovery and SQL generation. It reads a
catalog over the ermrest HTTP API, identifies every pure binary
association table, looks up the RIDs needed to compose history-table
index expressions, and writes a single `.sql` file that a database
administrator can review and run via pgAdmin (or `psql -f`) against
the live Postgres backing the catalog. It does **not** execute any
DDL itself — generation and application are deliberately separated so
the admin retains control of the maintenance window.

## 2. Goals and non-goals

### Goals

- Single CLI command, fits the existing `deriva-ml-*` pattern and the
  `tools/` directory.
- Read-only against the catalog. Talks only to the ermrest HTTP API
  using the same Globus auth path the rest of deriva-ml uses. No
  Postgres connection.
- Output a single self-contained `.sql` file with a header identifying
  the catalog, generation timestamp, and tool version, plus per-
  association sections that an admin can review or selectively
  apply.
- Use `CREATE INDEX CONCURRENTLY IF NOT EXISTS` so the file can be run
  against a live catalog without acquiring an `ACCESS EXCLUSIVE` lock,
  and so re-running is a safe no-op for already-indexed associations.
- Cover both the live association table and its corresponding history
  table in `_ermrest_history`, generating expression indexes on the
  history side that match the live-side composite indexes.
- Index pair composite indexes only — `(c1_cols, c2_cols)` and
  `(c2_cols, c1_cols)` — for both join orderings. Single-column
  indexes are skipped (already covered by ERMrest's per-FK index on
  the live side; not the bottleneck on the history side).

### Non-goals

- Applying the SQL. The admin runs the generated file.
- Detecting and dropping stale indexes (e.g. indexes pointing at
  column RIDs that no longer exist because a column was dropped and
  recreated). Re-running the tool periodically is the recommended
  hygiene; explicit cleanup of stale RID-keyed indexes would require
  reading `pg_indexes` over a Postgres connection, which is out of
  scope for v1.
- Indexes for impure associations (associations that carry their own
  metadata columns), n-ary associations (arity ≥ 3), or any
  non-association tables. The default `Table.is_association()` matches
  pure, unqualified, non-overlapping, binary associations — that is
  the intended scope.
- Single-column indexes. Skipped intentionally, per scope decision in
  §3 (covered by ERMrest's per-FK index on the live side, not the
  bottleneck on the history side).
- Indexes for tables in system schemas (`_ermrest`, `_ermrest_history`,
  `pg_catalog`, `public`, `www`).

## 3. Approach

A pure read-then-emit pipeline:

1. **Connect.** Open an `ErmrestCatalog` for the target catalog using
   `DerivaServer.connect_ermrest`, the same path `DerivaML` already
   uses internally. The CLI takes `--hostname` and `--catalog-id`;
   credentials come from the user's existing Deriva/Globus session.
2. **Pull the model.** `catalog.getCatalogModel()` returns an
   `ermrest_model.Model`. Iterate `model.schemas`, skip the system
   schemas listed above.
3. **Find associations.** For each table, call
   `Table.is_association(return_fkeys=True)` with all defaults (pure,
   unqualified, non-overlapping, binary). Skip if falsy. The truthy
   return is a 2-element set of `ForeignKey` objects, FK1 and FK2.
4. **Gather RID lookups.** ERMrest does not expose the `_ermrest`
   schema (where `known_tables` and `known_columns` live) as a
   queryable schema over the data API — it returns "Schema _ermrest
   does not exist." Instead, the raw ERMrest schema document (GET
   `/ermrest/catalog/<id>/schema`) carries the table and column RIDs
   inline as `"RID"` fields on each table and column entry, even
   though the deriva-py `Model` object strips them out at parse time.

   The implementation fetches the raw schema doc once via
   `catalog.get("/schema").json()`, then walks it to build:

   - `table_rid: dict[(schema_name, table_name), str]`
   - `column_rid: dict[(table_rid, column_name), str]`

   The deriva-py `Model` is still used for association detection
   (`Table.is_association(return_fkeys=True)`); the raw doc is used
   only for RID lookup. Both come from the same single HTTP fetch
   under the hood (deriva-py's `getCatalogModel` and `cat.get("/schema")`
   both hit `/schema`), but we do two calls to keep the code paths
   simple — the schema doc is small enough that a duplicate fetch is
   negligible. (Future optimization: parse the raw doc once and feed
   it to `Model.fromcatalog` to avoid the second round-trip.)
5. **Emit SQL.** For each association `A` with FK pair
   `(FK1, FK2)`:

   - Resolve FK column lists. `c1_cols = FK1.foreign_key_columns`,
     `c2_cols = FK2.foreign_key_columns` — each is a list of
     `Column` objects, typically length 1, but may be composite.
     Resolve each to its column RID via the lookup built in step 4.
   - Resolve `A`'s table RID via the same lookup.
   - Emit a header comment block identifying the association table,
     the two referenced tables, and the column lists for each FK.
   - Emit four `CREATE INDEX CONCURRENTLY IF NOT EXISTS` statements:
     two for the live table covering `(c1_cols..., c2_cols...)` and
     `(c2_cols..., c1_cols...)`, and two equivalent expression
     indexes on `_ermrest_history."t<table_rid>"` using
     `(rowdata->>'<col_rid>')` per FK column.
6. **Write the file.** Default path
   `association-indexes-<hostname>-<catalog_id>-<UTC-timestamp>.sql`
   in cwd, overridable with `--output`. UTF-8, LF line endings.

`CREATE INDEX CONCURRENTLY` cannot run inside a transaction block, so
no `BEGIN/COMMIT` wrappers. Each `CREATE INDEX` is its own statement;
a failure on one does not affect the others when the file is run
through `psql -f` (psql continues past failures by default unless
`-v ON_ERROR_STOP=1` is set, which the admin can choose).

## 4. Output format

```sql
-- Generated by deriva-ml-generate-association-indexes vX.Y.Z
-- Catalog:    https://<hostname>/ermrest/catalog/<catalog_id>
-- Generated:  2026-04-30T14:23:00Z
-- Schemas scanned:    <count>
-- Associations found: <count>
--
-- Review before running. Apply via pgAdmin or `psql -f`.
-- Each CREATE INDEX uses CONCURRENTLY so it does not block live
-- traffic, and IF NOT EXISTS so re-runs are safe no-ops.
-- Expression indexes on _ermrest_history.* tables key on column
-- RIDs, not column names — see ermrest_schema.sql for details.

-- ========================================================================
-- Subject_Image  (eye-ai.Subject_Image)
--   Table RID: 1-ABCD
--   FK1: ("Subject_RID")  ->  eye-ai.Subject       [col RID 1-EEEE]
--   FK2: ("Image_RID")    ->  eye-ai.Image         [col RID 1-FFFF]
-- ========================================================================

CREATE INDEX CONCURRENTLY IF NOT EXISTS "Subject_Image_assoc_fwd_idx"
  ON "eye-ai"."Subject_Image" ("Subject_RID", "Image_RID");

CREATE INDEX CONCURRENTLY IF NOT EXISTS "Subject_Image_assoc_rev_idx"
  ON "eye-ai"."Subject_Image" ("Image_RID", "Subject_RID");

CREATE INDEX CONCURRENTLY IF NOT EXISTS "Subject_Image_hist_assoc_fwd_idx"
  ON _ermrest_history."t1-ABCD"
     (((rowdata->>'1-EEEE')), ((rowdata->>'1-FFFF')));

CREATE INDEX CONCURRENTLY IF NOT EXISTS "Subject_Image_hist_assoc_rev_idx"
  ON _ermrest_history."t1-ABCD"
     (((rowdata->>'1-FFFF')), ((rowdata->>'1-EEEE')));

-- ========================================================================
-- ... next association ...
```

### Index naming

- Live: `<table_name>_assoc_fwd_idx`, `<table_name>_assoc_rev_idx`.
- History: `<table_name>_hist_assoc_fwd_idx`,
  `<table_name>_hist_assoc_rev_idx`.

Postgres caps identifiers at 63 bytes. If a name would exceed that, it
is truncated and a deterministic 8-character hex suffix (md5 of the
full intended name, first 8 chars) is appended in place of the
truncated tail. Implementation note: the truncation point is chosen so
the suffix and trailing `_idx` always fit; this guarantees uniqueness
even when two long table names share a long common prefix.

### Composite-FK handling

If FK1 has multiple columns (e.g., a 2-column composite FK), the index
column list expands to include all columns of FK1 in their declared
order, then all columns of FK2 in their declared order. The reverse
index expands FK2's columns first, then FK1's. The history-side
expression indexes mirror this with one `(rowdata->>'<rid>')`
expression per column.

## 5. Components and module layout

```
src/deriva_ml/tools/generate_association_indexes.py   # CLI + orchestration
src/deriva_ml/tools/_index_sql.py                     # pure SQL builders
tests/tools/test_generate_association_indexes.py      # tests
```

### `_index_sql.py` (pure functions, no I/O)

- `live_index_sql(schema, table, index_name, columns) -> str` — emits
  one `CREATE INDEX CONCURRENTLY IF NOT EXISTS` for the live table.
- `history_index_sql(table_rid, index_name, column_rids) -> str` —
  emits the equivalent expression index on
  `_ermrest_history."t<rid>"`.
- `truncate_index_name(name: str) -> str` — applies the 63-byte cap
  with hash suffix.
- `quote_ident(name: str) -> str` — Postgres double-quoting,
  doubling embedded `"` characters. Used for schema, table, and column
  names. RIDs are also quoted because they contain `-`.
- `quote_literal(text: str) -> str` — Postgres single-quoting, used
  for the inside of `rowdata->>'…'`. RIDs are alphanumeric plus `-`,
  so escaping is defensive but cheap.

These are unit-testable without a catalog.

### `generate_association_indexes.py` (orchestration + CLI)

- `main()` — argparse entry point: `--hostname`, `--catalog-id`,
  `--output`, `--verbose`. Wires up logging via
  `deriva_ml.core.logging_config.get_logger`.
- `generate(hostname, catalog_id, output_path)` — public function:
  connects, pulls model, builds RID lookups, walks associations, emits
  file. Returns a small summary dataclass (counts, output path) for
  testability and machine consumption if a notebook caller wants it.
- `_build_rid_lookup(catalog) -> tuple[TableRidMap, ColumnRidMap]` —
  fetches the raw schema doc via `catalog.get("/schema").json()` and
  walks `schemas[*].tables[*]["RID"]` and
  `schemas[*].tables[*].column_definitions[*]["RID"]` to build the
  two in-memory dicts.
- `_walk_associations(model) -> Iterator[AssociationInfo]` — yields a
  small dataclass per association: `(schema_name, table_name,
  fk1_columns, fk1_target, fk2_columns, fk2_target)`. Internal-only
  value object (no user-facing surface), so plain `@dataclass` per the
  project preference.

### CLI registration

In `pyproject.toml`, alongside the existing tools entries:

```toml
deriva-ml-generate-association-indexes = "deriva_ml.tools.generate_association_indexes:main"
```

## 6. Data flow

```
$ deriva-ml-generate-association-indexes --hostname www.eye-ai.org --catalog-id eye-ai

  ErmrestCatalog.connect (Globus token from local creds)
    └─> Model = catalog.getCatalogModel()
    └─> _build_rid_lookup() via raw catalog.get("/schema"):
          schemas[*].tables[*].RID            -> {(sname, tname): rid}
          schemas[*].tables[*].column_definitions[*].RID
                                              -> {(table_rid, cname): rid}
    └─> for schema in model.schemas, skip system schemas:
          for table in schema.tables:
            fkeys = table.is_association(return_fkeys=True)
            if not fkeys: continue
            yield AssociationInfo(...)
    └─> writer:
          write header (catalog URL, timestamp, counts)
          for assoc in associations:
            write per-association comment block
            write 4 CREATE INDEX statements
    └─> close file, log summary

  -> association-indexes-www.eye-ai.org-eye-ai-20260430T142300Z.sql
```

## 7. Error handling

- **Connection / auth failure.** Bubble the underlying
  `deriva.core` error with a CLI-friendly message ("could not connect
  to <hostname>/<catalog_id>: <reason>"). Exit code 2.
- **RID lookup miss.** If a column referenced by an association FK
  cannot be resolved in the parsed schema doc — either because the
  column was renamed mid-fetch, or because the ERMrest server is too
  old to populate `"RID"` on column entries — that single association
  is skipped, a warning is logged, and processing continues. The
  summary at the end reports the count of skipped associations and
  the reasons. If the *table*'s RID is missing, only the live-side
  indexes are emitted (live indexes don't need the table RID; only
  history indexes do); the history-side indexes are skipped and a
  warning is logged.
- **No associations found.** Emit the header-only file with the
  "Associations found: 0" count and exit 0. This is a valid outcome
  for a catalog that has no association tables yet.
- **File write failure.** Bubble `OSError`. Exit code 2.

The tool does not retry. Network flakiness is the operator's problem
to handle.

## 8. Idempotence and re-running

The tool is safe to re-run any time:

- Generation produces a new file each run (timestamped name); old
  files are not touched. The admin can diff two runs to see what
  associations have appeared.
- Applied SQL is idempotent at the database level via
  `CREATE INDEX … IF NOT EXISTS`. Running the same file twice is a
  no-op for the second run.

### When indexes go stale

History indexes are bound to column RIDs. If a column is dropped and a
new column with the same name is added, the new column gets a new
RID, the JSONB key for new history rows changes, and the previously-
generated index no longer covers them. Re-running the tool generates a
new index against the current RIDs; the old index continues to exist
(and continues to cover history rows written before the drop). v1
does not emit `DROP INDEX` for the obsolete index — see Non-goals §2.
Live-table indexes do not have this issue because they reference
columns by name.

## 9. Testing strategy

Unit tests (no catalog required):

- `_index_sql.live_index_sql` — verifies output for single-column and
  composite-column FKs, and for table/column names that need quoting
  (embedded spaces, hyphens, mixed case, embedded `"`).
- `_index_sql.history_index_sql` — verifies the
  `(rowdata->>'<rid>')` expression form, single-column and composite,
  and that the table RID is used in the table name.
- `_index_sql.truncate_index_name` — verifies the 63-byte cap, the
  hash suffix shape, and that two distinct long names always produce
  distinct truncated names.
- `_walk_associations` — fed a hand-built `Model` with mixed table
  shapes (pure binary association, impure binary, ternary, regular
  FK-bearing table, vocabulary), verifies only the pure binary one is
  yielded.

Integration test (catalog required, gated on `DERIVA_HOST`):

- Against a populated test catalog, run `generate(...)` end-to-end and
  verify the produced SQL parses cleanly with `sqlparse`, contains the
  expected number of `CREATE INDEX` statements per known association
  in the test fixture, and that every generated index name is ≤63
  bytes.
- Re-run against the same catalog and assert byte-identical output
  (modulo the timestamp in the header) — confirms the walk is
  deterministic.

The integration test does **not** apply the SQL. We don't have
Postgres write access from the test harness, and the fixture has no
expectation that history-table indexes exist or don't exist.

## 10. Open questions

None blocking. Two minor items the implementer can decide:

- Whether to emit a brief usage hint at the bottom of the SQL file
  (e.g., a comment showing the recommended `psql -f -v
  ON_ERROR_STOP=1` invocation). Cheap to add.
- Whether `--verbose` should emit per-association timing as it walks.
  Probably overkill for v1; standard logger output is enough.

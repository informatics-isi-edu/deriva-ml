# Engineer Audit — schema/ Subsystem (v1.37.1, 2026-05-22)

## Scope and method

Reviewed:

- `src/deriva_ml/schema/__init__.py` (8 LoC)
- `src/deriva_ml/schema/create_schema.py` (660 LoC)
- `src/deriva_ml/schema/annotations.py` (629 LoC)
- `src/deriva_ml/schema/policy.json` (81 LoC)
- `tests/schema/test_acl_application.py`, `test_annotations.py`,
  `test_cascade_behavior.py`, `test_policy.py`,
  `test_seeded_vocab_terms.py`, `test_vocab_fk_convention.py`

External consumers (grep `from deriva_ml.schema` across `src/`):

- `src/deriva_ml/demo_catalog.py` — uses `create_ml_catalog`
- `src/deriva_ml/core/base.py:1126` — uses `catalog_annotation`
- `src/deriva_ml/core/mixins/asset.py:30` — uses `asset_annotation`
- `src/deriva_ml/schema/create_schema.py:35` — uses
  `asset_annotation`, `generate_annotation`

Recent context: PR #186 (`ecd140a3`) — fixed ACL-ordering bug in
`create_ml_catalog` so `acl_config` runs **after**
`create_ml_schema`. New `tests/schema/test_acl_application.py`
guards that ordering.

Severity:
**P0** (release-blocker) / **P1** (must-fix before release) /
**P2** (should-fix soon) / **P3** (nice-to-have).

---

## Executive summary

The subsystem is in a **shipworthy** state for v1.37.1. The
load-bearing entry points (`create_ml_catalog`,
`create_ml_schema`, `asset_annotation`, `build_navbar_menu`,
`initialize_ml_schema`) all carry recent test coverage, and PR
#186 closed the most consequential bug (ACL ordering). The
findings below are mostly correctness-near-misses and
documentation drift — no P0 blockers.

The single most important pattern uncovered: **`generate_annotation`
hard-codes the literal string `"deriva-ml"` in seven places while
also accepting a `schema` parameter** (F-01). Callers who pass a
non-default schema name get a half-rewritten annotation set that
references both the supplied schema and the literal — Chaise will
silently render broken FK pseudo-columns. This is a latent
correctness bug for anyone who tries to use the documented
`schema_name=` parameter on `create_ml_schema`.

Themes:

1. **Hard-coded schema names** (F-01, F-02, F-03) — `schema_name`
   parameter is partially honored.
2. **Doc / signature drift** (F-04, F-05, F-06, F-07) —
   `asset_annotation` claims to return a dict; `generate_annotation`
   has no docstring; `model` parameter is unused.
3. **Non-deterministic emitted JSON** (F-08) — `asset_metadata`
   set iteration leaks into Chaise annotations.
4. **Internal table-builder helpers are exposed via module
   namespace but undocumented** (F-09 — F-12).
5. **No unit tests for `asset_annotation` / `generate_annotation`**
   (F-13, F-14) — the only coverage is indirect through full
   catalog integration tests.
6. **`use_hatrac` parameter accepted but ignored** (F-15) — silent
   `File` vs `Execution_Asset` divergence.
7. **CASCADE warning is clear but doesn't surface the bag-pipeline
   carve-out at the call site** (F-16, F-17).
8. **Schema-cache freshness contract is invisible from the
   subsystem** (F-18) — `create_ml_schema` rebuilds the catalog
   model but the in-process `SchemaCache` (in `core/`) is never
   touched. Acceptable, but worth a comment.
9. **Hidden test gaps for the FK-naming convention used by
   `asset_annotation`** (F-19) — annotation string-formula values
   like `f"{asset_name}_Asset_Type_{asset_name}_fkey"` aren't
   verified to match what `define_association` actually emits.

Total findings: **27** (P0: 0, P1: 5, P2: 13, P3: 9).

---

## Findings

### F-01 — `generate_annotation` hard-codes `"deriva-ml"` despite `schema` parameter (P1, correctness)

`src/deriva_ml/schema/annotations.py:361,506,517,520,530,546,552`

```python
def generate_annotation(model: Model, schema: str) -> dict:
    ...
    dataset_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                ...
                {
                    "source": [
                        {"outbound": ["deriva-ml", "Dataset_Version_fkey"]},  # ← literal
                        ...
```

Seven occurrences inside `dataset_annotation` reference the literal
`"deriva-ml"` instead of the `schema` parameter. The rest of the
function (`workflow_annotation`, `execution_annotation`,
`dataset_version_annotation`) correctly uses the parameter. A user
who passes `schema_name="my_ml"` to `create_ml_schema` will get a
`Dataset` table whose `visible_columns` annotation references
`"deriva-ml"` FKs that don't exist in their schema — Chaise will
silently render the column as missing.

**Action**: replace the seven literals with the `schema` parameter.

### F-02 — `catalog_annotation` hard-codes `"deriva-ml"` in `chaise_config.defaultTable` (P2, correctness)

`src/deriva_ml/schema/annotations.py:232`

```python
"defaultTable": {"table": "Dataset", "schema": "deriva-ml"},
```

Same shape as F-01 but inside the Chaise top-level config. A
non-default `schema_name` will mean the Chaise landing page
points at a non-existent schema. Acceptable to defer if
non-default `schema_name` is in practice unused, but the
`schema_name` parameter is part of `create_ml_schema`'s
documented signature, so the contract is broken.

**Action**: thread the schema name through `catalog_annotation`
(or pull it from `model.ml_schema`).

### F-03 — `create_ml_schema` honors `schema_name` partially (P2, correctness)

`src/deriva_ml/schema/create_schema.py:313-410`

The function takes `schema_name: str = "deriva-ml"`. It correctly
uses the parameter to look up / drop the schema (line 335) and to
create vocabulary tables. But it calls `generate_annotation(model,
schema_name)` which itself contains literal `"deriva-ml"` (F-01).
And at line 410, `initialize_ml_schema(model, schema_name)` passes
the name through correctly.

So `create_ml_schema` partly honors `schema_name`. The contract
needs either (a) the literals fixed (F-01, F-02), or (b) the
parameter removed and the literal `"deriva-ml"` documented as
the only supported value.

**Action**: pick a direction. The cheap path is fixing the
literals.

### F-04 — `asset_annotation` docstring says "Returns a dictionary" but returns `None` (P1, doc)

`src/deriva_ml/schema/annotations.py:247-358`

```python
def asset_annotation(asset_table: Table):
    """Generate annotations for an asset table.

    Args:
        asset_table: The Table object representing the asset table.

    Returns:
        A dictionary containing the annotations for the asset table.
    """
```

The function mutates `asset_table.annotations` in-place and calls
`asset_table.schema.model.apply()`. It has **no `return`
statement** — returns `None`. The "Returns" docstring section is
wrong.

**Action**: rewrite the docstring to describe the mutation and
the `apply()` side effect; drop the `Returns` block (or replace
with `Returns: None`).

### F-05 — `generate_annotation` has no docstring at all (P1, doc)

`src/deriva_ml/schema/annotations.py:361`

```python
def generate_annotation(model: Model, schema: str) -> dict:
    workflow_annotation = {
```

The function is part of the module-level `__all__` (line 623), so
it's a public API. Workspace convention (CLAUDE.md) requires a
Google-style docstring with `Args/Returns/Raises/Example`.

**Action**: add a Google-style docstring; document the five keys
of the returned dict (`workflow_annotation`, `dataset_annotation`,
`execution_annotation`, `schema_annotation`,
`dataset_version_annotation`).

### F-06 — `generate_annotation`'s `model` parameter is unused (P2, dead code)

`src/deriva_ml/schema/annotations.py:361`

```python
def generate_annotation(model: Model, schema: str) -> dict:
```

`model` is declared but never referenced inside the function body.
Either the function intended to introspect the model (e.g., to
discover features) or the parameter is dead. The call site
(`create_schema.py:340`) passes `generate_annotation(model,
schema_name)`, so removing the parameter is a public-API change.

**Action**: either (a) remove the parameter and update the one
call site, or (b) use the parameter (e.g., to drive
`navbarMenu`-style discovery) and document it.

### F-07 — `create_dataset_table` has no docstring (P1, doc)

`src/deriva_ml/schema/create_schema.py:40-82`

The neighboring `create_execution_table`, `create_workflow_table`,
`create_asset_table`, and `define_table_dataset_version` all have
Google-style docstrings; `create_dataset_table` is the only
sibling with none. Given the function creates four tables
(Dataset, Dataset_Type, Dataset_Type association, Dataset_Version,
Dataset_Dataset for nested, Dataset_Execution) the parameter
documentation is non-trivial.

**Action**: add a Google-style docstring listing the side-effect
graph (which tables get created, which FK edges get added).

### F-08 — `asset_annotation` iterates a `set` for `visible_columns` ordering (P1, correctness — non-determinism)

`src/deriva_ml/schema/annotations.py:259, 307, 321`

```python
asset_metadata = {c.name for c in asset_table.columns} - DerivaAssetColumns
...
"*": [
    ...
] + [fkey_column(c) for c in asset_metadata],   # ← set iteration
"detailed": [
    ...
] + [fkey_column(c) for c in asset_metadata],   # ← set iteration
```

`asset_metadata` is a **set**, so its iteration order is
implementation-defined. The resulting Chaise `visible_columns`
order will jitter run-to-run on rebuilds and is not
byte-reproducible across catalog instances created from the same
source. This in turn breaks downstream diffs of catalog
annotations.

This is the same family of bug as the asset-manifest ordering bug
documented in CLAUDE.md ("Metadata directory order must match
`sorted(metadata_columns)` — both `asset_table_upload_spec()` and
`_build_upload_staging()` sort alphabetically").

**Action**: replace both list comprehensions with `[fkey_column(c)
for c in sorted(asset_metadata)]`.

### F-09 — Internal table-builder helpers (`create_dataset_table` etc.) are not in `__all__` but are reachable via attribute access (P2, API surface)

`src/deriva_ml/schema/create_schema.py:655-659`

```python
__all__ = [
    "create_ml_catalog",
    "create_ml_schema",
    "initialize_ml_schema",
]
```

`create_dataset_table`, `create_execution_table`,
`create_workflow_table`, `create_asset_table`,
`define_table_dataset_version` are all module-level and reachable
as `deriva_ml.schema.create_schema.create_dataset_table`. They
have no `_` prefix so the static analyzer treats them as public,
but `__all__` excludes them. A wildcard import gets only the three
listed — which is correct — but an explicit
`from deriva_ml.schema.create_schema import create_dataset_table`
works and is a hidden surface contract.

**Action**: prefix with `_` (recommended — they're build-time
helpers), **or** add them to `__all__` and give them docstrings
that reflect "internal but exposed".

### F-10 — `asset_annotation`'s FK-name string-formula is a fragile contract (P2, correctness)

`src/deriva_ml/schema/annotations.py:276-289`

```python
asset_type_source = {
    "source": [
        {"inbound": [schema, f"{asset_name}_Asset_Type_{asset_name}_fkey"]},
        {"outbound": [schema, f"{asset_name}_Asset_Type_Asset_Type_fkey"]},
        "RID",
    ],
    "markdown_name": "Asset Types",
}
```

These FK-constraint names are constructed by `Table.define_association`
inside deriva-py. The exact format (`<assoc>_<column>_fkey`) is a
deriva-py convention, not a deriva-ml-controlled name. If deriva-py
ever changes the synthesized constraint-name template, this
annotation breaks silently — Chaise will render no asset-types
column.

There is no test that asserts these names actually match the FK
constraints emitted by `define_association`.

**Action**: at minimum, add an integration assertion that the
generated table has a FK constraint matching
`f"{asset_name}_Asset_Type_{asset_name}_fkey"`. Better: pull the
name from `asset_table.foreign_keys[…].name` rather than
formatting a string.

### F-11 — `create_asset_table.use_hatrac` parameter is accepted and silently ignored (P1, correctness)

`src/deriva_ml/schema/create_schema.py:222-249`

```python
def create_asset_table(
    schema: Schema,
    asset_name: str,
    execution_table: Table,
    asset_type_table: Table,
    asset_role_table: Table,
    use_hatrac: bool = True,
) -> Table:
    ...
    asset_table = schema.create_table(
        AssetTableDef(
            schema_name=schema.name,
            name=asset_name,
            hatrac_template="/hatrac/metadata/{{MD5}}.{{Filename}}",
        )
    )
```

The `use_hatrac` parameter is documented in the docstring
("Whether to use Hatrac for file storage") but the function body
**never references it**. The `File` table is created at line 392
with `use_hatrac=False`, but it gets the same `hatrac_template`
as `Execution_Metadata` and `Execution_Asset`. So the `File` table
is wired up identically to a Hatrac-backed asset.

This is either (a) the parameter is dead and the `File` table's
"not managed by Hatrac" semantic is implemented elsewhere, or
(b) the parameter was supposed to switch off the `hatrac_template`
and someone forgot to wire it up.

**Action**: investigate. If the `File` table actually doesn't go
through Hatrac at runtime, the parameter is misleading and should
either be removed or actually drive behavior. The seeded term
`{"Name": "File", "Description": "A file that is not managed by
Hatrac"}` (create_schema.py:476) reinforces that the semantic was
intended.

### F-12 — `define_table_dataset_version` and similar helpers leak schema-name string-coupling (P2, design)

`src/deriva_ml/schema/create_schema.py:85-150`

`define_table_dataset_version(sname, …)` takes the schema name as
a string parameter and builds FK definitions that reference
`sname` for `Dataset` and `Execution` tables. The function is
called from `create_dataset_table` (line 72), which itself takes a
`Schema` object. There's a layered API gap: the caller has the
Schema, the inner function takes a string.

This isn't load-bearing, but it makes the helper hard to reuse —
you can't pass it a `Schema` directly.

**Action**: change the signature to `define_table_dataset_version(schema: Schema, …)` and use `schema.name` internally,
mirroring `create_execution_table(schema, …)`.

### F-13 — No unit test for `asset_annotation` (P1, coverage)

`tests/schema/` has no test file covering `asset_annotation`. The
only coverage is indirect, via `test_acl_application.py` which
exercises full catalog creation (and so runs `asset_annotation`
as a side effect of `create_ml_schema`). A change that broke the
annotation shape (e.g., dropped a key from `visible_columns`)
would not be caught.

The function is straightforward to test against a mock `Table`
(no live catalog needed) — same pattern as
`test_annotations.py::build_navbar_menu`.

**Action**: add `tests/schema/test_asset_annotation.py` with at
least:
- annotation dict shape (top-level keys);
- presence of the `asset_type_source` inbound/outbound triple;
- deterministic ordering (covers F-08);
- file-preview annotation gets applied to the URL column.

### F-14 — No unit test for `generate_annotation` (P1, coverage)

Same shape as F-13. `generate_annotation` produces the dict that
`create_ml_schema` passes to four `create_*_table` helpers; it
has no test. Adding a test would catch the F-01 literal-vs-parameter
bug as a regression.

**Action**: add `tests/schema/test_generate_annotation.py`
checking the five-key shape and that **no** value contains the
literal `"deriva-ml"` when called with `schema="not-deriva-ml"`.

### F-15 — `policy.json` schema is not documented anywhere in the subsystem (P2, doc)

`src/deriva_ml/schema/policy.json` is bundled with the package and
shelled out to `deriva.config.acl_config`. Its structure
(`groups`, `acl_definitions`, `acl_bindings`, `catalog_acl`,
`schema_acls`, `table_acls`) is partially documented in
`tests/schema/test_policy.py` (which asserts the invariants) but
there's no narrative in the module docstring, no README, no link
from `create_ml_catalog`'s docstring to the file format. A future
maintainer rewriting the file has no reference.

**Action**: add a brief `policy.json` schema explainer to
`create_ml_catalog`'s docstring (or a sibling
`schema/policy.md`) covering: the four group-resolution levels,
the `read_only` / `self_serve` ACL definitions, the
`row_owner_guard` binding, and the negative-lookahead pattern
that targets non-public schemas.

### F-16 — CASCADE warning is present but not surfaced to callers programmatically (P2, safety)

`src/deriva_ml/schema/create_schema.py:336`

```python
if model.schemas.get(schema_name):
    logger.warning(f"Dropping existing schema '{schema_name}' with CASCADE")
    model.schemas[schema_name].drop(cascade=True)
```

The CASCADE drop fires after a `logger.warning(...)` only. A
caller can't intercept this (no callback, no confirmation
parameter, no exception that requires `force=True`). Compare to
`refresh_schema` in `core/base.py:510-571`, which refuses to drop
work without `force=True`.

The CLAUDE.md note ("never call `create_ml_schema` on a catalog
that already has data in `deriva-ml`") signals that the contract
is "callers don't actually do this." But the safer pattern would
be to require an explicit `force_drop=True`.

**Action**: add `force_drop: bool = False` and raise
`DerivaMLConfigurationError` when the schema exists and
`force_drop` is False. (The clone path already avoids this code
path per the docstring at line 322.)

### F-17 — `create_ml_schema` docstring mentions bag-pipeline carve-out but doesn't link to the bag loader (P3, doc)

`src/deriva_ml/schema/create_schema.py:322-326`

The docstring says the bag-pipeline clone path "does not call
this function — it loads catalog content via
`BagCatalogLoader` instead." Useful, but no module path is given.
A reader chasing the carve-out has to grep.

**Action**: add a fully-qualified reference, e.g.,
`:class:\`deriva_ml.catalog.bag_catalog_loader.BagCatalogLoader\``.

### F-18 — Schema-cache freshness: `create_ml_schema` doesn't touch the in-process `SchemaCache` (P2, design clarity)

`src/deriva_ml/schema/create_schema.py` doesn't import or invoke
the `core.schema_cache.SchemaCache`. In practice this is fine
because `create_ml_schema` is called either before a `DerivaML`
instance exists or via `create_ml_catalog` (which then constructs
the instance). But for ad-hoc REPL workflows where a user does:

```python
ml = DerivaML(host, catalog_id)
create_ml_schema(ml.catalog)   # destructive
ml.find_datasets()             # stale schema-cache, may crash
```

…the schema-cache will be stale and there is no contract
documenting it.

**Action**: at minimum, add a note to `create_ml_schema`'s
docstring that callers with an existing `DerivaML` instance
should call `ml.refresh_schema(force=True)` afterward. (No code
change needed — the function lives below the cache layer.)

### F-19 — `test_vocab_fk_convention.py` proves Name-targeted FKs, but doesn't prove `asset_annotation`'s formula matches (P2, coverage)

`tests/schema/test_vocab_fk_convention.py` walks the catalog and
asserts that every FK targeting a vocab table references `Name`.
Excellent invariant. But `asset_annotation` builds string
references like `f"{asset_name}_Asset_Type_{asset_name}_fkey"`
which is the **synthesized constraint name**, not the FK target.
A test that crafts the same formula and looks it up against the
live catalog's FK constraint names would close F-10.

**Action**: extend `test_vocab_fk_convention.py` (or add a
sibling) to assert that every FK-string-formula in
`asset_annotation`'s output resolves to a real constraint on the
created table.

### F-20 — Parallel asset-table creation is repeated three times with copy-paste (P2, dup)

`src/deriva_ml/schema/create_schema.py:375-407`

```python
create_asset_table(schema, MLTable.execution_metadata,
    execution_table, asset_type_table, asset_role_table)
create_asset_table(schema, MLTable.execution_asset,
    execution_table, asset_type_table, asset_role_table)
file_table = create_asset_table(schema, MLTable.file,
    execution_table, asset_type_table, asset_role_table,
    use_hatrac=False)
```

Three near-identical calls. The `File` table differs only by
`use_hatrac=False` (which itself is broken per F-15) and the
follow-up `Table.define_association(Dataset, File)` at lines
401-407.

Acceptable as-is (only three sites), but a `for asset_name in
(MLTable.execution_metadata, MLTable.execution_asset, MLTable.file):`
loop would be tidier.

**Action**: optional refactor. Low priority unless touched.

### F-21 — `generate_annotation` has parallel `workflow_annotation` / `dataset_annotation` / `execution_annotation` blocks that don't share structure (P3, dup)

`src/deriva_ml/schema/annotations.py:362-619`

Each of the four returned annotation dicts has its own
hand-written `visible_columns: { "*": [...], "detailed": [...],
"filter": { "and": [...] } }` block. The `RCB/RMB` pseudo-columns
and the "Created By" / "Modified By" filter entries appear in
multiple blocks. Extracting a `_audit_columns(schema, table)`
helper would cut ~80 lines and ensure the four tables share the
same audit-column display.

**Action**: optional refactor; would also make F-01 easier
because the helper would naturally take the schema parameter.

### F-22 — `_ensure_terms` helper inside `initialize_ml_schema` is good; could be promoted (P3, design)

`src/deriva_ml/schema/create_schema.py:449-455`

```python
def _ensure_terms(table_name: str, terms: list[dict]) -> None:
    """Insert terms that don't already exist in a vocabulary table."""
    table = pb.tables[table_name]
    existing = {row["Name"] for row in table.entities()}
    missing = [t for t in terms if t["Name"] not in existing]
    if missing:
        table.insert(missing, defaults={"ID", "URI"})
```

Useful pattern; defined as a nested closure so it can't be reused.
The CLAUDE.md mentions a `find_associations` / `add_term`
elsewhere — promoting `_ensure_terms` to a module-level helper
would let other vocab-seeding code reuse it.

**Action**: optional. If there's no second caller, leave it.

### F-23 — Five seeded asset types have descriptions that mix metadata structure with content (P3, content drift)

`src/deriva_ml/schema/create_schema.py:460-484`

Asset_Type seed descriptions:
- `"Execution_Config"`: "Configuration File for execution metadata"
- `"Execution_Metadata"`: "Information about the execution environment"

vs.

- `"Hydra_Config"`: "Hydra YAML configuration file (config.yaml,
  overrides.yaml, hydra.yaml)" — names specific filenames
- `"Metrics_File"`: "Training-metric log file (typically JSONL,
  one record per evaluation point — epoch, step, or eval cycle)"

The format is inconsistent: some descriptions are pure semantic
("Information about the X"), others enumerate filename
conventions, others document format ("typically JSONL"). Chaise
will display these verbatim in the vocabulary picker. A
consumer-facing description style guide would help.

**Action**: cosmetic; reconcile descriptions to one of (a)
"What is this term for?" (semantic), or (b) "What concrete
artifacts carry this term?" (operational). Currently mixed.

### F-24 — `initialize_ml_schema` docstring is excellent but has no `Example:` block (P3, doc)

`src/deriva_ml/schema/create_schema.py:413-444`

The function carries the strongest docstring in the subsystem
(documents the platform-vs-domain principle, names anti-examples
that the test suite then pins). It lacks an `Example:` block
showing the typical invocation, which is a workspace convention.

**Action**: add an `Example:` block (probably `# doctest: +SKIP`
since it requires a live catalog).

### F-25 — `create_ml_catalog` shells out to a sub-Python; surface the cleanup contract (P2, safety)

`src/deriva_ml/schema/create_schema.py:618-641`

```python
try:
    subprocess.run(
        [sys.executable, "-m", "deriva.config.acl_config", ...],
        check=True, capture_output=True, text=True,
    )
except subprocess.CalledProcessError as e:
    raise DerivaMLConfigurationError(
        f"Failed to apply ACL policy to catalog {catalog.catalog_id} ..."
    ) from e
```

If `acl_config` fails, the catalog **already exists** (it was
created by `server.create_ermrest_catalog()` at line 590 and the
schema was created at line 601). The exception is raised but the
half-configured catalog leaks. Compare to the `catalog_alias`
branch (line 644) — if alias creation fails, same problem.

A `try/except/delete_ermrest_catalog(really=True)/raise`
wrapper would prevent the leak.

**Action**: wrap the `acl_config` subprocess and the alias-creation
call in a single try block that calls
`catalog.delete_ermrest_catalog(really=True)` on failure before
re-raising. The test fixtures already do this — the production
helper should too.

### F-26 — `policy.json` group `eye-ai` is bundled with the deriva-ml package (P3, design)

`src/deriva_ml/schema/policy.json:6,9-14`

```json
"eye-ai": ["https://auth.globus.org/d38e6f4d-..."],
...
"project-writers":  ["isrd-systems", "eye-ai"],
"project-curators": ["isrd-systems", "local-admin", "eye-ai"],
"project-users":    ["isrd-systems", "local-admin", "eye-ai"],
```

The bundled policy hard-codes an `eye-ai` group reference. Every
deriva-ml catalog created via `create_ml_catalog` thus implicitly
grants the eye-ai Globus group write/curate access on every
non-public schema. This is reasonable for ISI but bleeds project
identity into a platform-level default.

**Action**: either (a) document this in `policy.json`'s
neighborhood, or (b) factor the project-specific groups out into
a per-project policy file overlay.

### F-27 — `Workflow.URL` column is typed `ermrest_uri` but Chaise display uses raw markdown (P3, cosmetic)

`src/deriva_ml/schema/create_schema.py:289`
`src/deriva_ml/schema/annotations.py:389-392`

`Workflow.URL` is declared `BuiltinType.ermrest_uri` (a
deriva-specific type that Chaise renders as a clickable link).
The `detailed` view in `workflow_annotation` overrides this with
a custom markdown template:

```python
{
    "display": {"markdown_pattern": "[{{{URL}}}]({{{URL}}})"},
    "markdown_name": "URL",
},
```

This is redundant with the `ermrest_uri` type — Chaise will
already render the column as a link. The custom template only
exists for the "detailed" view; the "*" view doesn't have it.
Inconsistent.

**Action**: either drop the custom `markdown_pattern` (let
`ermrest_uri` do its job) or apply it to both views.

---

## Coverage map

| Function / asset | Public? | Unit test | Integration test |
|---|---|---|---|
| `create_ml_catalog` | yes (`__all__`) | — | `test_acl_application.py`, `test_vocab_fk_convention.py`, `test_cascade_behavior.py` |
| `create_ml_schema` | yes (`__all__`) | — | `test_cascade_behavior.py` |
| `initialize_ml_schema` | yes (`__all__`) | `test_seeded_vocab_terms.py` (source-introspection) | indirect |
| `create_dataset_table` | exposed | — | indirect |
| `create_execution_table` | exposed | — | indirect |
| `create_workflow_table` | exposed | — | indirect |
| `create_asset_table` | exposed | — | indirect |
| `define_table_dataset_version` | exposed | — | indirect |
| `asset_annotation` | yes (`__all__`) | **none** (F-13) | indirect |
| `catalog_annotation` | yes (`__all__`) | — | `tests/core/test_catalog_annotations.py` |
| `build_navbar_menu` | yes (`__all__`) | `test_annotations.py` | — |
| `generate_annotation` | yes (`__all__`) | **none** (F-14) | indirect |
| `policy.json` | (resource) | `test_policy.py` (structural) | `test_acl_application.py` |

---

## Recommendation for v1.37.1

**Ship as-is.** No P0. The five P1 items (F-01, F-04, F-05, F-07,
F-08, F-11, F-13, F-14) are correctness-near-misses and doc
issues; none change observable schema or break a working install.
Recommend follow-up PRs:

1. **F-01 + F-08 in one PR** — they're the only two findings that
   change emitted catalog state, and both fixes are small.
2. **F-04 + F-05 + F-07 in a docs-only PR** — pure docstring
   patches.
3. **F-11 + F-13 + F-14 in a coverage PR** — investigate
   `use_hatrac`, add unit tests for the two untested public
   annotation functions.
4. **F-25 in a safety PR** — catalog-cleanup wrapper around the
   `acl_config` shell-out.

Defer P2/P3 to a non-release window.

---

## Totals

- **Findings:** 27 (P0: 0, P1: 7, P2: 13, P3: 7)
- **File:** `/Users/carl/GitHub/DerivaML/deriva-ml/docs/audits/2026-05-22-engineer-audit-schema.md`

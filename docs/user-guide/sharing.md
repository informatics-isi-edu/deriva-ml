# Sharing and collaboration

DerivaML gives you three ways to share data: a citable persistent identifier (MINID) for a versioned dataset, a portable BDBag archive that a collaborator can unpack without catalog access, and a partial catalog clone (`create_ml_workspace`) for setting up a fully functional multi-site workspace.

These mechanisms are not mutually exclusive. A typical handoff might share a MINID for citation purposes, attach a BDBag archive for immediate offline use, and maintain a clone for ongoing collaboration on a shared server.

## The three mechanisms at a glance

| Mechanism | What you share | Catalog required by recipient |
|---|---|---|
| MINID | A persistent URL (registered with FAIR identifier service) | No — URL resolves to a BDBag archive |
| BDBag archive | A `.bag.zip` file containing all tables and asset URLs | No — local inspection only |
| Catalog clone | A new catalog on any Deriva server | Yes — recipient connects with `DerivaML` |

Choose a MINID when you want a stable, citable reference that will survive a URL change. Choose a BDBag archive when you need a portable snapshot that a collaborator can hand to code directly. Choose a catalog clone when your collaborator needs to run workflows, add features, or browse data interactively.

## How to share a dataset with a MINID

A MINID (Minimal Viable Identifier) is a globally unique, resolvable identifier registered with a FAIR identifier service. When you download a dataset with `use_minid=True`, DerivaML uploads the bag archive to S3 and registers the resulting URL with the MINID service. The returned identifier remains stable even if your catalog server moves.

**Requirements.** Your catalog must be configured with an S3 bucket. If it is not, `download_dataset_bag(use_minid=True)` raises `DerivaMLException` with a message explaining the missing configuration.

```python
from deriva_ml import DerivaML

ml = DerivaML(hostname="catalog.example.org", catalog_id="1")
dataset = ml.lookup_dataset("1-ABC4")

# Download version 2.0.0 and register a MINID for it
bag = dataset.download_dataset_bag(
    version="2.0.0",
    use_minid=True,
    materialize=True,
)

# The MINID URL is stored on the dataset version record
print(bag.version)  # "2.0.0"
```

The MINID URL is stored in the catalog's dataset version record. Calling `download_dataset_bag(use_minid=True)` again for the same version returns the cached MINID rather than re-uploading, unless the dataset's export spec has changed.

A collaborator who receives the MINID URL can resolve it to the BDBag archive and materialize it without any DerivaML installation:

```bash
# Collaborator's machine — bdbag CLI only
bdbag --resolve-fetch all /path/to/unpacked/bag
```

**Notes**

- `use_minid=True` requires `s3_bucket` to be configured on the catalog; confirm this with your catalog administrator before attempting.
- MINIDs are version-specific — each call with a different `version` argument creates a separate identifier.
- MINID registration is idempotent for a given version and export spec hash: DerivaML checks whether the existing MINID is still current before re-registering.
- Collaborators who receive only the MINID URL can inspect and use the data without a Deriva account.

## How to share a bag archive

`download_dataset_bag` always produces a `.bag.zip` archive regardless of whether a MINID is registered. Without `use_minid=True`, the archive is built client-side from the catalog and stored in the local cache directory.

```python
bag = dataset.download_dataset_bag(
    version="2.0.0",
    materialize=False,   # metadata tables only; omit assets for a smaller archive
)
print(bag.path)  # Path to the local unpacked bag directory
```

The archive at `bag.path.parent / f"Dataset_{dataset_rid}_{version}.bag.zip"` contains:

- `data/` — CSV exports of every table reachable from the dataset via foreign key paths
- `fetch.txt` — a list of remote asset URLs (Hatrac) for any files that were not materialized
- `bag-info.txt` — BDBag metadata: dataset RID, version, checksum, creation date
- `bagit.txt` — BagIt specification marker

To inspect the archive without DerivaML:

```bash
# Install the bdbag CLI tool
pip install bdbag

bdbag --validate fast /path/to/Dataset_1-ABC4_2.0.0.bag.zip
bdbag --resolve-fetch all /path/to/Dataset_1-ABC4_2.0.0.bag.zip
```

A collaborator who has DerivaML can open the bag directly with `DatasetBag`:

```python
from deriva_ml.dataset import DatasetBag

bag = DatasetBag("/path/to/Dataset_1-ABC4")
df = bag.get_table_as_dataframe("Subject")
```

**Notes**

- Asset files are not included in the archive unless `materialize=True`; `fetch.txt` contains their remote URLs instead.
- The `bdbag` CLI tool is available on PyPI (`pip install bdbag`) and is the only dependency needed to validate and materialize bags without DerivaML.
- Bag contents are determined by the catalog's export annotation on the dataset's root table, which follows all FK paths and stops at other dataset element types.

## How to clone a subset of a catalog

`create_ml_workspace` creates a new catalog on a Deriva server containing only the data reachable from a specified root RID. This is the right choice when a collaborator needs a fully functional catalog they can connect to with `DerivaML` and use for further analysis or model training.

```python
from deriva_ml.catalog.clone import create_ml_workspace, OrphanStrategy, AssetCopyMode

result = create_ml_workspace(
    source_hostname="catalog.example.org",
    source_catalog_id="1",
    root_rid="3-HXMC",                        # Starting point for data reachability
    dest_hostname="local.example.org",         # Where to create the new catalog
    alias="my-project-workspace",             # Optional human-readable alias
    asset_mode=AssetCopyMode.REFERENCES,      # Keep asset URLs pointing to source
    orphan_strategy=OrphanStrategy.DELETE,    # Delete rows with broken FK references
    add_ml_schema=True,                       # Add deriva-ml tracking schema
    reinitialize_dataset_versions=True,       # Fix dataset version snapshots
)

print(f"Created catalog {result.catalog_id} on {result.hostname}")
print(result.report.to_text())               # Human-readable clone report
```

### The three-stage approach

`create_ml_workspace` solves a practical problem with real-world catalogs: row-level access policies frequently hide some rows from the cloning user, which would cause foreign key violations if the clone were applied naively. The three-stage approach handles this safely:

**Stage 1 — Schema without foreign keys.** The destination catalog is created with all tables and columns present, but no FK constraints yet. This allows the data copy to proceed even if some referenced rows are missing.

**Stage 2 — Async data copy.** Data is copied concurrently across all discovered tables. Discovery uses the root table's export annotation (if present) to follow the same FK paths that bag export uses. When no export annotation exists, it falls back to FK graph traversal. Vocabulary tables and association tables are auto-included by default.

**Stage 3 — FK application with orphan handling.** After all data is loaded, FK constraints are applied one by one. Rows that would violate a constraint (because their referenced row was hidden from the cloning user) are handled according to the `orphan_strategy` you specify.

### Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_hostname` | `str` | — | Source catalog server |
| `source_catalog_id` | `str` | — | Source catalog ID |
| `root_rid` | `str` | — | Starting RID; determines data reachability |
| `dest_hostname` | `str \| None` | Same as source | Destination server |
| `alias` | `str \| None` | None | Human-readable alias for the new catalog |
| `asset_mode` | `AssetCopyMode` | `REFERENCES` | How to handle asset files (see below) |
| `orphan_strategy` | `OrphanStrategy` | `FAIL` | What to do with broken FK references |
| `add_ml_schema` | `bool` | `True` | Add `deriva-ml` tracking schema to clone |
| `include_tables` | `list[str]` | None | Extra tables to include (`"schema:table"` format) |
| `exclude_objects` | `list[str]` | None | Tables to exclude (`"schema:table"` format) |
| `prune_hidden_fkeys` | `bool` | `False` | Drop FKs whose reference table is hidden |
| `truncate_oversized` | `bool` | `False` | Truncate values that exceed index size limits |

**Asset modes:**

- `AssetCopyMode.NONE` — Asset table rows are copied but file content is not. URLs are left pointing to the source server but the files are not accessible.
- `AssetCopyMode.REFERENCES` — Asset table rows are copied with their original URLs intact. Files remain on the source Hatrac server and are still accessible if the source is reachable.
- `AssetCopyMode.FULL` — Asset files are downloaded from the source and re-uploaded to the destination Hatrac. Fully self-contained but slow for large catalogs.

## How to choose an orphan-handling strategy

An **orphan row** is a row whose FK reference points to a row that is absent from the clone — either because the referenced row was hidden by access policy or because it was excluded by the `exclude_objects` filter. `create_ml_workspace` cannot silently ignore orphan rows because FK constraints would reject them.

The `orphan_strategy` parameter controls what happens when an orphan is detected at Stage 3:

### `OrphanStrategy.FAIL` (default)

The clone operation reports an error for every FK violation and leaves the constraint unapplied. The clone is still usable, but missing FK constraints mean the Chaise web interface may not render relationships correctly.

Use `FAIL` when:
- You need to understand the full scope of incoherence before deciding how to handle it.
- You are cloning from a well-controlled catalog where orphan rows indicate a real problem.
- You want to review the `CloneReport` before deciding whether to re-run with a different strategy.

### `OrphanStrategy.DELETE`

Rows with dangling FK references are deleted from the clone. The resulting catalog is fully coherent — all FK constraints are applied — but some rows from the source are absent from the clone.

Use `DELETE` when:
- You are doing analysis on a subset and do not need the orphan rows.
- The orphan rows are peripheral (metadata for hidden subjects, associations to excluded tables).
- You have reviewed the `CloneReport` from a `FAIL` run and confirmed the deletions are acceptable.

!!! warning
    `DELETE` can cascade. If a table in the middle of a FK chain has orphans deleted, rows in dependent tables that pointed to those deleted rows may also become orphans and be deleted in turn. Use the `CloneReport` to verify what was removed.

### `OrphanStrategy.NULLIFY`

Dangling FK column values are set to `NULL` rather than deleting the row. This preserves all rows but removes the cross-table link for references that have no valid target.

Use `NULLIFY` when:
- The FK column is optional (`nullok=True`) and the relationship is not essential.
- You want to preserve all rows for analysis even when some relationships are broken.
- The orphan references are incidental metadata (e.g., a `Created_By` FK to a user that is not visible).

!!! note
    `NULLIFY` only works when the FK column allows NULL. If the column is required (`nullok=False`), the strategy falls back to the behavior of `FAIL` for that constraint and logs a warning.

**Recommended workflow for unfamiliar catalogs:** Run `create_ml_workspace` once with `orphan_strategy=OrphanStrategy.FAIL`, inspect `result.report.to_text()`, identify the affected tables, then re-run with `DELETE` or `NULLIFY` as appropriate for each table's semantics.

## How to copy assets after a clone

When you clone with `asset_mode=AssetCopyMode.REFERENCES`, asset URLs in the destination catalog still point to the source Hatrac server. This is efficient — no file transfer at clone time — but requires the source server to remain accessible.

When you need the clone to be self-contained, use `localize_assets` to copy files from the source Hatrac to the destination Hatrac and update the catalog URLs:

```python
from deriva_ml.catalog.localize import localize_assets

ml = DerivaML(hostname="local.example.org", catalog_id=result.catalog_id)

# Get the RIDs of all assets you want to localize
pb = ml.pathBuilder()
asset_rids = [
    r["RID"]
    for r in pb.schemas["myschema"].tables["Image"].entities().fetch()
]

localize_result = localize_assets(
    catalog=ml,
    asset_table="Image",
    asset_rids=asset_rids,
    schema_name="myschema",
    source_hostname="catalog.example.org",  # Where the files currently live
    chunk_size=50 * 1024 * 1024,            # 50 MB chunks for large files
)

print(f"Localized {localize_result.assets_processed} assets")
print(f"Skipped   {localize_result.assets_skipped}")
print(f"Failed    {localize_result.assets_failed}")
if localize_result.errors:
    for err in localize_result.errors:
        print(f"  Error: {err}")
```

`localize_assets` returns a `LocalizeResult` with counts and a `localized_assets` list of `(rid, old_url, new_url)` tuples for audit purposes.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `catalog` | `DerivaML \| ErmrestCatalog` | — | Connected catalog to localize into |
| `asset_table` | `str` | — | Name of the asset table |
| `asset_rids` | `list[str]` | — | RIDs of the assets to copy |
| `schema_name` | `str \| None` | None | Schema containing the table; searches all if omitted |
| `source_hostname` | `str \| None` | None | Required when URLs are relative (e.g., cloned with `REFERENCES`) |
| `hatrac_namespace` | `str \| None` | None | Destination namespace; defaults to `/hatrac/{asset_table}` |
| `chunk_size` | `int \| None` | None | Chunk size in bytes; auto-chunked above 100 MB |
| `dry_run` | `bool` | `False` | Report what would be done without making changes |

**Notes**

- Assets already pointing at the local server are detected by hostname comparison and skipped automatically.
- Relative URLs (e.g., `/hatrac/...` without a hostname) require `source_hostname` to be specified; without it, they are treated as already-local and skipped.
- `localize_assets` is optimized for bulk operations: it fetches all asset records in one query and batches the catalog URL updates.
- For very large asset sets, consider processing in chunks (split `asset_rids` into batches) to avoid long-running operations that could be interrupted.

## Access control

DerivaML delegates access control entirely to Deriva's policy engine and Globus Auth. A catalog clone inherits the source catalog's ACLs and ACL bindings when `copy_policy=True` (the default). Changing who can read or write a catalog requires using the Deriva web interface or ERMrest policy API directly — there is no DerivaML Python API for this.

For Globus-based sharing (sharing a catalog with external collaborators using their institutional credentials), refer to the [Deriva documentation](https://docs.derivacloud.org) and the [Globus Auth documentation](https://docs.globus.org/api/auth/). DerivaML assumes that authentication is already configured when a `DerivaML` instance is constructed.

!!! tip
    BDBag archives and MINID URLs are the most practical way to share data with collaborators who do not have a Deriva account. No catalog access policy changes are needed.

---

## Common pitfalls

!!! warning "create_ml_schema drops all deriva-ml data with CASCADE"
    `create_ml_schema()` checks whether the `deriva-ml` schema already exists and, if it does, **drops it with CASCADE before recreating it**. This destroys every Execution, Dataset, Feature, and Workflow record in the schema. Never call `create_ml_schema` directly on a catalog that already has DerivaML data. The `add_ml_schema=True` parameter in `create_ml_workspace` uses an internal guard that checks for existing data before calling `create_ml_schema`; only use `create_ml_schema` on a brand-new empty catalog.

!!! warning "OrphanStrategy.DELETE is per-clone, not per-table"
    `orphan_strategy` applies the same strategy to every FK violation encountered during Stage 3. Applying `DELETE` globally can remove large amounts of data that is legitimate but simply unreachable from the root RID. Always run a `FAIL` clone first to review the `CloneReport`, then make a targeted decision before re-running with `DELETE`.

!!! note "bdbag CLI is required to inspect archives without DerivaML"
    Collaborators who receive a `.bag.zip` archive and do not have DerivaML installed need the `bdbag` command-line tool (`pip install bdbag`) to validate and materialize the bag. The archive is a standard BagIt container and can be inspected with any BagIt-compatible tool.

---

## See also

- [Working with datasets](datasets.md) — `download_dataset_bag`, `estimate_bag_size`, `DatasetSpec` with `timeout`
- [Working offline](offline.md) — `DatasetBag`, `restructure_assets`, offline feature access
- [Running an experiment](executions.md) — how executions link inputs and outputs for provenance
- [Deriva documentation](https://docs.derivacloud.org) — catalog administration, ACLs, Hatrac

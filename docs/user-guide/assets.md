# Working with assets

Assets are file-backed catalog records — model weights, images, segmentation
masks, embeddings, plots, logs. Each asset row stores the file's metadata
(filename, byte length, MD5, object-store URL, type tags) and, for files
produced or consumed inside an execution, a provenance link to that execution.
By the end of this chapter you will know how to **upload** a file as an asset,
**download** an asset's bytes, reference assets correctly by RID, and read
asset metadata without moving any bytes.

## What is an asset?

An asset is a row in an **asset table** (e.g. `Model`, `Image`, `Execution_Metadata`)
whose file content lives in the object store (Hatrac), not in the relational
row. The row carries:

- `RID` — the stable identifier. **Always reference an asset by RID, never by
  filename** — filenames are not unique and can change; the RID is immutable.
- `Filename`, `Length`, `MD5`, `URL` — file metadata.
- `Asset_Type` — a set of vocabulary tags (e.g. `Model_File`, plus the
  directional `Input_File` / `Output_File` tags described below).
- A provenance link to the execution that produced or consumed it, via the
  `{Asset}_Execution` association row.

The metadata half of the asset lifecycle (listing, looking up, retagging) is
catalog state you can read and edit directly. Moving file **bytes** — upload and
download — happens through an execution, in local Python, so the transfer is
recorded as walkable provenance.

## How to upload a file as an asset

Asset *creation from a local file* happens inside an execution context, using
`exe.asset_file_path()` to obtain a path to write to, followed by
`exe.commit_output_assets()` after the context exits. This is the same flow
covered in detail under "How to write asset files" and "How execution-asset
roles work" in [Chapter 5 — Running an execution](executions.md); the short
version:

```python
with ml.create_execution(config) as exe:
    # asset_file_path registers the file for upload and returns a Path to write.
    model_path = exe.asset_file_path("Model", "best_model.pt", asset_types=["Model_File"])
    torch.save(model.state_dict(), model_path)

# Upload happens AFTER the context manager exits:
exe.commit_output_assets()
```

The new `Model` row is tagged `Output_File` automatically (in addition to any
`asset_types` you pass) because it was produced by this execution.

**Notes**

- `asset_file_path` signature:
  `asset_file_path(asset_name, file_name, asset_types=None, copy_file=False, rename_file=None, metadata=None, description=None)`.
- Bytes are not uploaded until `commit_output_assets()` runs *after* the `with`
  block — calling it inside the block, or omitting it, leaves the file unuploaded.
- See [executions.md](executions.md) for the five `asset_file_path` modes
  (new file, symlink, copy, rename, metadata-only).

## How to download an asset

To pull an asset's bytes to a local directory, use `exe.download_asset()` from
inside an execution. This is the provenance-tracked counterpart to upload: the
asset is recorded as an **input** of the execution.

```python
with ml.create_execution(config) as exe:
    # Downloads to dest_dir / <asset's Filename>; records the asset as an input.
    local = exe.download_asset(asset_rid="2-ABCD", dest_dir=exe.working_dir / "downloads")
    weights = torch.load(local.path)
```

`download_asset` writes the file to `dest_dir / asset_record["Filename"]`. With
the default `update_catalog=True`, deriva-ml adds the `Input_File` Asset_Type tag
and writes `Asset_Role="Input"` on the `{Asset}_Execution` row — symmetric with
the `Output_File` tag that upload adds. The asset's pre-existing content tags
(e.g. `Model_File`) are **preserved**; the directional tag is additive.

**Notes**

- `download_asset` signature:
  `download_asset(asset_rid, dest_dir, update_catalog=True, use_cache=False)`.
- Re-downloading to a path whose bytes already match the catalog's MD5 is a
  silent no-op; a byte-different existing file is overwritten with a logged
  WARNING. For collision-free downloads, use a per-asset directory — the
  canonical pattern is `dest_dir = working_dir / "downloads" / asset_rid`.
- Pass `update_catalog=False` only for ad-hoc fetches you do *not* want recorded
  as an execution input.

### Downloading outside an execution

If you only need the bytes and do not want a provenance edge — e.g. a one-off
inspection script — use the bare `Asset.download()` primitive:

```python
asset = ml.lookup_asset("2-ABCD")
path = asset.download(dest_dir="/tmp/inspect")
```

This writes the file and returns its path **without** tagging any execution.
Prefer `exe.download_asset()` whenever the download is part of real work whose
provenance should be reproducible; reserve `Asset.download()` for throwaway reads.

**Notes**

- `Asset.download` signature: `download(dest_dir) -> Path`.
- No `Input_File` tag, no `{Asset}_Execution` row is written — this is
  deliberately outside the provenance graph.

## How to read asset metadata without moving bytes

Most questions about assets — what type is it, how big, which execution produced
it, what's its URL — are answered from catalog metadata alone, with no download.

```python
asset = ml.lookup_asset("2-ABCD")
print(asset.filename, asset.length, asset.md5)
print(asset.asset_types)              # e.g. ["Model_File", "Output_File"]
print(asset.url)                      # object-store URL
```

To enumerate assets in a table, use `ml.list_assets(asset_table)`; to search the
catalog, use `ml.find_assets(...)`.

**Filtering by type — use membership, not equality.** Because deriva-ml appends
the directional `Input_File` / `Output_File` tags, an asset's `asset_types` is
usually a *set* of tags, not a single value. Test membership:

```python
# Correct:
models = [a for a in ml.list_assets("Model") if "Model_File" in a.asset_types]

# Wrong — silently misses every asset that also carries a directional tag:
models = [a for a in ml.list_assets("Model") if a.asset_types == ["Model_File"]]
```

**Notes**

- `list_assets` is parent-scoped (takes an asset table); `find_assets` is a
  catalog-wide search. See [exploring.md](exploring.md) for the `find_*` vs
  `list_*` distinction.
- Retag or re-describe an existing asset with `ml.update_asset(asset_rid,
  asset_types=[...], description=...)` — the type list is set-style (the tool
  diffs against the current tags). Do not use raw `update_entities` on asset
  rows; it bypasses the tag-diff and provenance handling.

## Summary

| Task | Use | Provenance edge? |
|---|---|---|
| Upload a produced file | `exe.asset_file_path()` + `exe.commit_output_assets()` | yes — `Output_File` |
| Download for real work | `exe.download_asset(rid, dest_dir)` | yes — `Input_File` |
| Download for a throwaway read | `asset.download(dest_dir)` | no |
| Read metadata only | `ml.lookup_asset(rid)` / `list_assets` / `find_assets` | n/a (no bytes moved) |

The two byte-moving operations (`asset_file_path`, `download_asset`) run in local
Python inside an execution because they need filesystem access and must be
recorded against the execution's provenance. The metadata operations are plain
catalog reads/edits available directly (and over the MCP surface).

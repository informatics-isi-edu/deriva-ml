# API Changes Requiring MCP/Script Updates

Tracking document for all API changes made in this session that need to be reflected in the MCP server and user-facing scripts.

## New APIs

| API | Module | Description | MCP Impact |
|-----|--------|-------------|------------|
| `ml.asset_record_class(table)` | `core/mixins/asset.py` | Generate typed Pydantic model for asset metadata | New MCP tool or resource |
| `AssetFilePath.metadata` property | `asset/aux_classes.py` | Set/get typed metadata on asset paths | Document in MCP instructions |
| `AssetFilePath.set_asset_types()` | `asset/aux_classes.py` | Set asset types with manifest persistence | Document in MCP instructions |
| `AssetManifest` class | `asset/manifest.py` | Persistent JSON manifest for execution assets | Internal — no direct MCP exposure |
| `AssetRecord` base class | `asset/asset_record.py` | Base for dynamically generated asset metadata models | Referenced by asset_record_class |

## Changed APIs

| API | Change | MCP Impact |
|-----|--------|------------|
| `asset_file_path()` | New `metadata` param (AssetRecord or dict) | Update MCP `asset_file_path` tool |
| `download_dataset_bag()` | New `fetch_concurrency` param (default 8) | Update MCP `download_dataset` tool |
| `prefetch()` | New `fetch_concurrency` param (default 8) | No MCP tool for this yet |

## Completed Renames (backward-compatible aliases kept)

| Current Name | New Name | Files Affected | MCP Impact |
|-------------|----------|----------------|------------|
| `Dataset.prefetch()` | `Dataset.cache()` | dataset.py | Update MCP tool if exposed; old name still works |
| `DerivaML.prefetch_dataset()` | `DerivaML.cache_dataset()` | dataset mixin | Update MCP tool if exposed; old name still works |

## Storage Layout Changes

| Before | After | Impact |
|--------|-------|--------|
| `asset/{schema}/{table}/{meta1}/{meta2}/.../file` | `assets/{table}/file` + `asset-manifest.json` | Upload staging creates symlinks at upload time |
| `asset-type/{schema}/{table}.jsonl` | Still written (backward compat) + manifest | No MCP impact |

## Scripts to Update

- [ ] MCP server `asset_file_path` tool — add `metadata` parameter
- [ ] MCP server `download_dataset` tool — add `fetch_concurrency` parameter
- [ ] MCP server instructions — document AssetRecord pattern
- [ ] Any user scripts that directly access the shadow directory structure

"""Localize remote hatrac assets to a local catalog server."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, quote as urlquote

from deriva.core import ErmrestCatalog, HatracStore, get_credential

if TYPE_CHECKING:
    from deriva_ml import DerivaML

logger = logging.getLogger("deriva_ml")


@dataclass
class LocalizeResult:
    """Result of an asset localization operation.

    Attributes:
        assets_processed: Number of assets successfully localized.
        assets_skipped: Number of assets skipped (already local or errors).
        assets_failed: Number of assets that failed to localize.
        errors: List of error messages for failed assets.
        localized_assets: List of (RID, old_url, new_url) tuples for successfully localized assets.
    """

    assets_processed: int = 0
    assets_skipped: int = 0
    assets_failed: int = 0
    errors: list[str] = field(default_factory=list)
    localized_assets: list[tuple[str, str, str]] = field(default_factory=list)


def localize_assets(
    catalog: DerivaML | ErmrestCatalog,
    asset_table: str,
    asset_rids: list[str],
    schema_name: str | None = None,
    hatrac_namespace: str | None = None,
    chunk_size: int | None = None,
    dry_run: bool = False,
) -> LocalizeResult:
    """Localize remote hatrac assets to the local catalog server.

    Downloads assets from remote hatrac servers (determined from the URL in each
    asset record) and uploads them to the local hatrac server, updating the asset
    table URLs to point to the local copies.

    This is useful after cloning a catalog with asset_mode="refs" where the
    asset URLs still point to the source server. Use this function to make
    the assets fully local.

    The source hatrac server for each asset is determined automatically from
    the URL stored in the asset record.

    This function is optimized for bulk operations:
    - Fetches all asset records in a single query
    - Caches connections to remote hatrac servers
    - Batches catalog updates for efficiency
    - Supports chunked uploads for large files

    Args:
        catalog: A DerivaML instance or ErmrestCatalog connected to the catalog.
        asset_table: Name of the asset table containing the assets to localize.
        asset_rids: List of asset RIDs to localize. Each RID should be a record
            in the asset table.
        schema_name: Schema containing the asset table. If None, searches all schemas.
        hatrac_namespace: Optional hatrac namespace for uploaded files. If None,
            uses "/hatrac/{asset_table}/{md5}.{filename}" pattern.
        chunk_size: Optional chunk size in bytes for large file uploads. If None,
            uses default chunking behavior.
        dry_run: If True, only report what would be done without making changes.

    Returns:
        LocalizeResult with counts and details of the operation.

    Raises:
        ValueError: If asset_table is not found.

    Examples:
        Localize specific assets using DerivaML:
            >>> from deriva_ml import DerivaML
            >>> ml = DerivaML("localhost", "42")
            >>> result = localize_assets(
            ...     ml,
            ...     asset_table="Image",
            ...     asset_rids=["1-ABC", "2-DEF", "3-GHI"],
            ... )
            >>> print(f"Localized {result.assets_processed} assets")

        Localize using ErmrestCatalog:
            >>> from deriva.core import DerivaServer
            >>> server = DerivaServer("https", "localhost")
            >>> catalog = server.connect_ermrest("42")
            >>> result = localize_assets(
            ...     catalog,
            ...     asset_table="Model_Weights",
            ...     asset_rids=["4-JKL"],
            ...     dry_run=True,
            ... )
    """
    result = LocalizeResult()

    # Extract catalog and hostname from the input
    ermrest_catalog, hostname, credential = _get_catalog_info(catalog)

    # Create pathbuilder for datapath queries
    pb = ermrest_catalog.getPathBuilder()

    # Find the asset table
    table_path, found_schema = _find_asset_table_path(pb, asset_table, schema_name)
    if table_path is None:
        raise ValueError(f"Asset table '{asset_table}' not found in catalog")

    # Set up local hatrac
    local_hatrac = HatracStore("https", hostname, credentials=credential)

    # Determine hatrac namespace
    if hatrac_namespace is None:
        hatrac_namespace = f"/hatrac/{asset_table}"

    # Fetch all asset records in a single query
    logger.info(f"Fetching {len(asset_rids)} asset records...")
    all_records = _fetch_asset_records(table_path, asset_rids)

    # Build a map of RID -> record for easy lookup
    records_by_rid = {r["RID"]: r for r in all_records}

    # Identify which assets need to be localized
    assets_to_localize = []
    for rid in asset_rids:
        record = records_by_rid.get(rid)
        if record is None:
            logger.warning(f"Asset {rid} not found")
            result.assets_skipped += 1
            continue

        current_url = record.get("URL")
        if not current_url:
            logger.warning(f"Asset {rid} has no URL, skipping")
            result.assets_skipped += 1
            continue

        # Parse the URL to get source hostname
        parsed_url = urlparse(current_url)
        source_hostname = parsed_url.netloc

        if not source_hostname:
            logger.info(f"Asset {rid} has relative URL, already local")
            result.assets_skipped += 1
            continue

        if source_hostname == hostname:
            logger.info(f"Asset {rid} is already local, skipping")
            result.assets_skipped += 1
            continue

        # Extract the hatrac path from the URL
        source_path = _extract_hatrac_path(current_url)
        if not source_path:
            logger.warning(f"Could not extract hatrac path from URL: {current_url}")
            result.assets_skipped += 1
            continue

        assets_to_localize.append({
            "rid": rid,
            "record": record,
            "source_hostname": source_hostname,
            "source_path": source_path,
            "current_url": current_url,
        })

    if not assets_to_localize:
        logger.info("No assets need to be localized")
        return result

    logger.info(f"Localizing {len(assets_to_localize)} assets...")

    if dry_run:
        for asset_info in assets_to_localize:
            logger.info(
                f"[DRY RUN] Would download {asset_info['source_path']} from "
                f"{asset_info['source_hostname']} and upload to {hatrac_namespace}"
            )
            result.assets_processed += 1
        return result

    # Cache for remote hatrac connections (keyed by hostname)
    remote_hatrac_cache: dict[str, HatracStore] = {}

    # Ensure local namespace exists
    _ensure_hatrac_namespace(local_hatrac, hatrac_namespace)

    # Collect updates for batch catalog update
    catalog_updates: list[dict] = []

    # Process each asset
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch_dir = Path(tmpdir)

        for i, asset_info in enumerate(assets_to_localize):
            rid = asset_info["rid"]
            record = asset_info["record"]
            source_hostname = asset_info["source_hostname"]
            source_path = asset_info["source_path"]
            current_url = asset_info["current_url"]
            filename = record.get("Filename")
            md5 = record.get("MD5")

            logger.info(f"[{i+1}/{len(assets_to_localize)}] Localizing {rid}: {filename} from {source_hostname}")

            try:
                # Get or create remote hatrac connection
                if source_hostname not in remote_hatrac_cache:
                    source_cred = get_credential(source_hostname)
                    remote_hatrac_cache[source_hostname] = HatracStore(
                        "https", source_hostname, credentials=source_cred
                    )
                source_hatrac = remote_hatrac_cache[source_hostname]

                # Download from source
                local_file = scratch_dir / (md5 or rid) / (filename or "asset")
                local_file.parent.mkdir(parents=True, exist_ok=True)

                source_hatrac.get_obj(path=source_path, destfilename=str(local_file))

                # Upload to local hatrac
                dest_path = f"{hatrac_namespace}/{md5}.{filename}" if md5 and filename else f"{hatrac_namespace}/{rid}"

                new_url = local_hatrac.put_loc(
                    dest_path,
                    str(local_file),
                    headers={"Content-Disposition": f"filename*=UTF-8''{urlquote(filename or 'asset')}"},
                    chunked=chunk_size is not None,
                    chunk_size=chunk_size or 0,
                )

                # Queue the catalog update
                catalog_updates.append({"RID": rid, "URL": new_url})

                logger.info(f"Localized asset {rid}: {current_url} -> {new_url}")
                result.assets_processed += 1
                result.localized_assets.append((rid, current_url, new_url))

                # Clean up scratch file
                if local_file.exists():
                    local_file.unlink()

            except Exception as e:
                error_msg = f"Failed to localize asset {rid}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.assets_failed += 1

    # Batch update the catalog records
    if catalog_updates:
        logger.info(f"Updating {len(catalog_updates)} catalog records...")
        try:
            table_path.path.update(catalog_updates)
            logger.info("Catalog records updated successfully")
        except Exception as e:
            # If batch update fails, try individual updates as fallback
            logger.warning(f"Batch update failed ({e}), falling back to individual updates...")
            for update in catalog_updates:
                try:
                    table_path.path.filter(table_path.RID == update["RID"]).update([update])
                except Exception as e2:
                    error_msg = f"Failed to update catalog record {update['RID']}: {e2}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

    return result


def _get_catalog_info(
    catalog: DerivaML | ErmrestCatalog,
) -> tuple[ErmrestCatalog, str, dict | None]:
    """Extract catalog, hostname, and credential from a DerivaML or ErmrestCatalog.

    Args:
        catalog: DerivaML instance or ErmrestCatalog.

    Returns:
        Tuple of (ErmrestCatalog, hostname, credential).
    """
    # Check if it's a DerivaML instance
    if hasattr(catalog, "catalog") and hasattr(catalog, "host_name"):
        # It's a DerivaML instance
        hostname = catalog.host_name
        ermrest_catalog = catalog.catalog
        credential = getattr(catalog, "credential", None) or get_credential(hostname)
        return (ermrest_catalog, hostname, credential)

    # It's an ErmrestCatalog
    ermrest_catalog = catalog
    # Extract hostname from the catalog's server_uri
    server_uri = ermrest_catalog.get_server_uri()
    parsed = urlparse(server_uri)
    hostname = parsed.netloc

    credential = get_credential(hostname) if hostname else None
    return (ermrest_catalog, hostname, credential)


def _find_asset_table_path(
    pb,
    table_name: str,
    schema_name: str | None,
) -> tuple | None:
    """Find an asset table using pathbuilder.

    Args:
        pb: PathBuilder instance.
        table_name: Name of the table to find.
        schema_name: Optional schema name. If None, searches all schemas.

    Returns:
        Tuple of (table_path, schema_name) if found, (None, None) otherwise.
    """
    if schema_name:
        try:
            table_path = pb.schemas[schema_name].tables[table_name]
            return (table_path, schema_name)
        except KeyError:
            return (None, None)

    # Search all schemas
    for sname in pb.schemas:
        try:
            table_path = pb.schemas[sname].tables[table_name]
            return (table_path, sname)
        except KeyError:
            continue

    return (None, None)


def _fetch_asset_records(table_path, rids: list[str]) -> list[dict]:
    """Fetch multiple asset records in a single query.

    Args:
        table_path: Datapath table object.
        rids: List of RIDs to fetch.

    Returns:
        List of record dictionaries.
    """
    if not rids:
        return []

    # Use datapath to fetch all records with RIDs in the list
    # Build a filter for RID in (rid1, rid2, ...)
    from deriva.core.datapath import DataPathException

    try:
        # Fetch records where RID is in our list
        # Use multiple OR conditions since datapath doesn't have an "in" operator
        path = table_path.path

        # Build filter: RID == rid1 OR RID == rid2 OR ...
        filter_expr = None
        for rid in rids:
            condition = table_path.RID == rid
            if filter_expr is None:
                filter_expr = condition
            else:
                filter_expr = filter_expr | condition

        if filter_expr is not None:
            path = path.filter(filter_expr)

        return list(path.entities().fetch())
    except DataPathException as e:
        logger.warning(f"Bulk fetch failed: {e}, falling back to individual fetches")
        # Fallback: fetch records individually
        records = []
        for rid in rids:
            try:
                result = list(table_path.path.filter(table_path.RID == rid).entities().fetch())
                records.extend(result)
            except Exception:
                pass
        return records


def _extract_hatrac_path(url: str) -> str | None:
    """Extract the hatrac path from a full URL.

    Args:
        url: Full URL like "https://host/hatrac/namespace/file"

    Returns:
        Hatrac path like "/hatrac/namespace/file" or None if not a hatrac URL.
    """
    parsed = urlparse(url)
    path = parsed.path

    if "/hatrac/" in path:
        # Find the /hatrac/ part and return from there
        idx = path.find("/hatrac/")
        return path[idx:]

    if path.startswith("/hatrac/"):
        return path

    return None


def _ensure_hatrac_namespace(hatrac: HatracStore, namespace: str) -> None:
    """Ensure a hatrac namespace exists, creating it if necessary.

    Args:
        hatrac: HatracStore instance.
        namespace: Namespace path like "/hatrac/MyTable".
    """
    try:
        # Try to create the namespace (will fail if exists, which is fine)
        hatrac.create_namespace(namespace, parents=True)
    except Exception:
        # Namespace likely already exists
        pass

"""Localize remote Hatrac assets to a local catalog server.

Copies assets referenced in a cloned catalog's asset tables from the source
Hatrac store to the local Hatrac instance. Intended for use after
``create_ml_workspace`` is called with ``asset_mode=REFERENCES``.

Three-stage flow:
1. Enumerate all rows in asset tables (tables with URL, Filename, MD5 columns).
2. For each row, download the file from the source Hatrac URL to a temp file,
   then upload it to the local Hatrac namespace.
3. Update the catalog row's URL to point to the new local Hatrac path.

The ``LocalizeResult`` model summarizes counts of processed/skipped/failed
assets and provides the old-to-new URL mapping for auditing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote as urlquote
from urllib.parse import urlparse

from deriva.core import ErmrestCatalog, HatracStore, get_credential
from pydantic import BaseModel, Field

from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.validation import VALIDATION_CONFIG

if TYPE_CHECKING:
    from deriva_ml import DerivaML

logger = get_logger(__name__)


class LocalizeResult(BaseModel):
    """Result of an asset localization operation.

    Attributes:
        assets_processed: Number of assets successfully localized.
        assets_skipped: Number of assets skipped (already local or errors).
        assets_failed: Number of assets that failed to localize.
        errors: List of error messages for failed assets.
        localized_assets: List of (RID, old_url, new_url) tuples for
            successfully localized assets.
    """

    model_config = VALIDATION_CONFIG

    assets_processed: int = 0
    assets_skipped: int = 0
    assets_failed: int = 0
    errors: list[str] = Field(default_factory=list)
    localized_assets: list[tuple[str, str, str]] = Field(default_factory=list)


def localize_assets(
    catalog: DerivaML | ErmrestCatalog,
    asset_table: str,
    asset_rids: list[str],
    schema_name: str | None = None,
    hatrac_namespace: str | None = None,
    chunk_size: int | None = None,
    dry_run: bool = False,
    source_hostname: str | None = None,
) -> LocalizeResult:
    """Phase-2 leg of split-phase slice copy: move asset bytes server-to-server.

    DerivaML's catalog-slice copy is intentionally split into two
    phases so the metadata clone and the asset-bytes movement can
    run independently (different schedules, different bandwidth
    budgets, different operator authority):

    - **Phase 1** — :func:`~deriva_ml.catalog.clone_via_bag.clone_via_bag`
      copies catalog rows from source server A → destination
      server B. With ``asset_mode=ROWS_ONLY`` (or
      ``REFERENCES``-style URL preservation) the rows in B reference
      assets that still live in A's Hatrac.
    - **Phase 2** (this function) — for the asset tables / RIDs
      the operator chooses, download each object from A's Hatrac,
      upload it to B's Hatrac, then rewrite the row's URL column
      to point at B. The Hatrac protocol has no
      server-to-server copy primitive, so this function mediates
      the transfer client-side.

    On completion (when ``dry_run=False``) the destination
    catalog's :class:`~deriva_ml.catalog.provenance.CatalogProvenance`
    annotation is updated with the phase-2 stats
    (``assets_localized=True``, ``assets_localized_at``,
    ``asset_source_hostname``, ``assets_copied``,
    ``assets_skipped``, ``assets_failed``) so the two-phase
    completion state is durably observable.

    The source Hatrac server for each asset is determined:

    1. From the URL when it's absolute (e.g.,
       ``https://source.example.org/hatrac/...``).
    2. From the ``source_hostname`` parameter when the URL is
       relative (e.g., ``/hatrac/...``) — needed because the
       cloned rows don't carry the source hostname inline.

    Optimized for bulk operations:

    - Fetches all asset records in a single query.
    - Caches connections to remote Hatrac servers.
    - Batches catalog updates (one ``table.update(...)`` for all
      successfully localized rows).
    - Supports chunked uploads for large files.

    Args:
        catalog: A DerivaML instance or ErmrestCatalog connected
            to the destination catalog.
        asset_table: Name of the asset table containing the
            assets to localize.
        asset_rids: List of asset RIDs to localize. Each must be
            a row in ``asset_table``.
        schema_name: Schema containing the asset table. If None,
            searches all schemas.
        hatrac_namespace: Hatrac namespace for uploaded files. If
            None, defaults to ``/hatrac/{asset_table}``.
        chunk_size: Chunk size in bytes for large file uploads.
            ``None`` uses Hatrac's default chunking. A positive
            integer overrides; a value of ``0`` is treated as
            ``None`` because zero chunks make no sense.
        dry_run: If True, log what would be done without making
            changes. Provenance is **not** updated in dry-run.
        source_hostname: Hostname to use for assets with relative
            URLs (e.g., ``"www.facebase.org"``). Required when
            localizing assets cloned with ``asset_mode=REFERENCES``
            from a different server.

    Returns:
        :class:`LocalizeResult` with counts and per-asset details.

    Raises:
        ValueError: If ``asset_table`` is not found.

    Examples:
        Localize specific assets using DerivaML:
            >>> from deriva_ml import DerivaML  # doctest: +SKIP
            >>> ml = DerivaML("localhost", "42")  # doctest: +SKIP
            >>> result = localize_assets(  # doctest: +SKIP
            ...     ml,
            ...     asset_table="Image",
            ...     asset_rids=["1-ABC", "2-DEF", "3-GHI"],
            ... )
            >>> print(f"Localized {result.assets_processed} assets")  # doctest: +SKIP

        Localize assets cloned from another server with relative URLs:
            >>> result = localize_assets(  # doctest: +SKIP
            ...     ml,
            ...     asset_table="file",
            ...     asset_rids=["TG0", "TG2"],
            ...     schema_name="isa",
            ...     source_hostname="www.facebase.org",
            ... )

        Localize using ErmrestCatalog:
            >>> from deriva.core import DerivaServer  # doctest: +SKIP
            >>> server = DerivaServer("https", "localhost")  # doctest: +SKIP
            >>> catalog = server.connect_ermrest("42")  # doctest: +SKIP
            >>> result = localize_assets(  # doctest: +SKIP
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
    logger.info("Fetching %d asset records...", len(asset_rids))
    all_records = _fetch_asset_records(table_path, asset_rids)

    # Build a map of RID -> record for easy lookup
    records_by_rid = {r["RID"]: r for r in all_records}

    # Detect URL column name from first record (try URL first, then url)
    url_column = "URL"
    if all_records:
        if "URL" not in all_records[0] and "url" in all_records[0]:
            url_column = "url"

    # Identify which assets need to be localized
    assets_to_localize = []
    for rid in asset_rids:
        record = records_by_rid.get(rid)
        if record is None:
            logger.warning("Asset %s not found", rid)
            result.assets_skipped += 1
            continue

        # Try both URL and url column names (different catalogs use different conventions)
        current_url = record.get("URL") or record.get("url")
        if not current_url:
            logger.warning("Asset %s has no URL column, skipping", rid)
            result.assets_skipped += 1
            continue

        # Parse the URL to get source hostname
        parsed_url = urlparse(current_url)
        asset_source_hostname = parsed_url.netloc

        if not asset_source_hostname:
            # URL is relative (e.g., /hatrac/facebase/data/...)
            if source_hostname:
                # Use provided source_hostname for relative URLs
                asset_source_hostname = source_hostname
                logger.info(
                    "Asset %s has relative URL, using source_hostname=%s",
                    rid,
                    source_hostname,
                )
            else:
                logger.info(
                    "Asset %s has relative URL, already local "
                    "(specify source_hostname to localize)",
                    rid,
                )
                result.assets_skipped += 1
                continue

        if asset_source_hostname == hostname:
            logger.info("Asset %s is already local, skipping", rid)
            result.assets_skipped += 1
            continue

        # Extract the hatrac path from the URL
        source_path = _extract_hatrac_path(current_url)
        if not source_path:
            logger.warning("Could not extract hatrac path from URL: %s", current_url)
            result.assets_skipped += 1
            continue

        assets_to_localize.append(
            {
                "rid": rid,
                "record": record,
                "source_hostname": asset_source_hostname,
                "source_path": source_path,
                "current_url": current_url,
            }
        )

    if not assets_to_localize:
        logger.info("No assets need to be localized")
        return result

    logger.info("Localizing %d assets...", len(assets_to_localize))

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
            # Per-asset source host (may differ across rows in a
            # mixed-source slice). Distinct from the function-scoped
            # ``source_hostname`` parameter, which is the fallback
            # used when the asset URL is relative.
            asset_src_host = asset_info["source_hostname"]
            source_path = asset_info["source_path"]
            current_url = asset_info["current_url"]
            # Handle case variations in column names
            filename = record.get("Filename") or record.get("filename")
            md5 = record.get("MD5") or record.get("md5")

            logger.info(
                "[%d/%d] Localizing %s: %s from %s",
                i + 1,
                len(assets_to_localize),
                rid,
                filename,
                asset_src_host,
            )

            try:
                # Get or create remote hatrac connection
                if asset_src_host not in remote_hatrac_cache:
                    source_cred = get_credential(asset_src_host)
                    remote_hatrac_cache[asset_src_host] = HatracStore(
                        "https", asset_src_host, credentials=source_cred
                    )
                source_hatrac = remote_hatrac_cache[asset_src_host]

                # Download from source
                local_file = scratch_dir / (md5 or rid) / (filename or "asset")
                local_file.parent.mkdir(parents=True, exist_ok=True)

                source_hatrac.get_obj(path=source_path, destfilename=str(local_file))

                # Upload to local hatrac
                dest_path = f"{hatrac_namespace}/{md5}.{filename}" if md5 and filename else f"{hatrac_namespace}/{rid}"

                # Enable chunking for large files (> 100MB) by default.
                # ``chunk_size=0`` from the caller is treated as
                # "use the default" rather than "no chunks" — a
                # zero-chunk upload makes no sense and is almost
                # certainly a caller passing a sentinel they expected
                # to be ignored.
                file_size = local_file.stat().st_size
                default_chunk_size = 50 * 1024 * 1024  # 50MB chunks
                use_chunked = (
                    chunk_size is not None and chunk_size > 0
                ) or file_size > 100 * 1024 * 1024
                actual_chunk_size = (
                    chunk_size if (chunk_size is not None and chunk_size > 0) else default_chunk_size
                )

                new_url = local_hatrac.put_loc(
                    dest_path,
                    str(local_file),
                    headers={"Content-Disposition": f"filename*=UTF-8''{urlquote(filename or 'asset')}"},
                    chunked=use_chunked,
                    chunk_size=actual_chunk_size if use_chunked else 0,
                )

                # Queue the catalog update using the detected URL column name
                catalog_updates.append({"RID": rid, url_column: new_url})

                logger.info("Localized asset %s: %s -> %s", rid, current_url, new_url)
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

    # Batch update the catalog records using datapath
    if catalog_updates:
        logger.info("Updating %d catalog records...", len(catalog_updates))
        try:
            # Use datapath update - table_path.update() handles the update correctly
            table_path.update(catalog_updates)
            logger.info("Updated %d catalog records successfully", len(catalog_updates))
        except Exception as e:
            # If batch update fails, try individual updates as fallback
            logger.warning("Batch update failed (%s), falling back to individual updates...", e)
            for update in catalog_updates:
                rid = update["RID"]
                try:
                    table_path.update([update])
                    logger.info("Updated catalog record %s", rid)
                except Exception as e2:
                    error_msg = f"Failed to update catalog record {rid}: {e2}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

    # Phase 2 of split-phase clone: record asset-localization stats
    # on the catalog's provenance annotation so an operator can see
    # that the asset bytes have been moved (and from where). The
    # clone_via_bag step wrote phase-1 provenance; we mutate it
    # rather than replace.
    if not dry_run:
        _record_localize_provenance(
            ermrest_catalog,
            result=result,
            asset_source_hostname=source_hostname,
        )

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

    # Fetch the requested rows via the datapath ``column.in_(values)``
    # operator (added in deriva-py #242). The bulk-fetch path issues
    # one ``RID=any(...)`` request; on failure, fall back to one
    # fetch per RID.
    from deriva.core.datapath import DataPathException

    try:
        return list(table_path.path.filter(table_path.RID.in_(list(rids))).entities().fetch())
    except DataPathException as e:
        logger.warning("Bulk fetch failed: %s, falling back to individual fetches", e)
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

    Only returns a path for absolute HTTP(S) URLs or pure paths
    (no scheme). Other schemes (``ftp://``, ``file://`` etc.)
    return ``None`` because they cannot be served by a Hatrac
    object store and almost always indicate corrupted asset data
    that should be skipped rather than fetched.

    Args:
        url: Full URL like ``"https://host/hatrac/namespace/file"``,
            or a relative URL like ``"/hatrac/namespace/file"``.

    Returns:
        Hatrac path like ``"/hatrac/namespace/file"`` or ``None``
        when the URL doesn't reference a Hatrac path or uses an
        unsupported scheme.
    """
    parsed = urlparse(url)
    # Reject non-HTTP schemes. An empty scheme is allowed (it's a
    # relative URL — the only kind that's allowed alongside http/https).
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return None

    path = parsed.path
    if "/hatrac/" in path:
        # Find the /hatrac/ part and return from there. This also
        # handles the path.startswith("/hatrac/") case, which the
        # previous code split into a second unreachable branch.
        idx = path.find("/hatrac/")
        return path[idx:]

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


def _record_localize_provenance(
    catalog: ErmrestCatalog,
    *,
    result: LocalizeResult,
    asset_source_hostname: str | None,
) -> None:
    """Update the destination's provenance annotation with phase-2 stats.

    Phase 1 (:func:`~deriva_ml.catalog.clone_via_bag.clone_via_bag`)
    writes a ``CatalogProvenance`` annotation with
    ``creation_method=CLONE`` and partially-populated
    ``CloneDetails``. This function fills in the localization-leg
    fields (``assets_localized``, ``assets_localized_at``,
    ``asset_source_hostname``, ``assets_copied``,
    ``assets_skipped``, ``assets_failed``) so a future reader can
    see that the bytes have been moved server-to-server and from
    where.

    Best-effort: a missing or malformed phase-1 annotation does
    not block localization. In that case we write a fresh
    annotation with ``creation_method=CLONE`` and only the
    phase-2 stats populated — the catalog was clearly cloned
    (we just localized assets in it) but we don't know the
    phase-1 details.

    Args:
        catalog: Destination ERMrest catalog handle.
        result: The completed :class:`LocalizeResult`.
        asset_source_hostname: The hostname the assets were
            moved *from*. ``None`` is preserved as ``None`` on
            the annotation — useful when the caller couldn't
            determine a single source (e.g., mixed-source slice).
    """
    # Lazy import: avoids a circular reference between
    # catalog/localize.py and catalog/provenance.py if either ever
    # grows a dependency in the other direction.
    from datetime import datetime, timezone

    from deriva_ml.catalog.provenance import (
        CatalogCreationMethod,
        CloneDetails,
        get_catalog_provenance,
        set_catalog_provenance,
    )

    try:
        existing = get_catalog_provenance(catalog)
        if existing is not None and existing.clone_details is not None:
            # Phase 1 already wrote a CloneDetails; update its
            # localization-leg fields in place.
            details = existing.clone_details
            updated = details.model_copy(
                update={
                    "assets_localized": True,
                    "assets_localized_at": datetime.now(timezone.utc).isoformat(),
                    "asset_source_hostname": asset_source_hostname,
                    "assets_copied": result.assets_processed,
                    "assets_skipped": result.assets_skipped,
                    "assets_failed": result.assets_failed,
                }
            )
        else:
            # No phase-1 details (catalog wasn't cloned via
            # clone_via_bag, or the annotation was lost). Write a
            # fresh CloneDetails with only the phase-2 fields.
            updated = CloneDetails(
                source_hostname=asset_source_hostname or "",
                source_catalog_id="",
                assets_localized=True,
                assets_localized_at=datetime.now(timezone.utc).isoformat(),
                asset_source_hostname=asset_source_hostname,
                assets_copied=result.assets_processed,
                assets_skipped=result.assets_skipped,
                assets_failed=result.assets_failed,
            )

        set_catalog_provenance(
            catalog,
            creation_method=CatalogCreationMethod.CLONE,
            clone_details=updated,
            # Preserve descriptive fields from the existing annotation
            # if they were set; otherwise leave None.
            name=existing.name if existing else None,
            description=existing.description if existing else None,
            workflow_url=existing.workflow_url if existing else None,
            workflow_version=existing.workflow_version if existing else None,
        )
    except Exception as e:
        # Provenance failures must not break localization.
        logger.warning("localize_assets: failed to update provenance: %s", e)

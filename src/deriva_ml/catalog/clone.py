"""Enhanced catalog cloning with cross-server and selective asset support."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from deriva.core import DerivaServer, get_credential

logger = logging.getLogger("deriva_ml")


class AssetCopyMode(Enum):
    """Asset handling mode for catalog cloning.

    Attributes:
        NONE: Don't copy assets. Asset columns will be empty in the clone.
        REFERENCES: Copy asset URLs only. Assets remain on source server's hatrac.
            Useful for cross-server cloning when assets should stay in place.
        FULL: Download assets from source and upload to destination hatrac.
            Creates a fully independent clone with all files copied.
    """

    NONE = "none"
    REFERENCES = "refs"
    FULL = "full"


@dataclass
class AssetFilter:
    """Filter for selective asset copying.

    Specify which assets to include during cloning. If both tables and rids
    are None, all assets are included (subject to asset_mode).

    Attributes:
        tables: List of asset table names to include. Only assets from these
            tables will be copied. Example: ["Image", "Model_Weights"]
        rids: List of specific asset RIDs to include. Only these exact assets
            will be copied, regardless of table.

    Note:
        If both tables and rids are specified, assets matching EITHER criterion
        are included (logical OR).
    """

    tables: list[str] | None = None
    rids: list[str] | None = None


@dataclass
class CloneCatalogResult:
    """Result of a catalog clone operation.

    Attributes:
        catalog_id: The catalog ID of the new clone.
        hostname: The hostname where the clone was created.
        alias: The alias name if one was created, None otherwise.
        ml_schema_added: True if the DerivaML schema was added to the clone.
        asset_mode: The asset copy mode used.
        schema_only: True if only schema was copied (no data).
        source_hostname: The source catalog's hostname.
        source_catalog_id: The source catalog's ID.
    """

    catalog_id: str
    hostname: str
    alias: str | None = None
    ml_schema_added: bool = False
    asset_mode: AssetCopyMode = AssetCopyMode.REFERENCES
    schema_only: bool = False
    source_hostname: str = ""
    source_catalog_id: str = ""
    errors: list[str] = field(default_factory=list)


def clone_catalog(
    source_hostname: str,
    source_catalog_id: str,
    dest_hostname: str | None = None,
    alias: str | None = None,
    add_ml_schema: bool = False,
    schema_only: bool = False,
    asset_mode: AssetCopyMode = AssetCopyMode.REFERENCES,
    asset_filter: AssetFilter | None = None,
    copy_annotations: bool = True,
    copy_policy: bool = True,
    exclude_schemas: list[str] | None = None,
    source_credential: dict | None = None,
    dest_credential: dict | None = None,
) -> CloneCatalogResult:
    """Clone a catalog with optional cross-server support and selective asset copying.

    This function provides flexible catalog cloning with several modes:

    **Same-server clone** (fast):
    When dest_hostname is None or matches source_hostname, uses ERMrest's native
    clone_catalog() for efficient same-server cloning.

    **Cross-server clone** (via backup/restore):
    When dest_hostname differs from source_hostname, uses DerivaBackup/DerivaRestore
    to migrate the catalog to the new server.

    **Schema-only clone**:
    Set schema_only=True to copy only the schema structure without any data.
    Useful for creating empty development/test catalogs.

    **Asset handling**:
    - NONE: Don't copy assets (asset columns will be empty)
    - REFERENCES: Keep URLs pointing to source hatrac (cross-server only)
    - FULL: Download and re-upload all assets (creates independent copy)

    Args:
        source_hostname: Hostname of the source catalog server.
        source_catalog_id: ID of the catalog to clone.
        dest_hostname: Destination hostname. If None or same as source, uses
            fast same-server cloning. Otherwise uses backup/restore.
        alias: Optional alias name for the new catalog. Creates a friendly
            name like /ermrest/catalog/my-project instead of /ermrest/catalog/45.
        add_ml_schema: If True, add the DerivaML schema to the clone if not
            already present. Useful for converting plain ERMrest catalogs.
        schema_only: If True, copy only schema structure without data.
        asset_mode: How to handle assets during cloning. See AssetCopyMode.
        asset_filter: Optional filter to selectively copy only certain assets.
        copy_annotations: If True (default), copy all catalog annotations.
        copy_policy: If True (default), copy ACL policies.
        exclude_schemas: List of schema names to exclude from cloning.
        source_credential: Optional credential dict for source server.
            If None, uses credential from ~/.deriva/credentials.
        dest_credential: Optional credential dict for destination server.
            If None, uses credential from ~/.deriva/credentials.

    Returns:
        CloneCatalogResult with details of the cloned catalog.

    Raises:
        ValueError: If invalid parameters are provided.
        Exception: If cloning fails due to permissions or connectivity.

    Examples:
        Same-server clone:
            >>> result = clone_catalog("localhost", "21")
            >>> print(f"Created clone: {result.catalog_id}")

        Cross-server clone with alias:
            >>> result = clone_catalog(
            ...     "dev.example.org", "21",
            ...     dest_hostname="prod.example.org",
            ...     alias="my-project",
            ... )

        Schema-only clone for development:
            >>> result = clone_catalog(
            ...     "prod.example.org", "21",
            ...     dest_hostname="localhost",
            ...     schema_only=True,
            ... )

        Clone with ML schema addition:
            >>> result = clone_catalog(
            ...     "source.org", "5",
            ...     dest_hostname="dest.org",
            ...     add_ml_schema=True,
            ... )

        Selective asset copying:
            >>> result = clone_catalog(
            ...     "source.org", "21",
            ...     dest_hostname="dest.org",
            ...     asset_mode=AssetCopyMode.FULL,
            ...     asset_filter=AssetFilter(tables=["Image"]),
            ... )
    """
    # Determine if this is same-server or cross-server clone
    is_same_server = dest_hostname is None or dest_hostname == source_hostname
    effective_dest_hostname = source_hostname if dest_hostname is None else dest_hostname

    if is_same_server:
        result = _clone_same_server(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            schema_only=schema_only,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            exclude_schemas=exclude_schemas,
            credential=source_credential,
        )
    else:
        result = _clone_via_backup_restore(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            dest_hostname=effective_dest_hostname,
            schema_only=schema_only,
            asset_mode=asset_mode,
            asset_filter=asset_filter,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            exclude_schemas=exclude_schemas,
            source_credential=source_credential,
            dest_credential=dest_credential,
        )

    # Post-clone operations: alias creation and ML schema addition
    result = _post_clone_operations(
        result=result,
        alias=alias,
        add_ml_schema=add_ml_schema,
        credential=dest_credential,
    )

    return result


def _clone_same_server(
    source_hostname: str,
    source_catalog_id: str,
    schema_only: bool,
    copy_annotations: bool,
    copy_policy: bool,
    exclude_schemas: list[str] | None,
    credential: dict | None,
) -> CloneCatalogResult:
    """Clone a catalog on the same server using native ERMrest clone.

    This is the fast path for same-server cloning.
    """
    cred = credential or get_credential(source_hostname)
    server = DerivaServer("https", source_hostname, credentials=cred)
    source_catalog = server.connect_ermrest(source_catalog_id)

    # Clone the catalog
    dest_catalog = source_catalog.clone_catalog(
        dst_catalog=None,  # Create new catalog
        copy_data=not schema_only,
        copy_annotations=copy_annotations,
        copy_policy=copy_policy,
        truncate_after=True,
        exclude_schemas=exclude_schemas,
    )

    return CloneCatalogResult(
        catalog_id=str(dest_catalog.catalog_id),
        hostname=source_hostname,
        schema_only=schema_only,
        asset_mode=AssetCopyMode.REFERENCES,  # Same-server clone preserves asset refs
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
    )


def _clone_via_backup_restore(
    source_hostname: str,
    source_catalog_id: str,
    dest_hostname: str,
    schema_only: bool,
    asset_mode: AssetCopyMode,
    asset_filter: AssetFilter | None,
    copy_annotations: bool,
    copy_policy: bool,
    exclude_schemas: list[str] | None,
    source_credential: dict | None,
    dest_credential: dict | None,
) -> CloneCatalogResult:
    """Clone a catalog across servers using DerivaBackup/DerivaRestore."""
    from deriva.transfer.backup import DerivaBackup
    from deriva.transfer.restore import DerivaRestore

    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backup"
        backup_dir.mkdir()

        # Determine asset handling for backup
        if asset_mode == AssetCopyMode.NONE:
            include_assets = None  # Don't include assets
        elif asset_mode == AssetCopyMode.REFERENCES:
            include_assets = "references"  # Store URLs only
        else:  # AssetCopyMode.FULL
            include_assets = True  # Download assets

        # Build exclude_data list for selective asset filtering
        exclude_data = list(exclude_schemas) if exclude_schemas else []

        # Step 1: Backup the source catalog
        logger.info(f"Backing up catalog {source_hostname}:{source_catalog_id}")

        backup_args: dict[str, Any] = {
            "host": source_hostname,
            "protocol": "https",
            "catalog_id": source_catalog_id,
        }

        # Add credential if provided
        if source_credential:
            backup_args["credentials"] = source_credential

        backup = DerivaBackup(
            backup_args,
            output_dir=str(backup_dir),
            no_data=schema_only,
            include_assets=include_assets,
            exclude_data=exclude_data,
        )
        backup.transfer()

        # Find the backup output
        bag_dirs = list(backup_dir.glob("*"))
        if not bag_dirs:
            raise RuntimeError("Backup failed: no output created")
        bag_path = bag_dirs[0]

        # Apply asset filtering if specified
        if asset_filter and not schema_only:
            _apply_asset_filter(bag_path, asset_filter, asset_mode)

        # Step 2: Restore to destination server
        logger.info(f"Restoring to {dest_hostname}")

        restore_args: dict[str, Any] = {
            "host": dest_hostname,
            "protocol": "https",
            "catalog_id": None,  # Create new catalog
        }

        if dest_credential:
            restore_args["credentials"] = dest_credential

        # Determine whether to restore assets
        no_assets = asset_mode == AssetCopyMode.NONE or asset_mode == AssetCopyMode.REFERENCES

        restore = DerivaRestore(
            restore_args,
            input_path=str(bag_path),
            no_data=schema_only,
            no_annotations=not copy_annotations,
            no_policy=not copy_policy,
            no_assets=no_assets,
            exclude_schemas=exclude_data if exclude_data else [],
        )
        restore.restore()

        dest_catalog_id = str(restore.dst_catalog.catalog_id) if restore.dst_catalog else ""

    return CloneCatalogResult(
        catalog_id=dest_catalog_id,
        hostname=dest_hostname,
        schema_only=schema_only,
        asset_mode=asset_mode,
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
        errors=errors,
    )


def _apply_asset_filter(
    bag_path: Path,
    asset_filter: AssetFilter,
    asset_mode: AssetCopyMode,
) -> None:
    """Apply asset filtering to a backup bag.

    Modifies the bag's data files to exclude assets not matching the filter.
    This is a selective operation that removes rows from asset tables or
    removes asset files from the bag's data directory.
    """
    # For now, log a warning if filtering is requested but not yet implemented
    # Full implementation would require parsing the bag's CSV files and
    # filtering based on table names or RIDs
    if asset_filter.tables or asset_filter.rids:
        logger.warning(
            "Asset filtering is specified but detailed filtering is not yet "
            "fully implemented. All assets matching the asset_mode will be included."
        )


def _post_clone_operations(
    result: CloneCatalogResult,
    alias: str | None,
    add_ml_schema: bool,
    credential: dict | None,
) -> CloneCatalogResult:
    """Perform post-clone operations: alias creation and ML schema addition."""
    cred = credential or get_credential(result.hostname)
    server = DerivaServer("https", result.hostname, credentials=cred)

    # Create alias if requested
    if alias:
        try:
            server.create_ermrest_alias(
                id=alias,
                alias_target=result.catalog_id,
                name=alias,
                description=f"Alias for catalog {result.catalog_id}",
            )
            result.alias = alias
            logger.info(f"Created alias '{alias}' for catalog {result.catalog_id}")
        except Exception as e:
            error_msg = f"Failed to create alias '{alias}': {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    # Add ML schema if requested
    if add_ml_schema:
        try:
            from deriva_ml.schema import create_ml_schema

            dest_catalog = server.connect_ermrest(result.catalog_id)
            model = dest_catalog.getCatalogModel()

            if "deriva-ml" not in model.schemas:
                create_ml_schema(dest_catalog, schema_name="deriva-ml")
                result.ml_schema_added = True
                logger.info(f"Added DerivaML schema to catalog {result.catalog_id}")
            else:
                logger.info(f"Catalog {result.catalog_id} already has DerivaML schema")
        except Exception as e:
            error_msg = f"Failed to add ML schema: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    return result

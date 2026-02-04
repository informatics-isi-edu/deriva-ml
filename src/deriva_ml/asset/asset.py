"""Asset management for DerivaML.

This module provides the Asset class for managing assets in a Deriva catalog.
An asset represents a file-based record (image, model, data file, etc.) with
associated metadata, types, and provenance tracking.

The Asset class parallels the Dataset class, providing:
- Catalog-backed entity access via RID
- Type management (add/remove asset types)
- Provenance tracking (which executions created/used the asset)
- Feature discovery (features defined on the asset or its referenced tables)
- Download capability for offline access

Typical usage:
    >>> # Look up an existing asset
    >>> asset = ml.lookup_asset("3JSE")
    >>> print(f"Asset: {asset.filename} ({asset.asset_table})")
    >>> print(f"Types: {asset.asset_types}")

    >>> # Find the execution that created this asset
    >>> executions = asset.list_executions(asset_role="Output")
    >>> creator = executions[0] if executions else None

    >>> # Download for offline use
    >>> local_path = asset.download(Path("/tmp/assets"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deriva_ml.core.definitions import RID

if TYPE_CHECKING:
    from deriva_ml.execution.execution_record import ExecutionRecord
    from deriva_ml.feature import Feature, FeatureRecord
    from deriva_ml.interfaces import DerivaMLCatalog

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Table = _ermrest_model.Table


class Asset:
    """Manages asset operations in a Deriva catalog.

    The Asset class provides functionality for accessing and managing assets
    in a Deriva catalog. It handles metadata, type associations, and provenance.

    An Asset is a file-based record in an asset table (Image, Model, etc.)
    with associated metadata, controlled vocabulary types, and execution tracking.

    The class provides a consistent interface parallel to Dataset, allowing
    code to work uniformly with both data collections and individual assets.

    Attributes:
        asset_rid (RID): The unique Resource Identifier for this asset.
        asset_table (str): Name of the asset table containing this asset.
        filename (str): Original filename of the asset.
        url (str): URL to access the asset file.
        length (int): Size of the asset file in bytes.
        md5 (str): MD5 checksum of the asset file.
        asset_types (list[str]): List of vocabulary terms describing the asset type.
        description (str): Human-readable description of the asset.
        _ml_instance (DerivaMLCatalog): Reference to the catalog containing this asset.

    Example:
        >>> # Look up an existing asset
        >>> asset = ml.lookup_asset("3JSE")
        >>> print(f"File: {asset.filename}, Size: {asset.length} bytes")
        >>> print(f"Types: {asset.asset_types}")

        >>> # Find executions that used this asset
        >>> for exe in asset.list_executions():
        ...     print(f"Execution {exe.execution_rid}: {exe.configuration.description}")
    """

    def __init__(
        self,
        catalog: "DerivaMLCatalog",
        asset_rid: RID,
        asset_table: str,
        filename: str = "",
        url: str = "",
        length: int = 0,
        md5: str = "",
        description: str = "",
        asset_types: list[str] | None = None,
        execution_rid: RID | None = None,
    ):
        """Initialize an Asset object from an existing asset in the catalog.

        This constructor wraps an existing asset record. To create a new asset
        in the catalog, use Execution.asset_file_path() and upload_execution_outputs().

        Args:
            catalog: The DerivaMLCatalog instance containing this asset.
            asset_rid: The RID of the existing asset record.
            asset_table: Name of the asset table (e.g., "Image", "Model").
            filename: Original filename of the asset.
            url: URL to access the asset file.
            length: Size of the asset file in bytes.
            md5: MD5 checksum of the asset file.
            description: Human-readable description.
            asset_types: List of asset type vocabulary terms.
            execution_rid: RID of the execution that created this asset (if known).

        Example:
            >>> # Usually created via ml.lookup_asset()
            >>> asset = ml.lookup_asset("3JSE")
        """
        self._logger = logging.getLogger("deriva_ml")
        self._ml_instance = catalog
        self.asset_rid = asset_rid
        self.asset_table = asset_table
        self.filename = filename
        self.url = url
        self.length = length
        self.md5 = md5
        self.description = description
        self._asset_types = asset_types or []
        self._execution_rid = execution_rid

    def __repr__(self) -> str:
        """Return a string representation of the Asset for debugging."""
        return (
            f"<deriva_ml.Asset at {hex(id(self))}: rid='{self.asset_rid}', "
            f"table='{self.asset_table}', file='{self.filename}', types={self._asset_types}>"
        )

    @property
    def asset_types(self) -> list[str]:
        """Get the asset types for this asset.

        Returns:
            List of asset type vocabulary terms.
        """
        if not self._asset_types:
            self._load_asset_types()
        return self._asset_types

    def _load_asset_types(self) -> None:
        """Load asset types from the catalog."""
        # Find the asset type association table
        asset_table_obj = self._ml_instance.model.name_to_table(self.asset_table)
        try:
            type_assoc_table, asset_fk, _ = self._ml_instance.model.find_association(
                asset_table_obj, "Asset_Type"
            )
        except Exception:
            # No type association for this asset table
            self._asset_types = []
            return

        pb = self._ml_instance.pathBuilder()
        type_path = pb.schemas[type_assoc_table.schema.name].tables[type_assoc_table.name]

        types = list(
            type_path.filter(type_path.columns[asset_fk] == self.asset_rid)
            .attributes(type_path.Asset_Type)
            .fetch()
        )
        self._asset_types = [t["Asset_Type"] for t in types]

    @property
    def execution_rid(self) -> RID | None:
        """Get the RID of the execution that created this asset.

        Returns:
            RID of the creating execution, or None if not tracked.
        """
        if self._execution_rid is None:
            # Try to find the execution that created this asset (Output role)
            executions = self.list_executions(asset_role="Output")
            if executions:
                self._execution_rid = executions[0].execution_rid
        return self._execution_rid

    def list_executions(self, asset_role: str | None = None) -> list["ExecutionRecord"]:
        """List all executions associated with this asset.

        Returns all executions that created or used this asset, along with
        the role (Input/Output) in each execution.

        Args:
            asset_role: Optional filter for asset role ('Input' or 'Output').
                If None, returns all associations.

        Returns:
            List of ExecutionRecord objects for the executions associated
            with this asset.

        Example:
            >>> # Find the execution that created this asset
            >>> creators = asset.list_executions(asset_role="Output")
            >>> if creators:
            ...     print(f"Created by execution {creators[0].execution_rid}")

            >>> # Find all executions that used this asset as input
            >>> users = asset.list_executions(asset_role="Input")
        """
        return self._ml_instance.list_asset_executions(self.asset_rid, asset_role=asset_role)

    def find_features(self) -> list["Feature"]:
        """Find all features defined on this asset's table.

        Returns:
            List of Feature objects defined on this asset's table.

        Example:
            >>> features = asset.find_features()
            >>> for f in features:
            ...     print(f"Feature: {f.feature_name}")
        """
        return self._ml_instance.find_features(self.asset_table)

    def list_feature_values(self, feature_name: str) -> list["FeatureRecord"]:
        """Get feature values for this specific asset.

        Args:
            feature_name: Name of the feature to query.

        Returns:
            List of FeatureRecord instances for this asset. Each record has:
                - Execution: RID of the execution that created this feature value
                - Feature_Name: Name of the feature
                - All feature-specific columns as typed attributes
                - model_dump() method to convert back to a dictionary

        Example:
            >>> values = asset.list_feature_values("quality_score")
            >>> for v in values:
            ...     print(f"Score: {v.Score}, Execution: {v.Execution}")
            >>> # Or convert to dict:
            >>> dicts = [v.model_dump() for v in values]
        """
        return list(self._ml_instance.list_feature_values(self.asset_table, feature_name))

    def add_asset_type(self, type_name: str) -> None:
        """Add an asset type to this asset.

        Args:
            type_name: Name of the asset type vocabulary term to add.

        Example:
            >>> asset.add_asset_type("Training_Data")
        """
        asset_table_obj = self._ml_instance.model.name_to_table(self.asset_table)
        type_assoc_table, asset_fk, _ = self._ml_instance.model.find_association(
            asset_table_obj, "Asset_Type"
        )

        pb = self._ml_instance.pathBuilder()
        type_path = pb.schemas[type_assoc_table.schema.name].tables[type_assoc_table.name]

        # Insert the association
        type_path.insert([{asset_fk: self.asset_rid, "Asset_Type": type_name}])

        # Update local cache
        if type_name not in self._asset_types:
            self._asset_types.append(type_name)

    def remove_asset_type(self, type_name: str) -> None:
        """Remove an asset type from this asset.

        Args:
            type_name: Name of the asset type vocabulary term to remove.

        Example:
            >>> asset.remove_asset_type("Temporary")
        """
        asset_table_obj = self._ml_instance.model.name_to_table(self.asset_table)
        type_assoc_table, asset_fk, _ = self._ml_instance.model.find_association(
            asset_table_obj, "Asset_Type"
        )

        pb = self._ml_instance.pathBuilder()
        type_path = pb.schemas[type_assoc_table.schema.name].tables[type_assoc_table.name]

        # Delete the association
        type_path.filter(
            (type_path.columns[asset_fk] == self.asset_rid) &
            (type_path.Asset_Type == type_name)
        ).delete()

        # Update local cache
        if type_name in self._asset_types:
            self._asset_types.remove(type_name)

    def download(self, dest_dir: Path, update_catalog: bool = False) -> Path:
        """Download the asset file to a local directory.

        Args:
            dest_dir: Directory to download the asset to.
            update_catalog: If True and called within an execution context,
                track this asset as an input to the current execution.

        Returns:
            Path to the downloaded file.

        Example:
            >>> local_path = asset.download(Path("/tmp/assets"))
            >>> print(f"Downloaded to: {local_path}")
        """

        # Use hatrac to download the file
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / self.filename
        self._ml_instance.hatrac.get_obj(self.url, destfilename=str(dest_path))

        return dest_path

    def get_metadata(self) -> dict[str, Any]:
        """Get all metadata for this asset from the catalog.

        Returns:
            Dictionary of all columns/values for this asset record.

        Example:
            >>> metadata = asset.get_metadata()
            >>> print(f"Created: {metadata.get('RCT')}")
        """
        pb = self._ml_instance.pathBuilder()
        asset_path = pb.schemas[self._ml_instance.model.name_to_table(self.asset_table).schema.name].tables[self.asset_table]

        records = list(asset_path.filter(asset_path.RID == self.asset_rid).entities().fetch())
        return records[0] if records else {}

    def get_chaise_url(self) -> str:
        """Get the Chaise URL for viewing this asset in the web interface.

        Returns:
            URL to view this asset in Chaise.

        Example:
            >>> url = asset.get_chaise_url()
            >>> print(f"View at: {url}")
        """
        table_obj = self._ml_instance.model.name_to_table(self.asset_table)
        schema_name = table_obj.schema.name
        catalog_id = self._ml_instance.catalog_id
        hostname = self._ml_instance.host_name

        return (
            f"https://{hostname}/chaise/record/#{catalog_id}/"
            f"{schema_name}:{self.asset_table}/RID={self.asset_rid}"
        )

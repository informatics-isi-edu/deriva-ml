"""Asset management mixin for DerivaML.

This module provides the AssetMixin class which handles
asset table operations including creating, listing, and looking up assets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Table = _ermrest_model.Table

from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import ColumnDefinition, MLVocab, RID, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.schema.annotations import asset_annotation

if TYPE_CHECKING:
    from deriva_ml.asset.asset import Asset
    from deriva_ml.model.catalog import DerivaModel


class AssetMixin:
    """Mixin providing asset management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - domain_schema: str - name of the domain schema
        - pathBuilder(): method returning catalog path builder
        - add_term(): method for adding vocabulary terms (from VocabularyMixin)
        - apply_catalog_annotations(): method to update navbar (from DerivaML base class)

    Methods:
        create_asset: Create a new asset table
        list_assets: List contents of an asset table
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    domain_schema: str
    pathBuilder: Callable[[], Any]
    add_term: Callable[..., VocabularyTerm]
    apply_catalog_annotations: Callable[[], None]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_asset(
        self,
        asset_name: str,
        column_defs: Iterable[ColumnDefinition] | None = None,
        fkey_defs: Iterable[ColumnDefinition] | None = None,
        referenced_tables: Iterable[Table] | None = None,
        comment: str = "",
        schema: str | None = None,
        update_navbar: bool = True,
    ) -> Table:
        """Creates an asset table.

        Args:
            asset_name: Name of the asset table.
            column_defs: Iterable of ColumnDefinition objects to provide additional metadata for asset.
            fkey_defs: Iterable of ForeignKeyDefinition objects to provide additional metadata for asset.
            referenced_tables: Iterable of Table objects to which asset should provide foreign-key references to.
            comment: Description of the asset table. (Default value = '')
            schema: Schema in which to create the asset table.  Defaults to domain_schema.
            update_navbar: If True (default), automatically updates the navigation bar to include
                the new asset table. Set to False during batch asset creation to avoid redundant
                updates, then call apply_catalog_annotations() once at the end.

        Returns:
            Table object for the asset table.
        """
        # Initialize empty collections if None provided
        column_defs = column_defs or []
        fkey_defs = fkey_defs or []
        referenced_tables = referenced_tables or []
        schema = schema or self.domain_schema

        # Add an asset type to vocabulary
        self.add_term(MLVocab.asset_type, asset_name, description=f"A {asset_name} asset")

        # Create the main asset table
        asset_table = self.model.schemas[schema].create_table(
            Table.define_asset(
                schema,
                asset_name,
                column_defs=[c.model_dump() for c in column_defs],
                fkey_defs=[fk.model_dump() for fk in fkey_defs],
                comment=comment,
            )
        )

        # Create an association table between asset and asset type
        self.model.schemas[self.domain_schema].create_table(
            Table.define_association(
                [
                    (asset_table.name, asset_table),
                    ("Asset_Type", self.model.name_to_table("Asset_Type")),
                ]
            )
        )

        # Create references to other tables if specified
        for t in referenced_tables:
            asset_table.create_reference(self.model.name_to_table(t))

        # Create an association table for tracking execution
        atable = self.model.schemas[self.domain_schema].create_table(
            Table.define_association(
                [
                    (asset_name, asset_table),
                    (
                        "Execution",
                        self.model.schemas[self.ml_schema].tables["Execution"],
                    ),
                ]
            )
        )
        atable.create_reference(self.model.name_to_table("Asset_Role"))

        # Add asset annotations
        asset_annotation(asset_table)

        # Update navbar to include the new asset table
        if update_navbar:
            self.apply_catalog_annotations()

        return asset_table

    def list_assets(self, asset_table: Table | str) -> list[dict[str, Any]]:
        """Lists contents of an asset table.

        Returns a list of assets with their types for the specified asset table.

        Args:
            asset_table: Table or name of the asset table to list assets for.

        Returns:
            list[dict[str, Any]]: List of asset records, each containing:
                - RID: Resource identifier
                - Type: Asset type
                - Metadata: Asset metadata

        Raises:
            DerivaMLException: If the table is not an asset table or doesn't exist.

        Example:
            >>> assets = ml.list_assets("tissue_types")
            >>> for asset in assets:
            ...     print(f"{asset['RID']}: {asset['Type']}")
        """
        # Validate and get asset table reference
        asset_table = self.model.name_to_table(asset_table)
        if not self.model.is_asset(asset_table):
            raise DerivaMLException(f"Table {asset_table.name} is not an asset")

        # Get path builders for asset and type tables
        pb = self.pathBuilder()
        asset_path = pb.schemas[asset_table.schema.name].tables[asset_table.name]
        (
            asset_type_table,
            _,
            _,
        ) = self.model.find_association(asset_table, MLVocab.asset_type)
        type_path = pb.schemas[asset_type_table.schema.name].tables[asset_type_table.name]

        # Build a list of assets with their types
        assets = []
        for asset in asset_path.entities().fetch():
            # Get associated asset types for each asset
            asset_types = (
                type_path.filter(type_path.columns[asset_table.name] == asset["RID"])
                .attributes(type_path.Asset_Type)
                .fetch()
            )
            # Combine asset data with its types
            assets.append(
                asset | {MLVocab.asset_type.value: [asset_type[MLVocab.asset_type.value] for asset_type in asset_types]}
            )
        return assets

    def list_asset_executions(
        self, asset_rid: str, asset_role: str | None = None
    ) -> list[dict[str, Any]]:
        """List all executions associated with an asset.

        Given an asset RID, returns a list of executions that created or used
        the asset, along with the role (Input/Output) in each execution.

        Args:
            asset_rid: The RID of the asset to look up.
            asset_role: Optional filter for asset role ('Input' or 'Output').
                If None, returns all associations.

        Returns:
            list[dict[str, Any]]: List of records containing:
                - Execution: RID of the associated execution
                - Asset_Role: Role of the asset in the execution ('Input' or 'Output')

        Raises:
            DerivaMLException: If the asset RID is not found or not an asset.

        Example:
            >>> # Find all executions that created this asset
            >>> executions = ml.list_asset_executions("1-abc123", asset_role="Output")
            >>> for record in executions:
            ...     print(f"Created by execution {record['Execution']}")

            >>> # Find all executions that used this asset as input
            >>> executions = ml.list_asset_executions("1-abc123", asset_role="Input")
        """
        # Resolve the RID to find which asset table it belongs to
        rid_info = self.retrieve_rid(asset_rid)  # type: ignore[attr-defined]
        asset_table = self.model.name_to_table(rid_info["table"])

        if not self.model.is_asset(asset_table):
            raise DerivaMLException(f"RID {asset_rid} is not an asset (table: {rid_info['table']})")

        # Find the association table between this asset table and Execution
        asset_exe_table, asset_fk, execution_fk = self.model.find_association(asset_table, "Execution")

        # Build the query
        pb = self.pathBuilder()
        asset_exe_path = pb.schemas[asset_exe_table.schema.name].tables[asset_exe_table.name]

        # Filter by asset RID
        query = asset_exe_path.filter(asset_exe_path.columns[asset_fk] == asset_rid)

        # Optionally filter by asset role
        if asset_role:
            query = query.filter(asset_exe_path.Asset_Role == asset_role)

        return list(query.entities().fetch())

    def lookup_asset(self, asset_rid: RID) -> "Asset":
        """Look up an asset by its RID.

        Returns an Asset object for the specified RID. The asset can be from
        any asset table in the catalog.

        Args:
            asset_rid: The RID of the asset to look up.

        Returns:
            Asset object for the specified RID.

        Raises:
            DerivaMLException: If the RID is not found or is not an asset.

        Example:
            >>> asset = ml.lookup_asset("3JSE")
            >>> print(f"File: {asset.filename}, Table: {asset.asset_table}")
        """
        from deriva_ml.asset.asset import Asset

        # Resolve the RID to find which table it belongs to
        rid_info = self.retrieve_rid(asset_rid)  # type: ignore[attr-defined]
        asset_table = self.model.name_to_table(rid_info["table"])

        if not self.model.is_asset(asset_table):
            raise DerivaMLException(f"RID {asset_rid} is not an asset (table: {rid_info['table']})")

        # Query the asset table for this record
        pb = self.pathBuilder()
        asset_path = pb.schemas[asset_table.schema.name].tables[asset_table.name]

        records = list(asset_path.filter(asset_path.RID == asset_rid).entities().fetch())
        if not records:
            raise DerivaMLException(f"Asset {asset_rid} not found in table {asset_table.name}")

        record = records[0]

        # Get asset types
        asset_types = []
        try:
            type_assoc_table, asset_fk, _ = self.model.find_association(asset_table, "Asset_Type")
            type_path = pb.schemas[type_assoc_table.schema.name].tables[type_assoc_table.name]
            types = list(
                type_path.filter(type_path.columns[asset_fk] == asset_rid)
                .attributes(type_path.Asset_Type)
                .fetch()
            )
            asset_types = [t["Asset_Type"] for t in types]
        except Exception:
            pass  # No type association for this asset table

        return Asset(
            catalog=self,  # type: ignore[arg-type]
            asset_rid=asset_rid,
            asset_table=asset_table.name,
            filename=record.get("Filename", ""),
            url=record.get("URL", ""),
            length=record.get("Length", 0),
            md5=record.get("MD5", ""),
            description=record.get("Description", ""),
            asset_types=asset_types,
        )

    def find_assets(
        self,
        asset_table: Table | str | None = None,
        asset_type: str | None = None,
    ) -> Iterable["Asset"]:
        """Find assets in the catalog.

        Returns an iterable of Asset objects matching the specified criteria.
        If no criteria are specified, returns all assets from all asset tables.

        Args:
            asset_table: Optional table or table name to search. If None, searches
                all asset tables.
            asset_type: Optional asset type to filter by. Only returns assets
                with this type.

        Returns:
            Iterable of Asset objects matching the criteria.

        Example:
            >>> # Find all assets in the Model table
            >>> models = list(ml.find_assets(asset_table="Model"))

            >>> # Find all assets with type "Training_Data"
            >>> training = list(ml.find_assets(asset_type="Training_Data"))

            >>> # Find all assets across all tables
            >>> all_assets = list(ml.find_assets())
        """
        from deriva_ml.asset.asset import Asset

        # Determine which tables to search
        if asset_table is not None:
            tables = [self.model.name_to_table(asset_table)]
        else:
            # Find all asset tables
            tables = [
                t for t in self.model.schemas[self.domain_schema].tables.values()
                if self.model.is_asset(t)
            ]
            # Also include ML schema asset tables (like Execution_Asset)
            tables.extend([
                t for t in self.model.schemas[self.ml_schema].tables.values()
                if self.model.is_asset(t)
            ])

        for table in tables:
            # Get all assets from this table
            assets_data = self.list_assets(table)

            for asset_data in assets_data:
                # Filter by asset type if specified
                if asset_type is not None:
                    if asset_type not in asset_data.get(MLVocab.asset_type.value, []):
                        continue

                yield Asset(
                    catalog=self,  # type: ignore[arg-type]
                    asset_rid=asset_data["RID"],
                    asset_table=table.name,
                    filename=asset_data.get("Filename", ""),
                    url=asset_data.get("URL", ""),
                    length=asset_data.get("Length", 0),
                    md5=asset_data.get("MD5", ""),
                    description=asset_data.get("Description", ""),
                    asset_types=asset_data.get(MLVocab.asset_type.value, []),
                )

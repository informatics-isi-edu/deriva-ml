"""Asset management mixin for DerivaML.

This module provides the AssetMixin class which handles
asset table operations including creating and listing assets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

from deriva.core.ermrest_model import Table
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import ColumnDefinition, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.schema.annotations import asset_annotation

if TYPE_CHECKING:
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

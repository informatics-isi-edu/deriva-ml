"""Feature management mixin for DerivaML.

This module provides the FeatureMixin class which handles
feature operations including creating, looking up, deleting,
and listing feature values.
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Callable

import deriva.core.datapath as datapath
from deriva.core.ermrest_model import Key, Table
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import ColumnDefinition, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import Feature, FeatureRecord

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


class FeatureMixin:
    """Mixin providing feature management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - domain_schema: str - name of the domain schema
        - pathBuilder(): method returning catalog path builder
        - add_term(): method for adding vocabulary terms (from VocabularyMixin)

    Methods:
        create_feature: Create a new feature definition
        feature_record_class: Get pydantic model class for feature records
        delete_feature: Remove a feature definition
        lookup_feature: Retrieve a Feature object
        list_feature_values: Get all values for a feature
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    domain_schema: str
    pathBuilder: Callable[[], Any]
    add_term: Callable[..., VocabularyTerm]

    def create_feature(
        self,
        target_table: Table | str,
        feature_name: str,
        terms: list[Table | str] | None = None,
        assets: list[Table | str] | None = None,
        metadata: list[ColumnDefinition | Table | Key | str] | None = None,
        optional: list[str] | None = None,
        comment: str = "",
    ) -> type[FeatureRecord]:
        """Creates a new feature definition.

        A feature represents a measurable property or characteristic that can be associated with records in the target
        table. Features can include vocabulary terms, asset references, and additional metadata.

        Args:
            target_table: Table to associate the feature with (name or Table object).
            feature_name: Unique name for the feature within the target table.
            terms: Optional vocabulary tables/names whose terms can be used as feature values.
            assets: Optional asset tables/names that can be referenced by this feature.
            metadata: Optional columns, tables, or keys to include in a feature definition.
            optional: Column names that are not required when creating feature instances.
            comment: Description of the feature's purpose and usage.

        Returns:
            type[FeatureRecord]: Feature class for creating validated instances.

        Raises:
            DerivaMLException: If a feature definition is invalid or conflicts with existing features.

        Examples:
            Create a feature with confidence score:
                >>> feature_class = ml.create_feature(
                ...     target_table="samples",
                ...     feature_name="expression_level",
                ...     terms=["expression_values"],
                ...     metadata=[ColumnDefinition(name="confidence", type=BuiltinTypes.float4)],
                ...     comment="Gene expression measurement"
                ... )
        """
        # Initialize empty collections if None provided
        terms = terms or []
        assets = assets or []
        metadata = metadata or []
        optional = optional or []

        def normalize_metadata(m: Key | Table | ColumnDefinition | str) -> Key | Table | dict:
            """Helper function to normalize metadata references."""
            if isinstance(m, str):
                return self.model.name_to_table(m)
            elif isinstance(m, ColumnDefinition):
                return m.model_dump()
            else:
                return m

        # Validate asset and term tables
        if not all(map(self.model.is_asset, assets)):
            raise DerivaMLException("Invalid create_feature asset table.")
        if not all(map(self.model.is_vocabulary, terms)):
            raise DerivaMLException("Invalid create_feature asset table.")

        # Get references to required tables
        target_table = self.model.name_to_table(target_table)
        execution = self.model.schemas[self.ml_schema].tables["Execution"]
        feature_name_table = self.model.schemas[self.ml_schema].tables["Feature_Name"]

        # Add feature name to vocabulary
        feature_name_term = self.add_term("Feature_Name", feature_name, description=comment)
        atable_name = f"Execution_{target_table.name}_{feature_name_term.name}"
        # Create an association table implementing the feature
        atable = self.model.schemas[self.domain_schema].create_table(
            target_table.define_association(
                table_name=atable_name,
                associates=[execution, target_table, feature_name_table],
                metadata=[normalize_metadata(m) for m in chain(assets, terms, metadata)],
                comment=comment,
            )
        )
        # Configure optional columns and default feature name
        for c in optional:
            atable.columns[c].alter(nullok=True)
        atable.columns["Feature_Name"].alter(default=feature_name_term.name)

        # Return feature record class for creating instances
        return self.feature_record_class(target_table, feature_name)

    def feature_record_class(self, table: str | Table, feature_name: str) -> type[FeatureRecord]:
        """Returns a pydantic model class for feature records.

        Creates a typed interface for creating new instances of the specified feature. The returned class includes
        validation and type checking based on the feature's definition.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to create a record class for.

        Returns:
            type[FeatureRecord]: A pydantic model class for creating validated feature records.

        Raises:
            DerivaMLException: If the feature doesn't exist or the table is invalid.

        Example:
            >>> ExpressionFeature = ml.feature_record_class("samples", "expression_level")
            >>> feature = ExpressionFeature(value="high", confidence=0.95)
        """
        # Look up a feature and return its record class
        return self.lookup_feature(table, feature_name).feature_record_class()

    def delete_feature(self, table: Table | str, feature_name: str) -> bool:
        """Removes a feature definition and its data.

        Deletes the feature and its implementation table from the catalog. This operation cannot be undone and
        will remove all feature values associated with this feature.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to delete.

        Returns:
            bool: True if the feature was successfully deleted, False if it didn't exist.

        Raises:
            DerivaMLException: If deletion fails due to constraints or permissions.

        Example:
            >>> success = ml.delete_feature("samples", "obsolete_feature")
            >>> print("Deleted" if success else "Not found")
        """
        # Get table reference and find feature
        table = self.model.name_to_table(table)
        try:
            # Find and delete the feature's implementation table
            feature = next(f for f in self.model.find_features(table) if f.feature_name == feature_name)
            feature.feature_table.drop()
            return True
        except StopIteration:
            return False

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """Retrieves a Feature object.

        Looks up and returns a Feature object that provides an interface to work with an existing feature
        definition in the catalog.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to look up.

        Returns:
            Feature: An object representing the feature and its implementation.

        Raises:
            DerivaMLException: If the feature doesn't exist in the specified table.

        Example:
            >>> feature = ml.lookup_feature("samples", "expression_level")
            >>> print(feature.feature_name)
            'expression_level'
        """
        return self.model.lookup_feature(table, feature_name)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_feature_values(self, table: Table | str, feature_name: str) -> datapath._ResultSet:
        """Retrieves all values for a feature.

        Returns all instances of the specified feature that have been created, including their associated
        metadata and references.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to retrieve values for.

        Returns:
            datapath._ResultSet: A result set containing all feature values and their metadata.

        Raises:
            DerivaMLException: If the feature doesn't exist or cannot be accessed.

        Example:
            >>> values = ml.list_feature_values("samples", "expression_level")
            >>> for value in values:
            ...     print(f"Sample {value['RID']}: {value['value']}")
        """
        # Get table and feature references
        table = self.model.name_to_table(table)
        feature = self.lookup_feature(table, feature_name)

        # Build and execute query for feature values
        pb = self.pathBuilder()
        return pb.schemas[feature.feature_table.schema.name].tables[feature.feature_table.name].entities().fetch()

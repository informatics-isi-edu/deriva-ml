"""Feature implementation for deriva-ml.

This module provides classes for defining and managing features in deriva-ml. Features represent measurable
properties or characteristics that can be associated with records in a table. The module includes:

- Feature: Main class for defining and managing features
- FeatureRecord: Base class for feature records using pydantic models

Typical usage example:
    >>> feature = Feature(association_result, model)
    >>> FeatureClass = feature.feature_record_class()
    >>> record = FeatureClass(value="high", confidence=0.95)
"""

from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, ClassVar, Optional, Type

from deriva.core.ermrest_model import Column, FindAssociationResult
from pydantic import BaseModel, create_model

if TYPE_CHECKING:
    from model.catalog import DerivaModel


class FeatureRecord(BaseModel):
    """Base class for dynamically generated feature record models.

    This class serves as the base for pydantic models that represent feature records. Each feature record
    contains the values and metadata associated with a feature instance.

    Attributes:
        Execution (Optional[str]): RID of the execution that created this feature record.
        Feature_Name (str): Name of the feature this record belongs to.
        feature (ClassVar[Optional[Feature]]): Reference to the Feature object that created this record.

    Example:
        >>> class GeneFeature(FeatureRecord):
        ...     value: str
        ...     confidence: float
        >>> record = GeneFeature(
        ...     Feature_Name="expression",
        ...     value="high",
        ...     confidence=0.95
        ... )
    """

    # model_dump of this feature should be compatible with feature table columns.
    Execution: Optional[str] = None
    Feature_Name: str
    feature: ClassVar[Optional["Feature"]] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def feature_columns(cls) -> set[Column]:
        """Returns all columns specific to this feature.

        Returns:
            set[Column]: Set of feature-specific columns, excluding system and relationship columns.
        """
        return cls.feature.feature_columns

    @classmethod
    def asset_columns(cls) -> set[Column]:
        """Returns columns that reference asset tables.

        Returns:
            set[Column]: Set of columns that contain references to asset tables.
        """
        return cls.feature.asset_columns

    @classmethod
    def term_columns(cls) -> set[Column]:
        """Returns columns that reference vocabulary terms.

        Returns:
            set[Column]: Set of columns that contain references to controlled vocabulary terms.
        """
        return cls.feature.term_columns

    @classmethod
    def value_columns(cls) -> set[Column]:
        """Returns columns that contain direct values.

        Returns:
            set[Column]: Set of columns containing direct values (not references to assets or terms).
        """
        return cls.feature.value_columns


class Feature:
    """Manages feature definitions and their relationships in the catalog.

    A Feature represents a measurable property or characteristic that can be associated with records in a table.
    Features can include asset references, controlled vocabulary terms, and custom metadata fields.

    Attributes:
        feature_table: Table containing the feature implementation.
        target_table: Table that the feature is associated with.
        feature_name: Name of the feature (from Feature_Name column default).
        feature_columns: Set of columns specific to this feature.
        asset_columns: Set of columns referencing asset tables.
        term_columns: Set of columns referencing vocabulary tables.
        value_columns: Set of columns containing direct values.

    Example:
        >>> feature = Feature(association_result, model)
        >>> print(f"Feature {feature.feature_name} on {feature.target_table.name}")
        >>> print("Asset columns:", [c.name for c in feature.asset_columns])
    """

    def __init__(self, atable: FindAssociationResult, model: "DerivaModel") -> None:
        self.feature_table = atable.table
        self.target_table = atable.self_fkey.pk_table
        self.feature_name = atable.table.columns["Feature_Name"].default
        self._model = model

        skip_columns = {
            "RID",
            "RMB",
            "RCB",
            "RCT",
            "RMT",
            "Feature_Name",
            self.target_table.name,
            "Execution",
        }
        self.feature_columns = {c for c in self.feature_table.columns if c.name not in skip_columns}

        assoc_fkeys = {atable.self_fkey} | atable.other_fkeys

        # Determine the role of each column in the feature outside the FK columns.
        self.asset_columns = {
            fk.foreign_key_columns[0]
            for fk in self.feature_table.foreign_keys
            if fk not in assoc_fkeys and self._model.is_asset(fk.pk_table)
        }

        self.term_columns = {
            fk.foreign_key_columns[0]
            for fk in self.feature_table.foreign_keys
            if fk not in assoc_fkeys and self._model.is_vocabulary(fk.pk_table)
        }

        self.value_columns = self.feature_columns - (self.asset_columns | self.term_columns)

    def feature_record_class(self) -> type[FeatureRecord]:
        """Create a pydantic model for entries into the specified feature table

        Returns:
            A Feature class that can be used to create instances of the feature.
        """

        def map_type(c: Column) -> UnionType | Type[str] | Type[int] | Type[float]:
            """Maps a Deriva column type to a Python/pydantic type.

            Converts ERMrest column types to appropriate Python types for use in pydantic models.
            Special handling is provided for asset columns which can accept either strings or Path objects.

            Args:
                c: ERMrest column to map to a Python type.

            Returns:
                UnionType | Type[str] | Type[int] | Type[float]: Appropriate Python type for the column:
                    - str | Path for asset columns
                    - str for text columns
                    - int for integer columns
                    - float for floating point columns
                    - str for all other types

            Example:
                >>> col = Column(name="score", type="float4")
                >>> typ = map_type(col)  # Returns float
            """
            if c.name in {c.name for c in self.asset_columns}:
                return str | Path

            match c.type.typename:
                case "text":
                    return str
                case "int2" | "int4" | "int8":
                    return int
                case "float4" | "float8":
                    return float
                case _:
                    return str

        featureclass_name = f"{self.target_table.name}Feature{self.feature_name}"

        # Create feature class. To do this, we must determine the python type for each column and also if the
        # column is optional or not based on its nullability.
        feature_columns = {
            c.name: (
                Optional[map_type(c)] if c.nullok else map_type(c),
                c.default or None,
            )
            for c in self.feature_columns
        } | {
            "Feature_Name": (
                str,
                self.feature_name,
            ),  # Set default value for Feature_Name
            self.target_table.name: (str, ...),
        }
        docstring = (
            f"Class to capture fields in a feature {self.feature_name} on table {self.target_table}. "
            "Feature columns include:\n"
        )
        docstring += "\n".join([f"    {c.name}" for c in self.feature_columns])

        model = create_model(
            featureclass_name,
            __base__=FeatureRecord,
            __doc__=docstring,
            **feature_columns,
        )
        model.feature = self  # Set value of class variable within the feature class definition.

        return model

    def __repr__(self) -> str:
        return (
            f"Feature(target_table={self.target_table.name}, feature_name={self.feature_name}, "
            f"feature_table={self.feature_table.name})"
        )

"""
This module provides the implementation of the Feature capability in deriva-ml
"""

from deriva.core.ermrest_model import FindAssociationResult, Column
from pathlib import Path
from pydantic import BaseModel, create_model
from typing import Optional, Type, ClassVar, TYPE_CHECKING
from types import UnionType

if TYPE_CHECKING:
    from .deriva_model import DerivaModel


class FeatureRecord(BaseModel):
    """Base class for feature records.  Feature records are pydantic models which are dynamically generated and
    describe all the columns of a feature.

    Attributes:
        Execution (str):
        Feature_Name (str):
        feature:
    Returns:

    """

    # model_dump of this feature should be compatible with feature table columns.
    Execution: str
    Feature_Name: str
    feature: ClassVar[Optional["Feature"]] = None

    class Config:

        arbitrary_types_allowed = True

    @classmethod
    def feature_columns(cls) -> set[Column]:
        """

        Returns:
          A set of feature column names.

        """
        return cls.feature.feature_columns

    @classmethod
    def asset_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          A set of asset column names.

        """
        return cls.feature.asset_columns

    @classmethod
    def term_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          :return: set of term column names.

        """
        return cls.feature.term_columns

    @classmethod
    def value_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          A set of value column names.

        """
        return cls.feature.value_columns


class Feature:
    """Wrapper for results of Table.find_associations()"""

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
        self.feature_columns = {
            c for c in self.feature_table.columns if c.name not in skip_columns
        }

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

        self.value_columns = self.feature_columns - (
            self.asset_columns | self.term_columns
        )

    def feature_record_class(self) -> type[FeatureRecord]:
        """Create a pydantic model for entries into the specified feature table

        Returns:
            A Feature class that can be used to create instances of the feature.
        """

        def map_type(c: Column) -> UnionType | Type[str] | Type[int] | Type[float]:
            """Map a deriva type into a pydantic model type.

            Args:
                c: column to be mapped
                c: Column:

            Returns:
                A pydantic model type
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
        docstring = f"Class to capture fields in a feature {self.feature_name} on table {self.target_table}. Feature columns include:\n"
        docstring += "\n".join([f"    {c.name}" for c in self.feature_columns])

        model = create_model(
            featureclass_name,
            __base__=FeatureRecord,
            __doc__=docstring,
            **feature_columns,
        )
        model.feature = (
            self  # Set value of class variable within the feature class definition.
        )

        return model

    def __repr__(self) -> str:
        return (
            f"Feature(target_table={self.target_table.name}, feature_name={self.feature_name}, "
            f"feature_table={self.feature_table.name})"
        )

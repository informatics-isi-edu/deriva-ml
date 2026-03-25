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

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, ClassVar, Optional, Type

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Column = _ermrest_model.Column
FindAssociationResult = _ermrest_model.FindAssociationResult

from pydantic import BaseModel, create_model

if TYPE_CHECKING:
    from model.catalog import DerivaModel


class FeatureRecord(BaseModel):
    """Base class for dynamically generated feature record models.

    This class serves as the base for pydantic models that represent feature
    records. Each feature record contains the values and metadata associated
    with a feature instance. Subclasses are created dynamically by
    ``Feature.feature_record_class()`` with fields corresponding to the
    feature's vocabulary terms, asset references, and metadata columns.

    Feature records are returned by ``list_feature_values()`` and
    ``fetch_table_features()``. They can also be constructed manually and
    passed to ``Execution.add_features()`` to insert new values into the
    catalog.

    **Handling multiple values per target:**

    When the same target object (e.g., an Image) has multiple feature values
    — for example, labels from different annotators or model runs — use
    a ``selector`` function to choose one. Pass it to ``fetch_table_features``
    or ``list_feature_values``. A selector receives a list of FeatureRecord
    instances for the same target and returns the selected one::

        # Built-in: pick the most recently created record
        features = ml.fetch_table_features(
            "Image", selector=FeatureRecord.select_newest
        )

        # Custom: pick the record with highest confidence
        def select_best(records):
            return max(records, key=lambda r: getattr(r, "Confidence", 0))

        features = ml.fetch_table_features("Image", selector=select_best)

    Attributes:
        Execution (Optional[str]): RID of the execution that created this
            feature record. Links to the ``Execution`` table for provenance
            tracking — use this to trace which workflow run produced this value.
        Feature_Name (str): Name of the feature this record belongs to.
        RCT (Optional[str]): Row Creation Time — an ISO 8601 timestamp string
            (e.g., ``"2024-06-15T10:30:00.000000+00:00"``). Populated
            automatically when reading from the catalog or a dataset bag.
            Used by ``select_newest`` to determine recency.
        feature (ClassVar[Optional[Feature]]): Reference to the Feature
            definition object. Set automatically by ``feature_record_class()``
            when the dynamic subclass is created. ``None`` on the base class.
            Provides access to the feature's column metadata, target table,
            and vocabulary/asset column sets.
    """

    # model_dump of this feature should be compatible with feature table columns.
    Execution: Optional[str] = None
    Feature_Name: str
    RCT: Optional[str] = None
    feature: ClassVar[Optional["Feature"]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @staticmethod
    def select_newest(records: list["FeatureRecord"]) -> "FeatureRecord":
        """Select the feature record with the most recent creation time.

        Uses the RCT (Row Creation Time) field to determine recency. RCT is
        an ISO 8601 timestamp string, so lexicographic comparison correctly
        identifies the most recent record. Records with ``None`` RCT are
        treated as older than any timestamped record.

        This method is designed to be passed directly as the ``selector``
        argument to ``fetch_table_features`` or ``list_feature_values``::

            features = ml.fetch_table_features(
                "Image", selector=FeatureRecord.select_newest
            )

        Args:
            records: List of FeatureRecord instances for the same target
                object. Must be non-empty.

        Returns:
            The FeatureRecord with the latest RCT value.
        """
        return max(records, key=lambda r: r.RCT or "")

    @staticmethod
    def select_by_execution(execution_rid: str):
        """Return a selector that picks the newest record from a specific execution.

        Creates a selector function that filters records to those produced by
        the given execution, then returns the newest match by RCT. This is
        useful when multiple executions have produced values for the same
        feature and you want results from a specific run.

        Unlike ``select_by_workflow`` (which requires catalog access and lives
        on the DerivaML class), this selector works purely on the
        ``Execution`` field of each record and can be passed directly as the
        ``selector`` argument to ``fetch_table_features`` or
        ``list_feature_values``::

            features = ml.fetch_table_features(
                "Image",
                feature_name="FooBar",
                selector=FeatureRecord.select_by_execution("3WY2"),
            )

        Args:
            execution_rid: RID of the execution to filter by.

        Returns:
            A selector function ``(list[FeatureRecord]) -> FeatureRecord``
            suitable for use with ``fetch_table_features`` or
            ``list_feature_values``.

        Raises:
            DerivaMLException: If no records in the group match the
                given execution RID.

        Examples:
            Select values from a specific execution::

                >>> features = ml.fetch_table_features(
                ...     "Image",
                ...     feature_name="Classification",
                ...     selector=FeatureRecord.select_by_execution("3WY2"),
                ... )

            Use with list_feature_values::

                >>> values = ml.list_feature_values(
                ...     "Image", "Classification",
                ...     selector=FeatureRecord.select_by_execution("3WY2"),
                ... )
        """

        def _selector(records: list["FeatureRecord"]) -> "FeatureRecord":
            filtered = [r for r in records if r.Execution == execution_rid]
            if not filtered:
                from deriva_ml.core.exceptions import DerivaMLException

                raise DerivaMLException(
                    f"No feature records match execution '{execution_rid}'."
                )
            return FeatureRecord.select_newest(filtered)

        return _selector

    @staticmethod
    def select_first(records: list["FeatureRecord"]) -> "FeatureRecord":
        """Select the feature record with the earliest creation time.

        Uses the RCT (Row Creation Time) field. Records with ``None`` RCT
        are treated as older than any timestamped record (since empty string
        sorts before any ISO 8601 timestamp).

        Useful when you want to preserve the original annotation and ignore
        later revisions.

        This method is designed to be passed directly as the ``selector``
        argument to ``fetch_table_features`` or ``list_feature_values``::

            features = ml.fetch_table_features(
                "Image", selector=FeatureRecord.select_first
            )

        Args:
            records: List of FeatureRecord instances for the same target
                object. Must be non-empty.

        Returns:
            The FeatureRecord with the earliest RCT value.
        """
        return min(records, key=lambda r: r.RCT or "")

    @staticmethod
    def select_latest(records: list["FeatureRecord"]) -> "FeatureRecord":
        """Select the most recently created feature record.

        Alias for ``select_newest``. Included for API symmetry with
        ``select_first``.

        Args:
            records: List of FeatureRecord instances for the same target
                object. Must be non-empty.

        Returns:
            The FeatureRecord with the latest RCT value.
        """
        return FeatureRecord.select_newest(records)

    @classmethod
    def select_majority_vote(cls, column: str | None = None):
        """Return a selector that picks the most common value for a column.

        Creates a selector function that counts the values of the specified
        column across all records, picks the most frequent one, and breaks
        ties by most recent RCT.

        For single-term features, the column can be auto-detected from the
        feature's metadata. For multi-term features, the column must be
        specified explicitly.

        This is useful for consensus labeling, where multiple annotators
        have labeled the same record and you want the majority opinion::

            selector = RecordClass.select_majority_vote()
            features = ml.fetch_table_features(
                "Image",
                feature_name="Diagnosis",
                selector=selector,
            )

        Args:
            column: Name of the column to count values for. If None,
                auto-detects the first term column from feature metadata.

        Returns:
            A selector function ``(list[FeatureRecord]) -> FeatureRecord``.

        Raises:
            DerivaMLException: If column is None and the feature has no
                term columns or multiple term columns.
        """

        def _selector(records: list["FeatureRecord"]) -> "FeatureRecord":
            col = column
            if col is None:
                # Auto-detect from feature metadata on the record class
                record_cls = type(records[0])
                if (
                    hasattr(record_cls, "feature")
                    and record_cls.feature
                    and record_cls.feature.term_columns
                ):
                    if len(record_cls.feature.term_columns) == 1:
                        col = record_cls.feature.term_columns[0].name
                    else:
                        from deriva_ml.core.exceptions import (
                            DerivaMLException,
                        )

                        raise DerivaMLException(
                            "select_majority_vote requires a column name for "
                            "features with multiple term columns. "
                            f"Available: {[c.name for c in record_cls.feature.term_columns]}"
                        )
                else:
                    from deriva_ml.core.exceptions import (
                        DerivaMLException,
                    )

                    raise DerivaMLException(
                        "select_majority_vote requires a column name — "
                        "could not auto-detect from feature metadata."
                    )

            from collections import Counter

            counts = Counter(getattr(r, col, None) for r in records)
            max_count = max(counts.values())
            majority_values = {v for v, c in counts.items() if c == max_count}
            candidates = [
                r for r in records if getattr(r, col, None) in majority_values
            ]
            return max(candidates, key=lambda r: r.RCT or "")

        return _selector

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
        feature_table (Table): Table containing the feature implementation.
        target_table (Table): Table that the feature is associated with.
        feature_name (str): Name of the feature (from Feature_Name column default).
        feature_columns (set[Column]): All columns specific to this feature.
        asset_columns (set[Column]): Columns referencing asset tables.
        term_columns (set[Column]): Columns referencing vocabulary tables.
        value_columns (set[Column]): Columns containing direct values (not FK references).

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
        """Create a dynamically generated Pydantic model class for this feature.

        The returned class is a subclass of FeatureRecord with fields derived from
        the feature table's columns. Term columns accept vocabulary term names (str),
        asset columns accept file paths (str | Path), and value columns are typed
        according to their database column type (int, float, str).

        Returns:
            A FeatureRecord subclass with validated fields matching this feature's schema.
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
                case "boolean":
                    return bool
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

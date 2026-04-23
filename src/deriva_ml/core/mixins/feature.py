"""Feature management mixin for DerivaML.

This module provides the FeatureMixin class which handles
feature operations including creating, looking up, deleting,
and listing feature values.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterable

datapath = importlib.import_module("deriva.core.datapath")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Key = _ermrest_model.Key
Table = _ermrest_model.Table

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
        - apply_catalog_annotations(): method to update navbar (from DerivaML base class)

    Methods:
        create_feature: Create a new feature definition
        feature_record_class: Get pydantic model class for feature records
        delete_feature: Remove a feature definition
        lookup_feature: Retrieve a Feature object
        find_features: Find all features in the catalog, optionally filtered by table
        list_feature_values: Get all values for a feature
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    domain_schemas: frozenset[str]
    default_schema: str | None
    pathBuilder: Callable[[], Any]
    add_term: Callable[..., VocabularyTerm]
    apply_catalog_annotations: Callable[[], None]
    lookup_workflow: Callable[..., Any]
    find_executions: Callable[..., Iterable[Any]]

    def create_feature(
        self,
        target_table: Table | str,
        feature_name: str,
        terms: list[Table | str] | None = None,
        assets: list[Table | str] | None = None,
        metadata: list[ColumnDefinition | Table | Key | str] | None = None,
        optional: list[str] | None = None,
        comment: str = "",
        update_navbar: bool = True,
    ) -> type[FeatureRecord]:
        """Creates a new feature definition.

        A feature represents a measurable property or characteristic that can be associated with records in the target
        table. Features can include vocabulary terms, asset references, and additional metadata.

        **Side Effects**:
        This method dynamically creates:
        1. A new association table in the domain schema to store feature values
        2. A Pydantic model class (subclass of FeatureRecord) for creating validated feature instances

        The returned Pydantic model class provides type-safe construction of feature records with
        automatic validation of values against the feature's definition (vocabulary terms, asset
        references, etc.). Use this class to create feature instances that can be inserted into
        the catalog.

        Args:
            target_table: Table to associate the feature with (name or Table object).
            feature_name: Unique name for the feature within the target table.
            terms: Optional vocabulary tables/names whose terms can be used as feature values.
            assets: Optional asset tables/names that can be referenced by this feature.
            metadata: Optional columns, tables, or keys to include in a feature definition.
            optional: Column names that are not required when creating feature instances.
            comment: Description of the feature's purpose and usage.
            update_navbar: If True (default), automatically updates the navigation bar to include
                the new feature table. Set to False during batch feature creation to avoid
                redundant updates, then call apply_catalog_annotations() once at the end.

        Returns:
            type[FeatureRecord]: A dynamically generated Pydantic model class for creating
                validated feature instances. The class has fields corresponding to the feature's
                terms, assets, and metadata columns.

        Raises:
            DerivaMLException: If a feature definition is invalid or conflicts with existing features.

        Examples:
            Create a feature with confidence score:
                >>> DiagnosisFeature = ml.create_feature(
                ...     target_table="Image",
                ...     feature_name="Diagnosis",
                ...     terms=["Diagnosis_Type"],
                ...     metadata=[ColumnDefinition(name="confidence", type=BuiltinTypes.float4)],
                ...     comment="Clinical diagnosis label"
                ... )
                >>> # Use the returned class to create validated feature instances
                >>> record = DiagnosisFeature(
                ...     Image="1-ABC",  # Target record RID
                ...     Diagnosis_Type="Normal",  # Vocabulary term
                ...     confidence=0.95,
                ...     Execution="2-XYZ"  # Execution that produced this value
                ... )
        """
        # Initialize empty collections if None provided
        terms = terms or []
        assets = assets or []
        metadata = metadata or []
        optional = optional or []

        def normalize_metadata(m: Key | Table | ColumnDefinition | str | dict) -> Key | Table | dict:
            """Helper function to normalize metadata references.

            Handles:
            - str: Table name, converted to Table object
            - ColumnDefinition: Dataclass with to_dict() method
            - dict: Already in dict format (from Column.define())
            - Key/Table: Passed through unchanged
            """
            if isinstance(m, str):
                return self.model.name_to_table(m)
            elif isinstance(m, dict):
                # Already a dict (e.g., from Column.define())
                return m
            elif hasattr(m, 'to_dict'):
                # ColumnDefinition or similar dataclass
                return m.to_dict()
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
        atable = self.model.create_table(
            self.model._define_association(
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

        # Update navbar to include the new feature table
        if update_navbar:
            self.apply_catalog_annotations()

        # Return feature record class for creating instances
        return self.feature_record_class(target_table, feature_name)

    def feature_record_class(self, table: str | Table, feature_name: str) -> type[FeatureRecord]:
        """Returns a dynamically generated Pydantic model class for creating feature records.

        Each feature has a unique set of columns based on its definition (terms, assets, metadata).
        This method returns a Pydantic class with fields corresponding to those columns, providing:

        - **Type validation**: Values are validated against expected types (str, int, float, Path)
        - **Required field checking**: Non-nullable columns must be provided
        - **Default values**: Feature_Name is pre-filled with the feature's name

        **Field types in the generated class:**
        - `{TargetTable}` (str): Required. RID of the target record (e.g., Image RID)
        - `Execution` (str, optional): RID of the execution for provenance tracking
        - `Feature_Name` (str): Pre-filled with the feature name
        - Term columns (str): Accept vocabulary term names
        - Asset columns (str | Path): Accept asset RIDs or file paths
        - Value columns: Accept values matching the column type (int, float, str)

        Use `lookup_feature()` to inspect the feature's structure and see what columns
        are available.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to create a record class for.

        Returns:
            type[FeatureRecord]: A Pydantic model class for creating validated feature records.
                The class name follows the pattern `{TargetTable}Feature{FeatureName}`.

        Raises:
            DerivaMLException: If the feature doesn't exist or the table is invalid.

        Example:
            >>> # Get the dynamically generated class
            >>> DiagnosisFeature = ml.feature_record_class("Image", "Diagnosis")
            >>>
            >>> # Create a validated feature record
            >>> record = DiagnosisFeature(
            ...     Image="1-ABC",           # Target record RID
            ...     Diagnosis_Type="Normal", # Vocabulary term
            ...     confidence=0.95,         # Metadata column
            ...     Execution="2-XYZ"        # Provenance
            ... )
            >>>
            >>> # Convert to dict for insertion
            >>> record.model_dump()
            {'Image': '1-ABC', 'Diagnosis_Type': 'Normal', 'confidence': 0.95, ...}
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
        """Look up a feature definition by table and name.

        Returns a ``Feature`` object that describes the **schema structure**
        of a feature — not the feature values themselves. A Feature is a
        schema-level descriptor derived by inspecting the catalog's
        association tables. It tells you:

        - **What table the feature annotates** (``target_table``) — e.g., Image
        - **Where values are stored** (``feature_table``) — the association
          table linking targets to values and executions
        - **What kind of values it holds**, classified by column role:

          - ``term_columns``: columns referencing controlled vocabulary
            tables (e.g., a ``Diagnosis_Type`` column pointing to a
            vocabulary of diagnosis terms)
          - ``asset_columns``: columns referencing asset tables (e.g., a
            ``Segmentation_Mask`` column)
          - ``value_columns``: columns holding direct values like floats,
            ints, or text (e.g., a ``confidence`` score)

        The Feature object also provides ``feature_record_class()``, which
        returns a dynamically generated Pydantic model for constructing
        validated feature records to insert into the catalog.

        To retrieve actual feature **values**, use ``fetch_table_features``
        or ``list_feature_values`` instead.

        Args:
            table: The table the feature is defined on (name or Table object).
            feature_name: Name of the feature to look up.

        Returns:
            A Feature schema descriptor.

        Raises:
            DerivaMLException: If the feature doesn't exist on the specified
                table.

        Example:
            >>> feature = ml.lookup_feature("Image", "Classification")
            >>> print(f"Feature: {feature.feature_name}")
            >>> print(f"Stored in: {feature.feature_table.name}")
            >>> print(f"Term columns: {[c.name for c in feature.term_columns]}")
            >>> print(f"Value columns: {[c.name for c in feature.value_columns]}")
        """
        return self.model.lookup_feature(table, feature_name)

    def find_features(self, table: str | Table | None = None) -> list[Feature]:
        """Find feature definitions in the schema.

        Discovers features by inspecting the catalog schema for association tables
        that have ``Feature_Name`` and ``Execution`` columns. Returns Feature objects
        describing each feature's structure (target table, term/asset/value columns),
        not the feature values themselves.

        Use ``fetch_table_features`` or ``list_feature_values`` to retrieve actual
        feature values.

        Args:
            table: Optional table to find features for. If None, returns all feature
                definitions across all tables.

        Returns:
            A list of Feature instances describing the feature definitions.

        Examples:
            Find all feature definitions:
                >>> all_features = ml.find_features()
                >>> for f in all_features:
                ...     print(f"{f.target_table.name}.{f.feature_name}")

            Find features defined on a specific table:
                >>> image_features = ml.find_features("Image")
                >>> print([f.feature_name for f in image_features])
        """
        return list(self.model.find_features(table))

    def add_features(self, *args, **kwargs) -> int:
        """Retired — use ``exe.add_features(records)`` inside an execution context.

        ``DerivaML.add_features`` has been removed. Feature writes must go
        through the execution context so that provenance is tracked and values
        are staged for atomic upload.

        Replacement::

            with ml.create_execution(config).execute() as exe:
                exe.add_features(records)

        Raises:
            DerivaMLException: Always. Points at the replacement API.
        """
        raise DerivaMLException(
            "DerivaML.add_features() has been retired. "
            "Use exe.add_features(records) inside an execution context: "
            "``with ml.create_execution(config).execute() as exe: exe.add_features(records)``"
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Yield feature values for a single feature, one record per target RID.

        Returns an iterator of typed ``FeatureRecord`` instances. Each record is
        wide in shape — target RID, all value columns (vocab terms, asset
        references, metadata columns), and provenance columns (``Execution``,
        ``RCT``) — exposed as typed attributes.

        When a ``selector`` is provided, records are grouped by target RID and
        the selector collapses each group to a single survivor. Target RIDs
        whose group's selector returns ``None`` are omitted. When no selector
        is provided, every raw record is yielded — multiple records per target
        RID are possible.

        This method has identical signatures and semantics across ``DerivaML``,
        ``Dataset``, and ``DatasetBag``. The bag implementation reads from a
        per-feature denormalization cache populated on first access; subsequent
        calls are cheap.

        All rows for the feature are fetched from the catalog before the first
        record is yielded — this method is iterator-shaped for composability,
        not for streaming of very large feature tables.

        Args:
            table: Target table the feature is defined on (name or Table).
            feature_name: Name of the feature to read.
            selector: Optional callable
                ``(list[FeatureRecord]) -> FeatureRecord | None`` used to
                reduce multi-value groups. Built-ins include
                ``FeatureRecord.select_newest``,
                ``FeatureRecord.select_first``, and the factory
                ``FeatureRecord.select_by_workflow(workflow, container=...)``.
                Return ``None`` from a selector to omit that target RID.

        Returns:
            Iterator of ``FeatureRecord`` — one record per target RID after
            selector reduction, or all raw records if no selector.

        Raises:
            DerivaMLTableNotFound: ``table`` does not exist.
            DerivaMLException: ``feature_name`` is not a feature on ``table``.

        Example:
            Get the newest Glaucoma label per image::

                >>> from deriva_ml.feature import FeatureRecord
                >>> for rec in ml.feature_values(
                ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
                ... ):
                ...     print(f"{rec.Image}: {rec.Glaucoma} (by {rec.Execution})")

            Filter by a specific workflow — works identically on a downloaded bag::

                >>> workflow = ml.lookup_workflow("Glaucoma_Training_v2")
                >>> sel = FeatureRecord.select_by_workflow(workflow, container=ml)
                >>> labels = [r.Glaucoma for r in ml.feature_values(
                ...     "Image", "Glaucoma", selector=sel,
                ... )]

            Convert to a pandas DataFrame when needed::

                >>> import pandas as pd
                >>> df = pd.DataFrame(
                ...     r.model_dump()
                ...     for r in ml.feature_values("Image", "Glaucoma")
                ... )
        """
        table_obj = self.model.name_to_table(table)
        feat = self.lookup_feature(table_obj, feature_name)
        record_class = feat.feature_record_class()
        field_names = set(record_class.model_fields.keys())
        target_col = feat.target_table.name

        # Fetch raw rows via datapath
        pb = self.pathBuilder()
        raw_values = (
            pb.schemas[feat.feature_table.schema.name]
            .tables[feat.feature_table.name]
            .entities()
            .fetch()
        )

        # Materialize to FeatureRecord instances
        records: list[FeatureRecord] = [
            record_class(**{k: v for k, v in raw.items() if k in field_names})
            for raw in raw_values
        ]

        if selector is None:
            # No reduction — yield everything.
            yield from records
            return

        # Group by target RID, apply selector, skip None results.
        grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
        for rec in records:
            target_rid = getattr(rec, target_col, None)
            if target_rid is not None:
                grouped[target_rid].append(rec)

        for group in grouped.values():
            chosen = selector(group)
            if chosen is not None:
                yield chosen

    def fetch_table_features(self, *args, **kwargs):
        """Retired — use ``feature_values(table, name)`` or ``Denormalizer``.

        ``DerivaML.fetch_table_features`` has been removed. To read feature
        values for a single feature, use the new ``feature_values`` method::

            for rec in ml.feature_values("Image", "Quality"):
                ...

        For wide-table denormalization across all features use the
        ``Denormalizer`` subsystem.

        Raises:
            DerivaMLException: Always. Points at the replacement API.
        """
        raise DerivaMLException(
            "DerivaML.fetch_table_features() has been retired. "
            "Use feature_values(table, feature_name) to read a single feature, "
            "or Denormalizer for multi-feature wide tables."
        )

    def list_feature_values(self, *args, **kwargs) -> Iterable[FeatureRecord]:
        """Retired — renamed to ``feature_values``.

        ``DerivaML.list_feature_values`` has been removed. Use the new
        ``feature_values`` method instead::

            for rec in ml.feature_values("Image", "Quality"):
                ...

        The signature is identical (``table``, ``feature_name``, optional
        ``selector``).

        Raises:
            DerivaMLException: Always. Points at the replacement API.
        """
        raise DerivaMLException(
            "DerivaML.list_feature_values() has been retired and renamed. "
            "Use feature_values(table, feature_name, selector=...) instead."
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_workflow_executions(self, workflow: str) -> list[str]:
        """Return execution RIDs that ran the given workflow.

        The ``workflow`` argument resolves in two steps: first as a Workflow
        RID, and if that fails, as a Workflow_Type name. The returned list
        contains every execution RID for every workflow that matches.

        This method is the catalog-backed building block for
        ``FeatureRecord.select_by_workflow(workflow, container=ml)`` — it
        resolves the workflow's execution set once, and the selector closes
        over the result for cheap per-group membership testing.

        Entries are unique by construction (each execution runs one workflow).
        Consumers that need O(1) membership testing convert to ``set`` at the
        call site.

        Args:
            workflow: Workflow RID (e.g., ``"2-ABC1"``) or Workflow_Type name
                (e.g., ``"Training"``).

        Returns:
            List of execution RIDs, in insertion order. May be empty if the
            workflow exists but has no executions yet.

        Raises:
            DerivaMLException: If ``workflow`` does not resolve as a Workflow
                RID nor as a Workflow_Type name.

        Example:
            List all executions of a workflow and count them::

                >>> rids = ml.list_workflow_executions("Glaucoma_Training_v2")
                >>> print(f"{len(rids)} executions of this workflow")

            Use as the catalog-backed resolver for the selector factory::

                >>> from deriva_ml.feature import FeatureRecord
                >>> sel = FeatureRecord.select_by_workflow(
                ...     "Glaucoma_Training_v2", container=ml,
                ... )
        """
        # Try RID first — narrow scope so catalog errors inside find_executions propagate.
        try:
            wf = self.lookup_workflow(workflow)
        except DerivaMLException:
            wf = None

        if wf is not None:
            return [r.execution_rid for r in self.find_executions(workflow=wf)]

        # Fallback: treat `workflow` as a Workflow_Type name.
        # find_executions(workflow_type=...) returns an empty generator (not raise)
        # for unknown type names, so we check whether the result is empty and raise
        # only when neither the RID path nor the type-name path matched anything.
        rids = [r.execution_rid for r in self.find_executions(workflow_type=workflow)]
        if not rids:
            raise DerivaMLException(
                f"No workflow resolved for '{workflow}' — tried as Workflow RID "
                f"and Workflow_Type name."
            )
        return rids

    def select_by_workflow(self, *args, **kwargs) -> FeatureRecord:
        """Retired — use ``FeatureRecord.select_by_workflow(workflow, container=...)`` factory.

        ``DerivaML.select_by_workflow`` has been removed. The replacement is a
        classmethod factory that returns a selector callable compatible with the
        ``selector`` parameter of ``feature_values``::

            from deriva_ml.feature import FeatureRecord
            sel = FeatureRecord.select_by_workflow(workflow, container=ml)
            for rec in ml.feature_values("Image", "Quality", selector=sel):
                ...

        Raises:
            DerivaMLException: Always. Points at the replacement API.
        """
        raise DerivaMLException(
            "DerivaML.select_by_workflow() has been retired. "
            "Use FeatureRecord.select_by_workflow(workflow, container=ml) to build a selector, "
            "then pass it to feature_values(..., selector=sel)."
        )

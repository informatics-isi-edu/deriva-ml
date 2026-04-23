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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_features(self, features: list[FeatureRecord]) -> int:
        """Add feature values to the catalog in batch.

        Inserts a list of FeatureRecord instances into the appropriate feature table.
        All records must be from the same feature (i.e., created by the same
        ``feature_record_class()``). Records are batch-inserted for efficiency.

        Args:
            features: List of FeatureRecord instances to insert. All must share
                the same feature definition (same ``feature`` class variable).
                Create records using the class returned by
                ``Feature.feature_record_class()``.

        Returns:
            Number of feature records inserted.

        Raises:
            ValueError: If features list is empty.

        Example:
            >>> feature = ml.lookup_feature("Image", "Classification")
            >>> RecordClass = feature.feature_record_class()
            >>> records = [
            ...     RecordClass(Image="1-ABC", Image_Class="Normal", Execution=exe_rid),
            ...     RecordClass(Image="1-DEF", Image_Class="Abnormal", Execution=exe_rid),
            ... ]
            >>> count = ml.add_features(records)
            >>> print(f"Inserted {count} feature values")
        """
        if not features:
            raise ValueError("features list must not be empty")

        feature_table = features[0].feature.feature_table
        feature_path = self.pathBuilder().schemas[feature_table.schema.name].tables[feature_table.name]
        entries = feature_path.insert([f.model_dump() for f in features])
        return len(list(entries))

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

    def fetch_table_features(
        self,
        table: Table | str,
        feature_name: str | None = None,
        selector: Callable[[list[FeatureRecord]], FeatureRecord] | None = None,
    ) -> dict[str, list[FeatureRecord]]:
        """Fetch all feature values for a table, grouped by feature name.

        Returns a dictionary mapping feature names to lists of FeatureRecord
        instances. This is useful for retrieving all annotations on a table
        in a single call — for example, getting all classification labels,
        quality scores, and bounding boxes for a set of images at once.

        **Selector for resolving multiple values:**

        An asset may have multiple values for the same feature — for example,
        labels from different annotators, or predictions from successive model
        runs. When a ``selector`` is provided, records are grouped by target
        RID and the selector is called once per group to pick a single value.
        Groups with only one record are passed through unchanged.

        A selector is any callable with signature
        ``(list[FeatureRecord]) -> FeatureRecord``. Built-in selectors:

        - ``FeatureRecord.select_newest`` — picks the record with the most
          recent ``RCT`` (Row Creation Time).

        Custom selector example::

            def select_highest_confidence(records):
                return max(records, key=lambda r: getattr(r, "Confidence", 0))

        For workflow-aware selection, see ``select_by_workflow()``.

        Args:
            table: The table to fetch features for (name or Table object).
            feature_name: If provided, only fetch values for this specific
                feature. If ``None``, fetch all features on the table.
            selector: Optional function to select among multiple feature values
                for the same target object. Receives a list of FeatureRecord
                instances (all for the same target RID) and returns the selected
                one.

        Returns:
            dict[str, list[FeatureRecord]]: Keys are feature names, values are
            lists of FeatureRecord instances. When a selector is provided, each
            target object appears at most once per feature.

        Raises:
            DerivaMLException: If a specified ``feature_name`` doesn't exist
                on the table.

        Examples:
            Fetch all features for a table::

                >>> features = ml.fetch_table_features("Image")
                >>> for name, records in features.items():
                ...     print(f"{name}: {len(records)} values")

            Fetch a single feature with newest-value selection::

                >>> features = ml.fetch_table_features(
                ...     "Image",
                ...     feature_name="Classification",
                ...     selector=FeatureRecord.select_newest,
                ... )

            Convert results to a DataFrame::

                >>> features = ml.fetch_table_features("Image", feature_name="Quality")
                >>> import pandas as pd
                >>> df = pd.DataFrame([r.model_dump() for r in features["Quality"]])
        """
        table = self.model.name_to_table(table)
        features = self.find_features(table)
        if feature_name is not None:
            features = [f for f in features if f.feature_name == feature_name]
            if not features:
                raise DerivaMLException(
                    f"Feature '{feature_name}' not found on table '{table.name}'."
                )

        result: dict[str, list[FeatureRecord]] = {}

        for feat in features:
            record_class = feat.feature_record_class()
            field_names = set(record_class.model_fields.keys())
            target_col = feat.target_table.name

            # Query all feature values
            pb = self.pathBuilder()
            raw_values = (
                pb.schemas[feat.feature_table.schema.name]
                .tables[feat.feature_table.name]
                .entities()
                .fetch()
            )

            records: list[FeatureRecord] = []
            for raw_value in raw_values:
                filtered_data = {k: v for k, v in raw_value.items() if k in field_names}
                records.append(record_class(**filtered_data))

            if selector and records:
                # Group by target RID and apply selector
                grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
                for rec in records:
                    target_rid = getattr(rec, target_col, None)
                    if target_rid is not None:
                        grouped[target_rid].append(rec)
                records = [
                    selector(group) if len(group) > 1 else group[0]
                    for group in grouped.values()
                ]

            result[feat.feature_name] = records

        return result

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Retrieve all values for a single feature as typed FeatureRecord instances.

        Convenience wrapper around ``fetch_table_features()`` for the common
        case of querying a single feature by name. Returns a flat list of
        FeatureRecord objects — one per feature value (or one per target object
        when a ``selector`` is provided).

        Each returned record is a dynamically-generated Pydantic model with
        typed fields matching the feature's definition. For example, an
        ``Image_Classification`` feature might produce records with fields
        ``Image`` (str), ``Image_Class`` (str), ``Execution`` (str),
        ``RCT`` (str), and ``Feature_Name`` (str).

        Args:
            table: The table the feature is defined on (name or Table object).
            feature_name: Name of the feature to retrieve values for.
            selector: Optional function to resolve multiple values per target.
                See ``fetch_table_features`` for details on how selectors work.
                Use ``FeatureRecord.select_newest`` to pick the most recently
                created value.

        Returns:
            Iterable[FeatureRecord]: FeatureRecord instances with:

            - ``Execution``: RID of the execution that created this value
            - ``Feature_Name``: Name of the feature
            - ``RCT``: Row Creation Time (ISO 8601 timestamp)
            - Feature-specific columns as typed attributes (vocabulary terms,
              asset references, or value columns depending on the feature)
            - ``model_dump()``: Convert to a dictionary

        Raises:
            DerivaMLException: If the feature doesn't exist on the table.

        Examples:
            Get typed feature records::

                >>> for record in ml.list_feature_values("Image", "Quality"):
                ...     print(f"Image {record.Image}: {record.ImageQuality}")
                ...     print(f"Created by execution: {record.Execution}")

            Select newest when multiple values exist::

                >>> records = list(ml.list_feature_values(
                ...     "Image", "Quality",
                ...     selector=FeatureRecord.select_newest,
                ... ))

            Convert to a list of dicts::

                >>> dicts = [r.model_dump() for r in
                ...          ml.list_feature_values("Image", "Classification")]
        """
        result = self.fetch_table_features(table, feature_name=feature_name, selector=selector)
        return result.get(feature_name, [])

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

    def select_by_workflow(
        self,
        records: list[FeatureRecord],
        workflow: str,
    ) -> FeatureRecord:
        """Select the newest feature record created by a specific workflow.

        Filters a list of FeatureRecord instances to only those whose
        ``Execution`` was created by a matching workflow, then returns the
        newest match by RCT. This is useful when multiple model runs or
        annotators have labeled the same data and you want to use values
        from a particular workflow.

        **Resolution chain:**

        The ``workflow`` argument is first tried as a Workflow RID. If no
        workflow is found with that RID, it is treated as a Workflow_Type
        name (e.g., ``"Training"``, ``"Feature_Creation"``). The resolution
        chain is:

        1. ``workflow`` → ``Workflow.RID`` → all Executions for that workflow
        2. ``workflow`` → ``Workflow_Type.Name`` → all Workflows of that type
           → all Executions for those workflows

        Matching records are then filtered by ``Execution`` and the newest
        (by RCT) is returned.

        Note: Unlike ``FeatureRecord.select_newest``, this method cannot be
        passed directly as a ``selector`` argument because it requires catalog
        access. Call it directly on a list of records instead.

        Args:
            records: List of FeatureRecord instances to select from. Typically
                all values for a single target object from one feature.
            workflow: Either a Workflow RID (e.g., ``"2-ABC1"``) or a
                Workflow_Type name (e.g., ``"Training"``). Auto-detected:
                tries RID lookup first, falls back to type name.

        Returns:
            The newest FeatureRecord whose execution matches the workflow.

        Raises:
            DerivaMLException: If no workflows match the given identifier,
                no executions exist for the matched workflow(s), or no
                records in the input list were created by matching executions.

        Examples:
            Select the newest label from any Training workflow::

                >>> all_values = ml.list_feature_values("Image", "Classification")
                >>> from collections import defaultdict
                >>> by_image = defaultdict(list)
                >>> for v in all_values:
                ...     by_image[v.Image].append(v)
                >>> selected = {
                ...     img: ml.select_by_workflow(recs, "Training")
                ...     for img, recs in by_image.items()
                ... }

            Select by a specific workflow RID::

                >>> record = ml.select_by_workflow(records, "2-ABC1")
        """
        # Determine matching execution RIDs
        matching_execution_rids: set[str] = set()

        # Try as a Workflow RID first
        try:
            wf = self.lookup_workflow(workflow)
            # Found a workflow — get all executions for this workflow
            for exec_record in self.find_executions(workflow=wf):
                matching_execution_rids.add(exec_record.execution_rid)
        except DerivaMLException:
            # Not a valid workflow RID — treat as Workflow_Type name
            pb = self.pathBuilder()
            wt_assoc = pb.schemas[self.ml_schema].Workflow_Workflow_Type
            matching_workflows = {
                row["Workflow"]
                for row in wt_assoc.filter(
                    wt_assoc.Workflow_Type == workflow
                ).entities().fetch()
            }
            if not matching_workflows:
                raise DerivaMLException(
                    f"No workflows found for workflow type '{workflow}'."
                )
            for exec_record in self.find_executions():
                if exec_record.workflow_rid in matching_workflows:
                    matching_execution_rids.add(exec_record.execution_rid)

        if not matching_execution_rids:
            raise DerivaMLException(
                f"No executions found for workflow '{workflow}'."
            )

        # Filter records to those matching the workflow's executions
        filtered = [r for r in records if r.Execution in matching_execution_rids]
        if not filtered:
            raise DerivaMLException(
                f"No feature records match workflow '{workflow}'."
            )

        return FeatureRecord.select_newest(filtered)

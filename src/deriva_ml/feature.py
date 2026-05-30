"""Feature implementation for deriva-ml.

This module provides classes for defining and managing features in deriva-ml.
Features represent measurable properties or characteristics associated with
records in a target table (e.g., a diagnostic label on an Image row).

Exported classes:
    Feature: Encapsulates a feature's schema — target table, vocabulary columns,
        asset columns, and value columns. Obtained via ``DerivaML.create_feature``
        or ``DerivaML.lookup_feature``. Not constructed directly.
    FeatureRecord: Pydantic base class for dynamically generated feature record
        models. Subclasses are created by ``Feature.feature_record_class()``.

Selector classmethod suite (``FeatureRecord`` class methods):
    ``FeatureRecord.select_newest(records)`` — Returns the record with the most
        recent ``RCT`` (Row Creation Time). Useful when multiple annotators have
        labelled the same object.
    ``FeatureRecord.select_first(records)`` — Returns the record with the
        earliest ``RCT``. Useful to preserve the original annotation.
    ``FeatureRecord.select_latest(records)`` — Alias for ``select_newest``.
    ``FeatureRecord.select_by_execution(execution_rid)`` — Returns a selector
        that picks the newest record from a specific execution run.
    ``FeatureRecord.select_by_workflow(workflow, *, container)`` — Returns a
        selector that picks the newest record from any execution of the named
        workflow. Resolves the execution list eagerly at construction time.
    ``FeatureRecord.select_majority_vote(column)`` — Returns a selector that
        picks the most common value for a column (consensus labeling).

Typical usage (the generated record carries one field per feature
column, plus a *required* field named after the target table — here
``Image`` — that holds the target row's RID)::

    >>> feature = ml.lookup_feature("Image", "Diagnosis")  # doctest: +SKIP
    >>> DiagnosisRecord = feature.feature_record_class()  # doctest: +SKIP
    >>> record = DiagnosisRecord(Image="1-abc123", Diagnosis="benign", Confidence=0.97)  # doctest: +SKIP
"""

from collections import Counter, defaultdict
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Callable, ClassVar, Iterable, Iterator, Optional, Type

from deriva.core.ermrest_model import Column, FindAssociationResult
from pydantic import BaseModel, create_model

from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


def reduce_with_selector(
    records: Iterable["FeatureRecord"],
    target_col: str,
    selector: Callable[[list["FeatureRecord"]], "FeatureRecord | None"],
    qualifier_cols: Iterable[str] = (),
) -> Iterator["FeatureRecord"]:
    """Group records by feature identity, apply ``selector`` per group, drop ``None`` results.

    Centralises the three-step pattern that previously lived inline
    in three places — ``feature_values`` on
    :class:`~deriva_ml.core.mixins.feature.FeatureMixin`,
    :class:`~deriva_ml.dataset.dataset.Dataset`, and
    :class:`~deriva_ml.dataset.dataset_bag.DatasetBag`:

    1. Group raw records by their *feature identity*: the target-RID
       column (the FK to the target table, e.g. ``Image`` for an
       Image feature) plus any ``qualifier_cols`` (value FKs that
       participate in the association table's compound uniqueness
       key, e.g. ``Image_Side`` for a per-eye feature).
    2. Apply the selector to each group; the selector picks one
       record from the group (or returns ``None`` to drop).
    3. Yield surviving selections.

    A selector exists to reduce *redundant* records describing the
    same logical observation (the same Image labelled by three
    annotators, say) down to one. The grouping unit must therefore
    be the feature's identity, not the target RID alone. For an
    ordinary (unqualified) feature, identity *is* the target RID, so
    with the default empty ``qualifier_cols`` the composite key
    degenerates to ``(target_rid,)`` and the behavior is byte-identical
    to grouping on the target RID alone. For a *key-qualified* feature
    (a value FK in the uniqueness key), the schema author declared
    ``(target, *qualifiers)`` to be the identity — e.g. a left-eye and
    a right-eye ``Chart_Label`` for the same Subject are two distinct
    observations, not redundant copies — so the selector must reduce
    *within* each qualifier bucket and preserve both.

    Records whose ``target_col`` is missing or ``None`` are
    silently dropped before grouping — they have no target to
    group under. ``None`` qualifier values are tolerated and form
    their own bucket.

    Args:
        records: Raw feature records to group + reduce. Iterated
            once.
        target_col: Name of the FK column on each record that
            identifies its target row (e.g. ``"Image"``).
        selector: Callable that takes a non-empty list of records
            sharing one feature identity and returns the chosen
            record (or ``None`` to skip). Examples:
            :meth:`FeatureRecord.select_newest`,
            :meth:`FeatureRecord.select_by_workflow` (returned by
            the closure factory of the same name).
        qualifier_cols: Names of additional record fields that, with
            ``target_col``, form the feature's composite identity key
            (a :class:`~deriva_ml.feature.Feature`'s
            ``qualifier_columns``). Defaults to empty — group by target
            RID alone, the historical behavior for unqualified features.

    Yields:
        One :class:`FeatureRecord` per ``(target_rid, *qualifiers)``
        identity for which the selector returned a non-``None`` choice.
        Order is unspecified (dict iteration order, i.e. insertion
        order of first-seen identities in ``records``).

    Example:
        >>> from deriva_ml.feature import FeatureRecord, reduce_with_selector
        >>> callable(reduce_with_selector)
        True
    """
    qualifier_cols = tuple(qualifier_cols)
    grouped: dict[tuple, list[FeatureRecord]] = defaultdict(list)
    for rec in records:
        target_rid = getattr(rec, target_col, None)
        if target_rid is not None:
            key = (target_rid, *(getattr(rec, q, None) for q in qualifier_cols))
            grouped[key].append(rec)
    for group in grouped.values():
        chosen = selector(group)
        if chosen is not None:
            yield chosen


class FeatureRecord(BaseModel):
    """Base class for dynamically generated feature record models.

    This class serves as the base for pydantic models that represent feature
    records. Each feature record contains the values and metadata associated
    with a feature instance. Subclasses are created dynamically by
    ``Feature.feature_record_class()`` with fields corresponding to the
    feature's vocabulary terms, asset references, and metadata columns.

    Feature records are returned by ``feature_values()``. They can also be
    constructed manually and passed to ``Execution.add_features()`` to insert
    new values into the catalog.

    **Handling multiple values per target:**

    When the same target object (e.g., an Image) has multiple feature values
    — for example, labels from different annotators or model runs — use
    a ``selector`` function to choose one. Pass it to ``feature_values``.
    A selector receives a list of FeatureRecord instances for the same target
    and returns the selected one::

        # Built-in: pick the most recently created record
        for rec in ml.feature_values("Image", selector=FeatureRecord.select_newest):
            ...

        # Custom: pick the record with highest confidence
        def select_best(records):
            return max(records, key=lambda r: getattr(r, "Confidence", 0))

        for rec in ml.feature_values("Image", selector=select_best):
            ...

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
        argument to ``feature_values``::

            for rec in ml.feature_values("Image", selector=FeatureRecord.select_newest):
                ...

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

        Unlike ``select_by_workflow`` (a factory that resolves the workflow's
        execution set from a container), this selector filters on a known
        ``Execution`` RID with no container dependency and can be passed
        directly as the ``selector`` argument to ``feature_values``::

            for rec in ml.feature_values(
                "Image",
                feature_name="FooBar",
                selector=FeatureRecord.select_by_execution("3WY2"),
            ):
                ...

        Args:
            execution_rid: RID of the execution to filter by.

        Returns:
            A selector function ``(list[FeatureRecord]) -> FeatureRecord | None``
            suitable for use as the ``selector=`` argument to
            ``feature_values``. Returns ``None`` when no record in the group
            matches the execution; returns the newest matching record (by RCT)
            otherwise. Mirrors ``select_by_workflow``'s no-match semantics:
            ``feature_values`` invokes the selector once per target row, so
            targets whose record group lacks a match are silently omitted
            from the output rather than aborting the whole query.

        Examples:
            Select values from a specific execution::

                >>> for rec in ml.feature_values(  # doctest: +SKIP
                ...     "Image",
                ...     feature_name="Classification",
                ...     selector=FeatureRecord.select_by_execution("3WY2"),
                ... ):
                ...     print(rec)
        """

        def _selector(records: list["FeatureRecord"]) -> "FeatureRecord | None":
            filtered = [r for r in records if r.Execution == execution_rid]
            if not filtered:
                # Return None so feature_values omits this target RID silently.
                # Parallel to select_by_workflow's behavior; raising used to
                # abort the whole query when any target row lacked a record
                # from the target execution, which made the selector
                # unusable in any catalog where the feature is sparsely
                # annotated by execution (the common case).
                return None
            return FeatureRecord.select_newest(filtered)

        return _selector

    @classmethod
    def select_by_workflow(cls, workflow: str, *, container) -> "Callable[[list[FeatureRecord]], FeatureRecord | None]":
        """Return a selector that picks the newest record from a specific workflow.

        Creates a selector function that filters records to those produced by
        executions of the given workflow, then returns the newest match by RCT.
        This is the recommended replacement for the retired
        ``DerivaML.select_by_workflow(records, workflow)`` method.

        Unlike ``select_by_execution``, which requires knowing a specific
        execution RID, this selector works at the workflow level — it accepts
        any record produced by any execution of the named workflow.

        **Eager resolution:** the workflow's execution list is resolved once
        at factory-construction time by calling
        ``container.list_workflow_executions(workflow)``. Unknown-workflow
        errors therefore surface immediately (at factory-call time), not
        lazily during iteration.

        **None return semantics:** when no record in a group matches the
        workflow, the selector returns ``None``. ``feature_values`` treats
        ``None`` as "feature absent for this target RID" and omits the target
        from the iterator. ``select_by_execution`` follows the same
        convention.

        Args:
            workflow: Name (or RID) of the workflow to filter by. Must be a
                workflow known to ``container``; an unknown name raises
                ``DerivaMLException`` immediately.
            container: Required keyword-only argument. An object that
                implements ``list_workflow_executions(workflow) -> list[str]``.
                Typically a ``DerivaML``, ``Dataset``, or ``DatasetBag``
                instance. The container determines which executions are in
                scope (all catalog executions for ``DerivaML``; dataset-scoped
                executions for ``Dataset`` / ``DatasetBag``).

        Returns:
            A selector callable ``(list[FeatureRecord]) -> FeatureRecord | None``
            suitable for use as the ``selector=`` argument to
            ``feature_values``. Returns ``None``
            when no record in the group matches the workflow; returns the
            newest matching record (by RCT) otherwise.

        Raises:
            DerivaMLException: If ``workflow`` is not known to ``container``.
                Raised at factory-construction time (eager resolution).
            TypeError: If ``container`` is passed positionally (it is
                keyword-only).

        Example:
            Select Glaucoma labels produced by a specific training workflow::

                >>> selector = FeatureRecord.select_by_workflow(  # doctest: +SKIP
                ...     "Glaucoma_Training_v2", container=ml
                ... )
                >>> for rec in ml.feature_values(  # doctest: +SKIP
                ...     "Image", "Glaucoma", selector=selector
                ... ):
                ...     print(f"{rec.Image}: {rec.Glaucoma}")

            Works identically on a downloaded bag (offline)::

                >>> selector = FeatureRecord.select_by_workflow(  # doctest: +SKIP
                ...     "Glaucoma_Training_v2", container=bag
                ... )
                >>> labels = list(bag.feature_values("Image", "Glaucoma", selector=selector))  # doctest: +SKIP
        """
        # Eager resolution: fail fast on unknown workflow at construction time,
        # not lazily during iteration. Convert to a set for O(1) membership
        # testing inside the closure.
        execution_rids: set[str] = set(container.list_workflow_executions(workflow))

        def _selector(records: list["FeatureRecord"]) -> "FeatureRecord | None":
            matched = [r for r in records if r.Execution in execution_rids]
            if not matched:
                # Return None so feature_values omits this target RID silently.
                return None
            return FeatureRecord.select_newest(matched)

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
        argument to ``feature_values``::

            for rec in ml.feature_values("Image", selector=FeatureRecord.select_first):
                ...

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
            for rec in ml.feature_values(
                "Image",
                feature_name="Diagnosis",
                selector=selector,
            ):
                ...

        Args:
            column: Name of the column to count values for. If None,
                auto-detects the first term column from feature metadata.

        Returns:
            A selector function ``(list[FeatureRecord]) -> FeatureRecord``.

        Raises:
            DerivaMLException: If column is None and the feature has no
                term columns or multiple term columns.
        """

        def _selector(records: list["FeatureRecord"]) -> "FeatureRecord | None":
            # Empty input → return ``None`` per the selector
            # convention (matches :meth:`select_by_workflow` and
            # :meth:`select_by_execution`; ``feature_values`` skips
            # ``None`` survivors when reducing groups). Pre-fix
            # (audit F-2) this hit ``records[0]`` below and
            # raised ``IndexError``. Audit F-20 added the test.
            if not records:
                return None
            col = column
            if col is None:
                # Auto-detect from feature metadata on the record class.
                # ``term_columns`` is a ``set[Column]`` (see Feature.__init__),
                # so we can't index it. Single-term features have exactly
                # one element to pull out; multi-term features need an
                # explicit ``column=`` from the caller.
                record_cls = type(records[0])
                if hasattr(record_cls, "feature") and record_cls.feature and record_cls.feature.term_columns:
                    term_cols = record_cls.feature.term_columns
                    if len(term_cols) == 1:
                        col = next(iter(term_cols)).name
                    else:
                        # Sort for a deterministic error message.
                        available = sorted(c.name for c in term_cols)
                        raise DerivaMLException(
                            "select_majority_vote requires a column name for "
                            "features with multiple term columns. "
                            f"Available: {available}"
                        )
                else:
                    raise DerivaMLException(
                        "select_majority_vote requires a column name — could not auto-detect from feature metadata."
                    )

            counts = Counter(getattr(r, col, None) for r in records)
            max_count = max(counts.values())
            majority_values = {v for v, c in counts.items() if c == max_count}
            candidates = [r for r in records if getattr(r, col, None) in majority_values]
            return max(candidates, key=lambda r: r.RCT or "")

        return _selector

    @classmethod
    def _require_feature(cls) -> "Feature":
        """Return ``cls.feature``, raising a helpful error if unset.

        The four column-accessor classmethods below all need
        ``cls.feature`` to be populated, which only happens on
        subclasses returned by ``Feature.feature_record_class()``.
        Calling them on the ``FeatureRecord`` base class would
        otherwise crash with ``AttributeError: 'NoneType' object
        has no attribute 'feature_columns'`` — this guard
        substitutes a useful message naming the actual problem.
        """
        if cls.feature is None:
            raise DerivaMLException(
                f"{cls.__name__}.feature is None — the column-accessor "
                "classmethods (feature_columns, asset_columns, "
                "term_columns, value_columns) are only valid on "
                "subclasses returned by Feature.feature_record_class(), "
                "not on the FeatureRecord base class."
            )
        return cls.feature

    @classmethod
    def feature_columns(cls) -> set[Column]:
        """Return all columns specific to this feature.

        Returns the full set of feature-specific columns — the union of
        ``asset_columns``, ``term_columns``, and ``value_columns``. System
        columns (``RID``, ``RCT``, ``RMT``, ``RCB``, ``RMB``) and structural
        association columns (``Feature_Name``, the target-table FK, and
        ``Execution``) are excluded.

        Returns:
            set[Column]: Feature-specific ERMrest ``Column`` objects. Equivalent
            to ``cls.feature.feature_columns``.

        Raises:
            DerivaMLException: Called on the ``FeatureRecord`` base class
                rather than a subclass returned by
                ``Feature.feature_record_class()``.
        """
        return cls._require_feature().feature_columns

    @classmethod
    def asset_columns(cls) -> set[Column]:
        """Return columns that reference asset tables.

        Asset columns are FK columns whose referent table is classified as an
        asset table (e.g., ``Image``, ``Scan``). In a generated
        ``FeatureRecord`` subclass these fields accept ``str | Path`` values.

        Returns:
            set[Column]: ERMrest ``Column`` objects that are FK references to
            asset tables. A subset of ``feature_columns()``.

        Raises:
            DerivaMLException: Called on the ``FeatureRecord`` base class
                rather than a subclass returned by
                ``Feature.feature_record_class()``.
        """
        return cls._require_feature().asset_columns

    @classmethod
    def term_columns(cls) -> set[Column]:
        """Return columns that reference controlled vocabulary terms.

        Term columns are FK columns whose referent table is classified as a
        vocabulary table. In a generated ``FeatureRecord`` subclass these
        fields accept ``str`` values (the term name, not the RID).

        Returns:
            set[Column]: ERMrest ``Column`` objects that are FK references to
            vocabulary tables. A subset of ``feature_columns()``.

        Raises:
            DerivaMLException: Called on the ``FeatureRecord`` base class
                rather than a subclass returned by
                ``Feature.feature_record_class()``.
        """
        return cls._require_feature().term_columns

    @classmethod
    def value_columns(cls) -> set[Column]:
        """Return columns that contain direct (non-FK) values.

        Value columns hold scalar data — integers, floats, booleans, or text
        — rather than FK references to other tables. In a generated
        ``FeatureRecord`` subclass these fields are typed according to the
        ERMrest column type (``int``, ``float``, ``bool``, or ``str``).

        Returns:
            set[Column]: ERMrest ``Column`` objects that contain direct data
            values. Computed as ``feature_columns() - asset_columns() -
            term_columns()``.

        Raises:
            DerivaMLException: Called on the ``FeatureRecord`` base class
                rather than a subclass returned by
                ``Feature.feature_record_class()``.
        """
        return cls._require_feature().value_columns


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
        qualifier_columns (list[str]): Names of value-FK columns that participate
            in the association table's compound uniqueness key (e.g. ``Image_Side``
            on a per-eye feature). Empty for ordinary features, where the target
            RID alone is the identity. Used by ``reduce_with_selector`` to group on
            ``(target, *qualifiers)`` so a selector preserves distinct qualified
            observations instead of collapsing them.

    Example:
        >>> feature = ml.lookup_feature("Image", "Diagnosis")  # doctest: +SKIP
        >>> print(f"Feature {feature.feature_name} on {feature.target_table.name}")  # doctest: +SKIP
        >>> print("Asset columns:", [c.name for c in feature.asset_columns])  # doctest: +SKIP
    """

    def __init__(self, atable: FindAssociationResult, model: "DerivaModel") -> None:
        """Initialize a Feature from an association table result.

        Classifies the feature table's FK columns into three disjoint sets:
        ``asset_columns`` (FK to an asset table), ``term_columns`` (FK to a
        vocabulary table), and ``value_columns`` (everything else). The
        association FKs linking back to the target table and to the feature
        name vocabulary are excluded before classification.

        Args:
            atable: Result from ``deriva.core.ermrest_model.FindAssociationResult``
                describing the feature association table. Provides the feature
                table, the self-FK back to the target, and the set of other FKs.
            model: ``DerivaModel`` instance used to classify FK targets as
                asset or vocabulary tables.

        Note:
            This constructor is not part of the public API. Obtain ``Feature``
            instances via ``DerivaML.create_feature`` or
            ``DerivaML.lookup_feature``.
        """
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

        # Exclude the two FKs that are structural parts of the association table
        # itself — the self-FK pointing back to the target table (e.g., Image)
        # and the other-FKs pointing to Feature_Name and Execution — before
        # classifying the remaining FKs as asset, term, or value columns. Without
        # this subtraction, those structural FKs would be misclassified as feature
        # columns and create spurious fields in the generated FeatureRecord class.
        assoc_fkeys = {atable.self_fkey} | atable.other_fkeys

        # Qualifier columns are the value FKs that participate in the
        # association table's compound uniqueness key — i.e. ``other_fkeys``
        # minus the structural ``Feature_Name`` / ``Execution`` FKs (the
        # self-FK to the target is never in ``other_fkeys``). Their presence
        # means the schema author declared ``(target, *qualifiers)`` — not the
        # target alone — to be the feature's identity (e.g. ``Image_Side`` on
        # ``Execution_Subject_Chart_Label``: one Chart_Label row per eye per
        # Subject). ``reduce_with_selector`` must group on this composite
        # identity so a selector reduces redundant records *within* each
        # qualifier bucket instead of collapsing distinct observations.
        #
        # For an unqualified feature ``other_fkeys`` is exactly
        # ``{Feature_Name FK, Execution FK}``, so this set is empty and
        # grouping degenerates to group-by-target — the historical behavior.
        # Exposed as the FeatureRecord *field names* (the association table's
        # own column names) so ``reduce_with_selector`` can ``getattr`` them.
        structural_fk_names = {"Feature_Name", "Execution"}
        self.qualifier_columns: list[str] = [
            fk.foreign_key_columns[0].name
            for fk in atable.other_fkeys
            if fk.foreign_key_columns[0].name not in structural_fk_names
        ]

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

        Builds a ``FeatureRecord`` subclass with fields derived from the feature
        table's columns. Column types are mapped as follows:

        - Term columns (FK to vocabulary): ``str`` (vocabulary term name)
        - Asset columns (FK to asset table): ``str | Path`` (file path)
        - Value columns (direct data): typed per the database column type
          (``int``, ``float``, ``bool``, or ``str``)

        All feature-specific fields are ``Optional`` with a default of ``None``
        to allow partial construction when building records for insertion.
        The ``Feature_Name`` field defaults to this feature's name.

        Args:
            self: The ``Feature`` instance whose schema drives field generation.

        Returns:
            A subclass of ``FeatureRecord`` whose fields match this feature's
            schema. The class's ``feature`` ClassVar is set to ``self``.

        Raises:
            DerivaMLException: If the feature table schema cannot be read.

        Example:
            The generated model has one field per feature column plus a
            *required* field named after the target table (``Image``)
            that holds the target row's RID::

            >>> feature = ml.lookup_feature("Image", "Diagnosis")  # doctest: +SKIP
            >>> DiagnosisRecord = feature.feature_record_class()  # doctest: +SKIP
            >>> rec = DiagnosisRecord(Image="1-abc123", Diagnosis="benign")  # doctest: +SKIP
        """
        # Compute the asset-column-name set once; map_type is called
        # for every column on the feature, and re-computing this
        # comprehension on each call is N² for a feature with N
        # columns.
        asset_col_names = {c.name for c in self.asset_columns}

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
                >>> col = Column(name="score", type="float4")  # doctest: +SKIP
                >>> typ = map_type(col)  # Returns float  # doctest: +SKIP
            """
            if c.name in asset_col_names:
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

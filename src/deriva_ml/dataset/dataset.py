"""Dataset management for DerivaML.

This module provides functionality for managing datasets in DerivaML. A dataset represents a collection
of related data that can be versioned, downloaded, and tracked. The module includes:

- Dataset class: Core class for dataset operations
- Version management: Track and update dataset versions
- History tracking: Record dataset changes over time
- Download capabilities: Export datasets as BDBags
- Relationship management: Handle dataset dependencies and hierarchies

The Dataset class serves as a base class in DerivaML, making its methods accessible through
DerivaML class instances.

Typical usage example:
    >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
    >>> with ml.create_execution(config) as exe:  # doctest: +SKIP
    ...     dataset = exe.create_dataset(
    ...         dataset_types=['experiment'],
    ...         description='Experimental data'
    ...     )
    ...     dataset.add_dataset_members(members=['1-abc123', '1-def456'])
    ...     dataset.release(
    ...         bump=VersionPart.minor,
    ...         description='Added new samples'
    ...     )
"""

from __future__ import annotations

from collections import defaultdict

# Standard library imports
from graphlib import TopologicalSorter

# Local imports
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Iterator, Self

# Deriva imports
from deriva.core.asyncio import AsyncErmrestCatalog
from deriva.core.asyncio.async_catalog import AsyncErmrestSnapshot

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution
    from deriva_ml.feature import FeatureRecord

# Third-party imports
import pandas as pd
from deriva.core.ermrest_model import Table
from pydantic import validate_call

from deriva_ml.core.async_helpers import run_async
from deriva_ml.core.constants import RID
from deriva_ml.core.definitions import (
    MLVocab,
    VocabularyTerm,
)
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.mixins.rid_resolution import AnyQuantifier
from deriva_ml.core.validation import VALIDATION_CONFIG
from deriva_ml.dataset.aux_classes import (
    DatasetHistory,
    DatasetSpec,
    DatasetVersion,
    VersionPart,
)
from deriva_ml.dataset.bag_builder import DatasetBagBuilder
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.feature import Feature
from deriva_ml.interfaces import DerivaMLCatalog
from deriva_ml.model.database import DatabaseModel


class Dataset:
    """Manages dataset operations in a Deriva catalog.

    The Dataset class provides functionality for creating, modifying, and tracking datasets
    in a Deriva catalog. It handles versioning, relationships between datasets, and data export.

    A Dataset is a versioned collection of related data elements. Each dataset:
    - Has a unique RID (Resource Identifier) within the catalog
    - Maintains a version history using semantic versioning (major.minor.patch)
    - Can contain nested datasets, forming a hierarchy
    - Can be exported as a BDBag for offline use or sharing

    The class implements the DatasetLike protocol, allowing code to work uniformly
    with both live catalog datasets and downloaded DatasetBag objects.

    Attributes:
        dataset_rid (RID): The unique Resource Identifier for this dataset.
        dataset_types (list[str]): List of vocabulary terms describing the dataset type.
        description (str): Human-readable description of the dataset.
        execution_rid (RID | None): Optional RID of the execution that created this dataset.
        _ml_instance (DerivaMLCatalog): Reference to the catalog containing this dataset.

    Example:
        >>> # Create a new dataset via an execution
        >>> with ml.create_execution(config) as exe:  # doctest: +SKIP
        ...     dataset = exe.create_dataset(
        ...         dataset_types=["training_data"],
        ...         description="Image classification training set"
        ...     )
        ...     # Add members to the dataset (lands on a dev version)
        ...     dataset.add_dataset_members(members=["1-abc", "1-def"])
        ...     # Promote the dev period to a released version
        ...     new_version = dataset.release(VersionPart.minor, "Added samples")
        >>> # Download for offline use
        >>> bag = dataset.download_dataset_bag(version=new_version)  # doctest: +SKIP
    """

    @validate_call(config=VALIDATION_CONFIG)
    def __init__(
        self,
        catalog: DerivaMLCatalog,
        dataset_rid: RID,
        description: str = "",
        execution_rid: RID | None = None,
    ):
        """Initialize a Dataset object from an existing dataset in the catalog.

        This constructor wraps an existing dataset record. To create a new dataset
        in the catalog, use the static method Dataset.create_dataset() instead.

        Args:
            catalog: The DerivaMLCatalog instance containing this dataset.
            dataset_rid: The RID of the existing dataset record.
            description: Human-readable description of the dataset's purpose and contents.
            execution_rid: Optional execution RID that created or is associated with this dataset.

        Example:
            >>> # Wrap an existing dataset
            >>> dataset = Dataset(catalog=ml, dataset_rid="4HM")  # doctest: +SKIP
        """
        self._logger = get_logger(__name__)
        self.dataset_rid = dataset_rid
        self.execution_rid = execution_rid
        self._ml_instance = catalog
        self.description = description

    def __repr__(self) -> str:
        """Return a string representation of the Dataset for debugging."""
        return (
            f"<deriva_ml.Dataset object at {hex(id(self))}: rid='{self.dataset_rid}', "
            f"version='{self.current_version}', types={self.dataset_types}>"
        )

    def __hash__(self) -> int:
        """Return hash based on dataset RID for use in sets and as dict keys.

        This allows Dataset objects to be stored in sets and used as dictionary keys.
        Two Dataset objects with the same RID will hash to the same value.
        """
        return hash(self.dataset_rid)

    def __eq__(self, other: object) -> bool:
        """Check equality based on dataset RID.

        Two Dataset objects are considered equal if they reference the same
        dataset RID, regardless of other attributes like version or types.

        Args:
            other: Object to compare with.

        Returns:
            True if other is a Dataset with the same RID, False otherwise.
            Returns NotImplemented for non-Dataset objects.
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.dataset_rid == other.dataset_rid

    def _get_dataset_type_association_table(self) -> tuple[str, Any]:
        """Get the association table for dataset types.

        Returns:
            Tuple of (table_name, table_path) for the Dataset-Dataset_Type association table.
        """
        associations = list(
            self._ml_instance.model.schemas[self._ml_instance.ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        )
        atable_name = associations[0].name if associations else None
        pb = self._ml_instance.pathBuilder()
        atable_path = pb.schemas[self._ml_instance.ml_schema].tables[atable_name]
        return atable_name, atable_path

    @property
    def dataset_types(self) -> list[str]:
        """Get the dataset types from the catalog.

        This property fetches the current dataset types directly from the catalog,
        ensuring consistency when multiple Dataset instances reference the same
        dataset or when types are modified externally.

        Returns:
            List of dataset type term names from the Dataset_Type vocabulary.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> types = dataset.dataset_types  # doctest: +SKIP
            >>> print(types)  # doctest: +SKIP
        """
        _, atable_path = self._get_dataset_type_association_table()
        ds_types = (
            atable_path.filter(atable_path.Dataset == self.dataset_rid).attributes(atable_path.Dataset_Type).fetch()
        )
        return [ds[MLVocab.dataset_type] for ds in ds_types]

    @staticmethod
    @validate_call(config=VALIDATION_CONFIG)
    def create_dataset(
        ml_instance: DerivaMLCatalog,
        execution_rid: RID,
        dataset_types: str | list[str] | None = None,
        description: str = "",
        version: DatasetVersion | None = None,
    ) -> Self:
        """Creates a new dataset in the catalog.

        Creates a dataset with specified types and description. The dataset must be
        associated with an execution for provenance tracking.

        Args:
            ml_instance: DerivaMLCatalog instance.
            execution_rid: Execution RID to associate with dataset creation (required).
            dataset_types: One or more dataset type terms from Dataset_Type vocabulary.
            description: Description of the dataset's purpose and contents.
            version: Optional initial version number. Defaults to 0.1.0.

        Returns:
            Dataset: The newly created dataset.

        Raises:
            DerivaMLException: If dataset_types are invalid or creation fails.

        Example:
            >>> with ml.create_execution(config) as exe:  # doctest: +SKIP
            ...     dataset = exe.create_dataset(
            ...         dataset_types=["experiment", "raw_data"],
            ...         description="RNA sequencing experiment data",
            ...         version=DatasetVersion(1, 0, 0)
            ...     )
        """

        version = version or DatasetVersion(0, 1, 0)

        # Validate dataset types
        ds_types = [dataset_types] if isinstance(dataset_types, str) else dataset_types
        dataset_types = [ml_instance.lookup_term(MLVocab.dataset_type, t) for t in ds_types]

        # Create the entry for the new dataset_table and get its RID.
        pb = ml_instance.pathBuilder()
        dataset_table_path = pb.schemas[ml_instance._dataset_table.schema.name].tables[ml_instance._dataset_table.name]
        dataset_rid = dataset_table_path.insert(
            [
                {
                    "Description": description,
                    "Deleted": False,
                }
            ]
        )[0]["RID"]

        pb.schemas[ml_instance.model.ml_schema].Dataset_Execution.insert(
            [{"Dataset": dataset_rid, "Execution": execution_rid}]
        )
        dataset = Dataset(
            catalog=ml_instance,
            dataset_rid=dataset_rid,
            description=description,
        )

        # Insert the dataset-type associations *before* the version row is
        # finalised, so the version row's RMT ends up as the last write of
        # the create-time sequence. Drift detection (PR 6) uses the
        # released version row's RMT as the time anchor: with this
        # ordering, a freshly-created dataset has nothing reachable with
        # ``RMT > version_row.RMT`` and is correctly reported as not
        # dirty. (`add_dataset_types` is told not to flip the dataset to
        # dev — the version row doesn't exist yet anyway.)
        dataset.add_dataset_types(dataset_types, _skip_version_increment=True)

        Dataset._insert_dataset_versions(
            ml_instance=ml_instance,
            dataset_list=[DatasetSpec(rid=dataset_rid, version=version)],
            execution_rid=execution_rid,
            description="Initial dataset creation.",
        )
        return dataset

    def add_dataset_type(
        self,
        dataset_type: str | VocabularyTerm,
        _skip_version_increment: bool = False,
    ) -> None:
        """Add a dataset type to this dataset.

        Adds a type term to this dataset if it's not already present.
        The term must exist in the Dataset_Type vocabulary. Flips the
        dataset to a dev version to record the metadata change (per
        ADR-0003: every mutation lands on dev). If the term is already
        present, the call is a no-op and does not advance the dev
        counter.

        Args:
            dataset_type: Term name (string) or VocabularyTerm object from Dataset_Type vocabulary.
            _skip_version_increment: Internal parameter to skip the
                dev-row update when called from ``add_dataset_types`` or
                during initial dataset creation, which handle the
                version transition themselves.

        Raises:
            DerivaMLInvalidTerm: If the term doesn't exist in the Dataset_Type vocabulary.

        Example:
            >>> dataset.add_dataset_type("Training")  # doctest: +SKIP
            >>> dataset.add_dataset_type("Validation")  # doctest: +SKIP
        """
        # Convert to VocabularyTerm if needed (validates the term exists)
        if isinstance(dataset_type, VocabularyTerm):
            vocab_term = dataset_type
        else:
            vocab_term = self._ml_instance.lookup_term(MLVocab.dataset_type, dataset_type)

        # Check if already present
        if vocab_term.name in self.dataset_types:
            return

        # Insert into association table
        _, atable_path = self._get_dataset_type_association_table()
        atable_path.insert([{MLVocab.dataset_type: vocab_term.name, "Dataset": self.dataset_rid}])

        # Flip to dev to record the metadata change (unless called from add_dataset_types
        # during initial creation, which sets _skip_version_increment).
        if not _skip_version_increment:
            self._create_or_advance_dev_row(
                description=f"Added dataset type: {vocab_term.name}",
            )

    def remove_dataset_type(self, dataset_type: str | VocabularyTerm) -> None:
        """Remove a dataset type from this dataset.

        Removes a type term from this dataset if it's currently
        associated. The term must exist in the Dataset_Type vocabulary.
        Flips the dataset to a dev version on successful removal (per
        ADR-0003: every mutation lands on dev). If the term isn't
        currently associated with this dataset, the call is a no-op
        and does not advance the dev counter.

        Args:
            dataset_type: Term name (string) or VocabularyTerm object from Dataset_Type vocabulary.

        Raises:
            DerivaMLInvalidTerm: If the term doesn't exist in the Dataset_Type vocabulary.

        Example:
            >>> dataset.remove_dataset_type("Training")  # doctest: +SKIP
        """
        # Convert to VocabularyTerm if needed (validates the term exists)
        if isinstance(dataset_type, VocabularyTerm):
            vocab_term = dataset_type
        else:
            vocab_term = self._ml_instance.lookup_term(MLVocab.dataset_type, dataset_type)

        # Check if present — no-op input doesn't advance the dev counter.
        if vocab_term.name not in self.dataset_types:
            return

        # Delete from association table
        _, atable_path = self._get_dataset_type_association_table()
        atable_path.filter(
            (atable_path.Dataset == self.dataset_rid) & (atable_path.Dataset_Type == vocab_term.name)
        ).delete()
        self._create_or_advance_dev_row(
            description=f"Removed dataset type: {vocab_term.name}",
        )

    def add_dataset_types(
        self,
        dataset_types: str | VocabularyTerm | list[str | VocabularyTerm],
        _skip_version_increment: bool = False,
    ) -> None:
        """Add one or more dataset types to this dataset.

        Convenience method for adding multiple types at once. Each term
        must exist in the Dataset_Type vocabulary. Types that are
        already associated with the dataset are silently skipped.
        Flips the dataset to a dev version once after all new types are
        added (per ADR-0003: every mutation lands on dev). If every
        supplied type is already associated, the call is a no-op and
        does not advance the dev counter.

        Args:
            dataset_types: Single term or list of terms. Can be strings (term names)
                or VocabularyTerm objects.
            _skip_version_increment: Internal parameter to skip the
                dev-row update during initial dataset creation.

        Raises:
            DerivaMLInvalidTerm: If any term doesn't exist in the Dataset_Type vocabulary.

        Example:
            >>> dataset.add_dataset_types(["Training", "Image"])  # doctest: +SKIP
            >>> dataset.add_dataset_types("Testing")  # doctest: +SKIP
        """
        # Normalize input to a list
        types_to_add = [dataset_types] if not isinstance(dataset_types, list) else dataset_types

        # Track which types were actually added (not already present)
        added_types: list[str] = []
        for term in types_to_add:
            # Get term name before calling add_dataset_type
            if isinstance(term, VocabularyTerm):
                term_name = term.name
            else:
                term_name = self._ml_instance.lookup_term(MLVocab.dataset_type, term).name

            # Check if already present before adding
            if term_name not in self.dataset_types:
                self.add_dataset_type(term, _skip_version_increment=True)
                added_types.append(term_name)

        # Flip to dev once for all added types (if any were added).
        if added_types and not _skip_version_increment:
            type_names = ", ".join(added_types)
            self._create_or_advance_dev_row(
                description=f"Added dataset type(s): {type_names}",
            )

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table from the catalog schema.

        Returns:
            Table: The Deriva Table object for the Dataset table in the ML schema.
        """
        return self._ml_instance.model.schemas[self._ml_instance.ml_schema].tables["Dataset"]

    def _element_to_association_map(self) -> dict[str, str]:
        """Return ``{member_table_name: association_table_name}`` for the Dataset.

        For each ``Dataset_X`` association on the Dataset table,
        record the *other* (non-Dataset) endpoint's table name as the
        key and the association table name as the value. Used by
        :meth:`add_dataset_members` and :meth:`delete_dataset_members`
        to route a member RID to the correct ``Dataset_X`` row to
        insert / delete.

        Returns:
            ``{element_type_name -> Dataset_ElementType_name}`` map.
        """
        return {
            a.other_fkeys.pop().pk_table.name: a.table.name
            for a in self._dataset_table.find_associations()
        }

    # ==================== Read Interface Methods ====================
    # These methods implement the DatasetLike protocol for read operations.
    # They delegate to the catalog instance for actual data retrieval.
    # This allows Dataset and DatasetBag to share a common interface.

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of elements that can be contained in this dataset.

        Returns:
            Iterable of Table objects representing element types.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> for t in dataset.list_dataset_element_types():  # doctest: +SKIP
            ...     print(t.name)  # doctest: +SKIP
        """
        return self._ml_instance.list_dataset_element_types()

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features associated with a table.

        Args:
            table: Table to find features for.

        Returns:
            Iterable of Feature objects.

        Raises:
            DerivaMLTableNotFound: If ``table`` does not exist in the schema.

        Example:
            >>> features = list(dataset.find_features("Image"))  # doctest: +SKIP
        """
        return self._ml_instance.find_features(table)

    def list_members(self, table: str | Table) -> list[str]:
        """Return the RIDs of dataset members belonging to the given table.

        Convenience wrapper around :meth:`list_dataset_members` that returns
        a flat list of RID strings for a single table rather than the full
        ``dict[table_name, list[record]]`` mapping.

        Args:
            table: Table name (str) or Table object whose member RIDs to return.

        Returns:
            List of RID strings for members of this dataset that belong to
            ``table``. Returns an empty list if no members of that type exist.

        Example:
            >>> image_rids = dataset.list_members("Image")  # doctest: +SKIP
            >>> print(f"{len(image_rids)} images in dataset")  # doctest: +SKIP
        """
        table_name = table if isinstance(table, str) else table.name
        members = self.list_dataset_members()
        return [r["RID"] for r in members.get(table_name, [])]

    def feature_values(
        self,
        table: str | Table,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Dataset-scoped feature values — identical signature to DerivaML.feature_values.

        Yields only records whose target RID is a member of this dataset.
        Filtering is applied to the raw feature table query before selector
        reduction — a target RID outside the dataset's member set is never
        presented to the selector.

        See :meth:`deriva_ml.core.mixins.feature.FeatureMixin.feature_values`
        for the full contract (return type, selector semantics, exceptions).

        Args:
            table: Target table the feature is defined on (name or Table).
            feature_name: Name of the feature to read.
            selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
                used to reduce multi-value groups. See ``FeatureRecord`` for built-ins.
                Return ``None`` from a selector to omit that target RID.
            materialize_limit: Optional cap on the upstream catalog
                query's row materialization. Forwarded to
                ``DerivaML.feature_values``; raises
                ``DerivaMLMaterializeLimitExceeded`` if exceeded.
                Default ``None`` preserves unbounded behavior.
            execution_rids: Optional filter forwarded to the upstream
                catalog query. When set, only feature rows whose
                ``Execution`` value is in this list are materialized.
                Empty list short-circuits to an empty result.

        Returns:
            Iterator of ``FeatureRecord`` — filtered to dataset members, then
            reduced by selector if provided.

        Raises:
            DerivaMLTableNotFound: ``table`` does not exist.
            DerivaMLException: ``feature_name`` is not a feature on ``table``.
            DerivaMLMaterializeLimitExceeded: If the upstream
                materialization exceeds ``materialize_limit``.

        Example:
            >>> from deriva_ml.feature import FeatureRecord  # doctest: +SKIP
            >>> records = list(dataset.feature_values(  # doctest: +SKIP
            ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
            ... ))
        """
        members = set(self.list_members(table))
        target_col = table if isinstance(table, str) else table.name

        # Filter upstream raw records to dataset members. Forward
        # materialize_limit and execution_rids to the catalog query so
        # the upstream materialization is bounded too. The dataset-scope
        # filter is applied AFTER the catalog query, so the limit check
        # in the upstream guards us against memory blow-up before we
        # filter further.
        raw_in_scope = [
            rec
            for rec in self._ml_instance.feature_values(
                table,
                feature_name,
                selector=None,
                materialize_limit=materialize_limit,
                execution_rids=execution_rids,
            )
            if getattr(rec, target_col, None) in members
        ]

        if selector is None:
            yield from raw_in_scope
            return

        grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
        for rec in raw_in_scope:
            target_rid = getattr(rec, target_col, None)
            if target_rid is not None:
                grouped[target_rid].append(rec)

        for group in grouped.values():
            chosen = selector(group)
            if chosen is not None:
                yield chosen

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """Look up a Feature definition — delegates to the owning DerivaML.

        Identical signature and return to ``DerivaML.lookup_feature``. Provided
        for API symmetry so dataset-scoped code does not need to reach back
        through ``self._ml_instance``.

        Args:
            table: The table the feature is defined on (name or Table object).
            feature_name: Name of the feature to look up.

        Returns:
            A Feature schema descriptor.

        Raises:
            DerivaMLException: If the feature doesn't exist on the specified table.

        Example:
            >>> feat = dataset.lookup_feature("Image", "Glaucoma")  # doctest: +SKIP
            >>> RecordClass = feat.feature_record_class()  # doctest: +SKIP
        """
        return self._ml_instance.lookup_feature(table, feature_name)

    def list_workflow_executions(self, workflow: str) -> list[str]:
        """Dataset-scoped list_workflow_executions — see DerivaML.list_workflow_executions.

        Current implementation returns the full workflow execution list from the
        catalog. Target-RID filtering at selection time (via ``feature_values``)
        ensures that records from executions outside the dataset's member set
        are excluded. A stricter scope (executions whose outputs touch dataset
        members) is a performance optimization deferred to a later change.

        Args:
            workflow: Workflow RID or Workflow_Type name. See
                ``DerivaML.list_workflow_executions`` for the resolution rules.

        Returns:
            List of execution RIDs. May be empty.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> rids = dataset.list_workflow_executions("Glaucoma_Training_v2")  # doctest: +SKIP
            >>> print(f"{len(rids)} training runs in catalog")  # doctest: +SKIP
        """
        return self._ml_instance.list_workflow_executions(workflow)

    def dataset_history(self) -> list[DatasetHistory]:
        """List every ``Dataset_Version`` row for this dataset, oldest first.

        Returns one entry per row, including the dev row when one exists.
        Released and dev rows are not separated — they are different states
        of the same row type, and filtering belongs at the call site.
        Callers who want released-only filter on the typed property::

            released = [
                h for h in ds.dataset_history()
                if not h.dataset_version.is_devrelease
            ]

        Results are sorted by ``dataset_version`` in ascending PEP 440
        order. With the dev-versioning model, that reads
        chronologically: ``[0.1.0, ..., 0.4.0, 0.4.0.post1.devN]``.

        Returns:
            A list of :class:`DatasetHistory` entries. Each carries the
            parsed version, the version row's RID, the dataset's RID,
            the description and execution link, and the snapshot ID
            (``None`` for dev rows).

        Raises:
            DerivaMLException: If this dataset's RID is not a valid
                dataset RID.

        Example:
            >>> history = dataset.dataset_history()  # doctest: +SKIP
            >>> for entry in history:  # doctest: +SKIP
            ...     mark = " (dev)" if entry.dataset_version.is_devrelease else ""
            ...     print(f"v{entry.dataset_version}{mark}: {entry.description}")
        """
        if not self._ml_instance.model.is_dataset_rid(self.dataset_rid):
            raise DerivaMLException(f"RID is not for a data set: {self.dataset_rid}")
        version_path = self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Version"]
        entries = [
            DatasetHistory(
                dataset_version=DatasetVersion.parse(v["Version"]),
                minid=v["Minid"],
                spec_hash=v.get("Minid_Spec_Hash"),
                snapshot=v["Snapshot"],
                dataset_rid=self.dataset_rid,
                version_rid=v["RID"],
                description=v["Description"],
                execution_rid=v["Execution"],
            )
            for v in version_path.filter(version_path.Dataset == self.dataset_rid).entities().fetch()
        ]
        entries.sort(key=lambda h: h.dataset_version)
        return entries

    @property
    @validate_call(config=VALIDATION_CONFIG)
    def current_version(self) -> DatasetVersion:
        """Return the dataset's current version label.

        The current version is the highest entry in
        :meth:`dataset_history` under PEP 440 ordering. When a dev row
        exists, the dev label sorts after every released label and is
        returned. Otherwise the latest released label is returned.

        Released versions pin a catalog snapshot — reading the dataset
        at a released version always returns the same content. Dev
        versions track live catalog state — the row's ``Snapshot`` is
        ``NULL`` and reads resolve against the current catalog. Callers
        who want a stable reference should use a released version, not
        a dev version.

        Returns:
            The most recent ``DatasetVersion``, dev or released.

        Raises:
            DerivaMLException: If the dataset has no ``Dataset_Version``
                rows. ``create_dataset`` always inserts an initial
                released row, so an empty history indicates a catalog
                inconsistency rather than a normal state.

        Example:
            >>> v = dataset.current_version  # doctest: +SKIP
            >>> if v.is_devrelease:  # doctest: +SKIP
            ...     print(f"in dev period at {v}; call release() to mint a release")
            ... else:  # doctest: +SKIP
            ...     print(f"at released {v}")
        """
        history = self.dataset_history()
        if not history:
            raise DerivaMLException(
                f"Dataset {self.dataset_rid} has no Dataset_Version rows. "
                "Every dataset is initialised with an initial released row "
                "at creation time; an empty history indicates a catalog "
                "inconsistency."
            )
        return max(h.dataset_version for h in history)

    def get_chaise_url(self) -> str:
        """Get the Chaise URL for viewing this dataset in the browser.

        Returns:
            URL string for the dataset record in Chaise.
        """
        return (
            f"https://{self._ml_instance.host_name}/chaise/record/"
            f"#{self._ml_instance.catalog_id}/deriva-ml:Dataset/RID={self.dataset_rid}"
        )

    def to_markdown(self, show_children: bool = False, indent: int = 0) -> str:
        """Generate a markdown representation of this dataset.

        Returns a formatted markdown string with a link to the dataset,
        version, types, and description. Optionally includes nested children.

        Args:
            show_children: If True, include direct child datasets.
            indent: Number of indent levels (each level is 2 spaces).

        Returns:
            Markdown-formatted string.

        Example:
            >>> ds = ml.lookup_dataset("4HM")  # doctest: +SKIP
            >>> print(ds.to_markdown())  # doctest: +SKIP
        """
        prefix = "  " * indent
        version = str(self.current_version) if self.current_version else "n/a"
        types = ", ".join(self.dataset_types) if self.dataset_types else ""
        desc = self.description or ""

        line = f"{prefix}- [{self.dataset_rid}]({self.get_chaise_url()}) v{version}"
        if types:
            line += f" [{types}]"
        if desc:
            line += f": {desc}"

        lines = [line]

        if show_children:
            children = self.list_dataset_children(recurse=False)
            for child in children:
                lines.append(child.to_markdown(show_children=False, indent=indent + 1))

        return "\n".join(lines)

    def display_markdown(self, show_children: bool = False, indent: int = 0) -> None:
        """Display a formatted markdown representation of this dataset in Jupyter.

        Convenience method that calls to_markdown() and displays the result
        using IPython.display.Markdown.

        Args:
            show_children: If True, include direct child datasets.
            indent: Number of indent levels (each level is 2 spaces).

        Example:
            >>> ds = ml.lookup_dataset("4HM")  # doctest: +SKIP
            >>> ds.display_markdown(show_children=True)  # doctest: +SKIP
        """
        from IPython.display import Markdown, display

        display(Markdown(self.to_markdown(show_children, indent)))

    def _build_dataset_graph(self) -> Iterable[Dataset]:
        """Build a dependency graph of all related datasets and return in topological order.

        This method is used when incrementing dataset versions. Because datasets can be
        nested (parent-child relationships), changing the version of one dataset may
        require updating related datasets.

        The topological sort ensures that children are processed before parents,
        so version updates propagate correctly through the hierarchy.

        Returns:
            Iterable[Dataset]: Datasets in topological order (children before parents).

        Example:
            If dataset A contains nested dataset B, which contains C:
            A -> B -> C
            The returned order would be [C, B, A], ensuring C's version is
            updated before B's, and B's before A's.
        """
        ts: TopologicalSorter = TopologicalSorter()
        self._build_dataset_graph_1(ts, set())
        return ts.static_order()

    def _build_dataset_graph_1(self, ts: TopologicalSorter, visited: set[str]) -> None:
        """Recursively build the dataset dependency graph.

        Uses topological sort where parents depend on their children, ensuring
        children are processed before parents in the resulting order.

        Args:
            ts: TopologicalSorter instance to add nodes and dependencies to.
            visited: Set of already-visited dataset RIDs to avoid cycles.
        """
        if self.dataset_rid in visited:
            return

        visited.add(self.dataset_rid)
        # Use current catalog state for graph traversal, not version snapshot.
        # Parent/child relationships need to reflect current state for version updates.
        children = self.list_dataset_children(version=None)
        parents = self.list_dataset_parents(version=None)

        # Add this node with its children as dependencies.
        # This means: self depends on children, so children will be ordered before self.
        ts.add(self, *children)

        # Recursively process children
        for child in children:
            child._build_dataset_graph_1(ts, visited)

        # Recursively process parents (they will depend on this node)
        for parent in parents:
            parent._build_dataset_graph_1(ts, visited)

    @validate_call(config=VALIDATION_CONFIG)
    def _increment_dataset_version(
        self,
        component: VersionPart,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> DatasetVersion:
        """Force a release-version bump on this dataset and its related datasets.

        Internal helper preserved from the pre-dev-versioning model.
        Walks the dataset graph (parents/children) and inserts a new
        released ``Dataset_Version`` row for each, with the supplied
        component bumped and a stamped catalog snapshot. Bypasses the
        dev-versioning model entirely — there's no precondition that
        the dataset be in a dev period.

        Used internally by the catalog-clone path, where dataset
        versions need to be reinitialised after a catalog snapshot is
        copied. Not for user-facing release work; use :meth:`release`
        for that.

        Args:
            component: Which release-segment part to bump.
            description: Optional description for the new version row.
            execution_rid: Optional RID of the calling execution.

        Returns:
            The new ``DatasetVersion`` for this dataset (related
            datasets are also bumped but their versions are not in the
            return).

        Raises:
            DerivaMLException: If the catalog write fails.
        """

        # Find all the datasets that are reachable from this dataset and determine their new version numbers.
        related_datasets = list(self._build_dataset_graph())
        version_update_list = [
            DatasetSpec(
                rid=ds.dataset_rid,
                version=ds.current_version.next_release(component),
            )
            for ds in related_datasets
        ]
        Dataset._insert_dataset_versions(
            self._ml_instance, version_update_list, description=description, execution_rid=execution_rid
        )
        return next((d.version for d in version_update_list if d.rid == self.dataset_rid))

    def release(
        self,
        bump: VersionPart,
        description: str,
        execution: "Execution | None" = None,
    ) -> DatasetVersion:
        """Promote this dataset's dev period to a released version.

        Per ADR-0003, ``release`` is the only operation that produces
        a released ``Dataset_Version`` row. It promotes the existing
        dev row in place: rewrites ``Version`` to the released label,
        stamps ``Snapshot`` with the catalog snapshot at release time,
        replaces ``Description`` with release notes, and overwrites
        the row's ``Execution`` link with the supplied execution (or
        ``NULL`` if none).

        Concurrency: the promotion uses a conditional update on the
        dev row's observed ``RMT``. If a competing writer landed
        between this method's read and write — another mutation, or
        another concurrent ``release`` — the call raises
        :class:`DerivaMLException` with a clear retry message rather
        than silently overwriting the other writer's work.

        Args:
            bump: Which release-segment part to advance from the
                last released version. ``VersionPart.minor`` is the
                common case; ``VersionPart.major`` for
                schema-breaking changes; ``VersionPart.patch`` for
                small clean-ups.
            description: Release notes. **Replaces** the dev row's
                accumulated description, not appended.
            execution: Optional ``Execution`` that called release.
                Stored on the released row's ``Execution`` link.
                Mutator authorship during the dev period is not
                captured here — it's recoverable from the catalog's
                audit trail (``RMT`` on changed rows + per-row
                provenance).

        Returns:
            The new released ``DatasetVersion``.

        Raises:
            DerivaMLException: If this dataset has no dev row to
                promote, or if a concurrent writer modified the dev
                row between this call's read and write.

        Example:
            >>> dataset.add_dataset_members(["1-abc", "1-def"])  # doctest: +SKIP
            >>> dataset.add_dataset_members(["1-ghi"])  # doctest: +SKIP
            >>> # Now at e.g. 0.4.0.post1.dev2; cut a real release:
            >>> v = dataset.release(  # doctest: +SKIP
            ...     bump=VersionPart.minor,
            ...     description="Added 3 new samples for v0.5.0",
            ... )
            >>> print(v)  # doctest: +SKIP
            0.5.0
        """
        history = self.dataset_history()
        dev_entries = [h for h in history if h.dataset_version.is_devrelease]
        if not dev_entries:
            raise DerivaMLException(
                f"Dataset {self.dataset_rid} has no dev period to release "
                f"(current_version={self.current_version}). To release a "
                "no-op change, call mark_dev() first to declare a dev "
                "period, then release() to promote it."
            )
        if len(dev_entries) > 1:
            raise DerivaMLException(
                f"Dataset {self.dataset_rid} has {len(dev_entries)} dev rows; expected at most one."
            )
        dev_row = dev_entries[0]

        # Find the last released version to anchor the next-release bump.
        released_entries = [h for h in history if not h.dataset_version.is_devrelease]
        if not released_entries:
            # Defensive: every dataset has at least one released row at
            # creation time. If this fires, the catalog is in an
            # inconsistent state.
            raise DerivaMLException(
                f"Dataset {self.dataset_rid} is in a dev period but has "
                "no released version to anchor the next release against."
            )
        last_released = max(h.dataset_version for h in released_entries)
        next_label = str(last_released.next_release(bump))

        # Re-fetch the dev row to get its current RMT for the conditional update.
        schema_path = self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema]
        version_table = schema_path.tables["Dataset_Version"]
        current_rows = list(version_table.filter(version_table.RID == dev_row.version_rid).entities().fetch())
        if not current_rows:
            raise DerivaMLException(
                f"Dev version row {dev_row.version_rid} disappeared between read and update — concurrent deletion?"
            )
        observed_rmt = current_rows[0]["RMT"]

        # Stamp the catalog snapshot at release time.
        snapshot = self._ml_instance.catalog.get("/").json()["snaptime"]

        execution_rid = execution.execution_rid if execution is not None else None
        updated = list(
            version_table.update(
                [
                    {
                        "RID": dev_row.version_rid,
                        "RMT": observed_rmt,
                        "Version": next_label,
                        "Snapshot": snapshot,
                        "Description": description,
                        "Execution": execution_rid,
                    }
                ],
                correlation={"RID", "RMT"},
            )
        )
        if not updated:
            raise DerivaMLException(
                f"Concurrent modification of dev row {dev_row.version_rid} "
                f"for dataset {self.dataset_rid}: another writer advanced "
                "the row between this call's read and write. Re-read the "
                "dataset and retry if the new state is still what you "
                "intended."
            )

        return DatasetVersion.parse(next_label)

    def is_dirty(self) -> bool:
        """Return ``True`` if catalog drift has occurred since the last released version.

        A dataset's contents include not just its members but everything those
        members reference — feature values, asset metadata, classifications,
        and any other rows reachable from the dataset via the catalog's foreign
        keys. When any of those rows change, the bag this dataset would
        download today differs from the bag at the last released version, even
        if the dataset's member list hasn't been touched.

        Mechanism: walks the same foreign-key paths used to build the dataset's
        bag, short-circuiting on the first table that has any row with
        ``RMT`` greater than the last released version's row-modified time.

        Limitations:
            Deletions are not detected. A row that was reachable at the last
            release but has since been deleted will not flip ``is_dirty`` to
            ``True``. Users who delete catalog rows that affect a dataset
            should call ``mark_dev`` manually to record the dirty state.

        Returns:
            ``True`` if any reachable row has been modified or added since
            the last released version's snapshot. ``False`` if no drift is
            detected.

        Raises:
            DerivaMLException: If the dataset has no released version
                (every dataset is created with an initial release row, so
                this indicates a catalog inconsistency).

        Example:
            >>> if dataset.is_dirty():  # doctest: +SKIP
            ...     print("Drift since last release; consider mark_dev/release")
        """
        for _table_name, count in self._iter_drift_counts(
            *self._release_diff_bounds(),
            short_circuit=True,
        ):
            if count > 0:
                return True
        return False

    def release_diff(self) -> dict[str, int]:
        """Show which catalog changes since the last release would alter this dataset's contents.

        A dataset's contents include not just its members but everything those
        members reference — feature values, asset metadata, classifications,
        and any other rows reachable from the dataset via the catalog's foreign
        keys. When any of those rows change, the bag this dataset would
        download today differs from the bag at the last released version, even
        if the dataset's member list hasn't been touched.

        This method reports those changes, grouped by the table the changed
        rows live in. Use it as a follow-up to ``is_dirty()`` returning True,
        to see *where* the drift is.

        Mechanism: walks the same foreign-key paths used to build the dataset's
        bag, counting rows in each reachable table whose row-modified time
        (``RMT``) is later than the last released version's row-modified time.
        The walk is bounded by what's reachable from this dataset — changes
        elsewhere in the catalog that don't affect this dataset are not
        counted.

        Implemented as a thin wrapper around
        :meth:`compare_versions` with the last released version as the
        lower bound and the current version as the upper bound.

        Limitations:
            Deletions are not detected. A row that was reachable at the last
            release but has since been deleted will not appear in the result.
            Users who delete catalog rows that affect a dataset should call
            ``mark_dev`` manually to record the dirty state.

        Returns:
            Mapping of fully-qualified table name → number of changed rows in
            that table. Tables with zero changes are omitted; an empty dict
            means no drift is detected.

        Raises:
            DerivaMLException: If the dataset has no released version.

        Example:
            >>> # Find out what's changed and decide whether to release
            >>> if dataset.is_dirty():  # doctest: +SKIP
            ...     for table, count in dataset.release_diff().items():
            ...         print(f"{table}: {count} rows changed")
        """
        result: dict[str, int] = {}
        for table_name, count in self._iter_drift_counts(
            *self._release_diff_bounds(),
            short_circuit=False,
        ):
            if count > 0:
                # If multiple FK paths reach the same table, sum their counts.
                result[table_name] = result.get(table_name, 0) + count
        return result

    def compare_versions(
        self,
        v_a: DatasetVersion | str,
        v_b: DatasetVersion | str,
    ) -> dict[str, int]:
        """Show catalog rows that changed between two versions of this dataset.

        Walks the same foreign-key paths used to build the dataset's bag,
        counting rows in each reachable table whose row-modified time
        (``RMT``) falls between the two versions' bounds. Order of *v_a*
        and *v_b* doesn't matter — the predicate uses
        ``min(t_a, t_b) < RMT <= max(t_a, t_b)``.

        Each argument may independently be a released label (resolves to
        that version row's ``RMT`` as the time bound) or the current dev
        label (resolves to "now"). Stale or non-current dev labels error
        per ADR-0003's addressability rule.

        Args:
            v_a: A version label, released or the current dev.
            v_b: A version label, released or the current dev.

        Returns:
            Mapping of fully-qualified table name → number of changed rows
            in that table between the two endpoints. Empty dict if both
            endpoints resolve to the same time (e.g., both are the same
            version).

        Raises:
            DerivaMLException: If either argument doesn't resolve to a
                version of this dataset, or if a dev label doesn't match
                the current dev row.

        Example:
            >>> # Diff between two historical released versions
            >>> changes = dataset.compare_versions("0.3.0", "0.5.0")  # doctest: +SKIP
            >>> for table, count in changes.items():  # doctest: +SKIP
            ...     print(f"{table}: {count} rows added/changed")
        """
        t_a = self._resolve_version_to_rmt(v_a)
        t_b = self._resolve_version_to_rmt(v_b)
        # Symmetric in argument order: lower bound is min, upper is max.
        # `None` means "live" — when present, it's always the upper bound.
        if t_a is None and t_b is None:
            # Both are the current dev row → no time window, no drift.
            return {}
        if t_a is None:
            t_lower, t_upper = t_b, None
        elif t_b is None:
            t_lower, t_upper = t_a, None
        else:
            t_lower, t_upper = (t_a, t_b) if t_a <= t_b else (t_b, t_a)

        if t_lower == t_upper:
            return {}

        result: dict[str, int] = {}
        for table_name, count in self._iter_drift_counts(t_lower, t_upper, short_circuit=False):
            if count > 0:
                result[table_name] = result.get(table_name, 0) + count
        return result

    def _release_diff_bounds(self) -> tuple[str, str | None]:
        """Return ``(lower_rmt, upper_rmt)`` bounds for ``release_diff`` / ``is_dirty``.

        Lower bound is the last released version's ``RMT``. Upper bound
        is ``None`` (live — no upper bound). Whether or not a dev row
        exists doesn't affect these bounds: drift is "everything
        changed since the last release," and the dev row itself is one
        of those changes.
        """
        history = self.dataset_history()
        released = [h for h in history if not h.dataset_version.is_devrelease]
        if not released:
            raise DerivaMLException(
                f"Dataset {self.dataset_rid} has no released version to anchor drift detection against."
            )
        last_released = max(released, key=lambda h: h.dataset_version)
        return self._fetch_version_row_rmt(last_released.version_rid), None

    def _resolve_version_to_rmt(self, version: DatasetVersion | str) -> str | None:
        """Resolve a version label to a ``Dataset_Version.RMT`` time anchor.

        Returns the version row's ``RMT`` for a released label, or
        ``None`` for the current dev label (meaning "live"). Stale or
        non-current dev labels raise.
        """
        version_str = str(version)
        history = self.dataset_history()
        try:
            entry = next(h for h in history if str(h.dataset_version) == version_str)
        except StopIteration:
            available = [str(h.dataset_version) for h in history]
            raise DerivaMLException(
                f"Version {version_str!r} does not exist for dataset {self.dataset_rid}. Available: {available}"
            )
        if entry.dataset_version.is_devrelease:
            # Dev versions resolve only when they match the current dev
            # row. By construction `dataset_history` only returns one dev
            # row (the current one), so this is the matching case.
            return None
        return self._fetch_version_row_rmt(entry.version_rid)

    def _fetch_version_row_rmt(self, version_rid: RID) -> str:
        """Fetch the ``RMT`` for a ``Dataset_Version`` row by RID."""
        version_table = self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Version"]
        rows = list(version_table.filter(version_table.RID == version_rid).entities().fetch())
        if not rows:
            raise DerivaMLException(f"Dataset_Version row {version_rid} not found while resolving time anchor.")
        return rows[0]["RMT"]

    def _iter_drift_counts(
        self,
        t_lower: str,
        t_upper: str | None,
        short_circuit: bool,
    ) -> "Iterator[tuple[str, int]]":
        """Yield ``(table_name, count)`` for rows reachable from this dataset with ``t_lower < RMT <= t_upper``.

        Walks the dataset's FK paths via
        :meth:`DatasetBagBuilder.aggregate_queries` and counts rows
        in each terminal table that fall within the given RMT window.
        ``t_upper=None`` means "no upper bound" — i.e., live state.

        If *short_circuit* is True, yields the first non-zero count and
        stops. Otherwise yields one ``(table_name, count)`` per (table,
        path) reached.
        """
        from deriva.core.datapath import Cnt

        builder = DatasetBagBuilder(ml_instance=self._ml_instance)
        table_queries = builder.aggregate_queries(self)

        for table_name, path_entries in table_queries.items():
            for dp, target_pb_table, _is_asset in path_entries:
                # Build the RMT filter: t_lower < RMT <= t_upper
                rmt_col = target_pb_table.RMT
                rmt_filter = rmt_col > t_lower
                if t_upper is not None:
                    rmt_filter = rmt_filter & (rmt_col <= t_upper)
                filtered = dp.filter(rmt_filter)
                try:
                    rows = list(filtered.aggregates(Cnt(target_pb_table.RID).alias("n")).fetch())
                    count = rows[0]["n"] if rows else 0
                except Exception as exc:
                    self._logger.debug(
                        "drift count query failed for table %s: %s",
                        table_name,
                        exc,
                    )
                    count = 0
                yield table_name, count
                if short_circuit and count > 0:
                    return

    def _create_or_advance_dev_row(
        self,
        description: str,
        execution_rid: RID | None = None,
    ) -> None:
        """Create or advance this dataset's dev row.

        Internal primitive that backs :meth:`mark_dev` and (in a later
        PR) the member-mutation operations. Implements the dev-row
        lifecycle from ADR-0003:

        * If the dataset has no dev row, INSERT one at
          ``<last_released>.post1.dev1`` with ``Snapshot=NULL``,
          ``Description`` set to *description*, and ``Execution`` set to
          *execution_rid* (or ``NULL`` if not supplied). Update the
          dataset's ``Version`` FK to point at the new row.
        * If the dataset has a dev row, UPDATE that row in place:
          advance ``.devN`` by 1, replace ``Description``, overwrite
          ``Execution``. Use a conditional update on the row's observed
          ``RMT`` so concurrent writers can be detected.

        Concurrency: a competing writer that landed between this
        method's read and write is detected by the ``RMT`` predicate
        on the UPDATE — if zero rows match, the call raises
        :class:`DerivaMLException` rather than silently overwriting
        the other writer's work.

        Args:
            description: Description of the change being recorded.
                Replaces any existing dev-row description (not
                appended — the catalog's audit log preserves prior
                values).
            execution_rid: Optional RID of the calling execution.
                Stored on the dev row's ``Execution`` column.

        Raises:
            DerivaMLException: If a concurrent writer modified the dev
                row between this call's read and write.
        """
        history = self.dataset_history()
        dev_entries = [h for h in history if h.dataset_version.is_devrelease]
        schema_path = self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema]
        version_table = schema_path.tables["Dataset_Version"]

        if not dev_entries:
            # No dev row yet: anchor a new one at the latest release.
            released = max(
                (h.dataset_version for h in history if not h.dataset_version.is_devrelease),
                default=None,
            )
            if released is None:
                # Defensive: every dataset has at least one released row at
                # creation time. If this fires, the catalog is in an
                # inconsistent state that this method cannot fix.
                raise DerivaMLException(
                    f"Dataset {self.dataset_rid} has no released version to anchor a dev row against."
                )
            new_label = f"{released}.post1.dev1"
            inserted = list(
                version_table.insert(
                    [
                        {
                            "Dataset": self.dataset_rid,
                            "Version": new_label,
                            "Description": description,
                            "Execution": execution_rid,
                        }
                    ]
                )
            )
            new_rid = inserted[0]["RID"]
            schema_path.tables["Dataset"].update([{"RID": self.dataset_rid, "Version": new_rid}])
            return

        # Dev row exists: advance .devN, replace description, overwrite
        # execution. Use a conditional update keyed on the row's observed
        # RMT so a concurrent writer can be detected.
        if len(dev_entries) > 1:
            # Should be impossible — at most one dev row per dev period.
            raise DerivaMLException(
                f"Dataset {self.dataset_rid} has {len(dev_entries)} dev rows; expected at most one."
            )
        current_dev = dev_entries[0]
        # Re-fetch the row by RID to capture its RMT for the conditional
        # update. dataset_history doesn't expose RMT.
        current_rows = list(version_table.filter(version_table.RID == current_dev.version_rid).entities().fetch())
        if not current_rows:
            raise DerivaMLException(
                f"Dev version row {current_dev.version_rid} disappeared between read and update — concurrent deletion?"
            )
        observed_rmt = current_rows[0]["RMT"]
        next_n = current_dev.dataset_version.dev + 1
        next_label = (
            f"{current_dev.dataset_version.major}."
            f"{current_dev.dataset_version.minor}."
            f"{current_dev.dataset_version.micro}"
            f".post{current_dev.dataset_version.post}.dev{next_n}"
        )
        updated = list(
            version_table.update(
                [
                    {
                        "RID": current_dev.version_rid,
                        "RMT": observed_rmt,
                        "Version": next_label,
                        "Description": description,
                        "Execution": execution_rid,
                    }
                ],
                correlation={"RID", "RMT"},
            )
        )
        if not updated:
            raise DerivaMLException(
                f"Concurrent modification of dev row {current_dev.version_rid} "
                "for dataset {self.dataset_rid}: another writer advanced the "
                "row between this call's read and write. Re-read the dataset "
                "and retry if the new state is still what you intended."
            )

    def mark_dev(
        self,
        description: str,
        execution: "Execution | None" = None,
    ) -> None:
        """Flip this dataset to a dev version, recording catalog drift.

        Use ``mark_dev`` when the catalog has changed in a way that
        affects this dataset's contents, but no dataset-API operation
        flipped the dataset to dev automatically. Typical case: a
        feature value was added by a separate execution against a row
        that's a member of this dataset — the dataset's own row and
        member list are untouched, but the bag the dataset would
        download today differs from the last release.

        On the first call after a release, a new dev row is created at
        ``<last_release>.post1.dev1`` with ``Snapshot=NULL``. Subsequent
        calls during the same dev period advance the ``.devN`` counter
        (the dev row is mutable; one row per dev period). The
        ``Description`` is replaced on each call — prior values are
        recoverable from the catalog's audit log.

        ``mark_dev`` returns ``None`` rather than the new dev label
        because dev labels are not addressable across time: by the
        time a caller might use a returned label for anything, the
        next mutation has advanced ``.devN`` and the returned value no
        longer resolves. Callers who want to display the new label
        read :attr:`current_version` after the call.

        Args:
            description: What changed. Replaces any existing dev-row
                description.
            execution: Optional execution that observed the drift.
                Stored on the dev row's ``Execution`` column.

        Raises:
            DerivaMLException: If a concurrent writer modified the dev
                row between this call's read and write.

        Example:
            >>> # A separate execution recorded labels for some images
            >>> # that are members of this dataset. Flag the drift:
            >>> dataset.mark_dev("Picked up classifier output for the test split")  # doctest: +SKIP
            >>> str(dataset.current_version)  # doctest: +SKIP
            '0.4.0.post1.dev1'
        """
        execution_rid = execution.execution_rid if execution is not None else None
        self._create_or_advance_dev_row(
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call(config=VALIDATION_CONFIG)
    def list_dataset_members(
        self,
        recurse: bool = False,
        limit: int | None = None,
        _visited: set[RID] | None = None,
        version: DatasetVersion | str | None = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """Lists members of a dataset.

        Returns a dictionary mapping member types to lists of member records. Can optionally
        recurse through nested datasets and limit the number of results.

        Args:
            recurse: Whether to include members of nested datasets. Defaults to False.
            limit: Maximum number of members to return per type. None for no limit.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list members from. Defaults to the current version.
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            dict[str, list[dict[str, Any]]]: Dictionary mapping member types to lists of members.
                Each member is a dictionary containing the record's attributes.

        Raises:
            DerivaMLException: If dataset_rid is invalid.

        Example:
            >>> members = ml.list_dataset_members("1-abc123", recurse=True)  # doctest: +SKIP
            >>> for type_name, records in members.items():  # doctest: +SKIP
            ...     print(f"{type_name}: {len(records)} records")
        """
        # Initialize visited set for recursion guard
        if _visited is None:
            _visited = set()

        # Prevent infinite recursion by checking if we've already visited this dataset
        if self.dataset_rid in _visited:
            return {}
        _visited.add(self.dataset_rid)

        # Look at each of the element types that might be in the dataset_table and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        pb = version_snapshot_catalog.pathBuilder()
        for assoc_table in self._dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            # Look at domain tables and nested datasets.
            if not self._ml_instance.model.is_domain_schema(target_table.schema.name) and not (
                target_table == self._dataset_table or target_table.name == "File"
            ):
                continue
            member_column = (
                "Nested_Dataset" if target_table == self._dataset_table else other_fkey.foreign_key_columns[0].name
            )
            # Use the actual referenced column from the FK definition, not always "RID".
            # e.g. isa:Dataset_file.file -> isa:file.id (integer), not RID.
            target_column = other_fkey.referenced_columns[0].name

            target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
            member_path = pb.schemas[member_table.schema.name].tables[member_table.name]

            path = member_path.filter(member_path.Dataset == self.dataset_rid).link(
                target_path,
                on=(member_path.columns[member_column] == target_path.columns[target_column]),
            )
            target_entities = list(path.entities().fetch(limit=limit) if limit else path.entities().fetch())
            members[target_table.name].extend(target_entities)
            if recurse and target_table == self._dataset_table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds_rid in nested_datasets:
                    # Nested datasets live on their own version timeline:
                    # the outer's 1.2.3 does NOT map to the inner's 1.2.3 in
                    # general. We already resolved the outer version to a
                    # catalog snapshot above (``version_snapshot_catalog``),
                    # and ``lookup_dataset`` returns a Dataset bound to THAT
                    # snapshot. Passing ``version=None`` on the recursive
                    # call means "use the ml instance you're already bound
                    # to" — i.e. the snapshot — instead of trying to look
                    # up the outer's version string in the inner's history,
                    # which would raise when the strings don't line up.
                    ds = version_snapshot_catalog.lookup_dataset(ds_rid)
                    for k, v in ds.list_dataset_members(version=None, recurse=recurse, _visited=_visited).items():
                        members[k].extend(v)
        return dict(members)

    def get_denormalized_as_dataframe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
        version: DatasetVersion | str | None = None,
    ) -> pd.DataFrame:
        """Return the dataset as a denormalized wide table (DataFrame).

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.as_dataframe`.
        See the ``Denormalizer`` class docstring for the full semantic
        rules (Rules 1-8) and cardinality details.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).
            ignore_unrelated_anchors: If True, silently drop anchors
                with no FK path (Rule 8).
            version: Optional dataset version. When given, queries run
                against the corresponding catalog snapshot for
                reproducibility (same semantics as
                :meth:`list_dataset_members`'s ``version`` kwarg). When
                None, uses whatever catalog binding the underlying
                DerivaML instance was constructed with (live or a
                previously-pinned snapshot).

        Returns:
            A :class:`pandas.DataFrame` with one row per ``row_per``
            instance in scope. Columns use ``Table.column`` notation.

        Example::

            dataset = ml.lookup_dataset("28CT")
            df = dataset.get_denormalized_as_dataframe(["Image", "Subject"])
            # Pinned to a specific version for reproducibility:
            df = dataset.get_denormalized_as_dataframe(
                ["Image", "Subject"], version="1.0.0"
            )
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self, version=version).as_dataframe(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )

    def get_denormalized_as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
        version: DatasetVersion | str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the denormalized dataset rows as dicts.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.as_dict`.
        Same rules and exceptions as
        :meth:`get_denormalized_as_dataframe` but yields one dict per
        row. Use this for large datasets where a full DataFrame won't
        fit in memory.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).
            ignore_unrelated_anchors: If True, silently drop anchors
                with no FK path (Rule 8).
            version: Optional dataset version (snapshot-bound queries).
                Same semantics as
                :meth:`get_denormalized_as_dataframe`'s ``version``.

        Yields:
            ``dict[str, Any]`` per row — keys are ``Table.column``
            labels, values are raw Python types.

        Example::

            for row in dataset.get_denormalized_as_dict(["Image", "Subject"]):
                print(row["Image.RID"], row["Subject.Name"])
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        yield from Denormalizer(self, version=version).as_dict(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )

    def list_denormalized_columns(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        version: DatasetVersion | str | None = None,
    ) -> list[tuple[str, str]]:
        """List the columns the denormalized table would have.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.columns`.
        Model-only — no data fetch, no catalog query. Runs the same
        Rule 2/5/6 validation as
        :meth:`get_denormalized_as_dataframe` so planner errors surface
        early.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).
            version: Optional dataset version. Rarely matters for column
                preview (the schema shape is usually stable across
                versions) but accepted for symmetry with the other
                denormalize methods.

        Returns:
            List of ``(column_name, column_type)`` tuples.

        Example::

            cols = dataset.list_denormalized_columns(["Image", "Subject"])
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self, version=version).columns(
            include_tables,
            row_per=row_per,
            via=via,
        )

    def describe_denormalized(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        version: DatasetVersion | str | None = None,
    ) -> dict[str, Any]:
        """Dry-run the denormalization; return planning metadata.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.describe` —
        returns a plan dict with ``row_per``, ``row_per_source``,
        ``row_per_candidates``, ``columns``, ``include_tables``, ``via``,
        ``join_path``, ``transparent_intermediates``, ``ambiguities``,
        ``estimated_row_count``, ``anchors``, and ``source``. This method
        never raises on ambiguity — ambiguities are reported in the dict.

        Args:
            include_tables: Tables whose columns would appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).
            version: Optional dataset version (snapshot-bound queries).
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self, version=version).describe(
            include_tables,
            row_per=row_per,
            via=via,
        )

    def list_schema_paths(
        self,
        tables: list[str] | None = None,
        *,
        version: DatasetVersion | str | None = None,
    ) -> dict[str, Any]:
        """List FK paths reachable from this dataset's members.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.list_paths`.
        Useful for schema exploration — answers "what tables could I
        include in a denormalization?"

        Args:
            tables: Optional filter — when given, ``schema_paths`` in
                the returned dict includes only entries involving at
                least one of these tables.
            version: Optional dataset version. When given, the member
                enumeration uses the corresponding catalog snapshot.

        Returns:
            Dict with 6 keys: ``member_types``, ``anchor_types``,
            ``reachable_tables``, ``association_tables``,
            ``feature_tables``, ``schema_paths``. See
            :meth:`Denormalizer.list_paths` for the detailed shape.

        Example::

            info = dataset.list_schema_paths()
            print(info["member_types"])       # e.g. ["Image", "Subject"]
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self, version=version).list_paths(tables=tables)

    def cache_denormalized(
        self,
        include_tables: list[str],
        version: str | None = None,
        force: bool = False,
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> pd.DataFrame:
        """Denormalize dataset tables and cache the result locally as SQLite.

        On first call, computes the denormalized join and stores it in the
        working data cache. Subsequent calls return the cached data without
        re-computing the join. Use ``force=True`` to re-compute.

        The cache key is derived from the dataset RID, sorted table names,
        version, and the planner knobs (``row_per`` / ``via`` /
        ``ignore_unrelated_anchors``), so each planner variant caches
        independently.

        Args:
            include_tables: List of table names to include in the join.
            version: Dataset version to query. Defaults to current version.
            force: If True, re-compute even if already cached.
            row_per: Optional explicit leaf table (Rule 2). If None,
                auto-inferred from sinks in ``include_tables``.
            via: Optional path-only intermediates to disambiguate FK paths
                (Rule 6) without adding their columns to the output.
            ignore_unrelated_anchors: If True, silently drop dataset
                members whose table has no FK path to ``include_tables``
                (Rule 8). Default False raises
                :class:`DerivaMLDenormalizeUnrelatedAnchor`.

        Returns:
            DataFrame with the denormalized wide table.

        Example::

            dataset = ml.lookup_dataset("28CT")
            df = dataset.cache_denormalized(["Image", "Diagnosis"], version="1.0.0")
            print(df["Image.Filename"].head())

            # Second call returns cached data instantly
            df = dataset.cache_denormalized(["Image", "Diagnosis"], version="1.0.0")

            # Diamond-schema disambiguation via an intermediate table:
            df = dataset.cache_denormalized(
                ["Image", "Subject"], via=["Observation"]
            )
        """
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        # Resolve version → snapshot-bound catalog so fetches are
        # reproducible per spec-pinned version. When version is None,
        # falls through to self._ml_instance (whatever the DerivaML was
        # constructed against — live or pre-pinned). Matches the
        # pattern used by Dataset.list_dataset_members / _version_snapshot_catalog.
        version_snapshot_ml = self._version_snapshot_catalog(version)
        paged_client = ErmrestPagedClient(catalog=version_snapshot_ml.catalog)
        children = [c.dataset_rid for c in self.list_dataset_children(recurse=True)]
        result = self._ml_instance.workspace.cache_denormalized(
            model=version_snapshot_ml.model,
            dataset_rid=self.dataset_rid,
            include_tables=include_tables,
            version=version,
            source="catalog",
            refresh=force,
            dataset=self,
            dataset_children_rids=children,
            paged_client=paged_client,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )
        return result.to_dataframe()

    @validate_call(config=VALIDATION_CONFIG)
    def add_dataset_members(
        self,
        members: list[RID] | dict[str, list[RID]],
        validate: bool = True,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Add records to this dataset.

        Associates one or more records with this dataset and flips the
        dataset to a dev version (per ADR-0003: every mutation lands on
        dev). Members can be provided in two forms:

        **List of RIDs (simpler but slower):**
        When `members` is a list of RIDs, each RID is resolved to determine which table
        it belongs to. This uses batch RID resolution for efficiency, but still requires
        querying the catalog to identify each RID's table.

        **Dictionary by table name (faster, recommended for large datasets):**
        When `members` is a dict mapping table names to lists of RIDs, no RID resolution
        is needed. The RIDs are inserted directly into the dataset. Use this form when
        you already know which table each RID belongs to.

        **Important:** Members can only be added from tables that have been registered as
        dataset element types. Use :meth:`DerivaML.add_dataset_element_type` to register
        a table before adding its records to datasets.

        After this call, ``current_version`` is a dev label of the form
        ``<last_release>.post1.devN``. Call :meth:`release` to mint a
        released version when the dev period is complete. A call with
        an empty ``members`` argument is a no-op and does not advance
        the dev counter.

        Args:
            members: Either:
                - list[RID]: List of RIDs to add. Each RID will be resolved to find its table.
                - dict[str, list[RID]]: Mapping of table names to RID lists. Skips resolution.
            validate: Whether to validate that members don't already exist. Defaults to True.
            description: Optional description of the member additions.
            execution_rid: Optional execution RID to associate with changes.

        Raises:
            DerivaMLException: If:
                - Any RID is invalid or cannot be resolved
                - Any RID belongs to a table that isn't registered as a dataset element type
                - Adding members would create a cycle (for nested datasets)
                - Validation finds duplicate members (when validate=True)

        See Also:
            :meth:`DerivaML.add_dataset_element_type`: Register a table as a dataset element type.
            :meth:`DerivaML.list_dataset_element_types`: List registered dataset element types.

        Examples:
            Using a list of RIDs (simpler):
                >>> dataset.add_dataset_members(  # doctest: +SKIP
                ...     members=["1-ABC", "1-DEF", "1-GHI"],
                ...     description="Added sample images"
                ... )

            Using a dict by table name (faster for large datasets):
                >>> dataset.add_dataset_members(  # doctest: +SKIP
                ...     members={
                ...         "Image": ["1-ABC", "1-DEF"],
                ...         "Subject": ["2-XYZ"]
                ...     },
                ...     description="Added images and subjects"
                ... )
        """
        description = description or "Updated dataset via add_dataset_members"

        def check_dataset_cycle(member_rid, path=None):
            """

            Args:
              member_rid:
              path: (Default value = None)

            Returns:

            """
            path = path or set(self.dataset_rid)
            return member_rid in path

        if validate:
            existing_rids = set(m["RID"] for ms in self.list_dataset_members().values() for m in ms)
            if overlap := set(existing_rids).intersection(members):
                raise DerivaMLException(
                    f"Attempting to add existing member to dataset_table {self.dataset_rid}: {overlap}"
                )

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements: dict[str, list[RID]] = {}

        # Map of valid element tables to their association tables.
        association_map = self._element_to_association_map()

        # Get a list of all the object types that can be linked to a dataset_table.
        if type(members) is list:
            members = set(members)

            # Get candidate tables for batch resolution (only tables that can be dataset elements)
            candidate_tables = [
                self._ml_instance.model.name_to_table(table_name) for table_name in association_map.keys()
            ]

            # Batch resolve all RIDs at once instead of one-by-one
            rid_results = self._ml_instance.resolve_rids(members, candidate_tables=candidate_tables)

            # Group by table and validate
            for rid, rid_info in rid_results.items():
                if rid_info.table_name not in association_map:
                    raise DerivaMLException(f"RID table: {rid_info.table_name} not part of dataset_table")
                if rid_info.table == self._dataset_table and check_dataset_cycle(rid_info.rid):
                    raise DerivaMLException("Creating cycle of datasets is not allowed")
                dataset_elements.setdefault(rid_info.table_name, []).append(rid_info.rid)
        else:
            dataset_elements = {t: list(set(ms)) for t, ms in members.items()}
        # Now make the entries into the association tables.
        pb = self._ml_instance.pathBuilder()
        any_added = False
        for table, elements in dataset_elements.items():
            # Determine schema: ML schema for Dataset/File, otherwise use the table's actual schema
            if table == "Dataset" or table == "File":
                schema_name = self._ml_instance.ml_schema
            else:
                # Find the table and use its schema
                table_obj = self._ml_instance.model.name_to_table(table)
                schema_name = table_obj.schema.name
            schema_path = pb.schemas[schema_name]
            fk_column = "Nested_Dataset" if table == "Dataset" else table
            if len(elements):
                # Find out the name of the column in the association table.
                schema_path.tables[association_map[table]].insert(
                    [{"Dataset": self.dataset_rid, fk_column: e} for e in elements]
                )
                any_added = True
        # Per ADR-0003 / Q18: a call that doesn't change any row is a no-op
        # and doesn't advance the dev counter. Only flip to dev if at least
        # one association row was inserted.
        if any_added:
            self._create_or_advance_dev_row(
                description=description,
                execution_rid=execution_rid,
            )

    @validate_call(config=VALIDATION_CONFIG)
    def delete_dataset_members(
        self,
        members: list[RID],
        description: str = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Remove members from this dataset.

        Removes the specified members and flips the dataset to a dev
        version (per ADR-0003: every mutation lands on dev).

        After this call, ``current_version`` is a dev label of the form
        ``<last_release>.post1.devN``. Call :meth:`release` to mint a
        released version when the dev period is complete. A call with
        an empty ``members`` argument is a no-op and does not advance
        the dev counter.

        Args:
            members: List of member RIDs to remove from the dataset.
            description: Optional description of the removal operation.
                Replaces the dev row's existing description.
            execution_rid: Optional RID of execution associated with
                this operation. Stored on the dev row.

        Raises:
            DerivaMLException: If any RID is invalid or not part of this dataset.

        Example:
            >>> dataset.delete_dataset_members(  # doctest: +SKIP
            ...     members=["1-ABC", "1-DEF"],
            ...     description="Removed corrupted samples"
            ... )
            >>> dataset.current_version  # doctest: +SKIP
            <Version('0.4.0.post1.dev1')>
        """
        members = set(members)
        # Per ADR-0003 / Q18: no-op input doesn't advance the dev counter.
        if not members:
            return
        description = description or "Deleted dataset members"

        # Go through every rid to be deleted and sort them based on what association table entries
        # need to be removed.
        dataset_elements: dict[str, list[RID]] = {}
        association_map = self._element_to_association_map()

        # Batch resolve all RIDs in one query per candidate table instead
        # of one network round-trip per RID. Mirrors the optimization in
        # add_dataset_members.
        candidate_tables = [self._ml_instance.model.name_to_table(table_name) for table_name in association_map.keys()]
        try:
            rid_results = self._ml_instance.resolve_rids(members, candidate_tables=candidate_tables)
        except DerivaMLException as e:
            # Preserve the legacy "Invalid RID" message shape for callers
            # that match on the prefix.
            raise DerivaMLException(f"Invalid RID: {e}") from e
        for rid, rid_info in rid_results.items():
            if rid_info.table_name not in association_map:
                raise DerivaMLException(f"RID table: {rid_info.table_name} not part of dataset")
            dataset_elements.setdefault(rid_info.table_name, []).append(rid_info.rid)

        # Delete the entries from the association tables. Use one
        # filtered DELETE per element table (Any-quantified IN-list)
        # rather than per-RID — collapses N round-trips into 1 per
        # element type.
        pb = self._ml_instance.pathBuilder()
        for table, elements in dataset_elements.items():
            if not elements:
                continue
            # Determine schema: ML schema for Dataset, otherwise use the table's actual schema
            if table == "Dataset":
                schema_name = self._ml_instance.ml_schema
            else:
                table_obj = self._ml_instance.model.name_to_table(table)
                schema_name = table_obj.schema.name
            schema_path = pb.schemas[schema_name]
            fk_column = "Nested_Dataset" if table == "Dataset" else table

            atable_path = schema_path.tables[association_map[table]]
            # Chain two .filter() calls rather than ``& (a == AnyQuantifier(*xs))``
            # — the compound-and form generates an URL the server rejects with
            # "URL parse error at token: None" for the Any-quantified clause.
            atable_path.filter(
                atable_path.Dataset == self.dataset_rid,
            ).filter(
                atable_path.columns[fk_column] == AnyQuantifier(*elements),
            ).delete()

        # Per ADR-0003: every mutation lands on dev. The caller passed a
        # non-empty `members` list, so this is a real mutation attempt
        # (the validation step above resolved the RIDs and confirmed they
        # belong to dataset element types). RIDs that didn't actually
        # match a membership row are a benign DELETE no-op at the table
        # level but still represent caller intent — flip to dev.
        self._create_or_advance_dev_row(
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call(config=VALIDATION_CONFIG)
    def list_dataset_parents(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: DatasetVersion | str | None = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Return the parent datasets that contain this dataset as a nested child.

        Queries the Dataset_Dataset association table to find datasets that include
        this dataset as a nested member. When ``recurse=True``, traverses the full
        ancestor chain (parents of parents, etc.).

        Args:
            recurse: If True, recursively return all ancestor datasets, not just
                direct parents.
            _visited: Internal parameter to track visited datasets and prevent
                infinite recursion in cyclic graphs. Callers should not set this.
            version: Dataset version to query against. If provided, uses a catalog
                snapshot from that version. Defaults to the current version.
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            list[Dataset]: Parent Dataset objects that contain this dataset. Empty
                list if this dataset is not nested inside any other dataset.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
            >>> parents = ds.list_dataset_parents()  # doctest: +SKIP
            >>> print([p.dataset_rid for p in parents])  # doctest: +SKIP
        """
        # Initialize visited set for recursion guard
        if _visited is None:
            _visited = set()

        # Prevent infinite recursion by checking if we've already visited this dataset
        if self.dataset_rid in _visited:
            return []
        _visited.add(self.dataset_rid)

        # Get association table for nested datasets
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        pb = version_snapshot_catalog.pathBuilder()
        atable_path = pb.schemas[self._ml_instance.ml_schema].Dataset_Dataset
        parents = [
            version_snapshot_catalog.lookup_dataset(p["Dataset"])
            for p in atable_path.filter(atable_path.Nested_Dataset == self.dataset_rid).entities().fetch()
        ]
        if recurse:
            for parent in parents.copy():
                # Each `parent` is already bound to `version_snapshot_catalog`
                # via the lookup above. Passing ``version=None`` on recursion
                # means "use the catalog this Dataset is already pinned to"
                # — i.e. the outer snapshot. Forwarding the outer version
                # string would make the nested dataset try to resolve the
                # outer's version in its OWN history, which raises when the
                # timelines don't line up (nested datasets have their own
                # version sequences).
                parents.extend(parent.list_dataset_parents(recurse=True, _visited=_visited, version=None))
        return parents

    @validate_call(config=VALIDATION_CONFIG)
    def list_dataset_children(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: DatasetVersion | str | None = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Return the child datasets nested inside this dataset.

        Queries the Dataset_Dataset association table to find datasets that are
        nested members of this dataset. When ``recurse=True``, traverses the full
        descendant tree (children of children, etc.).

        Args:
            recurse: If True, recursively return all descendant datasets, not just
                direct children.
            _visited: Internal parameter to track visited datasets and prevent
                infinite recursion in cyclic graphs. Callers should not set this.
            version: Dataset version to query against. If provided, uses a catalog
                snapshot from that version. Defaults to the current version.
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            list[Dataset]: Child Dataset objects nested in this dataset. Empty list
                if this dataset has no nested children.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
            >>> children = ds.list_dataset_children()  # doctest: +SKIP
        """
        # Initialize visited set for recursion guard
        if _visited is None:
            _visited = set()

        version = DatasetVersion.parse(version) if isinstance(version, str) else version
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        dataset_dataset_path = (
            version_snapshot_catalog.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Dataset"]
        )
        nested_datasets = list(dataset_dataset_path.entities().fetch())

        def find_children(rid: RID) -> list[RID]:
            # Prevent infinite recursion by checking if we've already visited this dataset
            if rid in _visited:
                return []
            _visited.add(rid)

            children = [child["Nested_Dataset"] for child in nested_datasets if child["Dataset"] == rid]
            if recurse:
                for child in children.copy():
                    children.extend(find_children(child))
            return children

        return [version_snapshot_catalog.lookup_dataset(rid) for rid in find_children(self.dataset_rid)]

    def list_executions(self) -> list["Execution"]:
        """List all executions associated with this dataset.

        Returns all executions that used this dataset as input. This is
        tracked through the Dataset_Execution association table.

        Returns:
            List of Execution objects associated with this dataset.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> dataset = ml.lookup_dataset("1-abc123")  # doctest: +SKIP
            >>> executions = dataset.list_executions()  # doctest: +SKIP
            >>> for exe in executions:  # doctest: +SKIP
            ...     print(f"Execution {exe.execution_rid}: {exe.status}")  # doctest: +SKIP
        """
        # Import here to avoid circular dependency

        pb = self._ml_instance.pathBuilder()
        dataset_execution_path = pb.schemas[self._ml_instance.ml_schema].Dataset_Execution

        # Query for all executions associated with this dataset
        records = list(
            dataset_execution_path.filter(dataset_execution_path.Dataset == self.dataset_rid).entities().fetch()
        )

        return [self._ml_instance.lookup_execution(record["Execution"]) for record in records]

    @staticmethod
    def _insert_dataset_versions(
        ml_instance: DerivaMLCatalog,
        dataset_list: list[DatasetSpec],
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Insert new version records for a list of datasets.

        This internal method creates Dataset_Version records in the catalog for
        each dataset in the list. It also captures a catalog snapshot timestamp
        to associate with these versions.

        The version record links:
        - The dataset RID to its new version number
        - An optional description of what changed
        - An optional execution that triggered the version change
        - The catalog snapshot time for reproducibility

        Args:
            ml_instance: The catalog instance to insert versions into.
            dataset_list: List of DatasetSpec objects containing RID and version info.
            description: Optional description of the version change.
            execution_rid: Optional execution RID to associate with the version.
        """
        schema_path = ml_instance.pathBuilder().schemas[ml_instance.ml_schema]

        # Insert version records for all datasets in the list
        version_records = schema_path.tables["Dataset_Version"].insert(
            [
                {
                    "Dataset": dataset.rid,
                    "Version": str(dataset.version),
                    "Description": description,
                    "Execution": execution_rid,
                }
                for dataset in dataset_list
            ]
        )
        version_records = list(version_records)

        # Update each dataset's current version pointer to the new version
        # record before stamping the snapshot. The Dataset.Version UPDATE
        # bumps Dataset.RMT; doing it first means the *Dataset_Version*
        # row ends up with the latest RMT after the Snapshot update below.
        # Drift detection (PR 6) uses the version row's RMT as the time
        # anchor for ``is_dirty`` / ``release_diff``; this ordering ensures
        # that anchor is later than every other create- or release-time
        # write, so a freshly-finalised dataset reads as not dirty.
        schema_path.tables["Dataset"].update([{"Version": v["RID"], "RID": v["Dataset"]} for v in version_records])

        # ERMrest does not return system-generated columns (including snaptime)
        # in the INSERT response — it only echoes back the columns you sent.
        # We need the snaptime to record the version's catalog snapshot for
        # point-in-time reads. Perform a separate GET immediately after the
        # INSERT to retrieve the server-assigned snaptime for this row.
        snap = ml_instance.catalog.get("/").json()["snaptime"]

        # Update version records with the snapshot timestamp.  This UPDATE
        # is the last write of the version-insertion sequence, so the
        # version row's RMT becomes the time anchor for drift detection.
        schema_path.tables["Dataset_Version"].update(
            [{"RID": v["RID"], "Dataset": v["Dataset"], "Snapshot": snap} for v in version_records]
        )

    @validate_call(config=VALIDATION_CONFIG)
    def download_dataset_bag(
        self,
        version: DatasetVersion | str,
        materialize: bool = True,
        use_minid: bool = False,
        exclude_tables: set[str] | None = None,
        timeout: tuple[int, int] | None = None,
        fetch_concurrency: int = 1,
    ) -> DatasetBag:
        """Downloads a dataset to the local filesystem and optionally creates a MINID.

        Downloads a dataset to the local file system. If the dataset has a version set, that version is used.
        If the dataset has a version and a version is provided, the version specified takes precedence.

        The exported bag contains all data reachable from this dataset's members by following
        foreign key relationships (both incoming and outgoing). Starting from each member element
        type, the export traverses all FK-connected tables, with vocabulary tables acting as
        natural path terminators. Only paths starting from element types that have members in
        this dataset are included.

        Args:
            version: Dataset version to download. If not specified, the version must be set in the dataset.
            materialize: If True, materialize the dataset after downloading.
            use_minid: If True, upload the bag to S3 and create a MINID for the dataset.
                Requires s3_bucket to be configured on the catalog. Defaults to False.
            exclude_tables: Optional set of table names to exclude from FK path traversal
                during bag export. Tables in this set will not be visited, pruning branches
                of the FK graph that pass through them. Useful for avoiding query timeouts
                caused by expensive joins through large or unnecessary tables.
            timeout: Optional (connect_timeout, read_timeout) in seconds for network
                requests. Defaults to (10, 610). Increase read_timeout for large datasets
                with deep FK joins that need more time to complete.
            fetch_concurrency: Maximum number of concurrent file downloads during
                materialization. Defaults to 8.

        Returns:
            DatasetBag: A ``DatasetBag`` instance wrapping the downloaded bag. Key attributes:
                ``bag.path`` (``Path``) — local directory containing the bag;
                ``bag.dataset_rid`` (str) — RID of the dataset;
                ``bag.current_version`` (``DatasetVersion``) — version downloaded.

        Raises:
            DerivaMLDatasetNotFound: If this dataset's RID no longer exists in the
                catalog (e.g., was deleted after this object was created).
            DerivaMLException: If use_minid=True but s3_bucket is not configured on
                the catalog, or if the bag export or materialization fails.

        Examples:
            Download without MINID (default):
                >>> bag = dataset.download_dataset_bag(version="1.0.0")  # doctest: +SKIP
                >>> print(f"Downloaded to {bag.path}")  # doctest: +SKIP

            Download with MINID (requires s3_bucket configured):
                >>> # Catalog must be created with s3_bucket="s3://my-bucket"
                >>> bag = dataset.download_dataset_bag(version="1.0.0", use_minid=True)  # doctest: +SKIP

            Exclude tables that cause query timeouts:
                >>> bag = dataset.download_dataset_bag(version="1.0.0", exclude_tables={"Process"})  # doctest: +SKIP
        """
        if isinstance(version, str):
            version = DatasetVersion.parse(version)

        # Validate use_minid requires s3_bucket configuration
        if use_minid and not self._ml_instance.s3_bucket:
            raise DerivaMLException(
                "Cannot use use_minid=True without s3_bucket configured. "
                "Configure s3_bucket when creating the DerivaML instance to enable MINID support."
            )

        # The bag-download orchestration lives in ``bag_download.py``
        # (extracted Phase 3, audit §3.A). ``download_dataset_bag``
        # stays on ``Dataset`` because users call it via
        # ``dataset.download_dataset_bag(...)``; the cluster of
        # private helpers it composes is now a free-function module.
        from deriva_ml.dataset.bag_download import (
            download_dataset_minid,
            get_dataset_minid,
            materialize_dataset_bag,
        )

        minid = get_dataset_minid(
            self, version, create=True, use_minid=use_minid, exclude_tables=exclude_tables, timeout=timeout
        )

        bag_path = (
            materialize_dataset_bag(self, minid, use_minid=use_minid, fetch_concurrency=fetch_concurrency)
            if materialize
            else download_dataset_minid(self, minid, use_minid)
        )
        from deriva_ml.model.deriva_ml_bag_view import DerivaMLBagView

        db_model = DatabaseModel(minid, bag_path, self._ml_instance.working_dir)
        return DerivaMLBagView(db_model).lookup_dataset(self.dataset_rid)

    @validate_call(config=VALIDATION_CONFIG)
    def estimate_bag_size(
        self,
        version: DatasetVersion | str,
        exclude_tables: set[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate the size of a dataset bag before downloading.

        Uses :meth:`DatasetBagBuilder.aggregate_queries` to build datapath
        objects for every FK path that reaches each table, then fetches RID
        lists from the snapshot catalog and computes the exact union across
        all paths.

        When the same table is reachable via multiple FK paths, **all** paths
        are queried and the RID sets are unioned to get the exact row count.
        For asset tables, ``(RID, Length)`` pairs are fetched and deduplicated
        by RID so that ``asset_bytes`` reflects the true total.

        Note:
            This is the one place in ``dataset/`` that bypasses
            :class:`~deriva.bag.catalog_builder.CatalogBagBuilder` for
            execution (it still shares the walker via
            :meth:`~DatasetBagBuilder.aggregate_queries`). The bypass
            is **deliberate** — see
            :doc:`docs/adr/0008-estimate-bag-size-bypasses-bag-pipeline`
            for the design decision, the rejected alternatives
            (lifting parallelism upstream vs. leaving the divergence
            undocumented), and practical contributor guidance. Do
            not "fix" the duplication without reading the ADR first.

        Note: this fetches complete RID lists rather than using server-side
        aggregates, which gives exact union counts but uses O(N) memory where
        N is the total rows across all paths.  This is suitable for datasets
        with up to hundreds of thousands of rows per table.

        Args:
            version: Dataset version to estimate.
            exclude_tables: Optional set of table names to exclude from FK path
                traversal, same as in download_dataset_bag.

        Returns:
            dict with keys:
                - tables: dict mapping table name to
                  {row_count, is_asset, asset_bytes, csv_bytes}
                - total_rows: total row count across all tables
                - total_asset_bytes: total size of asset files in bytes
                - total_asset_size: human-readable asset size (e.g., "1.2 GB")
                - total_csv_bytes: estimated size of CSV metadata in bytes
                - total_csv_size: human-readable CSV size
                - total_estimated_bytes: asset + CSV bytes combined
                - total_estimated_size: human-readable combined size
        """
        if isinstance(version, str):
            version = DatasetVersion.parse(version)

        # Build a DatasetBagBuilder on the version snapshot and collect
        # aggregate datapath objects grouped by target table.
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        builder = DatasetBagBuilder(
            ml_instance=version_snapshot_catalog,
            exclude_tables=exclude_tables,
        )
        table_queries = builder.aggregate_queries(self)

        # Connect to the snapshot catalog for queries using the async catalog,
        # which uses httpx.AsyncClient with connection pooling and is safe for
        # concurrent requests (unlike the sync ErmrestCatalog).
        snapshot_catalog_id = self._version_snapshot_catalog_id(version)
        from deriva.core import get_credential

        hostname = self._ml_instance.catalog.deriva_server.server
        protocol = self._ml_instance.catalog.deriva_server.scheme
        credentials = get_credential(hostname)

        # Parse snapshot catalog ID (format: "catalog_id@snaptime" or just "catalog_id")
        if "@" in snapshot_catalog_id:
            cat_id, snaptime = snapshot_catalog_id.split("@", 1)
            catalog = AsyncErmrestSnapshot(protocol, hostname, cat_id, snaptime, credentials)
        else:
            catalog = AsyncErmrestCatalog(protocol, hostname, snapshot_catalog_id, credentials)

        def _extract_path(uri: str) -> str:
            """Extract the catalog-relative path from a full datapath URI.

            Strips the ``https://host/ermrest/catalog/N`` prefix, returning the
            path starting from ``/aggregate/``, ``/entity/``, ``/attribute/``, etc.
            """
            for marker in ("/aggregate/", "/entity/", "/attribute/"):
                idx = uri.find(marker)
                if idx >= 0:
                    return uri[idx:]
            raise ValueError(f"Cannot extract catalog path from URI: {uri}")

        # Build query paths using the datapath API.  For each
        # (table_name, path_entries) we fetch RID lists (and RID+Length for
        # assets) so we can compute the exact union across all FK paths.
        # (table_name, query_path, query_type)
        query_items: list[tuple[str, str, str]] = []
        # Track which tables already have a sample query to avoid duplicates
        # when multiple FK paths reach the same table.
        sampled_tables: set[str] = set()

        for table_name, path_entries in table_queries.items():
            for dp, target_table, is_asset in path_entries:
                # Fetch RID list for row-count union
                rid_rs = dp.attributes(target_table.RID)
                query_items.append((table_name, _extract_path(rid_rs.uri), "csv"))

                if is_asset:
                    entity_path = _extract_path(dp.uri).removeprefix("/entity/")
                    fetch_path = f"/attribute/{entity_path}/RID,Length"
                    query_items.append((table_name, fetch_path, "fetch"))

                # Sample a few rows to estimate CSV serialization size.
                # Only one sample per table (first path wins).
                if table_name not in sampled_tables:
                    sampled_tables.add(table_name)
                    entity_path = _extract_path(dp.uri)
                    sample_path = f"{entity_path}?limit=100"
                    query_items.append((table_name, sample_path, "sample"))

        # Execute all queries concurrently using asyncio.gather
        import asyncio

        async def _run_query(table_name: str, query_path: str, query_type: str) -> tuple[str, str, Any]:
            try:
                response = await catalog.get_async(query_path)
                return table_name, query_type, response.json()
            except Exception as exc:
                self._logger.debug("estimate_bag_size query failed for %s (%s): %s", table_name, query_path, exc)
                return table_name, query_type, []

        async def _run_all_queries():
            tasks = [_run_query(name, path, qtype) for name, path, qtype in query_items]
            results = await asyncio.gather(*tasks)
            await catalog.close()
            return results

        # Run the async queries from the sync context. See
        # :func:`deriva_ml.core.async_helpers.run_async` — same
        # notebook-loop-fallback dance that the bag-commit path
        # uses for ``BagCatalogLoader.arun``.
        all_results = run_async(_run_all_queries())

        # Compute exact union of RIDs across all paths for each table.
        rids_by_table: dict[str, set[str]] = defaultdict(set)
        # For assets, collect {RID: Length} across paths (first wins; same asset = same Length).
        asset_lengths_by_table: dict[str, dict[str, int]] = defaultdict(dict)
        # Collect sample rows for CSV size estimation.
        sample_rows_by_table: dict[str, list[dict]] = {}

        for table_name, query_type, rows in all_results:
            if query_type == "csv":
                rids_by_table[table_name].update(r["RID"] for r in rows if "RID" in r)
            elif query_type == "fetch":
                for r in rows:
                    rid = r.get("RID")
                    if rid and rid not in asset_lengths_by_table[table_name]:
                        asset_lengths_by_table[table_name][rid] = r.get("Length") or 0
            elif query_type == "sample":
                # Keep only the first sample per table (set during query building)
                if table_name not in sample_rows_by_table and rows:
                    sample_rows_by_table[table_name] = rows

        # Estimate CSV size per table from sample rows.
        csv_bytes_by_table: dict[str, int] = {}
        for table_name, sample_rows in sample_rows_by_table.items():
            row_count = len(rids_by_table.get(table_name, set()))
            csv_bytes_by_table[table_name] = self._estimate_csv_bytes(sample_rows, row_count)

        # Determine which tables are assets from the original table_queries
        asset_tables = {
            table_name for table_name, entries in table_queries.items() if any(is_asset for _, _, is_asset in entries)
        }

        table_estimates: dict[str, dict[str, Any]] = {}
        total_rows = 0
        total_asset_bytes = 0
        total_csv_bytes = 0

        for table_name, rids in rids_by_table.items():
            row_count = len(rids)
            is_asset = table_name in asset_tables
            asset_bytes = sum(asset_lengths_by_table[table_name].values())
            csv_bytes = csv_bytes_by_table.get(table_name, 0)
            table_estimates[table_name] = {
                "row_count": row_count,
                "is_asset": is_asset,
                "asset_bytes": asset_bytes,
                "csv_bytes": csv_bytes,
            }
            total_rows += row_count
            total_asset_bytes += asset_bytes
            total_csv_bytes += csv_bytes

        # Handle tables that only appear in fetch results (unlikely but safe)
        for table_name, lengths in asset_lengths_by_table.items():
            if table_name not in table_estimates:
                csv_bytes = csv_bytes_by_table.get(table_name, 0)
                table_estimates[table_name] = {
                    "row_count": len(lengths),
                    "is_asset": True,
                    "asset_bytes": sum(lengths.values()),
                    "csv_bytes": csv_bytes,
                }
                total_rows += len(lengths)
                total_asset_bytes += sum(lengths.values())
                total_csv_bytes += csv_bytes

        total_size = total_asset_bytes + total_csv_bytes
        return {
            "tables": table_estimates,
            "total_rows": total_rows,
            "total_asset_bytes": total_asset_bytes,
            "total_asset_size": self._human_readable_size(total_asset_bytes),
            "total_csv_bytes": total_csv_bytes,
            "total_csv_size": self._human_readable_size(total_csv_bytes),
            "total_estimated_bytes": total_size,
            "total_estimated_size": self._human_readable_size(total_size),
        }

    @validate_call(config=VALIDATION_CONFIG)
    def bag_info(
        self,
        version: DatasetVersion | str,
        exclude_tables: set[str] | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive info about a dataset bag: size, contents, and cache status.

        Combines the size estimate from ``estimate_bag_size`` with local cache
        status from ``BagCache``. Use this to decide whether to prefetch a bag
        before running an experiment.

        Args:
            version: Dataset version to inspect.
            exclude_tables: Optional set of table names to exclude from FK path
                traversal, same as in download_dataset_bag.

        Returns:
            dict with keys:
                - tables: dict mapping table name to {row_count, is_asset, asset_bytes}
                - total_rows: total row count across all tables
                - total_asset_bytes: total size of asset files in bytes
                - total_asset_size: human-readable size string
                - cache_status: one of "not_cached", "cached_materialized",
                  "cached_holey"
                - cache_path: local path to cached bag (if cached), else None
        """
        # Get size estimate
        size_info = self.estimate_bag_size(version=version, exclude_tables=exclude_tables)

        # Get cache status
        from deriva_ml.dataset.bag_cache import BagCache

        cache = BagCache(self._ml_instance.cache_dir)
        cache_info = cache.cache_status(self.dataset_rid)

        return {**size_info, **cache_info}

    def cache(
        self,
        version: DatasetVersion | str,
        materialize: bool = True,
        exclude_tables: set[str] | None = None,
        timeout: tuple[int, int] | None = None,
        fetch_concurrency: int = 1,
    ) -> dict[str, Any]:
        """Download a dataset bag into the local cache without creating an execution.

        Use this to warm the cache before running experiments. No execution or
        provenance records are created — this is purely a local download operation.

        Internally calls ``download_dataset_bag`` with ``use_minid=False``.

        Args:
            version: Dataset version to cache.
            materialize: If True (default), download all asset files. If False,
                download only metadata (table data without binary assets).
            exclude_tables: Optional set of table names to exclude from FK path
                traversal during bag export.
            timeout: Optional (connect_timeout, read_timeout) in seconds.
            fetch_concurrency: Maximum number of concurrent file downloads during
                materialization. Defaults to 8.

        Returns:
            dict with bag_info results after caching (includes cache_status,
            cache_path, and size info).
        """
        self.download_dataset_bag(
            version=version,
            materialize=materialize,
            use_minid=False,
            exclude_tables=exclude_tables,
            timeout=timeout,
            fetch_concurrency=fetch_concurrency,
        )
        return self.bag_info(version=version, exclude_tables=exclude_tables)

    @staticmethod
    def _estimate_csv_bytes(sample_rows: list[dict], total_row_count: int) -> int:
        """Estimate the CSV file size for a table from a sample of rows.

        Serializes each sample row to CSV format, computes the average row
        size, then extrapolates to the full row count.  A header row (column
        names) is added to the estimate.

        Args:
            sample_rows: List of row dicts (from an entity query).
            total_row_count: Total number of rows in the table.

        Returns:
            Estimated CSV size in bytes.
        """
        if not sample_rows or total_row_count == 0:
            return 0

        import csv
        import io

        # Measure header size (column names)
        columns = list(sample_rows[0].keys())
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        header_bytes = len(buf.getvalue().encode("utf-8"))

        # Measure each sample row
        row_sizes: list[int] = []
        for row in sample_rows:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(str(v) if v is not None else "" for v in row.values())
            row_sizes.append(len(buf.getvalue().encode("utf-8")))

        avg_row_bytes = sum(row_sizes) / len(row_sizes)
        return int(header_bytes + avg_row_bytes * total_row_count)

    @staticmethod
    def _human_readable_size(size_bytes: int) -> str:
        """Convert bytes to human-readable string."""
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{size:.1f} {units[i]}"

    def _version_snapshot_catalog(self, dataset_version: DatasetVersion | str | None) -> DerivaMLCatalog:
        """Get a catalog instance bound to a specific version's snapshot.

        Dataset versions are associated with catalog snapshots, which represent
        the exact state of the catalog at the time the version was created.
        This method returns a catalog instance that queries against that snapshot,
        ensuring reproducible access to historical data.

        Args:
            dataset_version: The version to get a snapshot for, or None to use
                the current catalog state.

        Returns:
            DerivaMLCatalog: Either a snapshot-bound catalog or the current catalog.
        """
        if isinstance(dataset_version, str) and str:
            dataset_version = DatasetVersion.parse(dataset_version)
        if dataset_version:
            return self._ml_instance.catalog_snapshot(self._version_snapshot_catalog_id(dataset_version))
        else:
            return self._ml_instance

    def _version_snapshot_catalog_id(self, version: DatasetVersion | str) -> str:
        """Get the catalog ID with snapshot suffix for a specific version.

        Constructs a catalog identifier in the format "catalog_id@snapshot_time"
        that can be used to access the catalog state at the time the version
        was created.

        Args:
            version: The dataset version to get the snapshot for.

        Returns:
            str: Catalog ID with snapshot suffix (e.g., "1@2023-01-15T10:30:00").

        Raises:
            DerivaMLException: If the specified version doesn't exist.
        """
        version = str(version)
        history = self.dataset_history()
        try:
            version_record = next(h for h in history if str(h.dataset_version) == version)
        except StopIteration:
            available = [(str(h.dataset_version), h.snapshot) for h in history]
            raise DerivaMLException(
                f"Dataset version {version} not found for dataset {self.dataset_rid}. "
                f"Available versions in history: {available}"
            )
        return (
            f"{self._ml_instance.catalog.catalog_id}@{version_record.snapshot}"
            if version_record.snapshot
            else self._ml_instance.catalog.catalog_id
        )


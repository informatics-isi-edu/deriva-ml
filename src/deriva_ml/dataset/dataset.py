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
    ...     dataset.increment_dataset_version(
    ...         component=VersionPart.minor,
    ...         description='Added new samples'
    ...     )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict

# Standard library imports
from graphlib import TopologicalSorter
from pathlib import Path

# Local imports
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Self
from urllib.parse import urlparse

# Deriva imports
import deriva.core.utils.hash_utils as hash_utils
from deriva.core.asyncio import AsyncErmrestCatalog
from deriva.core.asyncio.async_catalog import AsyncErmrestSnapshot

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution
    from deriva_ml.feature import FeatureRecord

# Third-party imports
import pandas as pd
import requests
from bdbag import bdbag_api as bdb
from bdbag.fetch.fetcher import fetch_single_file
from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import format_exception
from deriva.transfer.download import (
    DerivaDownloadAuthenticationError,
    DerivaDownloadAuthorizationError,
    DerivaDownloadConfigurationError,
    DerivaDownloadError,
    DerivaDownloadTimeoutError,
)
from deriva.transfer.download.deriva_export import DerivaExport
from pydantic import ConfigDict, validate_call

try:
    from icecream import ic

    ic.configureOutput(
        includeContext=True,
        argToStringFunction=lambda x: pformat(x.model_dump() if hasattr(x, "model_dump") else x, width=80, depth=10),
    )

except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from deriva_ml.core.constants import RID
from deriva_ml.core.definitions import (
    DRY_RUN_RID,
    MLVocab,
    VocabularyTerm,
)
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.rid_resolution import AnyQuantifier
from deriva_ml.dataset.aux_classes import (
    DatasetHistory,
    DatasetMinid,
    DatasetSpec,
    DatasetVersion,
    VersionPart,
)
from deriva_ml.dataset.catalog_graph import CatalogGraph
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.feature import Feature
from deriva_ml.interfaces import DerivaMLCatalog
from deriva_ml.model.database import DatabaseModel


def _hash_spec(spec: Any) -> str:
    """SHA-256 hex digest of a download spec, keyed by sorted-JSON form.

    Used to key the bag cache and to compare a freshly computed spec
    against the ``Minid_Spec_Hash`` recorded on a historical version.
    ``sort_keys=True`` makes the hash order-insensitive across dict
    renderings so semantically-equal specs always hash identically.

    Args:
        spec: The download spec (any JSON-serializable Python object,
            typically the dict returned by
            ``CatalogGraph.generate_dataset_download_spec``).

    Returns:
        Lowercase hex-digest string (64 chars).
    """
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


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
        ...     # Add members to the dataset
        ...     dataset.add_dataset_members(members=["1-abc", "1-def"])
        ...     # Increment version after changes
        ...     new_version = dataset.increment_dataset_version(VersionPart.minor, "Added samples")
        >>> # Download for offline use
        >>> bag = dataset.download_dataset_bag(version=new_version)  # doctest: +SKIP
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
        self._logger = logging.getLogger("deriva_ml")
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
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
        Dataset._insert_dataset_versions(
            ml_instance=ml_instance,
            dataset_list=[DatasetSpec(rid=dataset_rid, version=version)],
            execution_rid=execution_rid,
            description="Initial dataset creation.",
        )
        dataset = Dataset(
            catalog=ml_instance,
            dataset_rid=dataset_rid,
            description=description,
        )

        # Skip version increment during initial creation (version already set above)
        dataset.add_dataset_types(dataset_types, _skip_version_increment=True)
        return dataset

    def add_dataset_type(
        self,
        dataset_type: str | VocabularyTerm,
        _skip_version_increment: bool = False,
    ) -> None:
        """Add a dataset type to this dataset.

        Adds a type term to this dataset if it's not already present. The term must
        exist in the Dataset_Type vocabulary. Also increments the dataset's minor
        version to reflect the metadata change.

        Args:
            dataset_type: Term name (string) or VocabularyTerm object from Dataset_Type vocabulary.
            _skip_version_increment: Internal parameter to skip version increment when
                called from add_dataset_types (which handles versioning itself).

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

        # Increment minor version to reflect metadata change (unless called from add_dataset_types)
        if not _skip_version_increment:
            self.increment_dataset_version(
                VersionPart.minor,
                description=f"Added dataset type: {vocab_term.name}",
            )

    def remove_dataset_type(self, dataset_type: str | VocabularyTerm) -> None:
        """Remove a dataset type from this dataset.

        Removes a type term from this dataset if it's currently associated. The term
        must exist in the Dataset_Type vocabulary.

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

        # Check if present
        if vocab_term.name not in self.dataset_types:
            return

        # Delete from association table
        _, atable_path = self._get_dataset_type_association_table()
        atable_path.filter(
            (atable_path.Dataset == self.dataset_rid) & (atable_path.Dataset_Type == vocab_term.name)
        ).delete()

    def add_dataset_types(
        self,
        dataset_types: str | VocabularyTerm | list[str | VocabularyTerm],
        _skip_version_increment: bool = False,
    ) -> None:
        """Add one or more dataset types to this dataset.

        Convenience method for adding multiple types at once. Each term must exist
        in the Dataset_Type vocabulary. Types that are already associated with the
        dataset are silently skipped. Increments the dataset's minor version once
        after all types are added.

        Args:
            dataset_types: Single term or list of terms. Can be strings (term names)
                or VocabularyTerm objects.
            _skip_version_increment: Internal parameter to skip version increment
                (used during initial dataset creation).

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

        # Increment version once for all added types (if any were added)
        if added_types and not _skip_version_increment:
            type_names = ", ".join(added_types)
            self.increment_dataset_version(
                VersionPart.minor,
                description=f"Added dataset type(s): {type_names}",
            )

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table from the catalog schema.

        Returns:
            Table: The Deriva Table object for the Dataset table in the ML schema.
        """
        return self._ml_instance.model.schemas[self._ml_instance.ml_schema].tables["Dataset"]

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
        """Retrieves the version history of a dataset.

        Returns a chronological list of dataset versions, including their version numbers,
        creation times, and associated metadata.

        Returns:
            list[DatasetHistory]: List of history entries, each containing:
                - dataset_version: Version number (major.minor.patch)
                - minid: Minimal Viable Identifier
                - snapshot: Catalog snapshot time
                - dataset_rid: Dataset Resource Identifier
                - version_rid: Version Resource Identifier
                - description: Version description
                - execution_rid: Associated execution RID

        Raises:
            DerivaMLException: If dataset_rid is not a valid dataset RID.

        Example:
            >>> history = ml.dataset_history("1-abc123")  # doctest: +SKIP
            >>> for entry in history:  # doctest: +SKIP
            ...     print(f"Version {entry.dataset_version}: {entry.description}")
        """

        if not self._ml_instance.model.is_dataset_rid(self.dataset_rid):
            raise DerivaMLException(f"RID is not for a data set: {self.dataset_rid}")
        version_path = self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Version"]
        return [
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

    @property
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def current_version(self) -> DatasetVersion:
        """Retrieve the current (most recent) version of this dataset.

        Returns the highest semantic version from the dataset's version history.
        If the dataset has no version history, returns ``0.1.0`` as the default.

        Note that each version captures the state of the catalog at the time the
        version was created, not the current state. Values associated with an
        object in the catalog may differ from the values in a given dataset version.

        Returns:
            DatasetVersion: The most recent semantic version of this dataset.

        Raises:
            DerivaMLException: If the catalog query fails.

        Example:
            >>> ver = dataset.current_version  # doctest: +SKIP
            >>> print(str(ver))  # doctest: +SKIP
        """
        history = self.dataset_history()
        if not history:
            return DatasetVersion(0, 1, 0)
        else:
            # Ensure we return a DatasetVersion, not a string
            versions = [h.dataset_version for h in history]
            return max(versions) if versions else DatasetVersion(0, 1, 0)

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
        children = self._list_dataset_children_current()
        parents = self._list_dataset_parents_current()

        # Add this node with its children as dependencies.
        # This means: self depends on children, so children will be ordered before self.
        ts.add(self, *children)

        # Recursively process children
        for child in children:
            child._build_dataset_graph_1(ts, visited)

        # Recursively process parents (they will depend on this node)
        for parent in parents:
            parent._build_dataset_graph_1(ts, visited)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def increment_dataset_version(
        self,
        component: VersionPart,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> DatasetVersion:
        """Increments a dataset's version number.

        Creates a new version of the dataset by incrementing the specified version component
        (major, minor, or patch). The new version is recorded with an optional description
        and execution reference.

        Args:
            component: Which version component to increment ('major', 'minor', or 'patch').
            description: Optional description of the changes in this version.
            execution_rid: Optional execution RID to associate with this version.

        Returns:
            DatasetVersion: The new version number.

        Raises:
            DerivaMLException: If dataset_rid is invalid or version increment fails.

        Example:
            >>> new_version = ml.increment_dataset_version(  # doctest: +SKIP
            ...     dataset_rid="1-abc123",
            ...     component="minor",
            ...     description="Added new samples"
            ... )
            >>> print(f"New version: {new_version}")  # e.g., "1.2.0"  # doctest: +SKIP
        """

        # Find all the datasets that are reachable from this dataset and determine their new version numbers.
        related_datasets = list(self._build_dataset_graph())
        version_update_list = [
            DatasetSpec(
                rid=ds.dataset_rid,
                version=ds.current_version.increment_version(component),
            )
            for ds in related_datasets
        ]
        Dataset._insert_dataset_versions(
            self._ml_instance, version_update_list, description=description, execution_rid=execution_rid
        )
        return next((d.version for d in version_update_list if d.rid == self.dataset_rid))

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_members(
        self,
        members: list[RID] | dict[str, list[RID]],
        validate: bool = True,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Adds members to a dataset.

        Associates one or more records with a dataset. Members can be provided in two forms:

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

        Adding members automatically increments the dataset's minor version.

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

        # Build map of valid element tables to their association tables
        associations = list(self._dataset_table.find_associations())
        association_map = {a.other_fkeys.pop().pk_table.name: a.table.name for a in associations}

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
        self.increment_dataset_version(
            VersionPart.minor,
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def delete_dataset_members(
        self,
        members: list[RID],
        description: str = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Remove members from this dataset.

        Removes the specified members from the dataset. In addition to removing members,
        the minor version number of the dataset is incremented and the description,
        if provided, is applied to that new version.

        Args:
            members: List of member RIDs to remove from the dataset.
            description: Optional description of the removal operation.
            execution_rid: Optional RID of execution associated with this operation.

        Raises:
            DerivaMLException: If any RID is invalid or not part of this dataset.

        Example:
            >>> dataset.delete_dataset_members(  # doctest: +SKIP
            ...     members=["1-ABC", "1-DEF"],
            ...     description="Removed corrupted samples"
            ... )
        """
        members = set(members)
        description = description or "Deleted dataset members"

        # Go through every rid to be deleted and sort them based on what association table entries
        # need to be removed.
        dataset_elements: dict[str, list[RID]] = {}
        associations = list(self._dataset_table.find_associations())
        association_map = {a.other_fkeys.pop().pk_table.name: a.table.name for a in associations}

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
            atable_path.filter(
                (atable_path.Dataset == self.dataset_rid)
                & (atable_path.columns[fk_column] == AnyQuantifier(*elements)),
            ).delete()

        self.increment_dataset_version(
            VersionPart.minor,
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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

    def _list_dataset_parents_current(self) -> list[Self]:
        """Return parent datasets using current catalog state (not version snapshot).

        Used by _build_dataset_graph_1 to find all related datasets for version updates.
        """
        pb = self._ml_instance.pathBuilder()
        atable_path = pb.schemas[self._ml_instance.ml_schema].Dataset_Dataset
        return [
            self._ml_instance.lookup_dataset(p["Dataset"])
            for p in atable_path.filter(atable_path.Nested_Dataset == self.dataset_rid).entities().fetch()
        ]

    def _list_dataset_children_current(self) -> list[Self]:
        """Return child datasets using current catalog state (not version snapshot).

        Used by _build_dataset_graph_1 to find all related datasets for version updates.
        """
        dataset_dataset_path = (
            self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Dataset"]
        )
        nested_datasets = list(dataset_dataset_path.entities().fetch())

        def find_children(rid: RID) -> list[RID]:
            return [child["Nested_Dataset"] for child in nested_datasets if child["Dataset"] == rid]

        return [self._ml_instance.lookup_dataset(rid) for rid in find_children(self.dataset_rid)]

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

        # ERMrest does not return system-generated columns (including snaptime)
        # in the INSERT response — it only echoes back the columns you sent.
        # We need the snaptime to record the version's catalog snapshot for
        # point-in-time reads. Perform a separate GET immediately after the
        # INSERT to retrieve the server-assigned snaptime for this row.
        snap = ml_instance.catalog.get("/").json()["snaptime"]

        # Update version records with the snapshot timestamp
        schema_path.tables["Dataset_Version"].update(
            [{"RID": v["RID"], "Dataset": v["Dataset"], "Snapshot": snap} for v in version_records]
        )

        # Update each dataset's current version pointer to the new version record
        schema_path.tables["Dataset"].update([{"Version": v["RID"], "RID": v["Dataset"]} for v in version_records])

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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

        minid = self._get_dataset_minid(
            version, create=True, use_minid=use_minid, exclude_tables=exclude_tables, timeout=timeout
        )

        bag_path = (
            self._materialize_dataset_bag(minid, use_minid=use_minid, fetch_concurrency=fetch_concurrency)
            if materialize
            else self._download_dataset_minid(minid, use_minid)
        )
        from deriva_ml.model.deriva_ml_database import DerivaMLDatabase

        db_model = DatabaseModel(minid, bag_path, self._ml_instance.working_dir)
        return DerivaMLDatabase(db_model).lookup_dataset(self.dataset_rid)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def estimate_bag_size(
        self,
        version: DatasetVersion | str,
        exclude_tables: set[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate the size of a dataset bag before downloading.

        Uses ``CatalogGraph._aggregate_queries`` to build datapath objects for
        every FK path that reaches each table, then fetches RID lists from the
        snapshot catalog and computes the exact union across all paths.

        When the same table is reachable via multiple FK paths, **all** paths
        are queried and the RID sets are unioned to get the exact row count.
        For asset tables, ``(RID, Length)`` pairs are fetched and deduplicated
        by RID so that ``asset_bytes`` reflects the true total.

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

        # Build a CatalogGraph on the version snapshot and collect aggregate
        # datapath objects grouped by target table.
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        graph = CatalogGraph(
            version_snapshot_catalog,
            exclude_tables=exclude_tables,
        )
        table_queries = graph._aggregate_queries(self)

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

        # Run the async queries from the sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g., Jupyter) -- use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
            all_results = loop.run_until_complete(_run_all_queries())
        else:
            all_results = asyncio.run(_run_all_queries())

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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
                - cache_status: one of "not_cached", "cached_metadata_only",
                  "cached_materialized", "cached_incomplete"
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

    # Backward compatibility alias
    def prefetch(self, *args, **kwargs) -> dict[str, Any]:
        """Deprecated: Use cache() instead."""
        return self.cache(*args, **kwargs)

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
            version_record = next(h for h in history if h.dataset_version == version)
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

    def _download_dataset_minid(self, minid: DatasetMinid, use_minid: bool) -> Path:
        """Download and extract a dataset bag archive into the local cache.

        Handles three source types based on how the bag was obtained:

        1. **Local cache hit** (``minid.checksum`` set by Tier 1 in ``_get_dataset_minid``):
           The cache directory ``{rid}_{checksum}`` already exists → return immediately.

        2. **S3 download** (``use_minid=True``):
           Download the bag archive from S3 via ``minid.bag_url``.

        3. **Client-side bag** (``use_minid=False``):
           The bag was already generated locally by ``_create_dataset_bag_client``
           and is referenced via a ``file://`` URI.

        After obtaining the archive, this method:
        - Extracts it to a staging directory (atomic — prevents corrupt caches)
        - Validates the BDBag structure
        - Moves the staging directory to the final cache location
        - Cleans up temporary files

        Cache directory naming:
        - Deterministic path (Tier 1/3): ``{rid}_{spec_hash[:16]}_{snapshot}``
        - MINID path (Tier 2): ``{rid}_{sha256_from_s3}``

        Both formats are found by the Tier 1 glob in ``_get_dataset_minid``
        (the deterministic format by snapshot suffix, the MINID format by
        its own SHA-256 lookup).

        Args:
            minid: DatasetMinid with bag URL and cache key (in checksum field).
            use_minid: If True, source is S3. If False, source is local file://.

        Returns:
            Path to the extracted bag directory: ``{cache_dir}/{key}/Dataset_{rid}``
        """
        # Check if the bag is already cached under the key provided by _get_dataset_minid.
        # For Tier 1 hits, this always succeeds (the directory was found by glob).
        # For Tier 2/3 first downloads, this is a miss and we proceed to download.
        bag_dir = self._ml_instance.cache_dir / f"{minid.dataset_rid}_{minid.checksum}"
        if bag_dir.exists():
            self._logger.info(f"Using cached bag for {minid.dataset_rid} Version:{minid.dataset_version}")
            return Path(bag_dir / f"Dataset_{minid.dataset_rid}")

        # ----- Download the archive -------------------------------------------
        with TemporaryDirectory() as tmp_dir:
            if use_minid:
                # Tier 2: Download bag archive from S3
                bag_path = Path(tmp_dir) / Path(urlparse(minid.bag_url).path).name
                archive_path = fetch_single_file(minid.bag_url, output_path=bag_path)
            elif minid.bag_url.startswith("file://"):
                # Tier 3: Client-side bag — already on local filesystem
                archive_path = urlparse(minid.bag_url).path
            else:
                # Legacy: Download from catalog export endpoint
                exporter = DerivaExport(host=self._ml_instance.catalog.deriva_server.server, output_dir=tmp_dir)
                archive_path = exporter.retrieve_file(minid.bag_url)

            # For non-MINID downloads without a pre-computed cache key (legacy
            # code path), fall back to SHA-256 of the archive as cache key.
            if not use_minid and not minid.checksum:
                hashes = hash_utils.compute_file_hashes(archive_path, hashes=["md5", "sha256"])
                checksum = hashes["sha256"][0]
                bag_dir = self._ml_instance.cache_dir / f"{minid.dataset_rid}_{checksum}"
                if bag_dir.exists():
                    self._logger.info(f"Using cached bag for {minid.dataset_rid} Version:{minid.dataset_version}")
                    return Path(bag_dir / f"Dataset_{minid.dataset_rid}")

            # ----- Extract to staging directory (atomic cache population) ------
            # Write to a temporary staging directory first. Only rename to the
            # final cache location after successful extraction and validation.
            # This prevents partial/corrupt cache entries if the process crashes.
            staging_dir = self._ml_instance.cache_dir / f"{bag_dir.name}_staging"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            try:
                extracted_bag_path = bdb.extract_bag(archive_path, staging_dir.as_posix())
                bdb.validate_bag_structure(extracted_bag_path)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

        # Atomic move: staging → final cache location.
        staging_dir.rename(bag_dir)

        # Clean up the client_export temp directory for local file:// bags.
        # After extraction to cache, the original archive is no longer needed.
        if not use_minid and minid.bag_url.startswith("file://"):
            export_dir = Path(archive_path).parent
            if "client_export" in export_dir.parts:
                shutil.rmtree(export_dir, ignore_errors=True)

        return Path(bag_dir / f"Dataset_{minid.dataset_rid}")

    def _create_dataset_minid(
        self,
        version: DatasetVersion,
        use_minid: bool = True,
        exclude_tables: set[str] | None = None,
        spec: dict | None = None,
        spec_hash: str | None = None,
        timeout: tuple[int, int] | None = None,
    ) -> str:
        """Create a new MINID (Minimal Viable Identifier) for the dataset.

        This method generates a BDBag export of the dataset and optionally
        registers it with a MINID service for persistent identification.
        The bag is uploaded to S3 storage when using MINIDs.

        Args:
            version: The dataset version to create a MINID for.
            use_minid: If True, register with MINID service and upload to S3.
                If False, just generate the bag and return a local URL.
            exclude_tables: Optional set of table names to exclude from FK traversal.
            spec: Optional pre-computed download spec dict. If None, the spec is
                generated from the snapshot catalog.
            spec_hash: Optional pre-computed SHA-256 hash of the spec. If None and
                spec is provided, it is computed from the spec.
            timeout: Optional (connect_timeout, read_timeout) in seconds for network
                requests. Defaults to (10, 610).

        Returns:
            str: URL to the MINID landing page (if use_minid=True) or
                the direct bag download URL.
        """
        with TemporaryDirectory() as tmp_dir:
            # Generate spec if not supplied (allows callers to reuse a spec they already computed).
            if spec is None:
                version_snapshot_catalog = self._version_snapshot_catalog(version)
                downloader = CatalogGraph(
                    version_snapshot_catalog,
                    s3_bucket=self._ml_instance.s3_bucket,
                    use_minid=use_minid,
                    exclude_tables=exclude_tables,
                )
                spec = downloader.generate_dataset_download_spec(self)

            if spec_hash is None:
                spec_hash = _hash_spec(spec)

            spec_file = Path(tmp_dir) / "download_spec.json"
            with spec_file.open("w", encoding="utf-8") as ds:
                json.dump(spec, ds)

            self._logger.info(
                "Downloading dataset %s for catalog: %s@%s"
                % (
                    "minid" if use_minid else "bag",
                    self.dataset_rid,
                    str(version),
                )
            )

            if use_minid:
                # Server-side export: generates bag, uploads to S3, registers MINID.
                try:
                    exporter = DerivaExport(
                        host=self._ml_instance.catalog.deriva_server.server,
                        config_file=spec_file,
                        output_dir=tmp_dir,
                        defer_download=True,
                        timeout=timeout or (10, 610),
                        envars={"RID": self.dataset_rid},
                    )
                    minid_page_url = exporter.export()[0]
                except (
                    DerivaDownloadError,
                    DerivaDownloadConfigurationError,
                    DerivaDownloadAuthenticationError,
                    DerivaDownloadAuthorizationError,
                    DerivaDownloadTimeoutError,
                ) as e:
                    raise DerivaMLException(format_exception(e))
                # Update version table with MINID and spec hash.
                version_path = (
                    self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Version"]
                )
                version_rid = [h for h in self.dataset_history() if h.dataset_version == version][0].version_rid
                version_path.update([{"RID": version_rid, "Minid": minid_page_url, "Minid_Spec_Hash": spec_hash}])
                return minid_page_url
            else:
                # Client-side download: runs queries locally with paged query support
                # for automatic retry on query timeout errors. This avoids server-side
                # export lock contention and gives better control over query execution.
                return self._create_dataset_bag_client(version, spec, timeout=timeout)

    def _create_dataset_bag_client(
        self, version: DatasetVersion, spec: dict, timeout: tuple[int, int] | None = None
    ) -> str:
        """Create a dataset bag using client-side download.

        Executes ERMrest queries directly using ErmrestCatalog.get_as_file() with
        paged query support, building a BDBag from the results.

        If any CSV data query fails (e.g., due to server-side query timeouts on deep
        multi-table joins), the method raises a DerivaMLException listing the failed
        tables and suggesting that the user add those records as direct dataset members.

        Args:
            version: The dataset version to export.
            spec: The download specification dict (from generate_dataset_download_spec).

        Returns:
            str: A file:// URI pointing to the generated bag zip archive.

        Raises:
            DerivaMLException: If any data query fails during export.
        """
        import csv
        import codecs
        import uuid

        from deriva.core import DerivaServer, get_credential, DEFAULT_SESSION_CONFIG

        snapshot_catalog_id = self._version_snapshot_catalog_id(version)
        hostname = self._ml_instance.catalog.deriva_server.server
        protocol = self._ml_instance.catalog.deriva_server.scheme

        # Connect to the snapshot catalog with optional custom timeout
        credentials = get_credential(hostname)
        session_config = None
        if timeout:
            session_config = dict(DEFAULT_SESSION_CONFIG)
            session_config["timeout"] = timeout
        server = DerivaServer(protocol, hostname, credentials=credentials, session_config=session_config)
        catalog = server.connect_ermrest(snapshot_catalog_id)

        # Build bag in a persistent directory (survives for _download_dataset_minid)
        tmp_dir = Path(self._ml_instance.working_dir) / "client_export" / str(uuid.uuid4())[:8]
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Format environment variables
        envars = {"RID": self.dataset_rid}
        bag_config = spec.get("bag", {})
        bag_name = bag_config.get("bag_name", f"Dataset_{self.dataset_rid}").format(**envars)
        bag_path = tmp_dir / bag_name
        bag_algorithms = bag_config.get("bag_algorithms", ["md5"])

        # Create the bag
        bdb.ensure_bag_path_exists(str(bag_path))
        bag = bdb.make_bag(str(bag_path), algs=bag_algorithms, idempotent=True)

        # Process query_processors from the spec
        query_processors = spec.get("catalog", {}).get("query_processors", [])
        failed_queries = []
        skipped_empty = []
        fetch_entries: dict[str, tuple[str, str, str, str]] = {}  # asset_rid -> (url, length, rel_path, md5)

        for qp in query_processors:
            processor_name = qp.get("processor", "")
            params = qp.get("processor_params", {})

            if processor_name == "env":
                # Environment variable processors — execute and capture values
                query_path = params.get("query_path", "")
                if not query_path:
                    continue
                query_path = query_path.format(**envars)
                query_keys = params.get("query_keys", [])
                try:
                    if query_path == "/":
                        # Root query returns catalog metadata including snaptime
                        resp = catalog.get("/").json()
                    else:
                        resp = catalog.get(query_path).json()
                    if isinstance(resp, list) and resp:
                        resp = resp[0]
                    if resp and query_keys:
                        for key in query_keys:
                            if key in resp:
                                envars[key] = resp[key]
                except Exception as e:
                    self._logger.warning("Failed to execute env query %s: %s", query_path, e)

            elif processor_name == "json":
                # JSON query (e.g., schema dump)
                query_path = params.get("query_path", "")
                output_path = params.get("output_path", "")
                if not query_path:
                    continue
                query_path = query_path.format(**envars)
                # Output path becomes filename with .json extension
                dest_file = bag_path / "data" / (output_path + ".json")
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    resp = catalog.get(query_path).json()
                    dest_file.write_text(json.dumps(resp, indent=2), encoding="utf-8")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download {output_path} from snapshot catalog ({query_path}): {e}"
                    ) from e

            elif processor_name == "csv":
                # Data query — use paged mode for resilience
                query_path = params.get("query_path", "")
                output_path = params.get("output_path", "")
                if not query_path:
                    continue
                query_path = query_path.format(**envars)
                paged = params.get("paged_query", False)

                dest_dir = bag_path / "data" / output_path
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = str(dest_dir) + ".csv"

                try:
                    catalog.get_as_file(
                        query_path,
                        dest_file,
                        headers={"accept": "text/csv"},
                        delete_if_empty=True,
                        paged=paged,
                        page_size=100000,
                    )
                    if not os.path.isfile(dest_file):
                        skipped_empty.append(output_path)
                except Exception as e:
                    # Tolerate individual query failures — log and continue.
                    # This handles snapshot catalog timeouts for large joins.
                    self._logger.warning(
                        "Query failed for %s (will be missing from bag): %s",
                        output_path,
                        e,
                    )
                    failed_queries.append(output_path)
                    # Clean up partial file if it exists
                    if os.path.isfile(dest_file):
                        os.remove(dest_file)

            elif processor_name == "fetch":
                # Asset file references — write entries to fetch.txt for lazy materialization.
                # The actual binary files are downloaded later by bdbag.materialize() when
                # materialize=True is set on download_dataset_bag().
                query_path = params.get("query_path", "")
                output_path = params.get("output_path", "")
                if not query_path:
                    continue
                query_path = query_path.format(**envars)

                try:
                    resp = catalog.get(query_path).json()
                    for record in resp:
                        url = record.get("url")
                        filename = record.get("filename", "unknown")
                        length = record.get("length", "")
                        md5 = record.get("md5", "")
                        asset_rid = record.get("asset_rid", "unknown")
                        if not url:
                            continue
                        # Build the full URL for the asset
                        if url.startswith("/"):
                            asset_url = f"{protocol}://{hostname}{url}"
                        else:
                            asset_url = url
                        # Build relative path within bag data directory
                        file_output_path = output_path.format(asset_rid=asset_rid)
                        rel_path = f"data/{file_output_path}/{filename}"
                        # Deduplicate by asset RID — the same asset may be
                        # reachable via multiple FK paths; keep the first entry
                        # (all paths produce the same URL and file content).
                        if asset_rid not in fetch_entries:
                            fetch_entries[asset_rid] = (asset_url, length, rel_path, md5)
                except Exception as e:
                    self._logger.warning("Asset query failed for %s: %s", output_path, e)

        # Remove empty directories left behind by empty/failed queries
        for dirpath, dirnames, filenames in os.walk(str(bag_path / "data"), topdown=False):
            if not dirnames and not filenames:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass

        if failed_queries:
            # Extract table names from output paths (format: "schema/table")
            failed_tables = [q.rsplit("/", 1)[-1] if "/" in q else q for q in failed_queries]
            raise DerivaMLException(
                f"Dataset bag export failed: {len(failed_queries)} queries timed out or "
                f"failed for tables: {failed_tables}. "
                f"This typically happens when deep multi-table joins exceed server query "
                f"time limits. To fix this, add the desired records as direct dataset "
                f"members using add_dataset_members() with the relevant table's RIDs. "
                f"For example, if Image data is missing, register Image as a dataset "
                f"element type (add_dataset_element_type('Image')) and add Image RIDs "
                f"as members so they are exported via a direct association path rather "
                f"than a deep FK join. Failed paths: {failed_queries}"
            )

        # Write remote file manifest for BDBag to generate fetch.txt.
        # The manifest must be a JSON-stream file (one JSON object per line)
        # with url, length, filename (without data/ prefix), and a hash.
        # Passing it to make_bag(remote_file_manifest=...) ensures fetch.txt
        # is generated correctly and not destroyed by make_bag(update=True).
        remote_manifest_path = None
        if fetch_entries:
            remote_manifest_path = str(bag_path / "remote-file-manifest.json")
            with open(remote_manifest_path, "w", encoding="utf-8") as f:
                for url, length, rel_path, md5 in fetch_entries.values():
                    # rel_path has "data/" prefix; bdbag expects filename without it
                    filename = rel_path.removeprefix("data/")
                    entry = {
                        "url": url,
                        "length": int(length) if length else 0,
                        "filename": filename,
                    }
                    if md5:
                        entry["md5"] = md5
                    f.write(json.dumps(entry) + "\n")
            self._logger.info("Wrote %d remote file manifest entries", len(fetch_entries))

        # Update and archive the bag
        bdb.make_bag(
            str(bag_path),
            algs=bag_algorithms,
            remote_file_manifest=remote_manifest_path,
            update=True,
            idempotent=True,
        )
        archive_path = bdb.archive_bag(str(bag_path), bag_config.get("bag_archiver", "zip"))
        return Path(archive_path).as_uri()

    def _get_dataset_minid(
        self,
        version: DatasetVersion,
        create: bool,
        use_minid: bool,
        exclude_tables: set[str] | None = None,
        timeout: tuple[int, int] | None = None,
    ) -> DatasetMinid | None:
        """Locate or create a dataset bag, using a three-tier caching strategy.

        The download algorithm proceeds through three tiers, stopping at the
        first success. This applies identically to both MINID and non-MINID paths:

        **Tier 1 — Local deterministic cache (filesystem lookup, no network beyond spec)**

        The cache key is ``{rid}_{spec_hash[:16]}_{snapshot}``, combining:
        - **spec_hash**: SHA-256 of the download spec (captures FK traversal plan).
          Changes when the schema changes (new tables, new FKs).
        - **snapshot**: Immutable catalog snapshot ID (captures data state).

        Both must match for a cache hit — a snapshot-only match would return
        stale bags created before schema changes.

        Cost: one schema introspection query (to compute spec_hash) + one stat.

        **Tier 2 — MINID / S3 (when ``use_minid=True``)**

        If no local cache exists, check whether a MINID (Minimal Viable
        Identifier) was previously registered for this version.  The stored
        ``Minid_Spec_Hash`` is compared to the current download spec hash:

        - Match → fetch MINID metadata (HTTP GET), download bag from S3.
        - Mismatch or missing → regenerate bag server-side, upload to S3,
          register new MINID.

        The spec hash detects schema or FK-path changes that would alter bag
        contents even for the same snapshot.

        **Tier 3 — Client-side bag generation (when ``use_minid=False``)**

        Build the bag locally by running ERMrest queries against the snapshot
        catalog. The bag is stored under ``{rid}_{spec_hash[:16]}_{snapshot}``
        so Tier 1 finds it on subsequent calls.

        Args:
            version: The dataset version to download.
            create: If True, create a new bag/MINID when none is cached.
                If False, raise an exception when nothing is available.
            use_minid: If True, use S3 + MINID service (Tier 2) on cache miss.
                If False, generate bag client-side (Tier 3) on cache miss.
            exclude_tables: Table names to exclude from FK path traversal.
            timeout: Optional (connect_timeout, read_timeout) in seconds.

        Returns:
            DatasetMinid with the bag URL (local ``file://`` or S3) and a
            checksum that doubles as the cache directory suffix.

        Raises:
            DerivaMLException: If the version doesn't exist, or if
                ``create=False`` and no cached/registered bag is available.
        """
        # ----- Resolve version record -----------------------------------------
        version_str = str(version)
        history = self.dataset_history()
        try:
            version_record = next(v for v in history if v.dataset_version == version_str)
        except StopIteration:
            raise DerivaMLException(f"Version {version_str} does not exist for RID {self.dataset_rid}")

        snapshot = version_record.snapshot
        minid_url = version_record.minid

        # =====================================================================
        # Compute spec_hash upfront (required for all tiers).
        #
        # The download spec defines which FK paths and tables are included in
        # the bag. If the schema changes (new tables, new FKs), the spec_hash
        # changes even for the same snapshot. We MUST include the spec_hash in
        # the cache key to avoid returning stale bags that are missing tables
        # added after the cached bag was created.
        #
        # Cost: one schema introspection query (no data queries). This is
        # cheap and necessary for correctness.
        # =====================================================================
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        downloader = CatalogGraph(
            version_snapshot_catalog,
            s3_bucket=self._ml_instance.s3_bucket,
            use_minid=use_minid,
            exclude_tables=exclude_tables,
        )
        spec = downloader.generate_dataset_download_spec(self)
        spec_hash = _hash_spec(spec)

        # The deterministic cache key: {spec_hash[:16]}_{snapshot}
        # - spec_hash[:16] captures the FK traversal plan (schema-dependent)
        # - snapshot captures the catalog data state (immutable point-in-time)
        # Together they uniquely identify the bag contents.
        cache_suffix = f"{spec_hash[:16]}_{snapshot}"

        # =====================================================================
        # Tier 1: Local deterministic cache (filesystem lookup, no network).
        #
        # Look for a cached bag with BOTH the same spec_hash and snapshot.
        # A snapshot-only match would return stale bags created before schema
        # changes (e.g., new tables added to the FK traversal).
        # =====================================================================
        cache_dir_name = f"{self.dataset_rid}_{cache_suffix}"
        cached_dir = self._ml_instance.cache_dir / cache_dir_name
        cached_bag_path = cached_dir / f"Dataset_{self.dataset_rid}"
        if cached_bag_path.exists():
            self._logger.info(
                "Local cache hit for %s version %s (spec+snapshot match: %s)",
                self.dataset_rid,
                version,
                cache_dir_name,
            )
            return DatasetMinid(
                dataset_version=version,
                RID=f"{self.dataset_rid}@{snapshot}",
                location=cached_bag_path.parent.as_uri(),
                checksums=[{"function": "sha256", "value": cache_suffix}],
            )

        # =====================================================================
        # Tier 2: MINID / S3 download (use_minid=True only).
        #
        # Compare spec_hash to the stored Minid_Spec_Hash. If they match,
        # the S3 bag is still current. If not, regenerate.
        # =====================================================================
        if use_minid:
            if minid_url and version_record.spec_hash == spec_hash:
                # S3 bag is current — download it (populates local cache for Tier 1).
                return self._fetch_minid_metadata(version, minid_url)

            # No MINID, or spec has changed — need to regenerate.
            if not create:
                raise DerivaMLException(f"Minid for dataset {self.dataset_rid} doesn't exist")
            if minid_url:
                self._logger.info(
                    "Spec hash changed for dataset %s version %s — regenerating MINID bag.",
                    self.dataset_rid,
                    version,
                )
            else:
                self._logger.info("Creating new MINID for dataset %s", self.dataset_rid)
            minid_url = self._create_dataset_minid(
                version,
                use_minid=True,
                exclude_tables=exclude_tables,
                spec=spec,
                spec_hash=spec_hash,
                timeout=timeout,
            )
            return self._fetch_minid_metadata(version, minid_url)

        # =====================================================================
        # Tier 3: Client-side bag generation (use_minid=False).
        #
        # Build the bag locally. Store under the deterministic cache key
        # {rid}_{spec_hash[:16]}_{snapshot} so Tier 1 finds it next time.
        # =====================================================================
        if not create and not minid_url:
            raise DerivaMLException(f"Minid for dataset {self.dataset_rid} doesn't exist")

        self._logger.info(
            "Cache miss for %s version %s — generating bag client-side",
            self.dataset_rid,
            version,
        )
        minid_url = self._create_dataset_minid(
            version,
            use_minid=False,
            exclude_tables=exclude_tables,
            spec=spec,
            spec_hash=spec_hash,
            timeout=timeout,
        )
        return DatasetMinid(
            dataset_version=version,
            RID=f"{self.dataset_rid}@{snapshot}",
            location=minid_url,
            checksums=[{"function": "sha256", "value": cache_suffix}],
        )

    def _fetch_minid_metadata(self, version: DatasetVersion, url: str) -> DatasetMinid:
        """Fetch MINID metadata from the MINID service.

        Args:
            version: The dataset version associated with this MINID.
            url: The MINID landing page URL.

        Returns:
            DatasetMinid: Parsed metadata including bag URL, checksum, and identifiers.

        Raises:
            requests.HTTPError: If the MINID service request fails.
        """
        r = requests.get(url, headers={"accept": "application/json"})
        r.raise_for_status()
        return DatasetMinid(dataset_version=version, **r.json())

    @staticmethod
    def _bag_is_fully_materialized(bag_path: Path) -> bool:
        """Check whether all fetch.txt entries have been downloaded locally.

        Uses bdbag's validate_bag_structure for a quick structural check, then
        verifies that every file referenced in fetch.txt is present on disk.

        Args:
            bag_path: Path to the BDBag directory.

        Returns:
            True if the bag has no fetch.txt or all fetch.txt entries exist locally.
            False if any referenced file is missing.
        """
        try:
            bdb.validate_bag_structure(bag_path.as_posix())
        except Exception as e:
            logging.getLogger("deriva_ml").debug(f"Bag validation check failed for {bag_path}: {e}")
            return False
        fetch_file = bag_path / "fetch.txt"
        if not fetch_file.exists():
            return True
        with fetch_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    rel_path = parts[2]
                    if not (bag_path / rel_path).exists():
                        return False
        return True

    def _materialize_dataset_bag(
        self,
        minid: DatasetMinid,
        use_minid: bool,
        fetch_concurrency: int = 1,
    ) -> Path:
        """Materialize a dataset bag by downloading all referenced files.

        This method downloads a BDBag and then "materializes" it by fetching
        all files referenced in the bag's fetch.txt manifest. This includes
        data files, assets, and any other content referenced by the bag.

        Progress is reported through callbacks that update the execution status
        if this download is associated with an execution.

        Args:
            minid: DatasetMinid containing the bag URL and metadata.
            use_minid: If True, download from S3 using the MINID URL.

        Returns:
            Path: The path to the fully materialized bag directory.

        Note:
            Materialization status is cached via a 'validated_check.txt' marker
            file to avoid re-downloading already-materialized bags.
        """

        def fetch_progress_callback(current, total):
            msg = f"Materializing bag: {current} of {total} file(s) downloaded."
            self._logger.info(msg)
            return True

        def validation_progress_callback(current, total):
            msg = f"Validating bag: {current} of {total} file(s) validated."
            self._logger.info(msg)
            return True

        # request metadata
        bag_path = self._download_dataset_minid(minid, use_minid)
        bag_dir = bag_path.parent
        validated_check = bag_dir / "validated_check.txt"

        # If this bag has already been validated, verify completeness using bdbag before trusting the cache.
        # This guards against caches that were marked valid but have missing fetch.txt assets.
        if validated_check.exists():
            if self._bag_is_fully_materialized(bag_path):
                self._logger.info(
                    f"Cached bag {minid.dataset_rid} Version:{minid.dataset_version} verified as complete."
                )
                return Path(bag_path)
            else:
                self._logger.warning(
                    f"Cached bag {minid.dataset_rid} Version:{minid.dataset_version} is incomplete "
                    f"(fetch.txt entries missing). Re-materializing."
                )
                validated_check.unlink(missing_ok=True)

        self._logger.info(f"Materializing bag {minid.dataset_rid} Version:{minid.dataset_version}")
        # Ensure parent directories exist for all fetch entries
        fetch_file = bag_path / "fetch.txt"
        if fetch_file.exists():
            with fetch_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        rel_path = parts[2]
                        (bag_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
        bdb.materialize(
            bag_path.as_posix(),
            fetch_callback=fetch_progress_callback,
            validation_callback=validation_progress_callback,
            fetch_concurrency=fetch_concurrency,
        )
        validated_check.touch()
        return Path(bag_path)

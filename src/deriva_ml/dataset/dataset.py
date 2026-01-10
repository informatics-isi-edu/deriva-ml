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
    >>> ml = DerivaML('deriva.example.org', 'my_catalog')
    >>> dataset_rid = ml.create_dataset('experiment', 'Experimental data')
    >>> ml.add_dataset_members(dataset_rid=dataset_rid, members=['1-abc123', '1-def456'])
    >>> ml.increment_dataset_version(dataset_rid=dataset_rid, component=VersionPart.minor,
    ...     description='Added new samples')
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

# Standard library imports
from graphlib import TopologicalSorter
from pathlib import Path

# Local imports
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import Any, Generator, Iterable, Self
from urllib.parse import urlparse

# Deriva imports
import deriva.core.utils.hash_utils as hash_utils
import requests

# Third-party imports
import pandas as pd
from bdbag import bdbag_api as bdb
from bdbag.fetch.fetcher import fetch_single_file
from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import format_exception
from deriva.transfer.download.deriva_download import (
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
    Status,
    VocabularyTerm,
)
from deriva_ml.core.exceptions import DerivaMLException
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


class Dataset:
    """Manages dataset operations in a Deriva catalog.

    The Dataset class provides functionality for creating, modifying, and tracking datasets
    in a Deriva catalog. It handles versioning, relationships between datasets, and data export.

    Attributes:
        _ml_instance (DerivaModel): Catalog model instance.

    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        catalog: DerivaMLCatalog,
        dataset_rid: RID,
        dataset_types: str | list[str] | None = None,
        description: str = "",
        execution_rid: RID | None = None,
    ):
        """Creates a new dataset in the catalog.

        Creates a dataset with specified types and description. The dataset can be associated
        with an execution and initialized with a specific version.

        If no version is specified, the current version of the dataset in the current catalog snapshot is used.
        If a version is specified, the dataset is initialized with that version.

        Args:
            dataset_types: One or more dataset type terms from Dataset_Type vocabulary.
            description: Description of the dataset's purpose and contents.
            execution_rid: Optional execution RID to associate with dataset creation.

        Returns:
            RID: Resource Identifier of the newly created dataset.

        Raises:
            DerivaMLException: If dataset_types are invalid or creation fails.

        Example:
            >>> rid = ml.create_dataset(
            ...     dataset_types=["experiment", "raw_data"],
            ...     description="RNA sequencing experiment data",
            ...     version=DatasetVersion(1, 0, 0)
            ... )
        """
        self._logger = logging.getLogger("deriva_ml")
        self.dataset_rid = dataset_rid
        self.execution_rid = execution_rid
        self._ml_instance = catalog
        self.description = description

        # Normalize dataset_types to always be a list of strings
        if dataset_types is None:
            self.dataset_types: list[str] = []
        elif isinstance(dataset_types, str):
            self.dataset_types: list[str] = [dataset_types]
        else:
            self.dataset_types: list[str] = dataset_types

    def __repr__(self) -> str:
        return (f"<deriva_ml.Dataset object at {hex(id(self))}: rid='{self.dataset_rid}', "
                f"version='{self.current_version}', types={self.dataset_types}>")

    def __hash__(self) -> int:
        """Hash based on dataset RID for use in sets and as dict keys."""
        return hash(self.dataset_rid)

    def __eq__(self, other: object) -> bool:
        """Two Dataset objects are equal if they have the same RID."""
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.dataset_rid == other.dataset_rid

    @staticmethod
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_dataset(
        ml_instance: DerivaMLCatalog,
        dataset_types: str | list[str] | None = None,
        description: str = "",
        execution_rid: RID | None = None,
        version: DatasetVersion | None = None,
    ) -> Self:
        """Creates a new dataset in the catalog.

        Creates a dataset with specified types and description. The dataset can be associated
        with an execution and initialized with a specific version.

        Args:
            ml_instance: DerivaMLCatalog instance.
            dataset_types: One or more dataset type terms from Dataset_Type vocabulary.
            description: Description of the dataset's purpose and contents.
            execution_rid: Optional execution RID to associate with dataset creation.
            version: Optional initial version number. Defaults to 0.1.0.

        Returns:
            RID: Resource Identifier of the newly created dataset.

        Raises:
            DerivaMLException: If dataset_types are invalid or creation fails.

        Example:
            >>> rid = ml.create_dataset(
            ...     dataset_types=["experiment", "raw_data"],
            ...     description="RNA sequencing experiment data",
            ...     version=DatasetVersion(1, 0, 0)
            ... )
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

        if execution_rid is not None:
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

        dataset.add_dataset_types(dataset_types)
        return dataset

    def add_dataset_types(self, dataset_types: str | VocabularyTerm | list[str | VocabularyTerm]) -> None:
        """Adds one or more dataset types to an existing dataset.

        Args:
            dataset_types: Single term or list of terms. Can be strings (term names) or VocabularyTerm objects.
        """

        # Normalize input to a list
        types_to_add = [dataset_types] if not isinstance(dataset_types, list) else dataset_types

        # Convert all to VocabularyTerm objects and collect new ones to insert
        new_terms = []
        for term in types_to_add:
            # If it's already a VocabularyTerm, use it; otherwise look it up by name
            if isinstance(term, VocabularyTerm):
                vocab_term = term
            else:
                vocab_term = self._ml_instance.lookup_term(MLVocab.dataset_type, term)

            # Check if this term is already associated with the dataset
            # Store as string names in self.dataset_types for consistency with __init__
            term_name = vocab_term.name
            if term_name not in self.dataset_types:
                new_terms.append(vocab_term)
                # dataset_types is always a list now
                self.dataset_types.append(term_name)

        # Only insert if there are new terms to add
        if new_terms:
            # Get the name of the association table between dataset_table and dataset_type.
            associations = list(
                self._ml_instance.model.schemas[self._ml_instance.ml_schema]
                .tables[MLVocab.dataset_type]
                .find_associations()
            )
            pb = self._ml_instance.pathBuilder()
            atable = associations[0].name if associations else None
            pb.schemas[self._ml_instance.ml_schema].tables[atable].insert(
                [{MLVocab.dataset_type: term.name, "Dataset": self.dataset_rid} for term in new_terms]
            )

    @property
    def _dataset_table(self) -> Table:
        return self._ml_instance.model.schemas[self._ml_instance.ml_schema].tables["Dataset"]

    # ==================== Read Interface Methods ====================
    # These methods implement the DatasetLike protocol for read operations.
    # They delegate to the catalog instance for actual data retrieval.

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of elements that can be contained in this dataset.

        Returns:
            Iterable of Table objects representing element types.
        """
        return self._ml_instance.list_dataset_element_types()

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features associated with a table.

        Args:
            table: Table to find features for.

        Returns:
            Iterable of Feature objects.
        """
        return self._ml_instance.find_features(table)

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
            >>> history = ml.dataset_history("1-abc123")
            >>> for entry in history:
            ...     print(f"Version {entry.dataset_version}: {entry.description}")
        """

        if not self._ml_instance.model.is_dataset_rid(self.dataset_rid):
            raise DerivaMLException(f"RID is not for a data set: {self.dataset_rid}")
        version_path = self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Version"]
        return [
            DatasetHistory(
                dataset_version=DatasetVersion.parse(v["Version"]),
                minid=v["Minid"],
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
        """Retrieve the current version of the specified dataset_table.

        Return the most recent version of the dataset. It is important to remember that this version
        captures the state of the catalog at the time the version was created, not the current state of the catalog.
        This means that its possible that the values associated with an object in the catalog may be different
        from the values of that object in the dataset.

        Returns:
            A tuple with the semantic version of the dataset_table.
        """
        history = self.dataset_history()
        if not history:
            return DatasetVersion(0, 1, 0)
        else:
            # Ensure we return a DatasetVersion, not a string
            versions = [h.dataset_version for h in history]
            return max(versions) if versions else DatasetVersion(0, 1, 0)

    def _build_dataset_graph(self) -> Iterable[Dataset]:
        """Build a dependency graph of all related datasets and return in topological order.

        Returns datasets in an order where children come before parents, ensuring
        that when versions are incremented, all related datasets are updated together.
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
            >>> new_version = ml.increment_dataset_version(
            ...     dataset_rid="1-abc123",
            ...     component="minor",
            ...     description="Added new samples"
            ... )
            >>> print(f"New version: {new_version}")  # e.g., "1.2.0"
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
    def list_dataset_members(self, version: DatasetVersion | str | None = None,
                             recurse: bool = False,
                             limit: int | None = None,
                             _visited: set[RID] | None = None) -> dict[str, list[dict[str, Any]]]:
        """Lists members of a dataset.

        Returns a dictionary mapping member types to lists of member records. Can optionally
        recurse through nested datasets and limit the number of results.

        Args:
            version: Dataset version to list members from. Defaults to the current version.
            recurse: Whether to include members of nested datasets. Defaults to False.
            limit: Maximum number of members to return per type. None for no limit.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.

        Returns:
            dict[str, list[dict[str, Any]]]: Dictionary mapping member types to lists of members.
                Each member is a dictionary containing the record's attributes.

        Raises:
            DerivaMLException: If dataset_rid is invalid.

        Example:
            >>> members = ml.list_dataset_members("1-abc123", recurse=True)
            >>> for type_name, records in members.items():
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
            if target_table.schema.name != self._ml_instance.domain_schema and not (
                target_table == self._dataset_table or target_table.name == "File"
            ):
                continue
            member_column = (
                "Nested_Dataset" if target_table == self._dataset_table else other_fkey.foreign_key_columns[0].name
            )

            target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
            member_path = pb.schemas[member_table.schema.name].tables[member_table.name]

            path = member_path.filter(member_path.Dataset == self.dataset_rid).link(
                target_path,
                on=(member_path.columns[member_column] == target_path.columns["RID"]),
            )
            target_entities = list(path.entities().fetch(limit=limit) if limit else path.entities().fetch())
            members[target_table.name].extend(target_entities)
            if recurse and target_table == self._dataset_table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds_rid in nested_datasets:
                    ds = version_snapshot_catalog.lookup_dataset(ds_rid)
                    for k, v in ds.list_dataset_members(version=version, recurse=recurse, _visited=_visited).items():
                        members[k].extend(v)
        return dict(members)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_members(
        self,
        members: list[RID] | dict[str, list[RID]],
        validate: bool = True,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Adds members to a dataset.

        Associates one or more records with a dataset. Can optionally validate member types
        and create a new dataset version to track the changes.

        Args:
            members: List of RIDs to add as dataset members. Can be organized into a dictionary that indicates the
                table that the member rids belong to.
            validate: Whether to validate member types. Defaults to True.
            description: Optional description of the member additions.
            execution_rid: Optional execution RID to associate with changes.

        Raises:
            DerivaMLException: If:
                - dataset_rid is invalid
                - members are invalid or of wrong type
                - adding members would create a cycle
                - validation fails

        Example:
            >>> ml.add_dataset_members(
            ...     dataset_rid="1-abc123",
            ...     members=["1-def456", "1-ghi789"],
            ...     description="Added sample data"
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
        dataset_elements = {}
        association_map = {
            a.other_fkeys.pop().pk_table.name: a.table.name for a in self._dataset_table.find_associations()
        }

        # Get a list of all the object types that can be linked to a dataset_table.
        if type(members) is list:
            members = set(members)
            for m in members:
                try:
                    rid_info = self._ml_instance.resolve_rid(m)
                except KeyError:
                    raise DerivaMLException(f"Invalid RID: {m}")
                if rid_info.table.name not in association_map:
                    raise DerivaMLException(f"RID table: {rid_info.table.name} not part of dataset_table")
                if rid_info.table == self._dataset_table and check_dataset_cycle(rid_info.rid):
                    raise DerivaMLException("Creating cycle of datasets is not allowed")
                dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        else:
            dataset_elements = {t: set(ms) for t, ms in members.items()}
        # Now make the entries into the association tables.
        pb = self._ml_instance.pathBuilder()
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[
                self._ml_instance.ml_schema
                if (table == "Dataset" or table == "File")
                else self._ml_instance.domain_schema
            ]
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
        dataset_rid: RID,
        members: list[RID],
        description: str = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Remove elements to an existing dataset_table.

        Delete elements from an existing dataset. In addition to deleting members, the minor version number of the
        dataset is incremented and the description, if provide is applied to that new version.

        Args:
            dataset_rid: RID of dataset_table to extend or None if a new dataset_table is to be created.
            members: List of member RIDs to add to the dataset_table.
            description: Markdown description of the updated dataset.
            execution_rid: Optional RID of execution associated with this operation.
        """

        members = set(members)
        description = description or "Deletes dataset members"

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        association_map = {
            a.other_fkeys.pop().pk_table.name: a.table.name for a in self._dataset_table.find_associations()
        }
        # Get a list of all the object types that can be linked to a dataset_table.
        for m in members:
            try:
                rid_info = self._ml_instance.resolve_rid(m)
            except KeyError:
                raise DerivaMLException(f"Invalid RID: {m}")
            if rid_info.table.name not in association_map:
                raise DerivaMLException(f"RID table: {rid_info.table.name} not part of dataset_table")
            dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        pb = self._ml_instance.pathBuilder()
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[
                self._ml_instance.ml_schema if table == "Dataset" else self._ml_instance.domain_schema
            ]
            fk_column = "Nested_Dataset" if table == "Dataset" else table

            if len(elements):
                atable_path = schema_path.tables[association_map[table]]
                # Find out the name of the column in the association table.
                for e in elements:
                    entity = atable_path.filter(
                        (atable_path.Dataset == dataset_rid) & (atable_path.columns[fk_column] == e),
                    )
                    entity.delete()
        self.increment_dataset_version(
            dataset_rid,
            VersionPart.minor,
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_dataset_parents(self,
                             version: DatasetVersion | str | None = None,
                             recurse: bool = False,
                             _visited: set[RID] | None = None) -> list[Self]:
        """Given a dataset_table RID, return a list of RIDs of the parent datasets if this is included in a
        nested dataset.

        Args:
            version: Dataset version to list parents from. Defaults to the current version.
            recurse: If True, recursively return all ancestor datasets.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.

        Returns:
            List of parent datasets.
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
                parents.extend(parent.list_dataset_parents(version, recurse=True, _visited=_visited))
        return parents

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_dataset_children(self,
                              version: DatasetVersion | str | None = None,
                              recurse: bool = False,
                              _visited: set[RID] | None = None) -> list[Self]:
        """Given a dataset_table RID, return a list of RIDs for any nested datasets.

        Args:
            version: Dataset version to list children from. Defaults to the current version.
            recurse: If True, return a list of nested datasets RIDs.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.

        Returns:
          list of nested dataset RIDs.

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

    @staticmethod
    def _insert_dataset_versions(
        ml_instance: DerivaMLCatalog,
        dataset_list: list[DatasetSpec],
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> None:
        schema_path = ml_instance.pathBuilder().schemas[ml_instance.ml_schema]
        # determine snapshot after changes were made

        # Construct version records for insert
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
        snap = ml_instance.catalog.get("/").json()["snaptime"]
        schema_path.tables["Dataset_Version"].update(
            [{"RID": v["RID"], "Dataset": v["Dataset"], "Snapshot": snap} for v in version_records]
        )

        # And update the dataset records.
        schema_path.tables["Dataset"].update([{"Version": v["RID"], "RID": v["Dataset"]} for v in version_records])

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_dataset_bag(
        self,
        version: DatasetVersion | str,
        materialize: bool = True,
        use_minid: bool = True,
    ) -> DatasetBag:
        """Downloads a dataset to the local filesystem and creates a MINID if needed.

        Downloads a dataset to the local file system.  If the dataset has a version set, that version is used.
        If the dataset has a version and a version is provided, the version specified takes precedence.

        If the dataset doesn't have a MINID (Minimal Viable Identifier), one will be created.
        The dataset can optionally be associated with an execution record.


        Args:
            version: Dataset version to download. If not specified, the version must be set in the dataset.
            materialize: If True, materialize the dataset after downloading.
            use_minid: If True, create a MINID for the dataset if one doesn't already exist.

        Returns:
            DatasetBag: Object containing:
                - path: Local filesystem path to downloaded dataset
                - rid: Dataset's Resource Identifier
                - minid: Dataset's Minimal Viable Identifier

        Examples:
            Download with default options:
                >>> spec = DatasetSpec(rid="1-abc123")
                >>> bag = ml.download_dataset_bag(dataset=spec)
                >>> print(f"Downloaded to {bag.path}")

            Download with execution tracking:
                >>> bag = ml.download_dataset_bag(
                ...     dataset=DatasetSpec(rid="1-abc123", materialize=True),
                ... )
        """
        if isinstance(version, str):
            version = DatasetVersion.parse(version)

        minid = self._get_dataset_minid(version, create=True, use_minid=use_minid)

        bag_path = (
            self._materialize_dataset_bag(minid, use_minid=use_minid)
            if materialize
            else self._download_dataset_minid(minid, use_minid)
        )
        return DatabaseModel(minid, bag_path, self._ml_instance.working_dir).lookup_dataset(self.dataset_rid)

    def _version_snapshot_catalog(self, dataset_version: DatasetVersion | str | None) -> DerivaMLCatalog:
        if isinstance(dataset_version, str) and str:
            dataset_version = DatasetVersion.parse(dataset_version)
        if dataset_version:
            return self._ml_instance.catalog_snapshot(self._version_snapshot_catalog_id(dataset_version))
        else:
            return self._ml_instance

    def _version_snapshot_catalog_id(self, version: DatasetVersion | str) -> str:
        """Return a catalog with snapshot for the specified dataset version"""

        version = str(version)
        try:
            version_record = next(h for h in self.dataset_history() if h.dataset_version == version)
        except StopIteration:
            raise DerivaMLException(f"Dataset version {version} not found for dataset {self.dataset_rid}")
        return (
            f"{self._ml_instance.catalog.catalog_id}@{version_record.snapshot}"
            if version_record.snapshot
            else self._ml_instance.catalog.catalog_id
        )

    def _download_dataset_minid(self, minid: DatasetMinid, use_minid: bool) -> Path:
        """Given a RID to a dataset_table, or a MINID to an existing bag, download the bag file, extract it, and
        validate that all the metadata is correct

        Args:
            minid: The RID of a dataset_table or a minid to an existing bag.
        Returns:
            the location of the unpacked and validated dataset_table bag and the RID of the bag and the bag MINID
        """

        # Check to see if we have an existing idempotent materialization of the desired bag. If so, then reuse
        # it.  If not, then we need to extract the contents of the archive into our cache directory.
        bag_dir = self._ml_instance.cache_dir / f"{minid.dataset_rid}_{minid.checksum}"
        if bag_dir.exists():
            self._logger.info(f"Using cached bag for  {minid.dataset_rid} Version:{minid.dataset_version}")
            return Path(bag_dir / f"Dataset_{minid.dataset_rid}")

        # Either bag hasn't been downloaded yet, or we are not using a Minid, so we don't know the checksum yet.
        with TemporaryDirectory() as tmp_dir:
            if use_minid:
                # Get bag from S3
                bag_path = Path(tmp_dir) / Path(urlparse(minid.bag_url).path).name
                archive_path = fetch_single_file(minid.bag_url, output_path=bag_path)
            else:
                exporter = DerivaExport(host=self._ml_instance.catalog.deriva_server.server, output_dir=tmp_dir)
                archive_path = exporter.retrieve_file(minid.bag_url)
                hashes = hash_utils.compute_file_hashes(archive_path, hashes=["md5", "sha256"])
                checksum = hashes["sha256"][0]
                bag_dir = self._ml_instance.cache_dir / f"{minid.dataset_rid}_{checksum}"
                if bag_dir.exists():
                    self._logger.info(f"Using cached bag for  {minid.dataset_rid} Version:{minid.dataset_version}")
                    return Path(bag_dir / f"Dataset_{minid.dataset_rid}")
            bag_path = bdb.extract_bag(archive_path, bag_dir.as_posix())
        bdb.validate_bag_structure(bag_path)
        return Path(bag_path)

    def _create_dataset_minid(self, version: DatasetVersion, use_minid=True) -> str:
        with TemporaryDirectory() as tmp_dir:
            # Generate a download specification file for the current catalog schema. By default, this spec
            # will generate a minid and place the bag into S3 storage.
            spec_file = Path(tmp_dir) / "download_spec.json"
            version_snapshot_catalog = self._version_snapshot_catalog(version)
            with spec_file.open("w", encoding="utf-8") as ds:
                downloader = CatalogGraph(version_snapshot_catalog, use_minid=use_minid)
                json.dump(downloader.generate_dataset_download_spec(self), ds)
            try:
                self._logger.info(
                    "Downloading dataset %s for catalog: %s@%s"
                    % (
                        "minid" if use_minid else "bag",
                        self.dataset_rid,
                        str(version),
                    )
                )
                # Generate the bag and put into S3 storage.
                exporter = DerivaExport(
                    host=self._ml_instance.catalog.deriva_server.server,
                    config_file=spec_file,
                    output_dir=tmp_dir,
                    defer_download=True,
                    timeout=(10, 610),
                    envars={"RID": self.dataset_rid},
                )
                minid_page_url = exporter.export()[0]  # Get the MINID launch page
            except (
                DerivaDownloadError,
                DerivaDownloadConfigurationError,
                DerivaDownloadAuthenticationError,
                DerivaDownloadAuthorizationError,
                DerivaDownloadTimeoutError,
            ) as e:
                raise DerivaMLException(format_exception(e))
            # Update version table with MINID.
            if use_minid:
                version_path = (
                    self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Version"]
                )
                version_rid = [h for h in self.dataset_history() if h.dataset_version == version][0].version_rid
                version_path.update([{"RID": version_rid, "Minid": minid_page_url}])
        return minid_page_url

    def _get_dataset_minid(
        self,
        version: DatasetVersion,
        create: bool,
        use_minid: bool,
    ) -> DatasetMinid | None:
        """Return a MINID for the specified dataset. If no version is specified, use the latest.

        Args:
            create: Create a new MINID if one doesn't already exist.

        Returns:
            New or existing MINID for the dataset.
        """

        # Find dataset version record
        version_str = str(version)
        history = self.dataset_history()
        try:
            version_record = next(v for v in history if v.dataset_version == version_str)
        except StopIteration:
            raise DerivaMLException(f"Version {version_str} does not exist for RID {self.dataset_rid}")

        # Check or create MINID
        minid_url = version_record.minid
        # If we either don't have a MINID, or we have a MINID, but we don't want to use it, generate a new one.
        if (not minid_url) or (not use_minid):
            if not create:
                raise DerivaMLException(f"Minid for dataset {self.dataset_rid} doesn't exist")
            if use_minid:
                self._logger.info("Creating new MINID for dataset %s", self.dataset_rid)
            minid_url = self._create_dataset_minid(version, use_minid=use_minid)

        # Return based on MINID usage
        if use_minid:
            return self._fetch_minid_metadata(version, minid_url)
        return DatasetMinid(
            dataset_version=version,
            RID=f"{self.dataset_rid}@{version_record.snapshot}",
            location=minid_url,
        )

    def _fetch_minid_metadata(self, version: DatasetVersion, url: str) -> DatasetMinid:
        r = requests.get(url, headers={"accept": "application/json"})
        r.raise_for_status()
        return DatasetMinid(dataset_version=version, **r.json())

    def _materialize_dataset_bag(
        self,
        minid: DatasetMinid,
        use_minid: bool,
    ) -> Path:
        """Materialize a dataset_table bag into a local directory

        Args:
            minid: A MINID to an existing bag or a RID of the dataset_table that should be downloaded.

        Returns:
            A tuple containing the path to the bag, the RID of the bag, and the MINID to the bag.
        """

        def update_status(status: Status, msg: str) -> None:
            """Update the current status for this execution in the catalog"""
            if self.execution_rid and self.execution_rid != DRY_RUN_RID:
                self._ml_instance.pathBuilder().schemas[self._ml_instance.ml_schema].Execution.update(
                    [
                        {
                            "RID": self.execution_rid,
                            "Status": status.value,
                            "Status_Detail": msg,
                        }
                    ]
                )
            self._logger.info(msg)

        def fetch_progress_callback(current, total):
            msg = f"Materializing bag: {current} of {total} file(s) downloaded."
            if self.execution_rid:
                update_status(Status.running, msg)
            return True

        def validation_progress_callback(current, total):
            msg = f"Validating bag: {current} of {total} file(s) validated."
            if self.execution_rid:
                update_status(Status.running, msg)
            return True

        # request metadata
        bag_path = self._download_dataset_minid(minid, use_minid)
        bag_dir = bag_path.parent
        validated_check = bag_dir / "validated_check.txt"

        # If this bag has already been validated, our work is done.  Otherwise, materialize the bag.
        if not validated_check.exists():
            self._logger.info(f"Materializing bag {minid.dataset_rid} Version:{minid.dataset_version}")
            bdb.materialize(
                bag_path.as_posix(),
                fetch_callback=fetch_progress_callback,
                validation_callback=validation_progress_callback,
            )
            validated_check.touch()
        return Path(bag_path)

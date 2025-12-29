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

import copy
import json
import logging
from collections import defaultdict

# Standard library imports
from graphlib import TopologicalSorter
from pathlib import Path

# Local imports
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Self
from urllib.parse import urlparse

# Deriva imports
import deriva.core.utils.hash_utils as hash_utils
import requests

# Third-party imports
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
        version: DatasetVersion | str | None = None,
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
        self._logger = logging.getLogger("deriva_ml")
        self.dataset_rid = dataset_rid
        self.execution_rid = execution_rid
        self._ml_instance = catalog
        self.description = description
        self._version: DatasetVersion | None = None
        self._version_snapshot: DerivaMLCatalog = self._ml_instance

        self.set_version(version)
        self.dataset_types = dataset_types or []

    def __repr__(self) -> str:
        return f"<Dataset rid='{self.dataset_rid}', version='{self.version}', types={self.dataset_types}>"

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
        dataset_types = dataset_types or []

        type_path = ml_instance.pathBuilder().schemas[ml_instance.ml_schema].tables[MLVocab.dataset_type.value]
        defined_types = list(type_path.entities().fetch())

        def check_dataset_type(dtype: str) -> bool:
            for term in defined_types:
                if dtype == term["Name"] or (term["Synonyms"] and ds_type in term["Synonyms"]):
                    return True
            return False

        # Create the entry for the new dataset_table and get its RID.
        ds_types = [dataset_types] if isinstance(dataset_types, str) else dataset_types
        pb = ml_instance.pathBuilder()
        for ds_type in ds_types:
            if not check_dataset_type(ds_type):
                raise DerivaMLException("Dataset type must be a vocabulary term.")
        dataset_table_path = pb.schemas[ml_instance._dataset_table.schema.name].tables[ml_instance._dataset_table.name]
        dataset_rid = dataset_table_path.insert(
            [
                {
                    "Description": description,
                    "Deleted": False,
                }
            ]
        )[0]["RID"]

        # Get the name of the association table between dataset_table and dataset_type.
        associations = list(
            ml_instance.model.schemas[ml_instance.ml_schema].tables[MLVocab.dataset_type].find_associations()
        )
        atable = associations[0].name if associations else None
        pb.schemas[ml_instance.ml_schema].tables[atable].insert(
            [{MLVocab.dataset_type: ds_type, "Dataset": dataset_rid} for ds_type in ds_types]
        )
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
            dataset_types=dataset_types,
            description=description,
            version=version,
        )
        return dataset

    @property
    def version(self) -> DatasetVersion:
        """
        If version is set, return it. Otherwise, return the most recent version of the dataset.
        Returns:
        """
        return self._version or self.current_version

    def set_version(self, version: DatasetVersion | str | None) -> Self:
        """
        Sets the version of the dataset. If a version is provided, it will be parsed and
        saved along with its corresponding snapshot. If no version is provided, the
        existing version and snapshot will be reset to None.

        Args:
            version: An instance of DatasetVersion, a string representation of the
                version, or None. If a string is provided, it will be parsed into a
                DatasetVersion object. None will reset the version data.

        Returns:
            A new copy of the current object, with the updated version details applied.
        """
        versioned_dataset = copy.copy(self)

        if version:
            versioned_dataset._version = DatasetVersion.parse(version) if isinstance(version, str) else version
            versioned_dataset._version_snapshot = versioned_dataset._version_snapshot_catalog(version)
        else:
            versioned_dataset._version = None
            versioned_dataset._version_snapshot = self._ml_instance
        return versioned_dataset

    @property
    def _dataset_table(self) -> Table:
        return self._ml_instance.model.schemas[self._ml_instance.ml_schema].tables["Dataset"]

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
        ts: TopologicalSorter = TopologicalSorter()
        self._build_dataset_graph_1(ts, set())
        return ts.static_order()

    def _build_dataset_graph_1(self, ts: TopologicalSorter, visited) -> None:
        """Use topological sort to return bottom up list of nested datasets"""
        if self.dataset_rid not in visited:
            ts.add(self)
            visited.add(self.dataset_rid)
            children = self.list_dataset_children()
            parents = self.list_dataset_parents()
            for parent in parents:
                parent._build_dataset_graph_1(ts, visited)
            for child in children:
                child._build_dataset_graph_1(ts, visited)

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
                version=ds.version.increment_version(component),
            )
            for ds in related_datasets
        ]
        Dataset._insert_dataset_versions(
            self._ml_instance, version_update_list, description=description, execution_rid=execution_rid
        )
        return next((d.version for d in version_update_list if d.rid == self.dataset_rid))

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def delete_dataset(self, dataset_rid: RID, recurse: bool = False) -> None:
        """Delete a dataset_table from the catalog.

        Args:
            dataset_rid: RID of the dataset_table to delete.
            recurse: If True, delete the dataset_table along with any nested datasets. (Default value = False)
        """
        # Get association table entries for this dataset_table
        # Delete association table entries
        if not self._ml_instance.model.is_dataset_rid(dataset_rid):
            raise DerivaMLException("Dataset_rid is not a dataset.")

        if parents := self.list_dataset_parents():
            raise DerivaMLException(f'Dataset_rid "{dataset_rid}" is in a nested dataset: {parents}.')

        pb = self._ml_instance.pathBuilder()
        dataset_path = pb.schemas[self._dataset_table.schema.name].tables[self._dataset_table.name]

        rid_list = [dataset_rid] + (self.list_dataset_children() if recurse else [])
        dataset_path.update([{"RID": r, "Deleted": True} for r in rid_list])

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_dataset_members(self, recurse: bool = False, limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
        """Lists members of a dataset.

        Returns a dictionary mapping member types to lists of member records. Can optionally
        recurse through nested datasets and limit the number of results.

        Args:
            recurse: Whether to include members of nested datasets. Defaults to False.
            limit: Maximum number of members to return per type. None for no limit.

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

        # Look at each of the element types that might be in the dataset_table and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)

        pb = self._version_snapshot.pathBuilder()
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
                    ds = self._version_snapshot.lookup_dataset(ds_rid)
                    for k, v in ds.list_dataset_members(recurse=recurse).items():
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
    def list_dataset_parents(self) -> list[Self]:
        """Given a dataset_table RID, return a list of RIDs of the parent datasets if this is included in a
        nested dataset.

        Returns:
            RID of the parent dataset_table.
        """
        if not self._ml_instance.model.is_dataset_rid(self.dataset_rid):
            raise DerivaMLException(
                f"RID: {self.dataset_rid} does not belong to dataset_table {self._dataset_table.name}"
            )
        # Get association table for nested datasets
        pb = self._version_snapshot.pathBuilder()
        atable_path = pb.schemas[self._ml_instance.ml_schema].Dataset_Dataset
        return [
            self._version_snapshot.lookup_dataset(p["Dataset"])
            for p in atable_path.filter(atable_path.Nested_Dataset == self.dataset_rid).entities().fetch()
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_dataset_children(self, recurse: bool = False) -> list[Self]:
        """Given a dataset_table RID, return a list of RIDs for any nested datasets.

        Args:
            recurse: If True, return a list of nested datasets RIDs.

        Returns:
          list of nested dataset RIDs.

        """
        dataset_dataset_path = (
            self._version_snapshot.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Dataset"]
        )
        nested_datasets = list(dataset_dataset_path.entities().fetch())

        def find_children(rid: RID) -> list[Self]:
            children = [child["Nested_Dataset"] for child in nested_datasets if child["Dataset"] == rid]
            if recurse:
                for child in children.copy():
                    children.extend(find_children(child))
            return children

        return [self._version_snapshot.lookup_dataset(rid) for rid in find_children(self.dataset_rid)]

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
        version: DatasetVersion | str | None = None,
        materialize: bool = True,
        use_minid: bool = True,
    ) -> DatasetBag:
        """Downloads a dataset to the local filesystem and creates a MINID if needed.

        Downloads a dataset to the local file system.  If the dataset has a version set, that version is used.
        If the dataset has a version and a version is provided, the version specified in the argument must match..
        Otherwise, the version to be downloaded must be specified via the version argument.
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
                ...     execution_rid="1-xyz789"
                ... )
        """
        if isinstance(version, str):
            version = DatasetVersion.parse(version)

        if not (self._version or version):
            raise DerivaMLException(
                "Dataset version not specified.  Version must either be set in the dataset or provided as an argument."
            )

        if self._version and version and self._version != version:
            raise DerivaMLException(
                f"Dataset version specified in dataset ({self._version}) "
                f"does not match version provided as argument ({version})."
            )

        # Get Dataset object that corresponds to the version of the dataset.
        versioned_dataset = self.set_version(version)
        return versioned_dataset._download_dataset_bag(
            materialize=materialize,
            use_minid=use_minid,
        )

    def _download_dataset_bag(
        self,
        materialize: bool,
        use_minid: bool = True,
    ) -> DatasetBag:
        """Download a dataset onto the local file system.  Create a MINID for the dataset if one doesn't already exist.

        Args:
            materialize: Download all of the assets in the dataset.

        Returns:
            Tuple consisting of the path to the dataset, the RID of the dataset that was downloaded and the MINID
            for the dataset.
        """
        if (
            self.execution_rid
            and self.execution_rid != DRY_RUN_RID
            and self._ml_instance.resolve_rid(self.execution_rid).table.name != "Execution"
        ):
            raise DerivaMLException(f"RID {self.execution_rid} is not an execution")
        minid = self._get_dataset_minid(create=True, use_minid=use_minid)

        bag_path = (
            self._materialize_dataset_bag(minid, use_minid=use_minid)
            if materialize
            else self._download_dataset_minid(minid, use_minid)
        )
        return DatabaseModel(minid, bag_path, self._ml_instance.working_dir).get_dataset()

    def _version_snapshot_catalog(self, dataset_version: DatasetVersion | str | None) -> DerivaMLCatalog:
        if isinstance(dataset_version, str) and str:
            dataset_version = DatasetVersion.parse(dataset_version)
        if dataset_version:
            return self._ml_instance.catalog_snapshot(self._version_snapshot_catalog_id())
        else:
            return self._ml_instance

    def _version_snapshot_catalog_id(self) -> str:
        """Return a catalog with snapshot for the specified dataset version"""
        try:
            version_record = next(h for h in self.dataset_history() if h.dataset_version == self._version)
        except StopIteration:
            raise DerivaMLException(f"Dataset version {self._version} not found for dataset {self.dataset_rid}")
        return f"{self._ml_instance.catalog.catalog_id}@{version_record.snapshot}"

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

    def _create_dataset_minid(self, use_minid=True) -> str:
        with TemporaryDirectory() as tmp_dir:
            # Generate a download specification file for the current catalog schema. By default, this spec
            # will generate a minid and place the bag into S3 storage.
            spec_file = Path(tmp_dir) / "download_spec.json"
            with spec_file.open("w", encoding="utf-8") as ds:
                downloader = CatalogGraph(self._ml_instance, use_minid=use_minid)
                json.dump(downloader.generate_dataset_download_spec(self), ds)
            try:
                self._logger.info(
                    "Downloading dataset %s for catalog: %s@%s"
                    % (
                        "minid" if use_minid else "bag",
                        self.dataset_rid,
                        str(self._version),
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
                version_rid = [h for h in self.dataset_history() if h.dataset_version == self._version][0].version_rid
                version_path.update([{"RID": version_rid, "Minid": minid_page_url}])
        return minid_page_url

    def _get_dataset_minid(
        self,
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
        version_str = str(self.version)
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
            minid_url = self._create_dataset_minid(use_minid=use_minid)

        # Return based on MINID usage
        if use_minid:
            return self._fetch_minid_metadata(minid_url)
        return DatasetMinid(
            dataset_version=self._version,
            RID=f"{self.dataset_rid}@{version_record.snapshot}",
            location=minid_url,
        )

    def _fetch_minid_metadata(self, url: str) -> DatasetMinid:
        r = requests.get(url, headers={"accept": "application/json"})
        r.raise_for_status()
        return DatasetMinid(dataset_version=self._version, **r.json())

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

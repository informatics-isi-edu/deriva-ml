"""
This module defines the DataSet class with is used to manipulate datasets in DerivaML,
The intended use of this class is as a base class in DerivaML so all the methods documented here are
accessible via a DerivaML class instance.


"""

from bdbag.fetch.fetcher import fetch_single_file
from bdbag import bdbag_api as bdb
from collections import defaultdict

from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import tag as deriva_tags, format_exception
from deriva.transfer.download.deriva_export import DerivaExport
from deriva.transfer.download.deriva_download import (
    DerivaDownloadConfigurationError,
    DerivaDownloadError,
    DerivaDownloadAuthenticationError,
    DerivaDownloadAuthorizationError,
    DerivaDownloadTimeoutError,
)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from graphlib import TopologicalSorter
import json
import logging
from pathlib import Path
from pydantic import (
    validate_call,
    ConfigDict,
)
import requests

from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Any, Callable, Optional, Iterable, Iterator

from deriva_ml import DatasetBag
from .deriva_definitions import ML_SCHEMA, DerivaMLException, MLVocab, Status, RID
from .history import iso_to_snap
from .deriva_model import DerivaModel
from .database_model import DatabaseModel
from .dataset_aux_classes import (
    DatasetVersion,
    DatasetMinid,
    DatasetHistory,
    VersionPart,
    DatasetSpec,
)


class Dataset:
    """
    Class to manipulate a dataset.

    Attributes:
        dataset_table (Table): ERMRest table holding dataset information.
    """

    _Logger = logging.getLogger("deriva_ml")

    def __init__(self, model: DerivaModel, cache_dir: Path):
        self._model = model
        self._ml_schema = ML_SCHEMA
        self.dataset_table = self._model.schemas[self._ml_schema].tables["Dataset"]
        self._cache_dir = cache_dir
        self._logger = logging.getLogger("deriva_ml")

    def _is_dataset_rid(self, dataset_rid: RID, deleted: bool = False) -> bool:
        try:
            rid_info = self._model.catalog.resolve_rid(dataset_rid, self._model.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {dataset_rid}")
        if rid_info.table != self.dataset_table:
            return False
        elif deleted:
            # Got a dataset rid. Now check to see if its deleted or not.
            return True
        else:
            return not list(rid_info.datapath.entities().fetch())[0]["Deleted"]

    def _insert_dataset_version(
        self,
        dataset_rid: RID,
        dataset_version: DatasetVersion,
        description: Optional[str] = "",
        execution_rid: Optional[RID] = None,
    ) -> RID:
        schema_path = self._model.catalog.getPathBuilder().schemas[self._ml_schema]
        version_path = schema_path.tables["Dataset_Version"]
        version_rid = version_path.insert(
            [
                {
                    "Dataset": dataset_rid,
                    "Version": str(dataset_version),
                    "Description": description,
                    "Execution": execution_rid,
                }
            ]
        )[0]["RID"]
        schema_path.tables["Dataset"].update(
            [{"RID": dataset_rid, "Version": version_rid}]
        )
        return version_rid

    def _bootstrap_versions(self):
        datasets = [ds["RID"] for ds in self.find_datasets()]
        ds_version = [
            {
                "Dataset": d,
                "Version": "0.1.0",
                "Description": "Dataset at the time of conversion to versioned datasets",
            }
            for d in datasets
        ]
        schema_path = self._model.catalog.getPathBuilder().schemas[self._ml_schema]
        version_path = schema_path.tables["Dataset_Version"]
        dataset_path = schema_path.tables["Dataset"]
        history = list(version_path.insert(ds_version))
        dataset_versions = [
            {"RID": h["Dataset"], "Version": h["Version"]} for h in history
        ]
        dataset_path.update(dataset_versions)

    def _synchronize_dataset_versions(self):
        datasets = [ds["RID"] for ds in self.find_datasets()]
        for ds in datasets:
            self.dataset_version(ds)
        schema_path = self._model.catalog.getPathBuilder().schemas[self._ml_schema]
        dataset_version_path = schema_path.tables["Dataset_Version"]
        # Get the maximum version number for each dataset.
        versions = {}
        for v in dataset_version_path.entities().fetch():
            if v["Version"] > versions.get("Dataset", DatasetVersion(0, 0, 0)):
                versions[v["Dataset"]] = v
        dataset_path = schema_path.tables["Dataset"]

        dataset_path.update(
            [
                {"RID": dataset, "Version": version["RID"]}
                for dataset, version in versions.items()
            ]
        )

    def dataset_history(self, dataset_rid: RID) -> list[DatasetHistory]:
        """Return a list of DatasetHistory objects representing the dataset

        Args:
            dataset_rid: A RID to the dataset for which history is to be fetched.

        Returns:
            A list of DatasetHistory objects which indicate the version-number, creation time, and bag instantiation of the dataset.
        """
        version_path = (
            self._model.catalog.getPathBuilder()
            .schemas[self._ml_schema]
            .tables["Dataset_Version"]
        )
        return [
            DatasetHistory(
                dataset_version=DatasetVersion.parse(v["Version"]),
                minid=v["Minid"],
                timestamp=v["RCT"],
                dataset_rid=dataset_rid,
                version_rid=v["RID"],
                description=v["Description"],
                execution_rid=v["Execution"],
            )
            for v in version_path.filter(version_path.Dataset == dataset_rid)
            .entities()
            .fetch()
        ]

    @validate_call
    def dataset_version(self, dataset_rid: RID) -> DatasetVersion:
        """Retrieve the current version of the specified dataset_table.

        Given a rid, return the most recent version of the dataset. It is important to remember that this version
        captures the state of the catalog at the time the version was created, not the current state of the catalog.
        This means that its possible that the values associated with an object in the catalog may be different
        from the values of that object in the dataset.

        Args:
            dataset_rid: The RID of the dataset to retrieve the version for.

        Returns:
            A tuple with the semantic version of the dataset_table.
        """
        history = self.dataset_history(dataset_rid)
        if not history:
            return DatasetVersion(0, 1, 0)
        else:
            return max([h.dataset_version for h in self.dataset_history(dataset_rid)])

    def _build_dataset_graph(self, dataset_rid: RID) -> Iterable[RID]:
        ts = TopologicalSorter()
        self._build_dataset_graph_1(dataset_rid, ts, set())
        return ts.static_order()

    def _build_dataset_graph_1(self, dataset_rid: RID, ts, visited) -> None:
        """Use topological sort to return bottom up list of nested datasets"""
        ts.add(dataset_rid)
        if dataset_rid not in visited:
            visited.add(dataset_rid)
            children = self.list_dataset_children(dataset_rid=dataset_rid)
            parents = self.list_dataset_parents(dataset_rid=dataset_rid)
            for parent in parents:
                self._build_dataset_graph_1(parent, ts, visited)
            for child in children:
                self._build_dataset_graph_1(child, ts, visited)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def increment_dataset_version(
        self,
        dataset_rid: RID,
        component: VersionPart,
        description: Optional[str] = "",
        execution_rid: Optional[RID] = None,
    ) -> DatasetVersion:
        """Increment the version of the specified dataset_table.

        Args:
          dataset_rid: RID to a dataset_table
          component: Which version of the dataset_table to increment.
          dataset_rid: RID of the dataset whose version is to be incremented.
          component: Major, Minor or Patch
          description: Description of the version update of the dataset_table.
          execution_rid: Which execution is performing increment.

        Returns:
          new semantic version of the dataset_table as a 3-tuple

        Raises:
          DerivaMLException: if provided RID is not to a dataset_table.
        """
        for dataset in self._build_dataset_graph(dataset_rid=dataset_rid):
            version = self.dataset_version(dataset)
            new_version = version.increment_version(component)
            self._insert_dataset_version(
                dataset,
                new_version,
                description=description,
                execution_rid=execution_rid,
            )
        return self.dataset_version(dataset_rid)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_dataset(
        self,
        type: str | list[str],
        description: str,
        execution_rid: Optional[RID] = None,
        version: Optional[DatasetVersion] = None,
    ) -> RID:
        """Create a new dataset_table from the specified list of RIDs.

        Args:
            type: One or more dataset_table types.  Must be a term from the DatasetType controlled vocabulary.
            description: Description of the dataset_table.
            execution_rid: Execution under which the dataset_table will be created.
            version: Version of the dataset_table.
            type: str | list[str]:
            description: str:


        Returns:
            New dataset_table RID.

        """

        version = version or DatasetVersion(0, 1, 0)

        type_path = (
            self._model.catalog.getPathBuilder()
            .schemas[self._ml_schema]
            .tables[MLVocab.dataset_type.value]
        )
        defined_types = list(type_path.entities().fetch())

        def check_dataset_type(dtype: str) -> bool:
            for term in defined_types:
                if dtype == term["Name"] or (
                    term["Synonyms"] and ds_type in term["Synonyms"]
                ):
                    return True
            return False

        # Create the entry for the new dataset_table and get its RID.
        ds_types = [type] if isinstance(type, str) else type
        pb = self._model.catalog.getPathBuilder()
        for ds_type in ds_types:
            if not check_dataset_type(ds_type):
                raise DerivaMLException("Dataset type must be a vocabulary term.")
        dataset_table_path = pb.schemas[self.dataset_table.schema.name].tables[
            self.dataset_table.name
        ]
        dataset_rid = dataset_table_path.insert(
            [
                {
                    "Description": description,
                    "Deleted": False,
                }
            ]
        )[0]["RID"]

        # Get the name of the association table between dataset_table and dataset_type.
        atable = next(
            self._model.schemas[self._ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name
        pb.schemas[self._ml_schema].tables[atable].insert(
            [
                {MLVocab.dataset_type: ds_type, "Dataset": dataset_rid}
                for ds_type in ds_types
            ]
        )
        if execution_rid is not None:
            pb.schemas[self._ml_schema].Dataset_Execution.insert(
                [{"Dataset": dataset_rid, "Execution": execution_rid}]
            )
        self._insert_dataset_version(
            dataset_rid,
            dataset_version=version,
            execution_rid=execution_rid,
            description="Initial dataset creation.",
        )
        return dataset_rid

    @validate_call
    def delete_dataset(self, dataset_rid: RID, recurse: bool = False) -> None:
        """Delete a dataset_table from the catalog.

        Args:
            dataset_rid: RID of the dataset_table to delete.
            recurse: If True, delete the dataset_table along with any nested datasets. (Default value = False)
            dataset_rid: RID:
        """
        # Get association table entries for this dataset_table
        # Delete association table entries
        if not self._is_dataset_rid(dataset_rid):
            raise DerivaMLException("Dataset_rid is not a dataset.")

        if parents := self.list_dataset_parents(dataset_rid):
            raise DerivaMLException(
                f'Dataset_rid "{dataset_rid}" is in a nested dataset: {parents}.'
            )

        pb = self._model.catalog.getPathBuilder()
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[
            self.dataset_table.name
        ]

        rid_list = [dataset_rid] + (
            self.list_dataset_children(dataset_rid) if recurse else []
        )
        dataset_path.update([{"RID": r, "Deleted": True} for r in rid_list])

    def find_datasets(self, deleted: bool = False) -> Iterable[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Arguments:
            deleted: If True, included the datasets that have been deleted.

        Returns:
             list of currently available datasets.
        """
        # Get datapath to all the tables we will need: Dataset, DatasetType and the association table.
        pb = self._model.catalog.getPathBuilder()
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[
            self.dataset_table.name
        ]
        atable = next(
            self._model.schemas[self._ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name
        ml_path = pb.schemas[self._ml_schema]
        atable_path = ml_path.tables[atable]

        if deleted:
            filtered_path = dataset_path
        else:
            filtered_path = dataset_path.filter(
                (dataset_path.Deleted == False) | (dataset_path.Deleted == None)
            )

        # Get a list of all the dataset_type values associated with this dataset_table.
        datasets = []
        for dataset in filtered_path.entities().fetch():
            ds_types = (
                atable_path.filter(atable_path.Dataset == dataset["RID"])
                .attributes(atable_path.Dataset_Type)
                .fetch()
            )
            datasets.append(
                dataset
                | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in ds_types]}
            )
        return datasets

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of entities that can be added to a dataset_table.

        Returns:
          :return: An iterable of Table objects that can be included as an element of a dataset_table.
        """

        def domain_table(table: Table) -> bool:
            return (
                table.schema.name == self._model.domain_schema
                or table.name == self.dataset_table.name
            )

        return [
            t
            for a in self.dataset_table.find_associations()
            if domain_table(t := a.other_fkeys.pop().pk_table)
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_element_type(self, element: str | Table) -> Table:
        """A dataset_table is a heterogeneous collection of objects, each of which comes from a different table. This
        routine makes it possible to add objects from the specified table to a dataset_table.

        Args:
            element: Name or the table or table object that is to be added to the dataset_table.
            element: str | Table:

        Returns:
            The table object that was added to the dataset_table.
        """
        # Add table to map
        element_table = self._model.name_to_table(element)
        table = self._model.schemas[self._model.domain_schema].create_table(
            Table.define_association([self.dataset_table, element_table])
        )

        # self.model = self.catalog.getCatalogModel()
        self.dataset_table.annotations.update(self._generate_dataset_annotations())
        self._model.model.apply()
        return table

    # @validate_call
    def list_dataset_members(
        self, dataset_rid: RID, recurse: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a list of entities associated with a specific dataset_table.

        Args:
            dataset_rid: param recurse: If this is a nested dataset_table, list the members of the contained datasets
            dataset_rid: RID:
            recurse:  (Default value = False)

        Returns:
            Dictionary of entities associated with a specific dataset_table.  Key is the table from which the elements
            were taken.
        """

        if not self._is_dataset_rid(dataset_rid):
            raise DerivaMLException(f"RID is not for a dataset_table: {dataset_rid}")

        # Look at each of the element types that might be in the dataset_table and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        pb = self._model.catalog.getPathBuilder()
        for assoc_table in self.dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            # Look at domain tables and nested datasets.
            if (
                target_table.schema.name != self._model.domain_schema
                and target_table != self.dataset_table
            ):
                continue
            member_column = (
                "Nested_Dataset"
                if target_table == self.dataset_table
                else other_fkey.foreign_key_columns[0].name
            )

            target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
            member_path = pb.schemas[member_table.schema.name].tables[member_table.name]

            path = member_path.filter(member_path.Dataset == dataset_rid).link(
                target_path,
                on=(member_path.columns[member_column] == target_path.columns["RID"]),
            )
            target_entities = list(path.entities().fetch())
            members[target_table.name].extend(target_entities)
            if recurse and target_table == self.dataset_table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds in nested_datasets:
                    for k, v in self.list_dataset_members(ds, recurse=False).items():
                        members[k].extend(v)
        return dict(members)

    @validate_call
    def add_dataset_members(
        self,
        dataset_rid: RID,
        members: list[RID],
        validate: bool = True,
        description: Optional[str] = "",
        execution_rid: Optional[RID] = None,
    ) -> None:
        """Add additional elements to an existing dataset_table.

        Add new elements to an existing dataset. In addition to adding new members, the minor version number of the
        dataset is incremented and the description, if provide is applied to that new version.

        Args:
            dataset_rid: RID of dataset_table to extend or None if new dataset_table is to be created.
            members: List of RIDs of members to add to the  dataset_table.
            validate: Check rid_list to make sure elements are not already in the dataset_table.
            description: Markdown description of the updated dataset.
            execution_rid: Optional RID of execution associated with this dataset.
        """
        members = set(members)
        description = description or "Updated dataset via add_dataset_members"

        def check_dataset_cycle(member_rid, path=None):
            """

            Args:
              member_rid:
              path:  (Default value = None)

            Returns:

            """
            path = path or set(dataset_rid)
            return member_rid in path

        if validate:
            existing_rids = set(
                m["RID"]
                for ms in self.list_dataset_members(dataset_rid).values()
                for m in ms
            )
            if overlap := set(existing_rids).intersection(members):
                raise DerivaMLException(
                    f"Attempting to add existing member to dataset_table {dataset_rid}: {overlap}"
                )

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        association_map = {
            a.other_fkeys.pop().pk_table.name: a.table.name
            for a in self.dataset_table.find_associations()
        }
        # Get a list of all the types of objects that can be linked to a dataset_table.
        for m in members:
            try:
                rid_info = self._model.catalog.resolve_rid(m)
            except KeyError:
                raise DerivaMLException(f"Invalid RID: {m}")
            if rid_info.table.name not in association_map:
                raise DerivaMLException(
                    f"RID table: {rid_info.table.name} not part of dataset_table"
                )
            if rid_info.table == self.dataset_table and check_dataset_cycle(
                rid_info.rid
            ):
                raise DerivaMLException("Creating cycle of datasets is not allowed")
            dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        pb = self._model.catalog.getPathBuilder()
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[
                self._ml_schema if table == "Dataset" else self._model.domain_schema
            ]
            fk_column = "Nested_Dataset" if table == "Dataset" else table
            if len(elements):
                # Find out the name of the column in the association table.
                schema_path.tables[association_map[table]].insert(
                    [{"Dataset": dataset_rid, fk_column: e} for e in elements]
                )
        self.increment_dataset_version(
            dataset_rid,
            VersionPart.minor,
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call
    def delete_dataset_members(
        self,
        dataset_rid: RID,
        members: list[RID],
        description: str = "",
        execution_rid: Optional[RID] = None,
    ) -> None:
        """Remove elements to an existing dataset_table.

        Delete elements from an existing dataset. In addition to deleting members, the minor version number of the
        dataset is incremented and the description, if provide is applied to that new version.

        Args:
            dataset_rid: RID of dataset_table to extend or None if new dataset_table is to be created.
            members: List of RIDs of members to add to the  dataset_table.
            description: Markdown description of the updated dataset.
            execution_rid: Optional RID of execution associated with this operation.
        """

        members = set(members)
        description = description or "Deletes dataset members"

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        association_map = {
            a.other_fkeys.pop().pk_table.name: a.table.name
            for a in self.dataset_table.find_associations()
        }
        # Get a list of all the types of objects that can be linked to a dataset_table.
        for m in members:
            try:
                rid_info = self._model.catalog.resolve_rid(m)
            except KeyError:
                raise DerivaMLException(f"Invalid RID: {m}")
            if rid_info.table.name not in association_map:
                raise DerivaMLException(
                    f"RID table: {rid_info.table.name} not part of dataset_table"
                )
            dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        pb = self._model.catalog.getPathBuilder()
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[
                self._ml_schema if table == "Dataset" else self._model.domain_schema
            ]
            fk_column = "Nested_Dataset" if table == "Dataset" else table

            if len(elements):
                atable_path = schema_path.tables[association_map[table]]
                # Find out the name of the column in the association table.
                for e in elements:
                    entity = atable_path.filter(
                        (atable_path.Dataset == dataset_rid)
                        & (atable_path.columns[fk_column] == e),
                    )
                    entity.delete()
        self.increment_dataset_version(
            dataset_rid,
            VersionPart.minor,
            description=description,
            execution_rid=execution_rid,
        )

    @validate_call
    def list_dataset_parents(self, dataset_rid: RID) -> list[RID]:
        """Given a dataset_table RID, return a list of RIDs of the parent datasets if this is included in a
        nested dataset.

        Args:
            dataset_rid: return: RID of the parent dataset_table.
            dataset_rid: RID:

        Returns:
            RID of the parent dataset_table.
        """
        if not self._is_dataset_rid(dataset_rid):
            raise DerivaMLException(
                f"RID: {dataset_rid} does not belong to dataset_table {self.dataset_table.name}"
            )
        # Get association table for nested datasets
        pb = self._model.catalog.getPathBuilder()
        atable_path = pb.schemas[self._ml_schema].Dataset_Dataset
        return [
            p["Dataset"]
            for p in atable_path.filter(atable_path.Nested_Dataset == dataset_rid)
            .entities()
            .fetch()
        ]

    @validate_call
    def list_dataset_children(self, dataset_rid: RID, recurse=False) -> list[RID]:
        """Given a dataset_table RID, return a list of RIDs of any nested datasets.

        Args:
            dataset_rid: A dataset_table RID.
            recurse: If True, return a list of RIDs of any nested datasets.

        Returns:
          list of RIDs of nested datasets.

        """
        children = [d["RID"] for d in self.list_dataset_members(dataset_rid)["Dataset"]]
        if recurse:
            for child in children.copy():
                children.extend(self.list_dataset_children(child, recurse=recurse))
        return children

    def _vocabulary_specification(
        self, writer: Callable[[str, str, Table], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """

        Args:
          writer: Callable[[list[Table]]: list[dict[str: Any]]]:

        Returns:

        """
        vocabs = [
            table
            for s in self._model.schemas.values()
            for table in s.tables.values()
            if self._model.is_vocabulary(table)
        ]
        return [
            o
            for table in vocabs
            for o in writer(f"{table.schema.name}:{table.name}", table.name, table)
        ]

    def _table_paths(self) -> Iterator[tuple[list[str], list[str], list[Table]]]:

        dataset_dataset = self._model.schemas[self._ml_schema].tables["Dataset_Dataset"]
        paths = self._model._schema_to_paths()
        nested_paths = paths

        for i in range(self._dataset_nesting_depth()):
            if i == 0:
                paths.extend([[self.dataset_table, dataset_dataset]])
            nested_paths = [
                [self.dataset_table, dataset_dataset] + p for p in nested_paths
            ]
            paths.extend(nested_paths)

        def source_path(path):
            p = [f"{self._model.ml_schema}:Dataset/RID={{Dataset_RID}}"]
            for table in path[1:]:
                if table == dataset_dataset:
                    p.append("(RID)=(deriva-ml:Dataset_Dataset:Dataset)")
                elif table == self.dataset_table:
                    p.append("(Nested_Dataset)=(deriva-ml:Dataset:RID)")
                elif table.name == "Dataset_Version":
                    p.append(f"(RID)=({self._model.ml_schema}:Dataset_Version:Dataset)")
                else:
                    p.append(f"{table.schema.name}:{table.name}")
            return p

        src_paths = ["/".join(source_path(p)) for p in paths]
        dest_paths = ["/".join([t.name for t in p]) for p in paths]
        target_tables = [p[-1] for p in paths]

        return zip(src_paths, dest_paths, target_tables)

    def _dataset_nesting_depth(self):
        """Determine the maximum dataset nesting depth in the current catalog.

        Returns:

        """

        def children_depth(
            dataset_rid: RID, nested_datasets: dict[RID, list[RID]]
        ) -> int:
            """Return the number of nested datasets in the current catalog"""
            try:
                children = nested_datasets[dataset_rid]
                return (
                    max(map(lambda x: children_depth(x, nested_datasets), children)) + 1
                    if children
                    else 1
                )
            except KeyError:
                return 0

        # Build up the dataset_table nesting graph...
        pb = (
            self._model.catalog.getPathBuilder()
            .schemas[self._ml_schema]
            .tables["Dataset_Dataset"]
        )
        nested_dataset = defaultdict(list)
        for ds in pb.entities().fetch():
            nested_dataset[ds["Dataset"]].append(ds["Nested_Dataset"])
        return (
            max(map(lambda d: children_depth(d, dict(nested_dataset)), nested_dataset))
            if nested_dataset
            else 0
        )

    def _dataset_specification(
        self, writer: Callable[[str, str, Table], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Output a download/export specification for a dataset_table.  Each element of the dataset_table will be placed in its own dir
        The top level data directory of the resulting BDBag will have one subdirectory for element type. the subdirectory
        will contain the CSV indicating which elements of that type are present in the dataset_table, and then there will be a
        subdirectories for each object that is reachable from the dataset_table members.

        To simplify reconstructing the relationship between tables, the CVS for each
        The top level data directory will also contain a subdirectory for any controlled vocabularies used in the dataset_table.
        All assets will be placed into a directory named asset in a subdirectory with the asset table name.

        For example, consider a dataset_table that consists of two element types, T1 and T2. T1 has foreign key relationships to
        objects in tables T3 and T4.  There are also two controlled vocabularies, CV1 and CV2.  T2 is an asset table
        which has two asset in it. The layout of the resulting bdbag would be:
              data
                CV1/
                    cv1.csv
                CV2/
                    cv2.csv
                Dataset/
                    T1/
                        t1.csv
                        T3/
                            t3.csv
                        T4/
                            t4.csv
                    T2/
                        t2.csv
                asset/
                  T2
                    f1
                    f2

        Args:
          writer: Callable[[list[Table]]: list[dict[str:  Any]]]:

        Returns:
            A dataset_table specification.
        """
        element_spec = []
        for path in self._table_paths():
            element_spec.extend(writer(*path))
        return self._vocabulary_specification(writer) + element_spec

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_dataset_bag(
        self,
        dataset: DatasetSpec,
        execution_rid: Optional[RID] = None,
    ) -> DatasetBag:
        """Download a dataset onto the local file system.  Create a MINID for the dataset if one doesn't already exist.

        Args:
            dataset: Specification of the dataset to be downloaded.
            execution_rid: Execution RID for the dataset.

        Returns:
            Tuple consisting of the path to the dataset, the RID of the dataset that was downloaded and the MINID
            for the dataset.
        """
        if (
            execution_rid
            and self._model.catalog.resolve_rid(execution_rid).table.name != "Execution"
        ):
            raise DerivaMLException(f"RID {execution_rid} is not an execution")
        minid = self.get_dataset_minid(dataset)

        bag_path = (
            self._materialize_dataset_bag(minid, execution_rid=execution_rid)
            if dataset.materialize
            else self._download_dataset_bag(minid)
        )
        return DatabaseModel(minid, bag_path).get_dataset()

    def _version_snapshot(self, dataset: DatasetSpec) -> str:
        version_record = [
            h
            for h in self.dataset_history(dataset_rid=dataset.rid)
            if h.dataset_version == dataset.version
        ][0]
        return f"{self._model.catalog.catalog_id}@{iso_to_snap(version_record.timestamp.isoformat())}"

    def _create_dataset_minid(self, dataset: DatasetSpec) -> str:
        with TemporaryDirectory() as tmp_dir:
            # Generate a download specification file for the current catalog schema. By default, this spec
            # will generate a minid and place the bag into S3 storage.
            spec_file = f"{tmp_dir}/download_spec.json"
            with open(spec_file, "w", encoding="utf-8") as ds:
                json.dump(self._generate_dataset_download_spec(dataset), ds)
            try:
                self._logger.info(
                    f"Downloading dataset minid for catalog: {dataset.rid}@{str(dataset.version)}"
                )
                # Generate the bag and put into S3 storage.
                exporter = DerivaExport(
                    host=self._model.catalog.deriva_server.server,
                    config_file=spec_file,
                    output_dir=tmp_dir,
                    defer_download=True,
                    timeout=(10, 610),
                    envars={"Dataset_RID": dataset.rid},
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
            version_path = (
                self._model.catalog.getPathBuilder()
                .schemas[self._ml_schema]
                .tables["Dataset_Version"]
            )
            version_rid = [
                h
                for h in self.dataset_history(dataset_rid=dataset.rid)
                if h.dataset_version == dataset.version
            ][0].version_rid
            version_path.update([{"RID": version_rid, "Minid": minid_page_url}])
        return minid_page_url

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_dataset_minid(
        self, dataset: DatasetSpec, create: bool = True
    ) -> DatasetMinid:
        """Return a MINID to the specified dataset.  If no version is specified, use the latest.

        Args:
            dataset: Specification of the dataset.
            create: Create a new MINID if one doesn't already exist.

        Returns:
            New or existing MINID for the dataset.
        """
        if dataset.rid.startswith("minid"):
            minid_url = f"https://identifiers.org/{dataset.rid}"
        elif dataset.rid.startswith("http"):
            minid_url = dataset.rid
        else:
            if not any([dataset.rid == ds["RID"] for ds in self.find_datasets()]):
                raise DerivaMLException(f"RID {dataset.rid} is not a dataset_table")

            # Get the history record for the version we are looking for.
            dataset_version_record = [
                v
                for v in self.dataset_history(dataset.rid)
                if v.dataset_version == str(dataset.version)
            ][0]
            if not dataset_version_record:
                raise DerivaMLException(
                    f"Version {str(dataset.version)} does not exist for RID {dataset.rid}"
                )
            minid_url = dataset_version_record.minid
            if not minid_url:
                if not create:
                    raise DerivaMLException(
                        f"Minid for dataset {dataset.rid} doesn't exist"
                    )
                self._logger.info("Creating new MINID for dataset %s", dataset.rid)
                minid_url = self._create_dataset_minid(dataset)
            # If provided a MINID, use the MINID metadata to get the checksum and download the bag.
        r = requests.get(minid_url, headers={"accept": "application/json"})
        return DatasetMinid(dataset_version=dataset.version, **r.json())

    def _download_dataset_bag(self, minid: DatasetMinid) -> Path:
        """Given a RID to a dataset_table, or a MINID to an existing bag, download the bag file, extract it and validate
        that all the metadata is correct

        Args:
            minid: The RID of a dataset_table or a minid to an existing bag.
        Returns:
            the location of the unpacked and validated dataset_table bag and the RID of the bag and the bag MINID
        """

        # Check to see if we have an existing idempotent materialization of the desired bag. If so, then just reuse
        # it.  If not, then we need to extract the contents of the archive into our cache directory.
        bag_dir = self._cache_dir / f"{minid.dataset_rid}_{minid.checksum}"
        if bag_dir.exists():
            bag_path = (bag_dir / f"Dataset_{minid.dataset_rid}").as_posix()
        else:
            bag_dir.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                delete=False, suffix=f"Dataset_{minid.dataset_rid}.zip"
            ) as zip_file:
                archive_path = fetch_single_file(minid.bag_url, zip_file.name)
                bag_path = bdb.extract_bag(archive_path, bag_dir.as_posix())
            bdb.validate_bag_structure(bag_path)
        return Path(bag_path)

    def _materialize_dataset_bag(
        self,
        minid: DatasetMinid,
        execution_rid: Optional[RID] = None,
    ) -> Path:
        """Materialize a dataset_table bag into a local directory

        Args:
            minid: A MINID to an existing bag or a RID of the dataset_table that should be downloaded.

        Returns:
            A tuple containing the path to the bag, the RID of the bag, and the MINID to the bag.
        """

        def update_status(status: Status, msg: str) -> None:
            """Update the current status for this execution in the catalog"""
            self._model.catalog.getPathBuilder().schemas[
                self._ml_schema
            ].Execution.update(
                [
                    {
                        "RID": execution_rid,
                        "Status": status.value,
                        "Status_Detail": msg,
                    }
                ]
            )
            self._logger.info(msg)

        def fetch_progress_callback(current, total):
            msg = f"Materializing bag: {current} of {total} file(s) downloaded."
            if execution_rid:
                update_status(Status.running, msg)
            return True

        def validation_progress_callback(current, total):
            msg = f"Validating bag: {current} of {total} file(s) validated."
            if execution_rid:
                update_status(Status.running, msg)
            return True

        # request metadata
        bag_path = self._download_dataset_bag(minid)
        bag_dir = bag_path.parent
        validated_check = bag_dir / "validated_check.txt"

        # If this bag has already been validated, our work is done.  Otherwise, materialize the bag.
        if not validated_check.exists():
            bdb.materialize(
                bag_path.as_posix(),
                fetch_callback=fetch_progress_callback,
                validation_callback=validation_progress_callback,
            )
            validated_check.touch()
        return Path(bag_path)

    def _export_outputs(self) -> list[dict[str, Any]]:
        """Return and output specification for the datasets in the provided model

        Returns:
          An export specification suitable for Chaise.
        """

        def writer(spath: str, dpath: str, table: Table) -> list[dict[str, Any]]:
            """

            Args:
              spath: list[Table]:
              dpath: list[Table]:
              table: Table

            Returns:
                An export specification suitable for Chaise.
            """
            return self._export_dataset_element(spath, dpath, table)

        # Export specification is a specification for the datasets, plus any controlled vocabulary
        return [
            {
                "source": {"api": False, "skip_root_path": True},
                "destination": {"type": "env", "params": {"query_keys": ["snaptime"]}},
            },
            {
                "source": {"api": "entity"},
                "destination": {
                    "type": "env",
                    "params": {"query_keys": ["RID", "Description"]},
                },
            },
            {
                "source": {"api": "schema", "skip_root_path": True},
                "destination": {"type": "json", "name": "schema"},
            },
        ] + self._dataset_specification(writer)

    def _processor_params(self) -> list[dict[str, Any]]:
        """
        Returns:
          a download specification for the datasets in the provided model.

        """

        def writer(spath: str, dpath: str, table: Table) -> list[dict[str, Any]]:
            """

            Args:
              spath:
              dpath:
              table: Table

            Returns:

            """
            return self._download_dataset_element(spath, dpath, table)

        # Download spec is the spec for any controlled vocabulary and for the dataset_table.
        return [
            {
                "processor": "json",
                "processor_params": {"query_path": "/schema", "output_path": "schema"},
            }
        ] + self._dataset_specification(writer)

    @staticmethod
    def _download_dataset_element(
        spath: str, dpath: str, table: Table
    ) -> list[dict[str, Any]]:
        """Return the download specification for the data object indicated by a path through the data model.

        Args:
          spath: Source path
          dpath: Destination path
          table: Table referenced to by the path

        Returns:
          The download specification that will retrieve that data from the catalog and place it into a BDBag.
        """
        exports = [
            {
                "processor": "csv",
                "processor_params": {
                    "query_path": f"/entity/{spath}?limit=none",
                    "output_path": dpath,
                },
            }
        ]

        # If this table is an asset table, then we need to output the files associated with the asset.
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        if asset_columns.issubset({c.name for c in table.columns}):
            exports.append(
                {
                    "processor": "fetch",
                    "processor_params": {
                        "query_path": f"/attribute/{spath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5?limit=none",
                        "output_path": f"asset/{table.name}",
                    },
                }
            )
        return exports

    @staticmethod
    def _export_dataset_element(
        spath: str, dpath: str, table: Table
    ) -> list[dict[str, Any]]:
        """Given a path in the data model, output an export specification for the path taken to get to the current table.

        Args:
          spath: Source path
          dpath: Destination path
          table: Table referenced to by the path

        Returns:
          The export specification that will retrieve that data from the catalog and place it into a BDBag.
        """
        # The table is the last element of the path.  Generate the ERMRest query by converting the list of tables
        # into a path in the form of /S:T1/S:T2/S:Table
        # Generate the destination path in the file system using just the table names.

        exports = [
            {
                "source": {"api": "entity", "path": spath},
                "destination": {"name": dpath, "type": "csv"},
            }
        ]

        # If this table is an asset table, then we need to output the files associated with the asset.
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        if asset_columns.issubset({c.name for c in table.columns}):
            exports.append(
                {
                    "source": {
                        "api": "attribute",
                        "path": f"{spath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5",
                    },
                    "destination": {"name": f"asset/{table.name}", "type": "fetch"},
                }
            )
        return exports

    def _generate_dataset_download_spec(self, dataset: DatasetSpec) -> dict[str, Any]:
        """

        Returns:
        """
        s3_target = "s3://eye-ai-shared"
        minid_test = False

        catalog_id = self._version_snapshot(dataset)
        return {
            "env": {"Dataset_RID": "{Dataset_RID}"},
            "bag": {
                "bag_name": "Dataset_{Dataset_RID}",
                "bag_algorithms": ["md5"],
                "bag_archiver": "zip",
                "bag_metadata": {},
                "bag_idempotent": True,
            },
            "post_processors": [
                {
                    "processor": "cloud_upload",
                    "processor_params": {
                        "acl": "public-read",
                        "target_url": s3_target,
                    },
                },
                {
                    "processor": "identifier",
                    "processor_params": {
                        "test": minid_test,
                        "env_column_map": {
                            "Dataset_RID": "{RID}@{snaptime}",
                            "Description": "{Description}",
                        },
                    },
                },
            ],
            "catalog": {
                "host": f"{self._model.catalog.deriva_server.scheme}://{self._model.catalog.deriva_server.server}",
                "catalog_id": catalog_id,
                "query_processors": [
                    {
                        "processor": "env",
                        "processor_params": {
                            "output_path": "Dataset",
                            "query_keys": ["snaptime"],
                            "query_path": "/",
                        },
                    },
                    {
                        "processor": "env",
                        "processor_params": {
                            "query_path": "/entity/M:=deriva-ml:Dataset/RID={Dataset_RID}?limit=none",
                            "output_path": "Dataset",
                            "query_keys": ["RID", "Description"],
                        },
                    },
                ]
                + self._processor_params(),
            },
        }

    def dataset_visible_columns(self) -> dict[str, Any]:
        dataset_table = self._model.schemas["deriva-ml"].tables["Dataset"]
        rcb_name = next(
            [fk.name[0].name, fk.name[1]]
            for fk in dataset_table.foreign_keys
            if fk.name[1] == "Dataset_RCB_fkey"
        )
        rmb_name = next(
            [fk.name[0].name, fk.name[1]]
            for fk in dataset_table.foreign_keys
            if fk.name[1] == "Dataset_RMB_fkey"
        )
        return {
            "*": [
                "RID",
                "Description",
                {
                    "display": {
                        "markdown_pattern": "[Annotate Dataset](https://www.eye-ai.org/apps/grading-interface/main?dataset_rid={{{RID}}}){: .btn}"
                    },
                    "markdown_name": "Annotation App",
                },
                rcb_name,
                rmb_name,
            ],
            "detailed": [
                "RID",
                "Description",
                {
                    "source": [
                        {"inbound": ["deriva-ml", "Dataset_Dataset_Type_Dataset_fkey"]},
                        {
                            "outbound": [
                                "deriva-ml",
                                "Dataset_Dataset_Type_Dataset_Type_fkey",
                            ]
                        },
                        "RID",
                    ],
                    "markdown_name": "Dataset Types",
                },
                {
                    "display": {
                        "markdown_pattern": "[Annotate Dataset](https://www.eye-ai.org/apps/grading-interface/main?dataset_rid={{{RID}}}){: .btn}"
                    },
                    "markdown_name": "Annotation App",
                },
                rcb_name,
                rmb_name,
            ],
            "filter": {
                "and": [
                    {"source": "RID"},
                    {"source": "Description"},
                    {
                        "source": [
                            {
                                "inbound": [
                                    "deriva-ml",
                                    "Dataset_Dataset_Type_Dataset_fkey",
                                ]
                            },
                            {
                                "outbound": [
                                    "deriva-ml",
                                    "Dataset_Dataset_Type_Dataset_Type_fkey",
                                ]
                            },
                            "RID",
                        ],
                        "markdown_name": "Dataset Types",
                    },
                    {
                        "source": [{"outbound": rcb_name}, "RID"],
                        "markdown_name": "Created By",
                    },
                    {
                        "source": [{"outbound": rmb_name}, "RID"],
                        "markdown_name": "Modified By",
                    },
                ]
            },
        }

    def _dataset_visible_fkeys(self) -> dict[str, Any]:
        def fkey_name(fk):
            return [fk.name[0].name, fk.name[1]]

        dataset_table = self._model.schemas["deriva-ml"].tables["Dataset"]

        source_list = [
            {
                "source": [
                    {"inbound": fkey_name(fkey.self_fkey)},
                    {"outbound": fkey_name(other_fkey := fkey.other_fkeys.pop())},
                    "RID",
                ],
                "markdown_name": other_fkey.pk_table.name,
            }
            for fkey in dataset_table.find_associations(max_arity=3, pure=False)
        ]
        return {"detailed": source_list}

    def _generate_dataset_annotations(self) -> dict[str, Any]:
        return {
            deriva_tags.export_fragment_definitions: {
                "dataset_export_outputs": self._export_outputs()
            },
            deriva_tags.visible_columns: self.dataset_visible_columns(),
            deriva_tags.visible_foreign_keys: self._dataset_visible_fkeys(),
            deriva_tags.export_2019: {
                "detailed": {
                    "templates": [
                        {
                            "type": "BAG",
                            "outputs": [{"fragment_key": "dataset_export_outputs"}],
                            "displayname": "BDBag Download",
                            "bag_idempotent": True,
                            "postprocessors": [
                                {
                                    "processor": "identifier",
                                    "processor_params": {
                                        "test": False,
                                        "env_column_map": {
                                            "Dataset_RID": "{RID}@{snaptime}",
                                            "Description": "{Description}",
                                        },
                                    },
                                }
                            ],
                        },
                        {
                            "type": "BAG",
                            "outputs": [{"fragment_key": "dataset_export_outputs"}],
                            "displayname": "BDBag to Cloud",
                            "bag_idempotent": True,
                            "postprocessors": [
                                {
                                    "processor": "cloud_upload",
                                    "processor_params": {
                                        "acl": "public-read",
                                        "target_url": "s3://eye-ai-shared/",
                                    },
                                },
                                {
                                    "processor": "identifier",
                                    "processor_params": {
                                        "test": False,
                                        "env_column_map": {
                                            "Dataset_RID": "{RID}@{snaptime}",
                                            "Description": "{Description}",
                                        },
                                    },
                                },
                            ],
                        },
                    ]
                }
            },
        }

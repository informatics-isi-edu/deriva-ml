"""
THis module defines the DataSet class with is used to manipulate n
"""

from collections import defaultdict
from typing import Any, Callable, Optional, Iterable
from deriva.core.ermrest_model import Model, Table
from deriva.core.datapath import DataPathException

from .deriva_definitions import (
    ML_SCHEMA,
    RID,
    SemanticVersion,
    DerivaMLException,
    MLVocab,
)
from .schema_setup.dataset_annotations import generate_dataset_annotations
from .deriva_definitions import VocabularyTerm

from pydantic import validate_call, ConfigDict


class Dataset:
    """
    Class to manipulate a dataset.

    Attributes:
        table: ERMrest table holding dataset information.
    """

    def __init__(self, model: Model):
        self._model = model
        self._ml_schema = ML_SCHEMA
        self._domain_schema = [
            s for s in model.schemas if s not in ["deriva-ml", "www", "public"]
        ].pop()
        self.table = self._model.schemas[self._ml_schema].tables["Dataset"]

    def _is_dataset_rid(self, rid: RID) -> bool:
        rid_record = self._model.catalog.resolve_rid(rid)
        return rid_record.table == self.table

    def _resolve_rid(self, rid: RID):
        try:
            return self._model.catalog.resolve_rid(rid, self._model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")

    def dataset_version(self, dataset_rid: RID) -> tuple[int, ...]:
        """Retrieve the version of the specified dataset.

        Args:
            dataset_rid: return: A tuple with the semantic version of the dataset.
            dataset_rid: RID:

        Returns:
            A tuple with the semantic version of the dataset.
        """
        if not self._is_dataset_rid(dataset_rid):
            raise DerivaMLException(
                f"RID: {dataset_rid} does not belong to dataset {self.table.name}"
            )
        return tuple(
            map(
                int,
                self._model.catalog.retrieve_rid(dataset_rid, self._model)[
                    "Version"
                ].split("."),
            )
        )

    def dataset_version_history(self, dataset_rid: RID) -> list[SemanticVersion]:
        pass

    def increment_dataset_version(
        self,
        dataset_rid: RID,
        component: SemanticVersion,
        description: Optional[str] = "",
    ) -> tuple[int, ...]:
        """Increment the version of the specified dataset.

        Args:
          dataset_rid: RID to a dataset
          component: Which version of the dataset to increment.
          description: Description of the version of the dataset.

        Returns:
          new semantic ersion of the dataset as a 3-tuple

        Raises:
          DerivaMLException: if provided RID is not to a dataset.
        """

        # Makesure that the RID is to a dataset
        if self._model.catalog.resolve_rid(dataset_rid).table != self.table.name:
            raise DerivaMLException(f'RID "{dataset_rid}" is not a dataset')

        major, minor, patch = self.dataset_version(dataset_rid)
        schema_path = self._model.catalog.getPathBuilder().schemas[self._ml_schema]
        match component:
            case SemanticVersion.major:
                major += 1
            case SemanticVersion.minor:
                minor += 1
            case SemanticVersion.patch:
                patch += 1
        dataset_path = schema_path.tables[self.table.name]
        semantic_version = f"{major}.{minor}.{patch}"
        dataset_path.update(
            [{"RID": dataset_rid, "Version": semantic_version}]
        )
        snapshot = self._model.catalog.latest_snapshot().snaptime
        schema_path.tables["DatasetVersion"].insert(
            [
                {
                    "Dataset": dataset_rid,
                    "Version": semantic_version,
                    "SnapshotID": snapshot,
                    "Description": description,
                }
            ]
        )
        dataset_path.update(
            [{"RID": dataset_rid, "Version": f"{major}.{minor}.{patch}"}]
        )
        return major, minor, patch

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of entities that can be added to a dataset.

        Returns:
          :return: An iterable of Table objects that can be included as an element of a dataset.
        """

        def domain_table(table: Table) -> bool:
            return (
                table.schema.name == self._domain_schema
                or table.name == self.table.name
            )

        return [
            t
            for a in self.table.find_associations()
            if domain_table(t := a.other_fkeys.pop().pk_table)
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_element_type(self, element_table: Table) -> Table:
        """A dataset is a heterogeneous collection of objects, each of which comes from a different table. This
        routine makes it possible to add objects from the specified table to a dataset.

        Args:
            element_table: Table object that is to be added to the dataset.

        Returns:
            The table object that was added to the dataset.
        """
        # Add table to map
        table = self._model.schemas[self._domain_schema].create_table(
            Table.define_association([self.table, element_table])
        )

        # self.model = self.catalog.getCatalogModel()
        self.table.annotations.update(generate_dataset_annotations(self._model))
        self._model.apply()
        return table

    def find_datasets(self) -> Iterable[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Returns:
             list of currently available datasets.
        """
        # Get datapath to all the tables we will need: Dataset, DatasetType and the association table.
        pb = self._model.catalog.getPathBuilder()
        dataset_path = pb.schemas[self.table.schema.name].tables[self.table.name]
        atable = next(
            self._model.schemas[self._ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name
        ml_path = pb.schemas[self._ml_schema]
        atable_path = ml_path.tables[atable]

        # Get a list of all the dataset_type values associated with this dataset.
        datasets = []
        for dataset in dataset_path.entities().fetch():
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

    @validate_call
    def create_dataset(
        self,
        ds_type: str | list[str],
        description: str,
        execution_rid: Optional[RID] = None,
        version: tuple[int, int, int] = (1, 0, 0),
    ) -> RID:
        """Create a new dataset from the specified list of RIDs.

        Args:
            ds_type: One or more dataset types.  Must be a term from the DatasetType controlled vocabulary.
            description: Description of the dataset.
            execution_rid: Execution under which the dataset will be created.
            version: Version of the dataset.
            ds_type: str | list[str]:
            description: str:
            execution_rid: Optional[RID]:  (Default value = None)
            version: tuple[int: int: int]

        Returns:
            New dataset RID.

        """
        # Create the entry for the new dataset and get its RID.
        ds_types = [ds_type] if isinstance(ds_type, str) else ds_type

        pb = self._model.catalog.getPathBuilder()
        for ds_type in ds_types:
            vocab_table = self._model.schemas[self._ml_schema].tables[MLVocab.dataset_type]
            if not self._lookup_term(vocab_table, ds_type):
                raise DerivaMLException(f"Dataset type must be a vocabulary term.")
        dataset_table_path = pb.schemas[self.table.schema.name].tables[self.table.name]
        dataset = dataset_table_path.insert(
            [
                {
                    "Description": description,
                    MLVocab.dataset_type: ds_type,
                    "Version": f"{version[0]}.{version[1]}.{version[2]}",
                }
            ]
        )[0]["RID"]

        # Get the name of the association table between dataset and dataset_type.
        atable = next(
            self._model.schemas[self._ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name
        pb.schemas[self._ml_schema].tables[atable].insert(
            [
                {MLVocab.dataset_type: ds_type, "Dataset": dataset}
                for ds_type in ds_types
            ]
        )
        if execution_rid is not None:
            pb.schemas[self._ml_schema].Dataset_Execution.insert(
                [{"Dataset": dataset, "Execution": execution_rid}]
            )
        return dataset

    @validate_call
    def delete_dataset(self, dataset_rid: RID, recurse: bool = False) -> None:
        """Delete a dataset from the catalog.

        Args:
            dataset_rid: RID of the dataset to delete.
            recurse: If True, delete the dataset along with any nested datasets. (Default value = False)
            dataset_rid: RID:
        """
        # Get association table entries for this dataset
        # Delete association table entries
        pb = self._model.catalog.getPathBuilder()
        for assoc_table in self.table.find_associations(self.table):
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            schema_path = pb.schemas[member_table.schema.name]
            tpath = schema_path.tables[assoc_table.name]
            dataset_column_path = tpath.columns[self_fkey.columns[0].name]
            dataset_entries = tpath.filter(dataset_column_path == dataset_rid)
            if recurse and target_table == self.table:
                # Nested table
                for dataset in dataset_entries:
                    self.delete_dataset(dataset["RID"], recurse)
            try:
                dataset_entries.delete()
            except DataPathException:
                pass

        # Delete dataset.
        dataset_path = pb.schemas[self.table.schema.name].tables[self.table.name]
        dataset_path.filter(dataset_path.columns["RID"] == dataset_rid).delete()

    @validate_call
    def list_dataset_parents(self, dataset_rid: RID) -> list[RID]:
        """Given a dataset RID, return a list of RIDs of the parent datasets.

        Args:
            dataset_rid: return: RID of the parent dataset.

        Returns:
            RID of the parent dataset.
        """
        rid_record = self._resolve_rid(dataset_rid)
        if rid_record.table.name != self.table.name:
            raise DerivaMLException(
                f"RID: {dataset_rid} does not belong to dataset {self.table.name}"
            )
        # Get association table for nested datasets
        atable_path = (
            self._model.catalog.getPathBuilder().schemas[self._ml_schema].Dataset_Dataset
        )
        return [
            p["Dataset"]
            for p in atable_path.filter(atable_path.Nested_Dataset == dataset_rid)
            .entities()
            .fetch()
        ]

    @validate_call
    def list_dataset_children(self, dataset_rid: RID) -> list[RID]:
        """Given a dataset RID, return a list of RIDs of any nested datasets.

        Args:
            dataset_rid: A dataset RID.

        Returns:
          list of RIDs of nested datasets.

        """
        return [d["RID"] for d in self.list_dataset_members(dataset_rid)["Dataset"]]

    @validate_call
    def add_dataset_members(
        self, dataset_rid: Optional[RID], members: list[RID], validate: bool = True
    ) -> None:
        """Add additional elements to an existing dataset.

        Args:
            dataset_rid: RID of dataset to extend or None if new dataset is to be created.
            members: List of RIDs of members to add to the  dataset.
            validate: Check rid_list to make sure elements are not already in the dataset.
        """

        members = set(members)

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
                    f"Attempting to add existing member to dataset {dataset_rid}: {overlap}"
                )

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        association_map = {
            a.other_fkeys.pop().pk_table.name: a.table.name
            for a in self.table.find_associations()
        }
        # Get a list of all the types of objects that can be linked to a dataset.
        for m in members:
            rid_info = self._resolve_rid(m)
            if rid_info.table.name not in association_map:
                raise DerivaMLException(
                    f"RID table: {rid_info.table.name} not part of dataset"
                )
            if rid_info.table == self.table and check_dataset_cycle(rid_info.rid):
                raise DerivaMLException("Creating cycle of datasets is not allowed")
            dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        pb = self._model.catalog.getPathBuilder()
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[
                self._ml_schema if table == "Dataset" else self._domain_schema
            ]
            fk_column = "Nested_Dataset" if table == "Dataset" else table

            if len(elements):
                # Find out the name of the column in the association table.
                schema_path.tables[association_map[table]].insert(
                    [{"Dataset": dataset_rid, fk_column: e} for e in elements]
                )

    @validate_call
    def list_dataset_members(
        self, dataset_rid: RID, version=Optional[SemanticVersion], recurse: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a list of entities associated with a specific dataset.

        Args:
            dataset_rid: param recurse: If this is a nested dataset, list the members of the contained datasets
            version: Semantic version of the dataset to be returned.
            recurse:  (Default value = False)

        Returns:
            Dictionary of entities associated with a specific dataset.  Key is the table from which the elements
            were taken.

        Raises:
            DerivaMLException: if the RID doesn't exist. or semantic version doesn't match existing version.
        """

        try:
            if not self._is_dataset_rid(dataset_rid):
                raise DerivaMLException(f"RID is not for a dataset: {dataset_rid}")
        except DerivaMLException:
            raise DerivaMLException(f"Invalid RID: {dataset_rid}")

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        pb = self._model.catalog.getPathBuilder()
        for assoc_table in self.table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            if (
                target_table.schema.name != self._domain_schema
                and target_table != self.table
            ):
                # Look at domain tables and nested datasets.
                continue
            if target_table == self.table:
                # find_assoc gives us the keys in the wrong position, so swap.
                self_fkey, other_fkey = other_fkey, self_fkey

            target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
            member_path = pb.schemas[member_table.schema.name].tables[member_table.name]
            # Get the names of the columns that we are going to need for linking
            member_link = tuple(
                c.name for c in next(iter(other_fkey.column_map.items()))
            )
            path = pb.schemas[member_table.schema.name].tables[member_table.name].path
            path.filter(member_path.Dataset == dataset_rid)
            path.link(
                target_path,
                on=(
                    member_path.columns[member_link[0]]
                    == target_path.columns[member_link[1]]
                ),
            )
            target_entities = path.entities().fetch()
            members[target_table.name].extend(target_entities)
            if recurse and target_table == self.table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds in nested_datasets:
                    for k, v in self.list_dataset_members(ds, recurse=False).items():
                        members[k].extend(v)
        return members

    @staticmethod
    def export_dataset_element(
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
        # The table is the last element of the path.  Generate the ERMrest query by conversting the list of tables
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

    def download_dataset_element(
        self, spath, dpath, table: Table
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
    def _is_vocabulary(t: Table) -> bool:
        """

        Args:
          t: Table:

        Returns:
            True if the table has a vocabulary, False otherwise.
        """
        vocab_columns = {"Name", "URI", "Synonyms", "Description", "ID"}
        return vocab_columns.issubset({c.name for c in t.columns}) and t

    def _lookup_term(self, vocab_table: Table, term_name: str) -> VocabularyTerm:
        """Given a term name, return the vocabulary record.  Can provide either the term name
         or a synonym for the term.  Generate an exception if the term is not in the vocabulary.

        Args:
            vocab_table: The name of the controlled vocabulary table or a ERMrest table object..
            term_name: The name of the term to look up.

        Returns:
          The entry the associated term or synonym.

        Raises:
          DerivaException: If the schema or vocabulary table doesn't exist, or if the term is not
            found in the vocabulary.
        """
        if not self._is_vocabulary(vocab_table):
            raise DerivaMLException(f"{vocab_table.name} not a vocabulary")
        schema_name, table_name = vocab_table.schema.name, vocab_table.name
        schema_path = self._model.catalog.getPathBuilder().schemas[schema_name]
        for term in schema_path.tables[table_name].entities():
            if term_name == term["Name"] or (
                term["Synonyms"] and term_name in term["Synonyms"]
            ):
                return VocabularyTerm.model_validate(term)
        raise DerivaMLException(f"Term {term_name} is not in vocabulary {table_name}")

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
            if self._is_vocabulary(table)
        ]
        return [
            o
            for table in vocabs
            for o in writer(f"{table.schema.name}:{table.name}", table.name, table)
        ]

    def _domain_table_paths(
        self,
        graph: dict[Table, list[dict[Table, Any]]],
        spath: str = None,
        dpath: str = None,
        sprefix: str = "deriva-ml:Dataset/RID={Dataset_RID}",
        dprefix: str = "Dataset",
    ) -> list[tuple[str, str, Table]]:
        """Recursively walk over the domain schema graph and extend the current path.

        Args:
            graph: An undirected, acyclic graph of schema.  Represented as a dictionary whose name is the table name.
                and whose values are the child nodes of the table.
            spath: Source path so far
            dpath: Destination path so far
            sprefix: Initial path to be included.  Allows for nested datasets
            dprefix: Initial path to be included.  Allows for nested datasets

        Returns:
          A list of all the paths through the graph.  Each path is a list of tables.

        """
        source_path = spath or sprefix
        dest_path = dpath or dprefix
        paths = []
        for node, children in graph.items():
            if node.name == "Dataset":
                new_spath = sprefix
                new_dpath = dprefix
            else:
                new_spath = source_path + f"/{node.schema.name}:{node.name}"
                new_dpath = dest_path + f"/{node.name}"
            paths.append((new_spath, new_dpath, node))
            for child in children:
                paths.extend(self._domain_table_paths(child, new_spath, new_dpath))
        return paths

    def _table_paths(self, graph) -> list[tuple[str, str, Table]]:
        sprefix, dprefix = (
            "deriva-ml:Dataset/RID={Dataset_RID}",
            "Dataset",
        )
        table_paths = self._domain_table_paths(graph, sprefix=sprefix, dprefix=dprefix)
        dataset_dataset_table = self._model.schemas[self._ml_schema].tables[
            "Dataset_Dataset"
        ]
        nested_sprefix = sprefix
        nested_dprefix = dprefix
        for i in range(1, 3):
            #    nested_sprefix += f'/DD{i}:=deriva-ml:Dataset_Dataset/D{i+1}:=(Nested_Dataset)=(deriva-ml:Dataset:RID)'
            nested_sprefix += f"/(RID)=(deriva-ml:Dataset_Dataset:Dataset)"
            nested_dprefix += f"/Dataset_Dataset"
            table_paths.append((nested_sprefix, nested_dprefix, dataset_dataset_table))
            nested_sprefix += f"/(Nested_Dataset)=(deriva-ml:Dataset:RID)"
            nested_dprefix += f"/Dataset"
            table_paths.append((nested_sprefix, nested_dprefix, self.table))
            # Get CSV for nested datasets.
            table_paths.extend(
                self._domain_table_paths(
                    graph, sprefix=nested_sprefix, dprefix=nested_dprefix
                )[1:]
            )
        return table_paths

    def _dataset_nesting_depth(self) -> int:
        ds_path = (
            self._model.catalog.getPathBuilder()
            .schemas[ML_SCHEMA]
            .tables["Dataset_Dataset"]
        )
        dsets = list(
            ds_path.attributes(ds_path.Dataset, ds_path.Nested_Dataset).fetch()
        )
        tree_depth = 3
        return 2

    def _schema_graph(
        self, node: Table, visited_nodes: Optional[set] = None
    ) -> dict[Table, list[dict[Table, list]]]:
        """Generate an undirected, acyclic graph of domain schema. We do this by traversing the schema foreign key
        relationships.  We stop when we hit the deriva-ml schema or when we reach a node that we have already seen.

        Nested datasets need to be unfolded

        Args:
          node: Current (starting) node in the graph.
          visited_nodes: param nested_dataset: Are we in a nested dataset, (i.e. have we seen the DataSet table)?

        Returns:
            Graph of the schema, starting from node.
        """

        visited_nodes = visited_nodes or set()
        graph = {node: []}

        def include_node(child: Table) -> bool:
            """Indicate if the table should be included in the graph.

            Include node in the graph if it's not a loopback from fk<-> referred_by, you have not already been to the
            node.
            """
            return (
                child != node
                and child not in visited_nodes
                and child.schema.name == self._domain_schema
            )

        # Get all the tables reachable from the end of the path avoiding loops from T1<->T2 via referenced_by
        nodes = {fk.pk_table for fk in node.foreign_keys if include_node(fk.pk_table)}
        nodes |= {fk.table for fk in node.referenced_by if include_node(fk.table)}

        for t in nodes:
            new_visited_nodes = visited_nodes.copy()
            new_visited_nodes.add(t)
            if self._is_vocabulary(t):
                # If the end of the path is a vocabulary table, we are at a terminal node in the ERD, so stop
                continue
            # Get all the paths that extend the current path
            graph[node].append(self._schema_graph(t, new_visited_nodes))
        return graph

    def _dataset_specification(
        self, writer: Callable[[str, str, Table], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Output a download/export specification for a dataset.  Each element of the dataset will be placed in its own dir
        The top level data directory of the resulting BDBag will have one subdirectory for element type. the subdirectory
        will contain the CSV indicating which elements of that type are present in the dataset, and then there will be a
        subdirectories for each object that is reachable from the dataset members.

        To simplify reconstructing the relationship between tables, the CVS for each
        The top level data directory will also contain a subdirectory for any controlled vocabularies used in the dataset.
        All assets will be placed into a directory named asset in a subdirectory with the asset table name.

        For example, consider a dataset that consists of two element types, T1 and T2. T1 has foreign key relationships to
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
            A dataset specification.
        """
        element_spec = []
        for path in self._table_paths(self._schema_graph(self.table)):
            element_spec.extend(writer(*path))
        return self._vocabulary_specification(writer) + element_spec

    def export_outputs(self) -> list[dict[str, Any]]:
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
            return self.export_dataset_element(spath, dpath, table)

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
            return self.download_dataset_element(spath, dpath, table)

        # Download spec is the spec for any controlled vocabulary and for the dataset.
        return [
            {
                "processor": "json",
                "processor_params": {"query_path": f"/schema", "output_path": "schema"},
            }
        ] + self._dataset_specification(writer)

    def generate_dataset_download_spec(self) -> dict[str, Any]:
        """

        Returns:
        """
        return {
            "bag": {
                "bag_name": "Dataset_{Dataset_RID}",
                "bag_algorithms": ["md5"],
                "bag_archiver": "zip",
                "bag_metadata": {},
                "bag_idempotent": True,
            },
            "catalog": {
                "host": f"{self._model.catalog.deriva_server.scheme}://{self._model.catalog.deriva_server.server}",
                "catalog_id": f"{self._model.catalog.catalog_id}",
                "query_processors": [
                    {
                        "processor": "env",
                        "processor_params": {
                            "query_path": "/",
                            "output_path": "Dataset",
                            "query_keys": ["snaptime"],
                        },
                    }
                ]
                + self._processor_params(),
            },
        }

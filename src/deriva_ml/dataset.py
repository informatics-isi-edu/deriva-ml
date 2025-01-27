"""
THis module defines the DataSet class with is used to manipulate n
"""

from typing import Any, Callable, Optional

from deriva.core.ermrest_model import Model, Table

from deriva_ml.deriva_definitions import ML_SCHEMA


class Dataset:
    """
    Class to manipulate a dataset.

    Attributes:
        table: ERMrest table holding dataset information.
    """

    def __init__(self, model: Model):
        self._model = model
        self.ml_schema = ML_SCHEMA
        self._domain_schema = [
            s for s in model.schemas if s not in ["deriva-ml", "www", "public"]
        ].pop()
        self.table = self._model.schemas[self.ml_schema].tables["Dataset"]

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
        dataset_dataset_table = self._model.schemas[self.ml_schema].tables['Dataset_Dataset']
        nested_sprefix = sprefix
        nested_dprefix = dprefix
        for i in range(1, 3):
        #    nested_sprefix += f'/DD{i}:=deriva-ml:Dataset_Dataset/D{i+1}:=(Nested_Dataset)=(deriva-ml:Dataset:RID)'
            nested_sprefix += f'/(RID)=(deriva-ml:Dataset_Dataset:Dataset)'
            nested_dprefix += f'/Dataset_Dataset'
            table_paths.append((nested_sprefix, nested_dprefix, dataset_dataset_table))
            nested_sprefix += f'/(Nested_Dataset)=(deriva-ml:Dataset:RID)'
            nested_dprefix += f'/Dataset'
            table_paths.append((nested_sprefix, nested_dprefix, self.table))
        # Get CSV for nested datasets.
            table_paths.extend(self._domain_table_paths(graph, sprefix=nested_sprefix, dprefix=nested_dprefix)[1:])
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

from csv import reader
from deriva.core.ermrest_model import Model, Table
import logging
from pathlib import Path
import pandas as pd
from urllib.parse import urlparse
import sqlite3
from typing import Optional, Any, Generator, Callable


def export_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    """
    Given a path to the data model, output an export specification for the path taken to get to the current table.
    :param path:
    :return:
    """

    # The table is the last element of the path.  Generate the ERMrest query by conversting the list of tables
    # into a path in the form of /S:T1/S:T2/S:Table
    # Generate the destination path in the file system using just the table names.
    table = path[-1]
    npath = "Dataset/RID={Dataset_RID}/" if path[0].name == "Dataset" else path[0].name
    npath += '/'.join([f'{t.schema.name}:{t.name}' for t in path[1:]])

    dname =  '/'.join([t.name for t in path if not t.is_association()] + [table.name])
    exports = [
        {
            'source': {'api': 'entity', 'path': f'{npath}'},
            'destination': {
                'name': dname,
                'type': 'csv'
            }
        }
    ]

    # If this table is an asset table, then we need to output the files associated with the asset.
    asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
    if asset_columns.issubset({c.name for c in table.columns}):
        exports.append({
            'source': {
                'api': 'attribute',
                'path': f'{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5'
            },
            'destination': {'name': f'assets/{table.name}', 'type': 'fetch'}
        }
        )
    return exports


def download_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    table = path[-1]
    npath = "Dataset/RID={Dataset_RID}/" if path[0].name == "Dataset" else path[0].name
    npath += '/'.join([f'{t.schema.name}:{t.name}' for t in path[1:]])
    output_path = '/'.join([p.name for p in path if not p.is_association()] + [table.name])
    exports = [
        {
            "processor": "csv",
            "processor_params": {
                'query_path': f'/entity/{npath}?limit=none',
                'output_path': output_path
            }
        }
    ]
    asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
    if asset_columns.issubset({c.name for c in table.columns}):
        exports.append({
            'processor': 'fetch',
            'processor_params': {
                'query_path': f'/attribute/{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5?limit=none',
                'output_path': f'assets/{table.name}'
            }
        }
        )
    return exports


def is_vocabulary(t):
    vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
    return vocab_columns.issubset({c.name for c in t.columns}) and t


def vocabulary_specification(model, writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    vocabs = [table for s in model.schemas.values() for table in s.tables.values() if is_vocabulary(table)]
    return [o for table in vocabs for o in writer([table])]


def dataset_specification(model: Model,
                          writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Output a download/export specification for a dataset.  Each element of the dataset will be placed in its own dir
    The top level data directory of the resulting BDBag will have one sub-directory for element type. the subdirectory
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
            assets/
              T2
                f1
                f2


    :param model:
    :param writer:
    :return:
    """
    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()

    element_spec = []
    for element in dataset_table.find_associations():
        # A dataset may have may other object associated with it. We only want to consider those association tables
        #  that are in the domain schema, or the table Dataset_Dataset, which is used for nested datasets.
        if element.table.schema.name == domain_schema or element.name == "Dataset_Dataset":
            # Now generate all of the paths reachable from this node.
            for path in DatasetBag.table_paths(DatasetBag.schema_graph(model, element.table), [dataset_table]):
                table = path[-1]
               # if table.is_association(max_arity=3, pure=False):
               #     continue
                element_spec.extend(writer(path))
    return vocabulary_specification(model, writer) + element_spec


def export_outputs(model: Model) -> list[dict[str, Any]]:
    """
    Return and output specification for the datasets in the provided model
    :param model: An ermrest model.
    :return: An export specification suitable for Chaise.
    """
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return export_dataset_element(path)

    # Export specification is a specification for the datasets, plus any controlled vocabulary
    return [
        {'source': {'api': False, 'skip_root_path': True},
         'destination': {'type': 'env', 'params': {'query_keys': ['snaptime']}}
         },
        {'source': {'api': 'entity'},
         'destination': {'type': 'env', 'params': {'query_keys': ['RID', 'Description']}}
         },
        {'source': {"api": "schema", "skip_root_path": True},
         'destination': {'type': 'json', 'name': 'schema'},
        }
    ] + dataset_specification(model, writer)


def processor_params(model: Model) -> list[dict[str, Any]]:
    """

    :param model: current ERMrest Model
    :return: a download specification for the datasets in the provided model.
    """
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return download_dataset_element(path)

    # Download spec is the spec for any controlled vocabulary and for the dataset.
    return [
        {
            "processor": "json",
            "processor_params": {'query_path': f'/schema', 'output_path': 'schema'}
        }
    ] + dataset_specification(model, writer)

def generate_dataset_download_spec(model: Model) -> dict[str, Any]:
    return {
        "bag": {
            "bag_name": "Dataset_{Dataset_RID}",
            "bag_algorithms": ["md5"],
            "bag_archiver": "zip",
            "bag_metadata": {},
            "bag_idempotent": True
        },
        "catalog": {
            "host": f"{model.catalog.deriva_server.scheme}://{model.catalog.deriva_server.server}",
            "catalog_id": f"{model.catalog.catalog_id}",
            "query_processors":
                [
                    {
                        "processor": "env",
                        "processor_params": {"query_path": "/", "output_path": "Dataset", "query_keys": ["snaptime"]}
                    }
                 ] + processor_params(model)
        }
    }


class DatasetBag(object):
    def __init__(self, bag_path: Path | str):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.bag_path = Path(bag_path)
        self.dataset_rid = self.bag_path.name.replace('Dataset_','')
        self.model = Model.fromfile('file-system', self.bag_path / 'data/schema.json')
        self.domain_schema = [s for s in self.model.schemas if s not in ['deriva-ml', 'public', 'www']][0]
        self.dbase = sqlite3.connect(f"{self.bag_path / self.domain_schema}.db")
        with self.dbase:
            for t in self.model.schemas[self.domain_schema].tables.values():
                self.dbase.execute(t.sqlite3_ddl())
            for t in self.model.schemas['deriva-ml'].tables.values():
                self.dbase.execute(t.sqlite3_ddl())
        self._load_sqllite()

    @staticmethod
    def table_paths(graph: dict[str, Any], path: Optional[list[Table]] = None) -> list[list[Table]]:
        """
        Recursively walk over the domain schema and extend the current path.
        :param graph:
        :return:
        """
        path = path or []
        paths = []

        for node, children in graph.items():
            new_path = path + [node]
            paths.append(new_path)
            for child in children:
                paths.extend(DatasetBag.table_paths(child, new_path))
        return paths

    @staticmethod
    def schema_graph(model: Model,
                     node: str | Table,
                     visited_nodes: Optional[set] = None,
                     nested_dataset: bool = False) -> dict[str, Any]:
        domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()
        dataset_table = model.schemas['deriva-ml'].tables["Dataset"]

        def domain_table(table: Table) -> bool:
            return table.schema.name == domain_schema or table.name == "Dataset"


        dataset_associations = [a.table for a in dataset_table.find_associations() if
                                domain_table(a.other_fkeys.pop().pk_table)]
        visited_nodes = visited_nodes or set()
        graph = {node: []}

        def include_node(child: Table):
            # Include node in the graph if its not a loopback from fk<-> refered_by, you have not already been to the
            # node, its not a association table back to the dataset, or it is a nested dataset.
            return (
                    (
                            child != node and
                            child not in visited_nodes and
                            child.schema.name == domain_schema and
                            child not in dataset_associations
                    ) or
                    # Include nested datasets of level 1
                    (child.name == "Dataset" and nested_dataset)
            )

        # Get all the tables reachable from the end of the path avoiding loops from T1<->T2 via referenced_by
        nodes = {fk.pk_table for fk in node.foreign_keys if include_node(fk.pk_table)}
        nodes |= {fk.table for fk in node.referenced_by if include_node(fk.table)}
        for t in nodes:
            new_visited_nodes = visited_nodes.copy()
            new_visited_nodes.add(t)
            if is_vocabulary(t):
                # If the end of the path is a vocabulary table, we are at a terminal node in the ERD, so stop
                continue
            # Get all the paths that extend the current path
            graph[node].append(DatasetBag.schema_graph(model, t,
                                                 new_visited_nodes,
                                                 nested_dataset=nested_dataset or t.name == "Dataset"))
        return graph

    def localize_asset_table(self) -> dict[str, str]:
        fetch_map = {}
        try:
            with open(self.bag_path / 'fetch.txt', newline='\n') as fetchfile:
                for row in fetchfile:
                    fields = row.split('\t')
                    fetch_map[urlparse(fields[0]).path] = fields[2].replace('\n', '')
        except FileNotFoundError:
            logging.info(f'No downloaded assets in bag {self.dataset_rid}')
        return fetch_map

    def _reset_dbase(self):
        DatasetBag.delete_database(self.bag_path, self.domain_schema)
        self.dbase.execute("DROP DATABASE")

    def _load_sqllite(self) -> None:
        dpath = self.bag_path / "data"
        asset_map = self.localize_asset_table()

        def is_asset(table_name: str) -> bool:
            asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
            schema = self.domain_schema if table_name in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
            asset_table = self.model.schemas[schema].tables[table_name]
            return asset_columns.issubset({c.name for c in asset_table.columns})

        def localize_asset(o: list, asset_indexes: Optional[tuple[int, int]]) -> tuple:
            if asset_indexes:
                file_column, url_column = asset_indexes
                o[file_column] = asset_map[o[url_column]] if o[url_column] else ''
            return tuple(o)

        # for path, subdirs, files in dpath.walk():
        for csv_file in Path(dpath).rglob('*.csv'):          # table = path.name
            table = csv_file.stem
            schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
            # if f"{table}.csv" not in files:
            #     continue   # Some directories might be empty.
            # with open(path / f"{table}.csv", newline='') as csvfile:
            with csv_file.open(newline='') as csvfile:
                csv_reader = reader(csvfile)
                column_names = next(csv_reader)
                asset_indexes = (column_names.index('Filename'), column_names.index('URL')) if is_asset(table) else None
                object_table = [localize_asset(o, asset_indexes) for o in csv_reader]
                value_template = ','.join(['?'] * len(column_names))  # SQL placeholder for row (?,?..)
                column_list = ','.join([f'"{c}"' for c in column_names])
                with self.dbase:
                    self.dbase.executemany(
                        f'INSERT OR REPLACE INTO "{schema}:{table}" ({column_list}) VALUES ({value_template})',
                        object_table
                    )

    def list_tables(self) -> list[str]:
        with self.dbase:
            return [t[0] for t in self.dbase.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;").fetchall()]

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
        result = self.dbase.execute(f'SELECT * FROM "{schema}:{table}"')
        while row := result.fetchone():
            yield row

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
        return pd.read_sql(f'SELECT * FROM "{schema}:{table}"', con=self.dbase)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
        with self.dbase:
            col_names = [c[1] for c in self.dbase.execute(f'PRAGMA table_info("{schema}:{table}")').fetchall()]
            result = self.dbase.execute(f'SELECT * FROM "{schema}:{table}"')
            while row := result.fetchone():
                yield dict(zip(col_names, row))

    @staticmethod
    def delete_database(bag_path, schema):
        dbase_path = Path(bag_path) / f'{schema}.db'
        dbase_path.unlink()



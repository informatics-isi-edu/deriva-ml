from csv import reader
from deriva.core.ermrest_model import Model, Table
import logging
from pathlib import Path
import pandas as pd
from urllib.parse import urlparse
import sqlite3
from typing import Optional, Any

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
    def schema_graph(model: Model,
                     node: str | Table,
                     visited_nodes: Optional[set] = None,
                     nested_dataset: bool = False) -> dict[str, Any]:
        domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()
        dataset_table = model.schemas['deriva-ml'].tables["Dataset"]

        def domain_table(table: Table) -> bool:
            return table.schema.name == domain_schema or table.name == "Dataset"

        def is_vocabulary(t):
            vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
            return vocab_columns.issubset({c.name for c in t.columns}) and t

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
        dpath = self.bag_path / "data/Dataset"
        asset_map = self.localize_asset_table()

        def is_asset(table_name: str) -> bool:
            asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
            schema = self.domain_schema if table_name in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
            table = self.model.schemas[schema].tables[table_name]
            return asset_columns.issubset({c.name for c in table.columns})

        def localize_asset(o: list, asset_indexes: Optional[tuple[int, int]]) -> tuple:
            if asset_indexes:
                file_column, url_column = asset_indexes
                o[file_column] = asset_map[o[url_column]]
            return tuple(o)

        for path, subdirs, files in dpath.walk():
            table = path.name
            schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'

            if f"{table}.csv" not in files:
                continue   # Some directories might be empty.
            with open(path / f"{table}.csv", newline='') as csvfile:
                csv_reader = reader(csvfile)
                column_names = next(csv_reader)
                asset_indexes = (column_names.index('Filename'), column_names.index('URL')) if is_asset(table) else None
                object_table = [localize_asset(o, asset_indexes) for o in csv_reader]
                value_template = ','.join(['?'] * len(column_names))  # SQL placeholder for row (?,?..)
                column_list = ','.join(column_names)
                with self.dbase:
                    self.dbase.executemany(
                        f'INSERT OR REPLACE INTO "{schema}:{table}" ({column_list}) VALUES ({value_template})',
                                object_table
                    )

    def get_table(self, table: str):
        schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
        return self.dbase.execute(f'SELECT * FROM "{schema}:{table}"').fetchall()

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
        return pd.read_sql(f'SELECT * FROM "{schema}:{table}"', con=self.dbase)

    @staticmethod
    def delete_database(bag_path, schema):
        dbase_path = Path(bag_path) / f'{schema}.db'
        dbase_path.unlink()



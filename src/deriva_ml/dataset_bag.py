from csv import reader
from deriva.core.ermrest_model import Model
from deriva_ml.deriva_ml_base import DerivaMLException
import logging
from pathlib import Path
import pandas as pd
import sqlite3
from typing import Optional, Any, Generator
from urllib.parse import urlparse

class DatasetBag(object):
    """
    DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized BDBag for a
    dataset, which is created either by DerivaML.create_execution, or directly by calling DerivaML.download_dataset.

    As part of its initialization, this routine will create a sqlite database that has the contents of all the tables
    in the dataset.  In addition, any asset tables will the `Filename` column remapped to have the path of the local
    copy of the file. In addition, a local version of the ERMRest model that as used to generate the dataset is
    available.

   The sqllite database will not have any foreign key constraints applied, however, foreign-key relationships can be
    found by looking in the ERMrest model.  In addition, as sqllite doesn't support schema, Ermrest schema are added
    to the table name using the convention SchemaName:TableName.  Methods in DatasetBag that have table names as the
    argument will perform the appropriate name mappings.

    Attributes:
        dbase: A connection to the sqlite database.
        domain_schema The name of the domain schema for the dataset.
        dataset_rid: The name of the dataset

    Methods:
        get_table(self, table: str) -> Generator[tuple, None, None]
        get_table_as_dataframe(self, table: str) -> pd.DataFrame
        get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]
        list_tables(self) -> list[str]


    """
    def __init__(self, bag_path: Path | str):
        """
        Initialize a DatasetBag instance.

        :param bag_path: A path to a materialized BDbag as returned by download_dataset_bag or create_execution.
        """
        self.bag_path = Path(bag_path)
        self.dataset_rid = self.bag_path.name.replace('Dataset_','')
        self.model = Model.fromfile('file-system', self.bag_path / 'data/schema.json')

        # Guess the domain schema name by eliminating all the "builtin" schema.
        self.domain_schema = [s for s in self.model.schemas if s not in ['deriva-ml', 'public', 'www']][0]
        self.dbase = sqlite3.connect(f"{self.bag_path / self.domain_schema}.db")

        # Create a sqlite database schema that contains all the tables within the catalog from which the
        # BDBag was created.
        with self.dbase:
            for t in self.model.schemas[self.domain_schema].tables.values():
                self.dbase.execute(t.sqlite3_ddl())
            for t in self.model.schemas['deriva-ml'].tables.values():
                self.dbase.execute(t.sqlite3_ddl())

        # Load the database from the bag contents.
        self._load_sqllite()

    def _localize_asset_table(self) -> dict[str, str]:
        """
        Use the fetch.txt file in a bdbag to create a map from a URL to a local file path.

        :return: Dictionary that maps a URL to a local file path.
        """
        fetch_map = {}
        try:
            with open(self.bag_path / 'fetch.txt', newline='\n') as fetchfile:
                for row in fetchfile:
                    # Rows in fetch.text are tab seperated with URL filename.
                    fields = row.split('\t')
                    fetch_map[urlparse(fields[0]).path] = fields[2].replace('\n', '')
        except FileNotFoundError:
            logging.info(f'No downloaded assets in bag {self.dataset_rid}')
        return fetch_map

    def _reset_dbase(self):
        DatasetBag.delete_database(self.bag_path, self.domain_schema)
        self.dbase.execute("DROP DATABASE")

    def _load_sqllite(self) -> None:
        """
        Load a SQLite database from a bdbag.  THis is done by looking for all the CSV files in the bdbag directory.
        If the file is for an asset table, update the FileName column of the table to have the local file path for
        the materialized file.  Then load into the sqllite database.
        Note: none of the foreign key constraints are included in the database.

        :return:
        """
        dpath = self.bag_path / "data"
        asset_map = self._localize_asset_table()

        def is_asset(table_name: str) -> bool:
            asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
            sname = self.domain_schema if table_name in self.model.schemas[self.domain_schema].tables else 'deriva-ml'
            asset_table = self.model.schemas[sname].tables[table_name]
            return asset_columns.issubset({c.name for c in asset_table.columns})

        def localize_asset(o: list, indexes: Optional[tuple[int, int]]) -> tuple:
            """
            Given a list of column values for a table, replace the FileName column with the local file name based on
            the URL value.

            :param o: List of values for each column in a table row.
            :param indexes:  A tuple whose first element is the column index of the file name and whose second element
                             is the index of the URL in an asset table.  Tuple is None if table is not an asset table.
            :return: Tuple of updated column values.
            """
            if indexes:
                file_column, url_column = asset_indexes
                o[file_column] = asset_map[o[url_column]] if o[url_column] else ''
            return tuple(o)

        # Find all the CSV files in the subdirectory and load each file into the database.
        for csv_file in Path(dpath).rglob('*.csv'):
            table = csv_file.stem
            schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else 'deriva-ml'

            with csv_file.open(newline='') as csvfile:
                csv_reader = reader(csvfile)
                column_names = next(csv_reader)

                # Determine which columns in the table has the Filename and the URL
                asset_indexes = (column_names.index('Filename'), column_names.index('URL')) if is_asset(table) else None

                value_template = ','.join(['?'] * len(column_names))  # SQL placeholder for row (?,?..)
                column_list = ','.join([f'"{c}"' for c in column_names])
                with self.dbase:
                    object_table = (localize_asset(o, asset_indexes) for o in csv_reader)
                    self.dbase.executemany(
                        f'INSERT OR REPLACE INTO "{schema}:{table}" ({column_list}) VALUES ({value_template})',
                        object_table
                    )

    def list_tables(self) -> list[str]:
        """
        Return a list of all the table names in the dataset. The schema name is included in the table name seperated
        by a ':'

        :return:  List of table names.
        """
        with self.dbase:
            return [t[0] for t in self.dbase.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;").fetchall()]

    def _normalize_table_name(self, table: str) -> str:
        """
        Attempt to insert the schema into a table name if its not provided.
        :param table_name:
        :return: table name with schema included.
        """
        sname = ""
        try:
            [sname,tname] = table.split(':')
        except ValueError:
            tname = table
            for sname, s in self.model.schemas.items():
                if table in s.tables:
                    break
        try:
            _ = self.model.schemas[sname].tables[tname]
            return f'{sname}:{tname}'
        except KeyError:
            raise DerivaMLException(f'Table name "{table}" does not exist.')

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        """
        Retrieve the contents of the specified table. If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        :param table:
        :return: A generator that yields tuples of column values.
        """
        table_name = self._normalize_table_name(table)
        result = self.dbase.execute(f'SELECT * FROM "{table_name}"')
        while row := result.fetchone():
            yield row

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """
        Retrieve the contents of the specified table as a dataframe.
        If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        :param table: Table to retrieve data from.
        :return: A dataframe containing the contents of the specified table.
        """
        table_name = self._normalize_table_name(table)
        print(table_name)
        return pd.read_sql(f'SELECT * FROM "{table_name}"', con=self.dbase)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """
        Retrieve the contents of the specified table as a dictionary.
        :param table: Table to retrieve data from.
         If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        :return: A generator producing dictionaries containing the contents of the specified table as name/value pairs.
        """
        table_name = self._normalize_table_name(table)
        with self.dbase:
            col_names = [c[1] for c in self.dbase.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
            result = self.dbase.execute(f'SELECT * FROM "{table_name}"')
            while row := result.fetchone():
                yield dict(zip(col_names, row))

    @staticmethod
    def delete_database(bag_path, schema):
        dbase_path = Path(bag_path) / f'{schema}.db'
        dbase_path.unlink()



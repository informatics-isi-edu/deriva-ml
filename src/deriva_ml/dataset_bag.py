import logging
import sqlite3
from collections import defaultdict
from copy import copy
from csv import reader
from pathlib import Path
import tempfile
from typing import Any, Generator, Optional
from urllib.parse import urlparse

import pandas as pd
from deriva.core.ermrest_model import Model
from pydantic import validate_call

from .deriva_definitions import ML_SCHEMA, MLVocab, RID
from .deriva_ml_base import DerivaMLException


class DatasetBag(object):
    """DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized BDBag for a
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
        dbase: A sqlite database that has the contents of all the tables in the dataset.

    Methods:
        get_table(table: str) -> Generator[tuple, None, None]
        get_table_as_dataframe(table: str) -> pd.DataFrame
        get_table_as_dict(table: str) -> Generator[dict[str, Any], None, None]
        list_tables() -> list[str]
    """

    dbase: Optional[sqlite3.Connection] = None
    _model: Optional[Model] = None
    _rids_loaded: dict[RID:Path] = {}
    _paths_loaded: set[Path] = set()

    @validate_call
    def __init__(self, dataset: Path | RID, dbase_dir: Optional[Path] = None) -> None:
        """
        Initialize a DatasetBag instance.

        Args:
            dataset: A path to a BDBag or a dataset, or the RID of an already loaded dataset.
            dbase_dir: Optional path as to where to place the sqllite database file.
        """

        if dataset in DatasetBag._rids_loaded:
            # We have already loaded this RID already, so just pick up the info and return.
            self.bag_path = DatasetBag._rids_loaded[dataset]
            self.dataset_rid = dataset
            self._ml_schema = ML_SCHEMA
        else:
            self.bag_path = Path(dataset) if isinstance(dataset, str) else dataset
            self.dataset_rid = dataset.name.replace("Dataset_", "")
            self._ml_schema = ML_SCHEMA
            if self.bag_path not in DatasetBag._paths_loaded:
                # This is the first time we have seen this bag, so we need to create a database for it and
                # load it up.
                self._create_database(dbase_dir)
                self._domain_schema = self._guess_domain_schema()
                self._ml_schema = ML_SCHEMA
                self._load_model()
                self._load_sqllite()
                DatasetBag._paths_loaded.add(self.bag_path)
                DatasetBag._rids_loaded[self.dataset_rid] = self.bag_path

                # If this is a nested dataset, we want to know what RIDs are in it.
                for d in self.list_dataset_children(recurse=True):
                    self._rids_loaded[d] = self.bag_path
        self._domain_schema = self._guess_domain_schema()
        self.dataset_table = DatasetBag._model.schemas[self._ml_schema].tables[
            "Dataset"
        ]

    @staticmethod
    def _guess_domain_schema():
        # Guess the domain schema name by eliminating all the "builtin" schema.
        return [
            s
            for s in DatasetBag._model.schemas
            if s not in ["deriva-ml", "public", "www"]
        ][0]

    def _create_database(self, dir_path: Optional[Path] = None) -> None:
        if not DatasetBag._model:
            DatasetBag._model = Model.fromfile(
                "file-system", self.bag_path / "data/schema.json"
            )
            dir_path = dir_path or Path(tempfile.mkdtemp())
            dbase_file = dir_path / "dataset.db"
            DatasetBag.dbase = sqlite3.connect(dbase_file)

    def _load_model(self):
        # Create a sqlite database schema that contains all the tables within the catalog from which the
        # BDBag was created.
        with DatasetBag.dbase:
            for t in DatasetBag._model.schemas[self._domain_schema].tables.values():
                DatasetBag.dbase.execute(t.sqlite3_ddl())
            for t in DatasetBag._model.schemas["deriva-ml"].tables.values():
                DatasetBag.dbase.execute(t.sqlite3_ddl())

    def _localize_asset_table(self) -> dict[str, str]:
        """Use the fetch.txt file in a bdbag to create a map from a URL to a local file path.

        Returns:
            Dictionary that maps a URL to a local file path.

        """
        fetch_map = {}
        try:
            with open(self.bag_path / "fetch.txt", newline="\n") as fetchfile:
                for row in fetchfile:
                    # Rows in fetch.text are tab seperated with URL filename.
                    fields = row.split("\t")
                    local_file = fields[2].replace('\n', "")
                    local_path =  f'{self.bag_path}/{local_file}'
                    fetch_map[urlparse(fields[0]).path] = local_path
        except FileNotFoundError:
            dataset_rid = self.bag_path.name.replace("Dataset_", "")
            logging.info(f"No downloaded assets in bag {dataset_rid}")
        return fetch_map

    def _is_asset(self, table_name: str) -> bool:
        """

        Args:
          table_name: str:

        Returns:

        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        sname = (
            self._domain_schema
            if table_name in DatasetBag._model.schemas[self._domain_schema].tables
            else self._ml_schema
        )
        asset_table = DatasetBag._model.schemas[sname].tables[table_name]
        return asset_columns.issubset({c.name for c in asset_table.columns})

    @staticmethod
    def _localize_asset(
        o: list, indexes: tuple[int, int], asset_map: dict[str, str]
    ) -> tuple:
        """Given a list of column values for a table, replace the FileName column with the local file name based on
        the URL value.

        Args:
          o: List of values for each column in a table row.
          indexes: A tuple whose first element is the column index of the file name and whose second element
        is the index of the URL in an asset table.  Tuple is None if table is not an asset table.
          o: list:
          indexes: Optional[tuple[int: int]]:

        Returns:
          Tuple of updated column values.

        """
        if indexes:
            file_column, url_column = indexes
            o[file_column] = asset_map[o[url_column]] if o[url_column] else ""
        return tuple(o)

    def _load_sqllite(self) -> None:
        """Load a SQLite database from a bdbag.  THis is done by looking for all the CSV files in the bdbag directory.

        If the file is for an asset table, update the FileName column of the table to have the local file path for
        the materialized file.  Then load into the sqllite database.
        Note: none of the foreign key constraints are included in the database.
        """
        dpath = self.bag_path / "data"
        asset_map = self._localize_asset_table()  # Map of remote to local assets.

        # Find all the CSV files in the subdirectory and load each file into the database.
        for csv_file in Path(dpath).rglob("*.csv"):
            table = csv_file.stem
            schema = (
                self._domain_schema
                if table in DatasetBag._model.schemas[self._domain_schema].tables
                else self._ml_schema
            )

            with csv_file.open(newline="") as csvfile:
                csv_reader = reader(csvfile)
                column_names = next(csv_reader)

                # Determine which columns in the table has the Filename and the URL
                asset_indexes = (
                    (column_names.index("Filename"), column_names.index("URL"))
                    if self._is_asset(table)
                    else None
                )

                value_template = ",".join(
                    ["?"] * len(column_names)
                )  # SQL placeholder for row (?,?..)
                column_list = ",".join([f'"{c}"' for c in column_names])
                with DatasetBag.dbase:
                    object_table = (
                        DatasetBag._localize_asset(o, asset_indexes, asset_map)
                        for o in csv_reader
                    )
                    DatasetBag.dbase.executemany(
                        f'INSERT OR REPLACE INTO "{schema}:{table}" ({column_list}) VALUES ({value_template})',
                        object_table,
                    )

    @staticmethod
    def list_tables() -> list[str]:
        """List the names of the tables in the catalog

        Returns:
            A list of table names.  These names are all qualified with the Deriva schema name.
        """
        with DatasetBag.dbase:
            return [
                t[0]
                for t in DatasetBag.dbase.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;"
                ).fetchall()
            ]

    @staticmethod
    def find_datasets() -> list[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Returns:
             list of currently available datasets.
        """
        atable = next(
            DatasetBag._model.schemas[ML_SCHEMA]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name

        # Get a list of all the dataset_type values associated with this dataset.
        datasets = []
        print(atable)
        ds_types = list(DatasetBag.get_table_as_dict(atable))
        print(ds_types)
        for dataset in DatasetBag.get_table_as_dict("Dataset"):
            my_types = [t for t in ds_types if t["Dataset"] == dataset["RID"]]
            datasets.append(
                dataset
                | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in my_types]}
            )
        return datasets

    @validate_call
    def list_dataset_members(self, recurse: bool = False) -> defaultdict:
        """Return a list of entities associated with a specific dataset.

        Args:
           recurse:  (Default value = False)

        Returns:
            Dictionary of entities associated with a specific dataset.  Key is the table from which the elements
            were taken.
        """

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        for assoc_table in self.dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            if (
                target_table.schema.name != self._domain_schema
                and target_table != self.dataset_table
            ):
                # Look at domain tables and nested datasets.
                continue
            if target_table == self.dataset_table:
                # find_assoc gives us the keys in the wrong position, so swap.
                self_fkey, other_fkey = other_fkey, self_fkey
            sql_target = DatasetBag._normalize_table_name(target_table.name)
            sql_member = DatasetBag._normalize_table_name(member_table.name)

            # Get the names of the columns that we are going to need for linking
            member_link = tuple(
                c.name for c in next(iter(other_fkey.column_map.items()))
            )

            with DatasetBag.dbase:
                sql_cmd = (
                    f'SELECT * FROM "{sql_member}" '
                    f'JOIN "{sql_target}" ON "{sql_member}".{member_link[0]} = "{sql_target}".{member_link[1]} '
                    f'WHERE "{self.dataset_rid}" = "{sql_member}".Dataset;'
                )
                target_entities = DatasetBag.dbase.execute(sql_cmd).fetchall()
                members[target_table.name].extend(target_entities)

            target_entities = []  # path.entities().fetch()
            members[target_table.name].extend(target_entities)
            if recurse and target_table.name == self.dataset_table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds in nested_datasets:
                    for k, v in DatasetBag.list_dataset_members(
                        ds, recurse=False
                    ).items():
                        members[k].extend(v)
        return members

    @validate_call
    def list_dataset_children(self, recurse: bool = False) -> list[RID]:
        """Given a dataset RID, return a list of RIDs of any nested datasets.

        Returns:
          list of RIDs of nested datasets.

        """
        return self._list_dataset_children(self.dataset_rid, recurse)

    def _list_dataset_children(self, dataset_rid, recurse: bool) -> list[RID]:
        ds_table = DatasetBag._normalize_table_name("Dataset_Dataset")
        with DatasetBag.dbase:
            nested = [
                r[0]
                for r in DatasetBag.dbase.execute(
                    f'SELECT Nested_Dataset FROM "{ds_table}" where Dataset = "{dataset_rid}"'
                ).fetchall()
            ]
        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(self._list_dataset_children(child, recurse))
        return result

    @staticmethod
    def _normalize_table_name(table: str) -> str:
        """Attempt to insert the schema into a table name if it's not provided.

        Args:
          table: str:

        Returns:
          table name with schema included.

        """
        sname = ""
        try:
            [sname, tname] = table.split(":")
        except ValueError:
            tname = table
            for sname, s in DatasetBag._model.schemas.items():
                if table in s.tables:
                    break
        try:
            _ = DatasetBag._model.schemas[sname].tables[tname]
            return f"{sname}:{tname}"
        except KeyError:
            raise DerivaMLException(f'Table name "{table}" does not exist.')

    @staticmethod
    def get_table(table: str) -> Generator[tuple, None, None]:
        """Retrieve the contents of the specified table. If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        Args:
            table: return: A generator that yields tuples of column values.

        Returns:
          A generator that yields tuples of column values.

        """
        table_name = DatasetBag._normalize_table_name(table)
        result = DatasetBag.dbase.execute(f'SELECT * FROM "{table_name}"')
        while row := result.fetchone():
            yield row

    @staticmethod
    def get_table_as_dataframe(table: str) -> pd.DataFrame:
        """Retrieve the contents of the specified table as a dataframe.


        If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        Args:
            table: Table to retrieve data from.

        Returns:
          A dataframe containing the contents of the specified table.
        """
        table_name = DatasetBag._normalize_table_name(table)
        print(table_name)
        return pd.read_sql(f'SELECT * FROM "{table_name}"', con=DatasetBag.dbase)

    @staticmethod
    def get_table_as_dict(table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve the contents of the specified table as a dictionary.

        Args:
            table: Table to retrieve data from. f schema is not provided as part of the table name,
                the method will attempt to locate the schema for the table.

        Returns:
          A generator producing dictionaries containing the contents of the specified table as name/value pairs.
        """
        table_name = DatasetBag._normalize_table_name(table)
        with DatasetBag.dbase:
            col_names = [
                c[1]
                for c in DatasetBag.dbase.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]
            result = DatasetBag.dbase.execute(f'SELECT * FROM "{table_name}"')
            while row := result.fetchone():
                yield dict(zip(col_names, row))

    @staticmethod
    def delete_database(bag_path: Path, schema: str):
        """

        Args:
          bag_path:
          schema:

        Returns:

        """
        dbase_path = Path(bag_path) / f"{schema}.db"
        dbase_path.unlink()

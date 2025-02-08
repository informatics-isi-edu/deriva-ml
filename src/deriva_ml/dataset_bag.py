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
from pydantic import validate_call, ConfigDict

from .deriva_definitions import ML_SCHEMA, MLVocab, RID, DerivaMLException


class DatabaseModel:
    """Read in the contents of a BDBag and create a local SQLite database.

        As part of its initialization, this routine will create a sqlite database that has the contents of all the tables
    in the dataset_table.  In addition, any asset tables will the `Filename` column remapped to have the path of the local
    copy of the file. In addition, a local version of the ERMRest model that as used to generate the dataset_table is
    available.

       The sqllite database will not have any foreign key constraints applied, however, foreign-key relationships can be
    found by looking in the ERMrest model.  In addition, as sqllite doesn't support schema, Ermrest schema are added
    to the table name using the convention SchemaName:TableName.  Methods in DatasetBag that have table names as the
    argument will perform the appropriate name mappings.
    """

    _paths_loaded: dict[Path:"DatabaseModel"] = {}

    @classmethod
    @validate_call
    def create(cls, bag_path: Path, dbase_dir: Optional[Path] = None):
        o = cls._paths_loaded.get(bag_path.as_posix())
        if o:
            return o
        return cls(bag_path, dbase_dir)

    def __init__(self, bag_path: Path, dbase_dir: Optional[Path] = None) -> None:
        """

        Args:
            bag_path: path to a bag
            dbase_dir:
        """

        self.bag_path = bag_path

        DatabaseModel._paths_loaded[self.bag_path.as_posix()] = self

        dir_path = dbase_dir or Path(tempfile.mkdtemp())
        dbase_file = dir_path / "dataset_table.db"
        self.dbase = sqlite3.connect(dbase_file)

        self._model = Model.fromfile("file-system", self.bag_path / "data/schema.json")
        self.domain_schema = self._guess_domain_schema()
        self._load_model()
        self.ml_schema = ML_SCHEMA
        self._load_sqllite()

        self.dataset_table = self._model.schemas[self.ml_schema].tables["Dataset"]
        # Now go through the database and pick out all the dataset_table RIDS, along with their versions.
        sql_dataset = self.normalize_table_name("Dataset")
        with self.dbase:
            self.dataset_rids = [
                t[0]
                for t in self.dbase.execute(
                    f'SELECT "RID" FROM "{sql_dataset}"'
                ).fetchall()
            ]

    def _load_model(self) -> None:
        # Create a sqlite database schema that contains all the tables within the catalog from which the
        # BDBag was created.
        with self.dbase:
            for t in self._model.schemas[self.domain_schema].tables.values():
                self.dbase.execute(t.sqlite3_ddl())
            for t in self._model.schemas["deriva-ml"].tables.values():
                self.dbase.execute(t.sqlite3_ddl())

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
                self.domain_schema
                if table in self._model.schemas[self.domain_schema].tables
                else self.ml_schema
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
                with self.dbase:
                    object_table = (
                        self._localize_asset(o, asset_indexes, asset_map)
                        for o in csv_reader
                    )
                    self.dbase.executemany(
                        f'INSERT OR REPLACE INTO "{schema}:{table}" ({column_list}) VALUES ({value_template})',
                        object_table,
                    )

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
                    local_file = fields[2].replace("\n", "")
                    local_path = f"{self.bag_path}/{local_file}"
                    fetch_map[urlparse(fields[0]).path] = local_path
        except FileNotFoundError:
            dataset_rid = self.bag_path.name.replace("Dataset_", "")
            logging.info(f"No downloaded assets in bag {dataset_rid}")
        return fetch_map

    def _guess_domain_schema(self):
        # Guess the domain schema name by eliminating all the "builtin" schema.
        return [
            s for s in self._model.schemas if s not in ["deriva-ml", "public", "www"]
        ][0]

    def _is_asset(self, table_name: str) -> bool:
        """

        Args:
          table_name: str:

        Returns:

        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        sname = (
            self.domain_schema
            if table_name in self._model.schemas[self.domain_schema].tables
            else self.ml_schema
        )
        asset_table = self._model.schemas[sname].tables[table_name]
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

    def get_dataset(self, rid: RID) -> "DatasetBag":
        """
        Return a datasetbag object for the specified RID.
        Args:
            rid: RID of the desired dataset_table.

        Returns:
            DatasetBag object for the specified RID.

        Raises:
            DerivaException if the RID doesn't exist.
        """
        if rid in self.dataset_rids:
            return DatasetBag(rid, self)
        else:
            raise DerivaMLException(f"RID {rid} does not exist in database catalog.")

    def list_tables(self) -> list[str]:
        """List the names of the tables in the catalog

        Returns:
            A list of table names.  These names are all qualified with the Deriva schema name.
        """
        with self.dbase:
            return [
                t[0]
                for t in self.dbase.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;"
                ).fetchall()
            ]

    def find_datasets(self) -> list[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Returns:
             list of currently available datasets.
        """
        atable = next(
            self._model.schemas[ML_SCHEMA]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name

        # Get a list of all the dataset_type values associated with this dataset_table.
        datasets = []
        print(atable)
        ds_types = list(self.get_table_as_dict(atable))
        print(ds_types)
        for dataset in self.get_table_as_dict("Dataset"):
            my_types = [t for t in ds_types if t["Dataset"] == dataset["RID"]]
            datasets.append(
                dataset
                | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in my_types]}
            )
        return datasets

    def normalize_table_name(self, table: str) -> str:
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
            for sname, s in self._model.schemas.items():
                if table in s.tables:
                    break
        try:
            _ = self._model.schemas[sname].tables[tname]
            return f"{sname}:{tname}"
        except KeyError:
            raise DerivaMLException(f'Table name "{table}" does not exist.')

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        """Retrieve the contents of the specified table. If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        Args:
            table: return: A generator that yields tuples of column values.

        Returns:
          A generator that yields tuples of column values.

        """
        table_name = self.normalize_table_name(table)
        result = self.dbase.execute(f'SELECT * FROM "{table_name}"')
        while row := result.fetchone():
            yield row

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Retrieve the contents of the specified table as a dataframe.


        If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        Args:
            table: Table to retrieve data from.

        Returns:
          A dataframe containing the contents of the specified table.
        """
        table_name = self.normalize_table_name(table)
        print(table_name)
        return pd.read_sql(f'SELECT * FROM "{table_name}"', con=self.dbase)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve the contents of the specified table as a dictionary.

        Args:
            table: Table to retrieve data from. f schema is not provided as part of the table name,
                the method will attempt to locate the schema for the table.

        Returns:
          A generator producing dictionaries containing the contents of the specified table as name/value pairs.
        """
        table_name = self.normalize_table_name(table)
        with self.dbase:
            col_names = [
                c[1]
                for c in self.dbase.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]
            result = self.dbase.execute(f'SELECT * FROM "{table_name}"')
            while row := result.fetchone():
                yield dict(zip(col_names, row))

    def delete_database(self, bag_path: Path, schema: str):
        """

        Args:
          bag_path:
          schema:

        Returns:

        """
        dbase_path = Path(bag_path) / f"{schema}.db"
        dbase_path.unlink()


class DatasetBag:
    """DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized BDBag for a
    dataset_table, which is created either by DerivaML.create_execution, or directly by calling DerivaML.download_dataset.

    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, rid: RID, dbase: DatabaseModel) -> None:
        """
        Initialize a DatasetBag instance.

        Args:
            rid: RID of dataset_table
            version: Version of dataset_table
            dbase: DatabaseModel
        """

        self.database = dbase
        self.dataset_rid = rid
        self.dataset_table = self.database.dataset_table
        with dbase.dbase as db:
            dataset_table = dbase.normalize_table_name("Dataset")
            sql_cmd = (
                f'SELECT Version from "{dataset_table}" where RID="{self.dataset_rid}"'
            )
            self.version = db.execute(sql_cmd).fetchone()[0]

    def list_tables(self) -> list[str]:
        """List the names of the tables in the catalog

        Returns:
            A list of table names.  These names are all qualified with the Deriva schema name.
        """
        return self.database.list_tables()

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        return self.database.get_table(table)

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        return self.database.get_table_as_dataframe(table)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        return self.database.get_table_as_dict(table)

    @validate_call
    def list_dataset_members(self, recurse: bool = False) -> dict[str, list[tuple]]:
        """Return a list of entities associated with a specific dataset_table.

        Args:
           recurse:  (Default value = False)

        Returns:
            Dictionary of entities associated with a specific dataset_table.  Key is the table from which the elements
            were taken.
        """

        # Look at each of the element types that might be in the dataset_table and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        for assoc_table in self.dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            if (
                target_table.schema.name != self.database.domain_schema
                and target_table != self.dataset_table
            ):
                # Look at domain tables and nested datasets.
                continue
            if target_table == self.dataset_table:
                # find_assoc gives us the keys in the wrong position, so swap.
                self_fkey, other_fkey = other_fkey, self_fkey
            sql_target = self.database.normalize_table_name(target_table.name)
            sql_member = self.database.normalize_table_name(member_table.name)

            # Get the names of the columns that we are going to need for linking
            member_link = tuple(
                c.name for c in next(iter(other_fkey.column_map.items()))
            )

            with self.database.dbase as db:
                sql_cmd = (
                    f'SELECT * FROM "{sql_member}" '
                    f'JOIN "{sql_target}" ON "{sql_member}".{member_link[0]} = "{sql_target}".{member_link[1]} '
                    f'WHERE "{self.dataset_rid}" = "{sql_member}".Dataset;'
                )
                target_entities = db.execute(sql_cmd).fetchall()
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
        return dict(members)

    @validate_call
    def list_dataset_children(self, recurse: bool = False) -> list[RID]:
        """Given a dataset_table RID, return a list of RIDs of any nested datasets.

        Returns:
          list of RIDs of nested datasets.

        """
        return self._list_dataset_children(self.dataset_rid, recurse)

    def _list_dataset_children(self, dataset_rid, recurse: bool) -> list[RID]:
        ds_table = self.database.normalize_table_name("Dataset_Dataset")
        with self.database.dbase as db:
            nested = [
                r[0]
                for r in db.execute(
                    f'SELECT Nested_Dataset FROM "{ds_table}" where Dataset = "{dataset_rid}"'
                ).fetchall()
            ]
        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(self._list_dataset_children(child, recurse))
        return result

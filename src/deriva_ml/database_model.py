"""Ths module constains the definition of the DatabaseModel class.  The role of this class is to provide an nterface between the BDBag representation
of a dataset and a sqllite database in which the contents of the bag are stored.
"""
import logging
import sqlite3

from csv import reader
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from deriva.core.ermrest_model import Model

from .deriva_definitions import ML_SCHEMA, MLVocab, RID, DerivaMLException
from .dataset_aux_classes import DatasetVersion, DatasetMinid
from .deriva_model import DerivaModel
from .dataset_bag import DatasetBag


class DatabaseModelMeta(type):
    """Use metaclass to ensure that there is onl one instance per path"""

    _paths_loaded: dict[Path:"DatabaseModel"] = {}

    def __call__(cls, *args, **kwargs):
        logger = logging.getLogger("deriva_ml")
        bag_path: Path = args[1]
        if bag_path.as_posix() not in cls._paths_loaded:
            logger.info(f"Loading {bag_path}")
            cls._paths_loaded[bag_path] = super().__call__(*args, **kwargs)
        return cls._paths_loaded[bag_path]


class DatabaseModel(DerivaModel, metaclass=DatabaseModelMeta):
    """Read in the contents of a BDBag and create a local SQLite database.

        As part of its initialization, this routine will create a sqlite database that has the contents of all the tables
    in the dataset_table.  In addition, any asset tables will the `Filename` column remapped to have the path of the local
    copy of the file. In addition, a local version of the ERMRest model that as used to generate the dataset_table is
    available.

       The sqlite database will not have any foreign key constraints applied, however, foreign-key relationships can be
    found by looking in the ERMRest model.  In addition, as sqllite doesn't support schema, Ermrest schema are added
    to the table name using the convention SchemaName:TableName.  Methods in DatasetBag that have table names as the
    argument will perform the appropriate name mappings.

    Because of nested datasets, it's possible that more than one dataset rid is in a bag, or that a dataset rid might
    appear in more than one database. To help manage this, a global list of all the datasets that have been loaded
    into DatabaseModels, is kept in the class variable `_rid_map`.
    
    Because you can load diffent versions of a dataset simultaniously, the dataset RID and version number are tracked, and a new
    sqllite instance is created for every new dataset version present.

    Attributes:
        bag_path (Path): path to the local copy of the BDBag
        minid (DatasetMinid): Minid for the specified bag
        dataset_rid (RID): RID for the specified dataset
        dbase (Connection): connection to the sqlite database holding table values
        domain_schema (str): Name of the domain schema
        dataset_table  (Table): the dataset table in the ERMRest model.
    """

    # Maintain a global map of RIDS to versions and databases.
    _rid_map: dict[RID, list[tuple[DatasetVersion, "DatabaseModel"]]] = {}

    @staticmethod
    def rid_lookup(dataset_rid: RID) -> list[tuple[DatasetVersion, "DatabaseModel"]]:
        """Return a list of DatasetVersion/DatabaseModel instances corresponding to the given RID.

        Args:
            dataset_rid: Rit to be looked up.

        Returns:
            List of DatasetVersion/DatabaseModel instances corresponding to the given RID.

        Raises:
            Raise a DerivaMLException if the given RID is not found.
        """
        try:
            return DatabaseModel._rid_map[dataset_rid]
        except KeyError:
            raise DerivaMLException(f"Dataset {dataset_rid} not found")

    def __init__(self, minid: DatasetMinid, bag_path: Path):
        """Create a new DatabaseModel.

        Args:
            minid: Minid for the specified bag.
            bag_path:  Path to the local copy of the BDBag.
        """

        self.bag_path = bag_path
        self.minid = minid
        self.dataset_rid = minid.dataset_rid
        dir_path = bag_path.parent
        self.dbase_file = dir_path / f"{minid.version_rid}.db"
        self.dbase = sqlite3.connect(self.dbase_file)

        super().__init__(
            Model.fromfile("file-system", self.bag_path / "data/schema.json")
        )

        self._logger = logging.getLogger("deriva_ml")
        self.domain_schema = self._guess_domain_schema()
        self._load_model()
        self.ml_schema = ML_SCHEMA
        self._load_sqlite()
        self._logger.info(
            "Creating new database for dataset: %s in %s",
            self.dataset_rid,
            self.dbase_file,
        )
        self.dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]
        # Now go through the database and pick out all the dataset_table RIDS, along with their versions.
        sql_dataset = self.normalize_table_name("Dataset_Version")
        with self.dbase:
            dataset_versions = [
                t
                for t in self.dbase.execute(
                    f'SELECT "Dataset", "Version" FROM "{sql_dataset}"'
                ).fetchall()
            ]
        dataset_versions = [
            (v[0], DatasetVersion.parse(v[1])) for v in dataset_versions
        ]

        # Get most current version of each rid
        self.bag_rids = {}
        for rid, version in dataset_versions:
            self.bag_rids[rid] = max(
                self.bag_rids.get(rid, DatasetVersion(0, 1, 0)), version
            )

        for dataset_rid, dataset_version in self.bag_rids.items():
            version_list = DatabaseModel._rid_map.setdefault(dataset_rid, [])
            version_list.append((dataset_version, self))

    def _load_model(self) -> None:
        """Create a sqlite database schema that contains all the tables within the catalog from which the BDBag was created."""
        with self.dbase:
            for t in self.model.schemas[self.domain_schema].tables.values():
                self.dbase.execute(t.sqlite3_ddl())
            for t in self.model.schemas["deriva-ml"].tables.values():
                self.dbase.execute(t.sqlite3_ddl())

    def _load_sqlite(self) -> None:
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
                if table in self.model.schemas[self.domain_schema].tables
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
            with open(self.bag_path / "fetch.txt", newline="\n") as fetch_file:
                for row in fetch_file:
                    # Rows in fetch.text are tab seperated with URL filename.
                    fields = row.split("\t")
                    local_file = fields[2].replace("\n", "")
                    local_path = f"{self.bag_path}/{local_file}"
                    fetch_map[urlparse(fields[0]).path] = local_path
        except FileNotFoundError:
            dataset_rid = self.bag_path.name.replace("Dataset_", "")
            logging.info(f"No downloaded assets in bag {dataset_rid}")
        return fetch_map

    def _guess_domain_schema(self) -> str:
        """Guess the domain schema name by eliminating all the "builtin" schema.

        Returns:
            String for domain schema name.
        """
        return [
            s for s in self.model.schemas if s not in ["deriva-ml", "public", "www"]
        ][0]

    def _is_asset(self, table_name: str) -> bool:
        """

        Args:
          table_name: str:

        Returns:
            Boolean that is true if the table looks like an asset table.
        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        sname = (
            self.domain_schema
            if table_name in self.model.schemas[self.domain_schema].tables
            else self.ml_schema
        )
        asset_table = self.model.schemas[sname].tables[table_name]
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

    def get_dataset(self, dataset_rid: Optional[RID] = None) -> DatasetBag:
        """Get a dataset, or nested dataset from the bag database

        Args:
            dataset_rid: Optional.  If not provided, use the main RID for the bag.  If a value is given, it must
            be the RID for a nested dataset.

        Returns:
            DatasetBag object for the specified dataset.
        """
        if dataset_rid and dataset_rid not in self.bag_rids:
            DerivaMLException(f"Dataset RID {dataset_rid} is not in model.")
        return DatasetBag(self, dataset_rid or self.dataset_rid)

    def dataset_version(self, dataset_rid: Optional[RID] = None) -> DatasetVersion:
        if dataset_rid and dataset_rid not in self.bag_rids:
            DerivaMLException(f"Dataset RID {dataset_rid} is not in model.")
        return self.bag_rids[dataset_rid]

    def find_datasets(self) -> list[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Returns:
             list of currently available datasets.
        """
        atable = next(
            self.model.schemas[ML_SCHEMA]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name

        # Get a list of all the dataset_type values associated with this dataset_table.
        datasets = []
        ds_types = list(self.get_table_as_dict(atable))
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
            for sname, s in self.model.schemas.items():
                if table in s.tables:
                    break
        try:
            _ = self.model.schemas[sname].tables[tname]
            return f"{sname}:{tname}"
        except KeyError:
            raise DerivaMLException(f'Table name "{table}" does not exist.')

    def delete_database(self):
        """

        Args:

        Returns:

        """
        self.dbase_file.unlink()

"""This module contains the definition of the DatabaseModel class.  The role of this class is to provide an interface
between the BDBag representation of a dataset and a sqlite database in which the contents of the bag are stored.
"""

from __future__ import annotations

import json
import logging
from csv import reader
from pathlib import Path
from typing import Any, Generator, Optional, Type
from urllib.parse import urlparse

from dateutil import parser
from deriva.core.ermrest_model import Column as DerivaColumn
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import Table as DerivaTable
from deriva.core.ermrest_model import Type as DerivaType
from pydantic import ConfigDict, validate_call
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    create_engine,
    event,
    inspect,
    select,
)
from sqlalchemy import Column as SQLColumn
from sqlalchemy import ForeignKeyConstraint as SQLForeignKeyConstraint
from sqlalchemy import Table as SQLTable
from sqlalchemy import UniqueConstraint as SQLUniqueConstraint
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import backref, configure_mappers, foreign, relationship
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import TypeDecorator

from deriva_ml.core.definitions import ML_SCHEMA, RID, MLVocab
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetMinid, DatasetVersion
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.model.catalog import DerivaModel

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class ERMRestBoolean(TypeDecorator):
    impl = Boolean
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value in ("Y", "y", 1, True, "t", "T"):
            return True
        elif value in ("N", "n", 0, False, "f", "F"):
            return False
        elif value is None:
            return None
        raise ValueError(f"Invalid boolean value: {value!r}")


class StringToFloat(TypeDecorator):
    impl = Float
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value == "" or value is None:
            return None
        else:
            return float(value)

class StringToInteger(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value == "" or value is None:
            return None
        else:
            return int(value)


class StringToDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value == "" or value is None:
            return None
        else:
            return parser.parse(value)

class StringToDate(TypeDecorator):
    impl = Date
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value == "" or value is None:
            return None
        else:
            return parser.parse(value).date()

class DatabaseModelMeta(type):
    """Use metaclass to ensure that there is only one instance of a database model per path"""

    _paths_loaded: dict[Path, "DatabaseModel"] = {}

    def __call__(cls, *args, **kwargs):
        logger = logging.getLogger("deriva_ml")
        bag_path: Path = args[1]
        if bag_path.as_posix() not in cls._paths_loaded:
            logger.info(f"Loading {bag_path}")
            cls._paths_loaded[bag_path] = super().__call__(*args, **kwargs)
        return cls._paths_loaded[bag_path]


class DatabaseModel(DerivaModel, metaclass=DatabaseModelMeta):
    """Read in the contents of a BDBag and create a local SQLite database.

        As part of its initialization, this routine will create a sqlite database that has the contents of all the
    tables in the dataset_table.  In addition, any asset tables will the `Filename` column remapped to have the path
    of the local copy of the file. In addition, a local version of the ERMRest model that as used to generate the
    dataset_table is available.

       The sqlite database will not have any foreign key constraints applied, however, foreign-key relationships can be
    found by looking in the ERMRest model.  In addition, as sqlite doesn't support schema, Ermrest schema are added
    to the table name using the convention SchemaName:TableName.  Methods in DatasetBag that have table names as the
    argument will perform the appropriate name mappings.

    Because of nested datasets, it's possible that more than one dataset rid is in a bag, or that a dataset rid might
    appear in more than one database. To help manage this, a global list of all the datasets that have been loaded
    into DatabaseModels, is kept in the class variable `_rid_map`.

    Because you can load different versions of a dataset simultaneously, the dataset RID and version number are tracked,
    and a new sqlite instance is created for every new dataset version present.

    Attributes:
        bag_path (Path): path to the local copy of the BDBag
        minid (DatasetMinid): Minid for the specified bag
        dataset_rid (RID): RID for the specified dataset
        engine (Connection): connection to the sqlalchemy database holding table values
        domain_schema (str): Name of the domain schema
        dataset_table  (Table): the dataset table in the ERMRest model.
    """

    # Maintain a global map of RIDS to versions and databases.
    _rid_map: dict[RID, list[tuple[DatasetVersion, "DatabaseModel"]]] = {}

    def __init__(self, minid: DatasetMinid, bag_path: Path, dbase_path: Path):
        """Create a new DatabaseModel.

        Args:
            minid: Minid for the specified bag.
            bag_path:  Path to the local copy of the BDBag.
        """

        super().__init__(Model.fromfile("file-system", bag_path / "data/schema.json"))

        self.bag_path = bag_path
        self.minid = minid
        self.dataset_rid = minid.dataset_rid
        self.dbase_path = dbase_path / f"{minid.version_rid}"
        self.dbase_path.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{(self.dbase_path / 'main.db').resolve()}", future=True)
        self.metadata = MetaData()
        self.Base = automap_base(metadata=self.metadata)

        # Attach event listener for *this instance's* engine
        event.listen(self.engine, "connect", self._attach_schemas)

        schema_file = self.bag_path / "data/schema.json"
        with schema_file.open("r") as f:
            self.snaptime = json.load(f)["snaptime"]

        self._logger = logging.getLogger("deriva_ml")
        self._load_model()
        self.ml_schema = ML_SCHEMA
        self._load_database()
        self._logger.info(
            "Creating new database for dataset: %s in %s",
            self.dataset_rid,
            self.dbase_path,
        )
        self.dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]

        # Now go through the database and pick out all the dataset_table RIDS, along with their versions.
        with self.engine.connect() as conn:
            dataset_version = self.metadata.tables[f"{self.ml_schema}.Dataset_Version"]
            result = conn.execute(select(dataset_version.c.Dataset, dataset_version.c.Version))
            dataset_versions = [t for t in result]

        dataset_versions = [(v[0], DatasetVersion.parse(v[1])) for v in dataset_versions]
        # Get most current version of each rid
        self.bag_rids = {}
        for rid, version in dataset_versions:
            self.bag_rids[rid] = max(self.bag_rids.get(rid, DatasetVersion(0, 1, 0)), version)

        for dataset_rid, dataset_version in self.bag_rids.items():
            version_list = DatabaseModel._rid_map.setdefault(dataset_rid, [])
            version_list.append((dataset_version, self))

    def _attach_schemas(self, dbapi_conn, _conn_record):
        cur = dbapi_conn.cursor()
        for schema in [self.domain_schema, self.ml_schema]:
            schema_file = (self.dbase_path / f"{schema}.db").resolve()
            cur.execute(f"ATTACH DATABASE '{schema_file}' AS '{schema}'")
        cur.close()

    @staticmethod
    def _sql_type(type: DerivaType) -> TypeEngine:
        """Return the SQL type for a Deriva column."""
        return {
            "boolean": ERMRestBoolean,
            "date": StringToDate,
            "float4": StringToFloat,
            "float8": StringToFloat,
            "int2": StringToInteger,
            "int4": StringToInteger,
            "int8": StringToInteger,
            "json": JSON,
            "jsonb": JSON,
            "timestamptz": StringToDateTime,
            "timestamp": StringToDateTime,
        }.get(type.typename, String)

    def _load_model(self) -> None:
        """Create a sqlite database schema that contains all the tables within the catalog from which the BDBag
        was created."""

        def is_key(column: DerivaColumn, table: DerivaTable) -> bool:
            return column in [key.unique_columns[0] for key in table.keys] and column.name == "RID"

        def col(model, name: str):
            # try ORM attribute first
            try:
                return getattr(model, name).property.columns[0]
            except AttributeError:
                # fall back to exact DB column key on the Table
                return model.__table__.c[name]

        def guess_attr_name(col_name: str) -> str:
            return col_name[:-3] if col_name.lower().endswith("_id") else col_name

        database_tables: list[SQLTable] = []
        for schema_name in [self.domain_schema, self.ml_schema]:
            for table in self.model.schemas[schema_name].tables.values():
                database_columns: list[SQLColumn] = []
                for c in table.columns:
                    # clone column (type, nullability, PK, defaults, unique)
                    database_column = SQLColumn(
                        name=c.name,
                        type_=self._sql_type(c.type),  # SQLAlchemy type object is reusable
                        comment=c.comment,
                        default=c.default,
                        primary_key=is_key(c, table),
                        nullable=c.nullok,
                        # NOTE: server_onupdate, computed, etc. can be added if you use them
                    )
                    database_columns.append(database_column)
                database_table = SQLTable(table.name, self.metadata, *database_columns, schema=schema_name)
                for key in table.keys:
                    key_columns = [c.name for c in key.unique_columns]
                    #     if key.name[0] == "RID":
                    #        continue
                    database_table.append_constraint(
                        SQLUniqueConstraint(
                            *key_columns,
                            name=key.name[1],
                        )
                    )
                for fk in table.foreign_keys:
                    if fk.pk_table.schema.name not in [self.domain_schema, self.ml_schema]:
                        continue
                    if fk.pk_table.schema.name != schema_name:
                        continue
                    # Attach FK to the chosen column
                    database_table.append_constraint(
                        SQLForeignKeyConstraint(
                            columns=[f"{c.name}" for c in fk.foreign_key_columns],
                            refcolumns=[f"{schema_name}.{c.table.name}.{c.name}" for c in fk.referenced_columns],
                            name=fk.name[1],
                            comment=fk.comment,
                        )
                    )
                database_tables.append(database_table)
        with self.engine.begin() as conn:
            self.metadata.create_all(conn, tables=database_tables)

        def name_for_scalar_relationship(_base, local_cls, referred_cls, constraint):
            cols = list(constraint.columns) if constraint is not None else []
            if len(cols) == 1:
                name = cols[0].key
                if name in {c.key for c in local_cls.__table__.columns}:
                    name += "_rel"
                return name
            return constraint.name or referred_cls.__name__.lower()

        def name_for_collection_relationship(_base, local_cls, referred_cls, constraint):
            backref_name = constraint.name.replace("_fkey", "_collection")
            return backref_name or (referred_cls.__name__.lower() + "_collection")

        # Now build ORM mappings for the tables.
        self.Base.prepare(
            self.engine,
            name_for_scalar_relationship=name_for_scalar_relationship,
            name_for_collection_relationship=name_for_collection_relationship,
            reflect=True,
        )

        for schema in [self.domain_schema, self.ml_schema]:
            for table in self.model.schemas[schema].tables.values():
                for fk in table.foreign_keys:
                    if fk.pk_table.schema.name not in [self.domain_schema, self.ml_schema]:
                        continue
                    if fk.pk_table.schema.name == schema:
                        continue
                    table_name = f"{schema}.{table.name}"
                    table_class = self.get_orm_class_by_name(table_name)
                    foreign_key_column_name = fk.foreign_key_columns[0].name
                    foreign_key_column = col(table_class, foreign_key_column_name)

                    referenced_table_name = f"{fk.pk_table.schema.name}.{fk.pk_table.name}"
                    referenced_class = self.get_orm_class_by_name(referenced_table_name)
                    referenced_column = col(referenced_class, fk.referenced_columns[0].name)

                    relationship_attr = guess_attr_name(foreign_key_column_name)
                    backref_attr = fk.name[1].replace("_fkey", "_collection")
                    setattr(
                        table_class,
                        relationship_attr,
                        relationship(
                            referenced_class,
                            foreign_keys=[foreign_key_column],
                            primaryjoin=foreign(foreign_key_column) == referenced_column,
                            backref=backref(backref_attr, viewonly=True),
                            viewonly=True,  # set False for write behavior, but best with proper FKs
                        ),
                    )

        # Reflect won't pick up the second FK in the dataset_dataset table.  We need to do it manually
        # dataset_dataset_class = self.get_orm_class_by_name("deriva-ml.Dataset_Dataset")
        # dataset_class = self.get_orm_class_by_name("deriva-ml.Dataset")
        #     dataset_dataset_class.Nested_Dataset = relationship(
        #         dataset_class,
        #         primaryjoin=foreign(dataset_dataset_class.__table__.c["Nested_Dataset"])
        #         == dataset_class.__table__.c["RID"],
        #         foreign_keys=[dataset_dataset_class.__table__.c["Nested_Dataset"]],
        #         backref=backref("nested_dataset_collection", viewonly=True),  # pick a distinct name
        #         viewonly=True,  # keep it read-only unless you truly want writes
        #         overlaps="Dataset,dataset_dataset_collection",  # optional: silence overlap warnings
        #     )
        configure_mappers()

    def _load_database(self) -> None:
        """Load a SQLite database from a bdbag.  THis is done by looking for all the CSV files in the bdbag directory.

        If the file is for an asset table, update the FileName column of the table to have the local file path for
        the materialized file.  Then load into the sqlite database.
        Note: none of the foreign key constraints are included in the database.
        """
        dpath = self.bag_path / "data"
        asset_map = self._localize_asset_table()  # Map of remote to local assets.

        # Find all the CSV files in the subdirectory and load each file into the database.
        for csv_file in Path(dpath).rglob("*.csv"):
            table = csv_file.stem
            schema = self.domain_schema if table in self.model.schemas[self.domain_schema].tables else self.ml_schema
            sql_table = self.metadata.tables[f"{schema}.{table}"]

            with csv_file.open(newline="") as csvfile:
                csv_reader = reader(csvfile)
                column_names = next(csv_reader)

                # Determine which columns in the table has the Filename and the URL
                asset_indexes = (
                    (column_names.index("Filename"), column_names.index("URL")) if self._is_asset(table) else None
                )

                with self.engine.begin() as conn:
                    object_table = [
                        self._localize_asset(o, asset_indexes, asset_map, table == "Dataset")
                        for o in csv_reader
                    ]
                    conn.execute(
                        sqlite_insert(sql_table).on_conflict_do_nothing(),
                        [dict(zip(column_names, row)) for row in object_table],
                    )

    def _localize_asset_table(self) -> dict[str, str]:
        """Use the fetch.txt file in a bdbag to create a map from a URL to a local file path.

        Returns:
            Dictionary that maps a URL to a local file path.

        """
        fetch_map = {}
        try:
            with Path.open(self.bag_path / "fetch.txt", newline="\n") as fetch_file:
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

    def _is_asset(self, table_name: str) -> bool:
        """

        Args:
          table_name: str:

        Returns:
            Boolean that is true if the table looks like an asset table.
        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        sname = self.domain_schema if table_name in self.model.schemas[self.domain_schema].tables else self.ml_schema
        asset_table = self.model.schemas[sname].tables[table_name]
        return asset_columns.issubset({c.name for c in asset_table.columns})

    @staticmethod
    def _localize_asset(o: list, indexes: tuple[int, int], asset_map: dict[str, str], debug: bool = False) -> tuple:
        """Given a list of column values for a table, replace the FileName column with the local file name based on
        the URL value.

        Args:
          o: List of values for each column in a table row.
          indexes: A tuple whose first element is the column index of the file name and whose second element
        is the index of the URL in an asset table.  Tuple is None if table is not an asset table.
          o: list:
          indexes: Optional[tuple[int, int]]:

        Returns:
          Tuple of updated column values.

        """
        if indexes:
            file_column, url_column = indexes
            o[file_column] = asset_map[o[url_column]] if o[url_column] else ""
        return tuple(o)

    def find_table(self, table_name: str) -> SQLTable:
        """Find a table in the catalog."""
        return [t for t in self.metadata.tables if t == table_name or t.split(".")[1] == table_name][0]

    def list_tables(self) -> list[str]:
        """List the names of the tables in the catalog

        Returns:
            A list of table names.  These names are all qualified with the Deriva schema name.
        """
        tables = list(self.metadata.tables.keys())
        tables.sort()
        return tables

    def get_dataset(self, dataset_rid: Optional[RID] = None) -> DatasetBag:
        """Get a dataset, or nested dataset from the bag database

        Args:
            dataset_rid: Optional.  If not provided, use the main RID for the bag.  If a value is given, it must
            be the RID for a nested dataset.

        Returns:
            DatasetBag object for the specified dataset.
        """
        if dataset_rid and dataset_rid not in self.bag_rids:
            raise DerivaMLException(f"Dataset RID {dataset_rid} is not in model.")
        return DatasetBag(self, dataset_rid or self.dataset_rid)

    def dataset_version(self, dataset_rid: Optional[RID] = None) -> DatasetVersion:
        """Return the version of the specified dataset."""
        if dataset_rid and dataset_rid not in self.bag_rids:
            DerivaMLException(f"Dataset RID {dataset_rid} is not in model.")
        return self.bag_rids[dataset_rid]

    def find_datasets(self) -> list[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Returns:
             list of currently available datasets.
        """
        atable = next(self.model.schemas[ML_SCHEMA].tables[MLVocab.dataset_type].find_associations()).name

        # Get a list of all the dataset_type values associated with this dataset_table.
        datasets = []
        ds_types = list(self._get_table(atable))
        for dataset in self._get_table("Dataset"):
            my_types = [t for t in ds_types if t["Dataset"] == dataset["RID"]]
            datasets.append(dataset | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in my_types]})
        return datasets

    def list_dataset_members(self, dataset_rid: RID) -> dict[str, Any]:
        """Returns a list of all the dataset_table entries associated with a dataset."""
        return self.get_dataset(dataset_rid).list_dataset_members()

    def _get_table(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve the contents of the specified table as a dictionary.

        Args:
            table: Table to retrieve data from. f schema is not provided as part of the table name,
                the method will attempt to locate the schema for the table.

        Returns:
          A generator producing dictionaries containing the contents of the specified table as name/value pairs.
        """

        with self.engine.connect() as conn:
            result = conn.execute(select(self.metadata.tables[table]))
            for row in result.mappings():
                yield dict(row)

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

    def get_orm_class_by_name(self, table: str) -> Any | None:
        sql_table = self.metadata.tables.get(self.find_table(table))
        if sql_table is None:
            raise DerivaMLException(f"Table {table} not found")
        return self.get_orm_class_for_table(sql_table)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_orm_class_for_table(self, table: SQLTable | DerivaTable | str) -> Any | None:
        if isinstance(table, DerivaTable):
            table = self.metadata.tables[f"{table.schema.name}.{table.name}"]
            if isinstance(table, str):
                table = self.metadata.tables.get(self.find_table(table))
        for mapper in self.Base.registry.mappers:
            if mapper.persist_selectable is table or table in mapper.tables:
                return mapper.class_
        return None

    @staticmethod
    def _is_association(
        table_class, min_arity=2, max_arity=2, unqualified=True, pure=True, no_overlap=True, return_fkeys=False
    ):
        """Return (truthy) integer arity if self is a matching association, else False.

        min_arity: minimum number of associated fkeys (default 2)
        max_arity: maximum number of associated fkeys (default 2) or None
        unqualified: reject qualified associations when True (default True)
        pure: reject impure associations when True (default True)
        no_overlap: reject overlapping associations when True (default True)
        return_fkeys: return the set of N associated ForeignKeys if True

        The default behavior with no arguments is to test for pure,
        unqualified, non-overlapping, binary associations.

        An association is comprised of several foreign keys which are
        covered by a non-nullable composite row key. This allows
        specific combinations of foreign keys to appear at most once.

        The arity of an association is the number of foreign keys
        being associated. A typical binary association has arity=2.

        An unqualified association contains *only* the foreign key
        material in its row key. Conversely, a qualified association
        mixes in other material which means that a specific
        combination of foreign keys may repeat with different
        qualifiers.

        A pure association contains *only* row key
        material. Conversely, an impure association includes
        additional metadata columns not covered by the row key. Unlike
        qualifiers, impure metadata merely decorates an association
        without augmenting its identifying characteristics.

        A non-overlapping association does not share any columns
        between multiple foreign keys. This means that all
        combinations of foreign keys are possible. Conversely, an
        overlapping association shares some columns between multiple
        foreign keys, potentially limiting the combinations which can
        be represented in an association row.

        These tests ignore the five ERMrest system columns and any
        corresponding constraints.

        """
        if min_arity < 2:
            raise ValueError("An assocation cannot have arity < 2")
        if max_arity is not None and max_arity < min_arity:
            raise ValueError("max_arity cannot be less than min_arity")

        mapper = inspect(table_class).mapper

        # TODO: revisit whether there are any other cases we might
        # care about where system columns are involved?
        non_sys_cols = {col.name for col in mapper.columns if col.name not in {"RID", "RCT", "RMT", "RCB", "RMB"}}
        unique_columns = [
            {c.name for c in constraint.columns}
            for constraint in inspect(table_class).local_table.constraints
            if isinstance(constraint, SQLUniqueConstraint)
        ]

        non_sys_key_colsets = {
            frozenset(unique_column_set)
            for unique_column_set in unique_columns
            if unique_column_set.issubset(non_sys_cols) and len(unique_column_set) > 1
        }

        if not non_sys_key_colsets:
            # reject: not association
            return False

        # choose longest compound key (arbitrary choice with ties!)
        row_key = sorted(non_sys_key_colsets, key=lambda s: len(s), reverse=True)[0]
        foreign_keys = [constraint for constraint in inspect(table_class).relationships.values()]

        covered_fkeys = {fkey for fkey in foreign_keys if {c.name for c in fkey.local_columns}.issubset(row_key)}
        covered_fkey_cols = set()

        if len(covered_fkeys) < min_arity:
            # reject: not enough fkeys in association
            return False
        elif max_arity is not None and len(covered_fkeys) > max_arity:
            # reject: too many fkeys in association
            return False

        for fkey in covered_fkeys:
            fkcols = {c.name for c in fkey.local_columns}
            if no_overlap and fkcols.intersection(covered_fkey_cols):
                # reject: overlapping fkeys in association
                return False
            covered_fkey_cols.update(fkcols)

        if unqualified and row_key.difference(covered_fkey_cols):
            # reject: qualified association
            return False

        if pure and non_sys_cols.difference(row_key):
            # reject: impure association
            return False

        # return (truthy) arity or fkeys
        if return_fkeys:
            return covered_fkeys
        else:
            return len(covered_fkeys)

    def get_orm_association_class(
        self,
        left_cls: Type[Any],
        right_cls: Type[Any],
        min_arity=2,
        max_arity=2,
        unqualified=True,
        pure=True,
        no_overlap=True,
    ):
        """
        Find an association class C by: (1) walking rels on left_cls to a mid class C,
        (2) verifying C also relates to right_cls. Returns (C, C->left, C->right) or None.

        """
        for _, left_rel in inspect(left_cls).relationships.items():
            mid_cls = left_rel.mapper.class_
            is_assoc = self._is_association(mid_cls, return_fkeys=True)
            if not is_assoc:
                continue
            assoc_local_columns_left = list(is_assoc)[0].local_columns
            assoc_local_columns_right = list(is_assoc)[1].local_columns

            found_left = found_right = False
            for r in inspect(left_cls).relationships.values():
                remote_side = list(r.remote_side)[0]
                if remote_side in assoc_local_columns_left:
                    found_left = r
                if remote_side in assoc_local_columns_right:
                    found_left = r
                    # We have left and right backwards from the assocation, so swap them.
                    assoc_local_columns_left, assoc_local_columns_right = (
                        assoc_local_columns_right,
                        assoc_local_columns_left,
                    )
            for r in inspect(right_cls).relationships.values():
                remote_side = list(r.remote_side)[0]
                if remote_side in assoc_local_columns_right:
                    found_right = r
            if found_left != False and found_right != False:
                return mid_cls, found_left.class_attribute, found_right.class_attribute
        return None

    def delete_database(self):
        """

        Args:

        Returns:

        """
        self.dbase_file.unlink()

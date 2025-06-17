"""
The module implements the sqllite interface to a set of directories representing a dataset bag.
"""

from __future__ import annotations

import sqlite3

# Standard library imports
from collections import defaultdict
from copy import copy
from typing import TYPE_CHECKING, Any, Generator, Iterable, cast

import deriva.core.datapath as datapath

# Third-party imports
import pandas as pd

# Deriva imports
from deriva.core.ermrest_model import Column, Table
from pydantic import ConfigDict, validate_call

# Local imports
from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import Feature

if TYPE_CHECKING:
    from deriva_ml.model.database import DatabaseModel

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class DatasetBag:
    """DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized BDBag for a
    dataset_table, which is created either by DerivaML.create_execution, or directly by calling DerivaML.download_dataset.

    A general a bag may contain multiple datasets, if the dataset is nested. The DatasetBag is used to represent only
    one of the datasets in the bag.

    All the metadata associated with the dataset is stored in a SQLLite database that can be queried using SQL.

    Attributes:
        dataset_rid (RID): RID for the specified dataset
        version: The version of the dataset
        model (DatabaseModel): The Database model that has all the catalog metadata associated with this dataset.
            database:
        dbase (sqlite3.Connection): connection to the sqlite database holding table values
        domain_schema (str): Name of the domain schema
    """

    def __init__(self, database_model: DatabaseModel, dataset_rid: RID | None = None) -> None:
        """
        Initialize a DatasetBag instance.

        Args:
            database_model: Database version of the bag.
            dataset_rid: Optional RID for the dataset.
        """
        self.model = database_model
        self.database = cast(sqlite3.Connection, self.model.dbase)

        self.dataset_rid = dataset_rid or self.model.dataset_rid
        if not self.dataset_rid:
            raise DerivaMLException("No dataset RID provided")

        self.model.rid_lookup(self.dataset_rid)  # Check to make sure that this dataset is in the bag.

        self.version = self.model.dataset_version(self.dataset_rid)
        self._dataset_table = self.model.dataset_table

    def __repr__(self) -> str:
        return f"<deriva_ml.DatasetBag object {self.dataset_rid} at {hex(id(self))}>"

    def list_tables(self) -> list[str]:
        """List the names of the tables in the catalog

        Returns:
            A list of table names.  These names are all qualified with the Deriva schema name.
        """
        return self.model.list_tables()

    def _dataset_table_view(self, table: str) -> str:
        table_name = self.model.normalize_table_name(table)
        with self.database as dbase:
            select_args = ",".join(
                [f'"{table_name}"."{c[1]}"' for c in dbase.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
            )
        datasets = ",".join(
            [f'"{self.dataset_rid}"'] + [f'"{ds.dataset_rid}"' for ds in self.list_dataset_children(recurse=True)]
        )
        paths = [
            (
                [f'"{self.model.normalize_table_name(t.name)}"' for t in p],
                [self.model._table_relationship(t1, t2) for t1, t2 in zip(p, p[1:])],
            )
            for p in self.model._schema_to_paths()
            if p[-1].name == table
        ]

        sql = []
        dataset_table_name = f'"{self.model.normalize_table_name(self._dataset_table.name)}"'

        def column_name(col: Column) -> str:
            return f'"{self.model.normalize_table_name(col.table.name)}"."{col.name}"'

        for ts, on in paths:
            tables = " JOIN ".join(ts)
            on_expression = " and ".join([f"{column_name(left)}={column_name(right)}" for left, right in on])
            sql.append(
                f"SELECT {select_args} FROM {tables} ON {on_expression} WHERE {dataset_table_name}.RID IN ({datasets})"
            )
        sql = " UNION ".join(sql) if len(sql) > 1 else sql[0]
        return sql

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        """Retrieve the contents of the specified table. If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        Args:
            table: return: A generator that yields tuples of column values.

        Returns:
          A generator that yields tuples of column values.

        """
        result = self.database.execute(self._dataset_table_view(table))
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
        return pd.read_sql(self._dataset_table_view(table), self.database)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve the contents of the specified table as a dictionary.

        Args:
            table: Table to retrieve data from. f schema is not provided as part of the table name,
                the method will attempt to locate the schema for the table.

        Returns:
          A generator producing dictionaries containing the contents of the specified table as name/value pairs.
        """
        table_name = self.model.normalize_table_name(table)
        with self.database as dbase:
            col_names = [c[1] for c in dbase.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
            result = self.database.execute(self._dataset_table_view(table))
            while row := result.fetchone():
                yield dict(zip(col_names, row))

    @validate_call
    def list_dataset_members(self, recurse: bool = False) -> dict[str, dict[str, list[Any]]]:
        """Return a list of entities associated with a specific dataset.

        Args:
           recurse: Whether to include nested datasets.

        Returns:
            Dictionary of entities associated with the dataset.
        """

        # Look at each of the element types that might be in the _dataset_table and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        for assoc_table in self._dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            if target_table.schema.name != self.model.domain_schema and not (
                target_table == self._dataset_table or target_table.name == "File"
            ):
                # Look at domain tables and nested datasets.
                continue
            if target_table == self._dataset_table:
                # find_assoc gives us the keys in the wrong position, so swap.
                self_fkey, other_fkey = other_fkey, self_fkey
            sql_target = self.model.normalize_table_name(target_table.name)
            sql_member = self.model.normalize_table_name(member_table.name)

            # Get the names of the columns that we are going to need for linking
            member_link = tuple(c.name for c in next(iter(other_fkey.column_map.items())))

            with self.database as db:
                col_names = [c[1] for c in db.execute(f'PRAGMA table_info("{sql_target}")').fetchall()]
                select_cols = ",".join([f'"{sql_target}".{c}' for c in col_names])
                sql_cmd = (
                    f'SELECT {select_cols} FROM "{sql_member}" '
                    f'JOIN "{sql_target}" ON "{sql_member}".{member_link[0]} = "{sql_target}".{member_link[1]} '
                    f'WHERE "{self.dataset_rid}" = "{sql_member}".Dataset;'
                )

                target_entities = [dict(zip(col_names, e)) for e in db.execute(sql_cmd).fetchall()]
                members[target_table.name].extend(target_entities)

            target_entities = []  # path.entities().fetch()
            members[target_table.name].extend(target_entities)
            if recurse and target_table.name == self._dataset_table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds in nested_datasets:
                    for k, v in DatasetBag.list_dataset_members(ds, recurse=False).items():
                        members[k].extend(v)
        return dict(members)

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features for a table.

        Args:
            table: The table to find features for.

        Returns:
            An iterable of Feature instances.
        """
        return self.model.find_features(table)

    def list_feature_values(self, table: Table | str, feature_name: str) -> datapath._ResultSet:
        """Return feature values for a table.

        Args:
            table: The table to get feature values for.
            feature_name: Name of the feature.

        Returns:
            Feature values.
        """
        feature = self.model.lookup_feature(table, feature_name)
        feature_table = self.model.normalize_table_name(feature.feature_table.name)
        with self.database as db:
            sql_cmd = f'SELECT * FROM "{feature_table}"'
            return cast(datapath._ResultSet, db.execute(sql_cmd).fetchall())

    def list_dataset_children(self, recurse: bool = False) -> list[DatasetBag]:
        """Get nested datasets.

        Args:
            recurse: Whether to include children of children.

        Returns:
            List of child dataset bags.
        """
        ds_table = self.model.normalize_table_name("Dataset")
        nds_table = self.model.normalize_table_name("Dataset_Dataset")
        dv_table = self.model.normalize_table_name("Dataset_Version")
        with self.database as db:
            sql_cmd = (
                f'SELECT  "{nds_table}".Nested_Dataset, "{dv_table}".Version '
                f'FROM "{nds_table}" JOIN "{dv_table}" JOIN "{ds_table}" on '
                f'"{ds_table}".Version == "{dv_table}".RID AND '
                f'"{nds_table}".Nested_Dataset == "{ds_table}".RID '
                f'where "{nds_table}".Dataset == "{self.dataset_rid}"'
            )
            nested = [DatasetBag(self.model, r[0]) for r in db.execute(sql_cmd).fetchall()]

        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(child.list_dataset_children(recurse))
        return result


# Add annotations after definition to deal with forward reference issues in pydantic

DatasetBag.list_dataset_children = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)(DatasetBag.list_dataset_children)

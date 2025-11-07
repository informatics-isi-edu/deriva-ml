"""
The module implements the sqllite interface to a set of directories representing a dataset bag.
"""

from __future__ import annotations

# Standard library imports
from collections import defaultdict
from copy import copy
from typing import TYPE_CHECKING, Any, Generator, Iterable, cast

import deriva.core.datapath as datapath

# Third-party imports
import pandas as pd

# Local imports
from deriva.core.ermrest_model import Column, Table

# Deriva imports
from pydantic import ConfigDict, validate_call
from sqlalchemy import Engine, inspect, select
from sqlalchemy.orm import Session

from deriva_ml.core.definitions import RID, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm
from deriva_ml.feature import Feature
from deriva_ml.model.sql_mapper import SQLMapper

if TYPE_CHECKING:
    from deriva_ml.model.database import DatabaseModel

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class DatasetBag:
    """
    DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized
    BDBag for a dataset_table, which is created either by DerivaML.create_execution, or directly by
    calling DerivaML.download_dataset.

    A general a bag may contain multiple datasets, if the dataset is nested. The DatasetBag is used to
    represent only one of the datasets in the bag.

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
        self.engine = cast(Engine, self.model.engine)
        self.metadata = self.model.metadata

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
        """Return a SQL command that will return all of the elements in the specified table that are associated with
        dataset_rid"""

        def find_relationship_to(source_cls, target_cls):
            """ "
            Find the relationship between two classes."""
            mapper = inspect(source_cls)
            for name, rel in mapper.relationships.items():
                if rel.mapper.class_ is target_cls:
                    return rel.class_attribute
            return None

        table_class = self.model.get_orm_class_by_name(table)
        dataset_table_class = self.model.get_orm_class_by_name(self._dataset_table.name)
        dataset_rids = [c.dataset_rid for c in self.list_dataset_children(recurse=True)]
        paths = [[t.name for t in p] for p in self.model._schema_to_paths() if p[-1].name == table]
        sql_cmd = None
        for path in paths:
            path_sql = select(table_class)
            last_class = self.model.get_orm_class_by_name(path[0])
            for t in path[1:]:
                t_class = self.model.get_orm_class_by_name(t)
                path_sql = path_sql.join(find_relationship_to(last_class, t_class))
                last_class = t_class
            path_sql = path_sql.where(dataset_table_class.RID.in_(dataset_rids))
            if sql_cmd is not None:
                sql_cmd = sql_cmd.union(path_sql)
            else:
                sql_cmd = path_sql
        return sql_cmd

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        """Retrieve the contents of the specified table. If schema is not provided as part of the table name,
        the method will attempt to locate the schema for the table.

        Args:
            table: return: A generator that yields tuples of column values.

        Returns:
          A generator that yields tuples of column values.

        """
        with Session(self.engine) as session:
            result = session.execute(self._dataset_table_view(table))
            for row in result:
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
        return pd.read_sql(self._dataset_table_view(table), self.engine)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve the contents of the specified table as a dictionary.

        Args:
            table: Table to retrieve data from. f schema is not provided as part of the table name,
                the method will attempt to locate the schema for the table.

        Returns:
          A generator producing dictionaries containing the contents of the specified table as name/value pairs.
        """

        with Session(self.engine) as session:
            result = session.execute(self._dataset_table_view(table))
            for row in result.mappings():
                yield row

    @validate_call
    def list_dataset_members(self, recurse: bool = False) -> dict[str, list[dict[str, Any]]]:
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
            member_fkey = assoc_table.other_fkeys.pop()
            if member_fkey.pk_table.name == "Dataset" and member_fkey.foreign_key_columns[0].name != "Nested_Dataset":
                # Sometimes find_assoc gets confused on Dataset_Dataset.
                member_fkey = assoc_table.self_fkey

            target_table = member_fkey.pk_table
            member_table = assoc_table.table

            if target_table.schema.name != self.model.domain_schema and not (
                target_table == self._dataset_table or target_table.name == "File"
            ):
                # Look at domain tables and nested datasets.
                continue
            sql_target = self.model.normalize_table_name(target_table.name)
            sql_member = self.model.normalize_table_name(member_table.name)

            # Get the names of the columns that we are going to need for linking
            member_link = tuple(c.name for c in next(iter(member_fkey.column_map.items())))
            with self.database as db:
                col_names = [c[1] for c in db.execute(f'PRAGMA table_info("{sql_target}")').fetchall()]
                select_cols = ",".join([f'"{sql_target}".{c}' for c in col_names])
                sql_cmd = (
                    f'SELECT {select_cols} FROM "{sql_member}" '
                    f'JOIN "{sql_target}" ON "{sql_member}".{member_link[0]} = "{sql_target}".{member_link[1]} '
                    f'WHERE "{self.dataset_rid}" = "{sql_member}".Dataset;'
                )
                mapper = SQLMapper(self.model, sql_target)
                target_entities = [mapper.transform_tuple(e) for e in db.execute(sql_cmd).fetchall()]
            members[target_table.name].extend(target_entities)
            if recurse and (target_table.name == self._dataset_table.name):
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds in nested_datasets:
                    nested_dataset = self.model.get_dataset(ds)
                    for k, v in nested_dataset.list_dataset_members(recurse=recurse).items():
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
            col_names = [c[1] for c in db.execute(f'PRAGMA table_info("{feature_table}")').fetchall()]
            sql_cmd = f'SELECT * FROM "{feature_table}"'
            return cast(datapath._ResultSet, [dict(zip(col_names, r)) for r in db.execute(sql_cmd).fetchall()])

    def list_dataset_element_types(self) -> list[Table]:
        """
        Lists the data types of elements contained within a dataset.

        This method analyzes the dataset and identifies the data types for all
        elements within it. It is useful for understanding the structure and
        content of the dataset and allows for better manipulation and usage of its
        data.

        Returns:
            list[str]: A list of strings where each string represents a data type
            of an element found in the dataset.

        """
        return self.model.list_dataset_element_types()

    def list_dataset_children(self, recurse: bool = False) -> list[DatasetBag]:
        """Get nested datasets.

        Args:
            recurse: Whether to include children of children.

        Returns:
            List of child dataset bags.
        """
        ds_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset")
        nds_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset_Dataset")
        dv_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset_Version")

        with Session(self.engine) as session:
            sql_cmd = (
                select(nds_table.Nested_Dataset, dv_table.Version)
                .join_from(ds_table, nds_table, onclause=ds_table.RID == nds_table.Nested_Dataset)
                .join_from(ds_table, dv_table, onclause=ds_table.Version == dv_table.RID)
                .where(nds_table.Dataset == self.dataset_rid)
            )
            nested = [DatasetBag(self.model, r[0]) for r in session.execute(sql_cmd).all()]

        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(child.list_dataset_children(recurse))
        return result

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """Finds a term in a vocabulary table.

        Searches for a term in the specified vocabulary table, matching either the primary name
        or any of its synonyms.

        Args:
            table: Vocabulary table to search in (name or Table object).
            term_name: Name or synonym of the term to find.

        Returns:
            VocabularyTerm: The matching vocabulary term.

        Raises:
            DerivaMLVocabularyException: If the table is not a vocabulary table, or term is not found.

        Examples:
            Look up by primary name:
                >>> term = ml.lookup_term("tissue_types", "epithelial")
                >>> print(term.description)

            Look up by synonym:
                >>> term = ml.lookup_term("tissue_types", "epithelium")
        """
        # Get and validate vocabulary table reference
        vocab_table = self.model.normalize_table_name(table)
        if not self.model.is_vocabulary(table):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Search for term by name or synonym
        for term in self.get_table_as_dict(vocab_table):
            if term_name == term["Name"] or (term["Synonyms"] and term_name in term["Synonyms"]):
                term["Synonyms"] = list(term["Synonyms"])
                return VocabularyTerm.model_validate(term)

        # Term not found
        raise DerivaMLInvalidTerm(vocab_table, term_name)

    def _denormalize(self, include_tables: list[str] | None) -> str:
        """
        Generates an SQL statement for denormalizing the dataset based on the tables to include. Processes cycles in
        graph relationships, ensures proper join order, and generates selected columns for denormalization.

        Args:
            include_tables (list[str] | None): List of table names to include in the denormalized dataset. If None,
                all tables from the dataset will be included.

        Returns:
            str: SQL query string that represents the process of denormalization.
        """

        def column_name(col: Column) -> str:
            return f'"{self.model.normalize_table_name(col.table.name)}"."{col.name}"'

        # Skip over tables that we don't want to include in the denormalized dataset.
        # Also, strip off the Dataset/Dataset_X part of the path so we don't include dataset columns in the denormalized
        # table.

        join_tables, tables, denormalized_columns, dataset_rids, dataset_element_tables = (
            self.model._prepare_wide_table(self, self.dataset_rid, include_tables)
        )

        select_args = [
            # SQLlite will strip out the table name from the column in the select statement, so we need to add
            # an explicit alias to the column name.
            f'"{self.model.normalize_table_name(table_name)}"."{column_name}" AS "{table_name}.{column_name}"'
            for table_name, column_name in denormalized_columns
        ]

        # First table in the table list is the table specified in the method call.
        normalized_join_tables = [self.model.normalize_table_name(t) for t in join_tables]
        sql_statement = f'SELECT {",".join(select_args)} FROM "{normalized_join_tables[0]}"'
        for t in normalized_join_tables[1:]:
            on = tables[t]
            sql_statement += f' LEFT JOIN "{t}" ON '
            sql_statement += "OR ".join([f"{column_name(o[0])} = {column_name(o[1])}" for o in on])

        # Select only rows from the datasets you wish to include.
        dataset_rid_list = ",".join([f'"{self.dataset_rid}"'] + [f'"{b.dataset_rid}"' for b in dataset_rids])
        sql_statement += f'WHERE  "{self.model.normalize_table_name("Dataset")}"."RID" IN ({dataset_rid_list})'

        # Only include rows that have actual values in them.
        real_row = [f'"{self.model.normalize_table_name(t)}".RID IS NOT NULL ' for t in dataset_element_tables]
        sql_statement += f" AND ({' OR '.join(real_row)})"
        return sql_statement

    def denormalize_as_dataframe(self, include_tables: list[str] | None = None) -> pd.DataFrame:
        """
        Denormalize the dataset and return the result as a dataframe.

        This routine will examine the domain schema for the dataset, determine which tables to include and denormalize
        the dataset values into a single wide table.  The result is returned as a dataframe.

        The optional argument include_tables can be used to specify a subset of tables to include in the denormalized
        view.  The tables in this argument can appear anywhere in the dataset schema.  The method will determine which
        additional tables are required to complete the denormalization process.  If include_tables is not specified,
        all of the tables in the schema will be included.

        The resulting wide table will include a column for every table needed to complete the denormalization process.

        Args:
            include_tables: List of table names to include in the denormalized dataset. If None, than the entire schema
            is used.

        Returns:
            Dataframe containing the denormalized dataset.
        """
        return pd.read_sql(self._denormalize(include_tables=include_tables), self.database)

    def denormalize_as_dict(self, include_tables: list[str] | None = None) -> Generator[dict[str, Any], None, None]:
        """
        Denormalize the dataset and return the result as a set of dictionarys.

        This routine will examine the domain schema for the dataset, determine which tables to include and denormalize
        the dataset values into a single wide table.  The result is returned as a generateor that returns a dictionary
        for each row in the denormlized wide table.

        The optional argument include_tables can be used to specify a subset of tables to include in the denormalized
        view.  The tables in this argument can appear anywhere in the dataset schema.  The method will determine which
        additional tables are required to complete the denormalization process.  If include_tables is not specified,
        all of the tables in the schema will be included.

        The resulting wide table will include a column for every table needed to complete the denormalization process.

        Args:
            include_tables: List of table names to include in the denormalized dataset. If None, than the entire schema
            is used.

        Returns:
            A generator that returns a dictionary representation of each row in the denormalized dataset.
        """
        with self.database as dbase:
            cursor = dbase.execute(self._denormalize(include_tables=include_tables))
            columns = [desc[0] for desc in cursor.description]
            for row in cursor:
                yield dict(zip(columns, row))


# Add annotations after definition to deal with forward reference issues in pydantic

DatasetBag.list_dataset_children = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)(DatasetBag.list_dataset_children)

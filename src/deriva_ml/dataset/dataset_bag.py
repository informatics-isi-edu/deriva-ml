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
from deriva.core.ermrest_model import Table

# Deriva imports
from pydantic import ConfigDict, validate_call
from sqlalchemy import CompoundSelect, Engine, RowMapping, Select, and_, inspect, or_, select, union_all
from sqlalchemy.orm import RelationshipProperty, Session
from sqlalchemy.orm.util import AliasedClass

from deriva_ml.core.definitions import RID, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm
from deriva_ml.feature import Feature

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

    @staticmethod
    def _find_relationship_attr(source, target):
        """
        Return the relationship attribute (InstrumentedAttribute) on `source`
        that points to `target`. Works with classes or AliasedClass.
        Raises LookupError if not found.
        """
        src_mapper = inspect(source).mapper
        tgt_mapper = inspect(target).mapper

        # collect relationships on the *class* mapper (not on alias)
        candidates: list[RelationshipProperty] = [rel for rel in src_mapper.relationships if rel.mapper is tgt_mapper]

        if not candidates:
            raise LookupError(f"No relationship from {src_mapper.class_.__name__} → {tgt_mapper.class_.__name__}")

        # Prefer MANYTOONE when multiple paths exist (often best for joins)
        candidates.sort(key=lambda r: r.direction.name != "MANYTOONE")
        rel = candidates[0]

        # Bind to the actual source (alias or class)
        return getattr(source, rel.key) if isinstance(source, AliasedClass) else rel.class_attribute

    def _dataset_table_view(self, table: str) -> CompoundSelect[Any]:
        """Return a SQL command that will return all of the elements in the specified table that are associated with
        dataset_rid"""
        table_class = self.model.get_orm_class_by_name(table)
        dataset_table_class = self.model.get_orm_class_by_name(self._dataset_table.name)
        dataset_rids = [self.dataset_rid] + [c.dataset_rid for c in self.list_dataset_children(recurse=True)]

        paths = [[t.name for t in p] for p in self.model._schema_to_paths() if p[-1].name == table]
        sql_cmds = []
        for path in paths:
            path_sql = select(table_class)
            last_class = self.model.get_orm_class_by_name(path[0])
            for t in path[1:]:
                t_class = self.model.get_orm_class_by_name(t)
                path_sql = path_sql.join(self._find_relationship_attr(last_class, t_class))
                last_class = t_class
            path_sql = path_sql.where(dataset_table_class.RID.in_(dataset_rids))
            sql_cmds.append(path_sql)
        return union_all(*sql_cmds)

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

    # @validate_call
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

        dataset_class = self.model.get_orm_class_for_table(self._dataset_table)
        for element_table in self.model.list_dataset_element_types():
            element_class = self.model.get_orm_class_for_table(element_table)

            assoc_class, dataset_rel, element_rel = self.model.get_orm_association_class(dataset_class, element_class)

            element_table = inspect(element_class).mapped_table
            if element_table.schema != self.model.domain_schema and element_table.name not in ["Dataset", "File"]:
                # Look at domain tables and nested datasets.
                continue
            # Get the names of the columns that we are going to need for linking
            with Session(self.engine) as session:
                sql_cmd = (
                    select(element_class)
                    .join(element_rel)
                    .where(self.dataset_rid == assoc_class.__table__.c["Dataset"])
                )
                # Get back the list of ORM entities and convert them to dictionaries.
                element_entities = session.scalars(sql_cmd).all()
                element_rows = [{c.key: getattr(obj, c.key) for c in obj.__table__.columns} for obj in element_entities]
            members[element_table.name].extend(element_rows)
            if recurse and (element_table.name == self._dataset_table.name):
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in element_rows]
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
        feature_class = self.model.get_orm_class_for_table(feature.feature_table)
        with Session(self.engine) as session:
            sql_cmd = select(feature_class)
            return cast(datapath._ResultSet, [row for row in session.execute(sql_cmd).mappings()])

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
        if not self.model.is_vocabulary(table):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Search for term by name or synonym
        for term in self.get_table_as_dict(table):
            if term_name == term["Name"] or (term["Synonyms"] and term_name in term["Synonyms"]):
                term["Synonyms"] = list(term["Synonyms"])
                return VocabularyTerm.model_validate(term)

        # Term not found
        raise DerivaMLInvalidTerm(table, term_name)

    def _denormalize(self, include_tables: list[str] | None, allow_duplicates: bool = False) -> Select:
        """
        Generates an SQL statement for denormalizing the dataset based on the tables to include. Processes cycles in
        graph relationships, ensures proper join order, and generates selected columns for denormalization.

        Args:
            include_tables (list[str] | None): List of table names to include in the denormalized dataset. If None,
                all tables from the dataset will be included.
            allow_duplicates (bool): Whether to allow duplicate rows in the denormalized dataset. Default is False.

        Returns:
            str: SQL query string that represents the process of denormalization.
        """
        # Skip over tables that we don't want to include in the denormalized dataset.
        # Also, strip off the Dataset/Dataset_X part of the path so we don't include dataset columns in the denormalized
        # table.

        def find_relationship(table, join_condition):
            side1 = (join_condition[0].table.name, join_condition[0].name)
            side2 = (join_condition[1].table.name, join_condition[1].name)

            for relationship in inspect(table).relationships:
                local_columns = list(relationship.local_columns)[0].table.name, list(relationship.local_columns)[0].name
                remote_side = list(relationship.remote_side)[0].table.name, list(relationship.remote_side)[0].name
                if local_columns == side1 and remote_side == side2 or local_columns == side2 and remote_side == side1:
                    return relationship
            return None

        join_tables, join_conditions, denormalized_columns, dataset_rids, dataset_element_tables = (
            self.model._prepare_wide_table(self, self.dataset_rid, include_tables)
        )
        denormalized_columns = [
            self.model.get_orm_class_by_name(table_name)
            .__table__.columns[column_name]
            .label(f"{table_name}.{column_name}")
            for table_name, column_name in denormalized_columns
        ]

        sql_statement = select(*denormalized_columns).select_from(
            self.model.get_orm_class_for_table(self._dataset_table)
        )

        for table_name in join_tables[1:]:  # Skip over dataset table
            table_class = self.model.get_orm_class_by_name(table_name)
            on_clause = [
                getattr(table_class, r.key)
                for on_condition in join_conditions[table_name]
                if (r := find_relationship(table_class, on_condition))
            ]
            sql_statement = sql_statement.outerjoin(table_class, onclause=or_(*on_clause))
        dataset_rid_list = [self.dataset_rid] + [b.dataset_rid for b in dataset_rids]
        dataset_class = self.model.get_orm_class_by_name(self._dataset_table.name)

        # Only include rows that have actual values in them.
        real_row = or_(*[self.model.get_orm_class_by_name(t).RID.isnot(None) for t in dataset_element_tables])
        sql_statement = sql_statement.where(and_(dataset_class.RID.in_(dataset_rid_list)), real_row)
        if not allow_duplicates:
            sql_statement = sql_statement.distinct()
        return sql_statement

    def denormalize_as_dataframe(
        self, include_tables: list[str] | None = None, allow_duplicates: bool = False
    ) -> pd.DataFrame:
        """
        Denormalize the dataset and return the result as a dataframe.

         This routine will examine the domain schema for the dataset, determine which tables to include and denormalize
        the dataset values into a single wide table.  The result is returned as a generator that returns a dictionary
        for each row in the denormalized wide table.

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
        return pd.read_sql(
            self._denormalize(include_tables=include_tables, allow_duplicates=allow_duplicates), self.engine
        )

    def denormalize_as_dict(
        self, include_tables: list[str] | None = None, allow_duplicates: bool = False
    ) -> Generator[RowMapping, None, None]:
        """
        Denormalize the dataset and return the result as a set of dictionary's.

        This routine will examine the domain schema for the dataset, determine which tables to include and denormalize
        the dataset values into a single wide table.  The result is returned as a generator that returns a dictionary
        for each row in the denormalized wide table.

        The optional argument include_tables can be used to specify a subset of tables to include in the denormalized
        view.  The tables in this argument can appear anywhere in the dataset schema.  The method will determine which
        additional tables are required to complete the denormalization process.  If include_tables is not specified,
        all of the tables in the schema will be included.

        The resulting wide table will include a only those column for the tables listed in include_columns.

        Args:
            include_tables: List of table names to include in the denormalized dataset. If None, than the entire schema
            is used.
            allow_duplicates: Whether to allow duplicate rows in the denormalized dataset. Default is False.

        Returns:
            A generator that returns a dictionary representation of each row in the denormalized dataset.
        """
        with Session(self.engine) as session:
            cursor = session.execute(
                self._denormalize(include_tables=include_tables, allow_duplicates=allow_duplicates)
            )
            yield from cursor.mappings()
            for row in cursor.mappings():
                yield row


# Add annotations after definition to deal with forward reference issues in pydantic

DatasetBag.list_dataset_children = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)(DatasetBag.list_dataset_children)

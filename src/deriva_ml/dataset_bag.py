from collections import defaultdict
from copy import copy
from typing import Any, Generator, Optional

import pandas as pd
from pydantic import validate_call, ConfigDict

from .deriva_definitions import RID, DerivaMLException
from .dataset_aux_classes import DatasetVersion, DatasetMinid
from .database_model import DatabaseModel


class DatasetBag:
    """DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized BDBag for a
    dataset_table, which is created either by DerivaML.create_execution, or directly by calling DerivaML.download_dataset.

    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self, dataset: DatasetMinid | RID, version: Optional[DatasetVersion] = None
    ) -> None:
        """
        Initialize a DatasetBag instance.

        Args:
            dataset: Version of dataset_table
            version: Version of dataset_table
        """
        if isinstance(dataset, DatasetMinid):
            self.dataset_rid = dataset.dataset_rid
            self.dataset_version = dataset.dataset_version
        else:
            if not version:
                raise DerivaMLException(f"Must provide version if using dataset_rid")
            self.dataset_rid = dataset
            self.dataset_version = version
        self.model = DatabaseModel.rid_lookup(self.dataset_rid, self.dataset_version)
        self.database = self.model.dbase

        self.dataset_table = self.model.dataset_table

    def list_tables(self) -> list[str]:
        """List the names of the tables in the catalog

        Returns:
            A list of table names.  These names are all qualified with the Deriva schema name.
        """
        return self.model.list_tables()

    def get_table(self, table: str) -> Generator[tuple, None, None]:
        return self.model.get_table(table)

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        return self.model.get_table_as_dataframe(table)

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        return self.model.get_table_as_dict(table)

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
            sql_target = self.model.normalize_table_name(target_table.name)
            sql_member = self.model.normalize_table_name(member_table.name)

            # Get the names of the columns that we are going to need for linking
            member_link = tuple(
                c.name for c in next(iter(other_fkey.column_map.items()))
            )

            with self.database as db:
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
        ds_table = self.model.normalize_table_name("Dataset")
        nds_table = self.model.normalize_table_name("Dataset_Dataset")
        dv_table = self.model.normalize_table_name("Dataset_Version")
        with self.database as db:
            sql_cmd = (
                f'SELECT  "{nds_table}".Nested_Dataset, "{dv_table}".Version '
                f'FROM "{nds_table}" JOIN "{dv_table}" JOIN "{ds_table}" on '
                f'"{ds_table}".Version == "{dv_table}".RID AND '
                f'"{nds_table}".Nested_Dataset == "{ds_table}".RID '
                f'where "{nds_table}".Dataset == "{dataset_rid}"'
            )
            nested = [
                    DatasetBag(r[0], DatasetVersion.parse(r[1]))
                    for r in db.execute(sql_cmd).fetchall()
                ]

        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(self._list_dataset_children(child, recurse))
        return result

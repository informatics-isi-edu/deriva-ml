from typing import Any, Type
from deriva_ml import RID
from sqlalchemy import UniqueConstraint, inspect
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter

def _prepare_wide_table(self, dataset, dataset_rid: RID, include_tables: list[str]) -> tuple:
    """
    Generates details of a wide table from the model

    Args:
        include_tables (list[str] | None): List of table names to include in the denormalized dataset. If None,
            all tables from the dataset will be included.

    Returns:
        str: SQL query string that represents the process of denormalization.
    """

    # Skip over tables that we don't want to include in the denormalized dataset.
    # Also, strip off the Dataset/Dataset_X part of the path so we don't include dataset columns in the denormalized
    # table.
    include_tables = set(include_tables)
    for t in include_tables:
        # Check to make sure the table is in the catalog.
        _ = self.name_to_table(t)

    table_paths = [
        path
        for path in self._schema_to_paths()
        if path[-1].name in include_tables and include_tables.intersection({p.name for p in path})
    ]
    paths_by_element = defaultdict(list)
    for p in table_paths:
        paths_by_element[p[2].name].append(p)

    # Get the names of all of the tables that can be dataset elements.
    dataset_element_tables = {e.name for e in self.list_dataset_element_types() if e.schema.name == self.domain_schema}

    skip_columns = {"RCT", "RMT", "RCB", "RMB"}
    join_conditions = {}
    join_tables = {}
    for element_table, paths in paths_by_element.items():
        graph = {}
        for path in paths:
            for left, right in zip(path[0:], path[1:]):
                graph.setdefault(left.name, set()).add(right.name)

        # New lets remove any cycles that we may have in the graph.
        # We will use a topological sort to find the order in which we need to join the tables.
        # If we find a cycle, we will remove the table from the graph and splice in an additional ON clause.
        # We will then repeat the process until there are no cycles.
        graph_has_cycles = True
        element_join_tables = []
        element_join_conditions = {}
        while graph_has_cycles:
            try:
                ts = TopologicalSorter(graph)
                element_join_tables = list(reversed(list(ts.static_order())))
                graph_has_cycles = False
            except CycleError as e:
                cycle_nodes = e.args[1]
                if len(cycle_nodes) > 3:
                    raise DerivaMLException(f"Unexpected cycle found when normalizing dataset {cycle_nodes}")
                # Remove cycle from graph and splice in additional ON constraint.
                graph[cycle_nodes[1]].remove(cycle_nodes[0])

    # The Dataset_Version table is a special case as it points to dataset and dataset to version.
        if "Dataset_Version" in join_tables:
            element_join_tables.remove("Dataset_Version")

        for path in paths:
            for left, right in zip(path[0:], path[1:]):
                if right.name == "Dataset_Version":
                    # The Dataset_Version table is a special case as it points to dataset and dataset to version.
                    continue
                if element_join_tables.index(right.name) < element_join_tables.index(left.name):
                    continue
                table_relationship = self._table_relationship(left, right)
                element_join_conditions.setdefault(right.name, set()).add((table_relationship[0], table_relationship[1]))
        join_tables[element_table] = element_join_tables
        join_conditions[element_table] = element_join_conditions
    # Get the list of columns that will appear in the final denormalized dataset.
    denormalized_columns = [
        (table_name, c.name)
        for table_name in join_tables
        if not self.is_association(table_name)  # Don't include association columns in the denormalized view.'
        for c in self.name_to_table(table_name).columns
        if (not include_tables or table_name in include_tables) and (c.name not in skip_columns)
    ]

    # List of dataset ids to include in the denormalized view.
    dataset_rids = [dataset_rid] + dataset.list_dataset_children(recurse=True)
    return join_tables, join_conditions, denormalized_columns, dataset_rids, dataset_element_tables


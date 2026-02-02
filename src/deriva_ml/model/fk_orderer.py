"""Foreign key dependency ordering for safe data insertion.

This module provides the ForeignKeyOrderer class which computes a
topologically sorted insertion order for tables based on their
foreign key dependencies.

When loading data into a database, tables must be populated in an order
that satisfies foreign key constraints - referenced tables must be
populated before the tables that reference them.

Example:
    orderer = ForeignKeyOrderer(model, schemas=['domain', 'deriva-ml'])

    # Get safe insertion order for a set of tables
    tables = ['Image', 'Subject', 'Diagnosis']
    ordered = orderer.get_insertion_order(tables)
    # Returns: ['Subject', 'Image', 'Diagnosis']
    # (Subject first because Image references it)

    # Get deletion order (reverse of insertion)
    delete_order = orderer.get_deletion_order(tables)
"""

from __future__ import annotations

import logging
from graphlib import TopologicalSorter, CycleError
from typing import Any

from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import Table as DerivaTable


logger = logging.getLogger(__name__)


class ForeignKeyOrderer:
    """Computes insertion order for tables based on FK dependencies.

    Uses topological sort to ensure referenced tables are populated
    before tables that reference them. Handles cycles by either
    raising an error or breaking them.

    Example:
        orderer = ForeignKeyOrderer(model, schemas=['domain', 'deriva-ml'])

        # Get insertion order
        tables_to_fill = ['Image', 'Subject', 'Diagnosis']
        ordered = orderer.get_insertion_order(tables_to_fill)
        # Returns: ['Subject', 'Image', 'Diagnosis']

        # Get all tables in safe order
        all_ordered = orderer.get_insertion_order()

        # Get FK dependencies for a table
        deps = orderer.get_dependencies('Image')
        # Returns: {'Subject', 'Dataset', ...}
    """

    def __init__(
        self,
        model: Model,
        schemas: list[str],
    ):
        """Initialize the orderer.

        Args:
            model: ERMrest Model object.
            schemas: Schemas to consider for FK relationships.
        """
        self.model = model
        self.schemas = set(schemas)
        self._table_cache: dict[str, DerivaTable] = {}
        self._build_table_cache()

    def _build_table_cache(self) -> None:
        """Build cache mapping table names to Table objects."""
        for schema_name in self.schemas:
            if schema_name not in self.model.schemas:
                continue
            schema = self.model.schemas[schema_name]
            for table_name, table in schema.tables.items():
                # Store both qualified and unqualified names
                self._table_cache[f"{schema_name}.{table_name}"] = table
                # Only store unqualified if not already present (avoids conflicts)
                if table_name not in self._table_cache:
                    self._table_cache[table_name] = table

    def _to_table(self, t: str | DerivaTable) -> DerivaTable:
        """Convert table name to Table object.

        Args:
            t: Table name or Table object.

        Returns:
            DerivaTable object.

        Raises:
            ValueError: If table not found.
        """
        if isinstance(t, DerivaTable):
            return t

        if t in self._table_cache:
            return self._table_cache[t]

        raise ValueError(f"Table {t} not found in schemas {self.schemas}")

    def _table_key(self, t: DerivaTable) -> str:
        """Get unique key for a table."""
        return f"{t.schema.name}.{t.name}"

    def get_dependencies(self, table: str | DerivaTable) -> set[DerivaTable]:
        """Get tables that this table depends on (FK targets).

        Args:
            table: Table name or object.

        Returns:
            Set of tables that must be populated before this table.
        """
        t = self._to_table(table)
        dependencies = set()

        for fk in t.foreign_keys:
            pk_table = fk.pk_table
            # Only include dependencies within our schemas
            if pk_table.schema.name in self.schemas:
                # Don't include self-references as dependencies
                if self._table_key(pk_table) != self._table_key(t):
                    dependencies.add(pk_table)

        return dependencies

    def get_dependents(self, table: str | DerivaTable) -> set[DerivaTable]:
        """Get tables that depend on this table (FK sources).

        Args:
            table: Table name or object.

        Returns:
            Set of tables that reference this table.
        """
        t = self._to_table(table)
        dependents = set()

        for schema_name in self.schemas:
            if schema_name not in self.model.schemas:
                continue

            for other_table in self.model.schemas[schema_name].tables.values():
                if self._table_key(other_table) == self._table_key(t):
                    continue

                for fk in other_table.foreign_keys:
                    if self._table_key(fk.pk_table) == self._table_key(t):
                        dependents.add(other_table)
                        break

        return dependents

    def _build_dependency_graph(
        self,
        tables: list[str | DerivaTable] | None = None,
    ) -> dict[str, set[str]]:
        """Build FK dependency graph.

        Args:
            tables: Tables to include. If None, includes all tables.

        Returns:
            Dict mapping table key -> set of table keys it depends on.
        """
        if tables is None:
            # Include all tables in schemas
            table_objs = []
            for schema_name in self.schemas:
                if schema_name in self.model.schemas:
                    table_objs.extend(self.model.schemas[schema_name].tables.values())
        else:
            table_objs = [self._to_table(t) for t in tables]

        table_keys = {self._table_key(t) for t in table_objs}
        graph: dict[str, set[str]] = {}

        for t in table_objs:
            key = self._table_key(t)
            deps = set()

            for fk in t.foreign_keys:
                pk_key = self._table_key(fk.pk_table)
                # Only include deps within our table set
                if pk_key in table_keys and pk_key != key:
                    deps.add(pk_key)

            graph[key] = deps

        return graph

    def get_insertion_order(
        self,
        tables: list[str | DerivaTable] | None = None,
        handle_cycles: bool = True,
    ) -> list[DerivaTable]:
        """Compute FK-safe insertion order for the given tables.

        Returns tables ordered so that all FK dependencies are satisfied
        when inserting in order.

        Args:
            tables: Tables to order. If None, orders all tables in schemas.
            handle_cycles: If True, break cycles by removing edges.
                If False, raise CycleError on cycles.

        Returns:
            Ordered list of Table objects (insert from first to last).

        Raises:
            CycleError: If handle_cycles=False and cycles exist.
        """
        graph = self._build_dependency_graph(tables)

        try:
            ts = TopologicalSorter(graph)
            ordered_keys = list(ts.static_order())
        except CycleError as e:
            if handle_cycles:
                ordered_keys = self._break_cycles_and_sort(graph, e)
            else:
                raise

        # Convert keys back to Table objects
        return [self._table_cache[key] for key in ordered_keys]

    def get_deletion_order(
        self,
        tables: list[str | DerivaTable] | None = None,
        handle_cycles: bool = True,
    ) -> list[DerivaTable]:
        """Compute FK-safe deletion order for the given tables.

        Returns tables in reverse dependency order - tables that are
        referenced should be deleted last.

        Args:
            tables: Tables to order. If None, orders all tables in schemas.
            handle_cycles: If True, break cycles. If False, raise on cycles.

        Returns:
            Ordered list of Table objects (delete from first to last).
        """
        insertion_order = self.get_insertion_order(tables, handle_cycles)
        return list(reversed(insertion_order))

    def _break_cycles_and_sort(
        self,
        graph: dict[str, set[str]],
        error: CycleError,
    ) -> list[str]:
        """Handle cycles by breaking them and re-sorting.

        Uses a simple strategy of removing edges from cycle members
        until no cycles remain.

        Args:
            graph: Dependency graph.
            error: CycleError with cycle info.

        Returns:
            Ordered list of table keys.
        """
        # Get cycle from error message
        cycle = list(error.args[1]) if len(error.args) > 1 else []

        if cycle:
            logger.warning(f"Breaking cycle in FK dependencies: {' -> '.join(cycle)}")

            # Remove the last edge in the cycle
            if len(cycle) >= 2:
                from_node = cycle[-1]
                to_node = cycle[0]
                if from_node in graph and to_node in graph[from_node]:
                    graph[from_node].remove(to_node)
                    logger.debug(f"Removed edge {from_node} -> {to_node}")

        # Try again
        try:
            ts = TopologicalSorter(graph)
            return list(ts.static_order())
        except CycleError as e:
            # Recursively break more cycles
            return self._break_cycles_and_sort(graph, e)

    def validate_insertion_order(
        self,
        tables: list[str | DerivaTable],
    ) -> list[tuple[str, str, str]]:
        """Validate that a list of tables can be inserted in order.

        Checks each table to ensure all its FK dependencies are
        satisfied by tables earlier in the list.

        Args:
            tables: Ordered list of tables to validate.

        Returns:
            List of (table, missing_dependency, fk_name) tuples for
            any unsatisfied dependencies. Empty list if valid.
        """
        table_objs = [self._to_table(t) for t in tables]
        seen_keys = set()
        violations = []

        for t in table_objs:
            key = self._table_key(t)

            for fk in t.foreign_keys:
                pk_key = self._table_key(fk.pk_table)
                # Skip self-references and tables not in our set
                if pk_key == key:
                    continue
                if pk_key not in {self._table_key(x) for x in table_objs}:
                    continue

                if pk_key not in seen_keys:
                    violations.append((key, pk_key, fk.name[1]))

            seen_keys.add(key)

        return violations

    def get_all_tables(self) -> list[DerivaTable]:
        """Get all tables in configured schemas.

        Returns:
            List of all Table objects.
        """
        tables = []
        for schema_name in self.schemas:
            if schema_name in self.model.schemas:
                tables.extend(self.model.schemas[schema_name].tables.values())
        return tables

    def find_cycles(self) -> list[list[str]]:
        """Find all FK dependency cycles in the schema.

        Returns:
            List of cycles, each cycle is a list of table keys.
        """
        graph = self._build_dependency_graph()
        cycles = []

        # Use DFS to find cycles
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    idx = path.index(neighbor)
                    cycle = path[idx:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

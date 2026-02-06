"""Tests for ForeignKeyOrderer cycle-breaking logic.

These tests use mock Model/Table/ForeignKey objects to test the topological
sort and cycle-breaking behavior without requiring a live Deriva catalog.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from graphlib import CycleError

from deriva_ml.model.fk_orderer import ForeignKeyOrderer


def _make_mock_model(table_specs: dict[str, list[str]], schema_name: str = "test") -> MagicMock:
    """Create a mock Model with tables and FK relationships.

    Args:
        table_specs: Dict mapping table name -> list of FK target table names.
            E.g., {"Image": ["Subject"], "Subject": []} means Image has an FK to Subject.
        schema_name: Schema name for all tables.

    Returns:
        Mock Model object compatible with ForeignKeyOrderer.
    """
    model = MagicMock()

    # Create mock tables
    tables = {}
    for table_name in table_specs:
        table = MagicMock()
        table.name = table_name
        table.schema.name = schema_name
        tables[table_name] = table

    # Set up FK relationships
    for table_name, fk_targets in table_specs.items():
        fks = []
        for target_name in fk_targets:
            fk = MagicMock()
            fk.pk_table = tables[target_name]
            fks.append(fk)
        tables[table_name].foreign_keys = fks

    # Set up model schema
    schema = MagicMock()
    schema.tables = tables
    model.schemas = {schema_name: schema}

    return model


class TestForeignKeyOrderer:
    """Tests for ForeignKeyOrderer."""

    def test_simple_linear_order(self):
        """Test linear FK chain: C -> B -> A."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["B"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order()
        names = [t.name for t in result]

        # A must come before B, B must come before C
        assert names.index("A") < names.index("B")
        assert names.index("B") < names.index("C")

    def test_no_dependencies(self):
        """Test tables with no FK relationships."""
        model = _make_mock_model({
            "A": [],
            "B": [],
            "C": [],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order()
        assert len(result) == 3

    def test_simple_two_node_cycle(self):
        """Test breaking a simple A <-> B cycle."""
        model = _make_mock_model({
            "A": ["B"],
            "B": ["A"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order(handle_cycles=True)
        names = [t.name for t in result]

        assert set(names) == {"A", "B"}
        assert len(names) == 2

    def test_three_node_cycle(self):
        """Test breaking a 3-node cycle: A -> B -> C -> A."""
        model = _make_mock_model({
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order(handle_cycles=True)
        names = [t.name for t in result]

        assert set(names) == {"A", "B", "C"}
        assert len(names) == 3

    def test_cycle_with_dependent_table(self):
        """Test cycle with an additional dependent table.

        This mirrors the real-world Dataset/Dataset_Version cycle:
        Dataset <-> Dataset_Version, with Other_Table depending on Dataset.
        """
        model = _make_mock_model({
            "Dataset": ["Dataset_Version"],
            "Dataset_Version": ["Dataset"],
            "Other_Table": ["Dataset"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order(handle_cycles=True)
        names = [t.name for t in result]

        assert set(names) == {"Dataset", "Dataset_Version", "Other_Table"}
        assert len(names) == 3
        # Other_Table must come after Dataset (its FK target)
        assert names.index("Dataset") < names.index("Other_Table")

    def test_two_separate_cycles(self):
        """Test breaking two independent cycles in the same graph."""
        model = _make_mock_model({
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
            "D": ["E"],
            "E": ["D"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order(handle_cycles=True)
        names = [t.name for t in result]

        assert set(names) == {"A", "B", "C", "D", "E"}
        assert len(names) == 5

    def test_cycle_raises_when_handle_cycles_false(self):
        """Test that CycleError is raised when handle_cycles=False."""
        model = _make_mock_model({
            "A": ["B"],
            "B": ["A"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])

        with pytest.raises(CycleError):
            orderer.get_insertion_order(handle_cycles=False)

    def test_self_reference_ignored(self):
        """Test that self-referencing FKs are not included as dependencies."""
        model = _make_mock_model({
            "Tree": ["Tree"],  # Self-referencing FK (parent_id)
            "Leaf": ["Tree"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        result = orderer.get_insertion_order()
        names = [t.name for t in result]

        assert names.index("Tree") < names.index("Leaf")

    def test_deletion_order_is_reversed(self):
        """Test that deletion order is reverse of insertion order."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["B"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        insertion = orderer.get_insertion_order()
        deletion = orderer.get_deletion_order()

        assert [t.name for t in deletion] == list(reversed([t.name for t in insertion]))

    def test_get_dependencies(self):
        """Test getting dependencies for a single table."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["A", "B"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        deps = orderer.get_dependencies("C")
        dep_names = {t.name for t in deps}

        assert dep_names == {"A", "B"}

    def test_get_dependents(self):
        """Test getting tables that depend on a given table."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["A"],
            "D": ["B"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        dependents = orderer.get_dependents("A")
        dep_names = {t.name for t in dependents}

        assert dep_names == {"B", "C"}

    def test_validate_insertion_order_valid(self):
        """Test validation passes for correct insertion order."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["B"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        violations = orderer.validate_insertion_order(["A", "B", "C"])

        assert violations == []

    def test_validate_insertion_order_invalid(self):
        """Test validation catches wrong insertion order."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["B"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        violations = orderer.validate_insertion_order(["C", "B", "A"])

        # C before B (C depends on B) and B before A (B depends on A)
        assert len(violations) >= 1

    def test_find_cycles(self):
        """Test cycle detection."""
        model = _make_mock_model({
            "A": ["B"],
            "B": ["A"],
            "C": [],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        cycles = orderer.find_cycles()

        assert len(cycles) >= 1
        # The cycle should contain both A and B
        cycle_members = set()
        for cycle in cycles:
            cycle_members.update(cycle)
        assert "test.A" in cycle_members
        assert "test.B" in cycle_members

    def test_subset_ordering(self):
        """Test ordering a subset of tables."""
        model = _make_mock_model({
            "A": [],
            "B": ["A"],
            "C": ["B"],
            "D": ["A"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        # Only order A and C (B is between them but not included)
        result = orderer.get_insertion_order(["A", "C"])
        names = [t.name for t in result]

        # C doesn't directly reference A, and B isn't in the set,
        # so no ordering constraint between them
        assert set(names) == {"A", "C"}


class TestBreakCyclesRegression:
    """Regression tests for the _break_cycles_and_sort infinite recursion bug.

    The original bug: CycleError.args[1] returns ['A', 'B', 'A'] where
    first == last. The old code used cycle[-1] and cycle[0] as from/to nodes,
    which were the same node, so no edge was ever removed, causing infinite
    recursion.
    """

    def test_no_infinite_recursion_on_simple_cycle(self):
        """Regression: simple cycle must not cause infinite recursion."""
        model = _make_mock_model({
            "Dataset": ["Dataset_Version"],
            "Dataset_Version": ["Dataset"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        # This used to cause RecursionError
        result = orderer.get_insertion_order(handle_cycles=True)
        assert len(result) == 2

    def test_no_infinite_recursion_on_complex_graph(self):
        """Regression: complex graph with cycles must not cause infinite recursion."""
        model = _make_mock_model({
            "Dataset": ["Dataset_Version", "Dataset_Type"],
            "Dataset_Version": ["Dataset"],
            "Dataset_Type": [],
            "Dataset_Member": ["Dataset", "Dataset_Version"],
            "Image": [],
            "Subject": [],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        # This used to cause RecursionError
        result = orderer.get_insertion_order(handle_cycles=True)
        names = [t.name for t in result]

        assert set(names) == {"Dataset", "Dataset_Version", "Dataset_Type",
                              "Dataset_Member", "Image", "Subject"}
        # Dataset_Type must come before Dataset
        assert names.index("Dataset_Type") < names.index("Dataset")

    def test_max_depth_safety(self):
        """Test that the max depth safety guard works."""
        model = _make_mock_model({
            "A": ["B"],
            "B": ["A"],
        })
        orderer = ForeignKeyOrderer(model, schemas=["test"])
        graph = orderer._build_dependency_graph()

        # Simulate a broken CycleError that can't be resolved
        # by passing an empty cycle
        from graphlib import CycleError
        error = CycleError("nodes are in a cycle", [])
        # Should not infinite-recurse due to max depth guard
        result = orderer._break_cycles_and_sort(graph, error, _depth=999)
        assert isinstance(result, list)

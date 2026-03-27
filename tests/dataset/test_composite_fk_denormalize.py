"""
Tests for denormalization across composite (multi-column) foreign keys.

Composite FKs like file(biosample, dataset) → biosample(RID, dataset) are common
in FaceBase and other real-world catalogs. The denormalization pipeline must handle
these correctly — both in bag export (FK path traversal to populate tables) and in
local denormalization (SQL JOIN generation).

Bug reference: The file→biosample FK in FaceBase catalog 19 is composite, causing
bag.denormalize_as_dataframe(["file", "biosample"]) to return 0 rows because:
1. _table_relationship() uses [0] indexing, dropping the second FK column
2. Bag export follows incomplete join → biosample table exports with 0 rows
3. Denormalization can't join through an empty table
"""

import pytest
from deriva.core.typed import BuiltinType, ColumnDef, ForeignKeyDef, KeyDef, TableDef

from deriva_ml import DerivaML
from deriva_ml.execution.execution import ExecutionConfiguration
from deriva_ml.core.definitions import MLVocab


class TestCompositeFKDenormalize:
    """Test denormalization across composite foreign keys."""

    def _create_composite_fk_schema(self, ml: DerivaML) -> None:
        """Add tables with a composite FK to the test catalog.

        Creates:
        - Parent table with a natural compound key (RID + group_id)
        - Child table with a composite FK referencing both columns
        - Grandchild vocab table linked to Parent via single FK

        Schema:
            Group(RID, Name)
            Parent(RID, Name, Group [FK→Group.RID])
              - compound unique key on (RID, Group)
            Child(RID, Label, Parent_RID, Parent_Group)
              - composite FK: (Parent_RID, Parent_Group) → Parent(RID, Group)
        """
        schema_name = ml.default_schema
        model = ml.catalog.getCatalogModel()
        domain = model.schemas[schema_name]

        # Skip if tables already exist (session-scoped catalog reuse)
        if "Group" in domain.tables:
            return

        # Create Group table (grandchild in the join chain)
        domain.create_table(
            TableDef(
                name="Group",
                columns=[ColumnDef("Name", BuiltinType.text, nullok=False)],
            )
        )

        # Create Parent table with FK to Group and compound unique key
        domain.create_table(
            TableDef(
                name="Parent",
                columns=[
                    ColumnDef("Name", BuiltinType.text, nullok=False),
                    ColumnDef("Group", BuiltinType.text, nullok=False),
                ],
                foreign_keys=[
                    ForeignKeyDef(
                        columns=["Group"],
                        referenced_schema=schema_name,
                        referenced_table="Group",
                        referenced_columns=["RID"],
                    ),
                ],
            )
        )

        # Add compound unique key on (RID, Group) to Parent
        # This is needed so the composite FK from Child can reference it
        parent_table = model.schemas[schema_name].tables["Parent"]
        parent_table.create_key(KeyDef(columns=["RID", "Group"]))

        # Create Child table with composite FK to Parent
        domain.create_table(
            TableDef(
                name="Child",
                columns=[
                    ColumnDef("Label", BuiltinType.text, nullok=False),
                    ColumnDef("Parent_RID", BuiltinType.text, nullok=False),
                    ColumnDef("Parent_Group", BuiltinType.text, nullok=False),
                ],
                foreign_keys=[
                    ForeignKeyDef(
                        columns=["Parent_RID", "Parent_Group"],
                        referenced_schema=schema_name,
                        referenced_table="Parent",
                        referenced_columns=["RID", "Group"],
                    ),
                ],
            )
        )

    def _populate_composite_fk_data(self, ml: DerivaML) -> dict:
        """Insert test data into composite FK tables.

        Returns dict with RIDs for verification.
        """
        pb = ml.pathBuilder()
        domain = pb.schemas[ml.default_schema]

        # Insert Group
        groups = list(
            domain.tables["Group"].insert(
                [{"Name": "Alpha"}, {"Name": "Beta"}]
            )
        )
        group_a_rid = groups[0]["RID"]
        group_b_rid = groups[1]["RID"]

        # Insert Parents
        parents = list(
            domain.tables["Parent"].insert(
                [
                    {"Name": "Parent1", "Group": group_a_rid},
                    {"Name": "Parent2", "Group": group_b_rid},
                ]
            )
        )
        parent1_rid = parents[0]["RID"]
        parent2_rid = parents[1]["RID"]

        # Insert Children with composite FK
        children = list(
            domain.tables["Child"].insert(
                [
                    {"Label": "Child1A", "Parent_RID": parent1_rid, "Parent_Group": group_a_rid},
                    {"Label": "Child1B", "Parent_RID": parent1_rid, "Parent_Group": group_a_rid},
                    {"Label": "Child2A", "Parent_RID": parent2_rid, "Parent_Group": group_b_rid},
                ]
            )
        )

        return {
            "groups": groups,
            "parents": parents,
            "children": children,
            "group_a_rid": group_a_rid,
            "group_b_rid": group_b_rid,
            "parent1_rid": parent1_rid,
            "parent2_rid": parent2_rid,
        }

    def _create_dataset_with_children(self, ml: DerivaML, data: dict, description: str):
        """Helper: create a dataset with Child members via an execution."""
        # Refresh model to pick up dynamically created tables
        ml.model.refresh_model()
        ml.add_dataset_element_type("Child")
        # Ensure the workflow type exists (test_ml fixture starts with clean vocab)
        try:
            ml.lookup_term(MLVocab.workflow_type, "Test Workflow")
        except Exception:
            ml.add_term(MLVocab.workflow_type, "Test Workflow", description="Workflow type for testing")
        workflow = ml.create_workflow(name="Composite FK Test", workflow_type="Test Workflow")
        execution = ml.create_execution(ExecutionConfiguration(description="Test", workflow=workflow))
        dataset = execution.create_dataset(dataset_types=[], description=description)
        child_rids = [c["RID"] for c in data["children"]]
        dataset.add_dataset_members(members={"Child": child_rids})
        dataset.increment_version("Initial data")
        return dataset

    def test_table_relationship_composite_fk(self, test_ml: DerivaML):
        """_table_relationship must return all columns of a composite FK.

        Currently broken: uses [0] indexing which drops additional FK columns.
        """
        self._create_composite_fk_schema(test_ml)
        self._populate_composite_fk_data(test_ml)

        # _table_relationship should return ALL column pairs of the composite FK
        # Need to refresh the model to pick up newly created tables
        test_ml.model.refresh_model()
        col_pairs = test_ml.model._table_relationship("Child", "Parent")

        # With a composite FK (Parent_RID, Parent_Group) → (RID, Group),
        # we expect 2 column pairs
        assert len(col_pairs) == 2, (
            f"Expected 2 column pairs for composite FK, got {len(col_pairs)}. "
            f"_table_relationship may be dropping columns with [0] indexing."
        )
        fk_col_names = {fk.name for fk, _ in col_pairs}
        pk_col_names = {pk.name for _, pk in col_pairs}
        assert fk_col_names == {"Parent_RID", "Parent_Group"}, f"FK columns: {fk_col_names}"
        assert pk_col_names == {"RID", "Group"}, f"PK columns: {pk_col_names}"

    def test_denormalize_across_composite_fk(self, test_ml: DerivaML, tmp_path):
        """Denormalization must produce correct rows across composite FKs.

        This is the core bug: bag.denormalize_as_dataframe(["Child", "Parent"])
        should return 3 rows (one per Child), each joined to their Parent.
        Currently returns 0 rows because the composite FK join is incomplete.
        """
        self._create_composite_fk_schema(test_ml)
        data = self._populate_composite_fk_data(test_ml)

        dataset = self._create_dataset_with_children(test_ml, data, "Composite FK test dataset")

        # Download bag and denormalize
        version = dataset.current_version
        bag = dataset.download_dataset_bag(version=version, materialize=False)

        # Verify Child table is in the bag with data
        child_df = bag.get_table_as_dataframe("Child")
        assert len(child_df) == 3, f"Expected 3 children in bag, got {len(child_df)}"

        # Verify Parent table is in the bag with data (this is where the bug manifests)
        parent_df = bag.get_table_as_dataframe("Parent")
        assert len(parent_df) == 2, (
            f"Expected 2 parents in bag, got {len(parent_df)}. "
            "Composite FK path traversal likely failed to export parent data."
        )

        # Denormalize Child + Parent — should produce 3 rows
        df = bag.denormalize_as_dataframe(include_tables=["Child", "Parent"])
        assert len(df) == 3, (
            f"Expected 3 denormalized rows (one per child), got {len(df)}. "
            "Composite FK join in denormalization is likely incomplete."
        )

        # Verify join is correct — each child should have their parent's name
        child_labels = set(df["Child.Label"].tolist())
        assert child_labels == {"Child1A", "Child1B", "Child2A"}

        parent_names = set(df["Parent.Name"].tolist())
        assert parent_names == {"Parent1", "Parent2"}

    def test_denormalize_three_table_chain_with_composite_fk(
        self, test_ml: DerivaML, tmp_path
    ):
        """Denormalize through a chain: Child →(composite FK)→ Parent →(simple FK)→ Group.

        Tests that composite FKs don't break multi-hop denormalization.
        """
        self._create_composite_fk_schema(test_ml)
        data = self._populate_composite_fk_data(test_ml)

        dataset = self._create_dataset_with_children(test_ml, data, "Three-table composite FK test")

        version = dataset.current_version
        bag = dataset.download_dataset_bag(version=version, materialize=False)

        # Denormalize all three tables
        df = bag.denormalize_as_dataframe(include_tables=["Child", "Parent", "Group"])
        assert len(df) == 3, (
            f"Expected 3 rows for 3-table chain, got {len(df)}. "
            "Composite FK may be breaking the multi-hop join."
        )

        # Verify Group names are present
        group_names = set(df["Group.Name"].tolist())
        assert group_names == {"Alpha", "Beta"}

    def test_bag_export_populates_parent_table_via_composite_fk(
        self, test_ml: DerivaML, tmp_path
    ):
        """Bag export must follow composite FK paths to populate parent tables.

        The bag should contain Parent rows that are reachable via composite FK
        from Child members. This tests the FK path traversal in CatalogGraph.
        """
        self._create_composite_fk_schema(test_ml)
        data = self._populate_composite_fk_data(test_ml)

        dataset = self._create_dataset_with_children(test_ml, data, "Bag export composite FK test")

        version = dataset.current_version
        bag = dataset.download_dataset_bag(version=version, materialize=False)

        # Both parent rows should be in the bag (reachable from children)
        parent_df = bag.get_table_as_dataframe("Parent")
        assert len(parent_df) >= 2, (
            f"Expected at least 2 parents in bag via composite FK path, got {len(parent_df)}. "
            "Bag export FK traversal is not following composite FKs correctly."
        )

        # Group rows should also be reachable (Parent → Group is a simple FK)
        group_df = bag.get_table_as_dataframe("Group")
        assert len(group_df) >= 2, (
            f"Expected at least 2 groups in bag, got {len(group_df)}. "
            "Transitive FK path through composite FK is broken."
        )

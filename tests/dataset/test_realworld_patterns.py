"""Tests for FK patterns found in real-world catalogs (FaceBase, eye-ai, etc.).

These tests create schemas that mimic the structural patterns found in complex
production catalogs, without depending on those external catalogs. Each test
class creates its own schema extensions on top of the base test catalog.

Patterns tested:
  - Deep FK chains (5+ hops) with max_depth pruning
  - Nullable composite FKs (LEFT JOIN semantics)
  - Catalog-side composite FK denormalization
  - denormalize_columns() preview method
  - Self-referential FKs (hierarchical data)
  - Wide schema with many tables (path explosion guard)
"""

import pytest
from deriva.core.typed import BuiltinType, ColumnDef, ForeignKeyDef, KeyDef, TableDef

from deriva_ml import DerivaML
from deriva_ml.core.definitions import MLVocab
from deriva_ml.execution.execution import ExecutionConfiguration


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _ensure_workflow(ml: DerivaML, name: str = "Pattern Test") -> None:
    """Ensure the workflow type and a workflow exist for creating executions."""
    try:
        ml.lookup_term(MLVocab.workflow_type, "Test Workflow")
    except Exception:
        ml.add_term(MLVocab.workflow_type, "Test Workflow", description="Testing")


def _create_dataset_with_members(ml: DerivaML, element_table: str, rids: list[str], description: str):
    """Create a dataset with the given members."""
    ml.model.refresh_model()
    ml.add_dataset_element_type(element_table)
    _ensure_workflow(ml)
    workflow = ml.create_workflow(name="Pattern Test", workflow_type="Test Workflow")
    execution = ml.create_execution(ExecutionConfiguration(description="Test", workflow=workflow))
    dataset = execution.create_dataset(dataset_types=[], description=description)
    dataset.add_dataset_members(members={element_table: rids})
    return dataset


# ===========================================================================
# Deep FK chains (mimics FaceBase: Dataset → assoc → file → biosample →
# experiment → study → project)
# ===========================================================================


class TestDeepFKChains:
    """Test denormalization across deep FK chains (5+ hops).

    Mimics FaceBase-style schemas where data elements are connected through
    many intermediate tables: file → biosample → experiment → study → project.
    """

    def _create_deep_chain_schema(self, ml: DerivaML) -> None:
        """Create a 6-table linear FK chain: A → B → C → D → E → F."""
        schema_name = ml.default_schema
        model = ml.catalog.getCatalogModel()
        domain = model.schemas[schema_name]

        if "ChainF" in domain.tables:
            return

        # Build chain from leaf to root (F has no FKs, E→F, D→E, etc.)
        domain.create_table(TableDef(name="ChainF", columns=[ColumnDef("F_Name", BuiltinType.text)]))
        for prev, cur in [("E", "F"), ("D", "E"), ("C", "D"), ("B", "C"), ("A", "B")]:
            domain.create_table(
                TableDef(
                    name=f"Chain{prev}",
                    columns=[
                        ColumnDef(f"{prev}_Name", BuiltinType.text),
                        ColumnDef(f"Chain{cur}_Ref", BuiltinType.text, nullok=False),
                    ],
                    foreign_keys=[
                        ForeignKeyDef(
                            columns=[f"Chain{cur}_Ref"],
                            referenced_schema=schema_name,
                            referenced_table=f"Chain{cur}",
                            referenced_columns=["RID"],
                        ),
                    ],
                )
            )

    def _populate_deep_chain(self, ml: DerivaML) -> dict:
        pb = ml.pathBuilder()
        domain = pb.schemas[ml.default_schema]

        f_rows = list(domain.tables["ChainF"].insert([{"F_Name": "Root"}]))
        e_rows = list(domain.tables["ChainE"].insert([{"E_Name": "Level5", "ChainF_Ref": f_rows[0]["RID"]}]))
        d_rows = list(domain.tables["ChainD"].insert([{"D_Name": "Level4", "ChainE_Ref": e_rows[0]["RID"]}]))
        c_rows = list(domain.tables["ChainC"].insert([{"C_Name": "Level3", "ChainD_Ref": d_rows[0]["RID"]}]))
        b_rows = list(domain.tables["ChainB"].insert([{"B_Name": "Level2", "ChainC_Ref": c_rows[0]["RID"]}]))
        a_rows = list(domain.tables["ChainA"].insert([{"A_Name": "Level1", "ChainB_Ref": b_rows[0]["RID"]}]))

        return {"a": a_rows, "b": b_rows, "c": c_rows, "d": d_rows, "e": e_rows, "f": f_rows}

    def test_deep_chain_bag_denormalize(self, test_ml: DerivaML, tmp_path):
        """Denormalize across a 6-table deep FK chain via bag."""
        self._create_deep_chain_schema(test_ml)
        data = self._populate_deep_chain(test_ml)

        dataset = _create_dataset_with_members(test_ml, "ChainA", [r["RID"] for r in data["a"]], "Deep chain test")
        bag = dataset.download_dataset_bag(dataset.current_version, materialize=False)

        df = bag.denormalize_as_dataframe(include_tables=["ChainA", "ChainB", "ChainC", "ChainD", "ChainE", "ChainF"])
        assert len(df) == 1
        assert df["ChainA.A_Name"].iloc[0] == "Level1"
        assert df["ChainF.F_Name"].iloc[0] == "Root"

    def test_deep_chain_catalog_denormalize(self, test_ml: DerivaML, tmp_path):
        """Denormalize across a 6-table deep FK chain via live catalog."""
        self._create_deep_chain_schema(test_ml)
        data = self._populate_deep_chain(test_ml)

        dataset = _create_dataset_with_members(
            test_ml, "ChainA", [r["RID"] for r in data["a"]], "Deep chain catalog test"
        )
        df = dataset.denormalize_as_dataframe(
            include_tables=["ChainA", "ChainB", "ChainC", "ChainD", "ChainE", "ChainF"]
        )
        assert len(df) == 1
        assert df["ChainA.A_Name"].iloc[0] == "Level1"
        assert df["ChainF.F_Name"].iloc[0] == "Root"

    def test_schema_to_paths_max_depth(self, test_ml: DerivaML):
        """_schema_to_paths with max_depth limits path length."""
        self._create_deep_chain_schema(test_ml)
        data = self._populate_deep_chain(test_ml)

        # Register ChainA as a dataset element type so paths flow through it
        _create_dataset_with_members(test_ml, "ChainA", [r["RID"] for r in data["a"]], "max_depth test")
        test_ml.model.refresh_model()

        # Without depth limit, should find paths reaching ChainF
        # Path: Dataset → Dataset_ChainA → ChainA → ChainB → ... → ChainF (8 hops)
        all_paths = test_ml.model._schema_to_paths()
        deep_sigs = {" -> ".join(t.name for t in p) for p in all_paths if any(t.name == "ChainF" for t in p)}
        assert len(deep_sigs) > 0, "Should find paths to ChainF without depth limit"

        # With max_depth=4, should NOT reach ChainF (needs 8 hops from Dataset)
        shallow_paths = test_ml.model._schema_to_paths(max_depth=4)
        shallow_sigs = {" -> ".join(t.name for t in p) for p in shallow_paths if any(t.name == "ChainF" for t in p)}
        assert len(shallow_sigs) == 0, "max_depth=4 should not reach ChainF"

    def test_partial_chain_denormalize(self, test_ml: DerivaML, tmp_path):
        """Denormalize a subset of the chain (skip intermediate tables)."""
        self._create_deep_chain_schema(test_ml)
        data = self._populate_deep_chain(test_ml)

        dataset = _create_dataset_with_members(test_ml, "ChainA", [r["RID"] for r in data["a"]], "Partial chain test")
        bag = dataset.download_dataset_bag(dataset.current_version, materialize=False)

        # Include only endpoints: ChainA and ChainF, with intermediates auto-joined
        df = bag.denormalize_as_dataframe(include_tables=["ChainA", "ChainB", "ChainC", "ChainD", "ChainE", "ChainF"])
        assert len(df) == 1
        assert df["ChainF.F_Name"].iloc[0] == "Root"


# ===========================================================================
# Nullable composite FKs (LEFT JOIN semantics)
# ===========================================================================


class TestNullableCompositeFKs:
    """Test nullable composite FKs produce LEFT JOIN semantics.

    Mimics FaceBase patterns where file records may have nullable references
    to biosample via composite FKs. Records with NULL FK values should be
    preserved (not dropped by INNER JOIN).
    """

    def _create_nullable_composite_schema(self, ml: DerivaML) -> None:
        """Create tables with nullable composite FK.

        Schema:
            Target(RID, Name, Category)
              - compound key on (RID, Category)
            Source(RID, Label, Target_RID, Target_Category)
              - nullable composite FK: (Target_RID, Target_Category) → Target(RID, Category)
        """
        schema_name = ml.default_schema
        model = ml.catalog.getCatalogModel()
        domain = model.schemas[schema_name]

        if "NullTarget" in domain.tables:
            return

        domain.create_table(
            TableDef(
                name="NullTarget",
                columns=[
                    ColumnDef("Name", BuiltinType.text, nullok=False),
                    ColumnDef("Category", BuiltinType.text, nullok=False),
                ],
            )
        )

        target_table = model.schemas[schema_name].tables["NullTarget"]
        target_table.create_key(KeyDef(columns=["RID", "Category"]))

        domain.create_table(
            TableDef(
                name="NullSource",
                columns=[
                    ColumnDef("Label", BuiltinType.text, nullok=False),
                    ColumnDef("Target_RID", BuiltinType.text, nullok=True),
                    ColumnDef("Target_Category", BuiltinType.text, nullok=True),
                ],
                foreign_keys=[
                    ForeignKeyDef(
                        columns=["Target_RID", "Target_Category"],
                        referenced_schema=schema_name,
                        referenced_table="NullTarget",
                        referenced_columns=["RID", "Category"],
                    ),
                ],
            )
        )

    def _populate_nullable_composite(self, ml: DerivaML) -> dict:
        pb = ml.pathBuilder()
        domain = pb.schemas[ml.default_schema]

        targets = list(
            domain.tables["NullTarget"].insert(
                [
                    {"Name": "T1", "Category": "CatA"},
                ]
            )
        )

        sources = list(
            domain.tables["NullSource"].insert(
                [
                    {"Label": "Linked", "Target_RID": targets[0]["RID"], "Target_Category": "CatA"},
                    {"Label": "Orphan", "Target_RID": None, "Target_Category": None},
                ]
            )
        )

        return {"targets": targets, "sources": sources}

    def test_nullable_composite_fk_preserves_null_rows(self, test_ml: DerivaML, tmp_path):
        """Records with NULL composite FK should be preserved (LEFT JOIN)."""
        self._create_nullable_composite_schema(test_ml)
        data = self._populate_nullable_composite(test_ml)

        dataset = _create_dataset_with_members(
            test_ml, "NullSource", [s["RID"] for s in data["sources"]], "Nullable composite FK"
        )
        bag = dataset.download_dataset_bag(dataset.current_version, materialize=False)

        df = bag.denormalize_as_dataframe(include_tables=["NullSource", "NullTarget"])
        # Both rows should be present: the linked one AND the orphan with NULLs
        assert len(df) == 2, f"Expected 2 rows (LEFT JOIN preserves NULL FK rows), got {len(df)}"
        labels = set(df["NullSource.Label"].tolist())
        assert labels == {"Linked", "Orphan"}


# ===========================================================================
# Catalog-side composite FK denormalization
# ===========================================================================


class TestCatalogSideCompositeFKDenormalize:
    """Test composite FK denormalization via live catalog (not just bag).

    The existing composite FK tests only exercise the bag path. This verifies
    that the catalog-side denormalization also handles composite FKs.
    """

    def _create_composite_schema(self, ml: DerivaML) -> None:
        """Reuse the same composite FK schema as TestCompositeFKDenormalize."""
        schema_name = ml.default_schema
        model = ml.catalog.getCatalogModel()
        domain = model.schemas[schema_name]

        if "CatGroup" in domain.tables:
            return

        domain.create_table(TableDef(name="CatGroup", columns=[ColumnDef("Name", BuiltinType.text, nullok=False)]))
        domain.create_table(
            TableDef(
                name="CatParent",
                columns=[
                    ColumnDef("Name", BuiltinType.text, nullok=False),
                    ColumnDef("CatGroup_Ref", BuiltinType.text, nullok=False),
                ],
                foreign_keys=[
                    ForeignKeyDef(
                        columns=["CatGroup_Ref"],
                        referenced_schema=schema_name,
                        referenced_table="CatGroup",
                        referenced_columns=["RID"],
                    ),
                ],
            )
        )
        parent_table = model.schemas[schema_name].tables["CatParent"]
        parent_table.create_key(KeyDef(columns=["RID", "CatGroup_Ref"]))

        domain.create_table(
            TableDef(
                name="CatChild",
                columns=[
                    ColumnDef("Label", BuiltinType.text, nullok=False),
                    ColumnDef("Parent_RID", BuiltinType.text, nullok=False),
                    ColumnDef("Parent_Group", BuiltinType.text, nullok=False),
                ],
                foreign_keys=[
                    ForeignKeyDef(
                        columns=["Parent_RID", "Parent_Group"],
                        referenced_schema=schema_name,
                        referenced_table="CatParent",
                        referenced_columns=["RID", "CatGroup_Ref"],
                    ),
                ],
            )
        )

    def _populate(self, ml: DerivaML) -> dict:
        pb = ml.pathBuilder()
        domain = pb.schemas[ml.default_schema]

        groups = list(domain.tables["CatGroup"].insert([{"Name": "G1"}, {"Name": "G2"}]))
        parents = list(
            domain.tables["CatParent"].insert(
                [
                    {"Name": "P1", "CatGroup_Ref": groups[0]["RID"]},
                    {"Name": "P2", "CatGroup_Ref": groups[1]["RID"]},
                ]
            )
        )
        children = list(
            domain.tables["CatChild"].insert(
                [
                    {"Label": "C1", "Parent_RID": parents[0]["RID"], "Parent_Group": groups[0]["RID"]},
                    {"Label": "C2", "Parent_RID": parents[1]["RID"], "Parent_Group": groups[1]["RID"]},
                ]
            )
        )
        return {"groups": groups, "parents": parents, "children": children}

    def test_catalog_denormalize_composite_fk(self, test_ml: DerivaML, tmp_path):
        """Catalog-side denormalization across composite FK."""
        self._create_composite_schema(test_ml)
        data = self._populate(test_ml)

        dataset = _create_dataset_with_members(
            test_ml, "CatChild", [c["RID"] for c in data["children"]], "Catalog composite FK"
        )
        df = dataset.denormalize_as_dataframe(include_tables=["CatChild", "CatParent"])
        assert len(df) == 2
        assert set(df["CatChild.Label"].tolist()) == {"C1", "C2"}
        assert set(df["CatParent.Name"].tolist()) == {"P1", "P2"}

    def test_catalog_denormalize_composite_fk_three_table(self, test_ml: DerivaML, tmp_path):
        """Catalog-side three-table chain through composite FK."""
        self._create_composite_schema(test_ml)
        data = self._populate(test_ml)

        dataset = _create_dataset_with_members(
            test_ml, "CatChild", [c["RID"] for c in data["children"]], "3-table composite"
        )
        df = dataset.denormalize_as_dataframe(include_tables=["CatChild", "CatParent", "CatGroup"])
        assert len(df) == 2
        assert set(df["CatGroup.Name"].tolist()) == {"G1", "G2"}


# ===========================================================================
# denormalize_columns() preview tests
# ===========================================================================


class TestDenormalizeColumns:
    """Test denormalize_columns() returns correct column metadata without fetching data."""

    def test_columns_single_table(self, dataset_test, tmp_path):
        """denormalize_columns for a single table returns its columns."""
        dataset = dataset_test.dataset_description.dataset
        cols = dataset.denormalize_columns(include_tables=["Subject"])

        col_names = [name for name, _ in cols]
        assert "Subject.RID" in col_names
        assert "Subject.Name" in col_names
        # System columns should be excluded
        assert "Subject.RCT" not in col_names
        assert "Subject.RMB" not in col_names

    def test_columns_multiple_tables(self, dataset_test, tmp_path):
        """denormalize_columns for multiple tables returns all columns."""
        dataset = dataset_test.dataset_description.dataset
        cols = dataset.denormalize_columns(include_tables=["Subject", "Image", "Observation"])

        col_names = [name for name, _ in cols]
        # Should have columns from each table, dot-prefixed
        assert any(c.startswith("Subject.") for c in col_names)
        assert any(c.startswith("Image.") for c in col_names)
        assert any(c.startswith("Observation.") for c in col_names)

    def test_columns_includes_type_info(self, dataset_test, tmp_path):
        """denormalize_columns returns type information."""
        dataset = dataset_test.dataset_description.dataset
        cols = dataset.denormalize_columns(include_tables=["Subject"])

        col_dict = dict(cols)
        assert "Subject.Name" in col_dict
        # Name column should be text type
        assert col_dict["Subject.Name"] == "text"

    def test_columns_bag_matches_catalog(self, dataset_test, tmp_path):
        """Bag and catalog denormalize_columns return the same column names."""
        dataset = dataset_test.dataset_description.dataset
        version = dataset.current_version

        catalog_cols = dataset.denormalize_columns(include_tables=["Subject", "Image", "Observation"])
        bag = dataset.download_dataset_bag(version, use_minid=False)
        bag_cols = bag.denormalize_columns(include_tables=["Subject", "Image", "Observation"])

        catalog_names = {name for name, _ in catalog_cols}
        bag_names = {name for name, _ in bag_cols}
        assert catalog_names == bag_names, (
            f"Column name mismatch:\n"
            f"  Catalog only: {catalog_names - bag_names}\n"
            f"  Bag only: {bag_names - catalog_names}"
        )


# ===========================================================================
# Self-referential FK (hierarchical data)
# ===========================================================================


class TestSelfReferentialFK:
    """Test schemas with self-referential FKs (table references itself).

    Common in hierarchical data like Experiment.Parent → Experiment.RID.
    The path discovery should not infinite-loop on self-referential FKs.
    """

    def _create_hierarchical_schema(self, ml: DerivaML) -> None:
        schema_name = ml.default_schema
        model = ml.catalog.getCatalogModel()
        domain = model.schemas[schema_name]

        if "HierNode" in domain.tables:
            return

        domain.create_table(
            TableDef(
                name="HierNode",
                columns=[
                    ColumnDef("Name", BuiltinType.text, nullok=False),
                    ColumnDef("Parent_Node", BuiltinType.text, nullok=True),
                ],
                foreign_keys=[
                    ForeignKeyDef(
                        columns=["Parent_Node"],
                        referenced_schema=schema_name,
                        referenced_table="HierNode",
                        referenced_columns=["RID"],
                    ),
                ],
            )
        )

    def _populate_hierarchy(self, ml: DerivaML) -> dict:
        pb = ml.pathBuilder()
        domain = pb.schemas[ml.default_schema]

        root = list(
            domain.tables["HierNode"].insert(
                [
                    {"Name": "Root", "Parent_Node": None},
                ]
            )
        )
        child = list(
            domain.tables["HierNode"].insert(
                [
                    {"Name": "Child1", "Parent_Node": root[0]["RID"]},
                ]
            )
        )
        return {"root": root, "child": child}

    def test_self_referential_fk_no_infinite_loop(self, test_ml: DerivaML):
        """_schema_to_paths should not infinite-loop on self-referential FK."""
        self._create_hierarchical_schema(test_ml)
        self._populate_hierarchy(test_ml)
        test_ml.model.refresh_model()

        # Should complete without hanging. The cycle detection in
        # _schema_to_paths prevents infinite recursion.
        paths = test_ml.model._schema_to_paths()
        assert len(paths) > 0

    def test_self_referential_denormalize(self, test_ml: DerivaML, tmp_path):
        """Denormalize a single self-referential table."""
        self._create_hierarchical_schema(test_ml)
        data = self._populate_hierarchy(test_ml)

        all_rids = [r["RID"] for r in data["root"] + data["child"]]
        dataset = _create_dataset_with_members(test_ml, "HierNode", all_rids, "Hierarchical test")
        bag = dataset.download_dataset_bag(dataset.current_version, materialize=False)

        df = bag.denormalize_as_dataframe(include_tables=["HierNode"])
        assert len(df) == 2
        names = set(df["HierNode.Name"].tolist())
        assert names == {"Root", "Child1"}


# ===========================================================================
# Wide schema (many tables, path explosion guard)
# ===========================================================================


class TestWideSchema:
    """Test that path discovery handles wide schemas without path explosion.

    Mimics catalogs with many tables (like eye-ai with 20+ domain tables).
    Verifies _schema_to_paths terminates efficiently.
    """

    def _create_wide_schema(self, ml: DerivaML, n_tables: int = 8) -> None:
        """Create a star schema: Hub table with n_tables leaf tables pointing to it."""
        schema_name = ml.default_schema
        model = ml.catalog.getCatalogModel()
        domain = model.schemas[schema_name]

        if "Hub" in domain.tables:
            return

        domain.create_table(TableDef(name="Hub", columns=[ColumnDef("Name", BuiltinType.text)]))

        for i in range(n_tables):
            domain.create_table(
                TableDef(
                    name=f"Spoke{i}",
                    columns=[
                        ColumnDef("Value", BuiltinType.text),
                        ColumnDef("Hub_Ref", BuiltinType.text, nullok=False),
                    ],
                    foreign_keys=[
                        ForeignKeyDef(
                            columns=["Hub_Ref"],
                            referenced_schema=schema_name,
                            referenced_table="Hub",
                            referenced_columns=["RID"],
                        ),
                    ],
                )
            )

    def test_wide_schema_path_discovery_terminates(self, test_ml: DerivaML):
        """Path discovery on a wide schema should terminate without explosion."""
        self._create_wide_schema(test_ml, n_tables=8)
        test_ml.model.refresh_model()

        paths = test_ml.model._schema_to_paths()
        # Should complete. The star topology has bounded paths:
        # Dataset → assoc → Hub, Dataset → assoc → Hub → Spoke_i
        # Not exponential.
        assert len(paths) > 0

    def test_wide_schema_denormalize_subset(self, test_ml: DerivaML, tmp_path):
        """Denormalize a subset of tables from a wide schema."""
        self._create_wide_schema(test_ml, n_tables=8)
        test_ml.model.refresh_model()

        pb = test_ml.pathBuilder()
        domain = pb.schemas[test_ml.default_schema]

        hubs = list(domain.tables["Hub"].insert([{"Name": "Center"}]))
        spokes = list(
            domain.tables["Spoke0"].insert(
                [
                    {"Value": "S0", "Hub_Ref": hubs[0]["RID"]},
                ]
            )
        )

        dataset = _create_dataset_with_members(test_ml, "Spoke0", [s["RID"] for s in spokes], "Wide schema test")
        bag = dataset.download_dataset_bag(dataset.current_version, materialize=False)

        df = bag.denormalize_as_dataframe(include_tables=["Spoke0", "Hub"])
        assert len(df) == 1
        assert df["Hub.Name"].iloc[0] == "Center"

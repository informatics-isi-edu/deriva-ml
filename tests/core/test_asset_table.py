"""Tests for DerivaML.create_asset_table (issue #74).

The public method promotes the canonical asset-table wiring (the
internal ``schema.create_schema.create_asset_table`` used at catalog
bootstrap) to a user-facing ``DerivaML`` method: the five standard
hatrac columns, the ``<Asset>_Asset_Type`` tag association, the
``<Asset>_Execution`` association carrying the ``Asset_Role`` FK, and
the standard Chaise asset annotations -- plus ``additional_columns``
for domain extensions. Validation-by-construction: a table created
this way always satisfies ``model.is_asset``.

Unit shape test (no catalog) lives alongside the integration tests so
the deriva-py ``AssetTableDef`` contract this method relies on is
pinned where the method is tested.
"""

import pytest

from deriva_ml import DerivaMLException
from deriva_ml.core.definitions import AssetTableDef, BuiltinTypes, ColumnDefinition


class TestAssetTableDefShape:
    """Unit: the AssetTableDef contract create_asset_table relies on."""

    def test_hatrac_columns_auto_generated(self):
        """AssetTableDef.to_dict() carries the 5 standard hatrac columns."""
        d = AssetTableDef(schema_name="s", name="Widget_File").to_dict()
        names = [c["name"] for c in d["column_definitions"]]
        for required in ("URL", "Filename", "Length", "MD5", "Description"):
            assert required in names, f"missing standard column {required}"

    def test_additional_columns_extend_shape(self):
        """columns= extends, not replaces, the standard hatrac shape."""
        extra = ColumnDefinition(name="Frame_Count", type=BuiltinTypes.int4)
        d = AssetTableDef(schema_name="s", name="Widget_File", columns=[extra]).to_dict()
        names = [c["name"] for c in d["column_definitions"]]
        assert "Frame_Count" in names
        for required in ("URL", "Filename", "Length", "MD5", "Description"):
            assert required in names


class TestCreateAssetTable:
    """Integration: DerivaML.create_asset_table against the test catalog."""

    def test_create_asset_table_full_shape(self, test_ml):
        ml = test_ml
        table = ml.create_asset_table(
            "Scan_File",
            additional_columns=[
                ColumnDefinition(name="Scanner_Model", type=BuiltinTypes.text),
            ],
            comment="Raw scanner output files.",
        )
        # The created table satisfies the asset contract (hatrac columns,
        # NOT NULL constraints, URL asset annotation).
        assert ml.model.is_asset("Scan_File")
        # Domain extension column present.
        assert "Scanner_Model" in [c.name for c in table.columns]
        # Asset_Type tag association exists.
        assoc_type = ml.model.name_to_table("Scan_File_Asset_Type")
        assert ml.model.is_association(assoc_type)
        # Execution association exists and carries the Asset_Role FK.
        assoc_exec = ml.model.name_to_table("Scan_File_Execution")
        assert ml.model.is_association(assoc_exec, max_arity=3, pure=False)
        assert "Asset_Role" in [c.name for c in assoc_exec.columns]

    def test_create_asset_table_duplicate_raises(self, test_ml):
        ml = test_ml
        ml.create_asset_table("Dup_File", update_navbar=False)
        with pytest.raises(DerivaMLException):
            ml.create_asset_table("Dup_File", update_navbar=False)

    def test_create_asset_table_no_hatrac(self, test_ml):
        """use_hatrac=False still yields a valid asset table (plain URL column)."""
        ml = test_ml
        ml.create_asset_table("Plain_File", use_hatrac=False, update_navbar=False)
        assert ml.model.is_asset("Plain_File")

from deriva_ml import BuiltinTypes, ColumnDefinition


class TestAssets:
    def test_create_assets(self, test_ml, tmp_path):
        ml_instance = test_ml
        ml_instance.create_asset("FooAsset")
        assert "FooAsset" in [a.name for a in ml_instance.model.find_assets()]
        ml_instance.create_asset(
            "BarAsset",
            column_defs=[ColumnDefinition(name="foo", type=BuiltinTypes.int4)],
        )
        assert "BarAsset" in [a.name for a in ml_instance.model.find_assets()]
        assert ml_instance.model.asset_metadata("BarAsset") == {"foo"}

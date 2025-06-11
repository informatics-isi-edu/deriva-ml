import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition, DerivaMLException


class TestVocabulary:
    def test_add_term(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.create_vocabulary("CV2", "A vocab")
        assert len(ml_instance.list_vocabulary_terms("CV2")) == 0
        term = ml_instance.add_term("CV2", "T1", description="A vocab")
        assert len(ml_instance.list_vocabulary_terms("CV2")) == 1
        assert term.name == ml_instance.lookup_term("CV2", "T1").name

        # Check for redundant terms.
        with pytest.raises(DerivaMLException) as exc_info:
            ml_instance.add_term("CV2", "T1", description="A vocab", exists_ok=False)

        assert "T1" == ml_instance.add_term("CV2", "T1", description="A vocab").name


class TestAssets:
    def test_create_assets(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.create_asset("FooAsset")
        assert "FooAsset" in [a.name for a in ml_instance.model.find_assets()]
        ml_instance.create_asset(
            "BarAsset",
            column_defs=[ColumnDefinition(name="foo", type=BuiltinTypes.int4)],
        )
        assert "BarAsset" in [a.name for a in ml_instance.model.find_assets()]
        assert ml_instance.model.asset_metadata("BarAsset") == {"foo"}

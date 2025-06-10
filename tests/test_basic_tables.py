from derivaml_test import TestDerivaML

from deriva_ml import BuiltinTypes, ColumnDefinition, DerivaMLException


class TestVocabulary(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_add_term(self):
        self.ml_instance.create_vocabulary("CV2", "A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV2")), 0)
        term = self.ml_instance.add_term("CV2", "T1", description="A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV2")), 1)
        self.assertEqual(term.name, self.ml_instance.lookup_term("CV2", "T1").name)

        # Check for redundant terms.
        with self.assertRaises(DerivaMLException):
            self.ml_instance.add_term("CV2", "T1", description="A vocab", exists_ok=False)
        self.assertEqual("T1", self.ml_instance.add_term("CV2", "T1", description="A vocab").name)

    def test_create_assets(self):
        self.ml_instance.create_asset("FooAsset")
        self.assertIn("FooAsset", [a.name for a in self.ml_instance.model.find_assets()])
        self.ml_instance.create_asset(
            "BarAsset",
            column_defs=[ColumnDefinition(name="foo", type=BuiltinTypes.int4)],
        )
        self.assertIn("BarAsset", [a.name for a in self.ml_instance.model.find_assets()])
        self.assertEqual(1, len(self.ml_instance.model.asset_metadata("BarAsset")))

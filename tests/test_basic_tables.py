from derivaml_test import TestDerivaML

from deriva_ml import DerivaMLException, ColumnDefinition, BuiltinTypes


class TestVocabulary(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_find_vocabularies(self):
        # Look for a known vocabulary in the deriva-ml schema
        self.assertIn(
            "Dataset_Type", [v.name for v in self.ml_instance.model.find_vocabularies()]
        )

    def test_is_vocabulary(self):
        # Test the vocabulary table predicates.
        self.assertTrue(self.ml_instance.model.is_vocabulary("Dataset_Type"))
        self.assertFalse(self.ml_instance.model.is_vocabulary("Dataset"))
        self.assertRaises(
            DerivaMLException, self.ml_instance.model.is_vocabulary, "FooBar"
        )

    def test_create_vocabulary(self):
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertIn(
            "CV1", [v.name for v in self.ml_instance.model.find_vocabularies()]
        )
        self.assertTrue(self.ml_instance.model.is_vocabulary("Dataset_Type"))

    def test_add_term(self):
        self.ml_instance.create_vocabulary("CV2", "A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV2")), 0)
        term = self.ml_instance.add_term("CV2", "T1", description="A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV2")), 1)
        self.assertEqual(term.name, self.ml_instance.lookup_term("CV2", "T1").name)

        # Check for redundant terms.
        with self.assertRaises(DerivaMLException):
            self.ml_instance.add_term(
                "CV2", "T1", description="A vocab", exists_ok=False
            )
        self.assertEqual(
            "T1", self.ml_instance.add_term("CV2", "T1", description="A vocab").name
        )

    def test_find_assets(self):
        self.assertTrue(self.ml_instance.model.is_asset("Execution_Asset"))
        self.assertFalse(self.ml_instance.model.is_asset("Dataset"))
        self.assertIn(
            "Execution_Asset", [a.name for a in self.ml_instance.model.find_assets()]
        )

    def test_is_assoc(self):
        self.assertTrue(self.ml_instance.model.is_association("Dataset_Dataset"))
        self.assertFalse(self.ml_instance.model.is_association("Dataset"))

    def test_create_assets(self):
        self.ml_instance.create_asset("FooAsset")
        self.assertIn(
            "FooAsset", [a.name for a in self.ml_instance.model.find_assets()]
        )
        self.ml_instance.create_asset(
            "BarAsset",
            column_defs=[ColumnDefinition(name="foo", type=BuiltinTypes.int4)],
        )
        self.assertIn(
            "BarAsset", [a.name for a in self.ml_instance.model.find_assets()]
        )
        self.assertEqual(1, len(self.ml_instance.model.asset_metadata("BarAsset")))

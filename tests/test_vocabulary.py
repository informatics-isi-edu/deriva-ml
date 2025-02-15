import unittest
from deriva_ml.demo_catalog import populate_demo_catalog


@unittest.skipUnless(hostname, "Test host not specified")
class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.ml_instance = DerivaML(
            hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1"
        )
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def tearDown(self):
        pass

    def test_find_vocabularies(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        self.assertIn(
            "Dataset_Type", [v.name for v in self.ml_instance.find_vocabularies()]
        )

    def test_create_vocabulary(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertTrue(
            self.model.schemas[self.ml_instance.domain_schema].tables["CV1"]
        )

    def test_add_term(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 0)
        term = self.ml_instance.add_term("CV1", "T1", description="A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 1)
        self.assertEqual(term.name, self.ml_instance.lookup_term("CV1", "T1").name)

        # Check for redudent terms.
        with self.assertRaises(DerivaMLException) as context:
            self.ml_instance.add_term(
                "CV1", "T1", description="A vocab", exists_ok=False
            )
        self.assertEqual(
            "T1", self.ml_instance.add_term("CV1", "T1", description="A vocab").name
        )

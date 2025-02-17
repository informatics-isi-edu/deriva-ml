import unittest
from deriva_ml import DerivaML, DerivaMLException
from deriva.core import DerivaServer, get_credential
import os
from deriva_ml.demo_catalog import (
    create_demo_catalog,
    create_ml_schema,
    create_domain_schema,
)
import logging

hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
SNAME = os.getenv("DERIVA_PY_TEST_SNAME")
SNAME_DOMAIN = "deriva-test"

logger = logging.getLogger(__name__)
if os.getenv("DERIVA_PY_TEST_VERBOSE"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())


def setUpModule():
    print("Calling setupModule")
    global test_catalog
    logger.debug("setUpModule begin")
    credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
    server = DerivaServer("https", hostname, credentials=credential)
    try:
        test_catalog = server.create_ermrest_catalog()
        model = test_catalog.getCatalogModel()
        create_ml_schema(model)
        create_domain_schema(model, SNAME_DOMAIN)
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
    logger.debug("setUpModule  done")


def tearDownModule():
    logger.debug("tearDownModule begin")
    try:
        test_catalog.delete_ermrest_catalog(really=True)
    except Exception:
        pass
    logger.debug("tearDownModule done")


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
        print(self.ml_instance.find_vocabularies())
        self.assertIn(
            "Dataset_Type", [v.name for v in self.ml_instance.find_vocabularies()]
        )

    def test_create_vocabulary(self):
        # populate_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertTrue(
            self.model.schemas[self.ml_instance.domain_schema].tables["CV1"]
        )

    def test_add_term(self):
        # populate_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 0)
        term = self.ml_instance.add_term("CV1", "T1", description="A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 1)
        self.assertEqual(term.name, self.ml_instance.lookup_term("CV1", "T1").name)

        # Check for redundant terms.
        with self.assertRaises(DerivaMLException) as context:
            self.ml_instance.add_term(
                "CV1", "T1", description="A vocab", exists_ok=False
            )
        self.assertEqual(
            "T1", self.ml_instance.add_term("CV1", "T1", description="A vocab").name
        )

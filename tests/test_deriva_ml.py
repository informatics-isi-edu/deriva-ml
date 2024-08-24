# Tests for the datapath module.
#
# Environment variables:
#  DERIVA_PY_TEST_HOSTNAME: hostname of the test server
#  DERIVA_PY_TEST_CREDENTIAL: user credential, if none, it will attempt to get credentail for given hostname
#  DERIVA_PY_TEST_VERBOSE: set for verbose logging output to stdout
import builtins
from copy import deepcopy
import logging
from operator import itemgetter
import os
import unittest
import sys
from deriva.core import DerivaServer, ErmrestCatalog, get_credential, ermrest_model as _em
from deriva.core.datapath import DataPathException
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, FileUploadState, UploadState
from deriva_ml.schema_setup.create_schema import setup_ml_workflow
from deriva.chisel import Model, Schema, Table, Column, ForeignKey, builtin_types


try:
    from pandas import DataFrame
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SNAME_DOMAIN = 'ml-test'

hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
logger = logging.getLogger(__name__)
if os.getenv("DERIVA_PY_TEST_VERBOSE"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())


def define_domain_schema(model: Model) -> None:
    domain = model.create_schema(Schema.define(SNAME_DOMAIN))
    domain.create_table(_em.Table.define("Subject", column_defs=[Column.define('Name', builtin_types.text)]))

    image_table_def = Table.define_asset(
        sname=SNAME_DOMAIN,
        tname='Images',
        hatrac_template='/hatrac/execution_assets/{{MD5}}.{{Filename}}',
        column_defs=[Column.define('Subject', builtin_types.text)],
        fkey_defs=[
            ForeignKey.define(['RCB'], 'public', 'ERMrest_Client', ['ID']),
            ForeignKey.define(['RMB'], 'public', 'ERMrest_Client', ['ID']),
            ForeignKey.define(['Subject'], SNAME_DOMAIN, 'Subject', ['RID'])
        ],
    )
    domain.create_table(image_table_def)

def populate_test_catalog(catalog: ErmrestCatalog) -> None:
    pass

@unittest.skipUnless(hostname, "Test host not specified")
class DerivaMLTests (unittest.TestCase):
    catalog = None

    @classmethod
    def setUpClass(cls):
        logger.debug("setupUpClass begin")
        credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
        server = DerivaServer('https', hostname, credentials=credential)
        cls.catalog = server.create_ermrest_catalog()
        model = Model.from_catalog(server.connect_ermrest(cls.catalog.catalog_id))
        try:
            setup_ml_workflow(model, 'deriva-ml', cls.catalog.catalog_id, curie_prefix=SNAME_DOMAIN)
            define_domain_schema(model)
            populate_test_catalog(cls.catalog)
        except Exception:
            # on failure, delete catalog and re-raise exception
            cls.catalog.delete_ermrest_catalog(really=True)
            raise
        logger.debug("setupUpClass done")

    @classmethod
    def tearDownClass(cls):
        logger.debug("tearDownClass begin")
        cls.catalog.delete_ermrest_catalog(really=True)
        logger.debug("tearDownClass done")

    def setUp(self):
        self.ml_instance = DerivaML(hostname, self.catalog.catalog_id, SNAME_DOMAIN, None, None, "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]

    def tearDown(self):
        pass

    def test_find_vocabularies(self):
        self.assertIn("Dataset_Type", [v.name for v in self.ml_instance.find_vocabularies()])

    def test_create_vocabulary(self):
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertTrue(self.domain_schema.tables["CV1"])
        self.domain_schema.tables["CV1"].drop()


    def test_add_term(self):
        try:
            self.ml_instance.create_vocabulary("CV1", "A vocab")
            self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 0)
            rid = self.ml_instance.add_term("CV1", "T1", description="A vocab")
            self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 1)
            self.assertEqual(rid, self.ml_instance.lookup_term("CV1", "T1"))
        finally:
            self.domain_schema.tables["CV1"].drop()

    def test_find_features(self):
        self.ml_instance.find_features()

if __name__ == '__main__':
    sys.exit(unittest.main())

# Tests for the datapath module.
#
# Environment variables:
#  DERIVA_PY_TEST_HOSTNAME: hostname of the test server
#  DERIVA_PY_TEST_CREDENTIAL: user credential, if none, it will attempt to get credentail for given hostname
#  DERIVA_PY_TEST_VERBOSE: set for verbose logging output to stdout

from copy import deepcopy
import logging
from operator import itemgetter
import os
import unittest
import sys
from deriva.core import DerivaServer, ErmrestCatalog, get_credential, ermrest_model as _em, __version__
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, FileUploadState, UploadState
from deriva_ml.schema_setup.create_schema import setup_ml_workflow
from deriva.chisel import Model, Schema, Table, Column, ForeignKey




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


def define_test_schema(catalog: ErmrestCatalog) -> None:
    model = catalog.getCatalogModel()
    setup_ml_workflow(model, 'deriva-ml', catalog.catalog_id)
    domain = model.create_schema(_em.Schema.define(SNAME_DOMAIN))

    domain.create_table(_em.Table.define("Subject"))
    domain.create_table(_em.Table.define("Image"))

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
        try:
            define_test_schema(cls.catalog)
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
        self.ml_instance = DerivaML('www.eye-ai.org', 'feature-test', 'eye-ai', None, None, "1")

    def tearDown(self):
        try:
            self.experiment_copy.path.delete()
        except DataPathException:
            # suppresses 404 errors when the table is empty
            pass

    def test_catalog_dir_base(self):
        self.assertIn('schemas', dir(self.paths))


if __name__ == '__main__':
    sys.exit(unittest.main())

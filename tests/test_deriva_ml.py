# Tests for the datapath module.
#
# Environment variables:
#  DERIVA_PY_TEST_HOSTNAME: hostname of the test server
#  DERIVA_PY_TEST_CREDENTIAL: user credential, if none, it will attempt to get credentail for given hostname
#  DERIVA_PY_TEST_VERBOSE: set for verbose logging output to stdout
import logging
import os
import sys
import unittest

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from typing import Optional

from src.deriva_ml import (
    DerivaML,
    DerivaMLException,
    RID,
    ColumnDefinition,
    BuiltinTypes,
)
from src.deriva_ml import create_ml_schema
from test_catalog import create_domain_schema, populate_test_catalog
from src.deriva_ml import ExecutionConfiguration

try:
    from pandas import DataFrame

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SNAME_DOMAIN = "ml-test"

hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
logger = logging.getLogger(__name__)
if os.getenv("DERIVA_PY_TEST_VERBOSE"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())


test_catalog: Optional[ErmrestCatalog] = None

import unittest

# Discover and run all test cases in the "tests" directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover("tests")

test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)

def setUp()

def setUpModule():
    global test_catalog
    logger.debug("setUpModule begin")
    credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
    server = DerivaServer("https", hostname, credentials=credential)
    test_catalog = server.create_ermrest_catalog()
    model = test_catalog.getCatalogModel()
    try:
        create_ml_schema(model)
        create_domain_schema(model, SNAME_DOMAIN)
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    logger.debug("setUpModule  done")


def tearDownModule():
    logger.debug("tearDownModule begin")
    test_catalog.delete_ermrest_catalog(really=True)
    logger.debug("tearDownModule done")


if __name__ == "__main__":
    sys.exit(unittest.main())

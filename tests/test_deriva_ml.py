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
from deriva_ml.schema_setup.create_schema import initialize_ml_schema, create_ml_schema
from deriva_ml.demo_catalog import create_demo_catalog, populate_demo_catalog

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

    logger.debug("setUpModule begin")
    credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
    try:
        test_catalog = create_demo_catalog(hostname)
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    logger.debug("setUpModule  done")

# test_catalog = create_demo_catalog(hostname)


class DerivaMLTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        print("callign DerivaMLTest")
        self.test_catalog = test_catalog
        super().__init__(*args, **kwargs)


# Example assertion

# Discover and run all test cases in the "tests" directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover("tests")

print(test_suite)
test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)


if __name__ == "__main__":
    sys.exit(unittest.main())

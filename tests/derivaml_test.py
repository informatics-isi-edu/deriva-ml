import os
import unittest
from deriva_ml import DerivaML
from demo_catalog import create_demo_catalog


hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
SNAME = os.getenv("DERIVA_PY_TEST_SNAME")
SNAME_DOMAIN = "deriva-test"

TestCatalog = create_demo_catalog(
    hostname=hostname,
    domain_schema=SNAME_DOMAIN,
    create_features=False,
    create_datasets=False,
    populate=False,
)

ML_INSTANCE = DerivaML(hostname, TestCatalog.catalog_id, SNAME_DOMAIN)


class TestDerivaML(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_catalog = TestCatalog
        self.hostname = hostname
        self.domain_schema = SNAME_DOMAIN

    def setUp(self):
        self.ml_instance = ML_INSTANCE
        self.model = self.ml_instance.model

import os
import unittest
from deriva_ml import DerivaML, RID
from demo_catalog import (
    create_demo_catalog,
    populate_demo_catalog,
    create_demo_datasets,
    create_demo_features,
)
import logging

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

logging.basicConfig(level=logging.WARNING)


class TestDerivaML(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_catalog = TestCatalog
        self.hostname = hostname
        self.domain_schema = SNAME_DOMAIN
        self.catalog_populated = False

    def setUp(self):
        self.ml_instance = DerivaML(
            hostname,
            TestCatalog.catalog_id,
            SNAME_DOMAIN,
            logging_level=logging.WARNING,
        )
        self.model = self.ml_instance.model

    def populate_catalog(self):
        if not self.catalog_populated:
            logging.info("Populating catalog")
            populate_demo_catalog(self.ml_instance, self.domain_schema)
            self.catalog_populated = True

    def create_nested_dataset(self) -> tuple[RID, list[RID], list[RID]]:
        self.populate_catalog()
        logging.info("Creating nested dataset")
        return create_demo_datasets(self.ml_instance)

    def create_features(self):
        self.populate_catalog()
        create_demo_features(self.ml_instance)

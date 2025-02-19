import unittest

from deriva_ml import DerivaML, RID
from deriva.core import DerivaServer, get_credential
import os
from deriva_ml.demo_catalog import (
    create_ml_schema,
    create_domain_schema,
    reset_demo_catalog,
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


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(
            hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1"
        )
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def tearDown(self):
        pass

    def test_add_element_type(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.add_dataset_element_type("Subject")
        self.assertEqual(len(list(self.ml_instance.list_dataset_element_types())), 2)

    def test_create_dataset(self) -> RID:
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.add_dataset_element_type("Subject")
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        dataset_rid = self.ml_instance.create_dataset(
            type_rid.name, description="A Dataset"
        )
        self.assertEqual(len(list(self.ml_instance.find_datasets())), 1)
        return dataset_rid

    def test_add_dataset_members(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        subject_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[SNAME_DOMAIN]
            .tables["Subject"]
        )

        dataset_rid = self.test_create_dataset()
        subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(5)])
        subject_rids = [i["RID"] for i in subject_path.entities().fetch()]
        self.ml_instance.add_dataset_members(
            dataset_rid=dataset_rid, members=subject_rids
        )
        self.assertEqual(
            len(self.ml_instance.list_dataset_members(dataset_rid)["Subject"]),
            len(subject_rids),
        )
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 2)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "0.2.0")

    def test_remove_dataset_members(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        subject_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[SNAME_DOMAIN]
            .tables["Subject"]
        )

        dataset_rid = self.test_create_dataset()
        subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(5)])
        subject_rids = [i["RID"] for i in subject_path.entities().fetch()]
        self.ml_instance.add_dataset_members(
            dataset_rid=dataset_rid, members=subject_rids
        )

        subject_rids = self.ml_instance.list_dataset_members(dataset_rid)["Subject"]
        self.assertEqual(len(subject_rids), 5)
        self.ml_instance.delete_dataset_members(subject_rids[0:1])
        subject_rids = self.ml_instance.list_dataset_members(dataset_rid)["Subject"]
        self.assertEqual(len(subject_rids), 3)
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 3)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "0.3.0")

import os
import unittest
from deriva_ml import DerivaML, DatasetVersion, RID
from demo_catalog import create_demo_catalog, populate_demo_catalog
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

ML_INSTANCE = DerivaML(
    hostname, TestCatalog.catalog_id, SNAME_DOMAIN, logging_level=logging.WARN
)


class TestDerivaML(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_catalog = TestCatalog
        self.hostname = hostname
        self.domain_schema = SNAME_DOMAIN

    def setUp(self):
        self.ml_instance = ML_INSTANCE
        self.model = self.ml_instance.model

    def create_nested_dataset(self) -> tuple[RID, set[RID], set[RID]]:
        populate_demo_catalog(self.ml_instance, self.domain_schema)
        self.ml_instance.add_dataset_element_type("Subject")
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        table_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["Subject"]
        )
        subject_rids = [i["RID"] for i in table_path.entities().fetch()]
        dataset_rids = []
        for r in subject_rids[0:4]:
            d = self.ml_instance.create_dataset(
                type_rid.name,
                description=f"Dataset {r}",
                version=DatasetVersion(1, 0, 0),
            )
            self.ml_instance.add_dataset_members(d, [r])
            dataset_rids.append(d)
        nested_datasets = []
        for i in range(0, 4, 2):
            nested_dataset = self.ml_instance.create_dataset(
                type_rid.name,
                description=f"Nested Dataset {i}",
                version=DatasetVersion(1, 0, 0),
            )
            self.ml_instance.add_dataset_members(
                nested_dataset, dataset_rids[i : i + 1]
            )
            nested_datasets.append(nested_dataset)
        double_nested_dataset = self.ml_instance.create_dataset(
            type_rid.name,
            description=f"Double nested dataset",
            version=DatasetVersion(1, 0, 0),

        )
        self.ml_instance.add_dataset_members(double_nested_dataset, nested_datasets)
        return double_nested_dataset, set(nested_datasets), set(dataset_rids)

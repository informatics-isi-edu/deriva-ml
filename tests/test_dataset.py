class TestDataset(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(
            hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1"
        )
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def test_add_element_type(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.add_dataset_element_type("Subject")
        self.assertEqual(
            len(list(self.ml_instance.dataset_table.find_associations())), 2
        )

    def test_create_dataset(self) -> RID:
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.add_dataset_element_type("Subject")
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        dataset_rid = self.ml_instance.create_dataset(
            type_rid.name, description="A Dataset"
        )
        self.assertEqual(len(self.ml_instance.find_datasets()), 1)
        return dataset_rid

    def test_add_dataset_members(self):
        dataset_rid = self.test_create_dataset()
        subject_rids = [
            i["RID"]
            for i in self.ml_instance.catalog.getPathBuilder()
            .schemas[SNAME_DOMAIN]
            .tables["Subject"]
            .entities()
            .fetch()
        ]
        self.ml_instance.add_dataset_members(
            dataset_rid=dataset_rid, members=subject_rids
        )
        self.assertEqual(
            len(self.ml_instance.list_dataset_members(dataset_rid)["Subject"]),
            len(subject_rids),
        )

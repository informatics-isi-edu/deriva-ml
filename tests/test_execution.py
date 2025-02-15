class TestExecution(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(
            hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1"
        )
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model
        self.files = os.path.dirname(__file__) + "/files"

    def test_upload_configuration(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        config_file = self.files + "/test-workflow-1.json"
        config = ExecutionConfiguration.load_configuration(config_file)
        rid = self.ml_instance.upload_execution_configuration(config)
        self.assertEqual(rid, self.ml_instance.retrieve_rid(rid)["RID"])

    def test_execution_1(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        exec_config = ExecutionConfiguration.load_configuration(
            self.files + "/test-workflow-1.json"
        )
        configuration_record = self.ml_instance.initialize_execution(
            configuration=exec_config
        )
        with self.ml_instance.execution(configuration=configuration_record) as exec:
            output_dir = self.ml_instance.execution_assets_path / "testoutput"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "test.txt", "w+") as f:
                f.write("Hello there\n")
        upload_status = self.ml_instance.upload_execution(
            configuration=configuration_record
        )
        e = (
            list(
                self.ml_instance.catalog.getPathBuilder()
                .deriva_ml.Execution.entities()
                .fetch()
            )
        )[0]
        self.assertEqual(e["Status"], "Completed")

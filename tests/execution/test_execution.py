"""
Tests for the execution module.
"""

from tempfile import TemporaryDirectory

from deriva_ml import (
    DatasetSpec,
    DerivaML,
    ExecAssetType,
    ExecutionConfiguration,
    MLAsset,
)
from deriva_ml import (
    MLVocab as vc,
)


class TestExecution:
    def test_execution_no_download(self, test_ml):
        ml_instance = test_ml
        ml_instance.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
        ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")

        api_workflow = ml_instance.create_workflow(
            name="Test Workflow One",
            workflow_type="Test Workflow",
            description="A test operation",
        )

        manual_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )

        with manual_execution:
            pass
        manual_execution.upload_execution_outputs()
        assert "Completed" == ml_instance.retrieve_rid(manual_execution.execution_rid)["Status"]

        pb = ml_instance.pathBuilder.schemas[ml_instance.ml_schema]
        execution_metadata_execution = pb.Execution_Metadata_Execution
        execution_metadata = pb.Execution_Metadata
        metadata_path = execution_metadata_execution.path
        metadata_path.filter(execution_metadata_execution.Execution == manual_execution.execution_rid)
        metadata_path.link(execution_metadata)
        execution_metadata = [
            {
                "RID": a["RID"],
                "Filename": a["Filename"],
            }
            for a in metadata_path.entities().fetch()
        ]
        assert 2 == len(execution_metadata)

    def test_execution_upload(self, test_ml):
        ml_instance = test_ml
        ml_instance.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
        ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")

        api_workflow = ml_instance.create_workflow(
            name="Test Workflow One",
            workflow_type="Test Workflow",
            description="A test operation",
        )

        self.create_execution_asset(ml_instance, api_workflow)

    def test_execution_workflow_download(self, dataset_test, tmp_path):
        """Test creating and configuring an execution workflow."""
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        ml_instance.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
        ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")

        dataset_rid = dataset_test.dataset_description.rid

        # Create a workflow
        api_workflow = ml_instance.create_workflow(
            name="Test Workflow One",
            workflow_type="Test Workflow",
            description="A test operation",
        )

        model_rid = self.create_execution_asset(ml_instance, api_workflow)
        # Create execution configuration
        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(
                    rid=dataset_rid,
                    version=ml_instance.dataset_version(dataset_rid),
                ),
            ],
            assets=[model_rid],
            description="Sample Execution",
            workflow=api_workflow,
        )

        # Create manual execution
        test_execution = ml_instance.create_execution(config)
        with test_execution.execute() as execution:
            assert 1 == len(execution.asset_paths)
            assert 1 == len(execution.datasets)
            assert execution.datasets[0].dataset_rid == dataset_rid

    @staticmethod
    def create_execution_asset(ml_instance: DerivaML, api_workflow):
        manual_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )

        with manual_execution.execute() as execution:
            model_file = execution.asset_file_path(
                MLAsset.execution_asset, "API_Model/modelfile.txt", asset_types=ExecAssetType.model_file
            )
            with model_file.open("w") as fp:
                fp.write("My model")
            # Now upload the file and retrieve the RID of the new asset from the returned results.
        uploaded_assets = manual_execution.upload_execution_outputs()
        assert 1 == len(uploaded_assets["deriva-ml/Execution_Asset"])

        pb = ml_instance.pathBuilder.schemas[ml_instance.ml_schema]
        execution_asset_execution = pb.Execution_Asset_Execution
        execution_asset = pb.Execution_Asset
        asset_path = execution_asset_execution.path
        asset_path.filter(execution_asset_execution.Execution == manual_execution.execution_rid)
        asset_path.link(execution_asset)

        execution_asset = [
            {
                "RID": a["RID"],
                "Filename": a["Filename"],
            }
            for a in asset_path.entities().fetch()
        ]

        assert 1 == len(execution_asset)
        with TemporaryDirectory() as tmpdir:
            file = manual_execution.download_asset(execution_asset[0]["RID"], tmpdir, update_catalog=False)
            assert file.name == "modelfile.txt"
        return execution_asset[0]["RID"]

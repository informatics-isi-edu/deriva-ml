"""
Tests for the execution module.
"""

from deriva_ml import (
    DatasetSpec,
    ExecutionConfiguration,
    Workflow,
)
from deriva_ml import (
    MLVocab as vc,
)


def test_execution_workflow(test_ml_catalog):
    """Test creating and configuring an execution workflow."""
    # Get dataset RIDs
    training_dataset_rid = [ds["RID"] for ds in test_ml_catalog.find_datasets() if "Training" in ds["Dataset_Type"]][0]
    testing_dataset_rid = [ds["RID"] for ds in test_ml_catalog.find_datasets() if "Testing" in ds["Dataset_Type"]][0]
    nested_dataset_rid = [ds["RID"] for ds in test_ml_catalog.find_datasets() if "Partitioned" in ds["Dataset_Type"]][0]

    # Add required terms
    test_ml_catalog.add_term(vc.workflow_type, "Manual Workflow", description="Initial setup of Model File")
    test_ml_catalog.add_term(vc.execution_asset_type, "API_Model", description="Model for our API workflow")
    test_ml_catalog.add_term(vc.workflow_type, "ML Demo", description="A ML Workflow that uses Deriva ML API")

    # Create manual workflow
    api_workflow = test_ml_catalog.add_workflow(
        Workflow(
            name="Manual Workflow",
            url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/Notebooks/DerivaML%20Execution.ipynb",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )
    )

    # Create manual execution
    manual_execution = test_ml_catalog.create_execution(
        ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
    )

    # Create model file
    model_file = manual_execution.execution_asset_path("API_Model") / "modelfile.txt"
    model_file.write_text("My model")

    # Upload outputs
    uploaded_assets = manual_execution.upload_execution_outputs()
    assert "API_Model/modelfile.txt" in uploaded_assets
    training_model_rid = uploaded_assets["API_Model/modelfile.txt"].result["RID"]

    # Create ML workflow
    api_workflow = Workflow(
        name="ML Demo",
        url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/pyproject.toml",
        workflow_type="ML Demo",
        description="A workflow that uses Deriva ML",
    )

    # Create execution configuration
    config = ExecutionConfiguration(
        datasets=[
            DatasetSpec(
                rid=nested_dataset_rid,
                version=test_ml_catalog.dataset_version(nested_dataset_rid),
            ),
            DatasetSpec(
                rid=testing_dataset_rid,
                version=test_ml_catalog.dataset_version(testing_dataset_rid),
            ),
        ],
        assets=[training_model_rid],
        description="Sample Execution",
        workflow=api_workflow,
    )

    assert len(config.datasets) == 2
    assert len(config.assets) == 1
    assert config.workflow.name == "ML Demo"


class TestExecution(TestDerivaML):
    def test_execution_no_download(self):
        self.ml_instance.add_term(
            vc.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        self.ml_instance.add_term(
            vc.execution_asset_type,
            "API_Model",
            description="Model for our API workflow",
        )
        self.ml_instance.add_term(
            vc.workflow_type,
            "ML Demo",
            description="A ML Workflow that uses Deriva ML API",
        )

        api_workflow = self.ml_instance.create_workflow(
            name="Manual Workflow",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )

        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )
        with manual_execution:
            pass
        manual_execution.upload_execution_outputs()

    def test_execution_download(self):
        self.populate_catalog()
        double_nested, nested, datasets = self.create_nested_dataset()

        self.ml_instance.add_term(
            vc.execution_asset_type,
            "API_Model",
            description="Model for our API workflow",
        )
        self.ml_instance.add_term(
            vc.workflow_type,
            "ML Demo",
            description="A ML Workflow that uses Deriva ML API",
        )
        api_workflow = self.ml_instance.create_workflow(
            name="ML Demo",
            workflow_type="ML Demo",
            description="A workflow that uses Deriva ML",
        )

        execution_model = self.create_execution_asset(api_workflow)

        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(
                    rid=nested[0],
                    version=self.ml_instance.dataset_version(nested[0]),
                ),
                DatasetSpec(
                    rid=nested[1],
                    version=self.ml_instance.dataset_version(nested[1]),
                ),
            ],
            assets=[execution_model],
            description="Sample Execution",
            workflow=api_workflow,
        )
        exec = self.ml_instance.create_execution(config)
        with exec as e:
            print(e.asset_paths)
            print(e.datasets)
            self.assertEqual(1, len(e.asset_paths))
            self.assertEqual(2, len(e.datasets))
        exec.upload_execution_outputs()
        pb = self.ml_instance.pathBuilder.schemas[self.ml_instance.ml_schema]
        execution_asset_execution = pb.Execution_Asset_Execution
        execution_metadata_execution = pb.Execution_Metadata_Execution
        execution_asset = pb.Execution_Asset
        execution_metadata = pb.Execution_Metadata

        assets_execution = [
            {
                "RID": a["RID"],
                "Execution_Asset": a["Execution_Asset"],
                "Execution": a["Execution"],
            }
            for a in execution_asset_execution.entities().fetch()
            if a["Execution"] == exec.execution_rid
        ]
        metadata_execution = [
            {
                "RID": a["RID"],
                "Execution": a["Execution"],
                "Execution_Metadata": a["Execution_Metadata"],
            }
            for a in execution_metadata_execution.entities().fetch()
            if a["Execution"] == exec.execution_rid
        ]
        execution_assets = [{"RID": a["RID"], "Filename": a["Filename"]} for a in execution_asset.entities().fetch()]
        execution_metadata = [
            {"RID": a["RID"], "Filename": a["Filename"]} for a in execution_metadata.entities().fetch()
        ]
        print(assets_execution)
        print(metadata_execution)
        print(execution_assets)
        print(execution_metadata)
        self.assertEqual(1, len(assets_execution))
        self.assertEqual(2, len(metadata_execution))

    def create_execution_asset(self, api_workflow):
        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )
        model_file = manual_execution.execution_asset_path("API_Model") / "modelfile.txt"
        with open(model_file, "w") as fp:
            fp.write("My model")
        # Now upload the file and retrieve the RID of the new asset from the returned results.
        uploaded_assets = manual_execution.upload_execution_outputs()
        self.ml_instance._execution = None
        return uploaded_assets["API_Model/modelfile.txt"].result["RID"]

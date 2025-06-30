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


def test_execution_workflow(test_ml_catalog_dataset):
    """Test creating and configuring an execution workflow."""
    ml_instance = test_ml_catalog_dataset.ml_instance
    # Get dataset RIDs
    print(ml_instance.find_datasets())
    testing_dataset_rid = [ds["RID"] for ds in ml_instance.find_datasets() if "Testing" in ds["Dataset_Type"]][0]
    nested_dataset_rid = [ds["RID"] for ds in ml_instance.find_datasets() if "Partitioned" in ds["Dataset_Type"]][0]

    # Add required terms
    ml_instance.add_term(vc.workflow_type, "Manual Workflow", description="Initial setup of Model File")
    ml_instance.add_term(vc.asset_type, "API_Model", description="Model for our API workflow")
    ml_instance.add_term(vc.workflow_type, "ML Demo", description="A ML Workflow that uses Deriva ML API")

    # Create manual workflow
    api_workflow = ml_instance.add_workflow(
        Workflow(
            name="Manual Workflow",
            url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/Notebooks/DerivaML%20Execution.ipynb",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )
    )

    # Create manual execution
    manual_execution = ml_instance.create_execution(
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
                version=ml_instance.dataset_version(nested_dataset_rid),
            ),
            DatasetSpec(
                rid=testing_dataset_rid,
                version=ml_instance.dataset_version(testing_dataset_rid),
            ),
        ],
        assets=[training_model_rid],
        description="Sample Execution",
        workflow=api_workflow,
    )

    assert len(config.datasets) == 2
    assert len(config.assets) == 1
    assert config.workflow.name == "ML Demo"


class TestExecution:
    def test_execution_no_download(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.add_term(
            vc.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        ml_instance.add_term(
            vc.asset_type,
            "API_Model",
            description="Model for our API workflow",
        )
        ml_instance.add_term(
            vc.workflow_type,
            "ML Demo",
            description="A ML Workflow that uses Deriva ML API",
        )

        api_workflow = ml_instance.create_workflow(
            name="Manual Workflow",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )

        manual_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )
        with manual_execution:
            pass
        manual_execution.upload_execution_outputs()

    def test_execution_download(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.ml_instance
        dataset_description = test_ml_catalog_dataset.dataset_description
        double_nested = dataset_description.rid
        nested = ml_instance.list_dataset_children(double_nested)
        # datasets = [ds for nested_dataset in nested for ds in nested_dataset]

        ml_instance.add_term(
            vc.asset_type,
            "API_Model",
            description="Model for our API workflow",
        )
        ml_instance.add_term(
            vc.workflow_type,
            "ML Demo",
            description="A ML Workflow that uses Deriva ML API",
        )
        api_workflow = ml_instance.create_workflow(
            name="ML Demo",
            workflow_type="ML Demo",
            description="A workflow that uses Deriva ML",
        )

        execution_model = self.create_execution_asset(ml_instance, api_workflow)

        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(
                    rid=nested[0],
                    version=ml_instance.dataset_version(nested[0]),
                ),
                DatasetSpec(
                    rid=nested[1],
                    version=ml_instance.dataset_version(nested[1]),
                ),
            ],
            assets=[execution_model],
            description="Sample Execution",
            workflow=api_workflow,
        )
        exec = ml_instance.create_execution(config)
        with exec as e:
            print(e.asset_paths)
            print(e.datasets)
            assert 1 == len(e.asset_paths)
            assert 2 == len(e.datasets)
        exec.upload_execution_outputs()
        pb = ml_instance.pathBuilder.schemas[ml_instance.ml_schema]
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
        assert 1 == len(assets_execution)
        assert 2 == len(metadata_execution)

    def create_execution_asset(self, ml_instance, api_workflow):
        manual_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )
        model_file = manual_execution.asset_path("API_Model") / "modelfile.txt"
        with open(model_file, "w") as fp:
            fp.write("My model")
        # Now upload the file and retrieve the RID of the new asset from the returned results.
        uploaded_assets = manual_execution.upload_execution_outputs()
        ml_instance._execution = None
        return uploaded_assets["API_Model/modelfile.txt"].result["RID"]

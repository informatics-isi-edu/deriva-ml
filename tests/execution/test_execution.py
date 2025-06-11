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

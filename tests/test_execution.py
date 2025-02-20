from derivaml_test import TestDerivaML
from deriva_ml import MLVocab as vc, Workflow, ExecutionConfiguration, DatasetSpec
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
    populate_demo_catalog,
    create_demo_datasets,
)


class TestExecution(TestDerivaML):
    def test_execution_no_download(self):
        reset_demo_catalog(self.ml_instance, self.domain_schema)
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

        api_workflow = Workflow(
            name="Manual Workflow",
            url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/tests/test_execution.py",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )

        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(
                description="Sample Execution", workflow=api_workflow
            )
        )
        manual_execution.upload_execution_outputs()

    def test_execution_download(self):
        populate_demo_catalog(self.ml_instance, self.domain_schema)
        create_demo_datasets(self.ml_instance)
        exec_config = execution_test(self.ml_instance)
        exec = self.ml_instance.create_execution(exec_config)


def execution_test(ml_instance):
    training_dataset_rid = [
        ds["RID"]
        for ds in ml_instance.find_datasets()
        if "Training" in ds["Dataset_Type"]
    ][0]
    testing_dataset_rid = [
        ds["RID"]
        for ds in ml_instance.find_datasets()
        if "Testing" in ds["Dataset_Type"]
    ][0]

    nested_dataset_rid = [
        ds["RID"]
        for ds in ml_instance.find_datasets()
        if "Partitioned" in ds["Dataset_Type"]
    ][0]

    ml_instance.add_term(
        vc.workflow_type, "Manual Workflow", description="Initial setup of Model File"
    )
    ml_instance.add_term(
        vc.execution_asset_type, "API_Model", description="Model for our API workflow"
    )
    ml_instance.add_term(
        vc.workflow_type, "ML Demo", description="A ML Workflow that uses Deriva ML API"
    )
    api_workflow = Workflow(
        name="Manual Workflow",
        url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/Notebooks/DerivaML%20Execution.ipynb",
        workflow_type="Manual Workflow",
        description="A manual operation",
    )

    manual_execution = ml_instance.create_execution(
        ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
    )

    # Now lets create model configuration for our program.
    model_file = manual_execution.execution_asset_path("API_Model") / "modelfile.txt"
    with open(model_file, "w") as fp:
        fp.write(f"My model")

    # Now upload the file and retrieve the RID of the new asset from the returned results.
    uploaded_assets = manual_execution.upload_execution_outputs()

    training_model_rid = uploaded_assets["API_Model/modelfile.txt"].result["RID"]
    api_workflow = Workflow(
        name="ML Demo",
        url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/pyproject.toml",
        workflow_type="ML Demo",
        description="A workflow that uses Deriva ML",
    )

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
    return config

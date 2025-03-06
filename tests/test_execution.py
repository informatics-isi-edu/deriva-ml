from derivaml_test import TestDerivaML
from deriva_ml import (
    MLVocab as vc,
    Workflow,
    ExecutionConfiguration,
    DatasetSpec,
)


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

        api_workflow = self.ml_instance.add_workflow(
            Workflow(
                name="Manual Workflow",
                url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/tests/test_execution.py",
                workflow_type="Manual Workflow",
                description="A manual operation",
            )
        )

        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(
                description="Sample Execution", workflow=api_workflow
            )
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
        api_workflow = self.ml_instance.add_workflow(
            Workflow(
                name="ML Demo",
                url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/pyproject.toml",
                workflow_type="ML Demo",
                description="A workflow that uses Deriva ML",
            )
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
        execution_assets = [
            {"RID": a["RID"], "Filename": a["Filename"]}
            for a in execution_asset.entities().fetch()
        ]
        execution_metadata = [
            {"RID": a["RID"], "Filename": a["Filename"]}
            for a in execution_metadata.entities().fetch()
        ]
        print(assets_execution)
        print(metadata_execution)
        print(execution_assets)
        print(execution_metadata)
        self.assertEqual(1, len(assets_execution))
        self.assertEqual(2, len(metadata_execution))

    def create_execution_asset(self, api_workflow):
        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(
                description="Sample Execution", workflow=api_workflow
            )
        )
        model_file = (
            manual_execution.execution_asset_path("API_Model") / "modelfile.txt"
        )
        with open(model_file, "w") as fp:
            fp.write("My model")
        # Now upload the file and retrieve the RID of the new asset from the returned results.
        uploaded_assets = manual_execution.upload_execution_outputs()
        self.ml_instance._execution = None
        return uploaded_assets["API_Model/modelfile.txt"].result["RID"]

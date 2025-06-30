from random import random
from tempfile import TemporaryDirectory

from deriva_ml import ExecutionConfiguration, MLVocab
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
)


class TestUpload:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_upload_directory(self):
        reset_demo_catalog(ml_instance, self.domain_schema)
        self.ml_instance.create_asset("FooBar")
        with TemporaryDirectory() as tmpdir:
            asset_dir = ml_instance.asset_dir("FooBar", prefix=tmpdir)
            for s in range(2):
                asset_file = asset_dir.create_file(f"test_{s}.txt", metadata={})
                with open(asset_file, "w+") as f:
                    f.write(f"Hello there {random()}\n")
            ml_instance.upload_assets(asset_dir)
        assets = list(
            ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["FooBar"].entities().fetch()
        )
        assert len(assets) == 2

    def test_upload_directory_metadata(self):
        reset_demo_catalog(ml_instance, self.domain_schema)
        subject_path = ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["Subject"]
        ss = list(subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(2)]))
        with TemporaryDirectory() as tmpdir:
            image_dir = ml_instance.asset_dir("Image", prefix=tmpdir)
            for s in ss:
                image_file = image_dir.create_file(f"test_{s['RID']}.txt", {"Subject": s["RID"]})
                with open(image_file, "w+") as f:
                    f.write(f"Hello there {random()}\n")
            ml_instance.upload_assets(image_dir)
        assets = list(
            ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["Image"].entities().fetch()
        )
        assert assets[0]["Subject"] in [s["RID"] for s in ss]
        assert len(assets) == 2

    def test_upload_execution_outputs(self):
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        ml_instance.add_term(
            MLVocab.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        ml_instance.add_term(
            MLVocab.asset_type,
            "API_Model",
            description="Model for our API workflow",
        )

        api_workflow = self.ml_instance.create_workflow(
            name="Manual Workflow",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )

        manual_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )

        # Now let us create model configuration for our program.
        model_file = manual_execution.execution_asset_path("API_Model") / "modelfile.txt"
        with open(model_file, "w") as fp:
            fp.write("My model")

        # Now upload the file and retrieve the RID of the new asset from the returned results.
        manual_execution.upload_execution_outputs()
        path = ml_instance.catalog.getPathBuilder().schemas["deriva-ml"]
        assert 1 == len(list(path.Execution_Asset.entities().fetch()))

        execution_metadata = list(path.Execution_Metadata.entities().fetch())
        print([m for m in execution_metadata])
        assert 2 == len(execution_metadata)

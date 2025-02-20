from derivaml_test import TestDerivaML
from deriva_ml import MLVocab as vc, Workflow, ExecutionConfiguration
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
)


class TestExecution(TestDerivaML):
    def test_execution_1(self):
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

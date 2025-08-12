import sys

from deriva_ml import DerivaML
from deriva_ml import MLVocab as vc

hostname = sys.argv[1]
catalog_id = sys.argv[2]


ml_instance = DerivaML(hostname, catalog_id)

ml_instance.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")
api_workflow = ml_instance.create_workflow(
    name="Test Workflow One",
    workflow_type="Test Workflow",
    description="A test operation",
)
rid = ml_instance.add_workflow(api_workflow)
print(rid)

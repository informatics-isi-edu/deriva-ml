
from .deriva_ml import DatasetBag
form .deriva_ml import DerivaML
from deriva_ml import MLVocab as vc
from deriva_ml.dataset import Dataset
from deriva_ml.execution_configuration import ExecutionConfiguration, Workflow
from deriva_ml.demo_catalog import create_demo_catalog, DemoML

host = 'dev.eye-ai.org'
catalog_id = "eye-ai"
source_dataset = '2-39FY'
create_catalog = False

gnl = GlobusNativeLogin(host=host)
if gnl.is_logged_in([host]):
    print("You are already logged in.")
else:
    gnl.login([host], no_local_server=True, no_browser=True, refresh_tokens=True, update_bdbag_keychain=True)
    print("Login Successful")

test_workflow = Workflow(
    name="LAC data template",
    url="https://github.com/informatics-isi-edu/eye-ai-exec/blob/main/notebooks/templates/template_lac.ipynb",
    workflow_type="Test Workflow"
)

# Configuration instance.
config = ExecutionConfiguration(
    datasets=[{'rid':source_dataset, 'materialize':False}],
    # Materialize set to False if you only need the metadata from the bag, and not the assets.
    assets=['2-4JR6'],
    workflow=test_workflow,
    description="Template instance of a feature creation workflow")

if create_catalog:
    test_catalog = create_demo_catalog(host, 'demo-schema', create_features=True, create_datasets=True)
    catalog_id = test_catalog.catalog_id
ml_instance = DerivaML(hostname=host, catalog_id=catalog_id)

ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A test Workflow for new DM")

def create_execution():
    execution = ml_instance.create_execution(config)
    return execution





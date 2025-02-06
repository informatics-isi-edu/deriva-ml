from execution_configuration import ExecutionConfiguration, Workflow
from deriva_ml_base import DerivaML
from deriva_definitions import MLVocab as vc, DerivaSystemColumns
from deriva_ml.demo_catalog import create_demo_catalog, DemoML
import pandas as pd

host = 'dev.eye-ai.org'
catalog_id = "eye-ai"


#source_dataset = '2-7K8W'
source_dataset = '3R6'
create_catalog = False
test_catalog = create_demo_catalog('dev.eye-ai.org', 'demo-schema', create_features=True, create_datasets=True)
catalog_id = test_catalog.catalog_id
ml_instance = DerivaML(hostname=host, catalog_id=catalog_id)

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
    test_catalog = create_demo_catalog('dev.eye-ai.org', 'demo-schema', create_features=True, create_datasets=True)
    catalog_id = test_catalog.catalog_id
ml_instance = DerivaML(hostname=host, catalog_id=catalog_id)

ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A test Workflow for new DM")

def create_execution():
    execution = ml_instance.create_execution(config)
    return execution



from deriva.core import ErmrestCatalog, get_credential
from deriva.core.deriva_server import DerivaServer
from deriva_ml.schema_setup.create_schema import create_ml_schema
from demo_catalog import create_domain_schema, create_demo_datasets
from deriva_ml import DerivaML
server = DerivaServer("https", 'dev.eye-ai.org', credentials=get_credential('dev.eye-ai.org'))
test_catalog = server.create_ermrest_catalog()
model = test_catalog.getCatalogModel()
create_ml_schema(model, project_name='foo')
create_domain_schema(model, 'demo')
deriva_ml = DerivaML(
    hostname='dev.eye-ai.org',
    catalog_id=test_catalog.catalog_id,
    project_name='foo',
)
create_demo_datasets(deriva_ml)

test_catalog = create_demo_catalog(host, 'test-schema', create_features=True, create_datasets=True)
ml_instance = DemoML(host, test_catalog.catalog_id)
datasets = pd.DataFrame(ml_instance.find_datasets()).drop(columns=DerivaSystemColumns)
training_dataset_rid = [ds['RID'] for ds in ml_instance.find_datasets() if 'Training' in ds['Dataset_Type']][0]
testing_dataset_rid = [ds['RID'] for ds in ml_instance.find_datasets() if 'Testing' in ds['Dataset_Type']][0]

def execution_test():
    ml_instance.add_term(vc.workflow_type, "Manual Workflow", description="Inital setup of Model File")
    ml_instance.add_term(vc.execution_asset_type, "API_Model", description="Model for our API workflow")
    ml_instance.add_term(vc.workflow_type, "ML Demo", description="A ML Workflow that uses Deriva ML API")

    api_workflow = Workflow(
        name="Manual Workflow",
        url='https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/Notebooks/DerivaML%20Execution.ipynb',
        workflow_type="Manual Workflow",
        description="A manual operation"
    )

    manual_execution = ml_instance.create_execution(
        ExecutionConfiguration(description="Sample Execution", workflow=api_workflow))

    # Now lets create model configuration for our program.
    model_file = manual_execution.execution_asset_path('API_Model') / 'modelfile.txt'
    with open(model_file, "w") as fp:
        fp.write(f"My model")

    # Now upload the file and retrieve the RID of the new asset from the returned results.
    uploaded_assets = manual_execution.upload_execution_outputs()
    training_model_rid = uploaded_assets['API_Model/modelfile.txt'].result['RID']
    api_workflow = Workflow(
        name="ML Demo",
        url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/pyproject.toml",
        workflow_type="ML Demo",
        description="A workflow that uses Deriva ML"
    )

    config = ExecutionConfiguration(
        datasets=[training_dataset_rid, {'rid': testing_dataset_rid, 'materialize': False}],
        assets=[training_model_rid],
        description="Sample Execution",
        workflow=api_workflow
    )
    return ml_instance.create_execution(config)
host = 'dev.eye-ai.org'
catalog_id = "eye-ai"

#source_dataset = '2-7K8W'
source_dataset = '3R6'
create_catalog = False

from deriva_ml.demo_catalog import create_demo_catalog, DemoML
from deriva_ml import Workflow, ExecutionConfiguration, MLVocab as vc

def setup():
    host = 'dev.eye-ai.org'
    test_catalog = create_demo_catalog(host, 'test-schema', create_features=True, create_datasets=True)
    ml_instance = DemoML(host, test_catalog.catalog_id)
    config = execution_test(ml_instance)
    return ml_instance, config

def lac_test():
    source_dataset = '2-7K8W'
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
    return config

def create_execution():
    execution = ml_instance.create_execution(config)
    return execution


def create_demo_ml():
    host = 'dev.eye-ai.org'
    test_catalog = create_demo_catalog(host, 'test-schema', create_features=True, create_datasets=True)
    return DemoML(host, test_catalog.catalog_id)


def execution_test(ml_instance):
    training_dataset_rid = [ds['RID'] for ds in ml_instance.find_datasets() if 'Training' in ds['Dataset_Type']][0]
    testing_dataset_rid = [ds['RID'] for ds in ml_instance.find_datasets() if 'Testing' in ds['Dataset_Type']][0]

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
        datasets=[training_dataset_rid, {'rid': testing_dataset_rid, 'materialize': True}],
        assets=[training_model_rid],
        description="Sample Execution",
        workflow=api_workflow
    )
    return config



from deriva_ml import DatasetBag
from deriva_ml.dataset import Dataset
from deriva_ml.demo_catalog import create_demo_catalog, DemoML
test_catalog = create_demo_catalog('dev.eye-ai.org', 'demo-schema', create_features=True, create_datasets=True)
ml_instance = DemoML('dev.eye-ai.org', test_catalog.catalog_id)
bag = ml_instance.download_dataset_bag('3QY')
ds = DatasetBag(bag[0])

ds = DatasetBag(bag[1])
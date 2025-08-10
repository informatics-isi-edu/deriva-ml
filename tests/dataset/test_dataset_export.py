from pathlib import Path

from catalog import DerivaModel
from dataset import Dataset
from deriva.core.ermrest_model import Model


class TestDatasetVersion:

    def test_dataset_version_simple(self, test_ml, tmp_path):

        # Monkey patch
        def _dataset_nesting_depth(self, dataset_rid = None) -> int:
            return 1
        Dataset._dataset_nesting_depth = _dataset_nesting_depth

        schema_file = Path("dataset/demo-catalog-eye-ai-catalog-schema.json").absolute()
        schema_file = Path("dataset/eye-ai-catalog-schema.json").absolute()

        model = Model.fromfile("file-system", schema_file)
        deriva_model = DerivaModel(model)

        dataset = Dataset(deriva_model,
                          cache_dir=tmp_path,
                          working_dir=tmp_path,
                          use_minid=False)
        annotations = dataset._generate_dataset_download_annotations()['tag:isrd.isi.edu,2021:export-fragment-definitions']['dataset_export_outputs']
        outputs =  [o['destination']['name'] for o in annotations if 'name' in o['destination']]
        assert 'Dataset/Dataset_Version' in outputs
        print(outputs)
from tempfile import TemporaryDirectory
from urllib.parse import quote as urlquote

from demo_catalog import create_demo_features
from deriva.core.datapath import DataPathException

from deriva_ml import DerivaML
from deriva_ml.demo_catalog import (
    DatasetDescription,
    create_demo_catalog,
    create_demo_datasets,
    populate_demo_catalog,
)


class MLCatalog:
    def __init__(self, hostname):
        self.catalog = create_demo_catalog(
            hostname,
            domain_schema="test-schema",
            project_name="ml-test",
            populate=False,
            create_features=False,
            create_datasets=False,
            on_exit_delete=False,
        )
        self.catalog_id = self.catalog.catalog_id
        self.hostname = hostname
        self.domain_schema = "test-schema"
        print(f"🚀 Created demo catalog {self.catalog_id}")

    def cleanup(self):
        print("Deleting demo catalog")
        self.catalog.delete_ermrest_catalog(really=True)

    def reset_demo_catalog(self):
        """Reset the demo catalog to a clean state."""
        # Remove executions
        # Remove datasets
        # Remove features
        print("Resetting demo catalog")
        pb = self.catalog.getPathBuilder()
        ml_path = pb.schemas["deriva-ml"]
        domain_path = pb.schemas[self.domain_schema]
        for t in ["Dataset_Execution", "Dataset_Version", "Dataset_Dataset"]:
            try:
                ml_path.tables[t].path.delete()
            except DataPathException:
                pass
            except Exception:
                pass
        for t in ["Dataset_Subject", "Image_Subject"]:
            try:
                domain_path.tables[t].path.delete()
            except KeyError:
                pass
            except DataPathException:
                pass

        for t in [
            "Execution_Image_BoundingBox",
            "Execution_Image_Quality",
            "Execution_Subject_Health",
        ]:
            try:
                domain_path.tables[t].path.delete()
            except DataPathException:
                pass
            except KeyError:
                pass
        print("Resetting history...")
        cat_desc = self.catalog.get("/").json()
        latest = cat_desc["snaptime"]
        self.catalog.delete("/history/,%s" % (urlquote(latest),))


class MLDatasetCatalog:
    def __init__(self, catalog: MLCatalog, features: bool = False):
        self.features = features
        self.catalog = catalog

        with TemporaryDirectory() as tmpdir:
            ml_instance = DerivaML(catalog.hostname, catalog.catalog_id, use_minid=False, working_dir=tmpdir)
            populate_demo_catalog(ml_instance)
            self.dataset_description = create_demo_datasets(ml_instance)
            create_demo_features(ml_instance)

    def list_datasets(self, dataset_description: DatasetDescription) -> list[DatasetDescription]:
        """Return a set of RIDs whose members are members of the given dataset description."""
        nested_datasets = [
            ds
            for dset_member in dataset_description.members.get("Dataset", [])
            for ds in self.list_datasets(dset_member)
        ]
        return [dataset_description] + nested_datasets

    def collect_rids(self, description: DatasetDescription) -> set[str]:
        """Collect rids for a dataset and its nested datasets."""
        rids = {description.rid}
        for member_type, member_descriptor in description.members.items():
            rids |= set(description.member_rids.get(member_type, []))
            if member_type == "Dataset":
                for dataset in member_descriptor:
                    rids |= self.collect_rids(dataset)
        return rids

    def reset_catalog(self):
        """Reset the demo catalog to a clean state."""
        self.catalog.reset_demo_catalog()
        with TemporaryDirectory() as tmp_dir:
            ml_instance = DerivaML(self.catalog.hostname, self.catalog.catalog_id, working_dir=tmp_dir, use_minid=False)
            self.dataset_description: DatasetDescription = create_demo_datasets(ml_instance)

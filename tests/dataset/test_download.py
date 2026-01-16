from pathlib import Path
from pprint import pformat

from icecream import ic

# Local imports
from deriva_ml import DerivaML, MLVocab, TableDefinition
from deriva_ml.dataset.aux_classes import DatasetSpec, VersionPart
from deriva_ml.dataset.dataset import Dataset, DatasetBag
from deriva_ml.demo_catalog import DatasetDescription
from deriva_ml.model.deriva_ml_database import DerivaMLDatabase
from tests.test_utils import MLDatasetCatalog

ic.configureOutput(
    argToStringFunction=lambda x: pformat(x.model_dump() if hasattr(x, "model_dump") else x, width=80, depth=10)
)


class TestDatasetDownload:
    def list_datasets(self, dataset_description: DatasetDescription) -> set[Dataset]:
        nested_datasets = {
            ds
            for dset_member in dataset_description.members.get("Dataset", [])
            for ds in self.list_datasets(dset_member)
        }
        return {dataset_description.dataset} | nested_datasets

    def compare_datasets(
        self, ml_instance: DerivaML, dataset: MLDatasetCatalog, dataset_spec: DatasetSpec, recurse=False
    ):
        reference_datasets = self.list_datasets(dataset.dataset_description)
        versioned_dataset = ml_instance.lookup_dataset(dataset=dataset_spec.rid)
        bag = versioned_dataset.download_dataset_bag(version=dataset_spec.version, use_minid=False)
        # Check to see if all of the files have been downloaded.
        # Use list_dataset_members to get dataset-scoped data
        members = bag.list_dataset_members()
        files = [Path(r["Filename"]) for r in members.get("Image", [])]
        for f in files:
            assert f.exists()
        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r.dataset_rid for r in reference_datasets}

        # Now look at each dataset to see if they line up.
        ic("checking elements")
        db_catalog = DerivaMLDatabase(bag.model)
        # Sort reference_datasets by RID to ensure deterministic iteration order
        for ds in sorted(reference_datasets, key=lambda d: d.dataset_rid):
            dataset_bag = db_catalog.lookup_dataset(ds.dataset_rid)  # Get nested bag from the dataset.
            snapshot_ds = ml_instance.lookup_dataset(dataset=ds.dataset_rid)
            catalog_elements = snapshot_ds.list_dataset_members(version=dataset_spec.version, recurse=recurse)
            del catalog_elements["File"]  # Files is not in the bag.
            bag_elements = dataset_bag.list_dataset_members(recurse=recurse)

            assert len(catalog_elements) == len(bag_elements)  # Files is not in the bag.

            for t, members in catalog_elements.items():
                bag_members = bag_elements[t]
                bag_members.sort(key=lambda x: x["RID"])
                members.sort(key=lambda x: x["RID"])
                assert len(members) == len(bag_elements[t])
                for m, bm in zip(members, bag_members):
                    skip_keys = ["Description", "RMT", "RCT", "RCB", "RMB", "Filename", "Acquisition_Date",
                                 "Acquisition_Time"]
                    # For Dataset table entries, also skip Version since it can differ between
                    # the catalog snapshot and the bag
                    if t == "Dataset":
                        skip_keys.append("Version")
                    m = {k: v for k, v in m.items() if k not in skip_keys}
                    bm = {k: v for k, v in bm.items() if k not in skip_keys}
                    assert m == bm, f"Mismatch for dataset {ds.dataset_rid}, type {t}: catalog={m} vs bag={bm}"

    def test_bag_dataset_find(self, dataset_test, tmp_path):
        dataset_description = dataset_test.dataset_description
        dataset = dataset_test.dataset_description.dataset
        current_version = dataset.current_version
        bag = dataset.download_dataset_bag(current_version, use_minid=False)
        reference_datasets = {ds.dataset.dataset_rid for ds in dataset_test.list_datasets(dataset_description)}

        # Use DerivaMLDatabase for catalog-level operations
        db_catalog = DerivaMLDatabase(bag.model)
        bag_datasets = {ds.dataset_rid for ds in db_catalog.find_datasets()}
        assert reference_datasets == bag_datasets

        for ds in db_catalog.find_datasets():
            dataset_types = ds.dataset_types
            for t in dataset_types:
                assert db_catalog.lookup_term(MLVocab.dataset_type, t) is not None

        # Now check top level nesting
        assert set(dataset_description.member_rids["Dataset"]) == set(
            ds.dataset_rid for ds in bag.list_dataset_children()
        )
        # Now look two levels down
        for ds in dataset_description.members["Dataset"]:
            bag_child = db_catalog.lookup_dataset(ds.dataset.dataset_rid)
            assert set(ds.member_rids["Dataset"]) == set(c.dataset_rid for c in
                                                         bag_child.list_dataset_children())

        def check_relationships(description: DatasetDescription, bg: DatasetBag):
            """Check relationships between datasets."""
            dataset_children = set(ds.dataset_rid for ds in bg.list_dataset_children())
            assert set(description.member_rids.get("Dataset", [])) == dataset_children
            for child in bg.list_dataset_children():
                assert child.list_dataset_parents()[0].dataset_rid == bg.dataset_rid
            for nested_dataset in description.members.get("Dataset", []):
                nested_bag = db_catalog.lookup_dataset(nested_dataset.dataset.dataset_rid)
                check_relationships(nested_dataset, nested_bag)

        check_relationships(dataset_description, bag)

    def test_dataset_download_nested(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        current_version = dataset_description.dataset.current_version
        dataset_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        self.compare_datasets(ml_instance, dataset_test, dataset_spec)

    def test_dataset_download_recurse(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description
        reference_datasets = dataset_test.list_datasets(dataset_description)

        current_version = dataset_description.dataset.current_version
        dataset_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        bag = ml_instance.download_dataset_bag(dataset_spec)
        db_catalog = DerivaMLDatabase(bag.model)

        for dataset in reference_datasets:
            reference_members = dataset_test.collect_rids(dataset)
            member_rids = {dataset.dataset.dataset_rid}
            dataset_bag = db_catalog.lookup_dataset(dataset.dataset.dataset_rid)
            for member_type, dataset_members in dataset_bag.list_dataset_members(recurse=True).items():
                if member_type == "File":
                    continue
                member_rids |= {e["RID"] for e in dataset_members}
            assert reference_members == member_rids

    def test_dataset_download_versions(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description
        dataset = dataset_description.dataset

        current_version = dataset.current_version
        current_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        self.compare_datasets(ml_instance, dataset_test, current_spec)

        pb = ml_instance.pathBuilder()
        subjects = [s["RID"] for s in pb.schemas[ml_instance.domain_schema].tables["Subject"].path.entities().fetch()]

        dataset_description.dataset.add_dataset_members(subjects[-2:])
        new_version = dataset_description.dataset.current_version
        assert new_version == current_version.increment_version(VersionPart.minor)

        current_bag = dataset.download_dataset_bag(current_version, use_minid=False)
        new_bag = dataset.download_dataset_bag(new_version, use_minid=False)
        print([m["RID"] for m in dataset_description.dataset.list_dataset_members()["Subject"]])
        # Use list_dataset_members to get dataset-scoped data
        subjects_current = current_bag.list_dataset_members().get("Subject", [])
        subjects_new = new_bag.list_dataset_members().get("Subject", [])

        # Make sure that there is a difference between to old and new catalogs.
        assert len(subjects_new) == len(subjects_current) + 2

    def test_dataset_download_schemas(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        current_version = dataset_description.dataset.current_version

        ml_instance.create_table(
            TableDefinition(
                name="NewTable",
                column_defs=[],
            )
        )
        new_version = dataset_description.dataset.increment_dataset_version(component=VersionPart.minor)

        current_bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)
        new_bag = dataset_description.dataset.download_dataset_bag(new_version, use_minid=False)

        assert "NewTable" in new_bag.model.schemas[ml_instance.domain_schema].tables
        assert "NewTable" not in current_bag.model.schemas[ml_instance.domain_schema].tables

    def test_dataset_types_preserved_in_bag(self, dataset_test, tmp_path):
        """Test that dataset types in downloaded bag match the original catalog dataset types."""
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        # Get reference datasets with their types from the catalog
        reference_datasets = self.list_datasets(dataset_description)

        # Download the bag
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Use DerivaMLDatabase to access datasets in the bag
        db_catalog = DerivaMLDatabase(bag.model)

        # Check that dataset types match for all datasets in the hierarchy
        for catalog_dataset in reference_datasets:
            catalog_types = set(catalog_dataset.dataset_types)

            # Look up the same dataset in the downloaded bag
            bag_dataset = db_catalog.lookup_dataset(catalog_dataset.dataset_rid)
            bag_types = set(bag_dataset.dataset_types)

            assert catalog_types == bag_types, (
                f"Dataset types mismatch for dataset {catalog_dataset.dataset_rid}: "
                f"catalog={catalog_types}, bag={bag_types}"
            )

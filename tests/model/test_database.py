from pprint import pformat

from deriva.core.datapath import Any

from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.model.deriva_ml_database import DerivaMLDatabase
from tests.test_utils import DatasetDescription, DerivaML, MLDatasetCatalog

try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None

class TestDataBaseModel:
    def make_frozen(self, e) -> frozenset:
        e.pop("RMT")
        e.pop("RCT")
        e.pop("Acquisition_Date", None)
        e.pop("Acquisition_Time", None)
        e.pop("Filename", None)
        if "Description" in e and not e["Description"]:
            e["Description"] = ""
        return frozenset(e.items())

    def list_datasets(self, dataset_description: DatasetDescription) -> set[str]:
        nested_datasets = {
            ds
            for dset_member in dataset_description.members.get("Dataset", [])
            for ds in self.list_datasets(dset_member)
        }
        return {dataset_description.dataset.dataset_rid} | nested_datasets

    def compare_catalogs(self, ml_instance: DerivaML, dataset: MLDatasetCatalog, dataset_spec: DatasetSpec):
        reference_datasets = self.list_datasets(dataset.dataset_description)
        snapshot_catalog = dataset.dataset_description.dataset._version_snapshot_catalog(dataset_spec.version)
        bag = ml_instance.download_dataset_bag(dataset_spec)
        db_catalog = DerivaMLDatabase(bag.model)

        pb = snapshot_catalog.pathBuilder()
        ds = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset"]
        subject = pb.schemas[snapshot_catalog.default_schema].tables["Subject"]
        image = pb.schemas[snapshot_catalog.default_schema].tables["Image"]
        ds_ds = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset_Dataset"]
        ds_subject = pb.schemas[snapshot_catalog.default_schema].tables["Dataset_Subject"]
        ds_image = pb.schemas[snapshot_catalog.default_schema].tables["Dataset_Image"]

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}
        for dataset_rid in reference_datasets:
            dataset_bag = db_catalog.lookup_dataset(dataset_rid)
            dset = ml_instance.lookup_dataset(dataset_rid)
            dataset_rids = tuple([ds.dataset_rid for ds in dset.list_dataset_children(recurse=True)] + [dataset_rid])
            bag_rids = [b.dataset_rid for b in dataset_bag.list_dataset_children(recurse=True)] + [
                dataset_bag.dataset_rid
            ]
            assert list(dataset_rids).sort() == bag_rids.sort()

            ds_path = ds.path.link(ds_ds).filter(ds_ds.Dataset == Any(*dataset_rids)).link(ds)
            subject_path = ds.path.link(ds_subject).filter(ds_subject.Dataset == Any(*dataset_rids)).link(subject)
            image_path = ds.path.link(ds_image).filter(ds_image.Dataset == Any(*dataset_rids)).link(image)
            dataset_path_1 = ds.path.filter(ds.RID == Any(*dataset_rids))
            datasets = list(ds_path.entities().fetch()) + list(dataset_path_1.entities().fetch())
            subjects = list(subject_path.entities().fetch())
            images = list(image_path.entities().fetch())

            # Get RIDs from catalog results
            catalog_dataset_rids = {d["RID"] for d in datasets}
            catalog_subject_rids = {s["RID"] for s in subjects}
            catalog_image_rids = {i["RID"] for i in images}

            catalog_dataset = set([self.make_frozen(d) for d in datasets])
            catalog_subject = set([self.make_frozen(s) for s in subjects])
            catalog_image = set([self.make_frozen(i) for i in images])

            # Get bag data filtered to RIDs that appear in the catalog results
            # This verifies the bag contains the same data for those specific records
            all_bag_datasets = list(db_catalog.get_table_as_dict("Dataset"))
            all_bag_subjects = list(db_catalog.get_table_as_dict("Subject"))
            all_bag_images = list(db_catalog.get_table_as_dict("Image"))
            dataset_table = set([self.make_frozen(d) for d in all_bag_datasets if d["RID"] in catalog_dataset_rids])
            subject_table = set([self.make_frozen(s) for s in all_bag_subjects if s["RID"] in catalog_subject_rids])
            image_table = set([self.make_frozen(i) for i in all_bag_images if i["RID"] in catalog_image_rids])

            assert len(catalog_dataset) == len(dataset_table)
            assert len(catalog_subject) == len(subject_table)
            assert len(catalog_image) == len(image_table)

            assert catalog_dataset == dataset_table
            assert catalog_subject == subject_table
            assert catalog_image == image_table

    def test_database_methods(self, dataset_test):
        ml_instance = DerivaML(dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, use_minid=False)
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        current_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        current_bag = ml_instance.download_dataset_bag(current_spec)
        tables = current_bag.model.list_tables()
        schemas = ml_instance.model.schemas
        catalog_tables = len(schemas[ml_instance.default_schema].tables) + len(schemas[ml_instance.ml_schema].tables)
        assert catalog_tables == len(tables)

    def test_table_as_dict(self, dataset_test):
        ml_instance = DerivaML(dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, use_minid=False)
        dataset_description = dataset_test.dataset_description

        current_version = dataset_description.dataset.current_version
        dataset_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        self.compare_catalogs(ml_instance, dataset_test, dataset_spec)

    def test_table_versions(self, dataset_test):
        ml_instance = DerivaML(dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, use_minid=False)
        dataset_description = dataset_test.dataset_description

        current_version = dataset_description.dataset.current_version
        current_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        self.compare_catalogs(
            ml_instance, dataset_test, DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=current_version)
        )

        pb = ml_instance.pathBuilder()
        subject = pb.schemas[ml_instance.default_schema].Subject
        new_subjects = [s["RID"] for s in subject.insert([{"Name": f"Mew Thing{t + 1}"} for t in range(2)])]
        dataset_description.dataset.add_dataset_members(new_subjects)
        print("Adding subjects: ", new_subjects)
        new_version = dataset_description.dataset.current_version
        new_spec = DatasetSpec(rid=dataset_description.dataset.dataset_rid, version=new_version)
        current_bag = ml_instance.download_dataset_bag(current_spec)
        new_bag = ml_instance.download_dataset_bag(new_spec)
        current_db_catalog = DerivaMLDatabase(current_bag.model)
        new_db_catalog = DerivaMLDatabase(new_bag.model)
        subjects_current = list(current_db_catalog.get_table_as_dict("Subject"))
        subjects_new = list(new_db_catalog.get_table_as_dict("Subject"))

        # Make sure that there is a difference between to old and new catalogs.
        print(new_subjects)
        print([s["RID"] for s in subjects_current])
        print([s["RID"] for s in subjects_new])
        assert len(subjects_new) == len(subjects_current) + 2
        print("compare")
        self.compare_catalogs(ml_instance, dataset_test, current_spec)
        self.compare_catalogs(ml_instance, dataset_test, new_spec)

    def test_fk_traversal_with_explicit_images(self, dataset_test):
        """Test that bag includes FK-reachable images from Subject members plus explicit Image members.

        Creates a dataset with:
        - 2 Subject members (each has 1 Image via Image.Subject FK)
        - 2 explicit Image members: one already FK-reachable from a Subject member,
          and one whose Subject is NOT in the dataset

        The bag should contain:
        - 2 subjects (explicit members)
        - 3 images: 2 FK-reachable from subjects + 1 extra explicit image
          (the other explicit image overlaps with an FK-reachable one)
        """
        ml_instance = DerivaML(dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, use_minid=False)
        pb = ml_instance.pathBuilder()
        domain = ml_instance.default_schema

        # Get all subjects and images from catalog
        all_subjects = list(pb.schemas[domain].Subject.entities().fetch())
        all_images = list(pb.schemas[domain].Image.entities().fetch())
        assert len(all_subjects) >= 4, f"Need at least 4 subjects, got {len(all_subjects)}"
        assert len(all_images) >= 4, f"Need at least 4 images, got {len(all_images)}"

        # Pick 2 subjects to be dataset members
        subject_a = all_subjects[0]
        subject_b = all_subjects[1]
        subject_member_rids = [subject_a["RID"], subject_b["RID"]]

        # Find images linked to these subjects (FK-reachable)
        image_for_a = [i for i in all_images if i["Subject"] == subject_a["RID"]]
        image_for_b = [i for i in all_images if i["Subject"] == subject_b["RID"]]
        assert image_for_a, f"Subject {subject_a['RID']} has no images"
        assert image_for_b, f"Subject {subject_b['RID']} has no images"

        # Pick a subject NOT in the dataset and find its image
        subject_c = all_subjects[2]
        assert subject_c["RID"] not in subject_member_rids
        image_for_c = [i for i in all_images if i["Subject"] == subject_c["RID"]]
        assert image_for_c, f"Subject {subject_c['RID']} has no images"

        # Explicit Image members:
        # 1. image_for_a[0] — already FK-reachable from subject_a (overlap)
        # 2. image_for_c[0] — NOT FK-reachable (subject_c not in dataset)
        explicit_image_rids = [image_for_a[0]["RID"], image_for_c[0]["RID"]]

        # Create the dataset
        workflow = ml_instance.create_workflow(
            name="FK Traversal Test", workflow_type="Test Workflow"
        )
        config = ExecutionConfiguration(workflow=workflow)
        with ml_instance.create_execution(config) as execution:
            dataset = execution.create_dataset(
                dataset_types=["Complete"],
                description="Test FK traversal with explicit images",
            )
            dataset.add_dataset_members(
                {"Subject": subject_member_rids, "Image": explicit_image_rids},
                description="Subjects + mixed explicit images",
            )

        # Download the bag
        version = str(dataset.current_version)
        bag_spec = DatasetSpec(rid=dataset.dataset_rid, version=version)
        bag = ml_instance.download_dataset_bag(bag_spec)
        db = DerivaMLDatabase(bag.model)

        # Verify subjects in bag
        bag_subjects = list(db.get_table_as_dict("Subject"))
        bag_subject_rids = {s["RID"] for s in bag_subjects}
        assert bag_subject_rids >= set(subject_member_rids), (
            f"Bag missing subject members: {set(subject_member_rids) - bag_subject_rids}"
        )

        # Verify images in bag
        bag_images = list(db.get_table_as_dict("Image"))
        bag_image_rids = {i["RID"] for i in bag_images}

        # Expected images: FK-reachable from subject_a and subject_b, plus explicit image_for_c
        expected_image_rids = {image_for_a[0]["RID"], image_for_b[0]["RID"], image_for_c[0]["RID"]}
        assert bag_image_rids >= expected_image_rids, (
            f"Bag missing expected images: {expected_image_rids - bag_image_rids}"
        )

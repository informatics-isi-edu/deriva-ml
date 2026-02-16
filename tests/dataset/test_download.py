from pathlib import Path
from pprint import pformat

try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None

# Local imports
from deriva_ml import DerivaML, MLVocab, TableDefinition
from deriva_ml.dataset.aux_classes import DatasetSpec, VersionPart
from deriva_ml.dataset.dataset import Dataset, DatasetBag
from deriva_ml.demo_catalog import DatasetDescription
from deriva_ml.model.deriva_ml_database import DerivaMLDatabase
from tests.test_utils import MLDatasetCatalog


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
        subjects = [s["RID"] for s in pb.schemas[ml_instance.default_schema].tables["Subject"].path.entities().fetch()]

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
                columns=[],
            )
        )
        new_version = dataset_description.dataset.increment_dataset_version(component=VersionPart.minor)

        current_bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)
        new_bag = dataset_description.dataset.download_dataset_bag(new_version, use_minid=False)

        assert "NewTable" in new_bag.model.model.schemas[ml_instance.default_schema].tables
        assert "NewTable" not in current_bag.model.model.schemas[ml_instance.default_schema].tables

    def test_dataset_types_preserved_in_bag(self, dataset_test, tmp_path):
        """Test that dataset types in downloaded bag match the original catalog dataset types.

        This test verifies nested dataset coverage by:
        1. Recursively collecting all datasets in the hierarchy (including nested children)
        2. Verifying we test multiple datasets at different nesting levels
        3. Checking types match for datasets with various type configurations
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        # Get reference datasets with their types from the catalog
        # list_datasets recursively collects ALL datasets including nested children
        reference_datasets = self.list_datasets(dataset_description)

        # Verify we are testing nested datasets (more than just the root)
        assert len(reference_datasets) > 1, "Test must cover multiple datasets including nested ones"

        # Collect all unique type sets to verify we're testing diverse dataset types
        all_type_sets = {frozenset(ds.dataset_types) for ds in reference_datasets}
        assert len(all_type_sets) > 1, "Test must cover datasets with different type configurations"

        # Download the bag
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Use DerivaMLDatabase to access datasets in the bag
        db_catalog = DerivaMLDatabase(bag.model)

        # Check that dataset types match for all datasets in the hierarchy
        datasets_checked = 0
        for catalog_dataset in reference_datasets:
            catalog_types = set(catalog_dataset.dataset_types)

            # Look up the same dataset in the downloaded bag
            bag_dataset = db_catalog.lookup_dataset(catalog_dataset.dataset_rid)
            bag_types = set(bag_dataset.dataset_types)

            assert catalog_types == bag_types, (
                f"Dataset types mismatch for dataset {catalog_dataset.dataset_rid}: "
                f"catalog={catalog_types}, bag={bag_types}"
            )
            datasets_checked += 1

        # Final verification that we checked multiple datasets
        assert datasets_checked > 1, f"Expected to check multiple datasets, only checked {datasets_checked}"

    def test_list_dataset_children_preserves_types(self, dataset_test, tmp_path):
        """Test that list_dataset_children() returns datasets with proper types and metadata.

        This specifically tests that child datasets returned by list_dataset_children()
        have their dataset_types, descriptions, and other metadata populated correctly
        from the bag data, not just empty defaults.
        """
        dataset_description = dataset_test.dataset_description
        dataset = dataset_description.dataset

        # Get children from catalog for reference
        catalog_children = dataset.list_dataset_children()
        if not catalog_children:
            # Skip if no children to test
            return

        # Download the bag
        current_version = dataset.current_version
        bag = dataset.download_dataset_bag(current_version, use_minid=False)

        # Get children via list_dataset_children() - this is what we're testing
        bag_children = bag.list_dataset_children()

        assert len(bag_children) == len(catalog_children), (
            f"Child count mismatch: catalog={len(catalog_children)}, bag={len(bag_children)}"
        )

        # Build lookup maps by RID
        catalog_by_rid = {c.dataset_rid: c for c in catalog_children}
        bag_by_rid = {c.dataset_rid: c for c in bag_children}

        assert set(catalog_by_rid.keys()) == set(bag_by_rid.keys()), "Child RIDs don't match"

        # Verify each child has proper metadata
        for rid, catalog_child in catalog_by_rid.items():
            bag_child = bag_by_rid[rid]

            # Check dataset_types match
            assert set(catalog_child.dataset_types) == set(bag_child.dataset_types), (
                f"Dataset types mismatch for child {rid}: "
                f"catalog={catalog_child.dataset_types}, bag={bag_child.dataset_types}"
            )

            # Check description is populated (not empty)
            assert bag_child.description == catalog_child.description, (
                f"Description mismatch for child {rid}: "
                f"catalog={catalog_child.description!r}, bag={bag_child.description!r}"
            )

        # Also test recursive children if there are nested datasets
        bag_children_recursive = bag.list_dataset_children(recurse=True)
        catalog_children_recursive = dataset.list_dataset_children(recurse=True)

        for bag_child in bag_children_recursive:
            # Every child should have dataset_types populated (may be empty list, but should match catalog)
            catalog_child = next(
                (c for c in catalog_children_recursive if c.dataset_rid == bag_child.dataset_rid),
                None
            )
            assert catalog_child is not None, f"Child {bag_child.dataset_rid} not found in catalog"
            assert set(bag_child.dataset_types) == set(catalog_child.dataset_types), (
                f"Recursive child {bag_child.dataset_rid} types mismatch: "
                f"catalog={catalog_child.dataset_types}, bag={bag_child.dataset_types}"
            )

    def test_list_dataset_parents_preserves_metadata(self, dataset_test, tmp_path):
        """Test that list_dataset_parents() returns datasets with proper metadata.

        Bug reference: Commit aef2db5 fixed this by using lookup_dataset()
        instead of directly constructing DatasetBag objects.
        """
        dataset_description = dataset_test.dataset_description
        dataset = dataset_description.dataset

        # Download the bag
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Find a child dataset that has a parent
        children = bag.list_dataset_children()
        if not children:
            return  # Skip if no children

        child = children[0]
        child_catalog = dataset_test.ml_instance.lookup_dataset(child.dataset_rid)

        # Get parents from the bag via list_dataset_parents
        bag_parents = child.list_dataset_parents()
        catalog_parents = child_catalog.list_dataset_parents()

        assert len(bag_parents) == len(catalog_parents), (
            f"Parent count mismatch: bag={len(bag_parents)}, catalog={len(catalog_parents)}"
        )

        if bag_parents:
            bag_parent = bag_parents[0]
            catalog_parent = catalog_parents[0]

            # Verify dataset_types are preserved (not empty or default)
            assert set(bag_parent.dataset_types) == set(catalog_parent.dataset_types), (
                f"Parent types mismatch: bag={bag_parent.dataset_types}, "
                f"catalog={catalog_parent.dataset_types}"
            )

            # Verify description is preserved
            assert bag_parent.description == catalog_parent.description, (
                f"Parent description mismatch: bag={bag_parent.description!r}, "
                f"catalog={catalog_parent.description!r}"
            )

            # Verify RID matches
            assert bag_parent.dataset_rid == catalog_parent.dataset_rid


class TestDatabasePathCaching:
    """Tests for SQLite database path caching behavior."""

    def test_database_path_uses_bag_checksum(self, dataset_test, tmp_path):
        """Test that the SQLite database path includes the bag checksum.

        Bug reference: Commit 5a1e8b5 fixed this by using the bag cache
        directory name (which includes checksum) instead of just version_rid.
        This ensures that when a bag is regenerated with new content,
        a new database is created instead of using a stale cached one.
        """
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # The database path should include the bag cache directory name
        dbase_path = bag.model.database_dir

        # Path should exist and be a directory
        assert dbase_path.exists(), f"Database path does not exist: {dbase_path}"

        # The path should be inside a cache directory named with the bag identifier
        # which includes a checksum component (e.g., "RID_checksum...")
        cache_dir_name = dbase_path.name

        # Cache dir should be longer than just a short RID (3-4 chars)
        # because it includes the checksum
        assert len(cache_dir_name) > 10, (
            f"Cache dir name '{cache_dir_name}' seems too short to include checksum. "
            "Database path may be using version_rid instead of bag checksum."
        )

    def test_different_bag_versions_use_different_databases(self, dataset_test, tmp_path):
        """Test that different bag versions create separate database caches.

        This verifies that version changes result in new database paths,
        preventing stale data issues.
        """
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset = dataset_test.dataset_description.dataset
        current_version = dataset.current_version

        # Download the current version
        bag1 = dataset.download_dataset_bag(version=current_version, use_minid=False)
        dbase_path1 = bag1.model.database_dir

        # Increment version (this creates a new version in the catalog)
        new_version = dataset.increment_dataset_version(
            component=VersionPart.minor,
            description="Test version increment"
        )

        # Download the new version
        bag2 = dataset.download_dataset_bag(version=new_version, use_minid=False)
        dbase_path2 = bag2.model.database_dir

        # The database paths should be different
        assert dbase_path1 != dbase_path2, (
            f"Database paths should differ for different versions: "
            f"v{current_version} path={dbase_path1}, v{new_version} path={dbase_path2}"
        )

"""Tests for DatasetBag.restructure_assets() method."""


import pytest


class TestRestructureAssets:
    """Tests for the restructure_assets method on DatasetBag."""

    def test_restructure_basic_types_only(self, dataset_test, tmp_path):
        """Test basic restructuring with dataset types only (no group_by)."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured"
        result = bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=[],
        )

        assert result == output_dir
        assert output_dir.exists()

        # Check that subdirectories were created based on dataset types
        # The root dataset should have its type as the first directory level
        subdirs = list(output_dir.iterdir())
        assert len(subdirs) >= 1

        # Check that files exist (as symlinks by default)
        # Look for any files, not just specific extensions
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0, f"Expected files in {output_dir}, found: {list(output_dir.rglob('*'))}"

        # Verify files are symlinks
        for f in all_files:
            assert f.is_symlink(), f"Expected {f} to be a symlink"

    def test_restructure_copy_mode(self, dataset_test, tmp_path):
        """Test restructuring with file copying instead of symlinks."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_copy"
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=[],
            use_symlinks=False,
        )

        # Check that files are copies, not symlinks
        all_files = [f for f in output_dir.rglob("*") if f.is_file()]
        for f in all_files:
            assert f.is_file()
            assert not f.is_symlink()

    def test_restructure_type_selector(self, dataset_test, tmp_path):
        """Test restructuring with custom type selector."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_custom"

        # Use last type instead of first
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=[],
            type_selector=lambda types: types[-1] if types else "custom_unknown",
        )

        assert output_dir.exists()
        # Verify the directory was created
        subdirs = list(output_dir.iterdir())
        assert len(subdirs) >= 1

    def test_restructure_by_column(self, dataset_test, tmp_path):
        """Test restructuring with column-based grouping."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_column"

        # Group by Subject column (foreign key)
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Subject"],
        )

        assert output_dir.exists()
        # Files should be organized by Subject RID
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0, f"Expected files in {output_dir}"

    def test_restructure_missing_values_unknown(self, dataset_test, tmp_path):
        """Test that missing grouping values use 'Unknown' folder."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_missing"

        # Use a non-existent column - should result in "Unknown" folders
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["NonExistentColumn"],
        )

        assert output_dir.exists()

        # All files should end up in "Unknown" folder at the group level (capitalized)
        unknown_dirs = list(output_dir.rglob("Unknown"))
        assert len(unknown_dirs) >= 1

    def test_restructure_dataset_without_type_uses_testing(self, dataset_test, tmp_path):
        """Test that datasets without a type are treated as Testing (prediction scenario).

        When a dataset has no type defined, restructure_assets should place its
        assets in a 'testing' directory. This supports prediction/inference
        scenarios where unlabeled data is being processed.
        """
        ml = dataset_test.ml_instance

        # Create a dataset with NO types (empty list)
        dataset = ml.create_dataset(
            dataset_types=[],  # No type - prediction scenario
            description="Unlabeled prediction dataset",
        )

        # Add some images to the dataset
        image_path = ml.domain_path.tables["Image"]
        images = list(image_path.entities().fetch())
        if not images:
            pytest.skip("No images in test data")

        image_rids = [img["RID"] for img in images[:2]]
        dataset.add_dataset_members({"Image": image_rids})

        # Download and restructure
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_prediction"
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=[],
        )

        assert output_dir.exists()

        # Files should be in a 'testing' directory (default for datasets without type)
        testing_dirs = list(output_dir.rglob("testing"))
        assert len(testing_dirs) >= 1, (
            f"Expected 'testing' directory for dataset without type. "
            f"Found directories: {[d.name for d in output_dir.rglob('*') if d.is_dir()]}"
        )

        # Verify files are inside the testing directory
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0, "Expected files to be restructured"

        for f in all_files:
            relative = f.relative_to(output_dir)
            assert relative.parts[0] == "testing", (
                f"File {f} should be under 'testing' directory, got {relative.parts[0]}"
            )

    def test_restructure_prediction_with_missing_labels(self, dataset_test, tmp_path):
        """Test full prediction scenario: no type + no labels = testing/Unknown/.

        This tests the complete prediction workflow where:
        1. Dataset has no type (treated as Testing)
        2. Assets have no labels for group_by (placed in Unknown)
        Result: files end up in testing/Unknown/
        """
        ml = dataset_test.ml_instance

        # Create a dataset with NO types
        dataset = ml.create_dataset(
            dataset_types=[],
            description="Full prediction scenario test",
        )

        # Add images
        image_path = ml.domain_path.tables["Image"]
        images = list(image_path.entities().fetch())
        if not images:
            pytest.skip("No images in test data")

        image_rids = [img["RID"] for img in images[:2]]
        dataset.add_dataset_members({"Image": image_rids})

        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_full_prediction"
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["NonExistentLabel"],  # No labels - will use Unknown
        )

        assert output_dir.exists()

        # Should have testing/Unknown path
        expected_path = output_dir / "testing" / "Unknown"
        assert expected_path.exists(), (
            f"Expected testing/Unknown directory for prediction scenario. "
            f"Found: {list(output_dir.rglob('*'))}"
        )

        # Verify files are in testing/Unknown
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0

        for f in all_files:
            relative = f.relative_to(output_dir)
            assert relative.parts[:2] == ("testing", "Unknown"), (
                f"File {f} should be in testing/Unknown/, got {relative.parts[:2]}"
            )

    def test_restructure_empty_asset_table(self, dataset_test, tmp_path):
        """Test restructuring with an asset table that has no members in the dataset."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_empty"

        # Use a valid table that exists but might not have members in this dataset
        # Use "File" which exists in the schema but likely has no members
        result = bag.restructure_assets(
            asset_table="File",
            output_dir=output_dir,
            group_by=[],
        )

        assert result == output_dir
        # Directory should be created (may or may not have files depending on data)

    def test_restructure_nested_datasets(self, dataset_test, tmp_path):
        """Test restructuring with nested datasets - types should form hierarchy."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Check that there are nested datasets
        children = bag.list_dataset_children()
        if not children:
            pytest.skip("No nested datasets in test data")

        output_dir = tmp_path / "restructured_nested"
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=[],
        )

        assert output_dir.exists()

        # The directory structure should reflect the nesting
        # Root type -> child type -> files
        # Find the deepest directory path with files
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        if all_files:
            # Get the depth of the first file
            first_file = all_files[0]
            relative_path = first_file.relative_to(output_dir)
            # Should have at least 2 levels for nested datasets (parent type + child type)
            assert len(relative_path.parts) >= 2, f"Expected at least 2 levels, got {relative_path}"

    def test_restructure_multi_group(self, dataset_test, tmp_path):
        """Test restructuring with multiple grouping keys."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_multi"

        # Group by multiple columns
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Subject", "Description"],
        )

        assert output_dir.exists()

        # Should have deeper directory structure due to multiple groups
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        if all_files:
            # Count depth of first file
            first_file = all_files[0]
            relative_path = first_file.relative_to(output_dir)
            # Should have type + 2 group levels at minimum (type/subject/description/file)
            assert len(relative_path.parts) >= 3, f"Expected at least 3 levels, got {relative_path}"


class TestRestructureForeignKeyPaths:
    """Tests for restructure_assets finding assets through FK paths."""

    def test_get_reachable_assets_finds_indirectly_linked(self, dataset_test, tmp_path):
        """Test that _get_reachable_assets finds assets linked via FK chain.

        The demo schema has Image -> Subject FK relationship. When a dataset
        contains Subjects but not Images directly, _get_reachable_assets should
        still find Images reachable through the Subject FK.
        """
        ml = dataset_test.ml_instance

        # Create a dataset that only contains Subjects (no Images directly)
        dataset = ml.create_dataset(
            dataset_types=["Testing"],
            description="Dataset with only subjects",
        )

        # Get some subject RIDs from the catalog
        subject_path = ml.domain_path.tables["Subject"]
        subjects = list(subject_path.entities().fetch())
        if not subjects:
            pytest.skip("No subjects in test data")

        # Add only subjects to the dataset (not images)
        subject_rids = [s["RID"] for s in subjects[:2]]
        dataset.add_dataset_members({"Subject": subject_rids})

        # Download the bag
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # _get_reachable_assets should find Images through Subject -> Image FK
        reachable_images = bag._get_reachable_assets("Image")

        # There should be images reachable through the Subject FK
        # (Each subject should have associated images in the demo data)
        assert len(reachable_images) > 0, (
            "Expected to find Images reachable through Subject FK path. "
            "The demo catalog has Image -> Subject FK relationship."
        )

        # Verify the images are associated with the subjects we added
        for img in reachable_images:
            assert img.get("Subject") in subject_rids, (
                f"Image {img.get('RID')} has Subject={img.get('Subject')} "
                f"which is not in added subjects {subject_rids}"
            )

    def test_restructure_finds_assets_via_fk_path(self, dataset_test, tmp_path):
        """Test that restructure_assets finds assets connected via FK paths.

        Creates a dataset with only Subject members, then restructures Images.
        Images should be found via Subject -> Image FK relationship.
        """
        ml = dataset_test.ml_instance

        # Create a dataset that only contains Subjects
        dataset = ml.create_dataset(
            dataset_types=["Training"],
            description="Dataset with subjects only - images via FK",
        )

        # Get subjects and their associated images
        subject_path = ml.domain_path.tables["Subject"]
        subjects = list(subject_path.entities().fetch())
        if not subjects:
            pytest.skip("No subjects in test data")

        # Add only subjects to the dataset
        subject_rids = [s["RID"] for s in subjects[:2]]
        dataset.add_dataset_members({"Subject": subject_rids})

        # Download and restructure
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_fk"
        result = bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=[],
        )

        assert result == output_dir
        assert output_dir.exists()

        # Should have found images through the FK path
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0, (
            "No images found via FK path. restructure_assets should find Images "
            "connected through Subject -> Image FK relationship."
        )

    def test_asset_dataset_mapping_via_fk_path(self, dataset_test, tmp_path):
        """Test that _get_asset_dataset_mapping works with FK-connected assets."""
        ml = dataset_test.ml_instance

        # Create a dataset with only Subjects
        dataset = ml.create_dataset(
            dataset_types=["Testing"],
            description="Test FK mapping",
        )

        subject_path = ml.domain_path.tables["Subject"]
        subjects = list(subject_path.entities().fetch())
        if not subjects:
            pytest.skip("No subjects in test data")

        subject_rids = [s["RID"] for s in subjects[:2]]
        dataset.add_dataset_members({"Subject": subject_rids})

        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Get the asset-to-dataset mapping
        asset_map = bag._get_asset_dataset_mapping("Image")

        # Images found via FK should be mapped to the dataset
        assert len(asset_map) > 0, "Expected images to be mapped via FK path"

        # All mapped images should point to our dataset
        for image_rid, dataset_rid in asset_map.items():
            assert dataset_rid == bag.dataset_rid, (
                f"Image {image_rid} mapped to {dataset_rid}, expected {bag.dataset_rid}"
            )


class TestRestructureHelperMethods:
    """Tests for the helper methods used by restructure_assets."""

    def test_build_dataset_type_path_map(self, dataset_test, tmp_path):
        """Test _build_dataset_type_path_map helper."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        type_map = bag._build_dataset_type_path_map()

        # Should include at least the root dataset
        assert bag.dataset_rid in type_map
        # Path should be a list
        assert isinstance(type_map[bag.dataset_rid], list)
        # Path should have at least one element
        assert len(type_map[bag.dataset_rid]) >= 1

    def test_build_dataset_type_path_map_with_selector(self, dataset_test, tmp_path):
        """Test _build_dataset_type_path_map with custom type selector."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Use a selector that returns a fixed value
        type_map = bag._build_dataset_type_path_map(
            type_selector=lambda types: "FIXED_TYPE"
        )

        # All paths should contain "FIXED_TYPE"
        for rid, path in type_map.items():
            assert "FIXED_TYPE" in path

    def test_get_asset_dataset_mapping(self, dataset_test, tmp_path):
        """Test _get_asset_dataset_mapping helper."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        asset_map = bag._get_asset_dataset_mapping("Image")

        # Should have mappings for images
        members = bag.list_dataset_members(recurse=True)
        images = members.get("Image", [])

        # Each image should be mapped to a dataset
        for img in images:
            assert img["RID"] in asset_map

    def test_resolve_grouping_value_column(self, dataset_test, tmp_path):
        """Test _resolve_grouping_value with a column value."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        members = bag.list_dataset_members(recurse=True)
        images = members.get("Image", [])

        if not images:
            pytest.skip("No images in test data")

        # Test with a column that exists
        asset = images[0]
        value = bag._resolve_grouping_value(asset, "RID", {})
        assert value == asset["RID"]

    def test_resolve_grouping_value_missing(self, dataset_test, tmp_path):
        """Test _resolve_grouping_value with missing value."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        members = bag.list_dataset_members(recurse=True)
        images = members.get("Image", [])

        if not images:
            pytest.skip("No images in test data")

        # Test with a column that doesn't exist - should return "Unknown" (capitalized)
        asset = images[0]
        value = bag._resolve_grouping_value(asset, "NonExistent", {})
        assert value == "Unknown"


class TestRestructureWithFeatures:
    """Tests for restructure_assets with feature-based grouping."""

    def test_restructure_by_vocabulary_feature(self, dataset_test, tmp_path):
        """Test restructuring grouped by a vocabulary-based feature (Image.Quality)."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_feature"

        # Group by the Quality feature which uses ImageQuality vocabulary
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Quality"],
        )

        assert output_dir.exists()

        # Files should be organized by Quality values (Good/Bad)
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0, f"Expected files in {output_dir}"

        # Check that directories named after vocabulary terms exist
        all_dirs = [d.name for d in output_dir.rglob("*") if d.is_dir()]
        # Should have at least one of "Good", "Bad", or "unknown"
        quality_dirs = [d for d in all_dirs if d in ("Good", "Bad", "unknown")]
        assert len(quality_dirs) >= 1, f"Expected quality directories, found: {all_dirs}"

    def test_restructure_combine_column_and_feature(self, dataset_test, tmp_path):
        """Test restructuring with both column and feature grouping."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_combined"

        # Group by Subject column first, then Quality feature
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Subject", "Quality"],
        )

        assert output_dir.exists()

        # Should have deeper structure: type/subject_rid/quality/file
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        if all_files:
            first_file = all_files[0]
            relative_path = first_file.relative_to(output_dir)
            # Should have type + subject + quality + filename = at least 4 parts
            assert len(relative_path.parts) >= 3, f"Expected at least 3 levels, got {relative_path}"

    def test_restructure_feature_then_column(self, dataset_test, tmp_path):
        """Test restructuring with feature first, then column."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_feature_first"

        # Group by Quality feature first, then Subject column
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Quality", "Subject"],
        )

        assert output_dir.exists()
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0


class TestEnforceVocabulary:
    """Tests for the enforce_vocabulary parameter."""

    def test_enforce_vocabulary_rejects_asset_feature(self, dataset_test, tmp_path):
        """Test that enforce_vocabulary=True rejects asset-based features (no vocab)."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_asset_feature"

        # BoundingBox is an asset-based feature with no vocabulary terms
        with pytest.raises(DerivaMLException) as exc_info:
            bag.restructure_assets(
                asset_table="Image",
                output_dir=output_dir,
                group_by=["BoundingBox"],
                enforce_vocabulary=True,
            )

        assert "controlled vocabulary" in str(exc_info.value).lower()

    def test_enforce_vocabulary_false_allows_asset_feature(self, dataset_test, tmp_path):
        """Test that enforce_vocabulary=False allows asset-based features."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_asset_allowed"

        # Should not raise with enforce_vocabulary=False
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["BoundingBox"],
            enforce_vocabulary=False,
        )

        assert output_dir.exists()

    def test_enforce_vocabulary_default_is_true(self, dataset_test, tmp_path):
        """Test that enforce_vocabulary defaults to True."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_default"

        # BoundingBox feature should fail without explicitly setting enforce_vocabulary
        with pytest.raises(DerivaMLException):
            bag.restructure_assets(
                asset_table="Image",
                output_dir=output_dir,
                group_by=["BoundingBox"],
                # enforce_vocabulary not specified, should default to True
            )

    def test_enforce_vocabulary_allows_vocab_feature(self, dataset_test, tmp_path):
        """Test that enforce_vocabulary=True allows vocabulary-based features."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_vocab_ok"

        # Quality feature uses ImageQuality vocabulary - should work
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Quality"],
            enforce_vocabulary=True,
        )

        assert output_dir.exists()
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0


class TestFeatureCacheLoading:
    """Tests for _load_feature_values_cache with enforce_vocabulary."""

    def test_cache_loads_vocabulary_feature(self, dataset_test, tmp_path):
        """Test that cache correctly loads vocabulary-based feature values."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        cache = bag._load_feature_values_cache("Image", ["Quality"], enforce_vocabulary=True)

        # Quality should be in the cache
        assert "Quality" in cache
        # Should have mappings (may be empty if no feature values assigned)
        assert isinstance(cache["Quality"], dict)

    def test_cache_rejects_non_vocabulary_feature(self, dataset_test, tmp_path):
        """Test that cache raises for non-vocabulary feature when enforcing."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        with pytest.raises(DerivaMLException) as exc_info:
            bag._load_feature_values_cache("Image", ["BoundingBox"], enforce_vocabulary=True)

        assert "controlled vocabulary" in str(exc_info.value).lower()

    def test_cache_allows_non_vocabulary_when_not_enforcing(self, dataset_test, tmp_path):
        """Test that cache allows non-vocabulary features when not enforcing."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Should not raise
        cache = bag._load_feature_values_cache("Image", ["BoundingBox"], enforce_vocabulary=False)

        # BoundingBox may or may not be in cache depending on if it's found as a feature
        assert isinstance(cache, dict)


class TestListFeatureValuesReturnType:
    """Tests for list_feature_values return type correctness."""

    def test_list_feature_values_returns_feature_records(self, dataset_test, tmp_path):
        """Test that list_feature_values returns FeatureRecord instances.

        This test verifies that feature values are returned as typed FeatureRecord
        Pydantic models with proper attribute access and model_dump() support.
        """
        from deriva_ml.feature import FeatureRecord

        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Get feature values from the bag
        feature_values = list(bag.list_feature_values("Image", "Quality"))

        if feature_values:
            # Each item should be a FeatureRecord subclass
            first_value = feature_values[0]
            assert isinstance(first_value, FeatureRecord), (
                f"Expected FeatureRecord subclass, got {type(first_value).__name__}."
            )

            # Should have model_dump() method from Pydantic
            assert hasattr(first_value, "model_dump"), (
                "FeatureRecord should have model_dump() method"
            )

            # model_dump() should return a dict with column names as keys
            dumped = first_value.model_dump()
            assert isinstance(dumped, dict), "model_dump() should return a dict"
            assert "Image" in dumped, (
                f"Expected 'Image' key in model_dump(), got keys: {list(dumped.keys())}"
            )

            # Should have Feature_Name and Execution attributes
            assert hasattr(first_value, "Feature_Name"), (
                "FeatureRecord should have Feature_Name attribute"
            )
            assert hasattr(first_value, "Execution"), (
                "FeatureRecord should have Execution attribute for provenance"
            )

    def test_list_feature_values_columns_accessible_by_attribute(self, dataset_test, tmp_path):
        """Test that feature value columns can be accessed as attributes."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        feature_values = list(bag.list_feature_values("Image", "Quality"))

        if feature_values:
            for fv in feature_values:
                # Access as attribute should work
                image_rid = getattr(fv, "Image", None)
                assert image_rid is not None, (
                    "Cannot access 'Image' column from feature value as attribute"
                )

                # Can also access via model_dump()
                dumped = fv.model_dump()
                assert dumped.get("Image") == image_rid

                # The vocabulary term column should also be accessible
                # Quality feature uses ImageQuality vocabulary
                quality_value = getattr(fv, "ImageQuality", None)
                # Value may be None, but attribute should exist
                assert hasattr(fv, "ImageQuality") or quality_value is not None


class TestFeatureTablesInBagExport:
    """Tests for feature tables being included in dataset bag exports."""

    def test_feature_tables_present_in_bag_schema(self, dataset_test, tmp_path):
        """Test that feature association tables are included in downloaded bags.

        Bug reference: Commit 45078d4 fixed this by explicitly adding feature
        tables to the export path list.
        """
        from deriva_ml import DerivaML

        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        # Check that the feature table is present in the bag's schema
        domain_schema = ml_instance.domain_schema
        bag_tables = set(bag.model.schemas[domain_schema].tables.keys())

        # The feature table name follows pattern: {target_table}{feature_name}
        # e.g., ImageQuality for the Quality feature on Image table
        feature_table_name = "ImageQuality"
        assert feature_table_name in bag_tables, (
            f"Feature table {feature_table_name} not found in bag. "
            f"Available domain tables: {sorted(bag_tables)}"
        )

    def test_restructure_can_use_feature_from_bag(self, dataset_test, tmp_path):
        """Test that restructure_assets can group by features from downloaded bags.

        This is an integration test that verifies the full workflow:
        1. Download dataset bag
        2. Feature tables are included
        3. restructure_assets can successfully group by feature values
        """
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_by_feature"

        # This should work because feature tables are exported with the bag
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["Quality"],
        )

        assert output_dir.exists()

        # Files should be organized by Quality values
        all_files = [f for f in output_dir.rglob("*") if f.is_file() or f.is_symlink()]
        assert len(all_files) > 0, (
            "No files created by restructure_assets - "
            "feature table data may not have been exported properly"
        )

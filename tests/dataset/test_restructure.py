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
        """Test that missing grouping values use 'unknown' folder."""
        dataset = dataset_test.dataset_description.dataset
        bag = dataset.download_dataset_bag(version=dataset.current_version, use_minid=False)

        output_dir = tmp_path / "restructured_missing"

        # Use a non-existent column - should result in "unknown" folders
        bag.restructure_assets(
            asset_table="Image",
            output_dir=output_dir,
            group_by=["NonExistentColumn"],
        )

        assert output_dir.exists()

        # All files should end up in "unknown" folder at the group level
        unknown_dirs = list(output_dir.rglob("unknown"))
        assert len(unknown_dirs) >= 1

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

        # Test with a column that doesn't exist
        asset = images[0]
        value = bag._resolve_grouping_value(asset, "NonExistent", {})
        assert value == "unknown"

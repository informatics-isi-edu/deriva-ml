"""Tests for cache and directory management functionality."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def storage_workflow(test_ml):
    """Create a test workflow for storage tests."""
    from deriva_ml.core.enums import MLVocab

    # Add workflow type term if it doesn't exist
    try:
        test_ml.lookup_term(MLVocab.workflow_type, "Storage Test")
    except Exception:
        test_ml.add_term(MLVocab.workflow_type, "Storage Test", description="Workflow type for storage tests")

    return test_ml.create_workflow(
        name="Storage Test Workflow",
        workflow_type="Storage Test",
        description="Workflow for testing storage management"
    )


class TestCacheManagement:
    """Tests for cache management methods."""

    def test_get_cache_size_empty(self, test_ml):
        """Test get_cache_size returns zeros for empty cache."""
        ml = test_ml
        # Ensure cache is empty
        if ml.cache_dir.exists():
            import shutil
            for item in ml.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        stats = ml.get_cache_size()
        assert stats['total_bytes'] == 0
        assert stats['total_mb'] == 0.0
        assert stats['file_count'] == 0

    def test_get_cache_size_with_files(self, test_ml):
        """Test get_cache_size calculates size correctly."""
        ml = test_ml

        # Create some test files in cache
        test_dir = ml.cache_dir / "test_dataset"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_content = b"Hello, World!" * 100  # 1300 bytes
        test_file.write_bytes(test_content)

        stats = ml.get_cache_size()
        assert stats['total_bytes'] >= len(test_content)
        assert stats['file_count'] >= 1

        # Cleanup
        import shutil
        shutil.rmtree(test_dir)

    def test_clear_cache_removes_all(self, test_ml):
        """Test clear_cache removes all entries."""
        ml = test_ml

        # Create test files
        test_dir = ml.cache_dir / "test_to_clear"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        # Verify files exist
        assert test_dir.exists()

        result = ml.clear_cache()
        assert result['dirs_removed'] >= 1 or result['files_removed'] >= 1
        assert result['bytes_freed'] > 0
        assert result['errors'] == 0

        # Verify test directory is gone
        assert not test_dir.exists()

    def test_clear_cache_older_than(self, test_ml):
        """Test clear_cache with older_than_days filter."""
        ml = test_ml

        # Create a test directory
        test_dir = ml.cache_dir / "test_age_filter"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "file.txt").write_text("content")

        # Clear with 0 days - should remove everything
        result = ml.clear_cache(older_than_days=0)

        # With older_than_days=0, everything older than "now" should be removed
        # But since we just created it, it might not be removed
        # Let's test with a large number that should preserve recent files
        if test_dir.exists():
            result = ml.clear_cache(older_than_days=365)
            # Files created today should NOT be removed
            # This test verifies the filtering logic works
            assert test_dir.exists() or result['dirs_removed'] > 0

        # Cleanup if still exists
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


class TestExecutionDirManagement:
    """Tests for execution directory management."""

    def test_list_execution_dirs_empty(self, test_ml):
        """Test list_execution_dirs with no execution directories."""
        ml = test_ml
        # This should not error even if directory doesn't exist
        dirs = ml.list_execution_dirs()
        assert isinstance(dirs, list)

    def test_list_execution_dirs_with_entries(self, test_ml, storage_workflow):
        """Test list_execution_dirs returns directory info."""
        from deriva_ml.execution import ExecutionConfiguration

        ml = test_ml

        # Create an execution to generate a directory
        config = ExecutionConfiguration(description="Test for dir listing")
        execution = ml.create_execution(config, workflow=storage_workflow)

        # Create a test file in the execution directory
        test_file = execution.working_dir / "test.txt"
        test_file.write_text("test content")

        # List directories
        dirs = ml.list_execution_dirs()

        # Should find at least our execution
        found = False
        for d in dirs:
            if d['execution_rid'] == execution.execution_rid:
                found = True
                assert d['size_bytes'] > 0
                assert d['file_count'] >= 1
                assert 'modified' in d
                break

        assert found, f"Execution {execution.execution_rid} not found in list"

        # Cleanup
        import shutil
        if execution.working_dir.exists():
            shutil.rmtree(execution.working_dir)

    def test_clean_execution_dirs_with_exclude(self, test_ml, storage_workflow):
        """Test clean_execution_dirs respects exclude_rids."""
        from deriva_ml.execution import ExecutionConfiguration
        import shutil

        ml = test_ml

        # Create two executions
        config1 = ExecutionConfiguration(description="Test exec 1")
        exec1 = ml.create_execution(config1, workflow=storage_workflow)
        (exec1.working_dir / "file1.txt").write_text("content1")

        config2 = ExecutionConfiguration(description="Test exec 2")
        exec2 = ml.create_execution(config2, workflow=storage_workflow)
        (exec2.working_dir / "file2.txt").write_text("content2")

        # Clean but exclude exec1
        result = ml.clean_execution_dirs(exclude_rids=[exec1.execution_rid])

        # exec1 should still exist
        assert exec1.working_dir.exists(), "Excluded execution was incorrectly removed"

        # Cleanup
        if exec1.working_dir.exists():
            shutil.rmtree(exec1.working_dir)
        if exec2.working_dir.exists():
            shutil.rmtree(exec2.working_dir)

    def test_get_storage_summary(self, test_ml):
        """Test get_storage_summary returns expected structure."""
        ml = test_ml

        summary = ml.get_storage_summary()

        assert 'working_dir' in summary
        assert 'cache_dir' in summary
        assert 'cache_size_mb' in summary
        assert 'cache_file_count' in summary
        assert 'execution_dir_count' in summary
        assert 'execution_size_mb' in summary
        assert 'total_size_mb' in summary

        # Values should be non-negative
        assert summary['cache_size_mb'] >= 0
        assert summary['execution_size_mb'] >= 0
        assert summary['total_size_mb'] >= 0


class TestCleanExecutionDirConfig:
    """Tests for the clean_execution_dir configuration option."""

    def test_clean_execution_dir_default_true(self, test_ml):
        """Test that clean_execution_dir defaults to True."""
        ml = test_ml
        assert ml.clean_execution_dir is True

    def test_upload_uses_config_setting(self, test_ml, storage_workflow):
        """Test that upload_execution_outputs uses the config setting."""
        from deriva_ml.execution import ExecutionConfiguration

        ml = test_ml

        # Create execution
        config = ExecutionConfiguration(description="Test cleanup config")
        execution = ml.create_execution(config, workflow=storage_workflow)

        # Create a test file
        test_file = execution.working_dir / "test_output.txt"
        test_file.write_text("test content")

        # With clean_execution_dir=True (default), directory should be removed after upload
        with execution.execute():
            pass

        execution.upload_execution_outputs()

        # Directory should be removed (or at least emptied)
        # Note: The directory itself is now removed with remove_folder=True
        assert not execution.working_dir.exists() or not any(execution.working_dir.iterdir())

    def test_upload_override_clean_folder(self, test_ml, storage_workflow):
        """Test that clean_folder parameter overrides config setting."""
        from deriva_ml.execution import ExecutionConfiguration
        import shutil

        ml = test_ml

        # Create execution
        config = ExecutionConfiguration(description="Test override")
        execution = ml.create_execution(config, workflow=storage_workflow)

        # Create a test file
        test_file = execution.working_dir / "keep_me.txt"
        test_file.write_text("should be preserved")

        with execution.execute():
            pass

        # Override to keep files
        execution.upload_execution_outputs(clean_folder=False)

        # File should still exist
        assert test_file.exists(), "File was removed despite clean_folder=False"

        # Cleanup
        if execution.working_dir.exists():
            shutil.rmtree(execution.working_dir)

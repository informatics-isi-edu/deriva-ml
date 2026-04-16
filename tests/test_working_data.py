"""Tests for the working data cache and DerivaML.from_context()."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# =============================================================================
# WorkingDataCache Tests
# =============================================================================


class TestWorkingDataCache:
    """Tests for the WorkingDataCache class."""

    def test_cache_table_and_read(self, tmp_path):
        """Cache a DataFrame and read it back."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        df = pd.DataFrame({"RID": ["1-A", "1-B"], "Name": ["Alice", "Bob"]})

        result_path = cache.cache_table("Subject", df)

        assert result_path == cache.db_path
        assert cache.db_path.exists()

        read_back = cache.read_table("Subject")
        assert len(read_back) == 2
        assert list(read_back["Name"]) == ["Alice", "Bob"]

    def test_has_table(self, tmp_path):
        """has_table returns True only for cached tables."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        assert cache.has_table("Subject") is False

        df = pd.DataFrame({"RID": ["1-A"]})
        cache.cache_table("Subject", df)
        assert cache.has_table("Subject") is True
        assert cache.has_table("Image") is False

    def test_list_tables(self, tmp_path):
        """list_tables returns names of all cached tables."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        assert cache.list_tables() == []

        cache.cache_table("Subject", pd.DataFrame({"RID": ["1-A"]}))
        cache.cache_table("Image", pd.DataFrame({"RID": ["2-A"]}))
        assert sorted(cache.list_tables()) == ["Image", "Subject"]

    def test_table_info(self, tmp_path):
        """table_info returns metadata about a cached table."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        df = pd.DataFrame({"RID": ["1-A", "1-B"], "Name": ["Alice", "Bob"], "Age": [30, 25]})
        cache.cache_table("Subject", df)

        info = cache.table_info("Subject")
        assert info["table_name"] == "Subject"
        assert info["row_count"] == 2
        assert len(info["columns"]) == 3
        col_names = [c["name"] for c in info["columns"]]
        assert "RID" in col_names
        assert "Name" in col_names
        assert "Age" in col_names

    def test_table_info_missing_raises(self, tmp_path):
        """table_info raises ValueError for missing table."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            cache.table_info("Missing")

    def test_read_table_missing_raises(self, tmp_path):
        """read_table raises ValueError for missing table."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            cache.read_table("Missing")

    def test_query(self, tmp_path):
        """query executes SQL against the cache."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        df = pd.DataFrame({"RID": ["1-A", "1-B", "1-C"], "Age": [30, 25, 40]})
        cache.cache_table("Subject", df)

        result = cache.query("SELECT * FROM Subject WHERE Age > 28")
        assert len(result) == 2
        assert set(result["RID"]) == {"1-A", "1-C"}

    def test_drop_table(self, tmp_path):
        """drop_table removes a single table."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        cache.cache_table("Subject", pd.DataFrame({"RID": ["1-A"]}))
        cache.cache_table("Image", pd.DataFrame({"RID": ["2-A"]}))
        assert len(cache.list_tables()) == 2

        cache.drop_table("Subject")
        assert cache.list_tables() == ["Image"]

    def test_clear(self, tmp_path):
        """clear deletes the database file."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        cache.cache_table("Subject", pd.DataFrame({"RID": ["1-A"]}))
        assert cache.db_path.exists()

        cache.clear()
        assert not cache.db_path.exists()
        assert cache.list_tables() == []

    def test_replace_existing_table(self, tmp_path):
        """cache_table replaces an existing table."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        cache.cache_table("Subject", pd.DataFrame({"RID": ["1-A"]}))
        assert len(cache.read_table("Subject")) == 1

        cache.cache_table("Subject", pd.DataFrame({"RID": ["1-A", "1-B", "1-C"]}))
        assert len(cache.read_table("Subject")) == 3

    def test_status(self, tmp_path):
        """status returns overall cache info."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path)
        cache.cache_table("Subject", pd.DataFrame({"RID": ["1-A"]}))

        status = cache.status()
        assert status["table_count"] == 1
        assert status["db_size_bytes"] > 0
        assert "Subject" in status["tables"]

    def test_db_directory_created(self, tmp_path):
        """WorkingDataCache creates the working-data directory."""
        from deriva_ml.core.working_data import WorkingDataCache

        cache = WorkingDataCache(tmp_path / "deep" / "nested")
        assert (tmp_path / "deep" / "nested" / "working-data").is_dir()


# =============================================================================
# DerivaML.from_context() Tests
# =============================================================================


class TestFromContext:
    """Tests for DerivaML.from_context()."""

    def test_reads_context_file(self, tmp_path):
        """from_context reads .deriva-context.json from the given directory."""
        from deriva_ml.core.base import _find_context_file, CONTEXT_FILENAME

        context = {
            "hostname": "test.example.org",
            "catalog_id": "42",
            "default_schema": "my_schema",
        }
        (tmp_path / CONTEXT_FILENAME).write_text(json.dumps(context))

        found = _find_context_file(tmp_path)
        assert found == tmp_path / CONTEXT_FILENAME

        with open(found) as f:
            data = json.load(f)
        assert data["hostname"] == "test.example.org"
        assert data["catalog_id"] == "42"

    def test_walks_up_directories(self, tmp_path):
        """from_context searches parent directories."""
        from deriva_ml.core.base import _find_context_file, CONTEXT_FILENAME

        context = {"hostname": "test.example.org", "catalog_id": "42"}
        (tmp_path / CONTEXT_FILENAME).write_text(json.dumps(context))

        # Create a subdirectory and search from there
        subdir = tmp_path / "a" / "b" / "c"
        subdir.mkdir(parents=True)

        found = _find_context_file(subdir)
        assert found == tmp_path / CONTEXT_FILENAME

    def test_raises_when_not_found(self, tmp_path):
        """from_context raises FileNotFoundError when no context file exists."""
        from deriva_ml.core.base import _find_context_file

        with pytest.raises(FileNotFoundError, match="No .deriva-context.json"):
            _find_context_file(tmp_path)

"""Tests for BagDataSource, specifically multi-CSV handling for nested datasets.

Nested datasets produce multiple CSV files for the same table at different
directory depths in a BDBag. BagDataSource must read rows from ALL CSV files,
not just the last one found.
"""

from pathlib import Path

import pytest

from deriva_ml.model.data_sources import BagDataSource


@pytest.fixture
def nested_bag(tmp_path: Path) -> Path:
    """Create a minimal BDBag directory with multiple CSVs for the same table.

    Simulates the structure produced by a nested dataset export:
        data/
            Dataset/
                Dataset_Version.csv          (parent dataset versions)
                Dataset_Dataset/
                    Dataset/
                        Dataset_Version.csv  (child dataset versions)
    """
    data_dir = tmp_path / "data"

    # Parent-level Dataset_Version (e.g., for the root split dataset)
    parent_dv = data_dir / "Dataset" / "Dataset_Version.csv"
    parent_dv.parent.mkdir(parents=True)
    parent_dv.write_text(
        "RID,Dataset,Version\n"
        "V1,DS-ROOT,0.4.0\n"
        "V2,DS-ROOT,0.3.0\n"
    )

    # Child-level Dataset_Version (e.g., for Training/Test children)
    child_dv = data_dir / "Dataset" / "Dataset_Dataset" / "Dataset" / "Dataset_Version.csv"
    child_dv.parent.mkdir(parents=True)
    child_dv.write_text(
        "RID,Dataset,Version\n"
        "V3,DS-TRAIN,3.11.0\n"
        "V4,DS-TEST,5.11.0\n"
    )

    return tmp_path


@pytest.fixture
def nested_bag_with_duplicates(tmp_path: Path) -> Path:
    """Create a bag where the same RID appears in multiple CSV files.

    This can happen when a record is reachable through multiple paths
    in the dataset hierarchy.
    """
    data_dir = tmp_path / "data"

    # First CSV with some subjects
    csv1 = data_dir / "Dataset" / "Subject.csv"
    csv1.parent.mkdir(parents=True)
    csv1.write_text(
        "RID,Name\n"
        "S1,Alice\n"
        "S2,Bob\n"
    )

    # Second CSV with overlapping and new subjects
    csv2 = data_dir / "Dataset" / "Dataset_Dataset" / "Dataset" / "Subject.csv"
    csv2.parent.mkdir(parents=True)
    csv2.write_text(
        "RID,Name\n"
        "S2,Bob\n"
        "S3,Carol\n"
    )

    return tmp_path


class TestBagDataSourceMultiCSV:
    """Tests that BagDataSource correctly handles multiple CSVs per table."""

    def test_csv_cache_collects_all_paths(self, nested_bag: Path):
        """_csv_cache should map each table name to a list of ALL matching CSV paths."""
        source = BagDataSource(nested_bag, asset_localization=False)

        assert "Dataset_Version" in source._csv_cache
        assert len(source._csv_cache["Dataset_Version"]) == 2

    def test_get_table_data_yields_all_rows(self, nested_bag: Path):
        """get_table_data should yield rows from all CSV files for a table."""
        source = BagDataSource(nested_bag, asset_localization=False)
        rows = list(source.get_table_data("Dataset_Version"))

        # Should have all 4 rows: 2 from parent + 2 from child
        assert len(rows) == 4
        rids = {r["RID"] for r in rows}
        assert rids == {"V1", "V2", "V3", "V4"}

    def test_get_table_data_preserves_all_datasets(self, nested_bag: Path):
        """All dataset RIDs should be present in the yielded rows."""
        source = BagDataSource(nested_bag, asset_localization=False)
        rows = list(source.get_table_data("Dataset_Version"))

        datasets = {r["Dataset"] for r in rows}
        assert datasets == {"DS-ROOT", "DS-TRAIN", "DS-TEST"}

    def test_has_table_with_multiple_csvs(self, nested_bag: Path):
        """has_table should return True when multiple CSVs exist for a table."""
        source = BagDataSource(nested_bag, asset_localization=False)
        assert source.has_table("Dataset_Version") is True

    def test_list_available_tables(self, nested_bag: Path):
        """list_available_tables should include tables with multiple CSVs."""
        source = BagDataSource(nested_bag, asset_localization=False)
        tables = source.list_available_tables()
        assert "Dataset_Version" in tables

    def test_get_row_count_sums_all_csvs(self, nested_bag: Path):
        """get_row_count should return total rows across all CSV files."""
        source = BagDataSource(nested_bag, asset_localization=False)
        count = source.get_row_count("Dataset_Version")
        assert count == 4

    def test_duplicate_rows_are_yielded(self, nested_bag_with_duplicates: Path):
        """Duplicate RIDs across CSVs should all be yielded.

        Deduplication is handled downstream by DataLoader's on_conflict policy,
        not by BagDataSource.
        """
        source = BagDataSource(nested_bag_with_duplicates, asset_localization=False)
        rows = list(source.get_table_data("Subject"))

        # S2 appears in both CSVs, so we get 4 rows total (dedup is not BagDataSource's job)
        assert len(rows) == 4
        rids = [r["RID"] for r in rows]
        assert rids.count("S2") == 2

    def test_single_csv_still_works(self, tmp_path: Path):
        """Tables with a single CSV should work as before."""
        data_dir = tmp_path / "data"
        csv_file = data_dir / "Simple.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text(
            "RID,Value\n"
            "R1,hello\n"
            "R2,world\n"
        )

        source = BagDataSource(tmp_path, asset_localization=False)
        rows = list(source.get_table_data("Simple"))
        assert len(rows) == 2
        assert source.get_row_count("Simple") == 2

    def test_missing_table_returns_empty(self, nested_bag: Path):
        """Requesting a table with no CSVs should yield nothing."""
        source = BagDataSource(nested_bag, asset_localization=False)
        rows = list(source.get_table_data("NonExistent"))
        assert rows == []
        assert source.has_table("NonExistent") is False
        assert source.get_row_count("NonExistent") == 0

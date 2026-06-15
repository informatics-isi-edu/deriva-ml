"""Unit tests for _extract_archive_to_cache (client-bag staging refactor).

The helper owns the atomic cache-population dance: extract a bag archive
into a staging dir, validate, atomically rename into the checksum-keyed
cache location, and record in the BagCacheIndex. Both the S3 arm and the
client arm of the download path call it.

Pure-Python; no live catalog required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from deriva_ml.dataset.bag_download import _extract_archive_to_cache


def test_extract_archive_to_cache_atomic_move_and_record(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Fake index: bag_dir_for returns a checksum-keyed dir under cache_dir.
    bag_root = cache_dir / "bags" / "deadbeef_2-SNAP"
    index = MagicMock()
    index.bag_dir_for.return_value = bag_root

    # bdb.extract_bag creates the staging dir contents; simulate it by
    # populating the directory it is handed and returning that path.
    def fake_extract(archive, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "bag-info.txt").write_text("ok")
        return dest

    with (
        patch("deriva_ml.dataset.bag_download.bdb.extract_bag", side_effect=fake_extract),
        patch("deriva_ml.dataset.bag_download.bdb.validate_bag_structure"),
    ):
        result = _extract_archive_to_cache(
            index=index,
            archive_path="/tmp/whatever.zip",
            checksum="deadbeef_2-SNAP",
            dataset_rid="2-ABCD",
            dataset_version="1.0.0",
        )

    # Returns the bag dir inside the cache root.
    assert result == bag_root / "Dataset_2-ABCD"
    # The checksum-keyed cache dir now exists (atomic move happened).
    assert bag_root.exists()
    # Index recorded the bag under the Dataset anchor + version summary.
    index.record.assert_called_once()
    _, kwargs = index.record.call_args
    assert kwargs["checksum"] == "deadbeef_2-SNAP"
    assert kwargs["anchors"] == [("Dataset", "2-ABCD")]
    assert kwargs["anchor_summary"] == {"version": "1.0.0"}


def test_extract_archive_to_cache_cleans_staging_on_failure(tmp_path):
    """If validation raises, the staging dir is removed and the error propagates."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    bag_root = cache_dir / "bags" / "beef_2-SNAP"
    index = MagicMock()
    index.bag_dir_for.return_value = bag_root

    def fake_extract(archive, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        return dest

    import pytest

    with (
        patch("deriva_ml.dataset.bag_download.bdb.extract_bag", side_effect=fake_extract),
        patch(
            "deriva_ml.dataset.bag_download.bdb.validate_bag_structure",
            side_effect=RuntimeError("bad bag"),
        ),
    ):
        with pytest.raises(RuntimeError, match="bad bag"):
            _extract_archive_to_cache(
                index=index,
                archive_path="/tmp/whatever.zip",
                checksum="beef_2-SNAP",
                dataset_rid="2-ABCD",
                dataset_version="1.0.0",
            )

    # Staging dir cleaned up; no partial cache entry left behind.
    assert not (bag_root.parent / f"{bag_root.name}_staging").exists()
    assert not bag_root.exists()
    index.record.assert_not_called()

"""Client-arm bag staging lives under cache_dir, not working_dir, and is
always cleaned up (TemporaryDirectory guarantee), even on extract failure.

Regression coverage for the staging-spill fix: the client-side bag build
(``use_minid=False``) used to stage a multi-GB zip under
``working_dir/client_export/`` and clean it up only on the success path. It
now builds into a ``TemporaryDirectory(dir=cache_dir)`` and extracts in place,
so nothing lands in ``working_dir`` and the context manager guarantees cleanup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deriva_ml.dataset.bag_download import create_dataset_minid


def _fake_dataset(tmp_path):
    ds = MagicMock()
    ds.dataset_rid = "2-ABCD"
    ds._ml_instance.cache_dir = tmp_path / "cache"
    ds._ml_instance.working_dir = tmp_path / "work"
    ds._ml_instance.cache_dir.mkdir()
    ds._ml_instance.working_dir.mkdir()
    ds._ml_instance.s3_bucket = None
    return ds


def test_client_arm_stages_under_cache_dir_not_working_dir(tmp_path):
    ds = _fake_dataset(tmp_path)
    captured = {}

    def fake_build_bag(self, dataset, output_dir, timeout=None):
        captured["output_dir"] = Path(output_dir)
        zip_path = Path(output_dir) / "Dataset_2-ABCD.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("zip")
        return zip_path

    cache_bag = ds._ml_instance.cache_dir / "bags" / "x" / "Dataset_2-ABCD"

    with (
        patch("deriva_ml.dataset.bag_download.DatasetBagBuilder.build_bag", new=fake_build_bag),
        patch(
            "deriva_ml.dataset.bag_download._extract_archive_to_cache",
            return_value=cache_bag,
        ),
        patch("deriva_ml.dataset.bag_download.BagCacheIndex"),
    ):
        result = create_dataset_minid(
            ds,
            "1.0.0",
            use_minid=False,
            spec={"k": "v"},
            spec_hash="deadbeefdeadbeef",
            cache_suffix="deadbeefdeadbeef_2-SNAP",
        )

    # Staging output_dir is under cache_dir, never working_dir.
    assert ds._ml_instance.cache_dir in captured["output_dir"].parents
    assert ds._ml_instance.working_dir not in captured["output_dir"].parents
    # No client_export dir was created under working_dir.
    assert not (ds._ml_instance.working_dir / "client_export").exists()
    # Returns the final cache path (a Path, not a file:// URI string).
    assert isinstance(result, Path)
    assert "bags" in result.parts


def test_client_arm_cleans_staging_on_extract_failure(tmp_path):
    ds = _fake_dataset(tmp_path)
    seen_staging = []

    def fake_build_bag(self, dataset, output_dir, timeout=None):
        seen_staging.append(Path(output_dir))
        zip_path = Path(output_dir) / "Dataset_2-ABCD.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("zip")
        return zip_path

    with (
        patch("deriva_ml.dataset.bag_download.DatasetBagBuilder.build_bag", new=fake_build_bag),
        patch(
            "deriva_ml.dataset.bag_download._extract_archive_to_cache",
            side_effect=RuntimeError("extract boom"),
        ),
        patch("deriva_ml.dataset.bag_download.BagCacheIndex"),
    ):
        with pytest.raises(RuntimeError, match="extract boom"):
            create_dataset_minid(
                ds,
                "1.0.0",
                use_minid=False,
                spec={"k": "v"},
                spec_hash="deadbeefdeadbeef",
                cache_suffix="deadbeefdeadbeef_2-SNAP",
            )

    # The TemporaryDirectory(dir=cache_dir) staging is gone despite the failure.
    for staging in seen_staging:
        assert not staging.exists()
    # Nothing orphaned under working_dir.
    assert not (ds._ml_instance.working_dir / "client_export").exists()

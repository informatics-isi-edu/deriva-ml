"""Unit tests for the ``get_dataset_minid`` helpers (audit P1 Ds-minid).

Pre-extraction, ``get_dataset_minid`` was a 220-line monolith
in ``dataset/bag_download.py``. Its three tiers (local cache,
MINID/S3, client-side bag generation) were not unit-testable
without standing up a live catalog and S3 bucket.

Post-extraction it dispatches to four free helpers:

- ``_resolve_version_record`` — version-record lookup with
  ``DerivaMLException`` on miss.
- ``_build_version_rid`` — ``{rid}@{snap}`` formatting that
  handles ``snapshot=None``. Pinned because it was duplicated
  across Tier 1 and Tier 3 pre-extraction.
- ``_tier1_local_cache_lookup`` — local cache index check.
- ``_tier2_minid_path`` — MINID/S3 dispatch.
- ``_tier3_client_path`` — client-side bag generation.

These tests pin the small contracts each helper carries. Live
catalog and S3 are mocked.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetVersion
from deriva_ml.dataset.bag_download import (
    _build_version_rid,
    _resolve_version_record,
    _tier2_minid_path,
    _tier3_client_path,
)


class TestBuildVersionRid:
    """``_build_version_rid`` formats the RID with optional snapshot."""

    def test_with_snapshot(self):
        # Real snapshots are RID-shaped (catalog timestamps encoded
        # as short alphanumeric tokens), so the test fixture uses
        # one. The format isn't enforced by ``_build_version_rid``
        # itself but the downstream ``DatasetMinid.RID`` pattern
        # requires it.
        assert _build_version_rid("3WX", "2-7XYZ") == "3WX@2-7XYZ"

    def test_without_snapshot(self):
        """``snapshot=None`` returns the bare RID — not ``{rid}@None``.

        Pre-extraction this guard was duplicated across Tier 1
        and Tier 3. The ``DatasetMinid.RID`` pattern rejects
        ``{rid}@None`` with a cryptic regex error.
        """
        assert _build_version_rid("3WX", None) == "3WX"

    def test_empty_string_snapshot_is_treated_as_a_real_snapshot(self):
        """The guard is ``is None``, not falsy. An empty-string snapshot
        is unusual but should produce ``{rid}@``, not ``{rid}``.

        This is a pin against a future "is None or not snapshot" drift.
        """
        assert _build_version_rid("3WX", "") == "3WX@"


class TestResolveVersionRecord:
    """``_resolve_version_record`` looks up the history row by version."""

    def test_returns_matching_record(self):
        record = MagicMock(dataset_version="1.0.0")
        dataset = MagicMock()
        dataset.dataset_history.return_value = [
            MagicMock(dataset_version="0.1.0"),
            record,
            MagicMock(dataset_version="2.0.0"),
        ]
        version = DatasetVersion.parse("1.0.0")
        result = _resolve_version_record(dataset, version)
        assert result is record

    def test_unknown_version_raises(self):
        dataset = MagicMock()
        dataset.dataset_history.return_value = [MagicMock(dataset_version="0.1.0")]
        dataset.dataset_rid = "3WX"
        version = DatasetVersion.parse("99.0.0")
        with pytest.raises(DerivaMLException, match="99.0.0"):
            _resolve_version_record(dataset, version)


# ---------------------------------------------------------------------------
# _tier2_minid_path — MINID-or-regenerate dispatch
# ---------------------------------------------------------------------------


class TestTier2MinidPath:
    """``_tier2_minid_path`` fetches existing MINID or regenerates."""

    def test_existing_minid_with_matching_spec_hash_is_fetched(self):
        """When the stored ``spec_hash`` matches, the MINID is fetched."""
        dataset = MagicMock()
        version_record = MagicMock(spec_hash="abc123")
        with patch("deriva_ml.dataset.bag_download.fetch_minid_metadata") as fetch:
            fetch.return_value = "FETCHED"
            result = _tier2_minid_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                version_record=version_record,
                spec={"k": "v"},
                spec_hash="abc123",
                minid_url="ark://minid",
                create=True,
                exclude_tables=None,
                timeout=None,
            )
        assert result == "FETCHED"
        fetch.assert_called_once_with(dataset, DatasetVersion.parse("1.0.0"), "ark://minid")

    def test_no_minid_no_create_raises(self):
        dataset = MagicMock()
        dataset.dataset_rid = "3WX"
        version_record = MagicMock(spec_hash=None)
        with pytest.raises(DerivaMLException, match="doesn't exist"):
            _tier2_minid_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                version_record=version_record,
                spec={},
                spec_hash="abc",
                minid_url=None,
                create=False,
                exclude_tables=None,
                timeout=None,
            )

    def test_spec_drift_regenerates_when_create_true(self):
        """Stored spec_hash mismatch → regenerate; ``create=False`` would raise."""
        dataset = MagicMock()
        version_record = MagicMock(spec_hash="OLD")
        with (
            patch("deriva_ml.dataset.bag_download.fetch_minid_metadata") as fetch,
            patch("deriva_ml.dataset.bag_download.create_dataset_minid") as create_fn,
        ):
            fetch.return_value = "REFRESHED"
            create_fn.return_value = "ark://new"
            result = _tier2_minid_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                version_record=version_record,
                spec={"k": "v"},
                spec_hash="NEW",
                minid_url="ark://stale",
                create=True,
                exclude_tables={"ExcludedTable"},
                timeout=(5, 30),
            )
        assert result == "REFRESHED"
        create_fn.assert_called_once()
        # The regenerated URL flows into the metadata fetch.
        fetch.assert_called_once_with(dataset, DatasetVersion.parse("1.0.0"), "ark://new")

    def test_spec_drift_create_false_raises(self):
        dataset = MagicMock()
        dataset.dataset_rid = "3WX"
        version_record = MagicMock(spec_hash="OLD")
        with pytest.raises(DerivaMLException, match="doesn't exist"):
            _tier2_minid_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                version_record=version_record,
                spec={},
                spec_hash="NEW",
                minid_url="ark://stale",
                create=False,
                exclude_tables=None,
                timeout=None,
            )


# ---------------------------------------------------------------------------
# _tier3_client_path — client-side bag generation
# ---------------------------------------------------------------------------


class TestTier3ClientPath:
    """``_tier3_client_path`` generates bags locally under the cache suffix."""

    def test_generates_bag_and_returns_dataset_minid(self, tmp_path):
        dataset = MagicMock()
        dataset.dataset_rid = "3WX"
        # The client arm now extracts into the cache itself and returns the
        # final cache PATH; Tier 3 wraps its PARENT (the checksum cache root)
        # as the DatasetMinid location, matching the Tier-1 shape so the
        # downstream download_dataset_minid call is a no-op cache hit.
        cache_bag = tmp_path / "cache" / "bags" / "abcdef1234567890_2-7XYZ" / "Dataset_3WX"
        cache_bag.mkdir(parents=True)
        # Snapshot must be RID-shaped to satisfy the
        # ``DatasetMinid.RID`` pattern.
        with patch("deriva_ml.dataset.bag_download.create_dataset_minid") as create_fn:
            create_fn.return_value = cache_bag
            result = _tier3_client_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                spec={"k": "v"},
                spec_hash="abcdef1234567890abcdef1234567890",
                snapshot="2-7XYZ",
                minid_url=None,
                create=True,
                cache_suffix="abcdef1234567890_2-7XYZ",
                exclude_tables=None,
                timeout=None,
            )
        # cache_suffix is forwarded to the client arm so its write key matches
        # what Tier 1 reads.
        assert create_fn.call_args.kwargs["cache_suffix"] == "abcdef1234567890_2-7XYZ"
        # ``DatasetMinid`` aliases ``location`` → ``bag_url``. The location is
        # the file:// URI of the cache root (parent of the extracted bag dir).
        assert result.bag_url == cache_bag.parent.as_uri()
        assert result.version_rid == "3WX@2-7XYZ"
        assert result.checksum == "abcdef1234567890_2-7XYZ"

    def test_no_minid_no_create_raises(self):
        dataset = MagicMock()
        dataset.dataset_rid = "3WX"
        with pytest.raises(DerivaMLException, match="doesn't exist"):
            _tier3_client_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                spec={},
                spec_hash="abc",
                snapshot=None,
                minid_url=None,
                create=False,
                cache_suffix="abc_None",
                exclude_tables=None,
                timeout=None,
            )

    def test_existing_minid_url_allows_create_false(self):
        """``create=False`` is OK when a prior MINID URL exists — the URL
        carries enough information that the bag can still be rebuilt.

        Pin the existing semantic so a future "tighten the guard"
        change is intentional.
        """
        dataset = MagicMock()
        dataset.dataset_rid = "3WX"
        with patch("deriva_ml.dataset.bag_download.create_dataset_minid") as create_fn:
            create_fn.return_value = Path("/tmp/cache/bags/abc_None/Dataset_3WX")
            result = _tier3_client_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                spec={},
                spec_hash="abc",
                snapshot=None,
                minid_url="ark://existing",
                create=False,
                cache_suffix="abc_None",
                exclude_tables=None,
                timeout=None,
            )
        # Bare RID — no snapshot.
        assert result.version_rid == "3WX"

    def test_passes_through_exclude_tables_and_timeout(self):
        dataset = MagicMock()
        dataset.dataset_rid = "3WX"
        with patch("deriva_ml.dataset.bag_download.create_dataset_minid") as create_fn:
            create_fn.return_value = Path("/tmp/cache/bags/abc_None/Dataset_3WX")
            _tier3_client_path(
                dataset,
                version=DatasetVersion.parse("1.0.0"),
                spec={},
                spec_hash="abc",
                snapshot=None,
                minid_url=None,
                create=True,
                cache_suffix="abc_None",
                exclude_tables={"Skip"},
                timeout=(7, 42),
            )
        kwargs = create_fn.call_args.kwargs
        assert kwargs["exclude_tables"] == {"Skip"}
        assert kwargs["timeout"] == (7, 42)
        assert kwargs["use_minid"] is False
        # cache_suffix is forwarded so the client arm's write key matches Tier 1.
        assert kwargs["cache_suffix"] == "abc_None"

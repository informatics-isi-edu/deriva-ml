"""Tests for strict asset table detection."""
import pytest
from tempfile import TemporaryDirectory


class TestIsAsset:
    """Test that is_asset enforces NOT NULL constraints on asset columns."""

    def test_proper_asset_table_detected(self, catalog_manager):
        """Image table (proper asset) should be detected as asset."""
        with TemporaryDirectory() as tmp:
            ml = catalog_manager.get_ml_instance(tmp)
            assert ml.model.is_asset("Image") is True

    def test_table_with_nullable_url_not_asset(self, catalog_manager):
        """Report table (nullable URL/Length/MD5) should NOT be an asset."""
        with TemporaryDirectory() as tmp:
            ml = catalog_manager.get_ml_instance(tmp)
            # Report has asset-like columns but nullable URL/Length/MD5
            assert ml.model.is_asset("Report") is False

    def test_regular_table_not_asset(self, catalog_manager):
        """Subject table (no asset columns) should NOT be an asset."""
        with TemporaryDirectory() as tmp:
            ml = catalog_manager.get_ml_instance(tmp)
            assert ml.model.is_asset("Subject") is False

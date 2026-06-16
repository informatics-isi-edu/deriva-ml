"""Unit tests for ``restructure_assets`` helpers (audit P1 Ds-restr).

Pre-extraction, ``restructure_assets`` was a 500-line function
(250+ LOC body) that mixed input validation, target
classification, per-asset grouping resolution, source-path
lookup, and file placement. The audit asked for the inner
resolution and placement steps to be extracted so they could
be unit-tested without a live catalog or bag.

Post-extraction the helpers are:

- ``_validate_targets`` — reject dotted ``"Feature.column"``.
- ``_classify_targets`` — feature vs column probe.
- ``_resolve_source_path`` — locate the on-disk file, using
  ``bag.path`` (the public API; pre-extraction this site
  reached through two private attributes).
- ``_resolve_grouping_path`` — per-asset directory components.
- ``_place_asset`` — symlink / copy / transformer dispatch.
- ``_default_dir_name_from_target`` — drive-by P2 fix: use
  ``DerivaSystemColumns`` instead of a hardcoded set.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import (
    DerivaMLException,
    DerivaMLValidationError,
)
from deriva_ml.dataset.restructure import (
    _classify_targets,
    _default_dir_name_from_target,
    _place_asset,
    _resolve_grouping_path,
    _resolve_source_path,
    _validate_targets,
)


# ---------------------------------------------------------------------------
# _validate_targets — dotted syntax rejection
# ---------------------------------------------------------------------------


class TestValidateTargets:
    def test_none_is_ok(self):
        _validate_targets(None)  # no raise

    def test_list_without_dots_is_ok(self):
        _validate_targets(["Diagnosis", "Severity"])  # no raise

    def test_dict_without_dots_is_ok(self):
        _validate_targets({"Diagnosis": lambda recs: recs[0]})  # no raise

    def test_list_with_dot_raises_with_migration_hint(self):
        with pytest.raises(DerivaMLValidationError, match="Replace with"):
            _validate_targets(["Classification.Label"])

    def test_dict_key_with_dot_raises(self):
        with pytest.raises(DerivaMLValidationError):
            _validate_targets({"Classification.Label": lambda r: r[0]})


# ---------------------------------------------------------------------------
# _classify_targets — feature vs column probe
# ---------------------------------------------------------------------------


class TestClassifyTargets:
    def test_none_returns_empty(self):
        bag = MagicMock()
        assert _classify_targets(bag, "Image", None) == (None, [])

    def test_all_features(self):
        bag = MagicMock()
        bag.lookup_feature.return_value = MagicMock()  # always succeeds
        spec, cols = _classify_targets(bag, "Image", ["Diagnosis", "Severity"])
        assert spec == ["Diagnosis", "Severity"]
        assert cols == []

    def test_all_columns(self):
        bag = MagicMock()
        bag.lookup_feature.side_effect = Exception("not a feature")
        spec, cols = _classify_targets(bag, "Image", ["Filename", "Description"])
        assert spec is None
        assert cols == ["Filename", "Description"]

    def test_mixed_split(self):
        bag = MagicMock()

        # Diagnosis is a feature; Filename isn't.
        def fake_lookup(table, name):
            if name == "Diagnosis":
                return MagicMock()
            raise Exception("not a feature")

        bag.lookup_feature.side_effect = fake_lookup
        spec, cols = _classify_targets(bag, "Image", ["Diagnosis", "Filename"])
        assert spec == ["Diagnosis"]
        assert cols == ["Filename"]

    def test_dict_input_preserves_selector_for_features(self):
        bag = MagicMock()

        def fake_lookup(table, name):
            if name == "Diagnosis":
                return MagicMock()
            raise Exception()

        bag.lookup_feature.side_effect = fake_lookup
        sel = lambda recs: recs[0]
        spec, cols = _classify_targets(bag, "Image", {"Diagnosis": sel, "Filename": sel})
        assert spec == {"Diagnosis": sel}
        assert cols == ["Filename"]


# ---------------------------------------------------------------------------
# _default_dir_name_from_target — P2: SYSTEM_COLUMNS sourcing
# ---------------------------------------------------------------------------


class TestDefaultDirNameFromTarget:
    def test_none_target(self):
        assert _default_dir_name_from_target(None, []) == "Unknown"

    def test_string_target_returned_as_is(self):
        assert _default_dir_name_from_target("benign", []) == "benign"

    def test_dict_target_raises_with_migration_hint(self):
        with pytest.raises(DerivaMLException, match="target_transform"):
            _default_dir_name_from_target({"a": 1}, ["a"])

    def test_skips_canonical_system_columns(self):
        """``DerivaSystemColumns`` are skipped; the first content column wins.

        Pin the P2 fix: the skip set is sourced from the
        canonical constant ``DerivaSystemColumns`` plus the
        feature-table-specific ``Feature_Name`` / ``Execution``
        columns. A future ``RVT`` added to the canonical constant
        would propagate automatically.
        """
        record = MagicMock()
        record.model_dump.return_value = {
            "RID": "REC-1",
            "RCT": "ignored",
            "RMT": "ignored",
            "RCB": "ignored",
            "RMB": "ignored",
            "Feature_Name": "Diagnosis",
            "Execution": "EXE-1",
            "Diagnosis": "Benign",
        }
        assert _default_dir_name_from_target(record, ["Diagnosis"]) == "Benign"

    def test_falls_through_to_unknown_when_only_system_cols(self):
        record = MagicMock()
        record.model_dump.return_value = {
            "RID": "REC-1",
            "RCT": "x",
            "RMT": "x",
            "RCB": "x",
            "RMB": "x",
            "Feature_Name": "x",
            "Execution": "x",
        }
        assert _default_dir_name_from_target(record, ["Whatever"]) == "Unknown"


# ---------------------------------------------------------------------------
# _resolve_source_path — public bag.path fallback
# ---------------------------------------------------------------------------


class TestResolveSourcePath:
    def test_returns_filename_when_it_exists(self, tmp_path):
        f = tmp_path / "real.jpg"
        f.write_bytes(b"img")
        result = _resolve_source_path({"Filename": str(f), "RID": "ASSET-1"}, "Image", MagicMock())
        assert result == f

    def test_falls_back_to_bag_path_layout(self, tmp_path):
        """When ``Filename`` is a bare basename, looks under ``bag.path``."""
        bag_root = tmp_path / "bag"
        canonical_dir = bag_root / "data" / "asset" / "ASSET-1" / "Image"
        canonical_dir.mkdir(parents=True)
        canonical_file = canonical_dir / "img.jpg"
        canonical_file.write_bytes(b"img")

        bag = MagicMock()
        bag.path = bag_root  # public property
        # ``Filename`` is a bare name (no path) that doesn't exist as-is.
        result = _resolve_source_path({"Filename": "img.jpg", "RID": "ASSET-1"}, "Image", bag)
        assert result == canonical_file

    def test_no_filename_returns_none(self):
        assert _resolve_source_path({"RID": "ASSET-1"}, "Image", MagicMock()) is None

    def test_neither_location_exists_returns_none(self, tmp_path):
        bag = MagicMock()
        bag.path = tmp_path / "no-such-bag"
        result = _resolve_source_path({"Filename": "nope.jpg", "RID": "ASSET-1"}, "Image", bag)
        assert result is None

    def test_attribute_error_on_path_is_handled(self, tmp_path):
        """Bags constructed by test mocks may not have ``.path``.

        Pre-extraction this site reached through
        ``bag._catalog._database_model.bag_path`` with the same
        try/except. The helper preserves the guard against
        truncated mocks.
        """
        bag = MagicMock()
        del bag.path  # simulate missing attribute
        type(bag).path = property(lambda self: (_ for _ in ()).throw(AttributeError()))

        result = _resolve_source_path({"Filename": "img.jpg", "RID": "ASSET-1"}, "Image", bag)
        assert result is None


# ---------------------------------------------------------------------------
# _resolve_grouping_path — per-asset directory components
# ---------------------------------------------------------------------------


class TestResolveGroupingPath:
    """The core dispatch of feature/column targets + missing policy."""

    def test_column_target_with_value(self):
        path, skip = _resolve_grouping_path(
            asset={"RID": "ASSET-1"},
            target_names_list=["Quality"],
            column_targets=["Quality"],
            feature_targets_spec=None,
            feature_target_map={},
            column_value_map={"Quality": {"ASSET-1": "Sharp"}},
            target_transform=None,
            missing="unknown",
        )
        assert path == ["Sharp"]
        assert skip is False

    def test_column_target_missing_with_unknown(self):
        path, skip = _resolve_grouping_path(
            asset={"RID": "ASSET-1"},
            target_names_list=["Quality"],
            column_targets=["Quality"],
            feature_targets_spec=None,
            feature_target_map={},
            column_value_map={"Quality": {}},
            target_transform=None,
            missing="unknown",
        )
        assert path == ["Unknown"]
        assert skip is False

    def test_column_target_missing_with_skip(self):
        path, skip = _resolve_grouping_path(
            asset={"RID": "ASSET-1"},
            target_names_list=["Quality"],
            column_targets=["Quality"],
            feature_targets_spec=None,
            feature_target_map={},
            column_value_map={"Quality": {}},
            target_transform=None,
            missing="skip",
        )
        assert skip is True

    def test_column_target_missing_with_error_raises(self):
        with pytest.raises(DerivaMLException, match="Quality"):
            _resolve_grouping_path(
                asset={"RID": "ASSET-1"},
                target_names_list=["Quality"],
                column_targets=["Quality"],
                feature_targets_spec=None,
                feature_target_map={},
                column_value_map={"Quality": {}},
                target_transform=None,
                missing="error",
            )

    def test_feature_target_with_record(self):
        record = MagicMock()
        record.model_dump.return_value = {
            "RID": "FEAT-1",
            "RCT": "x",
            "RMT": "x",
            "RCB": "x",
            "RMB": "x",
            "Feature_Name": "Diagnosis",
            "Execution": "x",
            "Diagnosis": "Benign",
        }
        path, skip = _resolve_grouping_path(
            asset={"RID": "ASSET-1"},
            target_names_list=["Diagnosis"],
            column_targets=[],
            feature_targets_spec=["Diagnosis"],
            feature_target_map={"ASSET-1": record},
            column_value_map={},
            target_transform=None,
            missing="unknown",
        )
        assert path == ["Benign"]
        assert skip is False

    def test_feature_target_missing_skip(self):
        path, skip = _resolve_grouping_path(
            asset={"RID": "ASSET-1"},
            target_names_list=["Diagnosis"],
            column_targets=[],
            feature_targets_spec=["Diagnosis"],
            feature_target_map={},  # ASSET-1 not present
            column_value_map={},
            target_transform=None,
            missing="skip",
        )
        assert skip is True

    def test_feature_target_missing_unknown(self):
        path, skip = _resolve_grouping_path(
            asset={"RID": "ASSET-1"},
            target_names_list=["Diagnosis"],
            column_targets=[],
            feature_targets_spec=["Diagnosis"],
            feature_target_map={},  # ASSET-1 absent → Unknown branch
            column_value_map={},
            target_transform=None,
            missing="unknown",
        )
        assert path == ["Unknown"]
        assert skip is False

    def test_feature_target_missing_error_raises(self):
        with pytest.raises(DerivaMLException, match="ASSET-1"):
            _resolve_grouping_path(
                asset={"RID": "ASSET-1"},
                target_names_list=["Diagnosis"],
                column_targets=[],
                feature_targets_spec=["Diagnosis"],
                feature_target_map={},
                column_value_map={},
                target_transform=None,
                missing="error",
            )

    def test_target_transform_must_return_str(self):
        """Non-str return from target_transform raises ``DerivaMLValidationError``."""
        with pytest.raises(DerivaMLValidationError, match="must return str"):
            _resolve_grouping_path(
                asset={"RID": "ASSET-1"},
                target_names_list=["Quality"],
                column_targets=["Quality"],
                feature_targets_spec=None,
                feature_target_map={},
                column_value_map={"Quality": {"ASSET-1": "raw"}},
                target_transform=lambda val: 42,  # returns int, not str
                missing="unknown",
            )


# ---------------------------------------------------------------------------
# _place_asset — file placement dispatch
# ---------------------------------------------------------------------------


class TestPlaceAsset:
    def test_symlink_mode(self, tmp_path):
        src = tmp_path / "src.jpg"
        src.write_bytes(b"img")
        dst = tmp_path / "dst.jpg"
        result = _place_asset(src, dst, file_transformer=None, use_symlinks=True)
        assert result == dst
        assert dst.is_symlink()
        assert dst.read_bytes() == b"img"

    def test_copy_mode(self, tmp_path):
        src = tmp_path / "src.jpg"
        src.write_bytes(b"img")
        dst = tmp_path / "dst.jpg"
        result = _place_asset(src, dst, file_transformer=None, use_symlinks=False)
        assert result == dst
        assert not dst.is_symlink()
        assert dst.read_bytes() == b"img"

    def test_transformer_overrides_symlink(self, tmp_path):
        """``file_transformer`` takes precedence over ``use_symlinks``."""
        src = tmp_path / "src.jpg"
        src.write_bytes(b"img")
        dst = tmp_path / "dst.jpg"
        actual = tmp_path / "transformed.png"  # the transformer chose a new path

        def my_transformer(src_p: Path, dst_p: Path) -> Path:
            actual.write_bytes(b"converted")
            return actual

        result = _place_asset(src, dst, file_transformer=my_transformer, use_symlinks=True)
        assert result == actual
        assert actual.read_bytes() == b"converted"
        # The suggested destination wasn't touched.
        assert not dst.exists()

    def test_symlink_falls_back_to_copy_on_oserror(self, tmp_path, monkeypatch):
        """Platforms refusing symlinks fall back to ``shutil.copy2``."""
        src = tmp_path / "src.jpg"
        src.write_bytes(b"img")
        dst = tmp_path / "dst.jpg"

        original_symlink_to = Path.symlink_to

        def boom(self, target):
            raise OSError("symlinks not supported")

        monkeypatch.setattr(Path, "symlink_to", boom)
        # ``shutil.copy2`` should still produce the file.
        try:
            result = _place_asset(src, dst, file_transformer=None, use_symlinks=True)
            assert result == dst
            assert dst.read_bytes() == b"img"
            assert not dst.is_symlink()
        finally:
            monkeypatch.setattr(Path, "symlink_to", original_symlink_to)

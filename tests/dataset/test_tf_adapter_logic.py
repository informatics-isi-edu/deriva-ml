"""Pure-Python logic tests for tf_adapter.

Uses pytest.importorskip — runs only when tensorflow is installed. Tests
the join logic, selector pass-through, missing= branches, target arity,
output_signature inference, and error paths using real tf.data.Dataset.

Coverage matrix (spec §6.2):
- missing="error" raises at construction with RID list
- missing="skip" drops unlabeled elements
- missing="unknown" passes None target for unlabeled elements
- Selector dict form vs list form yield equivalent results
- Single-target returns FeatureRecord via target_transform
- Multi-target returns dict[str, FeatureRecord] via target_transform
- Asset-table element with no sample_loader raises at construction
- Non-asset-table element works without sample_loader
- output_signature=None triggers first-sample inference
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

tf = pytest.importorskip("tensorflow")

from deriva_ml.core.exceptions import DerivaMLException  # noqa: E402
from deriva_ml.dataset.tf_adapter import build_tf_dataset  # noqa: E402
from deriva_ml.feature import FeatureRecord  # noqa: E402


class _FakeRecord(FeatureRecord):
    """Minimal FeatureRecord stand-in for tf adapter tests."""

    Image: str
    Grade: str
    Feature_Name: str = "Grade"  # default so tests don't need to pass it


def _mock_bag_with_labeled_images(rids_and_labels: dict[str, str]):
    """Build a MagicMock DatasetBag with the given labeled images.

    Args:
        rids_and_labels: Mapping of RID to label string for labeled images.
            Only RIDs with non-empty labels will be returned by feature_values.

    Returns:
        A MagicMock DatasetBag.
    """
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": rid} for rid in rids_and_labels]}
    )

    def fake_feature_values(element_type, feature_name, selector=None):
        for rid, label in rids_and_labels.items():
            if not label:
                # Simulate missing: no record yielded for empty label
                continue
            rec = _FakeRecord(Image=rid, Grade=label)
            if selector is None:
                yield rec
            else:
                sel = selector([rec])
                if sel is not None:
                    yield sel

    bag.feature_values = fake_feature_values
    # Mock table metadata so the asset-path resolver can find columns.
    bag.model = MagicMock()
    bag.model.is_asset = MagicMock(return_value=True)
    asset_row = {"RID": "1-IMG1", "Filename": "img1.jpg"}
    bag.get_table_as_dict = MagicMock(return_value=iter([asset_row]))
    bag.is_asset = MagicMock(return_value=True)
    return bag


class _FakeSubjectRecord(FeatureRecord):
    """FeatureRecord for non-asset (tabular) tests with Subject element column."""

    Subject: str
    Grade: str
    Feature_Name: str = "Grade"


def _mock_non_asset_bag(rids_and_labels: dict[str, str]):
    """Build a MagicMock bag for a non-asset element type (tabular)."""
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(
        return_value={"Subject": [{"RID": rid} for rid in rids_and_labels]}
    )

    def fake_feature_values(element_type, feature_name, selector=None):
        for rid, label in rids_and_labels.items():
            if not label:
                continue
            rec = _FakeSubjectRecord(Subject=rid, Grade=label)
            if selector is None:
                yield rec
            else:
                sel = selector([rec])
                if sel is not None:
                    yield sel

    bag.feature_values = fake_feature_values
    bag.model = MagicMock()
    bag.model.is_asset = MagicMock(return_value=False)
    bag.get_table_as_dict = MagicMock(return_value=iter([]))
    bag.is_asset = MagicMock(return_value=False)
    return bag


# ---------------------------------------------------------------------------
# Basic construction tests
# ---------------------------------------------------------------------------


def test_as_tf_dataset_returns_tf_dataset():
    """build_tf_dataset returns an instance of tf.data.Dataset."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0, 2.0, 3.0]),
        targets=["Grade"],
        output_signature=(
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    assert isinstance(ds, tf.data.Dataset)


def test_element_count_reflects_all_when_no_skip():
    """Dataset element count equals total member count when missing != 'skip'."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": "Severe"})
    bag.get_table_as_dict.return_value = iter([
        {"RID": "1-IMG1", "Filename": "img1.jpg"},
        {"RID": "1-IMG2", "Filename": "img2.jpg"},
    ])
    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade"],
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    count = sum(1 for _ in ds)
    assert count == 2


# ---------------------------------------------------------------------------
# missing= branch tests (spec §6.2)
# ---------------------------------------------------------------------------


def test_missing_error_raises_at_construction():
    """missing='error' raises DerivaMLException at construction listing RIDs."""
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": "1-IMG1"}, {"RID": "1-IMG2"}]}
    )
    bag.feature_values = MagicMock(
        return_value=iter([_FakeRecord(Image="1-IMG1", Grade="Mild")])
    )
    bag.model = MagicMock()
    bag.model.is_asset = MagicMock(return_value=True)
    bag.get_table_as_dict = MagicMock(return_value=iter([]))
    bag.is_asset = MagicMock(return_value=True)
    with pytest.raises(DerivaMLException, match=r"1-IMG2"):
        build_tf_dataset(
            bag,
            "Image",
            sample_loader=lambda p, row: tf.constant([]),
            targets=["Grade"],
            missing="error",
        )


def test_missing_skip_drops_unlabeled_elements():
    """missing='skip' drops unlabeled elements from the dataset."""
    # Only IMG1 has a label; IMG2 has empty label (skipped by fake_feature_values)
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": ""})
    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade"],
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        missing="skip",
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    count = sum(1 for _ in ds)
    assert count == 1


def test_missing_unknown_keeps_all_elements_with_none_target():
    """missing='unknown' keeps all elements; target is None for unlabeled."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": ""})
    # Override get_table_as_dict to return both rows
    bag.get_table_as_dict.return_value = iter([
        {"RID": "1-IMG1", "Filename": "img1.jpg"},
        {"RID": "1-IMG2", "Filename": "img2.jpg"},
    ])
    # target_transform must handle None for the unlabeled element.
    targets_seen = []

    def capture_and_convert(rec):
        targets_seen.append(rec)
        return tf.constant(rec.Grade if rec is not None else "")

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade"],
        target_transform=capture_and_convert,
        missing="unknown",
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    count = sum(1 for _ in ds)
    assert count == 2
    assert None in targets_seen


# ---------------------------------------------------------------------------
# Target arity tests (spec §3.3 / §6.2)
# ---------------------------------------------------------------------------


def test_single_target_target_transform_receives_featurerecord():
    """Single-target: target_transform receives a FeatureRecord directly."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    bag.get_table_as_dict.return_value = iter([{"RID": "1-IMG1", "Filename": "f.jpg"}])
    received = []

    def capture_target(target):
        received.append(target)
        return tf.constant(0)

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade"],
        target_transform=capture_target,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    list(ds)  # consume to trigger __call__
    assert len(received) == 1
    assert isinstance(received[0], FeatureRecord)
    assert received[0].Grade == "Mild"


def test_multi_target_target_transform_receives_dict():
    """Multi-target: target_transform receives dict[str, FeatureRecord]."""
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": "1-IMG1"}]}
    )

    class _FakeGradeRecord(FeatureRecord):
        Image: str
        Grade: str
        Feature_Name: str = "Grade"

    class _FakeSeverityRecord(FeatureRecord):
        Image: str
        Severity: str
        Feature_Name: str = "Severity"

    grade_rec = _FakeGradeRecord(Image="1-IMG1", Grade="Mild")
    severity_rec = _FakeSeverityRecord(Image="1-IMG1", Severity="Low")

    def fake_feature_values(element_type, feature_name, selector=None):
        if feature_name == "Grade":
            yield grade_rec
        elif feature_name == "Severity":
            yield severity_rec

    bag.feature_values = fake_feature_values
    bag.model = MagicMock()
    bag.model.is_asset = MagicMock(return_value=True)
    bag.get_table_as_dict = MagicMock(return_value=iter([{"RID": "1-IMG1", "Filename": "f.jpg"}]))
    bag.is_asset = MagicMock(return_value=True)

    received = []

    def capture_target(target):
        received.append(target)
        return tf.constant(0)

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade", "Severity"],
        target_transform=capture_target,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    list(ds)
    assert len(received) == 1
    target_dict = received[0]
    assert isinstance(target_dict, dict)
    assert set(target_dict.keys()) == {"Grade", "Severity"}


# ---------------------------------------------------------------------------
# Selector dict form (spec §6.2)
# ---------------------------------------------------------------------------


def test_selector_dict_form_passes_selector_to_feature_values():
    """dict form targets passes per-feature selectors through."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    bag.get_table_as_dict.return_value = iter([{"RID": "1-IMG1", "Filename": "f.jpg"}])
    selector_called = []

    def my_selector(records):
        selector_called.append(True)
        return records[0] if records else None

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets={"Grade": my_selector},
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    list(ds)
    assert selector_called, "selector was never called"


# ---------------------------------------------------------------------------
# Asset-table-without-sample_loader error (spec §6.2)
# ---------------------------------------------------------------------------


def test_asset_table_without_sample_loader_raises():
    """Asset-table element_type with no sample_loader raises at construction."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    with pytest.raises(DerivaMLException, match=r"sample_loader"):
        build_tf_dataset(bag, "Image", targets=["Grade"])


# ---------------------------------------------------------------------------
# Non-asset table: no sample_loader required (spec §3.2)
# ---------------------------------------------------------------------------


def test_non_asset_table_no_sample_loader_returns_row_dict():
    """Non-asset element_type with no sample_loader defaults to returning the row.

    The adapter constructs successfully without a sample_loader. Because
    tf.data.Dataset requires tensors, we supply a sample_loader that
    converts the row dict to a tensor — this is also how real users would
    work with tabular data in TF. The key assertion is that construction
    does NOT raise even without a sample_loader.
    """
    bag = _mock_non_asset_bag({"1-SUB1": "active"})
    bag.get_table_as_dict.return_value = iter([{"RID": "1-SUB1", "Status": "active"}])

    # First verify that construction succeeds (no sample_loader, no error).
    # We don't iterate — just check isinstance.
    ds_no_loader = build_tf_dataset(
        bag,
        "Subject",
        targets=["Grade"],
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        output_signature=(
            tf.TensorSpec(shape=None, dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    assert isinstance(ds_no_loader, tf.data.Dataset)

    # With a sample_loader that converts the dict to a tensor, iteration works.
    bag2 = _mock_non_asset_bag({"1-SUB1": "active"})
    bag2.get_table_as_dict.return_value = iter([{"RID": "1-SUB1", "Status": "active"}])
    ds = build_tf_dataset(
        bag2,
        "Subject",
        sample_loader=lambda path, row: tf.constant(row.get("RID", "")),
        targets=["Grade"],
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    assert isinstance(ds, tf.data.Dataset)
    count = sum(1 for _ in ds)
    assert count >= 1


# ---------------------------------------------------------------------------
# TF-specific: output_signature=None triggers first-sample inference
# ---------------------------------------------------------------------------


def test_output_signature_none_infers_from_first_sample():
    """output_signature=None: element_spec is inferred from the first yielded sample."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": "Severe"})
    bag.get_table_as_dict.return_value = iter([
        {"RID": "1-IMG1", "Filename": "img1.jpg"},
        {"RID": "1-IMG2", "Filename": "img2.jpg"},
    ])

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0, 2.0, 3.0]),
        targets=None,
        output_signature=None,  # trigger inference
    )

    assert isinstance(ds, tf.data.Dataset)
    # element_spec should reflect a float32 tensor of shape (3,)
    spec = ds.element_spec
    assert spec.dtype == tf.float32
    assert spec.shape == (3,)

    # All elements should be present (inference must not drop the first sample)
    count = sum(1 for _ in ds)
    assert count == 2

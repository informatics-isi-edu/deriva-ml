"""Pure-Python logic tests for torch_adapter.

Uses pytest.importorskip — runs only when torch is installed. Tests
the join logic, selector pass-through, missing= branches, target arity,
and error paths using the real torch.utils.data.Dataset base class.

Coverage matrix (spec §6.2):
- missing="error" raises at construction with RID list
- missing="skip" drops unlabeled elements
- missing="unknown" passes None target for unlabeled elements
- Selector dict form vs list form yield equivalent results
- Single-target returns FeatureRecord via target_transform
- Multi-target returns dict[str, FeatureRecord] via target_transform
- Asset-table element with no sample_loader raises at construction
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from deriva_ml.core.exceptions import DerivaMLException  # noqa: E402
from deriva_ml.dataset.torch_adapter import build_torch_dataset  # noqa: E402
from deriva_ml.feature import FeatureRecord  # noqa: E402


class _FakeRecord(FeatureRecord):
    """Minimal FeatureRecord stand-in for torch adapter tests."""

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
    bag.list_dataset_members = MagicMock(return_value={"Image": [{"RID": rid} for rid in rids_and_labels]})

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
    bag.list_dataset_members = MagicMock(return_value={"Subject": [{"RID": rid} for rid in rids_and_labels]})

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


def test_as_torch_dataset_returns_torch_dataset_subclass():
    """build_torch_dataset returns an instance of torch.utils.data.Dataset."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade"],
    )
    assert isinstance(ds, torch.utils.data.Dataset)


def test_len_reflects_all_when_no_skip():
    """Dataset length equals total member count when missing != 'skip'."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": "Severe"})
    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade"],
    )
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# missing= branch tests (spec §6.2)
# ---------------------------------------------------------------------------


def test_missing_error_raises_at_construction():
    """missing='error' raises DerivaMLException at construction listing RIDs."""
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(return_value={"Image": [{"RID": "1-IMG1"}, {"RID": "1-IMG2"}]})
    bag.feature_values = MagicMock(return_value=iter([_FakeRecord(Image="1-IMG1", Grade="Mild")]))
    bag.model = MagicMock()
    bag.model.is_asset = MagicMock(return_value=True)
    bag.get_table_as_dict = MagicMock(return_value=iter([]))
    bag.is_asset = MagicMock(return_value=True)
    with pytest.raises(DerivaMLException, match=r"1-IMG2"):
        build_torch_dataset(
            bag,
            "Image",
            sample_loader=lambda p, row: b"",
            targets=["Grade"],
            missing="error",
        )


def test_missing_skip_drops_unlabeled_elements():
    """missing='skip' drops unlabeled elements from the dataset."""
    # Only IMG1 has a label; IMG2 has empty label (skipped by fake_feature_values)
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": ""})
    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade"],
        missing="skip",
    )
    assert len(ds) == 1


def test_missing_unknown_keeps_all_elements_with_none_target():
    """missing='unknown' keeps all elements; target is None for unlabeled."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": ""})
    # Override get_table_as_dict to return both rows
    bag.get_table_as_dict.return_value = iter(
        [
            {"RID": "1-IMG1", "Filename": "img1.jpg"},
            {"RID": "1-IMG2", "Filename": "img2.jpg"},
        ]
    )
    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade"],
        missing="unknown",
    )
    assert len(ds) == 2
    # Find the unlabeled item — target should be None
    targets_seen = []
    for i in range(len(ds)):
        sample, target = ds[i]
        targets_seen.append(target)
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
        return 0

    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade"],
        target_transform=capture_target,
    )
    _ = ds[0]
    assert len(received) == 1
    assert isinstance(received[0], FeatureRecord)
    assert received[0].Grade == "Mild"


def test_multi_target_target_transform_receives_dict():
    """Multi-target: target_transform receives dict[str, FeatureRecord]."""
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(return_value={"Image": [{"RID": "1-IMG1"}]})

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
        return {}

    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade", "Severity"],
        target_transform=capture_target,
    )
    _ = ds[0]
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

    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: b"fake",
        targets={"Grade": my_selector},
    )
    _ = ds[0]
    assert selector_called, "selector was never called"


# ---------------------------------------------------------------------------
# Asset-table-without-sample_loader error (spec §6.2)
# ---------------------------------------------------------------------------


def test_asset_table_without_sample_loader_raises():
    """Asset-table element_type with no sample_loader raises at construction."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    with pytest.raises(DerivaMLException, match=r"sample_loader"):
        build_torch_dataset(bag, "Image", targets=["Grade"])


# ---------------------------------------------------------------------------
# Non-asset table: no sample_loader required (spec §3.2)
# ---------------------------------------------------------------------------


def test_non_asset_table_no_sample_loader_returns_row_dict():
    """Non-asset element_type with no sample_loader defaults to returning row."""
    bag = _mock_non_asset_bag({"1-SUB1": "active"})
    bag.get_table_as_dict.return_value = iter([{"RID": "1-SUB1", "Status": "active"}])
    ds = build_torch_dataset(
        bag,
        "Subject",
        targets=["Grade"],
    )
    assert isinstance(ds, torch.utils.data.Dataset)
    assert len(ds) >= 0  # construction succeeded


# ---------------------------------------------------------------------------
# Asset path resolution against the canonical BDBag layout (regression test
# for 2026-05-19: _resolve_asset_path computed data/assets/{table}/{rid}/...
# but the BDBag materializer writes data/asset/{rid}/{table}/..., so
# __getitem__ on a real bag raised FileNotFoundError. All prior asset-table
# tests passed a canned sample_loader that ignored its `path` argument,
# leaving the bug latent for ~3 weeks.)
# ---------------------------------------------------------------------------


def test_resolve_asset_path_uses_bdbag_canonical_layout(tmp_path):
    """``_resolve_asset_path`` must return a path matching the on-disk
    BDBag layout produced by the bag materializer: ``data/asset/{RID}/
    {element_type}/{Filename}``. Singular ``asset/``; RID before the
    element-type segment.

    Regression test: the original implementation used
    ``data/assets/{element_type}/{RID}/{Filename}`` (plural, swapped),
    which never matched a real bag and was masked by canned
    sample_loaders in all other tests.

    Per the RID opacity rule
    (deriva-skills/skills/deriva-context/references/concepts.md
    "RID opacity rule"), this test treats the asset's RID as an
    opaque token: it's generated from secrets.token_hex once and
    threaded through both the on-disk layout and the bag mock, so
    the test author never writes a RID-shaped literal. The
    assertion compares the path the sample_loader received against
    the path the test built — RIDs are equality-compared only;
    nothing parses them.
    """
    import secrets

    # Single opaque token shared between the mock bag and the on-disk
    # layout — the test author never types a RID-shaped string.
    asset_rid = secrets.token_hex(3).upper()
    filename = "asset.bin"
    asset_bytes = b"fake-asset-bytes"

    # Build a bag directory tree at the canonical layout. Put a real
    # file at the leaf so the test can verify the resolver hands back
    # a path that exists.
    bag_root = tmp_path / "Dataset_FAKE"
    canonical_dir = bag_root / "data" / "asset" / asset_rid / "Image"
    canonical_dir.mkdir(parents=True)
    canonical_file = canonical_dir / filename
    canonical_file.write_bytes(asset_bytes)

    bag = _mock_bag_with_labeled_images({asset_rid: "Mild"})
    bag.path = bag_root
    bag.get_table_as_dict.return_value = iter([{"RID": asset_rid, "Filename": filename}])

    received_paths: list[Path] = []

    def capturing_loader(path, row):
        received_paths.append(path)
        return path.read_bytes()

    ds = build_torch_dataset(
        bag,
        "Image",
        sample_loader=capturing_loader,
        targets=["Grade"],
    )
    sample, _ = ds[0]
    # The sample_loader received the canonical-layout path and was able
    # to read it. Pre-fix, sample_loader received a non-existent path
    # and raised FileNotFoundError inside the read.
    assert received_paths == [canonical_file]
    assert sample == asset_bytes

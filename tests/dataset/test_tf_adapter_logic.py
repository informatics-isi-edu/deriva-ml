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
- global_shuffle= reorders the RID list; shuffle_seed= makes it reproducible
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
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
        ),
    )
    assert isinstance(ds, tf.data.Dataset)


def test_element_count_reflects_all_when_no_skip():
    """Dataset element count equals total member count when missing != 'skip'."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": "Severe"})
    bag.get_table_as_dict.return_value = iter(
        [
            {"RID": "1-IMG1", "Filename": "img1.jpg"},
            {"RID": "1-IMG2", "Filename": "img2.jpg"},
        ]
    )
    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade"],
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
    bag.list_dataset_members = MagicMock(return_value={"Image": [{"RID": "1-IMG1"}, {"RID": "1-IMG2"}]})
    bag.feature_values = MagicMock(return_value=iter([_FakeRecord(Image="1-IMG1", Grade="Mild")]))
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
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
        ),
    )
    count = sum(1 for _ in ds)
    assert count == 1


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
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
            tf.TensorSpec(shape=(), dtype=tf.int32),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
        return tf.constant(0)

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade", "Severity"],
        target_transform=capture_target,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
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
    bag.get_table_as_dict.return_value = iter(
        [
            {"RID": "1-IMG1", "Filename": "img1.jpg"},
            {"RID": "1-IMG2", "Filename": "img2.jpg"},
        ]
    )

    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0, 2.0, 3.0]),
        targets=None,
        output_signature=None,  # trigger inference
    )

    assert isinstance(ds, tf.data.Dataset)
    # element_spec is now (sample_spec, rid_spec) — the rid is always
    # the last positional value in the yielded tuple.
    sample_spec, rid_spec = ds.element_spec
    assert sample_spec.dtype == tf.float32
    assert sample_spec.shape == (3,)
    assert rid_spec.dtype == tf.string

    # All elements should be present (inference must not drop the first sample)
    count = sum(1 for _ in ds)
    assert count == 2


# ---------------------------------------------------------------------------
# global_shuffle= / shuffle_seed= (issue #362)
#
# The bug these pin: a bag whose members are grouped by class yields
# single-class batches from the head of the stream, which collapses
# training. tf.data's own .shuffle() is a bounded reservoir over *decoded*
# samples and cannot un-group it without buffering the whole dataset, so
# the adapter shuffles the RID list before the generator closes over it.
#
# RIDs here come from the class-grouped mock bag the fixture helper builds,
# never from literals written into an assertion — the tests compare orders
# and set-identity against what they read back out of the bag.
# ---------------------------------------------------------------------------


def _class_grouped_bag(n_per_class: int = 6):
    """Build a mock bag whose members are grouped by class.

    Emulates a dataset assembled one class at a time: every class-A RID
    precedes every class-B RID. This is the ordering that collapses
    training when the stream is consumed un-shuffled.

    Args:
        n_per_class: Number of elements to generate per class.

    Returns:
        Tuple of ``(bag, rids_in_bag_order, labels_by_rid)``. The RID
        strings are generated here and returned, so tests assert against
        values the fixture produced rather than hard-coded literals.
    """
    labels_by_rid: dict[str, str] = {}
    for i in range(n_per_class):
        labels_by_rid[f"1-A{i:03d}"] = "ClassA"
    for i in range(n_per_class):
        labels_by_rid[f"1-B{i:03d}"] = "ClassB"

    rids_in_bag_order = list(labels_by_rid)
    bag = _mock_bag_with_labeled_images(labels_by_rid)
    bag.get_table_as_dict = MagicMock(
        side_effect=lambda *_a, **_kw: iter([{"RID": rid, "Filename": f"{rid}.jpg"} for rid in rids_in_bag_order])
    )
    return bag, rids_in_bag_order, labels_by_rid


def _emitted_rids(ds) -> list[str]:
    """Consume a tf.data.Dataset and return its RIDs in emission order.

    The RID is always the last positional value in a yielded element.
    """
    return [element[-1].numpy().decode() for element in ds]


def _build(bag, **kwargs):
    """Build an unlabeled tf dataset over the mock bag's Image elements.

    Uses ``reachable=False`` so RIDs come from the mocked
    ``list_dataset_members``. The ``reachable=True`` default walks FK
    paths through SQL, which a MagicMock bag cannot serve.
    """
    return build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=None,
        reachable=False,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
        ),
        **kwargs,
    )


def test_global_shuffle_default_preserves_bag_order():
    """Default (global_shuffle=False) iterates the bag's natural order.

    Pins the no-op default: existing callers must see byte-identical
    ordering after this parameter was added.
    """
    bag, rids_in_bag_order, _ = _class_grouped_bag()
    assert _emitted_rids(_build(bag)) == rids_in_bag_order


def test_global_shuffle_reorders_rids():
    """global_shuffle=True changes the emission order."""
    bag, rids_in_bag_order, _ = _class_grouped_bag()
    shuffled = _emitted_rids(_build(bag, global_shuffle=True, shuffle_seed=42))
    assert shuffled != rids_in_bag_order


def test_global_shuffle_preserves_the_rid_set():
    """Shuffling reorders elements without adding, dropping, or duplicating.

    The RID list is the dataset's contents; a shuffle that loses or
    repeats an element would silently change what the model trains on.
    """
    bag, rids_in_bag_order, _ = _class_grouped_bag()
    shuffled = _emitted_rids(_build(bag, global_shuffle=True, shuffle_seed=42))
    assert sorted(shuffled) == sorted(rids_in_bag_order)
    assert len(shuffled) == len(set(shuffled))


def test_global_shuffle_breaks_class_grouping():
    """The motivating case: head-of-stream batches stop being single-class.

    Un-shuffled, the first half of a class-grouped bag is entirely one
    class — the ordering that collapsed VGG-19 training in #362. After
    shuffling, a leading slice must contain both classes.
    """
    bag, rids_in_bag_order, labels_by_rid = _class_grouped_bag(n_per_class=25)

    # Precondition: the un-shuffled stream really is class-grouped, so a
    # passing assertion below reflects the shuffle and not a weak fixture.
    unshuffled = _emitted_rids(_build(bag))
    first_batch_unshuffled = {labels_by_rid[rid] for rid in unshuffled[:10]}
    assert len(first_batch_unshuffled) == 1, "fixture is not class-grouped"

    bag, _, _ = _class_grouped_bag(n_per_class=25)
    shuffled = _emitted_rids(_build(bag, global_shuffle=True, shuffle_seed=42))
    first_batch_shuffled = {labels_by_rid[rid] for rid in shuffled[:10]}
    assert len(first_batch_shuffled) == 2, (
        f"leading slice is still single-class after shuffling: {first_batch_shuffled}"
    )


def test_same_seed_reproduces_the_same_order():
    """Identical shuffle_seed values produce identical orders."""
    bag_a, _, _ = _class_grouped_bag()
    bag_b, _, _ = _class_grouped_bag()
    order_a = _emitted_rids(_build(bag_a, global_shuffle=True, shuffle_seed=1234))
    order_b = _emitted_rids(_build(bag_b, global_shuffle=True, shuffle_seed=1234))
    assert order_a == order_b


def test_different_seeds_produce_different_orders():
    """Different shuffle_seed values produce different orders."""
    bag_a, _, _ = _class_grouped_bag(n_per_class=25)
    bag_b, _, _ = _class_grouped_bag(n_per_class=25)
    order_a = _emitted_rids(_build(bag_a, global_shuffle=True, shuffle_seed=1))
    order_b = _emitted_rids(_build(bag_b, global_shuffle=True, shuffle_seed=2))
    assert order_a != order_b


def test_shuffle_seed_is_independent_of_the_global_rng():
    """Seeded order does not depend on the process-wide `random` state.

    The adapter must use a local `random.Random(seed)`, never the global
    `random` module: data ordering has to be reproducible from
    shuffle_seed alone, regardless of any other random.* call elsewhere
    in the process (a caller seeding `random` for augmentation, say).
    """
    import random as _random

    bag_a, _, _ = _class_grouped_bag()
    _random.seed(0)
    order_a = _emitted_rids(_build(bag_a, global_shuffle=True, shuffle_seed=99))

    bag_b, _, _ = _class_grouped_bag()
    _random.seed(999999)
    _random.random()  # perturb global state further
    order_b = _emitted_rids(_build(bag_b, global_shuffle=True, shuffle_seed=99))

    assert order_a == order_b


def test_shuffle_seed_ignored_when_global_shuffle_false():
    """shuffle_seed has no effect unless global_shuffle is True."""
    bag, rids_in_bag_order, _ = _class_grouped_bag()
    emitted = _emitted_rids(_build(bag, global_shuffle=False, shuffle_seed=42))
    assert emitted == rids_in_bag_order


def test_global_shuffle_does_not_mutate_the_resolved_rid_list():
    """The shuffle copies; it must not reorder the list it was handed.

    build_tf_dataset receives the list resolve_element_rids returned. An
    in-place shuffle would reorder a list the adapter does not own,
    leaking ordering into any caller that holds the same object.
    """
    from deriva_ml.dataset import tf_adapter

    bag, rids_in_bag_order, _ = _class_grouped_bag()
    captured: list[list[str]] = []
    real_resolve = tf_adapter.resolve_element_rids

    def capturing_resolve(*args, **kwargs):
        resolved = real_resolve(*args, **kwargs)
        captured.append(resolved)
        return resolved

    tf_adapter.resolve_element_rids = capturing_resolve
    try:
        _emitted_rids(_build(bag, global_shuffle=True, shuffle_seed=42))
    finally:
        tf_adapter.resolve_element_rids = real_resolve

    assert captured, "resolve_element_rids was never called"
    assert captured[0] == rids_in_bag_order


def test_global_shuffle_with_targets_keeps_rid_label_pairing():
    """Labels follow their RIDs through the shuffle.

    A shuffle that reordered RIDs but not the target lookup would train
    on systematically mismatched labels — silent and catastrophic.
    """
    bag, _, labels_by_rid = _class_grouped_bag()
    ds = build_tf_dataset(
        bag,
        "Image",
        sample_loader=lambda p, row: tf.constant([1.0]),
        targets=["Grade"],
        target_transform=lambda rec: tf.constant(rec.Grade if rec is not None else ""),
        global_shuffle=True,
        shuffle_seed=42,
        reachable=False,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),  # target
            tf.TensorSpec(shape=(), dtype=tf.string),  # rid
        ),
    )
    for _sample, target, rid in ds:
        rid_str = rid.numpy().decode()
        assert target.numpy().decode() == labels_by_rid[rid_str]

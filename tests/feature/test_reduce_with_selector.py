"""Tests for ``deriva_ml.feature.reduce_with_selector`` (audit P1 F-8).

Pre-fix this code lived inline in three call sites:

- ``mixins/feature.py:feature_values``
- ``dataset/dataset.py:Dataset.feature_values``
- ``dataset/dataset_bag.py:DatasetBag.feature_values``

Post-extraction all three delegate to this one helper. Pin its
contract directly so a regression at the helper level is caught
without needing to spin up the full feature_values machinery.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

from types import SimpleNamespace

from deriva_ml.feature import reduce_with_selector


class TestReduceWithSelector:
    """Contract for the group-by-RID + apply-selector reduction."""

    def test_groups_by_target_col_and_yields_one_per_group(self):
        """One record per distinct target RID, selector picks survivor."""
        records = [
            SimpleNamespace(Image="img-1", val=1, RCT="t1"),
            SimpleNamespace(Image="img-1", val=2, RCT="t2"),
            SimpleNamespace(Image="img-2", val=10, RCT="t1"),
        ]
        # Selector picks the highest `val` in each group.
        result = list(reduce_with_selector(records, "Image", selector=lambda group: max(group, key=lambda r: r.val)))

        # Two groups → two survivors. ``Image == "img-1"`` keeps val=2
        # (max), ``"img-2"`` keeps val=10.
        vals_by_img = {r.Image: r.val for r in result}
        assert vals_by_img == {"img-1": 2, "img-2": 10}

    def test_drops_records_with_missing_target(self):
        """Records whose target_col is ``None`` or missing are dropped before grouping."""
        records = [
            SimpleNamespace(Image="img-1", val=1),
            SimpleNamespace(Image=None, val=2),  # explicit None
            SimpleNamespace(val=3),  # attribute missing entirely
        ]
        result = list(reduce_with_selector(records, "Image", selector=lambda group: group[0]))
        assert len(result) == 1
        assert result[0].Image == "img-1"

    def test_selector_returning_none_drops_group(self):
        """A group whose selector returns ``None`` is skipped, not yielded."""
        records = [
            SimpleNamespace(Image="img-1", val=5),
            SimpleNamespace(Image="img-2", val=99),
        ]
        # Selector returns None for any group whose first record has val > 10.
        result = list(
            reduce_with_selector(
                records,
                "Image",
                selector=lambda group: group[0] if group[0].val <= 10 else None,
            )
        )
        assert len(result) == 1
        assert result[0].Image == "img-1"

    def test_empty_input_yields_nothing(self):
        """Empty input → empty output (no calls to selector)."""
        calls = []

        def tracking_selector(group):
            calls.append(group)
            return group[0]

        result = list(reduce_with_selector([], "Image", selector=tracking_selector))
        assert result == []
        assert calls == []

    def test_singleton_group_still_calls_selector(self):
        """A group of one still goes through ``selector`` — never short-circuit.

        This is the behaviour the inline ``dataset_bag.py`` comment
        called out explicitly ("always call selector — never
        short-circuit for single-element groups"). The helper must
        preserve it; if a future "optimisation" skips selector for
        singletons, selectors that side-effect (e.g. logging,
        statistics, validation) would silently break.
        """
        rec = SimpleNamespace(Image="img-1", val=1)
        called = []

        def tracking_selector(group):
            called.append(group)
            return group[0]

        result = list(reduce_with_selector([rec], "Image", selector=tracking_selector))
        assert result == [rec]
        assert len(called) == 1
        assert called[0] == [rec]

    def test_iteration_order_is_first_seen(self):
        """Output order is dict insertion order (first-seen target RID).

        Pinning iteration order is conservative — three call sites
        share the helper, and at least one downstream test might
        rely on deterministic ordering even if it shouldn't.
        Pin the documented behaviour.
        """
        records = [
            SimpleNamespace(Image="img-B", val=1),
            SimpleNamespace(Image="img-A", val=2),
            SimpleNamespace(Image="img-B", val=3),
        ]
        result = list(reduce_with_selector(records, "Image", selector=lambda group: group[0]))
        # img-B first seen at index 0, img-A first seen at index 1.
        assert [r.Image for r in result] == ["img-B", "img-A"]


class TestReduceWithSelectorQualifierColumns:
    """Composite-key grouping for key-qualified features (findings 10 + 12).

    A *key-qualified* feature puts a value FK (e.g. ``Image_Side``) in the
    association table's compound uniqueness key, declaring
    ``(target, *qualifiers)`` — not the target alone — to be the feature's
    identity. The same Subject legitimately has a left-eye row and a
    right-eye row; a selector must reduce *within* each qualifier bucket and
    keep both, not collapse the two into one.

    These tests pin the new ``qualifier_cols`` parameter directly. The empty
    default must be byte-identical to grouping on the target alone — the
    backward-compat invariant for every unqualified feature and every
    existing caller.
    """

    def test_empty_qualifier_cols_is_group_by_target(self):
        """Default ``qualifier_cols=()`` ⇒ identical to group-by-target.

        Two rows for the same Subject with no qualifier collapse to one,
        exactly as before this parameter existed.
        """
        records = [
            SimpleNamespace(Subject="subj-1", RCT="t1"),
            SimpleNamespace(Subject="subj-1", RCT="t2"),
            SimpleNamespace(Subject="subj-2", RCT="t1"),
        ]
        result = list(reduce_with_selector(records, "Subject", selector=lambda group: group[0]))
        assert {r.Subject for r in result} == {"subj-1", "subj-2"}
        assert len(result) == 2

    def test_qualifier_splits_one_target_into_per_qualifier_groups(self):
        """A qualifier in the key keeps both eyes for the same Subject.

        Without ``qualifier_cols`` these two rows (same Subject, different
        ``Image_Side``) would collapse to one — the Chart_Label bug. With
        ``qualifier_cols=["Image_Side"]`` they remain two distinct
        observations.
        """
        records = [
            SimpleNamespace(Subject="subj-1", Image_Side="Left", RCT="t1"),
            SimpleNamespace(Subject="subj-1", Image_Side="Right", RCT="t2"),
        ]
        # selector picks the single member of each (Subject, Image_Side) group.
        result = list(
            reduce_with_selector(
                records,
                "Subject",
                selector=lambda group: group[0],
                qualifier_cols=["Image_Side"],
            )
        )
        sides = {r.Image_Side for r in result}
        assert sides == {"Left", "Right"}
        assert len(result) == 2

    def test_qualifier_reduces_within_each_bucket(self):
        """Selector still reduces redundant records *within* a qualifier bucket.

        Two annotators labelled the Left eye and two the Right eye. The
        selector (newest by RCT) must reduce each eye to one record — two
        survivors total, one per eye — not four and not one.
        """
        records = [
            SimpleNamespace(Subject="s1", Image_Side="Left", RCT="t1"),
            SimpleNamespace(Subject="s1", Image_Side="Left", RCT="t2"),  # newer Left
            SimpleNamespace(Subject="s1", Image_Side="Right", RCT="t1"),
            SimpleNamespace(Subject="s1", Image_Side="Right", RCT="t3"),  # newer Right
        ]
        result = list(
            reduce_with_selector(
                records,
                "Subject",
                selector=lambda group: max(group, key=lambda r: r.RCT),
                qualifier_cols=["Image_Side"],
            )
        )
        by_side = {r.Image_Side: r.RCT for r in result}
        assert by_side == {"Left": "t2", "Right": "t3"}

    def test_multiple_qualifier_columns_form_composite_key(self):
        """The composite key spans all qualifier columns, not just the first."""
        records = [
            SimpleNamespace(Subject="s1", Q1="a", Q2="x"),
            SimpleNamespace(Subject="s1", Q1="a", Q2="y"),
            SimpleNamespace(Subject="s1", Q1="b", Q2="x"),
        ]
        result = list(
            reduce_with_selector(
                records,
                "Subject",
                selector=lambda group: group[0],
                qualifier_cols=["Q1", "Q2"],
            )
        )
        # Three distinct (Subject, Q1, Q2) identities → three survivors.
        assert len(result) == 3

    def test_none_qualifier_value_forms_its_own_bucket(self):
        """A ``None`` qualifier value is a distinct bucket, not dropped.

        Only a missing/``None`` *target* drops a record; a ``None``
        qualifier just groups separately.
        """
        records = [
            SimpleNamespace(Subject="s1", Image_Side="Left"),
            SimpleNamespace(Subject="s1", Image_Side=None),
        ]
        result = list(
            reduce_with_selector(
                records,
                "Subject",
                selector=lambda group: group[0],
                qualifier_cols=["Image_Side"],
            )
        )
        assert len(result) == 2
        assert {r.Image_Side for r in result} == {"Left", None}

    def test_qualifier_cols_accepts_any_iterable(self):
        """``qualifier_cols`` is normalised from any iterable (e.g. a generator)."""
        records = [
            SimpleNamespace(Subject="s1", Image_Side="Left"),
            SimpleNamespace(Subject="s1", Image_Side="Right"),
        ]
        result = list(
            reduce_with_selector(
                records,
                "Subject",
                selector=lambda group: group[0],
                qualifier_cols=(c for c in ["Image_Side"]),
            )
        )
        assert len(result) == 2

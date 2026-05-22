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
        result = list(
            reduce_with_selector(
                records, "Image", selector=lambda group: max(group, key=lambda r: r.val)
            )
        )

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
        result = list(
            reduce_with_selector(records, "Image", selector=lambda group: group[0])
        )
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

        result = list(
            reduce_with_selector([rec], "Image", selector=tracking_selector)
        )
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
        result = list(
            reduce_with_selector(records, "Image", selector=lambda group: group[0])
        )
        # img-B first seen at index 0, img-A first seen at index 1.
        assert [r.Image for r in result] == ["img-B", "img-A"]

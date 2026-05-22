"""Direct unit tests for ``deriva_ml.model.catalog`` + ``deriva_ml_bag_view``.

Closes audit P1 M-13 (find_features dedup), M-15 (is_dataset_rid),
M-36 (DerivaMLBagView write refusals), and M-44 (resolve_rid).
Replaces the empty 6-line ``test_models.py`` stub that audit P3
M-42 flagged.

The ``DerivaModel.is_dataset_rid`` and ``find_features`` tests
require a live catalog (they exercise FK introspection +
``resolve_rid``). The ``DerivaMLBagView`` tests use the
already-existing ``materialized_bag_with_feature`` fixture from
``tests/dataset/conftest.py``.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml.core.exceptions import DerivaMLException


# ---------------------------------------------------------------------------
# DerivaModel.is_dataset_rid (audit M-15)
# ---------------------------------------------------------------------------


class TestIsDatasetRid:
    """Direct unit coverage for ``DerivaModel.is_dataset_rid``.

    The method has a non-obvious shape: live RID → check
    ``Deleted`` column, deleted RID under ``deleted=True`` →
    True, invalid RID → ``DerivaMLException``. Pre-fix all
    seven call sites in ``core/mixins/dataset.py`` +
    ``dataset/dataset.py`` had only indirect coverage.
    """

    def test_returns_true_for_a_live_dataset_rid(self, dataset_test, tmp_path):
        """A non-deleted Dataset RID returns True under either ``deleted`` flag."""
        from deriva_ml import DerivaML

        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        rid = dataset_test.dataset_description.dataset.dataset_rid

        assert ml.model.is_dataset_rid(rid) is True
        assert ml.model.is_dataset_rid(rid, deleted=True) is True

    def test_returns_false_for_a_non_dataset_rid(self, dataset_test, tmp_path):
        """A RID that resolves to a non-Dataset table returns False.

        The demo fixture populates Image rows; pick one and assert
        the predicate says "not a dataset."
        """
        from deriva_ml import DerivaML

        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        # Pull an Image RID from the catalog (any non-Dataset RID
        # would do; Image is reliably present per the demo fixture).
        pb = ml.pathBuilder()
        image_rows = list(
            pb.schemas[ml.default_schema].tables["Image"].entities().fetch()
        )
        assert image_rows, "Demo fixture must seed Image rows."
        non_dataset_rid = image_rows[0]["RID"]

        assert ml.model.is_dataset_rid(non_dataset_rid) is False
        assert ml.model.is_dataset_rid(non_dataset_rid, deleted=True) is False

    def test_raises_for_invalid_rid(self, dataset_test, tmp_path):
        """A fabricated RID that doesn't resolve raises ``DerivaMLException``."""
        from deriva_ml import DerivaML

        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        with pytest.raises(DerivaMLException, match=r"Invalid RID"):
            ml.model.is_dataset_rid("Z" * 12)


# ---------------------------------------------------------------------------
# DerivaModel.find_features no-arg dedup (audit M-13)
# ---------------------------------------------------------------------------


class TestFindFeaturesDedupSmoke:
    """Smoke test for the no-arg ``find_features`` dedup walk.

    The detailed dedup-by-qname assertions live in
    ``tests/feature/test_features.py::test_find_features`` (PR
    #196). This smoke test lives in ``tests/model/`` so a
    reader looking next to ``catalog.py`` can find coverage of
    the critical-correctness branch at ``catalog.py:603-645``.
    """

    def test_no_arg_find_features_returns_each_feature_once(
        self, dataset_test, tmp_path
    ):
        """``model.find_features()`` (no-arg) dedups by feature_table."""
        from deriva_ml import DerivaML

        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        features = list(ml.model.find_features())
        # Dedup invariant: every feature_table qualified name
        # appears at most once.
        qnames = [
            f"{f.feature_table.schema.name}.{f.feature_table.name}"
            for f in features
        ]
        assert len(qnames) == len(set(qnames)), (
            f"find_features() returned duplicates by feature_table qname: "
            f"{qnames}. The dedup at catalog.py:603-645 has regressed."
        )


# ---------------------------------------------------------------------------
# DerivaMLBagView write-operation refusals (audit M-36) + resolve_rid (M-44)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="bag-view tests require a live catalog to materialize the bag",
)
class TestDerivaMLBagViewWriteRefusals:
    """Pin the three write-operation refusals on ``DerivaMLBagView``.

    Bags are immutable snapshots. ``create_dataset``,
    ``pathBuilder``, and ``catalog_snapshot`` exist purely to
    raise ``DerivaMLException`` and document the read-only
    contract. Pre-fix nothing asserted that the refusal
    actually fires; a future change that silently delegated to
    ``_database_model`` (instead of raising) would ship
    unnoticed.
    """

    def test_create_dataset_refuses(self, materialized_bag_with_feature):
        """``DerivaMLBagView.create_dataset`` always raises."""
        bag_view = materialized_bag_with_feature.bag._catalog
        with pytest.raises(DerivaMLException, match=r"Cannot create datasets"):
            bag_view.create_dataset(description="should not be allowed")

    def test_path_builder_refuses(self, materialized_bag_with_feature):
        """``DerivaMLBagView.pathBuilder`` always raises."""
        bag_view = materialized_bag_with_feature.bag._catalog
        with pytest.raises(DerivaMLException, match=r"pathBuilder.*not available"):
            bag_view.pathBuilder()

    def test_catalog_snapshot_refuses(self, materialized_bag_with_feature):
        """``DerivaMLBagView.catalog_snapshot`` always raises."""
        bag_view = materialized_bag_with_feature.bag._catalog
        with pytest.raises(DerivaMLException, match=r"catalog_snapshot.*not available"):
            bag_view.catalog_snapshot("some-version")


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="bag-view tests require a live catalog to materialize the bag",
)
class TestDerivaMLBagViewResolveRid:
    """Pin ``DerivaMLBagView.resolve_rid`` (audit M-44).

    The method is part of the ``DerivaMLCatalog`` protocol
    surface that the bag view implements. Returns a dict with
    ``RID`` and ``version`` keys when the RID is in the bag;
    raises ``DerivaMLDatasetNotFound`` (via
    ``_database_model.rid_lookup``) when the RID isn't found.
    """

    def test_resolves_known_dataset_rid_in_bag(self, materialized_bag_with_feature):
        """A dataset RID present in the bag returns the documented dict shape."""
        bag = materialized_bag_with_feature.bag
        result = bag._catalog.resolve_rid(bag.dataset_rid)
        assert result["RID"] == bag.dataset_rid
        assert "version" in result, (
            f"resolve_rid contract: must return {{'RID': ..., 'version': ...}}; "
            f"got {result!r}"
        )

    def test_raises_for_rid_not_in_bag(self, materialized_bag_with_feature):
        """An RID absent from the bag raises ``DerivaMLDatasetNotFound``.

        The bag view's ``resolve_rid`` delegates to
        ``DatabaseModel.rid_lookup``, which raises the typed
        ``DerivaMLDatasetNotFound`` per the M-23 commit in this
        sweep. We catch on the base class to stay robust to
        future hierarchy changes, but ``DerivaMLDatasetNotFound``
        is the specific class.
        """
        from deriva_ml.core.exceptions import DerivaMLDatasetNotFound

        bag = materialized_bag_with_feature.bag
        with pytest.raises(DerivaMLDatasetNotFound):
            bag._catalog.resolve_rid("Z" * 12)

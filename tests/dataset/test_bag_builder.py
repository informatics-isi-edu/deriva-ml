"""Tests for :class:`deriva_ml.dataset.bag_builder.DatasetBagBuilder`.

Two layers:

* :class:`TestSpecSmoke` — light live-catalog tests confirming the
  three public methods (:meth:`generate_dataset_download_spec`,
  :meth:`generate_dataset_download_annotations`,
  :meth:`aggregate_queries`) run end-to-end against
  ``catalog_with_datasets``. These catch construction-level
  errors that would otherwise surface only as download failures.

* :class:`TestAnchorsAndPolicy` — exercises the bag-pipeline-shaped
  helpers (:meth:`anchors_for`, :meth:`build_policy`) without
  driving a full export.

The bag-content equivalence harness that gated the cutover from
``CatalogGraph`` to :class:`CatalogBagBuilder` is **gone** —
it served its purpose (verified row-set + asset (RID, MD5)
equivalence on ``catalog_with_datasets``), then was deleted
along with ``CatalogGraph`` itself. See
``docs/design/dataset-bag-cutover-2026-05.md`` in deriva-ml for
the design and the verified-equivalence record.
"""

from __future__ import annotations

import pytest

from deriva.bag.anchors import RIDAnchor
from deriva.bag.traversal import FKTraversalPolicy, VocabExport
from deriva_ml.dataset.bag_builder import DatasetBagBuilder


# ---------------------------------------------------------------------------
# Spec smoke tests — confirm the new spec generator is callable
# ---------------------------------------------------------------------------


class TestSpecSmoke:
    """Light tests confirming the new spec generator runs end-to-end.

    Not equivalence tests — those live in :class:`TestBagEquivalence`.
    These exercise the spec/annotation methods enough to catch
    construction-level errors (missing imports, attribute access)
    that the harness would otherwise only surface as a download
    failure.
    """

    def test_spec_runs_without_error(
        self, catalog_with_datasets
    ) -> None:
        """``generate_dataset_download_spec`` produces a dict with the right keys."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)

        spec = DatasetBagBuilder(
            ml_instance=ml
        ).generate_dataset_download_spec(dataset)
        assert "env" in spec
        assert "bag" in spec
        assert "catalog" in spec
        assert spec["bag"]["bag_name"] == "Dataset_{RID}"
        assert spec["env"]["RID"] == "{RID}"
        assert "query_processors" in spec["catalog"]

    def test_annotations_run_without_error(
        self, catalog_with_datasets
    ) -> None:
        """``generate_dataset_download_annotations`` produces a dict with the right keys."""
        ml, _ = catalog_with_datasets
        ann = DatasetBagBuilder(
            ml_instance=ml
        ).generate_dataset_download_annotations()
        from deriva.core.utils.core_utils import tag as deriva_tags

        assert deriva_tags.export_fragment_definitions in ann
        assert deriva_tags.visible_foreign_keys in ann
        assert deriva_tags.export_2019 in ann

    def test_aggregate_queries_runs_without_error(
        self, catalog_with_datasets
    ) -> None:
        """``aggregate_queries`` produces a non-empty dict for a real dataset."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        result = DatasetBagBuilder(ml_instance=ml).aggregate_queries(dataset)
        # Should produce at least one entry for the Dataset table.
        assert "Dataset" in result
        for table_name, entries in result.items():
            for dp, pb_table, is_asset in entries:
                assert dp is not None
                assert pb_table is not None
                assert isinstance(is_asset, bool)


# ---------------------------------------------------------------------------
# Bag-pipeline-shaped helpers
# ---------------------------------------------------------------------------


class TestAnchorsAndPolicy:
    """The ``anchors_for`` + ``build_policy`` helpers expose the bag-pipeline form.

    These methods are the ADR-0006 surface — a future
    ``Dataset.download_dataset_bag`` rewrite that calls
    :class:`CatalogBagBuilder` directly will pull anchors + policy
    from here.
    """

    def test_anchors_for_dataset_no_children(
        self, catalog_with_datasets
    ) -> None:
        """A dataset with no nested children yields a single anchor."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        # Pick the first dataset that has no children; if all
        # datasets in the fixture have children, skip.
        candidate = None
        for ds_row in datasets:
            ds = ml.lookup_dataset(ds_row.dataset_rid)
            if not list(ds.list_dataset_children()):
                candidate = ds
                break
        if candidate is None:
            pytest.skip(
                "Need a dataset with no children to test the single-"
                "anchor path; fixture has only nested datasets."
            )

        builder = DatasetBagBuilder(ml_instance=ml)
        anchors = builder.anchors_for(candidate)
        assert len(anchors) == 1
        anchor = anchors[0]
        assert isinstance(anchor, RIDAnchor)
        assert anchor.table == "Dataset"
        assert anchor.rids == [candidate.dataset_rid]

    def test_build_policy_default(self, catalog_with_datasets) -> None:
        """Default policy enables full-vocab export."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        builder = DatasetBagBuilder(ml_instance=ml)
        policy = builder.build_policy(dataset)
        assert isinstance(policy, FKTraversalPolicy)
        assert policy.vocab_export is VocabExport.FULL

    def test_build_policy_respects_vocab_export_override(
        self, catalog_with_datasets
    ) -> None:
        """``vocab_export=REFERENCED_ONLY`` flows through to the policy."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        builder = DatasetBagBuilder(ml_instance=ml)
        policy = builder.build_policy(
            dataset, vocab_export=VocabExport.REFERENCED_ONLY
        )
        assert policy.vocab_export is VocabExport.REFERENCED_ONLY

    def test_build_policy_includes_user_exclude_tables(
        self, catalog_with_datasets
    ) -> None:
        """User-supplied ``exclude_tables`` make it into the policy."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)

        # Pick a real table from the catalog so the exclusion is
        # actually applied (the schema lookup needs to resolve it).
        # Dataset is always present; using it is harmless because
        # the user-exclude logic only flags it, not removes Dataset
        # from the walk (the anchor still pins it).
        builder = DatasetBagBuilder(
            ml_instance=ml, exclude_tables={"Dataset"}
        )
        policy = builder.build_policy(dataset)
        # We don't assert the exact set — Dataset_Version etc. may
        # also land in policy.exclude_tables via the empty-
        # association filter. We just assert the user's input made
        # it in.
        assert any(
            table_name == "Dataset"
            for _schema, table_name in policy.exclude_tables
        )

"""Tests for :class:`deriva_ml.dataset.bag_builder.DatasetBagBuilder`.

The headline test is *spec equivalence*: the new
:class:`DatasetBagBuilder` must produce byte-identical export
specs to the legacy :class:`CatalogGraph` for the same
``(catalog, dataset)`` input. That's the safety net that lets us
cut :meth:`Dataset.download_dataset_bag` over to the new class
in a follow-up commit without regression.

The equivalence test requires a live catalog (the spec embeds
the catalog's host URL, snapshot, and reachable-table set), so
it's gated on the ``catalog_manager`` fixture and skipped when
``DERIVA_HOST`` isn't set.

Two thinner unit tests verify the bag-pipeline-shaped helpers
(:meth:`DatasetBagBuilder.anchors_for`,
:meth:`DatasetBagBuilder.build_policy`) — they exercise the
dataset-anchor + policy logic without needing the full export-
engine spec.
"""

from __future__ import annotations

import pytest

from deriva.bag.anchors import RIDAnchor
from deriva.bag.traversal import FKTraversalPolicy, VocabExport
from deriva_ml.dataset.bag_builder import DatasetBagBuilder


# ---------------------------------------------------------------------------
# Spec equivalence — gated on catalog_manager fixture
# ---------------------------------------------------------------------------


class TestSpecEquivalence:
    """``DatasetBagBuilder`` must emit specs equivalent to ``CatalogGraph``.

    These tests are the migration's load-bearing safety net.
    When :meth:`Dataset.download_dataset_bag` is rewired to call
    ``DatasetBagBuilder`` instead of ``CatalogGraph`` (a follow-up
    commit), byte-equivalence ensures every existing cache,
    MINID, and downstream consumer keeps working.
    """

    def test_spec_byte_equivalent_to_catalog_graph(
        self, catalog_with_datasets
    ) -> None:
        """The two spec generators emit identical dicts for the same dataset.

        ``catalog_with_datasets`` provides a populated catalog
        with at least one dataset; we build both spec generators
        over it and compare the outputs verbatim.
        """
        from deriva_ml.dataset.catalog_graph import CatalogGraph

        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip(
                "Need at least one dataset in the catalog to test spec "
                "equivalence; catalog_with_datasets returned none."
            )
        dataset_rid = datasets[0].dataset_rid
        dataset = ml.lookup_dataset(dataset_rid)

        legacy = CatalogGraph(ml_instance=ml).generate_dataset_download_spec(
            dataset
        )
        new = DatasetBagBuilder(
            ml_instance=ml
        ).generate_dataset_download_spec(dataset)

        assert legacy == new, (
            "DatasetBagBuilder spec drifted from CatalogGraph spec. "
            "This breaks the migration safety net — Dataset."
            "download_dataset_bag relies on byte equivalence to cut "
            "over without regression."
        )

    def test_annotations_byte_equivalent(
        self, catalog_with_datasets
    ) -> None:
        """Chaise export annotations also match the legacy form.

        The annotations are written to the Dataset table so
        browser-based downloads from Chaise produce the same
        bags. A mismatch here would mean Chaise downloads diverge
        from Python-API downloads after the cutover.
        """
        from deriva_ml.dataset.catalog_graph import CatalogGraph

        ml, _ = catalog_with_datasets
        legacy = CatalogGraph(
            ml_instance=ml
        ).generate_dataset_download_annotations()
        new = DatasetBagBuilder(
            ml_instance=ml
        ).generate_dataset_download_annotations()
        assert legacy == new

    def test_spec_with_exclude_tables(self, catalog_with_datasets) -> None:
        """``exclude_tables`` flows through both generators identically."""
        from deriva_ml.dataset.catalog_graph import CatalogGraph

        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        # Pick a table name guaranteed not to exist so the
        # exclusion is a no-op but exercises the parameter
        # plumbing. A real test against a deep-join table would
        # require a richer fixture.
        excluded = {"NoSuchTable"}

        legacy = CatalogGraph(
            ml_instance=ml, exclude_tables=excluded
        ).generate_dataset_download_spec(dataset)
        new = DatasetBagBuilder(
            ml_instance=ml, exclude_tables=excluded
        ).generate_dataset_download_spec(dataset)
        assert legacy == new

    def test_spec_with_s3_bucket(self, catalog_with_datasets) -> None:
        """When ``s3_bucket`` is set, post_processors match."""
        from deriva_ml.dataset.catalog_graph import CatalogGraph

        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        if not datasets:
            pytest.skip("Need at least one dataset.")
        dataset = ml.lookup_dataset(datasets[0].dataset_rid)
        bucket = "s3://test-bucket-name"

        legacy = CatalogGraph(
            ml_instance=ml, s3_bucket=bucket, use_minid=True
        ).generate_dataset_download_spec(dataset)
        new = DatasetBagBuilder(
            ml_instance=ml, s3_bucket=bucket, use_minid=True
        ).generate_dataset_download_spec(dataset)
        assert legacy == new


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

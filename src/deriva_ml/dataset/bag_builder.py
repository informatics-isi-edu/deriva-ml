"""Deriva-ml-domain bag builder for datasets.

:class:`DatasetBagBuilder` is the deriva-ml-domain wrapper over
:class:`deriva.bag.catalog_builder.CatalogBagBuilder`, per ADR-0006.
``CatalogBagBuilder`` is deriva-ml-agnostic — it walks any
ERMrest catalog from a list of :class:`Anchor`\\s following an
:class:`FKTraversalPolicy`. ``DatasetBagBuilder`` adds the four
dataset-specific concerns that today's
:class:`~deriva_ml.dataset.catalog_graph.CatalogGraph` baked in:

1. **Association filtering by member element types.** Only
   include ``Dataset_X`` association tables whose target element
   type ``X`` actually has members in this dataset. Skips empty
   associations and prunes paths that would otherwise traverse
   them. (See ``CatalogGraph._collect_paths`` for the original.)
2. **Feature tables per element type.** For each member element
   type, the dataset's feature tables are added to the walk's
   path set even though the generic walk wouldn't reach them.
3. **Nested datasets recursively.** Each dataset member's child
   dataset gets its own walk, up to a computed nesting depth.
4. **Vocabulary export as standalone queries.** Vocabulary
   tables are exported as full-table queries (every term), not
   joined through the element-type graph. Paths ending in vocab
   tables are pruned from the main walk.

The result of :meth:`generate_dataset_download_spec` is byte-
compatible with :meth:`CatalogGraph.generate_dataset_download_spec`
so the existing :meth:`Dataset.download_dataset_bag` machinery
(three-tier caching, MINID minting, materialization) works
unchanged when (eventually) wired up.

Scope note: this commit **adds the class**; it does **not yet
rewire** ``Dataset.download_dataset_bag`` to use it. The cutover
from ``CatalogGraph`` to ``DatasetBagBuilder`` is a follow-up
once live-catalog tests confirm byte-for-byte spec equivalence
against today's behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from deriva.bag.anchors import Anchor, RIDAnchor
from deriva.bag.traversal import FKTraversalPolicy, VocabExport
from deriva.core.ermrest_model import Table

from deriva_ml.core.constants import RID
from deriva_ml.dataset.catalog_graph import CatalogGraph
from deriva_ml.interfaces import DatasetLike, DerivaMLCatalog

logger = logging.getLogger(__name__)


class DatasetBagBuilder:
    """Build a download spec for a deriva-ml dataset.

    Args:
        ml_instance: The DerivaML catalog handle. Used for the
            element-type / feature-table / nested-dataset lookups
            that the wrapper needs to specialize the generic
            :class:`CatalogBagBuilder`.
        s3_bucket: S3 bucket URL for MINID minting. ``None``
            disables the MINID post-processors.
        use_minid: Whether to enable the MINID service. Only
            effective when ``s3_bucket`` is provided.
        exclude_tables: Optional set of bare table names to
            exclude from FK path traversal. Maps to
            :attr:`FKTraversalPolicy.exclude_tables` after
            schema-qualification.

    Example:
        Build a download spec for a dataset::

            >>> from deriva_ml.dataset.bag_builder import DatasetBagBuilder
            >>> builder = DatasetBagBuilder(  # doctest: +SKIP
            ...     ml_instance=ml,
            ...     s3_bucket="s3://my-bucket",
            ... )
            >>> spec = builder.generate_dataset_download_spec(dataset)  # doctest: +SKIP
    """

    def __init__(
        self,
        ml_instance: DerivaMLCatalog,
        s3_bucket: str | None = None,
        use_minid: bool = True,
        exclude_tables: set[str] | None = None,
    ):
        self._ml_instance = ml_instance
        self._s3_bucket = s3_bucket
        self._use_minid = use_minid
        self._exclude_tables = exclude_tables or set()

        # Today's CatalogGraph already encodes every dataset-
        # specific decision we need (association filtering,
        # feature-table inclusion, nested-dataset recursion,
        # vocabulary handling). Re-implementing them on top of
        # CatalogBagBuilder would risk subtle divergences in the
        # generated spec; instead this wrapper *uses* CatalogGraph
        # to compute the spec and exposes the same surface
        # (generate_dataset_download_spec / generate_dataset_download_annotations).
        # The semantic intent of DatasetBagBuilder per ADR-0006 is
        # captured by the public API + the build_policy / anchors_for
        # helpers below, which are what future deriva.bag-aware
        # callers would use. The cutover to a fully CatalogBagBuilder-
        # backed implementation is a follow-up once live-catalog
        # tests confirm spec equivalence.
        self._catalog_graph = CatalogGraph(
            ml_instance=ml_instance,
            s3_bucket=s3_bucket,
            use_minid=use_minid,
            exclude_tables=exclude_tables,
        )

    # ------------------------------------------------------------------
    # Public surface (spec equivalent to CatalogGraph)
    # ------------------------------------------------------------------

    def generate_dataset_download_spec(
        self, dataset: DatasetLike
    ) -> dict[str, Any]:
        """Return the download spec for a specific dataset.

        Byte-equivalent to
        :meth:`CatalogGraph.generate_dataset_download_spec`. The
        spec is consumed by the deriva-py export engine
        (``GenericDownloader``) and includes:

        - Top-level ``env``, ``bag``, and ``catalog`` keys.
        - Optional ``post_processors`` for S3 upload + MINID
          minting when ``s3_bucket`` is configured.
        - One ``csv`` processor per reached FK path, plus one
          ``fetch`` processor per asset table (for byte
          downloads).
        - Full-table queries for vocabulary tables (per ADR-0006
          ``vocab_export=FULL`` semantics — vocabs are exported
          as standalone queries, not joined through the FK graph).

        Args:
            dataset: The dataset to generate the spec for. Must
                expose ``dataset_rid``.

        Returns:
            The export-engine spec dict.
        """
        return self._catalog_graph.generate_dataset_download_spec(dataset)

    def generate_dataset_download_annotations(self) -> dict[str, Any]:
        """Return the Chaise export annotations for the Dataset table.

        Byte-equivalent to
        :meth:`CatalogGraph.generate_dataset_download_annotations`.
        Used to write the export configuration into the catalog so
        browser-based downloads from Chaise produce the same bags
        the Python API does.
        """
        return self._catalog_graph.generate_dataset_download_annotations()

    def aggregate_queries(
        self,
        dataset: DatasetLike | None = None,
    ) -> dict[str, list[Any]]:
        """Return per-target-table datapaths for size estimation.

        Byte-equivalent to
        :meth:`CatalogGraph._aggregate_queries`. Returns a dict
        keyed by terminal table name; each value is a list of
        ``(datapath, target_pb_table, is_asset)`` tuples — one
        per FK path that reaches the target table from the
        dataset. Used by :meth:`Dataset.estimate_bag_size` to
        compute row counts via RID-union semantics before
        deciding whether to materialize.

        Args:
            dataset: Optional dataset to filter paths to. ``None``
                aggregates across every dataset reachable in the
                catalog.

        Returns:
            ``{target_table_name: [(datapath, pb_table, is_asset),
            ...]}``.
        """
        # ``_aggregate_queries`` is a private CatalogGraph method;
        # using it through the wrapper preserves byte equivalence
        # while moving callers off the CatalogGraph import. When
        # CatalogGraph is eventually retired in favor of a fully
        # CatalogBagBuilder-backed implementation, this method
        # will route through the same downstream as today's
        # CatalogGraph (the export engine has the same datapath
        # surface).
        return self._catalog_graph._aggregate_queries(dataset)

    # ------------------------------------------------------------------
    # ADR-0006 bag-pipeline-shaped helpers
    # ------------------------------------------------------------------

    def anchors_for(self, dataset: DatasetLike) -> list[Anchor]:
        """Return the :class:`Anchor` list for a dataset's bag walk.

        The dataset's root row is the primary anchor; each nested
        child dataset becomes an additional :class:`RIDAnchor`.
        The traversal depth is bounded by
        :meth:`DatasetLike.dataset_children` chains.

        This is the bag-pipeline-shaped representation of "what
        rows does the walker start from for this dataset?". When
        :class:`Dataset.download_dataset_bag` is rewired to use
        :class:`CatalogBagBuilder` directly, this method will
        feed into the new flow.

        Args:
            dataset: The dataset to anchor at.

        Returns:
            A list of :class:`Anchor` objects. The first is the
            dataset's own RID anchor; subsequent entries cover
            nested children up to the configured nesting depth.
        """
        # Dataset row itself is the primary anchor.
        anchors: list[Anchor] = [
            RIDAnchor(table="Dataset", rids=[dataset.dataset_rid])
        ]

        # Nested children — same recursion CatalogGraph uses for
        # path collection. Walk one level at a time so the
        # anchor list documents the dataset's structure.
        for child in self._iter_descendant_rids(dataset):
            anchors.append(RIDAnchor(table="Dataset", rids=[child]))

        return anchors

    def build_policy(
        self,
        dataset: DatasetLike,
        *,
        vocab_export: VocabExport = VocabExport.FULL,
    ) -> FKTraversalPolicy:
        """Return the :class:`FKTraversalPolicy` for this dataset.

        The policy encodes the dataset-specific traversal
        constraints in the bag-pipeline vocabulary:

        - :attr:`FKTraversalPolicy.exclude_tables` includes the
          association tables whose target element type has no
          members in *this* dataset (the dataset-specific
          association filter from
          :meth:`CatalogGraph._collect_paths`).
        - :attr:`FKTraversalPolicy.vocab_export` defaults to
          :attr:`VocabExport.FULL` so vocab terms are exported
          completely (today's
          :meth:`CatalogGraph._export_vocabulary` behavior).
        - Feature tables reach the walk via FK-following from
          member element rows; no separate field is needed.

        Args:
            dataset: The dataset to derive the policy for.
            vocab_export: Override the vocabulary-export mode.
                Default is :attr:`VocabExport.FULL` (every term);
                pass :attr:`VocabExport.REFERENCED_ONLY` to limit
                to terms reached by the FK walk.

        Returns:
            An :class:`FKTraversalPolicy` ready to hand to
            :class:`CatalogBagBuilder`.
        """
        exclude_tables = self._exclude_empty_associations(dataset)
        # User-supplied exclusions take precedence over our derived
        # ones (e.g., to skip a deep-join table that times out).
        for bare in self._exclude_tables:
            for schema_name in self._ml_instance.model.schemas:
                schema = self._ml_instance.model.schemas[schema_name]
                if bare in schema.tables:
                    exclude_tables.add((schema_name, bare))

        return FKTraversalPolicy(
            exclude_tables=exclude_tables,
            vocab_export=vocab_export,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exclude_empty_associations(
        self, dataset: DatasetLike
    ) -> set[tuple[str, str]]:
        """Return ``{(schema, table)}`` for associations with no members.

        Mirrors the dataset-specific association filter inside
        :meth:`CatalogGraph._collect_paths`: for each
        ``Dataset_X`` association table, include it in the walk
        only when the dataset has at least one member of element
        type X (or when the association links to a vocabulary
        table — those carry dataset metadata and must always be
        included).
        """
        model = self._ml_instance.model
        ml_schema_name = self._ml_instance.ml_schema
        dataset_table = model.schemas[ml_schema_name].tables["Dataset"]

        # Element types that have members in this dataset.
        dataset_obj = self._ml_instance.lookup_dataset(
            dataset.dataset_rid
        )
        member_element_types: set[Table] = {
            model.name_to_table(name)
            for name, members in dataset_obj.list_dataset_members().items()
            if members
        }

        # Every vocabulary table in any schema — associations into
        # vocabularies always come along for the ride.
        vocab_tables: set[Table] = {
            table
            for schema in model.schemas.values()
            for table in schema.tables.values()
            if model.is_vocabulary(table)
        }

        # Walk every Dataset_X association. The association links
        # to one or more "other" tables via ``other_fkeys``; include
        # the association only when at least one of those tables
        # is a member element type or a vocabulary.
        excluded: set[tuple[str, str]] = set()
        for assoc in dataset_table.find_associations():
            assoc_table = assoc.table
            links_to_member = any(
                fk.pk_table in member_element_types
                for fk in assoc.other_fkeys
            )
            links_to_vocab = any(
                fk.pk_table in vocab_tables
                for fk in assoc.other_fkeys
            )
            if not (links_to_member or links_to_vocab):
                excluded.add(
                    (assoc_table.schema.name, assoc_table.name)
                )
        return excluded

    def _iter_descendant_rids(
        self, dataset: DatasetLike
    ) -> Iterable[RID]:
        """Yield every descendant Dataset RID, depth-first.

        Walks ``DatasetLike.list_dataset_children`` recursively.
        Used by :meth:`anchors_for` to build the anchor list. The
        order matches what
        :meth:`CatalogGraph._collect_paths` traverses so anchor
        provenance round-trips cleanly.
        """
        dataset_obj = self._ml_instance.lookup_dataset(
            dataset.dataset_rid
        )
        for child in dataset_obj.list_dataset_children():
            yield child.dataset_rid
            child_proxy = self._ml_instance.lookup_dataset(
                child.dataset_rid
            )
            yield from self._iter_descendant_rids(child_proxy)


__all__ = ["DatasetBagBuilder"]

"""Deriva-ml-domain bag builder for datasets.

:class:`DatasetBagBuilder` is the deriva-ml-domain wrapper over
:class:`deriva.bag.catalog_builder.CatalogBagBuilder`, per ADR-0006.
``CatalogBagBuilder`` is deriva-ml-agnostic — it walks any
ERMrest catalog from a list of :class:`Anchor`\\s following an
:class:`FKTraversalPolicy`. ``DatasetBagBuilder`` adds the four
dataset-specific concerns to the generic builder:

1. **Association filtering by member element types.** Only
   include ``Dataset_X`` association tables whose target element
   type ``X`` actually has members in this dataset. Skips empty
   associations; pruning paths that traverse them falls out of
   the policy's ``exclude_tables``.
2. **Feature tables per element type.** Reached naturally by
   inbound-FK walking from member element rows; no special
   handling in the wrapper (the generic walker covers it).
3. **Nested datasets recursively.** Each member's child dataset
   becomes its own :class:`RIDAnchor` in :meth:`anchors_for`.
4. **Vocabulary export in full.** Set via
   :attr:`FKTraversalPolicy.vocab_export` =
   :attr:`VocabExport.FULL`. The walker emits one full-table CSV
   processor per reached vocab.

Three public surfaces:

* :meth:`generate_dataset_download_spec(dataset)` — the runtime
  export-engine spec for downloading this dataset. Drives a
  :class:`CatalogBagBuilder` scoped to the dataset's RID and
  overlays the dataset-specific top-level keys (``env``,
  ``bag.bag_name = "Dataset_{RID}"``, the preamble query
  processors that parameterize the template, optional MINID
  post-processors).
* :meth:`generate_dataset_download_annotations()` — the static
  Chaise export annotation written to the Dataset table at
  catalog setup. Drives a :class:`CatalogBagBuilder` scoped to
  the whole ``Dataset`` table (no specific RID) and consumes its
  symbolic path set (via :meth:`CatalogBagBuilder.iter_reached_paths`)
  through a deriva-ml-side annotation writer.
* :meth:`aggregate_queries(dataset=None)` — live-catalog
  datapaths per FK route for size estimation / drift detection.
  Drives a :class:`CatalogBagBuilder` (per-dataset or
  catalog-wide) and returns its
  :meth:`~CatalogBagBuilder.iter_table_datapaths` output.

All three share the same walker — ``CatalogBagBuilder._compute_reached_tables``
— so the "the drift walk is the bag walk" invariant from
CONTEXT.md is preserved by construction.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, NamedTuple

from deriva.bag.anchors import Anchor, RIDAnchor, TableAnchor
from deriva.bag.catalog_builder import CatalogBagBuilder
from deriva.bag.traversal import FKTraversalPolicy, VocabExport
from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import tag as deriva_tags

from deriva_ml.core.constants import INTENTIONAL_FK_CYCLES, PROVENANCE_TERMINAL_TABLES, RID
from deriva_ml.core.logging_config import get_logger
from deriva_ml.interfaces import DatasetLike, DerivaMLCatalog

# Import datapath via importlib to avoid shadowing by local 'deriva.py' files
# (mirrors dataset.py's pattern).
datapath = importlib.import_module("deriva.core.datapath")

logger = get_logger(__name__)


class RidSetComputation(NamedTuple):
    """Result of :meth:`DatasetBagBuilder._compute_rid_sets`.

    Bundles the per-table RID sets (for Format-B bag generation) with the
    auxiliary reachability data the estimate consumes — all from one walk.

    Attributes:
        rid_sets: ``{(schema, table): [RID,...]}`` for non-vocab reached
            tables — the shape ``CatalogBagBuilder(rid_sets=...)`` consumes
            (vocab excluded, lists sorted).
        rids_by_table: bare-name ``{table: set(RID)}`` (row counts).
        asset_lengths_by_table: ``{table: {RID: Length}}`` (asset bytes).
        fetched_rows: ``{(schema, table): rows}`` (CSV-byte sampling source).
        sample_rows_by_table: ``{table: [row,...]}`` derived sample.
        asset_tables: bare table names that are asset tables.
    """

    rid_sets: dict[tuple[str, str], list[str]]
    rids_by_table: dict[str, set[str]]
    asset_lengths_by_table: dict[str, dict[str, int]]
    fetched_rows: dict[tuple[str, str], list[dict]]
    sample_rows_by_table: dict[str, list[dict]]
    asset_tables: set[str]


def _rid_sets_from_reachability(
    reached: dict[tuple[str, str], Any],
    rids_by_table: dict[str, set[str]],
    vocab_tables: set[tuple[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """Map compute_reachability output to the upstream rid_sets shape.

    ``compute_reachability`` returns ``rids_by_table`` keyed by bare table
    name; ``CatalogBagBuilder(rid_sets=...)`` wants ``{(schema, table):
    [RID,...]}``. This re-keys using ``reached``'s ``(schema, table)`` keys,
    drops vocab tables (the upstream rid-set branch skips them -- vocab keeps
    its full/per-path query), and sorts each RID list for deterministic specs
    (sets are not JSON-serializable; the spec must be reproducible).

    Args:
        reached: ``{(schema, table): [fk_path, ...]}`` -- the reached-paths map
            (its keys enumerate the tables to consider).
        rids_by_table: ``{table_name: set(RID)}`` from compute_reachability.
        vocab_tables: ``(schema, table)`` pairs that are vocabulary tables.

    Returns:
        ``{(schema, table): [RID, ...]}`` for every non-vocab reached table.

    Example:
        >>> _rid_sets_from_reachability(
        ...     {("S", "T"): [()], ("S", "V"): [()]},
        ...     {"T": {"b", "a"}},
        ...     {("S", "V")},
        ... )
        {('S', 'T'): ['a', 'b']}
    """
    result: dict[tuple[str, str], list[str]] = {}
    for key in reached:
        if key in vocab_tables:
            continue
        table_name = key[1]
        result[key] = sorted(rids_by_table.get(table_name, set()))
    return result


class _SnapshotAwareCatalogBagBuilder(CatalogBagBuilder):
    """``CatalogBagBuilder`` that honors the catalog's snapshot in exports.

    Upstream's :class:`CatalogBagBuilder._run_export` passes
    ``str(self.catalog.catalog_id)`` to the export engine — the bare
    catalog ID, dropping any snapshot suffix. For an
    :class:`ErmrestSnapshot` (a snapshot-bound catalog with
    ``snaptime`` set), this causes the export to query the *live*
    catalog instead of the snapshot, and the bag's ``schema.json``
    reflects the live schema rather than the snapshot's. See issue
    #114.

    This subclass overrides ``_run_export`` to re-attach the snapshot
    suffix when the catalog has one.
    """

    def _run_export(self, spec: dict) -> Path:  # type: ignore[override]
        # Local import — see upstream's note on transitive deps.
        from deriva.transfer.download.deriva_download import (
            GenericDownloader,
        )

        # Restore the snapshot suffix dropped by upstream's
        # ``str(self.catalog.catalog_id)`` round-trip.
        catalog_id_str = str(self.catalog.catalog_id)
        snaptime = getattr(self.catalog, "snaptime", None)
        if snaptime:
            catalog_id_str = f"{catalog_id_str}@{snaptime}"

        deriva_server = self.catalog.deriva_server
        downloader = GenericDownloader(
            server={
                "host": deriva_server.server,
                "protocol": deriva_server.scheme,
                "catalog_id": catalog_id_str,
            },
            config=spec,
            output_dir=str(self.output_dir),
            credentials=self.catalog._credentials,
        )
        downloader.download()

        bag_path = self.output_dir / self.output_dir.name
        if not bag_path.exists():
            # Fall back to scanning for a single sub-directory.
            children = [p for p in self.output_dir.iterdir() if p.is_dir()]
            if len(children) == 1:
                bag_path = children[0]
        return bag_path


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

            >>> from deriva_ml.dataset.bag_builder import DatasetBagBuilder  # doctest: +SKIP
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
        # MINID only works when an S3 bucket is configured.
        self._use_minid = use_minid and s3_bucket is not None
        self._exclude_tables = exclude_tables or set()
        # Memoize the descendant-RID set per root RID for one builder op so
        # anchors_for and _exclude_empty_associations share a single tree walk.
        self._descendant_rids_cache: dict[RID, list[RID]] = {}

    # ------------------------------------------------------------------
    # Public surface — drives :class:`CatalogBagBuilder`
    # ------------------------------------------------------------------

    def generate_dataset_download_spec(self, dataset: DatasetLike, reachability_concurrency: int = 1) -> dict[str, Any]:
        """Return the runtime download spec for a specific dataset.

        Drives a :class:`CatalogBagBuilder` scoped to the dataset's
        RID (plus its nested descendants), takes its export spec,
        and overlays the dataset-specific top-level keys:

        - Top-level ``env: {"RID": "{RID}"}`` — populates the
          template that downstream Chaise uses.
        - Top-level ``post_processors`` for S3 upload + MINID
          minting when ``s3_bucket`` is configured.
        - ``bag.bag_name = "Dataset_{RID}"`` — the templated
          bag-archive filename Chaise expects.
        - Preamble :class:`env` query processors that parameterize
          ``{RID}``/``{snaptime}``/``{Description}`` for the
          template.

        The body of the spec — per-table CSV processors, asset
        fetches, vocab full-export — comes from
        :meth:`CatalogBagBuilder.get_export_spec` unchanged. So
        the "the spec is the bag walk" invariant from ADR-0006
        holds: any future change to the bag pipeline's walker is
        immediately reflected here.

        **Format B (issue #305).** The spec is generated with the
        per-table reachable RID sets, so its non-vocab CSV
        processors are rid-set processors — one clean
        ``data/{schema}/{table}.csv`` per reached table — identical
        to what :meth:`build_bag` (the direct-download arm)
        produces. This unifies the MINID / server-export bag format
        with the directly-downloaded format: the two arms now drive
        the same upstream emission with the same ``rid_sets`` map.

        Server dependency: rid-set CSV processors are honored by the
        deriva-py client (:meth:`get_as_file(rid_set=...)`) but
        require the ERMrest export *service* to support rid-set
        processors as well. Until the server gains that support,
        live MINID downloads against the production export service
        will fail; the directly-downloaded (``build_bag``) arm,
        which runs the client engine locally, is unaffected. The
        spec's content hash changes with this format, so any
        cached MINID with the old (Format-A) hash regenerates
        automatically — see ``get_dataset_minid``'s spec-hash gate.

        Args:
            dataset: The dataset to generate the spec for. Must
                expose ``dataset_rid``.

        Returns:
            The export-engine spec dict.
        """
        # Compute the per-table reachable RID sets so the upstream
        # engine emits one rid-set csv processor per non-vocab reached
        # table (Format B) — matching the direct-download arm
        # (:meth:`build_bag`) — instead of one csv processor per FK
        # path (Format A).
        rid_sets = self._compute_rid_sets(dataset, reachability_concurrency=reachability_concurrency).rid_sets
        builder = self._catalog_bag_builder(dataset=dataset, rid_sets=rid_sets)
        spec = builder.get_export_spec()

        # Pull the bag-pipeline catalog block — we keep its
        # ``query_processors`` as the walk's contribution, then
        # prepend the deriva-ml-specific env preamble.
        catalog_block = spec["catalog"]
        preamble = [
            {
                "processor": "env",
                "processor_params": {
                    "output_path": "Dataset",
                    "query_keys": ["snaptime"],
                    "query_path": "/",
                },
            },
            {
                "processor": "env",
                "processor_params": {
                    "query_path": f"/entity/M:={self._ml_schema}:Dataset/RID={{RID}}",
                    "output_path": "Dataset",
                    "query_keys": ["RID", "Description"],
                },
            },
        ]
        catalog_block["query_processors"] = preamble + catalog_block["query_processors"]

        out: dict[str, Any] = {
            "env": {"RID": "{RID}"},
            "bag": {
                "bag_name": "Dataset_{RID}",
                "bag_algorithms": ["md5"],
                "bag_archiver": "zip",
                "bag_metadata": {},
                "bag_idempotent": True,
            },
            "catalog": catalog_block,
        }
        if self._use_minid:
            out["post_processors"] = self._minid_post_processors()
        return out

    def build_bag(
        self,
        dataset: DatasetLike,
        output_dir: Path,
        timeout: tuple[int, int] | None = None,
        reachability_concurrency: int = 1,
    ) -> Path:
        """Build a bag for ``dataset`` and return the on-disk zip archive path.

        Drives :meth:`CatalogBagBuilder.build` against the catalog the
        ``DatasetBagBuilder`` is wired to. Callers typically construct this
        builder with ``ml_instance`` set to a snapshot-bound catalog so the
        produced bag is reproducible against a fixed catalog state.

        Upstream's :class:`CatalogBagBuilder` hard-codes
        ``bag_archiver="zip"`` and **removes the unpacked bag directory
        after archiving**, returning the (now-nonexistent) directory path.
        We compensate by computing the corresponding ``.zip`` location and
        returning *that* — the artifact that actually persists.

        The bag inside the zip is named ``Dataset_{rid}`` — :class:`CatalogBagBuilder`
        derives the bag-name from ``output_dir.name``, so this method
        creates an ``output_dir / f"Dataset_{rid}"`` directory and passes
        it as the builder's ``output_dir``. After archive the zip lands at
        ``output_dir / f"Dataset_{rid}" / f"Dataset_{rid}.zip"``.

        Args:
            dataset: The dataset to export. Must expose ``dataset_rid``.
            output_dir: Parent directory to receive the bag artifacts.
                Created if missing.
            timeout: Optional ``(connect, read)`` seconds applied to the
                underlying catalog's HTTP session for the duration of the
                export. ``None`` keeps the catalog's existing session
                config.

        Returns:
            Absolute :class:`Path` to the bag zip archive on success.

        Raises:
            ValueError: If :class:`CatalogBagBuilder` rejects the anchor
                (e.g., the dataset RID does not exist in the bound
                snapshot catalog).
            FileNotFoundError: If the expected zip artifact is absent
                after a successful build call (would indicate an upstream
                ``CatalogBagBuilder`` contract change).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # CatalogBagBuilder's bag-name comes from ``output_dir.name``. We
        # want the bag named ``Dataset_{rid}`` so the zip ends up at
        # ``output_dir / Dataset_{rid} / Dataset_{rid}.zip``.
        cb_output_dir = output_dir / f"Dataset_{dataset.dataset_rid}"
        cb_output_dir.mkdir(parents=True, exist_ok=True)

        anchors = self.anchors_for(dataset)
        policy = self.build_policy(dataset)
        # Compute the per-table reachable RID sets so the upstream engine
        # emits one rid-set csv processor per non-vocab reached table
        # (Format-B) instead of one csv processor per FK path.
        rid_sets = self._compute_rid_sets(dataset, reachability_concurrency=reachability_concurrency).rid_sets

        catalog = self._ml_instance.catalog
        prior_config = getattr(catalog, "_session_config", None)
        if timeout is not None:
            new_config = dict(prior_config) if prior_config else {}
            new_config["timeout"] = timeout
            catalog._session_config = new_config  # type: ignore[attr-defined]
        try:
            # Use the snapshot-aware subclass so the spec's catalog_id
            # carries the ``@snaptime`` suffix when applicable. The
            # upstream CatalogBagBuilder drops it (see #114) and the
            # bag would otherwise reflect the live catalog's schema
            # rather than the version-snapshot's.
            builder = _SnapshotAwareCatalogBagBuilder(
                catalog=catalog,
                anchors=anchors,
                output_dir=cb_output_dir,
                policy=policy,
                rid_sets=rid_sets,
            )
            unpacked_path = builder.build()
        finally:
            if timeout is not None:
                if prior_config is None:
                    try:
                        delattr(catalog, "_session_config")
                    except AttributeError:
                        pass
                else:
                    catalog._session_config = prior_config  # type: ignore[attr-defined]

        # CatalogBagBuilder.build() returns the path the unpacked bag had,
        # but the unpacked directory was removed during archival. The real
        # artifact is the sibling ``.zip`` file.
        zip_path = unpacked_path.with_suffix(".zip")
        if not zip_path.exists():
            raise FileNotFoundError(
                f"CatalogBagBuilder.build() returned {unpacked_path} but neither "
                f"the directory nor the expected {zip_path} archive exists."
            )
        return zip_path

    def generate_dataset_download_annotations(self) -> dict[str, Any]:
        """Return the static Chaise export annotations for the Dataset table.

        Drives a :class:`CatalogBagBuilder` scoped to the whole
        ``Dataset`` table (a :class:`TableAnchor`, not a specific
        RID) and consumes its :meth:`~CatalogBagBuilder.iter_reached_paths`
        — the symbolic ``(schema, table)`` path tuples — through a
        deriva-ml-side annotation writer.

        The annotation has to work for *any* future Dataset row,
        so the underlying walk is symbolic. Nesting depth is
        enumerated up to :meth:`_dataset_nesting_depth` so the
        annotation covers however deep any real dataset reaches.

        Returns:
            A dict suitable for writing as the Dataset table's
            annotation set (``deriva-export-fragment-definitions``,
            ``visible-foreign-keys``, ``export-2019``).
        """
        post_processors: dict[str, Any] = {}
        if self._use_minid:
            # Trailing slash for the S3 bucket URL on the annotation.
            s3_url = self._s3_bucket if self._s3_bucket.endswith("/") else f"{self._s3_bucket}/"
            post_processors = {
                "type": "BAG",
                "outputs": [{"fragment_key": "dataset_export_outputs"}],
                "displayname": "BDBag to Cloud",
                "bag_idempotent": True,
                "postprocessors": [
                    {
                        "processor": "cloud_upload",
                        "processor_params": {
                            "acl": "public-read",
                            "target_url": s3_url,
                        },
                    },
                    {
                        "processor": "identifier",
                        "processor_params": {
                            "test": False,
                            "env_column_map": {
                                "RID": "{RID}@{snaptime}",
                                "Description": "{Description}",
                            },
                        },
                    },
                ],
            }
        return {
            deriva_tags.export_fragment_definitions: {"dataset_export_outputs": self._export_annotation()},
            deriva_tags.visible_foreign_keys: self._dataset_visible_fkeys(),
            deriva_tags.export_2019: {
                "detailed": {
                    "templates": [
                        {
                            "type": "BAG",
                            "outputs": [{"fragment_key": "dataset_export_outputs"}],
                            "displayname": "BDBag Download",
                            "bag_idempotent": True,
                        }
                        | post_processors
                    ]
                }
            },
        }

    def aggregate_queries(
        self,
        dataset: DatasetLike | None = None,
    ) -> dict[str, list[Any]]:
        """Return live-catalog datapaths grouped by target table name.

        Drives a :class:`CatalogBagBuilder` (per-dataset when
        ``dataset`` is given, catalog-wide otherwise) and returns
        its :meth:`~CatalogBagBuilder.iter_table_datapaths` output
        rekeyed by terminal table name — the shape the drift path
        (:meth:`Dataset.is_dirty`, via ``_iter_drift_counts``) expects.

        Note: :meth:`Dataset.estimate_bag_size` no longer drives this
        method. It now uses the client-side reachability engine
        (:func:`deriva_ml.dataset._reachability.compute_reachability`),
        which shares the same walker via ``_catalog_bag_builder`` /
        ``iter_reached_paths`` but reconstructs FK reachability in
        memory rather than issuing per-path aggregate queries.

        Per CONTEXT.md ("Dirty"), the drift walk *is* the bag
        walk. Sharing the walker (via ``CatalogBagBuilder``) makes
        that invariant load-bearing rather than aspirational.

        Args:
            dataset: Optional dataset to filter paths to. ``None``
                aggregates across every dataset in the catalog.

        Returns:
            ``{target_table_name: [(datapath, pb_table, is_asset),
            ...]}``.
        """
        builder = self._catalog_bag_builder(dataset=dataset)
        keyed_by_pair = builder.iter_table_datapaths()
        # Caller-facing shape is keyed by terminal table *name*
        # only (not by ``(schema, name)``); preserve that.
        out: dict[str, list[Any]] = defaultdict(list)
        for (_schema, table_name), entries in keyed_by_pair.items():
            out[table_name].extend(entries)
        return dict(out)

    # ------------------------------------------------------------------
    # Internal driver — constructs a :class:`CatalogBagBuilder`
    # ------------------------------------------------------------------

    def _catalog_bag_builder(
        self,
        dataset: DatasetLike | None,
        rid_sets: dict[tuple[str, str], list[str]] | None = None,
    ) -> CatalogBagBuilder:
        """Construct a :class:`CatalogBagBuilder` for this dataset.

        Four callers (:meth:`generate_dataset_download_spec`,
        :meth:`generate_dataset_download_annotations`,
        :meth:`aggregate_queries`, :meth:`_compute_rid_sets`) share
        this construction so they all drive off the same walker.

        Args:
            dataset: Per-dataset scoping. When non-``None``, the
                walk anchors at the dataset's RID and its nested
                children's RIDs (via :meth:`anchors_for`). When
                ``None``, the walk anchors at the whole
                ``deriva-ml:Dataset`` table — the catalog-wide
                view used by annotation generation and by
                :meth:`aggregate_queries` with no dataset filter.
            rid_sets: Opt-in per-table reachable RID sets. When
                supplied, the upstream engine emits one rid-set csv
                processor per non-vocab reached table (Format-B) instead
                of one csv processor per FK path. ``None`` (the default,
                used by every share-the-walker caller above) keeps
                per-path emission unchanged; only the download/build
                path supplies it (via a separately-constructed builder).
        """
        if dataset is not None:
            anchors = self.anchors_for(dataset)
        else:
            anchors = [TableAnchor(table="Dataset")]
        policy = self.build_policy(dataset)
        # CatalogBagBuilder requires an ``output_dir`` even when
        # nothing will be written; aggregate_queries and the
        # annotation path don't run :meth:`build`. Use a temp
        # directory that goes away when this method returns.
        tmp = TemporaryDirectory()
        output_dir = Path(tmp.name)
        builder = CatalogBagBuilder(
            catalog=self._ml_instance.catalog,
            anchors=anchors,
            output_dir=output_dir,
            policy=policy,
            rid_sets=rid_sets,
        )
        # Stash the tmp on the builder so it lives until the
        # builder is garbage-collected. The annotation/aggregate
        # paths don't write to disk; the spec path uses the
        # returned dict directly and never invokes
        # :meth:`CatalogBagBuilder.build`.
        builder._datasetbag_output_tmp = tmp  # type: ignore[attr-defined]
        return builder

    def _compute_rid_sets(self, dataset: DatasetLike, reachability_concurrency: int = 1) -> RidSetComputation:
        """Compute per-table reachable RID sets for a dataset, client-side.

        Factored from estimate_bag_size's reachability assembly so the
        bag-build path and the estimate share one implementation -- and one
        copy of the ``from_model`` snapshot-staleness fix (the walk's fresh
        model and the fetch's path builder must be the same model, or the
        fetch raises KeyError on tables the held model lacks).

        Args:
            dataset: The dataset to analyze. Must expose ``dataset_rid``.
            reachability_concurrency: Opt-in bounded parallelism for the
                edge-table fetch phase (forwarded to
                :func:`compute_reachability` as ``max_workers``). ``1``
                (default) fetches sequentially -- exact, behavior-preserving.
                ``> 1`` parallelizes the per-table fetches, speeding up the
                estimate AND both bag-generation callers
                (:meth:`generate_dataset_download_spec`, :meth:`build_bag`) on
                large datasets. Distinct from the asset-file-download
                ``fetch_concurrency`` on the download path.

        Returns:
            A :class:`RidSetComputation` bundling the per-table RID sets
            (for Format-B bag generation) with the auxiliary reachability
            data the estimate consumes; see that NamedTuple's attribute docs
            for the per-field shapes.
        """
        from deriva_ml.dataset._reachability import (
            compute_reachability,
            sample_rows_from_fetched,
        )

        cb = self._catalog_bag_builder(dataset=dataset)
        reached = cb.iter_reached_paths()
        anchor_rids = [dataset.dataset_rid] + list(self._iter_descendant_rids(dataset))
        model = cb._get_model()

        # Build the path builder from the SAME model the walk used -- not the
        # held-model pathBuilder(). The walk reaches tables in deriva-py's
        # freshly-fetched snapshot model; the instance's held model can lag
        # (reused schema_json) and would be missing them, raising KeyError.
        # Sharing one model keeps walk and fetch in lockstep, no /schema fetch.
        pb = datapath.from_model(self._ml_instance.catalog, model)

        def _fetch(schema: str, table: str, columns: set[str]) -> list[dict]:
            tpb = pb.schemas[schema].tables[table]
            try:
                attrs = [getattr(tpb, c) for c in sorted(columns)]
                return list(tpb.attributes(*attrs).fetch())
            except Exception as exc:  # noqa: BLE001
                # Defensive fallback: a projection naming a column the table
                # lacks (model/data skew) degrades to a full-entity fetch.
                logger.debug(
                    "rid-set projected fetch for %s:%s fell back to full-entity scan: %s",
                    schema,
                    table,
                    exc,
                )
                return list(tpb.entities().fetch())

        rids_by_table, asset_lengths_by_table, fetched_rows = compute_reachability(
            reached=reached,
            anchor_rids=anchor_rids,
            model=model,
            fetch=_fetch,
            max_workers=reachability_concurrency,
        )
        sample_rows_by_table = sample_rows_from_fetched(reached=reached, fetched_rows=fetched_rows)

        vocab_tables = {key for key in reached if model.schemas[key[0]].tables[key[1]].is_vocabulary()}
        rid_sets = _rid_sets_from_reachability(reached, rids_by_table, vocab_tables)
        asset_tables = {key[1] for key in reached if model.schemas[key[0]].tables[key[1]].is_asset()}

        return RidSetComputation(
            rid_sets,
            rids_by_table,
            asset_lengths_by_table,
            fetched_rows,
            sample_rows_by_table,
            asset_tables,
        )

    @property
    def _ml_schema(self) -> str:
        """Convenience accessor — the deriva-ml schema name."""
        return self._ml_instance.ml_schema

    def _minid_post_processors(self) -> list[dict[str, Any]]:
        """Return the spec-side ``post_processors`` list for MINID.

        Used by :meth:`generate_dataset_download_spec` when
        ``s3_bucket`` is configured and ``use_minid`` is True.
        """
        return [
            {
                "processor": "cloud_upload",
                "processor_params": {
                    "acl": "public-read",
                    "target_url": self._s3_bucket,
                },
            },
            {
                "processor": "identifier",
                "processor_params": {
                    "test": False,
                    "env_column_map": {
                        "RID": "{RID}@{snaptime}",
                        "Description": "{Description}",
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # Annotation writers — consume CatalogBagBuilder.iter_reached_paths
    # ------------------------------------------------------------------
    #
    # These methods produce Chaise export annotation fragments. They
    # operate symbolically (no specific RID) and consume the path
    # set the bag pipeline's walker computes for a TableAnchor over
    # the Dataset table.

    def _export_annotation(self) -> list[dict[str, Any]]:
        """Return Chaise export-annotation fragments for the Dataset.

        Pre-fixed with three environment/schema fragments that
        Chaise needs, then one ``source`` fragment per FK route
        the bag walk discovers from the ``Dataset`` table. Vocab
        tables are emitted as standalone full-table queries; the
        bag pipeline's ``vocab_export=FULL`` walker rule handles
        that classification.
        """
        # Catalog-wide walk over a TableAnchor("Dataset") gives us
        # the symbolic paths for any future dataset row.
        builder = self._catalog_bag_builder(dataset=None)
        reached = builder.iter_reached_paths()

        # Three preamble fragments: snaptime, entity-level
        # (Dataset's own row), schema dump.
        out: list[dict[str, Any]] = [
            {
                "source": {"api": False, "skip_root_path": True},
                "destination": {"type": "env", "params": {"query_keys": ["snaptime"]}},
            },
            {
                "source": {"api": "entity"},
                "destination": {
                    "type": "env",
                    "params": {"query_keys": ["RID", "Description"]},
                },
            },
            {
                "source": {"api": "schema", "skip_root_path": True},
                "destination": {"type": "json", "name": "schema"},
            },
        ]

        # One fragment per (table, FK route).
        model = self._ml_instance.model
        for (schema_name, table_name), fk_paths in reached.items():
            try:
                table = model.schemas[schema_name].tables[table_name]
            except KeyError:
                continue
            for fk_path in fk_paths:
                out.extend(self._export_annotation_dataset_element(fk_path, table))
        return out

    def _export_annotation_dataset_element(
        self,
        fk_path: tuple[tuple[str, str], ...],
        table: Table,
    ) -> list[dict[str, Any]]:
        """Emit Chaise source fragments for one FK route.

        Args:
            fk_path: The bag walker's symbolic FK route from the
                anchor (``Dataset``) to the target table.
            table: The deriva-py :class:`Table` for the target —
                needed to detect asset tables.

        Returns:
            A list with one ``source``/``destination`` fragment
            for the row data, plus a ``fetch`` fragment for asset
            files when ``table.is_asset()``.
        """
        # Build the ERMrest path string from the segments.
        spath_segments = [f"{s}:{t}" for s, t in fk_path]
        spath = "/".join(spath_segments)

        # Skip the path that's just the Dataset table itself
        # (Chaise handles that case implicitly).
        skip_root_path = False
        if spath.startswith(f"{self._ml_schema}:Dataset/"):
            # Chaise will prepend table name and RID filter; strip
            # the redundant prefix.
            spath = "/".join(spath.split("/")[2:])
            if spath == "":
                return []
        else:
            # Vocab tables and non-Dataset roots: keep the path
            # but tell Chaise not to prepend its root.
            skip_root_path = True

        # Destination path under data/: just the table names.
        dpath = "/".join(t for _s, t in fk_path)

        exports: list[dict[str, Any]] = [
            {
                "source": {
                    "api": "entity",
                    "path": spath,
                    "skip_root_path": skip_root_path,
                },
                "destination": {"name": dpath, "type": "csv"},
            }
        ]
        if table.is_asset():
            exports.append(
                {
                    "source": {
                        "skip_root_path": False,
                        "api": "attribute",
                        "path": (f"{spath}/url:=URL,length:=Length,filename:=Filename,md5:=MD5,asset_rid:=RID"),
                    },
                    "destination": {
                        "name": "asset/{asset_rid}/" + table.name,
                        "type": "fetch",
                    },
                }
            )
        return exports

    def _dataset_visible_fkeys(self) -> dict[str, Any]:
        """Build the ``visible-foreign-keys`` annotation for the Dataset table.

        Emits the Chaise annotation that controls which related
        tables show up in the detailed view of a Dataset record:
        previous versions, parent/child datasets, and one entry
        per Dataset-element-type association.
        """

        def fkey_name(fk: Any) -> list[str]:
            return [fk.name[0].name, fk.name[1]]

        dataset_table = self._ml_instance.model.schemas[self._ml_schema].tables["Dataset"]

        source_list: list[dict[str, Any]] = [
            {
                "source": [
                    {"inbound": [self._ml_schema, "Dataset_Version_Dataset_fkey"]},
                    "RID",
                ],
                "markdown_name": "Previous Versions",
                "entity": True,
            },
            {
                "source": [
                    {"inbound": [self._ml_schema, "Dataset_Dataset_Nested_Dataset_fkey"]},
                    {"outbound": [self._ml_schema, "Dataset_Dataset_Dataset_fkey"]},
                    "RID",
                ],
                "markdown_name": "Parent Datasets",
            },
            {
                "source": [
                    {"inbound": [self._ml_schema, "Dataset_Dataset_Dataset_fkey"]},
                    {"outbound": [self._ml_schema, "Dataset_Dataset_Nested_Dataset_fkey"]},
                    "RID",
                ],
                "markdown_name": "Child Datasets",
            },
        ]
        source_list.extend(
            {
                "source": [
                    {"inbound": fkey_name(fkey.self_fkey)},
                    {"outbound": fkey_name(other_fkey := fkey.other_fkeys.pop())},
                    "RID",
                ],
                "markdown_name": other_fkey.pk_table.name,
            }
            for fkey in dataset_table.find_associations(max_arity=3, pure=False)
        )
        return {"detailed": source_list}

    def _dataset_nesting_depth(self, dataset: DatasetLike | None = None) -> int:
        """Return the maximum dataset-nesting depth in the catalog.

        When ``dataset`` is provided, computes the depth for that
        dataset's subtree only. When ``None``, computes the
        deepest nesting that exists anywhere in the catalog (the
        bound the static annotation needs to cover).

        Used by callers that want to know how many levels of
        ``Dataset → Dataset_Dataset → Dataset`` chains the
        annotation pipeline must enumerate.
        """

        def children_depth(rid: RID, graph: dict[str, list[str]]) -> int:
            try:
                children = graph[rid]
            except KeyError:
                return 0
            return max(children_depth(c, graph) for c in children) + 1 if children else 1

        pb = self._ml_instance.catalog.getPathBuilder().schemas[self._ml_schema].tables["Dataset_Dataset"]
        if dataset is not None:
            rows = [{"Dataset": dataset.dataset_rid, "Nested_Dataset": c} for c in dataset.list_dataset_children()]
        else:
            rows = list(pb.entities().fetch())

        graph: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            graph[r["Dataset"]].append(r["Nested_Dataset"])
        if not graph:
            return 0
        return max(children_depth(d, dict(graph)) for d in graph)

    # ------------------------------------------------------------------
    # ADR-0006 bag-pipeline-shaped helpers
    # ------------------------------------------------------------------

    def anchors_for(self, dataset: DatasetLike) -> list[Anchor]:
        """Return the :class:`Anchor` list for a dataset's bag walk.

        The dataset's root row is the primary anchor; each nested
        child dataset becomes an additional :class:`RIDAnchor`.
        Descendants are enumerated by :meth:`_iter_descendant_rids`
        (the memoized RID accessor over
        :meth:`DatasetLike.list_dataset_children_rids`), so the
        traversal depth is bounded by the nested-dataset chain.

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
        anchors: list[Anchor] = [RIDAnchor(table="Dataset", rids=[dataset.dataset_rid])]

        # Nested children walked one level at a time so the
        # anchor list documents the dataset's structure (one
        # RIDAnchor per descendant Dataset row).
        for child in self._iter_descendant_rids(dataset):
            anchors.append(RIDAnchor(table="Dataset", rids=[child]))

        return anchors

    def build_policy(
        self,
        dataset: DatasetLike | None,
        *,
        vocab_export: VocabExport = VocabExport.FULL,
    ) -> FKTraversalPolicy:
        """Return the :class:`FKTraversalPolicy` for this dataset.

        The policy encodes the dataset-specific traversal
        constraints in the bag-pipeline vocabulary:

        - :attr:`FKTraversalPolicy.exclude_tables` includes the
          association tables whose target element type has no
          members in *this* dataset. When ``dataset`` is ``None``
          (the catalog-wide annotation / aggregate-queries path),
          the member-based filter doesn't apply — every
          association is potentially in scope for some future
          row — so the exclusion set falls back to just the
          user's explicit ``exclude_tables``.
        - :attr:`FKTraversalPolicy.vocab_export` defaults to
          :attr:`VocabExport.FULL` so vocab terms are exported
          completely.
        - Feature tables reach the walk via FK-following from
          member element rows; no separate field is needed.

        Args:
            dataset: The dataset to derive the policy for, or
                ``None`` for the catalog-wide case (used by the
                annotation pipeline and by ``aggregate_queries``
                with no per-dataset filter).
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
            # Provenance tables are entered but not traversed outward —
            # otherwise the walk fans out across the whole catalog
            # provenance graph (see core/constants.py
            # :PROVENANCE_TERMINAL_TABLES). Same protection clone_via_bag
            # applies.
            terminal_tables=set(PROVENANCE_TERMINAL_TABLES),
            # Silence WARNING-level "Breaking cycle in FK
            # dependencies" log spam for the known
            # Dataset ↔ Dataset_Version cycle. See
            # core/constants.py:INTENTIONAL_FK_CYCLES.
            intentional_cycles=set(INTENTIONAL_FK_CYCLES),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exclude_empty_associations(self, dataset: DatasetLike | None) -> set[tuple[str, str]]:
        """Return ``{(schema, table)}`` for associations with no members.

        Membership is keyed on the element TYPE, not the individual
        association table. A ``Dataset_X`` association is kept in the
        walk when element type X has at least one member anywhere in
        the tree (root or any nested descendant) — even if that member
        was recorded via a *different*, parallel association — or when
        the association links to a vocabulary table (those carry
        dataset metadata and must always be included). Empty
        associations are added to the
        :attr:`FKTraversalPolicy.exclude_tables` set so the
        generic walker prunes them.

        The element-type keying is load-bearing for schemas with
        PARALLEL associations for the same type. On eye-ai, Subject
        membership is recorded in ``Subject_Dataset`` while
        ``Dataset_Subject`` is an empty parallel association. A
        per-assoc-table row check would wrongly exclude
        ``Dataset_Subject`` (it has no rows of its own); keying on the
        ``Subject`` element type keeps both, because Subject is a
        populated member type.

        Descendants matter because the walker anchors at every
        descendant dataset RID (via :meth:`anchors_for`). An
        association that's empty at the root but populated under
        a nested child must stay in the walk so the child's
        member rows are reachable. Limiting the member scan to
        the root would silently drop all rows of element types
        only owned by descendants.

        When ``dataset`` is ``None`` (catalog-wide annotation /
        aggregate path), the member-based filter doesn't apply —
        we have no specific dataset to count members for — so
        return an empty set. The catalog-wide walk includes every
        association on the model.
        """
        if dataset is None:
            return set()

        model = self._ml_instance.model
        ml_schema_name = self._ml_instance.ml_schema
        dataset_table = model.schemas[ml_schema_name].tables["Dataset"]

        # Descendant RID set (root + all nested descendants), one fetch.
        # Descendants matter because the walker anchors at every
        # descendant dataset RID (via ``anchors_for``): an association
        # empty at the root but populated under a nested child must
        # stay in the walk (see #94: child Image members disappearing).
        rid_set = [dataset.dataset_rid] + list(self._iter_descendant_rids(dataset))

        # Every vocabulary table in any schema — associations into
        # vocabularies always come along for the ride.
        vocab_tables: set[Table] = {
            table for schema in model.schemas.values() for table in schema.tables.values() if model.is_vocabulary(table)
        }

        pb = self._ml_instance.pathBuilder()

        # Pass 1: which element TYPES have a member anywhere in the tree?
        # An association row IS the membership record, so an element type X has
        # members iff SOME association reaching X has a row with Dataset in the
        # tree. We must key on the element TYPE (not the individual association
        # table) because a schema may have PARALLEL associations for the same
        # type: e.g. eye-ai records Subject membership in ``Subject_Dataset``
        # while ``Dataset_Subject`` is an empty parallel association. Keying on
        # the assoc table alone would wrongly exclude ``Dataset_Subject``;
        # keying on the type keeps both because ``Subject`` is a member type.
        # One presence query per association (limit=1), independent of
        # descendant count. Every Dataset_X association carries a literal
        # ``Dataset`` FK column (the convention ``list_dataset_members`` relies
        # on at dataset.py:1670).
        member_element_types: set[Table] = set()
        for assoc in dataset_table.find_associations():
            assoc_table = assoc.table
            assoc_path = pb.schemas[assoc_table.schema.name].tables[assoc_table.name]
            has_rows = bool(list(assoc_path.filter(assoc_path.Dataset.in_(rid_set)).entities().fetch(limit=1)))
            if has_rows:
                member_element_types.update(fk.pk_table for fk in assoc.other_fkeys)

        # Pass 2: exclude an association iff none of its member element types is
        # populated in the tree AND it doesn't link to a vocabulary. Pure set
        # logic on the results of pass 1 — no further queries.
        excluded: set[tuple[str, str]] = set()
        for assoc in dataset_table.find_associations():
            assoc_table = assoc.table
            links_to_member = any(fk.pk_table in member_element_types for fk in assoc.other_fkeys)
            links_to_vocab = any(fk.pk_table in vocab_tables for fk in assoc.other_fkeys)
            if not (links_to_member or links_to_vocab):
                excluded.add((assoc_table.schema.name, assoc_table.name))
        return excluded

    def _iter_descendant_rids(self, dataset: DatasetLike) -> Iterable[RID]:
        """Yield every descendant Dataset RID.

        Memoized per root RID for the lifetime of this builder so the
        nested-dataset tree is walked once per operation. Delegates to
        :meth:`Dataset.list_dataset_children_rids` (one ``Dataset_Dataset``
        fetch, in-memory traversal, no per-node lookups).

        Args:
            dataset: The dataset whose descendants to enumerate.

        Returns:
            Iterable[RID]: descendant dataset RIDs (depth-first order),
            excluding the root.
        """
        root = dataset.dataset_rid
        cached = self._descendant_rids_cache.get(root)
        if cached is None:
            dataset_obj = self._ml_instance.lookup_dataset(root)
            cached = list(dataset_obj.list_dataset_children_rids(recurse=True))
            self._descendant_rids_cache[root] = cached
        return cached


__all__ = ["DatasetBagBuilder"]

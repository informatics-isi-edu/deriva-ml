"""Dataset management mixin for DerivaML.

This module provides the DatasetMixin class which handles
dataset operations including finding, creating, looking up,
deleting, and managing dataset elements.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from typing import TYPE_CHECKING, Any, Callable, Iterable

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Table = _ermrest_model.Table

from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLTableTypeError
from deriva_ml.core.sort import SortSpec, resolve_sort
from deriva_ml.dataset.aux_classes import DatasetSpec

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.model.catalog import DerivaModel


class DatasetMixin:
    """Mixin providing dataset management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - domain_schema: str - name of the domain schema
        - s3_bucket: str | None - S3 bucket URL for dataset storage
        - use_minid: bool - whether to use MINIDs
        - pathBuilder(): method returning catalog path builder
        - _dataset_table: property returning the Dataset table

    Methods:
        find_datasets: List all datasets in the catalog
        create_dataset: Create a new dataset
        lookup_dataset: Look up a dataset by RID or spec
        delete_dataset: Delete a dataset
        list_dataset_element_types: List types that can be added to datasets
        add_dataset_element_type: Add a new element type to datasets
        download_dataset_bag: Download a dataset as a bag
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    domain_schemas: frozenset[str]
    default_schema: str | None
    s3_bucket: str | None
    use_minid: bool
    pathBuilder: Callable[[], Any]

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table. Must be provided by host class."""
        raise NotImplementedError

    def find_datasets(self, deleted: bool = False, sort: SortSpec = None) -> Iterable["Dataset"]:
        """List all datasets in the catalog.

        Args:
            deleted: If True, include datasets that have been marked as deleted.
            sort: Optional sort spec.
                - ``None`` (default): backend-determined order (no sort
                  clause applied; cheapest path).
                - ``True``: newest-first by record creation time
                  (``RCT desc``). Recommended for "show me the most
                  recent datasets" queries.
                - Callable ``(path) -> sort_keys``: receives the
                  Dataset table path and returns one or more
                  path-builder sort keys.

        Returns:
            Iterable of Dataset objects.

        Example:
            >>> datasets = list(ml.find_datasets())  # doctest: +SKIP
            >>> for ds in datasets:  # doctest: +SKIP
            ...     print(f"{ds.dataset_rid}: {ds.description}")

            Newest-first (most common):

            >>> recent = list(ml.find_datasets(sort=True))  # doctest: +SKIP
        """
        # Import here to avoid circular imports
        from deriva_ml.dataset.dataset import Dataset

        # Get datapath to the Dataset table
        pb = self.pathBuilder()
        dataset_path = pb.schemas[self._dataset_table.schema.name].tables[self._dataset_table.name]

        if deleted:
            filtered_path = dataset_path
        else:
            filtered_path = dataset_path.filter(
                (dataset_path.Deleted == False) | (dataset_path.Deleted == None)  # noqa: E711, E712
            )

        # Resolve sort spec against this method's default (newest-first
        # by record creation time). resolve_sort returns None when the
        # caller explicitly opted out of sorting (sort=None), in which
        # case we don't call .sort() at all -- backend default order.
        entity_set = filtered_path.entities()
        sort_keys = resolve_sort(sort, lambda p: p.RCT.desc, dataset_path)
        if sort_keys is not None:
            entity_set = entity_set.sort(*sort_keys)

        # Create Dataset objects - dataset_types is now a property that fetches from catalog
        datasets = []
        for dataset in entity_set.fetch():
            datasets.append(
                Dataset(
                    self,  # type: ignore[arg-type]
                    dataset_rid=dataset["RID"],
                    description=dataset["Description"],
                )
            )
        return datasets

    def lookup_dataset(self, dataset: RID | DatasetSpec, deleted: bool = False) -> "Dataset":
        """Look up a dataset by RID or DatasetSpec.

        Args:
            dataset: Dataset RID or DatasetSpec to look up.
            deleted: If True, include datasets that have been marked as deleted.

        Returns:
            Dataset: The dataset object for the specified RID.

        Raises:
            DerivaMLException: If the dataset is not found.

        Example:
            >>> dataset = ml.lookup_dataset("4HM")  # doctest: +SKIP
            >>> print(f"Version: {dataset.current_version}")  # doctest: +SKIP
        """
        if isinstance(dataset, DatasetSpec):
            dataset_rid = dataset.rid
        else:
            dataset_rid = dataset

        try:
            return [ds for ds in self.find_datasets(deleted=deleted) if ds.dataset_rid == dataset_rid][0]
        except IndexError:
            raise DerivaMLException(f"Dataset {dataset_rid} not found.")

    def delete_dataset(self, dataset: "Dataset", recurse: bool = False) -> None:
        """Soft-delete a dataset by marking it as deleted in the catalog.

        Sets the ``Deleted`` flag on the dataset record. The dataset's data is
        preserved but it will no longer appear in normal queries (e.g.,
        ``find_datasets()``). The dataset cannot be deleted if it is currently
        nested inside a parent dataset.

        Args:
            dataset (Dataset): The dataset to delete.
            recurse (bool): If True, also soft-delete all nested child datasets.
                If False (default), only this dataset is marked as deleted.

        Raises:
            DerivaMLException: If the dataset RID is not a valid dataset, or if the
                dataset is nested inside a parent dataset.

        Example:
            >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
            >>> ml.delete_dataset(ds, recurse=False)  # doctest: +SKIP
        """
        # Get association table entries for this dataset_table
        # Delete association table entries
        dataset_rid = dataset.dataset_rid
        if not self.model.is_dataset_rid(dataset.dataset_rid):
            raise DerivaMLException("Dataset_rid is not a dataset.")

        if parents := dataset.list_dataset_parents():
            raise DerivaMLException(f'Dataset "{dataset}" is in a nested dataset: {parents}.')

        pb = self.pathBuilder()
        dataset_path = pb.schemas[self._dataset_table.schema.name].tables[self._dataset_table.name]

        # list_dataset_children returns Dataset objects, so extract their RIDs
        child_rids = [ds.dataset_rid for ds in dataset.list_dataset_children()] if recurse else []
        rid_list = [dataset_rid] + child_rids
        dataset_path.update([{"RID": r, "Deleted": True} for r in rid_list])

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the table types that can be added as dataset members.

        Returns every table that has an association with the Dataset table,
        restricted to domain-schema tables and the Dataset table itself.
        These are the types accepted by ``add_dataset_members()``.

        Returns:
            Iterable of ``Table`` objects representing valid member types.

        Raises:
            DerivaMLException: If the catalog schema cannot be read.

        Example:
            >>> types = ml.list_dataset_element_types()  # doctest: +SKIP
            >>> print([t.name for t in types])  # doctest: +SKIP
        """

        def is_domain_or_dataset_table(table: Table) -> bool:
            return self.model.is_domain_schema(table.schema.name) or table.name == self._dataset_table.name

        return [
            t
            for a in self._dataset_table.find_associations()
            if is_domain_or_dataset_table(t := a.other_fkeys.pop().pk_table)
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_element_type(self, element: str | Table) -> Table:
        """Make it possible to add objects from ``element`` table to a dataset.

        Creates a new association table linking Dataset to the given table,
        then updates catalog annotations so the new type is included in
        bag-export specs. If the workspace ORM was already built, it is
        rebuilt to pick up the new association table — the ORM is eagerly
        constructed at init time and does not see DDL changes applied after
        that point.

        Args:
            element: Name of the table (str) or Table object to register as
                a valid dataset element type.

        Returns:
            The Table object that was registered.

        Raises:
            DerivaMLException: If ``element`` is not a valid table name.
            DerivaMLTableTypeError: If the table is a system or ML table
                and cannot be a dataset element type.

        Example:
            >>> ml.add_dataset_element_type("Image")  # doctest: +SKIP
        """
        # Import here to avoid circular imports
        from deriva_ml.dataset.catalog_graph import CatalogGraph

        # Add table to map.
        element_table = self.model.name_to_table(element)
        atable_def = self.model._define_association(
            associates=[self._dataset_table, element_table],
        )
        try:
            table = self.model.create_table(atable_def)
        except ValueError as e:
            if "already exists" in str(e):
                table = self.model.name_to_table(atable_def["table_name"])
            else:
                raise e

        # Rebuild the workspace ORM so it can resolve the new association table.
        # The workspace ORM is built eagerly at init time from the schema snapshot;
        # DDL applied after that point (like this new association table) is not
        # visible until the ORM is rebuilt from a fresh model fetch.
        if getattr(self, "_workspace", None) is not None:
            ls = getattr(self._workspace, "local_schema", None)
            if ls is not None:
                # Fresh model fetch so the rebuild sees the newly-added
                # association table (the local ermrest Model object may
                # lag behind if the test harness created this table via
                # a side channel).
                self.model.refresh_model()
                self._workspace.rebuild_schema(
                    model=self.model.model,
                    schemas=[self.ml_schema, *self.domain_schemas],
                )

        # self.model = self.catalog.getCatalogModel()
        annotations = CatalogGraph(
            self, s3_bucket=self.s3_bucket, use_minid=self.use_minid
        ).generate_dataset_download_annotations()  # type: ignore[arg-type]
        self._dataset_table.annotations.update(annotations)
        self.model.model.apply()
        return table

    def download_dataset_bag(
        self,
        dataset: DatasetSpec,
    ) -> "DatasetBag":
        """Downloads a dataset to the local filesystem.

        Downloads a dataset specified by DatasetSpec to the local filesystem. If the catalog
        has s3_bucket configured and use_minid is enabled, the bag will be uploaded to S3
        and registered with the MINID service.

        Args:
            dataset: Specification of the dataset to download, including version and materialization options.

        Returns:
            DatasetBag: Object containing:
                - path: Local filesystem path to downloaded dataset
                - rid: Dataset's Resource Identifier
                - minid: Dataset's Minimal Viable Identifier (if MINID enabled)

        Note:
            MINID support requires s3_bucket to be configured when creating the DerivaML instance.
            The catalog's use_minid setting controls whether MINIDs are created.

        Examples:
            Download with default options:
                >>> spec = DatasetSpec(rid="1-abc123")  # doctest: +SKIP
                >>> bag = ml.download_dataset_bag(dataset=spec)  # doctest: +SKIP
                >>> print(f"Downloaded to {bag.path}")  # doctest: +SKIP
        """
        if not self.model.is_dataset_rid(dataset.rid):
            raise DerivaMLTableTypeError("Dataset", dataset.rid)
        ds = self.lookup_dataset(dataset)
        return ds.download_dataset_bag(
            version=dataset.version,
            materialize=dataset.materialize,
            use_minid=self.use_minid,
            exclude_tables=dataset.exclude_tables,
            timeout=dataset.timeout,
            fetch_concurrency=dataset.fetch_concurrency,
        )

    def estimate_bag_size(
        self,
        dataset: "DatasetSpec",
    ) -> dict[str, Any]:
        """Estimate the size of a dataset bag before downloading.

        Generates the same download specification used by download_dataset_bag,
        then runs COUNT and SUM(Length) queries against the snapshot catalog
        to preview what a download will contain and how large it will be.

        Args:
            dataset: Specification of the dataset to estimate, including version
                and optional exclude_tables.

        Returns:
            dict with keys:
                - tables: dict mapping table name to {row_count, is_asset, asset_bytes}
                - total_rows: total row count across all tables
                - total_asset_bytes: total size of asset files in bytes
                - total_asset_size: human-readable size string (e.g., "1.2 GB")
        """
        if not self.model.is_dataset_rid(dataset.rid):
            raise DerivaMLTableTypeError("Dataset", dataset.rid)
        ds = self.lookup_dataset(dataset)
        return ds.estimate_bag_size(
            version=dataset.version,
            exclude_tables=dataset.exclude_tables,
        )

    def bag_info(
        self,
        dataset: "DatasetSpec",
    ) -> dict[str, Any]:
        """Get comprehensive info about a dataset bag: size, contents, and cache status.

        Combines the size estimate with local cache status. Use this to decide
        whether to prefetch a bag before running an experiment.

        Args:
            dataset: Specification of the dataset, including version and
                optional exclude_tables.

        Returns:
            dict with keys:
                - tables: dict mapping table name to {row_count, is_asset, asset_bytes}
                - total_rows, total_asset_bytes, total_asset_size
                - cache_status: one of "not_cached", "cached_metadata_only",
                  "cached_materialized", "cached_incomplete"
                - cache_path: local path to cached bag (if cached), else None
        """
        if not self.model.is_dataset_rid(dataset.rid):
            raise DerivaMLTableTypeError("Dataset", dataset.rid)
        ds = self.lookup_dataset(dataset)
        return ds.bag_info(
            version=dataset.version,
            exclude_tables=dataset.exclude_tables,
        )

    def estimate_denormalized_size(
        self,
        include_tables: list[str],
    ) -> dict[str, Any]:
        """Return schema shape + catalog-wide size estimates for a denormalized table.

        This is the **catalog-wide** analog of
        :meth:`Dataset.describe_denormalized`. It asks "if I were to
        denormalize these tables across the entire catalog (not scoped
        to any specific dataset), what would the result look like and
        how big would it be?" Useful for rough size estimation before
        committing to a bag export.

        The return shape is aligned with :meth:`estimate_bag_size` and
        is **NOT the same** as the dataset-scoped 12-key plan dict from
        :meth:`Dataset.describe_denormalized` (spec §5). Do not confuse
        the two.

        Args:
            include_tables: List of table names to include in the join.

        Returns:
            dict with these keys:

            - ``columns``: list of ``(column_name, column_type)`` tuples.
            - ``join_path``: ordered list of domain table names on the
                join chain (excludes the implicit ``Dataset`` root and
                any association tables).
            - ``tables``: ``{table_name: {row_count, is_asset,
                asset_bytes}}`` — per-table stats for every table in the
                join path.
            - ``total_rows``: sum of ``row_count`` across all included
                tables.
            - ``total_asset_bytes``: sum of ``asset_bytes``.
            - ``total_asset_size``: human-readable byte-count string
                (e.g., ``"1.2 GB"``).

        Example::

            info = ml.estimate_denormalized_size(["Image", "Subject"])
            print(f"{info['total_rows']} rows across "
                  f"{len(info['tables'])} tables, "
                  f"{info['total_asset_size']} of assets")

        See Also:
            Dataset.describe_denormalized: Dataset-scoped planning dict.
            Denormalizer.describe: Full dataset-scoped plan with
                ambiguity reporting.
            estimate_bag_size: Bag-level size estimation.
        """
        from deriva.core.datapath import Cnt, Sum

        from deriva_ml.dataset.dataset import Dataset
        from deriva_ml.model.catalog import denormalize_column_name

        model = self.model

        # _prepare_wide_table doesn't actually use dataset or dataset_rid
        # in its body — it only traverses the schema. Pass None for both.
        element_tables, column_specs, multi_schema = model._prepare_wide_table(None, None, list(include_tables))

        # Build columns list
        columns = [
            (
                denormalize_column_name(schema_name, table_name, col_name, multi_schema),
                type_name,
            )
            for schema_name, table_name, col_name, type_name in column_specs
        ]

        # Extract join path (domain tables only, no Dataset or association tables)
        join_path: list[str] = []
        for element_name, (path_names, _, _) in element_tables.items():
            for table_name in path_names:
                if table_name not in join_path and table_name != "Dataset":
                    if not model.is_association(table_name):
                        join_path.append(table_name)

        # Query global row counts per table
        pb = self.pathBuilder()
        tables_info: dict[str, Any] = {}
        total_rows = 0
        total_asset_bytes = 0

        for table_name in join_path:
            table = model.name_to_table(table_name)
            is_asset = model.is_asset(table_name)

            schema_name = table.schema.name
            table_path = pb.schemas[schema_name].tables[table_name]
            row_count = table_path.aggregates(Cnt(table_path.RID).alias("cnt")).fetch()[0]["cnt"]

            entry: dict[str, Any] = {
                "row_count": row_count,
                "is_asset": is_asset,
                "asset_bytes": 0,
            }

            if is_asset:
                result = table_path.aggregates(Sum(table_path.Length).alias("total")).fetch()
                asset_bytes = result[0]["total"] or 0
                entry["asset_bytes"] = asset_bytes
                total_asset_bytes += asset_bytes

            tables_info[table_name] = entry
            total_rows += row_count

        return {
            "columns": columns,
            "join_path": join_path,
            "tables": tables_info,
            "total_rows": total_rows,
            "total_asset_bytes": total_asset_bytes,
            "total_asset_size": Dataset._human_readable_size(total_asset_bytes),
        }

    def cache_dataset(
        self,
        dataset: "DatasetSpec",
        materialize: bool = True,
    ) -> dict[str, Any]:
        """Download a dataset bag into the local cache without creating an execution.

        Use this to warm the cache before running experiments. No execution or
        provenance records are created.

        Args:
            dataset: Specification of the dataset, including version and
                optional exclude_tables.
            materialize: If True (default), download all asset files. If False,
                download only table metadata.

        Returns:
            dict with bag_info results after caching.
        """
        if not self.model.is_dataset_rid(dataset.rid):
            raise DerivaMLTableTypeError("Dataset", dataset.rid)
        ds = self.lookup_dataset(dataset)
        return ds.cache(
            version=dataset.version,
            materialize=materialize,
            exclude_tables=dataset.exclude_tables,
            timeout=dataset.timeout,
            fetch_concurrency=dataset.fetch_concurrency,
        )

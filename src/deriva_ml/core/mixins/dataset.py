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

from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLTableTypeError
from deriva_ml.core.sort import SortSpec, resolve_sort
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.validation import (
    AssetSpecResult,
    CrossSpecIssue,
    DatasetSpecResult,
    DatasetSpecValidationReport,
    ExecutionConfigurationValidationReport,
    WorkflowSpecResult,
)

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.execution.execution_configuration import ExecutionConfiguration
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

    # ------------------------------------------------------------------
    # Pre-flight validation (metadata-only; cf. dry_run for full path).
    # ------------------------------------------------------------------

    def validate_dataset_specs(
        self,
        specs: list[DatasetSpec | str | dict[str, Any]],
    ) -> DatasetSpecValidationReport:
        """Validate a list of :class:`DatasetSpec` against the catalog.

        Replaces the per-RID ``ml.lookup_dataset(rid)`` +
        ``ds.dataset_history()`` cross-check loop a user would
        otherwise write while iterating on ``src/configs/datasets.py``.
        Each spec is checked for three orthogonal failure modes —
        ``rid_not_found``, ``not_a_dataset``, ``version_not_found`` —
        and all three are reported per spec rather than stopping at
        the first.

        This is a metadata-only catalog query. For the heavier
        full-path check (which materializes bags) see
        :meth:`Execution.dry_run`. ADR-0002 captures the rationale
        for keeping the two surfaces distinct.

        Input shorthands are accepted for ergonomics: a plain RID
        string is parsed via :meth:`DatasetSpec.from_shorthand` (so
        ``"1-XYZ@1.0.0"`` and ``"1-XYZ"`` both work), and a dict is
        coerced via ``DatasetSpec(**d)``.

        Duplicate specs in the input are validated independently
        (no deduplication). Cross-spec duplicate detection lives on
        the composite :meth:`validate_execution_configuration`.

        Args:
            specs: List of dataset specifications to validate. Each
                element may be a :class:`DatasetSpec`, a shorthand
                string parseable by :meth:`DatasetSpec.from_shorthand`,
                or a dict that coerces to :class:`DatasetSpec`.

        Returns:
            A :class:`DatasetSpecValidationReport` with one
            :class:`DatasetSpecResult` per input spec (in the same
            order) plus a top-level ``all_valid`` convenience flag.

        Raises:
            pydantic.ValidationError: If any input element cannot be
                coerced to a :class:`DatasetSpec` (e.g. malformed
                version string, missing required fields).

        Example:
            Validate two specs, one good and one with a typo'd version::

                >>> from deriva_ml.dataset.aux_classes import DatasetSpec
                >>> report = ml.validate_dataset_specs(specs=[  # doctest: +SKIP
                ...     DatasetSpec(rid="2-B4C8", version="0.4.0"),
                ...     DatasetSpec(rid="2-B4C8", version="9.9.9"),
                ... ])
                >>> report.all_valid  # doctest: +SKIP
                False
                >>> bad = report.results[1]  # doctest: +SKIP
                >>> bad.reasons  # doctest: +SKIP
                ['version_not_found']
                >>> bad.available_versions  # doctest: +SKIP
                ['0.4.0', '0.3.0']
        """
        # Coerce inputs once so cached lookups can use the canonical form.
        coerced: list[DatasetSpec] = [self._coerce_dataset_spec(s) for s in specs]

        # Per-RID caches so duplicate-RID inputs cost one round-trip each,
        # not N. The values mirror the partial state assembled during a
        # validation pass.
        rid_cache: dict[str, dict[str, Any]] = {}

        results = [self._validate_one_dataset_spec(s, rid_cache) for s in coerced]
        return DatasetSpecValidationReport(
            all_valid=all(r.valid for r in results),
            results=results,
        )

    def validate_execution_configuration(
        self,
        config: "ExecutionConfiguration",
    ) -> ExecutionConfigurationValidationReport:
        """Pre-flight validation for an :class:`ExecutionConfiguration`.

        Walks the contained ``datasets`` and ``assets`` lists, validates
        the workflow RID, and reports per-spec results plus cross-spec
        issues (duplicate RIDs across the dataset list, dataset-version
        conflicts, asset role conflicts). Designed to be cheap to run
        repeatedly while iterating on a config — only catalog metadata
        is touched, no bags are materialized.

        This method is the lightweight complement to
        :meth:`Execution.dry_run`. ``dry_run`` is the heavier full-path
        test that exercises the bag-download + materialization pipeline;
        ``validate_execution_configuration`` answers the cheaper
        upstream question of *"do the RIDs in this config refer to
        things that exist in the catalog the way I think they do?"*.
        See ADR-0002 for the full rationale.

        The dataset half of the work is delegated to
        :meth:`validate_dataset_specs` — the two methods share the
        per-spec dataset validation logic.

        Args:
            config: The execution configuration to validate. Its
                ``datasets``, ``assets``, and ``workflow`` fields are
                walked; other fields (``description``, ``argv``,
                ``config_choices``) are ignored.

        Returns:
            A :class:`ExecutionConfigurationValidationReport` with
            per-spec results, an optional workflow result (None if
            the config has no workflow set), cross-spec issues, and
            an ``all_valid`` convenience flag.

        Raises:
            pydantic.ValidationError: If ``config`` is not an
                :class:`ExecutionConfiguration`.

        Example:
            Validate a config before invoking ``deriva-ml-run``::

                >>> from deriva_ml.execution import ExecutionConfiguration
                >>> from deriva_ml.dataset.aux_classes import DatasetSpec
                >>> from deriva_ml.asset.aux_classes import AssetSpec
                >>> config = ExecutionConfiguration(  # doctest: +SKIP
                ...     workflow=workflow,
                ...     datasets=[DatasetSpec(rid="2-B4C8", version="0.4.0")],
                ...     assets=[AssetSpec(rid="3JSE")],
                ... )
                >>> report = ml.validate_execution_configuration(config)  # doctest: +SKIP
                >>> if not report.all_valid:  # doctest: +SKIP
                ...     for issue in report.cross_spec_issues:
                ...         print(issue.detail)
        """
        # Dataset half — delegate to the singular method.
        dataset_report = self.validate_dataset_specs(specs=list(config.datasets))

        # Asset half — per-spec.
        asset_results = [self._validate_asset_spec(a) for a in config.assets]

        # Workflow.
        workflow_result: WorkflowSpecResult | None
        if config.workflow is not None and config.workflow.rid is not None:
            workflow_result = self._validate_workflow_rid(config.workflow.rid)
        else:
            workflow_result = None

        # Cross-spec issues.
        cross_spec_issues = self._collect_cross_spec_issues(
            dataset_specs=list(config.datasets),
            asset_specs=list(config.assets),
        )

        all_valid = (
            dataset_report.all_valid
            and all(r.valid for r in asset_results)
            and (workflow_result is None or workflow_result.valid)
            and not cross_spec_issues
        )

        return ExecutionConfigurationValidationReport(
            all_valid=all_valid,
            dataset_results=dataset_report.results,
            asset_results=asset_results,
            workflow_result=workflow_result,
            cross_spec_issues=cross_spec_issues,
        )

    # -- private helpers ------------------------------------------------

    @staticmethod
    def _coerce_dataset_spec(value: DatasetSpec | str | dict[str, Any]) -> DatasetSpec:
        """Coerce one input into a :class:`DatasetSpec`.

        See :meth:`validate_dataset_specs` for the supported shorthands.
        """
        if isinstance(value, DatasetSpec):
            return value
        if isinstance(value, str):
            return DatasetSpec.from_shorthand(value)
        if isinstance(value, dict):
            return DatasetSpec(**value)
        # Fall through: let DatasetSpec raise a clear ValidationError.
        return DatasetSpec.model_validate(value)

    def _validate_one_dataset_spec(
        self,
        spec: DatasetSpec,
        rid_cache: dict[str, dict[str, Any]],
    ) -> DatasetSpecResult:
        """Validate one already-coerced :class:`DatasetSpec`.

        Uses ``rid_cache`` to amortize repeated lookups for the same
        RID across two specs. Mutates the cache.
        """
        cached = rid_cache.get(spec.rid)
        if cached is None:
            cached = self._lookup_dataset_metadata(spec.rid)
            rid_cache[spec.rid] = cached

        # rid_not_found short-circuits everything else.
        if cached["status"] == "rid_not_found":
            return DatasetSpecResult(
                spec=spec,
                valid=False,
                reasons=["rid_not_found"],
            )

        # not_a_dataset short-circuits version checking.
        if cached["status"] == "not_a_dataset":
            return DatasetSpecResult(
                spec=spec,
                valid=False,
                reasons=["not_a_dataset"],
                actual_table=cached["actual_table"],
            )

        # We have a real dataset. Now check the version.
        requested_version = str(spec.version)
        available_versions: list[str] = cached["versions"]
        warnings: list[str] = []
        if cached["deleted"]:
            warnings.append("dataset_deleted")

        if requested_version not in available_versions:
            # Cap at 20, newest first. Versions in cached["versions"] are
            # already in newest-first order.
            return DatasetSpecResult(
                spec=spec,
                valid=False,
                reasons=["version_not_found"],
                warnings=warnings,
                available_versions=available_versions[:20],
            )

        return DatasetSpecResult(
            spec=spec,
            valid=True,
            warnings=warnings,
            resolved_version=requested_version,
            dataset_name=cached["description"],
        )

    def _lookup_dataset_metadata(self, rid: str) -> dict[str, Any]:
        """Resolve a RID and gather just enough metadata to validate it.

        Returns a dict with ``status`` of ``rid_not_found``,
        ``not_a_dataset``, or ``ok``. When ``ok``, the dict also
        carries ``description``, ``deleted``, and ``versions`` (newest
        first). When ``not_a_dataset``, it carries ``actual_table``.
        """
        try:
            resolved = self.resolve_rid(rid)  # type: ignore[attr-defined]
        except DerivaMLException:
            return {"status": "rid_not_found"}

        table = resolved.table
        if table.name != "Dataset":
            return {"status": "not_a_dataset", "actual_table": table.name}

        # Fetch the dataset row and its version history in two cheap queries.
        try:
            row = list(resolved.datapath.entities().fetch())[0]
        except (IndexError, DerivaMLException):
            return {"status": "rid_not_found"}

        pb = self.pathBuilder()
        version_path = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        version_rows = list(version_path.filter(version_path.Dataset == rid).entities().fetch())

        # Sort newest-first (semver-aware) so the head of the slice is
        # the most useful piece of context for "did you mean..." UX.
        def _semver_key(v: str) -> tuple[int, ...]:
            try:
                return tuple(int(p) for p in v.split("."))
            except ValueError:
                return (0,)

        versions = sorted(
            (vr["Version"] for vr in version_rows if vr.get("Version")),
            key=_semver_key,
            reverse=True,
        )

        return {
            "status": "ok",
            "description": row.get("Description"),
            "deleted": bool(row.get("Deleted")),
            "versions": versions,
        }

    def _validate_asset_spec(self, spec: AssetSpec) -> AssetSpecResult:
        """Validate one :class:`AssetSpec` against the catalog.

        Mirrors the ``rid_not_found`` / ``not_an_asset`` checks that
        :meth:`AssetMixin.lookup_asset` performs internally, but
        returns a structured per-spec result instead of raising.
        """
        try:
            resolved = self.resolve_rid(spec.rid)  # type: ignore[attr-defined]
        except DerivaMLException:
            return AssetSpecResult(spec=spec, valid=False, reasons=["rid_not_found"])

        table = resolved.table
        if not self.model.is_asset(table):
            return AssetSpecResult(
                spec=spec,
                valid=False,
                reasons=["not_an_asset"],
                actual_table=table.name,
            )

        try:
            row = list(resolved.datapath.entities().fetch())[0]
        except (IndexError, DerivaMLException):
            return AssetSpecResult(spec=spec, valid=False, reasons=["rid_not_found"])

        return AssetSpecResult(
            spec=spec,
            valid=True,
            asset_table=table.name,
            filename=row.get("Filename"),
        )

    def _validate_workflow_rid(self, rid: str) -> WorkflowSpecResult:
        """Validate that ``rid`` points at a Workflow row."""
        try:
            resolved = self.resolve_rid(rid)  # type: ignore[attr-defined]
        except DerivaMLException:
            return WorkflowSpecResult(rid=rid, valid=False, reasons=["rid_not_found"])

        table = resolved.table
        if table.name != "Workflow":
            return WorkflowSpecResult(
                rid=rid,
                valid=False,
                reasons=["not_a_workflow"],
                actual_table=table.name,
            )

        try:
            row = list(resolved.datapath.entities().fetch())[0]
        except (IndexError, DerivaMLException):
            return WorkflowSpecResult(rid=rid, valid=False, reasons=["rid_not_found"])

        return WorkflowSpecResult(
            rid=rid,
            valid=True,
            workflow_name=row.get("Name"),
        )

    @staticmethod
    def _collect_cross_spec_issues(
        dataset_specs: list[DatasetSpec],
        asset_specs: list[AssetSpec],
    ) -> list[CrossSpecIssue]:
        """Compute cross-spec inconsistencies for the composite report.

        Detects:
            - duplicate dataset RIDs (with the same version)
            - dataset version conflicts (same RID, different versions)
            - duplicate asset RIDs (same role)
            - role conflicts (same asset RID with both Input and Output)
        """
        issues: list[CrossSpecIssue] = []

        # Datasets: group by RID, then by version-string.
        ds_by_rid: dict[str, list[str]] = {}
        for s in dataset_specs:
            ds_by_rid.setdefault(s.rid, []).append(str(s.version))

        for rid, versions in ds_by_rid.items():
            distinct = set(versions)
            if len(distinct) > 1:
                issues.append(
                    CrossSpecIssue(
                        issue="version_conflict",
                        rids=[rid],
                        detail=(f"Dataset {rid} listed with conflicting versions: {sorted(distinct)}."),
                    )
                )
            elif len(versions) > 1:
                issues.append(
                    CrossSpecIssue(
                        issue="duplicate_rid",
                        rids=[rid],
                        detail=(f"Dataset {rid} listed {len(versions)} times with the same version {versions[0]!r}."),
                    )
                )

        # Assets: group by RID, then by role.
        as_by_rid: dict[str, list[str]] = {}
        for s in asset_specs:
            as_by_rid.setdefault(s.rid, []).append(s.asset_role)

        for rid, roles in as_by_rid.items():
            distinct = set(roles)
            if len(distinct) > 1:
                issues.append(
                    CrossSpecIssue(
                        issue="role_conflict",
                        rids=[rid],
                        detail=(f"Asset {rid} listed with conflicting roles: {sorted(distinct)}."),
                    )
                )
            elif len(roles) > 1:
                issues.append(
                    CrossSpecIssue(
                        issue="duplicate_rid",
                        rids=[rid],
                        detail=(f"Asset {rid} listed {len(roles)} times with role {roles[0]!r}."),
                    )
                )

        return issues

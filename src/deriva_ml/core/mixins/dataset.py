"""Dataset management mixin for DerivaML.

This module provides the DatasetMixin class which handles
dataset operations including finding, creating, looking up,
deleting, and managing dataset elements.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
import os
from typing import TYPE_CHECKING, Any, Callable, Iterable

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Table = _ermrest_model.Table

from pydantic import validate_call

from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.config.ast_walker import parse_config_file
from deriva_ml.config.bootstrap import (
    DEFAULT_DATASET_TYPE_FILTER,
    BootstrapReport,
    BootstrapSkipped,
    BootstrapSuggestion,
    _format_asset_spec,
    _format_dataset_spec,
    _format_deriva_ml_spec,
    _format_workflow_spec,
    _sanitize_config_name,
)
from deriva_ml.config.validation import (
    ConfigEntry,
    ConfigEntryResult,
    ConfigFileParseError,
    ConfigValidationReport,
)
from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLTableTypeError
from deriva_ml.core.sort import SortSpec, resolve_sort
from deriva_ml.core.validation import VALIDATION_CONFIG
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


__all__ = ["DatasetMixin"]


class DatasetMixin:
    """Mixin providing dataset management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - domain_schemas: frozenset[str] - names of the domain schemas
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
    # Provided by the host class (DerivaML). Declared here so type
    # checkers see the contract; the real implementation is a
    # @property in the host that returns the catalog's Dataset table.
    _dataset_table: Table

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

        Thin wrapper over :meth:`DerivaModel.list_dataset_element_types`;
        the model layer owns the filter logic.

        Returns:
            Iterable of ``Table`` objects representing valid member types.

        Raises:
            DerivaMLException: If the catalog schema cannot be read.

        Example:
            >>> types = ml.list_dataset_element_types()  # doctest: +SKIP
            >>> print([t.name for t in types])  # doctest: +SKIP
        """
        return self.model.list_dataset_element_types()

    @validate_call(config=VALIDATION_CONFIG)
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
        # Import here to avoid circular imports.
        from deriva_ml.dataset.bag_builder import DatasetBagBuilder

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
        annotations = DatasetBagBuilder(
            ml_instance=self,
            s3_bucket=self.s3_bucket,
            use_minid=self.use_minid,
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
                - cache_status: one of "not_cached", "cached_materialized",
                  "cached_holey"
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
        element_tables, column_specs, multi_schema = model._planner._prepare_wide_table(
            None, None, list(include_tables)
        )

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

    def validate_config_file(
        self,
        path: str | os.PathLike[str],
    ) -> ConfigValidationReport:
        """Validate every spec constructor in one hydra-zen config file.

        Parses the file via AST (no execution) and validates each
        ``DatasetSpecConfig`` / ``AssetSpecConfig`` / ``Workflow`` /
        ``DerivaMLConfig`` constructor call against the catalog.

        Composes the existing :meth:`validate_dataset_specs`,
        :meth:`_validate_asset_spec`, :meth:`_validate_workflow_rid`
        primitives. For ``DerivaMLConfig`` entries the validator
        compares the entry's ``hostname`` and ``catalog_id`` against
        the catalog this :class:`DerivaML` instance is connected to;
        a mismatch is reported but doesn't make catalog calls.

        Args:
            path: Path to the file. Accepts ``str``, ``Path``, or any
                ``os.PathLike``.

        Returns:
            A :class:`ConfigValidationReport` with one
            :class:`ConfigEntryResult` per constructor call found
            (in source order). Files that fail to parse produce a
            single :class:`ConfigFileParseError`; the entry list is
            empty in that case.

        Does not raise on syntax errors or missing files -- both are
        reported structurally so a caller validating many files can
        record them without aborting the walk.

        Example:
            Validate one file::

                >>> report = ml.validate_config_file(  # doctest: +SKIP
                ...     "src/configs/datasets.py"
                ... )
                >>> for r in report.results:  # doctest: +SKIP
                ...     if not r.valid:
                ...         print(r.entry.file, r.entry.line, r.reasons)
        """
        entries, parse_error = parse_config_file(path)
        parse_errors: list[ConfigFileParseError] = (
            [parse_error] if parse_error is not None else []
        )
        results = self._validate_config_entries(entries)
        return ConfigValidationReport(
            file_count=1 if parse_error is None else 0,
            entry_count=len(entries),
            all_valid=(not parse_errors) and all(r.valid for r in results),
            results=results,
            parse_errors=parse_errors,
        )

    def validate_config_directory(
        self,
        configs_dir: str | os.PathLike[str],
        *,
        recursive: bool = True,
    ) -> ConfigValidationReport:
        """Validate every ``*.py`` config file under ``configs_dir``.

        Walks the directory, parses each Python file, validates every
        constructor call against the catalog, and aggregates the per-
        file reports into one :class:`ConfigValidationReport`. A
        single broken file does not abort the walk -- the error is
        recorded in ``parse_errors`` and the validator continues with
        the next file.

        Args:
            configs_dir: Path to the configs directory (typically
                ``src/configs``). ``__init__.py`` files are included;
                ``__pycache__`` and dot-prefixed directories are
                skipped.
            recursive: When ``True`` (default), recurse into
                subdirectories. ``configs/dev/`` per-environment
                overrides are picked up.

        Returns:
            A :class:`ConfigValidationReport` with all entries from
            all files, ordered first by file path then by line.

        Example:
            Validate the whole tree::

                >>> report = ml.validate_config_directory(  # doctest: +SKIP
                ...     "src/configs"
                ... )
                >>> if not report.all_valid:  # doctest: +SKIP
                ...     for r in report.results:
                ...         if not r.valid:
                ...             print(r.entry.file, r.entry.line, r.reasons)
                ...     for pe in report.parse_errors:
                ...         print("UNPARSEABLE:", pe.file, pe.message)
        """
        from pathlib import Path as _Path

        root = _Path(os.fspath(configs_dir))
        if not root.exists():
            return ConfigValidationReport(
                file_count=0,
                entry_count=0,
                all_valid=False,
                results=[],
                parse_errors=[
                    ConfigFileParseError(
                        file=str(root),
                        line=None,
                        message=f"directory not found: {root}",
                    )
                ],
            )

        py_files: list[_Path] = []
        if root.is_file():
            if root.suffix == ".py":
                py_files = [root]
        else:
            walker = root.rglob("*.py") if recursive else root.glob("*.py")
            for p in walker:
                # Skip __pycache__ and any dot-prefixed dir.
                if any(part == "__pycache__" or part.startswith(".") for part in p.parts):
                    continue
                py_files.append(p)
        py_files.sort()

        all_entries: list[ConfigEntry] = []
        parse_errors: list[ConfigFileParseError] = []
        file_count = 0
        for f in py_files:
            entries, parse_error = parse_config_file(f)
            if parse_error is not None:
                parse_errors.append(parse_error)
                continue
            file_count += 1
            all_entries.extend(entries)

        results = self._validate_config_entries(all_entries)
        return ConfigValidationReport(
            file_count=file_count,
            entry_count=len(all_entries),
            all_valid=(not parse_errors) and all(r.valid for r in results),
            results=results,
            parse_errors=parse_errors,
        )

    def bootstrap_config(
        self,
        *,
        kinds: list[str] | None = None,
        dataset_type_filter: list[str] | None = None,
    ) -> BootstrapReport:
        """Suggest config entries by reading the catalog.

        Walks the catalog and produces structured :class:`BootstrapSuggestion`
        objects -- one per dataset / asset / workflow row a fresh
        project's ``src/configs/`` might want to pin. Does NOT write
        files. The skill prose layer formats the suggestions into the
        right config file (per-skill ownership of "which file" --
        ``dataset-lifecycle`` for datasets, ``work-with-assets`` for
        assets, ``write-hydra-config`` for the umbrella).

        Three use cases:

        - **New project, empty configs/.** Run unfiltered to see every
          candidate entry; pick the subset that's relevant.
        - **Catalog clone or environment switch.** Bootstrap to repoint
          configs at the new catalog, then validate to catch any
          stragglers.
        - **Incremental update.** Pass ``kinds=["datasets"]`` to see
          fresh dataset suggestions after a release without
          enumerating assets / workflows.

        Args:
            kinds: Which config groups to suggest entries for. Default
                is all four (``deriva_ml``, ``datasets``, ``assets``,
                ``workflow``). Skipping ``experiments``,
                ``multiruns``, ``model_config`` is intentional --
                those are project code, not catalog state.
            dataset_type_filter: When suggesting datasets, restrict
                to these ``Dataset_Type`` terms. Default is
                ``["Training", "Testing", "Validation", "Complete",
                "Labeled"]`` -- the partition-role + annotation tags
                experiments typically pin. Pass ``[]`` (empty list)
                to include every type. Pass ``None`` (default) to
                use the default filter.

        Returns:
            A :class:`BootstrapReport` with suggestions grouped (by
            ``kind`` field), and a ``skipped`` list explaining why
            specific entities weren't suggested.

        Example:
            One-shot bootstrap::

                >>> report = ml.bootstrap_config()  # doctest: +SKIP
                >>> for s in report.suggestions:  # doctest: +SKIP
                ...     print(s.kind, s.config_name, s.spec_string)
        """
        requested_kinds = set(kinds) if kinds is not None else {
            "deriva_ml",
            "datasets",
            "assets",
            "workflow",
        }
        if dataset_type_filter is None:
            type_filter = set(DEFAULT_DATASET_TYPE_FILTER)
        else:
            type_filter = set(dataset_type_filter)  # may be empty (= no filter)

        suggestions: list[BootstrapSuggestion] = []
        skipped: list[BootstrapSkipped] = []

        if "deriva_ml" in requested_kinds:
            suggestions.append(
                BootstrapSuggestion(
                    kind="deriva_ml",
                    config_name="default_deriva",
                    rid="",  # connection groups don't pin a RID
                    spec_string=_format_deriva_ml_spec(
                        str(getattr(self, "host_name", "")),
                        str(getattr(self, "catalog_id", "")),
                    ),
                    description=(
                        f"Connection to {getattr(self, 'host_name', '?')} "
                        f"catalog {getattr(self, 'catalog_id', '?')}"
                    ),
                    rationale="Connection group; pin this DerivaML instance.",
                )
            )

        if "datasets" in requested_kinds:
            datasets_iter = self.find_datasets()  # type: ignore[attr-defined]
            for ds in datasets_iter:
                ds_rid = ds.dataset_rid
                ds_types = list(ds.dataset_types or [])
                # Apply type filter if non-empty.
                if type_filter and not (set(ds_types) & type_filter):
                    skipped.append(
                        BootstrapSkipped(
                            kind="datasets",
                            rid=ds_rid,
                            reason=(
                                f"dataset_types={ds_types} -- not in filter "
                                f"{sorted(type_filter)}"
                            ),
                        )
                    )
                    continue
                # Need a released version to pin -- dev labels would
                # break reproducibility on consumers.
                current = ds.current_version  # type: ignore[attr-defined]
                if current is None:
                    skipped.append(
                        BootstrapSkipped(
                            kind="datasets",
                            rid=ds_rid,
                            reason="no current version",
                        )
                    )
                    continue
                version_str = str(current)
                if ".dev" in version_str or ".post" in version_str:
                    skipped.append(
                        BootstrapSkipped(
                            kind="datasets",
                            rid=ds_rid,
                            reason=(
                                f"current_version={version_str!r} is a dev label; "
                                "call deriva_ml_release(...) to mint a released version"
                            ),
                        )
                    )
                    continue
                desc = getattr(ds, "description", "") or ""
                config_name = _sanitize_config_name(desc, fallback=ds_rid)
                primary_type = (
                    next(iter(set(ds_types) & type_filter), None)
                    if type_filter
                    else (ds_types[0] if ds_types else "Dataset")
                )
                rationale = (
                    f"Dataset type {primary_type or '?'}; latest released "
                    f"version {version_str}."
                )
                suggestions.append(
                    BootstrapSuggestion(
                        kind="datasets",
                        config_name=config_name,
                        rid=ds_rid,
                        version=version_str,
                        spec_string=_format_dataset_spec(ds_rid, version_str),
                        description=desc or None,
                        rationale=rationale,
                    )
                )

        if "assets" in requested_kinds:
            for table in self.list_asset_tables():  # type: ignore[attr-defined]
                # Skip the built-in DerivaML asset tables -- they hold
                # auto-generated metadata files (execution config dumps,
                # uploaded notebooks). Users don't pin those by RID
                # from experiment configs; they're navigated through
                # the producing execution.
                if table.name in {"Execution_Metadata", "Execution_Asset"}:
                    skipped.append(
                        BootstrapSkipped(
                            kind="assets",
                            rid=table.name,
                            reason=(
                                "built-in ml-schema asset table; navigate via "
                                "Execution_RID rather than pinning by asset RID"
                            ),
                        )
                    )
                    continue
                assets = self.list_assets(table)  # type: ignore[attr-defined]
                for asset in assets:
                    asset_rid = asset.asset_rid
                    filename = getattr(asset, "filename", None) or ""
                    config_name = _sanitize_config_name(filename, fallback=asset_rid)
                    suggestions.append(
                        BootstrapSuggestion(
                            kind="assets",
                            config_name=config_name,
                            rid=asset_rid,
                            spec_string=_format_asset_spec(asset_rid),
                            description=filename or None,
                            rationale=(
                                f"Asset in {table.name}"
                                + (f" ({filename})" if filename else "")
                            ),
                        )
                    )

        if "workflow" in requested_kinds:
            workflows = self.find_workflows()  # type: ignore[attr-defined]
            for wf in workflows:
                wf_rid = getattr(wf, "rid", None)
                if wf_rid is None:
                    continue  # in-memory Workflow without a catalog row
                wf_name = getattr(wf, "name", "") or ""
                config_name = _sanitize_config_name(wf_name, fallback=wf_rid)
                suggestions.append(
                    BootstrapSuggestion(
                        kind="workflow",
                        config_name=config_name,
                        rid=wf_rid,
                        spec_string=_format_workflow_spec(wf_rid),
                        description=wf_name or None,
                        rationale=(
                            "Existing Workflow row; pin by RID to reuse "
                            "across executions."
                        ),
                    )
                )

        return BootstrapReport(
            catalog={
                "hostname": str(getattr(self, "host_name", "")),
                "catalog_id": str(getattr(self, "catalog_id", "")),
            },
            suggestions=suggestions,
            skipped=skipped,
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

    def _validate_config_entries(
        self,
        entries: list[ConfigEntry],
    ) -> list[ConfigEntryResult]:
        """Validate parsed config entries against the catalog.

        Groups entries by kind and batches dataset validation through
        :meth:`validate_dataset_specs` (which has per-RID caching);
        loops the per-RID validators for assets and workflows;
        compares ``DerivaMLConfig`` entries against this connection's
        hostname/catalog_id.

        Args:
            entries: List of parsed entries from :func:`parse_config_file`.

        Returns:
            One :class:`ConfigEntryResult` per input entry, in the
            same order.
        """
        # Bucket by kind so dataset validation can be batched.
        dataset_entries: list[tuple[int, ConfigEntry]] = []
        asset_entries: list[tuple[int, ConfigEntry]] = []
        workflow_entries: list[tuple[int, ConfigEntry]] = []
        deriva_entries: list[tuple[int, ConfigEntry]] = []
        unresolvable_idx: set[int] = set()

        for i, e in enumerate(entries):
            if e.entry_kind == "DerivaMLConfig":
                deriva_entries.append((i, e))
                continue
            if e.rid is None:
                unresolvable_idx.add(i)
                continue
            if e.entry_kind == "DatasetSpecConfig":
                dataset_entries.append((i, e))
            elif e.entry_kind == "AssetSpecConfig":
                asset_entries.append((i, e))
            elif e.entry_kind == "Workflow":
                workflow_entries.append((i, e))

        results: list[ConfigEntryResult | None] = [None] * len(entries)

        # Unresolvable -- AST couldn't pull a RID. Surface, don't guess.
        for i in unresolvable_idx:
            results[i] = ConfigEntryResult(
                entry=entries[i],
                valid=False,
                reasons=["rid_unresolvable"],
            )

        # Datasets -- batch through validate_dataset_specs.
        if dataset_entries:
            specs_for_validate: list[DatasetSpec] = []
            for _, e in dataset_entries:
                # version_missing is OUR diagnostic; the singular
                # validator requires a version. Default missing
                # versions to a sentinel so they round-trip through
                # validate_dataset_specs as version_not_found, and we
                # rewrite the reason afterwards.
                version = e.version or "0.0.0"
                specs_for_validate.append(DatasetSpec(rid=e.rid, version=version))
            ds_report = self.validate_dataset_specs(specs=specs_for_validate)
            for (idx, entry), spec_result in zip(dataset_entries, ds_report.results):
                reasons: list[Any] = list(spec_result.reasons)
                if entry.version is None and "version_not_found" in reasons:
                    # Replace the misleading inner reason with our own.
                    reasons = [r for r in reasons if r != "version_not_found"]
                    reasons.append("version_missing")
                results[idx] = ConfigEntryResult(
                    entry=entry,
                    valid=spec_result.valid and entry.version is not None,
                    reasons=reasons,
                    actual_table=spec_result.actual_table,
                    available_versions=spec_result.available_versions,
                    resolved_name=spec_result.dataset_name,
                )

        # Assets -- per-RID loop (lookup_asset has no batched form yet).
        for idx, entry in asset_entries:
            asset_result = self._validate_asset_spec(AssetSpec(rid=entry.rid))
            results[idx] = ConfigEntryResult(
                entry=entry,
                valid=asset_result.valid,
                reasons=list(asset_result.reasons),
                actual_table=asset_result.actual_table,
                resolved_name=asset_result.filename,
            )

        # Workflows -- per-RID.
        for idx, entry in workflow_entries:
            wf_result = self._validate_workflow_rid(entry.rid)
            results[idx] = ConfigEntryResult(
                entry=entry,
                valid=wf_result.valid,
                reasons=list(wf_result.reasons),
                actual_table=wf_result.actual_table,
                resolved_name=wf_result.workflow_name,
            )

        # DerivaMLConfig -- compare against THIS connection's host/catalog.
        # No catalog call here; we just check the entry matches what
        # self is connected to. A heavy "is the catalog reachable"
        # check would re-issue every call validate_dataset_specs
        # already does -- caller decides separately whether to test
        # the connection (a single deriva_ml_list_datasets call is
        # enough).
        self_hostname = getattr(self, "host_name", None)
        self_catalog_id = getattr(self, "catalog_id", None)
        for idx, entry in deriva_entries:
            reasons: list[Any] = []
            if entry.hostname and self_hostname and entry.hostname != self_hostname:
                reasons.append("catalog_hostname_mismatch")
            if entry.catalog_id and self_catalog_id is not None:
                if str(entry.catalog_id) != str(self_catalog_id):
                    reasons.append("catalog_id_mismatch")
            results[idx] = ConfigEntryResult(
                entry=entry,
                valid=not reasons,
                reasons=reasons,
                resolved_name=(
                    f"{self_hostname or '?'}/{self_catalog_id or '?'}"
                    if not reasons
                    else None
                ),
            )

        # Replace any leftover Nones with an internal-error result.
        # Shouldn't happen if the bucketing covers every kind, but
        # defense in depth.
        return [
            r
            if r is not None
            else ConfigEntryResult(
                entry=entries[i],
                valid=False,
                reasons=["rid_unresolvable"],
            )
            for i, r in enumerate(results)
        ]

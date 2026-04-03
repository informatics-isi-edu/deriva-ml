from __future__ import annotations

import logging
from collections import defaultdict
from pprint import pformat
from typing import Any, Callable, Iterator

from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import tag as deriva_tags

from deriva_ml.core.constants import RID
from deriva_ml.interfaces import DatasetLike, DerivaMLCatalog

logger = logging.getLogger(__name__)

try:

    from icecream import ic

    ic.configureOutput(
        includeContext=True,
        argToStringFunction=lambda x: pformat(x.model_dump() if hasattr(x, "model_dump") else x, width=80, depth=10),
    )
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class CatalogGraph:
    """Generates export specifications and annotations for dataset downloads.

    This class creates the configuration needed for Deriva's export processor to
    download datasets as BDBags, optionally with S3 upload and MINID registration.

    Args:
        ml_instance: The DerivaML catalog instance.
        s3_bucket: S3 bucket URL for dataset bag storage (e.g., 's3://my-bucket').
            Required for MINID functionality. If None, MINID features are disabled.
        use_minid: Whether to use MINID service for persistent identification.
            Only effective when s3_bucket is provided.
        exclude_tables: Optional set of table names to exclude from FK path traversal.
            Tables in this set will not be visited during schema graph walking, pruning
            branches that pass through them. Useful for avoiding query timeouts caused
            by expensive joins through large or unnecessary tables.
    """

    def __init__(
        self,
        ml_instance: DerivaMLCatalog,
        s3_bucket: str | None = None,
        use_minid: bool = True,
        exclude_tables: set[str] | None = None,
    ):
        self._ml_schema = ml_instance.ml_schema
        self._ml_instance = ml_instance
        self._s3_bucket = s3_bucket
        # MINID only works if S3 bucket is configured
        self._use_minid = use_minid and s3_bucket is not None
        self._dataset_table = ml_instance._dataset_table
        self._exclude_tables = exclude_tables or set()

    def _export_annotation(
        self,
    ) -> list[dict[str, Any]]:
        """Return and output specification for the datasets in the provided model

        Returns:
          An export specification suitable for Chaise.
        """

        # Export specification is a specification for the datasets, plus any controlled vocabulary
        return [
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
        ] + self._dataset_specification(self._export_annotation_dataset_element, None)

    def _export_specification(self, dataset: DatasetLike) -> list[dict[str, Any]]:
        """
        Generate a specification for export engine for specific dataset.

        Returns:
          a download specification for the datasets in the provided model.

        """

        # Download spec is the spec for any controlled vocabulary and for the dataset_table.
        return [
            {
                "processor": "json",
                "processor_params": {"query_path": "/schema", "output_path": "schema"},
            }
        ] + self._dataset_specification(self._export_specification_dataset_element, dataset)

    @staticmethod
    def _export_specification_dataset_element(spath: str, dpath: str, table: Table) -> list[dict[str, Any]]:
        """Return the download specification for the data object indicated by a path through the data model.

        Args:
          spath: Source path
          dpath: Destination path
          table: Table referenced to by the path

        Returns:
          The download specification that will retrieve that data from the catalog and place it into a BDBag.
        """
        exports = [
            {
                "processor": "csv",
                "processor_params": {
                    "query_path": f"/entity/{spath}",
                    "output_path": dpath,
                    "paged_query": True,
                },
            }
        ]

        # If this table is an asset table, then we need to output the files associated with the asset.
        if table.is_asset():
            exports.append(
                {
                    "processor": "fetch",
                    "processor_params": {
                        "query_path": f"/attribute/{spath}/url:=URL,length:=Length,filename:=Filename,md5:=MD5,asset_rid:=RID",
                        "output_path": "asset/{asset_rid}/" + table.name,
                    },
                }
            )
        return exports

    def _export_annotation_dataset_element(self, spath: str, dpath: str, table: Table) -> list[dict[str, Any]]:
        """Given a path in the data model, output an export specification for the path taken to get to the
        current table.

        Args:
          spath: Source path
          dpath: Destination path
          table: Table referenced to by the path

        Returns:
          The export specification that will retrieve that data from the catalog and place it into a BDBag.
        """
        # The table is the last element of the path.  Generate the ERMRest query by converting the list of tables
        # into a path in the form of /S:T1/S:T2/S:Table
        # Generate the destination path in the file system using just the table names.

        skip_root_path = False
        if spath.startswith(f"{self._ml_schema}:Dataset/"):
            # Chaise will add table name and RID filter, so strip it off.
            spath = "/".join(spath.split("/")[2:])
            if spath == "":
                # This path is to just the dataset table.
                return []
        else:
            # A vocabulary table, so we don't want the root_path.
            skip_root_path = True
        exports = [
            {
                "source": {
                    "api": "entity",
                    "path": spath,
                    "skip_root_path": skip_root_path,
                },
                "destination": {"name": dpath, "type": "csv"},
            }
        ]

        # If this table is an asset table, then we need to output the files associated with the asset.
        if table.is_asset():
            exports.append(
                {
                    "source": {
                        "skip_root_path": False,
                        "api": "attribute",
                        "path": f"{spath}/url:=URL,length:=Length,filename:=Filename,md5:=MD5,asset_rid:=RID",
                    },
                    "destination": {"name": "asset/{asset_rid}/" + table.name, "type": "fetch"},
                }
            )
        return exports

    def generate_dataset_download_spec(self, dataset: DatasetLike) -> dict[str, Any]:
        """Generate a specification for downloading a specific dataset.

        This routine creates a download specification that can be used by the Deriva
        export processor to download a specific dataset as a BDBag. If s3_bucket is
        configured and use_minid is True, the bag will be uploaded to S3 and
        registered with the MINID service.

        Args:
            dataset: The dataset to generate the download spec for.

        Returns:
            A download specification dictionary for the Deriva export processor.
        """
        minid_test = False

        post_processors: dict[str, Any] = {}
        if self._use_minid and self._s3_bucket:
            post_processors = {
                "post_processors": [
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
                            "test": minid_test,
                            "env_column_map": {
                                "RID": "{RID}@{snaptime}",
                                "Description": "{Description}",
                            },
                        },
                    },
                ]
            }
        return post_processors | {
            "env": {"RID": "{RID}"},
            "bag": {
                "bag_name": "Dataset_{RID}",
                "bag_algorithms": ["md5"],
                "bag_archiver": "zip",
                "bag_metadata": {},
                "bag_idempotent": True,
            },
            "catalog": {
                "host": f"{self._ml_instance.catalog.deriva_server.scheme}://{self._ml_instance.catalog.deriva_server.server}",
                "catalog_id": self._ml_instance.catalog_id,
                "query_processors": [
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
                            "query_path": "/entity/M:=deriva-ml:Dataset/RID={RID}",
                            "output_path": "Dataset",
                            "query_keys": ["RID", "Description"],
                        },
                    },
                ]
                + self._export_specification(dataset),
            },
        }

    def generate_dataset_download_annotations(self) -> dict[str, Any]:
        """Generate export annotations for the Dataset table.

        These annotations configure Chaise's export functionality for datasets.
        If s3_bucket is configured and use_minid is True, includes post-processors
        for S3 upload and MINID registration.

        Returns:
            A dictionary of annotations to apply to the Dataset table.
        """
        post_processors: dict[str, Any] = {}
        if self._use_minid and self._s3_bucket:
            # Ensure the S3 bucket URL ends with a trailing slash for the annotation
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

    def _dataset_visible_fkeys(self) -> dict[str, Any]:
        """Build visible-foreign-keys annotation for the Dataset table.

        Generates the Chaise annotation that controls which related tables
        appear in the detailed view of a Dataset record, including parent/child
        dataset links and all dataset element type associations.

        Returns:
            A dict with a ``"detailed"`` key containing the ordered list of
            foreign key source specifications for Chaise.
        """
        def fkey_name(fk):
            return [fk.name[0].name, fk.name[1]]

        dataset_table = self._ml_instance.model.schemas["deriva-ml"].tables["Dataset"]

        source_list = [
            {
                "source": [
                    {"inbound": ["deriva-ml", "Dataset_Version_Dataset_fkey"]},
                    "RID",
                ],
                "markdown_name": "Previous Versions",
                "entity": True,
            },
            {
                "source": [
                    {"inbound": ["deriva-ml", "Dataset_Dataset_Nested_Dataset_fkey"]},
                    {"outbound": ["deriva-ml", "Dataset_Dataset_Dataset_fkey"]},
                    "RID",
                ],
                "markdown_name": "Parent Datasets",
            },
            {
                "source": [
                    {"inbound": ["deriva-ml", "Dataset_Dataset_Dataset_fkey"]},
                    {"outbound": ["deriva-ml", "Dataset_Dataset_Nested_Dataset_fkey"]},
                    "RID",
                ],
                "markdown_name": "Child Datasets",
            },
        ]
        source_list.extend(
            [
                {
                    "source": [
                        {"inbound": fkey_name(fkey.self_fkey)},
                        {"outbound": fkey_name(other_fkey := fkey.other_fkeys.pop())},
                        "RID",
                    ],
                    "markdown_name": other_fkey.pk_table.name,
                }
                for fkey in dataset_table.find_associations(max_arity=3, pure=False)
            ]
        )
        return {"detailed": source_list}

    def _collect_paths(
        self,
        dataset_rid: RID | None = None,
        dataset_nesting_depth: int | None = None,
    ) -> set[tuple[Table, ...]]:
        """
        Collects all schema paths relevant to a specific dataset, optionally filtered by dataset membership or nesting
        depth, and returns those paths. The paths represent relationships between tables in the schema and how they can
        be traversed based on the dataset's structure and context.

        When a dataset_rid is provided, paths are filtered by **association filtering**: only paths whose
        association table (p[1]) corresponds to an element type with members in this dataset are included.
        Paths starting from element types without members are excluded. Once a path starts from a member
        element type, all FK-reachable tables are followed (both incoming and outgoing foreign keys),
        with vocabulary tables acting as natural path terminators.

        Paths ending in vocabulary tables are removed since vocabulary tables are exported in full
        separately by ``_export_vocabulary()``. This avoids redundant multi-table joins that can
        cause server-side query timeouts on large catalogs.

        Feature tables are included for element types reachable via the filtered paths.

        Args:
            dataset_rid:
                An optional identifier for the specific dataset to filter paths. If provided,
                only paths traversing elements of this dataset will be included.
            dataset_nesting_depth:
                Specifies the depth to which nested datasets should be included. If not provided,
                a default depth is calculated based on the current instance.

        Returns:
            set[tuple[Table, ...]]:
                A set of tuples, where each tuple represents a valid path consisting of
                Tables. Each path defines how tables are connected and can be navigated
                through the schema.
        """

        dataset_table = self._ml_instance.model.schemas[self._ml_schema].tables["Dataset"]
        dataset_dataset = self._ml_instance.model.schemas[self._ml_schema].tables["Dataset_Dataset"]

        # Figure out what types of elements the dataset contains.
        # dataset_association_tables: all association linking tables (e.g., Dataset_Subject, Dataset_Image)
        # Include associations from ALL schemas so that core associations like File_Dataset,
        # Dataset_Execution, etc. are properly filtered when they have no members.
        dataset_association_tables = [a.table for a in self._dataset_table.find_associations()]

        if dataset_rid:
            # Get a list of the members of the dataset so we can figure out which tables to query.
            dataset = self._ml_instance.lookup_dataset(dataset_rid)
            dataset_elements = [
                self._ml_instance.model.name_to_table(e) for e, m in dataset.list_dataset_members().items() if m
            ]
            # included_associations: association tables whose target element type has members,
            # plus association tables whose target is a vocabulary (e.g., Dataset_Dataset_Type).
            # Vocabulary-targeting associations carry dataset metadata and must always be included.
            vocab_tables = {
                table
                for s in self._ml_instance.model.schemas.values()
                for table in s.tables.values()
                if self._ml_instance.model.is_vocabulary(table)
            }
            included_associations = [
                a.table for a in dataset_table.find_associations()
                if any(fk.pk_table in dataset_elements for fk in a.other_fkeys)
                or any(fk.pk_table in vocab_tables for fk in a.other_fkeys)
            ]
        else:
            included_associations = dataset_association_tables

        # Get the paths through the schema and filter out all the dataset paths not used by this dataset.
        paths = set()
        for p in self._ml_instance.model._schema_to_paths(exclude_tables=self._exclude_tables):
            tp = tuple(p)
            if (len(p) == 1) or (p[1] not in dataset_association_tables) or (p[1] in included_associations):
                paths.add(tp)

        if dataset_rid:
            # Get ALL dataset element types in the catalog (tables with Dataset_X associations)
            all_element_types = set(self._ml_instance.model.list_dataset_element_types())

            # Remove paths ending in vocabulary tables. Vocabulary tables are exported
            # in full by _export_vocabulary() as standalone queries, so path-based
            # joins to vocab tables are redundant and can cause server-side query
            # timeouts when the join chain is deep.
            vocab_tables = {
                table
                for s in self._ml_instance.model.schemas.values()
                for table in s.tables.values()
                if self._ml_instance.model.is_vocabulary(table)
            }
            paths = {p for p in paths if p[-1] not in vocab_tables}

            # Add feature table paths for member element types reachable via paths.
            reachable_element_types = {
                table for path in paths for table in path if table in all_element_types
            }
            for element_table in reachable_element_types:
                for feature in self._ml_instance.find_features(element_table):
                    for path in paths.copy():
                        if path[-1] == element_table:
                            paths.add(path + (feature.feature_table,))

        # Now get paths for nested datasets
        nested_paths = set()
        if dataset_rid:
            dataset = self._ml_instance.lookup_dataset(dataset_rid)
            for c in dataset.list_dataset_children():
                nested_paths |= self._collect_paths(c.dataset_rid)
        else:
            # Initialize nesting depth if not already provided.
            dataset_nesting_depth = (
                self._dataset_nesting_depth() if dataset_nesting_depth is None else dataset_nesting_depth
            )
            if dataset_nesting_depth:
                nested_paths = self._collect_paths(dataset_nesting_depth=dataset_nesting_depth - 1)
        if nested_paths:
            paths |= {
                tuple([dataset_table]),
                (dataset_table, dataset_dataset),
            }
        paths |= {(self._dataset_table, dataset_dataset) + p for p in nested_paths}
        return paths

    def _export_vocabulary(self, writer: Callable[[str, str, Table], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Generate export entries for all vocabulary tables in the catalog.

        Vocabulary tables are exported in full (not filtered by dataset membership)
        since they provide shared controlled terms used across all datasets.

        Args:
            writer: A callable that generates export spec entries for a single table.
                Receives (source_path, dest_path, table) and returns a list of
                export specification dictionaries.

        Returns:
            A list of export specification dictionaries for all vocabulary tables.
        """
        vocabs = [
            table
            for s in self._ml_instance.model.schemas.values()
            for table in s.tables.values()
            if self._ml_instance.model.is_vocabulary(table)
        ]
        return [o for table in vocabs for o in writer(f"{table.schema.name}:{table.name}", table.name, table)]

    def _build_linked_datapath(
        self,
        path: tuple[Table, ...],
        pb: Any,
        root_dp: Any = None,
    ) -> tuple[Any, Any]:
        """Build a linked datapath by walking an FK path through the schema.

        Handles Dataset_Dataset nesting and composite FK ON-clauses. Used by
        both ``_table_paths`` (for export specs) and ``_aggregate_queries``
        (for size estimation) to avoid duplicating the FK-linking logic.

        Args:
            path: Tuple of ermrest Table objects representing the FK path.
            pb: The pathBuilder instance for looking up tables.
            root_dp: Optional pre-built datapath for the root table (e.g., with
                a filter applied). If None, starts from ``pb_table(path[0])``.

        Returns:
            ``(datapath, root_pb_table)`` where *datapath* is the fully linked
            pathBuilder object and *root_pb_table* is the pathBuilder table
            for the first table in the path.
        """
        def _pb_table(table: Table):
            return pb.schemas[table.schema.name].tables[table.name]

        def _build_on_clause(prev_model_table, table, prev_pb_table, pb_table):
            """Build an explicit on= clause for composite FKs.

            Returns None for simple FKs (datapath resolves them implicitly).
            """
            try:
                col_pairs = self._ml_instance.model._table_relationship(prev_model_table, table)
            except Exception:
                return None

            if len(col_pairs) <= 1:
                return None

            conditions = []
            for fk_col, pk_col in col_pairs:
                if fk_col.table.name == prev_model_table.name:
                    left = getattr(prev_pb_table, fk_col.name)
                    right = getattr(pb_table, pk_col.name)
                else:
                    left = getattr(prev_pb_table, pk_col.name)
                    right = getattr(pb_table, fk_col.name)
                conditions.append(left == right)

            result = conditions[0]
            for cond in conditions[1:]:
                result = result & cond
            return result

        dd_table = _pb_table(
            self._ml_instance.model.schemas[self._ml_schema].tables["Dataset_Dataset"]
        )

        ds_pb = _pb_table(path[0])
        dp = root_dp if root_dp is not None else ds_pb

        prev_table = path[0]
        prev_pb = ds_pb
        for table in path[1:]:
            cur_pb = _pb_table(table)
            if table.name == "Dataset_Dataset":
                dp = dp.link(cur_pb)
            elif table.name == "Dataset" and prev_table.name == "Dataset_Dataset":
                dp = dp.link(cur_pb, on=(dd_table.Nested_Dataset == cur_pb.RID))
            else:
                on_clause = _build_on_clause(prev_table, table, prev_pb, cur_pb)
                if on_clause is not None:
                    dp = dp.link(cur_pb, on=on_clause)
                else:
                    dp = dp.link(cur_pb)
            prev_table = table
            prev_pb = cur_pb

        return dp, ds_pb

    def _table_paths(
        self,
        dataset: DatasetLike | None = None,
    ) -> Iterator[tuple[str, str, Table]]:
        """Build ERMrest source paths, destination paths, and target tables for export.

        For each FK path from ``_collect_paths``, constructs an ERMrest query path
        using the datapath API with correct FK resolution (including composite FKs).

        Args:
            dataset: If provided, paths are filtered to this dataset's members.

        Returns:
            Iterator of ``(source_path, dest_path, target_table)`` tuples.
        """
        # Sort paths for deterministic spec generation (set iteration order is arbitrary).
        paths = sorted(
            self._collect_paths(dataset and dataset.dataset_rid),
            key=lambda p: tuple(f"{t.schema.name}:{t.name}" for t in p),
        )
        pb = self._ml_instance.catalog.getPathBuilder()

        def source_path(path: tuple[Table, ...]) -> str:
            """Build an ERMrest query path with RID={RID} placeholder."""
            dp, _ = self._build_linked_datapath(path, pb)

            uri = dp.uri
            entity_prefix = "/entity/"
            idx = uri.index(entity_prefix)
            entity_path = uri[idx + len(entity_prefix):]

            parts = entity_path.split("/", 1)
            if len(parts) == 1:
                return f"{parts[0]}/RID={{RID}}"
            return f"{parts[0]}/RID={{RID}}/{parts[1]}"

        src_paths = [source_path(p) for p in paths]
        dest_paths = ["/".join([t.name for t in p]) for p in paths]
        target_tables = [p[-1] for p in paths]
        return zip(src_paths, dest_paths, target_tables)

    def _aggregate_queries(
        self,
        dataset: DatasetLike | None = None,
    ) -> dict[str, list[tuple[Any, Any, bool]]]:
        """Build datapath objects grouped by target table name.

        For each path collected by ``_collect_paths``, constructs a linked
        datapath (using the same FK-linking logic as ``_table_paths``) filtered
        on the dataset RID.  Results are grouped by the terminal table name so
        that callers can issue aggregate queries for every path that reaches a
        given table.

        Args:
            dataset: If provided, paths are filtered to this dataset's members.

        Returns:
            A dict mapping target table name to a list of
            ``(datapath, target_pb_table, is_asset)`` tuples, where *datapath*
            is the linked pathBuilder object, *target_pb_table* is the
            pathBuilder table reference for the terminal table (needed to
            access column attributes like ``.RID``), and *is_asset* is
            ``True`` when the target table contains asset columns
            (Filename, URL, Length, MD5, Description).
        """
        paths = self._collect_paths(dataset and dataset.dataset_rid)
        pb = self._ml_instance.catalog.getPathBuilder()

        def _pb_table(table: Table):
            return pb.schemas[table.schema.name].tables[table.name]

        result: dict[str, list[tuple[Any, bool]]] = defaultdict(list)

        for path in paths:
            ds_pb = _pb_table(path[0])
            root_dp = ds_pb.filter(ds_pb.RID == dataset.dataset_rid) if dataset and dataset.dataset_rid else None
            dp, _ = self._build_linked_datapath(path, pb, root_dp=root_dp)

            target_table = path[-1]
            target_pb_table = _pb_table(target_table) if len(path) > 1 else ds_pb
            is_asset = target_table.is_asset()
            result[target_table.name].append((dp, target_pb_table, is_asset))

        return dict(result)

    def _dataset_nesting_depth(self, dataset: DatasetLike | None = None) -> int:
        """Determine the maximum dataset nesting depth in the current catalog.

        Args:
            dataset: If provided, compute depth for this dataset's subtree only.
                If None, computes the maximum depth across all datasets.

        Returns:
            The maximum nesting depth (0 if no nested datasets exist).
        """

        def children_depth(dataset: RID, nested_datasets: dict[str, list[str]]) -> int:
            """Return the number of nested datasets for the dataset_rid if provided, otherwise in the current catalog"""
            try:
                children = nested_datasets[dataset]
                return max(map(lambda x: children_depth(x, nested_datasets), children)) + 1 if children else 1
            except KeyError:
                return 0

        # Build up the dataset_table nesting graph...
        pb = self._ml_instance.catalog.getPathBuilder().schemas[self._ml_schema].tables["Dataset_Dataset"]
        dataset_children = (
            [
                {
                    "Dataset": dataset.dataset_rid,
                    "Nested_Dataset": c,
                }  # Make uniform with return from datapath
                for c in dataset.list_dataset_children()
            ]
            if dataset
            else pb.entities().fetch()
        )
        nested_dataset = defaultdict(list)
        for ds in dataset_children:
            nested_dataset[ds["Dataset"]].append(ds["Nested_Dataset"])
        return max(map(lambda d: children_depth(d, dict(nested_dataset)), nested_dataset)) if nested_dataset else 0

    def _dataset_specification(
        self,
        writer: Callable[[str, str, Table], list[dict[str, Any]]],
        dataset: DatasetLike | None = None,
    ) -> list[dict[str, Any]]:
        """Output a download/export specification for a dataset_table.  Each element of the dataset_table
        will be placed in its own directory.
        The top level data directory of the resulting BDBag will have one subdirectory for element type.
        The subdirectory will contain the CSV indicating which elements of that type are present in the
        dataset_table, and then there will be a subdirectory for each object that is reachable from the
        dataset_table members.

        To simplify reconstructing the relationship between tables, the CVS for each element is included.
        The top level data directory will also contain a subdirectory for any controlled vocabularies used in
        the dataset_table. All assets will be placed into a directory named asset in a subdirectory with the
        asset table name.

        For example, consider a dataset_table that consists of two element types, T1 and T2. T1 has foreign
        key relationships to objects in tables T3 and T4. There are also two controlled vocabularies, CV1 and
        CV2. T2 is an asset table which has two assets in it. The layout of the resulting bdbag would be:
              data
                CV1/
                    cv1.csv
                CV2/
                    cv2.csv
                Dataset/
                    T1/
                        t1.csv
                        T3/
                            t3.csv
                        T4/
                            t4.csv
                    T2/
                        t2.csv
                asset/
                  T2
                    f1
                    f2

        Args:
          writer: Callable[[list[Table]]: list[dict[str:  Any]]]:

        Returns:
            A dataset_table specification.
        """
        element_spec = self._export_vocabulary(writer)
        for path in self._table_paths(dataset=dataset):
            element_spec.extend(writer(*path))
        return element_spec

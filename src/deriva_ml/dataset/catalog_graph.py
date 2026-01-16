from __future__ import annotations

from collections import defaultdict
from pprint import pformat
from typing import Any, Callable, Iterator

from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import tag as deriva_tags

from deriva_ml.core.constants import RID
from deriva_ml.interfaces import DatasetLike, DerivaMLCatalog

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
    """

    def __init__(
        self,
        ml_instance: DerivaMLCatalog,
        s3_bucket: str | None = None,
        use_minid: bool = True,
    ):
        self._ml_schema = ml_instance.ml_schema
        self._ml_instance = ml_instance
        self._s3_bucket = s3_bucket
        # MINID only works if S3 bucket is configured
        self._use_minid = use_minid and s3_bucket is not None
        self._dataset_table = ml_instance._dataset_table

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
                },
            }
        ]

        # If this table is an asset table, then we need to output the files associated with the asset.
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        if asset_columns.issubset({c.name for c in table.columns}):
            exports.append(
                {
                    "processor": "fetch",
                    "processor_params": {
                        "query_path": f"/attribute/{spath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5,asset_rid:=RID",
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
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        if asset_columns.issubset({c.name for c in table.columns}):
            exports.append(
                {
                    "source": {
                        "skip_root_path": False,
                        "api": "attribute",
                        "path": f"{spath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5, asset_rid:=RID",
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
        dataset_associations = [
            a
            for a in self._dataset_table.find_associations()
            if a.table.schema.name != self._ml_schema or a.table.name == "Dataset_Dataset"
        ]

        if dataset_rid:
            # Get a list of the members of the dataset so we can figure out which tables to query.
            dataset = self._ml_instance.lookup_dataset(dataset_rid)
            dataset_elements = [
                self._ml_instance.model.name_to_table(e) for e, m in dataset.list_dataset_members().items() if m
            ]
            included_associations = [
                a.table for a in dataset_table.find_associations() if a.other_fkeys.pop().pk_table in dataset_elements
            ]
        else:
            included_associations = [a.table for a in dataset_associations]

        # Get the paths through the schema and filter out all the dataset paths not used by this dataset.
        paths = {
            tuple(p)
            for p in self._ml_instance.model._schema_to_paths()
            if (len(p) == 1)
            or (p[1] not in dataset_associations)  # Tables in the domain schema
            or (p[1] in included_associations)  # Tables that include members of the dataset
        }

        # Add feature table paths for domain tables in the dataset
        # Feature tables (e.g., Execution_Image_Image_Classification) contain feature values
        # that need to be exported with the dataset
        if dataset_rid:
            for element_table in dataset_elements:
                for feature in self._ml_instance.find_features(element_table):
                    # Find the path to the element table and extend it with the feature table
                    for path in paths.copy():
                        if path[-1] == element_table:
                            # Add a path that goes through the element table to the feature table
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
        """

        Args:
          writer: Callable[[list[Table]]: list[dict[str: Any]]]:

        Returns:

        """
        vocabs = [
            table
            for s in self._ml_instance.model.schemas.values()
            for table in s.tables.values()
            if self._ml_instance.model.is_vocabulary(table)
        ]
        return [o for table in vocabs for o in writer(f"{table.schema.name}:{table.name}", table.name, table)]

    def _table_paths(
        self,
        dataset: DatasetLike | None = None,
    ) -> Iterator[tuple[str, str, Table]]:
        paths = self._collect_paths(dataset and dataset.dataset_rid)

        def source_path(path: tuple[Table, ...]) -> list[str]:
            """Convert a tuple representing a path into a source path component with FK linkage"""
            path = list(path)
            p = [f"{self._ml_instance.ml_schema}:Dataset/RID={{RID}}"]
            for table in path[1:]:
                if table.name == "Dataset_Dataset":
                    p.append("(RID)=(deriva-ml:Dataset_Dataset:Dataset)")
                elif table.name == "Dataset":
                    p.append("(Nested_Dataset)=(deriva-ml:Dataset:RID)")
                elif table.name == "Dataset_Version":
                    p.append(f"(RID)=({self._ml_instance.ml_schema}:Dataset_Version:Dataset)")
                else:
                    p.append(f"{table.schema.name}:{table.name}")
            return p

        src_paths = ["/".join(source_path(p)) for p in paths]
        dest_paths = ["/".join([t.name for t in p]) for p in paths]
        target_tables = [p[-1] for p in paths]
        return zip(src_paths, dest_paths, target_tables)

    def _dataset_nesting_depth(self, dataset: DatasetLike | None = None) -> int:
        """Determine the maximum dataset nesting depth in the current catalog.

        Returns:

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

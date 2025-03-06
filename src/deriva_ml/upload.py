"""
This module provides functions that help structure local directories for uploading to a DerivaML catalog, and
generating an upload specification for those directories.

Here is the directory layout we support:

  deriva-ml/
       execution
           <execution_rid>
               execution-asset
                   <asset_type>
                       file1, file2, ....   <- Need to update execution_asset association table.
               execution-metadata
                   <metadata_type>
               feature
                   <schema>
                       <target_table>
                            <feature_name>
                                   asset
                                       <asset_table>
                                           file1, file2, ...
                           <feature_name>.csv    <- needs to have asset_name column remapped before uploading
            table
               <schema>
                   <record_table>
                      record_table.csv
            asset
               <schema>
                   <asset_table>
                     file1, file2, ....
        asset
            <schema>
                <asset_table>
                    <metadata1>
                        <metadata2>
                            file1, file2, ....

"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import regex as re
from deriva.core import urlquote
from deriva.core.ermrest_model import Table
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.transfer.upload.deriva_upload import GenericUploader
from pydantic import validate_call, ConfigDict

from deriva_ml.deriva_definitions import (
    RID,
    DerivaMLException,
    FileUploadState,
    UploadState,
)
from deriva_ml.deriva_model import DerivaModel

upload_root_regex = r"(?i)^.*/deriva-ml"

exec_dir_regex = upload_root_regex + r"/execution/(?P<execution_rid>[-\w]+)"
exec_asset_dir_regex = (
    exec_dir_regex + r"/execution-asset/(?P<execution_asset_type>[-\w]+)"
)
exec_asset_regex = (
    exec_asset_dir_regex + r"/(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]+)$"
)
exec_metadata_dir_regex = (
    exec_dir_regex + r"/execution-metadata/(?P<execution_metadata_type>[-\w]+)"
)
exec_metadata_regex = (
    exec_metadata_dir_regex + r"/(?P<filename>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$"
)
feature_dir_regex = exec_dir_regex + r"/feature"
feature_table_dir_regex = (
    feature_dir_regex
    + r"/(?P<schema>[-\w]+)/(?P<target_table>[-\w]+)/(?P<feature_name>[-\w]+)"
)
feature_value_regex = (
    feature_table_dir_regex + r"/(?P=feature_name)[.](?P<file_ext>[(csv|json)]*)$"
)
feature_asset_dir_regex = feature_table_dir_regex + r"/asset/(?P<asset_table>[-\w]+)"
feature_asset_regex = (
    feature_asset_dir_regex
    + r"/(?P<file_name>[A-Za-z0-9_-]+)[.](?P<file_ext>[a-z0-9]*)$"
)
asset_path_regex = (
    upload_root_regex + r"/asset/(?P<schema>[-\w]+)/(?P<asset_table>[-\w]*)"
)

asset_file_regex = r"(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$"

table_regex = (
    exec_dir_regex
    + r"/table/(?P<schema>[-\w]+)/(?P<table>[-\w]+)/(?P=table)[.](csv|json)$"
)


def is_execution_metadata_dir(path: Path) -> Optional[re.Match]:
    """Path matches the patten for execution metadata directory."""
    return re.match(exec_metadata_dir_regex + "$", path.as_posix())


def is_execution_asset_dir(path: Path) -> Optional[re.Match]:
    """Path maths the pattern for execution asset directory"""
    return re.match(exec_asset_dir_regex + "$", path.as_posix())


def is_feature_dir(path: Path) -> Optional[re.Match]:
    """Path matches the pattern for where the table for a feature would go."""
    return re.match(feature_table_dir_regex + "$", path.as_posix())


def is_feature_asset_dir(path: Path) -> Optional[re.Match]:
    """Path matches the pattern for feature, asset, returns the feature components."""
    return re.match(feature_asset_dir_regex + "$", path.as_posix())


def upload_root(prefix: Path | str) -> Path:
    """Return the top level directory of where to put files to be uploaded."""
    path = Path(prefix) / "deriva-ml"
    path.mkdir(exist_ok=True, parents=True)
    return path


def asset_root(prefix: Path | str) -> Path:
    """Return the top level directory of where to put asset directories to be uploaded."""
    path = upload_root(prefix) / "asset"
    path.mkdir(exist_ok=True, parents=True)
    return path


def asset_table_dir(prefix: Path, asset_schema, asset_table) -> Path:
    """Return the top level directory of where to asset table will be uploaded."""
    path = asset_root(prefix) / asset_schema / asset_table
    path.mkdir(exist_ok=True, parents=True)
    return path


def execution_rids(prefix: Path | str) -> list[RID]:
    """Return a list of all the execution RIDS that have files waiting to be uploaded."""
    path = upload_root(prefix) / "execution"
    return [d.name for d in path.iterdir()]


def execution_root(prefix: Path | str, exec_rid) -> Path:
    """Path to directory to place execution specific upload files."""
    path = upload_root(prefix) / "execution" / exec_rid
    path.mkdir(exist_ok=True, parents=True)
    return path


def execution_asset_root(prefix: Path | str, exec_rid: str) -> Path:
    """Path to directory into which execution assets should be located"""

    path = execution_root(prefix, exec_rid) / "execution-asset"
    path.mkdir(parents=True, exist_ok=True)
    return path


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def execution_asset_dir(prefix: Path | str, exec_rid: str, asset_type: str) -> Path:
    """Return the path to a directory in which to place execution assets of a specified type are to be uploaded."""
    path = execution_asset_root(prefix, exec_rid) / asset_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_metadata_root(prefix: Path | str, exec_rid: str) -> Path:
    """Path to directory into which execution metadata should be located."""
    path = execution_root(prefix, exec_rid) / "execution-metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_metadata_dir(
    prefix: Path | str, exec_rid: str, metadata_type: str
) -> Path:
    """Return the path to a directory in which to place execution metadata of a specified type are to be uploaded."""

    path = execution_metadata_root(prefix, exec_rid) / metadata_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_root(prefix: Path | str, exec_rid: str) -> Path:
    """Return the path to the directory in which features for the specified execution should be placed."""
    path = execution_root(prefix, exec_rid) / "feature"
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_dir(
    prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str
) -> Path:
    """Return the path to eht directory in which a named feature for an execution should be placed."""
    path = feature_root(prefix, exec_rid) / schema / target_table / feature_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_value_path(
    prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str
) -> Path:
    """Return the path to a CSV file in which to place feature values that are to be uploaded.

    Args:
        prefix: Location of upload root directory
        exec_rid: RID of the execution to be associated with this feature.
        schema: Domain schema name
        target_table: Target table name for the feature.
        feature_name: Name of the feature.

    Returns:
        Path to CSV file in which to place feature values
    """
    return (
        feature_dir(prefix, exec_rid, schema, target_table, feature_name)
        / f"{feature_name}.csv"
    )


def feature_asset_dir(
    prefix: Path | str,
    exec_rid: str,
    schema: str,
    target_table: str,
    feature_name: str,
    asset_table: str,
) -> Path:
    """Return the path to a directory in which to place feature assets for a named feature are to be uploaded.

    Args:
        prefix: Location of upload root directory
        exec_rid: RID of the execution for the feature asset
        schema: Domain schema
        target_table: Name of the target table for the feature.
        feature_name: Name of the feature
        asset_table: Name of the asset table for the feature.

    Returns:
        Path to directory in which feature asset files are placed.
    """
    path = (
        feature_dir(prefix, exec_rid, schema, target_table, feature_name)
        / "asset"
        / asset_table
    )

    path.mkdir(parents=True, exist_ok=True)
    return path


def table_path(prefix: Path | str, schema: str, table: str) -> Path:
    """Return the path to a CSV file in which to place table values that are to be uploaded.

    Args:
        prefix: Location of upload root directory
        schema: Domain schema
        table: Name of the table to be uploaded.

    Returns:
        Path to the file in which to place table values that are to be uploaded.
    """
    path = upload_root(prefix) / "table" / schema / table
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{table}.csv"


def asset_table_upload_spec(model: DerivaModel, asset_table: str | Table):
    """Generate a pattern to an asset table that may include additional metadata columns.

    Args:
        model:
        asset_table:

    Returns:

    """
    metadata_columns = model.asset_metadata(asset_table)
    asset_table = model.name_to_table(asset_table)
    schema = model.name_to_table(asset_table).schema.name
    metadata_path = "/".join([rf"(?P<{c}>[-\w]+)" for c in metadata_columns])
    asset_path = f"{upload_root_regex}/asset/{schema}/{asset_table.name}/{metadata_path}/{asset_file_regex}"
    asset_table = model.name_to_table(asset_table)
    schema = model.name_to_table(asset_table).schema.name
    return {
        # Upload assets into an asset table of an asset table.
        "column_map": {
            "MD5": "{md5}",
            "URL": "{URI}",
            "Length": "{file_size}",
            "Filename": "{file_name}",
        }
        | {c: f"{{{c}}}" for c in metadata_columns},
        "file_pattern": asset_path,  # Sets schema, asset_table, file_name, file_ext
        "target_table": [schema, asset_table.name],
        "checksum_types": ["sha256", "md5"],
        "hatrac_options": {"versioned_urls": True},
        "hatrac_templates": {
            "hatrac_uri": f"/hatrac/{asset_table.name}/{{md5}}.{{file_name}}",
            "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}",
        },
        "record_query_template": "/entity/{target_table}/MD5={{md5}}&Filename={{file_name}}",
    }


def bulk_upload_configuration(model: DerivaModel) -> dict[str, Any]:
    """Return an upload specification for deriva-ml
    Arguments:
        model: Model from which to generate the upload configuration
    """
    asset_tables_with_metadata = [
        asset_table_upload_spec(model=model, asset_table=t)
        for t in model.find_assets()
        if model.asset_metadata(t)
    ]
    return {
        "asset_mappings": [
            {
                # Upload  any files that may have been created by the program execution.  These are  in the
                # Execution_Metadata directory
                "column_map": {
                    "MD5": "{md5}",
                    "URL": "{URI}",
                    "Length": "{file_size}",
                    "Filename": "{file_name}",
                    "Execution_Metadata_Type": "{execution_metadata_type_name}",
                },
                "file_pattern": exec_metadata_regex,
                "target_table": ["deriva-ml", "Execution_Metadata"],
                "checksum_types": ["sha256", "md5"],
                "hatrac_options": {"versioned_urls": True},
                "hatrac_templates": {
                    "hatrac_uri": "/hatrac/execution_metadata/{md5}.{file_name}",
                    "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}",
                },
                "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
                "metadata_query_templates": [
                    "/attribute/deriva-ml:Execution_Metadata_Type/Name={execution_metadata_type}/execution_metadata_type_name:=Name"
                ],
            },
            {
                # Upload the contents of the Execution_Asset directory.
                "column_map": {
                    "MD5": "{md5}",
                    "URL": "{URI}",
                    "Length": "{file_size}",
                    "Filename": "{file_name}",
                    "Execution_Asset_Type": "{execution_asset_type_name}",
                },
                "file_pattern": exec_asset_regex,
                "target_table": ["deriva-ml", "Execution_Asset"],
                "checksum_types": ["sha256", "md5"],
                "hatrac_options": {"versioned_urls": True},
                "hatrac_templates": {
                    "hatrac_uri": "/hatrac/execution_asset/{md5}.{file_name}",
                    "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}",
                },
                "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
                "metadata_query_templates": [
                    "/attribute/deriva-ml:Execution_Asset_Type/Name={execution_asset_type}/execution_asset_type_name:=Name"
                ],
            },
            {
                # Upload the assets for a feature table.
                "column_map": {
                    "MD5": "{md5}",
                    "URL": "{URI}",
                    "Length": "{file_size}",
                    "Filename": "{file_name}",
                },
                "file_pattern": feature_asset_regex,  # Sets target_table, feature_name, asset_table
                "target_table": ["{schema}", "{asset_table}"],
                "checksum_types": ["sha256", "md5"],
                "hatrac_options": {"versioned_urls": True},
                "hatrac_templates": {
                    "hatrac_uri": "/hatrac/{asset_table}/{md5}.{file_name}",
                    "content-disposition": "filename*=UTF-8''{file_name}",
                },
                "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
            },
        ]
        + asset_tables_with_metadata
        + [
            {
                # Upload assets into an asset table of an asset table without any metadata
                "column_map": {
                    "MD5": "{md5}",
                    "URL": "{URI}",
                    "Length": "{file_size}",
                    "Filename": "{file_name}",
                },
                "target_table": [model.domain_schema, "{asset_table}"],
                "file_pattern": asset_path_regex
                + "/"
                + asset_file_regex,  # Sets schema, asset_table, file_name, file_ext
                "checksum_types": ["sha256", "md5"],
                "hatrac_options": {"versioned_urls": True},
                "hatrac_templates": {
                    "hatrac_uri": "/hatrac/{asset_table}/{md5}.{file_name}",
                    "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}",
                },
                "record_query_template": "/entity/{target_table}/MD5={{md5}}&Filename={{file_name}}",
            },
            # {
            #  Upload the records into a  table
            #   "asset_type": "skip",
            ##   "default_columns": ["RID", "RCB", "RMB", "RCT", "RMT"],
            #  "file_pattern": feature_value_regex,  # Sets schema, table,
            #  "ext_pattern": "^.*[.](?P<file_ext>json|csv)$",
            #  "target_table": ["{schema}", "{table}"],
            # },
            {
                #  Upload the records into a  table
                "asset_type": "table",
                "default_columns": ["RID", "RCB", "RMB", "RCT", "RMT"],
                "file_pattern": table_regex,  # Sets schema, table,
                "ext_pattern": "^.*[.](?P<file_ext>json|csv)$",
                "target_table": ["{schema}", "{table}"],
            },
        ],
        "version_update_url": "https://github.com/informatics-isi-edu/deriva-client",
        "version_compatibility": [[">=1.4.0", "<2.0.0"]],
    }


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def upload_directory(
    model: DerivaModel, directory: Path | str
) -> dict[Any, FileUploadState] | None:
    """Upload assets from a directory. This routine assumes that the current upload specification includes a
    configuration for the specified directory.  Every asset in the specified directory is uploaded

    Args:
        model: Model to upload assets to.
        directory: Directory containing the assets and tables to upload.

    Returns:
        Results of the upload operation.

    Raises:
        DerivaMLException: If there is an issue uploading the assets.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DerivaMLException("Directory does not exist")

    # Now upload the files by creating an upload spec and then calling the uploader.
    with TemporaryDirectory() as temp_dir:
        spec_file = f"{temp_dir}/config.json"

        with open(spec_file, "w+") as cfile:
            json.dump(bulk_upload_configuration(model), cfile)
        uploader = GenericUploader(
            server={
                "host": model.hostname,
                "protocol": "https",
                "catalog_id": model.catalog.catalog_id,
            },
            config_file=spec_file,
        )
        try:
            uploader.getUpdatedConfig()
            uploader.scanDirectory(directory)
            results = {
                path: FileUploadState(
                    state=UploadState(result["State"]),
                    status=result["Status"],
                    result=result["Result"],
                )
                for path, result in uploader.uploadFiles().items()
            }
        finally:
            uploader.cleanup()
        return results


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def upload_asset(
    model: DerivaModel, file: Path | str, table: Table | str, **kwargs: Any
) -> dict:
    """Upload the specified file into Hatrac and update the associated asset table.

    Args:
        file: path to the file to upload.
        table: Name of the asset table
        model: Model to upload assets to.
        kwargs: Keyword arguments for values of additional columns to be added to the asset table.

    Returns:

    """
    if not model.is_asset(table):
        raise DerivaMLException(f"Table {table} is not an asset table.")

    file_path = Path(file)
    file_name = file_path.name
    file_size = file_path.stat().st_size

    hatrac_path = f"/hatrac/{table.name}/"
    hs = HatracStore(
        "https",
        server=model.catalog.deriva_server.server,
        credentials=model.catalog.deriva_server.credentials,
    )
    md5_hashes = hash_utils.compute_file_hashes(file, ["md5"])["md5"]
    sanitized_filename = urlquote(
        re.sub("[^a-zA-Z0-9_.-]", "_", md5_hashes[0] + "." + file_name)
    )
    hatrac_path = f"{hatrac_path}{sanitized_filename}"

    try:
        # Upload the file to hatrac.
        hatrac_uri = hs.put_obj(
            hatrac_path,
            file,
            md5=md5_hashes[1],
            content_type=mime_utils.guess_content_type(file),
            content_disposition="filename*=UTF-8''" + file_name,
        )
    except Exception as e:
        raise e
    try:
        # Now update the asset table.
        ipath = (
            model.catalog.getPathBuilder().schemas[table.schema.name].tables[table.name]
        )
        return list(
            ipath.insert(
                [
                    {
                        "URL": hatrac_uri,
                        "Filename": file_name,
                        "Length": file_size,
                        "MD5": md5_hashes[0],
                    }
                    | kwargs
                ]
            )
        )[0]
    except Exception as e:
        raise e


class UploadAssetDirectory:
    def __init__(self, model: DerivaModel, schema: str, table: str, prefix: Path):
        self.prefix = prefix
        self.path = asset_table_dir(
            prefix=prefix, asset_schema=schema, asset_table=table
        )
        self.table = table
        self.schema = schema
        self.model = model

    def create_file(self, file_name: str, metadata: dict[str, Any]) -> Path:
        """Return the file in which to place  assets of a specified type are to be uploaded.

        Args:
            file_name: Name of file to which the contents of the asset will be placed
            metadata: Any additional metadata to add to the asset
        Returns:
            Path to directory in which to place assets of type asset_type.
        """
        metadata = metadata or {}
        asset_metadata = self.model.asset_metadata(self.table)
        if not (asset_metadata >= set(metadata.keys())):
            raise DerivaMLException(
                f"Metadata {metadata} does not match asset metadata {asset_metadata}"
            )

        path = self.path
        for m in self.model.asset_metadata(self.table):
            path = path / metadata.get(m, "None")
        path.mkdir(parents=True, exist_ok=True)
        return path / file_name


def test_upload():
    """ """
    ead = execution_asset_dir("foo", "my-rid", "my-asset")
    emd = execution_metadata_dir("foo", "my-rid", "my-metadata")
    _fp = feature_value_path("foo", "my-rid", "my-schema", "my-target", "my-feature")
    fa = feature_asset_dir(
        "foo", "my-rid", "my-schema", "my-target", "my-feature", "my-asset"
    )
    _tp = table_path("foo", "my-schema", "my-table")
    #   _ad = create_asset_dir("foo", "my-schema", "my-asset")
    _is_md = is_execution_metadata_dir(emd)
    _is_ea = is_execution_asset_dir(ead)
    _is_fa = is_feature_asset_dir(fa)

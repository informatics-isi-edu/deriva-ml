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
                           <feature_name>.jsonl    <- needs to have asset_name column remapped before uploading
                table
                   <schema>
                       <record_table>
                          record_table.csv
                asset
                    <schema>
                        <asset_table>
                            <metadata1>
                                <metadata2>
                                    file1, file2, ....
                asset-type
                    <schema>
                        file1.jsonl, file2.jsonl
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional

import regex as re
from deriva.core import urlquote
from deriva.core.ermrest_model import Table
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.transfer.upload.deriva_upload import GenericUploader
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import (
    RID,
    DerivaSystemColumns,
    FileUploadState,
    UploadProgress,
    UploadState,
)
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.model.catalog import DerivaModel

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

# Use os.path.sep for OS-agnostic paths in regex patterns
SEP = re.escape(os.path.sep)
upload_root_regex = f"(?i)^.*{SEP}deriva-ml"

exec_dir_regex = upload_root_regex + f"{SEP}execution{SEP}(?P<execution_rid>[-\\w]+)"

feature_dir_regex = exec_dir_regex + f"{SEP}feature"
feature_table_dir_regex = (
    feature_dir_regex + f"{SEP}(?P<schema>[-\\w]+){SEP}(?P<target_table>[-\\w]+){SEP}(?P<feature_name>[-\\w]+)"
)
feature_value_regex = feature_table_dir_regex + f"{SEP}(?P=feature_name)[.](?P<ext>[(csv|json)]*)$"
feature_asset_dir_regex = feature_table_dir_regex + f"{SEP}asset{SEP}(?P<asset_table>[-\\w]+)"
feature_asset_regex = feature_asset_dir_regex + f"{SEP}(?P<file>[A-Za-z0-9_-]+)[.](?P<ext>[a-z0-9]*)$"

asset_path_regex = exec_dir_regex + rf"{SEP}asset{SEP}(?P<schema>[-\w]+){SEP}(?P<asset_table>[-\w]*)"

asset_file_regex = r"(?P<file>[-\w]+)[.](?P<ext>[a-z0-9]*)$"

table_regex = exec_dir_regex + rf"{SEP}table{SEP}(?P<schema>[-\w]+){SEP}(?P<table>[-\w]+){SEP}(?P=table)[.](csv|json)$"


def is_feature_dir(path: Path) -> Optional[re.Match]:
    """Path matches the pattern for where the table for a feature would go."""
    return re.match(feature_table_dir_regex + "$", path.as_posix())


def normalize_asset_dir(path: str | Path) -> Optional[tuple[str, str]]:
    """Parse a path to an asset file and return the asset table name and file name.

    Args:
        path: Path to the asset file

    Returns:
        Tuple of (schema/table, filename) or None if path doesn't match pattern
    """
    path = Path(path)
    if not (m := re.match(asset_path_regex, str(path))):
        return None
    return f"{m['schema']}/{m['asset_table']}", path.name


def upload_root(prefix: Path | str) -> Path:
    """Return the top level directory of where to put files to be uploaded."""
    path = Path(prefix) / "deriva-ml"
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


def feature_root(prefix: Path | str, exec_rid: str) -> Path:
    """Return the path to the directory in which features for the specified execution should be placed."""
    path = execution_root(prefix, exec_rid) / "feature"
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_root(prefix: Path | str, exec_rid: str) -> Path:
    """Return the path to the directory in which features for the specified execution should be placed."""
    path = execution_root(prefix, exec_rid) / "asset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_dir(prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str) -> Path:
    """Return the path to eht directory in which a named feature for an execution should be placed."""
    path = feature_root(prefix, exec_rid) / schema / target_table / feature_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_value_path(prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str) -> Path:
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
    return feature_dir(prefix, exec_rid, schema, target_table, feature_name) / f"{feature_name}.jsonl"


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
    """Generate upload specification for an asset table.

    Args:
        model: The DerivaModel instance.
        asset_table: The asset table name or Table object.

    Returns:
        A dictionary containing the upload specification for the asset table.
    """
    metadata_columns = model.asset_metadata(asset_table)
    asset_table = model.name_to_table(asset_table)
    schema = model.name_to_table(asset_table).schema.name

    # Be careful here as a metadata value might be a string with can contain special characters.
    metadata_path = "/".join([rf"(?P<{c}>[-:._ \w]+)" for c in metadata_columns])
    asset_path = f"{exec_dir_regex}/asset/{schema}/{asset_table.name}/{metadata_path}/{asset_file_regex}"
    asset_table = model.name_to_table(asset_table)
    schema = model.name_to_table(asset_table).schema.name

    # Create upload specification
    spec = {
        # Upload assets into an asset table of an asset table.
        "column_map": {
            "MD5": "{md5}",
            "URL": "{URI}",
            "Length": "{file_size}",
            "Filename": "{file_name}",
        }
        | {c: f"{{{c}}}" for c in metadata_columns},
        "file_pattern": asset_path,  # Sets schema, asset_table, file
        "asset_type": "file",
        "target_table": [schema, asset_table.name],
        "checksum_types": ["sha256", "md5"],
        "hatrac_options": {"versioned_urls": True},
        "hatrac_templates": {
            "hatrac_uri": f"/hatrac/{asset_table.name}/{{md5}}.{{file_name}}",
            "content-disposition": "filename*=UTF-8''{file_name}",
        },
        "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
    }
    return spec


def bulk_upload_configuration(model: DerivaModel) -> dict[str, Any]:
    """Return an upload specification for deriva-ml
    Arguments:
        model: Model from which to generate the upload configuration
    """
    asset_tables_with_metadata = [
        asset_table_upload_spec(model=model, asset_table=t) for t in model.find_assets() if model.asset_metadata(t)
    ]
    return {
        "asset_mappings": asset_tables_with_metadata
        + [
            {
                # Upload assets into an asset table of an asset table without any metadata
                "column_map": {
                    "MD5": "{md5}",
                    "URL": "{URI}",
                    "Length": "{file_size}",
                    "Filename": "{file_name}",
                },
                "asset_type": "file",
                "target_table": ["{schema}", "{asset_table}"],
                "file_pattern": asset_path_regex + "/" + asset_file_regex,  # Sets schema, asset_table, name, ext
                "checksum_types": ["sha256", "md5"],
                "hatrac_options": {"versioned_urls": True},
                "hatrac_templates": {
                    "hatrac_uri": "/hatrac/{asset_table}/{md5}.{file_name}",
                    "content-disposition": "filename*=UTF-8''{file_name}",
                },
                "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
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
    model: DerivaModel,
    directory: Path | str,
    progress_callback: Callable[[UploadProgress], None] | None = None,
) -> dict[Any, FileUploadState] | None:
    """Upload assets from a directory. This routine assumes that the current upload specification includes a
    configuration for the specified directory.  Every asset in the specified directory is uploaded

    Args:
        model: Model to upload assets to.
        directory: Directory containing the assets and tables to upload.
        progress_callback: Optional callback function to receive upload progress updates.
            Called with UploadProgress objects containing file information and progress.

    Returns:
        Results of the upload operation.

    Raises:
        DerivaMLException: If there is an issue with uploading the assets.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DerivaMLException("Directory does not exist")

    # Track upload progress across files
    # status_callback is called twice per file: once before upload starts, once after it completes
    upload_state = {"completed_files": 0, "total_files": 0, "status_calls": 0}

    # Count total files to upload
    for root, dirs, files in os.walk(directory):
        upload_state["total_files"] += len(files)

    # Create wrapper callbacks for GenericUploader if a progress callback was provided
    def file_callback(**kwargs) -> bool:
        """Callback for per-chunk progress updates from GenericUploader.

        The deriva GenericUploader passes kwargs with: completed, total, file_path, host, job_info.
        Note: This callback is only invoked for large files (> 25MB) that use chunked uploads.
        Small files are uploaded in a single request and this callback won't be called.
        """
        if progress_callback is not None:
            file_path = kwargs.get("file_path", "")
            completed_chunks = kwargs.get("completed", 0)
            total_chunks = kwargs.get("total", 0)

            progress = UploadProgress(
                file_path=file_path,
                file_name=Path(file_path).name if file_path else "",
                bytes_completed=completed_chunks,
                bytes_total=total_chunks,
                percent_complete=(completed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
                phase="uploading_chunks",
                message=f"Uploading large file: chunk {completed_chunks} of {total_chunks}",
            )
            progress_callback(progress)
        return True  # Continue upload

    def status_callback() -> None:
        """Callback for per-file status updates from GenericUploader.

        GenericUploader calls this twice per file: once before upload starts (odd calls)
        and once after upload completes (even calls). We use even calls to track completed files.
        """
        if progress_callback is not None:
            upload_state["status_calls"] += 1

            # Even calls indicate file completion (after upload)
            if upload_state["status_calls"] % 2 == 0:
                upload_state["completed_files"] += 1

            # Report progress with current file count
            current_file = (upload_state["status_calls"] + 1) // 2  # 1-indexed current file
            progress = UploadProgress(
                phase="uploading",
                message=f"Uploading file {current_file} of {upload_state['total_files']}",
                percent_complete=(upload_state["completed_files"] / upload_state["total_files"] * 100)
                if upload_state["total_files"] > 0
                else 0,
            )
            progress_callback(progress)

    # Now upload the files by creating an upload spec and then calling the uploader.
    with TemporaryDirectory() as temp_dir:
        spec_file = Path(temp_dir) / "config.json"
        with spec_file.open("w+") as cfile:
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
            uploader.scanDirectory(directory, purge_state=True)
            results = {
                path: FileUploadState(
                    state=UploadState(result["State"]),
                    status=result["Status"],
                    result=result["Result"],
                )
                for path, result in uploader.uploadFiles(
                    file_callback=file_callback if progress_callback else None,
                    status_callback=status_callback if progress_callback else None,
                ).items()
            }
        finally:
            uploader.cleanup()
        return results


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def upload_asset(model: DerivaModel, file: Path | str, table: Table, **kwargs: Any) -> dict:
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
    md5_hashes = hash_utils.compute_file_hashes(file, frozenset(["md5"]))["md5"]
    sanitized_filename = urlquote(re.sub("[^a-zA-Z0-9_.-]", "_", md5_hashes[0] + "." + file_name))
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
        ipath = model.catalog.getPathBuilder().schemas[table.schema.name].tables[table.name]
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


def asset_file_path(
    prefix: Path | str,
    exec_rid: RID,
    asset_table: Table,
    file_name: str,
    metadata: dict[str, Any],
) -> Path:
    """Return the file in which to place  assets of a specified type are to be uploaded.

    Args:
        prefix: Path prefix to use.
        exec_rid: RID to use.
        asset_table: Table in which to place assets.
        file_name: File name to use.
        metadata: Any additional metadata to add to the asset
    Returns:
        Path to directory in which to place assets of type asset_type.
    """
    schema = asset_table.schema.name
    asset_name = asset_table.name

    path = execution_root(prefix, exec_rid) / "asset" / schema / asset_name
    metadata = metadata or {}
    asset_columns = {
        "Filename",
        "URL",
        "Length",
        "MD5",
        "Description",
    }.union(set(DerivaSystemColumns))
    asset_metadata = {c.name for c in asset_table.columns} - asset_columns

    if not (asset_metadata >= set(metadata.keys())):
        raise DerivaMLException(f"Metadata {metadata} does not match asset metadata {asset_metadata}")

    for m in asset_metadata:
        path = path / str(metadata.get(m, "None"))
    path.mkdir(parents=True, exist_ok=True)
    return path / file_name


def asset_type_path(prefix: Path | str, exec_rid: RID, asset_table: Table) -> Path:
    """Return the path to a JSON line file in which to place asset_type information.

    Args:
        prefix: Location of upload root directory
        exec_rid: Execution RID
        asset_table: Table in which to place assets.

    Returns:
        Path to the file in which to place asset_type values for the named asset.
    """
    path = execution_root(prefix, exec_rid=exec_rid) / "asset-type" / asset_table.schema.name
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{asset_table.name}.jsonl"

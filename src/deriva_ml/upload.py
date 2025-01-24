"""
This module provides functions that help stucture local directories for uploading to a DerivaML catalog, and
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

"""

from pathlib import Path
from typing import Optional

import regex as re

upload_root_regex = r"(?i)^.*/deriva-ml"
exec_dir_regex = upload_root_regex + r"/execution/(?P<execution_rid>[-\w]+)"
exec_asset_dir_regex = exec_dir_regex + r"/execution-asset/(?P<execution_asset_type>[-\w]+)"
exec_asset_regex = exec_asset_dir_regex + r"/(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]+)$"
exec_metadata_dir_regex = exec_dir_regex + r'/execution-metadata/(?P<execution_metadata_type>[-\w]+)'
exec_metadata_regex = exec_metadata_dir_regex + r'/(?P<filename>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$'
feature_dir_regex = exec_dir_regex + r'/feature'
feature_table_dir_regex = feature_dir_regex + r"/(?P<schema>[-\w]+)/(?P<target_table>[-\w]+)/(?P<feature_name>[-\w]+)"
feature_value_regex = feature_table_dir_regex + r"/(?P=feature_name)[.](?P<file_ext>[(csv|json)]*)$"
feature_asset_dir_regex =  feature_table_dir_regex + r"/asset/(?P<asset_table>[-\w]+)"
feature_asset_regex = feature_asset_dir_regex + r'/(?P<file_name>[A-Za-z0-9_-]+)[.](?P<file_ext>[a-z0-9]*)$'
asset_path_regex = (
    upload_root_regex
    + r"/asset/(?P<schema>[-\w]+)/(?P<asset_table>[-\w]*)/(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$"
)
table_regex = exec_dir_regex + r"/table/(?P<schema>[-\w]+)/(?P<table>[-\w]+)/(?P=table)[.](csv|json)$"


def is_execution_metadata_dir(path: Path) -> Optional[re.Match]:
    """

    Args:
      path: Path: 

    Returns:
        Regex match object from metadata_dir_regex.
    """
    return re.match(exec_metadata_dir_regex + '$', path.as_posix())


def is_execution_asset_dir(path: Path) -> Optional[re.Match]:
    """
    Args:
      path: Path: 

    Returns:

    """
    return re.match(exec_asset_dir_regex + '$', path.as_posix())


def is_feature_dir(path: Path) -> Optional[re.Match]:
    """

    Args:
      path: Path: 

    Returns:

    """
    return re.match(feature_table_dir_regex + '$', path.as_posix())


def is_feature_asset_dir(path: Path) -> Optional[re.Match]:
    """

    Args:
      path: Path: 

    Returns:

    """
    return re.match(feature_asset_dir_regex + '$', path.as_posix())


def upload_root(prefix: Path | str) -> Path:
    """

    Args:
      prefix: Path | str: 

    Returns:

    """
    path = Path(prefix) / "deriva-ml"
    path.mkdir(exist_ok=True, parents=True)
    return path


def execution_root(prefix: Path | str, exec_rid) -> Path:
    """

    Args:
      prefix: Path | str: 
      exec_rid: 

    Returns:

    """
    path = upload_root(prefix) / "execution" / exec_rid
    path.mkdir(exist_ok=True, parents=True)
    return path


def execution_asset_root(prefix: Path | str, exec_rid: str) -> Path:
    """

    Args:
      prefix: Path | str: 
      exec_rid: str: 

    Returns:

    """
    path = execution_root(prefix, exec_rid) / "execution-asset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_metadata_root(prefix: Path | str, exec_rid: str) -> Path:
    """

    Args:
      prefix: Path | str: 
      exec_rid: str: 

    Returns:

    """
    path = execution_root(prefix, exec_rid) / "execution-metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_asset_dir(prefix: Path | str, exec_rid: str, asset_type: str) -> Path:
    """Return the path to a directory in which to place execution assets of a specified type are to be uploaded.

    Args:
      prefix: Location of upload root directory
      asset_type: Type of execution asset
      exec_rid: RID of the execution asset
      prefix: Path | str: 
      exec_rid: str: 
      asset_type: str: 

    Returns:

    """
    path = execution_asset_root(prefix, exec_rid) / asset_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_metadata_dir(prefix: Path | str, exec_rid: str, metadata_type: str) -> Path:
    """Return the path to a directory in which to place execution metadata of a specified type are to be uploaded.

    Args:
        prefix: Location in which to locate this directory
        exec_rid: Execution rid to be associated with this metadata
        metadata_type: Controlled vocabulary term from vocabulary Metadata_Type

    Returns:
        Path to the metadata directory
    """
    path = execution_metadata_root(prefix, exec_rid) / metadata_type
    path.mkdir(parents=True, exist_ok=True)
    return path

def feature_root(prefix: Path | str, exec_rid: str) -> Path:
    """

    Args:
      prefix: Path | str: 
      exec_rid: str: 

    Returns:

    """
    path = execution_root(prefix, exec_rid) / 'feature'
    path.mkdir(parents=True, exist_ok=True)
    return path

def feature_dir(
    prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str
) -> Path:
    """

    Args:
        prefix: Path | str:
        exec_rid: str:
        schema: str:
        target_table: str:
        feature_name: str:

    Returns:

    """
    path = feature_root(prefix, exec_rid) / schema / target_table / feature_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_value_path(
    prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str
) -> Path:
    """Return the path to a CSV file in which to place feature values that are to be uploaded.  Values will either be
    scalar, references to controlled vocabulary (Terms) or references to assets.

    Args:
        prefix: Location of upload root directory
        exec_rid: RID of the execution to be associated with this feature.
        schema: Domain schema name
        target_table: Target table name for the feature.
        feature_name: Name of the feature.

    Returns:
        Path to CSV file in which to place feature values
    """
    return feature_dir(prefix, exec_rid, schema, target_table, feature_name) / f"{feature_name}.csv"


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
    path = feature_dir(prefix, exec_rid, schema, target_table, feature_name) / 'asset' / asset_table

    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_dir(prefix: Path | str, schema: str, asset_table: str) -> Path:
    """Return the path to a directory in which to place assets that are to be uploaded.

    Args:
        prefix: Location of upload root directory
        schema: Domain schema
        asset_table: Name of the asset table

    Returns:
        Path to the directory in which to place assets
    """
    path = upload_root(prefix) / 'asset' / schema / asset_table
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
    path = upload_root(prefix) / 'table' / schema / table
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{table}.csv"


bulk_upload_configuration = {
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
        {
            # Upload assets into an asset table of an asset table.
            "column_map": {
                "MD5": "{md5}",
                "URL": "{URI}",
                "Length": "{file_size}",
                "Filename": "{file_name}",
            },
            "file_pattern": asset_path_regex,  # Sets schema, asset_table, file_name, file_ext
            "checksum_types": ["sha256", "md5"],
            "hatrac_options": {"versioned_urls": True},
            "hatrac_templates": {
                "hatrac_uri": "/hatrac/{asset_table}/{md5}.{file_name}",
                "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}",
            },
            "target_table": ["{schema}", "{asset_table}"],
            "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
        },
        #{
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

def test_upload():
    """ """
    ead = execution_asset_dir('foo', 'my-rid', 'my-asset')
    emd = execution_metadata_dir('foo', 'my-rid', 'my-metadata')
    _fp = feature_value_path('foo', 'my-rid', 'my-schema', 'my-target', 'my-feature')
    fa = feature_asset_dir('foo', 'my-rid', 'my-schema', 'my-target', 'my-feature', 'my-asset')
    _tp = table_path('foo', 'my-schema', 'my-table')
    _ad = asset_dir('foo', 'my-schema', 'my-asset')
    _is_md = is_execution_metadata_dir(emd)
    _is_ea = is_execution_asset_dir(ead)
    _is_fa = is_feature_asset_dir(fa)

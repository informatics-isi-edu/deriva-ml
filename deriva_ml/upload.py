from pathlib import Path
from typing import Optional
import regex as re

# Here is the directory layout we support:
#
#  deriva-ml/
#       execution
#           <execution_rid>
#               execution-assets
#                   <asset_type>
#                       file1, file2, ....   <- Need to update execution_asset association table.
#               execution-metadata
#                   <metadata_type>
#               feature
#                   <schema>
#                       <target_table>
#                            <feature_name>
#                                   assets
#                                       <asset_table>
#                                           file1, file2, ...
#                           <feature_name>.csv    <- needs to have asset_name column remapped before uploading
#            table
#               <schema>
#                   <record_table>
#                      record_table.csv
#            asset
#               <schema>
#                   <asset_table>
#                     file1, file2, ....
#

upload_root_regex = r"(?i)^.*/deriva-ml"
exec_dir_regex = upload_root_regex + r"/execution/(?P<execution_rid>[-\w]+)"
exec_asset_dir_regex = exec_dir_regex + r"/execution-assets/(?P<execution_asset_type>[-\w]+)"
exec_asset_regex = exec_asset_dir_regex + r"/(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]+)$"
exec_metadata_dir_regex = exec_dir_regex + r'/execution-metadata/(?P<execution_metadata_type>[-\w]+)'
exec_metadata_regex = exec_metadata_dir_regex + r'/(?P<filename>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$'
feature_dir_regex = exec_dir_regex + r'/feature'
feature_table_dir_regex = feature_dir_regex + r"/(?P<schema>[-\w]+)/(?P<target_table>[-\w]+)/(?P<feature_name>[-\w]+)"
feature_value_regex = feature_table_dir_regex + r"/(?P=feature_name)[.](?P<file_ext>[(csv|json)]*)$"
feature_asset_dir_regex =  feature_table_dir_regex + r"/assets/(?P<asset_table>[-\w]+)"
feature_asset_regex = feature_asset_dir_regex + r'/(?P<file_name>[A-Za-z0-9_-]+)[.](?P<file_ext>[a-z0-9]*)$'
asset_path_regex = (
    upload_root_regex
    + r"/assets/(?P<schema>[-\w]+)/(?P<asset_table>[-\w]*)/(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$"
)
table_regex = exec_dir_regex + r"/table/(?P<schema>[-\w]+)/(?P<table>[-\w]+)/(?P=table)[.](csv|json)$"


def is_execution_metadata_dir(path: Path) -> Optional[re.Match]:
    return re.match(exec_metadata_dir_regex + '$', path.as_posix())


def is_execution_asset_dir(path: Path) -> Optional[re.Match]:
    return re.match(exec_asset_dir_regex + '$', path.as_posix())


def is_feature_dir(path: Path) -> Optional[re.Match]:
    return re.match(feature_table_dir_regex + '$', path.as_posix())


def is_feature_asset_dir(path: Path) -> Optional[re.Match]:
    return re.match(feature_asset_dir_regex + '$', path.as_posix())


def upload_root(prefix: Path | str) -> Path:
    path = Path(prefix) / "deriva-ml"
    path.mkdir(exist_ok=True, parents=True)
    return path


def execution_root(prefix: Path | str, exec_rid) -> Path:
    path = upload_root(prefix) / "execution" / exec_rid
    path.mkdir(exist_ok=True, parents=True)
    return path


def execution_assets_root(prefix: Path | str, exec_rid: str) -> Path:
    path = execution_root(prefix, exec_rid) / "execution-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_metadata_root(prefix: Path | str, exec_rid: str) -> Path:
    path = execution_root(prefix, exec_rid) / "execution-metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_assets_dir(prefix: Path | str, exec_rid: str, asset_type: str) -> Path:
    """
    Return the path to a directory in which to place execution assets that are to be uploaded.
    :param prefix: Location of upload root directory
    :param asset_type: Type of execution asset
    :param exec_rid: RID of the execution asset
    :return:
    """
    path = execution_assets_root(prefix, exec_rid) / asset_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_metadata_dir(prefix: Path | str, exec_rid: str, metadata_type: str) -> Path:
    """
    Return the path to a directory in which to place execution metadata that are to be uploaded.
    :param prefix:  Location in which to locate this directory
    :param exec_rid: Execution rid to be associated with this metadata
    :param metadata_type: Controlled vocabulary term from vocabulary Metadata_Type
    :return:
    """
    path = execution_metadata_root(prefix, exec_rid) / metadata_type
    path.mkdir(parents=True, exist_ok=True)
    return path

def feature_root(prefix: Path | str, exec_rid: str) -> Path:
    path = execution_root(prefix, exec_rid) / 'feature'
    path.mkdir(parents=True, exist_ok=True)
    return path

def feature_dir(
    prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str
) -> Path:
    path = feature_root(prefix, exec_rid) / schema / target_table / feature_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_value_path(
    prefix: Path | str, exec_rid: str, schema: str, target_table: str, feature_name: str
) -> Path:
    return feature_dir(prefix, exec_rid, schema, target_table, feature_name) / f"{feature_name}.csv"


def feature_asset_dir(
    prefix: Path | str,
    exec_rid: str,
    schema: str,
    target_table: str,
    feature_name: str,
    asset_table: str,
) -> Path:
    path = feature_dir(prefix, exec_rid, schema, target_table, feature_name) / 'assets' / asset_table

    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_dir(prefix: Path | str, schema: str, asset_table: str) -> Path:
    path = upload_root(prefix) / 'assets' / schema / asset_table
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_path(prefix: Path | str, schema: str, table: str, exec_rid: str = "") -> Path:
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
            # Upload the contents of the Execution_Assets directory.
            "column_map": {
                "MD5": "{md5}",
                "URL": "{URI}",
                "Length": "{file_size}",
                "Filename": "{file_name}",
                "Execution_Asset_Type": "{execution_asset_type_name}",
            },
            "file_pattern": exec_asset_regex,
            "target_table": ["deriva-ml", "Execution_Assets"],
            "checksum_types": ["sha256", "md5"],
            "hatrac_options": {"versioned_urls": True},
            "hatrac_templates": {
                "hatrac_uri": "/hatrac/execution_assets/{md5}.{file_name}",
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
    ead = execution_assets_dir('foo', 'my-rid', 'my-asset')
    emd = execution_metadata_dir('foo', 'my-rid', 'my-metadata')
    fp = feature_value_path('foo', 'my-rid', 'my-schema', 'my-target', 'my-feature')
    fa = feature_asset_dir('foo', 'my-rid', 'my-schema', 'my-target', 'my-feature', 'my-asset')
    tp = table_path('foo', 'my-schema', 'my-table')
    ad = asset_dir('foo', 'my-schema', 'my-asset')
    is_md = is_execution_metadata_dir(emd)
    is_ea = is_execution_asset_dir(ead)
    is_fa = is_feature_asset_dir(fa)
    is_ad = is_asset_dir(ad)
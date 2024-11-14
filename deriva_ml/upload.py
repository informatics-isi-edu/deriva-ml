from pathlib import Path
import regex as re

# Here is the directory layout we support:
#
#  execution-assets
#     asset_type
#         file1, file2, ....   <- Need to update execution_asset association table.
#  execution-matadata
#        xxx
#   deriva-ml
#     <schema>
#         table
#            <record_table>
#                 record_table.csv
#         asset
#            <asset_table>
#                 file1, file2, ....
#         feature
#            <target_table>
#                <feature_name>
#                       <asset_table>
#                         file1, file2, ...
#                     <feature_name>.csv    <- needs to have asset_name column remapped before uploading
#

exec_asset_regex = r"(?i)^.*/deriva-ml/execution-asset/(?P<execution_asset_type>[A-Za-z0-9_]*)/(?P<file_name>[A-Za-z0-9_-]*)[.](?P<file_ext>[a-z0-9]*)$"
exec_metadata_regex = "(?i)^.*/execution-metadata/(?P<execution_metadata_type>[A-Za-z0-9_]*)-(?P<filename>[A-Za-z0-9_]*)[.](?P<file_ext>[a-z0-9]*)$",
feature_table = r'(?i)^.*/deriva-ml/(?P<schema>[-\w]+)/feature/(?P<target_table>[-\w]+)/(?P<feature_name>[-\w]+)/'
feature_value_regex = feature_table + r'(?P=feature_name)[.](?P<file_ext>[(csv|json)]*)$'
feature_asset_regex = feature_table + r'(?P<asset_table>[-\w]+)/(?P<file_name>[A-Za-z0-9_-]+)[.](?P<file_ext>[a-z0-9]*)$'
asset_dir_regex = r"(?i)^.*/deriva-ml/(?P<schema>[-\w]+)/asset/(?P<asset_table>[-\w]*)/(?P<file_name>[-\w]+)[.](?P<file_ext>[a-z0-9]*)$"
table_regex = r"(?i)^.*/deriva-ml/(?P<schema>[-\w]+)/table/(?P<table>[-\w]+)/(?P=table)[.](csv|json)$"

def upload_root_path(prefix: Path):
    path = prefix / 'deriva-ml'
    path.mkdir(parents=True, exist_ok=True)
    return path

def execution_asset_dir(prefix: Path, asset_type: str) -> Path:
    """
    Return the path to a directory in which to place execution assets that are to be uploaded.
    :param prefix: Location of upload root directory
    :param asset_type: Type of execution asset
    :return:
    """
    path = prefix / f'deriva-ml/execution-asset/{asset_type}'
    path.mkdir(parents=True, exist_ok=True)
    return path

def execution_metadata_dir(prefix: Path, metadata_type: str) -> Path:
    path = prefix / f'deriva-ml/execution-metadata/{metadata_type}'
    path.mkdir(parents=True, exist_ok=True)
    return path

def is_execution_asset(path: Path) -> re.Match:
    return re.match(exec_asset_regex, path.as_posix())

def is_feature_path(path: Path) -> re.Match:
    return re.match(feature_table, path.as_posix())


def feature_value_path(prefix: Path, schema: str, target_table: str, feature_name: str) -> Path:
    path = prefix / f'deriva-ml/{schema}/feature/{target_table}/{feature_name}'
    path.mkdir(parents=True, exist_ok=True)
    return path / f'{feature_name}.csv'

def feature_asset_dir(prefix: Path, schema: str, target_table: str, feature_name: str, asset_table: str) -> Path:
    path = prefix / f'deriva-ml/{schema}/feature/{target_table}/{feature_name}/{asset_table}'
    path.mkdir(parents=True, exist_ok=True)
    return path

def asset_dir(prefix: Path, schema: str, asset_table: str) -> Path:
    path = prefix / f'deriva-ml/{schema}/asset/{asset_table}'
    path.mkdir(parents=True, exist_ok=True)
    return path

def table_path(prefix: Path, schema: str, table: str) -> Path:
    path =  prefix / f'deriva-ml/{schema}/table/{table}'
    path.mkdir(parents=True, exist_ok=True)
    return path / f'{table}.csv'


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
                "Execution_Metadata_Type": "{execution_metadata_type_name}"
            },
            "file_pattern": exec_metadata_regex,
            "target_table": ["deriva-ml", "Execution_Metadata"],
            "checksum_types": ["sha256", "md5"],
            "hatrac_options": {"versioned_urls": True},
            "hatrac_templates": {
                "hatrac_uri": "/hatrac/execution_metadata/{md5}.{file_name}",
                "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}"
            },
            "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
            "metadata_query_templates": [
                "/entity/deriva-ml:Execution_Metadata_Type/Name={execution_metadata_type}/execution_metadata_type_name:=Name"
            ],
        },
        {
            # Upload the contents of the Execution_Assets directory.
            "column_map": {
                "MD5": "{md5}",
                "URL": "{URI}",
                "Length": "{file_size}",
                "Filename": "{file_name}",
                "Execution_Asset_Type": "{execution_asset_type_name}"
            },
            "file_pattern": exec_asset_regex,
            "target_table": ["deriva-ml", "Execution_Assets"],
            "checksum_types": ["sha256", "md5"],
            "hatrac_options": {"versioned_urls": True},
            "hatrac_templates": {
                "hatrac_uri": "/hatrac/execution_assets/{md5}.{file_name}",
                "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}"
            },
            "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
            "metadata_query_templates": [
                "/attribute/deriva-ml:Execution_Asset_Type/Name={execution_asset_type}/execution_asset_type_name:=Name"
            ],
        },
        {
            # Upload the assets for a feature table.
            "column_map": {"MD5": "{md5}", "URL": "{URI}", "Length": "{file_size}", "Filename": "{file_name}"},
            "file_pattern": feature_asset_regex, # Sets target_table, feature_name, asset_table
            "target_table": ["{schema}", "{asset_table}"],
            "checksum_types": ["sha256", "md5"],
            "hatrac_options": {"versioned_urls": True},
            "hatrac_templates": {
                "hatrac_uri": "/hatrac/{asset_table}/{md5}.{file_name}",
                "content-disposition": "filename*=UTF-8''{file_name}"
            },
            "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
        },
        {
            # Upload assets into an asset table of an asset table.
            "column_map": {"MD5": "{md5}", "URL": "{URI}", "Length": "{file_size}", "Filename": "{file_name}"},
            "file_pattern": asset_dir_regex,  # Sets schema, asset_table, file_name, file_ext
            "checksum_types": ["sha256", "md5"],
            "hatrac_options": {"versioned_urls": True},
            "hatrac_templates": {
                "hatrac_uri": "/hatrac/{asset_table}/{md5}.{file_name}",
                "content-disposition": "filename*=UTF-8''{file_name}.{file_ext}"
            },
            "target_table": ["{schema}", "{asset_table}"],
            "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
        },
        {
            #  Upload the records into a feature table
            "asset_type": "table",
            "default_columns": ["RID", "RCB", "RMB", "RCT", "RMT"],
            "file_pattern": feature_value_regex,  # Sets schema, target_table, feature_name
            "ext_pattern": "^.*[.](?P<file_ext>json|csv)$",
            "target_table": ["{schema}", "Execution_{target_table}_{feature_name}"]
        },
        {
            #  Upload the records into a feature table
            "asset_type": "table",
            "default_columns": ["RID", "RCB", "RMB", "RCT", "RMT"],
            "file_pattern": table_regex,  # Sets schema, table,
            "ext_pattern": "^.*[.](?P<file_ext>json|csv)$",
            "target_table": ["{schema}", "{table}"]
        },
    ],
                "version_update_url": "https://github.com/informatics-isi-edu/deriva-client",
                "version_compatibility": [[">=1.4.0", "<2.0.0"]]
}

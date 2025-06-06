"""
Enumeration classes used throughout the DerivaML package.
"""

from enum import Enum
from deriva.core.ermrest_model import builtin_types


class BaseStrEnum(str, Enum):
    """Base class for string enums in Python 3.10"""
    pass


class UploadState(Enum):
    """State of file upload"""
    success = 0
    failed = 1
    pending = 2
    running = 3
    paused = 4
    aborted = 5
    cancelled = 6
    timeout = 7


class Status(BaseStrEnum):
    """Enumeration class defining execution status.

    Attributes:
        running: Execution is currently running.
        pending: Execution is pending.
        completed: Execution has been completed successfully.
        failed: Execution has failed.
    """
    initializing = "Initializing"
    created = "Created"
    pending = "Pending"
    running = "Running"
    aborted = "Aborted"
    completed = "Completed"
    failed = "Failed"


class BuiltinTypes(Enum):
    text = builtin_types.text.typename
    int2 = builtin_types.int2.typename
    jsonb = builtin_types.json.typename
    float8 = builtin_types.float8.typename
    timestamp = builtin_types.timestamp.typename
    int8 = builtin_types.int8.typename
    boolean = builtin_types.boolean.typename
    json = builtin_types.json.typename
    float4 = builtin_types.float4.typename
    int4 = builtin_types.int4.typename
    timestamptz = builtin_types.timestamptz.typename
    date = builtin_types.date.typename
    ermrest_rid = builtin_types.ermrest_rid.typename
    ermrest_rcb = builtin_types.ermrest_rcb.typename
    ermrest_rmb = builtin_types.ermrest_rmb.typename
    ermrest_rct = builtin_types.ermrest_rct.typename
    ermrest_rmt = builtin_types.ermrest_rmt.typename
    markdown = builtin_types.markdown.typename
    longtext = builtin_types.longtext.typename
    ermrest_curie = builtin_types.ermrest_curie.typename
    ermrest_uri = builtin_types.ermrest_uri.typename
    color_rgb_hex = builtin_types.color_rgb_hex.typename
    serial2 = builtin_types.serial2.typename
    serial4 = builtin_types.serial4.typename
    serial8 = builtin_types.serial8.typename


class MLVocab(BaseStrEnum):
    """Names of controlled vocabulary for various types within DerivaML."""
    dataset_type = "Dataset_Type"
    workflow_type = "Workflow_Type"
    file_type = "File_Type"
    asset_type = "Asset_Type"
    asset_role = "Asset_Role"


class MLAsset(BaseStrEnum):
    execution_metadata = "Execution_Metadata"
    execution_asset = "Execution_Asset"


class ExecMetadataType(BaseStrEnum):
    """
    Predefined execution metadata types.
    """
    execution_config = "Execution_Config"
    runtime_env = "Runtime_Env"


class ExecAssetType(BaseStrEnum):
    """
    Predefined execution metadata types.
    """
    input_file = "Input_File"
    output_file = "Output_File"
    notebook_output = "Notebook_Output" 
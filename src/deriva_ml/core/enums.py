"""Enumeration classes for DerivaML.

This module provides enumeration classes used throughout DerivaML for representing states, statuses,
types, and vocabularies. Each enum class represents a specific set of constants used in the system.

Classes:
    BaseStrEnum: Base class for string-based enums.
    UploadState: States for file upload operations.
    Status: Execution status values.
    BuiltinTypes: ERMrest built-in data types.
    MLVocab: Controlled vocabulary types.
    MLAsset: Asset type identifiers.
    ExecMetadataType: Execution metadata type identifiers.
    ExecAssetType: Execution asset type identifiers.
"""

from enum import Enum

from deriva.core.ermrest_model import builtin_types


class BaseStrEnum(str, Enum):
    """Base class for string-based enumerations.

    Extends both str and Enum to create string enums that are both string-like and enumerated.
    This provides type safety while maintaining string compatibility.

    Example:
        >>> class MyEnum(BaseStrEnum):
        ...     VALUE = "value"
        >>> isinstance(MyEnum.VALUE, str)  # True
        >>> isinstance(MyEnum.VALUE, Enum)  # True
    """

    pass


class UploadState(Enum):
    """File upload operation states.

    Represents the various states a file upload operation can be in, from initiation to completion.

    Attributes:
        success (int): Upload completed successfully.
        failed (int): Upload failed.
        pending (int): Upload is queued.
        running (int): Upload is in progress.
        paused (int): Upload is temporarily paused.
        aborted (int): Upload was aborted.
        cancelled (int): Upload was cancelled.
        timeout (int): Upload timed out.
    """

    success = 0
    failed = 1
    pending = 2
    running = 3
    paused = 4
    aborted = 5
    cancelled = 6
    timeout = 7


class Status(BaseStrEnum):
    """Execution status values.

    Represents the various states an execution can be in throughout its lifecycle.

    Attributes:
        initializing (str): Initial setup is in progress.
        created (str): Execution record has been created.
        pending (str): Execution is queued.
        running (str): Execution is in progress.
        aborted (str): Execution was manually stopped.
        completed (str): Execution finished successfully.
        failed (str): Execution encountered an error.
    """

    initializing = "Initializing"
    created = "Created"
    pending = "Pending"
    running = "Running"
    aborted = "Aborted"
    completed = "Completed"
    failed = "Failed"


class BuiltinTypes(Enum):
    """ERMrest built-in data types.

    Maps ERMrest's built-in data types to their type names. These types are used for defining
    column types in tables and for type validation.

    Attributes:
        text (str): Text/string type.
        int2 (str): 16-bit integer.
        jsonb (str): Binary JSON.
        float8 (str): 64-bit float.
        timestamp (str): Timestamp without timezone.
        int8 (str): 64-bit integer.
        boolean (str): Boolean type.
        json (str): JSON type.
        float4 (str): 32-bit float.
        int4 (str): 32-bit integer.
        timestamptz (str): Timestamp with timezone.
        date (str): Date type.
        ermrest_rid (str): Resource identifier.
        ermrest_rcb (str): Record created by.
        ermrest_rmb (str): Record modified by.
        ermrest_rct (str): Record creation time.
        ermrest_rmt (str): Record modification time.
        markdown (str): Markdown text.
        longtext (str): Long text.
        ermrest_curie (str): Compact URI.
        ermrest_uri (str): URI type.
        color_rgb_hex (str): RGB color in hex.
        serial2 (str): 16-bit auto-incrementing.
        serial4 (str): 32-bit auto-incrementing.
        serial8 (str): 64-bit auto-incrementing.
    """

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
    """Controlled vocabulary type identifiers.

    Defines the names of controlled vocabulary tables used in DerivaML for various types
    of entities and attributes.

    Attributes:
        dataset_type (str): Dataset classification vocabulary.
        workflow_type (str): Workflow classification vocabulary.
        asset_type (str): Asset classification vocabulary.
        asset_role (str): Asset role classification vocabulary.
    """

    dataset_type = "Dataset_Type"
    workflow_type = "Workflow_Type"
    asset_type = "Asset_Type"
    asset_role = "Asset_Role"
    feature_name = "Feature_Name"


class MLAsset(BaseStrEnum):
    """Asset type identifiers.

    Defines the types of assets that can be associated with executions.

    Attributes:
        execution_metadata (str): Metadata about an execution.
        execution_asset (str): Asset produced by an execution.
    """

    execution_metadata = "Execution_Metadata"
    execution_asset = "Execution_Asset"


class MLTable(BaseStrEnum):
    dataset = "Dataset"
    workflow = "Workflow"
    file = "File"
    asset = "Asset"
    execution = "Execution"
    dataset_version = "Dataset_Version"
    execution_metadata = "Execution_Metadata"
    execution_asset = "Execution_Asset"


class ExecMetadataType(BaseStrEnum):
    """Execution metadata type identifiers.

    Defines the types of metadata that can be associated with an execution.

    Attributes:
        execution_config (str): Execution configuration data.
        runtime_env (str): Runtime environment information.
    """

    execution_config = "Execution_Config"
    runtime_env = "Runtime_Env"


class ExecAssetType(BaseStrEnum):
    """Execution asset type identifiers.

    Defines the types of assets that can be produced during an execution.

    Attributes:
        input_file (str): Input file used by the execution.
        output_file (str): Output file produced by the execution.
        notebook_output (str): Jupyter notebook output from the execution.
    """

    input_file = "Input_File"
    output_file = "Output_File"
    notebook_output = "Notebook_Output"
    model_file = "Model_File"

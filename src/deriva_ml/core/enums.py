"""Enumeration classes for DerivaML.

This module provides enumeration classes used throughout DerivaML for representing states, statuses,
types, and vocabularies. Each enum class represents a specific set of constants used in the system.

Classes:
    BaseStrEnum: Base class for string-based enums.
    UploadState: States for file upload operations.
    Status: Execution status values.
    BuiltinTypes: ERMRest built-in data types.
    MLVocab: Controlled vocabulary types.
    MLAsset: Asset type identifiers.
    ExecMetadataType: Execution metadata type identifiers.
    ExecAssetType: Execution asset type identifiers.
"""

from enum import Enum

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
builtin_types = _ermrest_model.builtin_types


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
    """ERMRest built-in data types.

    Maps ERMRest's built-in data types to their type names. These types are used for defining
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
    """Controlled vocabulary table identifiers.

    Defines the names of controlled vocabulary tables used in DerivaML. These tables
    store standardized terms with descriptions and synonyms for consistent data
    classification across the catalog.

    Attributes:
        dataset_type (str): Dataset classification vocabulary (e.g., "Training", "Test").
        workflow_type (str): Workflow classification vocabulary (e.g., "Python", "Notebook").
        asset_type (str): Asset/file type classification vocabulary (e.g., "Image", "CSV").
        asset_role (str): Asset role vocabulary for execution relationships (e.g., "Input", "Output").
        feature_name (str): Feature name vocabulary for ML feature definitions.
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
    """Core ML schema table identifiers.

    Defines the names of the core tables in the deriva-ml schema. These tables
    form the backbone of the ML workflow tracking system.

    Attributes:
        dataset (str): Dataset table for versioned data collections.
        workflow (str): Workflow table for computational pipeline definitions.
        file (str): File table for tracking individual files.
        asset (str): Asset table for domain-specific file types.
        execution (str): Execution table for workflow run tracking.
        execution_execution (str): Execution_Execution table for nested executions.
        dataset_version (str): Dataset_Version table for version history.
        execution_metadata (str): Execution_Metadata table for run metadata.
        execution_asset (str): Execution_Asset table for run outputs.
    """

    dataset = "Dataset"
    workflow = "Workflow"
    file = "File"
    asset = "Asset"
    execution = "Execution"
    execution_execution = "Execution_Execution"
    dataset_version = "Dataset_Version"
    execution_metadata = "Execution_Metadata"
    execution_asset = "Execution_Asset"


class ExecMetadataType(BaseStrEnum):
    """Execution metadata type identifiers.

    Defines the types of metadata that can be associated with an execution.

    Attributes:
        execution_config (str): General execution configuration data.
        runtime_env (str): Runtime environment information.
        hydra_config (str): Hydra YAML configuration files (config.yaml, overrides.yaml).
        deriva_config (str): DerivaML execution configuration (configuration.json).
    """

    execution_config = "Execution_Config"
    runtime_env = "Runtime_Env"
    hydra_config = "Hydra_Config"
    deriva_config = "Deriva_Config"


class ExecAssetType(BaseStrEnum):
    """Execution asset type identifiers.

    Defines the types of assets that can be produced or consumed during an execution.
    These types are used to categorize files associated with workflow runs.

    Attributes:
        input_file (str): Input file consumed by the execution.
        output_file (str): Output file produced by the execution.
        notebook_output (str): Jupyter notebook output from the execution.
        model_file (str): Machine learning model file (e.g., .pkl, .h5, .pt).
    """

    input_file = "Input_File"
    output_file = "Output_File"
    notebook_output = "Notebook_Output"
    model_file = "Model_File"

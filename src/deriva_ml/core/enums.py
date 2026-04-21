"""Enumeration classes for DerivaML.

This module provides enumeration classes used throughout DerivaML for representing states, statuses,
types, and vocabularies. Each enum class represents a specific set of constants used in the system.

Classes:
    UploadState: States for file upload operations.
    Status: Execution status values.
    BuiltinTypes: Alias for BuiltinType from deriva.core.typed.
    MLVocab: Controlled vocabulary types.
    MLAsset: Asset type identifiers.
    ExecMetadataType: Execution metadata type identifiers.
    ExecAssetType: Execution asset type identifiers.
"""

from enum import Enum, StrEnum

# Import BuiltinType from deriva.core.typed
from deriva.core.typed import BuiltinType

# Backwards compatibility alias - DerivaML uses plural form
BuiltinTypes = BuiltinType
"""Alias for BuiltinType from deriva.core.typed.

This maintains backwards compatibility with existing DerivaML code that uses
the plural form 'BuiltinTypes'. New code should use BuiltinType directly.
"""


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


# Note: the legacy `Status` StrEnum was deleted in Phase 2 Subsystem 1a.
# Use `deriva_ml.execution.state_store.ExecutionStatus` instead — it carries
# the canonical 7-state lifecycle (Created, Running, Stopped, Failed,
# Pending_Upload, Uploaded, Aborted) with title-case values that match the
# catalog Execution.Status column directly.


class MLVocab(StrEnum):
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


class MLAsset(StrEnum):
    """Asset type identifiers.

    Defines the types of assets that can be associated with executions.

    Attributes:
        execution_metadata (str): Metadata about an execution.
        execution_asset (str): Asset produced by an execution.
    """

    execution_metadata = "Execution_Metadata"
    execution_asset = "Execution_Asset"


class MLTable(StrEnum):
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


class ExecMetadataType(StrEnum):
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


class ExecAssetType(StrEnum):
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

# This module constains enume for controlled vocabulary names that are defined within DerivaML and that are likely to be
# used by end users.

try:
    from enum import Enum, StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

class MLVocab(StrEnum):
    """
    Names of controlled vocabulary for various types within DerivaML.
    """
    dataset_type = 'Dataset_Type'
    workflow_type = 'Workflow_Type'
    execution_asset_type = 'Execution_Asset_Type'
    execution_metadata_type = 'Execution_Metadata_Type'

class ExecMetadataVocab(StrEnum):
    """
    Predefined execution metatadata types.
    """
    execution_config = 'Execution_Config'
    runtime_env = 'Runtime_Env'
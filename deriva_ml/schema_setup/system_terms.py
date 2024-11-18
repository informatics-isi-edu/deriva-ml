
try:
    from enum import Enum, StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

class MLVocab(StrEnum):
    dataset_type = 'Dataset_Type'
    workflow_type = 'Workflow_Type'

class ExecMetadataVocab(StrEnum):
    execution_config = 'Execution_Config'
    runtime_env = 'Runtime_Env'
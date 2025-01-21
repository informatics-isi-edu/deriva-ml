from .deriva_definitions import RID
import json
from pydantic import BaseModel, conlist, model_validator
from typing import Optional, Any

class Workflow(BaseModel):
    """
    A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.  The workflow type
    needs to be an existing-controlled vocabulary term.

    :ivar name: The name of the workflow
    :ivar url: The URI to the workflow instance.  In most cases should be a GitHub URI to the code being executed.
    :ivar workflow_type: The type of the workflow.  Must be an existing controlled vocabulary term.
    :ivar version: The version of the workflow instance.  Should follow semantic versioning.
    :ivar description: A description of the workflow instance.  Can be in markdown format.
    """
    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: str = None

class DatasetSpec(BaseModel):
    rid: RID
    materialize: bool = True

    @model_validator(mode='before')
    @classmethod
    def check_card_number_not_present(cls, data: Any) -> dict[str, str|bool]:
        # If you are just given a string, assume its a rid and put into dict for further validation.
        return {'rid': data} if isinstance(data, str) else data

class ExecutionConfiguration(BaseModel):
    """
    Define the parameters that are used to configure a specific execution.
    :ivar datasets: List of dataset RIDS, MINIDS for datasets to be downloaded prior to execution.  By default,
                     all  the datasets are materialized. However, if the assets associated with a dataset are not
                     needed, a dictionary that defines the rid and the materialization parameter for the
                     download_dataset_bag method can be specified, e.g.  datasets=[{'rid': RID, 'materialize': True}].
    :ivar assets: List of assets to be downloaded prior to execution.  The values must be RIDs in an asset table
    :ivar workflow: A workflow instance.  Must have a name, URI to the workflow instance, and a type.
    :ivar description: A description of the execution.  Can use markdown format.
    """
    datasets: conlist(DatasetSpec) = []
    assets: list[RID|str] = []      # List of RIDs to model files.
    workflow: Workflow
    description: str = ""

    @staticmethod
    def load_configuration(file: str) -> "ExecutionConfiguration":
        """
        Create a ExecutionConfiguration from a JSON configuration file.
        :param file:
        :return:  An execution configuration whose values are loaded from the given file.
        """
        with open(file) as fd:
                return ExecutionConfiguration.model_validate(json.load(fd))
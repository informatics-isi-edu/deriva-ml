from docutils.frontend import validate_dependency_file

from .deriva_definitions import RID
import json
from pydantic import BaseModel, field_validator, conlist, model_validator
from typing import Optional, Any

class Workflow(BaseModel):
    """
    A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.  The workflow type
    needs to be an existing controlled vocabulary term.

    :param name: The name of the workflow
    :param type:  The name of an existing controlled vocabulary term.
    :param uri: The URI to the workflow instance.  In most cases should be a GitHub URI to the code being executed.
    :param version: The version of the workflow instance.  Should follow semantic versioning.
    :param description: A description of the workflow instance.  Can be in markdown format.
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
    :param datasets: List of dataset RIDS, MINIDS for datasets to be downloaded prior to execution.  By default,
                     all  the datasets are materialized. However, if the assets associated with a dataset are not
                     needed, a dictionary that defines the rid and the materialization parameter for the
                     download_dataset_bag method can be specified, e.g.  datasets=[{'rid': RID, 'materialize': True}].
    :param assets: List of assets to be downloaded prior to execution.  The values must be RIDs in an asset table
    :param workflow: A workflow instance.  Must have a name, URI to the workflow instance, and a type.
    :param description: A description of the execution.  Can use markdown format.
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
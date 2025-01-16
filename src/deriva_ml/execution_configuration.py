from .deriva_definitions import RID
import json
from pydantic import BaseModel
from typing import Optional

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


class ExecutionConfiguration(BaseModel):
    datasets: list[RID|str] = []
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
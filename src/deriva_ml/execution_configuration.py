from .deriva_definitions import RID
import json
from pydantic import BaseModel
from typing import Optional

class Workflow(BaseModel):
    """
    A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.
    """
    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: str = None


class ExecutionConfiguration(BaseModel):
    datasets: list[str] = []
    assets: list[RID] = []      # List of RIDs to model files.
    workflow: Workflow
    description: str = ""

    @staticmethod
    def load_configuration(file: str) -> "ExecutionConfiguration":
        with open(file) as fd:
                return ExecutionConfiguration.model_validate(json.load(fd))
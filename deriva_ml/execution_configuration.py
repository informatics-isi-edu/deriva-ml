from typing import Optional
from pydantic import BaseModel
import json


class Workflow(BaseModel):
    """
    A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.
    """
    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: str = None


class Execution(BaseModel):
    description: str


class ExecutionConfiguration(BaseModel):
    bdbag_url: list[str] = []
    models: list[str] = []      # List of RIDs to model files.
    workflow: Workflow
    execution: Execution
    description: str = ""


    @staticmethod
    def load_configuration(file: str) -> "ExecutionConfiguration":
        with open(file) as fd:
                return ExecutionConfiguration.model_validate(json.load(fd))

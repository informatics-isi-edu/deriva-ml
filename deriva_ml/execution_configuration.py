from typing import List, Optional
from pydantic import BaseModel
from enum import Enum
import json


class Workflow(BaseModel):
    """
    A specificiation of a workflow.  Must have a name, URI to the workflow instance, and a type.
    """
    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: str = None


class Execution(BaseModel):
    description: str


class Term(str, Enum):
    annotation = "Annotation_Type"
    workflow = "Workflow_Type"
    execution_asset_type = "Execution_Asset_Type"
    execution_metadata_type = "Execution_Metadata_Type"


class WorkflowTerm(BaseModel):
    term: Term
    name: str
    description: str


class ExecutionConfiguration(BaseModel):
    bdbag_url: list[str] = []
    models: list[str] = []
    workflow: Workflow
    execution: Execution
    workflow_terms: list[WorkflowTerm] = []

    @staticmethod
    def load_configuration(file: str):
        with open(file) as fd:
            return ExecutionConfiguration.model_validate(json.load(fd))

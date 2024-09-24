from typing import List, Optional, Self
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
    execution_asset = "Execution_Asset_Type"
    execution_metadata = "Execution_Metadata_Type"


class WorkflowTerm(BaseModel):
    term: Term          # The vocaublary in which the term is found
    name: str           # The term
    description: str    # A description


class ExecutionConfiguration(BaseModel):
    bdbag_url: list[str] = []
    models: list[str] = []      # List of RIDs to model files.
    workflow: Workflow
    execution: Execution
    workflow_terms: list[WorkflowTerm] = []
    description: str = ""


    @staticmethod
    def load_configuration(file: str) -> "ExecutionConfiguration":
        with open(file) as fd:
                return ExecutionConfiguration.model_validate(json.load(fd))

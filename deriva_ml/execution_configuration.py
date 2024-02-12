from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


class Workflow(BaseModel):
    name: str
    url: str
    workflow_type: str
    version: Optional[str]
    description: str


class Execution(BaseModel):
    description: str


class Term(str, Enum):
    annotation = "Annotation_Type"
    diagnosis = "Diagnosis_Tag"


class WorkflowTerm(BaseModel):
    term: Term
    name: str
    description: str


class ExecutionConfiguration(BaseModel):
    host: str
    catalog_id: str
    dataset_rid: List[str]
    bdbag_url: List[str]
    models: List[str]
    workflow: Workflow
    execution: Execution
    workflow_terms: List[WorkflowTerm]

from typing import List, Optional
from pydantic import BaseModel


class Workflow(BaseModel):
    name: str
    url: str
    workflow_type: str
    version: Optional[str]
    description: str


class Execution(BaseModel):
    description: str


class AnnotationTag(BaseModel):
    name: str
    description: str
    synonyms: List[str] = []


class DiagnosisTag(BaseModel):
    name: str
    description: str
    synonyms: List[str] = []


class ExecutionConfiguration(BaseModel):
    host: str
    catalog_id: str
    dataset_rid: List[str]
    bdbag_url: List[str]
    models: List[str]
    workflow: Workflow
    execution: Execution
    annotation_tag: Optional[AnnotationTag] = None
    diagnosis_tag: Optional[DiagnosisTag] = None

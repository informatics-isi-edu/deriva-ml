from typing import List
from pydantic import BaseModel

class Process(BaseModel):
    name: str
    process_tag_name: str
    description: str
    owner: str
    repo: str
    file_path: str


class Workflow(BaseModel):
    name: str
    description: str
    owner: str
    repo: str
    file_path: str


class Execution(BaseModel):
    name: str
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
    process: List[Process]
    workflow: Workflow
    execution: Execution
    annotation_tag: AnnotationTag
    diagnosis_tag: DiagnosisTag

"""Workflow management mixin for DerivaML.

This module provides the WorkflowMixin class which handles
workflow operations including adding, looking up, listing,
and creating workflows.
"""

from __future__ import annotations

from typing import Any, Callable

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_deriva_core = importlib.import_module("deriva.core")
format_exception = _deriva_core.format_exception

from deriva_ml.core.definitions import RID, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.workflow import Workflow


class WorkflowMixin:
    """Mixin providing workflow management operations.

    This mixin requires the host class to have:
        - ml_schema: str - name of the ML schema
        - pathBuilder(): method returning catalog path builder
        - lookup_term(): method for vocabulary term lookup (from VocabularyMixin)

    Methods:
        find_workflows: Find all workflows in the catalog
        add_workflow: Add a workflow to the catalog
        lookup_workflow: Find a workflow by URL or checksum
        create_workflow: Create a new workflow definition
    """

    # Type hints for IDE support - actual attributes/methods from host class
    ml_schema: str
    pathBuilder: Callable[[], Any]
    lookup_term: Callable[[str, str], VocabularyTerm]

    def find_workflows(self) -> list[Workflow]:
        """Find all workflows in the catalog.

        Catalog-level operation to find all workflow definitions, including their
        names, URLs, types, versions, and descriptions.

        Returns:
            list[Workflow]: List of workflow objects, each containing:
                - name: Workflow name
                - url: Source code URL
                - workflow_type: Type of workflow
                - version: Version identifier
                - description: Workflow description
                - rid: Resource identifier
                - checksum: Source code checksum

        Examples:
            >>> workflows = ml.find_workflows()
            >>> for w in workflows:
                    print(f"{w.name} (v{w.version}): {w.description}")
                    print(f"  Source: {w.url}")
        """
        # Get a workflow table path and fetch all workflows
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow
        return [
            Workflow(
                name=w["Name"],
                url=w["URL"],
                workflow_type=w["Workflow_Type"],
                version=w["Version"],
                description=w["Description"],
                rid=w["RID"],
                checksum=w["Checksum"],
            )
            for w in workflow_path.entities().fetch()
        ]

    def add_workflow(self, workflow: Workflow) -> RID:
        """Adds a workflow to the catalog.

        Registers a new workflow in the catalog or returns the RID of an existing workflow with the same
        URL or checksum.

        Each workflow represents a specific computational process or analysis pipeline.

        Args:
            workflow: Workflow object containing name, URL, type, version, and description.

        Returns:
            RID: Resource Identifier of the added or existing workflow.

        Raises:
            DerivaMLException: If workflow insertion fails or required fields are missing.

        Examples:
            >>> workflow = Workflow(
            ...     name="Gene Analysis",
            ...     url="https://github.com/org/repo/workflows/gene_analysis.py",
            ...     workflow_type="python_script",
            ...     version="1.0.0",
            ...     description="Analyzes gene expression patterns"
            ... )
            >>> workflow_rid = ml.add_workflow(workflow)
        """
        # Check if a workflow already exists by URL
        if workflow_rid := self.lookup_workflow(workflow.checksum or workflow.url):
            return workflow_rid

        # Get an ML schema path for the workflow table
        ml_schema_path = self.pathBuilder().schemas[self.ml_schema]

        try:
            # Create a workflow record
            workflow_record = {
                "URL": workflow.url,
                "Name": workflow.name,
                "Description": workflow.description,
                "Checksum": workflow.checksum,
                "Version": workflow.version,
                MLVocab.workflow_type: self.lookup_term(MLVocab.workflow_type, workflow.workflow_type).name,
            }
            # Insert a workflow and get its RID
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]["RID"]
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(f"Failed to insert workflow. Error: {error}")
        return workflow_rid

    def lookup_workflow(self, url_or_checksum: str) -> RID | None:
        """Finds a workflow by URL.

        Args:
            url_or_checksum: URL or checksum of the workflow.
        Returns:
            RID: Resource Identifier of the workflow if found, None otherwise.

        Example:
            >>> rid = ml.lookup_workflow("https://github.com/org/repo/workflow.py")
            >>> if rid:
            ...     print(f"Found workflow: {rid}")
        """
        # Get a workflow table path
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow
        workflow_rid = None
        for w in workflow_path.path.entities().fetch():
            if w['URL'] == url_or_checksum or w['Checksum'] == url_or_checksum:
                workflow_rid = w['RID']

        return workflow_rid

    def create_workflow(self, name: str, workflow_type: str, description: str = "") -> Workflow:
        """Creates a new workflow definition.

        Creates a Workflow object that represents a computational process or analysis pipeline. The workflow type
        must be a term from the controlled vocabulary. This method is typically used to define new analysis
        workflows before execution.

        Args:
            name: Name of the workflow.
            workflow_type: Type of workflow (must exist in workflow_type vocabulary).
            description: Description of what the workflow does.

        Returns:
            Workflow: New workflow object ready for registration.

        Raises:
            DerivaMLException: If workflow_type is not in the vocabulary.

        Examples:
            >>> workflow = ml.create_workflow(
            ...     name="RNA Analysis",
            ...     workflow_type="python_notebook",
            ...     description="RNA sequence analysis pipeline"
            ... )
            >>> rid = ml.add_workflow(workflow)
        """
        # Validate workflow type exists in vocabulary
        self.lookup_term(MLVocab.workflow_type, workflow_type)

        # Create and return a new workflow object
        return Workflow(name=name, workflow_type=workflow_type, description=description)

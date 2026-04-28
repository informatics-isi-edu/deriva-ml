"""Workflow management mixin for DerivaML.

This module provides the WorkflowMixin class which handles
workflow operations including adding, looking up, listing,
and creating workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from deriva.core import format_exception

from deriva_ml.core.definitions import RID, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.sort import SortSpec, resolve_sort
from deriva_ml.execution.workflow import Workflow

if TYPE_CHECKING:
    pass


class WorkflowMixin:
    """Mixin providing workflow management operations.

    This mixin requires the host class to have:
        - ml_schema: str - name of the ML schema
        - pathBuilder(): method returning catalog path builder
        - lookup_term(): method for vocabulary term lookup (from VocabularyMixin)

    Methods:
        find_workflows: Find all workflows in the catalog
        _add_workflow: Add a workflow to the catalog (internal factory)
        lookup_workflow: Look up a workflow by RID
        find_workflow_by_url: Find a workflow by URL or checksum
        create_workflow: Create a new workflow definition
        list_workflow_executions: List execution RIDs for a workflow (via FeatureMixin)
    """

    # Type hints for IDE support - actual attributes/methods from host class
    ml_schema: str
    pathBuilder: Callable[[], Any]
    lookup_term: Callable[[str, str], VocabularyTerm]

    def _get_workflow_types_for_rid(self, workflow_rid: RID) -> list[str]:
        """Get workflow types for a workflow RID from the association table.

        Args:
            workflow_rid: RID of the workflow.

        Returns:
            List of workflow type term names.
        """
        pb = self.pathBuilder()
        assoc_path = pb.schemas[self.ml_schema].Workflow_Workflow_Type
        types = assoc_path.filter(assoc_path.Workflow == workflow_rid).attributes(assoc_path.Workflow_Type).fetch()
        return [t["Workflow_Type"] for t in types]

    def _get_workflow_types_index(self) -> dict[RID, list[str]]:
        """Fetch the entire ``Workflow_Workflow_Type`` table once, indexed by Workflow RID.

        Used by ``find_workflows`` to avoid issuing one
        ``_get_workflow_types_for_rid`` query per workflow (1+N pattern).
        For catalogs with many workflows the savings are linear.

        Returns:
            Dict mapping each workflow RID to its list of workflow-type
            term names. Workflows with no type associations are absent
            from the dict (the caller treats that as ``[]``).
        """
        pb = self.pathBuilder()
        assoc_path = pb.schemas[self.ml_schema].Workflow_Workflow_Type
        index: dict[RID, list[str]] = {}
        for row in assoc_path.attributes(assoc_path.Workflow, assoc_path.Workflow_Type).fetch():
            index.setdefault(row["Workflow"], []).append(row["Workflow_Type"])
        return index

    def find_workflows(self, sort: SortSpec = None) -> list[Workflow]:
        """Find all workflows in the catalog.

        Catalog-level operation to find all workflow definitions, including their
        names, URLs, types, versions, and descriptions. Each returned Workflow
        is bound to the catalog, allowing its description to be updated.

        Args:
            sort: Optional sort spec.
                - ``None`` (default): backend-determined order (no sort
                  clause applied; cheapest path).
                - ``True``: newest-first by record creation time
                  (``RCT desc``). Recommended for "show me the most
                  recent workflows" queries.
                - Callable ``(path) -> sort_keys``: receives the
                  Workflow table path and returns one or more
                  path-builder sort keys.

        Returns:
            list[Workflow]: List of workflow objects, each containing:
                - name: Workflow name
                - url: Source code URL
                - workflow_type: Type(s) of workflow
                - version: Version identifier
                - description: Workflow description
                - rid: Resource identifier
                - checksum: Source code checksum

        Examples:
            List all workflows and their descriptions::

                >>> workflows = ml.find_workflows()
                >>> for w in workflows:
                ...     print(f"{w.name} (v{w.version}): {w.description}")
                ...     print(f"  Source: {w.url}")

            Update a workflow's description (workflows are catalog-bound)::

                >>> workflows = ml.find_workflows()
                >>> workflows[0].description = "Updated description"

            Newest-first (most common)::

                >>> recent = list(ml.find_workflows(sort=True))  # doctest: +SKIP
        """
        # Get a workflow table path and fetch all workflows.
        # Pre-fetch the full Workflow_Workflow_Type table once and index
        # it by RID instead of issuing one association-table query per
        # workflow — that 1+N pattern dominated find_workflows on
        # catalogs with hundreds of workflows.
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow
        entity_set = workflow_path.entities()
        sort_keys = resolve_sort(sort, lambda p: p.RCT.desc, workflow_path)
        if sort_keys is not None:
            entity_set = entity_set.sort(*sort_keys)
        types_index = self._get_workflow_types_index()
        workflows = []
        for w in entity_set.fetch():
            workflow = Workflow(
                name=w["Name"],
                url=w["URL"],
                workflow_type=types_index.get(w["RID"], []),
                version=w["Version"],
                description=w["Description"],
                rid=w["RID"],
                checksum=w["Checksum"],
            )
            # Bind the workflow to this catalog instance
            workflow._ml_instance = self  # type: ignore[assignment]
            workflows.append(workflow)
        return workflows

    def _add_workflow(self, workflow: Workflow) -> RID:
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
            >>> workflow_rid = ml._add_workflow(workflow)
        """
        # Check if a workflow already exists by URL or checksum
        if workflow_rid := self._find_workflow_rid_by_url(workflow.checksum or workflow.url):
            return workflow_rid

        # Get an ML schema path for the workflow table
        ml_schema_path = self.pathBuilder().schemas[self.ml_schema]

        try:
            # Create a workflow record (without Workflow_Type column)
            workflow_record = {
                "URL": workflow.url,
                "Name": workflow.name,
                "Description": workflow.description,
                "Checksum": workflow.checksum,
                "Version": workflow.version,
            }
            # Insert a workflow and get its RID
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]["RID"]

            # Insert workflow type associations in a single batched insert.
            # Resolving each type name still goes through the cached
            # lookup_term, but the catalog round-trip is consolidated.
            if workflow.workflow_type:
                assoc_path = ml_schema_path.Workflow_Workflow_Type
                assoc_rows = [
                    {
                        "Workflow": workflow_rid,
                        MLVocab.workflow_type: self.lookup_term(MLVocab.workflow_type, wt).name,
                    }
                    for wt in workflow.workflow_type
                ]
                assoc_path.insert(assoc_rows)
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(f"Failed to insert workflow. Error: {error}")
        return workflow_rid

    def lookup_workflow(self, rid: RID) -> Workflow:
        """Look up a workflow by its Resource Identifier (RID).

        Retrieves a workflow from the catalog by its RID and returns a Workflow
        object bound to the catalog. The returned Workflow can be modified (e.g.,
        updating its description) and changes will be reflected in the catalog.

        Args:
            rid: Resource Identifier of the workflow to look up.

        Returns:
            Workflow: The workflow object bound to this catalog, allowing
                properties like ``description`` to be updated.

        Raises:
            DerivaMLException: If the RID does not correspond to a workflow
                in the catalog.

        Examples:
            Look up a workflow and read its properties::

                >>> workflow = ml.lookup_workflow("2-ABC1")
                >>> print(f"Name: {workflow.name}")
                >>> print(f"Description: {workflow.description}")
                >>> print(f"Type: {workflow.workflow_type}")

            Update a workflow's description (persisted to catalog)::

                >>> workflow = ml.lookup_workflow("2-ABC1")
                >>> workflow.description = "Updated analysis pipeline for RNA sequences"
                >>> # The change is immediately written to the catalog

            Attempting to update on a read-only catalog raises an error::

                >>> snapshot = ml.catalog_snapshot("2023-01-15T10:30:00")
                >>> workflow = snapshot.lookup_workflow("2-ABC1")
                >>> workflow.description = "New description"
                DerivaMLException: Cannot update workflow description on a read-only
                    catalog snapshot. Use a writable catalog connection instead.
        """
        # Get the workflow table path
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow

        # Filter by RID
        records = list(workflow_path.filter(workflow_path.RID == rid).entities().fetch())

        if not records:
            raise DerivaMLException(f"Workflow with RID '{rid}' not found in the catalog")

        w = records[0]
        workflow_types = self._get_workflow_types_for_rid(w["RID"])
        workflow = Workflow(
            name=w["Name"],
            url=w["URL"],
            workflow_type=workflow_types,
            version=w["Version"],
            description=w["Description"],
            rid=w["RID"],
            checksum=w["Checksum"],
        )
        # Bind the workflow to this catalog instance for write-back support
        workflow._ml_instance = self  # type: ignore[assignment]
        return workflow

    def _find_workflow_rid_by_url(self, url_or_checksum: str) -> RID | None:
        """Internal method to find a workflow RID by URL or checksum.

        Args:
            url_or_checksum: URL or checksum of the workflow to find.

        Returns:
            RID: Resource Identifier of the workflow if found, None otherwise.
        """
        # Get a workflow table path
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow
        workflow_rid = None
        for w in workflow_path.path.entities().fetch():
            if w["URL"] == url_or_checksum or w["Checksum"] == url_or_checksum:
                workflow_rid = w["RID"]
                break

        return workflow_rid

    def lookup_workflow_by_url(self, url_or_checksum: str) -> Workflow:
        """Look up a workflow by URL or checksum and return the full Workflow object.

        Searches for a workflow in the catalog that matches the given URL or
        checksum and returns a Workflow object bound to the catalog. This allows
        you to both identify a workflow by its source code location and modify
        its properties (e.g., description).

        The URL should be a GitHub URL pointing to the specific version of the
        workflow source code. The format typically includes the commit hash::

            https://github.com/org/repo/blob/<commit_hash>/path/to/workflow.py

        Alternatively, you can search by the Git object hash (checksum) of the
        workflow file.

        Args:
            url_or_checksum: GitHub URL with commit hash, or Git object hash
                (checksum) of the workflow file.

        Returns:
            Workflow: The workflow object bound to this catalog, allowing
                properties like ``description`` to be updated.

        Raises:
            DerivaMLException: If no workflow with the given URL or checksum
                is found in the catalog.

        Examples:
            Look up a workflow by its GitHub URL::

                >>> url = "https://github.com/org/repo/blob/abc123/analysis.py"
                >>> workflow = ml.lookup_workflow_by_url(url)
                >>> print(f"Found: {workflow.name}")
                >>> print(f"Version: {workflow.version}")

            Look up by Git object hash (checksum)::

                >>> workflow = ml.lookup_workflow_by_url("abc123def456789...")
                >>> print(f"Name: {workflow.name}")
                >>> print(f"URL: {workflow.url}")

            Update the workflow's description after lookup::

                >>> workflow = ml.lookup_workflow_by_url(url)
                >>> workflow.description = "Updated analysis pipeline"
                >>> # The change is persisted to the catalog

            Typical GitHub URL formats supported::

                # Full blob URL with commit hash
                https://github.com/org/repo/blob/abc123def/src/workflow.py

                # The URL is matched exactly, so ensure it matches what was
                # recorded when the workflow was registered
        """
        # Find the RID first
        rid = self._find_workflow_rid_by_url(url_or_checksum)
        if rid is None:
            raise DerivaMLException(f"Workflow with URL or checksum '{url_or_checksum}' not found in the catalog")

        # Use lookup_workflow to get the full object with catalog binding
        return self.lookup_workflow(rid)

    def create_workflow(self, name: str, workflow_type: str | list[str], description: str = "") -> Workflow:
        """Creates a new workflow definition.

        Creates a Workflow object that represents a computational process or analysis pipeline. The workflow type(s)
        must be terms from the controlled vocabulary. This method is typically used to define new analysis
        workflows before execution.

        Args:
            name: Name of the workflow.
            workflow_type: Type(s) of workflow (must exist in workflow_type vocabulary).
                Can be a single string or a list of strings.
            description: Description of what the workflow does.

        Returns:
            Workflow: New workflow object ready for registration.

        Raises:
            DerivaMLException: If any workflow_type is not in the vocabulary.

        Examples:
            >>> workflow = ml.create_workflow(
            ...     name="RNA Analysis",
            ...     workflow_type="python_notebook",
            ...     description="RNA sequence analysis pipeline"
            ... )
            >>> rid = ml._add_workflow(workflow)

            Multiple types::

                >>> workflow = ml.create_workflow(
                ...     name="Training Pipeline",
                ...     workflow_type=["Training", "Embedding"],
                ...     description="Combined training and embedding pipeline"
                ... )
        """
        # Normalize to list and validate each type exists in vocabulary
        types = [workflow_type] if isinstance(workflow_type, str) else workflow_type
        for wt in types:
            self.lookup_term(MLVocab.workflow_type, wt)

        # Create and return a new workflow object
        return Workflow(name=name, workflow_type=workflow_type, description=description)

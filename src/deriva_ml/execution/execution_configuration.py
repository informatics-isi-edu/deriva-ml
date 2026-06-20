"""Configuration management for DerivaML executions.

This module provides functionality for configuring and managing execution parameters in DerivaML.
It includes:

- ExecutionConfiguration class: Core class for execution settings
- Parameter validation: Handles JSON and file-based parameters
- Dataset specifications: Manages dataset versions and materialization
- Asset management: Tracks required input files with optional caching

The module supports both direct parameter specification and JSON-based configuration files.

Typical usage example:
    >>> workflow = ml.lookup_workflow_by_url("https://github.com/my-org/my-repo")  # doctest: +SKIP
    >>> config = ExecutionConfiguration(  # doctest: +SKIP
    ...     workflow=workflow,
    ...     datasets=[DatasetSpec(rid="1-abc123", version="1.0.0")],
    ...     description="Process sample data"
    ... )
    >>> execution = ml.create_execution(config)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pydantic import BaseModel, Field, field_validator

from deriva_ml.asset.aux_classes import AssetSpec, LocalFile
from deriva_ml.core.validation import VALIDATION_CONFIG
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.workflow import Workflow


class ExecutionConfiguration(BaseModel):
    """Configuration for a DerivaML execution.

    Defines the complete configuration for a computational or manual process in DerivaML,
    including required datasets, input assets, workflow definition, and parameters.

    Attributes:
        datasets (list[DatasetSpec]): Dataset specifications, each containing:
            - rid: Dataset Resource Identifier
            - version: Version to use
            - materialize: Whether to extract dataset contents
        assets (list[AssetSpec]): Asset specifications. Each element can be:
            - A plain RID string (no caching)
            - An ``AssetSpec(rid=..., cache=True)`` for checksum-based caching
        workflow (Workflow | None): Workflow object defining the computational process.
            Use ``ml.lookup_workflow(rid)`` or ``ml.lookup_workflow_by_url(url)`` to get
            a Workflow object from a RID or URL. Defaults to ``None``, which means the
            workflow must be provided via the ``workflow`` parameter of
            ``ml.create_execution()`` instead. If no workflow is specified in either
            place, a ``DerivaMLException`` is raised at execution creation time.
        description (str): Description of execution purpose (supports Markdown).
        argv (list[str]): Command line arguments used to start execution.
        config_choices (dict[str, str]): Hydra config group choices that were selected.
            Maps group names to selected config names (e.g., {"model_config": "cifar10_quick"}).
            Automatically populated by run_model() and get_notebook_configuration().

    Example:
        >>> # Plain RIDs (backward compatible)
        >>> config = ExecutionConfiguration(assets=["6-EPNR", "6-EP56"])
        >>>
        >>> # Mixed: cached model weights + uncached embeddings
        >>> config = ExecutionConfiguration(
        ...     assets=[
        ...         AssetSpec(rid="6-EPNR", cache=True),
        ...         "6-EP56",
        ...     ]
        ... )
    """

    datasets: list[DatasetSpec] = []
    assets: list[AssetSpec | LocalFile] = []
    workflow: Workflow | None = None
    description: str = ""
    argv: list[str] = Field(default_factory=lambda: sys.argv)
    config_choices: dict[str, str] = Field(default_factory=dict)

    model_config = VALIDATION_CONFIG

    @field_validator("assets", mode="before")
    @classmethod
    def validate_assets(cls, value: Any) -> Any:
        """Normalize asset entries, routing on shape (the safety chokepoint).

        Routing rule:
          - a ``LocalFile`` instance, or a mapping with a ``path`` key → a
            ``LocalFile`` (a local-file input, registered during resolution);
          - an ``AssetSpec`` instance, a mapping with a ``rid`` key, or a
            **bare string** → an ``AssetSpec`` (a catalog asset, by RID).

        A bare string is *always* a RID — never type-sniffed into a path. A
        filesystem read happens only for an explicitly-typed ``LocalFile``
        (or a ``{path: ...}`` mapping), never for an arbitrary (possibly
        injected) string. This is the abuse-surface boundary.
        """
        def _drop_legacy_role(d: dict) -> dict:
            # ``AssetSpec`` no longer has an ``asset_role`` field (role is
            # context-derived, never specified) and now forbids extra fields.
            # A persisted config from before the removal may still carry an
            # ``asset_role`` key — drop it so old configs still load (the field
            # was always a no-op) rather than failing deserialization.
            return {k: val for k, val in d.items() if k != "asset_role"}

        result = []
        for v in value:
            if isinstance(v, (AssetSpec, LocalFile)):
                result.append(v)
            elif isinstance(v, (dict, DictConfig)):
                # A mapping (plain dict from JSON, or a Hydra DictConfig).
                # Route on shape: a ``path`` key is a local-file input; a
                # ``rid`` key (the default) is a catalog asset.
                d = _drop_legacy_role(dict(v))
                if "path" in d:
                    result.append(LocalFile(**d))
                else:
                    result.append(AssetSpec(**d))
            elif isinstance(v, str):
                # A bare string is ALWAYS a RID — never sniffed into a path.
                result.append(AssetSpec(rid=v))
            else:
                # Unknown type — coerce to a RID string (never a path).
                result.append(AssetSpec(rid=str(v)))
        return result

    @staticmethod
    def load_configuration(path: Path) -> ExecutionConfiguration:
        """Creates an ExecutionConfiguration from a JSON file.

        Loads and parses a JSON configuration file into an ExecutionConfiguration
        instance. The file should contain a valid configuration specification.

        Args:
            path: Path to JSON configuration file.

        Returns:
            ExecutionConfiguration: Loaded configuration instance.

        Raises:
            ValueError: If JSON file is invalid or missing required fields.
            FileNotFoundError: If configuration file doesn't exist.

        Example:
            >>> config = ExecutionConfiguration.load_configuration(Path("config.json"))  # doctest: +SKIP
            >>> print(f"Workflow: {config.workflow}")  # doctest: +SKIP
            >>> print(f"Datasets: {len(config.datasets)}")  # doctest: +SKIP
        """
        with Path(path).open() as fd:
            config = json.load(fd)
        return ExecutionConfiguration.model_validate(config)

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

from deriva_ml.asset.aux_classes import AssetSpec
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
    assets: list[AssetSpec] = []
    workflow: Workflow | None = None
    description: str = ""
    argv: list[str] = Field(default_factory=lambda: sys.argv)
    config_choices: dict[str, str] = Field(default_factory=dict)

    model_config = VALIDATION_CONFIG

    @field_validator("assets", mode="before")
    @classmethod
    def validate_assets(cls, value: Any) -> Any:
        """Normalize asset entries to AssetSpec objects.

        Accepts plain RID strings, DictConfig from Hydra, AssetSpec objects,
        or dicts with 'rid' and optional 'cache' keys.
        """
        result = []
        for v in value:
            if isinstance(v, AssetSpec):
                result.append(v)
            elif isinstance(v, dict):
                # Dict with rid/cache keys (e.g., from JSON config)
                result.append(AssetSpec(**v))
            elif isinstance(v, DictConfig):
                # OmegaConf DictConfig from Hydra — may have rid+cache or just rid
                d = dict(v)
                if "rid" in d:
                    result.append(AssetSpec(**d))
                else:
                    # Bare DictConfig with just a .rid attribute.
                    result.append(AssetSpec(rid=v.rid, cache=getattr(v, "cache", False)))
            elif isinstance(v, str):
                result.append(AssetSpec(rid=v))
            else:
                # Unknown type — try string coercion
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

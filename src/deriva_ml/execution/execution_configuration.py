"""Configuration management for DerivaML executions.

This module provides functionality for configuring and managing execution parameters in DerivaML.
It includes:

- ExecutionConfiguration class: Core class for execution settings
- Parameter validation: Handles JSON and file-based parameters
- Dataset specifications: Manages dataset versions and materialization
- Asset management: Tracks required input files

The module supports both direct parameter specification and JSON-based configuration files.

Typical usage example:
    >>> config = ExecutionConfiguration(
    ...     workflow="analysis_workflow",
    ...     datasets=[DatasetSpec(rid="1-abc123", version="1.0.0")],
    ...     parameters={"threshold": 0.5},
    ...     description="Process sample data"
    ... )
    >>> execution = ml.create_execution(config)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deriva_ml.core.definitions import RID
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
        assets (list[RID]): Resource Identifiers of required input assets.
        workflow (RID | Workflow): Workflow definition or its Resource Identifier.
        parameters (dict[str, Any] | Path): Execution parameters, either as:
            - Dictionary of parameter values
            - Path to JSON file containing parameters
        description (str): Description of execution purpose (supports Markdown).
        argv (list[str]): Command line arguments used to start execution.

    Example:
        >>> config = ExecutionConfiguration(
        ...     workflow=Workflow.create_workflow("analysis", "python_script"),
        ...     datasets=[
        ...         DatasetSpec(rid="1-abc123", version="1.0.0", materialize=True)
        ...     ],
        ...     parameters={"threshold": 0.5, "max_iterations": 100},
        ...     description="Process RNA sequence data"
        ... )
    """

    datasets: list[DatasetSpec] = []
    assets: list[RID] = []
    workflow: RID | Workflow
    parameters: dict[str, Any] | Path = {}
    description: str = ""
    argv: list[str] = Field(default_factory=lambda: sys.argv)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> Any:
        """Validates and loads execution parameters.

        If value is a file path, loads and parses it as JSON. Otherwise, returns
        the value as is.

        Args:
            value: Parameter value to validate, either:
                - Dictionary of parameters
                - Path to JSON file
                - String path to JSON file

        Returns:
            dict[str, Any]: Validated parameter dictionary.

        Raises:
            ValueError: If JSON file is invalid or cannot be read.
            FileNotFoundError: If parameter file doesn't exist.

        Example:
            >>> config = ExecutionConfiguration(parameters="params.json")
            >>> print(config.parameters)  # Contents of params.json as dict
        """
        if isinstance(value, str) or isinstance(value, Path):
            with Path(value).open("r") as f:
                return json.load(f)
        else:
            return value

    @field_validator("workflow", mode="before")
    @classmethod
    def validate_workflow(cls, value: Any) -> Any:
        """Validates workflow specification.

        Args:
            value: Workflow value to validate (RID or Workflow object).

        Returns:
            RID | Workflow: Validated workflow specification.
        """
        return value

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
            >>> config = ExecutionConfiguration.load_configuration(Path("config.json"))
            >>> print(f"Workflow: {config.workflow}")
            >>> print(f"Datasets: {len(config.datasets)}")
        """
        with Path(path).open() as fd:
            config = json.load(fd)
        return ExecutionConfiguration.model_validate(config)

    # def download_execution_configuration(
    #     self, configuration_rid: RID
    # ) -> ExecutionConfiguration:
    #     """Create an ExecutionConfiguration object from a catalog RID that points to a JSON representation of that
    #     configuration in hatrac
    #
    #     Args:
    #         configuration_rid: RID that should be to an asset table that refers to an execution configuration
    #
    #     Returns:
    #         A ExecutionConfiguration object for configured by the parameters in the configuration file.
    #     """
    #     AssertionError("Not Implemented")
    #     configuration = self.retrieve_rid(configuration_rid)
    #     with NamedTemporaryFile("w+", delete=False, suffix=".json") as dest_file:
    #         hs = HatracStore("https", self.host_name, self.credential)
    #         hs.get_obj(path=configuration["URL"], destfilename=dest_file.name)
    #         return ExecutionConfiguration.load_configuration(Path(dest_file.name))

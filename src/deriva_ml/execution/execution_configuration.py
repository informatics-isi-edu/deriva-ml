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
    """Define the parameters that are used to configure a specific execution.

    Attributes:
        datasets: List of dataset specifications which specify the dataset RID, version, and if the dataset
            should be materialized.
        assets: List of assets to be downloaded prior to execution.  The values must be RIDs in an asset table
        parameters: Either a dictionary or a path to a JSON file that contains configuration parameters for
            the execution.
        workflow: Either a Workflow object or a RID for a workflow instance.
        parameters: Either a dictionary or a path to a JSON file that contains configuration parameters
            for the execution.
        description: A description of the execution.  Can use Markdown format.
    """

    datasets: list[DatasetSpec] = []
    assets: list[RID] = []  # List of RIDs to model files.
    workflow: RID | Workflow
    parameters: dict[str, Any] | Path = {}
    description: str = ""
    argv: list[str] = Field(default_factory=lambda: sys.argv)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> Any:
        """If a parameter is a file, assume that it has JSON contents for configuration parameters"""
        if isinstance(value, str) or isinstance(value, Path):
            with open(value, "r") as f:
                return json.load(f)
        else:
            return value

    @field_validator("workflow", mode="before")
    @classmethod
    def validate_workflow(cls, value: Any) -> Any:
        return value

    @staticmethod
    def load_configuration(path: Path) -> ExecutionConfiguration:
        """Create an ExecutionConfiguration from a JSON configuration file.

        Args:
          path: File containing JSON version of execution configuration.

        Returns:
          An execution configuration whose values are loaded from the given file.
        """
        with open(path) as fd:
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

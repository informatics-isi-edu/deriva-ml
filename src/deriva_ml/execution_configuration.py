from __future__ import annotations

import json
from typing import Optional

from pydantic import (
    BaseModel,
    conlist,
    ConfigDict,
)
from pathlib import Path


from .dataset_aux_classes import DatasetSpec
from .deriva_definitions import RID


class Workflow(BaseModel):
    """A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.  The workflow type
    needs to be an existing-controlled vocabulary term.

    Attributes:
        name: The name of the workflow
        url: The URI to the workflow instance.  In most cases should be a GitHub URI to the code being executed.
        workflow_type: The type of the workflow.  Must be an existing controlled vocabulary term.
        version: The version of the workflow instance.  Should follow semantic versioning.
        description: A description of the workflow instance.  Can be in Markdown format.
    """

    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: Optional[str] = ""
    rid: Optional[RID] = None
    checksum: Optional[str]


class ExecutionConfiguration(BaseModel):
    """Define the parameters that are used to configure a specific execution.

    Attributes:
        datasets: List of dataset specifications which specify the dataset RID, version and if the dataset
            should be materialized.
        assets: List of assets to be downloaded prior to execution.  The values must be RIDs in an asset table
        workflow: A RID for a workflow instance.  Must have a name, URI to the workflow instance, and a type.
        description: A description of the execution.  Can use Markdown format.
    """

    datasets: conlist(DatasetSpec) = []
    assets: list[RID | str] = []  # List of RIDs to model files.
    workflow: RID
    description: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def load_configuration(path: Path) -> ExecutionConfiguration:
        """Create a ExecutionConfiguration from a JSON configuration file.

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

import logging
from pathlib import Path
from typing import Any

from hydra.conf import HydraConf, RunDir
from hydra_zen import store
from omegaconf import OmegaConf
from pydantic import BaseModel, model_validator

from deriva_ml.core.definitions import ML_SCHEMA


class DerivaMLConfig(BaseModel):
    hostname: str
    catalog_id: str | int = 1
    domain_schema: str | None = None
    project_name: str | None = None
    cache_dir: str | Path | None = None
    working_dir: str | Path | None = None
    ml_schema: str = ML_SCHEMA
    logging_level: Any = logging.WARNING
    deriva_logging_level: Any = logging.WARNING
    credential: Any = None
    use_minid: bool = True
    check_auth: bool = True

    @model_validator(mode="after")
    def init_working_dir(self):
        """
        Sets up the working directory for the model.

        This method configures the working directory, ensuring that all required
        file operations are performed in the appropriate location. If the user does not
        specify a directory, a default directory based on the user's home directory
        or username will be used.

        This is a repeat of what is in the DerivaML.__init__ bu we put this here so that the working
        directory is available to hydra.

        Returns:
            Self: The object instance with the working directory initialized.
        """

        self.working_dir = DerivaMLConfig.compute_workdir(self.working_dir)
        return self

    @staticmethod
    def compute_workdir(working_dir) -> Path:
        # Create a default working directory if none is provided
        working_dir = Path(working_dir) if working_dir else Path.home() / "deriva-ml"
        return working_dir.absolute()


OmegaConf.register_new_resolver("compute_workdir", DerivaMLConfig.compute_workdir, replace=True)
store(
    HydraConf(
        run=RunDir("${compute_workdir:${deriva_ml.working_dir}}/hydra/${now:%Y-%m-%d_%H-%M-%S}"),
        output_subdir="hydra-config",
    ),
    group="hydra",
    name="config",
)

store.add_to_hydra_store()

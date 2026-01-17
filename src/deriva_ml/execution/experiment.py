"""Experiment analysis for DerivaML.

This module provides the Experiment class for analyzing completed executions.
An Experiment wraps an execution RID and provides helper methods for extracting
configuration details, model parameters, and experiment metadata.

Typical usage example:
    >>> from deriva_ml import DerivaML
    >>> from deriva_ml.execution import Experiment
    >>>
    >>> ml = DerivaML("localhost", 45)
    >>> exp = Experiment(ml, "47BE")
    >>> print(exp.name)  # e.g., "cifar10_quick"
    >>> print(exp.config_choices)  # Hydra config names used
    >>> print(exp.model_config)  # Model hyperparameters
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML
    from deriva_ml.execution.execution import Execution
    from deriva_ml.asset.asset import Asset
    from deriva_ml.dataset.dataset import Dataset


@dataclass
class Experiment:
    """Wraps an execution for experiment analysis.

    Provides convenient access to execution metadata, configuration choices,
    model parameters, inputs, and outputs. Useful for comparing experiments
    and generating analysis reports.

    Attributes:
        ml: DerivaML instance for catalog access.
        execution_rid: RID of the execution to analyze.
        execution: The underlying Execution object (lazy-loaded).
        name: Experiment name from config_choices.model_config or execution RID.
        config_choices: Dictionary of Hydra config names used.
        model_config: Dictionary of model hyperparameters.
        description: Execution description.
        status: Execution status (e.g., "Completed").

    Example:
        >>> exp = Experiment(ml, "47BE")
        >>> print(f"Experiment: {exp.name}")
        >>> print(f"Config: {exp.config_choices}")
        >>> for ds in exp.input_datasets:
        ...     print(f"  Input: {ds.dataset_rid}")
    """

    ml: "DerivaML"
    execution_rid: str
    _execution: "Execution | None" = field(default=None, repr=False)
    _hydra_config: dict | None = field(default=None, repr=False)
    _config_choices: dict | None = field(default=None, repr=False)
    _model_config: dict | None = field(default=None, repr=False)
    _name: str | None = field(default=None, repr=False)

    @property
    def execution(self) -> "Execution":
        """Get the underlying Execution object (lazy-loaded)."""
        if self._execution is None:
            self._execution = self.ml.lookup_execution(self.execution_rid)
        return self._execution

    @property
    def hydra_config(self) -> dict:
        """Get the full Hydra configuration from execution metadata.

        Downloads and parses the hydra config YAML file from the execution's
        metadata assets.

        Returns:
            Dictionary containing the full Hydra configuration, or empty dict
            if no config file is found.
        """
        if self._hydra_config is None:
            self._hydra_config = self._load_hydra_config()
        return self._hydra_config

    def _load_hydra_config(self) -> dict:
        """Load the Hydra config YAML from execution metadata assets."""
        # Find the hydra config file in execution metadata
        for asset in self.execution.list_input_assets(asset_role="Output"):
            if asset.filename and "config.yaml" in asset.filename:
                # Download to temp location and parse
                with tempfile.TemporaryDirectory() as tmpdir:
                    dest = Path(tmpdir) / asset.filename
                    self.ml.download_asset(asset.asset_rid, dest.parent)
                    if dest.exists():
                        with open(dest) as f:
                            return yaml.safe_load(f) or {}
        return {}

    @property
    def config_choices(self) -> dict[str, str]:
        """Get the Hydra configuration choices (config names used).

        Returns:
            Dictionary mapping config group names to the selected config names,
            e.g., {"model_config": "cifar10_quick", "datasets": "cifar10_labeled_split"}
        """
        if self._config_choices is None:
            self._config_choices = self.hydra_config.get("config_choices", {})
        return self._config_choices

    @property
    def model_config(self) -> dict[str, Any]:
        """Get the model configuration parameters.

        Returns:
            Dictionary of model hyperparameters from the Hydra config,
            e.g., {"epochs": 3, "learning_rate": 0.001, "batch_size": 128}
        """
        if self._model_config is None:
            self._model_config = self.hydra_config.get("model_config", {})
        return self._model_config

    @property
    def name(self) -> str:
        """Get the experiment name.

        Returns the model_config name from config_choices if available,
        otherwise returns the execution RID.

        Returns:
            Experiment name string.
        """
        if self._name is None:
            self._name = self.config_choices.get("model_config", self.execution_rid)
        return self._name

    @property
    def description(self) -> str:
        """Get the execution description."""
        if self.execution.configuration:
            return self.execution.configuration.description or ""
        return ""

    @property
    def status(self) -> str:
        """Get the execution status."""
        if self.execution.status:
            return self.execution.status.value
        return ""

    @property
    def input_datasets(self) -> list["Dataset"]:
        """Get the input datasets for this experiment.

        Returns:
            List of Dataset objects used as inputs.
        """
        return self.execution.list_input_datasets()

    @property
    def input_assets(self) -> list["Asset"]:
        """Get the input assets for this experiment.

        Returns:
            List of Asset objects used as inputs.
        """
        return self.execution.list_input_assets(asset_role="Input")

    @property
    def output_assets(self) -> list["Asset"]:
        """Get the output assets from this experiment.

        Returns:
            List of Asset objects produced as outputs.
        """
        return self.execution.list_input_assets(asset_role="Output")

    def get_chaise_url(self) -> str:
        """Get the Chaise URL for viewing this execution in the browser.

        Returns:
            URL string for the execution record in Chaise.
        """
        return (
            f"https://{self.ml.host_name}/chaise/record/#{self.ml.catalog_id}/"
            f"deriva-ml:Execution/RID={self.execution_rid}"
        )

    def summary(self) -> dict[str, Any]:
        """Get a summary dictionary of the experiment.

        Returns:
            Dictionary with experiment metadata suitable for display or analysis.
        """
        return {
            "name": self.name,
            "execution_rid": self.execution_rid,
            "description": self.description,
            "status": self.status,
            "config_choices": self.config_choices,
            "model_config": {
                k: v for k, v in self.model_config.items() if not k.startswith("_")
            },
            "input_datasets": [
                {"rid": ds.dataset_rid, "description": ds.description}
                for ds in self.input_datasets
            ],
            "url": self.get_chaise_url(),
        }

    def __repr__(self) -> str:
        return f"Experiment(name={self.name!r}, rid={self.execution_rid!r})"

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
from deriva.core.hatrac_store import HatracStore

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML
    from deriva_ml.execution.execution_record import ExecutionRecord
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
    _execution: "ExecutionRecord | None" = field(default=None, repr=False)
    _hydra_config: dict | None = field(default=None, repr=False)
    _config_choices: dict | None = field(default=None, repr=False)
    _model_config: dict | None = field(default=None, repr=False)
    _name: str | None = field(default=None, repr=False)

    @property
    def execution(self) -> "ExecutionRecord":
        """Get the underlying ExecutionRecord (lazy-loaded)."""
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
        """Load Hydra configuration from execution metadata assets.

        Loads both the config.yaml (model parameters) and hydra.yaml (choices)
        and merges them into a single dictionary with:
        - config_choices: from hydra.yaml runtime.choices
        - model_config: from config.yaml model_config section
        - Full config.yaml contents
        """
        # Query Execution_Metadata_Execution to find metadata assets for this execution
        pb = self.ml.pathBuilder()
        meta_exec = pb.schemas[self.ml.ml_schema].Execution_Metadata_Execution
        metadata_table = pb.schemas[self.ml.ml_schema].Execution_Metadata

        # Find metadata assets linked to this execution with role "Output"
        query = meta_exec.filter(meta_exec.Execution == self.execution_rid)
        query = query.filter(meta_exec.Asset_Role == "Output")
        records = list(query.entities().fetch())

        # Collect metadata records
        metadata_files: dict[str, dict] = {}
        for record in records:
            metadata_rid = record.get("Execution_Metadata")
            if not metadata_rid:
                continue

            meta_records = list(
                metadata_table.filter(metadata_table.RID == metadata_rid)
                .entities()
                .fetch()
            )
            if meta_records:
                meta = meta_records[0]
                filename = meta.get("Filename", "")
                if filename:
                    metadata_files[filename] = meta

        # Create HatracStore for downloading
        hs = HatracStore(
            "https",
            self.ml.host_name,
            self.ml.credential,
        )

        result: dict = {}

        # Load config.yaml for model_config and full configuration
        for filename, meta in metadata_files.items():
            if filename.endswith("-config.yaml"):
                url = meta.get("URL")
                if url:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        dest = Path(tmpdir) / filename
                        hs.get_obj(url, destfilename=str(dest))
                        if dest.exists():
                            with open(dest) as f:
                                result = yaml.safe_load(f) or {}
                break

        # Load hydra.yaml for config_choices (runtime.choices)
        for filename, meta in metadata_files.items():
            if filename.endswith("-hydra.yaml"):
                url = meta.get("URL")
                if url:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        dest = Path(tmpdir) / filename
                        hs.get_obj(url, destfilename=str(dest))
                        if dest.exists():
                            with open(dest) as f:
                                hydra_data = yaml.safe_load(f) or {}
                            # Extract choices from hydra.runtime.choices
                            choices = (
                                hydra_data.get("hydra", {})
                                .get("runtime", {})
                                .get("choices", {})
                            )
                            # Filter out hydra internal choices
                            result["config_choices"] = {
                                k: v
                                for k, v in choices.items()
                                if not k.startswith("hydra/")
                            }
                break

        return result

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
        return self.execution.description or ""

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
                {
                    "dataset_rid": ds.dataset_rid,
                    "description": ds.description,
                    "version": str(ds.current_version) if ds.current_version else None,
                    "dataset_types": ds.dataset_types,
                }
                for ds in self.input_datasets
            ],
            "url": self.get_chaise_url(),
        }

    def to_markdown(self, show_datasets: bool = True, show_assets: bool = True) -> str:
        """Generate a markdown summary of this experiment.

        Returns a formatted markdown string with clickable links, configuration
        details, and optionally input datasets and assets.

        Args:
            show_datasets: If True, include input datasets with nested children.
            show_assets: If True, include input assets.

        Returns:
            Markdown-formatted string.

        Example:
            >>> exp = ml.lookup_experiment("47BE")
            >>> print(exp.to_markdown())
        """
        lines = []

        # Header with execution link
        lines.append(f"### {self.name} ([{self.execution_rid}]({self.get_chaise_url()}))")

        # Description
        if self.description:
            lines.append(f"**Description:** {self.description}")

        # Config choices
        if self.config_choices:
            choices_str = ", ".join(
                f"`{k}={v}`" for k, v in sorted(self.config_choices.items())
            )
            lines.append(f"**Configuration Choices:** {choices_str}")

        # Model configuration (filter internal fields)
        model_cfg = {
            k: v for k, v in self.model_config.items() if not k.startswith("_")
        }
        if model_cfg:
            lines.append("**Model Configuration:**")
            for k, v in sorted(model_cfg.items()):
                lines.append(f"- **{k}**: {v}")

        # Input datasets
        if show_datasets and self.input_datasets:
            lines.append("**Input Datasets:**")
            for ds in self.input_datasets:
                lines.append(ds.to_markdown(show_children=True, indent=0))

        # Input assets
        if show_assets and self.input_assets:
            lines.append("**Input Assets:**")
            for asset in self.input_assets:
                lines.append(
                    f"- [{asset.asset_rid}]({asset.get_chaise_url()}): {asset.filename}"
                )

        return "\n".join(lines)

    def display_markdown(self, show_datasets: bool = True, show_assets: bool = True) -> None:
        """Display a formatted markdown summary of this experiment in Jupyter.

        Convenience method that calls to_markdown() and displays the result
        using IPython.display.Markdown.

        Args:
            show_datasets: If True, display input datasets with nested children.
            show_assets: If True, display input assets.

        Example:
            >>> exp = ml.lookup_experiment("47BE")
            >>> exp.display_markdown()
        """
        from IPython.display import display, Markdown

        display(Markdown(self.to_markdown(show_datasets, show_assets)))

    def __repr__(self) -> str:
        return f"Experiment(name={self.name!r}, rid={self.execution_rid!r})"

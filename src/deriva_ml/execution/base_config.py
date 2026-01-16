"""Base configuration for DerivaML applications.

This module defines the base configuration that both script execution and
notebooks can inherit from. It provides the common hydra defaults structure
without tying to the run_model function.

Usage:
    # In notebooks or project configs
    from deriva_ml.execution import BaseConfig, base_defaults

    @dataclass
    class MyNotebookConfig(BaseConfig):
        my_param: str = "value"

    # Build and register with hydra-zen
    MyConfig = builds(MyNotebookConfig, hydra_defaults=[...])
    store(MyConfig, name="my_config")
"""

from dataclasses import dataclass, field

from hydra_zen import builds, store

from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.definitions import RID
from deriva_ml.dataset.aux_classes import DatasetSpec


# Standard hydra defaults for DerivaML applications.
# Projects can customize these or define their own defaults.
base_defaults = [
    "_self_",
    {"deriva_ml": "default_deriva"},
    {"datasets": "default_dataset"},
    {"assets": "default_asset"},
    {"workflow": "default_workflow"},
    {"model_config": "default_model"},
]


@dataclass
class BaseConfig:
    """Base configuration for DerivaML applications.

    This dataclass defines the common configuration structure shared by
    both script execution and notebook modes. Project-specific configs
    should inherit from this class to get the standard DerivaML fields.

    Attributes:
        deriva_ml: DerivaML connection configuration (hostname, catalog_id, etc.)
        datasets: List of dataset specifications to use for training/inference
        assets: List of asset RIDs to load (model weights, etc.)
        dry_run: If True, skip catalog writes (for testing/debugging)
        description: Human-readable description of this run

    Example:
        >>> from dataclasses import dataclass
        >>> from deriva_ml.execution import BaseConfig
        >>>
        >>> @dataclass
        ... class MyConfig(BaseConfig):
        ...     learning_rate: float = 0.001
        ...     epochs: int = 10
    """
    deriva_ml: DerivaMLConfig = None
    datasets: list[DatasetSpec] = field(default_factory=list)
    assets: list[RID] = field(default_factory=list)
    dry_run: bool = False
    description: str = ""


# Create and register the base config with hydra-zen store.
# This provides a ready-to-use base that experiments can inherit from.
DerivaBaseConfig = builds(
    BaseConfig,
    populate_full_signature=True,
    hydra_defaults=base_defaults,
)

store(DerivaBaseConfig, name="deriva_base")

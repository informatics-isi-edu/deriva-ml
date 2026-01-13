"""Configuration management for DerivaML.

This module provides the DerivaMLConfig class for managing DerivaML instance
configuration. It integrates with hydra-zen for configuration management and supports
both programmatic and structured configuration.

The configuration handles:
    - Server connection settings (hostname, catalog_id, credentials)
    - Schema configuration (domain_schema, ml_schema)
    - Directory paths (working_dir, cache_dir)
    - Logging levels for both DerivaML and underlying Deriva libraries
    - Feature toggles (use_minid, check_auth)

Integration with hydra-zen:
    The module registers a custom resolver for computing working directories
    and configures Hydra's output directory structure for reproducible runs.
    Use hydra-zen's `builds()` and `store` to create composable configurations.

Example:
    Programmatic configuration:
        >>> config = DerivaMLConfig(
        ...     hostname='deriva.example.org',
        ...     catalog_id='my_catalog',
        ...     working_dir='/path/to/work'
        ... )
        >>> ml = DerivaML.instantiate(config)

    With hydra-zen:
        >>> from hydra_zen import builds, instantiate, store, zen
        >>> from deriva_ml import DerivaML
        >>> from deriva_ml.core.config import DerivaMLConfig
        >>>
        >>> # Create a structured config for DerivaML
        >>> DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
        >>>
        >>> # Store configurations for different environments
        >>> store(DerivaMLConf(
        ...     hostname='dev.example.org',
        ...     catalog_id='1',
        ... ), name='dev')
        >>>
        >>> store(DerivaMLConf(
        ...     hostname='prod.example.org',
        ...     catalog_id='52',
        ... ), name='prod')
        >>>
        >>> # Use with Hydra's @hydra.main or zen() wrapper
        >>> @zen(DerivaMLConf)
        ... def my_task(cfg: DerivaMLConfig):
        ...     ml = DerivaML.instantiate(cfg)
        ...     # ... do work with ml instance
"""

import getpass
import logging
from pathlib import Path
from typing import Any

from hydra.conf import HydraConf, RunDir
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store
from omegaconf import OmegaConf
from pydantic import BaseModel, model_validator

from deriva_ml.core.definitions import ML_SCHEMA


class DerivaMLConfig(BaseModel):
    """Configuration model for DerivaML instances.

    This Pydantic model defines all configurable parameters for a DerivaML instance.
    It can be used directly or via Hydra configuration files.

    Attributes:
        hostname: Hostname of the Deriva server (e.g., 'deriva.example.org').
        catalog_id: Catalog identifier, either numeric ID or catalog name.
        domain_schema: Schema name for domain-specific tables. If None, auto-detected.
        project_name: Project name for organizing outputs. Defaults to domain_schema.
        cache_dir: Directory for caching downloaded datasets. Defaults to working_dir/cache.
        working_dir: Base directory for computation data. Defaults to ~/deriva-ml.
        hydra_runtime_output_dir: Hydra's runtime output directory (set automatically).
        ml_schema: Schema name for ML tables. Defaults to 'deriva-ml'.
        logging_level: Logging level for DerivaML. Defaults to WARNING.
        deriva_logging_level: Logging level for Deriva libraries. Defaults to WARNING.
        credential: Authentication credentials. If None, retrieved automatically.
        use_minid: Whether to use MINID service for dataset bags. Defaults to True.
        check_auth: Whether to verify authentication on connection. Defaults to True.

    Example:
        >>> config = DerivaMLConfig(
        ...     hostname='deriva.example.org',
        ...     catalog_id=1,
        ...     domain_schema='my_domain',
        ...     logging_level=logging.INFO
        ... )
    """

    hostname: str
    catalog_id: str | int = 1
    domain_schema: str | None = None
    project_name: str | None = None
    cache_dir: str | Path | None = None
    working_dir: str | Path | None = None
    hydra_runtime_output_dir: str | Path | None = None
    ml_schema: str = ML_SCHEMA
    logging_level: Any = logging.WARNING
    deriva_logging_level: Any = logging.WARNING
    credential: Any = None
    use_minid: bool = True
    check_auth: bool = True

    @model_validator(mode="after")
    def init_working_dir(self) -> "DerivaMLConfig":
        """Initialize working directory after model validation.

        Sets up the working directory path, computing a default if not specified.
        Also captures Hydra's runtime output directory for logging and outputs.

        This validator runs after all field validation and ensures the working
        directory is available for Hydra configuration resolution.

        Returns:
            Self: The configuration instance with initialized paths.
        """
        self.working_dir = DerivaMLConfig.compute_workdir(self.working_dir, self.catalog_id)
        self.hydra_runtime_output_dir = Path(HydraConfig.get().runtime.output_dir)
        return self

    @staticmethod
    def compute_workdir(working_dir: str | Path | None, catalog_id: str | int | None = None) -> Path:
        """Compute the effective working directory path.

        Creates a standardized working directory path. If a base directory is provided,
        appends the current username to prevent conflicts between users. If no directory
        is provided, uses ~/.deriva/deriva-ml. The catalog_id is appended to
        separate data from different catalogs.

        Args:
            working_dir: Base working directory path, or None for default.
            catalog_id: Catalog identifier to include in the path. If None, no
                       catalog subdirectory is created.

        Returns:
            Path: Absolute path to the working directory.

        Example:
            >>> DerivaMLConfig.compute_workdir('/shared/data', '52')
            PosixPath('/shared/data/username/deriva-ml/52')
            >>> DerivaMLConfig.compute_workdir(None, 1)
            PosixPath('/home/username/.deriva/deriva-ml/1')
        """
        # Append username to provided path, or use ~/.deriva as base
        base_dir = (Path(working_dir) / getpass.getuser() if working_dir else Path.home() / ".deriva") / "deriva-ml"
        # Append catalog_id if provided
        if catalog_id is not None:
            base_dir = base_dir / str(catalog_id)
        return base_dir.absolute()


# =============================================================================
# Hydra Integration
# =============================================================================

# Register custom resolver for computing working directories in Hydra configs
# This allows ${compute_workdir:${working_dir},${catalog_id}} syntax in YAML configuration files
OmegaConf.register_new_resolver("compute_workdir", DerivaMLConfig.compute_workdir, replace=True)

# Configure Hydra's output directory structure for reproducible runs
# Outputs are organized by timestamp under the computed working directory
store(
    HydraConf(
        run=RunDir("${compute_workdir:${deriva_ml.working_dir},${deriva_ml.catalog_id}}/hydra/${now:%Y-%m-%d_%H-%M-%S}"),
        output_subdir="hydra-config",
    ),
    group="hydra",
    name="config",
)

# Add the configuration to Hydra's store for discovery
store.add_to_hydra_store()

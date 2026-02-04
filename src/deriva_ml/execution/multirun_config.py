"""Multirun configuration for DerivaML experiments.

This module provides a way to define named multirun configurations that bundle
together Hydra overrides and a description. This allows you to document complex
experiment sweeps in code rather than on the command line.

Usage:
    # In configs/multiruns.py
    from deriva_ml.execution import multirun_config

    multirun_config(
        "quick_vs_extended",
        overrides=[
            "+experiment=cifar10_quick,cifar10_extended",
        ],
        description="## Quick vs Extended Comparison\\n\\nComparing training configs...",
    )

    multirun_config(
        "lr_sweep",
        overrides=[
            "+experiment=cifar10_lr_sweep",
            "model_config.learning_rate=0.0001,0.001,0.01,0.1",
        ],
        description="## Learning Rate Sweep\\n\\nExploring optimal learning rates...",
    )

Then run with:
    deriva-ml-run +multirun=quick_vs_extended
    deriva-ml-run +multirun=lr_sweep model_config.epochs=5  # Can still override

Benefits:
    - Explicit declaration of multirun experiments
    - Rich markdown descriptions for parent executions
    - Reproducible sweeps documented in code
    - Same Hydra override syntax as command line
"""

from dataclasses import dataclass, field


@dataclass
class MultirunSpec:
    """Specification for a multirun experiment.

    Attributes:
        name: Unique identifier for this multirun configuration.
        overrides: List of Hydra override strings (same syntax as command line).
            Examples:
            - "+experiment=cifar10_quick,cifar10_extended"
            - "model_config.learning_rate=0.0001,0.001,0.01"
            - "model_config.epochs=5,10,25,50"
        description: Rich description for the parent execution. Supports full
            markdown formatting (headers, tables, bold, etc.).
    """
    name: str
    overrides: list[str] = field(default_factory=list)
    description: str = ""


# Global registry of multirun configurations
_multirun_registry: dict[str, MultirunSpec] = {}


def multirun_config(
    name: str,
    overrides: list[str],
    description: str = "",
) -> MultirunSpec:
    """Register a named multirun configuration.

    This function registers a multirun specification that can be invoked with
    `deriva-ml-run +multirun=<name>`. The overrides use the same syntax as
    Hydra command-line overrides.

    Args:
        name: Unique name for this multirun configuration. Used to invoke it
            via `+multirun=<name>`.
        overrides: List of Hydra override strings. These are the same overrides
            you would pass on the command line after `--multirun`. Examples:
            - "+experiment=cifar10_quick,cifar10_extended" - run multiple experiments
            - "model_config.learning_rate=0.0001,0.001,0.01" - sweep a parameter
            - "datasets=small,medium,large" - sweep datasets
        description: Rich description for the parent execution. This supports
            full markdown formatting since it's defined in Python, not on the
            command line. Use this to document:
            - What experiments are being compared and why
            - Expected outcomes
            - Methodology and metrics to analyze

    Returns:
        The registered MultirunSpec instance.

    Example:
        >>> from deriva_ml.execution import multirun_config
        >>>
        >>> multirun_config(
        ...     "lr_sweep",
        ...     overrides=[
        ...         "+experiment=cifar10_lr_sweep",
        ...         "model_config.learning_rate=0.0001,0.001,0.01,0.1",
        ...     ],
        ...     description='''## Learning Rate Sweep
        ...
        ... **Objective:** Find optimal learning rate for CIFAR-10 CNN.
        ...
        ... | Learning Rate | Expected Behavior |
        ... |--------------|-------------------|
        ... | 0.0001 | Slow convergence |
        ... | 0.001 | Standard baseline |
        ... | 0.01 | Fast, may overshoot |
        ... | 0.1 | Likely unstable |
        ... ''',
        ... )
    """
    spec = MultirunSpec(
        name=name,
        overrides=overrides,
        description=description,
    )
    _multirun_registry[name] = spec
    return spec


def get_multirun_config(name: str) -> MultirunSpec | None:
    """Look up a registered multirun configuration by name.

    Args:
        name: The name of the multirun configuration.

    Returns:
        The MultirunSpec if found, None otherwise.
    """
    return _multirun_registry.get(name)


def list_multirun_configs() -> list[str]:
    """List all registered multirun configuration names.

    Returns:
        List of registered multirun config names.
    """
    return list(_multirun_registry.keys())


def get_all_multirun_configs() -> dict[str, MultirunSpec]:
    """Get all registered multirun configurations.

    Returns:
        Dictionary mapping names to MultirunSpec instances.
    """
    return dict(_multirun_registry)

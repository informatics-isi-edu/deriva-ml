"""
Deriva ML Model Runner
======================

Generic model runner for executing ML workflows within DerivaML execution contexts.

This module provides `run_model`, a reusable function for running ML models with:
- Automatic execution context creation and management
- Multirun/sweep support with parent-child execution nesting
- Hydra configuration integration
- Support for DerivaML subclasses (e.g., EyeAI, GUDMAP)

Usage
-----
In your project's main script:

    from deriva_ml.execution.runner import run_model, create_model_config
    from deriva_ml import DerivaML
    from hydra_zen import zen, store

    # Create the hydra-zen configuration
    model_config = create_model_config(DerivaML)  # or your subclass
    store(model_config, name="deriva_model")

    # Add your configs...
    store.add_to_hydra_store()

    # Run
    zen(run_model).hydra_main(config_name="deriva_model", version_base="1.3")

For subclasses like EyeAI:

    from eye_ai import EyeAI
    model_config = create_model_config(EyeAI)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar, TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig
from hydra_zen import builds

if TYPE_CHECKING:
    from deriva_ml import DerivaML
    from deriva_ml.core.config import DerivaMLConfig
    from deriva_ml.dataset import DatasetSpec
    from deriva_ml.execution import ExecutionConfiguration, Workflow
    from deriva_ml.core.definitions import RID


# Type variable for DerivaML and its subclasses
T = TypeVar("T", bound="DerivaML")


# =============================================================================
# Multirun State Management
# =============================================================================

class MultirunState:
    """Manages state for multirun (sweep) executions.

    In multirun mode, we create a parent execution that groups all the
    individual sweep jobs. This class holds the shared state needed to
    coordinate between jobs.

    Attributes:
        parent_execution_rid: RID of the parent execution (created on first job)
        parent_execution: The parent Execution object
        ml_instance: Shared DerivaML instance
        job_sequence: Counter for ordering child executions
        sweep_dir: Path to the sweep output directory
    """
    parent_execution_rid: str | None = None
    parent_execution: Any = None
    ml_instance: Any = None  # DerivaML or subclass
    job_sequence: int = 0
    sweep_dir: Path | None = None


# Global instance - persists across jobs in a multirun
_multirun_state = MultirunState()


def _is_multirun() -> bool:
    """Check if we're running in Hydra multirun mode."""
    try:
        hydra_cfg = HydraConfig.get()
        # RunMode.MULTIRUN has value 2
        return hydra_cfg.mode.value == 2
    except Exception:
        return False


def _get_job_num() -> int:
    """Get the current job number in a multirun."""
    try:
        hydra_cfg = HydraConfig.get()
        return hydra_cfg.job.num
    except Exception:
        return 0


def _create_parent_execution(
    ml_instance: "DerivaML",
    workflow: "Workflow",
    description: str,
    dry_run: bool = False,
) -> None:
    """Create the parent execution for a multirun sweep.

    This is called on the first job of a multirun to create the parent
    execution that will group all child executions together.

    Args:
        ml_instance: The DerivaML (or subclass) instance.
        workflow: The workflow to associate with the parent execution.
        description: Description for the parent execution.
        dry_run: If True, don't write to the catalog.
    """
    global _multirun_state

    # Import here to avoid circular imports
    from deriva_ml.execution import ExecutionConfiguration

    # Get sweep info from Hydra
    hydra_cfg = HydraConfig.get()
    sweep_overrides = list(hydra_cfg.overrides.task)

    # Create a description for the parent execution
    parent_description = f"Multirun sweep: {description}\n\nOverrides: {', '.join(sweep_overrides)}"

    # Create parent execution configuration (no datasets - those are for children)
    parent_config = ExecutionConfiguration(
        description=parent_description,
    )

    # Create the parent execution
    parent_execution = ml_instance.create_execution(
        parent_config,
        workflow=workflow,
        dry_run=dry_run,
    )

    # Start the parent execution
    parent_execution.execution_start()

    # Store in global state
    _multirun_state.parent_execution = parent_execution
    _multirun_state.parent_execution_rid = parent_execution.execution_rid
    _multirun_state.ml_instance = ml_instance

    logging.info(f"Created parent execution: {parent_execution.execution_rid}")


def run_model(
    deriva_ml: "DerivaMLConfig",
    datasets: list["DatasetSpec"],
    assets: list["RID"],
    description: str,
    workflow: "Workflow",
    model_config: Any,
    dry_run: bool = False,
    ml_class: type["DerivaML"] | None = None,
) -> None:
    """
    Execute a machine learning model within a DerivaML execution context.

    This function serves as the main entry point called by hydra-zen after
    configuration resolution. It orchestrates the complete execution lifecycle:
    connecting to Deriva, creating an execution record, running the model,
    and uploading results.

    In multirun mode, this function also:
    - Creates a parent execution on the first job to group all sweep jobs
    - Links each child execution to the parent with sequence ordering

    Parameters
    ----------
    deriva_ml : DerivaMLConfig
        Configuration for the DerivaML connection. Contains server URL,
        catalog ID, credentials, and other connection parameters.

    datasets : list[DatasetSpec]
        Specifications for datasets to use in this execution. Each DatasetSpec
        identifies a dataset in the Deriva catalog to be made available to
        the model.

    assets : list[RID]
        Resource IDs (RIDs) of assets to include in the execution. Typically
        used for model weight files, pretrained checkpoints, or other
        artifacts needed by the model.

    description : str
        Human-readable description of this execution run. Stored in the
        Deriva catalog for provenance tracking.

    workflow : Workflow
        The workflow definition to associate with this execution. Defines
        the computational pipeline and its metadata.

    model_config : Any
        A hydra-zen callable that wraps the actual model code. When called
        with `ml_instance` and `execution` arguments, it runs the model
        training or inference logic.

    dry_run : bool, optional
        If True, create the execution record but skip actual model execution.
        Useful for testing configuration without running expensive computations.
        Default is False.

    ml_class : type[DerivaML], optional
        The DerivaML class (or subclass) to instantiate. If None, uses the
        base DerivaML class. Use this to instantiate domain-specific classes
        like EyeAI or GUDMAP.

    Returns
    -------
    None
        Results are uploaded to the Deriva catalog as execution outputs.

    Examples
    --------
    This function is typically not called directly, but through hydra:

        # From command line:
        python deriva_run.py +experiment=cifar10_cnn dry_run=True

        # Multirun (creates parent + child executions):
        python deriva_run.py --multirun +experiment=cifar10_quick,cifar10_extended

        # With a custom DerivaML subclass (in your script):
        from functools import partial
        run_model_eyeai = partial(run_model, ml_class=EyeAI)
    """
    global _multirun_state

    # Import here to avoid circular imports
    from deriva_ml import DerivaML
    from deriva_ml.execution import ExecutionConfiguration

    # ---------------------------------------------------------------------------
    # Clear hydra's logging configuration
    # ---------------------------------------------------------------------------
    # Hydra sets up its own logging handlers which can interfere with DerivaML's
    # logging. Remove them to ensure consistent log output.
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # ---------------------------------------------------------------------------
    # Connect to the Deriva catalog
    # ---------------------------------------------------------------------------
    # Use the provided ml_class or default to DerivaML
    if ml_class is None:
        ml_class = DerivaML

    ml_instance = ml_class.instantiate(deriva_ml)

    # ---------------------------------------------------------------------------
    # Handle multirun mode - create parent execution on first job
    # ---------------------------------------------------------------------------
    is_multirun = _is_multirun()
    if is_multirun and _multirun_state.parent_execution is None:
        _create_parent_execution(ml_instance, workflow, description, dry_run)

    # ---------------------------------------------------------------------------
    # Create the execution context
    # ---------------------------------------------------------------------------
    # The ExecutionConfiguration bundles together all the inputs for this run:
    # which datasets to use, which assets (model weights, etc.), and metadata.

    # In multirun mode, enhance the description with job info
    job_description = description
    if is_multirun:
        job_num = _get_job_num()
        job_description = f"[Job {job_num}] {description}"

    execution_config = ExecutionConfiguration(
        datasets=datasets,
        assets=assets,
        description=job_description
    )

    # Create the execution record in the catalog. This generates a unique
    # execution ID and sets up the working directories for this run.
    execution = ml_instance.create_execution(
        execution_config,
        workflow=workflow,
        dry_run=dry_run
    )

    # ---------------------------------------------------------------------------
    # Link to parent execution in multirun mode
    # ---------------------------------------------------------------------------
    if is_multirun and _multirun_state.parent_execution is not None:
        if not dry_run:
            try:
                # Get the current job sequence from the global state
                job_sequence = _multirun_state.job_sequence
                _multirun_state.parent_execution.add_nested_execution(
                    execution,
                    sequence=job_sequence
                )
                logging.info(
                    f"Linked execution {execution.execution_rid} to parent "
                    f"{_multirun_state.parent_execution_rid} (sequence={job_sequence})"
                )
                # Increment the sequence for the next job
                _multirun_state.job_sequence += 1
            except Exception as e:
                logging.warning(f"Failed to link execution to parent: {e}")

    # ---------------------------------------------------------------------------
    # Run the model within the execution context
    # ---------------------------------------------------------------------------
    # The context manager handles setup (downloading datasets, creating output
    # directories) and teardown (recording completion status, timing).
    with execution.execute() as exec_context:
        if dry_run:
            # In dry run mode, skip model execution but still test the setup
            logging.info("Dry run mode: skipping model execution")
        else:
            # Invoke the model configuration callable. The model_config is a
            # hydra-zen wrapped function that has been partially configured with
            # all model-specific parameters (e.g., learning rate, batch size).
            # We provide the runtime context here.
            model_config(ml_instance=ml_instance, execution=exec_context)

    # ---------------------------------------------------------------------------
    # Upload results to the catalog
    # ---------------------------------------------------------------------------
    # After the model completes, upload any output files (metrics, predictions,
    # model checkpoints) to the Deriva catalog for permanent storage.
    if not dry_run:
        uploaded_assets = execution.upload_execution_outputs()

        # Print summary of uploaded assets
        total_files = sum(len(files) for files in uploaded_assets.values())
        if total_files > 0:
            print(f"\nUploaded {total_files} asset(s) to catalog:")
            for asset_type, files in uploaded_assets.items():
                for f in files:
                    print(f"  - {asset_type}: {f}")


def create_model_config(
    ml_class: type["DerivaML"] | None = None,
    description: str = "Model execution",
    hydra_defaults: list | None = None,
) -> Any:
    """Create a hydra-zen configuration for run_model.

    This helper creates a properly configured hydra-zen builds() for run_model
    with the specified DerivaML class bound via partial application.

    Parameters
    ----------
    ml_class : type[DerivaML], optional
        The DerivaML class (or subclass) to use. If None, uses the base DerivaML.

    description : str, optional
        Default description for executions. Can be overridden at runtime.

    hydra_defaults : list, optional
        Custom hydra defaults. If None, uses standard defaults for deriva_ml,
        datasets, assets, workflow, and model_config groups.

    Returns
    -------
    Any
        A hydra-zen builds() configuration ready to be registered with store().

    Examples
    --------
    Basic usage with DerivaML:

        >>> from deriva_ml.execution.runner import create_model_config
        >>> model_config = create_model_config()
        >>> store(model_config, name="deriva_model")

    With a custom subclass:

        >>> from eye_ai import EyeAI
        >>> model_config = create_model_config(EyeAI, description="EyeAI analysis")
        >>> store(model_config, name="eyeai_model")

    With custom hydra defaults:

        >>> model_config = create_model_config(
        ...     hydra_defaults=[
        ...         "_self_",
        ...         {"deriva_ml": "production"},
        ...         {"datasets": "full_dataset"},
        ...     ]
        ... )
    """
    from functools import partial

    if hydra_defaults is None:
        hydra_defaults = [
            "_self_",
            {"deriva_ml": "default_deriva"},
            {"datasets": "default_dataset"},
            {"assets": "default_asset"},
            {"workflow": "default_workflow"},
            {"model_config": "default_model"},
        ]

    # Create a partial function with ml_class bound
    if ml_class is not None:
        run_func = partial(run_model, ml_class=ml_class)
    else:
        run_func = run_model

    return builds(
        run_func,
        description=description,
        populate_full_signature=True,
        hydra_defaults=hydra_defaults,
    )


def reset_multirun_state() -> None:
    """Reset the global multirun state.

    This is primarily useful for testing to ensure clean state between tests.
    """
    global _multirun_state
    _multirun_state.parent_execution_rid = None
    _multirun_state.parent_execution = None
    _multirun_state.ml_instance = None
    _multirun_state.job_sequence = 0
    _multirun_state.sweep_dir = None

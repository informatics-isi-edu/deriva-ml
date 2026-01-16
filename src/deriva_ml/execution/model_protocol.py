"""
DerivaML Model Protocol
=======================

This module defines the protocol (interface) that model functions must follow
to work with DerivaML's execution framework and the run_model() function.

The DerivaMLModel protocol specifies that models must accept two special
keyword arguments that are injected at runtime:

- ml_instance: The DerivaML (or subclass) instance for catalog operations
- execution: The Execution context for managing inputs, outputs, and provenance

All other parameters are model-specific and configured via Hydra.

Example
-------
A compliant model function:

    def train_classifier(
        # Model-specific parameters (configured via Hydra)
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        # Runtime parameters (injected by run_model)
        ml_instance: DerivaML = None,
        execution: Execution = None,
    ) -> None:
        '''Train a classifier within the DerivaML execution context.'''

        # Download input datasets
        for dataset_spec in execution.datasets:
            bag = execution.download_dataset_bag(dataset_spec)
            images = load_images_from_bag(bag)

        # Train the model
        model = MyClassifier()
        for epoch in range(epochs):
            train_one_epoch(model, images, learning_rate, batch_size)

        # Save outputs (will be uploaded to catalog)
        model_path = execution.asset_file_path("Model", "model.pt")
        torch.save(model.state_dict(), model_path)

        metrics_path = execution.asset_file_path("Execution_Metadata", "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"final_accuracy": 0.95}, f)

Registering with Hydra-Zen
--------------------------
Wrap your model with builds() and zen_partial=True:

    from hydra_zen import builds, store

    TrainClassifierConfig = builds(
        train_classifier,
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
        zen_partial=True,  # Creates a partial function
    )

    # Register in the model_config group
    store(TrainClassifierConfig, group="model_config", name="default_model")

    # Create variants with different defaults
    store(TrainClassifierConfig, epochs=50, group="model_config", name="extended")
    store(TrainClassifierConfig, epochs=5, group="model_config", name="quick")

Type Checking
-------------
Use the DerivaMLModel protocol for type hints in utilities:

    from deriva_ml.execution.model_protocol import DerivaMLModel

    def validate_model(model: DerivaMLModel) -> bool:
        '''Check if a callable conforms to the model protocol.'''
        return isinstance(model, DerivaMLModel)

The protocol uses @runtime_checkable, so isinstance() checks work at runtime.
"""

from __future__ import annotations

from typing import Protocol, Any, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_ml import DerivaML
    from deriva_ml.execution.execution import Execution


@runtime_checkable
class DerivaMLModel(Protocol):
    """Protocol for model functions compatible with DerivaML's run_model().

    A model function must accept keyword arguments `ml_instance` and `execution`
    that are injected at runtime by run_model(). All other parameters are
    configured via Hydra and passed through the model_config.

    The model function is responsible for:
    1. Downloading input datasets via execution.download_dataset_bag()
    2. Performing the ML computation (training, inference, etc.)
    3. Registering output files via execution.asset_file_path()

    Output files registered with asset_file_path() are automatically uploaded
    to the catalog after the model completes.

    Attributes
    ----------
    This protocol defines a callable signature, not attributes.

    Examples
    --------
    Basic model function:

        def my_model(
            epochs: int = 10,
            ml_instance: DerivaML = None,
            execution: Execution = None,
        ) -> None:
            # Training logic here
            pass

    With domain-specific DerivaML subclass:

        def eyeai_model(
            threshold: float = 0.5,
            ml_instance: EyeAI = None,  # EyeAI is a DerivaML subclass
            execution: Execution = None,
        ) -> None:
            # Can use EyeAI-specific methods
            ml_instance.some_eyeai_method()

    Checking protocol compliance:

        >>> from deriva_ml.execution.model_protocol import DerivaMLModel
        >>> isinstance(my_model, DerivaMLModel)
        True
    """

    def __call__(
        self,
        *args: Any,
        ml_instance: "DerivaML",
        execution: "Execution",
        **kwargs: Any,
    ) -> None:
        """Execute the model within a DerivaML execution context.

        Parameters
        ----------
        *args : Any
            Positional arguments (typically not used; prefer keyword args).
        ml_instance : DerivaML
            The DerivaML instance (or subclass like EyeAI) connected to the
            catalog. Use this for catalog operations not available through
            the execution context.
        execution : Execution
            The execution context manager. Provides:
            - execution.datasets: List of input DatasetSpec objects
            - execution.download_dataset_bag(): Download dataset as BDBag
            - execution.asset_file_path(): Register output file for upload
            - execution.working_dir: Path to local working directory
        **kwargs : Any
            Model-specific parameters configured via Hydra.

        Returns
        -------
        None
            Models should not return values. Results are captured through:
            - Files registered with asset_file_path() (uploaded to catalog)
            - Datasets created with execution.create_dataset()
            - Status updates via execution.update_status()
        """
        ...

# Hydra-zen Configuration

DerivaML integrates with [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for configuration management, enabling reproducible ML workflows with structured, composable configurations.

## Overview

Hydra-zen provides a Pythonic way to configure applications using dataclasses and structured configs. DerivaML leverages this for:

- **Environment Configuration**: Different settings for dev/staging/production
- **Dataset Collections**: Named groups of datasets for experiments
- **Execution Parameters**: Reproducible execution configurations
- **Working Directory Management**: Automatic Hydra output organization

## Quick Start

```python
from hydra_zen import builds, instantiate, store
from deriva_ml import DerivaML, DerivaMLConfig

# Create a structured config for DerivaML
DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

# Configure for your environment
conf = DerivaMLConf(
    hostname='deriva.example.org',
    catalog_id='42',
    domain_schema='my_domain',
)

# Instantiate to get a DerivaMLConfig, then create DerivaML
config = instantiate(conf)
ml = DerivaML.instantiate(config)
```

## Configuration Classes

### DerivaMLConfig

The main configuration class for DerivaML instances:

```python
from deriva_ml import DerivaMLConfig
from hydra_zen import builds

DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

conf = DerivaMLConf(
    hostname='example.org',           # Deriva server hostname
    catalog_id='1',                   # Catalog ID or name
    domain_schema='my_project',       # Domain schema name
    working_dir='/shared/workspace',  # Base working directory
    use_minid=True,                   # Use MINID for dataset bags
    check_auth=True,                  # Verify authentication
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hostname` | str | required | Deriva server hostname |
| `catalog_id` | str/int | 1 | Catalog identifier |
| `domain_schema` | str | None | Domain schema (auto-detected if None) |
| `working_dir` | str/Path | None | Base directory for outputs |
| `ml_schema` | str | "deriva-ml" | ML schema name |
| `use_minid` | bool | True | Use MINID service for datasets |
| `check_auth` | bool | True | Verify authentication on connect |

### DatasetSpecConfig

For configuring dataset specifications in execution configurations:

```python
from deriva_ml.dataset import DatasetSpecConfig

# Create dataset specs for an experiment
training_data = DatasetSpecConfig(
    rid="1ABC",
    version="1.0.0",
    materialize=True,      # Download asset files
    description="Training images"
)

metadata_only = DatasetSpecConfig(
    rid="2DEF",
    version="2.0.0",
    materialize=False,     # Only download table data
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rid` | str | required | Dataset RID |
| `version` | str | required | Semantic version (e.g., "1.2.0") |
| `materialize` | bool | True | Download asset files |
| `description` | str | "" | Description for logging |

### AssetRIDConfig

For configuring input assets (model weights, config files, etc.):

```python
from deriva_ml.execution import AssetRIDConfig

# Define input assets
model_weights = AssetRIDConfig(rid="WXYZ", description="Pretrained model")
config_file = AssetRIDConfig(rid="ABCD", description="Hyperparameters")

assets = [model_weights, config_file]
```

## Working Directory Configuration

DerivaML automatically configures Hydra's output directory based on your `working_dir` setting:

```python
conf = DerivaMLConf(
    hostname='deriva.example.org',
    working_dir='/shared/ml_workspace',  # Custom working directory
)
```

The output structure is:
```
{working_dir}/{username}/deriva-ml/hydra/{timestamp}/
```

For example:
```
/shared/ml_workspace/jsmith/deriva-ml/hydra/2024-01-15_10-30-45/
```

This ensures:
- Each user has isolated workspace
- Outputs are organized by timestamp
- Hydra config files are preserved for reproducibility

## Using the Hydra Store

The hydra-zen store allows you to register named configurations:

### Environment Configurations

```python
from hydra_zen import store
from deriva_ml import DerivaMLConfig

DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

# Register different environments
deriva_store = store(group="deriva_ml")

deriva_store(DerivaMLConf(
    hostname='dev.example.org',
    catalog_id='1',
    use_minid=False,
), name='dev')

deriva_store(DerivaMLConf(
    hostname='prod.example.org',
    catalog_id='100',
    use_minid=True,
), name='prod')
```

### Dataset Collections

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Define dataset collections
training_v1 = [
    DatasetSpecConfig(rid="TRNA", version="1.0.0"),
    DatasetSpecConfig(rid="TRNB", version="1.0.0"),
]

training_v2 = [
    DatasetSpecConfig(rid="TRNA", version="2.0.0"),
    DatasetSpecConfig(rid="TRNB", version="2.0.0"),
    DatasetSpecConfig(rid="TRNC", version="1.0.0"),
]

# Store them
datasets_store = store(group="datasets")
datasets_store(training_v1, name="training_v1")
datasets_store(training_v2, name="training_v2")
```

### Asset Collections

```python
from hydra_zen import store
from deriva_ml.execution import AssetRIDConfig

# Define asset collections
resnet_weights = [
    AssetRIDConfig(rid="RSN1", description="ResNet50 pretrained"),
]

vit_weights = [
    AssetRIDConfig(rid="VIT1", description="ViT-B/16 pretrained"),
    AssetRIDConfig(rid="VIT2", description="ViT fine-tuned"),
]

# Store them
assets_store = store(group="assets")
assets_store(resnet_weights, name="resnet")
assets_store(vit_weights, name="vit")
```

## Complete Execution Configuration

Combine all components for a full execution configuration:

```python
from hydra_zen import builds, instantiate, make_config, store
from deriva_ml import DerivaML, DerivaMLConfig
from deriva_ml.execution import ExecutionConfiguration, Workflow
from deriva_ml.dataset import DatasetSpecConfig

# Build configs
DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)
WorkflowConf = builds(
    Workflow,
    name="Image Classification",
    workflow_type="Training",
    description="Train CNN classifier",
    populate_full_signature=True
)

# Create combined application config
AppConfig = make_config(
    deriva_ml=DerivaMLConf(hostname="example.org", catalog_id="1"),
    execution=ExecConf(
        description="Training run",
        datasets=[
            DatasetSpecConfig(rid="DATA", version="1.0.0"),
        ],
        assets=["WGTS"],
    ),
    workflow=WorkflowConf,
)

# Use in your application
def train(cfg: AppConfig):
    # Instantiate configs
    ml_config = instantiate(cfg.deriva_ml)
    exec_config = instantiate(cfg.execution)

    # Create DerivaML instance
    ml = DerivaML.instantiate(ml_config)

    # Run execution
    with ml.create_execution(exec_config) as exe:
        # ... training code ...
        pass
```

## Using with Hydra CLI

DerivaML provides [`deriva-ml-run`](cli-reference.md#deriva-ml-run) which handles
Hydra configuration composition, config module loading, and execution tracking
automatically. See [Running Models and Notebooks](running-models-and-notebooks.md)
for the recommended workflow.

The examples below show the underlying Hydra CLI patterns for reference:

```bash
# Use default config
python train.py

# Override hostname
python train.py deriva_ml.hostname=staging.example.org

# Use different dataset collection
python train.py +datasets=training_v2

# Multi-run with different configs
python train.py --multirun +datasets=training_v1,training_v2
```

## Example: Complete ML Script

!!! note
    In practice, `create_model_config()` and `deriva-ml-run` handle the
    boilerplate shown below automatically. See
    [Running Models and Notebooks](running-models-and-notebooks.md) for the
    recommended approach.

```python
"""train.py - Example training script with hydra-zen configuration."""
from hydra_zen import builds, instantiate, store, zen
from deriva_ml import DerivaML, DerivaMLConfig
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset import DatasetSpecConfig

# Define configs
DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

# Store environment configs
store(DerivaMLConf(hostname="localhost", catalog_id=1), group="deriva_ml", name="local")
store(DerivaMLConf(hostname="prod.org", catalog_id=100), group="deriva_ml", name="prod")

# Store dataset configs
store([DatasetSpecConfig(rid="DATA", version="1.0.0")], group="datasets", name="default")

# Main config combining all groups
Config = make_config(
    defaults=[
        "_self_",
        {"deriva_ml": "local"},
        {"datasets": "default"},
    ],
    deriva_ml=DerivaMLConf,
    datasets=list,
    description="Training run",
)

@zen(Config)
def main(cfg):
    # Instantiate DerivaML
    ml_config = instantiate(cfg.deriva_ml)
    ml = DerivaML.instantiate(ml_config)

    # Create execution config
    exec_config = ExecutionConfiguration(
        description=cfg.description,
        datasets=[instantiate(d) for d in cfg.datasets],
    )

    # Run
    with ml.create_execution(exec_config) as exe:
        for ds in exe.datasets:
            bag = exe.download_dataset_bag(ds)
            # Process data...

    exe.upload_execution_outputs()

if __name__ == "__main__":
    store.add_to_hydra_store()
    main()
```

## Configuring ML Models with DerivaML

A powerful pattern is to use `zen_partial=True` to create partially configured model functions that receive the DerivaML `Execution` object at runtime. This allows you to:

- Configure model hyperparameters via Hydra
- Access datasets and assets through the execution object
- Keep model code separate from configuration

### Model Protocol

Define a protocol for models that integrate with DerivaML:

```python
# models/model_protocol.py
from typing import Protocol, Any, runtime_checkable
from deriva_ml.execution import Execution
from deriva_ml import DerivaML

@runtime_checkable
class DerivaMLModel(Protocol):
    def __call__(self,
                 *args: Any,
                 ml_instance: DerivaML,
                 execution: Execution,
                 **kwargs: Any) -> None:
        """Interface for functions that integrate DerivaML with ML frameworks."""
        ...
```

### Model Implementation

Create model functions that follow the protocol:

```python
# models/my_model.py
from deriva_ml.execution import Execution
from deriva_ml import DerivaML, MLAsset, ExecAssetType

def train_classifier(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    ml_instance: DerivaML,
    execution: Execution | None = None,
) -> None:
    """Train a classifier using DerivaML execution context.

    Args:
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Training batch size
        ml_instance: DerivaML instance for catalog access
        execution: Execution object with datasets, assets, and working directory
    """
    # Access input assets (e.g., pretrained weights)
    for table, assets in execution.asset_paths.items():
        print(f"Loading assets from {table}:")
        for asset in assets:
            print(f"  {asset}")

    # Access datasets
    for dataset in execution.datasets:
        bag = execution.download_dataset_bag(dataset)
        # Process dataset...

    # Your training code here
    print(f"Training with lr={learning_rate}, epochs={epochs}, batch={batch_size}")

    # Register output files for upload
    model_path = execution.asset_file_path(
        MLAsset.execution_asset,
        "trained_model.pt",
        ExecAssetType.output_file
    )
    # Save model to model_path...
```

### Model Configuration with zen_partial

Use `zen_partial=True` to create a partially applied function:

```python
# configs/model.py
from hydra_zen import builds, store
from models.my_model import train_classifier

# Build the base configuration with zen_partial=True
# This creates a callable that waits for ml_instance and execution
ModelConfig = builds(
    train_classifier,
    learning_rate=1e-3,
    epochs=10,
    batch_size=32,
    populate_full_signature=True,
    zen_partial=True,  # Key: creates partial function
)

# Register configurations
model_store = store(group="model_config")
model_store(ModelConfig, name="default")

# Create variants by overriding specific parameters
model_store(ModelConfig, name="fast_training", epochs=5, learning_rate=1e-2)
model_store(ModelConfig, name="long_training", epochs=100, learning_rate=1e-4)
model_store(ModelConfig, name="large_batch", batch_size=128, epochs=50)
```

### Model Runner

Create a runner that instantiates the partial config with execution context:

```python
# model_runner.py
import logging
from typing import Any
from deriva_ml import DerivaML, DerivaMLConfig, RID
from deriva_ml.dataset import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration, Workflow

def run_model(
    deriva_ml: DerivaMLConfig,
    datasets: list[DatasetSpec],
    assets: list[RID],
    description: str,
    workflow: Workflow,
    model_config: Any,  # Partially configured model callable
    dry_run: bool = False,
) -> None:
    """Execute a configured model with DerivaML.

    Args:
        deriva_ml: DerivaML connection configuration
        datasets: List of dataset specifications
        assets: List of input asset RIDs
        description: Execution description
        workflow: Workflow definition
        model_config: Partially configured model (from zen_partial)
        dry_run: If True, don't record execution in catalog
    """
    # Connect to catalog
    ml_instance = DerivaML.instantiate(deriva_ml)

    # Create execution configuration
    execution_config = ExecutionConfiguration(
        datasets=datasets,
        assets=assets,
        description=description
    )

    execution = ml_instance.create_execution(
        execution_config,
        workflow=workflow,
        dry_run=dry_run
    )

    with execution.execute() as exe:
        # Complete the partial function with runtime arguments
        model_config(ml_instance=ml_instance, execution=exe)

    # Upload outputs after execution completes
    execution.upload_execution_outputs()
```

### Main Script

Tie everything together with a Hydra entry point:

```python
# train.py
from hydra_zen import store, zen, builds

from model_runner import run_model

# Build main application config with defaults
app_config = builds(
    run_model,
    description="Model training run",
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default"},
        {"datasets": "training"},
        {"assets": "weights"},
        {"workflow": "training_workflow"},
        {"model_config": "default"},
    ],
)
store(app_config, name="train_app")

# Import config modules to register them
import configs.deriva      # noqa: F401
import configs.datasets    # noqa: F401
import configs.assets      # noqa: F401
import configs.workflow    # noqa: F401
import configs.model       # noqa: F401

if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(run_model).hydra_main(
        config_name="train_app",
        version_base="1.3",
        config_path=None,
    )
```

### Running the Model

```bash
# Run with defaults
python train.py

# Override model config
python train.py model_config=long_training

# Override multiple parameters
python train.py model_config=fast_training datasets=validation

# Override individual hyperparameters
python train.py model_config.epochs=25 model_config.learning_rate=0.001

# Multi-run experiments
python train.py --multirun model_config=default,fast_training,long_training
```

### Key Benefits of zen_partial

1. **Separation of concerns**: Model hyperparameters are configured separately from runtime context
2. **Deferred execution**: The model function isn't called until `ml_instance` and `execution` are available
3. **Config variants**: Easy to create model variants by overriding specific parameters
4. **CLI flexibility**: All hyperparameters are exposed to Hydra's CLI
5. **Reproducibility**: Full configuration is logged by Hydra

## Configuration Descriptions

Adding descriptions to configurations helps users and AI assistants understand and discover the right configurations. DerivaML provides two mechanisms depending on the configuration type:

### For List-Based Configs (Assets, Datasets)

Use `with_description()` to wrap lists of RIDs or `DatasetSpecConfig` objects:

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

# Datasets with descriptions
datasets_store = store(group="datasets")
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="28D4", version="0.22.0")],
        "Split dataset with 10,000 images (5,000 train + 5,000 test). "
        "Testing images are unlabeled. Use for standard train/test workflows."
    ),
    name="cifar10_split",
)

# Assets with descriptions
asset_store = store(group="assets")
asset_store(
    with_description(
        ["3WMG", "3XPA"],
        "Model weights from quick (3WMG) and extended (3XPA) training runs. "
        "Use for comparison experiments."
    ),
    name="comparison_weights",
)

# Empty default
asset_store(
    with_description([], "No assets - empty default configuration"),
    name="default_asset",
)
```

After instantiation, `config.datasets` and `config.assets` behave like regular lists but have a `.description` attribute:

```python
# Normal list operations work
for dataset in config.datasets:
    print(dataset.rid)

# Access description
print(config.assets.description)  # "Model weights from quick..."
```

### For Model Configs (builds())

Use `zen_meta` parameter when storing `builds()` configs:

```python
from hydra_zen import builds, store
from models.my_model import train_classifier

model_store = store(group="model_config")

ModelConfig = builds(
    train_classifier,
    learning_rate=1e-3,
    epochs=10,
    populate_full_signature=True,
    zen_partial=True,
)

# Add description via zen_meta
model_store(
    ModelConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default training config: 10 epochs, lr=1e-3. "
            "Balanced for standard training runs."
        )
    },
)

# Variant with description
model_store(
    ModelConfig,
    name="quick",
    epochs=3,
    batch_size=128,
    zen_meta={
        "description": (
            "Quick validation: 3 epochs, batch 128. "
            "Use for rapid iteration and debugging."
        )
    },
)
```

### Summary: When to Use Each Mechanism

| Config Type | Storage Pattern | Description Mechanism |
|-------------|-----------------|----------------------|
| Assets (RID lists) | `store(["RID1", "RID2"], ...)` | `with_description(items, desc)` |
| Datasets (DatasetSpecConfig lists) | `store([DatasetSpecConfig(...)], ...)` | `with_description(items, desc)` |
| Model configs | `store(builds(...), ...)` | `zen_meta={"description": desc}` |
| Workflow configs | `store(builds(Workflow, ...), ...)` | `zen_meta={"description": desc}` |

### Writing Good Descriptions

Include:
- **What it contains**: Size, types, key parameters
- **Where it came from**: Source execution, version
- **When to use it**: Training, testing, debugging, production

Examples:

```python
# ✓ Good dataset description
"Training dataset with 5,000 labeled CIFAR-10 images (32x32 RGB). "
"All images have ground truth classifications."

# ✓ Good asset description
"Model weights (model.pt) from extended training: 50 epochs, "
"64→128 channels, dropout 0.25. Use for inference or fine-tuning."

# ✓ Good model config description
"Quick training: 3 epochs, batch 128. Use for rapid iteration "
"and verifying the training pipeline works."

# ✗ Bad (too vague)
"Training dataset"
"Model weights"
"Quick config"
```

## Best Practices

1. **Use `builds()` with `populate_full_signature=True`** to expose all parameters
2. **Use `zen_partial=True`** for model functions that need runtime context
3. **Store related configs in the same group** for easy composition
4. **Use descriptive names** for stored configurations
5. **Set `working_dir`** for reproducible output locations
6. **Use `DatasetSpecConfig`** instead of building `DatasetSpec` directly for cleaner configs
7. **Use `AssetRIDConfig`** for consistent asset specification
8. **Define a model protocol** for consistent model interfaces across your project
9. **Always add descriptions** using `with_description()` for lists or `zen_meta` for builds

## Related Documentation

- [Running Models and Notebooks](running-models-and-notebooks.md) — Practical guide for project setup and CLI usage
- [CLI Reference](cli-reference.md) — All DerivaML command-line tools
- [Execution Configuration](execution-configuration.md) — Execution lifecycle and workflows
- [Using Jupyter Notebooks](notebooks.md) — Notebook reproducibility guidelines
- [Datasets](datasets.md)
- [Hydra-zen Documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)

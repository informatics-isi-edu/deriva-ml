# Configuration Groups

DerivaML uses five standard configuration groups. Each group needs at least a
default entry. This page details each group and how to customize it.

For the underlying configuration classes and advanced patterns, see the
[Hydra-zen Configuration Overview](overview.md).

## Connection Settings (`deriva_ml`)

Define how to connect to Deriva catalogs:

```python
# configs/deriva.py
from hydra_zen import builds, store
from deriva_ml import DerivaMLConfig

DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
deriva_store = store(group="deriva_ml")

# Development catalog
deriva_store(
    DerivaMLConf(hostname="dev.example.org", catalog_id="1"),
    name="default_deriva",
)

# Production catalog
deriva_store(
    DerivaMLConf(hostname="prod.example.org", catalog_id="100"),
    name="production",
)
```

See [DerivaMLConfig](overview.md#derivamlconfig) for all parameters.

## Datasets (`datasets`)

Specify which datasets to download for each workflow:

```python
# configs/datasets.py
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# Required: default (used when no override is specified)
datasets_store([], name="default_dataset")

# A named dataset collection
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="1-ABC", version="1.0.0")],
        "Training dataset with 1000 labeled images.",
    ),
    name="training_data",
)
```

See [DatasetSpecConfig](overview.md#datasetspecconfig) for options.

## Assets (`assets`)

List input asset RIDs (model weights, configuration files, etc.):

```python
# configs/assets.py
from hydra_zen import store
from deriva_ml.execution import with_description

asset_store = store(group="assets")

# Required: default
asset_store([], name="default_asset")

# Model weights
asset_store(
    with_description(
        ["6-EPNR"],
        "ResNet50 pretrained weights from MAE pre-training.",
    ),
    name="resnet_weights",
)
```

For caching support, use `AssetSpecConfig` instead of plain RID strings.
See [Configuration Descriptions](overview.md#configuration-descriptions)
for details on `with_description()`.

## Workflows (`workflow`)

Define the computational process being tracked:

```python
# configs/workflow.py
from hydra_zen import builds, store
from deriva_ml.execution import Workflow

workflow_store = store(group="workflow")

workflow_store(
    builds(Workflow, name="default", workflow_type="Training",
           populate_full_signature=True),
    name="default_workflow",
)

workflow_store(
    builds(Workflow, name="Feature Extraction", workflow_type="Preprocessing",
           description="Extract features from raw data",
           populate_full_signature=True),
    name="feature_extraction",
)
```

See [Running an experiment](../user-guide/executions.md) for how workflows
track source code provenance.

## Model Configuration (`model_config`)

Configure model hyperparameters. This is where `zen_partial=True` is essential:

```python
# configs/my_model.py
from hydra_zen import builds, store
from models.my_model import train_classifier

model_store = store(group="model_config")

# Base config: partially applied, waits for ml_instance and execution
ModelConfig = builds(
    train_classifier,
    learning_rate=1e-3,
    epochs=10,
    batch_size=32,
    populate_full_signature=True,
    zen_partial=True,
)

model_store(ModelConfig, name="default_model")
model_store(ModelConfig, name="quick", epochs=3, learning_rate=1e-2)
model_store(ModelConfig, name="long_training", epochs=100, learning_rate=1e-4)
```

See [Model Configuration with zen_partial](overview.md#model-configuration-with-zen_partial)
for the full pattern.

## See Also

- [Hydra-zen Configuration Overview](overview.md) — Configuration class reference
- [Experiments and Multiruns](experiments.md) — Preset experiment configurations
- [Notebook Configuration](notebooks.md) — Notebook-specific patterns

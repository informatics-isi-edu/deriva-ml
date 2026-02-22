# Running Models and Notebooks

DerivaML provides two command-line tools for executing reproducible ML workflows:

- **`deriva-ml-run`** — runs Python model functions
- **`deriva-ml-run-notebook`** — runs Jupyter notebooks

Both tools use [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)
for composable configuration and automatically track executions in the Deriva
catalog. This guide walks through setting up a project, writing configuration,
and running workflows.

For detailed configuration class documentation, see
[Hydra-zen Configuration](hydra-zen-configuration.md). For execution lifecycle
details, see [Configuring and Running Executions](execution-configuration.md).

## Project Layout

A DerivaML project follows this standard directory structure:

```
my-project/
  pyproject.toml
  src/
    configs/             # Hydra-zen configuration modules (Python, no YAML)
      __init__.py
      base.py            # create_model_config() + store registration
      deriva.py          # Connection settings (deriva_ml group)
      datasets.py        # Dataset specifications (datasets group)
      assets.py          # Input asset RIDs (assets group)
      workflow.py        # Workflow definitions (workflow group)
      my_model.py        # Model configs (model_config group)
      my_notebook.py     # Notebook configs (notebook_config())
      experiments.py     # Experiment presets (loaded last)
      multiruns.py       # Named multirun/sweep configs
    models/
      my_model.py        # Model function implementations
  notebooks/
    my_notebook.ipynb
```

A repository template is available at
[DerivaML Repository Template](https://github.com/informatics-isi-edu/deriva-ml-model-template).

### How Configuration Discovery Works

When you run `deriva-ml-run` or `deriva-ml-run-notebook`, the tool calls
`load_configs()` which uses `pkgutil.iter_modules()` to discover and import all
Python modules in the `configs/` package. Each module registers its configurations
with the hydra-zen store as a side effect of being imported.

Modules are loaded in alphabetical order, with one exception: `experiments.py` is
always loaded last because experiments typically reference configurations from
other modules.

## Setting Up Configuration Groups

DerivaML uses five standard configuration groups. Each group needs at least a
default entry.

### Connection Settings (`deriva_ml`)

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

See [DerivaMLConfig](hydra-zen-configuration.md#derivamlconfig) for all parameters.

### Datasets (`datasets`)

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

See [DatasetSpecConfig](hydra-zen-configuration.md#datasetspecconfig) for options.

### Assets (`assets`)

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
See [Configuration Descriptions](hydra-zen-configuration.md#configuration-descriptions)
for details on `with_description()`.

### Workflows (`workflow`)

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

See [Workflows](execution-configuration.md#workflows) for how workflows track
source code provenance.

### Model Configuration (`model_config`)

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

See [Model Configuration with zen_partial](hydra-zen-configuration.md#model-configuration-with-zen_partial)
for the full pattern.

## Writing a Model Function

Model functions follow a simple protocol: they accept any number of configurable
parameters plus two keyword arguments injected at runtime:

```python
# models/my_model.py
from deriva_ml import DerivaML
from deriva_ml.execution import Execution

def train_classifier(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    ml_instance: DerivaML,
    execution: Execution | None = None,
) -> None:
    """Train a classifier using DerivaML execution context."""
    # Access downloaded datasets
    for dataset in execution.datasets:
        bag = execution.download_dataset_bag(dataset)
        # Process data from the bag...

    # Access downloaded input assets (model weights, etc.)
    for table_name, asset_paths in execution.asset_paths.items():
        for asset_path in asset_paths:
            print(f"Loaded {table_name}: {asset_path}")

    # Your training code here
    model = build_model()
    for epoch in range(epochs):
        train_epoch(model, learning_rate, batch_size)

    # Register output files for upload
    model_path = execution.asset_file_path("Model", "best_model.pt")
    save_model(model, model_path)

    metrics_path = execution.asset_file_path("Execution_Metadata", "metrics.json")
    save_metrics(metrics_path)
```

The `ml_instance` and `execution` parameters are injected by `run_model()` at
runtime. Hydra configures the remaining parameters (`learning_rate`, `epochs`,
`batch_size`) from the config store.

## The Base Configuration (`base.py`)

The `base.py` module ties everything together using `create_model_config()`:

```python
# configs/base.py
from hydra_zen import store
from deriva_ml import DerivaML
from deriva_ml.execution import create_model_config

DerivaModelConfig = create_model_config(
    DerivaML,                         # or your DerivaML subclass
    description="My project model run",
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)

store(DerivaModelConfig, name="deriva_model")
```

`create_model_config()` creates a hydra-zen `builds()` of the `run_model()`
function with all the standard groups wired up. The `deriva_model` name must
match the `--config-name` argument (the default).

If your project uses a DerivaML subclass (e.g., `EyeAI`), pass it as the first
argument so that the correct class is instantiated at runtime.

## Running with `deriva-ml-run`

### Basic Usage

```bash
# Run with all defaults from base.py
uv run deriva-ml-run

# Override a config group
uv run deriva-ml-run model_config=quick datasets=training_data

# Override individual parameters
uv run deriva-ml-run model_config.epochs=50 model_config.learning_rate=0.001

# Dry run: download inputs but skip catalog writes
uv run deriva-ml-run dry_run=true

# Show all available configuration groups and options
uv run deriva-ml-run --info
```

### Overriding Host and Catalog

You can override the Deriva connection from the command line without changing
configs:

```bash
uv run deriva-ml-run --host prod.example.org --catalog 100
```

Or using Hydra override syntax:

```bash
uv run deriva-ml-run deriva_ml=production
```

### Experiments

Experiments are preset configurations that bundle specific model, dataset, asset,
and workflow choices. Define them in `configs/experiments.py`:

```python
# configs/experiments.py
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

experiment_store = store(group="experiment", package="_global_")

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},
            {"override /datasets": "training_data"},
            {"override /assets": "resnet_weights"},
            {"override /workflow": "default_workflow"},
        ],
        description="Quick training with ResNet weights on training data.",
        bases=(DerivaModelConfig,),
    ),
    name="quick_training",
)
```

Run an experiment:

```bash
uv run deriva-ml-run +experiment=quick_training

# Override an experiment parameter
uv run deriva-ml-run +experiment=quick_training model_config.epochs=25
```

Note the `+` prefix: this adds the experiment group, which is not in the default
config. The `override /` prefix in `hydra_defaults` ensures the experiment's
choices replace (rather than conflict with) the base defaults.

### Multiruns and Sweeps

For parameter sweeps, use Hydra's multirun mode:

```bash
# Sweep a parameter
uv run deriva-ml-run --multirun model_config.learning_rate=0.0001,0.001,0.01

# Sweep across experiments
uv run deriva-ml-run --multirun +experiment=quick_training,long_training
```

For complex sweeps, define named multirun configurations in
`configs/multiruns.py`:

```python
# configs/multiruns.py
from deriva_ml.execution import multirun_config

multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=quick_training",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description='''## Learning Rate Sweep

    Exploring optimal learning rates for quick training config.

    | Learning Rate | Expected Behavior |
    |--------------|-------------------|
    | 0.0001 | Slow convergence |
    | 0.001 | Standard baseline |
    | 0.01 | Fast, may overshoot |
    | 0.1 | Likely unstable |
    ''',
)
```

Run a named multirun:

```bash
# Use the named multirun config (automatically enables multirun mode)
uv run deriva-ml-run +multirun=lr_sweep

# Override parameters from the multirun config
uv run deriva-ml-run +multirun=lr_sweep model_config.epochs=5
```

Named multiruns create a parent-child execution structure in the catalog:

- **Parent execution**: Contains the markdown description and links to all children
- **Child executions**: One per parameter combination, each with full provenance

Use `list_parent_executions()` and `list_nested_executions()` to navigate this
hierarchy.

### Code Provenance

DerivaML records the Git commit hash and source URL for each execution. Always
commit your code before running models:

```bash
git add -A && git commit -m "Ready for training run"
uv run bump-version patch  # Optional: create a release tag
uv run deriva-ml-run +experiment=quick_training
```

If the working tree has uncommitted changes, DerivaML issues a warning and the
execution record may not have a valid code reference.

See [Automatic Source Code Detection](execution-configuration.md#automatic-source-code-detection)
for details on how provenance works in scripts, notebooks, and Docker containers.

## Notebook Configuration

Notebooks use a slightly different pattern: instead of `zen_partial` model
functions, they use `notebook_config()` and `run_notebook()`.

### Defining a Notebook Configuration

For simple notebooks that only use the standard fields (datasets, assets, etc.):

```python
# configs/my_analysis.py
from deriva_ml.execution import notebook_config

notebook_config(
    "my_analysis",
    defaults={"assets": "comparison_weights", "datasets": "training_data"},
)
```

For notebooks with custom parameters:

```python
# configs/my_analysis.py
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class MyAnalysisConfig(BaseConfig):
    threshold: float = 0.5
    num_samples: int = 100

notebook_config(
    "my_analysis",
    config_class=MyAnalysisConfig,
    defaults={"assets": "comparison_weights"},
)
```

### Using `run_notebook()` in the Notebook

In the first code cell of your notebook:

```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook("my_analysis")
```

This single call:

1. Loads all config modules from `src/configs/`
2. Resolves the hydra-zen configuration (including any CLI overrides)
3. Connects to the Deriva catalog
4. Creates a workflow and execution record
5. Downloads any specified datasets and assets

After that, you can use `ml`, `execution`, and `config` throughout the notebook:

```python
# Access resolved config values
print(config.threshold)  # 0.5

# Access downloaded assets
for table, paths in execution.asset_paths.items():
    for path in paths:
        print(f"Asset: {path}")

# At the end of the notebook
execution.upload_execution_outputs()
```

## Running with `deriva-ml-run-notebook`

### Basic Usage

```bash
# Run with default configuration
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# Override Hydra config groups (positional overrides)
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
    assets=different_weights deriva_ml=production
```

Hydra overrides are passed as positional arguments after the notebook path. These
are forwarded to the notebook via the `DERIVA_ML_HYDRA_OVERRIDES` environment
variable, where `run_notebook()` picks them up automatically.

### Parameter Injection

You can also inject values directly into a notebook's parameter cell (tagged with
`parameters` in Jupyter) using papermill:

```bash
# Inject individual parameters
uv run deriva-ml-run-notebook notebooks/train.ipynb \
    -p learning_rate 0.001 -p epochs 50

# Load parameters from a file
uv run deriva-ml-run-notebook notebooks/train.ipynb --file params.yaml
```

### Inspecting Available Options

```bash
# Show notebook parameters (from the papermill parameters cell)
uv run deriva-ml-run-notebook notebooks/train.ipynb --inspect

# Show available Hydra config groups
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info
```

### Kernel Selection

The CLI automatically detects the Jupyter kernel for the current virtual
environment. If auto-detection fails or you want to use a different kernel:

```bash
uv run deriva-ml-run-notebook notebooks/train.ipynb --kernel my-ml-project
```

If no kernel is installed for your environment, use
[`deriva-ml-install-kernel`](cli-reference.md#deriva-ml-install-kernel).

### Output Handling

After execution, `deriva-ml-run-notebook`:

1. Converts the executed notebook to Markdown (with images embedded as base64
   data URIs and DataFrame tables converted to Markdown format)
2. Uploads both the `.ipynb` and `.md` files to the catalog as execution assets
   with type `notebook_output`
3. Prints the execution URL for easy access

### Streaming Output

To see cell outputs in your terminal as the notebook runs:

```bash
uv run deriva-ml-run-notebook notebooks/train.ipynb --log-output
```

## Complete Walkthrough

Here is a minimal end-to-end example showing how to set up a new DerivaML
project and run a model.

### 1. Project Setup

```bash
# Create project from template
gh repo create my-ml-project --template informatics-isi-edu/deriva-ml-model-template

# Set up environment
cd my-ml-project
uv sync
uv run deriva-ml-install-kernel  # If using notebooks
```

### 2. Define Configurations

Create `src/configs/deriva.py`:

```python
from hydra_zen import builds, store
from deriva_ml import DerivaMLConfig

DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
store(DerivaMLConf(hostname="dev.example.org", catalog_id="1"),
      group="deriva_ml", name="default_deriva")
```

Create `src/configs/datasets.py`:

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

store([], group="datasets", name="default_dataset")
store([DatasetSpecConfig(rid="1-ABC", version="1.0.0")],
      group="datasets", name="training")
```

Create `src/configs/assets.py`:

```python
from hydra_zen import store
store([], group="assets", name="default_asset")
```

Create `src/configs/workflow.py`:

```python
from hydra_zen import builds, store
from deriva_ml.execution import Workflow

store(builds(Workflow, name="default", workflow_type="Training",
             populate_full_signature=True),
      group="workflow", name="default_workflow")
```

### 3. Write a Model Function

Create `src/models/classifier.py`:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import Execution

def train(
    learning_rate: float = 0.001,
    epochs: int = 10,
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    print(f"Training for {epochs} epochs at lr={learning_rate}")
    # Your training code here...
```

### 4. Register the Model Config

Create `src/configs/classifier.py`:

```python
from hydra_zen import builds, store
from models.classifier import train

store(builds(train, populate_full_signature=True, zen_partial=True),
      group="model_config", name="default_model")
store(builds(train, epochs=3, populate_full_signature=True, zen_partial=True),
      group="model_config", name="quick")
```

### 5. Create the Base Config

Create `src/configs/base.py`:

```python
from hydra_zen import store
from deriva_ml import DerivaML
from deriva_ml.execution import create_model_config

store(create_model_config(DerivaML), name="deriva_model")
```

### 6. Run

```bash
# Authenticate
uv run deriva-globus-auth-utils login --host dev.example.org

# Verify configs
uv run deriva-ml-run --info

# Dry run
uv run deriva-ml-run dry_run=true

# Commit and run for real
git add -A && git commit -m "Initial model setup"
uv run deriva-ml-run model_config=quick datasets=training
```

## Troubleshooting

**"Config directory not found"** — Ensure `src/configs/` exists and contains an
`__init__.py`. The `--config-dir` flag defaults to `src/configs`.

**"Uncommitted changes" warning** — Commit your code before running. DerivaML
tracks the Git commit hash for code provenance.

**"Kernel not found"** — Run `uv run deriva-ml-install-kernel` to install a
Jupyter kernel for the current environment. Verify with
`jupyter kernelspec list`.

**Override syntax errors** — Remember: `group=option` overrides a group in the
defaults list, `group.param=value` overrides a specific parameter, `+group=option`
adds a group not in the defaults (like `+experiment=...`).

**Notebook ignores Hydra overrides** — Make sure the notebook calls
`run_notebook("config_name")` (not `get_notebook_configuration()` directly).
Hydra overrides are passed via the `DERIVA_ML_HYDRA_OVERRIDES` environment
variable, which `run_notebook()` reads automatically.

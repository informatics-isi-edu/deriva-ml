# Running Models

DerivaML provides two command-line tools for executing reproducible ML workflows:

- **`deriva-ml-run`** — runs Python model functions
- **`deriva-ml-run-notebook`** — runs Jupyter notebooks

Both tools use [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)
for composable configuration and automatically track executions in the Deriva
catalog. This page covers model function setup and the `deriva-ml-run` CLI.
For notebook-specific patterns, see
[Notebook Configuration](../configuration/notebooks.md).

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

## See Also

- [Configuration Groups](../configuration/groups.md) — Setting up the five standard groups
- [Experiments and Multiruns](../configuration/experiments.md) — Preset configurations and sweeps
- [Notebook Configuration](../configuration/notebooks.md) — Notebook-specific patterns
- [Execution Lifecycle](execution-lifecycle.md) — How executions are created and tracked
- [CLI Reference](../cli-reference.md) — Full CLI documentation

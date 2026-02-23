# Notebook Configuration

Notebooks use a slightly different pattern from model functions: instead of
`zen_partial` model configs, they use `notebook_config()` and `run_notebook()`.

For the model function pattern, see
[Configuration Groups — Model Configuration](groups.md#model-configuration-model_config).

## Defining a Notebook Configuration

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

## Using `run_notebook()` in the Notebook

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
[`deriva-ml-install-kernel`](../cli-reference.md#deriva-ml-install-kernel).

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

## See Also

- [Configuration Groups](groups.md) — The five standard config groups
- [Hydra-zen Configuration Overview](overview.md) — Configuration class reference
- [CLI Reference](../cli-reference.md) — Full CLI documentation

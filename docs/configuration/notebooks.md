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

### What `run_notebook()` Does Internally

That single call handles a multi-step initialization sequence:

1. **Loads all config modules** from the config package (default: `"configs"`). This
   imports every Python module under `src/configs/`, which registers their
   hydra-zen configurations with the store as a side effect. The `experiments`
   module is loaded last since it typically depends on other configs.

2. **Resolves Hydra configuration** via `get_notebook_configuration()`. This launches
   hydra-zen to compose the full config from defaults, config groups, and any
   overrides (including overrides passed through the `DERIVA_ML_HYDRA_OVERRIDES`
   environment variable when running from the CLI). The result is a fully
   instantiated config object with all defaults merged.

3. **Creates a DerivaML instance** connected to the catalog specified by
   `config.deriva_ml.hostname` and `config.deriva_ml.catalog_id`. If Hydra
   captured a runtime output directory during config resolution, it is passed
   along so the Hydra YAML files can be uploaded with the execution later.

4. **Validates all RIDs** referenced in the config (datasets and assets) exist in
   the catalog. If any RID is missing, a `DerivaMLException` is raised with
   details. Warnings (e.g., datasets missing descriptions) are logged but do
   not block execution.

5. **Creates a workflow** using `ml.create_workflow()`. The workflow name defaults
   to a title-cased version of the config name (e.g., `"roc_analysis"` becomes
   `"Roc Analysis"`). The workflow type defaults to `"Analysis Notebook"`.

6. **Creates an ExecutionConfiguration** from the resolved config, bundling the
   workflow, datasets, assets, and description.

7. **Creates and starts an Execution** context. This creates the execution record
   in the catalog and downloads all input datasets and assets to local storage.
   The downloaded assets are accessible through `execution.asset_paths`.

8. **Returns** the tuple `(ml, execution, config)`.

### Notebook Responsibilities

The notebook is responsible for using the returned objects to do its work and
for cleaning up when finished:

- **Use the execution context for provenance.** Any outputs (plots, CSVs, model
  weights) should be registered with `execution.asset_file_path()` so they are
  tracked as execution assets.

- **Call `execution.upload_execution_outputs()` at the end.** This uploads all
  registered output files to the catalog. If you skip this call, your outputs
  will not be recorded.

- **The execution context is not automatically cleaned up.** Unlike `run_model()`
  which manages the full lifecycle, `run_notebook()` leaves the execution open
  for the notebook to use interactively. The notebook must explicitly upload
  outputs when done.

## Running with `deriva-ml-run-notebook`

The CLI wraps notebook execution with additional infrastructure that goes beyond
what `run_notebook()` provides inside the notebook itself:

1. **Executes the notebook with papermill.** The notebook is run non-interactively
   as a subprocess, with parameter injection from `-p` flags or config files.

2. **Sets environment variables for workflow provenance.** Before execution, the
   CLI sets `DERIVA_ML_WORKFLOW_URL` (the Git source URL for the notebook),
   `DERIVA_ML_WORKFLOW_CHECKSUM` (MD5 of the notebook file),
   `DERIVA_ML_NOTEBOOK_PATH` (local filesystem path), and
   `DERIVA_ML_SAVE_EXECUTION_RID` (path where the notebook should write its
   execution metadata). These allow the CLI to reconnect to the execution after
   the notebook finishes.

3. **Forwards Hydra overrides.** Positional arguments after the notebook path
   (e.g., `assets=different_weights`) are serialized to the
   `DERIVA_ML_HYDRA_OVERRIDES` environment variable as JSON, where
   `run_notebook()` picks them up during config resolution.

4. **Handles kernel detection.** The CLI searches installed Jupyter kernels for
   one whose Python executable matches the current virtual environment. You can
   also specify a kernel explicitly with `--kernel`.

5. **Converts the executed notebook to Markdown.** After execution, the CLI
   converts the `.ipynb` to `.md` with images embedded as base64 data URIs and
   DataFrame tables converted to Markdown format, producing a self-contained
   archival document.

6. **Uploads notebook outputs as execution assets.** Both the executed `.ipynb`
   and the `.md` conversion are uploaded to the catalog as execution assets with
   type `notebook_output`. This happens on top of any outputs the notebook
   itself uploaded via `execution.upload_execution_outputs()`.

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
[`deriva-ml-install-kernel`](../user-guide/hydra-zen.md).

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
- [Integrating with hydra-zen](../user-guide/hydra-zen.md) — Full CLI documentation

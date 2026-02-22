# CLI Reference

DerivaML provides several command-line tools for running ML workflows, managing
versions, and administering catalogs. All commands are installed as console scripts
when you install the `deriva-ml` package.

When using a project managed with `uv`, prefix commands with `uv run`:

```bash
uv run deriva-ml-run +experiment=my_experiment
```

## Command Overview

| Command | Description |
|---------|-------------|
| [`deriva-ml-run`](#deriva-ml-run) | Execute ML models with Hydra configuration and execution tracking |
| [`deriva-ml-run-notebook`](#deriva-ml-run-notebook) | Execute Jupyter notebooks with parameter injection and tracking |
| [`bump-version`](#bump-version) | Bump semantic version tags and push to remote |
| [`deriva-ml-install-kernel`](#deriva-ml-install-kernel) | Install a Jupyter kernel for the current virtual environment |
| [`deriva-ml-split-dataset`](#deriva-ml-split-dataset) | Split a dataset into training and testing subsets |
| [`deriva-ml-create-schema`](#deriva-ml-create-schema) | Create the DerivaML schema in a catalog |
| [`deriva-ml-check-catalog-schema`](#deriva-ml-check-catalog-schema) | Validate a catalog's schema against DerivaML requirements |
| [`deriva-ml-table-comments-utils`](#deriva-ml-table-comments-utils) | Update table and column comments from documentation files |
| [`create-demo-catalog`](#create-demo-catalog) | Create a demo catalog with sample data for testing |

---

## Model and Notebook Execution

### deriva-ml-run

Execute ML models with Hydra-zen configuration composition and automatic execution
tracking in a Deriva catalog.

**Synopsis:**

```
deriva-ml-run [--host HOST] [--catalog CATALOG] [--config-dir DIR]
              [--config-name NAME] [--info] [--multirun|-m] [OVERRIDES...]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--host HOST` | *(from config)* | Deriva server hostname |
| `--catalog CATALOG` | *(from config)* | Catalog ID or identifier |
| `--config-dir DIR`, `-c` | `src/configs` | Path to the configs directory |
| `--config-name NAME` | `deriva_model` | Name of the main Hydra-zen config |
| `--info` | | Display available configuration groups and options |
| `--multirun`, `-m` | | Enable Hydra multirun mode for parameter sweeps |
| `OVERRIDES` | | Hydra-zen configuration overrides (positional) |

**Examples:**

```bash
# Run with default configuration
uv run deriva-ml-run

# Override a config group
uv run deriva-ml-run model_config=my_model datasets=full_training

# Override individual parameters
uv run deriva-ml-run model_config.epochs=50 model_config.learning_rate=0.001

# Use an experiment preset
uv run deriva-ml-run +experiment=cifar10_quick

# Dry run (download inputs, skip catalog writes)
uv run deriva-ml-run dry_run=true

# Show all available configs
uv run deriva-ml-run --info

# Override host and catalog from command line
uv run deriva-ml-run --host prod.example.org --catalog 100

# Multirun with comma-separated values
uv run deriva-ml-run --multirun model_config.learning_rate=0.0001,0.001,0.01

# Named multirun configuration
uv run deriva-ml-run +multirun=lr_sweep

# Named multirun with additional overrides
uv run deriva-ml-run +multirun=lr_sweep model_config.epochs=5
```

**See also:** [Running Models and Notebooks](running-models-and-notebooks.md)

---

### deriva-ml-run-notebook

Execute Jupyter notebooks with parameter injection, automatic kernel detection,
and execution tracking. The executed notebook and a Markdown conversion are uploaded
to the catalog as execution assets.

**Synopsis:**

```
deriva-ml-run-notebook NOTEBOOK [--host HOST] [--catalog CATALOG]
                       [--file FILE] [--parameter KEY VALUE]
                       [--kernel KERNEL] [--inspect] [--info]
                       [--log-output] [OVERRIDES...]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `NOTEBOOK` | *(required)* | Path to the `.ipynb` notebook file |
| `--host HOST` | *(from config)* | Deriva server hostname |
| `--catalog CATALOG` | *(from config)* | Catalog ID or identifier |
| `--file FILE`, `-f` | | JSON or YAML file with parameter values |
| `--parameter KEY VALUE`, `-p` | | Parameter name and value to inject (repeatable) |
| `--kernel KERNEL`, `-k` | *(auto-detected)* | Jupyter kernel name |
| `--inspect` | | Display notebook parameters and exit |
| `--info` | | Display available Hydra configuration groups |
| `--log-output` | | Stream cell outputs during execution |
| `OVERRIDES` | | Hydra-zen configuration overrides (positional) |

**Environment Variables Set During Execution:**

| Variable | Purpose |
|----------|---------|
| `DERIVA_ML_WORKFLOW_URL` | Git URL or local path to the notebook source |
| `DERIVA_ML_WORKFLOW_CHECKSUM` | MD5 checksum of the notebook file |
| `DERIVA_ML_NOTEBOOK_PATH` | Absolute filesystem path to the notebook |
| `DERIVA_ML_SAVE_EXECUTION_RID` | Path where the notebook saves execution metadata |
| `DERIVA_ML_HYDRA_OVERRIDES` | JSON-encoded list of Hydra overrides |

**Examples:**

```bash
# Run a notebook with default configuration
uv run deriva-ml-run-notebook notebooks/analyze_results.ipynb

# Override Hydra config groups (positional overrides)
uv run deriva-ml-run-notebook notebooks/analysis.ipynb \
    assets=my_assets deriva_ml=production

# Inject parameters into the notebook's parameter cell
uv run deriva-ml-run-notebook notebooks/train.ipynb \
    -p learning_rate 0.001 -p epochs 50

# Load parameters from a YAML file
uv run deriva-ml-run-notebook notebooks/train.ipynb --file params.yaml

# Inspect available notebook parameters without running
uv run deriva-ml-run-notebook notebooks/train.ipynb --inspect

# Show available Hydra config groups
uv run deriva-ml-run-notebook notebooks/analysis.ipynb --info

# Stream notebook output to terminal
uv run deriva-ml-run-notebook notebooks/train.ipynb --log-output

# Override host and catalog
uv run deriva-ml-run-notebook notebooks/analysis.ipynb \
    --host prod.example.org --catalog 100
```

**See also:** [Running Models and Notebooks](running-models-and-notebooks.md),
[Using Jupyter Notebooks](notebooks.md)

---

## Development Tools

### bump-version

Manage semantic version tags for your project. Creates an initial tag if none
exists, or bumps the existing version using
[bump-my-version](https://github.com/callowayproject/bump-my-version).

This tool works with [setuptools_scm](https://github.com/pypa/setuptools_scm)
for dynamic version derivation from git tags — there is no hardcoded version
string in the source code.

**Synopsis:**

```
bump-version [patch|minor|major]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `patch\|minor\|major` | `patch` | Which semantic version component to increment |

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `START` | `0.1.0` | Initial version if no tag exists |
| `PREFIX` | `v` | Tag prefix (e.g., `v` for tags like `v1.2.3`) |

**How Versioning Works:**

- **At a tag**: Version is clean, e.g., `1.2.3`
- **After a tag**: Includes distance and commit hash, e.g., `1.2.3.post2+g1234abc`
- **Dirty working tree**: Adds `.dirty` suffix

**Examples:**

```bash
# Bump patch version (1.2.3 -> 1.2.4)
uv run bump-version

# Bump minor version (1.2.3 -> 1.3.0)
uv run bump-version minor

# Bump major version (1.2.3 -> 2.0.0)
uv run bump-version major

# Check current version
uv run python -m setuptools_scm
```

**Requirements:** `git`, `uv`, and `bump-my-version` configured in `pyproject.toml`.

---

### deriva-ml-install-kernel

Install a Jupyter kernel for the current virtual environment. This allows
Jupyter notebooks to use the DerivaML environment with all its dependencies.

**Synopsis:**

```
deriva-ml-install-kernel [--install-local]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--install-local` | Install kernel to the venv's prefix directory instead of the user's Jupyter data directory |

The kernel name and display name are derived from the virtual environment's
`prompt` setting in `pyvenv.cfg`.

**Example Workflow:**

```bash
# Create virtual environment with a name
uv venv --prompt my-ml-project

# Activate it
source .venv/bin/activate

# Install the Jupyter kernel
uv run deriva-ml-install-kernel
# Output: Installed Jupyter kernel 'my-ml-project' with display name 'Python (my-ml-project)'

# The kernel now appears in Jupyter's kernel selector
jupyter lab
```

**Kernel location:** `~/.local/share/jupyter/kernels/` (Linux/macOS)
or `%APPDATA%\jupyter\kernels\` (Windows).

---

## Data Operations

### deriva-ml-split-dataset

Split a DerivaML dataset into training and testing subsets. Follows scikit-learn
conventions for split parameters and supports stratified splitting.

**Synopsis:**

```
deriva-ml-split-dataset --hostname HOST --catalog-id ID --dataset-rid RID
                        [--test-size SIZE] [--train-size SIZE] [--seed SEED]
                        [--stratify-by-column COL] [--element-table TABLE]
                        [--include-tables TABLES] [--training-types TYPES]
                        [--testing-types TYPES] [--description DESC]
                        [--workflow-type TYPE] [--dry-run] [--show-urls]
                        [--no-shuffle]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hostname` | *(required)* | Deriva server hostname |
| `--catalog-id` | *(required)* | Catalog ID to connect to |
| `--dataset-rid` | *(required)* | RID of the source dataset to split |
| `--domain-schema` | *(auto-detected)* | Domain schema name |
| `--test-size` | `0.2` | Test set size as fraction (0-1) or absolute count |
| `--train-size` | *(complement)* | Train set size as fraction (0-1) or absolute count |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-shuffle` | | Do not shuffle before splitting |
| `--stratify-by-column` | | Column name for stratified splitting (requires `--include-tables`) |
| `--element-table` | *(auto-detected)* | Element table to split (e.g., `Image`) |
| `--include-tables` | | Comma-separated tables for denormalization (required for stratified splitting) |
| `--training-types` | `Labeled` | Comma-separated dataset types for the training set |
| `--testing-types` | `Labeled` | Comma-separated dataset types for the testing set |
| `--description` | | Description for the parent split dataset |
| `--workflow-type` | `Dataset_Split` | Workflow type vocabulary term |
| `--dry-run` | | Print the split plan without modifying the catalog |
| `--show-urls` | | Show Chaise web interface URLs for created datasets |

**Examples:**

```bash
# Simple random 80/20 split
uv run deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0

# Stratified split by class label
uv run deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 \
    --stratify-by-column Image_Classification_Image_Class \
    --include-tables Image,Image_Classification

# Fixed-count split
uv run deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --train-size 400 --test-size 100

# Dry run (show plan without modifying catalog)
uv run deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --dry-run
```

---

### create-demo-catalog

Create a demonstration catalog with sample data for testing and development.

**Synopsis:**

```
create-demo-catalog --host HOST [--domain-schema SCHEMA]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | *(required)* | Deriva server hostname |
| `--domain-schema` | `demo-schema` | Name for the domain schema |

This command is primarily used for development and testing of DerivaML itself.

---

## Catalog Administration

### deriva-ml-create-schema

Create the DerivaML schema in a Deriva catalog. This is typically run once when
setting up a new catalog for ML workflows.

**Synopsis:**

```
deriva-ml-create-schema HOSTNAME PROJECT_NAME SCHEMA_NAME --curie_prefix PREFIX
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `HOSTNAME` | *(required)* | Deriva server hostname |
| `PROJECT_NAME` | *(required)* | Project name for the catalog |
| `SCHEMA_NAME` | `deriva-ml` | Schema name |
| `--curie_prefix` | *(required)* | CURIE prefix for identifiers |

---

### deriva-ml-check-catalog-schema

Validate a catalog's schema against the DerivaML reference schema. Reports
any missing tables, columns, or configuration issues.

**Synopsis:**

```
deriva-ml-check-catalog-schema --host HOST [--catalog CATALOG] [--dump]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | *(required)* | Deriva server hostname |
| `--catalog` | `1` | Catalog number |
| `--dump` | | Dump schema details |

---

### deriva-ml-table-comments-utils

Update table and column comments in a catalog from file-based documentation.
This is an administrative utility for maintaining schema documentation.

**Synopsis:**

```
deriva-ml-table-comments-utils --host HOST [--catalog CATALOG]
```

This command uses Deriva's `BaseCLI` for standard host/catalog arguments.

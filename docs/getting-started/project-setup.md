# Project Setup

This guide covers how DerivaML projects are organized and how to set up a new
one from the repository template.

## Creating a Project

The fastest way to start is from the
[DerivaML Repository Template](https://github.com/informatics-isi-edu/deriva-ml-model-template):

1. Click **Use this template** > **Create a new repository** on GitHub
2. Clone your new repository and install dependencies:

```bash
git clone https://github.com/your-org/your-project.git
cd your-project
uv sync
```

The template provides a working project structure with example configurations,
a sample model, and a sample notebook.

## Project Structure

A DerivaML project follows this layout:

```
my-project/
  pyproject.toml          # Dependencies and project metadata
  src/
    configs/              # Hydra-zen configuration (Python, no YAML)
      __init__.py
      base.py             # create_model_config() + store registration
      deriva.py           # Catalog connection settings
      datasets.py         # Dataset specifications
      assets.py           # Input asset RIDs
      workflow.py          # Workflow definitions
      my_model.py         # Model hyperparameters
      my_notebook.py      # Notebook configurations
      experiments.py      # Experiment presets (loaded last)
      multiruns.py        # Named multirun/sweep configs
    models/
      my_model.py         # Model function implementations
  notebooks/
    my_notebook.ipynb
```

### Key Directories

- **`src/configs/`** — All configuration is Python-first using hydra-zen.
  Config modules are auto-discovered at runtime. See
  [Configuration Groups](../configuration/groups.md) for details on each group.

- **`src/models/`** — Model function implementations. Each model accepts
  configurable parameters plus `ml_instance` and `execution` injected at
  runtime. See [Running Models](../workflows/running-models.md#writing-a-model-function).

- **`notebooks/`** — Jupyter notebooks that use `run_notebook()` for
  DerivaML integration. See
  [Notebook Configuration](../configuration/notebooks.md).

## DerivaML Subclasses

Projects that work with domain-specific catalogs typically create a subclass
of `DerivaML` that adds domain-specific functionality. For example:

```python
from deriva_ml import DerivaML

class EyeAI(DerivaML):
    """EyeAI-specific catalog operations."""

    def get_oct_images(self, dataset_rid: str):
        # Domain-specific data access...
        pass
```

Pass your subclass to `create_model_config()` in `base.py` so the correct
class is instantiated at runtime:

```python
from deriva_ml.execution import create_model_config
from my_project import EyeAI

DerivaModelConfig = create_model_config(EyeAI)
```

## Optional Dependency Groups

The template uses uv dependency groups for optional features:

```bash
# Jupyter notebook support
uv sync --group=jupyter

# PyTorch
uv sync --group=pytorch

# Documentation building
uv sync --group=docs
```

To always install certain groups, add them to `default-groups` in
`pyproject.toml`:

```toml
[tool.uv]
default-groups = ["dev", "jupyter"]
```

## Notebook Setup

For notebook development:

```bash
# Install Jupyter support
uv sync --group=jupyter

# Auto-strip output cells on commit (run once per repo)
uv run nbstripout --install

# Register a Jupyter kernel for this environment
uv run deriva-ml-install-kernel

# Verify
uv run jupyter kernelspec list
```

**Important:** Always strip notebook outputs before committing. The
`nbstripout` hook handles this automatically after the one-time install above.

## Updating Dependencies

```bash
# Update a specific package
uv sync --upgrade-package deriva-ml

# Update all packages (regenerate lock file)
uv lock --upgrade && uv sync
```

Always commit `uv.lock` after updating to ensure reproducible environments.

## Next Steps

- [Configuration Groups](../configuration/groups.md) — Set up the five standard
  config groups
- [Running Models](../workflows/running-models.md) — Write and run model
  functions
- [Git Workflow and Versioning](../workflows/git-and-versioning.md) — Code
  provenance practices

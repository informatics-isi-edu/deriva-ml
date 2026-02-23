# Quick Start

Get a DerivaML project running in five steps.

## Step 1: Install uv

If you haven't already, install the [uv](https://docs.astral.sh/uv/) package
manager:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Step 2: Create a Project

Use the DerivaML repository template:

1. Go to [deriva-ml-model-template](https://github.com/informatics-isi-edu/deriva-ml-model-template)
2. Click **Use this template** > **Create a new repository**
3. Clone your new repository:

```bash
git clone https://github.com/your-org/your-model-repo.git
cd your-model-repo
```

Set up the environment:

```bash
# Install dependencies
uv sync

# For notebook support
uv sync --group=jupyter
uv run nbstripout --install
uv run deriva-ml-install-kernel
```

## Step 3: Authenticate

```bash
uv run deriva-globus-auth-utils login --host dev.example.org
```

This opens a browser for Globus authentication. Credentials are cached locally.

## Step 4: Run a Model

```bash
# Show available configurations
uv run deriva-ml-run --info

# Dry run (downloads inputs, skips catalog writes)
uv run deriva-ml-run dry_run=true

# Run with defaults
uv run deriva-ml-run

# Override a config group
uv run deriva-ml-run model_config=quick
```

## Step 5: Run a Notebook

```bash
# Run a notebook and upload results to the catalog
uv run deriva-ml-run-notebook notebooks/my_notebook.ipynb

# With Hydra config overrides
uv run deriva-ml-run-notebook notebooks/my_notebook.ipynb \
    assets=my_weights deriva_ml=production
```

## Next Steps

- [Installation](install.md) — Detailed installation and environment setup
- [Project Setup](project-setup.md) — Project structure and configuration
- [Configuration Overview](../configuration/overview.md) — The hydra-zen
  configuration system
- [Running Models](../workflows/running-models.md) — Complete model setup
  walkthrough

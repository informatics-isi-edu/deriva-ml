# Installing deriva-ml

DerivaML is a Python package for building reproducible ML workflows on the
Deriva platform. It includes the core library, command-line tools, and
Jupyter notebook support.

## Prerequisites

DerivaML requires **Python 3.12 or later** and uses
[uv](https://docs.astral.sh/uv/) as its package and project manager.

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installing in a Project (Recommended)

The recommended way to use DerivaML is as a dependency in a uv-managed project.
Add it to your project's `pyproject.toml`:

```bash
uv add deriva-ml
```

Then install the project and all dependencies:

```bash
uv sync
```

To include Jupyter notebook support:

```bash
uv sync --group=jupyter
```

A project template is available at
[deriva-ml-model-template](https://github.com/informatics-isi-edu/deriva-ml-model-template)
to get started quickly.

## Installing Standalone

For quick experiments or interactive use outside of a project, you can install
DerivaML into a standalone virtual environment:

```bash
uv pip install deriva-ml
```

## Authentication

DerivaML uses Globus authentication to connect to Deriva catalogs. After
installation, log in to your target host:

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

For example:

```bash
uv run deriva-globus-auth-utils login --host www.eye-ai.org
```

## Jupyter Kernel

To use DerivaML in Jupyter notebooks, install a dedicated kernel that runs
in your project's virtual environment:

```bash
uv run deriva-ml-install-kernel
```

This ensures notebooks use the same package versions as your project.
Auto-stripping notebook outputs on commit is also recommended:

```bash
uv run nbstripout --install
```

## Checking the Installed Version

DerivaML uses semantic versioning. To check the installed version:

```bash
uv run python -c "from importlib.metadata import version; print(version('deriva-ml'))"
```

To upgrade to the latest version:

```bash
uv add --upgrade deriva-ml
```

Or, if pinned to a git source in `pyproject.toml`, update the lock file:

```bash
uv lock --upgrade-package deriva-ml && uv sync
```

## Verifying the Installation

Once installed, verify that DerivaML can be imported:

```python
from deriva_ml import DerivaML, DerivaMLConfig, ExecutionConfiguration, Workflow
```

In most projects, you won't use `DerivaML` directly. Instead, it serves as a
base class for domain-specific libraries. For example:

```python
from eye_ai import EyeAI
from deriva_ml import ExecutionConfiguration, Workflow
```

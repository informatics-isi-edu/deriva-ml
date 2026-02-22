# Using Jupyter Notebooks With DerivaML

DerivaML can be used to execute Jupyter notebooks in a reproducible and structured manner.
Although DerivaML provides numerous tools to support reproducible machine learning, users must adopt and maintain standardized development practices to fully benefit from these tools.

In general, achieving reproducibility with Jupyter notebooks will require some discipline on the part of the developer.
For an amusing take on some of the challenges associated with Jupyter notebooks, the following presentation is very helpful:
[I Don't Like Notebooks](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g3d168d2fd3_0_72)

To ensure that your Jupyter notebooks are reproducible, follow these recommended guidelines:

## Version Control and Semantic Versioning

Always store your notebook in a GitHub repository. A repository template for DerivaML projects can be found at [DerivaML Repository Template](https://github.com/informatics-isi-edu/deriva-ml-model-template).
To use a GitHub template select the *Use This Template* dropdown from the GitHub user interface, rather than clone.
The template contains examples of both a DerivaML Python script and Jupyter notebook.

Adopt [semantic versioning](https://semver.org) for your notebooks.
In addition to semantic versions, Git tags are also quite helpful.
The repository template provides a command to simplify managing version numbers and tags:

```bash
uv run bump-version major|minor|patch
```

See [`bump-version`](cli-reference.md#bump-version) for details.

## Clearing Notebook Outputs

Normal operation of a Jupyter notebook puts results in output cells in the notebook, modifying the notebook file and complicating reproducibility.
For this reason, we recommend that before committing a notebook to Git, to clear all output cells, ensuring that only code and markdown cells are version-controlled.

While you can always clear output cells manually from the notebook, DerivaML includes a script which automatically strips output cells upon commit.
To set this up, execute the following command once in your repository:

```bash
nbstripout --install
```
You only need to perform the install once per repository, and after that, the notebook output will be stripped before every commit.

## Setting Notebook Parameters

Another challenge for reproducibility is that the behavior of cells in a notebook is often modified by changing the values of global variables assigned in a code cell.
In order to impose some order on this potentially chaotic process, DerivaML adopts the use of [Papermill](https://papermill.readthedocs.io) to help manage configuring notebooks prior to execution.
The basic idea behind Papermill is to place all of the configuration variables for a notebook in a single cell, and then provide an interface that will substitute values in for those variables and run the notebook in its entirety.

To use Papermill in DerivaML:
- Define all configurable variables in a single "parameters" cell located immediately after your imports at the top of your notebook. The contents of this cell can be automatically updated when the notebook is executed.
For Papermill to work, you must have a Jupyter cell tagged with `parameters` to indicate which cell contains parameter values. The DerivaML template already has this cell tagged. See [Papermill](https://papermill.readthedocs.io/en/latest/usage-parameterize.html) for instructions on how to do this.
- The parameters cell should contain only comments and variable assignments. It is recommended to include type hints for clarity and usability.
- Avoid setting configuration variables elsewhere in your notebook.

## Notebook Structure and Execution Flow

The overall workflow supported by DerivaML is a phase in which notebooks are developed and debugged, followed by an experimental phase in which multiple model parameters might be evaluated, or alternative approaches explored.
The boundary between debugging and experimentation can be fuzzy; in general it is better to err on the side of considering a run of a notebook to be an experiment rather than debugging.

The following guidelines can help facilitate notebook reproducibility:
- Structure your notebook so that it runs sequentially, from the first to the last cell.
- Regularly restart the kernel, clear outputs, and execute all cells sequentially to confirm reproducibility.
- Keep each notebook focused on a single task; avoid combining multiple tasks within one notebook.
- Utilize the `dry_run` mode during debugging to avoid cluttering the catalog with unnecessary execution records.

Use `dry_run` only for debugging, not during model tuning, as recording all tuning attempts is crucial for transparency and reproducibility.

## Commit and Tagging Procedures

After validating your notebook, commit it and generate a corresponding version tag:

```bash
git add -A && git commit -m "Notebook ready for execution"
uv run bump-version patch
```

## Configuring Notebooks with Hydra

DerivaML provides `notebook_config()` and `run_notebook()` for integrating
notebooks with the hydra-zen configuration system. This allows notebooks to
use the same configuration groups (datasets, assets, etc.) as model scripts.

For the full guide on setting up notebook configurations and running them with
`deriva-ml-run-notebook`, see
[Running Models and Notebooks](running-models-and-notebooks.md#notebook-configuration).

### Quick Example

In `src/configs/my_analysis.py`:

```python
from deriva_ml.execution import notebook_config

notebook_config(
    "my_analysis",
    defaults={"assets": "comparison_weights", "datasets": "training_data"},
)
```

In the first code cell of `notebooks/my_analysis.ipynb`:

```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook("my_analysis")
```

Running the notebook:

```bash
# With default configuration
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# With Hydra overrides
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
    assets=different_weights deriva_ml=production
```

## Executing Notebooks with DerivaML

A reproducible notebook execution has these components:

1. A committed notebook file is specified.
2. Per-execution specific values for variables in the `parameters` cell are specified and a new cell with the specified parameter values is injected into the notebook.
3. The modified notebook is executed in its entirety, including the uploading of any notebook-generated assets.
4. On conclusion of the notebook execution, the resulting notebook file (including output cells) and a Markdown conversion are uploaded into the DerivaML catalog as execution assets.

DerivaML includes the `deriva-ml-run-notebook` command to conveniently execute notebooks, substitute parameters dynamically, and store the execution results as assets:

```bash
# Parameter injection
uv run deriva-ml-run-notebook notebooks/my-notebook.ipynb \
    -p parameter1 value1 -p parameter2 value2

# Hydra config overrides
uv run deriva-ml-run-notebook notebooks/my-notebook.ipynb \
    assets=my_assets deriva_ml=production

# Inspect available parameters
uv run deriva-ml-run-notebook notebooks/my-notebook.ipynb --inspect
```

Parameters can also be specified via a JSON or YAML configuration file using the `--file filename` option.

For the complete CLI reference, see [`deriva-ml-run-notebook`](cli-reference.md#deriva-ml-run-notebook).

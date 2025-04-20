# Using Jupyter Notebooks With DerivaML

DerivaML can be used to execute Jupyter notebooks in a reproducible and structured manner. 
Although DerivaML provides numerous tools to support reproducible machine learning, users must adopt and maintain standardized development practices to fully benefit from these tools.

In general, achieving reproducablity with Jupyter notebooks will require some disiplane on the part of the developer.  
For an amuzing take on some of the chalages assocated with Jupyter notebooks, the follow presentation is very helpful:
[I Don't Like Notebooks](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g3d168d2fd3_0_72)

To ensure that your Jupyter notebooks are reproducible, follow these recommended guidelines:

## Version Control and Semantic Versioning

Always store your notebook in a GitHub repository. A repository template for DerivaML project can be found at [DerivaML Repository Template](https://github.com/informatics-isi-edu/deriva-ml-model-template).
To use a GitHub template select the *Use This Template* dropdown from the GitHub user interface, rather than clone.
The template contains examples of both a DerivaML Python script and Jupyter notebook.

Adopt [semantic versioning](https://semver.org) for your notebooks. 
In addition to semantic versions, Git tags are also quite helpful.
The repository templates provides a script to simplify managing version numbers and tags.

```bash
bumpversion major|minor|patch
```

## Clearing Notebook Outputs

Normal operation of a Jupyter notebook puts results in output cells in the notebook, modifying the notebook file and complicating reproducablity.
For this reason, we recommend that before committing a notebook to Git, to clear all output cells, ensuring that only code and markdown cells are version-controlled.

While you can always clear output cells manaally from the notebook, DerivaML includes a script which automatically strips output cells upon commit. 
To set this up, execute the following command once in your repository:

```bash
nboutputstrip --install
```
You only need to perform the install once per repository, and after that, the notebook output will be striped before every commit.

## Setting Notebook Parameters

Another challange for reproducibility is that the behavior of cells in a notebook is often modifified by changing the values of global variables assigned in a code cell.
In order to impose some order on this potentially chaotic process, DerivaML adopts the use of [Papermill](https://papermill.readthedocs.io) to help manage configuring notebooks prior to execution.
The basic idea behind Papermill is to place all of the configuration variables for a notebook in a single cell, and then provide and interface that will substitute values in for those variables and run the notebook in its entirety.

To use Papermill in DerivaML:
- Define all configurable variables in a single "parameters" cell located immediately after your imports at the top of your notebook. The contents of this cell can be automatically updated when the notebook is executed. 
For Papermill to work, you must have a Jupyter cell tagged with `parameters` to indicate which cell contains parameter values. The DerivaML template already has this cell tagged. See [Papermill](https://papermill.readthedocs.io/en/latest/usage-parameterize.html) for instructions on how to do this.  .
- The parameters cell should contain only comments and variable assignments.  It is recommended to include type hints for clarity and usability.
- Avoid setting configuration variables elsewhere in your notebook.

## Notebook Structure and Execution Flow

The overall workflow supported by DerivaML is a phase in which notebooks are developed and debugged, followed by an experimental phase in which multiple model parameters might be evauated, or alternative approches explored.
The boundary between debugging and experimentation can be fuzzy, in general it is better to err on the side of considering a run of a notebook to be an experiment rather than debugging.

The following guidelines can help facilitate notebook reproducibility:
- Structure your notebook so that it runs sequentially, from the first to the last cell.
- Regularly restart the kernel, clear outputs, and execute all cells sequentially to confirm reproducibility.
- Keep each notebook focused on a single task; avoid combining multiple tasks within one notebook.
- Utilize the `dry_run` mode during debugging to avoid cluttering the catalog with unnecessary execution records. Example:

```python
exe = ml_instance.create_execution(configuration=ExecutionConfiguration(...), dry_run=True)
```

Use `dry_run` only for debugging, not during model tuning, as recording all tuning attempts is crucial for transparency and reproducibility.

## Commit and Tagging Procedures

After validating your notebook, commit it and generate a corresponding version tag using the provided scripts. For example:

```bash
./bump-version.sh  major|minor|patch
```

## Executing Notebooks with DerivaML

A reproducable notebook execution has three components. 
1. A commited notebook file is specified.
2. Per-execution specific values for variables in the `parameters` cell are specfified and a new cell with the specified parameter values is injected into the notebock.
3. The modified notebook is executed in its entirety, including the uploading of any notebook generated assets.
4. On conclusion of the notebook executiong, the resulting notebook file, including output cells is uploaded into the DerivaML catalog and stored in the *Execution_Assets* table.

DerivaML includes the `deriva-ml-run-notebook` command to conveniently execute notebooks, substitute parameters dynamically, and store the execution results as assets.
AN example of how this command is used is:

```bash
deriva-ml-run-notebook --parameter parameter1 value1 --parameter parameter2 value2 my-notebook.ipynb
```

This command substitutes `value1` and `value2` into the corresponding parameters within the notebook's parameters cell, executes the notebook entirely, and saves the resulting notebook as an execution asset in the catalog.

Alternatively, parameters can be specified via a JSON configuration file using the `--file filename` option.
You may also automate experiments using scripts stored in GitHub, ensuring reproducibility through version control and clear documentation.


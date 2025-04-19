# Using Jupyter Notebooks With DerivaML

DerivaML can be used to execute Jupyter notebooks in a reproducible and structured manner. Although DerivaML provides numerous tools to support reproducible machine learning, users must adopt and maintain standardized development practices to fully benefit from these tools.

To ensure that your Jupyter notebooks are reproducible, follow these recommended guidelines:

## 1. Version Control and Semantic Versioning

- Always store your notebook in a GitHub repository. A repository template for DerivaML project can be found at [DerivaML Repository Template](https://github.com/informatics-isi-edu/deriva-ml-model-template) 
- Adopt semantic versioning for your notebooks. To facilitate this, the repository templates provides a script to simplify managing version numbers and tags.

```bash
bumpversion major|minor|patch
```

## 2. Clearing Notebook Outputs

- When committing changes, clear all output cells, ensuring that only code and markdown cells are version-controlled.
- To automate this process, install the provided DerivaML script which automatically strips output cells upon commit. To set this up, execute the following command once in your repository:

```bash
nboutputstrip --install
```

## 3. Centralizing Notebook Parameters

- Define all configurable variables in a single "parameters" cell located immediately after your imports at the top of your notebook.
- The parameters cell should contain only variable assignments.
- Avoid setting configuration variables elsewhere in your notebook.
- It is recommended to include type hints for clarity and usability.

## 4. Notebook Structure and Execution Flow

- Structure your notebook so that it runs sequentially, from the first to the last cell.
- Regularly restart the kernel, clear outputs, and execute all cells sequentially to confirm reproducibility.
- Keep each notebook focused on a single task; avoid combining multiple tasks within one notebook.
- Utilize the `dry_run` mode during debugging to avoid cluttering the catalog with unnecessary execution records. Example:

```python
exe = ml_instance.create_execution(configuration=ExecutionConfiguration(...), dry_run=True)
```

Use `dry_run` only for debugging, not during model tuning, as recording all tuning attempts is crucial for transparency and reproducibility.

## 5. Commit and Tagging Procedures

- After validating your notebook, commit it and generate a corresponding version tag using the provided scripts. For example:

```bash
./bump-version.sh  major|minor|patch
```

## 6. Executing Notebooks with DerivaML

- DerivaML includes the `deriva-ml-run-notebook` command to conveniently execute notebooks, substitute parameters dynamically, and store the execution results as assets.
- Example command line execution:

```bash
deriva-ml-run-notebook --parameter parameter1 value1 --parameter parameter2 value2 my-notebook.ipynb
```

This command substitutes `value1` and `value2` into the corresponding parameters within the notebook's parameters cell, executes the notebook entirely, and saves the resulting notebook as an execution asset in the catalog.

- Alternatively, parameters can be specified via a JSON configuration file using the `--file filename` option.
- You may also automate experiments using scripts stored in GitHub, ensuring reproducibility through version control and clear documentation.

Following these practices helps leverage DerivaML’s full potential, maintaining clarity, reproducibility, and continuous fairness in your data science projects.


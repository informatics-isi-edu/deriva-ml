# DerivaML
Deriva-ML is a python library to simplify the process of creating and executing reproducible machine learning workflows
using a deriva catalog.


Complete on-line documentation for DerivaML can be found [here](https://informatics-isi-edu.github.io/deriva-ml/)

## Experiments

DerivaML organizes ML activities into **experiments**. An experiment is a
**GitHub repository** that holds the *executable* and *human-readable* sides
of a research project — model code, hydra-zen configurations, and a
`tacit-knowledge.md` capturing the *why* behind project decisions. The catalog
stores the *what* (data, RIDs, lineage); the experiment repository stores
the *how* and the *why*. Every execution from the repo records the git commit
hash on its Workflow row, so a result traces to the exact code that produced
it.

To bootstrap a new experiment, clone the
[deriva-ml-model-template repository](https://github.com/informatics-isi-edu/deriva-ml-model-template)
and modify it to suit your requirements.

## API verb conventions

DerivaML's public-API methods follow a verb-then-noun naming
convention. Knowing the rule means you don't have to guess which
method to reach for:

- `find_*` — schema-introspection / discovery with filtering or
  traversal logic. Examples: `find_features`, `find_datasets`,
  `find_workflows`, `find_executions`, `find_assets`.
- `list_*` — straightforward enumeration of "what's there" inside
  a known scope. Examples: `list_assets`, `list_executions`,
  `list_dataset_members`, `list_vocabulary_terms`.
- `get_*` — single-RID detail read.
- `lookup_*` — RID-or-name resolution helper.
- `create_*` / `update_*` / `delete_*` / `add_*` — write operations.

If you reach for `ml.list_features` you will get an `AttributeError`;
the correct call is `ml.find_features(table)`. Features require an
association walk, so they live under `find_*` not `list_*`.

See [`CLAUDE.md`](CLAUDE.md#key-patterns) for the full convention.
Also see the `deriva_ml_getting_started` MCP prompt for the
slightly-different wire-protocol convention used by the
`deriva-ml-mcp` server.

## References
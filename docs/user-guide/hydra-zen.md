# Integrating with hydra-zen

Hydra-zen is an optional layer that turns Python config objects into composable, CLI-overridable configurations without requiring YAML files. You do not need it for exploration or one-off runs — it becomes valuable when you want repeatable, grid-searchable experiments launched from the command line.

## When to add hydra-zen

Two patterns cover most users:

**Straight Python — no hydra-zen needed.** If you are exploring a catalog in a notebook or running a one-time script, construct everything directly:

```python
from deriva_ml import DerivaML, DerivaMLConfig
from deriva_ml.dataset import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration

ml = DerivaML.instantiate(DerivaMLConfig(hostname="deriva.example.org", catalog_id="42"))
cfg = ExecutionConfiguration(
    datasets=[DatasetSpec(rid="1ABC", version="1.0.0")],
    description="exploratory run",
)
with ml.create_execution(cfg) as exe:
    bag = exe.download_dataset_bag(exe.datasets[0])
    # ... process data ...
exe.commit_output_assets()
```

**hydra-zen — for reproducible project runs.** When you want to sweep hyperparameters, select dataset collections from the CLI, or let `deriva-ml-run` wire everything together automatically, use hydra-zen config classes registered in a `configs/` module:

```python
# configs/model.py
from hydra_zen import builds, store
from myproject.train import train_classifier

ModelConfig = builds(train_classifier, learning_rate=1e-3, epochs=10,
                     populate_full_signature=True, zen_partial=True)
store(ModelConfig, group="model_config", name="default")
```

```bash
# Run with default config, then sweep two variants:
deriva-ml-run model_config=default
deriva-ml-run --multirun model_config=default,fast_training
```

## The four config classes

These four classes are the bridge between Python and hydra-zen. Use them in `store()` calls and `ExecutionConfiguration` constructors.

| Class | Import | Purpose |
|---|---|---|
| `DerivaMLConfig` | `from deriva_ml import DerivaMLConfig` | Catalog connection: hostname, catalog ID, working directory |
| `DatasetSpecConfig` | `from deriva_ml.dataset import DatasetSpecConfig` | Dataset input: RID, semantic version, materialize flag |
| `AssetSpecConfig` | `from deriva_ml.asset import AssetSpecConfig` | Asset input: RID, optional MD5-based cache |
| `ExecutionConfiguration` | `from deriva_ml.execution import ExecutionConfiguration` | Full execution: datasets, assets, workflow, description |

`DatasetSpecConfig` and `AssetSpecConfig` are hydra-zen dataclasses (decorated with `@hydrated_dataclass`) designed to round-trip cleanly through hydra-zen's `instantiate()`. `DerivaMLConfig` and `ExecutionConfiguration` are Pydantic models and can also be constructed directly.

See [Configuration overview](../configuration/overview.md) for field-by-field documentation and complete examples.

## How to compose configs through the CLI

`deriva-ml-run` (for Python model functions) and `deriva-ml-run-notebook` (for Jupyter notebooks) are the standard entry points. Both load your project's `configs/` module, register its contents into the hydra store, and then hand control to Hydra for CLI composition.

```bash
# Basic run — uses whatever your configs/ module registers as defaults
deriva-ml-run

# Override a config group
deriva-ml-run model_config=long_training

# Override a specific field inside a group
deriva-ml-run model_config.learning_rate=0.0001

# Sweep two dataset versions (multirun)
deriva-ml-run --multirun datasets=training_v1,training_v2
```

Multirun creates a parent execution in the catalog with one child execution per sweep element. The parent-child relationship is recorded automatically — you do not need to manage it.

### Inspecting configs: three distinct operations

These three questions have three different commands. Confusing them is a common mistake (in particular, `--list-configs` does **not** show the resolved config — it shows the *menu of choices*):

```bash
# 1. "What group=value choices can I pass?" — the menu of registered options.
#    deriva-ml-specific; Hydra has no equivalent. Ignores any overrides.
deriva-ml-run --list-configs

# 2. "What does my config actually resolve to?" — the fully composed config a
#    run would use, given these overrides, WITHOUT executing. This is Hydra's
#    own --cfg flag.
deriva-ml-run +experiment=cifar10_quick --cfg job

# 3. "Resolve AND validate against the live catalog" — checks every referenced
#    RID / vocabulary term resolves, then stops before training. Heavier
#    (downloads dataset bags). deriva-ml-specific.
deriva-ml-run +experiment=cifar10_quick dry_run=true
```

### Every Hydra command-line flag works

`deriva-ml-run` forwards any flag it does not itself define straight to Hydra,
so the entire Hydra command-line surface works as documented upstream:

- Flags reference: <https://hydra.cc/docs/advanced/hydra-command-line-flags/>
- Override grammar (sweeps, deletions, package directives): <https://hydra.cc/docs/advanced/override_grammar/basic/>

For example `--cfg job`, `--cfg hydra`, `--resolve`, `--package <group>`, and
`--info config|defaults|searchpath` all behave exactly as the Hydra docs say.

**Notes:**

- `--multirun` requires comma-separated values with no spaces.
- Config groups are discovered alphabetically; by convention, name the file `experiments.py` so it sorts last, allowing experiment configs to override base configs safely.
- Pass `--host` and `--catalog` on the CLI to override the hostname and catalog ID without touching your config files (they become `deriva_ml.hostname=` / `deriva_ml.catalog_id=` overrides).
- deriva-ml's own flags are `--list-configs`, `--catalog`, `--host`, `--config-dir`, `--config-name`, `--multirun`, and `--allow-dirty`; everything else is forwarded to Hydra. Two short-flag collisions to know: deriva-ml's `-c` is `--config-dir` (use the long `--cfg` for Hydra's config dump), and on `deriva-ml-run-notebook` `-p` is papermill's `--parameter` (Hydra's `--package` is not exposed there — the notebook runner composes config kernel-side rather than forwarding argv, so it offers `--list-configs`, `--info <mode>`, and `--cfg <mode>` as render-only inspection but not full flag passthrough).

See [Config groups](../configuration/groups.md) and [Experiments and multi-run](../configuration/experiments.md) for the full composition model.

## Project structure conventions

A project that uses `deriva-ml-run` expects this layout:

```
myproject/
├── configs/
│   ├── __init__.py        # calls load_configs() to register modules
│   ├── base.py            # DerivaMLConfig + ExecutionConfiguration defaults
│   ├── datasets.py        # DatasetSpecConfig store entries
│   ├── assets.py          # AssetSpecConfig store entries
│   ├── model.py           # builds(..., zen_partial=True) entries
│   └── experiments.py     # composed experiment overrides (loaded last)
└── src/
    └── myproject/
        └── train.py       # model function: def train(..., ml_instance, execution)
```

Rather than setting this up from scratch, use the [deriva-ml-model-template](https://github.com/informatics-isi-edu/deriva-ml-model-template) repository. It provides the full skeleton including `configs/`, a sample model function, and CI configuration. Clone it and replace the sample model with your own code.

!!! warning "Common pitfall"
    When wrapping a function with keyword arguments using `builds()`, always pass
    `populate_full_signature=True`. Without it, hydra-zen only exposes positional
    parameters to the CLI and silently ignores the rest, so overrides like
    `model_config.learning_rate=0.001` have no effect.

    ```python
    # Wrong — keyword args not exposed to CLI
    ModelConfig = builds(train_classifier, zen_partial=True)

    # Correct
    ModelConfig = builds(train_classifier, populate_full_signature=True, zen_partial=True)
    ```

## See also

- [Configuration overview](../configuration/overview.md) — field-level documentation for all four config classes, `builds()` patterns, and the hydra-zen store
- [Config groups](../configuration/groups.md) — organizing datasets, assets, and model variants into named groups
- [Experiments and multi-run](../configuration/experiments.md) — composing experiments and launching sweeps
- [Notebook-driven configs](../configuration/notebooks.md) — using `deriva-ml-run-notebook` with the same config infrastructure
- [Chapter 4: Executions](executions.md) — the `create_execution` context manager and `commit_output_assets()`
- [deriva-ml-model-template](https://github.com/informatics-isi-edu/deriva-ml-model-template) — starter repository for new hydra-zen projects

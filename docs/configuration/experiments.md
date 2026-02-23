# Experiments and Multiruns

Experiments bundle specific model, dataset, asset, and workflow choices into
reusable presets. Multiruns sweep parameters across multiple values.

## Defining Experiments

Define experiments in `configs/experiments.py`:

```python
# configs/experiments.py
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

experiment_store = store(group="experiment", package="_global_")

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},
            {"override /datasets": "training_data"},
            {"override /assets": "resnet_weights"},
            {"override /workflow": "default_workflow"},
        ],
        description="Quick training with ResNet weights on training data.",
        bases=(DerivaModelConfig,),
    ),
    name="quick_training",
)
```

Run an experiment:

```bash
uv run deriva-ml-run +experiment=quick_training

# Override an experiment parameter
uv run deriva-ml-run +experiment=quick_training model_config.epochs=25
```

Note the `+` prefix: this adds the experiment group, which is not in the default
config. The `override /` prefix in `hydra_defaults` ensures the experiment's
choices replace (rather than conflict with) the base defaults.

## Multiruns and Sweeps

For parameter sweeps, use Hydra's multirun mode:

```bash
# Sweep a parameter
uv run deriva-ml-run --multirun model_config.learning_rate=0.0001,0.001,0.01

# Sweep across experiments
uv run deriva-ml-run --multirun +experiment=quick_training,long_training
```

### Named Multirun Configurations

For complex sweeps, define named multirun configurations in
`configs/multiruns.py`:

```python
# configs/multiruns.py
from deriva_ml.execution import multirun_config

multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=quick_training",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description='''## Learning Rate Sweep

    Exploring optimal learning rates for quick training config.

    | Learning Rate | Expected Behavior |
    |--------------|-------------------|
    | 0.0001 | Slow convergence |
    | 0.001 | Standard baseline |
    | 0.01 | Fast, may overshoot |
    | 0.1 | Likely unstable |
    ''',
)
```

Run a named multirun:

```bash
# Use the named multirun config (automatically enables multirun mode)
uv run deriva-ml-run +multirun=lr_sweep

# Override parameters from the multirun config
uv run deriva-ml-run +multirun=lr_sweep model_config.epochs=5
```

Named multiruns create a parent-child execution structure in the catalog:

- **Parent execution**: Contains the markdown description and links to all children
- **Child executions**: One per parameter combination, each with full provenance

Use `list_parent_executions()` and `list_nested_executions()` to navigate this
hierarchy.

## Code Provenance

DerivaML records the Git commit hash and source URL for each execution. Always
commit your code before running models:

```bash
git add -A && git commit -m "Ready for training run"
uv run bump-version patch  # Optional: create a release tag
uv run deriva-ml-run +experiment=quick_training
```

If the working tree has uncommitted changes, DerivaML issues a warning and the
execution record may not have a valid code reference.

See [Automatic Source Code Detection](../workflows/execution-lifecycle.md#automatic-source-code-detection)
for details on how provenance works in scripts, notebooks, and Docker containers.

## See Also

- [Configuration Groups](groups.md) — The five standard config groups
- [Hydra-zen Configuration Overview](overview.md) — Configuration class reference
- [Running Models](../workflows/running-models.md) — CLI usage and complete walkthrough

# Reproducibility

This chapter covers what deriva-ml records automatically to support reproducing a past run, and what you must do to keep that record accurate. By the end you will know how to pin dataset versions, capture workflow provenance, handle a dirty working tree, and re-run a past execution from its stored configuration.

## What reproducibility means in deriva-ml

Every execution record is a snapshot of three things: the code that ran (workflow checksum and git commit), the data that was consumed (versioned dataset RIDs with catalog snapshot timestamps), and the environment in which it ran (locked Python dependencies and a runtime snapshot). Together these let you answer two questions later: "what code and data produced this result?" and "can I reproduce it?"

The guarantees have limits. Catalog snapshots pin data state at a point in time — they do not pin library versions or external services. If your catalog has a snapshot-retention policy that deletes snapshots older than some age, pinned dataset versions older than that age become unresolvable. Check your catalog's policy before relying on long-term version pinning.

## What is captured automatically

When you enter the execution context manager, deriva-ml uploads four files to the `Execution_Metadata` asset table before your model code runs:

| Asset type | File | Contents |
|---|---|---|
| `Deriva_Config` | `configuration.json` | Fully resolved `ExecutionConfiguration`: datasets, assets, workflow reference, description, and Hydra config group choices |
| `Execution_Config` | `uv.lock` | Python dependency lockfile from the project root, recording exact package versions |
| `Hydra_Config` | `hydra-<timestamp>-*.yaml` | Hydra YAML files (`config.yaml`, `overrides.yaml`, `hydra.yaml`) — only present when running via `deriva-ml-run` |
| `Runtime_Env` | `environment_snapshot_<timestamp>.txt` | Python version, installed packages, OS, GPU availability, and environment variables |

These files are uploaded at initialization, not at upload time. If the execution fails, the metadata is still in the catalog.

The `config_choices` field in `configuration.json` records which named Hydra config was selected for each config group. For example:

```json
{
  "model_config": "cifar10_quick",
  "datasets": "cifar10_small_labeled_split",
  "workflow": "cifar10_training"
}
```

This lets you reproduce a run by selecting the same named configs, rather than reconstructing parameter values from scratch.

## How to pin a dataset version

Passing a `version` to `DatasetSpec` tells deriva-ml to fetch the dataset as it existed when that version was created, not as it exists today.

```python
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration

config = ExecutionConfiguration(
    datasets=[
        DatasetSpec(rid="1-ABC4", version="2.1.0"),
    ],
    workflow=workflow,
    description="Pinned training run",
)

with ml.create_execution(config) as exe:
    bag = exe.datasets[0]
    # bag contains exactly the rows present at version 2.1.0
```

Under the hood, each dataset version stores a catalog snapshot timestamp. When deriva-ml downloads a versioned dataset bag, it appends the snapshot timestamp to every query, so it reads the catalog as of that moment. The same RID at version `2.1.0` returns the same rows on every run, regardless of subsequent additions or deletions.

**Notes:**

- The `version` argument accepts a string (`"2.1.0"`), a three-element list, or a `DatasetVersion` object.
- Omitting `version` uses the dataset's current version, which changes as members are added.
- `materialize=False` downloads table metadata only, without fetching asset files. Use this for large datasets when you only need row counts or identifiers.
- Snapshot timestamps can age out if your catalog has a retention policy. Verify that the snapshot still exists before depending on a version in long-running production code.

## How to capture workflow checksums and git commits

Workflow provenance happens in two distinct steps: local object construction and catalog-side deduplication.

**Step 1 — Local object construction.** When you call `ml.create_workflow(...)` (or instantiate `Workflow(...)` directly), a Pydantic validator inspects the calling source and populates three fields:

- The GitHub blob URL pointing to the script at the current git commit (`https://github.com/org/repo/blob/<commit>/src/models/train.py`)
- A checksum computed from the file content (git object hash)
- The git version tag (from `setuptools_scm` or `pyproject.toml`)

No catalog contact happens at this stage. The `Workflow` object is a local value object.

```python
# Constructs a local Workflow object — no catalog write yet
workflow = ml.create_workflow(
    name="CNN Training v2",
    workflow_type="Training",
    description="ResNet-50 with dropout",
)
```

**Step 2 — Catalog-side deduplication.** When you pass the workflow to `ml.create_execution(config)`, derive-ml calls `ml.add_workflow(workflow)` internally. That function queries the catalog for an existing record with the same checksum. If one exists, it returns its RID; otherwise it inserts a new record. Either way the execution is linked to the canonical RID.

```python
# All three executions below share one workflow RID because
# add_workflow (called inside create_execution) finds the same
# checksum and reuses the existing record
with ml.create_execution(config_a) as exe: ...
with ml.create_execution(config_b) as exe: ...
with ml.create_execution(config_c) as exe: ...
```

If you call `ml.add_workflow(workflow)` directly (without going through `create_execution`), deduplication still applies — but `create_execution` is the typical path that triggers it automatically.

Use this property intentionally: when you want a new workflow record (for a meaningfully different version of the code), commit the changes first. The new commit hash produces a new checksum, which causes `add_workflow` to insert a new record.

You can verify the current version tag at any time:

```bash
uv run python -m setuptools_scm
# 1.2.0        — clean release tag
# 1.2.1.dev4+gabcd1234 — 4 commits after v1.2.1
```

Tag before important runs:

```bash
git add . && git commit -m "Finalize architecture for production run"
uv run bump-version minor
uv run deriva-ml-run +experiment=production_training
```

**Notes:**

- Jupyter notebooks use the same mechanism: for `.ipynb` files, deriva-ml pipes the notebook through `nbstripout -t` before computing the git object hash, so re-running a notebook without code changes produces the same checksum. Install `nbstripout` (`pip install nbstripout`) to enable this path.
- The `url` and `checksum` fields on `Workflow` can be set explicitly to override automatic detection.
- `bump-version` fetches tags, bumps the version, creates a new tag, and pushes — all in one step. Do not use `bump-my-version` directly; it does not push.

## How to capture a Docker image digest

When running inside a Docker container there is no local git repository. Deriva-ml reads provenance from environment variables that your CI/CD pipeline sets at image build time.

`DERIVA_MCP_IN_DOCKER=true` is the **gate**: unless this variable is set to `true`, none of the other Docker provenance variables are read and deriva-ml falls back to the standard git-based path (which will fail if there is no repo). Always set this variable first.

| Variable | Purpose |
|---|---|
| `DERIVA_MCP_IN_DOCKER` | **Gate.** Set to `true` to activate Docker provenance path |
| `DERIVA_MCP_IMAGE_NAME` | Image name, e.g. `ghcr.io/org/repo` |
| `DERIVA_MCP_IMAGE_DIGEST` | Image digest (`sha256:...`) used as the workflow checksum |
| `DERIVA_MCP_GIT_COMMIT` | Git commit hash at build time (fallback when digest is absent) |
| `DERIVA_MCP_VERSION` | Semantic version of the image |

A typical GitHub Actions build step looks like:

```yaml
- name: Build and push image
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: ghcr.io/org/repo:${{ github.sha }}
  env:
    DERIVA_MCP_IN_DOCKER: "true"
    DERIVA_MCP_IMAGE_DIGEST: ${{ steps.build.outputs.digest }}
    DERIVA_MCP_GIT_COMMIT: ${{ github.sha }}
    DERIVA_MCP_VERSION: ${{ github.ref_name }}
    DERIVA_MCP_IMAGE_NAME: ghcr.io/org/repo
```

Bake these variables into the image so they are present when the container runs. Deriva-ml constructs the workflow URL as `<image_name>@<image_digest>`, which serves as the permanent, content-addressed identifier for that image.

**Notes:**

- `DERIVA_MCP_IMAGE_DIGEST` takes precedence over `DERIVA_MCP_GIT_COMMIT` as the checksum.
- If neither variable is set and no local git repository is found, the workflow record is created without a checksum, which disables deduplication.
- Deduplication still applies: two containers built from the same image have the same digest and therefore share one workflow RID.

## How configuration.json and execution metadata are captured

`configuration.json` is the serialized `ExecutionConfiguration` as it exists at the moment `ml.create_execution(config)` is called. It includes:

- The list of `DatasetSpec` objects, each with its `rid`, `version`, `materialize`, and `timeout`
- The list of `AssetSpec` objects for input assets
- The `workflow` reference (RID and checksum)
- The `description` and `argv`
- The `config_choices` dict populated by `deriva-ml-run` from the Hydra runtime

`uv.lock` is copied from the project root at the same moment. It is skipped whenever the script is not tracked by git — specifically when `git_root` cannot be resolved. The most common cases are Docker containers without a mounted checkout and scripts run from outside a git repository.

`environment_snapshot_<timestamp>.txt` is generated from the live Python environment: `sys.version`, the output of `pip list`, the OS name and CPU architecture, GPU availability via `nvidia-smi`, and a filtered subset of environment variables. It is generated and uploaded during initialization regardless of execution environment.

All four files use asset types from the `ExecMetadataType` vocabulary (`Deriva_Config`, `Execution_Config`, `Hydra_Config`, `Runtime_Env`). You can query them via `ml.list_assets("Execution_Metadata")` or by inspecting a past execution in the Chaise web UI.

## How to handle a dirty working tree

If you have uncommitted changes when a `Workflow` object is constructed, deriva-ml issues a warning: the recorded URL still points to the last committed version of the file, but the checksum reflects the uncommitted state. The execution record exists, but the workflow URL and the actual code that ran are inconsistent — your run is not reproducible from the catalog record alone.

The correct fix is to commit before running:

```bash
git status                          # see what is modified
git add .
git commit -m "Work in progress — squash before merge"
uv run deriva-ml-run +experiment=my_experiment
```

For automated tests and CI pipelines you can set `DERIVA_ML_ALLOW_DIRTY=true` or pass `--allow-dirty` on the CLI. This suppresses the warning and allows the run to proceed:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/
```

!!! warning
    Do not set `DERIVA_ML_ALLOW_DIRTY=true` in production workflows or shared CI pipelines. It silently allows runs to proceed with a workflow record that does not accurately reflect the code that ran. The execution appears to have full provenance, but the captured workflow URL may not match the uncommitted code. Auditors and future you will not be able to distinguish these runs from clean ones.

Use `dry_run=true` during development instead of `DERIVA_ML_ALLOW_DIRTY`. A dry run downloads datasets and exercises your model code but writes no catalog records:

```bash
uv run deriva-ml-run dry_run=true +experiment=my_experiment
```

**Notes:**

- Notebook runs follow the same rule: commit the notebook before running so that the recorded URL and checksum are consistent. Install `nbstripout` once per repository (`nbstripout --install`) to keep output cells out of git history.

## How to re-run a past execution

To reproduce a past execution, find it in the catalog, read its stored configuration, and construct a new `ExecutionConfiguration` from the same inputs.

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.state_store import ExecutionStatus

ml = DerivaML(hostname="catalog.example.org", catalog_id="1")

# Step 1: find the execution you want to reproduce
for record in ml.find_executions(status=ExecutionStatus.Uploaded):
    print(record.execution_rid, record.status, record.description)

# Step 2: load its stored configuration from the catalog
past_exec = ml.lookup_execution("1-XYZ4")
```

The `configuration.json` file stored in `Execution_Metadata` contains the full `ExecutionConfiguration` that was used. You can retrieve it from the catalog, or reconstruct it manually using the same dataset RIDs and versions:

```python
# Re-run with identical inputs
config = ExecutionConfiguration(
    datasets=[
        DatasetSpec(rid="1-ABC4", version="2.1.0"),
    ],
    workflow=past_exec.workflow,
    description="Reproduction of 1-XYZ4",
)

with ml.create_execution(config) as exe:
    bag = exe.datasets[0]
    # run your model with bag.path ...

exe.upload_execution_outputs()
```

Using the same `DatasetSpec` version ensures the new execution reads the same rows as the original. Using the same `workflow` object links the new execution to the same workflow record, so both appear together when you query `ml.find_executions(workflow=past_exec.workflow)`.

If the original run used Hydra, the `config_choices` in `configuration.json` record which named configs were selected. Pass those same group selections to `deriva-ml-run` to reconstruct the Hydra configuration exactly:

```bash
uv run deriva-ml-run \
  +experiment=production_training \
  model_config=cifar10_quick \
  datasets=cifar10_small_labeled_split
```

**Notes:**

- `find_executions()` queries the live catalog. Its `status` parameter accepts a single `ExecutionStatus` value (or `None` for all statuses) — not a list. For offline inspection, `list_executions()` reads the local SQLite registry without contacting the server and accepts a list of statuses.
- `lookup_execution(rid)` returns a live `ExecutionRecord` whose `status` field writes through to the catalog on assignment.
- The reproduced execution gets a new RID — it is a new record linked to the same workflow and dataset versions, not an alias of the original.
- If the original execution used input assets (`AssetSpec`), include those in the new configuration using their RIDs.

## How to trace what an artifact was produced from

Reproducing a run is one direction; tracing an artifact backwards through its
provenance chain is the other. When you're staring at a model output and want to
verify "did this prediction trace back to the dataset I expected?", use
`lookup_lineage()`:

```python
ml = DerivaML(hostname="catalog.example.org", catalog_id="1")
lineage = ml.lookup_lineage("2-PRED1")    # the prediction asset RID

# What was the immediate producing execution?
print(lineage.root.producing_execution.description)
# "Train ResNet-50 on chest X-ray dataset 1-ABC4 v1.2.0..."

# What dataset versions did that execution actually consume?
for ds in lineage.lineage.consumed_datasets:
    print(f"  {ds.rid} {ds.version} — {ds.name}")

# Walk further back: which executions produced those datasets?
for parent in lineage.lineage.parents:
    print(parent.execution.description)
```

The full chain is walked server-side in one call, replacing what would otherwise be
5–15 manual `lookup_execution` / `find_executions` round-trips. Lineage is the
**verification** half of reproducibility: re-running gives you a new artifact;
tracing tells you which inputs and code produced an existing artifact, so you can
confirm a reproduction matches the original.

See [Running an experiment — How to trace an artifact's lineage](executions.md#how-to-trace-an-artifacts-lineage)
for the full reference, including depth control, cycle handling, and the
data-flow-vs-orchestration distinction (recorded in ADR-0001).

!!! note "Common pitfalls"
    **`DERIVA_ML_ALLOW_DIRTY=true` in production.** This is the most dangerous reproducibility mistake. The execution record looks complete, but the workflow URL does not match the code that ran. Future auditors (including you) cannot tell. Reserve it for tests.

    **Omitting `version` in `DatasetSpec`.** Without a pinned version, re-running the same configuration weeks later may read a different dataset if members have been added or removed. Always pin versions for runs you intend to reproduce.

    **Catalog snapshot expiry.** Catalog snapshots can age out under a retention policy. A dataset version whose snapshot no longer exists raises an error at download time. Archive critical bag zips locally if you need long-term reproducibility guarantees.

## See also

- [Running an experiment](executions.md) — execution context manager, `upload_execution_outputs()`, and `asset_file_path()`
- [Working with datasets](datasets.md) — dataset versioning, `set_version()`, and `estimate_bag_size()`
- [Integrating with hydra-zen](hydra-zen.md) — `config_choices`, `deriva-ml-run`, multirun sweeps, `bump-version`, and `--allow-dirty`

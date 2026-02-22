# Configuring and Running Executions

Executions are how DerivaML tracks ML workflow runs with full provenance. Every execution records:

- **Inputs**: Which datasets and assets were used
- **Outputs**: Which files and datasets were produced
- **Timing**: When the workflow started and stopped
- **Status**: Progress updates and completion state

## Workflows

A **Workflow** represents a reusable computational process or analysis pipeline. Workflows
are a key part of DerivaML's provenance model — every execution is linked to exactly one
workflow, which records *what code* was run. Understanding workflows is essential before
creating executions.

### Workflow, Execution, and Workflow Type

These three concepts form a hierarchy:

- **Workflow Type** — A controlled vocabulary term that categorizes workflows (e.g., "Training",
  "Inference"). Managed in the `Workflow_Type` vocabulary table.
- **Workflow** — A reusable definition of a computational process. It records the source code
  location (URL), Git checksum, version, and type. A single workflow can be used by many
  executions.
- **Execution** — A specific run of a workflow at a particular time, with particular inputs
  and outputs. Each execution references exactly one workflow.

```
Workflow_Type (vocabulary)
  └── Workflow (reusable definition)
        └── Execution (one specific run)
        └── Execution (another run)
        └── ...
```

### Creating a Workflow

Use `ml.create_workflow()` to create a new workflow. This validates the workflow type
against the catalog vocabulary and returns a `Workflow` object:

```python
workflow = ml.create_workflow(
    name="ResNet50 Training",
    workflow_type="Training",
    description="Fine-tune ResNet50 on medical images"
)
```

You can call `create_workflow()` from any Python script, notebook, or interactive
session. Because DerivaML automatically detects the calling source code (see
[Automatic Source Code Detection](#automatic-source-code-detection) below),
the same `create_workflow()` call in the same committed script always produces
a Workflow with the same URL and checksum.

The returned `Workflow` object is not yet registered in the catalog. Registration
happens automatically when you pass it to `ml.create_execution()`, or you can
register it explicitly with `ml.add_workflow(workflow)`. In either case, if a
workflow with the same URL or checksum already exists in the catalog, the existing
record is reused rather than creating a duplicate. This means you can place
`create_workflow()` at the top of every script or notebook without worrying
about duplicate records — running the same committed code repeatedly reuses
the same workflow.

### Implicit Workflow Creation

When using the hydra-zen configuration system (via `deriva-ml-run` or
`deriva-ml-run-notebook`), you typically do not call `create_workflow()` at all.
Instead, you define a workflow configuration and the framework handles the rest.
See [Running Models and Notebooks](running-models-and-notebooks.md) for the
complete guide to setting up and running with these tools.

```python
# In your configs/workflow.py
from hydra_zen import builds, store
from deriva_ml.execution import Workflow

workflow_store = store(group="workflow")

workflow_store(
    builds(
        Workflow,
        name="ResNet50 Training",
        workflow_type="Training",
        description="Fine-tune ResNet50 on medical images",
        populate_full_signature=True,
    ),
    name="resnet_training",
)
```

At runtime, hydra-zen instantiates the `Workflow` object from this configuration,
and `run_model()` passes it to `create_execution()` automatically. The source code
detection, catalog registration, and deduplication all happen behind the scenes.

From the command line, you select the workflow by name:

```bash
# Use a named workflow configuration
uv run deriva-ml-run workflow=resnet_training

# Or as part of an experiment preset that bundles workflow + model + data
uv run deriva-ml-run +experiment=resnet_imagenet
```

### Automatic Source Code Detection

When a `Workflow` object is created (either via `ml.create_workflow()` or the `Workflow`
constructor), DerivaML automatically detects the source code that is creating the workflow
and records it for provenance. The detection works differently depending on the execution
environment.

#### Python Scripts

When running from a Python script (e.g., `python train.py` or `uv run deriva-ml-run`),
DerivaML identifies the script file, constructs a GitHub blob URL that includes the
current commit hash, and computes a Git object hash of the file content:

```
URL:      https://github.com/org/repo/blob/a1b2c3d/src/models/train.py
Checksum: e5f6a7b8c9d0...  (git hash-object of file content)
Version:  0.3.1             (from setuptools-scm or pyproject.toml)
```

If the script has uncommitted changes, DerivaML issues a warning. The URL still points
to the last committed version, so the checksum may not match the code that actually ran.
Committing before running ensures reproducibility.

#### Jupyter Notebooks

When running inside a Jupyter notebook, DerivaML identifies the notebook file by
querying the running Jupyter server for the current kernel's notebook path. The
checksum is computed after stripping cell outputs with `nbstripout`, so re-running
a notebook without code changes produces the same checksum regardless of output
differences.

For notebooks launched via `deriva-ml-run-notebook`, the notebook path and URL are
passed through environment variables (`DERIVA_ML_WORKFLOW_URL`,
`DERIVA_ML_WORKFLOW_CHECKSUM`) so that detection works even when the notebook is
executed by a separate process.

To ensure clean checksums, install `nbstripout` in your repository:

```bash
pip install nbstripout
nbstripout --install
```

#### Docker Containers

When running inside a Docker container (with `DERIVA_MCP_IN_DOCKER=true`), there is
no local Git repository. Instead, DerivaML reads provenance from environment variables
set at image build time:

| Variable | Purpose |
|----------|---------|
| `DERIVA_MCP_IMAGE_NAME` | Docker image name (e.g., `ghcr.io/org/repo`) |
| `DERIVA_MCP_IMAGE_DIGEST` | Image digest (`sha256:...`) used as checksum |
| `DERIVA_MCP_GIT_COMMIT` | Git commit hash at build time (fallback checksum) |
| `DERIVA_MCP_VERSION` | Semantic version of the image |

#### Overriding Detection

You can bypass automatic detection by setting the `url` and `checksum` fields
explicitly when constructing a `Workflow`:

```python
workflow = Workflow(
    name="Custom Pipeline",
    workflow_type="Training",
    url="https://github.com/org/repo/blob/main/pipeline.py",
    checksum="abc123def456",
)
```

Or by setting environment variables before the workflow is created:

```bash
export DERIVA_ML_WORKFLOW_URL="https://github.com/org/repo/blob/main/pipeline.py"
export DERIVA_ML_WORKFLOW_CHECKSUM="abc123def456"
```

### Reusing Existing Workflows

If a workflow with the same URL or checksum already exists in the catalog,
`ml.add_workflow()` returns the existing workflow's RID rather than creating a duplicate.
This means running the same committed script multiple times reuses the same workflow record.

You can also look up existing workflows directly:

```python
# Look up by RID
workflow = ml.lookup_workflow("2-ABC1")

# Look up by URL or Git checksum
workflow = ml.lookup_workflow_by_url("https://github.com/org/repo/blob/abc123/train.py")

# List all workflows in the catalog
all_workflows = ml.find_workflows()
for w in all_workflows:
    print(f"{w.name} ({w.workflow_type}): {w.url}")
```

### Updating Workflow Properties

Workflows retrieved from the catalog (via `lookup_workflow`, `lookup_workflow_by_url`, or
`find_workflows`) are *bound* to the catalog. You can update their `description` and
`workflow_type` properties, and the changes are written to the catalog immediately:

```python
workflow = ml.lookup_workflow("2-ABC1")
workflow.description = "Updated: now includes data augmentation"
workflow.workflow_type = "Training"  # Must be a valid Workflow_Type term
```

### Workflow Types

Workflow types are controlled vocabulary terms that categorize workflows. Common types include:

| Type | Description |
|------|-------------|
| Training | Model training workflows |
| Inference | Running predictions on new data |
| Preprocessing | Data cleaning and transformation |
| Evaluation | Model evaluation and metrics |
| Annotation | Adding labels or features |

These are not fixed — add custom types for your project:

```python
ml.add_term(
    table="Workflow_Type",
    term_name="Data_Augmentation",
    description="Workflows that augment training data"
)
```

The workflow type must exist in the catalog *before* creating a workflow that uses it.
`ml.create_workflow()` validates this and raises a `DerivaMLException` if the type is
not found.

### Providing the Workflow to an Execution

A workflow must be provided when creating an execution. There are two ways:

```python
# Option 1: In the ExecutionConfiguration
config = ExecutionConfiguration(
    workflow=workflow,
    description="Training run",
    datasets=[DatasetSpec(rid="1-ABC")],
)
exe = ml.create_execution(config)

# Option 2: As a separate argument to create_execution
config = ExecutionConfiguration(
    description="Training run",
    datasets=[DatasetSpec(rid="1-ABC")],
)
exe = ml.create_execution(config, workflow=workflow)
```

If no workflow is provided in either place, a `DerivaMLException` is raised.

## Execution Lifecycle

The execution workflow follows these steps:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Create Execution Configuration                              │
│     - Specify workflow                                          │
│     - Declare input datasets and assets                         │
├─────────────────────────────────────────────────────────────────┤
│  2. Create and Start Execution                                  │
│     - Context manager handles timing automatically              │
│     - Input datasets/assets are recorded                        │
├─────────────────────────────────────────────────────────────────┤
│  3. Run ML Workflow                                             │
│     - Download datasets as needed                               │
│     - Process data, train models, run inference                 │
│     - Register output files with asset_file_path()              │
├─────────────────────────────────────────────────────────────────┤
│  4. Upload Outputs                                              │
│     - Call upload_execution_outputs() after context exits       │
│     - Files are uploaded to Hatrac object store                 │
│     - Provenance links are created                              │
└─────────────────────────────────────────────────────────────────┘
```

## Creating an Execution Configuration

The `ExecutionConfiguration` specifies what inputs your workflow will use:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset.aux_classes import DatasetSpec

ml = DerivaML(hostname, catalog_id)

# Create a workflow definition
workflow = ml.create_workflow(
    name="ResNet50 Training",
    workflow_type="Training",
    description="Train ResNet50 on image classification task"
)

# Configure the execution
config = ExecutionConfiguration(
    workflow=workflow,
    description="Training run with augmented data",
    datasets=[
        DatasetSpec(rid="1-ABC"),                    # Use current version
        DatasetSpec(rid="1-DEF", version="1.2.0"),  # Use specific version
    ],
    assets=["2-GHI", "2-JKL"],  # Additional input asset RIDs
)
```

### DatasetSpec Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rid` | str | required | Dataset RID |
| `version` | str | None | Specific version (default: current) |
| `materialize` | bool | True | Download asset files (False = metadata only) |

```python
# Download with all files
DatasetSpec(rid="1-ABC", materialize=True)

# Download metadata only (faster for large datasets)
DatasetSpec(rid="1-ABC", materialize=False)

# Use specific version
DatasetSpec(rid="1-ABC", version="2.1.0")
```

## Running an Execution

Use the context manager pattern for automatic timing:

```python
# Create execution with context manager
with ml.create_execution(config) as exe:
    print(f"Execution RID: {exe.execution_rid}")
    print(f"Working directory: {exe.working_dir}")

    # Download input datasets
    bag = exe.download_dataset_bag(DatasetSpec(rid="1-ABC"))

    # Access dataset elements
    images = bag.list_dataset_members()["Image"]
    for img in images:
        # img["Filename"] contains local path to the file
        process_image(img["Filename"])

    # Train your model
    model = train_model(images)

    # Register output files
    model_path = exe.asset_file_path("Model", "best_model.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = exe.asset_file_path("Execution_Metadata", "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({"accuracy": 0.95}, f)

# IMPORTANT: Upload after context exits
exe.upload_execution_outputs()
```

### What the Context Manager Does

- **On entry**: Records start time, sets status to "running"
- **On exit**: Records stop time, calculates duration
- **Exception handling**: If an exception occurs, status is set to "failed"

### Why Upload is Separate

`upload_execution_outputs()` is called outside the context manager because:

1. Upload can be done asynchronously for large files
2. You can inspect outputs before uploading
3. Partial uploads can be retried if they fail
4. Even failed executions should upload partial results

### Tuning Uploads for Large Files

When uploading large files (e.g., model checkpoints > 1 GB), the default timeouts may
not be sufficient. `upload_execution_outputs()` accepts parameters to control upload
behavior:

```python
# Default behavior (25 MB chunks, 10 min timeout per chunk, 3 retries)
exe.upload_execution_outputs()

# Increase timeout for large files on slow connections (30 min per chunk)
exe.upload_execution_outputs(timeout=(6, 1800))

# Use larger chunks to reduce overhead (50 MB chunks)
exe.upload_execution_outputs(chunk_size=50 * 1024 * 1024)

# More retries with longer initial delay
exe.upload_execution_outputs(max_retries=5, retry_delay=10.0)

# Combined: large file on slow connection
exe.upload_execution_outputs(
    timeout=(6, 1800),          # 30 min read timeout per chunk
    chunk_size=50 * 1024 * 1024, # 50 MB chunks
    max_retries=5,               # 5 retry attempts
    retry_delay=10.0,            # 10s initial delay (doubles each retry)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | `(6, 600)` | `(connect_timeout, read_timeout)` in seconds per chunk |
| `chunk_size` | 25 MB | Chunk size in bytes for hatrac uploads |
| `max_retries` | `3` | Maximum retry attempts for failed uploads |
| `retry_delay` | `5.0` | Initial delay between retries (doubles each attempt) |

## Registering Output Files

Use `asset_file_path()` to register files for upload:

```python
with ml.create_execution(config) as exe:
    # Method 1: Get a path for a new file
    output_path = exe.asset_file_path(
        asset_name="Model",        # Target asset table
        file_name="model.pt"       # Filename to create
    )
    torch.save(model, output_path)  # Write to the returned path

    # Method 2: Stage an existing file
    exe.asset_file_path(
        asset_name="Image",
        file_name="/path/to/existing/file.png",  # Existing file
        copy_file=True                           # Copy (default: symlink)
    )

    # Method 3: Rename during upload
    exe.asset_file_path(
        asset_name="Image",
        file_name="/path/to/temp.png",
        rename_file="processed_image.png"
    )

    # Method 4: Apply asset types
    exe.asset_file_path(
        asset_name="Image",
        file_name="mask.png",
        asset_types=["Segmentation_Mask", "Derived"]
    )
```

## Updating Status

Report progress during long-running workflows:

```python
from deriva_ml.core.definitions import Status

with ml.create_execution(config) as exe:
    exe.update_status(Status.running, "Loading data...")

    data = load_data()
    exe.update_status(Status.running, "Training model...")

    for epoch in range(100):
        train_epoch(model, data)
        exe.update_status(Status.running, f"Epoch {epoch+1}/100 complete")

    exe.update_status(Status.running, "Saving model...")
```

## Creating Output Datasets

If your workflow produces a new curated dataset:

```python
with ml.create_execution(config) as exe:
    # Process data and generate outputs
    processed_rids = process_data(input_data)

    # Create a new dataset linked to this execution
    output_dataset = exe.create_dataset(
        description="Augmented training images",
        dataset_types=["Training", "Augmented"]
    )

    # Add processed items to the output dataset
    output_dataset.add_dataset_members(processed_rids)

exe.upload_execution_outputs()
```

## Restoring Executions

Resume working with a previous execution:

```python
# Restore by RID
exe = ml.restore_execution("1-XYZ")

# Continue working
exe.asset_file_path("Model", "continued_model.pt")
exe.upload_execution_outputs()
```

## Complete Example

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset.aux_classes import DatasetSpec
import torch
import json

# Connect to catalog
ml = DerivaML("your-server.org", "1")

# Define workflow
workflow = ml.create_workflow(
    name="Image Classifier Training v3",
    workflow_type="Training",
    description="Train CNN classifier on medical images"
)

# Configure execution
config = ExecutionConfiguration(
    workflow=workflow,
    description="Training with learning rate 0.001",
    datasets=[DatasetSpec(rid="1-ABC")],
)

# Run execution
with ml.create_execution(config) as exe:
    # Download training data
    bag = exe.download_dataset_bag(DatasetSpec(rid="1-ABC"))
    train_loader = create_dataloader(bag)

    # Train model
    model = ResNet50()
    for epoch in range(50):
        loss = train_epoch(model, train_loader)
        exe.update_status(Status.running, f"Epoch {epoch}: loss={loss:.4f}")

    # Save model
    model_path = exe.asset_file_path("Model", "classifier.pt")
    torch.save(model.state_dict(), model_path)

    # Save metrics
    metrics_path = exe.asset_file_path("Execution_Metadata", "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({"final_loss": loss, "epochs": 50}, f)

# Upload all outputs
exe.upload_execution_outputs()

print(f"Execution complete: {exe.execution_rid}")
```

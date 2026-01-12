# Configuring and Running Executions

Executions are how DerivaML tracks ML workflow runs with full provenance. Every execution records:

- **Inputs**: Which datasets and assets were used
- **Outputs**: Which files and datasets were produced
- **Timing**: When the workflow started and stopped
- **Status**: Progress updates and completion state

## Execution Lifecycle

The execution workflow follows these steps:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Create Execution Configuration                              │
│     - Specify workflow type                                     │
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

## Workflow Types

Workflows are categorized by type from the `Workflow_Type` vocabulary:

| Type | Description |
|------|-------------|
| Training | Model training workflows |
| Inference | Running predictions on new data |
| Preprocessing | Data cleaning and transformation |
| Evaluation | Model evaluation and metrics |
| Annotation | Adding labels or features |

Add custom workflow types:

```python
ml.add_term(
    table="Workflow_Type",
    term_name="Data_Augmentation",
    description="Workflows that augment training data"
)
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

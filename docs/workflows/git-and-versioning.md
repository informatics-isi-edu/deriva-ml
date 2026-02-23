# Git Workflow and Versioning

DerivaML tracks code provenance by linking execution records to Git commits.
Following a disciplined Git workflow ensures accurate tracking and
reproducibility.

## Core Principle

**Always commit before running.** DerivaML captures the Git commit hash when
you run a model or notebook. If you have uncommitted changes, the execution
record won't accurately reflect the code that produced your results.

```bash
# Check for uncommitted changes
git status

# Stage and commit
git add .
git commit -m "Add dropout regularization to CNN model"

# Now run
uv run deriva-ml-run +experiment=my_experiment
```

If the working tree has uncommitted changes, DerivaML issues a warning and the
execution record may not have a valid code reference.

## Debugging Workflow

During development, use dry runs to avoid creating execution records for
incomplete code:

```bash
# Dry run — downloads data but doesn't create records
uv run deriva-ml-run dry_run=true

# Make changes based on results...

# Once satisfied, commit and do a real run
git add .
git commit -m "Fix model architecture"
uv run deriva-ml-run +experiment=my_experiment
```

Use `dry_run` only for debugging, not during model tuning. Recording all tuning
attempts is important for transparency and reproducibility.

## Branching Strategy

Even for solo projects, use feature branches:

```bash
# Create a feature branch
git checkout -b feature/add-new-model

# Make changes, commit, push
git push -u origin feature/add-new-model
```

A typical branch structure:

```
main
 │
 ├── feature/new-model
 │    ├── commit: "Add model skeleton"
 │    ├── commit: "Implement training loop"
 │    └── commit: "Add validation metrics"
 │
 └── experiment/hyperparameter-sweep
      ├── commit: "Set up sweep configs"
      └── commit: "Run sweep experiments"
```

## Semantic Versioning

Version numbers follow the format `MAJOR.MINOR.PATCH`:

| Component | When to Increment | Example |
|-----------|-------------------|---------|
| **MAJOR** | Breaking changes to model interface or outputs | 1.0.0 → 2.0.0 |
| **MINOR** | New features, backward compatible | 1.0.0 → 1.1.0 |
| **PATCH** | Bug fixes, small improvements | 1.0.0 → 1.0.1 |

### Creating Versions

Use the `bump-version` command:

```bash
# Bug fix or small tweak
uv run bump-version patch

# New feature or significant improvement
uv run bump-version minor

# Breaking change or major milestone
uv run bump-version major
```

This creates a Git tag, pushes it, and (with the template's GitHub Actions)
triggers an automatic release.

### Checking Current Version

```bash
uv run python -m setuptools_scm
```

Example outputs:

- `1.0.0` — Clean release
- `1.0.1.dev3+g1234567` — 3 commits after v1.0.1, at commit 1234567

### When to Version

- **Before important runs**: Tag a version so execution records reference a
  clean release
- **Before experiment sweeps**: All runs in a sweep share the same version,
  making comparison easy
- **During development**: Dry runs don't need versioning; create a version
  when you're ready for a real run

```bash
# Typical workflow for a significant run
git add . && git commit -m "Prepare for production run"
uv run bump-version minor
uv run deriva-ml-run +experiment=production_training
```

## Notebook Reproducibility

Jupyter notebooks require extra discipline for reproducibility.

### Strip Output Cells

Notebook output cells change the file on every run, complicating version
control. Install `nbstripout` to auto-strip outputs on commit:

```bash
# One-time setup per repository
uv run nbstripout --install
```

After this, notebook outputs are stripped before every commit automatically.

### Notebook Structure Guidelines

- Structure notebooks to run sequentially from first to last cell
- Keep each notebook focused on a single task
- Place all configurable variables in a single
  [Papermill parameters cell](https://papermill.readthedocs.io/en/latest/usage-parameterize.html)
- Regularly restart the kernel and run all cells to confirm reproducibility
- Use `dry_run` mode during debugging to avoid unnecessary execution records

### Commit Before Running Notebooks

The same commit-before-run rule applies to notebooks:

```bash
git add -A && git commit -m "Notebook ready for execution"
uv run bump-version patch
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb
```

## Working with Large Files

Don't commit large files to Git. Use DerivaML to manage them instead:

- **Model weights** — Upload as assets via execution outputs
- **Datasets** — Store in Deriva catalogs
- **Large outputs** — Upload via `execution.upload_execution_outputs()`

```python
# Register large output for upload
model_path = execution.asset_file_path("Model_Artifact", "weights.pt")
torch.save(model.state_dict(), model_path)
# Uploaded automatically when execution completes
```

## What DerivaML Records

For each execution, DerivaML captures:

1. **Git commit hash** — Exact code state
2. **Version tag** (if on a tagged commit) — Semantic version
3. **Repository URL** — Where the code lives
4. **Branch name** — Which branch was used

See [Automatic Source Code Detection](execution-lifecycle.md#automatic-source-code-detection)
for details on how provenance works in scripts, notebooks, and Docker containers.

## See Also

- [Running Models](running-models.md) — CLI usage and model setup
- [Execution Lifecycle](execution-lifecycle.md) — How executions are tracked
- [CLI Reference](../cli-reference.md) — `bump-version` and other commands

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DerivaML is a Python library for creating and executing reproducible machine learning workflows using a Deriva catalog. It provides:
- Dataset versioning and management with BDBag support
- Execution tracking with provenance
- Feature management for ML experiments
- Controlled vocabulary management
- Asset tracking and upload

## Build and Development Commands

```bash
# Install dependencies
uv sync

# Run all tests (requires DERIVA_HOST env var or defaults to localhost)
uv run pytest

# Run a single test file
uv run pytest tests/dataset/test_datasets.py

# Run a specific test
uv run pytest tests/dataset/test_datasets.py::test_function_name -v

# Run tests with coverage
uv run pytest --cov=deriva_ml --cov-report=term-missing

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Build documentation
uv run mkdocs serve
```

## Architecture

### Core Classes

**DerivaML** (`src/deriva_ml/core/base.py`): Main entry point for catalog operations. Provides:
- Catalog connection and authentication via Globus
- Vocabulary and feature management
- Dataset creation and lookup
- Workflow and execution management

**Execution** (`src/deriva_ml/execution/execution.py`): Manages ML workflow lifecycle:
- Downloads/materializes datasets specified in configuration
- Tracks execution status and provenance
- Handles asset upload after execution completes
- Used as context manager: `with ml.create_execution(config) as exe:`

**Dataset** (`src/deriva_ml/dataset/dataset.py`): Versioned dataset management:
- Semantic versioning (major.minor.patch)
- BDBag export with optional MINID creation
- Nested dataset support
- Version history tracking via catalog snapshots

**ExecutionConfiguration** (`src/deriva_ml/execution/execution_configuration.py`): Pydantic model for execution setup:
- Dataset specifications with version and materialization options
- Input asset RIDs
- Workflow reference
- Execution parameters

### Key Patterns

**Catalog Path Builder**: Most catalog queries use the fluent path builder API:
```python
pb = ml.pathBuilder()
results = pb.schemas[schema_name].tables[table_name].entities().fetch()
```

**Dataset Versioning**: Datasets use catalog snapshots for version isolation:
- Each version records a catalog snapshot timestamp
- `dataset.set_version(version)` returns a Dataset bound to that snapshot
- Version increments propagate to parent/child datasets via topological sort

**Asset Management**: Assets are tracked via association tables:
- `Asset_Type` vocabulary controls asset categorization
- `{Asset}_Execution` tables link assets to executions with Input/Output roles
- File uploads use Hatrac object store

### Testing

Tests require a running Deriva catalog. The test fixtures in `tests/conftest.py`:
- `deriva_catalog`: Creates an empty test catalog (session-scoped)
- `test_ml`: Provides a DerivaML instance, resets catalog between tests
- `catalog_with_datasets`: Provides a catalog with populated demo data

Set `DERIVA_HOST` environment variable to specify the test server (defaults to `localhost`).

## Schema Structure

The library uses two schemas:
- **deriva-ml** (`ML_SCHEMA`): Core ML tables (Dataset, Execution, Workflow, Feature_Name, etc.)
- **Domain schema**: Application-specific tables created by users

Controlled vocabularies: Dataset_Type, Asset_Type, Workflow_Type, Asset_Role, Feature_Name

## Exception Hierarchy

DerivaML uses a structured exception hierarchy for error handling:

```
DerivaMLException (base class)
├── DerivaMLConfigurationError (configuration/initialization)
│   ├── DerivaMLSchemaError (schema structure issues)
│   └── DerivaMLAuthenticationError (auth failures)
├── DerivaMLDataError (data access/validation)
│   ├── DerivaMLNotFoundError (entity not found)
│   │   ├── DerivaMLDatasetNotFound
│   │   ├── DerivaMLTableNotFound
│   │   └── DerivaMLInvalidTerm
│   ├── DerivaMLTableTypeError (wrong table type)
│   ├── DerivaMLValidationError (validation failures)
│   └── DerivaMLCycleError (relationship cycles)
├── DerivaMLExecutionError (execution lifecycle)
│   ├── DerivaMLWorkflowError
│   └── DerivaMLUploadError
└── DerivaMLReadOnlyError (writes on read-only)
```

Import from: `from deriva_ml.core.exceptions import ...`

## Protocol Hierarchy

The library uses protocols for type-safe polymorphism:

**Dataset Protocols:**
- `DatasetLike`: Read-only operations (Dataset and DatasetBag)
- `WritableDataset`: Write operations (Dataset only)

**Catalog Protocols:**
- `DerivaMLCatalogReader`: Read-only catalog operations
- `DerivaMLCatalog`: Full catalog operations with writes

Import from: `from deriva_ml.interfaces import ...`

## Shared Utilities

**Validation** (`deriva_ml.core.validation`):
- `VALIDATION_CONFIG`: Standard ConfigDict for `@validate_call`
- `STRICT_VALIDATION_CONFIG`: ConfigDict that forbids extra fields

**Logging** (`deriva_ml.core.logging_config`):
- `get_logger(name)`: Get a deriva_ml logger
- `configure_logging(level)`: Configure logging for all components
- `LoggerMixin`: Mixin providing `_logger` attribute

## Future Decomposition

The `DerivaML` class (~1700 lines) handles multiple concerns. Future refactoring could extract:
- `VocabularyManager`: Term and vocabulary CRUD
- `FeatureManager`: Feature definition and values
- `WorkflowManager`: Workflow tracking and Git integration
- `DatasetManager`: Dataset creation and lookup
- `AssetManager`: Asset table operations

Similarly, `Execution` (~1100 lines) could be decomposed into:
- `DatasetDownloader`: Dataset materialization
- `AssetUploader`: Result upload and cataloging
- `StatusTracker`: Execution status management
# Chapter 1: Exploring a Catalog

This chapter shows you how to connect to a DerivaML catalog and discover what's in it. By the end you will know how to list tables, find datasets and features, run ad-hoc queries, and open a record in the Chaise web interface.

## Concept setup

A `DerivaML` instance is a Python handle on a remote Deriva catalog — a structured data store backed by ERMrest (a RESTful entity-relationship data service). Every row in that catalog has a **Resource Identifier (RID)**: a short, opaque, globally unique string that looks like `1-000C`. RIDs are the lingua franca of the library. Methods return them, accept them as arguments, and use them to link records across tables. The rest of this chapter shows you how to navigate the catalog and collect the RIDs you need.

In real projects you will usually work with a domain-specific subclass of `DerivaML` — for example, `EyeAI(DerivaML)` — that adds catalog-specific helper methods. The base `DerivaML` class provides everything described in this chapter, and all examples apply equally to a subclass instance.

## Connecting to a catalog

Create a `DerivaML` instance with a hostname and catalog ID:

```python
from deriva_ml import DerivaML

ml = DerivaML(hostname="catalog.example.org", catalog_id="1")
```

That's the entire connection setup for interactive use. For project-structured runs with hydra-zen configuration, the template repo wires this up for you — see the [deriva-ml-model-template repository](https://github.com/informatics-isi-edu/deriva-ml-model-template) for project setup details.

!!! note
    `DerivaML` authenticates with Globus. The first time you connect from a machine, `deriva-auth` will open a browser window. Subsequent connections reuse cached credentials.

## Understanding RIDs

Every object in a Deriva catalog — rows in every table, regardless of schema — has a RID. RIDs are:

- **Opaque.** They are assigned by the catalog, not derived from the data. `1-000C` tells you nothing about the row's content.
- **Stable.** A RID is permanent for the lifetime of the catalog. Rows are never renumbered.
- **Universally present.** Every table, in both the domain schema and the ML schema, has a `RID` column.

An unqualified RID like `1-000C` refers to the current state of the object. A snapshot-qualified RID like `1-000C@32S-W6DS-GAMG` refers to the same object as it existed at a specific point in time. Snapshot-qualified RIDs are how DerivaML pins dataset versions to a catalog state.

To obtain a globally resolvable URI for a RID — for citations or cross-catalog references — use `ml.cite()`:

```python
# Permanent citation URL, snapshot-qualified
url = ml.cite("1-000C")
# Returns: 'https://catalog.example.org/id/1/1-000C@32S-W6DS-GAMG'

# Current-state URL (no snapshot)
url = ml.cite("1-000C", current=True)
# Returns: 'https://catalog.example.org/id/1/1-000C'
```

!!! warning
    Do not parse RIDs or derive meaning from their structure. The format is an implementation detail of the Deriva platform. Treat them as opaque tokens.

## Listing tables and browsing the schema

`ml.model` gives you a programmatic view of the catalog's schema. The model knows about every table in both the domain schema and the `deriva-ml` ML schema.

```python
# List all tables in the default domain schema
for table in ml.model.domain_schema.tables.values():
    print(table.name)

# Look up a specific table by name
subject_table = ml.model.name_to_table("Subject")
print(subject_table.columns.keys())  # column names
```

The ML schema — always named `deriva-ml` — contains the core tracking tables: `Dataset`, `Workflow`, `Execution`, `Feature_Name`, and their association tables. Domain-specific tables (Subject, Image, Observation, and so on) live in the domain schema configured for your project.

## Finding datasets, features, workflows, and executions

Four high-level methods let you discover the key objects in a catalog.

**Datasets** — versioned, named collections of records:

```python
datasets = ml.find_datasets()
for ds in datasets:
    print(ds.dataset_rid, ds.version, ds.description)
```

**Features** — structured, provenance-linked annotations on table rows:

```python
# All feature definitions in the catalog
all_features = ml.find_features()
for f in all_features:
    print(f"{f.target_table.name}.{f.feature_name}")

# Feature definitions for a specific table
image_features = ml.find_features("Image")
print([f.feature_name for f in image_features])
```

**Workflows** — versioned records of computation scripts or notebooks:

```python
workflows = ml.find_workflows()
for w in workflows:
    print(f"{w.name} (v{w.version}): {w.description}")
    print(f"  Source: {w.url}")
```

**Executions** — individual runs of a workflow, with status and provenance:

```python
from deriva_ml.execution.execution import ExecutionStatus

# All executions
for record in ml.find_executions():
    print(record.execution_rid, record.status)

# Filter by status
uploaded = list(ml.find_executions(status=ExecutionStatus.Uploaded))

# Filter by workflow type
training_runs = list(ml.find_executions(workflow_type="Training"))
```

Each of these methods returns the full set of matching objects from the catalog. They are read-only operations — nothing is created or modified.

## Querying with pathBuilder

The high-level `find_*` methods cover the most common discovery tasks. When you need to query a table they don't cover — for instance, a domain-specific table, or a join across multiple tables — reach for `ml.pathBuilder()`.

`pathBuilder()` returns a fluent path-builder object that follows the ERMrest datapath API. It supports filtering, projections, and linked-table traversal without constructing raw REST URLs.

```python
# Query a domain table directly
pb = ml.pathBuilder()
domain = pb.schemas["my_domain_schema"]

subjects = domain.tables["Subject"].entities().fetch()
for row in subjects:
    print(row["RID"], row["Species"])
```

You can traverse foreign-key relationships and filter results:

```python
# Find all images linked to a specific subject RID
subject_rid = "1-000C"
pb = ml.pathBuilder()
images = (
    pb.schemas["my_schema"]
    .tables["Image"]
    .filter(pb.schemas["my_schema"].tables["Image"].Subject == subject_rid)
    .entities()
    .fetch()
)
```

For per-column projections, use `attributes()`:

```python
# Fetch only RID and Filename columns
pb = ml.pathBuilder()
image_path = pb.schemas["my_schema"].tables["Image"]
rows = image_path.attributes(image_path.RID, image_path.Filename).fetch()
```

## When to reach for pathBuilder vs. the high-level APIs

Use the high-level APIs (`find_datasets`, `find_features`, `find_workflows`, `find_executions`) for:

- Any task covered by one of those four methods.
- Code that should work against both live catalogs and downloaded `DatasetBag` objects — the high-level APIs provide the same interface on both.

Use `pathBuilder()` when:

- You need to query a domain-specific table that the high-level APIs don't expose.
- You need a join, a filter, or a column projection that would require assembling RID lists manually if done with the high-level APIs.
- You are doing exploratory analysis or building a one-off report.

Prefer the datapath API over constructing raw ERMrest URL strings. Raw URLs bypass the path builder's type inference and require manual URL encoding. The path builder produces the same REST calls but is less brittle.

## Jumping to Chaise

Chaise is Deriva's web interface for browsing and editing catalog data. When you have a table name or a RID and want to inspect it visually, `ml.chaise_url()` gives you a direct link:

```python
# URL for a table's recordset view
url = ml.chaise_url("Image")
print(url)
# 'https://catalog.example.org/chaise/recordset/#1/my_schema:Image'

# URL for a specific record (pass a RID)
url = ml.chaise_url("1-000C")
print(url)
```

Open the returned URL in a browser to see the full record with all columns and related records. This is particularly useful when debugging: if an execution produced unexpected output, jump to the `Execution` table to inspect the status, linked assets, and metadata files directly.

!!! tip
    Copy `ml.chaise_url("Execution")` into your browser after a run to browse all execution records and click through to their linked datasets, assets, and feature values.

## Common pitfalls

!!! warning
    **RIDs are opaque. Do not parse them.**

    The string `1-000C` has internal structure (a catalog identifier prefix and a row identifier suffix separated by a dash), but that structure is an implementation detail. Future Deriva versions may use a different format. Never split on `-`, strip prefixes, or derive the catalog number from a RID. If you need to resolve a RID back to a full record, use `ml.resolve_rid(rid)`.

## See also

- Chapter 2 — Working with datasets (`user-guide/datasets.md`, coming next): creating, versioning, and downloading datasets
- Chapter 4 — Running an experiment (`user-guide/executions.md`): the execution lifecycle and provenance capture
- [API Reference — DerivaML](../code-docs/deriva_ml_base.md) — full method documentation for `DerivaML`
- [API Reference — Feature](../code-docs/feature.md) — `Feature` object returned by `find_features`

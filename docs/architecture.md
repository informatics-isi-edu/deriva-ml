# DerivaML System Architecture

This document describes the architecture of the DerivaML ecosystem, including all components and their relationships.

## System Overview

DerivaML is a comprehensive platform for reproducible machine learning workflows. It combines scientific data management (Deriva), ML workflow orchestration (deriva-ml), AI-assisted development (MCP server + Claude Code), and configuration management (Hydra-zen) into a unified system that captures complete provenance for all experiments.

```mermaid
flowchart TB
    subgraph "User Interface Layer"
        CC[Claude Code<br/>AI Assistant]
        CLI[deriva-ml-run<br/>CLI]
        NB[Jupyter<br/>Notebooks]
        CH[Chaise<br/>Web UI]
    end

    subgraph "AI Integration Layer"
        MCP[deriva-ml-mcp<br/>MCP Server]
    end

    subgraph "Workflow Layer"
        DML[deriva-ml<br/>Python Library]
        HZ[Hydra-zen<br/>Configuration]
    end

    subgraph "Data Access Layer"
        DPY[deriva-py<br/>Python Client]
    end

    subgraph "Storage Layer"
        subgraph "Deriva Catalog"
            PG[(PostgreSQL<br/>Database)]
            ER[ERMrest<br/>REST API]
            HT[Hatrac<br/>File Store]
        end
    end

    CC --> MCP
    MCP --> DML
    CLI --> HZ
    CLI --> DML
    NB --> DML
    CH --> ER
    DML --> DPY
    HZ --> DML
    DPY --> ER
    DPY --> HT
    ER --> PG
```

## Component Descriptions

### Storage Layer: Deriva Catalog

The Deriva catalog provides the persistent storage foundation:

- **PostgreSQL Database**: Stores all structured data including datasets, executions, features, and vocabularies
- **ERMrest**: Entity-Relationship Mapping REST API that provides a RESTful interface to the database with fine-grained access control
- **Hatrac**: Object storage service for large files (images, model weights, prediction outputs)
- **Chaise**: Web-based user interface for browsing and editing catalog data

### Data Access Layer: deriva-py

The `deriva-py` library provides Python bindings for Deriva services:

- ERMrest client for database operations (CRUD, queries, schema management)
- Hatrac client for file upload/download
- Authentication via Globus
- BDBag support for reproducible data packaging

### Workflow Layer: deriva-ml

The `deriva-ml` library provides ML-specific abstractions:

- **DerivaML**: Main class connecting to catalogs and orchestrating workflows
- **Execution**: Tracks individual experiment runs with inputs, outputs, and status
- **Dataset**: Versioned collections of data with semantic versioning
- **Workflow**: Reusable workflow definitions linked to source code
- **Feature**: Maps vocabulary terms to domain records (labels, annotations)
- **Asset**: Files with metadata (images, model weights, predictions)

### Configuration Layer: Hydra-zen

Hydra-zen provides Python-first configuration management:

- Configuration as code (no YAML files)
- Runtime composition and overrides
- Multirun support for hyperparameter sweeps
- Automatic config tracking for reproducibility

### AI Integration Layer: deriva-ml-mcp

The MCP (Model Context Protocol) server exposes DerivaML operations to AI assistants:

- 60+ tools for catalog operations
- Resources for read-only access to schema and data
- Prompts for guided workflows
- Secure credential handling (never exposed to AI)

### User Interface Layer

Multiple interfaces for different use cases:

- **Claude Code**: AI-powered development assistant using MCP tools
- **deriva-ml-run**: Command-line interface for running experiments
- **Jupyter Notebooks**: Interactive analysis with `run_notebook()` API
- **Chaise**: Web UI for browsing and editing data

## Data Model

The core entities and their relationships:

```mermaid
erDiagram
    Workflow ||--o{ Execution : "has many"
    Execution ||--o{ Dataset : "uses (input)"
    Execution ||--o{ Dataset : "produces (output)"
    Execution ||--o{ Asset : "uses (input)"
    Execution ||--o{ Asset : "produces (output)"
    Dataset ||--o{ Dataset_Version : "has versions"
    Dataset ||--o{ Dataset : "contains (nested)"
    Dataset }o--o{ Domain_Record : "contains members"
    Feature ||--o{ Feature_Value : "has values"
    Feature_Value }o--|| Domain_Record : "labels"
    Feature_Value }o--|| Vocabulary_Term : "uses term"
    Workflow }o--|| Workflow_Type : "has type"
    Dataset }o--o{ Dataset_Type : "has types"

    Workflow {
        RID rid PK
        string name
        string url
        string checksum
        string workflow_type FK
    }

    Execution {
        RID rid PK
        string workflow FK
        string status
        string description
        timestamp start_time
        timestamp end_time
    }

    Dataset {
        RID rid PK
        string description
        string version FK
        boolean deleted
    }

    Dataset_Version {
        RID rid PK
        RID dataset FK
        string version
        string snapshot
        string minid
    }

    Asset {
        RID rid PK
        string filename
        string url
        string md5
        int length
    }

    Feature {
        RID rid PK
        string name
        string target_table
        string vocabulary
    }

    Feature_Value {
        RID rid PK
        RID feature FK
        RID target FK
        string term FK
    }
```

## Workflow: Running an Experiment

The typical flow for running an ML experiment:

```mermaid
sequenceDiagram
    participant User
    participant CLI as deriva-ml-run
    participant Hydra as Hydra-zen
    participant DML as deriva-ml
    participant Catalog as Deriva Catalog

    User->>CLI: deriva-ml-run +experiment=cifar10_quick
    CLI->>Hydra: Load and compose configs
    Hydra->>CLI: Resolved configuration
    CLI->>DML: run_model(config)

    DML->>Catalog: Create Workflow record
    DML->>Catalog: Create Execution record
    DML->>Catalog: Link input Datasets

    DML->>Catalog: Download Dataset (BDBag)
    Catalog-->>DML: Dataset files

    Note over DML: Execute model code

    DML->>Catalog: Upload output files (Hatrac)
    DML->>Catalog: Create Asset records
    DML->>Catalog: Record Feature values
    DML->>Catalog: Update Execution status

    DML-->>CLI: Execution complete
    CLI-->>User: Results summary
```

## Workflow: AI-Assisted Development

Using Claude Code with the MCP server:

```mermaid
sequenceDiagram
    participant User
    participant Claude as Claude Code
    participant MCP as deriva-ml-mcp
    participant DML as deriva-ml
    participant Catalog as Deriva Catalog

    User->>Claude: "List my datasets"
    Claude->>MCP: connect_catalog(host, catalog_id)
    MCP->>DML: DerivaML(host, catalog_id)
    DML->>Catalog: Authenticate (Globus)
    Catalog-->>DML: Session established
    DML-->>MCP: Connected
    MCP-->>Claude: Connection confirmed

    Claude->>MCP: find_datasets()
    MCP->>DML: ml.find_datasets()
    DML->>Catalog: Query Dataset table
    Catalog-->>DML: Dataset records
    DML-->>MCP: Dataset list
    MCP-->>Claude: JSON results

    Claude-->>User: "You have 13 datasets..."
```

## Dataset Versioning and BDBags

Datasets support semantic versioning for reproducibility:

```mermaid
flowchart LR
    subgraph "Catalog State"
        D[Dataset<br/>RID: ABC]
        V1[Version 0.1.0<br/>Snapshot: T1]
        V2[Version 0.2.0<br/>Snapshot: T2]
        V3[Version 1.0.0<br/>Snapshot: T3]
    end

    subgraph "BDBag Export"
        B1[BDBag v0.1.0]
        B2[BDBag v0.2.0]
        B3[BDBag v1.0.0]
    end

    subgraph "External Storage"
        S3[S3 Bucket]
        MINID[MINID Registry]
    end

    D --> V1
    D --> V2
    D --> V3

    V1 -.-> B1
    V2 -.-> B2
    V3 -.-> B3

    B3 --> S3
    S3 --> MINID
```

**Version semantics:**
- **Major**: Breaking changes to dataset structure
- **Minor**: New data added (members, types, features)
- **Patch**: Metadata corrections

**BDBag features:**
- Self-describing archive format
- Cryptographic checksums for integrity
- Remote file references for large assets
- MINID registration for permanent identifiers

## Provenance Chain

Complete provenance tracking from code to results:

```mermaid
flowchart TB
    subgraph "Code Provenance"
        GH[GitHub Repository]
        COMMIT[Git Commit SHA]
        URL[Blob URL]
    end

    subgraph "Workflow Record"
        WF[Workflow]
        WT[Workflow Type]
    end

    subgraph "Execution Record"
        EX[Execution]
        CFG[Config Choices]
        PARAMS[Parameters]
    end

    subgraph "Input Provenance"
        DS_IN[Input Datasets]
        AS_IN[Input Assets]
        VER[Dataset Versions]
    end

    subgraph "Output Provenance"
        DS_OUT[Output Datasets]
        AS_OUT[Output Assets]
        FV[Feature Values]
    end

    GH --> COMMIT
    COMMIT --> URL
    URL --> WF
    WT --> WF
    WF --> EX
    CFG --> EX
    PARAMS --> EX
    DS_IN --> EX
    AS_IN --> EX
    VER --> DS_IN
    EX --> DS_OUT
    EX --> AS_OUT
    EX --> FV
```

## Configuration System

Hydra-zen configuration hierarchy:

```mermaid
flowchart TB
    subgraph "Config Modules (src/configs/)"
        DERIVA[deriva.py<br/>Connection configs]
        DATASETS[datasets.py<br/>Dataset specs]
        ASSETS[assets.py<br/>Asset RIDs]
        MODEL[cifar10_cnn.py<br/>Model configs]
        EXP[experiments.py<br/>Experiment presets]
        MULTI[multiruns.py<br/>Sweep configs]
    end

    subgraph "Hydra Store"
        STORE[hydra-zen store]
    end

    subgraph "Runtime"
        CLI[CLI Overrides]
        COMPOSED[Composed Config]
        MODEL_FN[Model Function]
    end

    DERIVA --> STORE
    DATASETS --> STORE
    ASSETS --> STORE
    MODEL --> STORE
    EXP --> STORE
    MULTI --> STORE

    STORE --> COMPOSED
    CLI --> COMPOSED
    COMPOSED --> MODEL_FN
```

**Configuration groups:**
- `deriva_ml`: Connection settings (hostname, catalog_id)
- `datasets`: Dataset specifications with RID and version
- `assets`: Asset RID lists for pre-trained weights, etc.
- `model_config`: Model hyperparameters
- `experiment`: Preset combinations of all settings
- `multirun`: Parameter sweep definitions

## Schema Organization

The catalog uses multiple schemas for organization:

```mermaid
flowchart TB
    subgraph "deriva-ml Schema"
        DS[Dataset]
        DV[Dataset_Version]
        EX[Execution]
        WF[Workflow]
        FT[Feature]
        FV[Feature_Value]
    end

    subgraph "Domain Schema (e.g., cifar10)"
        IMG[Image]
        IC[Image_Class<br/>Vocabulary]
        ICF[Image_Classification<br/>Feature]
    end

    subgraph "Association Tables"
        DE[Dataset_Execution]
        DD[Dataset_Dataset]
        DI[Dataset_Image]
        EA[Execution_Asset]
    end

    DS --> DE
    EX --> DE
    DS --> DD
    DS --> DI
    IMG --> DI
    EX --> EA

    FT --> FV
    IC --> FV
    IMG --> FV
```

## Security Model

Authentication and authorization flow:

```mermaid
flowchart LR
    subgraph "Client"
        USER[User]
        APP[Application]
        CRED[Local Credentials]
    end

    subgraph "Authentication"
        GLOBUS[Globus Auth]
        TOKEN[Access Token]
    end

    subgraph "Deriva Services"
        ER[ERMrest]
        HT[Hatrac]
        ACL[ACL Policies]
    end

    USER --> GLOBUS
    GLOBUS --> TOKEN
    TOKEN --> CRED
    APP --> CRED
    CRED --> ER
    CRED --> HT
    ACL --> ER
    ACL --> HT
```

**Security features:**
- Globus authentication with local token caching
- Fine-grained ACLs at table, row, and column level
- MCP server never exposes credentials to AI
- Docker isolation for containerized deployments

## Summary

The DerivaML ecosystem provides:

1. **Reproducibility**: Complete provenance from code to results
2. **Versioning**: Semantic versioning for datasets with BDBag export
3. **Flexibility**: Multiple interfaces (CLI, notebooks, AI assistant, web)
4. **Scalability**: Handles large datasets with remote file references
5. **Security**: Fine-grained access control with Globus authentication
6. **Discoverability**: AI-assisted exploration and development

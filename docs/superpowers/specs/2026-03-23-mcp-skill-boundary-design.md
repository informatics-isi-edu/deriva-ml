# MCP/Skill Boundary Design for Multi-Tenant DerivaML

**Date:** 2026-03-23
**Status:** Draft
**Authors:** Carl (design), Claude (analysis synthesis)

## Problem Statement

The DerivaML MCP server currently handles everything: catalog queries, dataset downloads (GB+), bag analytics, asset uploads, execution lifecycle, RAG indexing, notebook execution, and local dev tools. For multi-tenant remote deployment, this creates five problems:

1. **Unbounded storage** — Bags, execution working dirs, and caches consume GB per user on the shared server
2. **Context bloat** — `denormalize_dataset` and `query_table` can dump 50K+ tokens into context
3. **Compute coupling** — Notebook execution and model training require local GPU/venv
4. **State fragility** — In-memory task tracking, per-process ChromaDB, and SQLite caches are lost on restart
5. **Security risk** — Running arbitrary user code (notebooks, scripts) on a shared server

## Design Principle

**MCP = catalog API gateway. Skills = local workflow orchestrator. Library = all real logic.**

```
                     ┌─────────────┐
                     │  deriva-ml  │  ← All real logic lives here
                     │  (Python)   │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────┴─────┐ ┌────┴────┐ ┌──────┴──────┐
        │ MCP Tools │ │ Skills  │ │ Direct CLI  │
        │ (catalog  │ │ (local  │ │ (notebooks, │
        │  proxy)   │ │  state) │ │  scripts)   │
        └───────────┘ └─────────┘ └─────────────┘
```

The litmus test for each tool:

| | No large state | Large state |
|---|---|---|
| **Needs catalog** | **MCP** | **MCP orchestrates, skill stores** |
| **No catalog needed** | **Either** | **Skill** |

## Tool Classification

### Tier 1: Keep in Remote MCP (~72 tools, stateless catalog proxy)

All annotation, schema, vocabulary, workflow, feature, and most dataset/execution metadata tools. These are thin wrappers over ERMrest with small, bounded responses.

**Catalog CRUD:** `connect_catalog`, `query_table`, `count_table`, `get_record`, `insert_records`, `update_record`, `create_dataset`, `add_dataset_members`, `list_dataset_members`, `split_dataset`, `create_feature`, `add_feature_value`, `fetch_table_features`, `create_execution`, `start_execution`, `stop_execution`, `create_table`, `add_column`, `create_vocabulary`, `add_term`, all annotation tools.

**Lightweight queries:** `validate_rids`, `cite`, `estimate_bag_size`, `denormalize_columns`, `rag_search`.

### Tier 2: Redesign for Remote (~12 tools, hybrid)

Tools with potentially large responses that need pagination or summary modes.

| Tool | Issue | Fix |
|------|-------|-----|
| `denormalize_dataset` | Can return 50K+ tokens | Add `summary_only` mode; enforce strict pagination |
| `query_table` / `get_table` | Unbounded rows | Hard cap at 50 rows per response with pagination cursor |
| `fetch_table_features` | All features for table | Add pagination; default to summary |
| `clone_catalog_async` | Already async | Keep; ensure task isolation per user |
| `connect_catalog` | Triggers RAG indexing | Externalize RAG trigger to background service |
| Result cache tools | SQLite per connection | Replace with shared cache (Redis) or remove |

### Tier 3: Move to Client-Side Skills (~30 tools)

Operations that download large data, process locally, or manage the filesystem.

| Tool | Why Move | Skill Pattern |
|------|----------|---------------|
| `download_dataset` | Writes GB to disk | Script Generator |
| `cache_dataset` | Same | Script Generator |
| `download_execution_dataset` | Same | Script Generator |
| `restructure_assets` | Downloads bag + reorganizes files | Script Generator |
| `download_asset` | Writes files to disk | Script Generator |
| `validate_dataset_bag` | Downloads + cross-validates | Script Generator |
| `asset_file_path` | Stages files locally | Script Generator |
| `upload_execution_outputs` | Reads local files, uploads to Hatrac | Script Generator |
| `list_storage_contents` / `delete_storage` | Scans/mutates `~/.deriva-ml/` | MCP-Coordinated |
| `denormalize_dataset` (analytics) | Large intermediate state | Local-First Analysis |

### Tier 4: Remove from Remote MCP (~12 tools, local devtools)

Tools that are CLI wrappers or spawn local processes. Meaningless on a remote server.

| Tool | Alternative |
|------|-------------|
| `bump_version` / `get_current_version` | `uv run bump-version` via Bash |
| `install_jupyter_kernel` / `list_jupyter_kernels` | `uv run deriva-ml-install-kernel` |
| `start_app` / `stop_app` / `start_schema_workbench` / `stop_schema_workbench` | Direct CLI or skill |
| `run_notebook` / `inspect_notebook` | `uv run deriva-ml-run-notebook` |
| `preview_handlebars_template` / `validate_template_syntax` | Pure functions, no server needed |
| `get_execution_working_dir` | Local path, meaningless remotely |

## Skill Architecture

### Key Insight: Skills Are Prompt Documents

Claude Code skills are SKILL.md files — static markdown that gets loaded into context. They cannot:
- Execute code directly (only instruct Claude to call tools or run Bash)
- Persist state between invocations
- Call MCP tools directly (they instruct Claude to call them)

All persistent state lives in: the MCP server (connection state), the filesystem (`~/.deriva-ml/`), or the catalog itself. Skills don't add a fourth state location.

### The Script Generator Pattern

The highest-leverage pattern for moving heavy operations to client-side:

1. Skill loads into context with instructions
2. Claude calls MCP tools for small metadata queries (bag_info, validate_rids)
3. Skill instructs Claude to generate a short Python script using `deriva-ml`
4. Claude executes the script via Bash
5. Script writes compact summary to stdout or a temp file
6. Claude reads the summary (small) into context

**Context savings:** A denormalize-then-analyze via MCP dumps 1000 rows (50K tokens) into context. The Script Generator pattern returns a 20-line statistical summary (~200 tokens).

### Proposed Skills

| Skill | Trigger | Pattern | MCP Calls | Local Work | Context Return |
|-------|---------|---------|-----------|------------|----------------|
| `download-and-analyze` | "analyze dataset" | Script Generator | `bag_info`, `cache_dataset` | Python: download + profile | 20-line stats summary |
| `restructure-for-training` | "prepare training data" | Script Generator | `estimate_bag_size` | Python: download + restructure | Dir tree + counts |
| `run-training-pipeline` | "train model" | Script Generator | `validate_rids` | `deriva-ml-run` CLI | Execution RID + metrics |
| `bulk-feature-load` | "load labels from CSV" | Script Generator | None (after connect) | Python: batch insert | "Loaded N values" |
| `clone-catalog-workspace` | "clone catalog" | MCP-Coordinated | `clone_catalog_async`, `get_task_status` | None (server-side) | Clone summary |
| `analyze-cached-bag` | "what's in this bag" | Local-First | `bag_info` | Python: bag analysis | Stats summary |
| `execution-lifecycle` | "run experiment" | Script Generator | `create_execution`, status updates | Python: full lifecycle | Execution RID + summary |

### Portability: Three-Layer Architecture

Skills are Claude Code specific. The same functionality must work in other MCP clients (Cursor, Windsurf) that don't have skills.

```
Layer 1: MCP Tools (universal — works in all MCP clients)
  - catalog CRUD, metadata, bounded queries
  - Always available, always small responses

Layer 2: deriva-ml Python Library (universal — runs locally)
  - DerivaML class with download, denormalize, restructure, upload
  - Works from any Python environment (notebooks, scripts, CLI)

Layer 3: Claude Code Skills (Claude Code specific — optional optimization)
  - Orchestrate Layer 1 + Layer 2 for best UX
  - Context-aware (minimize token consumption)
  - Not required — Layer 1 + Layer 2 work without them
```

For non-skill clients, add `summary_only` parameter to large-response MCP tools:

```python
# Non-skill client gets compact response
denormalize_dataset(..., summary_only=True)
# → {"row_count": 5000, "columns": [...], "sample": [3 rows], "cache_key": "..."}

# Then paginate with cache key
query_cached_result(cache_key="...", limit=50, offset=0)
```

## Execution Lifecycle Split

Currently the execution lifecycle spans both environments:

```
MCP (catalog CRUD):                    Skill (local orchestration):
  create_execution()                     download input datasets (bags)
  update_execution_status()              download input assets
  stop_execution()                       stage output files (manifest)
                                         build upload staging tree
                                         upload to Hatrac
```

The MCP server coordinates catalog records. The skill (or Python library) handles all filesystem operations and data movement.

## RAG Subsystem

**Current:** In-process ChromaDB with fastembed ONNX model (~150MB RAM). Single collection shared across users. Schema indexed per-catalog, docs indexed globally.

**Decision:** Keep ChromaDB for now. A better multi-tenant RAG solution is being developed separately and will replace this when ready. No investment in Chroma server mode or external vector DB migration at this time.

## Background Tasks

**Current:** In-memory Python dict. Lost on restart. Not shared across instances.

**Multi-tenant:** Requires persistent storage. Options:
- Redis-backed task queue (simplest)
- Database-backed (PostgreSQL)
- Cloud task queue (SQS/Cloud Tasks)

**Recommendation:** Redis for first deployment. Task state is small (status, progress, result pointer).

## Read-Only Tools → MCP Resources

Several current Tools are pure read operations that fit the MCP Resource pattern better:

| Current Tool | Resource URI | Benefit |
|---|---|---|
| `count_table(name)` | `deriva://table/{name}/count` | Cacheable, no side effects |
| `list_dataset_children(rid)` | `deriva://dataset/{rid}/children` | Already partially exists |
| `list_dataset_executions(rid)` | `deriva://dataset/{rid}/executions` | Read-only provenance |
| `estimate_bag_size(rid, ver)` | `deriva://dataset/{rid}/bag-size?version={ver}` | Cacheable estimate |
| `bag_info(rid, ver)` | `deriva://dataset/{rid}/bag-info?version={ver}` | Cache status check |
| `get_execution_info()` | `deriva://execution/active` | Read-only status |
| `rag_status()` | `deriva://rag/status` | Index statistics |

## Migration Strategy

### Phase 1: Slim the MCP Server (non-breaking)

1. Add `summary_only` parameter to `denormalize_dataset`, `query_table`, `fetch_table_features`
2. Enforce response size caps (50 rows default, 100 max)
3. Convert read-only tools to Resources where beneficial
4. Add pagination cursors to all list tools

### Phase 2: Create Client-Side Skills

1. Build Script Generator skills for download, analyze, restructure, upload
2. Keep MCP tools functional (backward compat) but add deprecation warnings for Tier 3/4 tools
3. Skills call MCP for metadata, `deriva-ml` for heavy lifting

### Phase 3: Remove Local State from MCP

1. RAG: keep ChromaDB for now; replace with multi-tenant RAG service when available
2. Replace in-memory task tracking with Redis
3. Remove filesystem-dependent tools from remote MCP profile
4. Introduce `local` vs `remote` server profiles

### Phase 4: Multi-Tenant Deployment

1. Per-user connection isolation (already exists via `ConnectionManager` + `contextvars`)
2. Per-user RAG collection prefixes
3. Per-user task queues
4. Monitoring and quotas

## Open Questions

1. **Result cache location:** Move entirely to skills (SQLite on client), keep in MCP (shared Redis), or hybrid?
2. **RAG replacement:** A multi-tenant RAG service is being developed separately. Keep ChromaDB until then.
3. **Asset upload path:** Should skills upload directly to Hatrac, or go through MCP as proxy?
4. **Server profiles:** One codebase with feature flags, or separate packages?

## Appendix: Expert Analysis Sources

- **ML Workflow Expert:** Analyzed execution lifecycle, data flow, latency classification, reproducibility requirements
- **MCP Architecture Expert:** Complete tool inventory (115 tools), classification, response size analysis, protocol best practices
- **Skills Architecture Expert:** Skill state persistence analysis, interaction patterns, Script Generator design, portability assessment

# Design — `validate_dataset_specs` and `validate_execution_configuration`

Date: 2026-05-03
Status: Implemented
Companion ADR: [docs/adr/0002-validate-not-dryrun-pre-flight.md](../../adr/0002-validate-not-dryrun-pre-flight.md)

## Summary

Two related public methods on `DerivaML` for cheap, metadata-only
pre-flight validation:

- **`validate_dataset_specs(specs)`** — singular dataset validator.
  Confirms a list of `DatasetSpec` objects (or shorthand strings/dicts
  that coerce to one) actually resolves: RID exists, RID points at a
  Dataset, named version exists.
- **`validate_execution_configuration(config)`** — composite pre-flight
  validator for an `ExecutionConfiguration`. Walks the contained
  `datasets` and `assets` lists, validates the workflow, and reports
  per-spec results plus cross-spec issues (duplicate RIDs, version
  conflicts, role conflicts).

The composite delegates the dataset half to the singular method.

## Why these two methods exist

Two concrete pain points:

1. **A user iterating on `src/configs/datasets.py` and wants to confirm
   specific `(RID, version)` pairs resolve before saving the config.**
   Today the user calls `ml.lookup_dataset(rid)` and
   `ds.dataset_history()` per spec, then mentally cross-checks the
   versions. `validate_dataset_specs` does this in one call with a
   structured per-spec report.
2. **A user about to run `deriva-ml-run` and wants to confirm the full
   `ExecutionConfiguration` will resolve cleanly.** Today the only
   tool that "validates" a config is `dry_run=True` — but dry_run
   pays the **bag-download** cost (minutes-to-hours, bandwidth) to do
   so. `validate_execution_configuration` is the cheap metadata-only
   pre-flight that fills the gap. See ADR-0002 for why this is a
   separate method rather than an extension to `dry_run`.

## Resolved design questions

The design tree was walked one question at a time.

### Q1 — Method placement
Both go on `DatasetMixin` in `src/deriva_ml/core/mixins/dataset.py`.
Asset-only validation is a private helper (`_validate_asset_spec`) inside
the same file; same for workflow check (`_validate_workflow_rid`).
`lookup_asset(rid)` already covers asset-existence + asset-table check;
no public asset-validation method needed.

### Q2 — Singular method name
`validate_dataset_specs` (plural). Takes a list, returns a per-spec
result list. Matches `resolve_rids` convention.

### Q3 — Composite method name
`validate_execution_configuration` (full word). Matches the
`ExecutionConfiguration` class name; no `_config` short form anywhere
else in the codebase.

### Q4 — Singular input shape
Accept `list[DatasetSpec | str | dict]`. The Pydantic `before` validator
on the method coerces:
- `str` → `DatasetSpec.from_shorthand(s)`
- `dict` → `DatasetSpec(**d)`
- `DatasetSpec` → as-is
This matches `ExecutionConfiguration.assets` and the `create_execution`
ergonomics.

### Q5 — Failure-reason vocabulary
Final per-dataset reasons: `rid_not_found`, `not_a_dataset`,
`version_not_found`. Per-dataset warnings (orthogonal to validity):
`dataset_deleted`. Per-asset reasons: `rid_not_found`, `not_an_asset`.
Per-workflow reasons: `rid_not_found`, `not_a_workflow`. Cross-spec
issues: `duplicate_rid`, `version_conflict`, `role_conflict`.

`version_string_malformed` was dropped — Pydantic raises
`ValidationError` at the boundary before the method runs.

`version_unspecified` was considered (see Q12) and dropped as too
fragile to detect post-coercion. A bare RID input becomes
`version="0.0.0"`; if `0.0.0` doesn't exist, the user sees
`version_not_found` with `available_versions` populated.

All reason strings are `Literal` types for caller type-checking.

### Q6 — Failure-detail richness
- `version_not_found` → `available_versions: list[str]` (cap 20, newest
  first). Load-bearing helpful field.
- `not_a_dataset` / `not_an_asset` / `not_a_workflow` → `actual_table:
  str` so the user sees the kind they accidentally pointed at.
- Success → `dataset_name` (Description) + `resolved_version`
  (canonical normalized form, e.g. user wrote `"0.4"` and we coerced to
  `"0.4.0"`).
- Skipped: Levenshtein on version strings, member counts. Both add
  cost without commensurate value.

### Q7 — Response shape
Mirror the `LineageResult` idiom: top-level Pydantic models with nested
compact result models, all `extra="forbid"`. Top-level `all_valid: bool`
convenience flag.

```
DatasetSpecValidationReport     # singular method's return
  all_valid: bool
  results: list[DatasetSpecResult]

ExecutionConfigurationValidationReport   # composite's return
  all_valid: bool
  dataset_results: list[DatasetSpecResult]
  asset_results: list[AssetSpecResult]
  workflow_result: WorkflowSpecResult | None
  cross_spec_issues: list[CrossSpecIssue]
```

`all_valid` is `True` iff every nested `valid` is True AND
`cross_spec_issues` is empty.

### Q8 — Pydantic model location
Single file: `src/deriva_ml/dataset/validation.py`. Mirrors how
`lineage.py` lives next to the execution mixin that returns it.

### Q9 — Implementation strategy
Per-spec for v1; ~4-10 round-trips for typical configs is acceptable.
`resolve_rids` (the existing batch tool) raises if any RID is missing —
useless for partial-result validation, so we use `resolve_rid` (single)
inside try/except per spec.

Optimization: deduplicate by RID before issuing queries (cache results
per RID); two specs pointing at the same RID still emit two per-spec
results, but cost only one resolution + one version-history fetch.

A future batched mode (when configs grow >20 specs) can land without
changing the public API.

### Q10 — Cross-spec issues
- `duplicate_rid`: same RID, two specs (datasets list OR assets list).
- `version_conflict`: same dataset RID, two specs with different
  versions.
- `role_conflict`: same asset RID listed both as Input and Output.

Skipped: cross-list overlap detection (dataset RID also in assets list).
The per-spec `not_a_dataset`/`not_an_asset` checks already catch these.

### Q11 — Composite edge cases
- Empty datasets + empty assets list → validate just the workflow.
- Missing workflow → `workflow_result = None`, not an error.
- Workflow points at non-Workflow → `not_a_workflow`.
- `@validate_call` raises `ValidationError` for non-`ExecutionConfiguration`
  input.

### Q12 — Singular edge cases
- Empty specs list → `all_valid=True, results=[]`. Not an error.
- `version="0.0.0"` is valid IFF the dataset has a `0.0.0` row.

### Q13 — Round-trip serialization
`DatasetSpec.version` (`DatasetVersion`, semver subclass) has a
`field_serializer` producing `{major, minor, patch}`. The unit tests
exercise round-trip via `model_dump_json()` → `model_validate_json()`
to catch any pydantic-v2 surprise; this matters because the deriva-ml-mcp
Round 6b wrappers will ship these models over the MCP wire.

### Q14 — Documentation deliverables
- This addendum.
- ADR-0002 for the dry-run-vs-validate distinction (see Q14 rationale
  in the conversation; matches all three ADR criteria).

### Q15 — Test counts
- `tests/dataset/test_validate_dataset_spec_unit.py` — 12 unit tests.
- `tests/dataset/test_validate_execution_configuration_unit.py` — 12
  unit tests.
- `tests/dataset/test_validate_specs_live.py` — ~6 integration tests
  gated on `DERIVA_HOST`.

### Q16 — Asset role check
`AssetSpec.asset_role` is on the spec object directly. Group by RID;
flag any RID appearing with both `"Input"` and `"Output"` values.

## Coordination with deriva-ml-skills Round 6b

After this PR merges + a `bump-version minor` lands, the deriva-ml-skills
Round 6b session will:

- Add `deriva_ml_validate_dataset_specs` MCP tool wrapper.
- Add `deriva_ml_validate_execution_configuration` MCP tool wrapper.
- Add `deriva://catalog/{h}/{c}/ml/dataset/{rid}/spec` resource.
- Update the tier-2 `write-hydra-config` skill's "Validating Configs
  Against the Catalog" section.

That work is out of scope for this PR.

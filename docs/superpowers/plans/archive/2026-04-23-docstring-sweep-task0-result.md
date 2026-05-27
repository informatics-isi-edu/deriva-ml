# Docstring Sweep — Task 0 Result

**Date:** 2026-04-23
**Decision:** `start_upload` RENAME TO `_start_upload`

## Evidence

### Searches

| Search | Results |
|---|---|
| `grep deriva-ml-model-template` | no hits (repo directory not present / empty) |
| `grep deriva-mcp` | no hits (repo directory not present) |
| `grep deriva-ml-apps` | no hits (repo directory not present) |
| `grep deriva-ml-demo` | no hits (repo directory not present) |
| `grep deriva-ml/src` | 3 hits — definition (`execution.py:745`), docstring example (`execution.py:764`), docstring note (`upload_job.py:46`) |

### Reasoning

No external caller was found in the template repo or any sibling Deriva repo. The only
occurrences in `src/` are the method definition itself and its own docstring examples.
The test file `tests/execution/test_upload_public_api.py` exercises the method, but test
code is not a public API contract; the test file name can be updated as part of the rename.
Renaming to `_start_upload` is therefore safe.

## Action for Task 14

Perform rename: `start_upload` → `_start_upload` in `src/deriva_ml/core/mixins/execution.py`
and update the reference in `src/deriva_ml/execution/upload_job.py`. Update
`tests/execution/test_upload_public_api.py` call sites accordingly.

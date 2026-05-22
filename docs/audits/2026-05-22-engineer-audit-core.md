# deriva-ml pre-release engineer audit (v1.37.1) — `core/`

Reviewed the core subsystem under
`/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/core/` (21 modules
+ 11 mixins, ~10 800 LoC) and the corresponding test surface
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/core/` (20 files,
~3 100 LoC) at the v1.37.1 release point. Cross-references against
`src/deriva_ml/interfaces.py`, `src/deriva_ml/execution/`,
`src/deriva_ml/dataset/`, and `src/deriva_ml/model/` confirm the in-
tree consumers of the core surface. Severity scale: P0 (block release),
P1 (must fix this release), P2 (next release), P3 (cleanup when
convenient).

## Summary

Overall posture: **the foundation modules are good and the recently-
refactored schema-cache machinery is excellent**. `exceptions.py`
carries a well-organised hierarchy with consistent docstrings and
structured attributes on every typed exception. `schema_cache.py` (218
LoC) and its test suite (336 LoC) are a model of small, single-purpose
design — atomic writes, concurrent-writer regression coverage, pin/
unpin/status all exercised. `schema_diff.py` is pure-functional, well-
documented, and has deterministic-ordering tests. `connection_mode.py`
is 34 LoC and exercises every code path through `tests/core/
test_connection_mode.py`.

The weaker areas are:

1. **`base.py` is 1 675 LoC of public API and the user's first contact
   — and the docstrings have stale/wrong content in several places.**
   `list_execution_dirs` and `get_storage_summary` use lowercase
   `any` as a type hint (the builtin, not `typing.Any`). Several
   docstring `Returns:` claims (chaise_url, cite) put concrete catalog
   IDs that contradict the test fixture values. `catalog_snapshot`
   constructs a new `DerivaML` without forwarding `working_dir`,
   `cache_dir`, `mode`, `credential`, etc. — the returned instance
   loses every non-default init kwarg.

2. **`validation.py` (the public RID-validation surface) has zero
   tests in `tests/core/`.** `validate_rids`, `validate_execution_
   config`, and `ValidationResult` are exported and consumed via the
   hydra-zen entry path; the only coverage is incidental via
   `tests/dataset/test_validate_execution_configuration_unit.py` and
   `tests/dataset/test_validate_specs_live.py`. The error-message
   parsing at `validation.py:257` (`if "Invalid RIDs:" in error_msg`)
   is string-coupled to `RidResolutionMixin.resolve_rids`'s message —
   if either side changes, the validator silently degrades to
   "RID validation failed: ..." for every bad RID.

3. **`filespec.py:144` has a real bug.** Inside `create_filespecs`'s
   inner `create_spec(file_path)`, the `length` field is computed as
   `path.stat().st_size` — that's the *outer* `path` (the directory
   the user passed in), not `file_path` (the file being processed).
   When `create_filespecs` walks a directory, every produced FileSpec
   carries the directory's stat-size, not its own. No test exercises
   the multi-file `tmp_path` size invariant; the existing tests in
   `tests/core/test_file.py` only assert filename and type-list.

4. **The mixin layer's docstrings claim methods that don't exist or
   that live in other mixins.** `WorkflowMixin` class docstring lists
   `find_workflow_by_url` (actual: `lookup_workflow_by_url`) and
   `list_workflow_executions` ("via FeatureMixin" — and indeed
   `list_workflow_executions` is defined in `FeatureMixin`, not
   `WorkflowMixin`). `VocabularyMixin.lookup_term`'s `Raises:` section
   names `DerivaMLVocabularyException` — a class that has never
   existed.

5. **`logging_config.py` is configured by `DerivaML.__init__` but has
   zero test coverage.** No test exercises `configure_logging`,
   `is_hydra_initialized`, or `_apply_logger_overrides`. The
   `get_logger` doctests work, but the side-effect-laden
   `configure_logging` (sets levels on 4 deriva-py loggers, 4 hydra
   loggers, and one deriva-ml logger; adds a handler conditionally)
   is the actual surface that matters at `DerivaML.__init__:354`.

6. **The exception hierarchy has drift between `exceptions.py.__all__`
   and `definitions.py.__all__`.** `definitions.py` re-exports a
   *subset* of exceptions and omits `DerivaMLFeatureNotFound`,
   `DerivaMLOfflineError`, `DerivaMLStateInconsistency`,
   `DerivaMLDirtyWorkflowError`, `NoAssociationException`,
   `AmbiguousAssociationException`, `DerivaMLSchemaRefreshBlocked`,
   `DerivaMLSchemaPinned`, `DerivaMLMaterializeLimitExceeded`, and
   every `DerivaMLDenormalize*` class. Users following the documented
   `from deriva_ml.core.definitions import ...` pattern (constants.py
   docstring, definitions.py module header) cannot reach the typed
   exceptions added since Phase 2.

7. **Online-only operations have inconsistent exception types.**
   `refresh_schema`/`diff_schema` raise `DerivaMLReadOnlyError` for the
   "offline mode" guard at `base.py:555,707`; `create_execution`
   raises `DerivaMLOfflineError` for the same condition (`execution.
   py:170`). The dedicated `DerivaMLOfflineError` exists for this case;
   `DerivaMLReadOnlyError` is overloaded.

Counts: **51 findings** (1 P0, 9 P1, 25 P2, 16 P3) across the 21 core
modules + 11 mixins, plus 7 cross-module coverage gaps and 5
duplication candidates.

---

## `core/base.py` — DerivaML class (1 675 LoC)

### B1. `list_execution_dirs` and `get_storage_summary` annotate return type with builtin `any` [P1]

`base.py:1466`, `base.py:1589`. Both signatures use
`-> list[dict[str, any]]` and `-> dict[str, any]` — `any` is Python's
*builtin* `any()` function, not `typing.Any`. At runtime this is
silently ignored by Python's type system, but `mypy`/`pyright` will
flag it and the IDE autocompletion is wrong. Fix to `Any` (already
imported via `from typing import Any` at line 20).

### B2. `catalog_snapshot` drops every non-default kwarg [P1]

`base.py:756-774`. The returned instance is built with
`DerivaML(self.host_name, version_snapshot, logging_level=...,
deriva_logging_level=...)`. Every other kwarg the user passed —
`working_dir`, `cache_dir`, `domain_schemas`, `default_schema`,
`s3_bucket`, `use_minid`, `credential`, `mode`, etc. — is silently
re-defaulted. For a user doing `ml.catalog_snapshot(snap)`, the
returned object has a different `working_dir` than `ml`, may have
different `domain_schemas` (if auto-detection picks differently from
the snapshot), and loses cached credentials. Either forward all
kwargs (recommended) or document the contract that snapshot
instances are bare.

### B3. `__del__`'s catalog session check is correct but untested for the GC-ordering case [P2]

`base.py:376-407`. The docstring says the abort is skipped when the
catalog session may already be torn down. The tests at
`test_del_no_abort_terminal.py` cover the status-filter logic but
never exercise the actual GC-ordering torn-session crash that
prompted the fix. A test that builds a real `DerivaML`, monkey-
patches `self._execution.update_status` to assert the call is
suppressed when `self.catalog._session is None`, would pin the
guarantee.

### B4. `cite()` swallows the `KeyError` cause [P2]

`base.py:1040-1043`. The except clause raises a bare
`DerivaMLException("Entity {e} does not have RID column")` for both
`KeyError` (missing "RID" key) and the upstream `DerivaMLException`
("Entity RID does not exist"). The first message string-interpolates
the `KeyError` (so the user sees `Entity 'RID' does not have RID
column` — odd phrasing); the second drops the original cause. Add
`from e` and consider distinct messages.

### B5. `cite()` doctest example mixes two output shapes [P3]

`base.py:1015-1028`. The "Permanent citation (default)" example
shows the catalog ID `1` in `https://deriva.org/id/1/...`, while
the test harness in `tests/core/test_base.py::_CiteHarness` uses
catalog_id `42`. The doctest is `# doctest: +SKIP` so the mismatch
is harmless, but a user copy-pasting from the docstring will
puzzle at the discrepancy.

### B6. `download_dir` is a thin two-line method with a docstring three times its size [P3]

`base.py:797-813`. The doctstring is fine; it's just a candidate
for inlining at the few call sites (4 of them in
`dataset/dataset.py`).

### B7. `chaise_url` returns a string that the docstring claims is `'https://...'/schema:experiment_table'` but the test fixture produces `'https://deriva.example.org/chaise/recordset/#42@2026-01-01T12:00:00/...'` [P2]

`base.py:963-993`. The docstring `Returns:` says the URL format is
`https://{host}/chaise/recordset/#{catalog}/{schema}:{table}`, but
the real branch at line 989 uses `self.catalog.get_server_uri()`
which carries the snaptime when on a snapshot. The format the
docstring documents is the *non-snapshot* case; the snapshot case
isn't documented. Add the snapshot-form example or document the
two-branch behavior.

### B8. `_find_context_file` only searches up; no symlink resolution test [P3]

`base.py:1649-1674`. The implementation uses `Path.resolve()` on
the start dir, which follows symlinks. The unit tests
(`tests/core/test_base.py::test_find_context_file_in_start_dir` and
`_walks_parents`) only test the non-symlink case. A symlinked
working directory pointing at a different drive would silently
look up the wrong context file. Not high risk; document the
behavior or add a symlink test.

### B9. `pin_schema` online path swallows `cache.load()` exceptions silently if the cache file vanishes between `if cache.exists()` and `cache.load()` [P3]

`base.py:636-654`. The check at the top of `pin_schema` calls
`cache.load()` unconditionally to compare snapshot ids. If the
cache file is deleted by a concurrent writer between `__init__`'s
write and `pin_schema`'s read, `cache.load()` raises
`FileNotFoundError`. The docstring claims `FileNotFoundError` is
raised in that case, but the online branch ordering means the
exception propagates *after* the pin would have applied for the
offline path. Add a `try/except` consistent with `unpin_schema`'s
"no-op if not pinned" semantics.

### B10. `clear_cache` and `clean_execution_dirs` log via `self._logger` but no other base-class method does [P2]

`base.py:1427, 1431, 1584`. The `_logger` attribute is set in
`__init__` but only these three methods use it; everywhere else
the module-level `logger = get_logger(__name__)` is used. Pick
one; the dual-source pattern is the kind of thing that breaks
when someone adds a new error path and consults the wrong logger
namespace.

---

## `core/exceptions.py` — exception hierarchy (780 LoC)

### E1. `definitions.py.__all__` omits 10+ exception classes [P1]

`definitions.py:120-181`. The module re-exports only the legacy
exception set and misses every new class added since Phase 2:
`DerivaMLFeatureNotFound`, `DerivaMLOfflineError`,
`DerivaMLStateInconsistency`, `DerivaMLDirtyWorkflowError`,
`NoAssociationException`, `AmbiguousAssociationException`,
`DerivaMLSchemaRefreshBlocked`, `DerivaMLSchemaPinned`,
`DerivaMLMaterializeLimitExceeded`, `DerivaMLDenormalizeError` and
all 5 subclasses. The module header recommends importing from
`deriva_ml.core.definitions`, so users following that guidance can't
reach the new classes. Either expand the re-export to match
`exceptions.py.__all__` exactly, or delete the exception block in
definitions.py and document `from deriva_ml.core.exceptions import
...` as canonical.

### E2. `core/__init__.py` exports only three exception classes [P2]

`core/__init__.py:35,42-45`. Only `DerivaMLException`,
`DerivaMLInvalidTerm`, `DerivaMLTableTypeError` are re-exported from
`deriva_ml.core`. The package-level `deriva_ml.__init__` re-exports
more, but the `deriva_ml.core` namespace under-promises. A user
catching `from deriva_ml.core import DerivaMLNotFoundError` gets an
`ImportError`. Either drop the partial set (force the longer
qualified imports) or include the full hierarchy.

### E3. `DerivaMLException._msg` private attribute is dead [P3]

`exceptions.py:96, 105-107`. The base class stores `self._msg = msg`
but no subclass and no caller in the codebase reads it. `super().
__init__(msg)` already exposes the message via `str(exc)`. Either
drop `_msg` or document it as a public alias.

### E4. Most typed exceptions have docstring `Example:` blocks that all `+SKIP` doctest [P3]

`exceptions.py:101, 122, 135, 151, 167, 183, 201, 220, 232, 250,
269, ...`. Twenty-plus classes have raise-pattern examples that all
skip doctest because they involve a fictional `ml` or `raise`. A
handful — `DerivaMLMaterializeLimitExceeded` at line 379-384,
`NoAssociationException` at line 451-461 — show how runnable
examples can be written (attribute access on the raised exception).
Convert the others to runnable form (construct the exception, check
attributes), and only `+SKIP` the inline raises.

### E5. `NoAssociationException` is documented as `DerivaMLNotFoundError` but its sibling `AmbiguousAssociationException` is `DerivaMLDataError` [P2]

`exceptions.py:435-501`. The asymmetry is documented in
`exceptions.py:9-25` and tested in `test_exceptions.py:39-77`, so
the inheritance is intentional. But the inheritance pair
("missing → NotFound, multiple → DataError") makes it impossible
to catch both with a single typed `except`. Consider whether a
shared `AssociationException` parent (extending `DerivaMLDataError`)
would simplify the call-site catch in `find_association`'s callers.

### E6. `DerivaMLStateInconsistency` has no test coverage [P2]

`exceptions.py:415-432`, `tests/core/test_exceptions.py:17-24`.
Only the inheritance is tested. The class docstring promises
"enough information for a human to intervene" but doesn't define
the message contract — no example of the actual SQLite-vs-catalog
disagreement string that `state_machine.reconcile_with_catalog`
emits. Add a positive-path test that exercises the real
disagreement detector at `execution/state_machine.py` and
confirms the typed exception is raised with the documented info.

### E7. `DerivaMLDirtyWorkflowError` carries `self.path` but is constructed only from the path argument [P3]

`exceptions.py:553-557`. The constructor signature is
`__init__(self, path: str)`. The message includes the path; the
attribute makes the path retrievable structurally. Fine — but the
example in the docstring (line 549-551) uses only the message
format, never demonstrating `.path` access. Document the structured
attribute.

---

## `core/validation.py` — RID + config validation (427 LoC)

### V1. Module is exported but has zero direct tests in `tests/core/` [P1]

`validation.py:174-350` (`validate_rids`), `352-427`
(`validate_execution_config`), `88-171` (`ValidationResult`). The
old `tests/schema/test_validation.py` is deleted (only the `.pyc`
remains). Coverage is now incidental through
`tests/dataset/test_validate_execution_configuration_unit.py`. Add
direct unit tests for:

- `ValidationResult.merge` (no test exercises the `validated_rids`
  merge case)
- `ValidationResult.__repr__` with errors+warnings combinations
  (covers the formatted output users see)
- `validate_rids` with each of the five filter combinations
  (`dataset_rids`/`asset_rids`/`workflow_rids`/`execution_rids`
  /empty)

### V2. `validate_rids` parses an error message string to detect invalid RIDs [P1]

`validation.py:256-263`. The block

```python
error_msg = str(e)
if "Invalid RIDs:" in error_msg:
    for rid in all_rids:
        if rid not in result.validated_rids:
            ...
            result.add_error(...)
else:
    result.add_error(f"RID validation failed: {e}")
```

is coupled to `RidResolutionMixin.resolve_rids`'s message at
`rid_resolution.py:211` (`raise DerivaMLException(f"Invalid RIDs:
{remaining_rids}")`). Any future refactor to the resolver — even a
period change — silently breaks the per-RID error reporting and
collapses every bad-RID case to one generic message. Either:

1. Have `resolve_rids` raise a typed exception carrying the bad-RID
   set as a structured attribute (`DerivaMLNotFoundError` subclass);
   `validate_rids` then reads the attribute.
2. Move the partial-failure logic into `resolve_rids` itself by
   returning a result object instead of raising.

### V3. `validate_execution_config` does duck-typed attribute access on `datasets`/`assets` [P2]

`validation.py:391-410`. The function accepts dict, dataclass,
`DatasetSpec`, `AssetSpec`, and bare strings. The `getattr(x,
"rid", None) or (x.get("rid") if isinstance(x, dict) else None)`
pattern obscures the contract — neither the parameter type
annotation (`list[Any]`) nor the docstring tells the caller which
shapes are accepted. Replace `list[Any]` with the actual union or
add a Pydantic adapter.

### V4. `ValidationResult.is_valid` defaults to `True`; `add_error` flips it [P3]

`validation.py:121, 126-129`. Fine as a pattern, but tested only
indirectly. Add a one-line direct test:
`r = ValidationResult(); r.add_error("x"); assert r.is_valid is
False`.

### V5. `STRICT_VALIDATION_CONFIG` is exported but used nowhere in core/ [P3]

`validation.py:61-66`. Search hits zero call sites in
`src/deriva_ml/core/`. It's defensive infrastructure for future
strict-validation needs; document the intent or drop it.

---

## `core/logging_config.py` — logging setup (221 LoC)

### L1. `configure_logging` has no test coverage [P1]

`logging_config.py:122-199`. The function is invoked by
`DerivaML.__init__:354` and decides:

- Whether to add a `StreamHandler` (only when not under Hydra and
  no handler exists already)
- Which 8 loggers (deriva-ml + hydra + deriva-py + bdbag + bagit)
  to set levels on
- Whether to call `basicConfig` (it explicitly doesn't)

A test that monkeypatches `logging.getLogger` and asserts the
resulting level-and-handler state for `(deriva_ml,)`, `(hydra*,)`,
and `(deriva, bagit, bdbag)` separately under three runtime cases
(no-hydra, hydra-initialized, custom handler) would pin the
contract.

### L2. `is_hydra_initialized` swallows `ImportError` and `Exception` [P2]

`logging_config.py:72-77`. The `except (ImportError, Exception)`
swallows every error from `GlobalHydra.instance().is_initialized()`.
`Exception` already catches `ImportError`; the redundant catch
makes the bare `Exception` swallow look like a typo. Replace with
`except Exception:` and add a debug log for the failure.

### L3. `get_logger` doctest at lines 107-113 is informative but doesn't test the `__name__`-prefix-stripping path [P3]

`logging_config.py:114-119`. The form-3 path
(`get_logger("deriva_ml.dataset")` returns
`deriva_ml.dataset` unchanged) is doctested. The corresponding
"already-prefixed *and* invalid (no `.` separator)" path —
`get_logger("deriva_ml")` — is doctested. Add the
short-suffix-with-dot edge case: `get_logger("a.b")` should
become `deriva_ml.a.b` (the current implementation does this; just
no test pins it).

### L4. `DEFAULT_FORMAT`, `HYDRA_LOGGERS`, `DERIVA_LOGGERS` are module-level constants but absent from `__all__` [P3]

`logging_config.py:38, 42-46, 49-54, 216-221`. They're public-shaped
(uppercase, top-level), and a user customizing logging would
plausibly want to import them. Add to `__all__` or rename with a
leading underscore.

### L5. `LOGGER_NAME` is exported from `__all__` but `get_logger.__doc__` is the canonical entry [P3]

`logging_config.py:35, 215-221`. Minor — the constant doubles up
documentation responsibility with the `get_logger` docstring. Pick
one.

---

## `core/schema_cache.py` — workspace cache (218 LoC)

### S1. `pin_status` raises `FileNotFoundError` when cache is missing — but `snapshot_id` returns `None` for the same condition [P2]

`schema_cache.py:80-87, 194-218`. The asymmetry is intentional
(snapshot lookup is "informational", pin_status is "authoritative")
but is undocumented at the class level. Add a "Read API
conventions" note to the class docstring.

### S2. `_path` attribute is exposed by name only via `cache._path` in tests [P3]

`schema_cache.py:74, schema_diff.py: uses cache._path indirectly`.
Six tests in `test_schema_cache.py` reach for `cache._path` to
plant or inspect the underlying file. The leading underscore says
"private"; convert to a public `path` property or document the
test-friendly access pattern.

### S3. Pin/unpin operations don't log [P3]

`schema_cache.py:155-192`. `refresh_schema` logs at INFO; pin and
unpin don't. For audit-trail purposes — pin/unpin is exactly the
sort of thing a reproducibility check would want logged — add a
single INFO line per call.

---

## `core/schema_diff.py` — structural diff (321 LoC)

### D1. `_compute_diff` only inspects tables inside schemas present in **both** payloads [P2]

`schema_diff.py:246`. The `for schema_name in sorted(cached_names &
live_names):` line means that tables inside a newly-added schema
(reported via `added_schemas`) do not appear in `added_tables`.
Symmetric for removed schemas. This is defensible (the added/removed
*schema* implies its tables) but subtle — a user inspecting
`diff.added_tables` will miss tables in added schemas. Document the
behavior or expand the walk to enumerate tables of added schemas
into `added_tables`.

### D2. `_fkey_key` assumes all `referenced_columns` share `schema_name`/`table_name` [P3]

`schema_diff.py:194-204`. The function reads `ref_cols_raw[0]
["schema_name"]` to derive the referenced schema/table. ERMrest's
data shape guarantees this (every FK references one table), but
nothing in the code validates the invariant. If the catalog ever
emits a malformed FK with a multi-table reference array, the
`_fkey_key` silently picks the first entry. Add an `assert`.

### D3. `SchemaDiff.render()` returns empty string for empty diffs [P3]

`schema_diff.py:134-169, test_schema_diff.py:175-177`. The test
asserts `empty.render() == "" or "no changes" in empty.render().
lower()` — the `or` clause is a hedge against an implementation
that doesn't exist. Tighten the test to `assert empty.render() ==
""`.

---

## `core/config.py` — DerivaMLConfig (233 LoC)

### C1. `init_working_dir` validator silently overwrites the user-provided `working_dir` [P1]

`config.py:133-161`. The validator runs `compute_workdir(self.
working_dir, ...)`, which appends `username/deriva-ml/hostname/
catalog_id` to whatever path the user provided. A user who passes
`working_dir="/tmp/wd"` ends up at `/tmp/wd/<user>/deriva-ml/<host>/
<catalog>` and is not told. The same `working_dir` argument to
`DerivaML.__init__` at `base.py:307-310` is honored as-is:

```python
if working_dir is not None:
    self.working_dir = Path(working_dir).absolute()
```

So the same kwarg behaves differently depending on whether you go
through the Pydantic config or the bare constructor. Pick one
semantic. The bare-constructor behavior (use path as given) is the
less surprising of the two; the Pydantic mutation behavior is
hydra-specific path-organisation that should live in `compute_
workdir` only, not in the validator.

### C2. `init_working_dir` reads `HydraConfig.get().runtime.output_dir` unconditionally [P1]

`config.py:151`. If the config is instantiated outside a Hydra
run, this raises `ValueError: HydraConfig was not set`. The test
suite hides this via `with patch("deriva_ml.core.config.
HydraConfig")` (see `test_hydra_zen_config.py:64`). User code that
constructs `DerivaMLConfig(hostname=..., catalog_id=...)` without
a Hydra context (e.g., in a Jupyter notebook for ad-hoc config)
hits the same `ValueError`. Wrap in a try/except and default to
`None`, or document the Hydra-context prerequisite at the
docstring.

### C3. `compute_workdir`'s `working_dir` argument has surprising semantics [P2]

`config.py:163-203`. If `working_dir` is *truthy*, the function
appends `username/deriva-ml` to it; if falsy (`None` or empty
string), it uses `~/.deriva-ml`. The truthy/falsy branching is
fine, but the inserted `username` middle directory is documented
as "to prevent conflicts between users" — yet no other code in
the workspace honors this convention (e.g., `DerivaML.__init__`
on the bare path). Either remove the `username` insertion (the
catalog-id+hostname suffix already disambiguates) or apply it
uniformly.

### C4. `OmegaConf.register_new_resolver(..., replace=True)` at module import time [P2]

`config.py:212`. The `replace=True` flag means importing
`deriva_ml.core.config` silently overwrites any previously-
registered `compute_workdir` resolver. Two concurrent importers
(deriva-ml itself and a downstream project that registered its own
resolver under the same name) silently lose information. Either
drop `replace=True` and let the second importer get a clear
`ValueError`, or namespace the resolver name.

### C5. `store(HydraConf(...))` at module import time [P3]

`config.py:217-230`. Importing `deriva_ml.core.config` registers
Hydra config defaults *globally*. A user importing the module just
to read the `DerivaMLConfig` class — for type-checking or
validation purposes — also mutates the Hydra store. Move the side
effect into a `configure_hydra()` function or guard with
`if __name__ == ...` (not applicable here, but a flag like
`DERIVA_ML_REGISTER_HYDRA` would let the user opt out).

---

## `core/filespec.py` — file metadata (190 LoC)

### F1. `create_filespecs` length bug — uses outer `path.stat()`, not `file_path.stat()` [P0]

`filespec.py:144`. Inside `create_spec(file_path)`:

```python
return FileSpec(
    length=path.stat().st_size,  # <-- BUG: should be file_path.stat().st_size
    md5=md5,
    ...
    url=file_path.as_posix(),
    ...
)
```

When the input `path` is a directory, `path.stat().st_size` returns
the directory inode's size (typically 64-512 bytes) and every
FileSpec produced from the directory walk gets that wrong size —
not its own file's size. Single-file invocations are correct
because `path == file_path` in that case. The test in
`test_file.py::test_create_filespecs` doesn't assert the length
field, so the bug has gone undetected. Fix to
`file_path.stat().st_size` and add a regression test that creates
two files of different sizes in a tmp dir and asserts both
lengths.

### F2. `create_filespecs` doesn't handle empty `file_types` argument cleanly [P3]

`filespec.py:134-136, 142-149`. `file_types = file_types or []`
then `file_types_fn = file_types if callable(file_types) else
lambda _x: file_types`. The closure captures `file_types` (the
mutated local), but if a caller passes `file_types=[]` explicitly
(non-falsy after the `or`), the lambda returns `[]` and the `if
"File" in type_list` check at line 149 produces `["File"]`. Fine,
but the closure-vs-shadowing risk is real if the function is
extended. Convert the static path to `lambda _x: list(file_types)`.

### F3. `VocabularyTerm.__init__` pops `Name`/`name` and stores in `PrivateAttr` [P3]

`ermrest.py:213-228`. The pattern works but is the kind of
manual private-attr juggling Pydantic v2 added `model_validator`
to replace. The `Name`/`Synonyms`/`Description` columns are
catalog-system; consider exposing them as standard fields with
`alias=` instead of `PrivateAttr`-via-`__init__`.

---

## `core/connection_mode.py` — mode enum (34 LoC)

### CM1. The `__init__` of every `DerivaML` calls `ConnectionMode(mode)` to coerce [P3]

`base.py:296, mixins/execution.py:163`. The coercion happens at the
two real consumers; the module is otherwise standalone. Fine —
flagging only because the spec §2.1 reference in the docstring
points to a spec doc that isn't in-tree (it lives in
`docs/design/...` or similar). A pointer to the actual doc in
the module docstring would help future readers.

---

## `core/catalog_stub.py` — offline catalog stand-in (43 LoC)

### CS1. `__getattr__` returns `AttributeError` for `_` names but the test only checks `__repr__`/`__str__` [P2]

`catalog_stub.py:30-38`, `test_catalog_stub.py:30-43`. Line 36-37:

```python
if name.startswith("_"):
    raise AttributeError(name)
```

The test exercises `repr(stub)` and `str(stub)` (which go through
`__repr__`, not `__getattr__`). Add a positive test: `stub.
_some_attr` should `raise AttributeError`, not
`DerivaMLReadOnlyError`. Pin the dunder-vs-userattr contract.

---

## `core/async_helpers.py` — sync/async bridge (73 LoC)

### A1. `run_async` has a runnable doctest but no proper test [P2]

`async_helpers.py:35-73`. The module docstring example and the
function doctest both demonstrate the happy path. Missing:

- The `nest_asyncio.apply()` path (when a loop is already running).
  This is the actual reason the module exists — Jupyter/papermill
  compatibility.
- The `Awaitable that raises` path, asserting the exception
  surfaces unchanged.

A test using `asyncio.new_event_loop()` and a coroutine that
raises would exercise the loop-detection-and-fallback path.

---

## `core/upload_layout.py` — directory layout helpers (330 LoC)

### UL1. `asset_table_upload_spec` calls `model.name_to_table(asset_table)` four times [P2]

`upload_layout.py:215, 216, 226, 227`. The redundant assignments at
226-227

```python
asset_table = model.name_to_table(asset_table)
schema = model.name_to_table(asset_table).schema.name
```

are exact duplicates of lines 215-216 with no intervening mutation.
Lines 226-227 should be deleted; both `asset_table` and `schema`
are unchanged.

### UL2. `NULL_SENTINEL` is module-level but only consumed by `asset/null_sentinel_processor.py` [P3]

`upload_layout.py:67`. The constant is defined here as a
convenience for the cross-module producer/consumer pair, but the
production path (`bag_commit`) bypasses it (docstring: "does not
need this sentinel"). Either drop the constant or move it to
`asset/null_sentinel_processor.py` so the legacy/production path
distinction is co-located with the consumer.

### UL3. `DEFAULT_UPLOAD_TIMEOUT` at line 313 is not exported in `__all__` [P3]

`upload_layout.py:90-102`. It's referenced via dotted import by
upload code; either expose it via `__all__` or rename with a
leading underscore.

### UL4. `upload_root` mkdirs unconditionally on read [P2]

`upload_layout.py:105-109`. Calling `upload_root(prefix)` always
creates the `deriva-ml` directory under `prefix`. Read-side users
that only want to *check* whether the upload root exists are
forced to create it as a side effect. Consider an `exist_ok=True,
parents=True, create=False` knob, or a separate `upload_root_
path(prefix) -> Path` that doesn't mkdir.

---

## `core/mixins/path_builder.py` — path builder (164 LoC)

### PB1. `_domain_path` returns annotation is `datapath.DataPath` but the body returns `pathBuilder().schemas[schema]` which is a `_Schema` wrapper [P3]

`path_builder.py:77-98`. Minor type-annotation lie. Either correct
the return type or drop the annotation; the lying annotation
makes IDE autocompletion useless inside `_domain_path` users.

### PB2. `get_table_as_dict` is a generator that wraps a non-generator `entities().fetch()` [P3]

`path_builder.py:145-164`. The function `yield from`s the fetch
result; the return type is `Iterable[dict[str, Any]]`. Combined
with `get_table_as_dataframe`'s `list(self.get_table_as_dict(table))`
through `rows_to_dataframe`, the generator yields twice. Fine —
flagging only because the generator-ness is invisible to callers
who think they're calling a fetch and getting a list.

---

## `core/mixins/rid_resolution.py` — RID resolution (213 LoC)

### RR1. `resolve_rids` raises `DerivaMLException(f"Invalid RIDs: {remaining_rids}")` — should be a typed exception [P1]

`rid_resolution.py:210-211`. The message-string coupling at
`validation.py:257` (see V2) depends on this exact wording. The
codebase has `DerivaMLNotFoundError` and a `DerivaMLInvalidTerm`-
style pattern with structured attributes (the missing RIDs). A
proper `DerivaMLNotFoundError` subclass — e.g.,
`DerivaMLRidsNotFound(missing_rids: set[RID])` — would replace
the string-parsing hack at the validation layer.

### RR2. `BatchRidResult` is a `@dataclass` but is part of the public API [P2]

`rid_resolution.py:40-54`. Per the CLAUDE.md guidance, user-facing
return types should be Pydantic `BaseModel`, not `@dataclass`.
`BatchRidResult` is returned from `resolve_rids` (a public
method); callers may need to serialize it. Convert to a Pydantic
model.

### RR3. `resolve_rid` swallows `KeyError` and rewraps as `DerivaMLException("Invalid RID {rid}")` [P3]

`rid_resolution.py:103-104`. Lossy — the original `KeyError`'s
context is dropped. Add `from _e`.

### RR4. `resolve_rids` returns empty dict for empty input; doctest shows the call pattern but no test asserts the empty-in / empty-out invariant [P3]

`rid_resolution.py:155-157`, `tests/core/test_rid_resolution.py:
50-54`. There *is* a test (`test_resolve_rids_empty`), so this is
covered. Flagging that the test imports `test_ml` only to call
`.resolve_rids([])`, which doesn't need a live catalog — the test
could be a pure unit test against a `RidResolutionMixin`-only
harness.

---

## `core/mixins/vocabulary.py` — vocabulary management (436 LoC)

### VM1. `lookup_term` docstring claims `DerivaMLVocabularyException` is raised [P1]

`vocabulary.py:217`. The class doesn't exist. The actual exception
raised at line 235 is `DerivaMLException("The table {table} is
not a controlled vocabulary")`. Fix the docstring to name
`DerivaMLException` (or, better, add `DerivaMLTableTypeError` —
see VM2).

### VM2. Inconsistent exception type for "table is not a vocabulary" guard [P1]

`vocabulary.py:160-161` (`add_term`), `vocabulary.py:234-235`
(`lookup_term`), `vocabulary.py:327-328`
(`list_vocabulary_terms`). `add_term` raises
`DerivaMLTableTypeError("vocabulary", vocab_table.name)`; the
other two raise the generic `DerivaMLException("The table ... is
not a controlled vocabulary")`. Same condition, three call sites,
two exception types. Pick one (`DerivaMLTableTypeError`) and use
everywhere.

### VM3. `_get_vocab_cache` lazily initializes `self._vocab_cache` [P3]

`vocabulary.py:65-69`. `hasattr` check on `_vocab_cache` is
fragile if a subclass `__slots__`-restricts. Initialize in a
`_init_mixin_state` method called from `DerivaML.__init__`, or
use a class-level sentinel.

### VM4. `delete_term`'s association-count loop is O(N) over `vocab_table.find_associations()` × O(M) rows each [P2]

`vocabulary.py:417-429`. Each iteration does a separate filtered
fetch. For a vocabulary with many associations this is a 1+N
pattern. Either batch with a single `pathBuilder` query joined
across all association tables, or fetch only `RID` columns and
shortcircuit on first match. Not urgent (delete is rare) but the
same anti-pattern caused a perf bug in workflow types (see
`workflow.py:62-79` for the fixed version).

### VM5. `list_vocabulary_terms` raises generic `DerivaMLException` for non-vocab [P2]

`vocabulary.py:327-328`. See VM2 — same condition, wrong typed
exception.

### VM6. `VocabCache` type alias is exported but used only inside this module [P3]

`vocabulary.py:35, 40`. Either remove from `__all__` or document
intended consumers.

---

## `core/mixins/workflow.py` — workflow management (409 LoC)

### WF1. Class docstring claims `find_workflow_by_url` and `list_workflow_executions` [P1]

`workflow.py:37-40`. The actual method is `lookup_workflow_by_url`;
`list_workflow_executions` lives in `FeatureMixin`. Update the
docstring class summary; either move `list_workflow_executions`
to `WorkflowMixin` (its natural home) or stop documenting it in
the wrong class.

### WF2. No module-level logger; failures silently propagate [P2]

`workflow.py:1-30`. Other mixins (`rid_resolution`, `execution`)
have a module logger; this one doesn't. The `_add_workflow` catch-
all at line 212 (`except Exception as e:`) re-raises as
`DerivaMLException(f"Failed to insert workflow. Error: {error}")`
without logging. Add a logger.

### WF3. `_add_workflow` catches `Exception` and rewraps as `DerivaMLException` [P2]

`workflow.py:187-215`. The bare-except + rewrap loses stack info
that would be useful for debugging insertion failures. Either
narrow to the catalog-specific exception classes or add `from e`.

### WF4. `_add_workflow` uses `MLVocab.workflow_type` as a dict key (StrEnum → str coercion) [P3]

`workflow.py:206-207`. The `{MLVocab.workflow_type: ...}` form
relies on `StrEnum`'s `__eq__`/`__hash__` matching the string
value. Works but flags as "use `.value` explicitly" for clarity
and to match `assoc_path.filter(assoc_path.Workflow_Type == X)`
style.

---

## `core/mixins/dataset.py` — dataset management (1 481 LoC)

### DS1. `lookup_dataset` does a full table scan [P1]

`dataset.py:177-185`. The implementation:

```python
return [ds for ds in self.find_datasets(deleted=deleted)
        if ds.dataset_rid == dataset_rid][0]
```

This fetches **every** dataset row, builds `Dataset` objects for
all of them, then Python-filters to the one wanted. On a catalog
with hundreds of datasets this is O(N) network + O(N) Dataset
construction per `lookup_dataset` call. The natural fix is a
server-side filter:

```python
dataset_path = pb.schemas[...].tables["Dataset"]
records = list(dataset_path.filter(dataset_path.RID == dataset_rid)
              .entities().fetch())
```

`lookup_workflow` at `workflow.py:217-279` already uses this
pattern; mirror it here.

### DS2. `DatasetMixin` imports from `deriva_ml.config`, `deriva_ml.asset`, `deriva_ml.dataset` [P2]

`dataset.py:20-44`. A "core" mixin shouldn't depend on five other
top-level packages. The bootstrap/config-validation methods at
`dataset.py:563-1097` (~530 LoC) belong in their own module
(`deriva_ml.dataset.config_validation` or
`deriva_ml.config.bootstrap_runner`), not in the `DerivaML` base
class. Splitting the mixin would also halve the file size.

### DS3. Mixin docstring lists 7 methods; the file defines 21 [P2]

`dataset.py:75-83`. The class docstring claims 7 methods (`find_
datasets`, `create_dataset`, ...). The file actually defines 21
public methods including `validate_dataset_specs`,
`validate_execution_configuration`, `validate_config_file`,
`validate_config_directory`, `bootstrap_config`, `cache_dataset`,
`estimate_bag_size`, `bag_info`, `estimate_denormalized_size`,
`download_dataset_bag`. Update the docstring (and consider DS2).

### DS4. `add_dataset_element_type` mutates the model and rebuilds workspace ORM under a `getattr(... "_workspace", None) is not None` guard [P2]

`dataset.py:289-300`. The guard is defensive against the workspace
not having been initialized; fine. But the comment block above
(285-296) is 12 lines of explanation for what is effectively a
"refresh the local ORM after a DDL change" operation. Extract to
a named helper (`_refresh_workspace_orm_after_ddl()`) and inline-
document the helper, not the call site.

---

## `core/mixins/execution.py` — execution management (1 253 LoC)

### EX1. Module is 1 253 LoC with one mixin class — splits naturally [P2]

`execution.py`. The file mixes:

- Execution creation/resumption (`create_execution`,
  `resume_execution`)
- Registry queries (`list_executions`, `find_executions`,
  `pending_summary`, `find_incomplete_executions`,
  `gc_executions`)
- Experiment queries (`lookup_experiment`, `find_experiments`)
- Lineage queries (`lookup_lineage`, `_classify_rid`,
  `_producer_of_dataset`, etc.)
- Upload (`upload_pending`)

Four distinct concerns, each ~250 LoC. Split into
`execution_lifecycle.py`, `execution_registry.py`,
`experiment_queries.py`, `lineage_queries.py`. Same gain as DS2.

### EX2. `create_execution` raises `DerivaMLOfflineError`, but `refresh_schema` raises `DerivaMLReadOnlyError` for the same offline guard [P1]

`execution.py:170, base.py:555, 707`. Pick one. The dedicated
`DerivaMLOfflineError` (which extends `DerivaMLConfigurationError`)
exists specifically for "online-only operation"; `DerivaMLReadOnly
Error` is for "writes-on-read-only" (a different semantic).

---

## `core/mixins/feature.py` — feature management (647 LoC)

### FT1. `fetch_table_features`, `list_feature_values`, `select_by_workflow` are tombstone shims [P1]

`feature.py:525-565, 628-647`. Per the workspace CLAUDE.md rule
"No backwards-compat shims — if something is unused, delete it,"
these three methods that just `raise DerivaMLException("... has
been retired")` should be deleted. Callers see a clean
`AttributeError` from Python rather than a hand-rolled message;
either is fine, and the message can live in the changelog.

### FT2. `list_workflow_executions` lives in `FeatureMixin` [P2]

`feature.py:566-626`. The method resolves workflows and queries
executions — neither activity is feature-related. Move to
`WorkflowMixin` or `ExecutionMixin`. The current placement is the
historical artifact of the selector-factory pattern; the comment
at line 574-577 hints at this.

### FT3. No logger in this mixin [P3]

`feature.py:1-30`. See WF2.

---

## `core/mixins/annotation.py` — Chaise annotation helpers (980 LoC)

### AN1. Every public method has `@validate_call(config=VALIDATION_CONFIG)` [P3]

`annotation.py` (16 decorators across the file). The annotation
mixin is the only one that uniformly decorates every method.
Other mixins decorate selectively. This is fine — annotations
take user-supplied dicts that benefit from validation — but the
inconsistency with the rest of the codebase is jarring. Either
extend the pattern to the other mixins or note the rationale in
the module docstring.

### AN2. `STRICT_PREALLOCATED_RID_TAG` annotation tag is module-level [P3]

`annotation.py:45`. The other annotation tags (DISPLAY_TAG, etc.)
are also module-level — fine. The tag with `2026:` in its URI
makes me wonder if it'll be stable across releases. Document the
versioning policy.

---

## `core/mixins/file.py` — file management (268 LoC)

### FL1. `add_files`'s last dataset is returned but the loop builds many [P2]

`file.py:155-176`. The function creates one `Dataset` per
directory level, then returns `dataset` (the last one built). The
docstring says "Returns: Dataset that represents the newly added
files" — but if the input spans multiple directories, the returned
dataset only represents the *innermost* level. Either return the
top-level dataset or return a list of all created datasets.

### FL2. `add_files` re-builds `defined_types` per call [P3]

`file.py:101-106`. The set-built-from-list-comprehension is
recomputed on every `add_files` call. For a long-running script
with many `add_files` invocations, vocabulary-list fetches
dominate. Cache the result (with explicit invalidation on
`add_term`).

---

## Cross-module observations

### X1. Logger construction is inconsistent across the core/ subsystem [P2]

- `core/base.py`, `core/mixins/rid_resolution.py`,
  `core/mixins/execution.py`, `core/validation.py`: module logger
  via `logger = get_logger(__name__)` (canonical).
- `core/mixins/{annotation,asset,dataset,feature,file,
  path_builder,vocabulary,workflow}.py`: no logger.
- `dataset/restructure.py:63`: `logger = logging.getLogger(
  __name__)` — direct stdlib bypass of `get_logger`.

The canonical-vs-bypass split should be unified.

### X2. `@validate_call` adoption is uneven [P2]

26 decorators total. `annotation.py` has 16 (one per public
method); `vocabulary.py` has 3; `feature.py` has 2 (on
`feature_values` and `list_workflow_executions`); `dataset.py`
has 1; `asset.py` has zero with a comment saying decorators were
removed because Pydantic doesn't validate dataclass fields well
(`asset.py:66-67`); `upload_layout.py` has 1. The "validate user
input" guarantee is partial. Either commit to a uniform policy
(decorator everywhere, or compose Pydantic models at the boundary
and skip `@validate_call`) or document where the boundary is
intentional.

### X3. The "mode" guard pattern is duplicated across `refresh_schema`/`diff_schema`/`create_execution` [P3]

`base.py:554-555, 706-707, execution.py:168-173`. Three nearly-
identical "raise if mode is not online" guards. Extract to a
decorator (`@require_online`) or a guard helper (`self._require_
online("refresh_schema")`).

### X4. `MLVocab` enum value is sometimes used as a string and sometimes via `.value` [P3]

Throughout the mixins. `pathBuilder.schemas[ML_SCHEMA]` works
because StrEnum auto-coerces; `{MLVocab.workflow_type: ...}` as a
dict key works the same way. Inconsistency hurts readability —
pick one (the `.value` form is the unambiguous one, but the bare
form reads more cleanly). Document the convention.

### X5. `Status` is gone but `enums.py` module docstring still claims it [P3]

`enums.py:8`. The "Classes:" block lists `Status: Execution status
values.` Phase 2 deleted the class (per comment at lines 67-71);
the docstring summary lagged the change. Delete the line.

### X6. Pure-Python tests prefer `_StorageHarness`/`_CiteHarness`-style minimal mocks, but live tests still build full `test_ml` instances for things that don't need them [P3]

`test_storage_management.py`, `test_base.py::_CiteHarness` show
the pattern: build a `SimpleNamespace`-shaped object exposing only
the attributes the method under test reads. Apply the same
pattern to `test_rid_resolution.py::test_resolve_rids_empty` (no
catalog needed) and `test_catalog_annotations.py` tests where the
assertion only inspects `annotations` dicts.

### X7. The Async path in `async_helpers.py` is imported lazily but never tested in the lazy-import branch [P3]

`async_helpers.py:67-72`. If `nest_asyncio` is *not* installed,
the lazy import inside the if-loop branch raises `ImportError`.
Document this as a graceful-degradation contract and add a
package extra (`pip install deriva-ml[notebook]`).

---

## Tests that need to be added

(consolidating the test gaps surfaced above)

1. **`tests/core/test_validation.py`** — direct coverage for
   `ValidationResult`, `validate_rids`, `validate_execution_config`.
   See V1.
2. **`tests/core/test_logging_config.py`** — `configure_logging`
   side effects under no-Hydra / Hydra / custom-handler. See L1.
3. **`tests/core/test_filespec.py`** — regression test for F1
   (multi-file length bug), plus tests for `read_filespec` round-
   trip and URL normalisation paths.
4. **`tests/core/test_async_helpers.py`** — `run_async` with no
   loop, with a running loop, and with a coroutine that raises.
   See A1.
5. **`tests/core/test_constants.py`** — pin the `rid_regex`
   against expected positive/negative cases.
6. **`tests/core/test_upload_layout.py`** — no test file exists.
   `asset_table_upload_spec`, `bulk_upload_configuration`,
   `manifest_path`, `table_path` all uncovered.
7. **Module-level `__all__` audit** — a simple test that imports
   each module and asserts every name in `__all__` resolves. The
   `DerivaMLVocabularyException` ghost (VM1) would surface
   immediately.

---

## Files touched

- `src/deriva_ml/core/base.py` (1 675 LoC)
- `src/deriva_ml/core/exceptions.py` (780 LoC)
- `src/deriva_ml/core/validation.py` (427 LoC)
- `src/deriva_ml/core/logging_config.py` (221 LoC)
- `src/deriva_ml/core/schema_cache.py` (218 LoC)
- `src/deriva_ml/core/schema_diff.py` (321 LoC)
- `src/deriva_ml/core/config.py` (233 LoC)
- `src/deriva_ml/core/definitions.py` (181 LoC)
- `src/deriva_ml/core/enums.py` (182 LoC)
- `src/deriva_ml/core/constants.py` (182 LoC)
- `src/deriva_ml/core/ermrest.py` (325 LoC)
- `src/deriva_ml/core/filespec.py` (190 LoC)
- `src/deriva_ml/core/connection_mode.py` (34 LoC)
- `src/deriva_ml/core/catalog_stub.py` (43 LoC)
- `src/deriva_ml/core/async_helpers.py` (73 LoC)
- `src/deriva_ml/core/upload_layout.py` (330 LoC)
- `src/deriva_ml/core/sort.py` (100 LoC)
- `src/deriva_ml/core/pd_utils.py` (39 LoC)
- `src/deriva_ml/core/__init__.py` (67 LoC)
- `src/deriva_ml/core/mixins/__init__.py` (42 LoC)
- `src/deriva_ml/core/mixins/annotation.py` (980 LoC)
- `src/deriva_ml/core/mixins/asset.py` (452 LoC)
- `src/deriva_ml/core/mixins/dataset.py` (1 481 LoC)
- `src/deriva_ml/core/mixins/execution.py` (1 253 LoC)
- `src/deriva_ml/core/mixins/feature.py` (647 LoC)
- `src/deriva_ml/core/mixins/file.py` (268 LoC)
- `src/deriva_ml/core/mixins/path_builder.py` (164 LoC)
- `src/deriva_ml/core/mixins/rid_resolution.py` (213 LoC)
- `src/deriva_ml/core/mixins/vocabulary.py` (436 LoC)
- `src/deriva_ml/core/mixins/workflow.py` (409 LoC)
- `tests/core/` (20 files, ~3 100 LoC)

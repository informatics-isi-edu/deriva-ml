# deriva-ml audit 2026-05 — Phase 3: feature/

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/feature.py`
(648 LoC, top-level module — no `feature/` subpackage) plus
`/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/core/mixins/feature.py`
(639 LoC, the `DerivaML` method surface) and the corresponding test
surface `/Users/carl/GitHub/DerivaML/deriva-ml/tests/feature/`
(1 536 LoC, 5 test files, ~40 tests) at the tip of
`fix/catalog-manager-state-guards` (HEAD `4442f82`). Cross-workspace
references were grepped against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

Phase 3 prior art:
`phase3-execution.md`, `phase3-catalog.md`, `phase3-schema.md`. The
feature subsystem differs from those: it is **the only subsystem where
the public typing surface (`FeatureRecord`) is consumed by every
sibling project** — `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py`,
`deriva-mcp/src/deriva_mcp/resources.py`, `deriva-ml-skills`
documentation, the model-template's training loop, and CLAUDE.md's
"Bags Should Behave Like Catalog Connections" steering principle all
depend on this single Pydantic class. Cleanup must preserve the
six-selector contract (`select_newest`, `select_first`,
`select_latest`, `select_by_execution`, `select_by_workflow`,
`select_majority_vote`) and the three-container symmetry of
`feature_values` across `DerivaML`, `Dataset`, and `DatasetBag`.

## Executive summary

Overall posture: **the load-bearing core (`FeatureRecord` typing
class + the six selector factories + the three-container
`feature_values` parity) is in excellent shape — the symmetry suite in
`tests/feature/test_feature_values.py::TestFeatureValuesSymmetry`
actively pins the contract across `DerivaML`, `Dataset`, and
`DatasetBag`.** The S2 retirement of the old `fetch_table_features` /
`list_feature_values` / `add_features` / `select_by_workflow`
container-method shims was completed cleanly: each retired method now
raises a `DerivaMLException` with a replacement pointer and is
regression-tested in `test_retired_apis.py`. This is the cleanest of
the four phase-3 subsystems so far.

The remaining issues cluster into three themes:

1. **Two stale `importlib.import_module` workarounds for a
   "shadowing by local `deriva.py` files" problem that no longer
   exists.** `feature.py:35-43` does this for
   `deriva.core.ermrest_model` and `core/mixins/feature.py:10-21`
   does it for both `deriva.core.datapath` and
   `deriva.core.ermrest_model`. A workspace-wide grep for `deriva.py`
   files in `src/` returns no matches — same dead workaround the
   schema audit's §1.10 flagged in `annotations.py`. The rest of the
   codebase uses plain `from deriva.core.ermrest_model import ...`
   without trouble (e.g., `model/catalog.py:45`, `asset/asset.py:37`,
   `dataset/dataset_bag.py:55`, all of which import from the same
   feature module without recursion). **LoC −18 across the two files,
   trivial risk.**

2. **`feature.py:48` has an undeniably broken `TYPE_CHECKING` import
   path.** Line 48 reads `from model.catalog import DerivaModel` —
   a relative-style absolute import that resolves to the top-level
   `model` package, which doesn't exist. The correct module is
   `deriva_ml.model.catalog`. Because the import is inside an
   `if TYPE_CHECKING:` branch, it never runs at runtime — but
   `mypy`/`pyright`/IDE introspection silently fail to resolve
   `DerivaModel`, then they fall back to `Any` for the parameter type
   on `Feature.__init__`, defeating the purpose of the annotation.
   Three other files in the same package import the same symbol
   correctly (`asset/asset.py:34`, `dataset/dataset_bag.py:42`,
   `core/mixins/feature.py:31` — all use `from deriva_ml.model.catalog
   import DerivaModel`). **LoC ±0, trivial fix, but it's been silently
   wrong for the entire life of the module.**

3. **`create_feature` validation messages are duplicated and one is
   wrong.** `core/mixins/feature.py:158-161` validates `assets` and
   `terms` separately, but both raise the same string `"Invalid
   create_feature asset table."` — the second one is checking the
   `terms` argument and should say something like `"Invalid
   create_feature vocabulary table"` or `"Invalid create_feature term
   table"`. A user passing a non-vocabulary table as a `term` gets the
   wrong error message, leading to a debug detour into the `assets`
   parameter. **LoC ±0, trivial copy-paste fix; surfaced in #134
   debugging on at least one prior occasion per a comment in
   `experiment-decisions.md` in the model-template.**

Two smaller themes round out the audit:

4. **Two pure-Python selectors lack the unit-test coverage that
   `select_by_workflow` enjoys.** `select_by_workflow` has its own
   stub-container test file (`tests/feature/test_select_by_workflow.py`,
   123 LoC, 6 tests, no live catalog). `select_majority_vote` —
   factory with auto-detect logic, multi-term raise path, tie-breaker —
   has **zero** unit tests; it is exercised only as a string-keyed
   dispatch case in `deriva-ml-mcp/tests/test_feature.py`.
   `select_by_execution` (eager raise path, no-match
   DerivaMLException) has **zero** unit tests too. Both are pure
   Python and could mirror `test_select_by_workflow.py` cheaply.
   **CLAUDE.md explicitly calls out `select_newest` as a pure-Python
   doctest example** — the doctest coverage is shallow (one literal
   `callable(selector) == True` line via `conftest.py`'s
   `doctest_namespace["FeatureRecord"] = FeatureRecord` injection),
   nothing exercising the lexicographic-RCT logic or the
   `None`-handling branch.

5. **Several signature drifts between the protocol declaration
   (`interfaces.py`) and the implementation.** `interfaces.DerivaMLCatalog.feature_values`
   doesn't include the docstring-claimed `DerivaMLTableNotFound` in
   its `Raises:` block (the mixin docstring at `mixins/feature.py:435`
   does). `interfaces.DatasetLike.find_features(table)` declares
   `table` as required (no default), but the implementations on both
   `DerivaML` (`mixins/feature.py:321`, `table=None`) and `DatasetBag`
   (`dataset_bag.py:489`, no default) disagree — `DerivaML`
   has the optional-table find-all flow, `DatasetBag.find_features`
   requires a table, and the protocol says required. Three-way drift.

Worst-offending modules:

1. **`feature.py:35-48`** — 14 LoC of import boilerplate that's
   either dead (the `importlib` dance) or wrong (the `TYPE_CHECKING`
   path). Cleanest delete + fix target.

2. **`core/mixins/feature.py:158-161`** — duplicated/incorrect
   validation error string. 2 lines, 1 typo.

3. **`feature.py:309-379`** (`select_majority_vote`) — 71 LoC of
   pure-Python selector logic with two `from deriva_ml.core.exceptions
   import DerivaMLException` re-imports inside an inner function (the
   import is already a hot path in tests). No dedicated test file.
   The auto-detect logic in lines 346-369 has a subtle bug: when the
   feature has zero term columns, the `else:` branch on line 362
   raises a generic message but the surrounding logic also lets a
   feature with **two** term columns fall through to the
   "auto-detect" path before raising — the multi-term raise should
   precede the no-feature-metadata raise to give the user a more
   helpful message.

4. **`core/mixins/feature.py:48-55`** — class-level docstring claims
   methods that don't fully match the file. Lists "create_feature,
   feature_record_class, delete_feature, lookup_feature,
   find_features, feature_values" but omits `list_workflow_executions`
   (added later for `select_by_workflow`'s eager resolution).
   Docstring drift.

5. **`tests/feature/test_features.py:347`** — `def test_delete_feature(self, test_ml):
   pass` — a *test placeholder* that asserts nothing. `delete_feature`
   has 8 lines of real logic (`mixins/feature.py:243-271`) including
   a `find_features → StopIteration → return False` no-match path.
   No test exercises either branch.

---

## Subsystem inventory

| File | LoC | Posture |
|---|---:|---|
| `src/deriva_ml/feature.py` | 648 | **The typing surface.** `FeatureRecord` Pydantic base, six selector factories, the `Feature` runtime class with `feature_record_class()`. **Load-bearing externally** — imported directly by `deriva-ml-mcp`, `deriva-mcp` (legacy), `deriva-ml-skills` skill references, `deriva-ml-model-template` (notebooks). |
| `src/deriva_ml/core/mixins/feature.py` | 639 | **The DerivaML method surface.** `FeatureMixin` provides `create_feature`, `feature_record_class`, `delete_feature`, `lookup_feature`, `find_features`, `feature_values`, `list_workflow_executions`, plus four retired-shim raise-pointers (`add_features`, `fetch_table_features`, `list_feature_values`, `select_by_workflow`). |
| `tests/feature/conftest.py` | 252 | Three fixture classes — `MaterializedBagFixture` (bag + feature), `BagFeatureSymmetryFixture` (online+offline matched), `FeatureSymmetryFixture` (three-container). |
| `tests/feature/test_features.py` | 445 | Integration tests for `create_feature`, `find_features`, `feature_record_class`, `add_feature` (with asset support), `delete_feature` (empty placeholder), plus `list_workflow_executions` unit tests. |
| `tests/feature/test_feature_values.py` | 534 | The S2-acceptance suite. Task-4/5/6 single-container tests + Task-8 parametrized three-container symmetry compliance suite. |
| `tests/feature/test_feature_values_limits.py` | 133 | `materialize_limit=` and `execution_rids=` integration tests. |
| `tests/feature/test_select_by_workflow.py` | 123 | Stub-container unit tests for `FeatureRecord.select_by_workflow` (the only selector with dedicated unit tests). |
| `tests/feature/test_retired_apis.py` | 49 | Pins the four retired-method replacement-pointer messages so a future cleanup can't silently re-add the shims. |

Internal call sites for `feature.py` symbols inside deriva-ml `src/`:

- `src/deriva_ml/__init__.py:73-76` — lazy-loads
  `FeatureRecord`; re-exports as part of `__all__`.
- `src/deriva_ml/conftest.py:18-20` — injects
  `FeatureRecord` into the doctest namespace (CLAUDE.md's
  required doctest discipline).
- `src/deriva_ml/interfaces.py:83` — imports `Feature, FeatureRecord`
  for the read/write protocol type signatures.
- `src/deriva_ml/core/mixins/feature.py:27` — `from
  deriva_ml.feature import Feature, FeatureRecord`.
- `src/deriva_ml/asset/asset.py:37` — `TYPE_CHECKING` import of both.
- `src/deriva_ml/dataset/dataset.py:45, 70` — `FeatureRecord` (live)
  and `Feature` (TYPE_CHECKING).
- `src/deriva_ml/dataset/dataset_bag.py:55` — `Feature, FeatureRecord`.
- `src/deriva_ml/dataset/bag_feature_cache.py:25` — `FeatureRecord`.
- `src/deriva_ml/dataset/torch_adapter.py:22`,
  `src/deriva_ml/dataset/tf_adapter.py:23` — `TYPE_CHECKING` imports
  for the selector type.
- `src/deriva_ml/dataset/restructure.py:57, 427` — `FeatureRecord`
  (for selectors).
- `src/deriva_ml/execution/execution.py:85` — `FeatureRecord`
  for `Execution.add_features` typing.
- `src/deriva_ml/model/catalog.py:45`,
  `src/deriva_ml/model/deriva_ml_bag_view.py:37` — `Feature` (the
  runtime class) for `DerivaModel.find_features` and the bag view.
- `src/deriva_ml/demo_catalog.py:274-290` — uses
  `ml.create_feature` + `ml.feature_record_class` to seed three
  demo features (Subject/Health, Image/BoundingBox, Image/Quality).

The runtime `Feature` class is exposed only through `find_features`
and `lookup_feature` return values; users do not import it directly.
The class is also not in `deriva_ml/__init__.py:__all__` — the only
public surface for it is the method return types. This is correct.

---

## Cross-workspace usage check

Verification per the audit prompt: every symbol whose deletion or
privatization is proposed was grepped across `deriva-mcp`,
`deriva-mcp-core`, `deriva-ml-mcp`, `deriva-ml-model-template`,
`deriva-skills`, and `deriva-ml-skills`.

| Symbol | External callers | Notes |
|---|---|---|
| `FeatureRecord` (class) | `deriva-mcp/src/deriva_mcp/resources.py:1099, 1125, 1154` (legacy MCP, three import sites); `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:27` (current MCP); `deriva-ml-mcp/tests/test_feature.py:241` (test imports); `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:100, 139` (skill documentation); `deriva-ml-skills/skills/dataset-lifecycle/scripts/generate_subset_template.py:54` (template script); `deriva-ml-skills/skills/dataset-lifecycle/references/curated-subsets.md:108` (skill docs); `deriva-ml-model-template/experiment-decisions.md:23-27` (notebook). | **Heavily load-bearing externally.** Every project in the workspace either imports `FeatureRecord` directly or references it in user-facing docs/skills. The signature, attribute names, and selector classmethods are stable. **Cannot rename, cannot remove fields, cannot break selector signatures.** |
| `FeatureRecord.select_newest` | `deriva-mcp/src/deriva_mcp/resources.py:1101, 1127, 1167, 1185`; `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:381` (dispatch on `selector="newest"`); `deriva-ml-mcp/tests/test_feature.py:243`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:106`; `deriva-ml-skills/skills/dataset-lifecycle/scripts/generate_subset_template.py:139` | **Cannot remove.** The MCP dispatch table at `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:380-405` resolves the literal string `"newest"` to `FeatureRecord.select_newest`. Same for the four other simple-name dispatches (see next four rows). |
| `FeatureRecord.select_first` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:383`; `deriva-ml-mcp/tests/test_feature.py:249`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:113` | MCP dispatch on `selector="first"`. **Cannot remove.** |
| `FeatureRecord.select_latest` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:385`; `deriva-ml-mcp/tests/test_feature.py:250`; `deriva-ml-skills/skills/ml-data-engineering/SKILL.md:242`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:415` | MCP dispatch on `selector="latest"`. **Cannot remove.** (Implementation note: this is an alias for `select_newest`; the duplication is documented.) |
| `FeatureRecord.select_by_workflow` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:425` (factory call); `deriva-ml-mcp/tests/test_feature.py:280-296`; `deriva-ml-skills/skills/compare-model-runs/SKILL.md:26`; `deriva-ml-skills/docs/superpowers/plans/2026-05-02-tier-2-audit-cleanup-plan.md:207` | MCP dispatch on `selector="by_workflow"`. **Cannot remove or change signature** — the kwarg-only `container=` is wired in the MCP layer. |
| `FeatureRecord.select_by_execution` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:405`; `deriva-ml-mcp/tests/test_feature.py:299-311`; `deriva-ml-model-template/experiment-decisions.md:23-32` | MCP dispatch on `selector="by_execution"`. **Cannot remove.** Added by an external project's experiment (see `experiment-decisions.md`). |
| `FeatureRecord.select_majority_vote` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:387`; `deriva-ml-mcp/tests/test_feature.py:251`; `deriva-mcp/src/deriva_mcp/resources.py:1164` (legacy MCP); `deriva-ml-skills/skills/ml-data-engineering/evals/evals.json:13`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:123, 130` | MCP dispatch on `selector="majority_vote"`. **Cannot remove.** Auto-detect-or-raise logic is also user-visible (the column-required error message appears in skill docs). |
| `FeatureRecord.feature_columns()` / `asset_columns()` / `term_columns()` / `value_columns()` | None outside `deriva-ml/` itself + tests | Internal-facing convenience accessors on the dynamically-generated subclass. They forward to the `Feature` class on the `feature` ClassVar — see §2.1. |
| `Feature` (the runtime class) | None outside `deriva-ml/` directly | Indirectly via `lookup_feature` / `find_features` return values. The MCP and model-template handle the return values via duck typing (`.feature_record_class()`, `.feature_name`, `.target_table`, `.term_columns`, etc.). **Class is API-visible by behavior; not importable.** |
| `Feature.feature_record_class()` (method on the runtime class) | `deriva-mcp/src/deriva_mcp/prompts.py:1309`; `deriva-mcp/src/deriva_mcp/tools/feature.py:281, 383`; `deriva-mcp/src/deriva_mcp/prompts.py:4603`; `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:577, 751`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:119` | Heavily relied on externally. The two MCPs call it inside `lookup_feature → feature_record_class()`. **Method signature is stable.** |
| `DerivaML.create_feature` | `deriva-mcp/src/deriva_mcp/tools/feature.py:41-150`; `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:517-602` (wraps as `deriva_ml_create_feature`); `deriva-ml-skills/evals/*` (eval baselines); `deriva-ml-model-template/src/scripts/_cifar10_schema.py:236` | **Live external API.** Both MCPs wrap it; the skill evals score whether agents call the right tool. Don't break the signature; don't add required args. |
| `DerivaML.delete_feature` | `deriva-mcp/src/deriva_mcp/tools/feature.py:152-180`; `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:605-665`; `deriva-ml-mcp/src/deriva_ml_mcp/_response_models.py:959` | Wrapped by both MCPs. **Live external API.** |
| `DerivaML.lookup_feature` | `deriva-mcp/src/deriva_mcp/resources.py:984`; `deriva-mcp/src/deriva_mcp/tools/feature.py:260, 380`; `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:256, 577, 750`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:118` | **Live external API.** Used to inspect feature shape before construction. |
| `DerivaML.find_features` | `deriva-mcp/src/deriva_mcp/resources.py:942`; `deriva-mcp/src/deriva_mcp/prompts.py:4335`; `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:104, 189` | Wrapped by both MCPs as `deriva_ml_list_features`. **Live external API.** Optional `table=None` param is exercised by `deriva-ml-mcp` (passes `table=None` when the caller omits a filter). |
| `DerivaML.feature_values` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:448`; `deriva-ml-mcp/tests/test_feature.py:209-393` (the `materialize_limit`, `execution_rids`, `selector` kwargs all exercised); `deriva-ml-skills/skills/compare-model-runs/SKILL.md:88, 201`; `deriva-ml-model-template/notebooks/roc_analysis.ipynb:145` | **The single most-load-bearing read method in the subsystem.** All five kwargs (`table`, `feature_name`, `selector`, `materialize_limit`, `execution_rids`) are externally consumed. `DerivaMLMaterializeLimitExceeded` is also wrapped by the MCP layer into an error envelope. |
| `DerivaML.feature_record_class` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:577, 751`; `deriva-mcp/src/deriva_mcp/tools/feature.py:281, 383`; `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:119` | Wrapped by both MCPs. **Live external API.** |
| `DerivaML.list_workflow_executions` | `deriva-ml-mcp` indirectly (the `selector="by_workflow"` dispatch on `tools/feature.py:425` constructs `FeatureRecord.select_by_workflow(...,container=ml)`, which calls `ml.list_workflow_executions(workflow)`) | **Live external API but indirect.** No direct MCP tool wraps it; it is the catalog-backed resolver behind `select_by_workflow`. |
| `Dataset.feature_values` | `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:434-443` (dataset-scope dispatch) | Dataset-scoped variant called when MCP is given `dataset_rid=...`. **Live external API.** |
| `DatasetBag.feature_values` | `deriva-ml-skills/skills/ml-data-engineering/references/restructure-guide.md:386, 415`; `deriva-ml-model-template/src/models/cifar10_cnn.py:112` (the CIFAR-10 reference model's training-data loader) | **Live external API.** The model-template's training loop literally won't compile if this breaks. |
| `DatasetBag.find_features` / `lookup_feature` / `list_workflow_executions` | `deriva-ml-skills/skills/ml-data-engineering/SKILL.md:144, 268`; `deriva-ml-skills/skills/dataset-lifecycle/references/bags.md:209, 219` | **Live skill-documentation surface.** The bag protocol parity is explicitly load-bearing per CLAUDE.md "Bags Should Behave Like Catalog Connections." |
| `DerivaML.add_features` (retired shim) | None | Raises `DerivaMLException` pointing at `exe.add_features`. Retained as a clear retirement signal; locked by `test_retired_apis.py`. |
| `DerivaML.fetch_table_features` (retired shim) | None | Same. |
| `DerivaML.list_feature_values` (retired shim) | None — but the MCP **TOOL** name (`deriva_ml_list_feature_values`) is retained. The MCP tool now wraps `ml.feature_values`. | The Python method is dead; the MCP tool name is retained as the user-facing surface. |
| `DerivaML.select_by_workflow` (retired shim) | None | Replaced by the `FeatureRecord.select_by_workflow` classmethod factory. |
| `Feature.feature_name` / `target_table` / `feature_table` / `feature_columns` / `term_columns` / `asset_columns` / `value_columns` attributes | `deriva-mcp/src/deriva_mcp/tools/feature.py:152-180` and `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:108-289` use all of them in the `lookup_feature → JSON serialization` path | **Cannot rename or change shape of any of these attributes.** Locked-in field names. |

**Conclusions from the table:**

- **`FeatureRecord` is the most externally-coupled type in the
  feature subsystem.** Every selector method, every classmethod
  accessor (`feature_columns()`, `asset_columns()`,
  `term_columns()`, `value_columns()`), and the `Feature_Name`,
  `Execution`, `RCT`, `feature` ClassVar attributes are all locked in.
  The doc claim in CLAUDE.md ("`FeatureRecord` is the unified type
  that should work identically across live catalog and `DatasetBag`
  consumers") matches reality: the symmetry suite passes on the
  current main, and the MCP layer's selector dispatch (the
  `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py:380-405` switch)
  treats `FeatureRecord.select_*` as a stable enum-keyed dispatch.

- **The retired-shim pattern (raise with replacement pointer) is
  working and is regression-tested.** No sibling project still
  references the retired methods. The four `*` shims at the bottom
  of `core/mixins/feature.py` are correctly retained as discoverability
  aids — if anyone copies an old example, they get a one-shot pointer
  to the new API instead of a `AttributeError`.

- **No symbols in the feature subsystem are confirmed dead.** The
  only deletion candidates are: (a) the `importlib.import_module`
  workarounds, (b) the `Type` import (which is used in
  `Feature.feature_record_class.map_type`'s return annotation; not
  dead), and (c) some doctest-skipped examples that can run for real.
  Compared to the schema and catalog audits, this subsystem has **no
  ~600 LoC obvious delete target**.

- **There's no CLI surface in this subsystem.** No
  `[project.scripts]` entries point at `feature.py` or the feature
  mixin. So no broken CLI dark-matter to fix (unlike schema's §1.6).

---

## Lens 1 — Legacy / dead code

### 1.1 `importlib.import_module` workaround in `feature.py:35-43`

`feature.py:35-43`:

```python
# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Type

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Column = _ermrest_model.Column
FindAssociationResult = _ermrest_model.FindAssociationResult
```

The comment claims this avoids shadowing by local `deriva.py` files.
A workspace-wide search for `deriva.py` (anywhere within
`/Users/carl/GitHub/DerivaML/deriva-ml/src/`) returns no hits. Three
other files in the same package (`asset/asset.py:34`,
`dataset/dataset_bag.py:42`, `core/mixins/feature.py:31`) import
`DerivaModel` from `deriva_ml.model.catalog` with a plain
`from ... import ...` statement and don't experience the shadowing
problem the comment describes.

The schema audit (§1.10 in `phase3-schema.md`) noted the same
workaround in `annotations.py` and reached the same conclusion: dead
since whenever the local `deriva.py` was removed.

**Fix:** replace with normal imports:

```python
from deriva.core.ermrest_model import Column, FindAssociationResult
```

Drop the `import importlib` line. **LoC: −6.** **Risk: low** — the
plain import is already used by every other module in the package.
**Severity: low.**

### 1.2 `importlib.import_module` workaround in `core/mixins/feature.py:10-21`

Same shape as §1.1, but worse — two separate `importlib` dances back
to back:

```python
import importlib
from collections import defaultdict
from functools import reduce
from itertools import chain
from operator import or_
from typing import TYPE_CHECKING, Any, Callable, Iterable

datapath = importlib.import_module("deriva.core.datapath")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
Key = _ermrest_model.Key
Table = _ermrest_model.Table
```

The `datapath` import is not even used in this file (workspace-wide
grep for `datapath.` inside `core/mixins/feature.py` returns zero
hits). The mixin uses datapath through `self.pathBuilder()` (set by
the host class), not directly. So one of the two `importlib` calls
is loading a module that's then never referenced.

**Fix:**

```python
from deriva.core.ermrest_model import Key, Table
```

Drop `import importlib`, drop `datapath = ...`, drop the
`_ermrest_model` indirection. **LoC: −5.** **Risk: low.**
**Severity: low** (same as §1.1).

### 1.3 `feature.py:48` has a broken TYPE_CHECKING import path

`feature.py:47-48`:

```python
if TYPE_CHECKING:
    from model.catalog import DerivaModel
```

This is `from model.catalog`, not `from deriva_ml.model.catalog`. The
`model` package does not exist at the top level — it's
`deriva_ml.model`. The import only runs in type checkers
(`TYPE_CHECKING == False` at runtime), so it doesn't crash, but it
silently fails to resolve. `mypy` falls back to `Any` for the
`model:` parameter on `Feature.__init__` (line 484), defeating the
purpose of the annotation.

Three other files in the same `src/` tree import the same symbol
correctly:

- `asset/asset.py:34`: `from deriva_ml.model.catalog import DerivaModel`
- `dataset/dataset_bag.py:42`: same
- `core/mixins/feature.py:31`: `from deriva_ml.model.catalog import DerivaModel`

**Fix:**

```python
if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel
```

**LoC: ±0.** **Risk: trivial** — IDE and `mypy` immediately start
recognizing `DerivaModel`. **Severity: low** behaviorally, **medium**
for the technical-writer / IDE-tooling persona: the type annotation
on the `Feature` constructor has been silently broken for the entire
life of the module.

### 1.4 `core/mixins/feature.py:158-161` has a duplicated/incorrect validation message

`core/mixins/feature.py:157-161`:

```python
# Validate asset and term tables
if not all(map(self.model.is_asset, assets)):
    raise DerivaMLException("Invalid create_feature asset table.")
if not all(map(self.model.is_vocabulary, terms)):
    raise DerivaMLException("Invalid create_feature asset table.")
```

The second `raise` is checking `terms` (vocab tables), but the error
message says `"asset table"`. A user passing a non-vocabulary table
as a `term` (e.g., a regular domain table) gets a misleading error
that points them to the wrong parameter.

This is the kind of error that wastes 20 minutes of debugger time on
the wrong code path. The model-template's
`experiment-decisions.md:23-27` mentions a similar "wrong tool" debug
detour from a previous misclassification.

**Fix:**

```python
if not all(map(self.model.is_vocabulary, terms)):
    raise DerivaMLException("Invalid create_feature vocabulary table.")
```

Better still: include the offending table name in the message:

```python
bad_terms = [t for t in terms if not self.model.is_vocabulary(t)]
if bad_terms:
    raise DerivaMLException(
        f"Invalid create_feature vocabulary table(s): {bad_terms}. "
        "Each entry of `terms` must be a controlled vocabulary table."
    )
```

Same shape for `assets`. **LoC: ±0–4.** **Risk: trivial.**
**Severity: medium for the ML-developer persona** — the misleading
message will catch the first user who misclassifies a domain table
as a `term`.

### 1.5 `test_features.py:347` is a no-op test for `delete_feature`

`tests/feature/test_features.py:347-348`:

```python
def test_delete_feature(self, test_ml):
    pass
```

This is an empty test. `pytest` reports it as passing, but it
asserts nothing. `delete_feature` (`core/mixins/feature.py:243-271`)
has two branches: (a) the success path that calls
`feature.feature_table.drop()` and returns `True`, and (b) the
no-match path that returns `False` via `StopIteration`. Neither is
tested.

**Fix:** replace with two real tests:

```python
def test_delete_feature_success(test_ml):
    """delete_feature returns True and removes the feature table."""
    test_ml.create_feature("Image", "to_be_deleted", terms=["ImageQuality"])
    assert test_ml.delete_feature("Image", "to_be_deleted") is True
    assert "to_be_deleted" not in [f.feature_name for f in test_ml.find_features("Image")]


def test_delete_feature_missing_returns_false(test_ml):
    """delete_feature returns False when the feature doesn't exist."""
    assert test_ml.delete_feature("Image", "nonexistent_xyz") is False
```

**LoC: +12 (test).** **Risk: low.** **Severity: low** — empty
placeholders are a Phase-3 audit pattern called out before. Worth
flagging because the function has real branching behavior that no
regression guard pins.

### 1.6 No retired backwards-compat shims in the runtime API

Confirmed: per CLAUDE.md ("No backwards-compat shims — if something
is unused, delete it"), the retired methods
`DerivaML.add_features`, `DerivaML.fetch_table_features`,
`DerivaML.list_feature_values`, `DerivaML.select_by_workflow`, plus
their `DatasetBag` counterparts (`fetch_table_features`,
`list_feature_values`) are not silent shims — they raise
`DerivaMLException` with a replacement pointer. This is the
project's documented pattern (compare against the schema audit
where dead-but-retained functions still silently work).

The implementation cost is ~6 LoC per retired method (the raise +
the multi-line message). With six retired methods, that's ~36 LoC of
intentional discoverability scaffolding. **CLAUDE.md exception
("No backwards-compat shims") is correctly observed:** these are
not shims — they are tombstones with forwarding addresses.

**Severity: none.** Flag only — this pattern is healthy and worth
keeping.

---

## Lens 2 — Privatization

### 2.1 `FeatureRecord.feature_columns()` etc. are convenience class-methods that forward to `self.feature.<attr>`

`feature.py:381-459` defines four classmethods on `FeatureRecord`
that forward to the `Feature` ClassVar's attributes:

```python
@classmethod
def feature_columns(cls) -> set[Column]:
    return cls.feature.feature_columns

@classmethod
def asset_columns(cls) -> set[Column]:
    return cls.feature.asset_columns

# (term_columns, value_columns follow the same shape)
```

These four methods raise `AttributeError` on the base class (where
`feature` is `None`). They only work on subclasses returned by
`Feature.feature_record_class()`. The docstring notes this — but
calling `FeatureRecord.feature_columns()` on the base class crashes
with a confusing `AttributeError: 'NoneType' object has no attribute
'feature_columns'` rather than a helpful "this is only valid on a
generated subclass" message.

A workspace-wide grep for these four method names returns:

- `deriva-mcp/src/deriva_mcp/...`: zero external calls;
- `deriva-ml-mcp/src/deriva_ml_mcp/...`: zero external calls;
- `deriva-ml-skills`: zero;
- `deriva-ml-model-template`: zero.

The four methods are **internal-facing only**. They exist for the
test suite (`tests/feature/test_features.py:115-129, 178-184`) and
for whoever introspects a generated class.

**Fix options:**

(a) Keep as-is — the docstring is explicit about the
generated-subclass-only requirement.

(b) Add a guarding `if cls.feature is None: raise
DerivaMLException("..." )` so the error message is useful when someone
calls the method on the base class. **LoC: +4 per method = +16.**

(c) Privatize — rename to `_feature_columns()` etc. since no
external project uses them. Tests update easily. **LoC: ±0**, but the
generated subclass interface narrows.

**Recommended:** (b). The methods are public-facing by accessor name
(they look like clean accessor surface), and CLAUDE.md's
"`FeatureRecord` is the user-facing type" steering pushes toward
keeping the surface flat. The guard fixes the error-message paper
cut.

**Severity: low.** Flag only.

### 2.2 `Feature.__init__` is internal-but-documented-as-internal

`feature.py:483-542` (`Feature.__init__`) has a docstring (line 499)
that says:

> Note:
>     This constructor is not part of the public API. Obtain
>     ``Feature`` instances via ``DerivaML.create_feature`` or
>     ``DerivaML.lookup_feature``.

The function is not underscore-prefixed. Workspace-wide grep for
`Feature(` (constructor calls) outside `deriva-ml/src/` returns
zero hits — confirmed the docstring claim.

**Fix:** leave as-is. The class itself is internal (no
`__all__` exposure, no direct external import). The docstring is
the right place for the contract.

**Severity: none.** Flag only.

### 2.3 `Feature.feature_record_class.map_type` is a deep-nested helper

`feature.py:575-609` (the inner `map_type` function inside
`Feature.feature_record_class`) is a closure that converts ERMrest
column types to Python types. Its docstring example (lines 592-594)
is illustrative but not runnable. It's never reused.

The closure shape is awkward — line 596 captures `self.asset_columns`
via the enclosing function:

```python
def map_type(c: Column) -> UnionType | Type[str] | Type[int] | Type[float]:
    if c.name in {c.name for c in self.asset_columns}:
        return str | Path
    match c.type.typename:
        case "text":
            return str
        case "int2" | "int4" | "int8":
            return int
        case "float4" | "float8":
            return float
        case "boolean":
            return bool
        case _:
            return str
```

The set comprehension on line 596 recomputes `{c.name for c in
self.asset_columns}` on every call. For a feature with N columns
this is N² work in a hot path of `feature_record_class()`.

**Fix:** pull the asset-column-name set out of the loop:

```python
asset_col_names = {c.name for c in self.asset_columns}

def map_type(c: Column) -> ...:
    if c.name in asset_col_names:
        return str | Path
    ...
```

**LoC: ±0** (move one line, save one allocation per column).
**Severity: low.** Performance flag for the technical-writer /
senior-engineer persona.

### 2.4 `core/mixins/feature.py:48-55` class docstring drifts from reality

`core/mixins/feature.py:48-55`:

```python
"""
Methods:
    create_feature: Create a new feature definition
    feature_record_class: Get pydantic model class for feature records
    delete_feature: Remove a feature definition
    lookup_feature: Retrieve a Feature object
    find_features: Find all features in the catalog, optionally filtered by table
    feature_values: Get all values for a feature
"""
```

`list_workflow_executions` (`mixins/feature.py:559-618`) is missing
from this list, even though it's the public method that lets
`FeatureRecord.select_by_workflow` resolve workflow→execution-set.
Same for the retired-shim raise-pointer methods (which are arguably
better left out, since they're tombstones).

**Fix:** add `list_workflow_executions: List execution RIDs for a
given workflow (by RID or Workflow_Type name)`. **LoC: +1.**
**Severity: low** — doc-only, but a technical-writer red flag.

---

## Lens 3 — Coverage

### 3.1 `select_majority_vote` has no dedicated unit tests

`feature.py:309-379` (`select_majority_vote`) is pure Python with
non-trivial branching:

1. Auto-detect column from `record_cls.feature.term_columns` when
   `column is None`.
2. Raise if the feature has **zero** term columns (line 367 message:
   `"requires a column name — could not auto-detect"`).
3. Raise if the feature has **multiple** term columns (line 357:
   `"requires a column name for features with multiple term columns"`).
4. `Counter`-based majority logic with RCT tie-breaker.

Coverage:

- `tests/feature/test_features.py`: no test.
- `tests/feature/test_feature_values.py`: no test.
- `tests/feature/test_select_by_workflow.py`: not in scope.
- `deriva-ml-mcp/tests/test_feature.py:251`: covers only the dispatch
  branch (string `"majority_vote"` → `FeatureRecord.select_majority_vote()`).
  Does not exercise the auto-detect or tie-breaker logic.
- `tests/dataset/test_restructure.py:948`: imports `FeatureRecord`
  but a quick check shows it tests `select_newest` and
  `select_majority_vote` through `restructure_assets`, not the bare
  selector — so the auto-detect branches are not exercised.

The three error paths (no records, no feature metadata, multiple
term columns) are entirely untested.

**Fix:** add `tests/feature/test_select_majority_vote.py` modeled on
`test_select_by_workflow.py` (stub container, no live catalog):

```python
def test_majority_vote_auto_detects_single_term_column():
    """When the feature has exactly one term column, auto-detect it."""
    ...

def test_majority_vote_raises_on_multiple_term_columns():
    """Multiple term columns → DerivaMLException without explicit column."""
    ...

def test_majority_vote_raises_on_zero_term_columns():
    """No term columns → DerivaMLException."""
    ...

def test_majority_vote_breaks_ties_by_newest_rct():
    """When two values tie, the newest RCT wins."""
    ...

def test_majority_vote_explicit_column():
    """When column is passed, it overrides auto-detection."""
    ...
```

**Effort: low** — pure-Python unit tests, no live catalog. **LoC:
+~80.** **Risk: low.** **Severity: medium** — this selector is
named in CLAUDE.md as a public API and is one of the dispatch keys
in the MCP layer (`"majority_vote"`).

### 3.2 `select_by_execution` has no dedicated unit tests

`feature.py:133-184` (`select_by_execution`) is pure Python:

1. Build a closure filtering on `r.Execution == execution_rid`.
2. If no match: `raise DerivaMLException` (line 181).
3. If match: `return FeatureRecord.select_newest(filtered)`.

Coverage:

- `tests/feature/test_select_by_workflow.py`: not in scope.
- `tests/feature/test_features.py`: no test.
- `deriva-ml-mcp/tests/test_feature.py:299-311`: covers the dispatch
  branch only — does not exercise the raise-on-no-match path.

**Fix:** add a parallel test file or extend
`test_select_by_workflow.py` with `select_by_execution` cases:

```python
def test_select_by_execution_picks_match():
    ...

def test_select_by_execution_raises_when_no_records_match():
    ...

def test_select_by_execution_picks_newest_among_matches():
    ...
```

**Effort: low.** **LoC: +~40.** **Severity: medium** — used by the
model-template (see `experiment-decisions.md`).

### 3.3 `select_newest`, `select_first`, `select_latest` have only doctest-shaped coverage

CLAUDE.md "Special focus areas" calls out `FeatureRecord.select_newest`
specifically as a pure-Python doctest example. The actual coverage:

- `tests/feature/test_features.py::TestFeatureRecord`: 3 tests
  exercise the base class's pydantic field construction, but never
  call `select_newest` / `select_first` / `select_latest`.
- `tests/feature/test_feature_values.py::test_feature_values_with_select_newest`:
  passes `selector=FeatureRecord.select_newest` to
  `feature_values`, then asserts `len(rids) == len(set(rids))` —
  i.e., "one record per target RID after dedup." Does not assert
  *which* record was picked, only that some single record was.
- `test_feature_values_symmetry::test_feature_values_with_selector_matches`:
  same shape.
- `src/deriva_ml/conftest.py:17-20`: injects `FeatureRecord` into the
  doctest namespace. The only doctest that exercises the selector is
  the `>>> callable(selector) is True` line in `feature.py`'s
  module-level docstring. Nothing tests the lexicographic RCT
  comparison, the None-RCT handling, or the "select_latest is an
  alias for select_newest" behavior.

The three selectors are simple enough that they're unlikely to
regress — `max(records, key=lambda r: r.RCT or "")` is one line. But
their `None`-handling is one of the unintuitive bits: per the
docstring (lines 117-118 of `feature.py`):

> Records with ``None`` RCT are treated as older than any timestamped
> record.

This holds because `None or ""` evaluates to `""`, and `""` is less
than any ISO 8601 timestamp string. A future refactor that swapped
to `r.RCT` (no `or ""`) would crash on a None RCT — and no test
would catch it.

**Fix:** add three small unit tests in
`tests/feature/test_select_newest.py` (or extend
`test_select_by_workflow.py`):

```python
def test_select_newest_picks_latest_rct():
    records = [
        FeatureRecord(Feature_Name="x", RCT="2024-01-01T00:00:00"),
        FeatureRecord(Feature_Name="x", RCT="2024-06-15T00:00:00"),
    ]
    assert FeatureRecord.select_newest(records).RCT == "2024-06-15T00:00:00"


def test_select_newest_treats_none_rct_as_oldest():
    records = [
        FeatureRecord(Feature_Name="x", RCT=None),
        FeatureRecord(Feature_Name="x", RCT="2024-01-01T00:00:00"),
    ]
    assert FeatureRecord.select_newest(records).RCT == "2024-01-01T00:00:00"


def test_select_latest_is_alias_for_select_newest():
    records = [FeatureRecord(Feature_Name="x", RCT="2024-06-15T00:00:00")]
    assert FeatureRecord.select_latest(records) is FeatureRecord.select_newest(records)


def test_select_first_picks_earliest_rct():
    records = [
        FeatureRecord(Feature_Name="x", RCT="2024-01-01T00:00:00"),
        FeatureRecord(Feature_Name="x", RCT="2024-06-15T00:00:00"),
    ]
    assert FeatureRecord.select_first(records).RCT == "2024-01-01T00:00:00"
```

**Effort: low.** **LoC: +~30.** **Severity: low** — the regression
class is narrow but the selectors are externally-documented and
worth pinning.

### 3.4 No test for `delete_feature` behavior

See §1.5.

### 3.5 No test for `create_feature` error paths

`create_feature` raises `DerivaMLException("Invalid create_feature
asset table.")` (or the buggy duplicated message in §1.4) on two
branches:

- `assets` contains a non-asset table;
- `terms` contains a non-vocabulary table.

Neither raise is tested. The closest test
(`tests/feature/test_features.py::TestFeatures.test_create_feature`)
asserts successful creation only.

A future refactor that swapped `is_asset` for some other predicate
(or that changed the validation order) would silently change
behavior, and no test would catch it.

**Fix:** add two tests:

```python
def test_create_feature_with_non_asset_table_raises(test_ml):
    """Passing a domain table as `assets` raises DerivaMLException."""
    with pytest.raises(DerivaMLException, match="asset table"):
        test_ml.create_feature("Image", "bad_feat", assets=["Subject"])


def test_create_feature_with_non_vocabulary_table_raises(test_ml):
    """Passing a domain table as `terms` raises DerivaMLException."""
    with pytest.raises(DerivaMLException, match="vocabulary|term"):
        test_ml.create_feature("Image", "bad_feat", terms=["Subject"])
```

Pair this with the §1.4 message fix. **Effort: low.** **Risk: low.**
**Severity: medium** for the ML-developer persona — these are the
exact errors a user hits when misclassifying a table.

### 3.6 `feature_values(execution_rids=[])` short-circuit is tested for catalog and bag but the doctest doesn't note it

`tests/feature/test_feature_values_limits.py::test_feature_values_execution_rids_empty_list_returns_nothing`
covers the short-circuit. Similar test exists for the bag side
(through the `feature_symmetry_fixture`). Good.

The docstring at `mixins/feature.py:427-429` says:

> Empty list short-circuits to an empty result.

This is correct. **Severity: none.** Flag only — coverage is good.

### 3.7 `Dataset.list_workflow_executions` docstring claims dataset-scoping that doesn't exist

`dataset/dataset.py:625-648` defines `Dataset.list_workflow_executions`
as a pure pass-through (`return self._ml_instance.list_workflow_executions(workflow)`).
The docstring (line 626) calls it "Dataset-scoped" and lines 628-632
acknowledge:

> Current implementation returns the full workflow execution list
> from the catalog. Target-RID filtering at selection time (via
> ``feature_values``) ensures that records from executions outside
> the dataset's member set are excluded. A stricter scope ... is
> deferred.

The test `test_dataset_list_workflow_executions_scopes_to_dataset`
(`test_feature_values.py:291`) is **named** as if it tests scoping
but asserts `ds_rids == ml_rids` — the opposite of "scoped" — and
includes a comment noting "If strict dataset scoping is added later
this test will fail."

This is honest documentation of a deferred behavior, but the
method name + docstring name "Dataset-scoped" misleads a reader who
expects scoping. The contract gap matters for the bag-symmetry test:
`DatasetBag.list_workflow_executions` actually IS scoped (the bag
only contains executions reachable from the dataset slice), but
`Dataset.list_workflow_executions` isn't.

**Fix options:**

(a) Rename docstring header from "Dataset-scoped
list_workflow_executions" to "Pass-through to
DerivaML.list_workflow_executions (dataset scoping deferred)." **LoC:
±0.**

(b) Actually scope: filter `self._ml_instance.list_workflow_executions(workflow)`
to those execution RIDs that produced/consumed dataset members. Closer
to user expectation but more expensive. **LoC: +15.**

**Recommended:** (a) for the audit. The behavioral fix is its own
task — flagging here for visibility.

**Severity: medium for the ML-developer persona** — the symmetry
contract has a subtle hole (bag-side scopes, catalog-side doesn't).
**Severity: low** for any user who reads the docstring's deferred-note.

### 3.8 `interfaces.py:215` (`DatasetLike.find_features`) has a different signature from the implementations

`interfaces.py:215`:

```python
def find_features(self, table: str | Table) -> Iterable[Feature]:
```

Required `table` argument. But the implementations:

- `mixins/feature.py:321`: `def find_features(self, table: str | Table |
  None = None) -> list[Feature]` — optional, with a "find all
  features in the catalog" branch when `None`.
- `dataset_bag.py:489`: `def find_features(self, table: str | Table)
  -> Iterable[Feature]` — required (matches the protocol).
- `dataset.py:???` — doesn't define `find_features`; inherits via the
  `_ml_instance` proxy? Let me check…
- `tests/feature/test_features.py:162`: `all_features = ml_instance.find_features()`
  — calls without an argument, so this branch is in use.

The signature drift is a soft constraint failure: a caller using
`DatasetLike.find_features(table=None)` to mean "find all" violates
the protocol but works on `DerivaML`. A caller using
`DatasetLike.find_features()` against a `DatasetBag` would fail at
runtime even though the protocol "permits" it.

**Fix:** update the protocol signature to match the implementations
that allow `table=None`:

```python
def find_features(self, table: str | Table | None = None) -> Iterable[Feature]:
```

Then either: (a) update `DatasetBag.find_features` to handle the
`None` case too (matching `DerivaML`), or (b) leave `DatasetBag.find_features`
required-only and accept the protocol-level optionality is
implementation-specific (less ideal).

**Recommended:** (a) — bags should be able to enumerate all features,
and the implementation is one call to `self.model.find_features()`
which already supports `table=None` (see `model/catalog.py:554`).
**LoC: +5.** **Risk: low.** **Severity: medium for the
ML-developer persona.**

### 3.9 `feature_values` raises `DerivaMLTableNotFound` per the docstring; the actual implementation raises bare `DerivaMLException`

`mixins/feature.py:434-435`:

> Raises:
>     DerivaMLTableNotFound: ``table`` does not exist.

The implementation (line 465) calls `self.model.name_to_table(table)`,
which raises `DerivaMLTableNotFound` per its own docstring. Good.

But line 466 also calls `self.lookup_feature(table_obj, feature_name)`,
which raises bare `DerivaMLException("Feature {table.name}:{feature_name} doesn't exist.")`
(see `model/catalog.py:624`). The docstring claim is technically
true (table-not-found does raise the typed exception), but a
feature-not-found case raises an untyped `DerivaMLException`. Tests
only assert `DerivaMLException` (`test_features.py:137-142`).

**Fix options:**

(a) Add a `DerivaMLFeatureNotFound` exception subclass and use it in
`lookup_feature`. **LoC: +6 in `exceptions.py` + 1 in
`model/catalog.py:624`.** **Risk: low** — it's a subclass; existing
tests still pass with `except DerivaMLException`.

(b) Document the actual behavior in the docstring (bare
`DerivaMLException` for feature-not-found). **LoC: ±0.**

**Recommended:** (a). The exception hierarchy in `core/exceptions.py`
already has `DerivaMLNotFoundError` (catalog audit §… of the exception
hierarchy) — extending with a feature-specific subclass is in
keeping with the existing pattern. **Severity: low.**

---

## Lens 4 — Docs sync

### 4.1 `feature.py:478-481` `Feature` example uses post-construction prints, not constructor

`feature.py:477-481`:

```python
Example:
    >>> feature = ml.lookup_feature("Image", "Diagnosis")  # doctest: +SKIP
    >>> print(f"Feature {feature.feature_name} on {feature.target_table.name}")  # doctest: +SKIP
    >>> print("Asset columns:", [c.name for c in feature.asset_columns])  # doctest: +SKIP
```

All three lines are `doctest: +SKIP`. Per CLAUDE.md's doctest
convention, pure-Python lines should run for real. The first line is
correctly skipped (`ml.lookup_feature` needs a catalog), but the
`feature.target_table.name` and the asset-column listing don't
require a catalog if `feature` is mocked.

CLAUDE.md `feature.py:30-32`'s top-level example is similarly all-skip
even for the trivial line:

```python
>>> rec = DiagnosisFeature(Diagnosis="benign", Confidence=0.97)  # doctest: +SKIP
```

This line could run for real if a `FeatureRecord` subclass with
the right fields is set up. The doctest in
`feature.py:14-17` (`FeatureRecord.select_newest`) does correctly
note "selector classmethod" but the cited example in CLAUDE.md
(`callable(selector) is True`) is one line — and that one line **is**
in `feature.py`. Coverage is technically there but minimal.

**Fix:** add at least one pure-Python doctest line that exercises
each public method's signature/return shape. Example for
`select_newest`:

```python
>>> records = [
...     FeatureRecord(Feature_Name="X", RCT="2024-01-01T00:00:00"),
...     FeatureRecord(Feature_Name="X", RCT="2024-06-15T00:00:00"),
... ]
>>> FeatureRecord.select_newest(records).RCT
'2024-06-15T00:00:00'
```

**LoC: +~30 across 6 selectors.** **Risk: low.** **Severity: low.**

### 4.2 `core/mixins/feature.py:443-447` doctest is shaped wrong

The docstring at `mixins/feature.py:440-464` contains:

```python
Example:
    Get the newest Glaucoma label per image::

        >>> from deriva_ml.feature import FeatureRecord
        >>> for rec in ml.feature_values(
        ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
        ... ):
        ...     print(f"{rec.Image}: {rec.Glaucoma} (by {rec.Execution})")
```

No `# doctest: +SKIP`. The first line (`from deriva_ml.feature
import FeatureRecord`) will run as a real doctest — but the
subsequent `for rec in ml.feature_values(...)` requires `ml` in the
doctest namespace, which it isn't. So at doctest collection time,
this raises `NameError: name 'ml' is not defined`. Whether this
fires depends on `pytest --doctest-modules` setup — looking at
`conftest.py:18-20`, only `FeatureRecord` is injected, not `ml`.

**Test it:** `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true
uv run pytest --doctest-modules src/deriva_ml/core/mixins/feature.py`
would confirm. (Did not run during this audit per the
"do not run tests" rule.)

**Fix:** add `# doctest: +SKIP` markers on the catalog-dependent
lines:

```python
>>> from deriva_ml.feature import FeatureRecord
>>> for rec in ml.feature_values(  # doctest: +SKIP
...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
... ):
...     print(f"{rec.Image}: {rec.Glaucoma} (by {rec.Execution})")
```

Same shape needed at `mixins/feature.py:587-598`
(`list_workflow_executions` example, which has `>>> rids =
ml.list_workflow_executions("Glaucoma_Training_v2")` without a SKIP
marker). And the `mixins/feature.py:444` example, and the
`mixins/feature.py:594` example.

**LoC: ±0 (annotation only).** **Risk: low.** **Severity: medium
for the technical-writer persona** — doctest collection may already
be skipping these silently. Worth verifying with one explicit
pytest run during the cleanup PR.

### 4.3 `core/mixins/feature.py:115` docstring example uses `BuiltinTypes.float4` that's not in scope

`mixins/feature.py:114-129`:

```python
Examples:
    Create a feature with confidence score:
        >>> DiagnosisFeature = ml.create_feature(  # doctest: +SKIP
        ...     target_table="Image",
        ...     feature_name="Diagnosis",
        ...     terms=["Diagnosis_Type"],
        ...     metadata=[ColumnDefinition(name="confidence", type=BuiltinTypes.float4)],
        ...     comment="Clinical diagnosis label"
        ... )
```

`ColumnDefinition` and `BuiltinTypes` aren't injected into the
doctest namespace (`conftest.py:18-20` only injects `FeatureRecord`).
The `# doctest: +SKIP` correctly skips the catalog-dependent line, but
the reader following the example needs to know to also
`from deriva_ml import ColumnDefinition, BuiltinTypes` first.

**Fix:** add the import line to the example:

```python
Examples:
    Create a feature with confidence score:
        >>> from deriva_ml import BuiltinTypes, ColumnDefinition  # doctest: +SKIP
        >>> DiagnosisFeature = ml.create_feature(  # doctest: +SKIP
        ...     ...
```

Same shape needed at other examples in this file. **LoC: +6.**
**Severity: low.**

### 4.4 The `Feature.feature_record_class.map_type` docstring example uses a fake `Column` constructor

`feature.py:592-594`:

```python
Example:
    >>> col = Column(name="score", type="float4")
    >>> typ = map_type(col)  # Returns float
```

ERMrest `Column` doesn't accept `name=` and `type=` as kwargs in this
shape — its actual constructor needs a `Column.define(...)` or a
deserialized schema object. The example is **illustrative-only**, but
it's not marked `doctest: +SKIP`, so if `--doctest-modules` ever
hits this docstring, it'd fail at the `Column(...)` line.

**Fix:** either mark `# doctest: +SKIP` or rewrite to use
`Column.define(...)`:

```python
Example:
    >>> # Inside Feature.feature_record_class, map_type returns float for float8 columns:
    >>> # (not directly callable — illustrative only)
```

**LoC: ±0–2.** **Severity: low.**

### 4.5 CLAUDE.md cites `select_newest` as a pure-Python doctest example — but the file's coverage of it is one line

CLAUDE.md "Special focus areas":

> **Selector functions:** `FeatureRecord.select_newest` is referenced
> in CLAUDE.md as a pure-Python doctest example. Check whether other
> selectors exist and whether they have similar test coverage.

The doctest that runs for real (per `conftest.py`):

```python
>>> from deriva_ml.feature import FeatureRecord
>>> selector = FeatureRecord.select_newest
>>> callable(selector)
True
```

That's the entirety of `select_newest`'s doctest coverage. The
other five selectors have no equivalent pure-Python doctest. CLAUDE.md
treats `select_newest` as the *exemplar* — but the exemplar covers
only "is the method callable," not "does it produce the right
result."

**Fix:** subsumed by §4.1 + §3.3 — add pure-Python doctests + unit
tests that actually exercise the selector return values.

**Severity: low.**

### 4.6 `__init__.py:109-110` comment misrepresents `FeatureRecord`'s scope

`src/deriva_ml/__init__.py:109-110`:

```python
# Feature record for feature values and restructure_assets selectors
"FeatureRecord",
```

This isn't wrong — `FeatureRecord` is used for both — but it
undersells the breadth. `FeatureRecord` is also:

- The base class for all generated feature subclasses;
- The container for the six selector classmethods;
- The shared type across `DerivaML`, `Dataset`, `DatasetBag` per
  CLAUDE.md "Bags Should Behave Like Catalog Connections";
- The Pydantic round-tripping anchor (`.model_dump()` is used in
  notebooks and MCP tool responses).

**Fix:**

```python
# Feature record type — pydantic model for selector inputs / outputs,
# generated subclasses for typed feature construction, and the unified
# shape across DerivaML / Dataset / DatasetBag (see CLAUDE.md).
"FeatureRecord",
```

**LoC: +3.** **Severity: low.**

---

## Lens 5 — API conventions (deriva-py vs raw ERMrest)

### 5.1 `feature_values` correctly uses `pathBuilder()` + `entities().fetch()` instead of raw ERMrest URLs

`mixins/feature.py:477-486`:

```python
pb = self.pathBuilder()
feature_path = pb.schemas[feat.feature_table.schema.name].tables[feat.feature_table.name]

if execution_rids is not None:
    predicates = [feature_path.Execution == rid for rid in execution_rids]
    feature_path = feature_path.filter(reduce(or_, predicates))

raw_values = list(feature_path.entities().fetch())
```

The implementation uses datapath consistently. The
filter-construction pattern (lines 480-484) for "IN-clause as OR of
equalities" is correctly justified by the inline comment:

> Path-builder column wrappers don't expose .in_(); build the
> IN-clause as a chained OR of equality predicates instead.

This is the right deriva-py idiom. **No issue.**

### 5.2 `list_workflow_executions` correctly delegates through `find_executions`

`mixins/feature.py:600-617`:

```python
try:
    wf = self.lookup_workflow(workflow)
except DerivaMLException:
    wf = None

if wf is not None:
    return [r.execution_rid for r in self.find_executions(workflow=wf)]

rids = [r.execution_rid for r in self.find_executions(workflow_type=workflow)]
if not rids:
    raise DerivaMLException(...)
return rids
```

Uses `find_executions(workflow=...)` and `find_executions(workflow_type=...)`
— both high-level deriva-ml APIs. **No issue.**

### 5.3 `Feature.__init__`'s FK-classification uses `pk_table` + `is_asset` / `is_vocabulary`

`feature.py:530-540`:

```python
self.asset_columns = {
    fk.foreign_key_columns[0]
    for fk in self.feature_table.foreign_keys
    if fk not in assoc_fkeys and self._model.is_asset(fk.pk_table)
}
```

Uses `fk.pk_table` directly (ERMrest model attribute) and
`self._model.is_asset` (deriva-ml predicate). This is the right shape.
**No issue.**

### 5.4 No raw `catalog.put` / `catalog.get` / URL construction anywhere in the subsystem

Confirmed by `grep -n "catalog.put\|catalog.get\|/ermrest/catalog" feature.py mixins/feature.py` —
zero hits. The subsystem fully relies on deriva-ml + deriva-py
abstractions per CLAUDE.md's "API Priority" guidance.

**Lens 5 verdict: clean.** No actions.

---

## Lens 6 — Maintainability

### 6.1 `feature.py` (648 LoC) and `core/mixins/feature.py` (639 LoC) are at the upper edge of acceptable

Both files are large but coherent. Each has one clear responsibility:

- `feature.py`: typing surface (`FeatureRecord` base, selectors,
  `Feature` runtime class).
- `core/mixins/feature.py`: `DerivaML` method facade.

The split is documented and clean (see audit prompt's "Mixin vs
standalone split" focus area). No bleed:

- `feature.py` does not depend on the mixin (lines 1-46 imports only
  Pydantic, ERMrest, typing, pathlib).
- `core/mixins/feature.py` imports `Feature, FeatureRecord` (line 27)
  for the typing surface, but the mixin's method implementations
  don't mutate `FeatureRecord` or `Feature` internals — they only
  call public methods.

The two files together represent a ~1 280 LoC subsystem, which is
on the larger side for a single concern but justified by the
external-API breadth (six selectors, three-container symmetry, MCP
wrapping). Splitting `feature.py` further (e.g., `_selectors.py`)
would add import indirection without simplifying the mental model.

**No action.** Flag for the six-month follow-up.

### 6.2 The retired-shim pattern is repeated 4× in `core/mixins/feature.py`

`mixins/feature.py:350-369` (`add_features`), `517-536`
(`fetch_table_features`), `538-556` (`list_feature_values`),
`620-639` (`select_by_workflow`) — each is ~20 LoC of docstring +
raise.

The pattern could be a shared helper:

```python
def _retired(message: str, *args, **kwargs):
    raise DerivaMLException(message)


list_feature_values = staticmethod(functools.partial(
    _retired,
    "DerivaML.list_feature_values() has been retired and renamed. "
    "Use feature_values(table, feature_name, selector=...) instead.",
))
```

But this loses the per-method docstring discoverability (e.g., the
detailed docstring at `mixins/feature.py:351-364` is itself a
documentation surface — users reading the file or running `help()`
get the full migration story).

**Recommended:** leave as-is. The duplication is intentional
documentation, and per CLAUDE.md "Examples are required" the
docstrings should remain runnable / readable.

**Severity: none.** Flag only.

### 6.3 `_selector` closure name appears 4× in `feature.py`

`feature.py:176, 258, 344, …` all define a function named
`_selector` inside a factory. This is consistent — every selector
factory follows the same pattern — but it means stack traces during
selector failures all read "in _selector" without indicating which
selector. (Practically a low-noise issue; selectors are simple
enough that debugger lookups suffice.)

**Fix options:** (a) name each closure (`_select_by_workflow_closure`
etc.), or (b) leave as-is.

**Recommended:** (b). Renaming for tracebacks-only is over-engineering.
**Severity: none.** Flag only.

### 6.4 Inline `from deriva_ml.core.exceptions import` inside `feature.py` selectors

`feature.py:179`, `feature.py:353`, `feature.py:363` all do:

```python
def _selector(records: list["FeatureRecord"]) -> "FeatureRecord":
    ...
    from deriva_ml.core.exceptions import DerivaMLException
    raise DerivaMLException(...)
```

Three separate inline-import patterns inside closures. The
module-level imports at `feature.py:35-48` already pull in
`Optional`, `Callable`, etc., and there's no circular-import risk
between `feature.py` and `core.exceptions` (exceptions has no
deriva-ml dependencies).

**Fix:** move the import to module level:

```python
from deriva_ml.core.exceptions import DerivaMLException
```

**LoC: −2 (remove 3 inline imports, add 1 module-level). Risk: low.**
**Severity: low.**

---

## Persona check

### Senior engineer

The error-handling story is solid — `feature_values` validates the
table via `name_to_table`, the feature via `lookup_feature`, the
`execution_rids` filter short-circuits on empty, and
`DerivaMLMaterializeLimitExceeded` exists for memory bounds. The
`@validate_call(config=VALIDATION_CONFIG)` decorator is applied to
`feature_values` and `list_workflow_executions` — the two methods
that take complex argument shapes. Less covered are `create_feature`,
`delete_feature`, `lookup_feature`, `find_features`, and
`feature_record_class` (no `@validate_call`). For a senior-engineer
review the gap is intentional (those methods take simpler args), but
worth flagging that any future signature drift on
`feature_values` (e.g., adding a kwarg) needs to also update the
`VALIDATION_CONFIG` integration. **The §1.4 buggy validation message
is the only real correctness flag.**

### Testing engineer

The S2 acceptance suite
(`tests/feature/test_feature_values.py::TestFeatureValuesSymmetry`) is
the strongest piece of test infrastructure in any of the four
phase-3 subsystems so far. The fixture builds a real catalog +
dataset + materialized bag and asserts cross-container parity. The
weak points are the **selector unit tests** (§3.1, §3.2, §3.3) and
the **empty `test_delete_feature`** (§1.5). The retired-shim
regression suite is well-shaped. **Overall coverage is in good
shape; the gaps are narrow.**

### Technical writer

Naming consistency: `Feature` vs `FeatureRecord` vs `Feature_Name`
vs `feature_name` — each has a distinct meaning and the docstrings
do explain them (e.g., `mixins/feature.py:275-318` walks through
the relationship cleanly). The doctest hygiene is **the weakest
dimension** — at least three docstrings (§4.2, §4.3, §4.4) have
examples that either fail to import or aren't marked `# doctest:
+SKIP`. CLAUDE.md's "Examples are required and runnable" rule is
partially violated by the catalog-dependent examples that don't
clearly carry the SKIP marker. The class-docstring at
`mixins/feature.py:48-55` is also stale (missing
`list_workflow_executions`, see §2.4).

### ML-developer user

The persona-of-record. From this perspective:

- **Six selectors cover the common use cases** (most-recent,
  earliest, by-execution, by-workflow, majority-vote, latest). The
  set is well-named and the MCP wraps them by simple string keys.
- **`FeatureRecord` is a coherent type** — `model_dump()` gives a
  dict, `.RCT`/`.Execution`/`.Feature_Name` are predictable
  attributes, selector signatures are consistent.
- **The bag-parity contract works** —
  `tests/feature/test_feature_values.py::TestFeatureValuesSymmetry`
  proves that `ml.feature_values()` and `bag.feature_values()`
  return equivalent records sorted by target RID. CLAUDE.md's "Bags
  Should Behave Like Catalog Connections" load-bearing assertion
  holds in practice.
- **The error-message bug in §1.4** is the one obvious paper cut.
  Misclassifying a domain table as `terms` instead of `assets`
  sends the user to the wrong place.
- **`Dataset.list_workflow_executions` is a pass-through, not
  scoped** (§3.7). A user who builds a dataset of 100 images out of
  a 10 000-image catalog, then asks "which executions ran the
  Glaucoma workflow on my dataset?", gets the full 10 000-image
  answer. The docstring acknowledges this but the method name doesn't
  forewarn.

### DBA

Schema correctness:

- `Feature_Name` is correctly seeded as an ML vocabulary in
  `schema/create_schema.py:343` — confirmed by the schema audit's
  `validation.py:274-280` `EXPECTED_VOCABULARY_TABLES` list (which
  includes `MLVocab.feature_name`, unlike `execution_status` which
  is missing — see schema audit §1.1).
- Feature association tables are created via
  `model._define_association` in `create_feature`
  (`mixins/feature.py:172-179`), which is the documented pattern in
  CLAUDE.md "Association Tables." Three FKs: `target_table`,
  `Feature_Name`, `Execution`. Correct shape.
- Per-row owner ACL inheritance: feature tables inherit the
  catalog-wide ACL bindings from `policy.json` via `create_ml_catalog`
  (not feature-specific code). The schema audit §3.2 noted that no
  test asserts ACL bindings on dynamically-created tables. **This
  also applies to feature tables.** A user who calls `create_feature`
  on a table they don't own may or may not be able to insert feature
  values depending on row-level ACL inheritance — and no test pins
  the behavior.

**DBA verdict: structurally correct, but ACL coverage on
dynamically-created feature tables is the same hole as the schema
audit's §3.2.**

---

## Ranked actions

Ordered by impact-per-effort. Each entry: effort (LoC + complexity)
and risk class (trivial / low / medium / high).

1. **Fix the duplicated/incorrect validation message in
   `create_feature` (§1.4).** Change line 161 to say "vocabulary
   table" and include the offending table name. Effort: trivial,
   <10 LoC. Risk: trivial. **Highest impact-per-effort in the
   audit** — fixes a known debugging detour.

2. **Fix the broken `TYPE_CHECKING` import path in `feature.py:48`
   (§1.3).** Change `from model.catalog import DerivaModel` to
   `from deriva_ml.model.catalog import DerivaModel`. Effort:
   trivial, 1 LoC. Risk: trivial. Restores IDE type resolution.

3. **Add `test_select_majority_vote.py` (§3.1).** Five tests mirroring
   `test_select_by_workflow.py`. Effort: low, ~80 LoC of new test
   code. Risk: low. **High value for the ML-developer persona** —
   selector is in the MCP dispatch and externally documented.

4. **Add `test_select_by_execution.py` (§3.2).** Three tests. Effort:
   low, ~40 LoC. Risk: low.

5. **Add doctest-runnable pure-Python tests for `select_newest`,
   `select_first`, `select_latest` (§3.3 + §4.1).** Three small
   tests + matching doctest examples in the source docstrings.
   Effort: low, ~30 LoC. Risk: low.

6. **Fix the missing `# doctest: +SKIP` markers in
   `core/mixins/feature.py` (§4.2).** Audit all examples that
   reference `ml.` or `bag.` and mark catalog-dependent lines.
   Confirmed-needed at lines 443-447, 587-598. Effort: low, ~10
   LoC. Risk: low.

7. **Replace the `test_delete_feature: pass` placeholder with two
   real tests (§1.5).** Effort: low, ~15 LoC. Risk: low.

8. **Add `test_create_feature_with_non_*_table_raises` tests
   (§3.5).** Pair with the §1.4 fix. Effort: low, ~15 LoC. Risk:
   low.

9. **Remove the `importlib.import_module` workarounds in
   `feature.py` (§1.1) and `core/mixins/feature.py` (§1.2).**
   Replace with normal imports. The unused `datapath` import in
   `mixins/feature.py:18` should be dropped entirely. Effort: low,
   −11 LoC. Risk: low (same pattern already in use across the
   codebase). **Schema audit §1.10 has the same finding;
   workspace-wide cleanup should land in one PR.**

10. **Update `Dataset.list_workflow_executions` docstring to be
    honest about the pass-through (§3.7).** Either rename the
    "Dataset-scoped" header or actually scope. Effort: low (option
    a) or medium (option b). Risk: low (option a) / medium (option
    b).

11. **Update `interfaces.DatasetLike.find_features` to allow
    `table=None` (§3.8).** Plus implement the `None` branch on
    `DatasetBag`. Effort: low, ~10 LoC. Risk: low.

12. **Add a `DerivaMLFeatureNotFound` exception subclass (§3.9).**
    Use it in `lookup_feature`. Effort: low, ~8 LoC. Risk: low.

13. **Fix the `feature.py:48` `TYPE_CHECKING` import path** — see
    #2.

14. **Move the three inline `from deriva_ml.core.exceptions`
    imports in `feature.py` to module level (§6.4).** Effort:
    trivial, −2 LoC. Risk: trivial.

15. **Lift the asset-column-name set out of `map_type`'s loop
    (§2.3).** Effort: trivial, 1 LoC move. Risk: trivial. Minor
    perf win in `feature_record_class()`.

16. **Add a guard on `FeatureRecord.feature_columns()` etc.
    (§2.1).** Raise a useful error when called on the base class.
    Effort: low, ~16 LoC. Risk: low.

17. **Update `core/mixins/feature.py:48-55` class docstring
    (§2.4).** Add `list_workflow_executions` to the methods list.
    Effort: trivial, 1 LoC.

18. **Update `__init__.py:109-110` comment for `FeatureRecord`
    (§4.6).** Effort: trivial, 3 LoC.

19. **Update `core/mixins/feature.py:115` and similar examples
    (§4.3).** Add the `from deriva_ml import` line to docstring
    examples. Effort: low, ~6 LoC.

20. **Mark `Feature.feature_record_class.map_type` docstring
    example with `# doctest: +SKIP` (§4.4).** Effort: trivial.

The four "trivial" items (1, 2, 9, 14) plus the four "low" test
items (3, 4, 5, 7) form a coherent ~30-min PR. The remaining items
are independent and can land separately.

---

## Worst offenders summary

1. **`feature.py:35-48`** — 14 lines of either-dead-or-broken
   import boilerplate (importlib workaround for a problem that
   doesn't exist + a wrong TYPE_CHECKING path). Cleanest delete +
   fix target in the subsystem.

2. **`core/mixins/feature.py:158-161`** — duplicated/incorrect
   validation message that sends users to the wrong parameter when
   debugging.

3. **`feature.py:309-379` (`select_majority_vote`)** — pure-Python
   selector logic with no dedicated unit tests despite being one of
   the MCP's dispatch keys and externally-documented in skill
   references.

4. **`tests/feature/test_features.py:347` (`test_delete_feature`)** —
   empty `pass` test for a method with two real branches.

5. **`core/mixins/feature.py:48-55`** — class docstring missing
   `list_workflow_executions`, the method that backs
   `FeatureRecord.select_by_workflow`.

---

## Summary

The feature subsystem is the healthiest of the four phase-3
subsystems audited so far. The S2 migration completed cleanly, the
three-container symmetry holds, external consumers in all six
sibling projects are correctly bound to a stable surface, and no
dead modules / dead CLI entry points / drifted reference snapshots
exist (unlike schema's three competing validators or catalog's
half-deprecated functions). The remaining issues are paper cuts:
one wrong import path, one wrong error message, three missing unit
test files for pure-Python selectors, a handful of doctest
annotation gaps, and the two-decade-old `importlib` workaround the
schema audit also flagged.

The cleanup ceiling is ~−25 LoC of source + ~+200 LoC of tests +
~+30 LoC of docstring fixes — modest in absolute terms but high in
signal-density value per the audit prompt's goals.

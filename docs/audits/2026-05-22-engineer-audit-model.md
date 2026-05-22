# Engineer audit — model subsystem (v1.37.1)

**Date:** 2026-05-22
**Scope:**

- `src/deriva_ml/model/__init__.py` — re-exports + lazy loaders
- `src/deriva_ml/model/annotations.py` — annotation builders (public API for `deriva-skills`)
- `src/deriva_ml/model/catalog.py` — `DerivaModel` (introspection, `find_features`, `is_*`, `from_cached`, …)
- `src/deriva_ml/model/database.py` — `DatabaseModel` (bag-backed deriva-ml database)
- `src/deriva_ml/model/denormalize_planner.py` — `DenormalizePlanner` (~1840 LoC, Rule 2/5/6 enforcement)
- `src/deriva_ml/model/deriva_ml_bag_view.py` — read-only catalog view over a bag
- `tests/model/` — entire directory (`test_annotations.py`, `test_asset_metadata_columns.py`, `test_data_sources.py`, `test_database.py`, `test_derivamodel_from_cached.py`, `test_exceptions.py`, `test_find_association.py`, `test_fk_orderer.py`, `test_handles.py`, `test_is_asset.py`, `test_models.py`)

## Summary

The largest subsystem in `src/` (~6 800 LoC counting tests), and the one with the
most variance in code quality between files. The four files differ in
character: `annotations.py` is a clean public-API builder library with
strong coverage; `catalog.py` is the well-documented but coverage-poor
introspection layer most of `core/mixins/` reaches through; `denormalize_planner.py`
is a ~1800 LoC algorithm subsystem with rigorous docstrings but only
indirect test coverage routed through `tests/local_db/` and
`tests/dataset/`; and `deriva_ml_bag_view.py` is a thin protocol-shaped
view layer whose own write-refuses are completely uncovered.

The headline issues are coverage gaps and duplication, not correctness
bugs. The `tests/model/` directory has zero direct tests for
`DatabaseModel`, `DenormalizePlanner` (as a class), most of
`DerivaMLBagView`, and the majority of `DerivaModel`'s read methods —
all the integration-style verification lives in adjacent test
directories (`tests/local_db/`, `tests/dataset/`, `tests/feature/`).
`test_models.py` is a 7-line empty stub. The annotation builders have a
strong external-consumer-contract test suite that pins the
deriva-skills surface, which is the right pattern to copy for the
other model files.

Three real liabilities stand out:

- **P1 — `_build_join_tree` raises bare `DerivaMLException` while its
  sibling `_prepare_wide_table` raises the typed
  `DerivaMLDenormalizeAmbiguousPath`** for the same conceptual error.
- **P1 — `DatabaseModel.dataset_version` and `rid_lookup` raise bare
  `DerivaMLException`** where `DerivaMLDatasetNotFound` is available.
- **P1 — `tests/model/test_data_sources.py` and `tests/model/test_fk_orderer.py`
  belong upstream** in `deriva.bag` — they test `BagDataSource` and
  `ForeignKeyOrderer` which both live in `deriva.bag.sources` /
  `deriva.bag.loader`, not in `deriva_ml/model/`. They drag ~520 LoC of
  test code into the wrong directory.

The rest is polish: ~6 docstring/annotation drifts, ~8 duplication
spots in the annotation builders' `to_dict()` and in the planner's
predicate plumbing, a couple of dead-code spots
(`_system_schemas`, `TAG_SOURCE_DEFINITIONS`, the `WWW` magic schema),
and substantial coverage gaps in the `DerivaModel` introspection
methods that `core/mixins/` depends on.

Total findings: **45** (0 × P0, 15 × P1, 22 × P2, 8 × P3).

---

## Findings by module

### `src/deriva_ml/model/__init__.py`

#### M-1 [P3] `__all__` advertises lazy names but the only consumer path is direct import
`__init__.py:55-93` lists `"DatabaseModel"` and `"DerivaMLBagView"` in
`__all__`, supported by `__getattr__` at lines 96-106. Every consumer in
the workspace (`dataset/dataset.py:72`, `execution/execution.py:88`,
`tests/model/test_database.py:7`, `tests/dataset/test_download.py:14`)
imports from the concrete module — `from deriva_ml.model.database
import DatabaseModel` — not from the package root. The lazy machinery
exists but no caller uses it. Either delete `__getattr__` and drop the
two names from `__all__`, or document an intended consumer.

### `src/deriva_ml/model/annotations.py`

#### M-2 [P2] `AnnotationBuilder.to_dict` raises `NotImplementedError` but is never invoked polymorphically
`annotations.py:282-294` declares an `AnnotationBuilder` base class with
abstract `tag` and `to_dict()`. Five subclasses inherit
(`Display`, `VisibleColumns`, `VisibleForeignKeys`, `TableDisplay`,
`ColumnDisplay`). The only code that consults `AnnotationBuilder` as a
type is the test suite's `EXPORTED_BUILDERS` tuple
(`test_annotations.py:509`). The base class buys no polymorphism in
production. Either delete it (subclasses don't need to inherit;
they're tagged via class-level `tag = TAG_*`) or make it an actual
`Protocol`/`ABC` and have `apply_annotations()` accept the protocol.

#### M-3 [P2] `TAG_SOURCE_DEFINITIONS` is defined but never used or exported
`annotations.py:274` defines
`TAG_SOURCE_DEFINITIONS = "tag:isrd.isi.edu,2019:source-definitions"`.
It is not referenced anywhere in `src/`, `tests/`, or in the cross-workspace
grep (`deriva-mcp`, `deriva-skills`, `deriva-ml-skills`, `deriva-ml-mcp`,
`deriva-mcp-core`, `deriva-ml-model-template`). Delete.

#### M-4 [P2] 18 `to_dict()` methods all follow the same "if not None, copy" pattern
The pattern (`Display`, `NameStyle`, `PseudoColumn`, `PseudoColumnDisplay`,
`TableDisplayOptions`, `ColumnDisplayOptions`, `PreFormat`, `Facet`,
`FacetRange`, ...) repeats 18 times in identical form:
```python
def to_dict(self) -> dict[str, Any]:
    result = {}
    if self.field1 is not None:
        result["field1"] = self.field1
    if self.field2 is not None:
        result["field2"] = self.field2.value  # enum
    ...
```
A single helper `_dataclass_to_dict(self, *, exclude_none=True, enum_value=True)`
on a shared base would shrink these to one line each and stop the next
contributor having to remember to wire up the optional-field skip on
every new field added to a dataclass. Worth doing because the
annotations module is public-API and changes here propagate to
deriva-skills.

#### M-5 [P2] `Facet`, `FacetList`, `PseudoColumn`, `SortKey`, `FacetRange`, etc. don't inherit from `AnnotationBuilder`
`annotations.py:606-729 ("PseudoColumn")`, `annotations.py:1170-1242 ("Facet")`,
`annotations.py:1245-1264 ("FacetList")`, `annotations.py:1141-1167
("FacetRange")`, `annotations.py:428-448 ("SortKey")`,
`annotations.py:485-522 ("OutboundFK"/"InboundFK")`. These all carry
their own `to_dict()` but no `.tag` because they're **inside**
annotations, not top-level annotations. This is correct, but the
contrast with the `AnnotationBuilder`-bearing five is uncommented and
will read as missing-inheritance to the next reader. Either add a
one-line comment block separating "tagged annotations" from "helpers
embedded inside annotations", or unify the base class (see M-2).

#### M-6 [P3] `VisibleColumns._contexts` is a single-leading-underscore "internal" but documented for direct user access
`annotations.py:816` declares `_contexts: dict[str, list[ColumnEntry] | str]
= field(default_factory=dict)`. The class docstring at
`annotations.py:806-811` explicitly tells users to write to it directly
for faceted search:
```
>>> facets = FacetList()
>>> facets.add(Facet(source="Status", open=True))
>>> vc._contexts["filter"] = facets.to_dict()
```
A leading-underscore name that's also a documented public-API knob is a
contradictory signal. Either expose `vc.filter(...)` as a first-class
method (it would mirror `compact`/`detailed`/`entry`) or rename
`_contexts` to `contexts`.

#### M-7 [P2] `Display(name="X", markdown_name="Y")` test uses bare `pytest.raises(ValueError)`, not the typed exception path
`test_annotations.py:59-62` asserts `ValueError` is raised when both
`name` and `markdown_name` are passed to `Display`. The check at
`annotations.py:400-402` is `raise ValueError(...)`. This is fine, but
it's inconsistent with the rest of the codebase where typed
`DerivaMLValidationError` subclasses are the convention. Annotation
builders deliberately use `ValueError` because they're public-API
helpers and `ValueError` is the Pythonic signal for "bad combination of
args" — but document this in the class docstring so the choice is
explicit. Currently it reads as an oversight.

#### M-8 [P3] `Facet.choices: list[Any]` loses type information
`annotations.py:1199` declares `choices: list[Any] | None = None`. The
field accepts text choices in practice
(`Facet(source="Status", choices=["Active", "Inactive"])`). Narrow to
`list[str | int | float | bool | None] | None` so the schema is
self-documenting and IDE autocomplete works in deriva-skills.

#### M-9 [P3] `Facet.array_options: dict[str, Any]` is a structured config typed as a free-form dict
`annotations.py:696` (`PseudoColumn.array_options`) declares
`dict[str, Any] | None = None`. The valid sub-keys are
`max_length: int` and `order: list[SortKey]`. A nested dataclass
(`ArrayOptions`) would let IDE autocomplete catch typos here too.
Same pattern as M-8.

### `src/deriva_ml/model/catalog.py`

#### M-10 [P1] `asset_metadata_columns` uses raw `str | Table` while every sibling uses `TableInput`
`catalog.py:702` reads `def asset_metadata_columns(self, table: str | Table)`
while 10 sibling methods (`name_to_table`, `is_vocabulary`, `vocab_columns`,
`is_association`, `find_association`, `is_asset`, `find_features`,
`lookup_feature`, `asset_metadata`) all type as `TableInput`. The
type alias exists at `catalog.py:69` precisely for consistency. Drop
the inline form.

#### M-11 [P1] `DerivaModel.__init__` builds `self._system_schemas` and never reads it
`catalog.py:113` sets
`self._system_schemas = frozenset(SYSTEM_SCHEMAS | {ml_schema})`.
Cross-workspace grep finds zero readers (only this write line). Dead
code from an earlier refactor — `is_system_schema` delegates to
`_is_system_schema(schema_name, self.ml_schema)` in
`core/definitions.py` instead. Delete.

#### M-12 [P2] `name_to_table` hard-codes `"WWW"` as a search-path schema
`catalog.py:387-388`:
```python
# Search domain schemas (sorted for deterministic order), then ML schema, then WWW
search_order = [*sorted(self.domain_schemas), self.ml_schema, "WWW"]
```
The `"WWW"` schema is a Deriva convention for web-content tables, not a
`deriva-ml` concept. The hard-coded magic constant should at minimum
live in `core/definitions.py` next to `SYSTEM_SCHEMAS`, with a comment
explaining why `WWW` is searched (the rest of the function uses
`domain_schemas` + `ml_schema`, both configured per-instance). If
`WWW` is just legacy support for a single deployment, justify it in a
comment; otherwise extract.

#### M-13 [P1] `find_features` (no-arg branch) has subtle dedup logic but no `tests/model/` test
`catalog.py:603-645`. The cross-schema dedup at 628-644 (the fix for
the 2026-05-19 duplicate-features bug — see the in-line comment block)
is correctness-critical and lives entirely in this single function.
`tests/model/` does not exercise it; the regression test
(`docs/bugs/2026-05-19-find-features-duplicates.md`) is referenced in
the comment but the actual regression-pinning test lives at
`tests/feature/test_features.py:419-425` (4 lines: a happy-path assertion
that `Health` appears in the result). Nothing pins the dedup behavior
when the same association is reachable through three schemas. Add a
unit test in `tests/model/test_catalog.py` (new file) that constructs
the exact triangle described in the comment and asserts that
`find_features(None)` returns a single Feature.

#### M-14 [P1] `get_schema_description` has zero unit tests but is consumed by `deriva-mcp`
`catalog.py:242-349` defines a substantial JSON-shape method that
populates `is_vocabulary`, `is_asset`, `is_association`, FK info, and
optionally `features`. Cross-workspace grep confirms it is called by
`deriva-mcp/src/deriva_mcp/resources.py:445`,
`deriva-mcp/src/deriva_mcp/rag/helpers.py:88`,
`deriva-mcp/src/deriva_mcp/tools/rag.py:416`,
`deriva-mcp/src/deriva_mcp/rag/__init__.py:478`, plus four test files
that mock its return value. So this is a **shape contract**.
`tests/model/` has zero direct tests of this method. Any change to
the returned shape silently breaks the legacy MCP. Pin the shape with
a snapshot test that constructs a small catalog and asserts on the
returned dict keys + nested structure.

#### M-15 [P1] `is_dataset_rid` has no direct unit test
`catalog.py:752-784`. Used at 7 call sites in `core/mixins/dataset.py`
and `dataset/dataset.py`. The function has a non-obvious branch around
`deleted=True`/`False` and `Deleted` column reading
(`return not list(rid_info.datapath.entities().fetch())[0]["Deleted"]`).
The KeyError → `DerivaMLException` translation at 776-777 also has no
coverage. Add unit tests for the three branches: live RID, deleted
RID with `deleted=False`, deleted RID with `deleted=True`, and invalid
RID.

#### M-16 [P2] `refresh_model` has no docstring and no direct unit test
`catalog.py:234-235`. Two-line method; the docstring would describe
"re-fetch the catalog model from the server, replacing
`self.model`." Used at `core/base.py:867` and 5 sites in
`tests/dataset/`. Add docstring + one test that mutates the catalog
out-of-band, calls `refresh_model`, and asserts the change is visible.

#### M-17 [P2] `chaise_config` property has no docstring beyond "Return the chaise configuration"
`catalog.py:238-240`. The one-line docstring is fine for trivial
properties but `chaise_config` is the head-of-line knob for catalog UI
configuration and the doctest example would be runnable
(`>>> ml.model.chaise_config["navbarBrandText"]`).

#### M-18 [P2] `apply()` `CatalogStub` refusal is tested only implicitly
`catalog.py:727-750`. The offline-mode refusal at 746-749 is a
correctness-critical guard against silent schema changes. No test in
`tests/model/` constructs an offline DerivaModel and asserts the
`DerivaMLReadOnlyError`. Add one.

#### M-19 [P1] `is_association` return-type annotation is `bool | set[str] | int` but the docstring says "True/False" for the common path
`catalog.py:453`: `def is_association(...) -> bool | set[str] | int`.
The Args block discusses `unqualified=True`'s side effect on the
return type but the call sites (`catalog.py:338`,
`core/mixins/dataset.py`, etc.) treat it as `bool`. Splitting into
`is_association(...) -> bool` (with a narrower contract) and
`association_details(...) -> set[str] | int | None` would let the
common path stop wrapping the return in `bool(...)` like
`get_schema_description` does at `catalog.py:338`.

#### M-20 [P2] `_define_association` is a private method documented as internal but accessed from tests
`catalog.py:898-949`. The method has a substantial docstring and a
real responsibility (vocab-aware FK key resolution) but the leading
underscore signals "do not consume." `tests/model/test_find_association.py:76,111`
reach into it directly. Either expose it (`define_association`) so
test calls stop reading like internal hackery, or document explicitly
in the docstring that it's "test-only internal" — which it isn't
(zero non-test callers cross-workspace, but the implementation looks
load-bearing).

#### M-21 [P2] `find_assets` and `find_vocabularies` lack `Example:` blocks
`catalog.py:551-562`. Two of the most-used introspection methods. Add
`Example:` blocks (skipped — `# doctest: +SKIP`) for symmetry with
`find_features`.

#### M-22 [P3] `from_cached` docstring shows `catalog=CatalogStub()` only in the prose, not as a runnable doctest
`catalog.py:141-193`. The minimal-schema-dict path is exercised by
`test_derivamodel_from_cached.py:13-23` but the docstring example
shows no callable form. Add a doctest equivalent (the test pattern
fits inline: `>>> schema_dict = {"schemas": {...}, ...}; >>> model =
DerivaModel.from_cached(schema_dict, catalog=CatalogStub())`).

### `src/deriva_ml/model/database.py`

#### M-23 [P1] `dataset_version` and `rid_lookup` raise bare `DerivaMLException`
`database.py:193`: `raise DerivaMLException(f"Dataset RID {rid} is not in this bag")`
`database.py:217`: `raise DerivaMLException(f"Dataset {dataset_rid} not found in this bag")`
The codebase ships `DerivaMLDatasetNotFound` (`core/exceptions.py:239`)
exactly for this case. Inheritance preserves the legacy catch path
but the typed subclass surfaces the dataset RID via the structured
field — the same pattern issue #180 / commit a1f8e22 applied to
`find_association`. Convert both raises to `DerivaMLDatasetNotFound(rid)`.

#### M-24 [P1] `DatabaseModel.dataset_version` and `rid_lookup` are tested only indirectly through bag downloads
`database.py:178-217`. Both methods are public — `dataset_version()` is
in the public interface and `rid_lookup` is called from
`deriva_ml_bag_view.py:137,368`. No `tests/model/test_database.py`
function tests either directly. Their unit-testability is high (build
a `DatabaseModel` from a fixture bag), so the absence is a coverage
gap rather than a hard limitation.

#### M-25 [P2] Module-level `logger` and instance `self._logger` are both defined
`database.py:42`: `logger = get_logger(__name__)` (module-level)
`database.py:94`: `self._logger = get_logger(__name__)` (instance)
`database.py:141,145`: only `self._logger` is called. The module-level
`logger` is dead. Pick one (instance is needless because both share
`__name__`); delete the other.

#### M-26 [P2] `_build_bag_rids` "highest version wins" semantics are uncovered
`database.py:151-176`. The "keep highest version" rule at lines
175-176 (`if rid not in self.bag_rids or version > self.bag_rids[rid]`)
is a behavior contract for nested datasets — a downloaded bag can
contain multiple historical versions of nested datasets, and this
method picks one per RID. No test pins this behavior. The
`test_table_versions` test in `test_database.py:145-194` exercises bag
version-change semantics end-to-end but the specific "two versions of
the same RID in the same bag" case isn't constructed.

### `src/deriva_ml/model/denormalize_planner.py`

#### M-27 [P1] `_build_join_tree` raises bare `DerivaMLException` for an ambiguous path; sibling raises the typed exception
`denormalize_planner.py:1548-1551`:
```python
raise DerivaMLException(
    f"Ambiguous path between {element_name} and {target}: "
    f"found {len(unique)} FK paths:\n" + ...
)
```
The Rule 6 ambiguity at the top level (`_prepare_wide_table` →
`_find_path_ambiguities`) raises the typed
`DerivaMLDenormalizeAmbiguousPath`
(`denormalize_planner.py:1692`). The path here in `_build_join_tree`
is logically the same failure mode — multiple FK paths between two
tables that the caller didn't disambiguate — but it surfaces as a
bare exception. Convert to `DerivaMLDenormalizeAmbiguousPath(element_name, target, paths, suggested_intermediates)`.

#### M-28 [P2] `valid_schemas = self.model.domain_schemas | {self.model.ml_schema}` is recomputed three times
`denormalize_planner.py:517, 575, 666`. Same expression, same value
within a single call. Extract to a `_valid_schemas` cached property
on the planner. Bonus: makes the schema-membership rule one named
thing instead of a phrase repeated by hand.

#### M-29 [P2] `domain_fks = [fk for fk in fks if fk.pk_table.name not in ("ERMrest_Client", "ERMrest_Group")]` is duplicated
`denormalize_planner.py:378` (in `_is_topological_association`)
`denormalize_planner.py:443` (in `_is_feature_association`)
Both predicates need to ignore the system FKs to `ERMrest_Client` /
`ERMrest_Group`. Extract a helper:
```python
def _domain_fks(self, tbl) -> list:
    return [fk for fk in tbl.foreign_keys if fk.pk_table.name not in _SYSTEM_FK_TARGETS]
```
The two predicates also share the `try/except: return False` wrap (a
defensive pattern that should be documented or eliminated — see M-31).

#### M-30 [P1] `_is_topological_association` and `_is_feature_association` swallow all exceptions
`denormalize_planner.py:373-381, 438-452`:
```python
try:
    tbl = name_or_table if hasattr(name_or_table, "foreign_keys") else self.model.name_to_table(name_or_table)
    fks = list(tbl.foreign_keys)
    ...
    return len(domain_fks) == 2
except Exception:
    return False
```
The bare `except Exception: return False` masks real bugs (e.g. a
typo'd table name silently returns False instead of `DerivaMLException`).
The hot path is `name_to_table(name_or_table)` which raises
`DerivaMLException` if the name is unknown — catching only that
specific exception would preserve the "missing table is not an
association" semantics without hiding real errors.

#### M-31 [P2] Rule 2 (sink-finding) docstring claims "exactly one sink" but `_find_sinks` returns 0+
`denormalize_planner.py:953-1004`. The docstring at 982-985 says
"Normally exactly one. Multiple sinks → caller should raise
`DerivaMLDenormalizeMultiLeaf`. Zero sinks → cycle, caller should
raise `DerivaMLDenormalizeNoSink`." The method itself returns the
sorted list and lets `_determine_row_per` enforce the cardinality.
That's fine in code but the docstring reads as if the method enforces
it. Reword to "Sink table names (cardinality not enforced — see
`_determine_row_per` for Rule 2 enforcement)."

#### M-32 [P1] `DenormalizePlanner` itself has no direct unit tests; all coverage is indirect via `tests/local_db/`
`denormalize_planner.py` is ~1840 LoC of Rule 2/5/6 logic. The class
is consumed via `DerivaModel._planner` and tested at:

- `tests/local_db/test_planner_rules.py` — exercises `_find_sinks`,
  `_determine_row_per`, `_find_path_ambiguities`, `_prepare_wide_table`,
  `_is_topological_association`, `_is_feature_association` against a
  populated catalog
- `tests/dataset/test_schema_paths.py` — exercises `_schema_to_paths`
- `tests/dataset/test_denormalize.py:333` — one direct call
- `tests/dataset/test_composite_fk_denormalize.py:176` — one direct call

No file inside `tests/model/` mentions `_planner` or `DenormalizePlanner`.
Moving the planner-rules tests into `tests/model/test_denormalize_planner.py`
(or symlinking) would put the tests next to the code. At minimum,
add a `tests/model/test_planner.py` smoke test that constructs a
`DenormalizePlanner` directly (not via the lazy property) to pin the
1-arg constructor contract.

#### M-33 [P2] `_outbound_reachable` vs `_outbound_reachable_strict` distinction is essential but not tested in isolation
`denormalize_planner.py:586-694, 696-778`. The split between the two
primitives is the difference between Rule 5/2's enforcement
(directional fan-out — strict) and Denormalizer's anchor-classification
(connectivity probe — non-strict). The docstrings explain this at
length (75 lines combined). No test pins the divergence with a
ground-truth example like "bridge case (Image ← feature-assoc →
Image_Classification): `_outbound_reachable` returns
`{Image_Classification}`, `_outbound_reachable_strict` returns
`set()`." A 10-line test would lock in the rule that future
contributors won't accidentally unify.

#### M-34 [P2] Recursive `_schema_to_paths` has no `max_depth` default but the docstring at 816 implies one
`denormalize_planner.py:786, 829-842`. Signature: `max_depth: int | None
= None`. The docstring at line 816 reads "Use to protect against
pathological schemas with deep chains." A real catalog can have
20+ levels of FK chains in pathological cases. The cycle-detection
at line 890-893 catches simple cycles via `if child in path`. With
`max_depth=None` and a complex schema, the recursion can still blow
up. Set a default like `max_depth=20`; document the choice.

#### M-35 [P3] `_DEFAULT_SKIP_TABLES` is module-level frozenset; should be a class attribute
`denormalize_planner.py:294`. Currently a module-level
`_DEFAULT_SKIP_TABLES = frozenset({"Dataset_Dataset", "Execution"})`.
Subclasses or alternate planners can't customize. Promote to a
class attribute on `DenormalizePlanner`.

### `src/deriva_ml/model/deriva_ml_bag_view.py`

#### M-36 [P1] Write-operation refusals (`create_dataset`, `pathBuilder`, `catalog_snapshot`) have zero coverage
`deriva_ml_bag_view.py:316-352`. Three methods that exist purely to
raise `DerivaMLException` and document the bag-is-read-only contract.
No test asserts the refusal. Trivial to add — fixture bag,
construct `DerivaMLBagView`, call each, expect the exception. Without
this coverage, a future change that silently makes one succeed (e.g.
delegates to `_database_model` by accident) ships unnoticed.

#### M-37 [P2] `lookup_term` has a slow-path comment that says "scan for synonyms" but the slow path also matches `Name`
`deriva_ml_bag_view.py:218-239`. The fast path (218-229) does
`WHERE Name = term_name`. The slow path (231-239) iterates rows and
checks `term_name in term.get("Synonyms", [])` — but the loop also
iterates over the `Name` column without a name-match check on the
slow-path side. If the fast path missed a row (e.g. because the
`Name` column was cased differently than expected), the slow path
would miss it too. Worth a comment that says explicitly "slow path
ONLY catches synonym matches; name matches are owned by the fast
path."

#### M-38 [P2] `find_datasets` and `lookup_dataset` are tested only indirectly via `compare_catalogs`
`deriva_ml_bag_view.py:118-190`. `tests/model/test_database.py` is the
only file that touches these and it uses them as helpers in a larger
catalog-comparison flow. No direct test asserts the shape of
`find_datasets`'s return (e.g. that a bag with one dataset returns a
single `DatasetBag` with the expected RID, description, dataset_types).

#### M-39 [P2] `_get_dataset_execution` returns `None` for missing-RID but raises nothing for the bag-rids miss
`deriva_ml_bag_view.py:243-271`. The function pattern is "if version
is None → return None; else SQL select and return dict-or-None". A
miss inside `bag_rids` returns `None`; a miss in the
`Dataset_Version` SQL query also returns `None`. Callers can't
distinguish the two failure modes. Either a single typed-exception path
or a `Result`-shaped return would help.

### Tests (`tests/model/`)

#### M-40 [P1] `tests/model/test_data_sources.py` tests `deriva.bag.sources.BagDataSource` — it does not belong in `tests/model/`
`test_data_sources.py:12`: `from deriva.bag.sources import BagDataSource`.
The file is 141 LoC of `BagDataSource` tests (multi-CSV handling, nested
dataset CSVs, etc.). `BagDataSource` is upstream in `deriva.bag` and
has nothing to do with `deriva_ml/model/`. These tests should live in
`deriva.bag`'s test suite. Move (or, if the upstream coverage is
sufficient, delete).

#### M-41 [P1] `tests/model/test_fk_orderer.py` tests `deriva.bag.loader.ForeignKeyOrderer` — same issue
`test_fk_orderer.py:14`: `from deriva.bag.loader import ForeignKeyOrderer`.
The file is 373 LoC of `ForeignKeyOrderer` tests (topological sort,
cycle handling, etc.). Same dispatch as M-40 — move upstream or
delete. Together with `test_data_sources.py` that's ~520 LoC of
test code in the wrong directory.

#### M-42 [P3] `tests/model/test_models.py` is a 7-line empty class
`test_models.py:7`: `class TestCatalogModel: pass`. Empty stub.
Either populate (with the catalog tests M-13/M-14/M-15/M-19 ask for)
or delete.

#### M-43 [P2] `test_database.py::TestDataBaseModel` test class name uses non-standard case
`test_database.py:16`: `class TestDataBaseModel`. The convention is
`TestDatabaseModel` (matching the class under test). Trivial rename.

#### M-44 [P1] No tests for `DerivaMLBagView.resolve_rid`
`deriva_ml_bag_view.py:353-369`. The method is part of the
`DerivaMLCatalog` protocol surface that the bag view implements. No
test exercises it. The contract `{"RID": rid, "version": version}` is
not pinned.

#### M-45 [P3] `test_annotations.py::TestExternalConsumerContract` only covers 5 of the ~10 builders
`test_annotations.py:509-515`. `EXPORTED_BUILDERS` lists `Display`,
`VisibleColumns`, `VisibleForeignKeys`, `TableDisplay`, `ColumnDisplay`.
The annotation module also exports `PseudoColumn`, `Facet`,
`FacetList`, `PreFormat`, `SortKey`, `InboundFK`, `OutboundFK`,
`NameStyle`, `TableDisplayOptions`, `ColumnDisplayOptions`,
`PseudoColumnDisplay`, `FacetRange`. The contract test ("class has
`.tag`, instance has `.to_dict()`") doesn't apply to all of these
(no `.tag` on `PseudoColumn`, etc.) but the JSON-roundtrip check
should. Add a second contract test that exercises every exported
helper's `to_dict()`.

## Cross-module coverage gaps

The `tests/model/` directory **has no direct tests for**:

- `DatabaseModel` (all `tests/model/test_database.py` flows are
  `DerivaMLBagView`-routed integration tests; `DatabaseModel`'s own
  surface — `dataset_version`, `rid_lookup`, `_build_bag_rids` — is
  uncovered as a unit)
- `DenormalizePlanner` as a class (tested indirectly via
  `tests/local_db/test_planner_rules.py` and friends)
- `DerivaModel` methods: `is_dataset_rid`, `get_schema_description`,
  `refresh_model`, `apply` offline-mode refusal, `find_assets`,
  `find_vocabularies`, `is_association` (no live test — only
  mock-based), `vocab_columns`, `lookup_feature`
- `DerivaMLBagView` write-refusals (`create_dataset`, `pathBuilder`,
  `catalog_snapshot`), `resolve_rid`, `lookup_term` (one assertion in
  `tests/dataset/test_download.py`)
- `__getattr__` lazy-import paths in `model/__init__.py`

The annotation builders are well-covered. The catalog / database /
bag-view / planner surfaces lean on neighboring test directories
(`tests/local_db/`, `tests/dataset/`, `tests/feature/`,
`tests/schema/`) for indirect coverage. The collateral damage:
moving `model/` files becomes harder because the tests that pin their
behavior live elsewhere.

## Cross-module duplication candidates

- **18 × `to_dict()` "if not None: copy" pattern** in `annotations.py`
  (M-4). Extracting a single helper would cut ~120 LoC and make
  field-addition mechanical.
- **`valid_schemas = domain_schemas | {ml_schema}`** in
  `denormalize_planner.py` (M-28).
- **`domain_fks = [fk for fk in fks if fk.pk_table.name not in
  ("ERMrest_Client", "ERMrest_Group")]`** in two predicate methods of
  `denormalize_planner.py` (M-29).
- **`try: ... self.model.name_to_table(name_or_table) ...; except
  Exception: return False`** in two predicate methods (M-30).
- **`__getattr__`-style model delegation** appears once in
  `catalog.py:351` and once implicitly via `BagDatabase`. Not
  duplicated per se, but the interaction with `BagDatabase.schemas`
  (instance-attribute, blocks `@property`) is documented in a comment
  rather than enforced — a regression could re-add the property and
  silently break `DatabaseModel`.

## Notes

- `annotations.py` is the cleanest file in the subsystem and the right
  template for the rest: it has a strong public-API surface, the
  `TestExternalConsumerContract` test (`test_annotations.py:483-602`)
  pins the deriva-skills contract end-to-end, and the module docstring
  is genuinely useful as documentation. The dedup opportunities (M-4)
  are the largest single LoC reduction available in the subsystem but
  the current code is correct and the duplication is **mechanical, not
  semantic** — refactor before adding new annotation tags, not as a
  P0.
- `denormalize_planner.py` has the best in-file documentation in the
  whole codebase (the module docstring at 1-197, then per-method
  docstrings averaging 30 lines). Coverage routing it through
  `tests/local_db/` is acceptable in principle (the planner is private
  to the denormalization subsystem) but a few direct tests in
  `tests/model/` would make the boundary explicit.
- The `deriva-skills/use-annotation-builders` external contract is
  load-bearing (confirmed via cross-workspace grep at
  `deriva-skills/.../use-annotation-builders/{SKILL.md,references/builder-api.md}`).
  Any rename or signature change in `annotations.py` is a breaking
  change for that skill.
- `from_cached` and `CatalogStub` form the offline-mode entry point
  used by `core/base.py` lines 466, 502, 594, 864. The single test in
  `test_derivamodel_from_cached.py` confirms it works with a minimal
  schema dict, but the apply-refusal path
  (`catalog.py:746-749`) — the one that prevents accidental schema
  changes in offline mode — has zero coverage (M-18).

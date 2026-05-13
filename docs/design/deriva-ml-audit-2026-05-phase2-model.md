# deriva-ml audit 2026-05 — Phase 2: model/

Reviewed `src/deriva_ml/model/` (4 production files, 3 717 LoC) on
`fix/catalog-manager-state-guards` HEAD, with Phase 1's cleanup
sprint already applied (no shim modules, no `_get_table_contents`
/ `get_orm_association_class` / `delete_database` aliases, single
`list_dataset_element_types`, unified navbar, unified logger init).
External-boundary checks against
`/Users/carl/GitHub/deriva-py/deriva/{core,bag}/`. Findings below
are scoped to what is left after Phase 1; items the audit closed
are not re-flagged.

## Executive summary

**Posture: structurally settled, content overgrown.** Phase 1
collapsed the shim layer and the back-compat aliases, so `model/`
no longer hides its real surface area behind re-exports. What
remains is the original substance: a 2 083-line `DerivaModel` that
genuinely does three jobs, a 1 264-line `annotations.py` builder
library with **zero callers in `src/`**, a clean 250-line
`DatabaseModel` inheritance layer, and a 316-line `DerivaMLBagView`
that is mostly thin delegation but has one notable gap. Phase 1's
recommendation to extract the denormalization planner is still the
correct Phase-3 move; it is no smaller than it looked.

Top themes ranked by impact:

1. **`model/annotations.py` (1 264 LoC) has no production
   consumers.** Every annotation builder class (`Display`,
   `VisibleColumns`, `TableDisplay`, `ColumnDisplay`, `Facet`,
   `FacetList`, …) is exported from `model/__init__.py` and
   instantiated only by `tests/model/test_annotations.py`. The
   actual navbar builder in `schema/annotations.py` constructs raw
   dicts. Either there are external (MCP / skill) consumers that
   the audit can't see, or this is the largest unused file in the
   codebase. **Severity: high** — 1 264 LoC of test-only code.
2. **`DerivaModel` carries three roles plus a layer of dead type
   aliases.** Phase 1 §6.1 named the denormalization-planner split;
   reading the actual 800 lines confirms it. Beyond the split, the
   class also carries 8 `TypeAlias` declarations
   (`SchemaName`, `ColumnSet`, `AssociationResult`, `TableSet`,
   `PathList`, `FilterPredicate`, `SchemaDict`, `FeatureList`) that
   are referenced **nowhere** — neither inside `catalog.py` nor in
   any consumer. `self.configuration = None` is set in
   `__init__` and never touched again.
3. **`DerivaModel.__getattr__` silently delegates to the underlying
   ermrest `Model`** (`catalog.py:410-412`). This was the
   workaround for "missing methods inherited from `Model`," but it
   means every typo on `model.foo` either returns an unrelated
   attribute or raises `AttributeError` with a misleading message,
   and IDEs cannot complete deriva-py methods on a `DerivaModel`.
   Replacing the delegation with explicit pass-throughs for the
   methods actually used would surface the real boundary.
4. **Two association predicates with overlapping semantics
   survive** — `is_association` (8 callers, delegates to deriva-py
   `Table.is_association`) and `_is_association_table` (11 callers
   including external callers in `local_db/denormalizer.py`).
   Phase 1 §2.7 flagged renaming `_is_association_table` to
   `is_topological_association` and treating it as public. After
   reading the planner's usage, that recommendation still holds —
   the predicate has eight external call sites, the leading
   underscore is misleading, and the docstring at line 712-766
   already documents it as the canonical "topology, not purity"
   variant.
5. **`DerivaMLBagView` is mostly thin pass-through but the
   `lookup_term` path uses an O(n) scan and the
   `find_datasets` path materializes the entire association table
   twice.** Symptoms of "view written quickly to satisfy a
   protocol, not optimized." Low-severity but worth a note.

Worst-offending files within `model/`:

1. **`annotations.py`** — 1 264 LoC, zero `src/` consumers.
2. **`catalog.py`** — 2 083 LoC, three responsibilities, one
   `__getattr__` delegation hatch, one `configuration = None`,
   eight unused type aliases.
3. **`deriva_ml_bag_view.py`** — 316 LoC, clean inheritance but
   one bug (`_get_dataset_execution` is an `_`-prefixed method
   called from a different module) and two O(n²)-feeling loops.
4. **`database.py`** — 250 LoC, clean, no significant issues.

---

## Subsystem inventory

| File                       | LoC   | Posture |
|----------------------------|------:|---------|
| `__init__.py`              |   106 | **Clean post Phase 1.** Single annotation-builder re-export plus a 9-line `__getattr__` lazy-import for the two heavyweight classes. No shim aliases. |
| `annotations.py`           | 1 264 | **In-the-attic.** Comprehensive Chaise-annotation builder library; zero `src/` callers; only `tests/model/test_annotations.py` exercises it. Either delete-and-archive or document the external API contract. |
| `catalog.py`               | 2 083 | **God-class.** Three responsibilities (introspection, denormalization planner, schema mutation) plus dead type aliases and one `__getattr__` delegation escape hatch. Phase-3 split candidate. |
| `database.py`              |   250 | **Healthy.** `DatabaseModel(BagDatabase, DerivaModel)` with a clean `__init__` ordering and three deriva-ml-specific concerns (`_build_bag_rids`, `dataset_version`, `rid_lookup`, `_get_dataset_execution`). Post Phase 1 the back-compat surface is gone. |
| `deriva_ml_bag_view.py`    |   316 | **Mostly clean delegation.** 7 protocol property pass-throughs, ~10 read methods, 3 write-rejects. One method (`list_dataset_element_types`) inherited from `DatabaseModel` is also exposed directly here — duplication is harmless but inconsistent with the rest of the class. |

### `DerivaModel` responsibility map

| Role | Line range | Method count | External callers | Phase verdict |
|---|---|---:|---|---|
| **Introspection** | 254-816 | 21 public methods | All mixins (`core/mixins/*`), `feature.py`, `asset/*`, `dataset/upload.py`, `execution/bag_commit.py`, `schema/annotations.py` | Stays in `catalog.py`. |
| **Denormalization planner** | 819-1837 | 6 `_`-prefixed methods + 1 dataclass (`JoinNode`) + 1 module-level helper (`denormalize_column_name`) | `local_db/denormalize.py:274`, `local_db/denormalizer.py:525/621/625/638/654/828/832/856/920/928/1147/1153/1182/1250`, `core/mixins/dataset.py:449` | **Phase 3 split candidate** (Phase 1 §6.1 recommendation stands after reading the code). |
| **FK traversal primitive** | 1839-2009 | 2 methods (`_table_relationship`, `_schema_to_paths`) | The planner above + `dataset/dataset_bag.py:383` + tests | **Co-locate with the planner.** Both are FK-graph primitives. Keep the introspection-only schema accessors in `catalog.py`. |
| **Schema mutation** | 2011-2083 | 2 methods (`create_table`, `_define_association`) | `core/mixins/feature.py:169`, `core/mixins/asset.py:142-158`, `core/mixins/dataset.py:253`, `core/base.py:1343` | Stays in `catalog.py`. Light usage (≤6 callers). |

---

## Lens 1 — Dead code

### 1.1 `annotations.py` has zero `src/` consumers (1 264 LoC)

`src/deriva_ml/model/annotations.py:282-1264` defines
`AnnotationBuilder` plus 13 concrete builders + 3 enums + 12
context constants + 6 tag-URI constants. The only files in `src/`
that import any of these names are:

- `model/__init__.py:20-52` — re-exports every public name.
- `model/annotations.py` itself.

The only consumers are in `tests/`:

- `tests/model/test_annotations.py:5-25` instantiates every builder.

`schema/annotations.py:218-260` — the function that *actually*
writes Chaise annotations to the catalog — constructs raw dicts
(`{"navbarMenu": build_navbar_menu(model), ...}`), not builder
instances. The dispatcher `apply_catalog_annotations`
(`core/base.py:1061-1115`) delegates to that raw-dict path.

`core/model_handles.py:TableHandle` (re-exported by
`model/__init__.py:18`) is the intended attachment point — the
module docstring (`annotations.py:19-38`) shows
`TableHandle(...).set_annotation(Display(...))`. **No call to
`TableHandle.set_annotation` exists anywhere in `src/`**
(`grep -rn "set_annotation"` returns nothing).

There are three legitimate possibilities:

(a) The builders are a public API for external (MCP /
    skill / model template) consumers. In that case
    `annotations.py` should be in the documented public surface
    and CI should test against actual external usage — neither is
    true today.

(b) The builders were written speculatively for the navbar
    consolidation (Phase 1 §4.1) but `schema/annotations.py` ended
    up using raw dicts instead. In that case `annotations.py` is
    1 264 LoC of dead code.

(c) The builders are deliberate dev-ergonomics scaffolding for a
    future migration. In that case there should be a note (ADR or
    `CLAUDE.md` entry) saying so.

**Action:** classify (a/b/c) with the maintainer. If (b),
remove the module + its `__init__.py` re-exports + the
`tests/model/test_annotations.py` 480-line test file. **Risk:
low if (b); high if (a).** **LoC: −1 744 if (b).**
**Severity: high.**

### 1.2 Eight unused `TypeAlias` declarations in `catalog.py`

`catalog.py:117-126` declares:

```python
TableInput: TypeAlias = str | Table          # used (5 sites)
SchemaDict: TypeAlias = dict[str, Schema]    # not used
FeatureList: TypeAlias = Iterable[Feature]   # not used
SchemaName = NewType("SchemaName", str)      # not used
ColumnSet: TypeAlias = set[Column]           # not used
AssociationResult: TypeAlias = FindAssociationResult  # not used
TableSet: TypeAlias = set[Table]             # not used
PathList: TypeAlias = list[list[Table]]      # not used
FilterPredicate = Callable[[Table], bool]    # not used
```

Verified by `grep -rn` across `src/` and `tests/`: only
`TableInput` is referenced (in `catalog.py` itself, lines 414,
442, 467, 533, 564, 611, 1841-1842).

**Action:** delete the unused aliases. **Risk: trivial. LoC: −9.**
**Severity: low.**

### 1.3 `self.configuration = None` is dead

`catalog.py:166` — `DerivaModel.__init__` sets
`self.configuration = None` and the attribute is **never read or
written** anywhere else in `model/` or in any consumer. (`grep
-rn "model\.configuration"` and `grep -rn "self\.configuration"
src/deriva_ml/model/` confirm.) The neighboring
`core/base.py:352` sets `self.configuration = None` on the
`DerivaML` class itself — that one **is** later assigned and
read by the execution lifecycle, but it's an independent
attribute on a different class.

**Action:** delete the line. **Risk: trivial. LoC: −1.**
**Severity: low.**

### 1.4 `DerivaModel.find_assets(with_metadata=False)` parameter is ignored

`catalog.py:550-552`:

```python
def find_assets(self, with_metadata: bool = False) -> list[Table]:
    """Return the list of asset tables in the current model"""
    return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_asset(t)]
```

The `with_metadata` parameter is declared but the function body
never references it. Two callers (`dataset/upload.py:306`,
`tests/core/test_offline_mode_smoke.py:67-68`) call it with no
arguments, so the dead parameter is invisible — but it's still
in the signature.

**Action:** delete the parameter. **Risk: low** (no caller
passes it). **LoC: −1.** **Severity: low.**

### 1.5 `DerivaModel.apply()` has a degenerate string-equality guard

`catalog.py:665-670`:

```python
def apply(self) -> None:
    """Call ERMRestModel.apply"""
    if self.catalog == "file-system":
        raise DerivaMLException("Cannot apply() to non-catalog model.")
    else:
        self.model.apply()
```

`self.catalog` is typed as `ErmrestCatalog` in `__init__`
(`catalog.py:167`), not as a string. The only way the equality
test can ever fire is if `from_cached` is called with the
literal string `"file-system"` as the `catalog` argument
(`Model.fromfile("file-system", ...)`), which is exactly what
`DatabaseModel.__init__` does at `database.py:105`. But
`DatabaseModel` inherits this `apply()` and the guard works as
intended for bags.

That said: the type signature is wrong (the catalog is sometimes
a string, sometimes an `ErmrestCatalog`, sometimes a
`CatalogStub` per `from_cached`'s docstring at line 211-218),
the guard is a fragile string check, and the docstring is
terse. A `CatalogStub` instance would silently pass the `!=
"file-system"` test and call `model.apply()` on a stub catalog.

**Action:** introduce a typed `is_apply_capable` check (probably
on the catalog object itself) and replace the string comparison.
**Risk: low-medium.** **LoC: ±0.** **Severity: low.**

### 1.6 `JoinNode.walk_edges` has one caller; both `walk` and `walk_edges` are public on a private dataclass

`catalog.py:79-92` — `JoinNode` exposes `walk()` (one caller at
line 1782 in `_prepare_wide_table`) and `walk_edges()` (one
caller at line 1817). Both are public on a dataclass that is
private to the denormalization planner. When the planner is
extracted (Phase 3), keep the methods package-private to the
planner module.

**Severity: low (cosmetic).** **No action in Phase 2.**

---

## Lens 2 — Deriva-py interface usage

### 2.1 `DerivaModel.is_association` correctly delegates to deriva-py

`catalog.py:486-507` is a thin pass-through to
`Table.is_association(min_arity=2, max_arity=2, unqualified=True,
pure=True)`. The deriva-py implementation is at
`/Users/carl/GitHub/deriva-py/deriva/core/ermrest_model.py:2206-2307`.
Verified: same parameter names, same defaults, no behavior
divergence.

The deriva-py method does support `return_fkeys=True`
(`ermrest_model.py:2304-2307`) which returns the set of N
associated `ForeignKey` objects. This would in principle let
`_is_association_table` (the topological-only variant) be
rewritten as `is_association(pure=False, no_overlap=False,
return_fkeys=True)` plus a filter on the returned fkey set —
but the topological variant is **looser** than even that: it
intentionally ignores the unqualified/no_overlap constraints
and counts only the FKs pointing at non-`ERMrest_Client` /
`ERMrest_Group` targets (`catalog.py:759`). Folding it into
`is_association` would require a larger predicate, with three
new flags. **Not worth the surface change.** Keep
`_is_association_table` as is, but rename — see Lens 4.

### 2.2 `DerivaModel.find_features` uses `Table.find_associations(min_arity=3, max_arity=3, pure=False)` correctly

`catalog.py:563-609` calls
`Table.find_associations(min_arity=3, max_arity=3, pure=False)`
(line 595) — the canonical deriva-py API for "M:N associations
of arity 3 with metadata columns" (`ermrest_model.py:2309-2322`).
The `is_feature` predicate (line 578-590) then checks the column
set for the deriva-ml-specific `{Feature_Name, Execution,
<target>}` triple, which is the right place for the
deriva-ml-domain layer.

**No action.** Good example of how the integration should look.

### 2.3 `DerivaModel.is_dataset_rid` uses `catalog.resolve_rid` correctly

`catalog.py:672-684` calls
`self.model.catalog.resolve_rid(rid, self.model)` — the
canonical pattern from deriva-py's `ErmrestCatalog`. Already
noted as the counterexample in Phase 1 §2.8. **No action.**

### 2.4 `DerivaModel._table_relationship` is unique to deriva-ml — correctly so

`catalog.py:1839-1883`. The method enumerates FK constraints
between two tables (both directions: `table1.foreign_keys`
pointing at `table2`, and `table1.referenced_by` from `table2`).
deriva-py's `Table` exposes both fields directly
(`ermrest_model.py:1006-1011`, then `find_associations` for the
"between two tables" question) — but it does **not** expose a
canonical "give me the FK column pairs joining T1 and T2"
helper. The deriva-ml implementation is the right place for it.

**Note:** the method's docstring (line 1844-1851) is good but the
single-FK-vs-composite distinction is buried. Worth a one-line
Example block.

### 2.5 `DatabaseModel` inheritance ordering is correct

`database.py:45` declares
`class DatabaseModel(BagDatabase, DerivaModel)`. MRO has
`BagDatabase` first because:

- `BagDatabase.__init__` (deriva-py:bag/database.py:117) does
  the SQLite + ORM + schema_meta work that has to run first.
- `DerivaModel.__init__` reads `self.model` (set by the parent
  `__init__`), so it must run second — the explicit ordering
  on `database.py:117-132` enforces this.

`BagDatabase` exposes `get_table_contents`, `find_table`,
`is_association_table`, `get_association_class`, `dispose`,
`__enter__/__exit__` — all inherited cleanly post Phase 1.

**Subtle:** `BagDatabase.is_association_table` is a **static
method** (`deriva-py/deriva/bag/database.py:751`, a delegator to
`deriva.bag._orm_helpers.is_association_table`) operating on
**ORM classes**, while `DerivaModel._is_association_table` is an
**instance method** on `Table` objects. The two coexist on
`DatabaseModel` without clashing because their signatures are
distinct (one takes an ORM class, the other takes an
ermrest_model.Table). But a maintainer reading the class will
see two `is_association_table`-shaped methods with very
different semantics. **Recommend** the Lens-4 rename of
`_is_association_table` → `is_topological_association` to
defuse the naming collision.

### 2.6 `BagDatabase` already provides `_build_asset_map` / `resolve_asset_local_path`

`deriva-py/deriva/bag/database.py:353-457`. `DatabaseModel`
doesn't override these — good. `DerivaMLBagView` doesn't expose
them either, which is correct (they're SQLite-implementation
details, not deriva-ml-domain concerns).

**No action.**

---

## Lens 4 — Inconsistencies / duplication

### 4.1 `_is_association_table` is private-by-name, public-by-use

`catalog.py:712-766`. The leading underscore implies
package-private, but the method has **11 external callers
across two subsystems**:

- `core/mixins/dataset.py:449` (via `_prepare_wide_table` →
  `_is_association_table`)
- `local_db/denormalizer.py:776, 828, 832, 1182, …` — explicit
  callers using `model._is_association_table(...)` directly.
- Docstring at `denormalizer.py:776` *references the
  underscore-prefixed name* in a public-API comment, cementing
  the contract.

Plus the method's own docstring (`catalog.py:712-753`) reads as
a public-API contract — describes semantics, gives examples,
documents the relationship to `Table.is_association`.

Phase 1 §2.7 recommended renaming to
`is_topological_association`. After reading the planner's
usage, that recommendation still applies and is **low-risk**:

- Rename `_is_association_table` → `is_topological_association`
  (8 internal callers in `catalog.py`, 7 external callers in
  `local_db/denormalizer.py`).
- Drop the `_` prefix from the docstring references.
- Update `local_db/README.md:31` and any test files.

**Risk: low (mechanical). LoC: ±0.** **Severity: medium**
(removes a long-standing privacy lie).

### 4.2 `DerivaModel.__getattr__` is a silent delegation hatch

`catalog.py:410-412`:

```python
def __getattr__(self, name: str) -> Any:
    # Called only if `name` is not found in Manager.  Delegate attributes to model class.
    return getattr(self.model, name)
```

Used implicitly by every caller that does `self.model.schemas`,
`self.model.annotations`, `self.model.chaise_config`, etc. —
where `self.model.model.schemas` would be the canonical path
through the wrapped ermrest `Model`. `grep -rn "model\.schemas"
src/` returns 50+ hits across the codebase, all routing through
this delegation.

The delegation is convenient but it:

- Hides the boundary between deriva-ml-domain methods (defined
  on `DerivaModel`) and raw deriva-py methods (inherited via
  delegation from `Model`).
- Breaks IDE / pyright autocompletion on `model.<deriva-py
  method>`.
- Means typos (`model.foo`, `model.cataolg`) raise
  `AttributeError` with the wrong class name attached.
- Surprises maintainers — every method on the underlying
  ermrest `Model` is implicitly part of `DerivaModel`'s
  surface, but `DerivaModel` doesn't document this.

Phase 1 left this in place. **In Phase 2 it is the highest-value
cleanup short of the denormalization split.** The properly
"correct" form is to:

1. Replace the bare `__getattr__` with explicit `@property`
   accessors for the deriva-py attributes actually used by
   consumers (`schemas`, `annotations`, `chaise_config`,
   `bulk_upload`, `display`, `column_defaults`, `apply`,
   `digest_fkeys` — 8 attributes per a grep across `src/`).
2. Keep `DerivaModel.apply()` as it already exists (line 665)
   but make the underlying `model.apply` accessible without the
   string guard.

**Risk: medium** (lots of `model.X` callsites). **LoC: −0
catalog.py, +20 explicit accessors.** **Severity: high**
(boundary-clarity hazard).

### 4.3 `DerivaMLBagView.list_dataset_element_types` is a thin pass-through

`deriva_ml_bag_view.py:242-248`:

```python
def list_dataset_element_types(self) -> list[Table]:
    """List the types of elements that can be in datasets."""
    return self._database_model.list_dataset_element_types()
```

`DatabaseModel` already inherits `list_dataset_element_types`
from `DerivaModel` (via `database.py:45`'s multiple
inheritance). Consumers of `DerivaMLBagView` could call
`view._database_model.list_dataset_element_types()` directly,
but the protocol surface in
`interfaces.py:DerivaMLCatalogReader` expects the method on the
catalog itself, so the pass-through is correct — but it's
inconsistent with how other methods are exposed (e.g.,
`find_features` at line 250 *also* pass-throughs, but
`get_table_as_dict` at line 231 is the only place where
`self.get_table_as_dict` is the canonical name).

**Not an issue per se** — just worth a comment that the class is
a protocol-shaped pass-through layer.

### 4.4 Two `find_features` exposures on `DerivaMLBagView`

`deriva_ml_bag_view.py:250-259` defines `find_features(table:
str | Table)` requiring a `table` argument. The parent
`DatabaseModel.find_features` (inherited from `DerivaModel`,
`catalog.py:563-609`) allows `table=None` to return all features
across all schemas.

The view's version forces a required argument, narrowing the
protocol. `interfaces.py:DerivaMLCatalogReader.find_features`
(line 786) lists `table: str | Table` as required. **Mostly
fine**, but means `DatasetBag.find_features(None)` (allowed by
`DerivaModel`) is rejected by the view. Either widen the view
to `table: str | Table | None = None` or document the narrowing
in the protocol.

**Severity: low.** **LoC: ±0.**

### 4.5 `_get_dataset_execution` is private-by-name, called across modules

`database.py:219-247` defines `_get_dataset_execution` as an
underscore-prefixed method. It's called from
`deriva_ml_bag_view.py:154` and `:179` (in
`lookup_dataset` and `find_datasets`):

```python
execution_rid=(self._database_model._get_dataset_execution(rid) or {}).get("Execution"),
```

Same `_` pattern as Lens 4.1. Either:

(a) Make it public (`get_dataset_execution`) since two outside
    callers use it, or
(b) Pull the logic into `DerivaMLBagView` (since that's the only
    caller) and delete it from `DatabaseModel`.

**Recommendation:** (b). `_get_dataset_execution` is a
deriva-ml-domain query, not a generic bag-database
operation, and the only caller is the view. Moving it
clarifies the layering.

**Risk: low. LoC: ±0 (move).** **Severity: low.**

---

## Lens 5 — Simplification opportunities

### 5.1 Delete `annotations.py` if no external consumer

See Lens 1.1. **Risk: depends on classification. LoC: −1 264
plus tests.** **Severity: high.**

### 5.2 Phase-3 split of the denormalization planner

Phase 1 §6.1 flagged this. After reading the actual planner
(`catalog.py:819-1837`), the split is the right call:

- **Planner has its own dataclass** (`JoinNode`, lines 49-92)
  and its own module-level helper (`denormalize_column_name`,
  lines 95-112) that's already imported lazily by every
  consumer.
- **Planner has six private methods** (`_build_join_tree`,
  `_determine_row_per`, `_enumerate_paths`,
  `_find_path_ambiguities`, `_find_sinks`, `_prepare_wide_table`)
  plus three reachability primitives (`_downstream_fk_sources`,
  `_outbound_reachable`, `_is_association_table`).
- **Planner is consumed only by `local_db/`** (denormalizer,
  denormalize) and a single line in `core/mixins/dataset.py`.
- **Introspection methods are consumed by every mixin** — a
  fundamentally different fan-out.

**Phase-3 sketch** (NOT a Phase-2 recommendation):

```
src/deriva_ml/model/
    catalog.py                  # ~1 200 LoC — introspection + schema mutation
    denormalize_planner.py      # ~880 LoC — planner + JoinNode + denormalize_column_name
        class DenormalizePlanner:
            def __init__(self, model: DerivaModel): ...
            def prepare_wide_table(...): ...
            def build_join_tree(...): ...
            def find_path_ambiguities(...): ...
            ...
```

The planner takes a `DerivaModel` for introspection but lives
adjacent to `local_db/denormalizer.py` (which is its only
consumer). The Lens-4 rename of `_is_association_table` happens
during the move.

**Risk: medium-high** (popular import path; lots of method
references to chase). **LoC: ±0 net, ~880 lines move.**
**Severity: medium.** **Schedule as Phase 3.**

### 5.3 `DerivaModel.find_assets` parameter cleanup

Lens 1.4. **LoC: −1.** **Severity: low.**

### 5.4 Replace `__getattr__` with explicit deriva-py accessors

Lens 4.2. **Risk: medium. LoC: −2 +20.** **Severity: high**
in maintainability terms.

### 5.5 Move `_get_dataset_execution` to the view

Lens 4.5. **LoC: ±0 (move).** **Severity: low.**

### 5.6 `find_datasets` materializes `Dataset_Dataset_Type` once vs. twice

`deriva_ml_bag_view.py:158-183`:

```python
def find_datasets(self, deleted: bool = False) -> Iterable[DatasetBag]:
    atable = f"Dataset_{MLVocab.dataset_type.value}"
    ds_types = list(self._database_model.get_table_contents(atable))  # full materialization

    datasets = []
    for dataset in self._database_model.get_table_contents("Dataset"):
        my_types = [t[MLVocab.dataset_type.value] for t in ds_types if t["Dataset"] == dataset["RID"]]
        ...
```

The inner list comprehension is O(N×M). For each dataset, it
re-scans the full `ds_types` list. For 10 datasets × 100
dataset-type rows it's fine, but the obvious dict-grouping fix
is one line:

```python
types_by_rid: dict[str, list[str]] = defaultdict(list)
for row in self._database_model.get_table_contents(atable):
    types_by_rid[row["Dataset"]].append(row[MLVocab.dataset_type.value])
```

The same pattern repeats in `lookup_dataset`
(`deriva_ml_bag_view.py:144-148`) — that one is single-RID and
fine.

**Risk: trivial. LoC: ±0.** **Severity: low** (not load-bearing
today; symptom of "wrote quickly to pass the protocol test").

### 5.7 `lookup_term` uses linear scan instead of indexed lookup

`deriva_ml_bag_view.py:185-218`:

```python
for term in self.get_table_as_dict(table_obj.name):
    if term_name == term.get("Name") or ...
        return VocabularyTerm.model_validate(term)
```

`BagDatabase.find_table` returns a SQLAlchemy `Table` — the
view could `select(table).where(table.c.Name == term_name)` for
a real indexed query. The linear scan is N for every lookup
call, and `lookup_term` is called per-vocab-row during many
feature-record validations.

**Risk: low (mechanical, but does need a SQLAlchemy session).
LoC: ~+15.** **Severity: low** (works today, slow path).

---

## Lens 6 — Maintainability

### 6.1 `DerivaModel` docstrings have placeholder Args/Returns

Phase 1 §6.3 flagged this. Spot-check on the current code:

- `catalog.py:486-507` — `is_association` docstring has empty
  `Args:` placeholders ("`unqualified: param ...`",
  "`pure: return: (Default value = True)`") never filled in.
- `catalog.py:509-514` — `find_association` docstring is two
  lines with no `Args:` or `Returns:` block.
- `catalog.py:611-624` — `lookup_feature` docstring has
  placeholder shape (`"Args: table: param feature_name:"`).
- `catalog.py:631-638` — `asset_metadata` is one line.
- `catalog.py:672-684` — `is_dataset_rid` is one line.
- `catalog.py:686-710` — `list_dataset_element_types` is a 9-line
  docstring (good) but the return type description says
  `list[str]` while the actual return is `list[Table]`.

Compare with the **well-formed** docstrings on the planner
methods (`_is_association_table` at 712-753,
`_fk_neighbors` at 768-799, `_downstream_fk_sources` at
1063-1102, `_find_sinks` at 1211-1249, `_find_path_ambiguities`
at 1407-1473): these have proper `Args:`, `Returns:`,
`Example:`, and explain semantics. The planner code was clearly
written/touched more recently than the introspection surface.

**Action:** sweep the introspection-surface docstrings to match
the planner's quality. The methods touched daily by every mixin
should have the best documentation, not the worst.

**Risk: zero (doc-only). LoC: +60-80.** **Severity: medium**
(the introspection API is what every mixin builds against).

### 6.2 `DerivaMLBagView.list_dataset_element_types` typo in docstring

`deriva_ml_bag_view.py:242-248`:

```
"""List the types of elements that can be in datasets.

Returns:
    List of Table objects representing element types.
"""
```

This is fine. But `DerivaModel.list_dataset_element_types`
(`catalog.py:686-710`) returns the same data while saying
`list[str]` in its `Returns:` block. The mismatch is the bug.

**Action:** Phase 1 §4.6 noted Phase 1 already collapsed the
duplicate `list_dataset_element_types` — but the docstring is
still stale. Fix the return type in `catalog.py:697`.

**Risk: zero. LoC: +1.** **Severity: low.**

### 6.3 Naming consistency on `table_name` parameters

Public methods vary in their first-arg name:

- `name_to_table(self, table: TableInput)` (line 414) — the
  parameter is `table` but it accepts a string name.
- `is_vocabulary(self, table_name: TableInput)` (line 442) —
  parameter is `table_name`.
- `vocab_columns(self, table_name: TableInput)` (line 467) —
  `table_name`.
- `is_association(self, table_name: str | Table)` (line 486) —
  `table_name`.
- `is_asset(self, table_name: TableInput)` (line 533) —
  `table_name`.
- `find_association(self, table1: Table | str, table2: ...)`
  (line 509) — `table1` / `table2`.
- `find_features(self, table: TableInput | None = None)` (line
  564) — `table`.
- `lookup_feature(self, table: TableInput, feature_name: str)`
  (line 611) — `table`.
- `asset_metadata(self, table: str | Table)` (line 631) —
  `table`.
- `_is_association_table(self, name_or_table: ...)` (line 712)
  — `name_or_table`.
- `_fk_neighbors(self, table: str | Table)` (line 768) —
  `table`.

Five different conventions for the same concept. Pick one
(`table` is shortest and matches `is_dataset_rid`'s `rid`
naming), rename the rest.

**Risk: low (parameter renames are caller-visible only for
kwargs callers — verify with grep first). LoC: ±0 net.**
**Severity: low.**

### 6.4 `from_cached` is a 50-line method with three duplicate `from deriva_ml...` imports

`catalog.py:200-252`. The method is straightforward; the
docstring is comprehensive (43 lines). One nit: the inline
`from deriva.core.ermrest_model import Model` at line 241 is
shadowed by the module-level `Model = _ermrest_model.Model` at
line 25 — the inline import is redundant. Same `Model` either
way.

**Risk: trivial. LoC: −1.** **Severity: low (cosmetic).**

### 6.5 `denormalize_column_name` should move with the planner

`catalog.py:95-112` defines the column-naming helper at module
scope. Its three consumers all live in `local_db/`:

- `local_db/denormalize.py:47` — top-level import.
- `local_db/denormalizer.py:523, 608, 1245` — lazy imports
  inside methods.
- `core/mixins/dataset.py:443` — lazy import inside method.

When the planner is split (Lens 5.2), this helper moves with
it. **No action in Phase 2.**

### 6.6 `model/__init__.py:55-93` exports an enormous public surface

The `__all__` lists 33 names — 23 of them from `annotations.py`
(see Lens 1.1) plus `DerivaModel`, `DatabaseModel`,
`DerivaMLBagView`, `TableHandle`, `ColumnHandle`, and 5
enum/context constants. If the annotation-builders cleanup
lands (Lens 1.1), this collapses to ~7 names — a reasonable
package surface.

**No action independently; falls out of Lens 1.1.**

---

## DerivaModel responsibilities analysis (Phase 3 split candidates)

### Responsibility A — Introspection (lines 254-816)

**21 methods, ~565 LoC.** Every mixin and most subsystems
consume this. Fan-out (from grep across `src/`):

- `name_to_table`: 50+ callsites.
- `is_asset` / `is_vocabulary`: ~15 callsites each.
- `find_association`: 14 callsites (`asset/asset.py`,
  `core/mixins/*`, `execution/*`).
- `find_features`: 5 callsites (`feature.py`, `core/mixins/feature.py`,
  `dataset/dataset_bag.py`, `schema/annotations.py`).
- `asset_metadata` / `asset_metadata_columns`: 5 callsites
  (`asset_record.py`, `asset/manifest.py`, `dataset/upload.py`).
- `is_dataset_rid` / `list_dataset_element_types`: 5 callsites
  (mostly `core/mixins/dataset.py`).
- `find_assets` / `find_vocabularies`: 3 callsites.
- `get_schema_description`: 0 callsites in `src/`, 0 in
  `tests/` — **looks unused but is the kind of thing an MCP /
  skill consumer would reach for**; needs maintainer
  classification (cf. Lens 1.1).

**Verdict:** stays in `catalog.py`. This is the core surface.

### Responsibility B — Denormalization planner (lines 819-1610)

**6 methods + 1 dataclass + 1 module helper, ~795 LoC.** Phase 3
split candidate per Lens 5.2.

Consumers: 14 callsites in `local_db/denormalizer.py`, 3 in
`local_db/denormalize.py`, 1 in `core/mixins/dataset.py`. All
in the `local_db/` neighborhood.

Has its own well-formed docstrings (the planner code is the
best-documented in the file), its own dataclass (`JoinNode`),
and its own module-level helper (`denormalize_column_name`).
**Stands as a unit.**

### Responsibility C — FK traversal primitives (lines 1612-2009)

**3 methods: `_prepare_wide_table` (line 1612), `_table_relationship`
(line 1839), `_schema_to_paths` (line 1891).** ~390 LoC.

Despite the naming, `_prepare_wide_table` is the **planner's
top-level driver** (it composes planner methods from B above) —
it should move with B. `_table_relationship` and
`_schema_to_paths` are the **lower-level primitives** that the
planner stands on; they're also used directly by the planner.

`_schema_to_paths` is also used by:
- `dataset/dataset_bag.py:383` (one callsite).
- `tests/dataset/test_schema_paths.py` and
  `tests/dataset/test_realworld_patterns.py` — heavy test
  coverage.

**Verdict:** when split, both primitives go to the planner
module. The one external callsite in `dataset_bag.py:383` can
reach in via `bag.model._schema_to_paths(...)` exactly as today
— the planner module owns the schema-traversal API.

### Responsibility D — Schema mutation (lines 2011-2083)

**2 methods: `create_table`, `_define_association`.** ~73 LoC.

Consumers: 5 callsites total (`core/mixins/asset.py:142, 157`,
`core/mixins/feature.py:169-170`, `core/mixins/dataset.py:253`,
`core/base.py:1343`).

**Verdict:** stays in `catalog.py`. Small enough not to warrant
its own module, but the responsibility is distinct from
introspection — worth a section divider comment.

### Summary table

| Responsibility | LoC | Consumer count | Phase 3 verdict |
|---|---:|---:|---|
| A — Introspection | 565 | 100+ across all mixins | Stays in `catalog.py` |
| B — Denormalization planner | 795 | 18 (all in `local_db/`) | **Move to `denormalize_planner.py`** |
| C — FK primitives | 390 | ~10 (planner + tests + 1 src) | **Move with B** |
| D — Schema mutation | 73 | 5 | Stays in `catalog.py` |

Post-split `catalog.py` would be ~640 LoC (responsibility A + D)
plus class/init scaffolding, and `denormalize_planner.py` would
be ~900 LoC (responsibility B + C + helpers).

---

## Ranked actions (1–N)

Ranked by `(impact × confidence) / cost`. Items 1–3 are
mechanical quick wins; 4–7 are small focused refactors; 8–10
are scoped follow-ups.

| # | Action | Risk | LoC | Files | Rationale |
|---|---|---|---|---|---|
| 1 | **Lens 1.2** — Delete the 8 unused `TypeAlias` declarations in `catalog.py:117-126` | trivial | −9 | `model/catalog.py` | Pure dead code; no `grep` hits. |
| 2 | **Lens 1.3 + 1.4 + 6.2 + 6.4** — Delete `self.configuration = None`, the unused `with_metadata=False` parameter on `find_assets`, fix `list_dataset_element_types`'s return-type docstring (`list[str]` → `list[Table]`), drop the inline `from deriva.core.ermrest_model import Model` in `from_cached` | trivial | −5 +1 | `model/catalog.py` | Four independent cleanups in one PR. |
| 3 | **Lens 6.1** — Sweep introspection-method docstrings (`is_association`, `find_association`, `lookup_feature`, `asset_metadata`, `is_dataset_rid`) up to the planner's docstring quality | low | +60-80 | `model/catalog.py` | Most-consumed surface should have the best docs. |
| 4 | **Lens 4.1** — Rename `_is_association_table` → `is_topological_association` (8 internal + 7 external callers in `local_db/denormalizer.py`) | low | ±0 | `model/catalog.py`, `local_db/denormalizer.py`, `local_db/README.md` | Removes a "private by name, public by use" lie. |
| 5 | **Lens 4.5** — Move `DatabaseModel._get_dataset_execution` to `DerivaMLBagView._get_dataset_execution` (its only caller) | low | ±0 (move) | `model/database.py`, `model/deriva_ml_bag_view.py` | Clarifies the deriva-ml-domain layering. |
| 6 | **Lens 5.6** — Index `find_datasets`'s dataset-type lookup by RID (one-line dict-groupby) | trivial | ±0 | `model/deriva_ml_bag_view.py` | O(N×M) → O(N+M); not load-bearing today but obvious. |
| 7 | **Lens 6.3** — Pick one parameter name for the first arg of table-introspection methods (`table` is shortest), rename the others to match | low | ±0 | `model/catalog.py` | Five conventions for the same concept. Caller-visible only via kwargs — grep first. |
| 8 | **Lens 1.1** — Classify `annotations.py` (1 264 LoC) as (a) public surface for external consumers, (b) dead code, or (c) deliberate scaffolding. If (b): delete the module + `tests/model/test_annotations.py` + the `model/__init__.py` re-exports | depends | −1 744 if (b) | `model/annotations.py`, `model/__init__.py`, `tests/model/test_annotations.py` | Largest single LoC reduction in the audit if (b); blocked on maintainer call. |
| 9 | **Lens 4.2** — Replace `DerivaModel.__getattr__` (catalog.py:410-412) with explicit `@property` accessors for the ~8 deriva-py attributes actually used (`schemas`, `annotations`, `chaise_config`, `bulk_upload`, `display`, `column_defaults`, `digest_fkeys`, `apply`) | medium | +20 | `model/catalog.py` | Removes a silent-delegation hatch; fixes IDE completion across all mixins. |
| 10 | **Lens 5.2** — **Phase 3.** Extract the denormalization planner (Responsibilities B + C from the analysis above) to `model/denormalize_planner.py`. ~880 LoC moves. | medium-high | ±0 net | `model/catalog.py`, new `model/denormalize_planner.py`, every caller in `local_db/*` and `core/mixins/dataset.py:449` | Largest structural improvement; sized for a focused PR. Land action 4 first (rename happens during the move). |
| 11 | **Lens 5.7** — Convert `DerivaMLBagView.lookup_term`'s linear scan to a SQLAlchemy `select().where()` | low | +15 | `model/deriva_ml_bag_view.py` | Mechanical perf fix on a frequently-called code path. |
| 12 | **Lens 1.5** — Replace `apply`'s `if self.catalog == "file-system"` string guard with a typed catalog-capability check | low-medium | ±0 | `model/catalog.py` | Robustness fix; current string guard is fragile against `CatalogStub` types. |

Actions 1–3 are good to land in one cleanup PR. Action 4 is a
prerequisite for action 10. Action 8 is blocked on a maintainer
decision. Action 9 is the highest-value cleanup in this audit
short of action 10.

---

## Follow-up scope (Phase 3 candidates)

1. **Action 10 (denormalize-planner extraction).** The right
   shape is a `DenormalizePlanner` class that takes a
   `DerivaModel` and exposes the planner methods without the
   leading underscores. ~880 LoC moves. Coordinate with
   `local_db/denormalizer.py` since the planner's external
   contract is currently spelled in terms of underscore-prefixed
   methods on `model.`. ADR-0006-style planning doc helpful but
   not required.

2. **Action 8 (annotations.py classification).** Needs a
   maintainer decision. If kept, document the external-consumer
   contract in `CLAUDE.md` and add an integration test that
   actually consumes the builders. If dropped, delete cleanly
   in one PR — there are no internal callers.

3. **Action 9 (`__getattr__` removal).** Bigger PR than the
   line count suggests because every mixin uses
   `self.model.schemas` (etc.). Probably wants a
   "step 1: introduce explicit properties next to `__getattr__`
   so callers can migrate, step 2: assert no remaining
   `__getattr__` hits via a temporary log, step 3: drop the
   `__getattr__`" landing plan.

4. **Naming consistency across catalog/database/view (Lens
   6.3).** Five conventions for "table name parameter" is
   tractable as a one-time sweep; do it during action 3's
   docstring pass.

5. **`apply()` capability check (Lens 1.5).** Co-design with
   `core/schema_cache.py` and `CatalogStub` to land a typed
   "is this catalog applyable" predicate that `DerivaModel`
   can use without the string guard. Cross-cutting; deferred
   from the immediate cleanup.

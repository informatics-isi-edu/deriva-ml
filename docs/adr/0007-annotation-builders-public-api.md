# ADR-0007: `deriva_ml.model.annotations` is a public API for `deriva-skills`

Date: 2026-05-13
Status: Accepted

## Context

`src/deriva_ml/model/annotations.py` is 1 264 lines of Pydantic builder
classes for Chaise display annotations: `Display`, `VisibleColumns`,
`VisibleForeignKeys`, `TableDisplay`, `ColumnDisplay`, `PseudoColumn`,
`FacetList`, plus helper records (`InboundFK`, `OutboundFK`,
`fk_constraint`), enums (`TemplateEngine`, `Aggregate`, `ArrayUxMode`,
`FacetUxMode`), and context constants (`CONTEXT_DEFAULT`,
`CONTEXT_COMPACT`, etc.).

The Phase 2 cleanup audit (`docs/design/deriva-ml-audit-2026-05-phase2-model.md`,
§1.1) initially flagged the file for deletion: a workspace-wide grep
for the builder class names returned **zero hits inside
`src/deriva_ml/`**. From the deriva-ml package's perspective, every
exported class in `model/__init__.py`'s `# Annotation builders` block
appears unused.

A cross-workspace check resolved the discrepancy: the builders are an
**externally-consumed public API**, used by the
[`deriva-skills`](https://github.com/informatics-isi-edu/deriva-skills)
Claude Code plugin. The
`deriva-skills/skills/use-annotation-builders/SKILL.md` skill
documents the canonical usage pattern:

```python
from deriva_ml.model.annotations import Display, VisibleColumns
display = Display(name="Images", markdown_name="**Images**")
table.annotations[Display.tag] = display.to_dict()
ml.apply_annotations()
```

The pattern surfaces three contract points the audit's automated
deletion scan would have missed:

1. **Module path** (`deriva_ml.model.annotations` and the re-export from
   `deriva_ml.model`) — the skill imports by path.
2. **Class-level `.tag` attribute** — used as the dict key on
   `table.annotations`.
3. **Instance-level `.to_dict()` method** — produces the JSON-shaped
   payload Chaise expects.

The skill predates any in-`deriva-ml` test that exercises the apply
flow against a real catalog: existing `tests/model/test_annotations.py`
covers only **builder construction** (Pydantic validation, `.to_dict()`
output shape) — not the
`table.annotations[Builder.tag] = builder.to_dict()` boundary the skill
relies on. So a refactor in `deriva-ml` could drop `.tag`, rename
`.to_dict()`, or move the module path, and CI would stay green while
the skill's user-facing scripts would silently fail at runtime.

## Decision

Pin the annotation-builders public API contract in this repo so future
audits don't re-find it for deletion, and so test coverage catches
contract drift before users do.

### The contract

For each annotation builder re-exported from `deriva_ml.model`:

- **Import path stability.** `from deriva_ml.model.annotations import
  <BuilderName>` and `from deriva_ml.model import <BuilderName>` both
  resolve, identically, to the same class. Removing either path is a
  **breaking change** requiring a major version bump and migration
  notice in the deriva-skills marketplace.
- **`.tag` attribute.** Class-level `str` attribute that names the
  ERMrest annotation tag the builder produces (e.g.
  `Display.tag == "tag:misd.isi.edu,2015:display"`). Read-only from
  the caller's perspective; the value may change only when deriva-py
  itself renames the underlying tag.
- **`.to_dict()` method.** Instance-level, no required arguments,
  returns a JSON-serializable `dict`. The result is assigned directly
  to `table.annotations[Builder.tag]` and ultimately serialized to
  ERMrest's `/schema/.../annotation` endpoint.
- **Apply path.** `DerivaML.apply_annotations()` is the user-facing
  commit step. It takes no required positional arguments and pushes
  staged annotation changes to the catalog.

### The exported surface

Per `src/deriva_ml/model/__init__.py:55-93`:

- Builders: `Display`, `VisibleColumns`, `VisibleForeignKeys`,
  `TableDisplay`, `TableDisplayOptions`, `ColumnDisplay`,
  `ColumnDisplayOptions`, `PreFormat`, `PseudoColumn`,
  `PseudoColumnDisplay`, `Facet`, `FacetList`, `FacetRange`,
  `SortKey`, `NameStyle`.
- FK helpers: `InboundFK`, `OutboundFK`, `fk_constraint`.
- Enums: `TemplateEngine`, `Aggregate`, `ArrayUxMode`, `FacetUxMode`.
- Context constants: `CONTEXT_DEFAULT`, `CONTEXT_COMPACT`,
  `CONTEXT_DETAILED`, `CONTEXT_ENTRY`, `CONTEXT_FILTER`.

The Claude Code skill documents `Display`, `VisibleColumns`,
`VisibleForeignKeys`, `TableDisplay`, `ColumnDisplay`,
`PseudoColumn`, `FacetList` in `SKILL.md` directly. The rest are
supporting types (records, enums, context strings) reached via builder
constructors.

### Test coverage of the contract

`tests/model/test_annotations.py::TestExternalConsumerContract` covers
the four contract points:

- Every exported builder has `.tag` and `.to_dict()`.
- Each builder is constructable with a sensible default.
- The `table.annotations[Builder.tag] = builder.to_dict()` line works
  against a plain-dict stand-in for `Table.annotations`, and the
  resulting payload is JSON-stable.
- `DerivaML.apply_annotations()` exists and takes no required
  positional arguments.

These are dict-based tests — they need no live catalog. They run as
part of the standard unit test suite, so contract drift surfaces in
CI on the deriva-ml side, before a deriva-skills user notices.

### Audit hygiene

The audit's automated "no `grep` hits in `src/`" check is not safe to
apply to this module. Future audits must consult this ADR (or the
`CLAUDE.md` entry it links) before flagging `model/annotations.py` for
deletion.

## Consequences

**Positive:**

- The contract is captured explicitly. New maintainers can see the
  external dependency without having to grep across workspaces.
- The integration test catches contract drift in CI.
- The deriva-skills plugin can rely on `tag` / `to_dict()` /
  `apply_annotations` continuing to exist without coordinating every
  deriva-ml release.

**Negative:**

- We commit to keeping `model/annotations.py` in its current shape
  even though no internal code uses it. Refactoring constraints
  apply.
- A breaking change to the contract (e.g. renaming `.to_dict()` to
  `.model_dump()`) requires version coordination with deriva-skills.
- The Pydantic-class-as-builder pattern is the **shape** of the
  contract — switching to a different builder mechanism (e.g.
  TypedDict, dataclass) would be a breaking change even if the
  resulting JSON were byte-identical.

**Neutral:**

- The supporting types (enums, context constants, helper records) are
  in the contract too, since they appear in builder constructors —
  changing their names is visible to skill users.

## Alternatives considered

### Delete `model/annotations.py`

Mechanically possible — there are no in-`src/` callers. Rejected: the
deriva-skills plugin has at least one documented user-facing skill
that depends on the module. Deletion would break the skill at runtime
with no compile-time error.

### Move `model/annotations.py` into deriva-skills

Inverts the dependency: instead of deriva-skills consuming an API in
deriva-ml, deriva-ml would lose access to its own annotation builders
(no internal callers today, but the audit also flagged opportunities
to add some during the schema-mutation surface in Phase 3). Rejected:
the builders are a stable, general-purpose Deriva concept; they
belong in the Python package that ships with deriva-py, not in a
Claude Code plugin's resource tree.

### Move to deriva-py upstream

The builders are general-purpose Deriva (not deriva-ml-specific). The
right long-term home is `deriva-py`'s annotation namespace. Out of
scope for this ADR — would require coordination with the deriva-py
maintainers and a migration period during which both import paths
work. Tracked separately.

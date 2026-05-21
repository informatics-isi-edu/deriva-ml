# Archived implementation plans

These plans describe past implementation work on the denormalization
subsystem. They are kept for historical context — they record how
the code got into its current shape, decisions taken at the time,
and the test ladders used during execution — but they should NOT be
read as the current architecture or contract.

**For current design references, see:**

- `docs/design/denormalization.md` — architecture, state model,
  fetcher/INSERT contract, fragility map, test matrix. This is the
  one document new engineering should read.
- `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md` —
  the eight semantic Rules implemented by the `Denormalizer` /
  planner. Still authoritative.
- `docs/superpowers/specs/2026-04-15-unified-local-db-design.md` and
  `2026-04-15-unified-local-db-phase2-design.md` — workspace layout
  (ATTACH'd schema files, registries). Still authoritative for
  storage; superseded by `docs/design/denormalization.md` wherever
  they overlap on fetcher/INSERT-side behavior.

**Files archived here:**

| Plan | Original date | Why archived |
|---|---|---|
| `2026-03-20-fix-denormalize-multihop-fk.md` | 2026-03-20 | Multi-hop FK behavior subsumed by the join-tree refactor and then the unified-local-db Phase 2 work. Current FK traversal lives in `model/denormalize_planner.py`. |
| `2026-03-20-fix-duplicate-association-tables.md` | 2026-03-20 | Fix landed; current duplicate-association handling lives in `_prepare_wide_table`'s path dedup. |
| `2026-03-28-denormalize-join-tree-refactor.md` | 2026-03-28 | Superseded by `2026-04-16-unified-local-db-phase2.md`, which collapsed both engines into one. |
| `2026-04-15-unified-local-db-phase1.md` | 2026-04-15 | Phase 1 complete. The architecture it implemented is now described in `docs/design/denormalization.md` and the Phase 1 design spec. |
| `2026-04-16-unified-local-db-phase2.md` | 2026-04-16 | Phase 2 complete. The architecture it implemented is now described in `docs/design/denormalization.md` and the Phase 2 design spec. |
| `2026-04-17-denormalization-semantics.md` | 2026-04-17 | Implementation plan for the semantic-Rules design spec. The Rules themselves remain authoritative in `…/specs/2026-04-17-denormalization-semantics-design.md`. |

**Policy for future archiving** (from
`docs/design/denormalization.md` §9 commitment #5): when an
implementation plan is superseded by a new design or a new plan,
move it here, do not delete it. Update this README's table.

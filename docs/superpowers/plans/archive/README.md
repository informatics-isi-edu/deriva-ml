# Archived implementation plans

These plans describe past implementation work on the denormalization
subsystem and other 2026-04/-05 deriva-ml development streams. They
are kept for historical context — they record how the code got into
its current shape, decisions taken at the time, and the test ladders
used during execution — but they should NOT be read as the current
architecture or contract.

**For current design references, see:**

- `docs/user-guide/denormalization.md` — the single source of record
  for denormalize: architecture, state model, fetcher/INSERT
  contract, the nine semantic Rules, fragility map, and test matrix.
  This is the one document new engineering should read.
- `docs/superpowers/specs/2026-04-15-unified-local-db-design.md` and
  `2026-04-15-unified-local-db-phase2-design.md` — workspace layout
  (ATTACH'd schema files, registries). Still authoritative for
  storage; superseded by the user guide wherever they overlap on
  fetcher/INSERT-side behaviour or Denormalizer semantics.

**Policy for future archiving** (from
`docs/user-guide/denormalization.md` §10 process commitments): when
an implementation plan is superseded by a new design or a new plan,
move it here, do not delete it. The git history alone is
load-bearing — readers shouldn't have to dig through it to find why
a decision was made.

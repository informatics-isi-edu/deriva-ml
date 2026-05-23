# 2026-05 audit working drafts — archived

This directory holds the **working-draft audit notes** that fed
the 2026-05-22 pre-release audit. They were superseded by the
polished engineer-audit and writer-audit reports under
[`docs/audits/`](../../audits/).

## What's in here

The Phase 1 cross-cutting audit plus seven Phase 2/Phase 3
subsystem deep-dives:

| Working draft | Polished replacement |
|---|---|
| `deriva-ml-audit-2026-05.md` (Phase 1, cross-cutting) | Folded into all `docs/audits/2026-05-22-engineer-audit-*.md` files |
| `deriva-ml-audit-2026-05-phase2-dataset.md` | `docs/audits/2026-05-22-engineer-audit-dataset.md` |
| `deriva-ml-audit-2026-05-phase2-model.md` | `docs/audits/2026-05-22-engineer-audit-model.md` |
| `deriva-ml-audit-2026-05-phase3-execution.md` | `docs/audits/2026-05-22-engineer-audit-execution.md` |
| `deriva-ml-audit-2026-05-phase3-schema.md` | `docs/audits/2026-05-22-engineer-audit-schema.md` |
| `deriva-ml-audit-2026-05-phase3-feature.md` | `docs/audits/2026-05-22-engineer-audit-feature.md` |
| `deriva-ml-audit-2026-05-phase3-core.md` | `docs/audits/2026-05-22-engineer-audit-core.md` |
| `deriva-ml-audit-2026-05-phase3-catalog.md` | `docs/audits/2026-05-22-engineer-audit-catalog.md` |

## Why archived

The polished versions in `docs/audits/` are the canonical
references. The working drafts here are line-longer (1500–1800
LoC each) and contain scratch notes, intermediate priorities,
and provisional fixes that were either reconsidered before the
final reports or whose status moved as PRs landed.

For the running ledger of which items shipped vs. remain open,
see [`docs/audits/2026-05-22-audit-status.md`](../../audits/2026-05-22-audit-status.md).

## When to read these

- Tracing why a specific change made the cut into a final audit
  report (the working draft sometimes has the original
  rationale at higher fidelity).
- Reconstructing the chronology of a finding (working drafts
  predate the polished reports by ~2 weeks).
- Historical curiosity only — **do not use as a current
  reference** for any subsystem's state.

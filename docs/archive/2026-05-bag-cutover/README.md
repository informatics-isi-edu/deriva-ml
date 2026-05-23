# 2026-05 bag-cutover design docs — archived

This directory holds the **shipped** design docs for the bag-based
data-movement cutover. They are archived because their
implementations are now in `main` and the docs describe the
plan, not the current state.

## What's in here

| Design doc | Status | What landed |
|---|---|---|
| `bag-client-cutover-2026-05.md` | **Shipped** | `clone_via_bag` (catalog/clone_via_bag.py) — single-call bag pipeline replacing the legacy three-stage clone |
| `dataset-bag-cutover-2026-05.md` | **Shipped** | `dataset/bag_builder.py` — uses `deriva.bag.catalog_builder.CatalogBagBuilder` for dataset export and size estimation; replaced `CatalogGraph` |
| `bag-based-commit-execution.md` | **Shipped** | `execution/bag_commit.py` — per-execution upload via `BagBuilder` + `BagCatalogLoader` |

## Canonical replacements

- The shipped behaviour is documented in **CLAUDE.md** (the
  per-repo guide), in the affected subsystem's module
  docstrings, and in [`docs/adr/0006-bag-oriented-data-movement.md`](../../adr/0006-bag-oriented-data-movement.md)
  (the architectural decision record that anchors all three).
- For the multi-execution batch-upload path (`deriva-ml-upload`
  CLI) the corresponding scoping note
  [`bag-based-upload-engine.md`](../../design/bag-based-upload-engine.md)
  is **deliberately not archived** — that work has not yet
  shipped and the scoping note is still authoritative.

## Why archived

These docs describe the *plan* and the rationale at the time
the work was scoped. Since the implementations are now
load-bearing in `main`, the canonical reference is the code
itself (module docstrings) plus ADR-0006. Keeping the docs in
`docs/design/` would mislead future readers into thinking
either (a) the cutover is still pending or (b) the doc is the
authoritative description of the current code.

## When to read these

- Reconstructing why a specific approach was chosen (e.g., why
  hardlinks vs. copies in `BagBuilder.add_asset`).
- Tracing trade-offs that were considered but rejected during
  scoping.
- Cross-referencing against ADR-0006 to understand the broader
  bag-oriented data-movement direction.
- **Do not use as a current reference** for any of the
  affected subsystems' behaviour — the code and module
  docstrings are authoritative.

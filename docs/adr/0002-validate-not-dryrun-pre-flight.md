# ADR-0002: `validate_*` is the metadata-only pre-flight; `dry_run` is the full-path test

Date: 2026-05-03
Status: Accepted

## Context

DerivaML offers two ways to "check before you run" an
`ExecutionConfiguration`:

1. **`Execution(..., dry_run=True)`** (existing): goes through the
   full execution lifecycle — fetches every dataset bag, materializes
   assets, validates the workflow can be invoked — but skips the user
   payload (no model fit, no feature writes, no upload). Produces an
   actual `Execution` row.
2. **`validate_dataset_specs(...)` and
   `validate_execution_configuration(...)`** (new in this PR):
   metadata-only catalog queries that confirm the RIDs in a config
   resolve, the dataset versions exist, the workflow is a workflow,
   and there are no cross-spec inconsistencies. No bag download, no
   `Execution` row created.

These look superficially redundant. A reviewer (or future maintainer)
will reasonably ask "why don't we just tell users to run `dry_run=True`
to validate their config?"

## Decision

**Keep `validate_*` and `dry_run` as two distinct tools answering two
distinct questions.** They are complementary, not duplicative.

- `validate_*` answers: *"does this config refer to things that
  exist in the catalog the way I think they do?"* — a question about
  catalog metadata. Cost: O(N) metadata round-trips, ~hundreds of
  milliseconds for typical configs.
- `dry_run` answers: *"if I were to run this config end-to-end, would
  the runtime path succeed up to the point where my code starts?"* —
  a question about the bag-download + materialization path. Cost:
  the actual download cost, which can be minutes-to-hours and several
  GB-to-TB of bandwidth.

## Rationale

A real-world session iterating on `src/configs/datasets.py` produces
many small typos: a wrong character in a RID, a `0.4` where the user
meant `0.40`, a workflow RID swapped with a dataset RID. The user
needs feedback within seconds, not hours. `dry_run` cannot deliver
that — its cheapest possible execution still pays the bag download.

Conversely, when the user is ready to commit to a real run, they
care about a class of failures `validate_*` cannot see: bag export
quirks, slow asset downloads, transient network problems mid-fetch.
`dry_run` exists to surface those.

The two tools also have different blast radii. `validate_*` is a
read-only catalog query and is safe to call from a notebook in a
loop while iterating. `dry_run` creates an `Execution` row in the
catalog (and a working dir on disk), which is the right thing for a
real-pre-run check but the wrong thing for "let me check 30 versions
of this config in 5 minutes."

## Alternatives considered

### Add a `metadata_only` mode to `dry_run`

Push the lightweight check inside the existing `dry_run` machinery as
a flag (e.g. `dry_run="metadata"` vs `dry_run="full"`). Rejected
because:

- `dry_run` always creates an `Execution` row; making "metadata-only"
  a sub-mode either inherits that side effect (wrong for fast
  iteration) or makes the side-effect behaviour conditional on a
  string argument (surprising).
- The two methods need different return shapes: `validate_*` returns
  per-spec results with `available_versions` populated on failure,
  while `dry_run` returns an `Execution` object. Forcing them into
  one return shape is harmful to both call sites.
- Tier-2 skills (write-hydra-config, configure-experiment) want to
  recommend the cheap check by name; "use `dry_run` with a special
  flag" is harder to teach than "use `validate_*`."

### Replace `dry_run` with `validate_*`

Drop the heavy path entirely. Rejected — `dry_run` catches a real
class of failures (bag-export errors, asset materialization timeouts)
that metadata validation cannot see. Both are needed.

### Make `validate_*` a free function instead of a method

Put it in `deriva_ml.dataset.validation` as a module-level function
that takes a `DerivaML` instance. Rejected — it's a question about
the catalog, and the existing convention is method-on-DerivaML
(`lookup_dataset`, `lookup_lineage`, `find_datasets`). Free-function
form would be inconsistent with the rest of the API.

## Consequences

- **Documentation discipline**: every place that mentions `dry_run`
  for pre-flight checking should be cross-referenced to `validate_*`
  as the cheaper alternative for catalog-metadata questions. The
  `validate_execution_configuration` docstring says this explicitly,
  and the deriva-ml-skills Round 6b update to `write-hydra-config`
  carries the message into the tier-2 skill layer.
- **MCP wire**: deriva-ml-mcp Round 6b adds two thin wrappers for
  `validate_dataset_specs` and `validate_execution_configuration`,
  mirroring how `lookup_lineage` was wrapped. Other agents (Cursor,
  raw FastMCP clients) get the same fast-pre-flight surface.
- **Future change**: if someone proposes "let's just merge these into
  `dry_run`" again, they should re-read this ADR. The two answer
  different questions.

# Plan: native-Hydra CLI passthrough for `deriva-ml-run` and `deriva-ml-run-notebook`

**Date:** 2026-05-31
**Origin:** e2e finding `evaluator/01` (+ `modeler/01`) — `deriva-ml-run --info experiment` and `... --cfg job` rejected by the wrapper.
**Status:** Plan — awaiting approval before implementation.

## Goal (user-stated)

> "Hydra's documentation should just work against our runners to the maximum
> extent possible." Notebook is more complex (Hydra + papermill); there,
> prioritize Hydra for consistency.

## What we verified (corrects the finding's premise)

- Hydra **does** have `--info` (`hydra/_internal/utils.py:586-594`): `-i`,
  `nargs="?"`, `const="all"`, `choices=[all, config, defaults, defaults-tree,
  plugins, searchpath]`. There is **no** `--info <group-name>` form — so
  `--info experiment` is not a real Hydra feature and was never going to "just
  work." The finding's headline symptom was a false target.
- The *real* loss: deriva-ml's wrapper `--info` is `store_true` + a custom
  `_show_hydra_info()` that only lists config groups, **fully shadowing**
  Hydra's six native `--info` modes and all its other flags
  (`--cfg/-c`, `--resolve`, `--package/-p`, `--config-name/-cn`,
  `--config-dir/-cd`, `--experimental-rerun`, `--shell-completion`).
- **Model runner** (`run_model.py`) replaces `sys.argv` and calls
  `zen(run_model).hydra_main(...)` → `hydra.main` re-parses that argv with
  Hydra's *full* parser. So anything we put in that argv is honored by Hydra.
- **Notebook runner** (`run_notebook.py`) never hands argv to Hydra. It
  serializes overrides to `DERIVA_ML_HYDRA_OVERRIDES` (JSON env var); the
  kernel calls `hydra_zen.launch(config, overrides=[...])`, which takes an
  overrides list + multirun bool and **no** flag parameters. Papermill owns
  `--parameter/-p` (collides with Hydra's `--package/-p`), `--kernel`,
  `--file`, `--log-output`, and the `notebook_file` positional.
- Both runners use `parse_args` (via `BaseCLI.parse_cli`), not
  `parse_known_args`, so unknown flags are rejected before Hydra runs.

## Design decisions (approved by user)

1. **Model runner → `parse_known_args` passthrough (max native).** Stop
   intercepting Hydra's flags. Keep only deriva-ml-specific flags explicit;
   forward the unrecognized remainder into the Hydra argv. Every Hydra-doc flag
   works, present and future, with no per-flag maintenance.
2. **Notebook runner → Hydra-vocabulary inspection, kernel-side.** Implement
   `--info <mode>` (Hydra's choices) and a `--cfg`-equivalent by routing through
   Hydra's compose/render in the kernel; papermill keeps its orthogonal flags.
   Full flag-passthrough is architecturally out of reach (no argv) and not
   attempted.
3. **Rename the hydra-zen group menu `--info` → `--list-configs`** (both
   runners). This frees the `--info` name to flow natively to Hydra. Clean
   break, no compat alias (workspace rule: no backwards-compat shims). This is
   a deliberate behavior change documented across all three repos.
4. **One coordinated change across three repos:** deriva-ml (code + tests +
   docstrings + user-guide), deriva-ml-skills (configure-experiment +
   experiment-lifecycle), deriva-ml-model-template (docs that mention `--info`).

## The three CLI operations — the canonical table all docs/tests are written against

| Intent | Command | What it does | Owner |
|---|---|---|---|
| "What `group=value` can I pass?" | `deriva-ml-run --list-configs` | Lists hydra-zen registered config groups + their options + named multiruns. **deriva-ml-specific; Hydra has no equivalent.** | deriva-ml wrapper |
| "What config will actually run?" | `deriva-ml-run +experiment=X --cfg job` | Hydra renders the fully-composed/resolved config for the given overrides, without executing. | **Hydra native** |
| "Show Hydra internals" | `deriva-ml-run --info config\|defaults\|searchpath\|...` | Hydra's own info modes (composed config, defaults tree, search path, plugins). | **Hydra native** |
| "Resolve + validate against the live catalog" | `deriva-ml-run +experiment=X dry_run=true` | deriva-ml resolves config AND checks every RID/term against the catalog; stops before training. Heavier (downloads bag). | deriva-ml |
| "List notebook params" | `deriva-ml-run-notebook nb.ipynb --inspect` | papermill parameter-cell inspection. | papermill |

**The doc bug this fixes:** `configure-experiment/SKILL.md:39` currently says
`+experiment=X --info` "inspects resolved config." It does NOT — old `--info`
ignored the override and listed the menu. After this change, the correct command
for "inspect resolved config" is `--cfg job`; the menu is `--list-configs`.

---

## Part A — model runner (`run_model.py`)

### A1. Switch to `parse_known_args`

`BaseCLI.parse_cli()` calls `self.parser.parse_args()`. We need the unknown
remainder. Options:
- Add a `parse_known` path. Cleanest: call
  `args, unknown = self.parser.parse_known_args()` directly in `ModelRunner`
  rather than `BaseCLI.parse_cli()`, OR have `parse_cli` expose a
  `return_unknown` switch. Prefer a local `parse_known_args` in the runner to
  avoid changing the shared deriva-py `BaseCLI` (out of our repo).
- Keep all current explicit args: `--catalog`, `--config-dir/-c`,
  `--config-name`, `--multirun/-m`, `--allow-dirty`, and the `hydra_overrides`
  positional, plus inherited BaseCLI args (`--host`, `--debug`, etc.).

**Collision audit (deriva-ml flags vs Hydra flags):**
- `--config-dir/-c`: deriva-ml uses `-c` for config-dir (preloads the configs
  package); Hydra uses `-c` for `--cfg`. **Collision.** Resolution: deriva-ml's
  `-c` is consumed by our argparse (explicit), so it never reaches Hydra. A user
  wanting Hydra's `--cfg` must spell it `--cfg` (long form), which our
  `parse_known_args` won't recognize → forwarded to Hydra → works. Document:
  "`-c` is deriva-ml's config-dir; use `--cfg` (long) for Hydra's config dump."
- `--config-name`: deriva-ml owns it (passed to `hydra_main(config_name=...)`).
  Hydra's `--config-name/-cn` would be redundant; if a user passes `--config-name`
  it's ours. Fine.
- `--multirun/-m`: deriva-ml owns it (drives `+multirun=` expansion + inserts
  `--multirun` into argv). A user passing `-m`/`--multirun` directly still ends
  up with `--multirun` in the Hydra argv. Consistent.
- `--info/-i`: **remove deriva-ml's `store_true` `--info`** and let Hydra's own
  `--info` flow through (so `--info config`, `--info searchpath`, bare `--info`
  → all, all work natively). BUT we lose the group-listing convenience. Keep it
  as a deriva-ml-specific spelling that does NOT collide: see A3.

### A2. Forward the remainder into the Hydra argv

Current argv build (`run_model.py:244-251`):
```python
hydra_argv = [sys.argv[0]] + hydra_overrides
if use_multirun:
    hydra_argv.insert(1, "--multirun")
```
New: append the `unknown` remainder (the Hydra-native flags argparse didn't
recognize) so Hydra re-parses them:
```python
hydra_argv = [sys.argv[0]] + hydra_overrides + unknown
if use_multirun and "--multirun" not in unknown and "-m" not in unknown:
    hydra_argv.insert(1, "--multirun")
```
Ordering: overrides first, then flags — argparse-style flags can appear anywhere;
Hydra's parser handles interleaving. Keep `+multirun=` expansion BEFORE this
(it operates on `hydra_overrides`, not `unknown`).

### A3. The guard now runs only on genuine bare positionals

`validate_hydra_overrides(args.hydra_overrides, ...)` stays, but with
`parse_known_args`, Hydra-native flags land in `unknown`, not `hydra_overrides`.
So a bare `roc_analysis` still lands in `hydra_overrides` and is still caught;
`--cfg job` lands in `unknown` and is forwarded. The guard's false positive
(Gap C) disappears because `--info experiment` is no longer our concern — if a
user types `--info experiment`, Hydra's own `choices=` rejects `experiment`
with a clean "invalid choice" message. **Net: the guard keeps its real job
(catch bare positionals) and loses its bug.**

### A4. `_show_hydra_info()` disposition

Hydra's native `--info` now reaches Hydra. The wrapper's group-listing is still
useful (Hydra has no "list my registered groups + multirun configs" mode). Keep
it under a non-colliding deriva-ml spelling — proposal: `--list-configs`
(or keep a deriva-ml `--info` that we intercept and Hydra never sees, but that
re-shadows Hydra's `--info`, defeating the goal). **Decision needed at impl
time** (see Open Questions). Move the implementation into a shared
`cli/show_info.py` helper (Part C) so both runners share it.

---

## Part B — notebook runner (`run_notebook.py`)

### B1. Hydra-vocabulary `--info`

Make `--info` accept Hydra's choices (`nargs="?"`, `const="all"`,
`choices=[all, config, defaults, defaults-tree, plugins, searchpath]`) plus the
deriva-ml group-listing spelling (shared helper). Because the kernel uses
`launch()`, "native" here means: when `--info config` is requested, resolve the
config the kernel *would* use (compose the registered config with the same
overrides via `hydra.compose` / hydra-zen's config) and print it in Hydra's
format — implemented in the runner/kernel boundary, not by forwarding argv.

### B2. `--cfg`-equivalent

Add `--cfg [job|hydra|all]`. Implement by composing the config with the
resolved overrides and printing — the same render Hydra's `--cfg` produces —
without executing the notebook. This is the "resolve & show, don't run" preflight
the Modeler wanted, far cheaper than the current `dry_run=true` workaround
(which downloads the dataset bag).

### B3. The `-p` collision

Papermill's `--parameter/-p` stays as-is (notebook param injection). We do NOT
introduce Hydra's `--package/-p` on the notebook runner (no argv to forward to
anyway). Document that `-p` on the notebook runner is papermill parameters; the
model runner's `-p` (if forwarded) is Hydra's `--package`. Asymmetry is
inherent and documented.

### B4. Papermill flags untouched

`--kernel`, `--file`, `--log-output`, `--inspect`, `notebook_file` positional —
all unchanged.

---

## Part C — shared `cli/show_info.py`

Collapse the two drifted `_show_hydra_info()` copies (`run_model.py:269-327`,
`run_notebook.py:448-522`) into one helper:
`render_config_groups(*, include_multirun: bool) -> str`. Model runner passes
`include_multirun=True`; notebook passes `False`. Removes the dependency on
hydra-zen's private `store._queue` being reimplemented twice.

---

## Touch points (three repos)

### deriva-ml (code + tests + docs)
| File | Change |
|---|---|
| `src/deriva_ml/run_model.py` | `parse_known_args`; drop `store_true --info`; add `--list-configs`; forward `unknown` into argv; call shared show-configs helper |
| `src/deriva_ml/run_notebook.py` | `parse_known_args`; `--list-configs`; `--info [mode]` + `--cfg [job\|hydra\|all]` rendered kernel-side; call shared helper; papermill flags untouched |
| `src/deriva_ml/cli/show_info.py` (new) | shared `render_config_groups(*, include_multirun)` helper (collapses the 2 drifted copies) |
| `src/deriva_ml/cli/hydra_overrides.py` | unchanged (guard keeps its real job: catch bare positionals) |
| `src/deriva_ml/execution/base_config.py` | kernel-side `--cfg`/`--info <mode>` compose+render entry point |
| `docs/user-guide/hydra-zen.md` | the canonical 3-operation table; `--list-configs` vs `--info`/`--cfg`; link hydra.cc override-grammar + flags docs; note `-c`/`-p` collisions + model/notebook asymmetry |
| runner module + class docstrings | correct "Hydra config options" → the precise meaning; runnable `Example:` blocks for `--list-configs`, `--cfg job`, `--info config` |

### deriva-ml-skills
| File | Change |
|---|---|
| `skills/configure-experiment/SKILL.md` | FIX line 39 error (`--info` ≠ resolved config); detail the 3 operations; link Hydra docs |
| `skills/experiment-lifecycle/SKILL.md` | line 50 gate: `--list-configs` to see the menu / `--cfg job` to inspect the resolved config / `dry_run=true` to validate against catalog |

### deriva-ml-model-template
| File | Change |
|---|---|
| `docs/**` mentioning `--info` (configuration/*, getting-started/*, quick-start) | `--info` → `--list-configs` where the menu is meant; add Hydra-flag pointers |
| `CLAUDE.md` "Notebook runner specifics" | reconcile with new flag surface |

## Test strategy (TDD, RED first) — comprehensive matrix

The user explicitly requires comprehensive test cases. Organize as a new
`tests/cli/` package. Use a lightweight approach where possible: assert on the
**argv handed to Hydra** (mock/capture the `zen(...).hydra_main` boundary) so
most cases are fast unit tests, not full catalog runs. Reserve a few live cases
for the genuinely end-to-end paths.

### Model-runner argv construction (`tests/cli/test_run_model_argv.py`) — unit, mock the Hydra boundary
1. `--cfg job` → appears in the forwarded argv (and NOT in `hydra_overrides`/guard).
2. `--cfg hydra` / `--cfg all` → forwarded verbatim.
3. `--info config` / `--info defaults` / `--info searchpath` → forwarded.
4. bare `--info` → forwarded (Hydra defaults to `all`).
5. `--resolve`, `--package model_config` → forwarded.
6. `--list-configs` → intercepted by the wrapper, NOT forwarded, prints the menu, exits 0.
7. `model_config=cifar10_quick` (override) → in argv, passes guard.
8. `+experiment=X --cfg job` → both the override AND `--cfg job` reach Hydra.
9. `+multirun=<name>` → expands to the named overrides AND inserts `--multirun`.
10. `--multirun model=a,b` → `--multirun` survives, sweep override forwarded.
11. bare positional `roc_analysis` → guard raises the diagnostic ValueError (exit 1).
12. typo flag `--catlog` → forwarded to Hydra (Hydra errors), NOT silently swallowed — assert non-zero/Hydra-error, proving loud-wrong not silent-wrong.
13. `--catalog 45` → injects `deriva_ml.catalog_id=45`; `--host h` → `deriva_ml.hostname=h`.
14. `--allow-dirty` → sets env, not forwarded.
15. ordering: overrides + forwarded flags coexist (interleave) without loss.

### Notebook-runner (`tests/cli/test_run_notebook_cli.py`)
16. `--list-configs` → prints menu (shared helper), no notebook execution.
17. `--info config` → renders composed config kernel-side, no execution, exit 0.
18. `--cfg job` → renders resolved config, no execution.
19. `--parameter foo 1` / `-p foo 1` → papermill param injected; NO collision with Hydra `--package` (papermill `-p` wins on this runner, by design).
20. `--kernel`, `--file params.json`, `--inspect`, `--log-output` → unchanged behavior.
21. bare positional → guard raises.
22. overrides still serialize to `DERIVA_ML_HYDRA_OVERRIDES` and reach `launch()`.

### Shared helper (`tests/cli/test_show_info.py`)
23. `render_config_groups(include_multirun=True)` → lists groups + options + multirun section.
24. `render_config_groups(include_multirun=False)` → groups + options, NO multirun section.
25. handles empty store / no groups without crashing.

### Guard (existing `tests/cli/test_hydra_overrides.py`)
26. unchanged behavior — still catches bare positionals; regression-guard that Hydra flags are NOT passed to it (they're in `unknown` now).

### Doctest + full suite
27. New runnable docstring `Example:` blocks for `--list-configs`/`--cfg job` pass the doctest harness (or are clearly `# doctest: +SKIP` where a live catalog is needed).
28. Full `uv run python -m pytest` + doctest harness green.

### Cross-repo verification (manual / scripted, in the e2e worktree)
29. `deriva-ml-run --list-configs` prints the menu (was `--info`).
30. `deriva-ml-run +experiment=cifar10_quick --cfg job` prints resolved config (the thing the skill claimed `--info` did).
31. Skills + model-template docs grep-clean of stale `--info`-means-menu usage.

## Implementation refinements (discovered during TDD — these are now decided)

- **Drop the `nargs="*"` positional entirely.** A greedy `nargs="*"` positional
  STEALS the value of an interleaved unknown flag (`--cfg job model=x` →
  `unknown=['--cfg']`, `'job'` wrongly absorbed as an override). Verified. Fix:
  define only deriva-ml-specific flags on the parser, call `parse_known_args`,
  and treat the **entire ordered remainder** as overrides+Hydra-flags. Hydra's
  own parser handles them in any order, so we hand it the remainder verbatim
  (order preserved). `--catalog`/`--host` are still consumed by our explicit
  flags and injected as `deriva_ml.*` overrides.
- **Guard = flag-value-aware (Option 3), self-maintaining.** The friendly
  bare-positional error (PR #247) is kept and works regardless of flag
  position. The guard derives Hydra's flag→arity map at runtime from
  `hydra._internal.utils.get_args_parser()` (NOT hardcoded — auto-tracks new
  Hydra flags), walks the remainder skipping each known Hydra flag and its
  value(s), and only flags a genuine bare positional (no `=`/`~`, not a flag,
  not a flag's value). Arity map observed: `--cfg/-c`=1, `--package/-p`=1,
  `--info/-i`=0-or-1, `--resolve`=0, `--config-name/-cn`=1, `--config-dir/-cd`=1,
  `--config-path/-cp`=1, `--experimental-rerun`=1, `--multirun/-m`=0,
  `--run/-r`=0, `--shell-completion/-sc`=0.

## Open questions (resolve at implementation)

1. **Model runner group-listing spelling.** Hydra's `--info` now flows through,
   so we can't keep deriva-ml's `--info` for group-listing without re-shadowing.
   Options: `--list-configs` (clear, no collision) / `--info groups` as a
   deriva-ml *extension* intercepted before forwarding (risks confusion since
   `groups` isn't a Hydra choice) / drop the wrapper listing entirely and tell
   users `--info defaults`. Leaning `--list-configs`.
2. **Does `parse_known_args` swallow `--debug`/BaseCLI flags into `unknown`?**
   No — they're registered on the parser, so they're recognized. Only truly
   unregistered flags land in `unknown`. Verify in impl.
3. **Notebook `--cfg` rendering path.** Confirm hydra-zen exposes a compose-only
   entry (so we render without `launch()` executing the task). If not, use
   `hydra.compose` against the registered config name.

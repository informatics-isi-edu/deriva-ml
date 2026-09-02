# DerivaML Tacit Knowledge

Rationale that the catalog doesn't capture. Newest entries at the bottom
of each section.

## Execution / commit

### 2026-07-17 — RID-lease POST retry design (issue #360, PR #361)

**Bug:** `post_lease_batch` (`execution/rid_lease.py`) issues a bare
`catalog.post("/entity/public:ERMrest_RID_Lease", ...)` with no retry.
A *transient* `409 CONFLICT "Schema public does not exist"` on that
single bookkeeping POST propagated out of `commit_output_assets` and
marked an otherwise-complete 19-hour, 14k-asset execution **Failed**.
The data (assets + feature rows + provenance) had all committed; only
the status was a false negative.

**Why the existing retry machinery didn't cover it:**
- deriva-py's session-level urllib3 `Retry` (`DEFAULT_SESSION_CONFIG`)
  has `retry_status_forcelist = [500, 502, 503, 504]` (409 not in it)
  **and** `allow_retry_on_all_methods = False` — so POSTs get zero
  transport-level retry, and 409 is excluded even for GETs.
- deriva-py's app-level `datapath._request_with_retry`
  (the "loader retry machinery" the issue references) defaults
  `retry_codes = {408, 429, 500, 502, 503, 504}` — **409 deliberately
  excluded** because for a normal insert a 409 is a real unique-key
  conflict, not transient.

**The load-bearing fact (verified against localhost catalog):**
`ERMrest_RID_Lease` has a **composite unique key `(RCB, ID)`**, where
`RCB` = row-created-by (authenticated client) and `ID` = our
client-generated UUID4 lease token. `ID` alone is NOT unique.

Consequence for retry-safety: a naive full-POST retry is **unsafe** —
if the first POST landed but its response was lost, retrying the same
tokens hits `(RCB, ID)` and 409s on the rows that actually succeeded.
So "just wrap the POST in blind retry-on-409" would trade one false
failure for another.

**Correct design = reconcile, don't blind-retry.** The UUID4 `ID`
token exists precisely for this (per `generate_lease_token` docstring:
"so we can look up what we leased after a mid-flight crash") — the
current code just never used it. On failure, query
`ERMrest_RID_Lease` for `RCB=me AND ID in (our tokens)` to learn which
tokens already landed, then POST only the missing ones. Idempotent by
construction; safe to retry any number of times. Retry transient
failures (timeouts, 5xx, and 409 where reconciliation shows work
remains) with bounded exponential backoff; still raise on terminal
failure.

**Placement decision (Carl, 2026-07-17): keep the whole fix in
deriva-ml.** The `(RCB, ID)` idempotency is technically an ERMrest
guarantee, but we chose not to add a deriva-py primitive / pin bump
for this. Trade-off accepted: faster to land, contract not pinned
upstream. Mitigation: the reconcile-retry lives in
`execution/rid_lease.py`, and a deriva-ml test asserts the
`(RCB, ID)` composite-unique guarantee against a live catalog so our
own CI catches a future server/deriva-py change that would break the
idempotency assumption. Document the ERMrest dependency inline in the
lease module.

**Codex-review hardening (same day, PR #361):** an independent Codex
review caught two real gaps in the first cut, both fixed:
1. **Reconcile must be RCB-scoped.** The first reconcile query filtered
   on `ID` alone. Since `ID` is unique only within `(RCB, ID)`, a token
   value colliding under *another* client's RCB could be adopted —
   returning a RID we never leased. Fixed: scope the reconcile to
   `RCB == _client_rcb(catalog)` (whoami's `client.id`). UUID4 makes the
   collision astronomically unlikely, but the query now matches the
   stated contract instead of relying on that luck.
2. **The reconcile query itself must tolerate transients.** The retry
   hardened the POST but left `_fetch_landed_leases` unguarded — a 503/
   timeout on the reconcile GET propagated straight out, re-introducing
   the exact false-failure #360 fixes, one call over. Fixed: the
   reconcile is wrapped so a transient failure consumes a retry attempt
   (like a POST failure) instead of aborting; terminal reconcile errors
   still propagate. Lesson: when you add a bookkeeping query *inside* a
   retry loop to make it safe, that query inherits the same
   transient-failure obligation as the operation it guards.

## Dataset / ML-framework adapters

### 2026-08-03 — Sample-ordering shuffle belongs to `as_tf_dataset` only (issue #362)

**Decision:** implement the `shuffle=`/`seed=` parameters from #362 on
`as_tf_dataset` / `build_tf_dataset` **and nowhere else**. Do *not* add
a matching parameter to `as_torch_dataset` or `restructure_assets`.

**Why this is a deliberate asymmetry, not an oversight.** The three
bag-to-ML paths differ in whether the *consumer* can shuffle cheaply:

- **`as_torch_dataset`** returns a **map-style** dataset (`__len__` +
  `__getitem__`, `torch_adapter.py`). `DataLoader(shuffle=True)` wraps
  it in a `RandomSampler` that permutes *indices* — a true global
  shuffle costing one permuted index array, no decoded sample held in
  memory — and it reshuffles **every epoch**.
- **`as_tf_dataset`** returns `tf.data.Dataset.from_generator`
  (`tf_adapter.py`) — **iterator-style, no random access**. See the
  `.shuffle()` note below: TF *does* ship a shuffle, but it operates at
  the wrong layer, so deriva-ml must shuffle the RID list itself.
- **`restructure_assets`** writes `{training,testing}/{label}/file`.
  The class grouping there **is the contract**, not a hazard:
  `ImageFolder` derives its class list and integer label mapping from
  those directory names. A shuffle parameter would have nothing to
  shuffle — interleaving classes would destroy the labeling scheme.
  Shuffling belongs in the consumer's sampler.

**Why class-ordered data breaks training at all** (the prior question a
reviewer asks before "why global?"). Two distinct failures:
1. **Biased per-step gradients.** SGD assumes each minibatch is roughly
   an unbiased sample of the training distribution. An all-class-A
   batch yields a gradient pointing at "always predict A". With
   momentum and BatchNorm it compounds — BN statistics are computed
   per-batch, so a single-class batch calibrates the layer on a
   distribution never seen at inference. The #362 signature is exactly
   this: val_acc oscillating 0.623 ↔ 0.377 *is* the model alternating
   between "everything is A" and "everything is B", and those two
   numbers are the class proportions; AUC ~0.5 confirms no ranking was
   learned.
2. **Catastrophic forgetting.** All of A then all of B yields a
   B-predictor — nothing re-exposes A after B's updates.

**Why pure-TF users usually don't hit this** (so "TF has no such knob,
why do we need one?" has an answer): they do — it's a known footgun,
hence `image_dataset_from_directory(shuffle=True)` as the default and
`.shuffle()` in nearly every tutorial. It stays invisible because their
data is typically already shuffled on disk or across TFRecord shards;
**or their elements are cheap** (CSV rows, pre-decoded vectors), so
`buffer_size=50_000` is affordable and the reservoir *is* global in
practice; **or they shuffle file paths before the expensive decode**
(`list_files(shuffle=True)` → `.map(decode)`).

That last pattern is precisely what #362 does — the RID list *is* our
file-path list. So this is not deriva-ml inventing something un-TF-ish;
it reproduces the idiomatic TF pattern at the only layer where we still
hold cheap handles. Our case is the intersection of the two conditions
that make it bite: **class-grouped at the source AND elements too
expensive to buffer whole.** Neither alone is fatal.

**Why *global*, given weaker options exist.** The requirement is only
that batches be class-mixed. Alternatives: a `.shuffle()` buffer larger
than the longest single-class run (≥3000 decoded images here, and it
scales with the dataset — doesn't hold up); interleaving per-class
streams via `sample_from_datasets` (needs classes as separate datasets;
we have one RID list); class-balanced sampling (same problem, and it
alters the effective class distribution). Since we already hold the
full RID list in memory, a global permutation is simply the cheapest
correct option — one `random.Random(seed).shuffle()` over short
strings. The weaker alternatives cost more machinery for a worse
guarantee.

**Corollary — the fixed-order limitation is not a real weakness.** A
single build-time permutation already eliminates both failure modes
above: every batch is class-mixed from epoch 1. Per-epoch reshuffling
only reduces overfitting to a particular batch composition — real, but
second-order. That is the correct division of labor between
`shuffle_rids=` and a chained `.shuffle()`.

**`tf.data.Dataset.shuffle()` exists — and is not a substitute.** This
is the first objection anyone raises, so state the answer precisely.
`.shuffle(buffer_size)` is a **reservoir shuffle over a sliding
window**, not a global permutation: it fills a buffer, emits one
element at random, refills from the stream. An element can therefore
only move `buffer_size` positions from where it started. On a
class-grouped bag — `[3000 class-A][1800 class-B]` with a typical
`buffer_size=1000` — the buffer is *entirely class-A* for the first
~3000 elements, so every batch in that stretch is still single-class.
It shuffles within class A and changes nothing about the collapse. A
true global shuffle needs `buffer_size >= len(dataset)`, and because
`.shuffle()` sits **downstream of the generator**, that buffer holds
*decoded* samples — the whole image set in RAM.

The contrast with torch is the whole point: `DataLoader(shuffle=True)`
permutes an *index array* before any loading happens. `tf.data` cannot,
because a `from_generator` dataset is an opaque iterator with no
indices to permute. Shuffling the RID list inside `build_tf_dataset`
reorders short strings *before* the generator closes over them — the
same index-level trick, applied at the only layer where TF still has
cheap handles on the elements.

**The two compose; recommend both.** The build-time RID shuffle is
fixed for all epochs (#362 acknowledges this). Chaining a modest
`.shuffle(buffer_size=1000)` on top adds per-epoch variation *within*
an already class-mixed stream, which is cheap and effective precisely
because the global ordering was fixed upstream. Guidance is "both, for
different jobs" — never "adapter instead of `.shuffle()`".

**Name the parameter `global_shuffle`, not `shuffle` — deliberately
diverging from `tf.data` vocabulary.** #362 proposes `shuffle=` /
`seed=`. Those names collide with `tf.data.Dataset.shuffle(buffer_size,
seed=...)`, which a TF user already knows and which behaves
**differently on the axis they care about**: `.shuffle()` reshuffles
every epoch by default (`reshuffle_each_iteration=True`); ours is one
build-time permutation, identical every epoch. Same word, opposite
epoch semantics, and the mismatch is *silent* — training converges and
looks plausible while epoch 7 replays epoch 1's exact order.

The second-order trap is the real cost: a user who sees `shuffle=True`
in the constructor concludes shuffling is handled and **drops
`.shuffle()` from their pipeline**, trading a per-epoch shuffle for a
fixed one without being told. The parameter's presence invites removing
the thing doing the more useful work. `seed` compounds it — in
`tf.data` it seeds the reservoir, in ours a `random.Random` over the
RID list; two RNGs, two scopes, one name.

So: `global_shuffle: bool = False`, `shuffle_seed: int | None = None`.

**Why `global_shuffle` and not `shuffle_rids`** (the first cut, rejected
same day): `shuffle_rids` names the *mechanism*, `global_shuffle` names
the **guarantee** — and the guarantee is the axis the user actually
needs. "RIDs vs. samples" is deriva-ml-internal vocabulary a TF user
doesn't carry; "global vs. windowed" is precisely where `.shuffle()`
falls short. `global_shuffle=True` beside `.shuffle(buffer_size=1000)`
reads as a coherent pair — one global, one buffered — so a user who
understands why the buffer is bounded immediately understands why a
separate knob exists, without knowing what a RID is. It also survives
implementation drift: if we ever permute a row index instead of the RID
list, `shuffle_rids` becomes a lie while `global_shuffle` stays true.
Name the guarantee, not the mechanism.

**What the name still cannot carry:** epoch semantics. A user may read
"global" as "strictly better" and infer it subsumes `.shuffle()`,
dropping theirs and silently losing per-epoch variation. The docstring's
**first line** must say: shuffles once at construction, same order every
epoch, chain `.shuffle()` for per-epoch variation. No name fixes this.

Cost is divergence from the torch adapter's vocabulary — which is no
cost, since the torch adapter has no such parameter and shouldn't.
eye-ai-vgg19's `TypeError` guard fails loud on the wrong name, so the
rename surfaces at once instead of silently no-op'ing. Do not "fix"
this back toward TF's vocabulary for consistency; the divergence is
the point.

**Do NOT add a passthrough seed for the native `.shuffle()`.** Asked
and rejected. `.shuffle()` is called on the **returned** dataset, in
user code, after `as_tf_dataset` returns — the caller already has
direct access to `seed=` there. Accepting a `tf_shuffle_seed=` would
force us to also accept `buffer_size`, then
`reshuffle_each_iteration`, then `.batch()`'s arguments: wrapping the
`tf.data` builder API one parameter at a time, for zero gain, since
every knob is already reachable on the returned object. (This is the
"too many interfaces" failure the repo's class-idiom guidance warns
about.) It would also undo the naming decision above — the entire
argument is that `global_shuffle` and `.shuffle()` are *different
operations at different layers*; putting a `tf.data` concern back on
our constructor re-muddies the boundary we just drew.

Clean division: **deriva-ml owns everything before the generator**
(`global_shuffle`, `shuffle_seed`); **the user owns everything
downstream on the returned `tf.data.Dataset`** (`.shuffle()`,
`.batch()`, `.prefetch()`, and their seeds).

Two consequences for docs, not for the API:
- **Full reproducibility needs both seeds.** `shuffle_seed` pins the
  build-time order; `.shuffle(seed=...)` pins the per-epoch reservoir.
  Setting only ours still leaves epoch-to-epoch ordering
  nondeterministic. Show the complete two-seed recipe in the docstring
  — "I set the seed and still can't reproduce" is the predictable
  support question.
- **Provenance callers must record both.** eye-ai-vgg19 records
  `shuffle_seed` in `run_config.json` to make batch composition
  recoverable; that only holds if the rest of the pipeline is
  deterministic, so if it chains `.shuffle()`, that seed belongs in the
  recorded hyperparameters too. Caller-side note, not a library change.

**Why adding it to torch anyway would be actively harmful** (not merely
redundant): a user who sets *only* the adapter-level flag gets one
build-time permutation reused for **every epoch**, silently losing the
per-epoch reshuffling `DataLoader` gives by default — worse than the
status quo. It also makes `seed` ambiguous, since `DataLoader`
reproducibility is governed by torch's global RNG / `generator=`, not
by an adapter argument.

**Residual risk to document, not to code around:** a third-party
trainer that walks the `restructure_assets` tree *without* shuffling
(a hand-rolled `glob`-and-iterate) hits the same class-grouped
collapse. `ImageFolder` + `DataLoader(shuffle=True)` and
`tf.keras.utils.image_dataset_from_directory` (`shuffle=True` by
default) are both safe — and note *why* the latter is safe, since it
is a `tf.data` path: it shuffles the **file list** it builds by walking
the tree, before any decoding. That is a global index-level shuffle,
structurally like `DataLoader`'s, not a reservoir over decoded
samples. The `.shuffle()` limitation above applies to the *generator*
path only. This is a docs concern for the consumer, not an API gap.

**Rule of thumb for future adapters:** deriva-ml owns sample ordering
only when it hands the consumer an iterator with no cheap global
shuffle. If the adapter returns something randomly-addressable, the
framework's sampler is the correct place — leave it there.

### 2026-08-03 — Adapter logic tests were 100% broken behind `importorskip` (PR #363 follow-up)

**What was found:** every test in `test_tf_adapter_logic.py` (11) and
`test_torch_adapter_logic.py` (13) failed on `main` — 24 tests, both
files entirely. Root cause: their MagicMock bags predate the adapters'
`reachable=True` default, which routes through
`resolve_reachable_rows` →
`Session(bag.engine).execute(bag._dataset_table_view(t))` — real SQL.
A MagicMock returns a MagicMock where SQLAlchemy wants a `text()`
construct, so every test raised `ArgumentError`.

**Why it went unnoticed for so long — the load-bearing lesson.** Both
modules open with `pytest.importorskip`, and neither torch nor
tensorflow is in the default dev environment. **A module that skips
its entire contents is indistinguishable, in summary output, from one
that passes.** `uv run pytest` reported green while 24 tests were
uncollected. Discovering this required `uv sync --extra tf --extra
torch` — something no routine workflow does.

**Compounding factor:** this repo has **no CI test workflow at all**
(`.github/workflows/` holds only `validate-schema`, `publish-docs`,
`release`). So there is no pipeline where a "wholesale skip" check
would even run today. Adding one is a real decision with real cost —
the suite needs a live Deriva catalog and runs 30–90 min — so it was
left to the maintainer rather than bundled into a test-fix PR.

**Fix chosen: stub the SQLAlchemy Session, don't opt out.** The quick
fix was passing `reachable=False` everywhere, and the first cut of the
#362 shuffle tests did exactly that. Rejected on reflection: it opts
the logic tests out of the branch **real callers take**, leaving the
`reachable=True` default with zero logic coverage. Instead
`tests/dataset/_reachable_stub.py` patches
`target_resolution.Session` so mock bags resolve the reachable path to
their own `list_dataset_members` rows. Verified non-vacuous: breaking
`resolve_reachable_rows` to return `[]` fails 22 of 34 tests, which
`reachable=False` versions could not have caught.

**Rule this generalizes to:** when an optional-dependency module is
gated by `importorskip`, *someone must actually install the extra and
run it* — periodically, or in CI. Otherwise the gate silently converts
"broken" into "green". Treat a fully-skipped module as an untested
module, not a passing one.

## Lineage / provenance

### 2026-08-31 — Snapshot-schema lookups must be guarded as a group (issue #365, PR #366)

**Bug:** `lookup_lineage()` raised `KeyError: 'Image_Execution'` and
aborted for executions consuming a dataset whose member asset table
postdates the consumed version's snapshot (live eye-ai `7-QCAA`,
`5-E9WE`).

**The load-bearing invariant:** a snapshot pathBuilder reflects the
catalog schema *as of the dataset version's snaptime*. A table created
after that snaptime is **absent from the snapshot schema**, and the
snapshot necessarily holds **no rows** for it. So the correct answer
for a missing table is an empty result — never an exception. Schema
absence at a snaptime is normal catalog evolution, not an error
condition.

**Why one guard wasn't enough.** `_distinct_member_output_producers`
made three sibling lookups (membership table, member table,
`<member>_Execution` association) and guarded only the first. The
existing guard's own comment already stated the invariant covering all
three — the reasoning was right, the coverage wasn't. When several
lookups share one precondition, guard them as a **group**; a guard on
the first sibling reads as "handled" and hides that the rest are
exposed.

**Why the blast radius exceeded the bug.** The KeyError propagated out
of `lookup_lineage` instead of degrading locally, so an affected
execution contributed *nothing* to the walk — consumed datasets,
consumed assets, and parents, not just the member-producer set the
failing helper computed. It suppressed the provenance chain for the
graph's most provenance-complete execution (the only one registering
an `Asset_Role="Input"` asset). **A helper deep in a graph walk that
raises instead of degrading converts a narrow gap into total data
loss for the caller.** Prefer returning an empty result from
leaf helpers whose failure mode is "this branch has nothing".

**Detection note:** the callers most likely to surface this are
long-lived catalogs where schema evolved *after* datasets were
versioned — i.e. production, not fresh test catalogs. Offline unit
tests can reproduce it cheaply by building a snapshot pathBuilder whose
`schemas[...].tables` dict simply omits the table (see
`tests/execution/test_producers_of_dataset_members.py`).

### 2026-08-31 — Dataset origin = earliest-version author; trace root-only (issue #367 design)

**Decision (with Carl):** `lookup_lineage`'s `producing_execution` for
an unversioned dataset root resolves to the author of the **earliest**
`Dataset_Version` row (structural origin), not the latest
(last-writer-wins, the #367 bug where a data migration was reported as
producer of the LAC splits). A new `version_history` trace (ordered
`(version, execution, description)` list) rides on the **root
descriptor only**, plus `origin_recorded: bool` flagging
sentinel/None origins. **No schema change.**

**Why no schema change:** the semantics question in #367 was already
settled by the provenance contract — `Dataset_Version.Execution` means
"author of this version" (authorship-canonical model), and the 92%
sentinel population is the contract's documented adoption backfill,
not corruption. The defect was presentation: the unversioned path
answered "who last touched it?" while lineage presented it as "how did
this come to exist?". Origin is derivable because `create_dataset`
writes the v0.1.0 row with the creating execution — earliest-row
author = origin **by construction** going forward; pre-contract
datasets honestly resolve to the sentinel ("origin unrecorded",
unrecoverable).

**Why trace root-only, not on consumed-dataset mentions:** consumed
mentions are version-pinned and the pin IS the provenance of that
consumption; a popular dataset appears in every walk that trained on
it (payload multiplies across the MCP boundary); and any dataset's
full trace is one `lookup_lineage(rid, depth=0)` call away. Rejected
alternative: explicit origin modeling (Version_Kind column /
Produced_By FK) — schema change + 5,258-row backfill for something
derivable; revisit only if migration-tagging is needed beyond
presentation.

**Implementation traps recorded in the spec:** (1) the existing `_key`
collapses unparseable versions to `(0,)` — fine under `max`, but under
`min` a dev row `0.4.0.post1.dev3` would sort before `0.1.0` and steal
the origin; parse PEP 440 via `packaging` (already a dep), RCT
fallback. (2) sentinel classification must degrade, not raise, when
the sentinel is absent (#365 lesson: lineage leaf helpers never abort
the walk). (3) `origin_recorded` carries interpretation separately
from `producing_execution`, which keeps the sentinel value — truth and
judgment as separate fields.

**Related gap deliberately split out:** `Dataset_Dataset` has no
version pin (can't say which of a parent's 85 versions a child derived
from) — separate schema decision, own issue.

Spec: `docs/superpowers/specs/2026-08-31-dataset-origin-lineage-design.md`.

**Codex-review revision (same day):** an independent Codex review of
the spec forced one behavioral design change and one ordering pivot,
both verified against the code before accepting:
1. **Walk seed ≠ origin.** `lookup_lineage` seeds the walk from a
   member producer when no version-producer exists (tk-018), then
   overwrites `root.producing_execution` with that walk root
   (`execution.py:1226`) — under the new semantics that would claim a
   member producer as origin while `origin_recorded=False`. Fixed in
   design: walk seeding unchanged; `producing_execution` is built from
   origin resolution only and never overwritten from the walk.
2. **Order by RCT, not PEP 440 label.** `create_dataset` accepts an
   arbitrary initial `version=`, so the label is not creation order;
   RCT is total, parse-free chronology (label only as RCT tiebreak).
   Also corrected: the "zero added round-trips" claim (ExecutionSummary
   needs a batched Execution fetch — version rows hold RIDs, not
   summaries) and `origin_recorded` became tri-state (`None` = not a
   dataset root). Lesson repeated from #361: independent review of a
   *design* pays the same way it does for code — both real catches were
   in interactions with existing behavior, not in the new code itself.
3. **Round 2 added: the sentinel is never a walk root.** Seeding the
   walk from the unknown-provenance sentinel with member producers as
   its parents fabricates edges claiming the sentinel *consumed* those
   producers. The contract says lineage *terminates* at the sentinel —
   so a sentinel origin seeds the walk from member producers directly,
   like the no-origin branch. Also caught in round 2: the
   `create_schema.py` Dataset_Version.Execution comment still said the
   initial row "has no producing execution" — contradicting the
   contract it predates; comments are contract surface too.

**Codex code-review catch before merge (2026-08-31, commit 2ebbd429):**
cross-model review of the finished branch found the one hole every
per-task review and the whole-branch review missed: **the sentinel
enters `member_producers` too.** The provenance backfill attributes
producerless member *assets* to the unknown-provenance sentinel via
`Asset_Role="Output"` edges — so on any backfilled catalog the sentinel
RID sits in the member-producer set, and the member-fallback path could
seed the walk from it (or attach it as an extra parent), re-fabricating
exactly the sentinel-consumption edges the design forbids. The
origin-path guard alone was insufficient; the invariant is
**"the sentinel never enters the walk by ANY route"** — enforce it by
filtering the candidate set, not by guarding one path. Second catch:
candidate seeding must iterate ALL member producers (first-N-stale →
lineage silently lost). Lesson: an invariant stated as "X never happens
via path P" should be re-checked against every OTHER path that supplies
the same value — the backfill made the sentinel an ordinary member of a
set the design treated as sentinel-free.

### 2026-08-31 — Dataset_Dataset version pinning: implicit-not-absent (refines the #367 split-out)

**Correction to the recorded framing.** The #367 entry (and the issue)
said Dataset_Dataset "has no version pin — can't say which of a
parent's 85 versions a child derived from." Too strong. Carl's
question "why is Dataset_Dataset different from any other dataset
construction?" surfaced the real structure:

- **Mechanically it is NOT different**: every membership association
  (Dataset_Image, Dataset_Subject, Dataset_Dataset) is version-blind
  row-wise and versioned by **catalog snapshot** — each
  Dataset_Version row pins a Snapshot; membership-at-version = read
  the association at that snapshot.
- **The one real difference**: it is the only membership edge whose
  TARGET is itself versioned, so "which version of the member?" is
  askable — and answered *implicitly*: at the parent version's
  snapshot, the child's Dataset.Version pointer is readable. Version
  propagation (topological, children-before-parents,
  dataset.py:~1051) keeps this coherent: child changes force parent
  bumps whose snapshots see the new child state.
- **Residual gap (the honest version)**: the implicit answer is
  always "current at snaptime". It cannot express INTENT (a
  collection deliberately containing child @ an older version — the
  nesting analog of DatasetSpec's version pin on consumption), cannot
  distinguish "derived from vX" from "vX happened to be current", and
  fails where snapshots are missing (dev rows Snapshot=NULL,
  pre-snapshot-era versions — the same population lacking origin
  executions).

**Consequence for the prospective issue:** an explicit
Dataset_Version FK on Dataset_Dataset is a semantic EXTENSION
(pinnable containment, mirroring Dataset_Execution), not recovery of
lost data. File it, if at all, on the intent-pinning motivation —
and it may not clear YAGNI until someone actually needs old-version
containment. Do not re-file it as "derivation is unrecoverable";
snapshot reads recover it wherever snapshots exist.

### 2026-08-31 — Deploy-repo analysis → four provenance APIs (issues #370–#373)

**What happened:** analyzed deriva-ml-model-deploy's lineage scripts +
requirements against deriva-ml 1.57. Filed 6 issues (4 here, 2 there),
implemented the deriva-ml four as PRs #374–#377. Decisions a future
teammate needs:

- **find_feature_producers (#370/PR #376) contract = CANDIDATES, not
  dependencies.** Which features training code actually *read* is
  unknowable from the catalog, so the API promises a bounded superset
  with evidence (feature, element, count) — and **null-execution
  feature values are first-class results** (execution_rid=None), never
  dropped: they are the provenance gap the caller must see. Ported
  from the deploy repo's proven implementation rather than redesigned;
  membership discovery mirrors _producers_of_dataset_members (FK-derived
  link columns), NOT name-convention f"Dataset_{element}" lookup.
- **Metadata read API (#371/PR #375) split of failure modes is the
  point:** join failures PROPAGATE (a catalog problem must never read
  as "not recorded" — the absence-vs-timeout distinguishability is the
  API's reason to exist), per-row lookup failures degrade keeping the
  rest. Join first, then bounded per-row lookup_asset — the perf win is
  not streaming thousands of OUTPUT assets, not avoiding 5 lookups.
- **WorkflowSummary.version (#372/PR #374) ships WITH the #373 caveat
  in its docstring** — dedup means first-registration version, not
  per-run identity (observed 6 months stale on eye-ai 7-ZW3P). Never
  present Workflow.version as "the code this execution ran"; that
  truth lives in the environment snapshot.
- **Inference_Contract (#373/PR #377):** vocab term + enum only; the
  schema.md ↔ create_schema.py term agreement is enforced by the CI
  validator (it DOES check vocabulary terms — VOCAB_TERMS_MISMATCH),
  so seed-list changes always need both files.
- **Deliberately NOT moved upstream:** replay-vs-artifact declaration,
  commit selection (snapshot > workflow record), env-snapshot secrets
  allowlist, bakeability, arc-ranking policy — domain policy, stays in
  the deploy repo.

**Codex pre-merge P1 on find_feature_producers (2026-08-31, PR #376):**
membership-association FKs may reference **arbitrary member keys**
(`Dataset_file.file -> file.id` is a supported shape, handled explicitly
in `Dataset.list_dataset_members`), while feature-association FKs always
target the member's **RID** (deriva-ml creates those itself). Any query
composing the two must therefore route **through the member table**
(membership → member on `other_fkey.referenced_columns`, then
member.RID → feature) — a direct membership→feature comparison silently
returns no matches for non-RID shapes, and a per-feature degrade
swallows the failure. Generalizes: never assume an association FK
targets RID unless deriva-ml itself created the association.

### 2026-08-31 — Lineage HTML: adopt the pattern, not the script (issue #378, PR #379)

**Decision (with Carl):** the deploy repo's `visualize.py` HTML ledger
stays there — it renders the **WorkflowInventory** schema, i.e. the
deploy-side arc-ranking policy (DIRECT > DATA-FLOW > FEATURE-candidates
> ANCESTOR, gap labels) that two prior rulings kept out of deriva-ml.
Moving the renderer would have meant importing that policy's schema.
What deriva-ml adopted instead is the renderer's three PROVEN design
properties, applied to its own model: `lineage_result_to_html()` +
`deriva-ml-lineage` render a `LineageResult` — (1) **JSON-decoupled**
(rendering never touches the catalog; `model_dump()` is the audit
artifact of record, re-renderable/diffable forever), (2)
**self-contained single-file HTML** (inline CSS, no JS, no external
assets), (3) **stdlib-only**. Catalog text is untrusted → everything
escaped; workflow URLs link only when http(s). Rule of thumb this
extends: when downstream proves a *presentation* pattern, upstream the
pattern bound to upstream's own model — never the artifact bound to
downstream's policy schema.

### 2026-08-31 — Live validation of the 1.57–1.59 lineage stack (eye-ai, 7-ZW3P)

First real-catalog run, against the deploy repo's documented ground
truth (`deriva-ml-model-deploy/docs/design/provenance-resolution.md`):

- `deriva-ml-lineage --rid 7-ZW3P` walk matches §3.1/§8 exactly:
  7-ZW3P → 6-MWQE, visited=2, complete; datasets 5-WEBG/5-ZHRE/5-ZMGJ
  all @ v0.6.0 (consumed pins, not current). Execution root correctly
  shows origin fields as not-applicable (tri-state None).
- **`find_feature_producers("5-WEBG")` recovers by traversal exactly
  what #367 §6 said was unreachable**: the cropper `7-VZQY` and the
  detector `7-QCAA` (9,511 Annotation values each), plus five earlier
  annotation runs and four Image_Diagnosis graders — the bounded
  candidate set with evidence, live in seconds.
- Observation, not a defect: the rendered workflow badge shows the
  workflow-record version (`0.3.5.dev31+g893b98a55`) — the KNOWN-stale
  value for 7-ZW3P per #373 (run actually used g6d5865a82). The page
  reports the recorded value honestly; a future nicety could footnote
  the staleness caveat on the page itself.

**REVERSED (Carl, 2026-08-31, PR #381):** the Inference_Contract term
was dropped the same day it shipped — "never set, no producer
specified." The principle this establishes for future Asset_Type (and
other seeded-vocabulary) requests: **deriva-ml seeds only terms some
deriva-ml writer actually sets or some deriva-ml reader dispatches
on.** A term whose sole consumer is downstream tooling belongs in that
tooling's catalogs via per-catalog `add_term` — vocabularies are
extensible by design; the library-wide seed list is not the place to
park another repo's conventions. (The pre-existing Model_File/
Metrics_File terms are grandfathered taxonomy, not license to grow
it.) The Workflow.version staleness caveat — #377's other half —
stands. Note: v1.59.0/v1.60.0 installs still seed the term on catalog
init until the revert is released; catalogs initialized in that window
may carry a harmless orphan term.

## Provenance semantics

### 2026-08-31 — The boundary interview: capture-not-mining, and features as bindings (issues #383/#385, PR #384)

Settled with Carl in a structured boundary interview after the
lookup_provenance closure proposal (#383). Three load-bearing rulings:

1. **Capture, never mining.** deriva-ml owns facts derivable from its
   schema. When provenance needs a fact that isn't derivable, the fix
   is to extend CAPTURE so it becomes schema-recorded — never to teach
   readers to mine conventions (Hydra `key=RID` parsing rejected as a
   closure arc on this ground; it survives only as a downstream audit
   finding on legacy runs).

2. **Legacy is history, not requirements.** eye-ai's provenance holes
   (unrecorded splits, config-only cropper assertions) came from
   pre-contract deriva-ml. Going forward only contract-era records are
   assumed usable; closure completeness is defined against them, and
   legacy gaps are reported honestly (origin_recorded=False, sentinel
   terminations) — never compensated for.

3. **A feature value is the binding (member, value, execution).** Not
   an interpretation — the `is_feature` predicate structurally
   requires the Execution FK, so the execution is part of the
   binding's IDENTITY. This dissolved the feature-arc epistemics: the
   closure question was "did the consumer READ these features?"
   (unknowable, tk-023 → forced candidates-not-claims); Carl's
   reframing moves the attachment point — a dataset as consumed is
   members PLUS bindings, the bindings of D@vN are those at vN's
   snapshot, so their executions are provenance BY CONSTRUCTION.
   Uncertainty dissolved, not resolved. Corollaries: member scope is
   FK-reachable ("or referenced by an object in the dataset");
   construction vs content arcs stay distinct by attachment point
   (version row vs member bindings); null-execution bindings are
   MALFORMED (incomplete triples), reported never repaired. Canonical
   text now in provenance-contract.md Definitions (PR #384).

**Defect this surfaced:** find_feature_producers (#370) is
membership-only and live-state-only — the recurring
membership-vs-FK-reachable blind spot (#316/#318) reintroduced, plus
no version-snapshot scoping. Filed as #385; the closure (#383) depends
on it.

**Boundary sequence today, for the record:** Inference_Contract term
OUT of deriva-ml (#381 — no producer), closure traversal proposed IN
(#383 — all primitives native), config-mining kept OUT
(capture-not-mining), binding-constitutive feature provenance IN (it
was always in the schema). The pattern: the boundary follows the
SCHEMA, in both directions.

**Ruling 4 (same interview, Carl): there is no arc-strength ordering.**
The deploy repo's DIRECT > DATA-FLOW > FEATURE > ANCESTOR ranking was
an evidence-CERTAINTY gradient that only made sense while the feature
arc was a maybe-read candidate. Under binding semantics every arc kind
is an equally certain schema fact, so the gradient is flat; any
residual ordering is a relevance judgment = downstream presentation.
The closure model carries an UNORDERED typology of arc kinds
(distinguished by attachment point: consumption edge / version
authorship / member binding / nesting) with per-arc measurables (depth,
evidence counts) and orthogonal gap flags — no "strongest arc"
anywhere. Corollary: when a semantics upgrade converts candidates into
facts, audit any ranking built on the old uncertainty — it may now be
encoding nothing.

**Ruling 5 amended (Carl): shared traversal core from day one.** The
duplicate-code concern beats the refactor-risk concern because (a) the
walk core (inputs, producers, sentinel, cycle/cap) is identical between
lineage and closure — only arc selection and the accumulator differ,
and (b) the ~40 behavior-pinning lineage tests make the extraction safe
NOW; deferring means extracting later from two divergent walkers. One
arc-gated engine, two frontends; neither routine calls the other; no
cost inversion (engine walks only requested arcs); lookup_lineage's
contract byte-identical under refactored internals. Session evidence
for the drift risk: the _producer_of_dataset DRY finding and the
ordering-logic duplication both caught by review within hours of
creation.

**Ruling 6 (Carl, final): ancestry hops resolve at snapshots,
unbounded to source.** Each Dataset_Dataset hop reads the parent at the
child version's snaptime (chained points-in-time down the ancestry);
live-state hops rejected as the same causal error as live-state
feature scans. Snapshot-chain breaks (dev rows, pre-snapshot legacy) =
reported gaps, never live-state guesses. Depth: to SOURCE under the
global cap; a dedicated depth knob is YAGNI. This completed the
boundary interview — six rulings, all recorded on #383.

**Codex pre-merge round on #385 (PR #386): two P1s refined the
reachability semantics.** (1) Snapshot mode must bind DISCOVERY, not
just data — live-model path enumeration against a snapshot's data
silently omits post-snapshot-removed features and mis-joins changed
topology; version= now sources model+planner+find_features from the
snapshot catalog. (2) **Paths run to the feature's TARGET, and feature
tables are never reachability intermediates**: arriving at a feature
table via a value/asset FK would count bindings whose target member is
outside the dataset, and tunneling THROUGH a binding to its value
objects is not membership — reachability is over the object graph; the
target→feature hop is appended per feature. Also: schema-qualified
path keys (same-named tables in two domain schemas), and a hybrid
counting strategy (single route → server-side grouped distinct; only
multi-route features pay client-side RID-union). Live eye-ai results
unchanged by the rework — the fixes close pathological-schema holes,
not the common case.

**2026-08-31 — SVG tooltips must be CSS-drawn, not native `<title>`
(PR #387).** The lineage figure's hover tooltips were native SVG
`<title>` elements — structurally present on every node and correct in
a full browser, yet the user saw nothing: native tooltips are rendered
by the BROWSER CHROME, and embedded preview panes (Claude Code's side
panel, IDE webviews) never surface them. Lesson: "tooltip exists in
the DOM" and "tooltip the reviewer can see" are different claims, and
the test asserting `<title>` counts only pinned the first. The fix
draws overlays inside the SVG revealed by a pure-CSS `.hvN:hover~#ttN`
sibling rule — no JS, so self-containment holds; overlays appended
last for z-order and because `~` needs the target after the trigger.
Verification standard raised accordingly: hover + screenshot on the
live page (CSS overlays appear in screenshots; native tooltips never
did — which is also why the gap survived earlier "verify tooltips"
passes that only inspected markup).

**2026-08-31 addendum (same PR #387): keeping `<title>` "for
accessibility" alongside CSS overlays produces TWO tooltips.** In a
full browser the CSS overlay appears instantly and the browser chrome
then draws the native `<title>` tooltip on top a second later — two
different-looking tooltips for one element (user-reported). The
accessible text belongs in `aria-label` (with `role="img"`) on the
hover group: announced by screen readers, rendered by no browser.
Rule of thumb: `<title>` in SVG is a VISUAL feature (browser-chrome
tooltip), not merely a semantic one — if you draw your own tooltip,
`<title>` must go.

**2026-08-31 — lookup_provenance spec, Codex round 1: the snapshot-
strictness cluster (#383).** 17 P1s; 16 accepted. The load-bearing
catch: the draft spec would have violated ruling 6 AS WRITTEN through
three inherited behaviors — `_version_snapshot_catalog()` silently
falls back to the LIVE catalog on `Snapshot=NULL` version rows (spec
now requires a strict resolver that raises → `snapshot_chain_break`
gap); live `version_history` leaks post-vN authors into a vN closure
(authorship arc now bounded ≤ vN); and unpinned `Dataset_Execution`
inputs had no defined behavior (now quarantined: `unpinned_input` gap,
no snapshot-dependent expansion ever). Lesson: a ruling is only as
strong as the primitives that implement it — reusing an existing
helper imports its fallback semantics, so each ruling needs an audit
of every primitive on its path. Other accepted structure: ancestry is
dataset-discovery not an execution arc; datasets keyed per-version
(snapshot facts never conflated); assets are first-class members with
ALL producers followed (`_producer_of_asset` first-match is a bug in
scope); gap taxonomy grew to 12 kinds so nothing disappears silently;
`traversal_complete` ≠ gap-freedom. Partially rejected (1/17): the
named live artifact in the validation PLAN is documentation; the
committed form resolves it by lookup per the RID rule.

**2026-09-01 — #383 model idioms settled (Carl): StrEnum for closed
vocabularies; Pydantic public / dataclass internal.** Arc kinds, gap
kinds, input types, and ancestry states are `StrEnum`s (`ArcKind`,
`GapKind`, `ArcInputType`, `AncestryState`), not `Literal`s: consumers
DISPATCH on these (filter arcs, group gaps), so the vocabulary must be
importable and autocompletable with typos failing at authoring time —
the `ExecutionStatus` precedent, not the `RootDescriptor.type` one
(Literal is for read-and-display fields). Costs nil: StrEnum members
compare equal to their strings and dump as plain strings, so the JSON
envelope and string-literal comparisons in user code are unaffected.
Closure result classes are Pydantic per the CLAUDE.md class-idiom rule
(boundary-crossing, user-facing, all-Pydantic siblings); only
engine-private bookkeeping may be dataclasses. Extended on Carl's
follow-up ("more opportunities"): `RootType` replaces
`RootDescriptor.type`'s Literal (runtime-compatible — same string
values), and the engine's arc gate is `frozenset[ArcKind]`. The
STOPPING POINT matters as much as the rule: enums only for
vocabularies deriva-ml itself CLOSES — catalog-sourced open
vocabularies (status, element_type, asset_table) stay `str`, because
enum validation would reject older/foreign catalog values deriva-ml
doesn't control.

**2026-09-01 — #383 plan, Codex round: byte-compat vs new-semantics
splits are where plans rot.** 20 P1s on the implementation plan; 19
accepted, 1 amended. The recurring shape across the best findings: a
primitive serving BOTH the byte-frozen lineage frontend and the new
closure must split explicitly, never "improve" shared behavior —
(a) asset producer choice: lineage keeps FETCHED-first (sorted-first
would silently change output AND assign semantics to RID order,
violating our own RID rule); the closure sorts only at finalize;
(b) unpinned-input handling: lineage keeps its live-display fallback,
the closure quarantines — same engine, frontend-gated; (c) chunked
fetching: closure chunks, lineage keeps per-node fetches (documented
spec deviation). Also empirically verified by the reviewer: under our
non-strict VALIDATION_CONFIG, `Field(ge=1)` accepts `True` and `"2"` —
public int boundaries need `Field(strict=True, ge=1)`. And the
snapshot-strictness discipline extended again: truncating a LIVE
version-row list at vN is not snapshot-faithful (post-snapshot edits
leak); authorship facts read AT the strict snapshot like everything
else. Plan v2 adds a dedicated harness-extension task with seam-call
RECORDING (`ml.calls`) so quarantine tests can assert what was NOT
queried — absence-of-work claims need instrumentation, not hope.

**2026-09-01 — #383 Task 7: never infer snapshot-ness from the catalog-id
string.** The strict snapshot resolver first discriminated the NULL-snapshot
fallback via `"@" not in snapshot_id` — airtight-looking, but when the
Dataset's `_ml_instance` is ITSELF snapshot-bound its `catalog_id` is
already `"1@SNAP"`, so the bare-id branch composes a string containing "@"
and a NULL-snapshot row slips through as a live/mis-scoped read (and
`"1@SNAPA@SNAPA"` malformed ids). Reachable through the ancestry hop
chain's normal dev-row state (dev rows carry Snapshot=NULL). Fix:
discriminate on the RESOLVED VERSION RECORD's `.snapshot` value via a
shared `_resolve_version_record` helper, raising before any id is
composed. General rule: catalog-id strings are compositional (`id@snap`
nests); never parse them for semantics — read the source row.

**2026-09-01 — #383 live reconciliation: lookup_provenance vs the deploy
inventory (eye-ai).** Ran both closures against the SAME root, resolved
by catalog lookup at run time (workflow `VGG-19 Glaucoma Diagnosis
Training` + description `VGG-19 training on Kyle's full datasets` +
consuming exactly the Train/Validation/Test trio named in its own
description text + `Status=Uploaded` + newest RCT). `lookup_lineage`
reaches 2 executions on that root; `lookup_provenance` reaches 44 and
the deploy inventory 46 — the spec's 45/17/14 aggregate has drifted
slightly with catalog growth, which is why the reconciliation must run
BOTH sides live rather than compare against a remembered number.

| Measure | closure | deploy | difference |
|---|---|---|---|
| executions | 44 | 46 | 6 deploy-only, 4 closure-only |
| distinct workflows | 15 | 17 | 4 deploy-only, 2 closure-only |
| datasets | 12 | 14 | 5 deploy-only, 3 closure-only |
| assets | 1 | n/a | deploy has no asset domain |
| gaps | 142 (7 kinds) | 9 (2 kinds) | richer taxonomy |

Every difference is principled; none needed a code change:

- **Sentinel by identity, not name-matching (ruling 2).** Deploy counts
  the sentinel execution `6-0B3J` and workflows `6-0B3E`
  ('Unknown Provenance') / `6-FRFT` ('BlackBox unknown-asset producer')
  as closure MEMBERS. The closure never admits the sentinel: it
  terminates that branch with one `sentinel_origin` gap. So 1 execution
  + 2 workflows of the deploy-only delta are the sentinel triple, and
  deploy's 3 `sentinel` missing-arcs collapse into the closure's 1
  honest gap.
- **Snapshot-strict beats live-read (rulings 3 + 6) — the load-bearing
  one.** The remaining 4 deploy-only executions (`7-GDAA`, `7-GV6C`,
  `7-QCAA`, `7-VZQY`, workflows `7-GD96` / `7-ZX2J`) are exactly the 4
  rows by which `find_feature_producers('2-277G')` UNVERSIONED (14 rows,
  live) exceeds the same call at the walked pin v4.13.0 (10 rows). All
  four executions have RCT in 2026-07-08..07-18; the v4.13.0 version row
  has RCT 2026-07-02 — they bound their feature values AFTER the walked
  version's snapshot. Deploy reads live and picks them up; the closure
  reads at the pin and correctly does not. **The brief's spot-check
  ("the 14 rows should appear as member_binding evidence") is therefore
  satisfied at 10, not 14** — the correct assertion is
  `find_feature_producers(ds, version=walked_pin) ⊆ arc evidence`, and
  that holds exactly. Anyone re-running this must version-scope the
  spot-check or they will chase a phantom bug.
- **Closure-only executions = arcs deploy has no concept of.** `4-53ZE`
  (Fill diag exec_rid), `4-WRGW` (Cropping Image), `4-Z7XC`
  (Image_Grading), `5-278E` (Create Condition_Label) enter via
  `version_authorship` / `member_binding` on ancestry datasets deploy
  never walks (`2-N93J`, `4-S42W`, `4-Z6K8` — the 3 closure-only
  datasets). Deploy's 5 dataset-only rows (`4-4116`, `4-411G`,
  `5-WEBG`, `5-ZHRE`, `5-ZMGJ`) come from its Hydra `key=RID` config
  mining, which ruling 1 explicitly excludes as an arc.
- **Richer gap taxonomy (ruling 2).** Deploy reports 2 kinds; the
  closure reports 7 of the 12. Its 6 `null_feature_execution` map 1:1
  onto 6 of the closure's 7 `null_binding_execution` (closure adds
  `5-1W26`, a dataset deploy never reaches). The other 4 kinds are
  strictly new visibility deploy's heuristic never had: 109
  `no_version_author`, 10 `version_unresolvable`, 9
  `origin_unrecorded`, 1 `no_asset_producer` (`2-4JR6`). Gap COUNT going
  up is the feature, not a regression — `traversal_complete=True` and
  `cap_hit=False` throughout, confirming gaps are orthogonal to caps.

**Two schema-evolution bugs found and FIXED in this same PR.**
Old eye-ai snaptimes predate columns the current code assumes, and the
raw exception escapes `lookup_provenance` instead of becoming a gap:
(1) `core/mixins/dataset.py:200` — `lookup_dataset` unconditionally
filters on `Dataset.Deleted`, which does not exist at v0.1.0 snaptimes
→ `AttributeError` from datapath; `strict_parents_at` calls it AT the
historical snapshot, so this aborts the whole walk. (2)
`dataset/dataset.py:906` — `dataset_history` does `v["Snapshot"]`, and
`Dataset_Version` has no `Snapshot` column at v1.0.0–4.1.0 snaptimes →
`KeyError`; `strict_parents_at` catches only `DerivaMLException`, so it
escapes too. Repro without the closure:
`ml.lookup_dataset('2-277G').strict_parents_at('0.1.0')` and
`...strict_parents_at('1.0.0')`. Both are catchable into
`SnapshotUnavailable`, which the engine ALREADY converts into a
`snapshot_chain_break` gap. The fix wraps each snapshot-bound read at
the strict layer (scoped to `AttributeError`/`KeyError` at the point of
the read, never a blanket `except Exception`), naming the missing column
in the gap detail. Confirmed live with no monkeypatch: the walk
completes and produces precisely the 5 `snapshot_chain_break` gaps in
the table above (`2-277C/G/J/M`, `2-7KA2`, all at v0.1.0), with every
other number identical to the shimmed run. General
rule this reinforces: **a snapshot-strict walk must treat "the schema
itself differs at that snaptime" as a first-class gap source**, not
just missing rows — reading history means reading OLD SCHEMA, and every
column reference on a snapshot-bound path is a potential `AttributeError`.
The offline blind spot that let this ship is worth naming too: every
mocked harness presents CURRENT-schema tables, so "missing rows at a
snapshot" was well covered while "missing schema at a snapshot" had no
coverage at all. The harness now has a `set_schema_failure_at` seam and
the real `Dataset` is pinned directly in
`test_strict_snapshot.py::TestSnapshotSchemaEvolutionDegradesToGap`.

Timing on this root (www.eye-ai.org, warm): `lookup_lineage` 8.5s,
`lookup_provenance` 249s at the default `max_executions=500` with no cap
hit. The cost is dominated by per-version authorship + binding reads
across 31 visited dataset-versions, so it scales with ancestry breadth,
not execution count — budget minutes, not seconds, for real catalogs.

**2026-09-01 — Ruling 7 (Carl): fifth arc kind `member_production`
(#383 final review).** The closure inherited lineage's member-producer
fallback: executions that produced MEMBER ASSETS of a consumed dataset
are expanded, but the settled 4-kind typology had no slot for them —
they sat in closure.executions with empty arcs, breaking the model's
"arcs record every reason" promise. Carl ruled: add
`ArcKind.member_production` — "produced a member (asset) of the walked
dataset@version" — the asset analogue of member_binding under ruling
3's content reasoning (a dataset's content is members + bindings;
member producers are content contributors). Alternatives rejected:
arcs-may-be-empty (weakens the model), exclusion (closure would be
SMALLER than the lineage walk — contradicts its purpose). Attachment
point: the member relation at the walked (dataset, version); root-seed
member-fallback producers keep ArcKind.root (they are the seed, not a
mid-walk discovery). Lesson: a byte-frozen lift can carry BEHAVIOR into
a new semantic frame that the frame's typology never modeled — sweep
inherited behaviors against the new model's self-description.

**2026-09-01 — #383 Codex pre-merge round (PR #388): the pinned-root
path and three recurrences.** 2 P1 / 3 P2 / 1 P3, all accepted, fixed
in `1b603ae9`. (1) Pinned Dataset roots seeded/attributed from LIVE
classification — root attribution now snapshot-faithful (reuses the
authorship leg's snapshot rows, zero extra fetches); PARKED residual:
the walk seed itself is still live-derived — requires a post-hoc
rewrite of a version row's Execution FK to matter, which the writer
contract never does. (2) Recorded-but-UNREADABLE snapshots crashed the
member scan before the gap path — now degrade to snapshot_chain_break;
this moved the eye-ai reconciliation from 5 to **10 chain-break gaps
(142→147 total)**, paired 1:1 with the five already-broken v0.1.0
snapshots: pre-fix the member scans failed SILENTLY, so the closure
claimed member knowledge it did not have. 44 executions and
traversal_complete unchanged. (3) Recurrences of prior lessons caught
again by fresh review: live-model DISCOVERY under snapshot data reads
(the #385 P1 pattern, this time in _producers_of_dataset_members) and
recursion-depth ceilings (the lineage-HTML lesson, this time in
ancestry — now an explicit stack, mutant- and A/B-verified including
arc depths). Pattern worth naming: every snapshot-related fix this
cycle was the SAME rule — bind discovery, data, and attribution to one
snapshot handle, and turn every failure of that binding into a typed
gap.

**2026-09-01 — Confirmed (Carl's question): binding executions do not
leak sibling datasets into a closure.** Probe-verified on the harness:
if execution E bound features on dataset D and ALSO on sibling S, D's
closure contains E (member_binding arc on D) and E's own consumed
inputs (upstream), but S never enters — its binding scan is never even
queried. Structural reason: arc discovery runs dataset→executions and
execution expansion walks INPUTS only; nothing enumerates an
execution's other outputs or annotation targets. Forward reachability
("what else did E touch") is a different question
(find_executions_consuming), not provenance.

**2026-09-01 — lookup_provenance profiled on eye-ai: latency × request
count, NOT schema refetch.** 294s total; 61% (179s) in the 31 member-
binding scans; 2,703 sequential HTTP GETs at ~0.10s each account for
~all wall time. The multiplier is (dataset, version) PAIRS (31 for 12
datasets — ancestry walks each dataset at several snapshot-resolved
versions, each getting its own full #385 binding scan: FK-path walk +
one grouped query per feature-route). The schema-refetch hypothesis
(the estimate-perf trap) was WRONG here: 124 catalog_snapshot handles
cost only 7.8s — eye-ai serves schema cheaply. Levers, impact order:
(1) reuse FK-path discovery across versions of one dataset
(snapshot-checked); (2) batch/parallelize per-feature queries;
(3) expose the engine's arc gating publicly so callers can skip
binding scans for a fast structural pass. Batch-appropriate as-is;
optimize before any interactive use.

**2026-09-01 addendum — worked example: how "sibling-looking" datasets
enter a closure (eye-ai training run).** The 12 closure datasets
decompose as 3 consumed (G/39FY/M graded splits) + 4 via ancestry of
consumed (Test 2-277C, Validation 2-277J, Development 2-277E, LAC
Complete 2-1S12 — the graded subsets' PARENTS at their snaptimes,
verified against Dataset_Dataset edges) + 2 via consumption by
discovered annotator/grader executions (2-7KA2, 5-1W26) + 3 via THOSE
datasets' ancestry (4-Z6K8 → 4-S42W → 2-N93J). Nothing enters
sideways: what looks like a sibling split is either a subset-parent
(ancestry) or an input of a binding execution (upstream consumption).
Useful template for answering "why is dataset X in my provenance".

**2026-09-01 — Ruling 8 (Carl): ancestry expansion is OUT of the
provenance closure — provenance is execution-mediated.** Revisits
ruling 6 with live output in hand. Carl's framing: the execution
consumed three datasets at specific versions; other datasets sharing
members are structure, not provenance — the question is "what did this
execution use / what went into creating this artifact". Three
arguments, all accepted: (1) under proper capture the ancestry leg is
REDUNDANT — a split execution's consumption arc already records the
parent, causally and version-pinned; (2) walking Dataset_Dataset to
recover source linkage on legacy data is COMPENSATION, which ruling 2
forbids ("we should not worry about legacy at all here" — Carl);
(3) the containment edge does not reliably encode causal direction — a
collection dataset assembled FROM existing children inverts it.
Consequence: remove the ancestry leg from lookup_provenance (parents
enter only via a recorded execution's consumption); Dataset_Dataset
stays available through list_dataset_parents for structural questions.
Side effect: most of the 31 walked (dataset, version) pairs on eye-ai
were ancestry fan-out — this also addresses the 294s profile.

**2026-09-01 — Ruling 8 implemented (#389/PR): live confirmation on
eye-ai — the closure lost NOTHING, and got 48% faster.** Same root as
the reconciliation entry above (resolved by lookup at run time: workflow
`VGG-19 Glaucoma Diagnosis Training` + Kyle's-full-datasets description
+ `Status=Uploaded` + newest RCT → `7-H9ZT`, 18 candidates).

| Measure | before (ancestry in) | after (ruling 8) | delta |
|---|---|---|---|
| **executions** | 44 | **44** | **0** |
| walked (dataset, version) pairs | 31 | 19 | −12 |
| datasets | 12 | 9 | −3 |
| gaps | 142 | 97 | −45 |
| **runtime** | 294s | **153.4s** | **−48%** |
| traversal_complete / cap_hit | True / False | True / False | — |

**The headline is the zero.** Ruling 8's redundancy argument predicted
that a properly-captured split execution's consumption arc already
carries the parent — and the execution set is byte-for-byte unchanged.
Every execution the ancestry leg used to reach was ALREADY reachable
execution-mediated; ancestry was buying only extra *dataset-version*
fan-out, at 12 extra #385 binding scans. That is the empirical proof of
argument (1), not just its restatement.

The 3 dropped datasets are the ones the worked-example entry above
attributed to "ancestry of consumed" with no execution path
(2-277E Development, 2-1S12 LAC Complete, 4-S42W) — structure, now
answered by `list_dataset_parents` instead. The 9 that remain include
2-277C / 2-277J / 2-N93J / 4-Z6K8 / 5-1W26, i.e. former "ancestry-only"
datasets that ALSO have a real consumption arc: they were never
ancestry-dependent, which is why removing the leg did not lose them.

Gaps fell 142→97 purely by walking fewer pairs; the taxonomy is
unchanged (7 kinds, `snapshot_chain_break` still present at 5 — it now
comes from the authorship/binding snapshot legs and the member-scan
degrade, exactly as ruling 8 anticipated, so the GapKind is not
orphaned). `no_version_author` (74) still dominates: legacy version rows
without authors, reported not compensated (ruling 2).

Perf note, refining the 294s profile entry above: the remaining 153s is
still 19 sequential binding scans. Ancestry removal took the *pair
multiplier* down (31→19) but not the per-pair cost, so the levers named
there — reuse FK-path discovery across versions of one dataset, batch
per-feature queries, expose arc gating for a fast structural pass — are
all still on the table and are now the whole remaining story. Note four
datasets still walk at 3-4 versions each (2-277G, 2-277M, 2-39FY,
2-277J): multi-version fan-out survives ruling 8 because it is
execution-mediated (different executions consumed different pins), which
is exactly why the internal dataset budget is still meaningful.

**2026-09-01 — Ruling 9 (Carl): binding evidence is monotone across
dataset versions — scan once per dataset at the maximum walked
snaptime.** Carl, twice, as dataset SEMANTICS: "new dataset versions
will only add feature values, not remove them." Bindings at an older
version's snaptime are a subset of bindings at any newer version's
snaptime, so per-(dataset, version) binding scans are redundant: ONE
scan per dataset at the max walked snaptime subsumes every older
walked pin (19 scans → 9 on the eye-ai reference run, before other
optimizations). Evidence counts are reported "as of" that snapshot.
The one API that could violate monotonicity — delete_dataset_members
(curation, flips to dev per ADR-0003) — is governed by the ruling, not
an exception to it: a removal takes those members and their bindings
out of the dataset's story, and the newest-walked scan is the
authoritative view. Writers whose only bindings were on since-removed
members do not survive; that is the intended semantics, not an
approximation. Carl's sharper closing argument: from the PERSPECTIVE
of the specific execution or dataset version whose provenance was
requested, a member removed before the consumed version contributed
nothing to that artifact — its writers are simply not in this
closure's causal story; and a discovered execution's own full input
detail is ITS provenance, available via lookup_provenance(X). The
max-snaptime scan is therefore EXACT for the question asked. To be
codified in provenance-contract.md when implemented.

**2026-09-01 — Ruling 9 + #391 implemented (PR on feat/389-ancestry-out):
the closure is 39% faster and byte-identical, and the profile's centre of
gravity MOVED.** Four owner-approved changes, live-A/B'd on the eye-ai
reference training run (resolved by lookup at run time: workflow `VGG-19
Glaucoma Diagnosis Training` + `Status=Uploaded` + newest RCT → `7-ZYY8`,
28 candidates — note the RID and the candidate count both drift with
catalog growth, exactly as the earlier reconciliation entry warned, which
is why the A/B must re-resolve rather than pin).

| Measure | before (#389) | after (#391) | delta |
|---|---|---|---|
| **executions** | 44 | **44** (same RIDs) | **0** |
| **datasets** | 12 | **12** (same RIDs) | **0** |
| gaps / kinds | 99 / 7 kinds | **99 / same 7** | **0** |
| walked (dataset, version) pairs | 19 | 19 | 0 |
| binding SCANS | 19 | **12** | −7 |
| HTTP GETs | 1743 | **1397** | −20% |
| **runtime** | 150.9s | **92.0s** | **−39%** |
| structural pass (`arcs` w/o member_binding) | n/a | **7.2s / 89 GETs** | **21×** |

**The zero is again the headline.** Ruling 9 predicted the max-snaptime
scan is EXACT, not approximate, for the requested closure — and the
execution set, dataset set, gap count AND gap taxonomy are unchanged. The
7 collapsed scans (19→12) bought nothing but time, which is the whole
claim. No per-pin binding gap turned into a per-dataset as-of gap on this
run, because the multi-pin datasets' older pins had no bindings the newest
pin lacked — monotonicity holding in the data, not just in the contract.

**What each change actually bought** (measured, not assumed):

1. **C1 (ruling 9, max-snaptime scans)** — 19 scans → 12. Scans are now
   DEFERRED (`expand_dataset` registers the pin; a round-runner resolves
   the max via `_version_row_sort_key` over the live history) because
   which pin is the maximum is unknowable until the walk stops finding
   pins. Rounds alternate drain/scan; the run took **3 rounds** (3, 7, 2
   scans).
2. **C2 (route pruning)** — the real find, and NOT what the brief
   guessed. Every enumerated route starts `Dataset → Dataset_<Member> →
   …`: the first hop IS the membership edge, so a route whose first hop
   is empty for this dataset contributes nothing for ANY feature on that
   target. Probing each distinct first hop once per scan (cached, shared
   across features) took the reference dataset's scan from **45 GETs /
   5.74s to 18 / 3.22s with identical records**. On eye-ai's 8 features ×
   5 routes, **4 of 5 routes are empty** and 7 probes replace 32 route
   queries — and every feature collapses to ONE surviving route, i.e.
   onto the cheap server-side groupby, so the client-side multi-route
   union mostly stops running at all. The brief's proposed whole-table
   edge-fetch + in-memory BFS (the estimate-perf pattern) would have been
   WRONG here: the reachable `Image` set alone is 28,107 RIDs, so both a
   whole-table fetch and a RID-list filter are hazards. **Lesson: the
   estimate pattern transfers only where server joins are the slow part;
   here the cost was request COUNT, and the fix was to ask fewer
   questions, not to stop asking the server.**
3. **C3 (parallel scans)** — 105.1s of scan work compressed into **23.6s**
   of wall time (4.5× overlap, 8 workers). Workers read only; results
   apply single-threaded in sorted dataset order, so determinism is
   structural, not lucky (pinned by a test that inverts worker completion
   order and diffs `model_dump()`).
4. **C4 (public arc gating)** — a structural pass is **7.2s vs 92s**. It
   also revealed WHY the closure is expensive: gated to structure it finds
   **2 executions, not 44**. The binding leg is what DISCOVERS the other
   42.

**The ≤20s target was NOT met, and the profile now says why.** After the
scans stop dominating, the remaining ~70s is the **sequential execution
walk** of the 42 binding-discovered executions: 44 `lookup_execution`
(16.2s / 308 GETs), 31 `_producers_of_dataset_members` (15.8s / 189),
`_input_dataset_pairs` (5.2s / 96), plus their fan-out. Binding scans are
now only ~24s of the ~92s. So the next lever is NOT more scan
optimization — it is **batching or parallelizing execution expansion**
(the drain loop is strictly sequential and its per-node lookups are
independent), or snapshot-keyed result caching. Recorded here so the next
pass doesn't re-optimize the leg that is already fast: **the 2026-09-01
"levers, impact order" list above is now spent — items (1) and (2) are
done and item (3) is shipped as `arcs=`.**

**2026-09-01 — Authorship history stays (Carl), and the
worldline-vs-neighborhood distinction.** After ruling 8, Carl weighed
dropping previous-version authors; settled on KEEPING them: each
version-producing execution is a curation event on the artifact itself
— execution-mediated causation, unlike ancestry. The "versions span
snapshots" wrinkle resolves cleanly: the walk never visits old
snapshots — Dataset_Version rows ≤ v are DATA IN v's snapshot (the
history is carried forward as content), so one snapshot-closed read at
v yields the whole chain. The principled in/out line: Dataset_Version
is the WORLDLINE of one artifact (execution-attributed construction
events on the thing itself → provenance); Dataset_Dataset is a
RELATION between different artifacts with no reliable causal direction
(→ structure, ruling 8). Perf consequence: historical authors are not
trimmed for speed; the remaining runtime goes to parallel execution
expansion instead, and the #392 arc-gating knob serves anyone wanting
the narrow view.

**2026-09-01 — Parallel expansion: reuse the deriva-py asyncio
framework, not a third thread pool (Carl-approved design).** Codebase
audit corrected an earlier wrong claim ("deriva-py is sync all the way
down"): deriva-py ships `deriva.core.asyncio` — AsyncErmrestCatalog on
httpx (retries, connection limits), AsyncErmrestSnapshot, async
datapath, and clone.py's semaphore-bounded asyncio.gather fan-out
(used by the bag loader deriva-ml already calls). deriva-ml also has a
dormant notebook-safe `run_async` bridge (core/async_helpers.py) and
two ad-hoc ThreadPoolExecutor sites (_reachability, #392 scan pool).
Decision: orchestrate the closure's frontier expansion with the
deriva-py async pattern (gather + Semaphore), offload existing sync
mixin seams via executor (per-worker catalog handles — requests.Session
is NOT thread-safe; the #392 scans solved this with per-scan handles),
keep the public API sync via run_async, and migrate hot reads to
native async incrementally — converging on ONE concurrency framework
instead of three.

**2026-09-01 — Parallel expansion implemented (#391b): byte-identical,
147s → 99s, and the bottleneck moved AGAIN — to `expand_dataset`.** The
approved design shipped as specified: frontier rounds, `asyncio.gather`
under an `asyncio.Semaphore` (the deriva-py `clone.py` pattern), the
existing sync mixin seams offloaded via `loop.run_in_executor`, public
API kept sync through `run_async`, and per-worker `DerivaML` handles
(`_provenance_worker_handle`) because `requests.Session` is not
thread-safe. Live A/B on `lookup_provenance("7-ZW3P")`, www.eye-ai.org:

| Measure | before | after | delta |
|---|---|---|---|
| **executions** | 52 | **52** (same RIDs) | **0** |
| **datasets** | 15 | **15** (same RIDs) | **0** |
| assets | 1 | 1 | 0 |
| gap multiset | 129 | **129 (identical strings)** | **0** |
| full `model_dump` | — | **byte-identical, 118,155 B** | **0** |
| **runtime** | 147.3s | **99.2s** | **−33%** |

**The overlap is excellent; the coverage is not.** Instrumented:
**149.3s of `read_execution` work compressed into ~29.6s of wall time
(5.4× overlap)** across 48 of 52 reads. That leg is now essentially
solved. But the ≤50s target was still missed, and the profile says why:

- `expand_dataset` — **24.1s over 43 calls, fully sequential.** This is
  the strict-snapshot resolution per walked pin, and it is now the
  single largest un-parallelized leg. **Next lever.**
- binding scans — 26.6s (already pooled by #392; 4 rounds).
- inline reads — 16.0s over just **4** reads that missed the prefetch.

**The finding worth carrying forward: frontier WIDTH, not frontier
existence, is what pays.** Measured widths were `{1: 44 frontiers, 6: 1,
42: 1}`. One width-42 frontier (the seed set) did essentially all the
work; **44 of 46 frontiers were width 1** and correctly fell back to an
inline read. The walk's shape is one wide fan-out at the seeds and then
long serial chains — so "parallelize the expansion" bought a one-shot
33%, not a uniform speedup, and further gains must come from
parallelizing a *different* leg (`expand_dataset`), not from tuning
worker count. Raising `DERIVA_ML_PROVENANCE_WORKERS` above 8 cannot help
a width-1 frontier.

**A cap-honesty subtlety that cost a test rewrite.** "No over-fetch past
the cap" cannot mean "reads ≤ cap" for any read-AHEAD: a frontier's
siblings are fetched before the first sibling's own subtree has spent
its budget, so a capped walk reads a bounded handful it then declines to
expand (measured: 6 reads under a cap of 4, vs 4 sequential). What IS
guaranteed, and what the test now pins, is (a) no frontier exceeds the
remaining budget, (b) in-flight readouts are CHARGED against that budget
— without which nested frontiers each measure the budget as if the
others did not exist and the read-ahead runs away — and (c) the capped
closure is byte-identical to the capped sequential closure. Cap honesty
is about what ends up in the closure, never about how many rows were
read to decide that.

**Design note that made this cheap: split read from apply, don't
restructure the walk.** Rather than converting the recursive DFS into an
explicit BFS-by-frontier state machine, `expand_execution` was split
into `read_execution` (pure I/O, returns an immutable `ExecutionReadout`,
touches no engine or visitor state) and the unchanged apply half. The
prefetch is then a pure read-ahead that fills a cache the apply path
consumes; an empty cache degrades to the historical inline read. That is
why **zero existing tests needed editing** — 194 provenance/lineage
tests and 8/8 goldens passed untouched on the first run — and why
`DERIVA_ML_PROVENANCE_WORKERS=1` is a true sequential-equivalence
control rather than a second code path. Failures the read side hits are
CAPTURED into the readout and classified on the main thread
(`member_producers_from` mirrors `member_producers_or_gap`'s exception
taxonomy verbatim), so gap text, gap dedup and gap order are unchanged
by construction rather than by luck.

**2026-09-01 — Review correction to #391b: `hash(rid) % len(pool)` is
ROUTING, not isolation — and a byte-identical A/B cannot detect the
difference.** The first parallel-expansion build handed each read a
handle by hashing the RID into the pool. With `min(workers, len(rids))`
tasks over exactly `workers` handles, that collides constantly
(birthday paradox; ~100% at frontier width ≥ 8), and every collision
silently shares an unsynchronized `requests.Session` **and** the
unguarded `_snapshot_cache` dict on that `DerivaML`. Fixed by LEASING:
handles live in an `asyncio.Queue`, acquired before `run_in_executor`
and released in a `finally`, with the queue depth serving as the *only*
concurrency bound so "pool ≥ concurrency" is enforced rather than
asserted. The separate `Semaphore` was deleted — a second bound can
drift out of step with the pool and reintroduce sharing.

**The lesson that generalizes: a byte-identical live A/B is evidence
about SEMANTICS, not about TRANSPORT SAFETY.** The 118,155-byte
identical dump was produced by the racy build. It proves the apply side
is single-threaded; it says nothing about whether the reads underneath
were safe, because a session race can return a correct answer on any
given run. Thread-safety claims need a test that can observe sharing,
not a diff that can't.

**Corollary: a test harness that returns `self` as its "per-worker
handle" cannot detect handle sharing** — the pool becomes N references
to one object and every collision looks like ordinary reuse. The
harness now mints a DISTINCT probe per handle, each counting its own
occupancy and *holding* it for a configurable interval so concurrent
reads genuinely overlap. Mutate-and-revert is what proved the pin real:
the collision test fails against the hash-mod build with **33 recorded
concurrent entries** and passes against the lease. Two other pins in
that PR turned out to be similarly vacuous until mutation-tested — the
in-flight budget charge (deleting it passed the entire suite, because
the coarse `≤ 2*cap` bound never bit; and the obvious scenario ALSO
can't exercise it, since `_prefetchable` filters already-cached RIDs to
empty — you need each seed to have its OWN parents so a new frontier is
offered while other readouts are still unapplied), and read-failure
equivalence (needed pinning per exception KIND, including that
`RuntimeError` still PROPAGATES rather than being swallowed because it
happened on a worker). **Standing rule: for any concurrency or
budget-accounting invariant, mutate the implementation and confirm the
test fails before believing it.**

**And a bug the probe itself surfaced:** proxying handle attributes by
"is it callable" wrapped `_FakeML.model` — a `MagicMock`, hence
callable — so `ml.model.name_to_table(...)` raised `AttributeError`
inside the asset branch, which swallowed it as `resolution_failed` and
silently dropped every asset producer. Instrument an explicit seam
list, never a callable check, when proxying a duck-typed object.

**2026-09-01 — Codex stack review round 2: ruling 9 had a latent
correctness hole, and the second concurrency leg had the same session
bug.** Four findings, all accepted (PR #393).

**The one that matters: "scan once per dataset" was implemented as
"scan once per dataset EVER".** `_binding_scanned` was a set, so once a
dataset was scanned it never scanned again — but a later round can walk
a HIGHER pin (a scan-discovered execution's inputs walk D@v2 above the
already-scanned v1). Ruling 9 promises evidence as of the MAXIMUM
walked snaptime; the implementation delivered evidence as of the
*first-scanned* snaptime, silently dropping bindings that exist only at
the higher pin. Fixed by tracking the scanned VERSION per dataset and
rescanning when the max pin advances (compared with `max_walked_pin`'s
own total order, so "newer" has one definition). The rescan REPLACES
the prior arcs rather than merging: monotonicity makes the new view a
superset, so a merge would duplicate every surviving record and leave
one dataset's arcs carrying two different `input_version` labels —
"evidence as of one snaptime" broken. Convergence is free: pins only
advance, bounded by version count.

**The eye-ai A/B could not have caught it — and said so only when
asked.** Post-fix run was byte-identical (52/15/1/129, 178 binding
evidence records unchanged). Instrumenting the run explained why: **15
datasets, 15 scans, ZERO rescans** — no dataset's max pin advances
after its first scan on that catalog, so the bug is latent for that
root. Recording this because the null result is a trap: identical bytes
after a correctness fix look like "the fix was unnecessary" and are
actually "this root doesn't reach the broken path." Always instrument
whether the fixed path was EXERCISED before reading a null A/B as
reassurance.

**Second finding, same shape as the round-1 lease bug:** the #392
binding-scan `ThreadPoolExecutor` called seams on the shared `self.ml`
concurrently — the identical unsynchronized `requests.Session` +
`_snapshot_cache` hazard the expansion lease had just fixed, sitting
untouched on the other leg. Both legs now draw from ONE handle queue
(`_run_leased`), which makes pool depth the single global concurrency
bound and retires the "two concurrency frameworks" item. **Lesson: when
you fix a concurrency bug, grep for every other pool in the same
module** — the fix does not generalize itself.

**Harness lesson (again): a probe can only see what routes through
it.** The first scan-lease test PASSED against a deliberately-broken
mutant, because a worker bypassing the lease calls the seam on the
shared `_FakeML` where no `_HandleProbe` exists. Needed a separate
"direct concurrent use of the shared instance" tracker before the pin
was real (then: 23 recorded violations against the mutant). Corollary
to the round-1 rule: mutation-testing a concurrency pin has to confirm
the harness can OBSERVE the mutant, not merely that the test fails for
some reason.

**Also: `reuse_schema_json` never pinned anything** — `_init_online`
re-fetches `/schema` to validate it and replaces on difference. Right
for `catalog_snapshot` (a pre-migration snapshot must not build a model
its catalog cannot serve), wrong for worker handles on the same live
catalog, which each re-fetched and could diverge from the caller's
model mid-walk (one walk, two model identities). Now an explicit
`trust_schema_json` flag used only by the worker-handle factory;
default stays validating.

**2026-09-01 — Round 3: the rescan fix was half-done — replacing arcs
without replacing GAPS just moves a stale-view bug into the gap
stream.** Ruling 9's "one as-of view per dataset" has to hold across
*every* accumulator the scan writes to, not just the one you were
thinking about. Two reachable cases: a binding-leg `sentinel_origin`
whose detail embedded `{dataset}@{version}` emitted TWO gaps for one
fact on the ordinary MONOTONE rescan (the two-version-labels violation
the arc fix's own docstring named — the fix documented the rule and
then broke it one accumulator over); and a transient v1
`binding_scan_failed` survived a clean v2 rescan, permanently reporting
a failure that no longer applied.

**Two complementary fixes, and it is worth knowing they are not
redundant:** (1) tag gaps with the dataset whose scan emitted them and
drop them on rescan — *releasing the dedup keys too*, or the fresh
scan's identical re-emission is swallowed as "already seen" and the gap
vanishes entirely, turning one bug into a worse one; (2) stop embedding
the scanned pin in details that state version-INDEPENDENT facts. Each
alone fixes the sentinel case; only (1) fixes the transient-failure
case; (2) makes the gap dedupe naturally before any drop runs. **General
rule: put the as-of label on the ARCS (which are the as-of view) and
keep it out of gap details (which state facts).**

**A test can pin the defect.** `test_binding_sentinel_execution_emits_gap…`
asserted `{dataset}@{version}` appeared in the sentinel detail — it had
been written against the buggy behaviour and was actively defending it.
When a fix makes an existing assertion fail, check whether the
assertion was ever *right* before changing the code back.

**And: "unified" claims need reading twice.** The round-2 commit said
both legs run through one `_run_leased`; they actually shared a handle
POOL while duplicating the queue/acquire/release loop. Genuinely
unified now (`_gather_leased` is the single primitive, `_run_leased`
its sync wrapper), with the per-task exception difference pushed into
each caller's worker so the primitive stays exception-neutral and one
task's failure cannot cancel the gather.

**2026-09-02 — Live proof of the rescan, and the environment blocker
that stopped it landing green.** Wrote
`tests/execution/test_lookup_provenance_live.py`: the pin-advance
scenario built on a real catalog rather than `_FakeML` — D over three
Subject members; E1 binds a Quality feature; D released at v1 (snapshot
cut AFTER E1's bindings, so a v1 scan sees E1 and nothing later); E2
binds a SECOND, distinct feature; D released at v2; then the load-
bearing edge, `e1.add_input_dataset(D, version=v2)`, which is what makes
the maximum walked pin advance in a round *after* D was first scanned.
Root R consumes D@v1. Five tests: exactly two scans of D (v1 then v2),
all surviving `member_binding` arcs carry v2, E1 holds exactly ONE arc
(replaced not merged), E2 present, and no gap detail carries the
superseded v1 label.

**Two construction notes worth keeping.** (1) The pin advance needs a
PUBLIC API, and there is one — `Execution.add_input_dataset(rid,
version=)` writes the `Dataset_Execution` row with the `Dataset_Version`
FK resolved, so no hand-rolled association insert is needed; reaching
for a raw insert here would have been a hard-coded-schema smell. (2)
Use TWO distinct features, not two batches of one feature: E2's
bindings have to be invisible at v1 for the control assertion
(`find_feature_producers(D, version=v1)` lacks E2 while the closure
contains it) to mean anything. Scan counting is a `monkeypatch` wrapper
on `_find_feature_producers_impl` at the CLASS level — the engine leases
separate handles, so patching the instance would miss the worker calls.

**Blocker: the localhost demo stack's bearer token is expired and cannot
be refreshed non-interactively.** `POST /ermrest/catalog` → 401 ("Access
requires authentication"), while GETs still return 200 — exactly the
trap the `demo-catalog-auth-and-build` memory warns about: **read
success does not prove the token is good**, because anonymous read is
allowed. Confirmed environmental, not code: the pre-existing
`test_lookup_lineage_live.py` fails identically, and the 17 errors in
`test_provenance_contract.py` reproduce on a clean tree with my file
stashed. Also worth recording for next time: this stack now fronts
**Keycloak** (`/authn/login` → `/auth/realms/deriva/...`), not Globus —
there is no `/authn/session` endpoint at all (404), `/authn/discovery`
returns `{}`, and the realm does advertise the `password` grant, so a
refresh is *mechanically* possible but needs a human's realm
credentials. Offline gates are green (224 passed incl. the 150-test
provenance unit suite); the live module skips cleanly without
`DERIVA_HOST` and is ready to run the moment a token is minted.

**2026-09-02 — Live rescan proof OBTAINED (postscript to the Keycloak
blocker entry).** After Carl refreshed the localhost credential, the
committed live module ran green twice (5/5, ~80s each, idempotent):
the demo-catalog pin-advance scenario fired exactly 2 binding scans of
the dataset, the final closure's member_binding arcs all carry the v2
label (v1 view replaced), the v2-only binder E2 is present, no
binding gap carries a stale version label, and the live control
(find_feature_producers at v1 lacks E2) proves E2 was genuinely
invisible at the old pin. The ruling-9 rescan machinery is now
evidence-complete: mutation-pinned offline AND exercised live.

**2026-09-02 — Perf round 2 approved (Carl): batched frontier reads +
pooled dataset legs; guiding principle "anything that reduces the
number of queries is a win."** Levers chosen from the post-#393
profile (~100s on the reference root: ~70s width-1 dependency chains,
~24s sequential dataset legs): (1) per-frontier chunked RID=any()
queries per TABLE (the tk-023 chunked-summary pattern) replacing
per-node reads — 42-wide frontier: ~150 requests → a handful; (4)
expand_dataset snapshot/authorship legs routed through the leased
pool rounds like the binding scans. Deferred by choice: intra-read
concurrency (2) and speculative prefetch from authorship rows (3) —
revisit only if the measured result still disappoints. Target
~35-45s cold; floor is chain-depth × RTT without speculation.

**2026-09-02 — Perf round 2 implemented (#394, PR): the closure issues
HALF the requests and is byte-identical — and the live A/B earned its
keep twice.** Both approved levers shipped. (1) Frontier reads are
BATCHED per TABLE: five new `ExecutionMixin` seams
(`_execution_records_batch`, `_input_dataset_pairs_batch`,
`_input_assets_batch`, `_producer_of_datasets_batch`,
`_producers_of_assets_batch`) answer each question once for the whole
frontier with chunked `RID=any(...)` queries (tk-023 chunk 25), and
`WalkEngine.read_frontier` assembles the SAME `ExecutionReadout` the
per-node path produced — so the apply path never changed and every
semantic pin over it stayed valid. (2) `expand_dataset`'s legs
(strict-snapshot resolve + `_dataset_version_rows_at` + author
summaries) run in a pooled round through `_gather_leased`.

| Measure | main | #394 | delta |
|---|---|---|---|
| **executions / datasets / assets / gaps** | 52 / 15 / 1 / 129 | **identical RIDs** | **0** |
| **canonical `model_dump`** | 118,155 B `806ad64f…` | **byte-identical** | **0** |
| **HTTP requests (full closure)** | 1,882 | **898** | **−52%** |
| runtime (full closure) | 100.0s | **81.1s median** (3 runs) | **−19%** |

**The request cut is nearly triple the runtime cut, and that IS the
result.** Halving requests bought ~19% wall clock, which says the walk has
moved off request COUNT and onto per-request latency down the
dependency chain — exactly the "chain-depth × RTT without speculation"
floor the scope note predicted. The remaining levers are the deferred
ones (intra-read concurrency, speculative prefetch), not more batching.

**Finding 1 — a closure divergence the offline suite could not
produce.** The first A/B came back 2 assets / 130 gaps against main's
1 / 129. The extra was `File` row `6-0B3G`: `File` carries a
`File_Execution` association (source files registered BY REFERENCE via
`add_files`) so `find_asset_execution_tables()` returns it, but
`model.is_asset(File)` is **False** — so `lookup_asset` refuses it
("RID X is not an asset") and the per-node `list_assets` path drops the
row with a debug log. The batched reader never calls `lookup_asset`, so
it had no equivalent guard. **The harness cannot express this case at
all** — `_FakeML.add_asset` registers the table as an asset by
construction — so a green offline suite was not evidence, and the pin
had to go at the seam with a hand-built model stub. Whether `File` rows
*should* be closure assets is a real question with a defensible "yes";
it is a semantics change and belongs in its own issue, not in a perf PR
whose acceptance criterion is a byte-identical closure. **General rule:
when a batched reader replaces a per-node one, enumerate the per-node
path's SILENT DEGRADES — they are behavior, and the batch inherits none
of them for free.**

**Finding 2 — "fewer requests" is not "faster", and concurrency can be
NEGATIVE on expensive joins.** The narrow structural pass
(`arcs=` without `member_binding`) got *slower* on the branch at
unchanged request count (93 → 95). Per-URL profiling of both trees put
the entire difference in three `Image_Execution …/Asset_Role=Output/
Execution` member-producer joins — the same three queries on both:
**main 3 × 2,140ms sequential (6.4s); branch 3 × 5,728ms concurrent
(17.2s of work, ~11s wall)**. The server does not parallelize them;
under contention each becomes ~2.7× slower, so pooling a leg of three
very expensive joins converts 6.4s of sequential work into ~11s of
contended wall time. Everything else on the branch got cheaper (~7s →
~4s), and on the profiled pair the branch still won overall (11.3s vs
13.6s) — but these are the highest-variance queries on the catalog
(2.1-5.7s each), which is why one A/B pair read 5.4s vs 12.8s and
another read 13.6s vs 11.3s. **Corollary to the #391 C3 result: pooling
paid 4.5× for the BINDING scans and is at best neutral for the
member-producer scans on a narrow round. "Independent reads" is not
sufficient justification for concurrency — the server-side cost of the
individual query decides, and a leg of few-but-huge joins wants
sequencing, not a pool.** Recorded rather than fixed: the full closure
is unaffected (984 requests saved dominate) and the closure is
byte-identical either way.

**Resolved in-PR after 5-run medians made it undeniable (5.5s main vs
12.1s branch, 93 vs 95 requests — not variance).** The member-scan
round now pools only at `_MEMBER_SCAN_POOL_THRESHOLD` (8, the default
worker count) or above; below it the scans run sequentially on the
caller's handle, which is also what the pre-#394 walk did for a width-1
frontier. Structural pass went **12.1s -> 6.4s median** (main 5.5s);
the wide case, where dedup and overlap both pay, is untouched. The
generalizable rule: **a pooling decision is per-LEG and needs its own
measurement — "these reads are independent" justifies correctness, not
speed.**

**Finding 3 — a handle is a whole connection, so size the pool to
demand.** Chasing finding 2 surfaced a real (if smaller) cost:
`_worker_handles` eagerly built the full worker count on first use, so
#394's batching — which shrank the narrow pass's rounds to a handful of
items — meant 8 `DerivaML` constructions to serve 3 dataset legs. Now
grown lazily to `min(workers, needed)`, reused across rounds, with a
latch so a refusing factory is asked once. Did not explain finding 2,
but worth keeping on its own.

**Harness lesson (third time on this stack): a probe can only see what
routes through it — including what it HANDS OUT.** Two of the new
concurrency pins were vacuous until the harness was extended. The
member-scan seam needed `_note_direct_seam_use` (mutant then showed 23
unleased concurrent uses). The dataset leg was worse: its snapshot read
is reached *through* the object `lookup_dataset` returns, so the leased
handle had to TRAVEL WITH that object (`_ProbedDatasetHandle`) or the
read escaped every probe — with that, the mutant showed 47 violations;
without it, zero. Also: the pre-existing wide-frontier lease pin went
vacuous the moment the row reads were batched onto the main thread (a
parentless wide frontier now issues no concurrent read at all), so it
had to move to a scenario with dataset inputs. **When you change WHICH
leg is concurrent, re-check every concurrency pin for vacuity — a
passing test on a leg that no longer runs concurrently proves nothing.**

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

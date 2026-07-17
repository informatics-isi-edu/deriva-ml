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

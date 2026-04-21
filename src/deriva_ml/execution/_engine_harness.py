"""Documentation-only module: the Phase 1 engine-testing harness.

During Phase 1 the upload engine is built against
ExecutionStateStore.insert_pending_row directly. The full
user-facing TableHandle.insert path (spec §2.10) is Phase 2 —
finalized after the feature-consistency review.

Engine tests in G3–G8 intentionally bypass the handle API so that
the engine can be proven correct independently of the handle design.
The handle API, once final, will call insert_pending_row identically
to the harness — so engine tests transfer unchanged.
"""

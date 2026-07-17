"""Live-catalog contract test for the RID-lease idempotency guarantee.

The reconcile-based retry in :func:`deriva_ml.execution.rid_lease.post_lease_batch`
(issue #360) depends on an ERMrest guarantee about
``public:ERMrest_RID_Lease``:

    The table has a composite unique key ``(RCB, ID)``, where ``RCB``
    is the row-created-by (authenticated client) and ``ID`` is our
    client-generated UUID4 lease token.

That key is what makes a lease POST *idempotent per client*: a
partially-landed POST cannot be blindly re-sent (it would 409 on the
tokens that already succeeded), so the retry reconciles by querying
which tokens already exist and re-POSTs only the missing ones.

Because we chose to keep the fix downstream (no deriva-py pin for this
contract — see ``tacit-knowledge.md`` 2026-07-17), these tests are our
CI's guard: if a future server/deriva-py change drops the ``(RCB, ID)``
unique key or changes the duplicate-POST behavior, the reconcile logic
must be revisited and these tests fail loudly.
"""

from __future__ import annotations

import os

import pytest
import requests

requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lease idempotency contract tests require DERIVA_HOST",
)


@requires_catalog
def test_lease_table_has_composite_rcb_id_unique_key(test_ml):
    """ERMrest_RID_Lease must carry a unique key on (RCB, ID).

    ``ID`` alone must NOT be the unique key — the guarantee is
    per-client. This pins the exact model fact the reconcile-retry
    relies on.
    """
    model = test_ml.catalog.getCatalogModel()
    lease = model.schemas["public"].tables["ERMrest_RID_Lease"]

    unique_col_sets = {frozenset(c.name for c in key.unique_columns) for key in lease.keys}
    assert frozenset({"RCB", "ID"}) in unique_col_sets, (
        "ERMrest_RID_Lease lost its (RCB, ID) composite unique key — "
        "the reconcile-based lease retry (issue #360) is no longer safe. "
        f"Found unique keys: {sorted(sorted(s) for s in unique_col_sets)}"
    )
    assert frozenset({"ID"}) not in unique_col_sets, (
        "ERMrest_RID_Lease.ID became unique on its own; the reconcile "
        "logic assumes ID is unique only within a client (RCB, ID)."
    )


@requires_catalog
def test_duplicate_lease_post_conflicts(test_ml):
    """Re-POSTing the same token (same client) conflicts.

    Proves the idempotency mechanism the retry defends against: a
    naive blind re-POST is unsafe, which is *why* the retry reconciles.
    """
    from deriva_ml.execution.rid_lease import generate_lease_token

    token = generate_lease_token()
    body = [{"ID": token}]
    # First POST lands.
    first = test_ml.catalog.post("/entity/public:ERMrest_RID_Lease", json=body).json()
    assert len(first) == 1
    # Second identical POST (same RCB, same ID) must conflict.
    with pytest.raises(requests.HTTPError) as ei:
        test_ml.catalog.post("/entity/public:ERMrest_RID_Lease", json=body)
    assert ei.value.response is not None
    assert ei.value.response.status_code == 409


@requires_catalog
def test_reconcile_recovers_leased_rid_after_landed_post(test_ml):
    """A landed POST is fully recoverable by the reconcile query.

    Simulates the lost-response case: the POST succeeded server-side
    but the client "didn't see" the response. ``_fetch_landed_leases``
    must return the token→RID mapping so the retry does not re-POST.
    """
    from deriva_ml.execution.rid_lease import (
        _client_rcb,
        _fetch_landed_leases,
        generate_lease_token,
    )

    token = generate_lease_token()
    posted = test_ml.catalog.post("/entity/public:ERMrest_RID_Lease", json=[{"ID": token}]).json()
    assert len(posted) == 1
    leased_rid = posted[0]["RID"]

    rcb = _client_rcb(test_ml.catalog)
    recovered = _fetch_landed_leases(test_ml.catalog, [token], rcb=rcb)
    assert recovered.get(token) == leased_rid, (
        "reconcile query failed to recover a landed lease token → RID; the retry would incorrectly re-POST it."
    )


@requires_catalog
def test_post_lease_batch_is_idempotent_end_to_end(test_ml):
    """Calling post_lease_batch twice with the same tokens is safe.

    Second call reconciles the already-landed tokens instead of
    409-ing, returning the same RIDs — the end-to-end property the
    retry provides.
    """
    from deriva_ml.execution.rid_lease import (
        generate_lease_token,
        post_lease_batch,
    )

    tokens = [generate_lease_token() for _ in range(3)]
    first = post_lease_batch(catalog=test_ml.catalog, tokens=tokens)
    assert set(first.keys()) == set(tokens)

    # Second call with the same tokens: a blind re-POST would 409, but
    # the reconcile path recovers every RID. Force the reconcile branch
    # by clearing nothing — post_lease_batch will POST, hit 409 (a
    # transient status), then reconcile and return the same RIDs.
    second = post_lease_batch(catalog=test_ml.catalog, tokens=tokens)
    assert second == first, (
        "post_lease_batch was not idempotent across repeated calls with "
        "the same tokens; reconcile-retry did not recover the RIDs."
    )

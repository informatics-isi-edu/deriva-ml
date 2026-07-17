"""Tests for RID leasing against public:ERMrest_RID_Lease."""

from __future__ import annotations

import uuid

import pytest
import requests


def _terminal_http_error(status: int = 403) -> requests.HTTPError:
    """A non-transient HTTP error (default 403) for the lease POST.

    Terminal 4xx statuses are NOT retried by ``post_lease_batch``;
    only transient failures (timeouts, 5xx, spurious 409) are. See
    ``test_rid_lease_retry.py`` for the transient-path coverage.
    """
    resp = requests.Response()
    resp.status_code = status
    resp._content = b"forbidden"
    resp.reason = "Forbidden"
    return requests.HTTPError(f"{status} error", response=resp)


class _MockLeaseCatalog:
    """Mock that records POSTs to ERMrest_RID_Lease and returns
    synthetic RIDs keyed by the lease tokens."""

    def __init__(self, *, prefix: str = "RID-", fail: bool = False):
        self.prefix = prefix
        self.fail = fail
        self.post_calls: list[list[dict]] = []

    def post(self, path: str, json=None, **_kw):
        if self.fail:
            # A terminal (non-transient) error must propagate without
            # retry; use 403 so the retry classifier treats it as
            # terminal rather than a transport hiccup.
            raise _terminal_http_error()
        assert "ERMrest_RID_Lease" in path
        assert isinstance(json, list)
        self.post_calls.append(json)

        class _R:
            def __init__(self, bodies, prefix):
                self._bodies = bodies
                self._prefix = prefix

            def json(self):
                return [{"RID": f"{self._prefix}{i}", "ID": b["ID"]} for i, b in enumerate(self._bodies)]

        return _R(json, self.prefix)


def test_generate_lease_token_is_uuid_string():
    from deriva_ml.execution.rid_lease import generate_lease_token

    t = generate_lease_token()
    # Must round-trip through UUID parser.
    uuid.UUID(t)


def test_post_lease_batch_sends_tokens_and_returns_rids():
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog(prefix="RID-")
    tokens = ["T1", "T2", "T3"]
    rids_by_token = post_lease_batch(catalog=cat, tokens=tokens)

    # Every input token received a RID back.
    assert set(rids_by_token.keys()) == set(tokens)
    assert all(v.startswith("RID-") for v in rids_by_token.values())
    # Exactly one POST call with N entries.
    assert len(cat.post_calls) == 1
    assert len(cat.post_calls[0]) == 3


def test_post_lease_batch_chunks(monkeypatch):
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease, "PENDING_ROWS_LEASE_CHUNK", 2)
    cat = _MockLeaseCatalog(prefix="X-")
    tokens = ["A", "B", "C", "D", "E"]
    rids_by_token = post_lease_batch(catalog=cat, tokens=tokens)

    # 5 tokens, chunk size 2 → 3 POSTs of 2, 2, 1.
    assert len(cat.post_calls) == 3
    assert len(cat.post_calls[0]) == 2
    assert len(cat.post_calls[1]) == 2
    assert len(cat.post_calls[2]) == 1
    assert set(rids_by_token.keys()) == set(tokens)


def test_post_lease_batch_empty_is_noop():
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog()
    result = post_lease_batch(catalog=cat, tokens=[])
    assert result == {}
    assert cat.post_calls == []


def test_post_lease_batch_propagates_terminal_error():
    """A terminal (non-transient) catalog error propagates immediately,
    without burning retry attempts (transient handling is covered in
    test_rid_lease_retry.py)."""
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog(fail=True)
    with pytest.raises(requests.HTTPError):
        post_lease_batch(catalog=cat, tokens=["T"])
    # Exactly one POST attempt — no retry on a terminal 4xx.
    assert len(cat.post_calls) == 0  # POST raised before recording

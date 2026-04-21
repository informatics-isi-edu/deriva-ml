"""Tests for RID leasing against public:ERMrest_RID_Lease."""

from __future__ import annotations

import uuid

import pytest


class _MockLeaseCatalog:
    """Mock that records POSTs to ERMrest_RID_Lease and returns
    synthetic RIDs keyed by the lease tokens."""
    def __init__(self, *, prefix: str = "RID-", fail: bool = False):
        self.prefix = prefix
        self.fail = fail
        self.post_calls: list[list[dict]] = []

    def post(self, path: str, json=None, **_kw):
        if self.fail:
            raise RuntimeError("simulated lease failure")
        assert "ERMrest_RID_Lease" in path
        assert isinstance(json, list)
        self.post_calls.append(json)
        class _R:
            def __init__(self, bodies, prefix):
                self._bodies = bodies
                self._prefix = prefix
            def json(self):
                return [
                    {"RID": f"{self._prefix}{i}", "ID": b["ID"]}
                    for i, b in enumerate(self._bodies)
                ]
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


def test_post_lease_batch_propagates_catalog_error():
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog(fail=True)
    with pytest.raises(RuntimeError):
        post_lease_batch(catalog=cat, tokens=["T"])

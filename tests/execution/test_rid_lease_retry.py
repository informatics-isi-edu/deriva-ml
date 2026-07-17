"""Tests for reconcile-based retry in post_lease_batch (issue #360).

A transient failure on the lease POST (timeout, 5xx, or a spurious
409) must not fail an otherwise-complete commit. The retry is
*reconcile-based*: because ``ERMrest_RID_Lease`` has a composite
unique key ``(RCB, ID)`` and ``ID`` is our client-generated UUID4
token, a partially-landed POST cannot simply be re-sent (it would
409 on the tokens that already succeeded). Instead the retry queries
which of our tokens already exist and re-POSTs only the missing ones.

See ``tacit-knowledge.md`` (2026-07-17) and the module docstring of
``deriva_ml.execution.rid_lease`` for the contract.
"""

from __future__ import annotations

import pytest
import requests


def _http_error(status: int, detail: str = "") -> requests.HTTPError:
    """Build a requests.HTTPError carrying a Response with ``status``."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = detail.encode()
    resp.reason = detail
    return requests.HTTPError(f"{status} error: {detail}", response=resp)


class _ReconcileCatalog:
    """Mock catalog that drives the reconcile-retry path.

    - ``post`` fails for the first ``fail_times`` calls with the
      configured exception, then succeeds. On success it records
      each row it "persists" (keyed by token ID) so the reconcile
      query can report already-landed tokens.
    - ``fail_after_persist``: when True, the failing POST persists
      its rows on the server *before* raising — modelling the
      lost-response case where work landed but the client saw an
      error. The reconcile query then finds those tokens.
    - ``getPathBuilder`` exposes an ``ERMrest_RID_Lease`` table whose
      ``ID.in_(chunk)`` fetch returns the persisted (token, RID)
      rows intersecting the chunk.
    """

    def __init__(
        self,
        *,
        fail_times: int = 0,
        exc: Exception | None = None,
        fail_after_persist: bool = False,
        prefix: str = "RID-",
    ):
        self.fail_times = fail_times
        self.exc = exc or _http_error(409, "Schema public does not exist")
        self.fail_after_persist = fail_after_persist
        self.prefix = prefix
        self.post_calls: list[list[dict]] = []
        self._persisted: dict[str, str] = {}  # token ID -> RID
        self._counter = 0
        self._reconcile_chunks: list[list[str]] = []

    # --- POST -----------------------------------------------------
    def _assign(self, tokens: list[str]) -> None:
        for t in tokens:
            if t not in self._persisted:
                self._persisted[t] = f"{self.prefix}{self._counter}"
                self._counter += 1

    def post(self, path: str, json=None, **_kw):
        assert "ERMrest_RID_Lease" in path
        assert isinstance(json, list)
        self.post_calls.append(json)
        tokens = [b["ID"] for b in json]

        if len(self.post_calls) <= self.fail_times:
            if self.fail_after_persist:
                self._assign(tokens)
            raise self.exc

        self._assign(tokens)

        class _R:
            def __init__(self, rows):
                self._rows = rows

            def json(self):
                return self._rows

        return _R([{"RID": self._persisted[t], "ID": t} for t in tokens])

    # --- reconcile query -----------------------------------------
    def getPathBuilder(self):
        outer = self

        class _InPredicate:
            def __init__(self, chunk):
                self.chunk = list(chunk)

        class _Col:
            def in_(self, chunk):
                return _InPredicate(chunk)

        class _Chain:
            def __init__(self, chunk):
                self.chunk = chunk

            def attributes(self, *_cols):
                return self

            def fetch(self):
                outer._reconcile_chunks.append(list(self.chunk))
                return [{"RID": outer._persisted[t], "ID": t} for t in self.chunk if t in outer._persisted]

        class _Table:
            RID = _Col()
            ID = _Col()

            def filter(self, pred: _InPredicate):
                return _Chain(pred.chunk)

        class _Schemas:
            def __getitem__(self, _name):
                class _S:
                    tables = {"ERMrest_RID_Lease": _Table()}

                return _S()

        class _PB:
            schemas = _Schemas()

        return _PB()


def test_transient_409_retries_and_succeeds(monkeypatch):
    """A single transient 409 is retried; the batch ultimately succeeds."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    # No real sleeping in tests.
    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)

    cat = _ReconcileCatalog(fail_times=1)  # first POST 409s, second ok
    tokens = ["T1", "T2", "T3"]
    result = post_lease_batch(catalog=cat, tokens=tokens)

    assert set(result.keys()) == set(tokens)
    assert len(cat.post_calls) == 2  # one failed, one retried


def test_transient_5xx_retries_and_succeeds(monkeypatch):
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    cat = _ReconcileCatalog(fail_times=1, exc=_http_error(503, "unavailable"))
    result = post_lease_batch(catalog=cat, tokens=["A", "B"])
    assert set(result.keys()) == {"A", "B"}
    assert len(cat.post_calls) == 2


def test_lost_response_reconciles_without_duplicate_post(monkeypatch):
    """If a POST landed on the server but the response was lost, the
    retry must NOT re-POST the already-landed tokens (that would 409
    on the composite (RCB, ID) key). It reconciles and re-POSTs only
    the missing ones."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    # First POST persists all rows on the server, then raises (lost
    # response). The retry should discover every token already landed
    # and re-POST nothing.
    cat = _ReconcileCatalog(fail_times=1, fail_after_persist=True)
    tokens = ["X1", "X2", "X3"]
    result = post_lease_batch(catalog=cat, tokens=tokens)

    assert set(result.keys()) == set(tokens)
    # The reconcile query ran, and any retried POST carried only
    # tokens NOT already persisted (here: none).
    assert cat._reconcile_chunks, "reconcile query was never issued"
    if len(cat.post_calls) > 1:
        retried_tokens = {b["ID"] for b in cat.post_calls[1]}
        assert retried_tokens == set(), f"retry re-POSTed already-landed tokens: {retried_tokens}"


def test_terminal_4xx_does_not_retry(monkeypatch):
    """A non-transient 4xx (e.g. 403 forbidden) raises immediately,
    without burning retry attempts."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    cat = _ReconcileCatalog(fail_times=99, exc=_http_error(403, "forbidden"))
    with pytest.raises(requests.HTTPError):
        post_lease_batch(catalog=cat, tokens=["T"])
    assert len(cat.post_calls) == 1  # no retry on terminal error


def test_exhausted_retries_raise(monkeypatch):
    """When every attempt fails transiently, the last error propagates."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    monkeypatch.setattr(rid_lease, "LEASE_MAX_ATTEMPTS", 3)
    cat = _ReconcileCatalog(fail_times=99, exc=_http_error(503, "unavailable"))
    with pytest.raises(requests.HTTPError):
        post_lease_batch(catalog=cat, tokens=["T"])
    assert len(cat.post_calls) == 3  # exactly max attempts


def test_backoff_uses_configured_factor(monkeypatch):
    """Retries sleep with exponential backoff between attempts."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    sleeps: list[float] = []
    monkeypatch.setattr(rid_lease.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(rid_lease, "LEASE_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(rid_lease, "LEASE_BACKOFF_FACTOR", 2)
    cat = _ReconcileCatalog(fail_times=2)  # fail twice, succeed on 3rd
    post_lease_batch(catalog=cat, tokens=["A"])
    # Two failures → two backoff sleeps before the 2nd and 3rd tries:
    # 2**0 = 1, 2**1 = 2.
    assert sleeps == [1, 2]

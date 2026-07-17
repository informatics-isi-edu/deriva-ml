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


SELF_RCB = "https://example.org/auth/self-client"
OTHER_RCB = "https://example.org/auth/other-client"


class _ReconcileCatalog:
    """Mock catalog that drives the reconcile-retry path.

    - ``post`` fails for the first ``fail_times`` calls with the
      configured exception, then succeeds. On success it records
      each row it "persists" (keyed by token ID) so the reconcile
      query can report already-landed tokens. Persisted rows carry
      an ``RCB`` (this client, ``SELF_RCB``) so the reconcile query
      can be scoped by creator.
    - ``fail_after_persist``: when True, the failing POST persists
      its rows on the server *before* raising — modelling the
      lost-response case where work landed but the client saw an
      error. The reconcile query then finds those tokens.
    - ``reconcile_fail_times``: fail the first N reconcile ``fetch``
      calls with ``reconcile_exc`` (defaults to a transient 503),
      modelling a transient hiccup on the reconcile query itself.
    - ``seed_foreign``: dict of {token: rid} rows pre-existing under
      ``OTHER_RCB`` — used to prove the reconcile query does NOT
      adopt a lease created by a different client.
    - ``getPathBuilder`` exposes an ``ERMrest_RID_Lease`` table whose
      ``RCB == <id>`` + ``ID.in_(chunk)`` fetch returns the matching
      (token, RID) rows.
    - ``get_authn_session`` reports this client's id (``SELF_RCB``).
    """

    def __init__(
        self,
        *,
        fail_times: int = 0,
        exc: Exception | None = None,
        fail_after_persist: bool = False,
        reconcile_fail_times: int = 0,
        reconcile_exc: Exception | None = None,
        seed_foreign: dict[str, str] | None = None,
        prefix: str = "RID-",
    ):
        self.fail_times = fail_times
        self.exc = exc or _http_error(409, "Schema public does not exist")
        self.fail_after_persist = fail_after_persist
        self.reconcile_fail_times = reconcile_fail_times
        self.reconcile_exc = reconcile_exc or _http_error(503, "unavailable")
        self.prefix = prefix
        self.post_calls: list[list[dict]] = []
        # rows: token -> (rid, rcb)
        self._rows: dict[str, tuple[str, str]] = {}
        for tok, rid in (seed_foreign or {}).items():
            self._rows[tok] = (rid, OTHER_RCB)
        self._counter = 0
        self._reconcile_calls = 0
        self._reconcile_chunks: list[list[str]] = []

    # --- auth -----------------------------------------------------
    def get_authn_session(self):
        class _R:
            def json(self):
                return {"client": {"id": SELF_RCB}}

        return _R()

    # --- POST -----------------------------------------------------
    def _assign(self, tokens: list[str]) -> None:
        """Persist tokens under SELF_RCB, modelling the (RCB, ID) key.

        A token already present under SELF_RCB conflicts (409) — the
        composite unique key forbids a second row for the same client.
        A token present only under OTHER_RCB is free to insert for us
        (different RCB), getting a fresh RID distinct from the foreign
        row's.
        """
        for t in tokens:
            if t in self._rows and self._rows[t][1] == SELF_RCB:
                # Same (RCB, ID) already exists → real ERMrest 409.
                raise _http_error(409, f"duplicate (RCB, ID) for {t}")
            self._rows[t] = (f"{self.prefix}{self._counter}", SELF_RCB)
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

        return _R([{"RID": self._rows[t][0], "ID": t} for t in tokens])

    # --- reconcile query -----------------------------------------
    def getPathBuilder(self):
        outer = self

        class _Pred:
            """Either an equality or an in_ predicate; composable via filter()."""

            def __init__(self, *, rcb=None, chunk=None):
                self.rcb = rcb
                self.chunk = list(chunk) if chunk is not None else None

            def merge(self, other):
                return _Pred(
                    rcb=self.rcb if other.rcb is None else other.rcb,
                    chunk=self.chunk if other.chunk is None else other.chunk,
                )

        class _RCBCol:
            def __eq__(self, value):
                return _Pred(rcb=value)

        class _IDCol:
            def in_(self, chunk):
                return _Pred(chunk=chunk)

        class _Chain:
            def __init__(self, pred: "_Pred"):
                self.pred = pred

            def filter(self, pred: "_Pred"):
                return _Chain(self.pred.merge(pred))

            def attributes(self, *_cols):
                return self

            def fetch(self):
                outer._reconcile_calls += 1
                if outer._reconcile_calls <= outer.reconcile_fail_times:
                    raise outer.reconcile_exc
                chunk = self.pred.chunk or []
                rcb = self.pred.rcb
                outer._reconcile_chunks.append(list(chunk))
                out = []
                for t in chunk:
                    if t not in outer._rows:
                        continue
                    rid, row_rcb = outer._rows[t]
                    if rcb is not None and row_rcb != rcb:
                        continue  # scoped out — created by another client
                    out.append({"RID": rid, "ID": t})
                return out

        class _Table:
            RID = _IDCol()  # only .attributes() uses it; shape doesn't matter
            ID = _IDCol()
            RCB = _RCBCol()

            def filter(self, pred: "_Pred"):
                return _Chain(pred)

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


def test_reconcile_scopes_to_current_client(monkeypatch):
    """A token that exists ONLY under another client's RCB must not be
    adopted. The reconcile query is scoped to the current client, so
    such a token is treated as still-missing and re-POSTed for us.

    (Codex finding 1: ``(RCB, ID)`` is the unique key; ``ID`` alone is
    not, so reconciling on ``ID`` only could return a foreign RID.)
    """
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    # "FOREIGN" already exists under OTHER_RCB with a RID we must NOT
    # return. First POST 409s (transient) to force the reconcile path.
    cat = _ReconcileCatalog(fail_times=1, seed_foreign={"FOREIGN": "NOT-OURS"})
    result = post_lease_batch(catalog=cat, tokens=["FOREIGN"])

    assert result["FOREIGN"] != "NOT-OURS", "reconcile adopted a lease created by a different client"
    # We re-POSTed the token for ourselves (the retry's missing set
    # still contained it after RCB-scoped reconcile found nothing ours).
    assert len(cat.post_calls) == 2


def test_transient_reconcile_query_failure_is_retried(monkeypatch):
    """A transient failure on the reconcile *query* must not abort the
    retry — it consumes an attempt and is retried, same as a POST
    failure.

    (Codex finding 2: the reconcile query was previously unguarded, so
    a 503/timeout on it re-introduced the exact false-failure #360
    fixes, just one call over.)
    """
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    monkeypatch.setattr(rid_lease, "LEASE_MAX_ATTEMPTS", 5)
    # POST fails once (409) → enter reconcile path. The reconcile query
    # then fails transiently once (503) before succeeding. The batch
    # must still ultimately succeed rather than propagating the 503.
    cat = _ReconcileCatalog(
        fail_times=1,
        fail_after_persist=True,  # tokens actually landed on the failed POST
        reconcile_fail_times=1,
    )
    tokens = ["R1", "R2"]
    result = post_lease_batch(catalog=cat, tokens=tokens)
    assert set(result.keys()) == set(tokens)


def test_transient_reconcile_failure_exhaustion_raises(monkeypatch):
    """If the reconcile query keeps failing transiently until attempts
    are exhausted, the last error propagates (does not hang or silently
    return an incomplete map)."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease.time, "sleep", lambda _s: None)
    monkeypatch.setattr(rid_lease, "LEASE_MAX_ATTEMPTS", 3)
    cat = _ReconcileCatalog(
        fail_times=1,
        fail_after_persist=True,
        reconcile_fail_times=99,  # reconcile never recovers
    )
    with pytest.raises(requests.HTTPError):
        post_lease_batch(catalog=cat, tokens=["R"])

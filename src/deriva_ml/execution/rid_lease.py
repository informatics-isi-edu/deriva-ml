"""RID leasing against public:ERMrest_RID_Lease.

Pure helpers — no SQLite awareness here. The production consumer is
``bag_commit._add_asset_rows_to_bag``, which leases RIDs in batch
before adding asset / association rows to the transient commit bag.

Why a dedicated module: the POST body format, chunking, and
error-handling choices are specific to the lease table and worth
isolating from higher-level orchestration.

Transient-failure handling (issue #360)
---------------------------------------
The lease POST is the *last* step of an arbitrarily long commit
(assets + feature rows + provenance already landed). A single
transient ERMrest hiccup on this bookkeeping POST must not discard
the success signal of the whole run, so ``post_lease_batch`` retries
transient failures with bounded exponential backoff.

The retry is **reconcile-based**, not blind re-POST. It relies on an
ERMrest guarantee about ``public:ERMrest_RID_Lease``:

    The table has a composite unique key ``(RCB, ID)``, where ``RCB``
    is the row-created-by (authenticated client) and ``ID`` is our
    client-generated UUID4 lease token. ``ID`` alone is NOT unique.

Because of that key, a POST that *landed* on the server but whose
response was lost cannot simply be re-sent — re-POSTing the same
tokens would 409 against the rows that already succeeded. So before
each retry we query which of our tokens already exist and re-POST
only the missing ones. This is exactly what the UUID4 ``ID`` token
was designed for (see :func:`generate_lease_token`).

This ERMrest guarantee is pinned for *our* CI by
``tests/execution/test_rid_lease_idempotency_contract.py`` — if a
future server/deriva-py change drops the ``(RCB, ID)`` unique key,
that test fails and this reconcile logic must be revisited.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Iterable

from deriva_ml.core.logging_config import get_logger

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

logger = get_logger(__name__)
# Chunk size for batched POSTs. 500 keeps us comfortably under
# ERMrest URL and body-size limits while amortizing round-trip cost.
# See spec §2.6 — may be tuned by tests via monkeypatch.
PENDING_ROWS_LEASE_CHUNK = 500

# Reconcile-retry tuning for the lease POST (issue #360). Mirrors
# deriva-py's ``datapath._request_with_retry`` shape (exponential
# backoff of ``factor ** (attempt - 1)`` seconds), with 409 added to
# the transient set because the observed production failure was a
# *spurious* 409 ("Schema public does not exist") against a healthy
# lease table. Reconciliation makes retrying a 409 safe.
# Tunable by tests via monkeypatch.
LEASE_MAX_ATTEMPTS = 5
LEASE_BACKOFF_FACTOR = 2
# HTTP status codes we treat as transient for the lease POST. 409 is
# included here (unlike a generic insert) — see module docstring.
LEASE_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _http_status(exc: BaseException) -> int | None:
    """Return the HTTP status code carried by ``exc``, or None.

    Args:
        exc: An exception raised by a catalog POST.

    Returns:
        The integer status code if ``exc`` is a
        :class:`requests.HTTPError` whose ``response`` carries a
        status; otherwise ``None`` (e.g. a bare transport error).

    Example:
        >>> import requests
        >>> r = requests.Response()
        >>> r.status_code = 503
        >>> _http_status(requests.HTTPError(response=r))
        503
        >>> _http_status(RuntimeError("boom")) is None
        True
    """
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    return int(status) if status is not None else None


def _is_transient_lease_error(exc: BaseException) -> bool:
    """Classify a lease-POST exception as transient (retry) or terminal.

    Transient: an :class:`requests.HTTPError` whose status is in
    :data:`LEASE_TRANSIENT_STATUS`, or a non-HTTP exception presumed
    to be a transport error (timeout, connection reset) with no
    status attached. Terminal: any other 4xx (e.g. 403 forbidden,
    404) — re-sending will not help.

    Args:
        exc: The exception raised by the catalog POST.

    Returns:
        ``True`` if the caller should retry, ``False`` to raise.

    Example:
        >>> import requests
        >>> r = requests.Response(); r.status_code = 409
        >>> _is_transient_lease_error(requests.HTTPError(response=r))
        True
        >>> r2 = requests.Response(); r2.status_code = 403
        >>> _is_transient_lease_error(requests.HTTPError(response=r2))
        False
        >>> _is_transient_lease_error(ConnectionError("reset"))
        True
    """
    status = _http_status(exc)
    if status is None:
        # No HTTP status → transport-level error; presume transient.
        return True
    return status in LEASE_TRANSIENT_STATUS


def _fetch_landed_leases(
    catalog: "ErmrestCatalog",
    tokens: list[str],
) -> dict[str, str]:
    """Return token→RID for tokens already present in ERMrest_RID_Lease.

    Reconciliation query for the retry path: after a transient POST
    failure we cannot know which rows landed, so we ask the server.
    Filters by our client's tokens (the ``ID`` column); the
    ``(RCB, ID)`` unique key means each token maps to at most one row
    created by this client.

    Args:
        catalog: Live ErmrestCatalog to query.
        tokens: Lease tokens to look up.

    Returns:
        Mapping of every token found in the lease table to its
        server-assigned RID. Tokens not yet landed are absent.
    """
    found: dict[str, str] = {}
    if not tokens:
        return found
    pb = catalog.getPathBuilder()
    lease_table = pb.schemas["public"].tables["ERMrest_RID_Lease"]
    for i in range(0, len(tokens), PENDING_ROWS_LEASE_CHUNK):
        chunk = tokens[i : i + PENDING_ROWS_LEASE_CHUNK]
        rows = lease_table.filter(lease_table.ID.in_(chunk)).attributes(lease_table.RID, lease_table.ID).fetch()
        for row in rows:
            found[row["ID"]] = row["RID"]
    return found


def _lease_chunk_with_retry(
    *,
    catalog: "ErmrestCatalog",
    tokens: list[str],
) -> dict[str, str]:
    """POST one chunk of lease tokens, reconciling+retrying transients.

    Args:
        catalog: Live ErmrestCatalog to POST against.
        tokens: The chunk of lease tokens to POST (already sized to
            :data:`PENDING_ROWS_LEASE_CHUNK`).

    Returns:
        Mapping of every input token to its leased RID.

    Raises:
        Exception: The last exception seen, if a terminal (non-
            transient) error occurs or all attempts are exhausted.
    """
    resolved: dict[str, str] = {}
    last_exc: BaseException | None = None

    for attempt in range(LEASE_MAX_ATTEMPTS):
        if attempt > 0:
            delay = LEASE_BACKOFF_FACTOR ** (attempt - 1)
            logger.debug("lease POST retry %d: sleeping %ss", attempt, delay)
            time.sleep(delay)
            # Reconcile: learn which tokens already landed so we
            # don't re-POST them (that would 409 on (RCB, ID)).
            landed = _fetch_landed_leases(catalog, tokens)
            resolved.update(landed)

        missing = [t for t in tokens if t not in resolved]
        if not missing:
            return resolved

        try:
            response = catalog.post(
                "/entity/public:ERMrest_RID_Lease",
                json=[{"ID": t} for t in missing],
            )
            for row in response.json():
                # ERMrest echoes both ID (our token) and RID (assigned).
                resolved[row["ID"]] = row["RID"]
            return resolved
        except Exception as exc:  # noqa: BLE001 — reclassified below
            last_exc = exc
            if not _is_transient_lease_error(exc):
                raise
            logger.warning(
                "Transient lease-POST failure (attempt %d/%d): %s",
                attempt + 1,
                LEASE_MAX_ATTEMPTS,
                exc,
            )

    # Exhausted every attempt on transient failures. A final
    # reconcile may still have completed the batch (e.g. the last
    # POST landed but its response was lost); check before giving up.
    resolved.update(_fetch_landed_leases(catalog, tokens))
    if all(t in resolved for t in tokens):
        return resolved

    logger.error("Lease POST exhausted %d attempts", LEASE_MAX_ATTEMPTS)
    assert last_exc is not None  # loop ran at least once with a failure
    raise last_exc


def generate_lease_token() -> str:
    """Generate a fresh lease token.

    Returns:
        A UUID4 string. Used as the ERMrest_RID_Lease.ID column so
        we can look up what we leased after a mid-flight crash.

    Example:
        >>> token = generate_lease_token()
        >>> len(token) == 36
        True
    """
    return str(uuid.uuid4())


def post_lease_batch(
    *,
    catalog: "ErmrestCatalog",
    tokens: list[str],
) -> dict[str, str]:
    """POST to ERMrest_RID_Lease in chunks; return token→RID map.

    Args:
        catalog: Live ErmrestCatalog to POST against.
        tokens: Lease tokens (typically uuid4 strings from
            generate_lease_token). Empty list is a no-op.

    Returns:
        Dict mapping each input token to its server-assigned RID.

    Raises:
        Exception: Whatever the catalog raises on a *terminal*
            (non-transient) POST failure, or the last transient
            error after exhausting :data:`LEASE_MAX_ATTEMPTS`.
            Transient failures (timeouts, 5xx, spurious 409) are
            retried with reconcile-based backoff so a single hiccup
            on this finalize-step POST does not fail an otherwise
            complete commit (issue #360). Partial progress is not
            rolled back, but retries reconcile against the server's
            lease rows, so re-running is idempotent by construction.

    Example:
        >>> tokens = [generate_lease_token() for _ in range(100)]  # doctest: +SKIP
        >>> assigned = post_lease_batch(catalog=cat, tokens=tokens)  # doctest: +SKIP
        >>> assigned[tokens[0]]  # doctest: +SKIP
        'EXE-ABC'
    """
    if not tokens:
        return {}

    result: dict[str, str] = {}
    # Chunk to keep URL + body sizes bounded. Each chunk POST retries
    # transient failures independently, reconciling against the
    # server's lease rows so a landed-but-lost POST is not re-sent.
    for i in range(0, len(tokens), PENDING_ROWS_LEASE_CHUNK):
        chunk = tokens[i : i + PENDING_ROWS_LEASE_CHUNK]
        result.update(_lease_chunk_with_retry(catalog=catalog, tokens=chunk))
    return result


class LeaseAggregator:
    """Accumulate lease tokens from multiple call sites, flush in one POST.

    Pre-extraction (audit P1 Ex-batch), ``bag_commit`` made three
    separate ``post_lease_batch`` calls per commit — one for
    ``*_Execution`` association rows, one for ``*_Asset_Type``
    association rows, one for feature rows. Each is a serialized
    round trip to ERMrest. For a 1,000-asset commit with 3 types
    each + features, that's 3 sequential POSTs that could be one
    batch.

    This aggregator collapses them. Call sites:

    1. ``reserve(n)`` — get a list of ``n`` fresh tokens and
       register them with the aggregator. Returns the tokens so
       the caller can map them onto rows (the production pattern
       keeps token order aligned with row order).
    2. After every site has reserved its tokens, call ``flush()``
       once. This issues a single ``post_lease_batch`` for all
       accumulated tokens.
    3. ``resolve(token)`` — look up the leased RID for a token
       after ``flush()``. Raises ``KeyError`` for an unknown
       token (call ``reserve`` first) or if ``flush()`` hasn't
       been called yet.

    The aggregator is single-shot: ``flush()`` is intended to be
    called once at the end of a commit. Multiple ``reserve`` →
    one ``flush`` is the supported flow. Calling ``reserve``
    after ``flush()`` raises :class:`RuntimeError` — would
    create a token that was never POSTed.

    Note: ``post_lease_batch`` is still the underlying primitive.
    The aggregator just defers the call so multiple sites pay
    one round-trip instead of N.

    Example:
        >>> from deriva_ml.execution.rid_lease import LeaseAggregator
        >>> agg = LeaseAggregator()
        >>> tokens_a = agg.reserve(2)
        >>> tokens_b = agg.reserve(3)
        >>> len(tokens_a) == 2
        True
        >>> len(tokens_b) == 3
        True
        >>> # agg.flush(catalog=cat) would POST 5 tokens in one batch
    """

    def __init__(self) -> None:
        self._tokens: list[str] = []
        self._lease_map: dict[str, str] | None = None

    def reserve(self, n: int) -> list[str]:
        """Reserve ``n`` lease tokens and register them with this aggregator.

        Args:
            n: How many tokens to reserve. Non-negative.

        Returns:
            List of ``n`` UUID4 strings. Order matches caller
            insertion order; the production pattern is to zip
            this list against row data so the leased RIDs land
            in the right rows after ``flush()``.

        Raises:
            RuntimeError: If called after ``flush()`` —
                creating a token post-flush would leave it
                un-POSTed, breaking the invariant that
                ``resolve()`` can always answer for any
                reserved token.
            ValueError: If ``n`` is negative.
        """
        if self._lease_map is not None:
            raise RuntimeError(
                "LeaseAggregator.reserve() called after flush(); "
                "this aggregator is single-shot. Build a fresh aggregator "
                "for any additional leases."
            )
        if n < 0:
            raise ValueError(f"reserve() requires n >= 0, got {n}")
        new_tokens = [generate_lease_token() for _ in range(n)]
        self._tokens.extend(new_tokens)
        return new_tokens

    def flush(self, *, catalog: "ErmrestCatalog") -> dict[str, str]:
        """POST every accumulated token in one batch; return token → RID.

        Args:
            catalog: Live :class:`ErmrestCatalog` to POST against.

        Returns:
            Dict mapping every reserved token to its leased
            RID. Empty if ``reserve()`` was never called (a
            no-op flush; useful in code paths where the
            aggregator is unconditionally flushed but may
            have nothing to lease).

        Raises:
            RuntimeError: If called twice. The aggregator is
                single-shot.
        """
        if self._lease_map is not None:
            raise RuntimeError("LeaseAggregator.flush() called twice; this aggregator is single-shot.")
        self._lease_map = post_lease_batch(catalog=catalog, tokens=self._tokens)
        return self._lease_map

    def resolve(self, token: str) -> str:
        """Return the leased RID for ``token``.

        Args:
            token: A token previously returned by ``reserve()``.

        Returns:
            The RID assigned to this token at flush time.

        Raises:
            RuntimeError: If ``flush()`` hasn't been called yet.
            KeyError: If ``token`` was never reserved by this
                aggregator.
        """
        if self._lease_map is None:
            raise RuntimeError(
                "LeaseAggregator.resolve() called before flush(); no RID has been assigned to any token yet."
            )
        return self._lease_map[token]


def _validate_pending_asset_leases(
    catalog: "ErmrestCatalog",
    entries: "Iterable[tuple[str, str]]",
) -> None:
    """Confirm each (key, rid) pair's RID is still live in ERMrest_RID_Lease.

    Queries the lease table in batches of ``PENDING_ROWS_LEASE_CHUNK``.
    Aggregates missing RIDs and raises a single
    :class:`DerivaMLValidationError` listing every failure in sorted
    order. Returns ``None`` silently when every RID is present.

    Args:
        catalog: Live ErmrestCatalog for querying the lease table.
        entries: Iterable of (key, rid) tuples. Key is a
            human-readable identifier used in the error message.

    Raises:
        DerivaMLValidationError: If one or more RIDs are not found
            in ``ERMrest_RID_Lease``.
    """
    from deriva_ml.core.exceptions import DerivaMLValidationError

    entries_list = list(entries)
    if not entries_list:
        return

    # Build a reverse map so we can attribute a missing RID back to
    # its caller-supplied key. If the same RID appears under two keys
    # (shouldn't happen in practice), the forward list below produces
    # one missing-entry per occurrence.
    rid_to_keys: dict[str, list[str]] = {}
    for key, rid in entries_list:
        rid_to_keys.setdefault(rid, []).append(key)

    all_rids = list(rid_to_keys.keys())
    found_rids: set[str] = set()

    pb = catalog.getPathBuilder()
    lease_table = pb.schemas["public"].tables["ERMrest_RID_Lease"]
    for i in range(0, len(all_rids), PENDING_ROWS_LEASE_CHUNK):
        chunk = all_rids[i : i + PENDING_ROWS_LEASE_CHUNK]
        rows = lease_table.filter(lease_table.RID.in_(chunk)).attributes(lease_table.RID).fetch()
        for row in rows:
            found_rids.add(row["RID"])

    missing: list[tuple[str, str]] = []
    for key, rid in entries_list:
        if rid not in found_rids:
            missing.append((key, rid))
    if not missing:
        return

    lines = [f"Missing or invalid pre-allocated RIDs for {len(missing)} pending asset(s):"]
    for key, rid in sorted(missing):
        lines.append(f"  - {key}: RID {rid} not found in ERMrest_RID_Lease")
    lines.append(
        "A pre-leased RID has become invalid (e.g., cleared from the "
        "lease table or never successfully POSTed). Restart the "
        "execution to re-lease, or investigate lease-table state."
    )
    raise DerivaMLValidationError("\n".join(lines))

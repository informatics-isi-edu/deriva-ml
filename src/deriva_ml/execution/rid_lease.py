"""RID leasing against public:ERMrest_RID_Lease.

Pure helpers — no SQLite awareness here. The production consumer is
``bag_commit._add_asset_rows_to_bag``, which leases RIDs in batch
before adding asset / association rows to the transient commit bag.

Why a dedicated module: the POST body format, chunking, and
error-handling choices are specific to the lease table and worth
isolating from higher-level orchestration.
"""

from __future__ import annotations

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
        Exception: Whatever the catalog raises on POST failure.
            Partial progress is NOT rolled back — the caller is
            responsible for recording which tokens landed (via the
            two-phase SQLite write in Task F2).

    Example:
        >>> tokens = [generate_lease_token() for _ in range(100)]  # doctest: +SKIP
        >>> assigned = post_lease_batch(catalog=cat, tokens=tokens)  # doctest: +SKIP
        >>> assigned[tokens[0]]  # doctest: +SKIP
        'EXE-ABC'
    """
    if not tokens:
        return {}

    result: dict[str, str] = {}
    # Chunk to keep URL + body sizes bounded.
    for i in range(0, len(tokens), PENDING_ROWS_LEASE_CHUNK):
        chunk = tokens[i : i + PENDING_ROWS_LEASE_CHUNK]
        body = [{"ID": t} for t in chunk]
        response = catalog.post("/entity/public:ERMrest_RID_Lease", json=body)
        for row in response.json():
            # ERMrest echoes both ID (our token) and RID (assigned).
            result[row["ID"]] = row["RID"]
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
            raise RuntimeError(
                "LeaseAggregator.flush() called twice; this aggregator "
                "is single-shot."
            )
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
                "LeaseAggregator.resolve() called before flush(); "
                "no RID has been assigned to any token yet."
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

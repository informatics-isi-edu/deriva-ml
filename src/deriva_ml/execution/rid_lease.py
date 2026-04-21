"""RID leasing against public:ERMrest_RID_Lease.

Per spec §2.6. Pure helpers — no SQLite awareness here. The
acquire_leases_for_pending function in state_store composition (Task
F2) wires these into the two-phase SQLite protocol.

Why a dedicated module: the POST body format, chunking, and
error-handling choices are specific to the lease table and worth
isolating from the higher-level "take pending_rows with status=staged
and assign them RIDs" orchestration.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

logger = logging.getLogger(__name__)

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
        >>> tokens = [generate_lease_token() for _ in range(100)]
        >>> assigned = post_lease_batch(catalog=cat, tokens=tokens)
        >>> assigned[tokens[0]]
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

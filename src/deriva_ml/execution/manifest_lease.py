"""Lease RIDs for manifest-pending asset entries.

Parallel to :func:`~deriva_ml.execution.lease_orchestrator.acquire_leases_for_execution`
(which operates on SQLite pending rows), this helper operates on
:class:`~deriva_ml.asset.manifest.AssetManifest` entries. It's used by
the manifest-driven upload path (``Execution._upload_execution_dirs``)
to ensure every pending asset has a pre-allocated RID before staging.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from deriva_ml.execution.rid_lease import generate_lease_token, post_lease_batch

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml.asset.manifest import AssetManifest


def lease_manifest_pending_assets(
    catalog: "ErmrestCatalog",
    manifest: "AssetManifest",
) -> None:
    """Assign pre-allocated RIDs to manifest entries that lack one.

    Iterates ``manifest.pending_assets()`` and collects entries whose
    ``rid`` is None. POSTs a batch of lease tokens to
    ``public:ERMrest_RID_Lease`` and writes each server-assigned RID
    back to the manifest via ``manifest.set_asset_rid()``.

    No-op when every pending entry already has a RID (e.g., the
    engine-driven upload path lands here after having leased RIDs
    into SQLite; the manifest entries mirror the SQLite state).

    Args:
        catalog: Live ErmrestCatalog for POSTing to ERMrest_RID_Lease.
        manifest: AssetManifest whose pending entries may need RIDs.
    """
    pending = manifest.pending_assets()
    needs_lease = [
        (key, entry) for key, entry in pending.items() if entry.rid is None
    ]
    if not needs_lease:
        return

    # Map lease-token → manifest key so we can write RIDs back to the
    # right entry after the POST response.
    token_to_key: dict[str, str] = {}
    for key, _entry in needs_lease:
        token = generate_lease_token()
        token_to_key[token] = key

    token_to_rid = post_lease_batch(catalog=catalog, tokens=list(token_to_key.keys()))

    for token, rid in token_to_rid.items():
        key = token_to_key[token]
        manifest.set_asset_rid(key, rid)

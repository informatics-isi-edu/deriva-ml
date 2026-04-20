"""Connection mode enumeration for DerivaML.

See spec §2.1 — online mode talks to the catalog eagerly; offline mode
stages all work locally in SQLite until an upload operation drains it.
"""

from __future__ import annotations

from enum import StrEnum


class ConnectionMode(StrEnum):
    """How a DerivaML instance interacts with the catalog.

    Members:
        online: Writes reach the catalog by the time the call returns
            (plain rows). Asset files still stage and wait for upload.
            Execution status transitions sync to the catalog atomically.
        offline: Every write stages into the workspace SQLite and stays
            there until upload. No server contact except for RID leases
            and the final upload.

    Example:
        >>> from deriva_ml import ConnectionMode, DerivaML
        >>> ml = DerivaML(hostname="example.org", catalog_id="42",
        ...               mode=ConnectionMode.offline)
        >>> ml.mode is ConnectionMode.offline
        True
    """

    online = "online"
    offline = "offline"

    def __str__(self) -> str:
        return self.value

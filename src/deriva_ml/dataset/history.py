"""History tracking utilities for Deriva records.

This module provides functions for retrieving the history of records
from a Deriva catalog using snaptime-based queries.
"""

from datetime import datetime
from typing import Any

from deriva.core import urlquote
from deriva.core.deriva_server import DerivaServer
from deriva.core.ermrest_model import (
    datetime_to_epoch_microseconds,
    epoch_microseconds_to_snaptime,
    timestamptz_to_snaptime,
)


def get_record_history(
    server: DerivaServer,
    cid: str | int,
    sname: str,
    tname: str,
    kvals: list[str],
    kcols: list[str] | None = None,
    snap: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Get the history of a record from the catalog.

    Traverses backward through catalog snapshots to find all versions
    of a record, starting from the latest (or specified) snapshot.

    Args:
        server: The server instance.
        cid: The catalog ID.
        sname: The schema name.
        tname: The table name.
        kvals: The key values to look up.
        kcols: The key columns. Defaults to ["RID"].
        snap: Optional snapshot ID to start from. If None, uses latest.

    Returns:
        A dict mapping snapshot IDs to row data for each version found.

    Raises:
        ValueError: If more than one row is returned for a query.
    """
    if kcols is None:
        kcols = ["RID"]

    parts = {
        "cid": urlquote(cid),
        "sname": urlquote(sname),
        "tname": urlquote(tname),
        "filter": ",".join(
            [
                "%s=%s" % (urlquote(kcol), urlquote(kval))
                for kcol, kval in zip(kcols, kvals)
            ]
        ),
    }

    if snap is None:
        # Determine starting (latest) snapshot
        r = server.get("/ermrest/catalog/%(cid)s" % parts)
        snap = r.json()["snaptime"]
    parts["snap"] = snap

    path = "/ermrest/catalog/%(cid)s@%(snap)s/entity/%(sname)s:%(tname)s/%(filter)s"

    snap2rows: dict[str, dict[str, Any]] = {}
    while True:
        url = path % parts
        response_data = server.get(url).json()
        if len(response_data) > 1:
            raise ValueError("got more than one row for %r" % url)
        if len(response_data) == 0:
            break
        row = response_data[0]
        snap2rows[parts["snap"]] = row
        # Parse RMT and find snap ID prior to row version birth time
        rmt = datetime.fromisoformat(row["RMT"])
        rmt_us = datetime_to_epoch_microseconds(rmt)
        parts["snap"] = epoch_microseconds_to_snaptime(rmt_us - 1)

    return snap2rows


def iso_to_snaptime(iso_datetime: str) -> str:
    """Convert ISO datetime string to ERMrest snaptime format.

    Args:
        iso_datetime: The ISO datetime string (e.g., from RMT column).

    Returns:
        The ERMrest snaptime string.
    """
    return timestamptz_to_snaptime(iso_datetime)


# Deprecated aliases for backward compatibility
def datetime_epoch_us(dt: datetime) -> int:
    """Convert datetime to epoch microseconds.

    .. deprecated::
        Use `deriva.core.ermrest_model.datetime_to_epoch_microseconds` instead.

    Args:
        dt: The datetime object to convert.

    Returns:
        The epoch time in microseconds.
    """
    return datetime_to_epoch_microseconds(dt)


def iso_to_snap(iso_datetime: str) -> str:
    """Convert ISO datetime string to snapshot format.

    .. deprecated::
        Use `iso_to_snaptime` or `deriva.core.ermrest_model.timestamptz_to_snaptime` instead.

    Args:
        iso_datetime: The ISO datetime string.

    Returns:
        The snapshot timestamp string.
    """
    return timestamptz_to_snaptime(iso_datetime)

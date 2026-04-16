"""Pure path-helper functions for the local_db layout.

No side effects, no SQLite access, no filesystem mutation. These helpers
exist so both tests and runtime code derive layout paths from a single
canonical source.
"""

from __future__ import annotations

from pathlib import Path

_UNSAFE_CHARS = set('/\\:<>"|?*')


def _sanitise_component(raw: str) -> str:
    """Replace path-unsafe characters with '_'. Return '_' for empty input.

    Special case: '.' is only unsafe when it forms sequences like '..' that have
    special meaning in paths. Normal dots in filenames (like 'foo.bar') are preserved.
    """
    if not raw:
        return "_"
    # Handle special path components (sequences of dots)
    if raw and all(c == "." for c in raw):
        return "_" * len(raw)
    # Normal character replacement
    return "".join("_" if c in _UNSAFE_CHARS else c for c in raw)


def workspace_root(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the per-catalog workspace root directory.

    Layout: {working_dir}/catalogs/{host}__{catalog_id}/
    """
    host = _sanitise_component(hostname)
    cat = _sanitise_component(str(catalog_id))
    return Path(working_dir) / "catalogs" / f"{host}__{cat}"


def working_db_path(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the per-catalog working DB directory path.

    The working DB is a directory containing main.db plus per-schema .db files
    (created by SchemaBuilder's multi-schema ATTACH pattern).

    Layout: {workspace_root}/working/
    """
    return workspace_root(working_dir, hostname, catalog_id) / "working"


def working_main_db_path(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the main.db file inside the working DB directory.

    This is the file the SQLAlchemy engine opens. Per-schema .db files are
    ATTACH'd into connections on this engine.
    """
    return working_db_path(working_dir, hostname, catalog_id) / "main.db"


def slice_dir(working_dir: Path, hostname: str, catalog_id: str | int, slice_id: str) -> Path:
    """Return the directory for a single slice.

    Layout: {workspace_root}/slices/{slice_id}/
    """
    return workspace_root(working_dir, hostname, catalog_id) / "slices" / _sanitise_component(slice_id)


def slice_db_path(working_dir: Path, hostname: str, catalog_id: str | int, slice_id: str) -> Path:
    """Return the slice.db path for a single slice."""
    return slice_dir(working_dir, hostname, catalog_id, slice_id) / "slice.db"

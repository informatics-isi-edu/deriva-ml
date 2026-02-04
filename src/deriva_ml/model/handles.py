"""Handle wrappers for ERMrest model objects.

This module re-exports TableHandle and ColumnHandle from deriva-py
for backwards compatibility. New code should import directly from
deriva.core.model_handles.

Classes:
    ColumnHandle: Wrapper for ERMrest Column with simplified property access.
    TableHandle: Wrapper for ERMrest Table with simplified operations.
"""

from deriva.core.model_handles import ColumnHandle, TableHandle

__all__ = ["TableHandle", "ColumnHandle"]

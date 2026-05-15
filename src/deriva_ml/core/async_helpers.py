"""Async-runner helpers for deriva-ml's sync entry points.

DerivaML's public API is synchronous, but several pieces of the
underlying deriva-py stack expose async-only entry points
(:meth:`deriva.bag.catalog_loader.BagCatalogLoader.arun`, the
catalog-async queries in :class:`deriva.core.async_ermrest.AsyncErmrestCatalog`,
etc.). The naive bridge — ``asyncio.run(coro)`` — raises
``RuntimeError: asyncio.run() cannot be called from a running event
loop`` when called from inside a Jupyter / papermill kernel, which
runs its own event loop on the main thread.

This module centralises the loop-detection fallback so we don't
copy-paste the same try/except into every sync entry point.

See :func:`run_async` for the canonical pattern.

Example:
    >>> from deriva_ml.core.async_helpers import run_async
    >>> async def _work():
    ...     return 42
    >>> run_async(_work())  # works both inside and outside a running loop
    42
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")

__all__ = ["run_async"]


def run_async(coro: Awaitable[T]) -> T:
    """Run ``coro`` to completion from a sync caller.

    Detects whether the caller is already inside a running event
    loop (notebook context) and uses :mod:`nest_asyncio` to
    re-enter it. Outside a loop, falls back to plain
    :func:`asyncio.run`.

    Args:
        coro: An awaitable to run. Typically the return value of an
            ``async def`` call.

    Returns:
        The coroutine's return value.

    Raises:
        Whatever ``coro`` raises. No exceptions are introduced by
        this wrapper itself.

    Example:
        >>> from deriva_ml.core.async_helpers import run_async
        >>> async def _work():
        ...     return "hello"
        >>> run_async(_work())
        'hello'
    """
    try:
        loop: Any = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Notebook context: re-enter the active loop via
        # nest_asyncio. Imported lazily so the base library stays
        # importable without it installed.
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)

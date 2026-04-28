"""Sort-spec resolution helper for ``find_*`` methods.

A small, pure-Python helper that lets every ``find_*`` method on
``DerivaML`` (and its dataset / bag counterparts) accept a uniform
three-state ``sort=`` parameter:

- ``None`` (default): no sort is applied. The caller gets rows in
  whatever order the backend returns, which is cheapest. This
  preserves the pre-sort behavior of every existing caller.
- ``True``: the method's documented default sort applies. For
  activity-log methods (``find_executions``, ``find_datasets``,
  ``find_workflows``) this is "newest-first by record creation time"
  (``RCT desc``). The exact default is the method's responsibility;
  this module only routes the request.
- A callable ``(path) -> sort_keys``: caller-supplied override. The
  callable receives the path-builder context (an ERMrest entity-table
  path) and returns either a single sort key (a column wrapper, or
  ``column.desc``) or a list of them. The result is unpacked into
  ``path.entities().sort(*keys)`` by the calling implementation.

The ``Callable`` form is intentionally library-private -- it requires
knowledge of the deriva-py path-builder column-wrapper API. The MCP
plugin only forwards ``True`` / ``None`` over the wire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

SortSpec = Union[bool, Callable[[Any], Any], None]
"""Type alias for the ``sort=`` parameter on ``find_*`` methods.

See the module docstring for the three-state semantics.
"""


def resolve_sort(
    sort: SortSpec,
    default_callable: Callable[[Any], Any],
    path: Any,
) -> list[Any] | None:
    """Resolve a ``SortSpec`` against a method's default-sort callable.

    Used inside ``find_*`` implementations to translate the caller's
    ``sort=`` argument into the list of sort keys (or ``None``, meaning
    "do not call ``.sort()`` at all"). Each implementation supplies
    its own ``default_callable`` -- for activity-log methods that's
    ``lambda p: p.RCT.desc``; for other methods it can be anything.

    Args:
        sort: The caller's ``sort=`` parameter (None, True, or callable).
        default_callable: Method-supplied callable returning the
            method's documented default sort keys, used when
            ``sort=True``. Receives the same ``path`` object.
        path: The path-builder context to pass to the callable. Opaque
            to this helper -- whatever the implementation's entity-table
            path object is.

    Returns:
        ``None`` when ``sort=None`` -- caller should NOT call
        ``.sort()`` on the path. Otherwise a list of sort keys (one
        or more) suitable for unpacking into ``path.entities().sort(*keys)``.

    Raises:
        TypeError: If ``sort`` is neither ``None``, ``True``, nor a
            callable.

    Example:
        >>> from deriva_ml.core.sort import resolve_sort
        >>> # sort=None -> no sort applied
        >>> resolve_sort(None, lambda p: p, object()) is None
        True
        >>> # sort=True -> default callable runs, result wrapped in list
        >>> resolve_sort(True, lambda p: "RCT-desc", object())
        ['RCT-desc']
        >>> # sort=callable -> user callable runs
        >>> resolve_sort(lambda p: ["A", "B"], lambda p: ["default"], object())
        ['A', 'B']
        >>> # Single value gets wrapped in a list
        >>> resolve_sort(lambda p: "single", lambda p: "default", object())
        ['single']
    """
    if sort is None:
        return None
    if sort is True:
        keys = default_callable(path)
    elif callable(sort):
        keys = sort(path)
    else:
        raise TypeError(f"sort must be None, True, or a callable; got {type(sort).__name__}")
    if isinstance(keys, (list, tuple)):
        return list(keys)
    return [keys]

"""Shared helpers for the ``Execution`` / ``ExecutionRecord`` / ``Workflow`` surfaces.

These free functions replace parallel implementations that the
audit (``docs/audits/2026-05-22-engineer-audit-execution.md``)
flagged as "two places to keep in sync":

- ``check_writable_catalog`` — formerly ``_check_writable_catalog``
  on both :class:`Workflow` and :class:`ExecutionRecord`, two
  near-identical method bodies that differed only in the
  entity label and which field carries the RID.
- ``update_field_in_catalog`` — formerly
  ``_update_description_in_catalog`` on both, plus
  ``_update_status_in_catalog`` on :class:`ExecutionRecord`. All
  three were the same shape: write one or two columns to one
  RID-keyed row in one schema.table.
- ``fetch_nested_execution_rows`` — formerly four inline
  ``pb.schemas[ml_schema].Execution_Execution`` queries
  scattered across ``execution.py`` and ``execution_record.py``,
  each building a slightly different filter / link expression.

Everything here is module-level free functions: no inheritance,
no state, easy to test in isolation against a mock
``ml_instance``. The original callers now delegate to these
with their own ``self.*`` fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    pass


def check_writable_catalog(
    *,
    rid: str | None,
    ml_instance: Any,
    entity_label: str,
    operation: str,
) -> None:
    """Refuse a write when the entity isn't registered or the catalog is read-only.

    Replaces the two ``_check_writable_catalog`` methods on
    :class:`Workflow` and :class:`ExecutionRecord`. The original
    pair differed only in:

    - which field carried the RID (``self.rid`` vs ``self.execution_rid``);
    - the entity label in the error message (``"Workflow"`` vs ``"Execution"``);
    - whether ``ml_instance is None`` was checked explicitly
      (only :class:`ExecutionRecord` did, but the implicit
      ``isinstance(self._ml_instance.catalog, ...)`` would have
      raised ``AttributeError`` for None anyway — folded into
      one explicit check here).

    Args:
        rid: The entity's RID. ``None`` means "never registered"
            and triggers the not-registered refusal.
        ml_instance: The bound :class:`DerivaML` instance, or
            ``None`` when the entity wasn't attached to a catalog.
        entity_label: ``"Workflow"`` or ``"Execution"`` (or any
            future entity label). Interpolated into the error
            messages for caller-friendly diagnostics.
        operation: Short description of the attempted action
            (``"update description"``, ``"add nested execution"``,
            etc.). Surfaces in the error so a user sees what was
            refused.

    Raises:
        DerivaMLException: If the entity isn't registered, isn't
            bound to a catalog, or the catalog is a read-only
            snapshot.
    """
    # Local import — ``deriva.core.ErmrestSnapshot`` carries
    # heavy network deps; keep it lazy so unit tests that mock
    # the catalog don't pay the import.
    import importlib

    _deriva_core = importlib.import_module("deriva.core")
    ErmrestSnapshot = _deriva_core.ErmrestSnapshot

    if rid is None:
        raise DerivaMLException(
            f"Cannot {operation}: {entity_label} is not registered in the catalog (no RID)"
        )

    if ml_instance is None:
        raise DerivaMLException(
            f"Cannot {operation}: {entity_label} is not bound to a catalog"
        )

    if isinstance(ml_instance.catalog, ErmrestSnapshot):
        raise DerivaMLException(
            f"Cannot {operation} on a read-only catalog snapshot. "
            f"Use a writable catalog connection instead."
        )


def update_field_in_catalog(
    *,
    rid: str,
    ml_instance: Any,
    table_name: str,
    updates: dict[str, Any],
) -> None:
    """Write column updates to one RID-keyed catalog row.

    Replaces the per-class ``_update_description_in_catalog`` and
    ``_update_status_in_catalog`` methods. All three had the same
    shape: build ``{"RID": rid, "<col>": value}`` (plus optionally
    a second column) and call ``path.update([...])``.

    Args:
        rid: The row's RID.
        ml_instance: The bound :class:`DerivaML` instance — used
            to reach the pathBuilder + schema.
        table_name: The table to update under
            ``ml_instance.ml_schema``. Examples: ``"Execution"``,
            ``"Workflow"``.
        updates: Column → value dict. The ``"RID"`` key is added
            automatically; callers pass the per-column updates
            only.

    Raises:
        DerivaMLException: If ``ml_instance`` is None, ``rid`` is
            empty, or the catalog rejects the update.
    """
    pb = ml_instance.pathBuilder()
    table_path = pb.schemas[ml_instance.ml_schema].tables[table_name]
    payload = {"RID": rid, **updates}
    table_path.update([payload])


def fetch_nested_execution_rows(
    *,
    ml_instance: Any,
    execution_rid: str,
    direction: Literal["children", "parents"],
) -> list[dict]:
    """Walk the ``Execution_Execution`` association table once.

    Replaces four near-identical inline queries scattered across
    ``execution.py`` and ``execution_record.py`` that all built
    ``pb.schemas[ml_schema].Execution_Execution.filter(...).link(...).fetch()``
    expressions with subtly different filter/link sides.

    Args:
        ml_instance: The bound :class:`DerivaML` instance.
        execution_rid: The anchor execution's RID. Read from
            the ``Execution`` column (when walking children) or
            the ``Nested_Execution`` column (when walking parents).
        direction: ``"children"`` returns rows where
            ``Execution == execution_rid`` linked to their
            ``Nested_Execution`` Execution row; ``"parents"``
            returns rows where ``Nested_Execution == execution_rid``
            linked to their ``Execution`` Execution row.

    Returns:
        A list of dict rows; each carries every Execution column
        for the OTHER side of the association (the child rows
        when walking children, the parent rows when walking
        parents). The caller wraps these in
        :class:`ExecutionRecord` or :class:`Execution` per its
        own taste.
    """
    pb = ml_instance.pathBuilder()
    ml_schema = ml_instance.ml_schema
    exec_exec_path = pb.schemas[ml_schema].Execution_Execution
    execution_path = pb.schemas[ml_schema].Execution

    if direction == "children":
        # Walking children: filter on the parent's Execution column,
        # link out via the child Nested_Execution column.
        path = exec_exec_path.filter(exec_exec_path.Execution == execution_rid).link(
            execution_path,
            on=(exec_exec_path.Nested_Execution == execution_path.RID),
        )
    else:  # parents
        # Walking parents: filter on the child's Nested_Execution
        # column, link out via the parent Execution column.
        path = exec_exec_path.filter(
            exec_exec_path.Nested_Execution == execution_rid
        ).link(
            execution_path,
            on=(exec_exec_path.Execution == execution_path.RID),
        )

    return list(path.entities().fetch())


def insert_nested_execution_link(
    *,
    ml_instance: Any,
    parent_rid: str,
    child_rid: str,
    sequence: int | None = None,
) -> None:
    """Add a parent→child link to ``Execution_Execution``.

    Replaces the two inline ``Execution_Execution.insert([...])``
    blocks (one in :class:`Execution.add_nested_execution`'s
    fallback, one in :class:`ExecutionRecord.add_nested_execution`).

    Args:
        ml_instance: The bound :class:`DerivaML` instance.
        parent_rid: Parent execution RID; written to the
            ``Execution`` column of the association row.
        child_rid: Child execution RID; written to the
            ``Nested_Execution`` column.
        sequence: Optional ordering hint; written to the
            ``Sequence`` column when supplied.
    """
    pb = ml_instance.pathBuilder()
    exec_exec_path = pb.schemas[ml_instance.ml_schema].Execution_Execution
    record: dict[str, Any] = {
        "Execution": parent_rid,
        "Nested_Execution": child_rid,
    }
    if sequence is not None:
        record["Sequence"] = sequence
    exec_exec_path.insert([record])


__all__ = [
    "check_writable_catalog",
    "fetch_nested_execution_rows",
    "insert_nested_execution_link",
    "update_field_in_catalog",
]

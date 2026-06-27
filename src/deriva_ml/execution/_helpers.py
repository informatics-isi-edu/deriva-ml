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
        raise DerivaMLException(f"Cannot {operation}: {entity_label} is not registered in the catalog (no RID)")

    if ml_instance is None:
        raise DerivaMLException(f"Cannot {operation}: {entity_label} is not bound to a catalog")

    if isinstance(ml_instance.catalog, ErmrestSnapshot):
        raise DerivaMLException(
            f"Cannot {operation} on a read-only catalog snapshot. Use a writable catalog connection instead."
        )


def update_field_in_catalog(
    *,
    rid: str,
    ml_instance: Any,
    table_name: str,
    updates: dict[str, Any],
    schema_name: str | None = None,
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
        table_name: The table to update. Looked up under
            ``schema_name`` if given, otherwise
            ``ml_instance.ml_schema``. Examples: ``"Execution"``,
            ``"Workflow"``, ``"Dataset"`` (all in the ML schema);
            ``"Image"``, ``"Subject"`` (in a domain schema —
            require ``schema_name``).
        updates: Column → value dict. The ``"RID"`` key is added
            automatically; callers pass the per-column updates
            only.
        schema_name: Optional explicit schema. When ``None``
            (default), uses ``ml_instance.ml_schema`` — the
            common case for the ML-schema tables. Pass an
            explicit name for Asset rows that live in domain
            schemas.

    Raises:
        DerivaMLException: If ``ml_instance`` is None, ``rid`` is
            empty, or the catalog rejects the update.
    """
    pb = ml_instance.pathBuilder()
    schema = schema_name if schema_name is not None else ml_instance.ml_schema
    table_path = pb.schemas[schema].tables[table_name]
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
        path = exec_exec_path.filter(exec_exec_path.Nested_Execution == execution_rid).link(
            execution_path,
            on=(exec_exec_path.Execution == execution_path.RID),
        )

    return list(path.entities().fetch())


def list_input_datasets(
    *,
    ml_instance: Any,
    execution_rid: str,
) -> list:
    """Return the input :class:`Dataset` list for an execution.

    Filters the ``Dataset_Execution`` association table for rows
    referencing ``execution_rid``. Under the authorship-canonical
    model ``Dataset_Execution`` is **input-only** — output edges
    (a dataset an execution produced) live in
    ``Dataset_Version.Execution``, never here — so every row is an
    input and no producer subtraction is needed.

    Replaces parallel implementations on
    :meth:`Execution.list_input_datasets` (dry-run fallback)
    and :meth:`ExecutionRecord.list_input_datasets`. Both
    classes now delegate here.

    Args:
        ml_instance: The bound :class:`DerivaML` instance.
        execution_rid: The anchor execution RID.

    Returns:
        List of :class:`Dataset` objects. Empty when the
        execution has no input datasets.
    """
    pb = ml_instance.pathBuilder()
    dataset_exec = pb.schemas[ml_instance.ml_schema].Dataset_Execution
    records = dataset_exec.filter(dataset_exec.Execution == execution_rid).entities().fetch()
    return [ml_instance.lookup_dataset(record["Dataset"]) for record in records if record.get("Dataset")]


def list_input_datasets_with_versions(
    *,
    ml_instance: Any,
    execution_rid: str,
) -> list[tuple[Any, str | None]]:
    """Input datasets of an execution paired with the consumed version.

    Like :func:`list_input_datasets`, but also returns the
    ``Dataset_Execution.Dataset_Version`` recorded on each input edge — the
    version of the dataset that was actually consumed. Lineage uses this to walk
    the consumed version rather than the dataset's current state. The existing
    :func:`list_input_datasets` ``list[Dataset]`` contract is intentionally left
    unchanged; lineage is the only caller that needs the consumed version.

    ``Dataset_Execution.Dataset_Version`` is a **foreign key** to the
    ``Dataset_Version`` table — ERMrest returns the Dataset_Version row's RID
    (e.g. ``"4FP"``), not the version string (e.g. ``"1.0.0"``). This helper
    resolves that RID to the version string by fetching the ``Dataset_Version``
    table once and building a ``{RID: Version}`` map.

    Args:
        ml_instance: The bound :class:`DerivaML` instance.
        execution_rid: The anchor execution RID.

    Returns:
        List of ``(Dataset, consumed_version)`` tuples. ``consumed_version`` is
        the version string from the input edge, or ``None`` when the edge has no
        version pin. Empty when the execution has no input datasets.

    Example:
        >>> pairs = list_input_datasets_with_versions(  # doctest: +SKIP
        ...     ml_instance=ml, execution_rid="2-EXAA"
        ... )
        >>> [(ds.dataset_rid, v) for ds, v in pairs]  # doctest: +SKIP
        [('1-DSAA', '1.0.0')]
    """
    pb = ml_instance.pathBuilder()
    dataset_exec = pb.schemas[ml_instance.ml_schema].Dataset_Execution
    records = [
        record
        for record in dataset_exec.filter(dataset_exec.Execution == execution_rid).entities().fetch()
        if record.get("Dataset")
    ]
    if not records:
        return []

    # Dataset_Execution.Dataset_Version is an FK — the value is the
    # Dataset_Version row RID, not the version string. Resolve RID -> Version.
    version_path = pb.schemas[ml_instance.ml_schema].tables["Dataset_Version"]
    rid_to_version: dict[str, str | None] = {row["RID"]: row.get("Version") for row in version_path.entities().fetch()}

    result: list[tuple[Any, str | None]] = []
    for record in records:
        version_rid = record.get("Dataset_Version")
        consumed_version = rid_to_version.get(version_rid) if version_rid else None
        result.append((ml_instance.lookup_dataset(record["Dataset"]), consumed_version))
    return result


def list_assets(
    *,
    ml_instance: Any,
    execution_rid: str,
    asset_role: str | None = None,
    logger: Any = None,
) -> list:
    """Return the :class:`Asset` list associated with an execution.

    Walks the ``*_Execution`` association tables across the
    domain + ml schemas (excluding ``Dataset_Execution``,
    which is the dataset linkage and handled separately by
    :func:`list_input_datasets`), filters by the anchor
    execution and optionally by ``asset_role``, and looks up
    each matched asset.

    Uses :meth:`DerivaModel.find_asset_execution_tables` to
    discover the association tables once per model lifetime —
    repeated ``list_assets`` calls reuse the cached discovery
    so the schema walk cost is paid once, not per call.

    Replaces parallel implementations on
    :meth:`Execution.list_assets` (dry-run fallback path) and
    :meth:`ExecutionRecord.list_assets`.

    Args:
        ml_instance: The bound :class:`DerivaML` instance.
        execution_rid: The anchor execution RID.
        asset_role: Optional filter — ``"Input"`` or
            ``"Output"`` from the ``Asset_Role`` vocabulary.
            ``None`` returns all.
        logger: Optional logger for per-asset debug lines
            ("could not look up asset"). Defaults to the
            module logger if ``None``.

    Returns:
        List of :class:`Asset` objects. **Only** per-asset
        ``lookup_asset`` failures are swallowed with a debug
        log so a single asset's catalog issue doesn't break
        the whole listing. Connectivity errors on the
        outer association-table query propagate — silently
        returning an empty list for a real catalog problem
        would be misleading.
    """
    if logger is None:
        from deriva_ml.core.logging_config import get_logger

        logger = get_logger(__name__)

    assets = []
    pb = ml_instance.pathBuilder()

    # Cached once per ``DerivaModel`` lifetime. Pre-fix this
    # walked every table in every schema on every call (a
    # 200-table catalog → 400 catalog touches per call) and
    # wrapped the per-table query in a bare ``except Exception``
    # that hid real connectivity errors as "no assets". Now
    # the schema iteration is amortised and the outer try/except
    # is gone — catalog errors surface to the caller.
    for schema_name, table_name in ml_instance.model.find_asset_execution_tables():
        # ``Image_Execution`` → asset_table_name == ``"Image"``.
        asset_table_name = table_name.replace("_Execution", "")
        table_path = pb.schemas[schema_name].tables[table_name]
        query = table_path.filter(table_path.Execution == execution_rid)
        if asset_role:
            query = query.filter(table_path.Asset_Role == asset_role)
        records = list(query.entities().fetch())
        for record in records:
            asset_rid = record.get(asset_table_name)
            if not asset_rid:
                continue
            try:
                assets.append(ml_instance.lookup_asset(asset_rid))
            except Exception as e:
                # Per-row swallow only — one asset row's catalog
                # issue shouldn't break the whole listing. The
                # outer query failure (above) does propagate.
                logger.debug("Could not look up asset %s: %s", asset_rid, e)
    return assets


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
    "list_assets",
    "list_input_datasets",
    "list_input_datasets_with_versions",
    "update_field_in_catalog",
]

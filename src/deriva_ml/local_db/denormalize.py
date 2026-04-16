"""Unified denormalization engine for the local_db layer.

Replaces both ``Dataset._denormalize_datapath`` and ``DatasetBag._denormalize``
with a single function that builds SQLAlchemy JOINs against local SQLite.

The key abstraction is the ``orm_resolver`` callable which maps table names to
SQLAlchemy ORM classes.  This decouples the denormalizer from where the ORM
classes originate:

- For bags: ``database_model.get_orm_class_by_name``
- For the workspace: ``local_schema.get_orm_class``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import pandas as pd
from sqlalchemy import and_, literal, select, union
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deriva_ml.model.catalog import DerivaModel, denormalize_column_name

logger = logging.getLogger(__name__)


@dataclass
class DenormalizeResult:
    """Result of a denormalization operation.

    Attributes:
        columns: ``(name, type)`` pairs describing the output schema.
        row_count: Number of rows returned.
    """

    columns: list[tuple[str, str]]
    row_count: int
    _rows: list[dict[str, Any]] = field(repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result to a :class:`pandas.DataFrame`."""
        if not self._rows:
            # Return an empty DataFrame with the correct column names.
            return pd.DataFrame(columns=[name for name, _ in self.columns])
        return pd.DataFrame(self._rows)

    def iter_rows(self) -> Generator[dict[str, Any], None, None]:
        """Yield each row as a dictionary."""
        yield from self._rows


def denormalize(
    model: DerivaModel,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    dataset_rid: str,
    include_tables: list[str],
    dataset: Any = None,
    dataset_children_rids: list[str] | None = None,
) -> DenormalizeResult:
    """Unified denormalization: plan joins via ``_prepare_wide_table``, execute locally.

    Args:
        model: :class:`DerivaModel` used for join planning
            (``_prepare_wide_table``).
        engine: SQLAlchemy engine with ATTACH'd schemas for SQL execution.
        orm_resolver: Callable mapping *table_name* -> ORM class.  The
            returned classes must have ``__table__`` attributes referencing
            tables visible in *engine*.
        dataset_rid: RID of the dataset to denormalize.
        include_tables: Tables to include in the wide table.
        dataset: ``DatasetLike`` object passed to ``_prepare_wide_table``.
            If ``None``, a minimal mock is used (empty members, no children).
        dataset_children_rids: Extra dataset RIDs for the ``WHERE`` clause.
            If ``None``, only *dataset_rid* is used.

    Returns:
        :class:`DenormalizeResult` with rows and column metadata.
    """
    # Step 1: Plan the join.
    if dataset is None:
        dataset = _MinimalDatasetMock(dataset_rid=dataset_rid)

    join_tables, column_specs, multi_schema = model._prepare_wide_table(dataset, dataset_rid, include_tables)

    # Step 2: Build labelled columns from ORM classes.
    denormalized_columns = []
    for schema_name, table_name, column_name, _type_name in column_specs:
        orm_class = orm_resolver(table_name)
        if orm_class is None:
            logger.warning(
                "ORM class not found for table %s; skipping column %s",
                table_name,
                column_name,
            )
            continue
        col = orm_class.__table__.columns[column_name]
        label = denormalize_column_name(schema_name, table_name, column_name, multi_schema)
        denormalized_columns.append(col.label(label))

    # Pre-compute the output column metadata (used in all return paths).
    output_columns = [(denormalize_column_name(s, t, c, multi_schema), tp) for s, t, c, tp in column_specs]

    if not denormalized_columns:
        return DenormalizeResult(columns=output_columns, row_count=0, _rows=[])

    # Step 3: Build SQL for each element path.
    dataset_rid_list = [dataset_rid] + (dataset_children_rids or [])
    dataset_orm = orm_resolver("Dataset")

    sql_statements = []
    for _key, (path, join_conditions, join_types) in join_tables.items():
        stmt = select(*denormalized_columns).select_from(dataset_orm)

        for table_name in path[1:]:  # Skip "Dataset" (already in select_from).
            if table_name not in join_conditions:
                continue
            on_clause = _build_join_on_clause(join_conditions[table_name], orm_resolver)
            table_class = orm_resolver(table_name)
            if table_class is None:
                continue
            if join_types.get(table_name) == "left":
                stmt = stmt.outerjoin(table_class, onclause=on_clause)
            else:
                stmt = stmt.join(table_class, onclause=on_clause)

        # WHERE Dataset.RID IN (...)
        if dataset_orm is not None:
            stmt = stmt.where(dataset_orm.RID.in_(dataset_rid_list))

        sql_statements.append(stmt)

    if not sql_statements:
        return DenormalizeResult(columns=output_columns, row_count=0, _rows=[])

    # Step 4: Execute.
    final_query = union(*sql_statements) if len(sql_statements) > 1 else sql_statements[0]

    with Session(engine) as session:
        result = session.execute(final_query)
        rows = [dict(row._mapping) for row in result]

    return DenormalizeResult(columns=output_columns, row_count=len(rows), _rows=rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_join_on_clause(
    join_condition_pairs: set,
    orm_resolver: Callable[[str], Any],
):
    """Build a SQLAlchemy ON clause from FK/PK column pairs.

    Each pair is ``(fk_col, pk_col)`` where *fk_col* and *pk_col* are ERMrest
    ``Column`` objects with ``.table.name`` and ``.name`` attributes.  We look
    up the corresponding SQLAlchemy ORM classes via *orm_resolver* and build an
    ``AND`` of column equalities.
    """
    conditions = []
    for fk_col, pk_col in join_condition_pairs:
        fk_table_name = fk_col.table.name if hasattr(fk_col.table, "name") else str(fk_col.table)
        pk_table_name = pk_col.table.name if hasattr(pk_col.table, "name") else str(pk_col.table)
        fk_class = orm_resolver(fk_table_name)
        pk_class = orm_resolver(pk_table_name)
        if fk_class is None or pk_class is None:
            continue
        left = fk_class.__table__.columns[fk_col.name]
        right = pk_class.__table__.columns[pk_col.name]
        conditions.append(left == right)
    return and_(*conditions) if conditions else literal(True)


@dataclass
class _MinimalDatasetMock:
    """Minimal ``DatasetLike`` mock for ``_prepare_wide_table``.

    Returns empty members and no children, which is sufficient when the
    caller doesn't need member-based filtering.
    """

    dataset_rid: str

    def list_dataset_members(self, **kwargs: Any) -> dict:  # noqa: ARG002
        return {}

    def list_dataset_children(self, **kwargs: Any) -> list:  # noqa: ARG002
        return []

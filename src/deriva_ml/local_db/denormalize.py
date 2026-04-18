"""Unified denormalization engine for the local_db layer.

This module hosts the low-level ``_denormalize_impl`` primitive — the SQL
executor that materializes a wide table from a dataset's join chain.

**Public API**: :class:`deriva_ml.local_db.denormalizer.Denormalizer` is the
class-based public API. Callers should construct a ``Denormalizer`` from
their dataset or RID set; the ``Denormalizer`` internally uses
``_denormalize_impl``. The free function below is private.

The primitive replaces both ``Dataset._denormalize_datapath`` and
``DatasetBag._denormalize`` with a single function that builds SQLAlchemy
JOINs against local SQLite.

The key abstraction is the ``orm_resolver`` callable which maps table names to
SQLAlchemy ORM classes.  This decouples the denormalizer from where the ORM
classes originate:

- For bags: ``database_model.get_orm_class_by_name``
- For the workspace: ``local_schema.get_orm_class``

The ``source`` parameter controls how rows get into the local SQLite engine:

- ``"local"`` (default): caller has already populated rows; denormalize runs
  the SQL join against whatever is there. Used by tests with pre-populated
  fixtures.
- ``"catalog"``: denormalize uses a :class:`PagedClient` to fetch rows from a
  live ERMrest catalog into the local engine before running the join. This is
  the production path for :meth:`Dataset.get_denormalized_as_dataframe`.
- ``"slice"``: rows are assumed to already be visible via an attached slice
  database (ATTACH'd into the engine). Used by
  :meth:`DatasetBag.get_denormalized_as_dataframe`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import pandas as pd
from sqlalchemy import and_, literal, select, union
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deriva_ml.local_db.paged_fetcher import PagedClient, PagedFetcher
from deriva_ml.model.catalog import DerivaModel, denormalize_column_name

logger = logging.getLogger(__name__)


@dataclass
class DenormalizeResult:
    """Result of a denormalization operation.

    Holds the materialized wide-table rows and the column schema.  Rows are
    stored in-memory after the SQL query executes; for very large datasets
    consider using :meth:`iter_rows` in a streaming fashion rather than calling
    :meth:`to_dataframe` which loads everything into pandas at once.

    Attributes:
        columns: ``(name, type)`` pairs describing the output schema.  Column
            names use ``Table.Column`` dot notation (or
            ``schema.Table.Column`` when multi-schema).
        row_count: Number of rows in the result (``len(_rows)``).

    Example::

        result = _denormalize_impl(model=m, engine=e, orm_resolver=r,
                                   dataset_rid="DS-001", include_tables=["Image", "Subject"])
        df = result.to_dataframe()       # full DataFrame
        for row in result.iter_rows():   # streaming
            process(row)
    """

    columns: list[tuple[str, str]]
    row_count: int
    _rows: list[dict[str, Any]] = field(repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result to a :class:`pandas.DataFrame`.

        Returns:
            A DataFrame with one column per entry in :attr:`columns`.  Returns
            an empty DataFrame (with correct column names) if there are no rows.
        """
        if not self._rows:
            # Return an empty DataFrame with the correct column names.
            return pd.DataFrame(columns=[name for name, _ in self.columns])
        return pd.DataFrame(self._rows)

    def iter_rows(self) -> Generator[dict[str, Any], None, None]:
        """Yield each row as a dictionary.

        Keys are column names in ``Table.Column`` (or ``schema.Table.Column``)
        dot notation.  Values are raw Python types as returned by SQLAlchemy.
        """
        yield from self._rows

    def extend(self, rows: list[dict[str, Any]]) -> "DenormalizeResult":
        """Return a new :class:`DenormalizeResult` with additional rows appended.

        Used when combining phases of denormalization — specifically,
        :meth:`Denormalizer._run` appends orphan rows (Rule 7 case 3)
        emitted by :meth:`Denormalizer._emit_orphan_rows` to the main
        JOIN result.

        **Immutability**: ``self`` is NOT mutated. Columns and schema are
        shared by reference with the returned instance (they're
        metadata, not per-row data). Only the row list is a fresh copy.

        Args:
            rows: Row dicts to append. Each row should match the shape
                produced by :meth:`iter_rows` — keys are
                ``Table.column`` / ``schema.Table.column`` labels, values
                are raw Python types.

        Returns:
            New :class:`DenormalizeResult` with ``row_count = self.row_count +
            len(rows)`` and concatenated rows.

        Example::

            main_result = _denormalize_impl(...)
            orphan_rows = [{"Image.RID": None, "Subject.RID": "ORPHAN"}]
            combined = main_result.extend(orphan_rows)
            # combined.row_count == main_result.row_count + 1
            # main_result is unchanged
        """
        return DenormalizeResult(
            columns=self.columns,
            row_count=self.row_count + len(rows),
            _rows=list(self._rows) + list(rows),
        )


def _denormalize_impl(
    model: DerivaModel,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    dataset_rid: str,
    include_tables: list[str],
    dataset: Any = None,
    dataset_children_rids: list[str] | None = None,
    source: str = "local",
    paged_client: PagedClient | None = None,
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
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
        source: How rows get into the engine. ``"local"`` (default) assumes
            the caller has pre-populated rows; ``"catalog"`` fetches rows via
            *paged_client* before running the join; ``"slice"`` assumes rows
            are visible via an attached slice database.
        paged_client: Required when ``source="catalog"``. The client used to
            fetch rows from a live ERMrest catalog.
        row_per: Optional leaf table override. When None (default),
            auto-inferred by Rule 2 from sinks in ``include_tables``. See
            :class:`~deriva_ml.local_db.denormalizer.Denormalizer` for
            semantic details.
        via: Optional list of tables forced into the join chain without
            contributing columns. Used to disambiguate path ambiguity
            (Rule 6).

    Returns:
        :class:`DenormalizeResult` with rows and column metadata.

    Raises:
        ValueError: If ``source="catalog"`` but ``paged_client`` is ``None``.
        RuntimeError: If the ``"Dataset"`` ORM class cannot be resolved
            (likely because ``build_local_schema`` was not called with the
            ``deriva-ml`` schema).
        DerivaMLDenormalizeMultiLeaf / NoSink / DownstreamLeaf /
            AmbiguousPath: planner rule violations raised by
            ``_prepare_wide_table``.

    Example::

        # Typical local-mode call (rows already in the engine):
        result = _denormalize_impl(
            model=model,
            engine=engine,
            orm_resolver=local_schema.get_orm_class,
            dataset_rid="DS-001",
            include_tables=["Image", "Subject"],
            source="local",
        )
        df = result.to_dataframe()

        # Catalog-mode: fetches rows via PagedClient before the join.
        result = _denormalize_impl(
            model=model, engine=engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid="DS-001",
            include_tables=["Image", "Subject"],
            source="catalog",
            paged_client=ErmrestPagedClient(catalog),
            row_per="Image",                 # Rule 2 override
            via=["Observation"],             # Rule 6 disambiguation
        )

    Note:
        Most callers should use the higher-level
        :class:`~deriva_ml.local_db.denormalizer.Denormalizer` API
        instead of calling this function directly. The class handles
        anchor classification, orphan-row emission, and source-mode
        selection automatically.
    """
    # Validate source / paged_client combination up front so callers get a
    # clear error before we do any work.
    if source == "catalog" and paged_client is None:
        raise ValueError(
            "paged_client is required when source='catalog'. Pass an ErmrestPagedClient (or compatible) to fetch rows."
        )

    # Step 1: Plan the join.
    if dataset is None:
        dataset = _MinimalDatasetMock(dataset_rid=dataset_rid)

    join_tables, column_specs, multi_schema = model._prepare_wide_table(
        dataset,
        dataset_rid,
        include_tables,
        row_per=row_per,
        via=via,
    )

    # Build a quick lookup from table name to schema name so we can form
    # qualified ERMrest names ("schema:table") for PagedFetcher calls.
    # We can't rely on column_specs alone because association tables (like
    # Dataset_Image) appear in the join path but contribute no output
    # columns — so their schema won't be in column_specs. Use the
    # model.name_to_table() helper to look up every table's schema.
    table_to_schema: dict[str, str] = {}
    for schema_name, table_name, _col, _type in column_specs:
        table_to_schema[table_name] = schema_name
    # Ensure every table that appears in the join plan has a schema entry.
    for _key, (path, _join_conditions, _join_types) in join_tables.items():
        for table_name in path:
            if table_name in table_to_schema:
                continue
            try:
                t = model.name_to_table(table_name)
                table_to_schema[table_name] = t.schema.name
            except Exception:
                logger.warning(
                    "Could not resolve schema for table %s; catalog fetch may skip it.",
                    table_name,
                )

    # Step 2: Build labelled columns from ORM classes.
    # Previously this loop silently skipped columns whose ORM class couldn't
    # be resolved, which produced a result whose actual columns didn't match
    # the advertised `output_columns` metadata. Raise instead so callers get
    # a clear error and we don't hand back inconsistent data.
    denormalized_columns = []
    for schema_name, table_name, column_name, _type_name in column_specs:
        orm_class = orm_resolver(table_name)
        if orm_class is None:
            raise RuntimeError(
                f"ORM class not found for table {table_name!r} (needed for "
                f"column {column_name!r}). This usually means "
                f"build_local_schema() wasn't called with a schema that "
                f"includes this table. Include tables requested: "
                f"{include_tables!r}."
            )
        col = orm_class.__table__.columns[column_name]
        label = denormalize_column_name(schema_name, table_name, column_name, multi_schema)
        denormalized_columns.append(col.label(label))

    # Pre-compute the output column metadata (used in all return paths).
    output_columns = [(denormalize_column_name(s, t, c, multi_schema), tp) for s, t, c, tp in column_specs]

    if not denormalized_columns:
        return DenormalizeResult(columns=output_columns, row_count=0, _rows=[])

    # Step 3: Resolve the Dataset ORM class (required for the WHERE clause).
    # C3: fail loudly if missing so callers get a clear error instead of a
    # cryptic SQLAlchemy crash on select_from(None).
    dataset_orm = orm_resolver("Dataset")
    if dataset_orm is None:
        raise RuntimeError(
            "Dataset ORM class not found — ensure build_local_schema() was called with the 'deriva-ml' schema included."
        )

    dataset_rid_list = [dataset_rid] + (dataset_children_rids or [])

    # Step 3b: If source='catalog', fetch rows into the engine's tables
    # before we run the SQL join. Without this step, the join runs against
    # an empty working DB and returns zero rows.
    if source == "catalog":
        _populate_from_catalog(
            paged_client=paged_client,
            engine=engine,
            orm_resolver=orm_resolver,
            table_to_schema=table_to_schema,
            join_tables=join_tables,
            dataset_rid_list=dataset_rid_list,
        )

    # Step 4: Build SQL for each element path.
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
        stmt = stmt.where(dataset_orm.RID.in_(dataset_rid_list))

        sql_statements.append(stmt)

    if not sql_statements:
        return DenormalizeResult(columns=output_columns, row_count=0, _rows=[])

    # Step 5: Execute.
    final_query = union(*sql_statements) if len(sql_statements) > 1 else sql_statements[0]

    with Session(engine) as session:
        result = session.execute(final_query)
        rows = [dict(row._mapping) for row in result]

    return DenormalizeResult(columns=output_columns, row_count=len(rows), _rows=rows)


def _populate_from_catalog(
    *,
    paged_client: PagedClient,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    table_to_schema: dict[str, str],
    join_tables: dict,
    dataset_rid_list: list[str],
) -> None:
    """Fetch rows from a live catalog into the engine's local tables.

    Walks the join paths in order so that each table's fetch can use the RID
    values already loaded into the preceding table. The algorithm is:

    1. Fetch the Dataset row(s) — always needed for the WHERE clause join.
    2. Walk each join path in order. For each table after Dataset:
       - Read FK values from the already-loaded preceding table(s)
       - Fetch rows for this table filtered by those FK/RID values
       - The PagedFetcher deduplicates so multi-path traversals are safe.

    This uses ``fetch_by_rids`` with a configurable ``rid_column`` so both
    FK-by-target (e.g., Dataset_Image filtered by Dataset FK) and FK-by-RID
    (e.g., Image filtered by RID) are handled uniformly.

    Args:
        paged_client: The client used for all catalog HTTP calls.
        engine: Local SQLAlchemy engine (rows are inserted into this engine's
            tables via the ORM class returned by orm_resolver).
        orm_resolver: Maps table name -> ORM class (its ``__table__`` is the
            write target).
        table_to_schema: Maps table name -> ERMrest schema name (used to
            build ``"schema:table"`` qualified names).
        join_tables: Output from ``_prepare_wide_table`` — dict keyed by leaf
            table name, values are ``(path, join_conditions, join_types)``
            where ``path`` is a list of table names in join order starting
            with "Dataset".
        dataset_rid_list: RIDs to scope the denormalization to (the root
            dataset plus any children from recursive traversal).
    """
    fetcher = PagedFetcher(client=paged_client, engine=engine)

    # --- Step 1: Fetch the Dataset rows themselves -------------------------
    # These are needed so the WHERE Dataset.RID IN (...) clause finds the
    # rows during the SQL join.
    dataset_orm = orm_resolver("Dataset")
    dataset_schema = table_to_schema.get("Dataset", "deriva-ml")
    fetcher.fetch_by_rids(
        table=f"{dataset_schema}:Dataset",
        rids=dataset_rid_list,
        target_table=dataset_orm.__table__,
        rid_column="RID",
    )

    # --- Step 2: Walk each join path, fetching each table in turn ----------
    # We process tables in the order they appear along each path. Already-
    # fetched tables are skipped (PagedFetcher dedup handles this per-table
    # via its internal _seen map, but we also dedupe at the path level to
    # avoid unnecessary method-call overhead).
    processed: set[str] = {"Dataset"}

    for _key, (path, join_conditions, _join_types) in join_tables.items():
        # Walk in order — each table depends on rows loaded by the previous.
        prior_tables_in_path: list[str] = ["Dataset"]
        for table_name in path[1:]:
            if table_name in processed:
                prior_tables_in_path.append(table_name)
                continue

            target_orm = orm_resolver(table_name)
            if target_orm is None:
                logger.warning("Skipping fetch for %s: no ORM class resolved", table_name)
                prior_tables_in_path.append(table_name)
                continue

            target_schema = table_to_schema.get(table_name)
            if target_schema is None:
                logger.warning("Skipping fetch for %s: no schema known", table_name)
                prior_tables_in_path.append(table_name)
                continue

            qualified = f"{target_schema}:{table_name}"

            # To fetch rows of this table, we need to know which RIDs to
            # request. Look at the join_conditions[table_name] entry which
            # is a set of (fk_col, pk_col) pairs. For each pair:
            #   - fk_col is on one of the tables we've already loaded
            #   - pk_col is on `table_name` (the one we're about to fetch)
            # We collect the values of fk_col from the local DB, then fetch
            # rows of table_name where the pk_col equals any of those values.
            conditions = join_conditions.get(table_name, set())
            rids_to_fetch, fk_column_on_target = _collect_fk_values(
                engine=engine,
                orm_resolver=orm_resolver,
                conditions=conditions,
                target_table_name=table_name,
            )

            if rids_to_fetch and fk_column_on_target is not None:
                # Convert any non-string values (e.g., integer RIDs) to
                # strings since PagedFetcher's Iterable[str] contract expects
                # stringified RID values.
                str_rids = [str(r) for r in rids_to_fetch if r is not None]
                if str_rids:
                    fetcher.fetch_by_rids(
                        table=qualified,
                        rids=str_rids,
                        target_table=target_orm.__table__,
                        rid_column=fk_column_on_target,
                    )

            processed.add(table_name)
            prior_tables_in_path.append(table_name)


def _collect_fk_values(
    *,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    conditions: set,
    target_table_name: str,
) -> tuple[list[Any], str | None]:
    """Determine which rows of *target_table_name* we need to fetch.

    Given the join conditions for *target_table_name* (pairs of
    (fk_col, pk_col) where fk_col lives on one side and pk_col on the other),
    figure out:

    - Which column on the target table is the one we filter by, and
    - What values of that column we need (pulled from already-loaded rows on
      the other side of the join).

    Returns ``(values, column_name)`` where ``column_name`` is the name of the
    filter column on *target_table_name*, or ``(empty, None)`` if no workable
    condition is found.
    """
    for fk_col, pk_col in conditions:
        # Each pair has one column that belongs to target_table_name (the
        # "far side" for this join) and one that belongs to an already-loaded
        # table. Figure out which is which.
        fk_table_name = _col_table_name(fk_col)
        pk_table_name = _col_table_name(pk_col)

        if fk_table_name == target_table_name:
            # FK is on the target table, PK is on the prior table.
            # e.g., Image has FK "Subject" -> Subject.RID
            # We fetch Images where Image.Subject matches loaded Subject.RIDs.
            other_table_name = pk_table_name
            filter_col_on_target = fk_col.name
            pull_col_name = pk_col.name
        elif pk_table_name == target_table_name:
            # PK is on the target table, FK is on the prior table.
            # e.g., Dataset_Image.Image -> Image.RID
            # We fetch Images by RID where those RIDs come from loaded
            # Dataset_Image rows' Image column.
            other_table_name = fk_table_name
            filter_col_on_target = pk_col.name
            pull_col_name = fk_col.name
        else:
            # Neither side of the condition is the target table — skip.
            continue

        other_orm = orm_resolver(other_table_name)
        if other_orm is None:
            continue

        # Query the local DB for distinct non-null values of pull_col_name
        # on the other-side ORM table.
        other_col = other_orm.__table__.columns.get(pull_col_name)
        if other_col is None:
            continue
        stmt = select(other_col).distinct().where(other_col.isnot(None))
        with engine.connect() as conn:
            values = [row[0] for row in conn.execute(stmt).fetchall() if row[0] is not None]

        if values:
            return values, filter_col_on_target

    return [], None


def _col_table_name(col: Any) -> str:
    """Return the ERMrest table name for an ERMrest ``Column`` object."""
    table = getattr(col, "table", None)
    if table is None:
        return ""
    return getattr(table, "name", str(table))


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
    caller doesn't need member-based filtering (i.e., when the denormalizer
    is driven purely by ``dataset_rid`` without a live Dataset object).

    This exists to avoid importing the full ``Dataset`` class inside the
    local_db layer, keeping the dependency graph simple.
    """

    dataset_rid: str

    def list_dataset_members(self, **kwargs: Any) -> dict:  # noqa: ARG002
        """Return an empty member dict (no members known without a live catalog)."""
        return {}

    def list_dataset_children(self, **kwargs: Any) -> list:  # noqa: ARG002
        """Return an empty children list (no children known without a live catalog)."""
        return []

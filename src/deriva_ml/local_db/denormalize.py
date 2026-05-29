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

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Callable, Generator

import pandas as pd
from sqlalchemy import and_, event, literal, select, union
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from deriva_ml.core.logging_config import get_logger
from deriva_ml.local_db.paged_fetcher import PagedClient, PagedFetcher
from deriva_ml.model.catalog import DerivaModel, denormalize_column_name

logger = get_logger(__name__)


@contextmanager
def _foreign_keys_off(engine: Engine):
    """Temporarily disable ``PRAGMA foreign_keys`` on every checkout.

    SQLite enforces FK constraints per-connection. The local-db engine
    is a transport mirror, not the authoritative store — its FK
    constraints exist to *describe* the schema for the ORM layer, not
    to police consistency. When :func:`_populate_from_catalog` walks
    join paths in data-dependency order it can legitimately insert a
    referencing row before its referent (the data flow says we need
    the parent's RIDs *from the child* to know what to fetch). Real
    integrity comes from the source ERMrest catalog the rows arrived
    from.

    The hook fires on every pool checkout; ``create_wal_engine``'s
    connect-time hook still sets ``foreign_keys=ON`` once per
    physical connect, but our checkout-time hook overrides it for the
    duration of the ``with`` block. On exit we remove the hook so
    later callers regain normal FK enforcement.

    Args:
        engine: SQLAlchemy engine to patch (scoped to this ``with``).
    """

    def _off(dbapi_conn, _record, _proxy):
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA foreign_keys = OFF")
        finally:
            cur.close()

    event.listen(engine, "checkout", _off)
    try:
        yield
    finally:
        event.remove(engine, "checkout", _off)


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
        cache_age_seconds: Caller-visible freshness signal (SC-03, spec §6
            "Freshness caveat"). The wall-clock time delta, in seconds,
            between the start of this denormalize call and the *earliest*
            catalog fetch that participated in the SQL JOIN that built this
            result. ``None`` when no live catalog fetch happened
            (``source='bag'`` / ``source='local'`` / ``source='slice'``, or
            ``source='catalog'`` with all keys already deduped against
            cache — though in practice the dataset-row fetch always
            stamps the ledger). A user re-running denormalize in a
            long-lived process can use this to detect that results draw
            on cached data older than they're comfortable with — the
            local SQLite cache is write-through and does NOT observe
            server-side deletions or updates (see ``docs/user-guide/denormalization.md``
            §6.5 freshness caveat and §7 F3/F4).

    Example::

        result = _denormalize_impl(model=m, engine=e, orm_resolver=r,
                                   dataset_rid="DS-001", include_tables=["Image", "Subject"])
        df = result.to_dataframe()       # full DataFrame
        for row in result.iter_rows():   # streaming
            process(row)
        if result.cache_age_seconds is not None and result.cache_age_seconds > 600:
            warn("denormalize result built from cache >10min old")
    """

    columns: list[tuple[str, str]]
    row_count: int
    _rows: list[dict[str, Any]] = field(repr=False)
    cache_age_seconds: float | None = None

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
            cache_age_seconds=self.cache_age_seconds,
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
    selector: Callable[[list[Any]], Any] | None = None,
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
        selector: Optional callable
            ``(list[FeatureRecord]) -> FeatureRecord | None`` used to reduce
            multi-row feature groups after materialization. When provided,
            ``include_tables`` must contain exactly one feature-association
            table; materialized rows are grouped by the feature's target RID
            and the selector picks one row per group (or returns ``None``
            to omit the group). Same contract as
            :meth:`~deriva_ml.core.mixins.feature.FeatureMixin.feature_values`'s
            ``selector`` argument. See ``FeatureRecord`` for built-in
            selectors (``select_newest``, ``select_first``, etc.).

    Returns:
        :class:`DenormalizeResult` with rows and column metadata.

    Raises:
        ValueError: If ``source="catalog"`` but ``paged_client`` is ``None``;
            if ``selector`` is given and ``include_tables`` contains zero or
            more than one feature-association table.
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

    # Validate selector / include_tables shape up front. The selector
    # reduction in Step 6 needs exactly one feature-association table to
    # group rows by — zero gives nothing to reduce, more than one would
    # require a per-feature ``dict[name, selector]`` shape (future Stage 2
    # extension; not in this stage's scope).
    feature_assoc_table: str | None = None
    if selector is not None:
        feature_assoc_tables = [t for t in include_tables if model._planner._is_feature_association(t)]
        if not feature_assoc_tables:
            raise ValueError(
                "selector requires a feature-association table in include_tables; "
                f"none found in {include_tables!r}. Pass include_tables that contains "
                "exactly one feature-association table (or the feature name that resolves "
                "to one), or drop the selector."
            )
        if len(feature_assoc_tables) > 1:
            raise ValueError(
                "selector with multiple feature-association tables not yet supported "
                f"({feature_assoc_tables!r}); pass include_tables with one feature at a time."
            )
        feature_assoc_table = feature_assoc_tables[0]

    # Step 1: Plan the join.
    if dataset is None:
        dataset = _MinimalDatasetMock(dataset_rid=dataset_rid)

    join_tables, column_specs, multi_schema = model._planner._prepare_wide_table(
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
    #
    # The returned fetcher carries its freshness ledger
    # (``fetcher.fetch_ledger``) — a map of ``(table, rid_column, frozenset(rids))``
    # to the ``time.monotonic()`` of the first fetch attempt for that key.
    # We use the oldest timestamp in the ledger to compute
    # :attr:`DenormalizeResult.cache_age_seconds` after the SQL JOIN
    # materializes (SC-03; spec §6 freshness caveat).
    fetcher: PagedFetcher | None = None
    if source == "catalog":
        fetcher = _populate_from_catalog(
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

    # Compute the freshness signal (SC-03). `cache_age_seconds` is the
    # wall-clock delta between *now* (just before we hand back the result)
    # and the earliest fetch timestamp in the fetcher's ledger. `None`
    # when no live catalog fetch happened (`source != "catalog"`, or
    # `source == "catalog"` but no fetches were recorded — empty join
    # plan, etc.). See :attr:`DenormalizeResult.cache_age_seconds` for
    # the user-facing contract.
    cache_age_seconds = _compute_cache_age_seconds(fetcher)

    if not sql_statements:
        return DenormalizeResult(
            columns=output_columns,
            row_count=0,
            _rows=[],
            cache_age_seconds=cache_age_seconds,
        )

    # Step 5: Execute.
    final_query = union(*sql_statements) if len(sql_statements) > 1 else sql_statements[0]

    with Session(engine) as session:
        result = session.execute(final_query)
        rows = [dict(row._mapping) for row in result]

    # Step 6: Apply selector reduction (Stage 1 of the
    # Denormalizer / feature_values consolidation). When a selector is
    # provided, group materialized rows by their feature-assoc target RID
    # and let the selector pick one row per group. Same "filter then
    # reduce" ordering used by ``Dataset.feature_values`` — materialize
    # first, then reduce.
    if selector is not None and feature_assoc_table is not None and rows:
        rows = _apply_selector(
            rows=rows,
            model=model,
            engine=engine,
            orm_resolver=orm_resolver,
            feature_assoc_table=feature_assoc_table,
            schema_for_table=table_to_schema,
            multi_schema=multi_schema,
            selector=selector,
        )

    return DenormalizeResult(
        columns=output_columns,
        row_count=len(rows),
        _rows=rows,
        cache_age_seconds=cache_age_seconds,
    )


def _apply_selector(
    rows: list[dict[str, Any]],
    *,
    model: DerivaModel,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    feature_assoc_table: str,
    schema_for_table: dict[str, str],
    multi_schema: bool,
    selector: Callable[[list[Any]], Any],
) -> list[dict[str, Any]]:
    """Group materialized denormalize rows by feature target RID and reduce.

    Shared selector application for the ``Denormalizer`` surface
    (Stage 1 of the ``feature_values`` / ``Denormalizer`` consolidation).

    For each materialized row, builds a ``FeatureRecord`` instance from
    the feature-association columns and runs
    :func:`~deriva_ml.feature.reduce_with_selector` over groups keyed by
    the feature's target RID. The selected ``FeatureRecord`` is mapped
    back to its source row so the output preserves the full wide-table
    shape (all ``include_tables`` columns, not just the feature ones).

    Args:
        rows: Materialized denormalize rows (``Table.column`` /
            ``schema.Table.column`` labelled dicts) from the SQL JOIN.
        model: :class:`DerivaModel` used to look up the
            :class:`~deriva_ml.feature.Feature` for
            ``feature_assoc_table`` and the schema name for label
            construction.
        feature_assoc_table: Name of the feature-association table
            in ``include_tables`` (already validated by the caller).
        schema_for_table: ``{table_name: schema_name}`` map produced by
            ``_denormalize_impl`` for use with
            :func:`denormalize_column_name`.
        multi_schema: Whether output labels carry the schema prefix.
        selector: Callable
            ``(list[FeatureRecord]) -> FeatureRecord | None`` — the
            same shape as :meth:`Dataset.feature_values`'s ``selector``.

    Returns:
        Reduced rows — one per target RID for which the selector
        returned a non-``None`` choice. Order follows the surviving
        target RIDs in dict iteration order (insertion order of
        first-seen target RID in the input rows). When the selector
        drops every group, returns an empty list.

    Raises:
        DerivaMLException: If the feature-association table can't be
            resolved to a :class:`~deriva_ml.feature.Feature` (e.g. the
            model exposes no ``find_features`` or the table name doesn't
            match any feature).

    Example::

        # Materialized rows from Execution_Image_Quality + Image.
        reduced = _apply_selector(
            rows=rows,
            model=model,
            feature_assoc_table="Execution_Image_Quality",
            schema_for_table={"Image": "domain", "Execution_Image_Quality": "deriva-ml"},
            multi_schema=False,
            selector=FeatureRecord.select_newest,
        )
        # ``reduced`` has one row per Image RID — the newest Quality
        # record per Image — with the rest of the wide-table columns
        # preserved.
    """
    from collections import defaultdict

    from deriva_ml.core.exceptions import DerivaMLException

    # Locate the target table for the feature-association table. The
    # preferred path is ``DerivaModel.find_features`` — that returns
    # canonical :class:`Feature` objects with a real
    # ``feature_record_class()``, keeping this code in lockstep with
    # ``Dataset.feature_values``. Some model shapes (minimal offline
    # fixtures whose ``Model.fromfile`` doesn't wire ``referenced_by``)
    # don't surface features that way, so we fall back to a structural
    # lookup on the feature-association table's own foreign keys.
    find_feats = getattr(model, "find_features", None)
    feature = None
    if callable(find_feats):
        try:
            feature = next(
                (f for f in find_feats() if f.feature_table.name == feature_assoc_table),
                None,
            )
        except Exception:
            feature = None

    if feature is not None:
        record_class = feature.feature_record_class()
        target_table_name = feature.target_table.name
        field_names = set(record_class.model_fields.keys())
    else:
        # Structural fallback. The planner's
        # ``_is_feature_association`` predicate already guaranteed the
        # table has the feature-association shape (Execution FK +
        # target FK + value FK). Walk the FKs, identify the target
        # (the one that is neither Execution nor a vocabulary/asset),
        # and synthesize a minimal FeatureRecord subclass so the
        # selector contract still resolves.
        from typing import Optional

        from pydantic import create_model

        from deriva_ml.feature import FeatureRecord

        try:
            feat_tbl = model.name_to_table(feature_assoc_table)
        except Exception as e:
            raise DerivaMLException(
                f"Cannot apply selector: {feature_assoc_table!r} is not a "
                f"known table in the catalog model ({type(e).__name__}: {e})."
            ) from e
        ml_schema = getattr(model, "ml_schema", "deriva-ml")
        domain_fks = [fk for fk in feat_tbl.foreign_keys if fk.pk_table.name not in ("ERMrest_Client", "ERMrest_Group")]
        target_fk = None
        for fk in domain_fks:
            pk_table = fk.pk_table
            if pk_table.name == "Execution" and pk_table.schema.name == ml_schema:
                continue
            is_vocab = getattr(model, "is_vocabulary", None)
            if callable(is_vocab):
                try:
                    if is_vocab(pk_table):
                        continue
                except Exception:
                    pass
            is_asset = getattr(model, "is_asset", None)
            if callable(is_asset):
                try:
                    if is_asset(pk_table):
                        continue
                except Exception:
                    pass
            target_fk = fk
            break
        if target_fk is None:
            raise DerivaMLException(
                f"Cannot apply selector: could not identify the target FK on "
                f"{feature_assoc_table!r}. The table's domain FKs do not match "
                f"the feature-association shape (Execution + target + value)."
            )
        target_table_name = target_fk.pk_table.name
        # Build a minimal subclass keyed by the feature-assoc table's
        # columns. Field types collapse to ``Optional[str]`` — the
        # built-in selectors read string-shaped fields (``RCT``
        # lexicographic compare, ``Execution`` equality), so this is
        # type-safe for the contract. Customer selectors that need
        # typed feature columns should use ``feature_values`` or
        # ``lookup_feature``-built records instead.
        record_fields: dict[str, Any] = {}
        for col in feat_tbl.columns:
            if col.name in {"RID", "RMB", "RCB", "RMT", "Execution", "Feature_Name", "RCT"}:
                continue
            record_fields[col.name] = (Optional[str], None)
        # Target FK column is required so grouping always finds a key.
        record_fields[target_table_name] = (str, ...)
        feature_name_default = feat_tbl.columns["Feature_Name"].default or "default"
        record_fields["Feature_Name"] = (str, feature_name_default)
        record_class = create_model(  # type: ignore[call-overload]
            f"_Denormalize_{feature_assoc_table}_Record",
            __base__=FeatureRecord,
            **record_fields,
        )
        field_names = set(record_class.model_fields.keys())

    # Resolve the wide-table column prefix for the feature-assoc table.
    schema_name = schema_for_table.get(feature_assoc_table, "")

    def _label(col_name: str) -> str:
        return denormalize_column_name(schema_name, feature_assoc_table, col_name, multi_schema)

    target_label = _label(target_table_name)
    rid_label = _label("RID")

    # Recover the system columns the planner skipped (RCT, etc.) keyed by
    # the feature-assoc RID. Shared with ``Dataset.feature_values``'s
    # delegation via :func:`_recover_system_columns`.
    supplements = _recover_system_columns(
        rows=rows,
        engine=engine,
        orm_resolver=orm_resolver,
        feature_assoc_table=feature_assoc_table,
        rid_label=rid_label,
        field_names=field_names,
    )

    # Group rows by target RID. We carry source rows alongside their
    # built FeatureRecord shadow so the selector's choice can be mapped
    # back to a full wide-table row. Rows whose target RID is missing
    # (NULL on a LEFT JOIN, etc.) can't be grouped and are passed
    # through unchanged — mirrors ``reduce_with_selector``, which
    # silently drops records with no target.
    groups: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
    untouched: list[dict[str, Any]] = []
    for row in rows:
        target_rid = row.get(target_label)
        if target_rid is None:
            untouched.append(row)
            continue
        # Build a FeatureRecord shadow from just the feature-assoc
        # columns. The generated record class is constructed by
        # ``Feature.feature_record_class`` with ``extra="forbid"``
        # inherited from ``FeatureRecord.Config``, so we feed only the
        # fields the class declares — other include_tables columns are
        # excluded. Supplementary fetch values overlay onto the
        # wide-table row contributions, so the FeatureRecord sees RCT
        # etc. when the wide table dropped them.
        record_kwargs: dict[str, Any] = {}
        for field_name in field_names:
            label = _label(field_name)
            if label in row:
                record_kwargs[field_name] = row[label]
        supp = supplements.get(row.get(rid_label), {})
        for field_name, value in supp.items():
            if field_name in field_names and record_kwargs.get(field_name) is None:
                record_kwargs[field_name] = value
        try:
            record = record_class(**record_kwargs)
        except Exception:
            # If a row can't be coerced into the FeatureRecord (e.g.,
            # unexpected NULL where the schema declares non-null), keep
            # the raw row in untouched so the selector path doesn't
            # silently eat it.
            untouched.append(row)
            continue
        groups[target_rid].append((record, row))

    if not groups:
        # No feature rows present (every materialized row was a
        # passthrough). Return verbatim so this behaves identically to
        # selector=None on a result with no feature side.
        return untouched

    # Apply the selector per-group and map the chosen FeatureRecord
    # back to its source row by identity. Built-in selectors
    # (``select_newest``, ``select_by_execution``, etc.) always return
    # one of the inputs, so ``is`` comparison resolves the original
    # row deterministically. ``reduce_with_selector`` follows the same
    # contract; we inline the loop here instead of delegating so we can
    # walk (record, row) pairs together rather than threading an index
    # alongside a record-only iteration.
    reduced: list[dict[str, Any]] = []
    for pairs in groups.values():
        chosen = selector([rec for rec, _ in pairs])
        if chosen is None:
            continue
        match = next((row for rec, row in pairs if rec is chosen), None)
        if match is not None:
            reduced.append(match)

    # Preserve passthrough rows (no target RID) — they never
    # participated in reduction so it would be wrong to drop them
    # silently. A caller that wants them gone can filter the result.
    reduced.extend(untouched)
    return reduced


def _recover_system_columns(
    *,
    rows: list[dict[str, Any]],
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    feature_assoc_table: str,
    rid_label: str,
    field_names: set[str],
) -> dict[str, dict[str, Any]]:
    """Fetch the system columns the wide-table SELECT dropped, keyed by RID.

    The denormalize planner (``_prepare_wide_table``) skips the system
    columns ``{RCT, RMT, RCB, RMB}`` on every contributing table, so a
    materialized wide-table row carries the feature-association table's
    domain columns but not its ``RCT``. The selector built-ins read
    ``RCT`` (``select_newest`` / ``select_first`` compare it
    lexicographically) and ``Dataset.feature_values`` exposes ``RCT`` on
    every yielded ``FeatureRecord`` — both need it back.

    This is a single ``SELECT * FROM <feat_assoc> WHERE RID IN (:rids)``
    against the local engine, returning ``{feature_assoc_RID:
    {field: value}}`` for the ``FeatureRecord`` fields the wide table did
    not already carry. Timestamp-shaped values (``RCT`` may come back from
    SQLite as a Python ``datetime`` depending on column affinity) are
    normalized to ISO-8601 strings so the selector built-ins and the
    ``FeatureRecord.RCT: Optional[str]`` schema see the same shape the
    PathBuilder path materializes.

    Shared by :func:`_apply_selector` (the ``Denormalizer`` selector path)
    and :func:`materialize_feature_records` (the ``Dataset.feature_values``
    delegation) so RCT recovery lives in exactly one place.

    Args:
        rows: Materialized wide-table rows from the SQL JOIN.
        engine: SQLAlchemy engine to read the feature-association table.
        orm_resolver: Maps ``feature_assoc_table`` -> ORM class.
        feature_assoc_table: Name of the feature-association table.
        rid_label: The dotted output label for the feature-assoc table's
            ``RID`` column (e.g. ``"Execution_Image_Quality.RID"``).
        field_names: ``FeatureRecord`` field names — the wanted columns
            (minus ``feature`` / ``Feature_Name``) are pulled.

    Returns:
        ``{feature_assoc_RID: {field_name: coerced_value}}``. Empty when
        no feature RIDs are present in ``rows`` or the supplementary
        fetch fails (best-effort — a failure is logged, not raised).

    Example::

        supplements = _recover_system_columns(
            rows=rows, engine=engine, orm_resolver=resolver,
            feature_assoc_table="Execution_Image_Quality",
            rid_label="Execution_Image_Quality.RID",
            field_names={"Image", "Quality", "RCT", "Execution"},
        )
        # supplements["3-ABC"] == {"RCT": "2026-05-28T16:54:02+00:00", ...}
    """
    feature_rids = [row.get(rid_label) for row in rows if row.get(rid_label) is not None]
    supplements: dict[str, dict[str, Any]] = {}
    if not feature_rids:
        return supplements

    from sqlalchemy import select as sa_select

    feat_orm = orm_resolver(feature_assoc_table)
    if feat_orm is None:
        return supplements

    # Pull the FeatureRecord fields the wide table doesn't already carry.
    # The wide table preserves the feature-association table's domain
    # columns; we only need the system columns the planner skipped.
    wanted_cols = [c for c in field_names if c not in {"feature", "Feature_Name"}]
    try:
        with engine.connect() as conn:
            rows_supp = (
                conn.execute(sa_select(feat_orm.__table__).where(feat_orm.__table__.c.RID.in_(feature_rids)))
                .mappings()
                .all()
            )
        for r in rows_supp:
            rid = r.get("RID")
            if rid is None:
                continue
            coerced: dict[str, Any] = {}
            for col in wanted_cols:
                if col not in r:
                    continue
                val = r.get(col)
                if hasattr(val, "isoformat"):
                    # Deriva ``timestamptz`` columns (RCT, etc.) are stored
                    # as UTC, and the PathBuilder ``feature_values`` path
                    # surfaces them as ISO-8601 strings WITH a ``+00:00``
                    # offset. The local SQLite round-trip drops the tzinfo
                    # (the value comes back tz-naive), so a bare
                    # ``.isoformat()`` would emit ``...232194`` instead of
                    # ``...232194+00:00`` and diverge from the legacy shape.
                    # Re-attach UTC on naive datetimes so the recovered RCT
                    # is bit-identical to the PathBuilder output (selectors
                    # compare RCT lexicographically — a missing offset also
                    # breaks ordering against catalog-shaped timestamps).
                    if getattr(val, "tzinfo", None) is None:
                        val = val.replace(tzinfo=timezone.utc)
                    val = val.isoformat()
                coerced[col] = val
            supplements[rid] = coerced
    except Exception as e:
        # Best-effort: if the fetch fails, callers fall back to whatever
        # fields the wide-table row provides. Log so operators can see
        # the path was exercised but degraded.
        logger.warning(
            "Denormalizer: supplementary fetch of %s system columns failed (%s); "
            "RCT and other skipped fields may be None.",
            feature_assoc_table,
            e,
        )
    return supplements


def materialize_feature_records(
    rows: list[dict[str, Any]],
    *,
    record_class: type,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    feature_assoc_table: str,
    target_table_name: str,
    schema_for_table: dict[str, str],
    multi_schema: bool,
) -> list[Any]:
    """Materialize wide-table denormalize rows into typed ``FeatureRecord`` instances.

    The strip-prefix-and-project adapter from audit finding 08 §4: the
    wide-table rows carry dotted column labels
    (``Execution_Image_Quality.Quality``) plus the target table's own
    columns (``Image.URL``, etc.). This function:

    1. Strips the feature-association table's dotted prefix.
    2. Projects down to only the keys in ``record_class.model_fields``
       (so ``Image.URL`` and other non-FeatureRecord columns are dropped —
       ``FeatureRecord`` is ``extra="forbid"``).
    3. Overlays the recovered system columns (RCT) from
       :func:`_recover_system_columns`.
    4. Constructs ``record_class(**projected)``.

    Rows whose target FK is ``None`` are dropped — matching the
    ``reduce_with_selector`` / PathBuilder semantics where a feature row
    with a NULL target is never surfaced (audit §10). This guards against
    the Denormalizer's LEFT-JOIN emit producing an orphan row with
    ``target=None`` that the PathBuilder path could never return.

    Args:
        rows: Materialized wide-table rows (no selector reduction applied
            — that happens upstream of this call if requested).
        record_class: The ``FeatureRecord`` subclass from
            ``Feature.feature_record_class()``.
        engine: SQLAlchemy engine for the RCT supplementary fetch.
        orm_resolver: Maps table name -> ORM class.
        feature_assoc_table: Name of the feature-association table (the
            column-label prefix to strip).
        target_table_name: Name of the feature's target table (its FK
            column on the feature-assoc table — used for the null-target
            drop).
        schema_for_table: ``{table_name: schema_name}`` for label building.
        multi_schema: Whether output labels carry the schema prefix.

    Returns:
        List of ``record_class`` instances — one per input row whose
        target FK is non-null. Each has ``RCT`` populated from the
        supplementary fetch (the wide table drops it).

    Example::

        recs = materialize_feature_records(
            rows,
            record_class=feat.feature_record_class(),
            engine=engine, orm_resolver=resolver,
            feature_assoc_table="Execution_Image_Quality",
            target_table_name="Image",
            schema_for_table={"Image": "domain", "Execution_Image_Quality": "deriva-ml"},
            multi_schema=False,
        )
        # recs[0].Image == "1-ABC"; recs[0].RCT == "2026-05-28T..."
    """
    field_names = set(record_class.model_fields.keys())
    schema_name = schema_for_table.get(feature_assoc_table, "")

    def _label(col_name: str) -> str:
        return denormalize_column_name(schema_name, feature_assoc_table, col_name, multi_schema)

    target_label = _label(target_table_name)
    rid_label = _label("RID")

    supplements = _recover_system_columns(
        rows=rows,
        engine=engine,
        orm_resolver=orm_resolver,
        feature_assoc_table=feature_assoc_table,
        rid_label=rid_label,
        field_names=field_names,
    )

    records: list[Any] = []
    for row in rows:
        # Null target FK → drop, matching reduce_with_selector / PathBuilder
        # (audit §10). The Denormalizer's LEFT-JOIN emit could otherwise
        # surface a feature row with target=None that the PathBuilder path
        # never returns.
        if row.get(target_label) is None:
            continue
        # Strip prefix + project down to FeatureRecord fields only.
        record_kwargs: dict[str, Any] = {}
        for field_name in field_names:
            label = _label(field_name)
            if label in row:
                record_kwargs[field_name] = row[label]
        # Overlay recovered system columns (RCT) where the wide table
        # left a gap.
        supp = supplements.get(row.get(rid_label), {})
        for field_name, value in supp.items():
            if field_name in field_names and record_kwargs.get(field_name) is None:
                record_kwargs[field_name] = value
        records.append(record_class(**record_kwargs))
    return records


def _compute_cache_age_seconds(fetcher: PagedFetcher | None) -> float | None:
    """Compute the freshness signal from a fetcher's ledger.

    The contract (SC-03, spec §6 freshness caveat): the wall-clock time
    delta between *now* and the *earliest* fetch attempt the fetcher
    recorded. Returns ``None`` when no fetch happened (``fetcher is
    None`` — bag/local/slice sources don't construct a fetcher) or when
    the ledger is empty (catalog source with an empty join plan or a
    no-op call).

    Args:
        fetcher: The :class:`PagedFetcher` that populated rows for this
            denormalize call, or ``None`` if no live fetch happened.

    Returns:
        Wall-clock age in seconds of the oldest participating fetch, or
        ``None`` if no fetch was recorded.

    Example::

        cache_age = _compute_cache_age_seconds(fetcher)
        # cache_age >= 0 if at least one fetch happened
        # cache_age is None for source != "catalog"
    """
    if fetcher is None:
        return None
    ledger = fetcher.fetch_ledger
    if not ledger:
        return None
    return time.monotonic() - min(ledger.values())


def _populate_from_catalog(
    *,
    paged_client: PagedClient,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    table_to_schema: dict[str, str],
    join_tables: dict,
    dataset_rid_list: list[str],
) -> PagedFetcher:
    """Fetch rows from a live catalog into the engine's local tables.

    Data-dependency order: each table's fetch needs RID values that come
    from rows we've already loaded (e.g., to fetch ``Image`` we read
    ``Dataset_Image.Image`` from the local DB). The join-path walk gives
    us this naturally — we walk each path's tables in join order and
    fetch each table once.

    FK enforcement is disabled for the duration of the load via
    :func:`_foreign_keys_off`, so a join path that visits a referencing
    table before its referent (e.g. ``Dataset → Dataset_Image → Image``,
    which inserts ``Dataset_Image`` before ``Image``) does not raise on
    SQLite. The constraint is re-enabled on exit.

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

    Returns:
        The :class:`PagedFetcher` used for the catalog fetches. The
        caller reads its :attr:`PagedFetcher.fetch_ledger` to compute
        :attr:`DenormalizeResult.cache_age_seconds` (SC-03; spec §6
        freshness caveat).
    """
    fetcher = PagedFetcher(client=paged_client, engine=engine)

    # FK enforcement is off for the whole load — see
    # :func:`_foreign_keys_off` for the rationale. Re-enabled on exit.
    with _foreign_keys_off(engine):
        _populate_from_catalog_inner(
            fetcher=fetcher,
            engine=engine,
            orm_resolver=orm_resolver,
            table_to_schema=table_to_schema,
            join_tables=join_tables,
            dataset_rid_list=dataset_rid_list,
        )

    return fetcher


def _populate_from_catalog_inner(
    *,
    fetcher: PagedFetcher,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    table_to_schema: dict[str, str],
    join_tables: dict,
    dataset_rid_list: list[str],
) -> None:
    """Inner walk for :func:`_populate_from_catalog`.

    Extracted so the FK-off context manager wraps a clean inner
    function — easier to reason about than ``try/finally`` inline.
    """
    # --- Step 1: Fetch the Dataset rows themselves -------------------------
    # These are needed so the WHERE Dataset.RID IN (...) clause finds the
    # rows during the SQL join.
    dataset_orm = orm_resolver("Dataset")
    dataset_schema = table_to_schema.get("Dataset", "deriva-ml")
    # Stamp the freshness ledger BEFORE the fetch so the timestamp is the
    # fetch *start*, not the end (SC-03). The ledger key uses the
    # unqualified table name to match the dedup-processed key shape used
    # in Step 2 below — see the row-completeness invariant block.
    fetcher.record_fetch_start("Dataset", "RID", dataset_rid_list)
    fetcher.fetch_by_rids(
        table=f"{dataset_schema}:Dataset",
        rids=dataset_rid_list,
        target_table=dataset_orm.__table__,
        rid_column="RID",
    )

    # --- Step 2: Walk each join path, fetching each table in turn ----------
    # The row-completeness invariant (spec §6 step 3, audit SC-06): the local
    # cache must contain the union of rows every path's
    # ``(table, rid_column, rids)`` tuple would fetch. Two element paths can
    # reach the same table via different intermediate routes — e.g. one path
    # hits ``Image`` with ``rid_column="RID"`` and one rid set, another hits
    # ``Image`` with ``rid_column="Image"`` (a FK) and a different rid set.
    # The first walk's narrower fetch must NOT shadow the second's.
    #
    # We dedup on the full ``(table, rid_column, frozenset(rids))`` tuple,
    # which implements the invariant directly: each distinct parametrization
    # fires its own fetch, and only true duplicates (same table, same rid
    # column, same rid set) are skipped. The pre-seeded entry covers Step 1's
    # Dataset fetch so we don't redundantly re-issue it in the loop.
    processed: set[tuple[str, str, frozenset[str]]] = {("Dataset", "RID", frozenset(str(r) for r in dataset_rid_list))}

    for _key, (path, join_conditions, _join_types) in join_tables.items():
        # Walk in order — each table depends on rows loaded by the previous.
        for table_name in path[1:]:
            target_orm = orm_resolver(table_name)
            if target_orm is None:
                logger.warning("Skipping fetch for %s: no ORM class resolved", table_name)
                continue

            target_schema = table_to_schema.get(table_name)
            if target_schema is None:
                logger.warning("Skipping fetch for %s: no schema known", table_name)
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
                if not str_rids:
                    continue

                # Dedup on the full (table, rid_column, frozenset(rids)) tuple
                # — see the comment block above for why the prior table-name
                # key was wrong.
                fetch_key = (table_name, fk_column_on_target, frozenset(str_rids))
                if fetch_key in processed:
                    continue

                # Stamp the freshness ledger BEFORE the fetch (SC-03).
                # ``record_fetch_start`` is idempotent on key — though the
                # dedup check above already guarantees we won't restamp,
                # the idempotence is what makes the ledger report the
                # *first* fetch's timestamp rather than the most-recent
                # restamp.
                fetcher.record_fetch_start(table_name, fk_column_on_target, str_rids)
                fetcher.fetch_by_rids(
                    table=qualified,
                    rids=str_rids,
                    target_table=target_orm.__table__,
                    rid_column=fk_column_on_target,
                )
                processed.add(fetch_key)


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

    **Single-column FK assumption (RB-08).** This routine currently
    handles only **single-column** foreign keys. When the planner emits
    a join condition for a **composite** FK (multiple ``(fk_col, pk_col)``
    pairs all targeting *target_table_name*), the legacy implementation
    silently returned the first pair's values, producing an under-scoped
    fetch. The implementation now raises ``NotImplementedError`` when
    more than one workable condition is found, so the limitation
    surfaces loudly the day a schema introduces a composite FK rather
    than producing wrong join results.

    DerivaML schemas in active use today (CSA, CFDE, GPCR) all use
    single-column RID-based FKs, so this guard is latent and only fires
    against a future schema that breaks the assumption.
    """
    workable: list[tuple[list[Any], str]] = []
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
            workable.append((values, filter_col_on_target))

    if len(workable) > 1:
        # RB-08: surface composite FKs loudly so a future schema breaking
        # the single-column assumption produces an actionable error
        # rather than a silently under-scoped fetch.
        filter_cols = sorted({fc for _, fc in workable})
        raise NotImplementedError(
            f"_collect_fk_values: composite FK on {target_table_name!r} is "
            f"not yet supported; got {len(workable)} workable conditions on "
            f"filter columns {filter_cols}. Single-column FKs only — see "
            f"function docstring for context."
        )
    if workable:
        return workable[0]
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

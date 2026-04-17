"""Denormalizer — public API for producing wide tables from Deriva data.

Wraps the lower-level ``_denormalize_impl`` primitive in a class-based API
with support for auto-inferred ``row_per``, explicit ``via`` path routing,
orphan-row handling, and arbitrary RID anchor sets.

See ``docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md``
for the semantic rules this class implements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generator

import pandas as pd

from deriva_ml.local_db.denormalize import DenormalizeResult, _denormalize_impl

if TYPE_CHECKING:
    from deriva_ml.interfaces import DatasetLike

logger = logging.getLogger(__name__)


class Denormalizer:
    """Produce wide-table denormalizations from Deriva datasets or anchor sets.

    Construction:
      - ``Denormalizer(dataset_like)`` — use the dataset's members as anchors
        and derive catalog/workspace/model from the dataset's bindings.
      - ``Denormalizer.from_rids(rids, ml=...)`` — arbitrary RID anchors
        (see classmethod; implemented in a later task).

    Methods:
      - :meth:`as_dataframe` — materialize as pd.DataFrame.
      - :meth:`as_dict` — stream rows as dicts.
      - :meth:`columns` — preview column schema without fetching.
      - :meth:`describe` — dry-run the call; returns planning metadata.
      - :meth:`list_paths` — describe the FK graph (for exploration).
    """

    def __init__(self, dataset: "DatasetLike") -> None:
        """Construct from a ``DatasetLike`` object.

        The dataset's members (recursively via ``list_dataset_members``)
        become the anchor set. The underlying model, engine, and
        orm_resolver are derived from the dataset.

        Args:
            dataset: A :class:`Dataset` or :class:`DatasetBag` (or any
                object satisfying the ``DatasetLike`` protocol plus the
                attributes ``model``, ``engine``, ``_orm_resolver``).
        """
        self._dataset = dataset
        self._dataset_rid = dataset.dataset_rid
        self._model = dataset.model
        # engine / orm_resolver / paged_client are extracted lazily so
        # test fixtures can inject their own.
        self._engine = getattr(dataset, "engine", None)
        self._orm_resolver = getattr(dataset, "_orm_resolver", None)
        if self._orm_resolver is None:
            # Fall back to model's get_orm_class_by_name if available
            gocbn = getattr(self._model, "get_orm_class_by_name", None)
            self._orm_resolver = gocbn

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def as_dataframe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> pd.DataFrame:
        """Materialize the denormalized table as a pandas DataFrame.

        Args:
            include_tables: Tables whose columns appear in the output.
                Also determines ``row_per`` unless overridden.
            row_per: Explicit leaf table. Must be in ``include_tables``.
                If None, auto-inferred (Rule 2).
            via: Tables forced into the join chain without contributing
                columns. Useful for disambiguating path ambiguity without
                cluttering the output.
            ignore_unrelated_anchors: If True, silently drop anchors whose
                table has no FK path to any requested table. Default False
                raises :class:`DerivaMLDenormalizeUnrelatedAnchor`.

        Returns:
            A :class:`pandas.DataFrame` with one row per ``row_per`` instance
            in scope. See the semantic rules (Rules 1–8 in the spec) for
            the full cardinality and column-projection semantics.
        """
        result = self._run(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )
        return result.to_dataframe()

    def as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the denormalized table row-by-row as dicts.

        Same semantics as :meth:`as_dataframe` but yields one dict per row.
        Use for large datasets where a full DataFrame won't fit in memory.
        """
        result = self._run(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )
        yield from result.iter_rows()

    def columns(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Preview (column_name, type_name) pairs for the denormalized table.

        Model-only — no data fetch, no catalog query. Runs the same path
        validation as :meth:`as_dataframe` so ambiguity errors surface
        here too.
        """
        from deriva_ml.model.catalog import denormalize_column_name

        # Invokes the planner; raises on ambiguity.
        element_tables, column_specs, multi_schema = self._model._prepare_wide_table(
            self._dataset,
            self._dataset_rid,
            list(include_tables),
            row_per=row_per,
            via=via,
        )
        return [(denormalize_column_name(s, t, c, multi_schema), tp) for s, t, c, tp in column_specs]

    def describe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a planning-metadata dict describing what would happen.

        Stub implementation in this task — extended in Task 7 below.
        """
        # Minimal implementation: return just the row_per + columns.
        resolved_row_per = self._model._determine_row_per(
            include_tables=list(include_tables),
            via=list(via or []),
            row_per=row_per,
        )
        cols = self.columns(include_tables, row_per=row_per, via=via)
        return {
            "row_per": resolved_row_per,
            "row_per_source": "explicit" if row_per else "auto-inferred",
            "columns": cols,
            "include_tables": list(include_tables),
            "via": list(via or []),
            "ambiguities": [],
        }

    def list_paths(
        self,
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Describe the FK graph for exploration. Stub — filled in Task 8."""
        # Stub: full implementation in Task 8.
        return {
            "member_types": [],
            "reachable_tables": {},
            "association_tables": [],
            "feature_tables": [],
            "schema_paths": {},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        include_tables: list[str],
        *,
        row_per: str | None,
        via: list[str] | None,
        ignore_unrelated_anchors: bool,
    ) -> DenormalizeResult:
        """Four-phase pipeline: planner → anchor classification → SQL → orphans."""
        # Step 1: planner decisions (row_per, ambiguity checks)
        resolved_row_per = self._model._determine_row_per(
            include_tables=list(include_tables),
            via=list(via or []),
            row_per=row_per,
        )
        ambiguities = self._model._find_path_ambiguities(
            row_per=resolved_row_per,
            include_tables=list(include_tables),
            via=list(via or []),
        )
        if ambiguities:
            from deriva_ml.core.exceptions import DerivaMLDenormalizeAmbiguousPath

            a = ambiguities[0]
            raise DerivaMLDenormalizeAmbiguousPath(
                from_table=a["from_table"],
                to_table=a["to_table"],
                paths=a["paths"],
                suggested_intermediates=a["suggested_intermediates"],
            )

        # Step 2: anchor classification (Rule 7, Rule 8)
        anchors = self._anchors_as_dict()
        scoping, orphans, ignored = self._classify_anchors(
            anchors,
            include_tables=list(include_tables),
            via=list(via or []),
            row_per=resolved_row_per,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )

        # Step 3: main SQL via _denormalize_impl
        main_result = _denormalize_impl(
            model=self._model,
            engine=self._engine,
            orm_resolver=self._orm_resolver,
            dataset_rid=self._dataset_rid,
            include_tables=list(include_tables),
            dataset=self._dataset,
            source="local",
            row_per=resolved_row_per,
            via=list(via or []) or None,
        )

        # Step 4a: Augment orphans with scoping-upstream anchors whose specific
        # RIDs didn't appear in the main result. An upstream anchor (table in
        # include_tables but != row_per) may have RIDs with no matching
        # row_per row (e.g., a Subject that has no Image). Those RIDs need
        # orphan rows — the table-level classification can't catch this.
        per_rid_orphans: dict[str, list[str]] = {}
        for anchor_table, rids in scoping.items():
            if anchor_table == resolved_row_per:
                continue
            if anchor_table not in include_tables:
                continue
            # Collect RIDs of this anchor table that actually appear in main.
            rid_col = f"{anchor_table}.RID"
            seen: set[str] = set()
            for row in main_result.iter_rows():
                val = row.get(rid_col)
                if val is not None:
                    seen.add(val)
            missing = [r for r in rids if r not in seen]
            if missing:
                per_rid_orphans[anchor_table] = missing

        # Merge per-RID orphans with table-level orphans.
        combined_orphans: dict[str, list[str]] = {}
        for t, rids in orphans.items():
            combined_orphans.setdefault(t, []).extend(rids)
        for t, rids in per_rid_orphans.items():
            combined_orphans.setdefault(t, []).extend(rids)

        # Step 4b: orphan rows (Rule 7 case 3). Uses DenormalizeResult.extend
        # to keep the combine a one-liner.
        if combined_orphans:
            orphan_rows = self._emit_orphan_rows(
                combined_orphans,
                include_tables=list(include_tables),
                row_per=resolved_row_per,
            )
            main_result = main_result.extend(orphan_rows)

        return main_result

    def _anchors_as_dict(self) -> dict[str, list[str]]:
        """Return anchors as table_name -> list of RID strings."""
        members = self._dataset.list_dataset_members(recurse=True)
        return {table: [r["RID"] for r in rows] for table, rows in members.items()}

    def _classify_anchors(
        self,
        anchors: dict[str, list[str]],
        *,
        include_tables: list[str],
        via: list[str],
        row_per: str,
        ignore_unrelated_anchors: bool,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """Classify anchors by their relationship to row_per (Rule 7 + Rule 8).

        Args:
            anchors: table_name -> list of RIDs.
            include_tables, via, row_per: the planner inputs.
            ignore_unrelated_anchors: if False, raise on Rule-8 cases.

        Returns:
            Tuple of (scoping_anchors, orphan_anchors, ignored_anchors). Each
            is a table_name -> list of RIDs dict.

            - scoping_anchors: anchors that scope the main SQL query (their
              reachable row_per rows go in the output).
            - orphan_anchors: anchors whose table is in include_tables but can't
              reach row_per — they emit orphan rows.
            - ignored_anchors: anchors whose table isn't in include_tables and
              can't reach row_per — contribute nothing; only populated when
              ignore_unrelated_anchors=True.

        Raises:
            DerivaMLDenormalizeUnrelatedAnchor: if any anchor's table has no FK
                path to include_tables and ignore_unrelated_anchors=False.
        """
        from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor

        scoping: dict[str, list[str]] = {}
        orphans: dict[str, list[str]] = {}
        ignored: dict[str, list[str]] = {}
        unrelated: list[str] = []

        all_tables = set(include_tables) | set(via)

        for table, rids in anchors.items():
            if table == row_per:
                scoping[table] = list(rids)
                continue
            # Does this table have any FK path into the subgraph?
            reachable = self._model._outbound_reachable(table, all_tables | {table})
            if row_per in reachable:
                # Upstream of row_per and reaches it → scoping
                scoping[table] = list(rids)
            elif table in include_tables:
                # Upstream of row_per (in the subgraph) but no row_per reachable
                # from this specific anchor — we emit an orphan row per anchor.
                orphans[table] = list(rids)
            else:
                # Anchor's table is not in include_tables — and has no path to
                # row_per. It would contribute nothing.
                unrelated.append(table)

        if unrelated and not ignore_unrelated_anchors:
            raise DerivaMLDenormalizeUnrelatedAnchor(
                unrelated_tables=sorted(set(unrelated)),
                include_tables=list(include_tables),
            )
        if unrelated:
            for table in unrelated:
                ignored[table] = list(anchors[table])

        return scoping, orphans, ignored

    def _emit_orphan_rows(
        self,
        orphans: dict[str, list[str]],
        *,
        include_tables: list[str],
        row_per: str,
    ) -> list[dict[str, Any]]:
        """Emit one output row per orphan anchor.

        For each orphan RID, fetch its row from the local engine and construct
        an output dict with:
          - Anchor's columns populated with the row values.
          - All other columns (row_per and tables not equal to anchor_table)
            set to None.

        Uses the same NULL-init pattern as ``Dataset._denormalize_datapath``.
        """
        from sqlalchemy import select

        from deriva_ml.model.catalog import denormalize_column_name

        orphan_rows: list[dict[str, Any]] = []

        # Get the full column spec so we know what keys to populate.
        _, column_specs, multi_schema = self._model._prepare_wide_table(
            self._dataset,
            self._dataset_rid,
            list(include_tables),
        )

        # For each orphan anchor table, for each RID, emit one row.
        for anchor_table, rids in orphans.items():
            orm_cls = self._orm_resolver(anchor_table)
            if orm_cls is None:
                continue
            for rid in rids:
                with self._engine.connect() as conn:
                    row = (
                        conn.execute(select(orm_cls.__table__).where(orm_cls.__table__.c.RID == rid)).mappings().first()
                    )
                if row is None:
                    continue
                # Build the output dict: anchor cols populated, others None.
                out: dict[str, Any] = {}
                for schema_name, table_name, col_name, _type_name in column_specs:
                    label = denormalize_column_name(schema_name, table_name, col_name, multi_schema)
                    if table_name == anchor_table:
                        out[label] = row.get(col_name)
                    else:
                        out[label] = None
                orphan_rows.append(out)

        return orphan_rows

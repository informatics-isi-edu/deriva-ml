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

        Unlike :meth:`as_dataframe`, ``describe`` does NOT raise on ambiguity —
        it reports ambiguities in the ``ambiguities`` key so callers can
        inspect before committing to a real call.

        Returns a dict with these keys (see spec §5):
            row_per, row_per_source, row_per_candidates, columns,
            include_tables, via, join_path, transparent_intermediates,
            ambiguities, estimated_row_count, anchors, source
        """
        from deriva_ml.core.exceptions import DerivaMLDenormalizeError
        from deriva_ml.model.catalog import denormalize_column_name

        include = list(include_tables)
        via_list = list(via or [])

        # ── row_per resolution ─────────────────────────────────────────────
        # Dry-run invariant: describe() never raises. Catch all
        # DerivaMLDenormalizeError subclasses (MultiLeaf, NoSink,
        # DownstreamLeaf) plus ValueError (raised by _determine_row_per when
        # row_per is not in include_tables). In every failure mode the
        # caller still gets a well-formed dict with resolved_row_per=None.
        row_per_source = "explicit" if row_per else "auto-inferred"
        row_per_candidates = self._model._find_sinks(include, via_list)
        try:
            resolved_row_per: str | None = self._model._determine_row_per(
                include_tables=include,
                via=via_list,
                row_per=row_per,
            )
        except (DerivaMLDenormalizeError, ValueError):
            resolved_row_per = None

        # ── columns (may raise if row_per is None or ambiguity) ────────────
        try:
            element_tables, column_specs, multi_schema = self._model._prepare_wide_table(
                self._dataset,
                self._dataset_rid,
                include,
                row_per=row_per,
                via=via_list,
            )
            cols = [(denormalize_column_name(s, t, c, multi_schema), tp) for s, t, c, tp in column_specs]
        except Exception:
            element_tables = {}
            cols = []

        # ── ambiguities (reported, not raised) ─────────────────────────────
        ambiguities_raw = []
        if resolved_row_per is not None:
            ambiguities_raw = self._model._find_path_ambiguities(
                row_per=resolved_row_per,
                include_tables=include,
                via=via_list,
            )
        ambiguities = [
            {
                "type": "multiple_paths",
                "from": a["from_table"],
                "to": a["to_table"],
                "paths": [" → ".join(p) for p in a["paths"]],
                # Same set of intermediates on both keys today; the split
                # keeps the intent (include → columns, via → routing only)
                # explicit so callers can surface the choice to the user.
                "suggestions": {
                    "add_to_include_tables": a["suggested_intermediates"],
                    "add_to_via": a["suggested_intermediates"],
                },
            }
            for a in ambiguities_raw
        ]

        # ── join path + transparent intermediates ───────────────────────────
        join_path: list[str] = []
        transparent: list[str] = []
        for _, (path_names, _, _) in element_tables.items():
            for tn in path_names:
                if tn not in join_path and tn != "Dataset":
                    join_path.append(tn)
                    if tn not in include and self._model._is_association_table(tn):
                        transparent.append(tn)

        # ── anchors summary ─────────────────────────────────────────────────
        anchors = self._anchors_as_dict()
        anchors_by_type = {t: len(rids) for t, rids in anchors.items()}

        # ── estimated row count (crude — refined in future work) ────────────
        # For v1: report how many anchors would be scoping vs orphan.
        estimated = {
            "in_scope_row_per_rows": None,
            "orphan_rows": None,
            "total": None,
        }
        if resolved_row_per is not None:
            # Count anchors classified as scoping (includes row_per anchors)
            try:
                scoping, orphans, _ = self._classify_anchors(
                    anchors,
                    include_tables=include,
                    via=via_list,
                    row_per=resolved_row_per,
                    ignore_unrelated_anchors=True,  # describe shouldn't raise
                )
                in_scope = sum(len(rids) for table, rids in scoping.items() if table == resolved_row_per)
                orphan_count = sum(len(rids) for rids in orphans.values())
                estimated = {
                    "in_scope_row_per_rows": in_scope,
                    "orphan_rows": orphan_count,
                    "total": in_scope + orphan_count,
                }
            except Exception:
                pass

        return {
            "row_per": resolved_row_per,
            "row_per_source": row_per_source,
            "row_per_candidates": row_per_candidates,
            "columns": cols,
            "include_tables": include,
            "via": via_list,
            "join_path": join_path,
            "transparent_intermediates": transparent,
            "ambiguities": ambiguities,
            "estimated_row_count": estimated,
            "anchors": {"total": sum(anchors_by_type.values()), "by_type": anchors_by_type},
            "source": "local",  # Task 11 updates this when source is tracked
        }

    def list_paths(
        self,
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Describe the FK graph reachable from the dataset/anchors.

        Useful for picking ``include_tables`` when the user doesn't know the
        schema. Model-only analysis.

        Args:
            tables: If given, filter ``schema_paths`` to paths involving at
                least one of these tables.

        Returns:
            Dict with:
                member_types: list of dataset element types (if constructed
                    from a dataset); else empty.
                anchor_types: union of all distinct anchor table names.
                reachable_tables: mapping from each member/anchor type to
                    tables reachable from it via FK.
                association_tables: names of pure association tables in the
                    schema.
                feature_tables: names of feature tables (detected heuristically).
                schema_paths: mapping from (source_table, target_table) to a
                    list of path descriptions.
        """
        model = self._model
        anchors = self._anchors_as_dict()
        anchor_types = sorted(anchors.keys())

        # member_types: if constructed from a dataset, the dataset's members.
        # For from_rids, this is the anchor types. Same in both cases.
        member_types = anchor_types

        # Enumerate all tables in the relevant schemas.
        all_table_names: set[str] = set()
        try:
            ml_schema = getattr(model, "ml_schema", "deriva-ml")
            domain_schemas = getattr(model, "domain_schemas", [])
            for sname in [ml_schema, *domain_schemas]:
                if sname in model.schemas:
                    for t in model.schemas[sname].tables.values():
                        all_table_names.add(t.name)
        except Exception:
            pass

        # reachable_tables: from each anchor type, which domain tables are
        # reachable via FK (using the whole schema as the subgraph).
        reachable_tables: dict[str, list[str]] = {}
        for t in anchor_types:
            reach = model._outbound_reachable(t, all_table_names)
            reachable_tables[t] = sorted(reach)

        # association_tables: pure M-to-N linking tables
        association_tables = sorted(t for t in all_table_names if model._is_association_table(t))

        # feature_tables: heuristic — tables whose name contains "_" and
        # have an FK to a non-association, non-system table plus FK to a
        # vocabulary term. Approximated by: `is_feature_table` if the model
        # exposes it, else empty.
        feature_tables: list[str] = []
        is_feat = getattr(model, "is_feature_table", None)
        if callable(is_feat):
            for t in all_table_names:
                try:
                    if is_feat(t):
                        feature_tables.append(t)
                except Exception:
                    pass
        feature_tables.sort()

        # schema_paths: for every (source, target) pair among anchor_types ×
        # reachable_tables, enumerate FK paths.
        schema_paths: dict[tuple[str, str], list[dict]] = {}
        sources = set(anchor_types)
        if tables is not None:
            sources |= set(tables)
        for source in sources:
            for target in reachable_tables.get(source, []):
                if tables is not None and source not in tables and target not in tables:
                    continue
                paths = model._enumerate_paths(source, target, all_table_names)
                # Deduplicate
                unique = list({tuple(p): p for p in paths}.values())
                schema_paths[(source, target)] = [{"path": p, "direct": len(p) == 2} for p in unique]

        return {
            "member_types": member_types,
            "anchor_types": anchor_types,
            "reachable_tables": reachable_tables,
            "association_tables": association_tables,
            "feature_tables": feature_tables,
            "schema_paths": schema_paths,
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
        # Single-pass scan: build {anchor_table: {seen_rids}} in one iteration
        # rather than re-iterating the main result per anchor table.
        upstream_anchors = {
            t: set(rids) for t, rids in scoping.items() if t != resolved_row_per and t in include_tables
        }
        seen_by_table: dict[str, set[str]] = {t: set() for t in upstream_anchors}
        if upstream_anchors:
            for row in main_result.iter_rows():
                for t, seen in seen_by_table.items():
                    val = row.get(f"{t}.RID")
                    if val is not None:
                        seen.add(val)
        per_rid_orphans: dict[str, list[str]] = {}
        for anchor_table, rids in upstream_anchors.items():
            missing = [r for r in rids if r not in seen_by_table[anchor_table]]
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

        Uses a NULL-init pattern (all columns start as None; only the
        anchor's own columns are populated from its fetched row) — the
        LEFT-JOIN shape described by Rule 7 case 3.
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

        # Fetch all orphan rows for each anchor table in one query
        # (IN(rids)) rather than one connection per RID.
        for anchor_table, rids in orphans.items():
            orm_cls = self._orm_resolver(anchor_table)
            if orm_cls is None:
                logger.warning(
                    "orm_resolver returned None for anchor table %r; skipping %d orphan row(s).",
                    anchor_table,
                    len(rids),
                )
                continue
            if not rids:
                continue
            with self._engine.connect() as conn:
                rows = conn.execute(select(orm_cls.__table__).where(orm_cls.__table__.c.RID.in_(rids))).mappings().all()
            for row in rows:
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

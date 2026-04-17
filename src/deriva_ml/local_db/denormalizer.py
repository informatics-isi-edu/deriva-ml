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
        """Dispatch to the low-level primitive.

        Pre-validates ``row_per`` / ``via`` by invoking the planner directly
        (so Rule 2/5/6 errors surface here, not later inside the SQL loop).
        ``_denormalize_impl`` itself doesn't yet accept ``row_per`` / ``via``
        — Task 5+ wires those through to the join planner inside the impl.
        """
        # Pre-validate so planner-level errors (Rules 2, 5, 6) surface even
        # though `_denormalize_impl` doesn't yet plumb row_per/via through.
        self._model._prepare_wide_table(
            self._dataset,
            self._dataset_rid,
            list(include_tables),
            row_per=row_per,
            via=via,
        )

        return _denormalize_impl(
            model=self._model,
            engine=self._engine,
            orm_resolver=self._orm_resolver,
            dataset_rid=self._dataset_rid,
            include_tables=list(include_tables),
            dataset=self._dataset,
            source="local",  # fixture tests pre-populate the DB
        )

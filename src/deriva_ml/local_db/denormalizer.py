"""Denormalizer — public API for producing wide tables from Deriva data.

Wraps the lower-level ``_denormalize_impl`` primitive in a class-based API
with support for auto-inferred ``row_per``, explicit ``via`` path routing,
orphan-row handling, and arbitrary RID anchor sets.

Architecture, state model, fetcher/INSERT contract, fragility map,
the nine semantic Rules implemented here (auto-inferred ``row_per``,
downstream-leaf rejection, ambiguity detection, orphan emission,
unrelated-anchor handling, etc.), and the full test matrix all live in
the single source of record:
    ``docs/user-guide/denormalization.md``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generator

import pandas as pd

from deriva_ml.core.logging_config import get_logger
from deriva_ml.local_db.denormalize import DenormalizeResult, _denormalize_impl

if TYPE_CHECKING:
    from deriva_ml.feature import FeatureRecord
    from deriva_ml.interfaces import DatasetLike

logger = get_logger(__name__)


class Denormalizer:
    """Produce wide-table denormalizations from Deriva datasets or anchor sets.

    The ``Denormalizer`` wraps the lower-level :func:`_denormalize_impl`
    SQL executor in a class-based API that adds the new semantic rules
    (Rules 1-8 from the design spec):

    - Rule 2: auto-inferred ``row_per`` (sink-finding)
    - Rule 5: explicit ``row_per`` with a downstream table in
      ``include_tables`` raises :class:`DerivaMLDenormalizeDownstreamLeaf`
    - Rule 6: multiple FK paths between requested tables raises
      :class:`DerivaMLDenormalizeAmbiguousPath` unless disambiguated via
      ``include_tables`` or ``via=``
    - Rule 7: orphan anchors (upstream rows that don't reach ``row_per``)
      emit LEFT-JOIN-style rows with the row_per-side columns NULL
    - Rule 8: anchors with no FK path at all raise
      :class:`DerivaMLDenormalizeUnrelatedAnchor` unless the
      ``ignore_unrelated_anchors`` flag is set

    Construction:

    - ``Denormalizer(dataset_like)`` — use the dataset's members as the
      anchor set. Accepts a :class:`~deriva_ml.dataset.Dataset` (live
      catalog), a :class:`~deriva_ml.dataset.DatasetBag` (downloaded bag),
      or any object satisfying the :class:`~deriva_ml.interfaces.DatasetLike`
      protocol.
    - :meth:`from_rids` — construct from an explicit RID anchor set without
      a Dataset context (useful for ad-hoc exploration).

    Methods:

    - :meth:`as_dataframe` — materialize the wide table as a
      :class:`pandas.DataFrame`.
    - :meth:`as_dict` — yield rows as ``dict[str, Any]`` one at a time.
      Convenience for row-by-row iteration; the full result is still
      materialised internally before the first row is yielded, so this
      is *not* a memory-efficient alternative to :meth:`as_dataframe`
      (see the method's own docstring and audit finding SC-07).
    - :meth:`columns` — preview ``(column_name, column_type)`` pairs
      without fetching data (model-only).
    - :meth:`describe` — dry-run: returns a 12-key plan dict (spec §5);
      reports ambiguities rather than raising.
    - :meth:`list_paths` — describe the reachable FK graph from the
      anchor set (useful for discovering what can go into
      ``include_tables``).

    Example::

        # From a live Dataset:
        ml = DerivaML(hostname="example.org", catalog_id="42")
        dataset = ml.lookup_dataset("28CT")
        d = Denormalizer(dataset)
        df = d.as_dataframe(["Image", "Subject"])

        # Diamond-schema disambiguation via an intermediate:
        df = d.as_dataframe(["Image", "Subject"], via=["Observation"])

        # Dry-run to inspect the plan before committing:
        plan = d.describe(["Image", "Subject", "Diagnosis"])
        print(plan["row_per"], plan["join_path"])

        # From an explicit RID anchor set (no Dataset required):
        d = Denormalizer.from_rids(
            [("Image", "1-ABCD"), ("Image", "1-EFGH")], ml=ml,
        )
    """

    def __init__(self, dataset: "DatasetLike", *, version: Any = None) -> None:
        """Construct from a ``DatasetLike`` object.

        The dataset's members (recursively via ``list_dataset_members``)
        become the anchor set. The underlying model, engine, catalog, and
        orm_resolver are derived from the dataset. Two shapes are supported:

        - **Live `Dataset` (`dataset._ml_instance` is a DerivaML):** pulls
          ``model``, ``catalog``, ``engine``, and ``orm_resolver`` from the ml
          instance's ``workspace.local_schema``. ``source`` defaults to
          ``"catalog"`` so the planner fetches rows via a :class:`PagedClient`.
          If ``version`` is given, the :class:`ErmrestPagedClient` is built
          against the **version-snapshot catalog** (matching the pattern
          used by :meth:`Dataset.list_dataset_members`). Otherwise it uses
          whichever catalog the DerivaML instance was originally bound to
          (live or pre-pinned).
        - **`DatasetBag` or canned test fixture:** reads ``model``,
          ``engine``, and ``_orm_resolver`` attributes directly. ``source``
          defaults to ``"local"`` (rows are assumed already present in the
          engine). ``version`` has no effect for a bag — the bag is already
          materialized at whatever version it was built from.

        Args:
            dataset: A :class:`Dataset`, :class:`DatasetBag`, or any
                object satisfying the ``DatasetLike`` protocol.
            version: Optional ``DatasetVersion | str | None``. When given
                AND ``dataset`` is a live :class:`Dataset`, resolves to
                the corresponding catalog snapshot so the returned
                Denormalizer fetches reproducibly. Ignored for
                DatasetBag / fixtures. Follows the same semantics as
                :meth:`Dataset.list_dataset_members`'s ``version`` kwarg.
        """
        self._dataset = dataset
        self._dataset_rid = dataset.dataset_rid
        # Stash the version so _anchors_as_dict can pass it through to
        # Dataset.list_dataset_members for snapshot-consistent member
        # enumeration. None is a valid value and means "use the
        # dataset's default binding".
        self._version = version

        # Prefer direct attributes (DatasetBag, test fixture).
        # Fall back to `_ml_instance` (live Dataset).
        ml_instance = getattr(dataset, "_ml_instance", None)

        # If version is supplied and the dataset knows how to resolve it
        # to a snapshot-bound DerivaML instance, switch ml_instance to
        # that snapshot. This matches Dataset.list_dataset_members'
        # "snapshot-if-version-else-current-instance" pattern exactly.
        if version is not None and ml_instance is not None:
            resolver = getattr(dataset, "_version_snapshot_catalog", None)
            if callable(resolver):
                try:
                    ml_instance = resolver(version)
                except Exception:
                    # If snapshot resolution fails (bad version string,
                    # history lookup error), propagate — this is NOT the
                    # dry-run path, so callers should see the failure.
                    raise

        self._model = getattr(dataset, "model", None)
        if self._model is None and ml_instance is not None:
            self._model = getattr(ml_instance, "model", None)

        self._engine = getattr(dataset, "engine", None)
        self._orm_resolver = getattr(dataset, "_orm_resolver", None)
        self._paged_client: Any = None
        self._source = "local"
        # RB-05: diagnostic for the silent-fallback-to-local case. When
        # ErmrestPagedClient construction fails on a live-catalog path,
        # this gets populated with a single-line description so the
        # describe() envelope can surface it (planned: SC-01 warnings
        # envelope appends this to plan["warnings"]). Empty string means
        # "no fallback occurred."
        self._init_warning: str = ""

        if ml_instance is not None:
            workspace = getattr(ml_instance, "workspace", None)
            if self._engine is None and workspace is not None:
                self._engine = getattr(workspace, "engine", None)
            if self._orm_resolver is None and workspace is not None:
                local_schema = getattr(workspace, "local_schema", None)
                if local_schema is not None:
                    self._orm_resolver = getattr(local_schema, "get_orm_class", None)
            # Live-catalog path: build a PagedClient so _denormalize_impl
            # can fetch rows. The catalog here is the ml_instance's
            # current catalog — which is either the live catalog, the
            # snapshot chosen by the `version` kwarg above, or whatever
            # the DerivaML was pre-bound to at construction time. This
            # three-way source-of-truth mirrors list_dataset_members.
            catalog = getattr(ml_instance, "catalog", None)
            if catalog is not None:
                try:
                    from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

                    self._paged_client = ErmrestPagedClient(catalog=catalog)
                    self._source = "catalog"
                except Exception as e:
                    # RB-05: If the client can't be built (offline tests,
                    # mock catalog, transient auth race), fall back to
                    # local mode — but log at WARNING and stash an
                    # `_init_warning` so describe() can surface the
                    # fallback to callers. Previously this was silent
                    # and produced zero-row results on auth/network
                    # setup races with no diagnostic.
                    self._paged_client = None
                    self._source = "local"
                    self._init_warning = (
                        f"ErmrestPagedClient construction failed "
                        f"({type(e).__name__}: {e}); fell back to "
                        f"source='local'. Live-catalog fetches are "
                        f"disabled — results will reflect whatever is "
                        f"already in the local engine."
                    )
                    logger.warning("Denormalizer init: %s", self._init_warning)

        if self._orm_resolver is None and self._model is not None:
            # Last resort — model-level ORM resolver.
            gocbn = getattr(self._model, "get_orm_class_by_name", None)
            self._orm_resolver = gocbn

    @classmethod
    def from_rids(
        cls,
        anchors: list[str | tuple[str, str]],
        *,
        ml: Any = None,
        catalog: Any = None,
        workspace: Any = None,
        model: Any = None,
        engine: Any = None,
        orm_resolver: Any = None,
        dataset_rid: str | None = None,
    ) -> "Denormalizer":
        """Construct from an explicit anchor set (no Dataset required).

        Anchors may be bare RIDs (table looked up via catalog) or
        ``(table_name, RID)`` tuples (lookup skipped). Mixed forms supported.

        Pass either ``ml=`` (common path) or the separate ``catalog``,
        ``workspace``, ``model`` keyword args (escape hatch).

        The current ``_denormalize_impl`` primitive scopes its SQL query by
        ``Dataset.RID IN (dataset_rid)`` — that is, it always traverses from
        the Dataset root. ``from_rids`` therefore requires a real dataset
        RID against which the anchors are linked. Pass an explicit
        ``dataset_rid=<RID>`` that contains the anchor RIDs, or use the
        :class:`Denormalizer` constructor with a :class:`Dataset` so the
        parent dataset's RID is used automatically.

        Per-call behavior flags (``row_per``, ``via``,
        ``ignore_unrelated_anchors``) are supplied to the public methods
        (:meth:`as_dataframe`, :meth:`as_dict`, etc.) — not to the
        constructor.

        Args:
            anchors: list of bare RIDs or (table, RID) tuples.
            ml: Convenience: pass a DerivaML instance. catalog/workspace/model
                are derived from it.
            catalog, workspace, model, engine, orm_resolver: Explicit deps.
            dataset_rid: Real dataset RID to scope the SQL by. Required
                against a live catalog — otherwise the SQL ``Dataset.RID
                IN (dataset_rid, ...)`` predicate matches nothing and the
                caller gets a silent empty DataFrame (SC-02 / TC-05). May
                be omitted in fixture/local-only contexts where the engine
                has been pre-populated with arbitrary RIDs; a warning is
                logged in that path.

        Returns:
            A :class:`Denormalizer` bound to the given anchor set.

        Raises:
            ValueError: if neither ``ml`` nor ``model`` is provided; if a
                bare RID is passed without a catalog for lookup; if a bare
                RID cannot be resolved; if a tuple anchor does not have
                exactly two elements; or if ``dataset_rid`` is ``None``
                against a live catalog (pass an explicit ``dataset_rid``
                or use the :class:`Denormalizer` constructor with a
                :class:`Dataset` instead).
        """
        # Derive deps from ml if given.
        if ml is not None:
            catalog = catalog if catalog is not None else getattr(ml, "catalog", None)
            workspace = workspace if workspace is not None else getattr(ml, "workspace", None)
            model = model if model is not None else getattr(ml, "model", None)
            if engine is None and workspace is not None:
                engine = getattr(workspace, "engine", None)
            if orm_resolver is None and workspace is not None:
                ls = getattr(workspace, "local_schema", None)
                if ls is not None:
                    orm_resolver = getattr(ls, "get_orm_class", None)

        if model is None:
            raise ValueError("Denormalizer.from_rids requires either ml= or an explicit model=")

        # SC-02 / TC-05 guard: dataset_rid is None against a live catalog
        # is a silent-zero failure mode. _denormalize_impl scopes by
        # ``Dataset.RID IN (dataset_rid, ...)`` and no real dataset has
        # the placeholder RID, so the result is an empty DataFrame with
        # no diagnostic. Probe whether the catalog passed in is a real
        # live one (i.e., one ErmrestPagedClient can wrap) and refuse
        # the call there. Fixture/local contexts (no catalog, or a
        # mock that can't be wrapped) keep working with a warning —
        # those flows pre-populate the engine and the RID acts as an
        # opaque scoping key the in-memory SQL will match if rows exist.
        if dataset_rid is None:
            live_catalog = False
            if catalog is not None:
                try:
                    from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

                    ErmrestPagedClient(catalog=catalog)
                    live_catalog = True
                except Exception:
                    # Catalog can't be wrapped (mock, offline) — treat
                    # as local mode; warn but don't raise.
                    live_catalog = False
            if live_catalog:
                raise ValueError(
                    "Denormalizer.from_rids() requires an explicit dataset_rid against "
                    "a live catalog because the SQL primitive scopes by "
                    "Dataset.RID IN (dataset_rid, ...). Pass dataset_rid=<existing "
                    "dataset RID> that contains the anchor RIDs, or use "
                    "Denormalizer(dataset) where the parent Dataset's RID is "
                    "used automatically."
                )
            logger.warning(
                "Denormalizer.from_rids() called with dataset_rid=None in "
                "local/fixture mode; falling back to the first anchor's RID "
                "as a placeholder scope. This will return zero rows against "
                "a live catalog — pass dataset_rid=<RID> for production use."
            )

        # Normalize anchors to (table, RID) pairs. Validate tuple arity so
        # a 3-tuple (or 1-tuple) surfaces as a clear ValueError here rather
        # than an opaque unpack error later.
        resolved: list[tuple[str, str]] = []
        bare_rids: list[str] = []
        for a in anchors:
            if isinstance(a, tuple):
                if len(a) != 2:
                    raise ValueError(f"Anchor tuples must be (table, RID); got {a!r}")
                resolved.append(a)
            else:
                bare_rids.append(a)

        # Batch-resolve bare RIDs. Prefer the plural ``ml.resolve_rids``
        # entry point (one query per candidate table) over the per-RID
        # ``catalog.resolve_rid`` fallback — for N anchors that's the
        # difference between O(tables) round-trips and O(N) round-trips.
        # Exceptions from either path are translated into ValueError so
        # the public contract in the docstring's Raises: block holds.
        if bare_rids:
            if ml is not None and hasattr(ml, "resolve_rids"):
                try:
                    rid_results = ml.resolve_rids(bare_rids)
                except Exception as e:
                    raise ValueError(f"Cannot resolve one or more RIDs: {bare_rids!r}") from e
                for rid in bare_rids:
                    info = rid_results.get(rid)
                    if info is None:
                        raise ValueError(f"Cannot resolve RID {rid!r} to a table")
                    resolved.append((info.table_name, rid))
            else:
                if catalog is None:
                    raise ValueError(
                        "Bare RIDs given but no catalog available for lookup. "
                        "Pass (table, RID) tuples or provide catalog= or ml=."
                    )
                if not hasattr(catalog, "resolve_rid"):
                    raise ValueError("catalog= does not expose resolve_rid; cannot look up bare RIDs")
                for rid in bare_rids:
                    try:
                        info = catalog.resolve_rid(rid)
                    except Exception as e:
                        raise ValueError(f"Cannot resolve RID {rid!r} to a table") from e
                    resolved.append((info.table.name, rid))

        # Group by table.
        anchors_by_table: dict[str, list[str]] = {}
        for table, rid in resolved:
            anchors_by_table.setdefault(table, []).append(rid)

        # Effective dataset RID used by _denormalize_impl's WHERE clause.
        # If caller supplied one, use it; otherwise (local/fixture mode
        # only — live-catalog callers were rejected above) fall back to
        # the first anchor's RID as a placeholder scope.
        effective_dataset_rid = dataset_rid if dataset_rid is not None else (resolved[0][1] if resolved else "")

        # Create a pseudo-dataset that exposes the anchors dict as members.
        class _AnchorSet:
            def __init__(self, ds_rid: str, members: dict[str, list[str]], model_ref: Any):
                self.dataset_rid = ds_rid
                self.model = model_ref
                self._members = members

            def list_dataset_members(self, **_kwargs: Any) -> dict[str, list[dict]]:
                return {t: [{"RID": r} for r in rids] for t, rids in self._members.items()}

            def list_dataset_children(self, **_kwargs: Any) -> list:
                return []

        anchor_set = _AnchorSet(effective_dataset_rid, anchors_by_table, model)

        # Fall back to model's get_orm_class_by_name if orm_resolver is still None.
        if orm_resolver is None:
            orm_resolver = getattr(model, "get_orm_class_by_name", None)

        # Build the Denormalizer manually (bypasses __init__'s DatasetLike
        # assumptions since _AnchorSet is a lightweight shim).
        inst = object.__new__(cls)
        inst._dataset = anchor_set
        inst._dataset_rid = effective_dataset_rid
        inst._model = model
        inst._engine = engine
        inst._orm_resolver = orm_resolver
        # from_rids doesn't carry an _ml_instance, so default to the local
        # source mode. If callers want catalog-side fetching they should
        # pre-populate the engine or extend from_rids with an explicit
        # paged_client / source pair.
        inst._source = "local"
        inst._paged_client = None
        # from_rids has no Dataset context → no version resolution;
        # _anchors_as_dict uses the anchor dict directly anyway.
        inst._version = None
        return inst

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
        selector: Callable[[list["FeatureRecord"]], "FeatureRecord | None"] | None = None,
    ) -> pd.DataFrame:
        """Materialize the denormalized table as a :class:`pandas.DataFrame`.

        Runs the full 4-phase pipeline: planner decisions (Rules 2/5/6) →
        anchor classification (Rules 7/8) → main SQL join → orphan-row
        combine. When ``selector`` is provided, a final reduction pass
        groups rows by the feature-association table's target RID and
        applies the selector per group — same contract as
        :meth:`~deriva_ml.core.mixins.feature.FeatureMixin.feature_values`'s
        ``selector`` argument.

        Args:
            include_tables: Tables whose columns appear in the output.
                Also determines ``row_per`` unless overridden. Columns are
                labeled ``Table.column`` (or ``schema.Table.column`` when
                multi-schema).
            row_per: Explicit leaf table. Must be in ``include_tables``.
                If None, auto-inferred by Rule 2 (the unique sink in the
                FK subgraph over ``include_tables ∪ via``).
            via: Tables forced into the join chain without contributing
                columns. Used to resolve Rule 6 path ambiguity without
                cluttering the output.
            ignore_unrelated_anchors: If True, silently drop anchors whose
                table has no FK path to any requested table. Default
                False raises :class:`DerivaMLDenormalizeUnrelatedAnchor`
                (Rule 8).
            selector: Optional callable
                ``(list[FeatureRecord]) -> FeatureRecord | None`` used to
                reduce multi-row feature groups. See ``FeatureRecord`` for
                built-ins (``select_newest``, ``select_first``,
                ``select_by_execution``, ``select_by_workflow``,
                ``select_majority_vote``). Return ``None`` from a selector
                to omit that target RID. Requires ``include_tables`` to
                contain exactly one feature-association table; raises
                ``ValueError`` when zero or more than one are present
                (multi-feature reduction is a future extension).

        Returns:
            A :class:`pandas.DataFrame` with one row per ``row_per``
            instance in scope, plus any orphan rows from upstream anchors
            that don't reach a ``row_per`` row (Rule 7 case 3). Upstream
            table columns are hoisted onto each row; orphan rows have
            ``row_per``-side columns set to ``NaN``. When ``selector`` is
            provided, multi-row feature groups collapse to one row per
            target RID per the selector's choice.

        Raises:
            DerivaMLDenormalizeMultiLeaf: auto-inference finds multiple
                candidate sinks (Rule 2).
            DerivaMLDenormalizeNoSink: cycle in the FK subgraph (Rule 2).
            DerivaMLDenormalizeDownstreamLeaf: explicit ``row_per`` with a
                downstream table in ``include_tables`` (Rule 5).
            DerivaMLDenormalizeAmbiguousPath: multiple FK paths between
                ``row_per`` and another requested table (Rule 6).
            DerivaMLDenormalizeUnrelatedAnchor: anchor has no FK path to
                any table in ``include_tables`` (Rule 8) — unless the
                ``ignore_unrelated_anchors`` flag is set.
            ValueError: ``selector`` was provided but ``include_tables``
                does not contain exactly one feature-association table.

        Example::

            d = Denormalizer(dataset)

            # One row per Image, with Subject columns hoisted:
            df = d.as_dataframe(["Image", "Subject"])

            # Force an intermediate table's columns into the output:
            df = d.as_dataframe(["Image", "Observation", "Subject"])

            # Route through Observation without adding its columns:
            df = d.as_dataframe(["Image", "Subject"], via=["Observation"])

            # Pin row_per explicitly (must be the deepest requested table):
            df = d.as_dataframe(["Image", "Subject"], row_per="Image")

            # Heterogeneous dataset: drop members with no path to Image:
            df = d.as_dataframe(
                ["Image", "Subject"], ignore_unrelated_anchors=True
            )

            # Reduce multi-row feature groups (e.g., multiple annotators
            # per Image) to one row per Image — the newest by RCT:
            from deriva_ml.feature import FeatureRecord
            df = d.as_dataframe(
                ["Image", "Execution_Image_Image_Classification"],
                selector=FeatureRecord.select_newest,
            )
        """
        result = self._run(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
            selector=selector,
        )
        return result.to_dataframe()

    def as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
        selector: Callable[[list["FeatureRecord"]], "FeatureRecord | None"] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Convert the denormalized table to dicts and yield row-by-row.

        Same planner, same rules, same exceptions as :meth:`as_dataframe` —
        but yields one ``dict[str, Any]`` per row (keyed by the
        ``Table.column`` / ``schema.Table.column`` label) rather than
        materializing a DataFrame.

        **The full result is materialised before the first row is
        yielded.** This method is memory-equivalent to
        :meth:`as_dataframe`, not a streaming alternative — peak memory
        scales with the full result set, because ``_denormalize_impl``
        eagerly drains the SQLAlchemy cursor into a Python list before
        :meth:`as_dict` iterates it. Use this when downstream code
        processes rows one at a time but does not need DataFrame
        indexing; do **not** rely on it to bound memory for very large
        results.

        True row-by-row streaming over the cursor is a known gap (audit
        finding SC-07; see ``docs/user-guide/denormalization.md`` §8.2 for
        the design discussion). Until that gap is closed, treat
        ``as_dict`` as "iteration interface, materialised internals."

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Explicit leaf table (Rule 2 override).
            via: Path-only intermediates (Rule 6 disambiguation).
            ignore_unrelated_anchors: If True, silently drop unrelated
                anchors (Rule 8).
            selector: Optional callable
                ``(list[FeatureRecord]) -> FeatureRecord | None`` used to
                reduce multi-row feature groups after materialization.
                Same contract as :meth:`as_dataframe`. See
                ``FeatureRecord`` for built-in selectors.

        Yields:
            ``dict[str, Any]`` — one per output row. Keys are
            denormalized column names (``Table.column`` or
            ``schema.Table.column``); values are raw Python types as
            returned by SQLAlchemy.

        Raises:
            Same as :meth:`as_dataframe`. Exceptions surface on the first
            ``next()`` — all planner validation and the full SQL execution
            run before any row is yielded, since the pipeline builds the
            full result up front.

        Example::

            d = Denormalizer(dataset)
            for row in d.as_dict(["Image", "Subject"]):
                print(row["Image.RID"], row["Subject.Name"])

            # With selector reduction:
            from deriva_ml.feature import FeatureRecord
            for row in d.as_dict(
                ["Image", "Execution_Image_Image_Classification"],
                selector=FeatureRecord.select_newest,
            ):
                ...
        """
        result = self._run(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
            selector=selector,
        )
        yield from result.iter_rows()

    def columns(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Preview ``(column_name, type_name)`` pairs for the denormalized table.

        Model-only — no data fetch, no catalog query, no anchor classification.
        Invokes the same planner as :meth:`as_dataframe` so Rules 2/5/6
        errors surface here too; this makes ``columns`` useful as a cheap
        validator of ``include_tables`` before committing to a full run.

        For a dry-run that reports ambiguities without raising, use
        :meth:`describe` instead.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Explicit leaf table (Rule 2 override).
            via: Path-only intermediates (Rule 6 disambiguation).

        Returns:
            Sorted list of ``(column_name, column_type)`` tuples. Column
            names use dot notation (``Table.column`` or
            ``schema.Table.column`` in multi-schema mode) — same shape
            used by :meth:`as_dataframe` and :meth:`as_dict` output.

        Raises:
            DerivaMLDenormalizeMultiLeaf / NoSink / DownstreamLeaf /
            AmbiguousPath: same as :meth:`as_dataframe` (planner rules
            2/5/6). Rule 7 and Rule 8 errors do NOT fire here — anchor
            classification happens only when rows are materialized.

        Example::

            d = Denormalizer(dataset)
            cols = d.columns(["Image", "Subject"])
            # [("Image.RID", "text"), ("Image.Filename", "text"),
            #  ("Subject.RID", "text"), ("Subject.Name", "text"), ...]
        """
        from deriva_ml.model.catalog import denormalize_column_name

        # Translate feature names to feature-association table names so
        # ``columns`` accepts the same caller-friendly inputs as
        # ``as_dataframe`` (the symmetry is half the point of this
        # method — preview the run's column list cheaply).
        include_resolved, via_resolved, row_per_resolved = self._resolve_table_names(
            list(include_tables), via=via, row_per=row_per
        )
        # Invoke the planner on the model alone. _prepare_wide_table runs
        # the Rule 2/5/6 guards and returns the column spec list. Walking
        # that list with denormalize_column_name gives us the final
        # dot-prefixed output labels without touching any data.
        element_tables, column_specs, multi_schema = self._model._planner._prepare_wide_table(
            self._dataset,
            self._dataset_rid,
            include_resolved,
            row_per=row_per_resolved,
            via=via_resolved,
        )
        return [(denormalize_column_name(s, t, c, multi_schema), tp) for s, t, c, tp in column_specs]

    def describe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a 13-key planning-metadata dict describing what a
        corresponding :meth:`as_dataframe` call would do.

        **Dry-run invariant**: unlike :meth:`as_dataframe`, ``describe``
        never raises. Every failure mode (planner rule violation,
        catalog access error, network timeout) is swallowed and
        represented in the returned dict as ``None`` / ``[]`` / ``{}``
        in the affected positions. **Information-preservation
        invariant**: whenever a broad-except site swallows an
        exception, a short one-line diagnostic is appended to
        ``plan["warnings"]`` so the caller can tell why a key is
        empty (genuine empty result vs. a swallowed failure).
        Ambiguities are reported in the ``ambiguities`` list so the
        caller can inspect before committing to a real call.

        Args:
            include_tables: Tables whose columns would appear in the
                output. Also determines ``row_per`` unless overridden.
            row_per: Optional explicit leaf table.
            via: Optional path-only intermediates.

        Returns:
            A dict with these 12 keys (see design spec §5):

            - ``row_per``: resolved leaf table name, or ``None`` if the
              planner couldn't resolve one (e.g., multi-leaf or bad
              ``row_per`` argument).
            - ``row_per_source``: ``"explicit"`` if the caller passed
              ``row_per``, else ``"auto-inferred"``.
            - ``row_per_candidates``: list of sink tables from Rule 2
              sink-finding (the choices auto-inference considered).
            - ``columns``: list of ``(column_name, column_type)`` tuples
              that :meth:`as_dataframe` would produce.
            - ``include_tables`` / ``via``: echoes of the inputs.
            - ``join_path``: ordered list of table names on the join
              chain (includes intermediates, excludes the implicit
              ``Dataset`` root).
            - ``transparent_intermediates``: subset of ``join_path`` that
              the user did NOT name in ``include_tables`` — their
              columns are NOT in the output (joined through only).
            - ``ambiguities``: list of ``{type, from, to, paths,
              suggestions}`` dicts, one per Rule 6 ambiguity. Empty
              when the plan is unambiguous.
            - ``estimated_row_count``: ``{in_scope_row_per_rows,
              orphan_rows, total}``, optionally with a ``reason`` field.
              Anchor-based estimate; honest about unknowns. ``total``
              is exact when every scoping anchor sits at ``row_per``
              (case 1 of §3.7). When any scoping anchor is downstream
              of ``row_per`` (the common feature-table case), the
              per-anchor fan-out is unknown without a catalog query —
              ``in_scope_row_per_rows`` and ``total`` come back as
              ``None`` and a ``reason`` string names the downstream
              tables. ``orphan_rows`` is always exact (an orphan
              contributes exactly one row regardless of ``row_per``
              cardinality). Originally surfaced as 2026-05-21 finding
              A02; documented in ``docs/user-guide/denormalization.md`` §7
              row F6.
            - ``anchors``: ``{total, by_type}`` — counts of anchor RIDs
              grouped by table.
            - ``source``: ``"catalog"`` for live Datasets, ``"local"``
              for DatasetBags / canned fixtures, ``"slice"`` for
              attached slices.
            - ``warnings``: ``list[str]`` — one short human-readable
              entry per swallowed exception. Each entry names the
              call that failed (``"_resolve_table_names"``,
              ``"_find_sinks"``, etc.), what defaulted (``"reverted
              to original include"``, ``"row_per_candidates is
              empty"``), and the exception type plus a truncated
              one-line message. Empty list means describe ran
              clean. See spec §8.3.1 and audit findings SC-01 /
              RB-01 / TC-09.

        Example::

            d = Denormalizer(dataset)
            plan = d.describe(["Image", "Subject"])

            if plan["ambiguities"]:
                for amb in plan["ambiguities"]:
                    print(f"  {amb['from']} → {amb['to']}: {len(amb['paths'])} paths")
                    print(f"  try adding: {amb['suggestions']['add_to_via']}")
            else:
                print(f"row_per = {plan['row_per']}")
                print(f"join = {' → '.join(plan['join_path'])}")
                print(f"{plan['estimated_row_count']['total']} rows")
        """
        from deriva_ml.core.exceptions import DerivaMLDenormalizeError
        from deriva_ml.model.catalog import denormalize_column_name

        original_include = list(include_tables)
        original_via = list(via or [])

        # ── warnings accumulator ────────────────────────────────────────────
        # Per spec §8.3.1 ("describe() return shape — warnings field"),
        # every broad-except site below appends a short one-liner
        # naming the call that failed, what defaulted, and the
        # exception type + truncated message. This restores the
        # information-preservation invariant (a caller can tell why a
        # key is empty) without weakening the dry-run invariant
        # (describe still never raises). Closes audit findings SC-01,
        # RB-01, TC-09 in
        # ``docs/audits/2026-05-26-denormalize-audit.md``.
        warnings: list[str] = []

        # ── feature-name resolution ─────────────────────────────────────────
        # Translate feature names ("Image_Classification") to their
        # underlying feature-association tables
        # ("Execution_Image_Image_Classification") so describe and
        # ``as_dataframe`` agree on what's a valid ``include_tables``
        # entry. The dry-run invariant says describe never raises, so
        # any resolution failure (typo, ambiguous feature) collapses
        # to the original inputs and the downstream planner calls
        # surface the failure as empty fields in the returned dict.
        try:
            include, via_list, row_per = self._resolve_table_names(original_include, via=original_via, row_per=row_per)
        except Exception as e:
            include = original_include
            via_list = original_via
            warnings.append(
                f"_resolve_table_names: {type(e).__name__} ({str(e)[:120]}); reverted to original include/via"
            )

        # ── row_per resolution ─────────────────────────────────────────────
        # Dry-run invariant: describe() never raises. Every call that could
        # fail against planner/catalog/schema state is wrapped in a broad
        # try/except so the caller always gets a well-formed dict with
        # sensible defaults (None / [] / {}) in the positions that couldn't
        # be computed.
        row_per_source = "explicit" if row_per else "auto-inferred"
        try:
            row_per_candidates = self._model._planner._find_sinks(include, via_list)
        except Exception as e:
            row_per_candidates = []
            warnings.append(f"_find_sinks: {type(e).__name__} ({str(e)[:120]}); row_per_candidates is empty")
        try:
            resolved_row_per: str | None = self._model._planner._determine_row_per(
                include_tables=include,
                via=via_list,
                row_per=row_per,
            )
        except (DerivaMLDenormalizeError, ValueError):
            resolved_row_per = None
        except Exception:
            # Defensive: model access can raise (catalog unreachable, etc.).
            resolved_row_per = None

        # ── columns (may raise if row_per is None or ambiguity) ────────────
        try:
            element_tables, column_specs, multi_schema = self._model._planner._prepare_wide_table(
                self._dataset,
                self._dataset_rid,
                include,
                row_per=row_per,
                via=via_list,
            )
            cols = [(denormalize_column_name(s, t, c, multi_schema), tp) for s, t, c, tp in column_specs]
        except Exception as e:
            element_tables = {}
            cols = []
            warnings.append(
                f"_prepare_wide_table: {type(e).__name__} ({str(e)[:120]}); element_tables and columns are empty"
            )

        # ── ambiguities (reported, not raised) ─────────────────────────────
        ambiguities_raw: list[dict] = []
        if resolved_row_per is not None:
            try:
                ambiguities_raw = self._model._planner._find_path_ambiguities(
                    row_per=resolved_row_per,
                    include_tables=include,
                    via=via_list,
                )
            except Exception as e:
                # If path enumeration fails (e.g., model access error),
                # treat as "no ambiguities detected" rather than raising.
                ambiguities_raw = []
                warnings.append(
                    f"_find_path_ambiguities: {type(e).__name__} ({str(e)[:120]}); ambiguities not detected"
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
        # A "transparent intermediate" is any table on the join path that
        # the user did NOT name in include_tables — the join traverses
        # through it but its columns aren't projected. This includes both
        # pure association tables AND non-association tables reached only
        # via `via=` routing (spec §5 example shows "Observation" here —
        # not an association table).
        join_path: list[str] = []
        transparent: list[str] = []
        for _, (path_names, _, _) in element_tables.items():
            for tn in path_names:
                if tn not in join_path and tn != "Dataset":
                    join_path.append(tn)
                    if tn not in include:
                        transparent.append(tn)

        # ── anchors summary ─────────────────────────────────────────────────
        # Anchor retrieval can fail for live datasets (network/catalog
        # errors). Dry-run invariant: fall back to an empty summary rather
        # than raise.
        try:
            anchors = self._anchors_as_dict()
        except Exception as e:
            anchors = {}
            warnings.append(f"_anchors_as_dict: {type(e).__name__} ({str(e)[:120]}); anchors summary is empty")
        # RB-03: skip empty anchor sets, matching the ``if not rids: continue``
        # guard in :meth:`_classify_anchors`. ``list_dataset_members`` returns
        # ``{"File": []}`` for empty association tables; describe and run must
        # agree on the resulting anchor count.
        anchors_by_type = {t: len(rids) for t, rids in anchors.items() if rids}

        # ── estimated row count (anchor-based; honest about unknowns) ───────
        #
        # Three cases (see ``docs/user-guide/denormalization.md`` §6.5 freshness
        # caveat and the 2026-05-21 finding A02 history):
        #
        # 1. Anchor table == row_per → in_scope is exact (1 row per anchor).
        # 2. Anchor table reaches row_per via FK chain (downstream or
        #    upstream) but is NOT row_per itself → cardinality is N rows
        #    per anchor for some N ≥ 0 that depends on per-anchor data the
        #    estimator does not have without a catalog query. Honest answer:
        #    None (with a `reason` field so the caller knows why).
        # 3. Anchor table has no FK path to row_per → orphan (case 3 of
        #    Rule 7; counted in orphan_rows).
        #
        # The pre-A02 implementation only honored case 1 (filtered on
        # ``table == resolved_row_per``) and silently returned 0 for case
        # 2. That meant ``preflight_count`` reported 0 for every
        # feature-table denormalize (the common case where row_per is a
        # downstream feature table and anchors are Image / Subject /
        # whatever the dataset elements are). The 2026-05-21 e2e Analyst
        # arc consumed the silent 0 as truth and reported A02.
        estimated: dict[str, Any] = {
            "in_scope_row_per_rows": None,
            "orphan_rows": None,
            "total": None,
        }
        if resolved_row_per is not None:
            try:
                scoping, orphans, _ = self._classify_anchors(
                    anchors,
                    include_tables=include,
                    via=via_list,
                    row_per=resolved_row_per,
                    ignore_unrelated_anchors=True,  # describe shouldn't raise
                )
                # Split scoping into "anchors sitting at row_per"
                # (case 1, exact contribution) and "anchors elsewhere
                # but reaching row_per via FK chain" (case 2, unknown
                # fan-out).
                at_row_per_count = sum(len(rids) for table, rids in scoping.items() if table == resolved_row_per)
                downstream_anchor_tables = [
                    table for table, rids in scoping.items() if table != resolved_row_per and rids
                ]
                orphan_count = sum(len(rids) for rids in orphans.values())

                if downstream_anchor_tables:
                    # Case 2 dominates: we can't honestly state a total
                    # without per-anchor fan-out info. Surface what we
                    # DO know (orphan_rows is exact) and a reason for
                    # the unknown fields.
                    estimated = {
                        "in_scope_row_per_rows": None,
                        "orphan_rows": orphan_count,
                        "total": None,
                        "reason": (
                            f"anchor table(s) {downstream_anchor_tables} are "
                            f"downstream of row_per={resolved_row_per!r}; "
                            "cardinality is N rows per anchor for unknown N. "
                            "Run the actual fetch to count, or query the "
                            "catalog directly for an exact figure."
                        ),
                    }
                else:
                    # Case 1 only: estimate is exact.
                    estimated = {
                        "in_scope_row_per_rows": at_row_per_count,
                        "orphan_rows": orphan_count,
                        "total": at_row_per_count + orphan_count,
                    }
            except Exception as e:
                warnings.append(
                    f"estimated_row_count: {type(e).__name__} ({str(e)[:120]}); row-count estimate left as None"
                )

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
            "source": getattr(self, "_source", "local"),
            "warnings": warnings,
        }

    def list_paths(
        self,
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Describe the FK graph reachable from this dataset / anchor set.

        Useful for schema exploration — answers "what tables could I
        reasonably include in ``include_tables`` given my anchor set?"
        Model-only analysis: no catalog query, no data fetch.

        Args:
            tables: Optional filter. When given, ``schema_paths`` includes
                only entries where the source OR target table is in this
                list. Anchor tables are always kept as traversal starting
                points regardless of the filter.

        Returns:
            A dict with these 6 keys (see design spec §6):

            - ``member_types``: dataset element types (same as
              ``anchor_types`` for :meth:`from_rids`-constructed
              Denormalizers).
            - ``anchor_types``: sorted list of distinct anchor table
              names.
            - ``reachable_tables``: ``{anchor_table: [reachable tables
              downstream via FK, sorted]}``.
            - ``association_tables``: sorted list of pure M:N association
              tables in the schema (detected via the planner's
              ``_is_topological_association`` predicate).
            - ``feature_tables``: sorted list of feature tables
              (via :meth:`DerivaModel.find_features`). Empty if the
              model doesn't expose ``find_features`` or has no features.
            - ``schema_paths``: ``{(source, target): [{path, direct}]}``
              — one entry per reachable ``(source, target)`` pair, each
              value listing the FK paths between them with a ``direct``
              flag indicating whether the path is a single FK hop.

        Example::

            d = Denormalizer(dataset)
            info = d.list_paths()

            # What types are in my dataset?
            print(info["member_types"])  # e.g., ["Image", "Subject"]

            # What can I reach from each type?
            for anchor, reach in info["reachable_tables"].items():
                print(f"{anchor} → {reach}")

            # Investigate multiple FK paths to Subject:
            for (src, tgt), paths in info["schema_paths"].items():
                if tgt == "Subject":
                    for p in paths:
                        print(f"  {src} → {tgt}: {' → '.join(p['path'])}"
                              f" (direct={p['direct']})")
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
            reach = model._planner._outbound_reachable(t, all_table_names)
            reachable_tables[t] = sorted(reach)

        # association_tables: pure M-to-N linking tables
        association_tables = sorted(t for t in all_table_names if model._planner._is_topological_association(t))

        # feature_tables: derive from DerivaModel.find_features (the canonical
        # feature-discovery API — see model/catalog.py:510). Each Feature's
        # .feature_table.name gives the backing table name. Degrades to [] if
        # the model doesn't expose find_features or the call raises.
        feature_tables: list[str] = []
        find_feats = getattr(model, "find_features", None)
        if callable(find_feats):
            try:
                feature_tables = sorted({f.feature_table.name for f in find_feats()})
            except Exception:
                feature_tables = []

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
                paths = model._planner._enumerate_paths(source, target, all_table_names)
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

    def _resolve_table_names(
        self,
        include_tables: list[str],
        *,
        via: list[str] | None = None,
        row_per: str | None = None,
    ) -> tuple[list[str], list[str], str | None]:
        """Translate caller-supplied names into canonical table names.

        Both :meth:`describe` and :meth:`as_dataframe` accept tables in
        ``include_tables`` / ``via`` / ``row_per`` by name. The natural
        mental model — confirmed by the rest of the DerivaML surface
        (:meth:`find_features`, :meth:`feature_values`,
        :meth:`lookup_feature`) — uses the **feature name** (e.g.
        ``"Image_Classification"``) rather than the underlying
        feature-association table name (e.g.
        ``"Execution_Image_Image_Classification"``). This helper
        accepts either form: real table names pass through unchanged;
        feature names are silently substituted for their backing
        association table.

        Resolution algorithm for each name:

        1. If ``name_to_table(t)`` succeeds → keep ``t`` as-is.
        2. Otherwise consult :meth:`DerivaModel.find_features` and look
           for a feature whose ``feature_name == t``. If exactly one
           match → substitute with ``feature.feature_table.name``. If
           multiple matches across different target tables → raise
           :class:`DerivaMLDenormalizeError` (ambiguous — caller should
           name the feature table directly).
        3. If neither path works → raise
           :class:`DerivaMLTableNotFound`. When the name doesn't match
           any feature either, the error makes clear that no table or
           feature with that name exists, so the user can tell whether
           they typo'd a feature name or pointed at the wrong feature.

        Args:
            include_tables: Caller's ``include_tables`` (mixed table
                names and feature names allowed).
            via: Caller's ``via`` argument.
            row_per: Caller's ``row_per`` argument.

        Returns:
            Tuple of ``(resolved_include_tables, resolved_via,
            resolved_row_per)`` — all canonical table names that
            ``name_to_table`` will accept.

        Raises:
            DerivaMLTableNotFound: A name doesn't match any catalog
                table and isn't a known feature name either. The
                error message names the offending entry so the user
                knows whether to fix a typo or pick a different
                feature.
            DerivaMLDenormalizeError: A feature name matches multiple
                features on different target tables — the helper
                can't disambiguate so the caller must pass the
                feature-association table name directly.

        Example::

            # Feature name silently resolves to the feature-association table.
            d._resolve_table_names(["Image", "Image_Classification"])
            # (["Image", "Execution_Image_Image_Classification"], [], None)

            # Real table names pass through untouched.
            d._resolve_table_names(["Image", "Subject"])
            # (["Image", "Subject"], [], None)
        """
        from deriva_ml.core.exceptions import (
            DerivaMLDenormalizeError,
            DerivaMLException,
            DerivaMLTableNotFound,
        )

        # The model may be None on minimal fixtures; in that case we
        # have no way to resolve names, so return inputs unchanged and
        # let downstream planner validation produce its own error.
        if self._model is None:
            return list(include_tables), list(via or []), row_per

        # Build the {feature_name: [feature]} index once per call.
        # ``find_features()`` walks every association in the catalog
        # so we don't want to re-run it for each name in the inputs.
        # Failures (no find_features, network errors) collapse to an
        # empty index — the helper then falls through to the
        # table-not-found branch and the caller gets the original
        # error.
        feature_index: dict[str, list[Any]] = {}
        find_feats = getattr(self._model, "find_features", None)
        if callable(find_feats):
            try:
                for f in find_feats():
                    fname = getattr(f, "feature_name", None)
                    if fname:
                        feature_index.setdefault(fname, []).append(f)
            except Exception:
                feature_index = {}

        def _resolve_one(t: str) -> str:
            """Return canonical table name for ``t`` or raise."""
            try:
                self._model.name_to_table(t)
                return t
            except DerivaMLException:
                pass
            # Not a table — try the feature index.
            matches = feature_index.get(t, [])
            if len(matches) == 1:
                resolved_name = matches[0].feature_table.name
                logger.info(
                    "Denormalizer: resolved feature name %r to feature table %r",
                    t,
                    resolved_name,
                )
                return resolved_name
            if len(matches) > 1:
                target_tables = sorted({m.target_table.name for m in matches})
                feature_tables = sorted({m.feature_table.name for m in matches})
                raise DerivaMLDenormalizeError(
                    f"Feature name {t!r} is ambiguous — it is defined on "
                    f"multiple target tables ({target_tables}). Pass the "
                    f"feature-association table name directly instead: "
                    f"one of {feature_tables}."
                )
            # Not a table and not a feature name. Distinguish the
            # error so the user knows it isn't a near-miss for a
            # feature they typed badly.
            known_features = sorted(feature_index.keys())
            if known_features:
                raise DerivaMLTableNotFound(
                    t,
                    msg=(f"No table or feature named {t!r} exists. Known feature names: {known_features}"),
                )
            raise DerivaMLTableNotFound(t)

        # RB-07: feature-name substitution can collapse two distinct
        # caller-supplied names onto the same canonical table (e.g.
        # ["Image_Classification", "Execution_Image_Image_Classification"]
        # both resolve to the feature-association table). Dedup the
        # resolved lists in first-seen order so the planner doesn't get
        # the same table twice — duplicates would trigger spurious
        # ambiguity/cardinality errors downstream.
        def _dedup(seq: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        resolved_include = _dedup([_resolve_one(t) for t in include_tables])
        resolved_via = _dedup([_resolve_one(t) for t in (via or [])])
        resolved_row_per = _resolve_one(row_per) if row_per is not None else None
        return resolved_include, resolved_via, resolved_row_per

    def _run(
        self,
        include_tables: list[str],
        *,
        row_per: str | None,
        via: list[str] | None,
        ignore_unrelated_anchors: bool,
        selector: Callable[[list["FeatureRecord"]], "FeatureRecord | None"] | None = None,
    ) -> DenormalizeResult:
        """Execute the full 4-phase denormalization pipeline.

        Called by :meth:`as_dataframe` and :meth:`as_dict`. Raises on any
        planner-rule violation (unlike :meth:`describe`, which reports
        ambiguities without raising).

        Pipeline:

        1. **Planner decisions** (Rules 2/5/6): resolve ``row_per`` via
           sink-finding or validate the explicit value, and check for
           path ambiguities — raises at the first violation.
        2. **Anchor classification** (Rules 7/8): partition anchors into
           ``scoping`` (filter), ``orphans`` (Rule 7 case 3), and
           ``ignored`` (Rule 7 case 5 silent drop / Rule 8 with flag).
           Raises if Rule 8 fires and the flag is off.
        3. **Main SQL**: delegate to :func:`_denormalize_impl` with
           ``row_per`` / ``via`` / ``selector`` threaded through. For
           live Datasets ``source="catalog"`` and a :class:`PagedClient`
           fetches rows before the join; for DatasetBag / fixtures
           ``source="local"`` and the engine is assumed pre-populated.
           Selector reduction (when supplied) runs inside
           ``_denormalize_impl`` after materialization, before this
           method's orphan-row combine.
        4. **Orphan-row combine**: both table-level orphans (case 3)
           AND per-RID orphans (scoping anchors whose specific RIDs
           didn't appear in the main result) are emitted as LEFT-JOIN-
           shaped rows and appended via :meth:`DenormalizeResult.extend`.

        Args:
            include_tables, row_per, via, ignore_unrelated_anchors,
            selector: forwarded from the public method.

        Returns:
            The final :class:`DenormalizeResult` — main SQL rows plus
            any orphan rows appended.
        """
        # Step 0: translate feature names to feature-association table
        # names so the caller can pass either form. This is what makes
        # ``include_tables=["Image", "Image_Classification"]`` work
        # symmetrically with the rest of the DerivaML surface
        # (``find_features``, ``feature_values``, ``lookup_feature``) —
        # see ``_resolve_table_names`` for the algorithm. Validation
        # errors (typo'd name, ambiguous feature) surface here, before
        # any planner work.
        include_tables, via_resolved, row_per = self._resolve_table_names(
            list(include_tables), via=via, row_per=row_per
        )
        via = via_resolved

        # Step 1: planner decisions (row_per, ambiguity checks).
        # _determine_row_per either validates an explicit row_per
        # (raising DownstreamLeaf if a downstream table is in
        # include_tables) or auto-infers via sink-finding (raising
        # MultiLeaf / NoSink if ambiguous or cyclic).
        resolved_row_per = self._model._planner._determine_row_per(
            include_tables=list(include_tables),
            via=list(via or []),
            row_per=row_per,
        )
        # Rule 6: check every (row_per, T) pair for multiple FK paths.
        # Unlike describe(), we raise on the first ambiguity detected so
        # callers get a clear error rather than a silently-wrong join.
        ambiguities = self._model._planner._find_path_ambiguities(
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

        # Step 3: main SQL via _denormalize_impl. Source + paged_client are
        # chosen in __init__ — "catalog" for live Datasets (needs PagedClient
        # to fetch rows before the join), "local" for DatasetBag / fixtures.
        #
        # Nested-dataset scoping: _denormalize_impl's SQL emits
        # ``WHERE Dataset.RID IN (dataset_rid, ...children)``. Without the
        # children RIDs, nested-dataset members (rows whose Dataset_X.Dataset
        # points at a descendant, not the root) never pass the WHERE and the
        # result comes back empty. Pull the descendant RIDs from the dataset
        # itself — DatasetLike exposes ``list_dataset_children(recurse=True)``
        # returning a list of child objects with ``.dataset_rid`` attributes.
        # Failure here is non-fatal (fixture-style datasets may not implement
        # it) — fall back to root-only scoping and the caller sees the same
        # behavior they had before nested support was wired in.
        dataset_children_rids: list[str] | None = None
        list_children = getattr(self._dataset, "list_dataset_children", None)
        if callable(list_children):
            try:
                children = list_children(recurse=True)
                dataset_children_rids = [
                    getattr(c, "dataset_rid", None) for c in children if getattr(c, "dataset_rid", None)
                ] or None
            except (AttributeError, TypeError):
                # RB-06: Fixture-shaped datasets or unusual DatasetLike
                # implementations may not support recurse; silently fall
                # back to root-only scoping rather than break denormalize.
                dataset_children_rids = None
            except Exception as e:
                # RB-06: For a live catalog source, a transient failure
                # (network, permissions) silently dropping to root-only
                # produces the silent-zero pattern on nested datasets.
                # Surface the failure as a WARNING log so operators can
                # see why their nested-member rows are missing. Fixtures
                # are caught by the narrow except above, so reaching here
                # against ``source='catalog'`` is a real catalog issue.
                if self._source == "catalog":
                    logger.warning(
                        "list_dataset_children failed on live catalog: %s; "
                        "falling back to root-only scoping (nested-dataset "
                        "members will be absent from the result).",
                        e,
                    )
                dataset_children_rids = None

        main_result = _denormalize_impl(
            model=self._model,
            engine=self._engine,
            orm_resolver=self._orm_resolver,
            dataset_rid=self._dataset_rid,
            include_tables=list(include_tables),
            dataset=self._dataset,
            dataset_children_rids=dataset_children_rids,
            source=self._source,
            paged_client=self._paged_client,
            row_per=resolved_row_per,
            via=list(via or []) or None,
            selector=selector,
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
            # RB-04: Build per-anchor RID labels via ``denormalize_column_name``
            # so multi-schema datasets resolve to ``schema.Table.RID`` rather
            # than the bare ``Table.RID`` form. Previously this loop assumed
            # single-schema and would silently fail to match any row on
            # multi-schema datasets, marking every anchor RID as orphan.
            from deriva_ml.model.catalog import denormalize_column_name

            _, _column_specs, _multi_schema = self._model._planner._prepare_wide_table(
                self._dataset,
                self._dataset_rid,
                list(include_tables),
                row_per=resolved_row_per,
                via=list(via or []) or None,
            )
            table_to_schema: dict[str, str] = {t: s for s, t, _c, _tp in _column_specs}
            rid_label_by_anchor: dict[str, str] = {
                t: denormalize_column_name(table_to_schema.get(t, ""), t, "RID", _multi_schema) for t in seen_by_table
            }
            for row in main_result.iter_rows():
                for t, seen in seen_by_table.items():
                    val = row.get(rid_label_by_anchor[t])
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
        """Return the anchor set as ``{table_name: [rid, rid, ...]}``.

        Delegates to :meth:`~deriva_ml.dataset.Dataset.list_dataset_members`
        with ``recurse=True`` (so nested-dataset members are included
        per Rule 9) and — if this Denormalizer was constructed with a
        ``version`` — passes that version through so member enumeration
        runs against the same snapshot the main SQL join will use.

        Reshapes the result into the dict form used by
        :meth:`_classify_anchors`.
        """
        # Pass version through ONLY if it was supplied at construction;
        # list_dataset_members on DatasetBag doesn't accept version (the
        # bag is already at a fixed snapshot), so omit the kwarg when
        # version is None to stay compatible with both shapes.
        kwargs: dict[str, Any] = {"recurse": True}
        if self._version is not None:
            kwargs["version"] = self._version
        members = self._dataset.list_dataset_members(**kwargs)
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

        Implements spec §3.7's six cases:

        - Case 1 — table == row_per → scoping
        - Case 2 — table in include_tables, reaches row_per → scoping
        - Case 3 — table in include_tables, can't reach row_per → orphan
          (per-RID emission happens downstream in ``_run``)
        - Case 4 — table NOT in include_tables, reaches row_per → scoping
          (filter-only; no orphan row because columns not in output)
        - Case 5 — table NOT in include_tables, IS upstream of
          ``include_tables ∪ via`` but can't reach row_per → silently
          dropped (into ``ignored``) regardless of flag. Per §3.8: case 5
          contributes no output either way so there's nothing to warn about.
        - Case 6 — table has NO FK path to ``include_tables ∪ via`` →
          raise :class:`DerivaMLDenormalizeUnrelatedAnchor` unless
          ``ignore_unrelated_anchors=True``, in which case add to ``ignored``.

        Args:
            anchors: table_name -> list of RIDs.
            include_tables, via, row_per: the planner inputs.
            ignore_unrelated_anchors: if False, raise on Rule-8 (case 6) anchors.

        Returns:
            Tuple of (scoping_anchors, orphan_anchors, ignored_anchors). Each
            is a table_name -> list of RIDs dict.

        Raises:
            DerivaMLDenormalizeUnrelatedAnchor: if any case-6 anchor is
                present and ``ignore_unrelated_anchors=False``.
        """
        from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor

        scoping: dict[str, list[str]] = {}
        orphans: dict[str, list[str]] = {}
        ignored: dict[str, list[str]] = {}
        # case-6 tables: no FK path to include_tables ∪ via at all.
        unrelated_case_6: list[str] = []

        include_set = set(include_tables)
        via_set = set(via)
        subgraph = include_set | via_set

        for table, rids in anchors.items():
            # Skip empty anchor sets entirely. An anchor dict returned by
            # list_dataset_members may contain a key for every association
            # table on Dataset (e.g. Dataset_File) even when no members of
            # that type were added — the dict ends up with e.g.
            # ``{"File": []}``. Those zero-RID entries can't contribute
            # anything to the output and should not trigger Rule 8, which
            # warns about anchors "that would contribute nothing." If the
            # RID list is empty, there's nothing TO contribute and nothing
            # to warn about.
            if not rids:
                continue

            # Case 1: table == row_per → scoping
            if table == row_per:
                scoping[table] = list(rids)
                continue

            # Rule 7 reachability is direction-agnostic at the anchor-
            # classification level: an anchor "filters" row_per if there
            # is ANY FK chain connecting the two, in either direction.
            #
            # - Downstream reach (`_outbound_reachable`): anchor's rows
            #   are referenced by row_per rows (Dataset → Dataset_Image
            #   → Image reaches Image via membership, or Subject →
            #   Image reaches Image via Image.Subject FK).
            # - Upstream reach (`_outbound_reachable` from row_per to
            #   anchor): anchor is above row_per in the FK hierarchy
            #   (e.g., Image anchor when row_per=Subject — Image has
            #   Image.Subject pointing at Subject).
            #
            # Both shapes give the anchor's RIDs a valid filter role
            # over row_per rows. The current engine's join generation
            # handles both via dataset-membership scoping and FK-chain
            # traversal.
            downstream_from_anchor = self._model._planner._outbound_reachable(table, subgraph | {table})
            reaches_row_per_downstream = row_per in downstream_from_anchor
            # For the upstream side: does row_per's downstream-reach set
            # contain the anchor? (row_per's outbound set = everything
            # row_per hoists OR reaches; that is symmetric to "anchor
            # can reach row_per via along-FK chain" modulo direction.)
            downstream_from_row_per = self._model._planner._outbound_reachable(row_per, subgraph | {table})
            reaches_row_per_upstream = table in downstream_from_row_per
            reaches_row_per = reaches_row_per_downstream or reaches_row_per_upstream

            if reaches_row_per:
                # Case 2 (table in include_tables) or Case 4 (not): scoping.
                scoping[table] = list(rids)
                continue

            if table in include_set:
                # Case 3: in include_tables but can't reach row_per → orphan.
                orphans[table] = list(rids)
                continue

            # Case 5 vs Case 6: does this table have ANY FK connection
            # (either direction) to the subgraph? If yes → case 5
            # (silent drop). If no → case 6 (raise unless flag).
            connected_downstream = bool(downstream_from_anchor & subgraph)
            # Upstream connection: is the anchor reachable from any
            # subgraph table's downstream walk? Cheapest check is per-
            # subgraph-table, but we already computed it for row_per
            # above; do a quick survey of the remaining subgraph tables.
            connected_upstream = table in downstream_from_row_per
            if not connected_upstream and not connected_downstream:
                # One more pass: for any OTHER table in the subgraph,
                # does its downstream reach include the anchor?
                for s in subgraph:
                    if s == row_per or s == table:
                        continue
                    if table in self._model._planner._outbound_reachable(s, subgraph | {table}):
                        connected_upstream = True
                        break
            if connected_downstream or connected_upstream:
                # Case 5: table connected to include_tables ∪ via but
                # doesn't reach row_per. Silent drop per spec §3.8.
                ignored[table] = list(rids)
            else:
                # Case 6: no FK path at all.
                unrelated_case_6.append(table)

        if unrelated_case_6 and not ignore_unrelated_anchors:
            raise DerivaMLDenormalizeUnrelatedAnchor(
                unrelated_tables=sorted(set(unrelated_case_6)),
                include_tables=list(include_tables),
            )
        if unrelated_case_6:
            # Flag is set → add case-6 anchors to ignored alongside case-5.
            for table in unrelated_case_6:
                ignored[table] = list(anchors[table])

        return scoping, orphans, ignored

    def _emit_orphan_rows(
        self,
        orphans: dict[str, list[str]],
        *,
        include_tables: list[str],
        row_per: str,
    ) -> list[dict[str, Any]]:
        """Emit one output row per orphan anchor (Rule 7 case 3).

        Each orphan row is an :class:`as_dataframe`-shaped dict with:

        - The anchor's own columns populated from its fetched row.
        - All other columns (``row_per``-side and any other table in
          ``include_tables``) set to ``None``.

        This is the LEFT-JOIN shape spec'd by Rule 7 case 3: upstream
        anchors that don't reach a ``row_per`` row still contribute an
        output row, preserving their identity for the caller.

        **Implementation detail** — batched fetch: a single
        ``SELECT ... WHERE RID IN (:rids)`` per anchor table rather than
        one connection per RID. Skips anchor tables whose ORM class
        can't be resolved (logs a warning).

        Args:
            orphans: ``{anchor_table: [orphan RIDs]}`` from
                :meth:`_classify_anchors` plus the per-RID orphans
                discovered in :meth:`_run` Step 4a.
            include_tables: the user's requested tables — determines
                which output columns appear.
            row_per: the resolved leaf table (unused here but accepted
                for symmetry with other helpers).

        Returns:
            List of output dicts, one per orphan RID that was
            successfully fetched. Keys are ``Table.column`` /
            ``schema.Table.column`` labels.
        """
        from sqlalchemy import select

        from deriva_ml.model.catalog import denormalize_column_name

        orphan_rows: list[dict[str, Any]] = []

        # Get the full column spec so we know what keys to populate.
        _, column_specs, multi_schema = self._model._planner._prepare_wide_table(
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

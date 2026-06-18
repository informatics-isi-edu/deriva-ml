"""Denormalization planner — schema-graph reachability + JOIN tree construction.

**Purpose.** Given a Deriva schema and a set of "interesting" tables
(``include_tables``), compute *how* to join those tables into one wide
result. The planner produces a JOIN plan (which tables, in what order,
on which FK columns, ``inner`` vs. ``left``) and the projected
columns. It does **not** issue any SQL. The plan is consumed by
``local_db/denormalize.py`` (which emits the SQL) and by
``core/mixins/dataset.py:list_denormalized_columns`` (which only needs
the column list).

------------------------------------------------------------------
Transparency model
------------------------------------------------------------------

A *transparent* table is one the planner walks **through** without
asking the caller to name it in ``include_tables`` or ``via``. The
planner treats two kinds of tables as transparent:

1. **Topological association tables** (M:N link tables):
   :meth:`DenormalizePlanner._is_topological_association`. Exactly two
   domain FKs (system FKs to ``ERMrest_Client`` / ``ERMrest_Group`` are
   ignored). Examples: ``Dataset_Image``, ``Subject_Sample``. These
   exist purely to link two domain tables — the planner hops through
   them in both directions so ``A → assoc → B`` discovers B from A.

2. **Feature association tables** (DerivaML feature value tables):
   :meth:`DenormalizePlanner._is_feature_association`. The association
   ``create_feature`` builds — at least four domain FKs: the feature's
   target domain table, one or more value FKs (vocabulary term / asset
   / metadata), an FK to the ML schema's ``Feature_Name`` vocab, and an
   FK to the ML schema's ``Execution`` table. The ``Feature_Name`` and
   ``Execution`` FKs are the two reliable structural markers (the
   predicate keys off them, not the FK count). Examples:
   ``Execution_Image_Image_Classification``,
   ``Execution_Subject_Diagnosis``. These exist to record one
   feature value per (target, execution) — the Execution FK is an
   audit / provenance edge and the Feature_Name FK identifies the
   feature, *not* domain edges that callers ever want to filter on.
   From the planner's denormalization perspective they behave like a
   pure 2-FK bridge between the target and the value table.

In both cases, transparency means:

- the planner walks **through** the table in both directions when
  determining reachability (:meth:`_outbound_reachable`);
- the table does **not** count as an interior table for the
  diamond-disambiguation rule (Path ambiguity in :meth:`_find_path_ambiguities`);
- the table's own columns are excluded from the output **unless** the
  caller explicitly names it in ``include_tables``, at which point
  its non-FK metadata columns (``Feature_Name`` / ``Confidence`` /
  etc.) become regular projected columns and the table behaves like
  any requested table (it still routes the join, but it's no longer
  "transparent" for *that* call).

What is intentionally **not** transparent:

- Multi-way domain associations that lack the feature marker FKs
  (no FK to ``Feature_Name`` and/or ``Execution``) — even when they
  carry an ``Execution`` provenance edge, an association without a
  ``Feature_Name`` FK is a genuine multi-way join whose extra edges
  are domain semantics, not feature plumbing, and the caller should
  signal intent.
- Tables that satisfy neither predicate. The planner treats anything
  not in ``include_tables`` / ``via`` and not transparent as a dead
  end during reachability — that's how it stops itself from sliding
  off the requested star into the rest of the schema.

**This module is internal.** The user-facing API for denormalization
is :class:`local_db.denormalize.Denormalizer`. The named semantic rules
this planner enforces (Row grain, Column projection, Column hoisting,
Downstream-leaf rejection, Path ambiguity, Anchor disposition), the full pipeline
(planner → row-population → SQL emission), the state-ownership model,
and the fragility map for the denormalize subsystem all live in
``docs/user-guide/denormalization.md`` — start there when touching
anything in this module. All methods are
underscore-prefixed because they are private from the user's
perspective.

------------------------------------------------------------------
Algorithmic overview
------------------------------------------------------------------

Denormalization takes a star-shaped slice of the FK graph and turns
it into a wide table where each row corresponds to one *leaf-table*
row (``row_per``). The planner answers four interlocking questions:

1. **Which table is the leaf?** (Row grain — spec §3.2)
   Find the unique sink in the directed FK subgraph on
   ``include_tables ∪ via``: the table with no outbound FK to any
   other table in the set. Zero sinks → cycle (rare; raise).
   Multiple sinks → ambiguous (raise with candidates listed).
   Caller can override via ``row_per=`` and bypass auto-inference.

2. **Is the leaf legal?** (Downstream-leaf rejection — spec §3.5)
   When the caller passes an explicit ``row_per``, verify no table
   in ``include_tables`` is *downstream* of it. A downstream table
   would force an aggregation (one ``row_per`` row vs. many
   downstream rows), which this engine doesn't do.

3. **Is the FK path unique?** (Path ambiguity — spec §3.6)
   For every (``row_per``, ``T``) pair with ``T ∈ include_tables ∪
   via``, enumerate all simple FK paths between them. If more than
   one path exists, the user must disambiguate (add an intermediate
   to ``include_tables`` or ``via``). Pure association tables are
   *transparent* — they bridge two tables without contributing
   columns, and they don't count toward path multiplicity.

4. **How do we actually JOIN?** (Column hoisting — spec §3.4)
   Build a :class:`JoinNode` tree rooted at the leaf table, with
   one edge per FK along the selected paths. Nullable FK columns
   produce ``LEFT JOIN``\\s so upstream rows aren't dropped.
   Pre-order traversal of the tree gives the JOIN order.

------------------------------------------------------------------
Method layering
------------------------------------------------------------------

The methods compose in three layers, lowest first:

**Reachability primitives** — raw graph traversal over the FK graph:

- :meth:`DenormalizePlanner._fk_neighbors` — undirected FK adjacency.
- :meth:`DenormalizePlanner._downstream_fk_sources` — directional
  (inbound-only) version.
- :meth:`DenormalizePlanner._outbound_reachable` — BFS reachability
  with transparent association hops.
- :meth:`DenormalizePlanner._schema_to_paths` — DFS path enumeration
  with cycle detection and vocab termination.
- :meth:`DenormalizePlanner._is_topological_association` — 2-FK
  predicate (pure M:N bridges).
- :meth:`DenormalizePlanner._is_feature_association` — feature predicate
  (one FK targets the ML schema's ``Feature_Name`` vocab and one targets
  the ML schema's ``Execution`` table; the rest are domain/value edges).
- :meth:`DenormalizePlanner._is_transparent_intermediate` — union of
  the two predicates; this is what the reachability primitives consult.
- :meth:`DenormalizePlanner._table_relationship` — column-pair
  extraction for a specific (table1, table2) FK.

**Planner rules** — denormalization semantics layered on primitives:

- :meth:`DenormalizePlanner._find_sinks` — Row grain.
- :meth:`DenormalizePlanner._determine_row_per` — Row grain +
  Downstream-leaf rejection.
- :meth:`DenormalizePlanner._enumerate_paths` — Path ambiguity path discovery
  (with transparency filter).
- :meth:`DenormalizePlanner._find_path_ambiguities` — Path ambiguity detection.

**Top-level entries** — produce the consumer-facing plan:

- :meth:`DenormalizePlanner._build_join_tree` — per-element JoinNode
  tree.
- :meth:`DenormalizePlanner._prepare_wide_table` — the full plan
  (element trees × column projections × multi-schema flag).

------------------------------------------------------------------
Consumers
------------------------------------------------------------------

The planner is held on :class:`DerivaModel` as the lazy
``_planner`` property:

* ``local_db/denormalize.py:denormalize()`` — calls
  ``_prepare_wide_table`` to get the plan, then emits SQL.
* ``local_db/denormalizer.py`` — calls ``_prepare_wide_table``,
  ``_find_sinks``, ``_determine_row_per``,
  ``_find_path_ambiguities``, ``_outbound_reachable``, and
  ``_enumerate_paths`` from the ``Denormalizer`` public class.
* ``core/mixins/dataset.py:list_denormalized_columns`` — calls
  ``_prepare_wide_table`` purely to extract the column list (no
  data fetch).
* ``dataset/dataset_bag.py`` — calls ``_schema_to_paths`` to walk
  the bag's local schema.

------------------------------------------------------------------
Why a separate module
------------------------------------------------------------------

The planner is ~1100 LoC of algorithmic code with a narrow consumer
set (four call sites in total). ``DerivaModel`` is a wide-fan-out
class touched by every mixin in the codebase, and bundling the
planner in there muddied its responsibility. Phase 3 (audit §5.2 in
``docs/archive/2026-05-audit-working-drafts/deriva-ml-audit-2026-05-phase2-model.md``)
extracted the planner into this module so:

* ``DerivaModel`` becomes a pure introspection class (~1100 LoC,
  consumed everywhere).
* The planner becomes a focused algorithm class (~1100 LoC,
  consumed by ``local_db/`` + a couple of single-line sites).
* The split lines up with the fan-out: small consumer set →
  separate module, large consumer set → shared base class.

Example:
    >>> from deriva_ml.model.denormalize_planner import DenormalizePlanner  # doctest: +SKIP
    >>> planner = DenormalizePlanner(ml.model)  # doctest: +SKIP
    >>> element_tables, columns, multi = planner._prepare_wide_table(  # doctest: +SKIP
    ...     dataset=None,
    ...     dataset_rid=None,
    ...     include_tables=["Subject", "Image"],
    ... )
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib

# Standard library imports
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

Column = _ermrest_model.Column
Table = _ermrest_model.Table

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import get_logger

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel

logger = get_logger(__name__)


@dataclass
class JoinNode:
    """A node in the join tree used by ``DenormalizePlanner._prepare_wide_table``.

    The join tree is a rooted tree where each node represents a table that
    participates in the denormalized query.  The root is the element table
    (e.g., Image), and children are tables that should be JOINed to it by
    following FK relationships.

    Attributes:
        table: The ermrest ``Table`` object for this node.
        table_name: Human-readable name (``table.name``).
        join_type: ``"inner"`` or ``"left"`` -- LEFT JOIN is used when the
            FK column is nullable so that rows with NULL FK values are
            preserved.
        fk_columns: ``(fk_col, pk_col)`` pairs describing how this node
            joins to its parent.  ``None`` for the root node.
        is_association: If True, this table is needed for the JOIN chain
            but its columns are excluded from the output (e.g., M:N
            linking tables like ``ClinicalRecord_Observation``).
        children: Child nodes to join after this one.
    """

    table: Any  # ermrest Table
    table_name: str
    join_type: str = "inner"  # "inner" or "left"
    fk_columns: list[tuple] | None = None  # list[(fk_col, pk_col)]
    is_association: bool = False
    children: list["JoinNode"] = field(default_factory=list)

    def walk(self) -> list["JoinNode"]:
        """Return a pre-order traversal of the tree (self first, then children)."""
        result = [self]
        for child in self.children:
            result.extend(child.walk())
        return result

    def walk_edges(self) -> list[tuple["JoinNode", "JoinNode"]]:
        """Return (parent, child) pairs in pre-order traversal."""
        edges = []
        for child in self.children:
            edges.append((self, child))
            edges.extend(child.walk_edges())
        return edges


def denormalize_column_name(schema_name: str, table_name: str, column_name: str, multi_schema: bool) -> str:
    """Build a prefixed column name for denormalized output.

    Uses dot notation to avoid ambiguity with column names that contain
    underscores (e.g., ``Acquisition_Date``).

    Args:
        schema_name: Schema the table belongs to.
        table_name: Table the column belongs to.
        column_name: Raw column name.
        multi_schema: If True, include schema prefix for disambiguation.

    Returns:
        Prefixed column name, e.g. ``Image.Filename`` or ``test-schema.Image.Filename``.
    """
    if multi_schema:
        return f"{schema_name}.{table_name}.{column_name}"
    return f"{table_name}.{column_name}"


# Default tables to skip during FK path traversal.
# These are ML schema tables that create unwanted traversal branches:
# - Dataset_Dataset: nested dataset self-reference (handled separately)
# - Execution: execution tracking (not useful for data traversal)
_DEFAULT_SKIP_TABLES = frozenset({"Dataset_Dataset", "Execution"})


class DenormalizePlanner:
    """JOIN-plan generator for the denormalization subsystem.

    See the module docstring for the algorithmic overview and the
    layering of methods (reachability primitives → planner rules →
    top-level entries). The planner is the algorithm layer that
    :class:`Denormalizer` (the user-facing class in
    ``local_db/denormalize.py``) sits on top of.

    Args:
        model: The :class:`DerivaModel` to plan against. The planner
            reads schemas/tables through the model and never mutates
            it. One planner per model is enough; :class:`DerivaModel`
            holds the planner on its ``_planner`` property and caches
            the instance for reuse.

    Example:
        >>> from deriva_ml.model.denormalize_planner import DenormalizePlanner  # doctest: +SKIP
        >>> planner = DenormalizePlanner(ml.model)  # doctest: +SKIP
        >>> sinks = planner._find_sinks(["Subject", "Image"])  # doctest: +SKIP
    """

    def __init__(self, model: "DerivaModel") -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Reachability primitives
    # ------------------------------------------------------------------

    def _is_topological_association(self, name_or_table: str | Table) -> bool:
        """Predicate for the "transparent intermediate" rule.

        An association table (like ``Dataset_Image`` linking ``Dataset``
        and ``Image``) has exactly two **domain** FKs pointing at the
        tables it links. The denormalization planner treats such
        tables as transparent: they're joined through but their
        columns are excluded from the output unless the caller
        explicitly names them in ``include_tables``. Pure association
        tables don't count toward path multiplicity in the Path ambiguity
        check either.

        **Topology, not purity.** Association-ness is determined by FK
        arity alone, not by whether the table also carries metadata
        columns. Real Deriva linkage tables routinely carry annotation
        data (``Role``, ``Ordinal``, ``Comment``, etc.) while
        remaining semantically M:N bridges. Once the caller names the
        table in ``include_tables``, those metadata columns appear in
        the output and the table loses its transparent status for
        that call.

        **Domain FKs only.** Every Deriva table has system FKs to
        ``ERMrest_Client`` / ``ERMrest_Group`` (for ``RCB`` / ``RMB``).
        Those don't make a real domain table a 3-FK association; we
        skip them when counting.

        Compare with ``Table.is_association()`` (ermrest's
        ``find_associations(pure=True)``): we're stricter in one
        direction (ignoring system FKs) and looser in another (not
        requiring purity), which together make this the right
        predicate for the denormalization rules.

        Args:
            name_or_table: table name (looked up via
                ``self.model.name_to_table``) or a :class:`Table`
                instance.

        Returns:
            ``True`` if the table has exactly 2 domain FKs.

        Example::

            planner._is_topological_association("Dataset_Image")       # True
            planner._is_topological_association("Dataset_Image_Role")  # True — extra Role col OK
            planner._is_topological_association("Image")               # False (has ≤1 FK)
            planner._is_topological_association("Observation")         # False (has 1 FK)
        """
        # Catch only ``DerivaMLException`` here — that's the
        # specific failure mode for "table name not in the model"
        # (raised by ``name_to_table``). Any other exception
        # (attribute errors, type errors, real bugs) propagates
        # so it surfaces loudly instead of silently returning
        # False and producing wrong rule decisions downstream.
        try:
            tbl = name_or_table if hasattr(name_or_table, "foreign_keys") else self.model.name_to_table(name_or_table)
        except DerivaMLException:
            return False
        fks = list(tbl.foreign_keys)
        # Domain FKs exclude the system FKs to ERMrest_Client /
        # ERMrest_Group that every table carries (for RCB/RMB).
        domain_fks = [fk for fk in fks if fk.pk_table.name not in ("ERMrest_Client", "ERMrest_Group")]
        return len(domain_fks) == 2

    def _is_feature_association(self, name_or_table: str | Table) -> bool:
        """Predicate for DerivaML feature-association transparency.

        A feature-association table is the association
        :meth:`~deriva_ml.core.mixins.feature.FeatureMixin.create_feature`
        builds for every feature. Its canonical FK set is::

            Execution_Image_Image_Classification
                -> Image                    (feature target)
                -> Image_Class              (value: vocab term, asset, or metadata)
                -> Feature_Name             (-> ML Feature_Name vocab)
                -> Execution                (-> ML Execution, provenance / audit)

        That is **four or more domain FKs**: the target domain table,
        one or more value FKs (a vocabulary term, an asset, or a
        metadata table), the ``Feature_Name`` vocabulary FK, and the
        ``Execution`` provenance FK. ``create_feature`` always emits
        the ``Feature_Name`` and ``Execution`` FKs, so they are the two
        reliable structural markers of a feature; the target and value
        FKs vary per feature, so their *count* is not a usable
        discriminator.

        The ``Execution`` FK is an audit edge (it records *which run*
        asserted the value), and the ``Feature_Name`` FK identifies the
        feature — neither is a semantic relationship a caller would
        ever join through. The denormalization planner therefore treats
        feature-association tables as transparent in the same sense as
        topological 2-FK associations: the planner hops through them in
        both directions during reachability, and they don't count as
        interior tables when checking for diamond ambiguity.

        **Keyed off the marker FKs, not the FK count.** This predicate
        deliberately mirrors ``DerivaModel.find_features``'s
        ``is_feature`` column-presence test (``model/catalog.py``) —
        the DerivaML-native definition of a feature — by requiring an
        FK to the ML schema's ``Feature_Name`` vocab AND an FK to the
        ML schema's ``Execution`` table. An FK-count equality check is
        wrong: the value FKs make every real feature ≥4 domain FKs, and
        ``create_feature`` cannot produce a 3-FK feature.

        **Why we don't widen** :meth:`_is_topological_association` **instead**:
        the 2-FK predicate is intentionally strict because the diamond
        rule (Path ambiguity) uses it. Conflating "transparent bridge" with
        "feature" inside a single predicate makes the rule docs harder
        to read and risks accidentally treating other multi-FK shapes
        as transparent in the future. The two predicates compose at the
        call sites (mostly :meth:`_outbound_reachable`) where
        transparency is consulted.

        **Structural, not naming.** The predicate ignores the table
        name (``Execution_*`` is a convention, not a contract) and
        keys off the FK shape. Domain FKs exclude the system FKs to
        ``ERMrest_Client`` / ``ERMrest_Group``.

        Args:
            name_or_table: table name (looked up via
                ``self.model.name_to_table``) or a :class:`Table`
                instance.

        Returns:
            ``True`` if the table has exactly one FK to the ML schema's
            ``Feature_Name`` vocab AND exactly one FK to the ML schema's
            ``Execution`` table.

        Example::

            # Real 4-FK feature: target + value + Feature_Name + Execution.
            planner._is_feature_association(
                "Execution_Image_Image_Classification"
            )                                                          # True
            planner._is_feature_association("Dataset_Image")           # False (2-FK assoc)
            planner._is_feature_association("Image")                   # False (no marker FKs)
            # 4-FK domain association WITHOUT a Feature_Name FK
            # (e.g. an Execution-stamped multi-way join) is not a feature.
            planner._is_feature_association("FourWayAssoc")            # False (no Feature_Name FK)
        """
        # Same narrowing as ``_is_topological_association``: only
        # ``DerivaMLException`` from ``name_to_table`` is the
        # legitimate "unknown table → False" path; everything else
        # propagates.
        try:
            tbl = name_or_table if hasattr(name_or_table, "foreign_keys") else self.model.name_to_table(name_or_table)
        except DerivaMLException:
            return False
        fks = list(tbl.foreign_keys)
        # Domain FKs exclude the system FKs to ERMrest_Client /
        # ERMrest_Group that every table carries (for RCB/RMB).
        domain_fks = [fk for fk in fks if fk.pk_table.name not in ("ERMrest_Client", "ERMrest_Group")]
        # Key off the two defining markers of a DerivaML feature, NOT
        # the FK count: an FK to the ML schema's ``Feature_Name`` vocab
        # AND an FK to the ML schema's ``Execution`` table. This mirrors
        # ``DerivaModel.find_features``'s ``is_feature`` column-presence
        # test (see model/catalog.py) so the planner and the rest of
        # deriva-ml agree on what a feature is. A real feature carries
        # ``target + value(s) + Feature_Name + Execution`` — at least 4
        # domain FKs — so an FK-count equality check (the original
        # ``!= 3`` guard) rejected every real feature table.
        ml_schema = self.model.ml_schema
        execution_fks = [
            fk for fk in domain_fks if fk.pk_table.name == "Execution" and fk.pk_table.schema.name == ml_schema
        ]
        feature_name_fks = [
            fk for fk in domain_fks if fk.pk_table.name == "Feature_Name" and fk.pk_table.schema.name == ml_schema
        ]
        return len(execution_fks) == 1 and len(feature_name_fks) == 1

    def _is_transparent_intermediate(self, name_or_table: str | Table) -> bool:
        """Return ``True`` if the table is transparent (assoc or feature-assoc).

        Convenience predicate that ORs
        :meth:`_is_topological_association` and
        :meth:`_is_feature_association`. See the module-level
        "Transparency model" section for the conceptual contract.

        Args:
            name_or_table: table name or :class:`Table` instance.

        Returns:
            ``True`` if either transparency predicate matches.

        Example::

            planner._is_transparent_intermediate("Dataset_Image")
            # True — 2-FK topological association.

            planner._is_transparent_intermediate(
                "Execution_Image_Image_Classification"
            )
            # True — feature association (Feature_Name + Execution FKs).

            planner._is_transparent_intermediate("Image")
            # False — domain table, not a bridge.
        """
        return self._is_topological_association(name_or_table) or self._is_feature_association(name_or_table)

    def _fk_neighbors(self, table: str | Table) -> set[Table]:
        """Return FK-neighbor tables of *table* (outbound + inbound, deduplicated).

        The undirected FK-adjacency primitive used by schema traversal.
        Follows both ``table.foreign_keys`` (outbound: tables *table*
        points at) and ``table.referenced_by`` (inbound: tables that
        point at *table*), filters to valid schemas (``domain_schemas ∪
        {ml_schema}``), and deduplicates so that multiple FKs between
        the same two tables count as one edge.

        **Direction-agnostic**: use :meth:`_downstream_fk_sources` for
        the directional (inbound-only) variant when you need to
        distinguish upstream from downstream.

        Extracted from a nested ``find_arcs`` in :meth:`_schema_to_paths`
        so the denormalization planner can reuse it as the FK-traversal
        primitive.

        Args:
            table: table name (looked up via :meth:`name_to_table`) or
                :class:`Table` instance.

        Returns:
            Set of :class:`Table` objects reachable from *table* via one
            FK arc (either direction), deduplicated by target.

        Example::

            # For Image, which has Image.Subject → Subject and is
            # referenced by Dataset_Image.Image:
            planner._fk_neighbors("Image")
            # {<Table Subject>, <Table Dataset_Image>}
        """
        tbl = table if hasattr(table, "foreign_keys") else self.model.name_to_table(table)
        valid_schemas = self.model.domain_schemas | {self.model.ml_schema}
        # Outbound edges: tables this table's FKs point at.
        # Inbound edges: tables that have FKs pointing at this table.
        arc_list = [fk.pk_table for fk in tbl.foreign_keys] + [fk.table for fk in tbl.referenced_by]
        # Filter out system/auxiliary schemas (ERMrest_Client, public, etc.).
        arc_list = [t for t in arc_list if t.schema.name in valid_schemas]
        # Deduplicate: multi-FK targets (e.g., two FKs pointing at the
        # same table) should count as one neighbor. Downstream callers
        # handle specific FK selection via :meth:`_table_relationship`.
        seen: set[Table] = set()
        deduped: list[Table] = []
        for t in arc_list:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return set(deduped)

    def _downstream_fk_sources(self, table: str | Table) -> set[Table]:
        """Return tables that have an FK pointing AT *table* (directional downstream).

        Denormalization direction vocabulary:

        - **Upstream** = fewer rows per unit. Subject is upstream of Image
          because each Image has exactly one Subject.
        - **Downstream** = more rows per unit. Image is downstream of
          Subject because each Subject can have many Images.

        In ERMrest terms: if ``Image.Subject`` is an FK pointing at
        ``Subject.RID``, then Image is downstream of Subject — which
        means Image is in ``Subject.referenced_by``.

        This method returns direct downstream neighbors only — it does
        NOT do transparent association-table hopping. Callers that need
        "all reachable downstream tables, hopping through associations"
        should use :meth:`_outbound_reachable`.

        Compare with :meth:`_fk_neighbors`, which is direction-agnostic
        and returns both upstream and downstream neighbors.

        Args:
            table: table name (looked up via :meth:`name_to_table`) or
                :class:`Table` instance.

        Returns:
            Set of :class:`Table` objects whose FK points at *table*,
            filtered to the valid schemas (``domain_schemas ∪
            {ml_schema}``).

        Example::

            # Subject is pointed at by Image.Subject and Observation.Subject:
            planner._downstream_fk_sources("Subject")
            # {<Table Image>, <Table Observation>}

            # Image is pointed at by Dataset_Image.Image:
            planner._downstream_fk_sources("Image")
            # {<Table Dataset_Image>}
        """
        valid_schemas = self.model.domain_schemas | {self.model.ml_schema}
        tbl = table if hasattr(table, "foreign_keys") else self.model.name_to_table(table)
        targets: set[Table] = set()
        # Tables with FK pointing at us are downstream
        for fk in tbl.referenced_by:
            src = fk.table
            if src.schema.name not in valid_schemas:
                continue
            targets.add(src)
        return targets

    def _outbound_reachable(
        self,
        from_table: str,
        tables_in_set: set[str],
    ) -> set[str]:
        """Return tables in ``tables_in_set`` downstream of ``from_table``.

        BFS reachability over the FK graph in the one-to-many direction.
        Composes :meth:`_downstream_fk_sources` plus association-
        transparency logic — does NOT walk FKs directly.

        **Transparent hops**: when the walker hits a *transparent
        intermediate* (per :meth:`_is_transparent_intermediate`) that
        isn't in ``tables_in_set``, it hops through it in BOTH
        directions — both the tables that point at the intermediate
        (inbound) AND the tables the intermediate's FKs point at
        (outbound). This lets ``A → bridge → B`` discover B from A
        even when ``A → bridge`` is an inbound FK and ``bridge → B``
        is an outbound FK. Without this bidirectional hop, many-to-many
        relationships (Dataset ↔ Image via Dataset_Image) and feature
        targets (Image ↔ Image_Classification via the feature-assoc
        table) wouldn't be traversable.

        Both topological 2-FK associations (e.g. ``Dataset_Image``)
        and feature associations (e.g.
        ``Execution_Image_Image_Classification``, marked by FKs to the
        ML schema's ``Feature_Name`` vocab and ``Execution`` table)
        count as transparent —
        see the module docstring's "Transparency model" section.

        **Direction matters**: with ``Image.Subject → Subject.RID``:

        - ``_outbound_reachable('Subject', {'Image','Subject'})`` returns
          ``{'Image'}`` (Image is downstream of Subject).
        - ``_outbound_reachable('Image', {'Image','Subject'})`` returns
          ``set()`` (Subject is UPSTREAM of Image, not downstream).

        Args:
            from_table: starting table (the "upstream" side of the
                one-to-many relationship).
            tables_in_set: the subgraph — only tables in this set count
                as "destinations" in the result. Association tables
                outside the set are still traversable (transparent).

        Returns:
            Set of names in ``tables_in_set`` downstream of
            ``from_table`` (excluding ``from_table`` itself).

        Example::

            # Given schema: Image.Subject → Subject, Dataset ← Dataset_Image → Image
            subgraph = {"Image", "Subject"}
            planner._outbound_reachable("Subject", subgraph)  # {"Image"}
            planner._outbound_reachable("Image", subgraph)    # set()

            # With Dataset_Image as a transparent hop:
            subgraph = {"Dataset", "Image"}
            planner._outbound_reachable("Dataset", subgraph)  # {"Image"}
        """
        seen_names: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = [from_table]
        while stack:
            t = stack.pop()
            if t in visited:
                continue
            visited.add(t)
            try:
                tbl = self.model.name_to_table(t)
            except Exception:
                continue

            # When the current node is itself a transparent intermediate
            # AND it's not the starting point, hop through both directions:
            # both the tables that point at it (referenced_by) AND the
            # tables it points to (foreign_keys). This is the "transparent
            # bridge" semantics — M:N link tables and feature-assoc tables
            # should be traversable in both directions so that
            # A → bridge → B discovers B from A.
            hopping_through_bridge = t != from_table and self._is_transparent_intermediate(tbl)

            valid_schemas = self.model.domain_schemas | {self.model.ml_schema}
            neighbors: list[Table] = list(self._downstream_fk_sources(t))
            if hopping_through_bridge:
                # Add the bridge's outbound FK targets (the "other side"
                # of the link) so we can see past the bridge. For feature-
                # assoc tables this also exposes the Execution target,
                # but that's fine — Execution is in _DEFAULT_SKIP_TABLES
                # for path enumeration and the reachability set only
                # cares about whether requested tables are reachable.
                for fk in tbl.foreign_keys:
                    nxt = fk.pk_table
                    if nxt.schema.name in valid_schemas:
                        neighbors.append(nxt)

            for neighbor in neighbors:
                target_name = neighbor.name
                if target_name == from_table:
                    continue
                if target_name in tables_in_set:
                    seen_names.add(target_name)
                    # Continue only if this neighbor is itself transparent
                    # (so we can chain bridges).
                    if self._is_transparent_intermediate(neighbor):
                        stack.append(target_name)
                elif self._is_transparent_intermediate(neighbor):
                    # Transparent hop: continue through the bridge.
                    stack.append(target_name)
                # else: non-requested, non-transparent — dead end
        return {t for t in seen_names if t in tables_in_set and t != from_table}

    def _outbound_reachable_strict(
        self,
        from_table: str,
        tables_in_set: set[str],
    ) -> set[str]:
        """Strict-direction variant of :meth:`_outbound_reachable`.

        Walks downstream from ``from_table`` following only
        :meth:`_downstream_fk_sources` (direct ``referenced_by``
        edges). Does **NOT** perform the bidirectional transparent-
        bridge hop that :meth:`_outbound_reachable` uses. This means
        ``A ↔ assoc ↔ B`` (where both A and B are upstream of the
        bridge) does **not** treat B as downstream of A — neither
        side fans out into the other through the bridge.

        Used by **Downstream-leaf rejection** (in
        :meth:`_determine_row_per`) and **Row grain** (sink-finding in
        :meth:`_find_sinks`).

        **Why a separate primitive.** :meth:`_outbound_reachable` is
        used by :meth:`Denormalizer._classify_anchors` to answer the
        connectivity question "is anchor X related to row_per via
        *any* FK chain?" — there, the bidirectional bridge hop is the
        right behavior (an Image-anchored set IS related to a
        Dataset-anchored set, even though the bridge sits in between
        and points at both). For Row grain and Downstream-leaf rejection
        we need the strict
        directional notion: "if I pick X as row_per, will the rows
        fan out into Y?" A bridge whose two endpoints are both
        upstream of it does not fan out — it produces at most a
        1:1:1 join, so neither endpoint should be considered
        downstream of the other.

        Args:
            from_table: starting table (the candidate row_per).
            tables_in_set: subgraph — only tables in this set count
                as destinations in the result.

        Returns:
            Set of names in ``tables_in_set`` strictly downstream of
            ``from_table`` (excluding ``from_table`` itself).

        Example::

            # Direct FK Image -> Subject:
            planner._outbound_reachable_strict("Subject", {"Subject", "Image"})
            # {"Image"}    — Image references Subject (Image is fan-out)
            planner._outbound_reachable_strict("Image", {"Subject", "Image"})
            # set()        — Subject is upstream of Image, not downstream

            # Bridge case (Image <- feature-assoc -> Image_Classification):
            planner._outbound_reachable_strict(
                "Image", {"Image", "Image_Classification"}
            )
            # set()        — neither side is strictly downstream of the
            #               other; the bridge does not fan out.
        """
        seen: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = [from_table]
        while stack:
            t = stack.pop()
            if t in visited:
                continue
            visited.add(t)
            try:
                tbl = self.model.name_to_table(t)
            except Exception:
                continue
            for neighbor in self._downstream_fk_sources(tbl):
                target_name = neighbor.name
                if target_name == from_table:
                    continue
                if target_name in tables_in_set:
                    seen.add(target_name)
                # Recurse into downstream tables regardless of set
                # membership — a chain like Subject ← Observation ←
                # Image must surface Image when only {Subject, Image}
                # are in the set. We never bridge through assoc tables
                # bidirectionally here; only follow further
                # `referenced_by` edges.
                if target_name not in visited:
                    stack.append(target_name)
        return {t for t in seen if t in tables_in_set and t != from_table}

    def _schema_to_paths(
        self,
        root: Table | None = None,
        exclude_tables: set[str] | None = None,
        skip_tables: frozenset[str] | None = None,
        max_depth: int | None = None,
        stop_at: str | None = None,
    ) -> list[list[Table]]:
        """Discover all FK paths through the schema graph (DFS, every-prefix).

        Delegates the FK walk to deriva-py's
        :class:`~deriva.bag.path_walker.SchemaPathWalker` and layers
        the DerivaML-specific rules on top via the walker's
        ``edge_filter`` hook:

        - Default skip tables (``Dataset_Dataset``, ``Execution``).
        - Nested-dataset loopback prevention
          (``Subject → Dataset_Subject → Dataset`` is blocked).

        The generic walker handles the rest: bidirectional FK walk,
        vocabulary-as-leaf, multi-FK edge deduplication, simple-path
        cycle guarding, schema allow-list, and depth bounding.

        Args:
            root: Starting table. Defaults to the ML schema's
                ``Dataset`` table.
            exclude_tables: Caller-specified table names to skip.
                Pruned along with all paths through them.
            skip_tables: Infrastructure table names to skip. Defaults
                to ``_DEFAULT_SKIP_TABLES``
                (``{"Dataset_Dataset", "Execution"}``). Override to
                customize which ML-schema tables are excluded.
            max_depth: Maximum path length (number of tables). ``None``
                means unbounded.
            stop_at: If supplied, return only paths whose last table
                has this name. The full-prefix DFS still runs; this
                is a post-filter on the result.

        Returns:
            Every walked prefix as a list of :class:`Table` objects.
            If ``[Dataset, A, B, C]`` is walked, ``[Dataset]``,
            ``[Dataset, A]``, ``[Dataset, A, B]`` are all in the
            result too.
        """
        from deriva.bag.path_walker import SchemaPathWalker
        from deriva.bag.traversal import FKTraversalPolicy

        # Resolve defaults.
        exclude_names = exclude_tables or set()
        skip_names = skip_tables if skip_tables is not None else _DEFAULT_SKIP_TABLES
        if root is None:
            root = self.model.model.schemas[self.model.ml_schema].tables["Dataset"]

        # Scope to the domain schemas plus the ML schema — the same
        # filter the legacy ``_fk_neighbors`` enforced.
        valid_schemas = self.model.domain_schemas | {self.model.ml_schema}

        # The DerivaML edge filter implements the two domain-specific
        # rules: skip-table exclusion (Dataset_Dataset / Execution by
        # default), caller-supplied exclude_tables, and nested-dataset
        # loopback prevention.
        dataset_table = self.model.model.schemas[self.model.ml_schema].tables["Dataset"]

        def edge_filter(src: Table, tgt: Table) -> bool:
            # Schema scope is handled by the walker (via policy.schemas),
            # but skip_tables / exclude_tables / loopback are domain
            # rules — keep them here.
            if tgt.name in skip_names:
                return False
            if tgt.name in exclude_names:
                return False
            # Nested-dataset loopback: block ``X → assoc → Dataset``
            # when X is not Dataset itself. Allows
            # ``Dataset → Dataset_X → ...`` (the intended outbound
            # direction) but prevents the inverse walk back to the
            # root through an element association.
            if src is not dataset_table and self._is_topological_association(tgt):
                for fk in tgt.foreign_keys:
                    if fk.pk_table is dataset_table:
                        return False
            return True

        policy = FKTraversalPolicy(schemas=valid_schemas)
        walker = SchemaPathWalker(
            model=self.model.model,
            policy=policy,
            edge_filter=edge_filter,
        )

        prefixes = walker.walk_all_prefixes(root=root, max_depth=max_depth, stop_at=stop_at)

        # Convert tuples to lists for compatibility with existing
        # callers (they iterate and slice freely).
        result: list[list[Table]] = [list(p) for p in prefixes]
        if stop_at is not None:
            return [p for p in result if p and p[-1].name == stop_at]
        return result

    def _table_relationship(
        self,
        table1: str | Table,
        table2: str | Table,
    ) -> list[tuple[Column, Column]]:
        """Return column pairs used to relate two tables.

        For simple FKs, returns a single-element list: [(fk_col, pk_col)].
        For composite FKs, returns multiple pairs: [(fk_col1, pk_col1), (fk_col2, pk_col2)].

        Each FK constraint counts as one relationship (even if composite),
        so ambiguity is detected when multiple separate FK constraints exist
        between the same two tables.
        """
        table1 = self.model.name_to_table(table1)
        table2 = self.model.name_to_table(table2)
        # Each FK constraint produces a list of (fk_col, pk_col) pairs
        relationships: list[list[tuple[Column, Column]]] = []
        for fk in table1.foreign_keys:
            if fk.pk_table == table2:
                pairs = list(zip(fk.foreign_key_columns, fk.referenced_columns))
                relationships.append(pairs)
        for fk in table1.referenced_by:
            if fk.table == table2:
                pairs = list(zip(fk.referenced_columns, fk.foreign_key_columns))
                relationships.append(pairs)

        if len(relationships) == 0:
            raise DerivaMLException(
                f"No FK relationship found between {table1.name} and {table2.name}. "
                f"These tables may not be directly connected. Check your include_tables list."
            )
        if len(relationships) > 1:
            path_descriptions = []
            for col_pairs in relationships:
                desc = ", ".join(
                    f"{fk_col.table.name}.{fk_col.name} → {pk_col.table.name}.{pk_col.name}"
                    for fk_col, pk_col in col_pairs
                )
                path_descriptions.append(f"  {desc}")
            raise DerivaMLException(
                f"Ambiguous linkage between {table1.name} and {table2.name}: "
                f"found {len(relationships)} FK relationships:\n" + "\n".join(path_descriptions)
            )
        return relationships[0]

    # ------------------------------------------------------------------
    # Planner rules (Row grain, Downstream-leaf rejection, Path ambiguity)
    #
    # These methods compose ``_fk_neighbors`` / ``_schema_to_paths`` /
    # ``is_topological_association`` — they do NOT introduce new FK traversal.
    # ------------------------------------------------------------------

    def _find_sinks(
        self,
        include_tables: list[str],
        via: list[str] | None = None,
    ) -> list[str]:
        """Find sinks in the FK subgraph on ``include_tables ∪ via`` (Row grain).

        A **sink** is a table in ``include_tables`` with no outbound FK
        (in the one-to-many / downstream sense) to any other table in
        the set. Intuition: the "deepest" table in the requested join —
        the one that receives FKs from others but doesn't have any
        others downstream. In star-schema denormalization, the sink is
        the natural ``row_per`` — one output row per sink row, with
        upstream columns hoisted.

        Composes :meth:`_outbound_reachable_strict` — strict downstream
        only (no bidirectional bridge hop). Two tables linked only by
        a transparent bridge (M:N association or feature-assoc) are
        therefore both sink candidates and the caller must
        disambiguate with explicit ``row_per=``. Does not traverse FKs
        itself.

        Args:
            include_tables: requested tables — only these are candidates
                for the sink role (``via`` tables don't contribute columns).
            via: optional additional tables that participate in the
                subgraph for routing but aren't sink candidates.

        Returns:
            Sorted list of sink table names. Normally exactly one.
            Multiple sinks → caller should raise
            :class:`DerivaMLDenormalizeMultiLeaf`. Zero sinks → cycle,
            caller should raise :class:`DerivaMLDenormalizeNoSink`.

        Example::

            # Chain Subject ← Observation ← Image → sink is Image
            planner._find_sinks(["Subject", "Observation", "Image"])
            # ["Image"]

            # Unrelated tables → multi-leaf (both are sinks)
            planner._find_sinks(["Dataset", "Subject"])
            # ["Dataset", "Subject"]
        """
        via = via or []
        all_tables = set(include_tables) | set(via)
        # A sink is a requested table whose strict-downstream set,
        # minus itself, is empty — i.e., nothing else in the subgraph
        # is strictly downstream of it via direct FK chain.
        return sorted(
            t for t in all_tables if t in include_tables and not (self._outbound_reachable_strict(t, all_tables) - {t})
        )

    def _determine_row_per(
        self,
        include_tables: list[str],
        via: list[str] | None,
        row_per: str | None,
    ) -> str:
        """Resolve the ``row_per`` table, implementing Row grain and
        Downstream-leaf rejection.

        Two paths:

        - **Explicit** (``row_per`` not None): validate the caller's
          choice. ``row_per`` must be in ``include_tables``, and no
          table in ``include_tables`` may be downstream of it
          (Downstream-leaf rejection — that would require aggregation,
          which the current engine doesn't do).
        - **Auto-infer** (``row_per is None``): apply Row grain via
          sink-finding. Expect exactly one sink.

        Args:
            include_tables: requested tables.
            via: optional path-only tables.
            row_per: caller's explicit leaf, or None to auto-infer.

        Returns:
            The resolved ``row_per`` table name — guaranteed to be in
            ``include_tables`` and free of downstream conflicts.

        Raises:
            ValueError: ``row_per`` is not in ``include_tables``.
            DerivaMLDenormalizeDownstreamLeaf: explicit ``row_per`` has
                downstream table(s) in ``include_tables`` (Downstream-leaf
                rejection).
            DerivaMLDenormalizeNoSink: no sink found (FK cycle in the
                subgraph — pathological).
            DerivaMLDenormalizeMultiLeaf: auto-inference finds more
                than one candidate sink (Row grain).

        Example::

            planner._determine_row_per(
                include_tables=["Subject", "Image"], via=[], row_per=None
            )
            # "Image" (auto-inferred — Image is the sink)

            # Downstream-leaf rejection: Subject with Image downstream is rejected.
            planner._determine_row_per(
                include_tables=["Subject", "Image"], via=[], row_per="Subject"
            )
            # raises DerivaMLDenormalizeDownstreamLeaf
        """
        from deriva_ml.core.exceptions import (
            DerivaMLDenormalizeDownstreamLeaf,
            DerivaMLDenormalizeMultiLeaf,
            DerivaMLDenormalizeNoSink,
        )

        via = via or []
        all_tables = set(include_tables) | set(via)

        if row_per is not None:
            if row_per not in include_tables:
                raise ValueError(f"row_per={row_per!r} must be in include_tables={include_tables}")
            # Downstream-leaf rejection uses strict downstream (no
            # bidirectional bridge hop) — picking either side of a
            # transparent bridge as
            # row_per is legitimate, since the bridge produces at
            # most a 1:1:1 join, not a fan-out.
            downstream = self._outbound_reachable_strict(row_per, all_tables)
            downstream_in_inc = [t for t in include_tables if t in downstream and t != row_per]
            if downstream_in_inc:
                raise DerivaMLDenormalizeDownstreamLeaf(
                    row_per=row_per,
                    downstream_tables=sorted(downstream_in_inc),
                )
            return row_per

        sinks = self._find_sinks(include_tables, via)
        if not sinks:
            raise DerivaMLDenormalizeNoSink(
                f"No sink found in include_tables={include_tables}. The FK subgraph may contain a cycle."
            )
        if len(sinks) > 1:
            raise DerivaMLDenormalizeMultiLeaf(
                candidates=sinks,
                include_tables=list(include_tables),
            )
        return sinks[0]

    def _enumerate_paths(
        self,
        from_table: str,
        to_table: str,
        tables_in_set: set[str],
        max_depth: int = 6,
    ) -> list[list[str]]:
        """Enumerate simple FK paths from ``from_table`` to ``to_table``.

        **Delegates the DFS** to :meth:`_schema_to_paths` (the
        authoritative FK-graph enumerator — handles cycle detection,
        vocabulary termination, schema filtering, and multi-FK
        deduplication). Uses its ``stop_at`` kwarg so inner recursion
        frames can prune eagerly rather than emitting all prefixes and
        filtering at the top. **Do NOT write a fresh DFS here.**

        The only additional work is a **transparency filter**: a path
        is kept only if every intermediate table (non-endpoint nodes)
        is either in ``tables_in_set`` (the user's requested /
        via-routed set) or is a pure association table (which acts as
        a transparent bridge).

        Args:
            from_table: path start.
            to_table: path end.
            tables_in_set: ``include_tables ∪ via``. Paths passing
                through tables NOT in this set are accepted only if
                every intermediate is a pure association table.
            max_depth: forwarded to :meth:`_schema_to_paths` as a
                safety cap against pathological schemas.

        Returns:
            List of paths, each a list of table-name strings starting
            with ``from_table`` and ending with ``to_table``. Empty if
            no transparent-valid path exists.

        Example::

            # Diamond schema: Image → Subject direct AND Image → Observation → Subject.
            # With Observation in the set, both paths are valid:
            planner._enumerate_paths("Image", "Subject", {"Image", "Subject", "Observation"})
            # [["Image", "Subject"], ["Image", "Observation", "Subject"]]

            # With only Image and Subject in the set, the multi-hop path
            # requires Observation as intermediate but it's not in the
            # set and not an association → only the direct path survives:
            planner._enumerate_paths("Image", "Subject", {"Image", "Subject"})
            # [["Image", "Subject"]]
        """
        # Delegate the DFS — stop_at tells _schema_to_paths to only
        # keep paths ending at to_table (inner frames can prune early).
        paths = self._schema_to_paths(
            root=self.model.name_to_table(from_table),
            max_depth=max_depth,
            stop_at=to_table,
        )
        result: list[list[str]] = []
        for path in paths:
            names = [t.name for t in path]
            # Transparency filter: every intermediate must be either
            # requested (in tables_in_set) or a transparent intermediate
            # (a pure association table OR a feature-association table —
            # see the module-level "Transparency model" section).
            if all(mid in tables_in_set or self._is_transparent_intermediate(mid) for mid in names[1:-1]):
                result.append(names)
        return result

    def _find_path_ambiguities(
        self,
        row_per: str,
        include_tables: list[str],
        via: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate path ambiguities between ``row_per`` and other requested tables (Path ambiguity).

        For each ``T`` in ``include_tables ∪ via`` (``T ≠ row_per``),
        enumerate all simple FK paths between ``row_per`` and ``T``
        using :meth:`_schema_to_paths` (full undirected DFS — we do
        NOT apply the transparency filter here, because we need to see
        the full picture to detect diamonds the user hasn't yet
        disambiguated).

        **User-signal disambiguation**: a path is considered "signaled"
        by the user if at least one of its intermediate tables is in
        ``include_tables ∪ via`` (pure association tables don't count —
        those are transparent). If exactly one path is signaled, the
        user has picked it and there's no ambiguity. If zero or >1 are
        signaled, we cannot silently choose, so an ambiguity is
        reported.

        This is distinct from :meth:`_enumerate_paths`, which applies
        the transparency filter to produce only "routable" paths given
        the current set. Here we want to see ALL candidates so we can
        warn about the diamond.

        Args:
            row_per: the leaf table (resolved earlier by
                :meth:`_determine_row_per`).
            include_tables: tables whose paths to ``row_per`` are checked.
            via: additional tables whose paths are checked (their columns
                aren't in the output, but they still participate in
                disambiguation).

        Returns:
            List of ambiguity dicts — empty when no ambiguities are
            detected. Each dict has:

            - ``from_table``: always ``row_per``.
            - ``to_table``: the ``T`` with multiple paths.
            - ``paths``: list of path lists (each path a list of table
              names, first element ``row_per``, last element ``T``).
            - ``suggested_intermediates``: non-endpoint tables that
              appear in at least one path but are not in
              ``include_tables`` and are not pure association tables
              — user could add any of these to ``include_tables`` or
              ``via`` to disambiguate.

        Example::

            # Diamond: Image→Subject direct AND Image→Observation→Subject.
            planner._find_path_ambiguities(
                row_per="Image", include_tables=["Image", "Subject"]
            )
            # [{"from_table": "Image", "to_table": "Subject",
            #   "paths": [["Image", "Subject"],
            #             ["Image", "Observation", "Subject"]],
            #   "suggested_intermediates": ["Observation"]}]

            # Once Observation is added to include_tables, it "signals"
            # the multi-hop path → no ambiguity:
            planner._find_path_ambiguities(
                row_per="Image", include_tables=["Image", "Observation", "Subject"]
            )
            # []
        """
        via = via or []
        all_tables = set(include_tables) | set(via)
        ambiguities: list[dict[str, Any]] = []

        for t in sorted(all_tables):
            if t == row_per:
                continue
            # Enumerate ALL simple paths (no transparency filter) — we need
            # the full picture to detect diamonds even when the user has not
            # requested the intermediate table.
            #
            # Note: we intentionally do NOT call ``_enumerate_paths`` here.
            # That helper applies a transparency filter (intermediates must
            # be requested or be association tables), which would mask the
            # very diamonds this rule must warn about. ``_enumerate_paths``
            # is for consumers who want only "routable" paths given the
            # current include_tables/via set.
            all_path_tables = self._schema_to_paths(
                root=self.model.name_to_table(row_per),
                max_depth=6,
                stop_at=t,
            )
            all_paths_named: list[list[str]] = [[tbl.name for tbl in p] for p in all_path_tables]
            unique = list({tuple(p): p for p in all_paths_named}.values())
            if len(unique) <= 1:
                continue

            # Monotonic-direction filter for diamond detection:
            # A genuine diamond has MULTIPLE paths that each constitute a
            # valid FK join chain — all-outbound (downstream) hops, with
            # association tables acting as transparent bridges. Paths that
            # change direction at an interior vertex are common-neighbor
            # shortcuts, not join alternatives. For example, with::
            #
            #     Image.Observation → Observation  (direct FK)
            #     Image.Subject → Subject           (direct FK)
            #     Observation.Subject → Subject     (direct FK)
            #
            # the undirected walk ``Image → Subject → Observation`` hops
            # Image.Subject downstream then Observation.Subject UPSTREAM
            # (Subject is a shared neighbor). This does not represent an
            # FK chain from Image to Observation — it represents a
            # co-occurrence via shared Subject, which is a materially
            # different query. We exclude such paths from ambiguity
            # detection so the direct FK Image→Observation isn't
            # spuriously flagged.
            #
            # Association tables remain transparent: the walker handles
            # them correctly via ``is_topological_association`` check inside
            # the direction test.
            def _edge_direction(a: str, b: str) -> str | None:
                """Return 'down' if a has a direct FK to b (outbound from
                a); 'up' if b has a direct FK to a (inbound to a); None
                if there's no direct FK between them."""
                try:
                    ta = self.model.name_to_table(a)
                    tb = self.model.name_to_table(b)
                except Exception:
                    return None
                for fk in ta.foreign_keys:
                    if fk.pk_table == tb:
                        return "down"
                for fk in tb.foreign_keys:
                    if fk.pk_table == ta:
                        return "up"
                return None

            def _is_downstream_chain(p: list[str]) -> bool:
                """Check that the path is all-downstream, treating
                transparent intermediates (pure 2-FK associations AND
                feature-assoc tables) as transparent bridges. A
                transparent bridge Image ← assoc → Subject counts as a
                single downstream step (the assoc's referenced_by
                connects the two sides). Transparent intermediates at
                interior positions don't count as direction changes."""
                i = 0
                while i < len(p) - 1:
                    a, b = p[i], p[i + 1]
                    # If b is an interior transparent intermediate, hop
                    # across it: count the A → bridge → C edge as a
                    # single transparent step and move two steps forward.
                    if i + 2 < len(p) and self._is_transparent_intermediate(b):
                        # A → bridge → C: the bridge is legitimate
                        # regardless of internal direction; advance past.
                        i += 2
                        continue
                    d = _edge_direction(a, b)
                    if d != "down":
                        return False
                    i += 1
                return True

            downstream = [p for p in unique if _is_downstream_chain(p)]
            if len(downstream) <= 1:
                # Only 0 or 1 downstream paths means no genuine diamond;
                # other "paths" were common-neighbor shortcuts. Fall back
                # to the direct/signaled path and don't flag ambiguity.
                continue
            unique = downstream

            # Disambiguation rule:
            # - A path is "signaled" if at least one of its non-endpoint
            #   intermediates is in ``include_tables ∪ via`` (user explicitly
            #   routed through it). Transparent intermediates (pure
            #   associations and feature-assoc tables) don't count —
            #   they're transparent and the user shouldn't need to name them.
            # - If exactly one path is signaled, the user has picked it → no
            #   ambiguity.
            # - Otherwise (0 or >1 signaled), we cannot silently choose →
            #   ambiguity.
            def _is_signaled(p: list[str]) -> bool:
                intermediates = p[1:-1]
                for mid in intermediates:
                    if mid in all_tables and not self._is_transparent_intermediate(mid):
                        return True
                return False

            signaled = [p for p in unique if _is_signaled(p)]
            if len(signaled) == 1:
                # Exactly one user-signaled path — use it.
                continue

            # Ambiguity: either no user signal, or conflicting signals.
            reportable = signaled if len(signaled) > 1 else unique
            all_intermediates: set[str] = set()
            for p in reportable:
                for node in p[1:-1]:
                    if node not in include_tables and not self._is_transparent_intermediate(node):
                        all_intermediates.add(node)
            ambiguities.append(
                {
                    "from_table": row_per,
                    "to_table": t,
                    "paths": reportable,
                    "suggested_intermediates": sorted(all_intermediates),
                }
            )
        return ambiguities

    # ------------------------------------------------------------------
    # Top-level entries
    # ------------------------------------------------------------------

    def _build_join_tree(
        self,
        element_name: str,
        include_tables: set[str],
        all_paths: list[list[Table]],
        via: set[str] | None = None,
    ) -> JoinNode:
        """Build a JoinTree rooted at *element_name* that reaches all *include_tables*.

        The algorithm:

        1. Collect all FK paths from `_schema_to_paths()` that start at the element
           table and end at a table in *include_tables*.
        2. For each target table, pick the SHORTEST sub-path from the element.
           If a longer path exists but ALL its intermediates are in *include_tables*,
           prefer it (user disambiguated).  If multiple equally-short paths exist
           and cannot be disambiguated, raise an ambiguity error.
        3. Merge the selected paths into a tree rooted at the element.
        4. Mark association tables (``is_association=True``) so their columns are
           excluded from output but they are still JOINed through.
        5. Set ``join_type="left"`` when the FK column is nullable.

        Args:
            element_name: The dataset element table (tree root), e.g. ``"Image"``.
            include_tables: Set of table names the user wants in the output.
            all_paths: All FK paths from ``_schema_to_paths()``.
            via: Optional set of table names the caller passed as
                ``via=`` — path-only routing hints. Intermediates in
                this set count as "covered" during disambiguation so the
                user can route through an intermediate without adding
                its columns to the output.

        Returns:
            A ``JoinNode`` tree rooted at the element table.

        Raises:
            DerivaMLDenormalizeAmbiguousPath: If multiple FK paths
                between the element table and a target table can't
                be unambiguously resolved by ``include_tables`` /
                ``via``. The exception carries ``paths`` (the
                competing FK path lists) and
                ``suggested_intermediates`` so callers can
                introspect rather than parse the message.
        """
        via = via or set()
        covering = include_tables | via
        element_table = self.model.name_to_table(element_name)

        # ── Step 1: collect sub-paths from element to each include_table ─────
        # Each "all_path" has the structure [Dataset, assoc, element, ..., endpoint].
        # We extract the sub-path starting from the element: [element, ..., endpoint].
        subpaths_by_target: dict[str, list[list[Table]]] = defaultdict(list)

        for path in all_paths:
            if len(path) < 3:
                continue
            if path[2].name != element_name:
                continue
            endpoint = path[-1].name
            if endpoint not in include_tables:
                continue
            # Sub-path from element onward
            sub = path[2:]  # [element, ..., endpoint]
            subpaths_by_target[endpoint].append(sub)

        # The element itself (self-path of length 1)
        if element_name in include_tables:
            subpaths_by_target.setdefault(element_name, []).append([element_table])

        # ── Step 2: for each target, pick the best path ──────────────────────
        selected_subpaths: dict[str, list[Table]] = {}

        for target, subpaths in subpaths_by_target.items():
            if target == element_name:
                # Self-path: no join needed
                selected_subpaths[target] = [element_table]
                continue

            # Deduplicate by table-name signature
            seen_sigs: set[tuple[str, ...]] = set()
            unique: list[list[Table]] = []
            for sp in subpaths:
                sig = tuple(t.name for t in sp)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    unique.append(sp)

            if len(unique) == 1:
                selected_subpaths[target] = unique[0]
                continue

            # Multiple paths — disambiguate.
            # Intermediates are tables between element (sp[0]) and endpoint (sp[-1]).
            path_intermediates = [tuple(t.name for t in sp[1:-1]) for sp in unique]

            # If all have identical intermediates, no ambiguity
            if len(set(path_intermediates)) <= 1:
                selected_subpaths[target] = unique[0]
                continue

            # A path is "selected" if all its non-transparent intermediates
            # are in include_tables. Transparent intermediates (M:N link
            # tables and feature-assoc tables) are infrastructure that
            # the user shouldn't need to name explicitly — they are
            # transparently included in the join chain.
            #
            # We detect transparent intermediates via
            # ``self._is_transparent_intermediate`` (the union of the
            # 2-FK topological-association predicate and the
            # feature-association predicate; both ignore ERMrest system FKs).

            def _intermediates_covered(sp: list[Table], ints: tuple[str, ...]) -> bool:
                sp_tables = {t.name: t for t in sp}
                for t in ints:
                    if t in covering:
                        # In include_tables OR in via= — explicitly routed.
                        continue
                    tbl = sp_tables.get(t)
                    if tbl is not None and self._is_transparent_intermediate(tbl):
                        continue  # transparent — doesn't need to be in include_tables
                    return False
                return True

            fully_covered = [
                (sp, ints) for sp, ints in zip(unique, path_intermediates) if _intermediates_covered(sp, ints)
            ]

            if len(fully_covered) == 1:
                sp, ints = fully_covered[0]
                if len(ints) > 0:
                    # User explicitly included intermediates
                    selected_subpaths[target] = sp
                    continue
                # Direct path (no intermediates) — check if there are indirect paths
                has_indirect = any(len(i) > 0 for i in path_intermediates)
                if not has_indirect:
                    selected_subpaths[target] = sp
                    continue
                # Direct FK alongside indirect — prefer direct (shortest)
                selected_subpaths[target] = sp
                continue

            if len(fully_covered) > 1:
                # Multiple fully-covered paths
                has_explicit = [(sp, ints) for sp, ints in fully_covered if len(ints) > 0]
                if len(has_explicit) == 1:
                    selected_subpaths[target] = has_explicit[0][0]
                    continue
                elif len(has_explicit) == 0:
                    # All direct paths — pick shortest
                    shortest = min(fully_covered, key=lambda x: len(x[0]))
                    selected_subpaths[target] = shortest[0]
                    continue
                else:
                    # Multiple explicit — prefer longest (most specific)
                    max_ints = max(len(ints) for _, ints in has_explicit)
                    longest = [sp for sp, ints in has_explicit if len(ints) == max_ints]
                    if len(longest) == 1:
                        selected_subpaths[target] = longest[0]
                        continue

            if len(fully_covered) == 0:
                # No path is fully covered.  Check if direct path exists.
                direct = [sp for sp, ints in zip(unique, path_intermediates) if len(ints) == 0]
                if len(direct) == 1:
                    selected_subpaths[target] = direct[0]
                    continue

            # Ambiguity — raise the typed exception so callers can
            # introspect ``paths`` and ``suggested_intermediates``
            # rather than parsing the message string. The sibling
            # raise in ``_prepare_wide_table`` (Phase 0 ambiguity
            # detection) already uses ``DerivaMLDenormalizeAmbiguousPath``;
            # we mirror it here so both ambiguity surfaces report
            # the same shape.
            from deriva_ml.core.exceptions import DerivaMLDenormalizeAmbiguousPath

            path_name_lists: list[list[str]] = []
            all_ints: set[str] = set()
            for sp, ints in zip(unique, path_intermediates):
                path_name_lists.append([t.name for t in sp])
                all_ints.update(ints)

            suggestion_tables = sorted(all_ints - include_tables)
            raise DerivaMLDenormalizeAmbiguousPath(
                from_table=element_name,
                to_table=target,
                paths=path_name_lists,
                suggested_intermediates=suggestion_tables,
            )

        # ── Step 3: merge selected paths into a tree ─────────────────────────
        # Build the tree by inserting each selected sub-path into the tree.
        root = JoinNode(
            table=element_table,
            table_name=element_name,
            join_type="inner",
            fk_columns=None,
            is_association=self.model.is_association(element_name),
            children=[],
        )

        # Map table_name -> JoinNode for quick lookup during tree building
        node_map: dict[str, JoinNode] = {element_name: root}

        for target, subpath in selected_subpaths.items():
            if target == element_name:
                continue
            # subpath = [element, ..intermediate.., target]
            # Walk the subpath, creating nodes as needed
            for i in range(1, len(subpath)):
                child_table = subpath[i]
                child_name = child_table.name
                parent_table = subpath[i - 1]
                parent_name = parent_table.name

                if child_name in node_map:
                    continue  # Already in tree

                # Get FK column pairs
                col_pairs = self._table_relationship(parent_table, child_table)

                # Determine join type: LEFT for nullable FK columns
                join_type = "inner"
                for fk_col, pk_col in col_pairs:
                    if fk_col.nullok:
                        join_type = "left"
                        break

                node = JoinNode(
                    table=child_table,
                    table_name=child_name,
                    join_type=join_type,
                    fk_columns=col_pairs,
                    is_association=self.model.is_association(child_name),
                    children=[],
                )
                node_map[child_name] = node
                # Attach to parent
                if parent_name in node_map:
                    node_map[parent_name].children.append(node)
                else:
                    # Parent not yet in tree — this shouldn't happen since we
                    # process paths from element outward, but handle gracefully
                    logger.warning(f"Parent {parent_name} not in tree when adding {child_name}")

        return root

    def _prepare_wide_table(
        self,
        dataset,
        dataset_rid: RID,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        system_columns: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[tuple], bool]:
        """Generate a join plan for denormalizing a dataset into a wide table.

        Uses a **JoinTree** approach that preserves path-specific structure:

        1. **Planner guards** -- validate ``row_per`` (Row grain /
           Downstream-leaf rejection) and check for path ambiguity
           (Path ambiguity) before any join work.
        2. **Path discovery** -- ``_schema_to_paths()`` discovers all FK paths
           from Dataset through the schema.
        3. **Path filtering & deduplication** -- keep only paths relevant to
           *include_tables*, dedup duplicate association table routes.
        4. **JoinTree construction** -- for each element type, build a tree
           rooted at the element.  Each node is a table to JOIN; association
           tables are in the tree (for JOIN) but excluded from output columns.
           Nullable FK columns produce LEFT JOINs.
        5. **Flatten to consumer format** -- convert the tree to the
           ``(path, join_conditions, join_types)`` tuple that
           ``_denormalize_impl()`` in ``local_db/denormalize.py``
           consumes. The :class:`JoinNode` tree itself is internal
           to the planner; the consumer takes the flat tuple.

        Args:
            dataset: A DatasetLike object (DatasetBag or Dataset).
            dataset_rid: RID of the dataset.
            include_tables: List of table names to include in the output.
            row_per: Explicit leaf table (one row per this table). If None,
                the sink is auto-inferred from include_tables.
            via: Additional tables used only for path routing (their columns
                are NOT included in the output).

        Returns:
            ``(element_tables, denormalized_columns, multi_schema)`` where:

            - **element_tables** -- ``dict[str, (path, join_conditions, join_types)]``
              keyed by ``{element}#{route_index}`` -- one entry per distinct
              ``Dataset -> ... -> element`` route (membership AND FK-reachable),
              so a subject-partitioned dataset (empty membership association)
              still resolves rows via the FK route. The consumer UNIONs the
              entries (deduped on the ``row_per`` leaf), so the key is never
              read semantically; it only needs to be unique.
              *path* is a list of table name strings in JOIN order (the
              ``Dataset -> ... -> element`` prefix followed by a pre-order walk
              of the shared downstream JoinTree).
              *join_conditions* maps ``table_name -> set[(fk_col, pk_col)]``.
              *join_types* maps ``table_name -> "inner" | "left"``.
            - **denormalized_columns** -- list of
              ``(schema_name, table_name, column_name, type_name)`` for the output.
            - **multi_schema** -- True if output spans multiple domain schemas.

        Raises:
            DerivaMLDenormalizeMultiLeaf / DerivaMLDenormalizeNoSink /
            DerivaMLDenormalizeDownstreamLeaf: from :meth:`_determine_row_per`.
            DerivaMLDenormalizeAmbiguousPath: if more than one FK path exists
                between row_per and a requested table.
        """
        include_tables_set = set(include_tables)
        for t in include_tables_set:
            _ = self.model.name_to_table(t)  # validate existence
        via_list = list(via or [])
        for t in via_list:
            _ = self.model.name_to_table(t)  # validate existence

        # ── Phase 0: planner guards (Row grain, Downstream-leaf rejection, Path ambiguity) ──
        # Empty include_tables is a legal degenerate case (caller passes no
        # requested tables and expects an empty result). Skip guards then.
        # `resolved_row_per` is also the row grain: only routes that
        # END at this table may drive rows (Phase 1c). None when include_tables
        # is empty (no routes emitted at all).
        resolved_row_per: str | None = None
        if include_tables:
            resolved_row_per = self._determine_row_per(
                include_tables=list(include_tables),
                via=via_list,
                row_per=row_per,
            )
            ambiguities = self._find_path_ambiguities(
                row_per=resolved_row_per,
                include_tables=list(include_tables),
                via=via_list,
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

        # ── Phase 1: path discovery ──────────────────────────────────────────
        all_paths = self._schema_to_paths()

        # Filter paths: must end at a table in include_tables AND
        # have at least one table in include_tables along the path.
        table_paths = [
            path
            for path in all_paths
            if path[-1].name in include_tables_set and include_tables_set.intersection({p.name for p in path})
        ]

        # ── Phase 1b: deduplicate ONLY truly identical paths ─────────────────
        # Previously this collapsed to one route per (element, endpoint),
        # preferring the Dataset_{Element} membership association and DISCARDING
        # the FK-reachable chain (e.g. Dataset -> Subject_Dataset -> Subject ->
        # Image). On subject-partitioned datasets the membership association is
        # empty, so that discard produced a silently-empty result. Retain every
        # distinct route; the consumer UNIONs them (SQL `UNION` dedups identical
        # output rows). See docs/superpowers/specs/2026-06-16-denormalize-fk-reachable-paths-design.md.
        deduplicated_paths: list[list[Table]] = []
        seen_path_sigs: set[tuple[str, ...]] = set()
        for path in table_paths:
            sig = tuple(t.name for t in path)
            if sig not in seen_path_sigs:
                seen_path_sigs.add(sig)
                deduplicated_paths.append(path)

        table_paths = deduplicated_paths

        # ── Phase 1c: group routes by the include-table they SERVE (endpoint) ─
        # The element a route serves is its ENDPOINT (p[-1]), not its third
        # table (p[2]). Both the membership route Dataset -> Dataset_Image ->
        # Image (p[2] == Image) and the FK-reachable route Dataset ->
        # Dataset_Subject -> Subject -> Image (p[2] == Subject) serve element
        # "Image" — they share the same endpoint. Keying by p[2] dropped the
        # FK route because its third table (Subject) is not in include_tables.
        # Key by the endpoint so every route to an include-table is retained.
        #
        # Only routes that END at the ``row_per`` table may DRIVE rows (Row grain:
        # one row per ``row_per`` instance). A route ending at a *non*-``row_per``
        # include-table (e.g. Dataset -> ... -> Subject when row_per is Image)
        # is the wrong grain — those upstream columns are HOISTED into the
        # row_per routes' shared subtree (Phase 2 ``_build_join_tree``), not
        # emitted as their own row source. Emitting them produced spurious rows:
        # the consumer SELECTs the full column set for every route, so a
        # Subject-ending route that never joins Image cross-joins Image's
        # columns into a cartesian product (#322). Restrict to the row_per
        # element. ``resolved_row_per`` is always set here because this code is
        # only reached when ``include_tables`` is non-empty (empty returns the
        # degenerate empty plan in Phase 4).
        paths_by_element: dict[str, list[list[Table]]] = defaultdict(list)
        for p in table_paths:
            if len(p) >= 3 and p[-1].name == resolved_row_per:
                paths_by_element[p[-1].name].append(p)

        # ── Phase 2: build JoinTree per element ──────────────────────────────
        # System columns are dropped from the wide table by default. Callers
        # that need provenance (e.g. RCB = the row's creating user) can opt
        # specific ones back in via ``system_columns``.
        skip_columns = {"RCT", "RMT", "RCB", "RMB"} - set(system_columns or [])
        element_tables: dict[str, tuple[list[str], dict[str, set], dict[str, str]]] = {}

        for element_name, paths in paths_by_element.items():
            # The downstream subtree (element -> other include_tables) is the
            # SAME for every route that reaches this element — only the
            # Dataset -> ... -> element PREFIX differs. Build it once.
            tree = self._build_join_tree(element_name, include_tables_set, table_paths, via=set(via_list))
            tree_nodes = tree.walk()

            # ── Phase 3: emit one consumer entry per DISTINCT route ────────────
            # Each route gets its own Dataset -> ... -> element prefix joined to
            # the shared subtree. The consumer UNIONs all entries (deduped on
            # the row_per leaf), so a subject-partitioned dataset — whose
            # membership association is empty — still resolves rows through the
            # FK-reachable route. The dict KEY (element#i) is never read
            # semantically by the consumer; it only needs to be unique.
            for route_index, route in enumerate(paths):
                # Locate the element in this route; the prefix is everything up
                # to and including the element (Dataset -> ... -> element).
                # Phase 1c grouped routes by endpoint, so element_name is always
                # in route (route[-1]); next() can't exhaust.
                elem_pos = next(i for i, t in enumerate(route) if t.name == element_name)
                prefix = route[: elem_pos + 1]

                path_names: list[str] = [t.name for t in prefix]
                # Tables positioned by the prefix (Dataset -> ... -> element).
                # Each is joined at its prefix position with the prefix edge's
                # ON clause, which references the PRECEDING prefix table (already
                # joined). The subtree edges below must NOT overwrite these — a
                # subtree edge references the element (joined at/after the prefix
                # end), so applying it to a table that sits earlier in the prefix
                # would reference a not-yet-joined table (#320). See the
                # join-order invariant in test_planner_rules.TestJoinOrderValidity.
                prefix_names = set(path_names)
                join_conditions: dict[str, set[tuple]] = {}
                join_types: dict[str, str] = {}

                # Walk the prefix's consecutive (from, to) pairs. Each edge's
                # FK orientation is resolved by _table_relationship and may flip
                # along the chain (membership hop vs FK hop); we HONOR the
                # returned (fk_col, pk_col) ordering exactly. Keep inner joins
                # to match the prior 2-hop prefix behavior.
                for j in range(elem_pos):
                    from_table = route[j]
                    to_table = route[j + 1]
                    try:
                        col_pairs = self._table_relationship(from_table, to_table)
                        join_conditions[to_table.name] = set(col_pairs)
                        join_types[to_table.name] = "inner"
                    except DerivaMLException:
                        pass

                # Append the shared downstream subtree (element first, then its
                # children in pre-order). Skip nodes already in the prefix.
                for node in tree_nodes:
                    if node.table_name not in path_names:
                        path_names.append(node.table_name)

                # Add join conditions from the subtree edges, but ONLY for tables
                # the subtree genuinely appends. A table already positioned by the
                # prefix keeps its prefix-edge condition (valid for its position);
                # the subtree edge references the element (joined at/after the
                # prefix end) and would be invalid at an earlier prefix position
                # (#320). The root (element) has no parent edge in walk_edges().
                for _parent_node, child_node in tree.walk_edges():
                    if child_node.table_name in prefix_names:
                        continue
                    if child_node.fk_columns:
                        join_conditions[child_node.table_name] = set(child_node.fk_columns)
                        join_types[child_node.table_name] = child_node.join_type

                element_tables[f"{element_name}#{route_index}"] = (path_names, join_conditions, join_types)

        # ── Phase 4: build denormalized column list ──────────────────────────
        denormalized_columns = []
        for table_name in include_tables_set:
            if self.model.is_association(table_name):
                continue
            table = self.model.name_to_table(table_name)
            for c in table.columns:
                if c.name not in skip_columns:
                    denormalized_columns.append((table.schema.name, table_name, c.name, c.type.typename))

        output_schemas = {s for s, _, _, _ in denormalized_columns if self.model.is_domain_schema(s)}
        multi_schema = len(output_schemas) > 1

        return element_tables, denormalized_columns, multi_schema


__all__ = ["DenormalizePlanner", "JoinNode", "denormalize_column_name"]

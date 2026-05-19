"""
Model management for Deriva ML catalogs.

This module provides the DerivaModel class which augments the standard Deriva model class with
ML-specific functionality. It handles schema management, feature definitions, and asset tracking.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib

# Standard library imports
from typing import Any, Iterable, TypeAlias

_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
Column = _ermrest_model.Column
FindAssociationResult = _ermrest_model.FindAssociationResult
Key = _ermrest_model.Key
Model = _ermrest_model.Model
Schema = _ermrest_model.Schema
Table = _ermrest_model.Table

# Third-party imports
from pydantic import validate_call

from deriva_ml.core.catalog_stub import CatalogStub
from deriva_ml.core.definitions import (
    ML_SCHEMA,
    RID,
    SYSTEM_SCHEMAS,
    DerivaAssetColumns,
    TableDefinition,
    _get_domain_schemas,
    _is_system_schema,
)
from deriva_ml.core.exceptions import (
    DerivaMLException,
    DerivaMLFeatureNotFound,
    DerivaMLReadOnlyError,
    DerivaMLTableTypeError,
)
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.validation import VALIDATION_CONFIG

# Local imports
from deriva_ml.feature import Feature

# The denormalization planner (~1100 LoC) lives in its own module
# now (Phase 3, audit §5.2). ``JoinNode`` and ``denormalize_column_name``
# are re-exported here so existing imports continue to work without
# requiring callers to update their import path in lockstep with this
# extraction. The noqa markers below tell ruff these are re-exports,
# not unused imports.
from deriva_ml.model.denormalize_planner import (
    DenormalizePlanner,
    JoinNode,  # noqa: F401  re-export for back-compat
    denormalize_column_name,  # noqa: F401  re-export for back-compat
)

logger = get_logger(__name__)

# Define common types:
TableInput: TypeAlias = str | Table


class DerivaModel:
    """Augmented interface to deriva model class.

    This class provides a number of DerivaML specific methods that augment the interface in the deriva model class.

    Attributes:
        model: ERMRest model for the catalog.
        catalog: ERMRest catalog for the model.
        hostname: Hostname of the ERMRest server.
        ml_schema: The ML schema name for the catalog.
        domain_schemas: Frozenset of all domain schema names in the catalog.
        default_schema: The default schema for table creation operations.

    """

    def __init__(
        self,
        model: Model,
        ml_schema: str = ML_SCHEMA,
        domain_schemas: str | set[str] | None = None,
        default_schema: str | None = None,
    ):
        """Create and initialize a DerivaModel instance.

        This method will connect to a catalog and initialize schema configuration.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
            model: The ERMRest model for the catalog.
            ml_schema: The ML schema name.
            domain_schemas: Optional explicit set of domain schema names. If None,
                auto-detects all non-system schemas.
            default_schema: The default schema for table creation operations. If None
                and there is exactly one domain schema, that schema is used as default.
                If there are multiple domain schemas, default_schema must be specified.
        """
        self.model = model
        self.catalog: ErmrestCatalog = self.model.catalog
        self.hostname = self.catalog.deriva_server.server if isinstance(self.catalog, ErmrestCatalog) else "localhost"

        self.ml_schema = ml_schema
        self._system_schemas = frozenset(SYSTEM_SCHEMAS | {ml_schema})

        # Determine domain schemas
        if domain_schemas is not None:
            if isinstance(domain_schemas, str):
                domain_schemas = {domain_schemas}
            self.domain_schemas = frozenset(domain_schemas)
        else:
            # Auto-detect all domain schemas
            self.domain_schemas = _get_domain_schemas(self.model.schemas.keys(), ml_schema)

        # Determine default schema for table creation
        if default_schema is not None:
            if default_schema not in self.domain_schemas:
                raise DerivaMLException(
                    f"default_schema '{default_schema}' is not in domain_schemas: {self.domain_schemas}"
                )
            self.default_schema = default_schema
        elif len(self.domain_schemas) == 1:
            # Single domain schema - use it as default
            self.default_schema = next(iter(self.domain_schemas))
        elif len(self.domain_schemas) == 0:
            # No domain schemas - default_schema will be None
            self.default_schema = None
        else:
            # Multiple domain schemas, no explicit default
            self.default_schema = None

    @classmethod
    def from_cached(
        cls,
        schema_dict: dict,
        *,
        catalog,
        ml_schema: str = ML_SCHEMA,
        domain_schemas: "str | set[str] | None" = None,
        default_schema: "str | None" = None,
    ) -> "DerivaModel":
        """Construct a DerivaModel from a cached ermrest /schema dict.

        No network is touched. The ``catalog`` argument is passed to
        deriva-py's ``Model(catalog, model_doc)`` constructor as the
        first positional argument; in offline mode it will be a
        :class:`~deriva_ml.core.catalog_stub.CatalogStub`, in online
        mode it is a real ``ErmrestCatalog``. ``DerivaModel.__init__``
        then reads the catalog back off ``model.catalog`` as usual.

        This replicates what ``Model.fromcatalog(catalog)`` does
        online — the online call fetches the schema dict via
        ``catalog.getCatalogSchema()`` (cached and ETag-revalidated
        by deriva-py) and passes the result to ``Model(catalog, dict)``.
        Here we pass in the already-cached dict from
        :class:`~deriva_ml.core.schema_cache.SchemaCache`.

        Args:
            schema_dict: The JSON payload from a previous
                ``catalog.getCatalogSchema()`` call (or any equivalent
                ``/schema`` GET), as persisted by ``SchemaCache``.
            catalog: The catalog object to associate with the model.
                Pass a real ``ErmrestCatalog`` online, or a
                ``CatalogStub`` offline.
            ml_schema: ML schema name (default ``"deriva-ml"``).
            domain_schemas: Optional explicit set of domain schema
                names. If None, auto-detects all non-system schemas
                from the cached dict.
            default_schema: Optional default schema name.

        Returns:
            A ``DerivaModel`` wrapping a deriva-py ``Model``
            reconstructed from the dict.
        """
        # Model.__init__(catalog, model_doc) stores catalog as
        # self._catalog and exposes it via the .catalog property;
        # DerivaModel.__init__ then reads self.model.catalog.
        model = Model(catalog, schema_dict)
        return cls(
            model,
            ml_schema=ml_schema,
            domain_schemas=domain_schemas,
            default_schema=default_schema,
        )

    def is_system_schema(self, schema_name: str) -> bool:
        """Check if a schema is a system or ML schema.

        Args:
            schema_name: Name of the schema to check.

        Returns:
            True if the schema is a system or ML schema.
        """
        return _is_system_schema(schema_name, self.ml_schema)

    def is_domain_schema(self, schema_name: str) -> bool:
        """Check if a schema is a domain schema.

        Args:
            schema_name: Name of the schema to check.

        Returns:
            True if the schema is a domain schema.
        """
        return schema_name in self.domain_schemas

    def _require_default_schema(self) -> str:
        """Get default schema, raising an error if not set.

        Returns:
            The default schema name.

        Raises:
            DerivaMLException: If default_schema is not set.
        """
        if self.default_schema is None:
            raise DerivaMLException(
                f"No default_schema set. With multiple domain schemas {self.domain_schemas}, "
                "you must either specify a default_schema when creating DerivaML or "
                "pass an explicit schema parameter to this method."
            )
        return self.default_schema

    def refresh_model(self) -> None:
        self.model = self.catalog.getCatalogModel()

    @property
    def chaise_config(self) -> dict[str, Any]:
        """Return the chaise configuration."""
        return self.model.chaise_config

    def get_schema_description(self, include_system_columns: bool = False) -> dict[str, Any]:
        """Return a JSON description of the catalog schema structure.

        Provides a structured representation of the domain and ML schemas including
        tables, columns, foreign keys, and relationships. Useful for understanding
        the data model structure programmatically.

        Args:
            include_system_columns: If True, include RID, RCT, RMT, RCB, RMB columns.
                Default False to reduce output size.

        Returns:
            Dictionary with schema structure:
            {
                "domain_schemas": ["schema_name1", "schema_name2"],
                "default_schema": "schema_name1",
                "ml_schema": "deriva-ml",
                "schemas": {
                    "schema_name": {
                        "tables": {
                            "TableName": {
                                "comment": "description",
                                "is_vocabulary": bool,
                                "is_asset": bool,
                                "is_association": bool,
                                "columns": [...],
                                "foreign_keys": [...],
                                "features": [...]
                            }
                        }
                    }
                }
            }
        """
        system_columns = {"RID", "RCT", "RMT", "RCB", "RMB"}
        result = {
            "domain_schemas": sorted(self.domain_schemas),
            "default_schema": self.default_schema,
            "ml_schema": self.ml_schema,
            "schemas": {},
        }

        # Include all domain schemas and the ML schema
        for schema_name in [*self.domain_schemas, self.ml_schema]:
            schema = self.model.schemas.get(schema_name)
            if not schema:
                continue

            schema_info = {"tables": {}}

            for table_name, table in schema.tables.items():
                # Get columns
                columns = []
                for col in table.columns:
                    if not include_system_columns and col.name in system_columns:
                        continue
                    columns.append(
                        {
                            "name": col.name,
                            "type": str(col.type.typename),
                            "nullok": col.nullok,
                            "comment": col.comment or "",
                        }
                    )

                # Get foreign keys
                foreign_keys = []
                for fk in table.foreign_keys:
                    fk_cols = [c.name for c in fk.foreign_key_columns]
                    ref_cols = [c.name for c in fk.referenced_columns]
                    foreign_keys.append(
                        {
                            "columns": fk_cols,
                            "referenced_table": f"{fk.pk_table.schema.name}.{fk.pk_table.name}",
                            "referenced_columns": ref_cols,
                        }
                    )

                # Get features if this is a domain table
                features = []
                if self.is_domain_schema(schema_name):
                    try:
                        for f in self.find_features(table):
                            features.append(
                                {
                                    "name": f.feature_name,
                                    "feature_table": f.feature_table.name,
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Could not enumerate features for table {table.name}: {e}")

                table_info = {
                    "comment": table.comment or "",
                    "is_vocabulary": self.is_vocabulary(table),
                    "is_asset": self.is_asset(table),
                    "is_association": bool(self.is_association(table)),
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                }
                if features:
                    table_info["features"] = features

                schema_info["tables"][table_name] = table_info

            result["schemas"][schema_name] = schema_info

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute access to the underlying deriva-py Model.

        Called only when ``name`` is not already an attribute of the
        ``DerivaModel`` instance (per Python's attribute resolution order),
        so explicit properties on this class — ``chaise_config``,
        ``apply``, ``catalog``, ``schemas`` (inherited via :class:`DatabaseModel`
        from :class:`deriva.bag.database.BagDatabase`) — take precedence.

        Kept as a fallback because ``self.model.<attr>`` is reached at 50+
        call sites for ``schemas``, ``annotations`` and a long tail of
        deriva-py Model attributes. Replacing each with explicit
        accessors would collide with mixins (e.g. ``BagDatabase.schemas``
        is an instance-attribute set in its ``__init__``, which a
        ``@property`` would shadow and block assignment to).
        """
        return getattr(self.model, name)

    def name_to_table(self, table: TableInput) -> Table:
        """Return the table object corresponding to the given table name.

        Searches domain schemas first (in sorted order), then ML schema, then WWW.
        If the table name appears in more than one schema, returns the first match.

        Args:
          table: A ERMRest table object or a string that is the name of the table.

        Returns:
          Table object.

        Raises:
          DerivaMLException: If the table doesn't exist in any searchable schema.
        """
        if isinstance(table, Table):
            return table

        # Search domain schemas (sorted for deterministic order), then ML schema, then WWW
        search_order = [*sorted(self.domain_schemas), self.ml_schema, "WWW"]
        for sname in search_order:
            if sname not in self.model.schemas:
                continue
            s = self.model.schemas[sname]
            if table in s.tables:
                return s.tables[table]
        raise DerivaMLException(f"The table {table} doesn't exist.")

    def is_vocabulary(self, table: TableInput) -> bool:
        """Check if a given table is a controlled vocabulary table.

        Delegates to ``Table.is_vocabulary()`` in deriva-py, which enforces both
        the required column names AND their types (ermrest_curie, ermrest_uri,
        text, markdown). The type check is stricter than a column-name-only
        check — a table with an ``ID`` column of the wrong type correctly
        returns False here where the legacy name-only implementation would
        have returned True.

        Mirrors :meth:`is_asset`, which already delegates to ``Table.is_asset()``.

        Args:
            table: An ERMrest Table object or the name of the table.

        Returns:
            True if the table has the structure of a controlled vocabulary,
            False otherwise.

        Raises:
            DerivaMLException: If the table doesn't exist in any searchable
                schema (raised by :meth:`name_to_table`).
        """
        table = self.name_to_table(table)
        return table.is_vocabulary()

    def vocab_columns(self, table: TableInput) -> dict[str, str]:
        """Return mapping from canonical vocab column name to actual column name.

        Canonical names are TitleCase (Name, ID, URI, Description, Synonyms).
        Actual names reflect the table's schema — could be lowercase for
        FaceBase-style catalogs or TitleCase for DerivaML-native tables.

        Args:
            table: A table object or the name of the table.

        Returns:
            Dict mapping canonical name to actual column name in the table.
            E.g. ``{"Name": "name", "ID": "id", ...}`` for FaceBase tables
            or ``{"Name": "Name", "ID": "ID", ...}`` for DerivaML tables.

        Raises:
            DerivaMLException: If the table doesn't exist (raised by
                :meth:`name_to_table`).
        """
        table = self.name_to_table(table)
        col_map = {c.name.upper(): c.name for c in table.columns}
        return {canon: col_map[canon.upper()] for canon in ("Name", "ID", "URI", "Description", "Synonyms")}

    def is_association(
        self,
        table: TableInput,
        unqualified: bool = True,
        pure: bool = True,
        min_arity: int = 2,
        max_arity: int = 2,
    ) -> bool | set[str] | int:
        """Check whether ``table`` is an association (linking) table.

        Delegates to :meth:`deriva.core.ermrest_model.Table.is_association`.
        An association table mediates a many-to-many relationship between
        two (or more) tables via outbound FKs to each end.

        Args:
            table: Table name or :class:`Table` to inspect.
            unqualified: Per deriva-py — if True, the returned column set
                uses bare column names (no schema/table qualification).
                Only consulted when the return mode is the column-name set.
            pure: If True, require a *pure* association — no extra payload
                columns beyond the FK columns and system metadata (RID,
                RCT, RMT, RCB, RMB). Excludes feature tables, which carry
                their own non-FK columns.
            min_arity: Minimum number of outbound FKs that count as
                "associating." Defaults to 2 (a binary association).
            max_arity: Maximum number of outbound FKs. Defaults to 2.

        Returns:
            ``bool`` when the question is "is this *any* association at the
            requested arity," or ``set[str]`` / ``int`` when deriva-py's
            ``is_association`` returns the structural detail set instead.
            See :meth:`Table.is_association` for the full contract.

        Raises:
            DerivaMLException: If ``table`` doesn't exist in any searchable
                schema (raised by :meth:`name_to_table`).
        """
        table = self.name_to_table(table)
        return table.is_association(unqualified=unqualified, pure=pure, min_arity=min_arity, max_arity=max_arity)

    def find_association(self, table1: TableInput, table2: TableInput) -> tuple[Table, Column, Column]:
        """Return the unique association table linking ``table1`` and ``table2``.

        Searches all associations on ``table1`` for one whose other-side
        FK lands on ``table2``. The result lets callers JOIN through the
        link without re-deriving the column names by hand.

        Args:
            table1: Either endpoint of the association. Table name or
                :class:`Table`.
            table2: The other endpoint. Table name or :class:`Table`.

        Returns:
            ``(assoc_table, table1_link_column, table2_link_column)``
            — the association table itself plus the two FK columns on it
            (one referencing ``table1``, one referencing ``table2``).

        Raises:
            DerivaMLException: If no association connects the two tables,
                or if multiple associations connect them (in which case
                the caller should disambiguate by name).
        """
        table1 = self.name_to_table(table1)
        table2 = self.name_to_table(table2)

        tables = [
            (a.table, a.self_fkey.columns[0].name, other_key.columns[0].name)
            for a in table1.find_associations(pure=False)
            if len(a.other_fkeys) == 1 and (other_key := a.other_fkeys.pop()).pk_table == table2
        ]

        if len(tables) == 1:
            return tables[0]
        elif len(tables) == 0:
            raise DerivaMLException(f"No association tables found between {table1.name} and {table2.name}.")
        else:
            raise DerivaMLException(
                f"There are {len(tables)} association tables between {table1.name} and {table2.name}."
            )

    def is_asset(self, table: TableInput) -> bool:
        """Check whether ``table`` is a proper asset table.

        Delegates to :meth:`Table.is_asset` from deriva-py, which verifies:

        - Required columns exist (``URL``, ``Filename``, ``Length``, ``MD5``).
        - ``URL``, ``Length``, ``MD5`` are NOT NULL.
        - ``URL`` carries the ``asset`` annotation.

        Args:
            table: Table name or :class:`Table` to inspect.

        Returns:
            True if all asset-table requirements are satisfied.

        Raises:
            DerivaMLException: If ``table`` doesn't exist in any searchable
                schema (raised by :meth:`name_to_table`).
        """
        table = self.name_to_table(table)
        return table.is_asset()

    def find_assets(self) -> list[Table]:
        """Return the list of asset tables in the current model."""
        return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_asset(t)]

    def find_vocabularies(self) -> list[Table]:
        """Return a list of all controlled vocabulary tables in domain and ML schemas."""
        tables = []
        for schema_name in [*self.domain_schemas, self.ml_schema]:
            schema = self.model.schemas.get(schema_name)
            if schema:
                tables.extend(t for t in schema.tables.values() if self.is_vocabulary(t))
        return tables

    @validate_call(config=VALIDATION_CONFIG)
    def find_features(self, table: TableInput | None = None) -> Iterable[Feature]:
        """List features in the catalog.

        If a table is specified, returns only features for that table.
        If no table is specified, returns all features across all tables in the catalog.

        Args:
            table: Optional table to find features for. If None, returns all features
                in the catalog.

        Returns:
            An iterable of Feature instances describing the features.
        """

        def is_feature(a: FindAssociationResult) -> bool:
            """Check if association represents a feature.

            Args:
                a: Association result to check
            Returns:
                bool: True if association represents a feature
            """
            return {
                "Feature_Name",
                "Execution",
                a.self_fkey.foreign_key_columns[0].name,
            }.issubset({c.name for c in a.table.columns})

        def find_table_features(t: Table) -> list[Feature]:
            """Find all features for a single table."""
            return [
                Feature(a, self) for a in t.find_associations(min_arity=3, max_arity=3, pure=False) if is_feature(a)
            ]

        if table is not None:
            # Find features for a specific table
            return find_table_features(self.name_to_table(table))

        # No table arg: discover features across the whole catalog.
        #
        # ``find_associations`` walks ``Table.referenced_by`` from each
        # candidate table, so the same association table is visited
        # once per FK target. For a single ``Image.Image_Classification``
        # feature backed by ``Execution_Image_Image_Classification``
        # (an association with FKs to Image, Execution, and the
        # Image_Class vocab) the naive cross-schema scan yields three
        # Feature objects -- one with ``target_table=Image`` (the
        # actual target), one with ``target_table=Execution``, and one
        # with ``target_table=Image_Class``. Only the first is what
        # callers want. See
        # docs/bugs/2026-05-19-find-features-duplicates.md.
        #
        # The fix is twofold:
        # 1. Skip iteration over tables that can never be the actual
        #    feature target -- the Execution table and any vocabulary
        #    table. Every feature association references both, so
        #    scanning them only produces duplicates.
        # 2. Dedup the remaining list by the association table itself
        #    (qualified schema.name), in case multiple distinct target
        #    tables share an association in some non-canonical layout.
        ml_schema_obj = self.model.schemas.get(self.ml_schema)
        execution_table = ml_schema_obj.tables.get("Execution") if ml_schema_obj is not None else None

        seen_feature_tables: set[tuple[str, str]] = set()
        features: list[Feature] = []
        for schema_name in [*self.domain_schemas, self.ml_schema]:
            schema = self.model.schemas.get(schema_name)
            if schema is None:
                continue
            for t in schema.tables.values():
                if execution_table is not None and t is execution_table:
                    continue
                if self.is_vocabulary(t):
                    continue
                for f in find_table_features(t):
                    key = (f.feature_table.schema.name, f.feature_table.name)
                    if key in seen_feature_tables:
                        continue
                    seen_feature_tables.add(key)
                    features.append(f)
        return features

    def lookup_feature(self, table: TableInput, feature_name: str) -> Feature:
        """Look up the named feature on ``table``.

        Features are association tables (linking a target table to
        vocabulary terms, assets, and metadata) discovered by
        :meth:`find_features`. This is the by-name accessor.

        Args:
            table: The target table the feature is attached to. Name or
                :class:`Table`.
            feature_name: The feature's name as set in its
                ``Feature_Name`` column.

        Returns:
            The :class:`Feature` wrapper for the matching association.

        Raises:
            DerivaMLTableNotFound: If ``table`` doesn't exist.
            DerivaMLFeatureNotFound: If no feature with
                ``feature_name`` is defined on ``table``.
        """
        table = self.name_to_table(table)
        try:
            return [f for f in self.find_features(table) if f.feature_name == feature_name][0]
        except IndexError:
            raise DerivaMLFeatureNotFound(table.name, feature_name) from None

    def asset_metadata(self, table: TableInput) -> set[str]:
        """Return the non-asset columns of an asset table.

        Asset tables are ``Table.is_asset()`` tables: they carry the
        standard ``URL`` / ``Filename`` / ``Length`` / ``MD5`` columns
        plus arbitrary domain-specific metadata. This method returns
        the metadata column names — i.e. everything *except* the four
        standard asset columns (kept in
        :data:`~deriva_ml.core.definitions.DerivaAssetColumns`).

        Args:
            table: The asset table — name or :class:`Table` instance.

        Returns:
            Set of metadata column names. Empty if the asset table
            carries no extra columns.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not an asset table.
            DerivaMLException: If ``table`` doesn't exist (raised by
                :meth:`name_to_table`).
        """
        table = self.name_to_table(table)

        if not self.is_asset(table):
            raise DerivaMLTableTypeError("asset table", table.name)
        return {c.name for c in table.columns} - DerivaAssetColumns

    def asset_metadata_columns(self, table: str | Table) -> list[Column]:
        """Return Column objects for the asset-metadata columns of ``table``.

        Like :meth:`asset_metadata` but returns the :class:`Column`
        instances (not just names) so callers can inspect attributes
        such as ``nullok``. Results are sorted by column name for
        deterministic iteration.

        Args:
            table: Asset table name or Table object.

        Returns:
            Sorted list of Column objects.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not an asset table.
        """
        table = self.name_to_table(table)
        if not self.is_asset(table):
            raise DerivaMLTableTypeError("asset table", table.name)
        return sorted(
            (c for c in table.columns if c.name not in DerivaAssetColumns),
            key=lambda c: c.name,
        )

    def apply(self) -> None:
        """Apply pending annotation/schema changes via the underlying Model.

        Thin passthrough to ``self.model.apply()``. Kept explicit so the
        annotation/schema commit boundary is visible on the DerivaModel
        public surface rather than hiding behind generic ``__getattr__``
        delegation.

        Refuses to run when ``self.catalog`` is a
        :class:`~deriva_ml.core.catalog_stub.CatalogStub` (offline mode):
        applying a schema change without a live catalog connection is
        nonsensical, and the underlying ``Model.apply()`` would otherwise
        raise an unhelpful :class:`DerivaMLReadOnlyError` once it reached
        through the stub.

        Raises:
            DerivaMLReadOnlyError: If this DerivaML instance is in offline
                mode (``self.catalog`` is a ``CatalogStub``).
        """
        if isinstance(self.catalog, CatalogStub):
            raise DerivaMLReadOnlyError(
                "DerivaModel.apply() requires online mode; this DerivaML instance was constructed with mode=offline."
            )
        self.model.apply()

    def is_dataset_rid(self, rid: RID, deleted: bool = False) -> bool:
        """Check whether ``rid`` identifies a (non-deleted) Dataset row.

        Resolves ``rid`` against the live catalog via
        :meth:`ErmrestCatalog.resolve_rid` to determine which table it
        belongs to, then verifies it's the ``Dataset`` table. By default
        deleted datasets are treated as not-a-dataset; pass ``deleted=True``
        to include tombstoned rows in the positive set.

        Args:
            rid: The RID to test.
            deleted: If True, return ``True`` for soft-deleted datasets
                too. Defaults to False (deleted rows return ``False``).

        Returns:
            True if ``rid`` is a Dataset row (filtered by the ``deleted``
            flag), False if it points at a different table.

        Raises:
            DerivaMLException: If ``rid`` doesn't resolve in the catalog
                at all (typically an invalid or fabricated RID).
        """
        try:
            rid_info = self.model.catalog.resolve_rid(rid, self.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")
        if rid_info.table.name != "Dataset":
            return False
        elif deleted:
            # Got a dataset rid. Now check to see if its deleted or not.
            return True
        else:
            return not list(rid_info.datapath.entities().fetch())[0]["Deleted"]

    def list_dataset_element_types(self) -> list[Table]:
        """List the deriva-py ``Table`` types that can be dataset members.

        Walks ``Dataset.find_associations()`` and returns the
        ``other_fkey.pk_table`` for each association whose target is a
        domain-schema table or the Dataset table itself. Used by
        ``DerivaML.add_dataset_members`` to validate the kind of row
        a caller is trying to add to a dataset.

        Returns:
            A list of :class:`~deriva.core.ermrest_model.Table`
            objects — one per valid member type.
        """

        dataset_table = self.name_to_table("Dataset")

        def is_domain_or_dataset_table(table: Table) -> bool:
            return self.is_domain_schema(table.schema.name) or table.name == dataset_table.name

        return [
            t
            for a in dataset_table.find_associations()
            if is_domain_or_dataset_table(t := a.other_fkeys.pop().pk_table)
        ]

    # ------------------------------------------------------------------
    # Denormalization planner
    #
    # The planner — schema-graph reachability + JOIN tree construction —
    # was extracted into :mod:`deriva_ml.model.denormalize_planner` in
    # Phase 3 (audit §5.2). It's a ~1100 LoC algorithm subsystem with a
    # narrow consumer set (``local_db/`` + a couple of single-line
    # sites). The split keeps :class:`DerivaModel` focused on its
    # wide-fan-out role (introspection touched by every mixin) and
    # gives the planner its own focused module.
    #
    # Access the planner via :attr:`_planner`. All planner methods are
    # underscore-prefixed because the planner is internal to the
    # denormalization subsystem; the user-facing API is
    # :class:`local_db.denormalize.Denormalizer`.
    #
    # The :meth:`_schema_to_paths` forwarder below exists because
    # ``dataset/dataset_bag.py:381`` still reaches into the model's FK
    # traversal directly. A follow-up cleanup will migrate that caller
    # to ``model._planner._schema_to_paths`` and drop the forwarder.
    # ------------------------------------------------------------------

    @property
    def _planner(self) -> "DenormalizePlanner":
        """Lazily-constructed :class:`DenormalizePlanner` for this model.

        Cached on the instance after first access so reachability /
        join-tree computations don't repeat the construction cost. The
        planner reads schemas/tables through ``self`` and never mutates
        the model, so the cache is safe to share. The planner itself
        isn't documented as thread-safe — callers needing concurrent
        access should construct their own ``DenormalizePlanner``
        per-thread.

        Uses a single-underscore attribute name (``_planner_cache``)
        rather than double-underscore to avoid Python's name
        mangling and keep ``hasattr`` lookups straightforward.
        """
        if not hasattr(self, "_planner_cache"):
            self._planner_cache = DenormalizePlanner(self)
        return self._planner_cache

    def _schema_to_paths(
        self,
        root: "Table | None" = None,
        path: "list[Table] | None" = None,
        exclude_tables: "set[str] | None" = None,
        skip_tables: "frozenset[str] | None" = None,
        max_depth: int | None = None,
        stop_at: str | None = None,
    ) -> "list[list[Table]]":
        """Forwarder to :meth:`DenormalizePlanner._schema_to_paths`.

        Kept on ``DerivaModel`` because ``dataset/dataset_bag.py:381``
        still calls it on the model. New code should use
        ``model._planner._schema_to_paths``.
        """
        return self._planner._schema_to_paths(
            root=root,
            path=path,
            exclude_tables=exclude_tables,
            skip_tables=skip_tables,
            max_depth=max_depth,
            stop_at=stop_at,
        )

    def create_table(self, table_def: TableDefinition, schema: str | None = None) -> Table:
        """Create a new table from TableDefinition.

        Args:
            table_def: Table definition (dataclass or dict).
            schema: Schema to create the table in. If None, uses default_schema.

        Returns:
            The newly created Table.

        Raises:
            DerivaMLException: If no schema specified and default_schema is not set.

        Note: @validate_call removed because TableDefinition is now a dataclass from
        deriva.core.typed and Pydantic validation doesn't work well with dataclass fields.
        """
        schema = schema or self._require_default_schema()
        # Handle both TableDefinition (dataclass with to_dict) and plain dicts
        table_dict = table_def.to_dict() if hasattr(table_def, "to_dict") else table_def
        return self.model.schemas[schema].create_table(table_dict)

    def _define_association(
        self,
        associates: list,
        metadata: list | None = None,
        table_name: str | None = None,
        comment: str | None = None,
        **kwargs,
    ) -> dict:
        """Build an association table definition with vocab-aware key selection.

        Wraps Table.define_association to ensure non-vocabulary tables use RID
        as their foreign key target. The default key search heuristic in
        define_association prefers Name/ID keys over RID, which is correct for
        vocabulary tables (FK to human-readable Name) but wrong for domain
        tables that happen to have non-nullable Name or ID keys (e.g., tables
        in cloned catalogs like FaceBase).

        Args:
            associates: Reference targets being associated (Table, Key, or tuples).
            metadata: Additional metadata fields and/or reference targets.
            table_name: Name for the association table.
            comment: Comment for the association table.
            **kwargs: Additional arguments passed to Table.define_association.

        Returns:
            Table definition dict suitable for create_table.
        """
        metadata = metadata or []

        def _resolve_key(ref):
            """Convert non-vocabulary Table references to their RID Key."""
            if isinstance(ref, tuple):
                # (name, Table) or (name, nullok, Table) — resolve the Table element
                items = list(ref)
                table_obj = items[-1]
                if isinstance(table_obj, Table) and not table_obj.is_vocabulary():
                    items[-1] = table_obj.key_by_columns(["RID"])
                return tuple(items)
            elif isinstance(ref, Table) and not ref.is_vocabulary():
                return ref.key_by_columns(["RID"])
            return ref  # Key objects or vocabulary Tables pass through

        resolved_associates = [_resolve_key(a) for a in associates]
        resolved_metadata = [_resolve_key(m) for m in metadata]

        return Table.define_association(
            associates=resolved_associates,
            metadata=resolved_metadata,
            table_name=table_name,
            comment=comment,
            **kwargs,
        )

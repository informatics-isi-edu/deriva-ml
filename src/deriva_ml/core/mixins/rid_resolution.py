"""RID resolution mixin for DerivaML.

This module provides the RidResolutionMixin class which handles
Resource Identifier (RID) resolution and retrieval operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_datapath = importlib.import_module("deriva.core.datapath")
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

AnyQuantifier = _datapath.Any
ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
ErmrestSnapshot = _ermrest_catalog.ErmrestSnapshot
ResolveRidResult = _ermrest_catalog.ResolveRidResult
Table = _ermrest_model.Table

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


@dataclass
class BatchRidResult:
    """Result of batch RID resolution.

    Attributes:
        rid: The resolved RID (normalized form).
        table: The Table object containing this RID.
        table_name: The name of the table containing this RID.
        schema_name: The name of the schema containing this RID.
    """

    rid: RID
    table: Table
    table_name: str
    schema_name: str


class RidResolutionMixin:
    """Mixin providing RID resolution and retrieval operations.

    This mixin requires the host class to have:
        - catalog: ErmrestCatalog or ErmrestSnapshot instance
        - model: DerivaModel instance (with .model attribute for ermrest model)
        - pathBuilder(): method returning catalog path builder

    Methods:
        resolve_rid: Resolve a RID to its catalog location
        resolve_rids: Batch resolve multiple RIDs efficiently
        retrieve_rid: Retrieve the complete record for a RID
    """

    # Type hints for IDE support - actual attributes from host class
    catalog: ErmrestCatalog | ErmrestSnapshot
    model: "DerivaModel"
    pathBuilder: Any  # Callable returning path builder

    def resolve_rid(self, rid: RID) -> ResolveRidResult:
        """Resolves RID to catalog location.

        Looks up a RID and returns information about where it exists in the catalog, including schema,
        table, and column metadata.

        Args:
            rid: Resource Identifier to resolve.

        Returns:
            ResolveRidResult: Named tuple containing:
                - schema: Schema name
                - table: Table name
                - columns: Column definitions
                - datapath: Path builder for accessing the entity

        Raises:
            DerivaMLException: If RID doesn't exist in catalog.

        Examples:
            >>> result = ml.resolve_rid("1-abc123")
            >>> print(f"Found in {result.schema}.{result.table}")
            >>> data = result.datapath.entities().fetch()
        """
        try:
            # Attempt to resolve RID using catalog model
            return self.catalog.resolve_rid(rid, self.model.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")

    def retrieve_rid(self, rid: RID) -> dict[str, Any]:
        """Retrieves complete record for RID.

        Fetches all column values for the entity identified by the RID.

        Args:
            rid: Resource Identifier of the record to retrieve.

        Returns:
            dict[str, Any]: Dictionary containing all column values for the entity.

        Raises:
            DerivaMLException: If the RID doesn't exist in the catalog.

        Example:
            >>> record = ml.retrieve_rid("1-abc123")
            >>> print(f"Name: {record['name']}, Created: {record['creation_date']}")
        """
        # Resolve RID and fetch the first (only) matching record
        return self.resolve_rid(rid).datapath.entities().fetch()[0]

    def resolve_rids(
        self,
        rids: set[RID] | list[RID],
        candidate_tables: list[Table] | None = None,
    ) -> dict[RID, BatchRidResult]:
        """Batch resolve multiple RIDs efficiently.

        Resolves multiple RIDs in batched queries, significantly faster than
        calling resolve_rid() for each RID individually. Instead of N network
        calls for N RIDs, this makes one query per candidate table.

        Args:
            rids: Set or list of RIDs to resolve.
            candidate_tables: Optional list of Table objects to search in.
                If not provided, searches all tables in domain and ML schemas.

        Returns:
            dict[RID, BatchRidResult]: Mapping from each resolved RID to its
                BatchRidResult containing table information.

        Raises:
            DerivaMLException: If any RID cannot be resolved.

        Example:
            >>> results = ml.resolve_rids(["1-ABC", "2-DEF", "3-GHI"])
            >>> for rid, info in results.items():
            ...     print(f"{rid} is in table {info.table_name}")
        """
        rids = set(rids)
        if not rids:
            return {}

        results: dict[RID, BatchRidResult] = {}
        remaining_rids = set(rids)

        # Determine which tables to search
        if candidate_tables is None:
            # Search all tables in domain and ML schemas
            candidate_tables = []
            for schema_name in [self.model.domain_schema, self.model.ml_schema]:
                schema = self.model.model.schemas.get(schema_name)
                if schema:
                    candidate_tables.extend(schema.tables.values())

        pb = self.pathBuilder()

        # Query each candidate table for matching RIDs
        for table in candidate_tables:
            if not remaining_rids:
                break

            schema_name = table.schema.name
            table_name = table.name

            # Build a query with RID filter for all remaining RIDs
            table_path = pb.schemas[schema_name].tables[table_name]

            # Use ERMrest's Any quantifier for IN-style query
            # Query only for RID column to minimize data transfer
            try:
                # Filter: RID = any(rid1, rid2, ...) - ERMrest's way of doing IN clause
                found_entities = list(
                    table_path.filter(table_path.RID == AnyQuantifier(*remaining_rids))
                    .attributes(table_path.RID)
                    .fetch()
                )
            except Exception:
                # Table might not support this query, skip it
                continue

            # Process found RIDs
            for entity in found_entities:
                rid = entity["RID"]
                if rid in remaining_rids:
                    results[rid] = BatchRidResult(
                        rid=rid,
                        table=table,
                        table_name=table_name,
                        schema_name=schema_name,
                    )
                    remaining_rids.remove(rid)

        # Check if any RIDs were not found
        if remaining_rids:
            raise DerivaMLException(f"Invalid RIDs: {remaining_rids}")

        return results

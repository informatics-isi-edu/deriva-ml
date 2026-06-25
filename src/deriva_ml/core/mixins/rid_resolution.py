"""RID resolution mixin for DerivaML.

This module provides the RidResolutionMixin class which handles
Resource Identifier (RID) resolution and retrieval operations.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, Any

_datapath = importlib.import_module("deriva.core.datapath")
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

AnyQuantifier = _datapath.Any
ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
ErmrestSnapshot = _ermrest_catalog.ErmrestSnapshot
ResolveRidResult = _ermrest_catalog.ResolveRidResult
Table = _ermrest_model.Table


from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLRidsNotFound
from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)
if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


__all__ = [
    "RidResolutionMixin",
    "BatchRidResult",
]

# Maximum number of RIDs placed in a single ``RID = Any(...)`` filter.
#
# That filter renders into the GET URL *path* (deriva-py datapath builds
# ``base_uri + str(path_expression)`` and issues ``catalog.get(path)``), so over
# enough RIDs the request line exceeds the server's URI length limit and the
# query fails (a 10k-RID URL is ~70 KB → HTTP 414). Without chunking, resolving
# a few hundred+ RIDs in one shot overflowed the URL and the failure was
# silently turned into a spurious "RIDs not found".
#
# 500 matches deriva-py's ``RID_SET_CHUNK_SIZE`` (its URL-safe batch size for
# the bulk ``get_as_file(rid_set=...)`` path) — proven in production against
# www.eye-ai.org's long-form RIDs (20 chunks of 500 fetched cleanly, no 414).
# Kept in lockstep deliberately: both place ``RID=any(...)`` chunks in a GET URL
# and share the same limit. (A stricter localhost Apache caps ~4 KB ≈ 994
# three-char RIDs; production allows more, which is why 500 long RIDs is safe
# there. The test environment uses short RIDs, so 500 is well under even the
# localhost limit.)
_MAX_RIDS_PER_QUERY = 500


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
        _retrieve_rid: Retrieve the complete record for a RID (internal)
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
            >>> result = ml.resolve_rid("1-abc123")  # doctest: +SKIP
            >>> print(f"Found in {result.schema}.{result.table}")  # doctest: +SKIP
            >>> data = result.datapath.entities().fetch()  # doctest: +SKIP
        """
        try:
            # Attempt to resolve RID using catalog model
            return self.catalog.resolve_rid(rid, self.model.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")

    def _retrieve_rid(self, rid: RID) -> dict[str, Any]:
        """Retrieves complete record for RID.

        Fetches all column values for the entity identified by the RID.

        Args:
            rid: Resource Identifier of the record to retrieve.

        Returns:
            dict[str, Any]: Dictionary containing all column values for the entity.

        Raises:
            DerivaMLException: If the RID doesn't exist in the catalog.

        Example:
            >>> record = ml._retrieve_rid("1-abc123")  # doctest: +SKIP
            >>> print(f"Name: {record['name']}, Created: {record['creation_date']}")  # doctest: +SKIP
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
            >>> results = ml.resolve_rids(["1-ABC", "2-DEF", "3-GHI"])  # doctest: +SKIP
            >>> for rid, info in results.items():  # doctest: +SKIP
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
            for schema_name in [*self.model.domain_schemas, self.model.ml_schema]:
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
            table_path = pb.schemas[schema_name].tables[table_name]

            # The ``RID = Any(...)`` filter renders into the GET URL path, so a
            # single query over hundreds of RIDs overflows the server's request
            # URL limit and fails. Chunk the remaining RIDs into URL-safe
            # batches (``_MAX_RIDS_PER_QUERY``) and union the matches. Snapshot
            # the RID list first because ``remaining_rids`` is mutated below.
            for rid_chunk in batched(list(remaining_rids), _MAX_RIDS_PER_QUERY):
                # Filter: RID = any(rid1, rid2, ...) — ERMrest's IN clause.
                # Query only the RID column to minimize data transfer.
                try:
                    found_entities = list(
                        table_path.filter(table_path.RID == AnyQuantifier(*rid_chunk))
                        .attributes(table_path.RID)
                        .fetch()
                    )
                except Exception as e:
                    # Resilience across heterogeneous candidate tables: a query
                    # that errors on one table shouldn't abort the whole scan —
                    # the RIDs simply stay in ``remaining_rids`` and surface as a
                    # legitimate DerivaMLRidsNotFound if no table matched them.
                    # This is narrow now: the oversized-URL failure that used to
                    # land here (and get masked into a spurious not-found) is
                    # prevented by the chunking above, so an error reaching this
                    # point is a genuine per-table condition, logged loudly.
                    logger.warning(f"RID resolution query failed for {schema_name}.{table_name}: {e}")
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

        # Check if any RIDs were not found. Raise the typed
        # ``DerivaMLRidsNotFound`` so callers can pull the unresolved
        # set off ``e.missing_rids`` without string-parsing the
        # message — see ``DerivaML.validate_rids``, which previously
        # had to grep the message for ``"Invalid RIDs:"`` because
        # this raise site emitted a bare ``DerivaMLException``.
        if remaining_rids:
            raise DerivaMLRidsNotFound(remaining_rids)

        return results

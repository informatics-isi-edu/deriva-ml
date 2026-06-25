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

        Resolves multiple RIDs in batched queries, far faster than calling
        ``resolve_rid`` once per RID. RIDs are matched against a table with a
        single chunked ``RID = Any(...)`` query rather than N per-RID lookups.

        Two strategies depending on ``candidate_tables``:

        - **Explicit candidate tables** (e.g. dataset element types): each
          listed table is bulk-matched in turn until every RID is resolved.
        - **No candidate tables** (``None``): instead of scanning *every* table
          in the domain + ML schemas (a wasted zero-row query per table that
          holds none of the RIDs), the server is asked which table a sample RID
          lives in via ``resolve_rid`` (one ``/entity_rid`` call, catalog-wide),
          the rest are bulk-matched in that table, and the process repeats for
          any RIDs in other tables. Cost is ~one probe per distinct table the
          RIDs actually span, independent of catalog size.

        Args:
            rids: Set or list of RIDs to resolve.
            candidate_tables: Optional list of Table objects to search in.
                If not provided, the server resolves each RID's table directly
                (probe strategy above) — note this resolves catalog-wide, so a
                RID in any schema is found, not only domain/ML schemas.

        Returns:
            dict[RID, BatchRidResult]: Mapping from each resolved RID to its
                BatchRidResult containing table information.

        Raises:
            DerivaMLRidsNotFound: If any RID cannot be resolved; the unresolved
                set is available as ``e.missing_rids``.

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
        # RIDs the probe strategy proves don't exist anywhere in the catalog.
        # Tracked separately because they're removed from ``remaining_rids`` to
        # keep the probe loop progressing, but must still surface in the final
        # DerivaMLRidsNotFound.
        missing_rids: set[RID] = set()
        pb = self.pathBuilder()

        def match_in_table(table: Table) -> None:
            """Bulk-match ``remaining_rids`` against one table, recording hits.

            The ``RID = Any(...)`` filter renders into the GET URL path, so a
            single query over hundreds of RIDs overflows the server's request
            URL limit. Chunk into URL-safe batches (``_MAX_RIDS_PER_QUERY``) and
            union the matches; every matched RID is recorded and dropped from
            ``remaining_rids``.
            """
            schema_name = table.schema.name
            table_name = table.name
            table_path = pb.schemas[schema_name].tables[table_name]
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
                    # Resilience: a query that errors on one candidate table
                    # shouldn't abort the whole resolution — the RIDs stay in
                    # ``remaining_rids`` and surface as a legitimate
                    # DerivaMLRidsNotFound if nothing matches them. Narrow now:
                    # the oversized-URL failure that used to land here (and get
                    # masked into a spurious not-found) is prevented by the
                    # chunking above, so an error here is a genuine per-table
                    # condition, logged loudly.
                    logger.warning(f"RID resolution query failed for {schema_name}.{table_name}: {e}")
                    continue
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

        if candidate_tables is not None:
            # Caller already narrowed the search (e.g. dataset element types).
            # Scan exactly those tables, bulk-matching in each.
            for table in candidate_tables:
                if not remaining_rids:
                    break
                match_in_table(table)
        else:
            # No candidate tables given. Rather than scan EVERY table in the
            # domain + ML schemas (firing a zero-row query at each table that
            # holds none of the RIDs), ask the server which table a sample RID
            # lives in — ``resolve_rid`` hits ``/entity_rid`` and resolves
            # catalog-wide in one cheap call — then bulk-match the rest there,
            # and repeat. Cost is ~one probe per distinct table the RIDs span
            # plus the chunk queries, independent of catalog size.
            while remaining_rids:
                probe_rid = next(iter(remaining_rids))
                try:
                    probe_table = self.resolve_rid(probe_rid).table
                except DerivaMLException:
                    # The probe RID doesn't exist anywhere in the catalog. Move
                    # it to ``missing_rids`` (out of the working set so the loop
                    # progresses) — it surfaces in the final
                    # DerivaMLRidsNotFound. Don't abort: other RIDs may be valid
                    # and in other tables.
                    remaining_rids.discard(probe_rid)
                    missing_rids.add(probe_rid)
                    continue
                before = len(remaining_rids)
                match_in_table(probe_table)
                # ``match_in_table`` removed every RID it found in
                # ``probe_table``; ``probe_rid`` itself is guaranteed gone (the
                # server placed it there), so progress is guaranteed and the
                # loop terminates.
                if len(remaining_rids) == before:  # pragma: no cover - defensive
                    # Server resolved probe_rid to a table that the bulk match
                    # didn't return it from (should not happen). Treat it as
                    # unresolved to guarantee termination.
                    remaining_rids.discard(probe_rid)
                    missing_rids.add(probe_rid)

        # Check if any RIDs were not found. ``remaining_rids`` holds anything no
        # candidate table matched; ``missing_rids`` holds probes the server said
        # don't exist. Raise the typed ``DerivaMLRidsNotFound`` so callers can
        # pull the unresolved set off ``e.missing_rids`` without string-parsing
        # the message — see ``DerivaML.validate_rids``, which previously had to
        # grep the message for ``"Invalid RIDs:"`` because this raise site
        # emitted a bare ``DerivaMLException``.
        unresolved = remaining_rids | missing_rids
        if unresolved:
            raise DerivaMLRidsNotFound(unresolved)

        return results

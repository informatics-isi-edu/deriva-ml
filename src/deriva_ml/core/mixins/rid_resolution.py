"""RID resolution mixin for DerivaML.

This module provides the RidResolutionMixin class which handles
Resource Identifier (RID) resolution and retrieval operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deriva.core.ermrest_catalog import ErmrestCatalog, ErmrestSnapshot, ResolveRidResult

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


class RidResolutionMixin:
    """Mixin providing RID resolution and retrieval operations.

    This mixin requires the host class to have:
        - catalog: ErmrestCatalog or ErmrestSnapshot instance
        - model: DerivaModel instance (with .model attribute for ermrest model)

    Methods:
        resolve_rid: Resolve a RID to its catalog location
        retrieve_rid: Retrieve the complete record for a RID
    """

    # Type hints for IDE support - actual attributes from host class
    catalog: ErmrestCatalog | ErmrestSnapshot
    model: "DerivaModel"

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

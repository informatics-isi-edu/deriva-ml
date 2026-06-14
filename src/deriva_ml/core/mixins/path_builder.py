"""Path builder mixin for DerivaML.

This module provides the PathBuilderMixin class which handles
catalog path building and table access utilities.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

datapath = importlib.import_module("deriva.core.datapath")
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
ErmrestSnapshot = _ermrest_catalog.ErmrestSnapshot
Table = _ermrest_model.Table

import pandas as pd

from deriva_ml.core.pd_utils import rows_to_dataframe
from deriva_ml.core.upload_layout import table_path as _table_path

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


__all__ = ["PathBuilderMixin"]


class PathBuilderMixin:
    """Mixin providing path building and table access utilities.

    This mixin requires the host class to have:
        - catalog: ErmrestCatalog or ErmrestSnapshot instance
        - domain_schemas: frozenset[str] - names of the domain schemas
        - model: DerivaModel instance
        - working_dir: Path - working directory path

    Methods:
        pathBuilder: Get catalog path builder for queries
        _domain_path: Internal path builder for domain schema
        _table_path: Internal filesystem path for table CSV files
        get_table_as_dataframe: Get table contents as pandas DataFrame
        get_table_as_dict: Get table contents as dictionaries
        user_list: Get catalog users from public.ERMrest_Client
    """

    # Type hints for IDE support - actual attributes from host class
    catalog: ErmrestCatalog | ErmrestSnapshot
    domain_schemas: frozenset[str]
    default_schema: str | None
    model: "DerivaModel"
    working_dir: Path

    def pathBuilder(self) -> "datapath._CatalogWrapper":
        """Returns catalog path builder for queries.

        The path builder provides a fluent interface for constructing complex queries against the catalog.
        This is a core component used by many other methods to interact with the catalog.

        Returns:
            datapath._CatalogWrapper: A new instance of the catalog path builder.

        Raises:
            Exception: If the catalog connection is unavailable.

        Example:
            >>> pb = ml.pathBuilder()  # doctest: +SKIP
            >>> path = pb.schemas['my_schema'].tables['my_table']  # doctest: +SKIP
            >>> results = path.entities().fetch()  # doctest: +SKIP
        """
        # Build the path-builder wrapper from the model deriva-ml already
        # holds (self.model.model) instead of letting deriva-py re-fetch
        # /schema on every call. The wrapper's schema structure comes
        # entirely from the model; HTTP (reads AND writes) still routes
        # through self.catalog (the wrapper's _wrapped_catalog), so
        # writes, datapath joins, and snapshot-pinning are unchanged.
        #
        # Correctness under schema changes: self.model.model is the
        # authoritative in-memory Model. create_table mutates it in place
        # (a wrapper built afterward, on a cache MISS, sees the new
        # table); refresh_model()/refresh_schema() REBIND it to a new
        # object. We cache the wrapper keyed on the inner-model object
        # identity, so a rebind (refresh) invalidates the cache and the
        # next call rebuilds from the current model. The previous approach
        # cached deriva-py's getPathBuilder() result and went stale after
        # an in-place create_table (same identity, stale wrapper); building
        # from the held model via datapath.from_model avoids both the
        # staleness and the redundant /schema fetch.
        inner_model = self.model.model
        cached = getattr(self, "_path_builder_cache", None)
        if cached is not None and cached[0] is inner_model:
            return cached[1]
        wrapper = datapath.from_model(self.catalog, inner_model)
        self._path_builder_cache = (inner_model, wrapper)
        return wrapper

    def _domain_path(self, schema: str | None = None) -> datapath.DataPath:
        """Returns path builder for a domain schema.

        Provides a convenient way to access tables and construct queries within a domain-specific schema.

        Args:
            schema: Schema name to get path builder for. If None, uses default_schema.

        Returns:
            datapath._CatalogWrapper: Path builder object scoped to the specified domain schema.

        Raises:
            DerivaMLException: If no schema specified and default_schema is not set.

        Example:
            >>> domain = ml._domain_path()  # Uses default schema  # doctest: +SKIP
            >>> results = domain.my_table.entities().fetch()  # doctest: +SKIP
            >>> # Or with explicit schema:
            >>> domain = ml._domain_path("my_schema")  # doctest: +SKIP
        """
        schema = schema or self.model._require_default_schema()
        return self.pathBuilder().schemas[schema]

    def _table_path(self, table: str | Table, schema: str | None = None) -> Path:
        """Returns a local filesystem path for table CSV files.

        Generates a standardized path where CSV files should be placed when preparing to upload data to a table.
        The path follows the project's directory structure conventions.

        Args:
            table: Name of the table or Table object to get the path for.
            schema: Schema name for the path. If None, uses the table's schema or default_schema.

        Returns:
            Path: Filesystem path where the CSV file should be placed.

        Example:
            >>> path = ml._table_path("experiment_results")  # doctest: +SKIP
            >>> df.to_csv(path)  # Save data for upload  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)
        # Use table's schema if available, otherwise use provided schema or default
        schema = schema or table_obj.schema.name
        return _table_path(
            self.working_dir,
            schema=schema,
            table=table_obj.name,
        )

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Retrieves all contents of a table from the catalog.

        Args:
            table: Name of the table to retrieve.

        Returns:
            DataFrame containing all table contents.

        Raises:
            DerivaMLTableNotFound: If the table does not exist in any schema.

        Example:
            >>> df = ml.get_table_as_dataframe("Subject")  # doctest: +SKIP
        """
        return rows_to_dataframe(self.get_table_as_dict(table))

    def get_table_as_dict(self, table: str) -> Iterable[dict[str, Any]]:
        """Get table contents as dictionaries.

        Retrieves all contents of a table from the catalog.

        Args:
            table: Name of the table to retrieve.

        Returns:
            Iterable yielding dictionaries for each row.

        Raises:
            DerivaMLTableNotFound: If the table does not exist in any schema.

        Example:
            >>> rows = list(ml.get_table_as_dict("Subject"))  # doctest: +SKIP
        """
        table_obj = self.model.name_to_table(table)
        pb = self.pathBuilder()
        yield from pb.schemas[table_obj.schema.name].tables[table_obj.name].entities().fetch()

    def user_list(self) -> list[dict[str, str]]:
        """Returns the catalog user list.

        Retrieves basic information about all users who have access to the
        catalog, from the ``public.ERMrest_Client`` table.

        Note:
            The user table lives in the ``public`` schema, which is *outside*
            the domain/ML schema search path used by
            :meth:`get_table_as_dict` (``name_to_table`` searches
            ``domain_schemas → ml_schema → WWW``). This method is the
            supported accessor for catalog users; ``get_table_as_dict``
            cannot reach ``ERMrest_Client``.

        Returns:
            A list of user dictionaries, each with:
                - ``'ID'``: the user's globus identifier
                - ``'Full_Name'``: the user's full name

        Example:
            >>> users = ml.user_list()  # doctest: +SKIP
            >>> for user in users:  # doctest: +SKIP
            ...     print(f"{user['Full_Name']} ({user['ID']})")
        """
        user_path = self.pathBuilder().schemas["public"].tables["ERMrest_Client"]
        return [{"ID": u["ID"], "Full_Name": u["Full_Name"]} for u in user_path.entities().fetch()]

"""DerivaML-specific database model for downloaded BDBags.

This module provides the DatabaseModel class which creates a SQLite database
from a BDBag and provides DerivaML-specific functionality:

- Dataset version tracking
- Dataset RID resolution
- Integration with DerivaModel for schema analysis

The implementation uses a two-phase pattern:
1. Phase 1 (SchemaBuilder): Create SQLAlchemy ORM from schema.json
2. Phase 2 (DataLoader): Load data from CSV files

For the low-level components, see:
- schema_builder.py: SchemaBuilder, SchemaORM
- data_sources.py: DataSource, BagDataSource, CatalogDataSource
- data_loader.py: DataLoader
- fk_orderer.py: ForeignKeyOrderer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, Type

from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import Table as DerivaTable
from sqlalchemy import Table as SQLTable
from sqlalchemy import select
from sqlalchemy.orm import Session

from deriva_ml.core.definitions import ML_SCHEMA, RID, get_domain_schemas
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetMinid, DatasetVersion
from deriva_ml.model.catalog import DerivaModel
from deriva_ml.model.data_loader import DataLoader
from deriva_ml.model.data_sources import BagDataSource
from deriva_ml.model.schema_builder import SchemaBuilder, SchemaORM

logger = logging.getLogger(__name__)


class DatabaseModel(DerivaModel):
    """DerivaML database model for downloaded BDBags.

    This class creates a SQLite database from a BDBag and provides:
    - SQLAlchemy ORM access (engine, metadata, Base)
    - DerivaModel schema methods (find_features, is_asset, etc.)
    - Dataset version tracking (bag_rids, dataset_version)
    - Dataset RID validation (rid_lookup)

    The implementation uses a two-phase pattern:
    1. SchemaBuilder creates SQLAlchemy ORM from schema.json
    2. DataLoader fills the database from CSV files

    Attributes:
        bag_path: Path to the BDBag directory.
        minid: DatasetMinid for the downloaded bag.
        dataset_rid: Primary dataset RID in this bag.
        bag_rids: Dictionary mapping all dataset RIDs to their versions.
        dataset_table: The Dataset table from the ERMrest model.
        engine: SQLAlchemy engine for database access.
        metadata: SQLAlchemy MetaData with table definitions.
        Base: SQLAlchemy automap base for ORM classes.

    Example:
        >>> db = DatabaseModel(minid, bag_path, working_dir)
        >>> version = db.dataset_version("ABC123")
        >>> for row in db.get_table_contents("Image"):
        ...     print(row["Filename"])
    """

    def __init__(self, minid: DatasetMinid, bag_path: Path, dbase_path: Path):
        """Create a DerivaML database from a BDBag.

        Args:
            minid: DatasetMinid containing bag metadata (RID, version, etc.).
            bag_path: Path to the BDBag directory.
            dbase_path: Base directory for SQLite database files.
        """
        self._logger = logging.getLogger("deriva_ml")
        self.minid = minid
        self.dataset_rid = minid.dataset_rid
        self.bag_path = bag_path

        # Load the model from schema.json
        schema_file = bag_path / "data/schema.json"
        model = Model.fromfile("file-system", schema_file)

        # Determine schemas using schema classification
        ml_schema = ML_SCHEMA
        domain_schemas = get_domain_schemas(model.schemas.keys(), ml_schema)
        schemas = [*domain_schemas, ml_schema]

        # Extract bag checksum for unique database path
        bag_cache_dir = bag_path.parent.name
        self.database_dir = dbase_path / bag_cache_dir
        self.database_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Build ORM structure
        builder = SchemaBuilder(
            model=model,
            schemas=schemas,
            database_path=self.database_dir,
        )
        self._orm: SchemaORM = builder.build()

        # Phase 2: Load data from bag CSVs
        source = BagDataSource(bag_path, model=model)
        loader = DataLoader(self._orm, source)
        load_counts = loader.load_tables()

        total_rows = sum(load_counts.values())
        self._logger.debug(f"Loaded {total_rows} rows from bag")

        # Initialize DerivaModel (provides schema analysis methods)
        DerivaModel.__init__(
            self,
            model=model,
            ml_schema=ml_schema,
            domain_schemas=domain_schemas,
        )

        self.dataset_table = model.schemas[ml_schema].tables["Dataset"]

        # Build dataset RID -> version mapping from Dataset_Version table
        self._build_bag_rids()

        self._logger.info(
            "Created DerivaML database for dataset %s in %s",
            self.dataset_rid,
            self.database_dir,
        )

    # =========================================================================
    # Property delegates to SchemaORM
    # =========================================================================

    @property
    def engine(self):
        """SQLAlchemy engine for database access."""
        return self._orm.engine

    @property
    def metadata(self):
        """SQLAlchemy MetaData with table definitions."""
        return self._orm.metadata

    @property
    def Base(self):
        """SQLAlchemy automap base for ORM classes."""
        return self._orm.Base

    @property
    def schemas(self) -> list[str]:
        """List of schema names in the database."""
        return self._orm.schemas

    # =========================================================================
    # Dataset version tracking
    # =========================================================================

    def _build_bag_rids(self) -> None:
        """Build mapping of dataset RIDs to their versions in this bag."""
        self.bag_rids: dict[RID, DatasetVersion] = {}

        dataset_version_table = self.metadata.tables.get(f"{self.ml_schema}.Dataset_Version")
        if dataset_version_table is None:
            return

        with self.engine.connect() as conn:
            result = conn.execute(
                select(dataset_version_table.c.Dataset, dataset_version_table.c.Version)
            )
            for rid, version_str in result:
                version = DatasetVersion.parse(version_str)
                # Keep the highest version for each RID
                if rid not in self.bag_rids or version > self.bag_rids[rid]:
                    self.bag_rids[rid] = version

    def dataset_version(self, dataset_rid: RID | None = None) -> DatasetVersion:
        """Get the version of a dataset in this bag.

        Args:
            dataset_rid: Dataset RID to look up. If None, uses the primary dataset.

        Returns:
            DatasetVersion for the specified dataset.

        Raises:
            DerivaMLException: If the RID is not in this bag.
        """
        rid = dataset_rid or self.dataset_rid
        if rid not in self.bag_rids:
            raise DerivaMLException(f"Dataset RID {rid} is not in this bag")
        return self.bag_rids[rid]

    def rid_lookup(self, dataset_rid: RID) -> DatasetVersion | None:
        """Check if a dataset RID exists in this bag.

        Args:
            dataset_rid: RID to look up.

        Returns:
            DatasetVersion if found.

        Raises:
            DerivaMLException: If the RID is not found in this bag.
        """
        if dataset_rid in self.bag_rids:
            return self.bag_rids[dataset_rid]
        raise DerivaMLException(f"Dataset {dataset_rid} not found in this bag")

    # =========================================================================
    # Table/ORM access methods - delegate to SchemaORM
    # =========================================================================

    def list_tables(self) -> list[str]:
        """List all tables in the database.

        Returns:
            List of fully-qualified table names (schema.table), sorted.
        """
        return self._orm.list_tables()

    def find_table(self, table_name: str) -> SQLTable:
        """Find a table by name.

        Args:
            table_name: Table name, with or without schema prefix.

        Returns:
            SQLAlchemy Table object.

        Raises:
            KeyError: If table not found.
        """
        return self._orm.find_table(table_name)

    def get_table_contents(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve all rows from a table as dictionaries.

        Args:
            table: Table name (with or without schema prefix).

        Yields:
            Dictionary for each row with column names as keys.
        """
        yield from self._orm.get_table_contents(table)

    def get_orm_class_by_name(self, table_name: str) -> Any | None:
        """Get the ORM class for a table by name.

        Args:
            table_name: Table name, with or without schema prefix.

        Returns:
            SQLAlchemy ORM class for the table.

        Raises:
            KeyError: If table not found.
        """
        return self._orm.get_orm_class(table_name)

    def get_orm_class_for_table(self, table: SQLTable | DerivaTable | str) -> Any | None:
        """Get the ORM class for a table.

        Args:
            table: SQLAlchemy Table, Deriva Table, or table name.

        Returns:
            SQLAlchemy ORM class, or None if not found.
        """
        return self._orm.get_orm_class_for_table(table)

    @staticmethod
    def is_association_table(
        table_class,
        min_arity: int = 2,
        max_arity: int = 2,
        unqualified: bool = True,
        pure: bool = True,
        no_overlap: bool = True,
        return_fkeys: bool = False,
    ):
        """Check if an ORM class represents an association table.

        Delegates to SchemaORM.is_association_table.
        """
        return SchemaORM.is_association_table(
            table_class, min_arity, max_arity, unqualified, pure, no_overlap, return_fkeys
        )

    def get_association_class(
        self,
        left_cls: Type[Any],
        right_cls: Type[Any],
    ) -> tuple[Any, Any, Any] | None:
        """Find an association class connecting two ORM classes.

        Args:
            left_cls: First ORM class.
            right_cls: Second ORM class.

        Returns:
            Tuple of (association_class, left_relationship, right_relationship),
            or None if no association found.
        """
        return self._orm.get_association_class(left_cls, right_cls)

    # =========================================================================
    # Compatibility methods
    # =========================================================================

    def _get_table_contents(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve table contents as dictionaries.

        This method provides compatibility with existing code that uses
        _get_table_contents. New code should use get_table_contents instead.

        Args:
            table: Table name.

        Yields:
            Dictionary for each row.
        """
        yield from self.get_table_contents(table)

    def _get_dataset_execution(self, dataset_rid: str) -> dict[str, Any] | None:
        """Get the execution associated with a dataset version.

        Looks up the Dataset_Version record for the dataset's version in this bag
        and returns the associated execution information.

        Args:
            dataset_rid: Dataset RID to look up.

        Returns:
            Dataset_Version row as dict, or None if not found.
            The 'Execution' field contains the execution RID (may be None).
        """
        version = self.bag_rids.get(dataset_rid)
        if not version:
            return None

        dataset_version_table = self.find_table("Dataset_Version")
        cmd = select(dataset_version_table).where(
            dataset_version_table.columns.Dataset == dataset_rid,
            dataset_version_table.columns.Version == str(version),
        )

        with Session(self.engine) as session:
            result = session.execute(cmd).mappings().first()
            return dict(result) if result else None

    def get_orm_association_class(self, left_cls, right_cls, **kwargs):
        """Find association class between two ORM classes.

        Wrapper around get_association_class for compatibility.
        """
        return self.get_association_class(left_cls, right_cls)

    # =========================================================================
    # Resource management
    # =========================================================================

    def dispose(self) -> None:
        """Dispose of SQLAlchemy resources.

        Call this when done with the database to properly clean up connections.
        After calling dispose(), the instance should not be used further.
        """
        if hasattr(self, "_orm") and self._orm is not None:
            self._orm.dispose()

    def delete_database(self) -> None:
        """Delete the database files.

        Note: This method is deprecated. Use dispose() and manually remove
        the database directory if needed.
        """
        self.dispose()
        # Note: We don't actually delete files here to avoid data loss.
        # The caller should handle file deletion if needed.

    def __del__(self) -> None:
        """Cleanup resources when garbage collected."""
        self.dispose()

    def __enter__(self) -> "DatabaseModel":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - dispose resources."""
        self.dispose()
        return False

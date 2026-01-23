"""DerivaML-specific database model for downloaded BDBags.

This module provides the DatabaseModel class which extends the generic BagDatabase
from deriva-py with DerivaML-specific functionality:

- Dataset version tracking
- Dataset RID resolution
- Integration with DerivaModel for schema analysis

For schema-independent BDBag operations, see deriva.core.bag_database.BagDatabase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from deriva.core.bag_database import BagDatabase
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import Table as DerivaTable

from deriva_ml.core.definitions import ML_SCHEMA, RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetMinid, DatasetVersion
from deriva_ml.model.catalog import DerivaModel


class DatabaseModel(BagDatabase, DerivaModel):
    """DerivaML database model for downloaded BDBags.

    This class combines the generic BagDatabase functionality with DerivaML-specific
    features like dataset versioning and the DerivaModel schema utilities.

    It reads a BDBag and creates a SQLite database, then provides:
    - All BagDatabase query methods (list_tables, get_table_contents, etc.)
    - All DerivaModel schema methods (find_features, is_asset, etc.)
    - Dataset version tracking (bag_rids, dataset_version)
    - Dataset RID validation (rid_lookup)

    Attributes:
        bag_path: Path to the BDBag directory.
        minid: DatasetMinid for the downloaded bag.
        dataset_rid: Primary dataset RID in this bag.
        bag_rids: Dictionary mapping all dataset RIDs to their versions.
        dataset_table: The Dataset table from the ERMrest model.

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

        # Load the model first to determine schema names
        schema_file = bag_path / "data/schema.json"
        temp_model = Model.fromfile("file-system", schema_file)

        # Determine domain schema
        ml_schema = ML_SCHEMA
        builtin_schemas = {"public", ml_schema, "www", "WWW"}
        user_schemas = {k for k in temp_model.schemas.keys()} - builtin_schemas

        if len(user_schemas) == 1:
            domain_schema = user_schemas.pop()
        else:
            raise DerivaMLException(f"Ambiguous domain schema: {user_schemas}")

        # Initialize BagDatabase (creates SQLite DB)
        BagDatabase.__init__(
            self,
            bag_path=bag_path,
            database_dir=dbase_path,
            schemas=[domain_schema, ml_schema],
        )

        # Initialize DerivaModel (provides schema analysis methods)
        # Note: We pass self.model which was set by BagDatabase
        DerivaModel.__init__(
            self,
            model=self.model,
            ml_schema=ml_schema,
            domain_schema=domain_schema,
        )

        self.dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]

        # Build dataset RID -> version mapping from Dataset_Version table
        self._build_bag_rids()

        self._logger.info(
            "Created DerivaML database for dataset %s in %s",
            self.dataset_rid,
            self.database_dir,
        )

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

    def dataset_version(self, dataset_rid: Optional[RID] = None) -> DatasetVersion:
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

    # Compatibility aliases for methods that have different names in BagDatabase
    def get_orm_association_class(self, left_cls, right_cls, **kwargs):
        """Find association class between two ORM classes.

        Wrapper around BagDatabase.get_association_class for compatibility.
        """
        return self.get_association_class(left_cls, right_cls)

    def delete_database(self) -> None:
        """Delete the database files.

        Note: This method is deprecated. Use dispose() and manually remove
        the database directory if needed.
        """
        self.dispose()
        # Note: We don't actually delete files here to avoid data loss.
        # The caller should handle file deletion if needed.

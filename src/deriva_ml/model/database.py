"""DerivaML-specific bag-backed database model.

:class:`DatabaseModel` is the deriva-ml-domain extension of
:class:`deriva.bag.database.BagDatabase`. ``BagDatabase`` handles
the generic side (opening the bag, building the SQLAlchemy ORM
from ``data/schema.json``, loading CSVs, asset URL localization
via fetch.txt, the WAL engine + schema_meta versioning). This
class adds:

- :class:`~deriva_ml.model.catalog.DerivaModel` integration for
  schema-analysis methods (``find_features``, ``is_asset``,
  ``is_vocabulary``, etc.).
- Dataset version tracking — ``bag_rids`` mapping each dataset
  RID in the bag to its :class:`DatasetVersion`.
- :meth:`dataset_version` / :meth:`rid_lookup` lookups for callers
  that need to resolve a dataset RID to its version.
- :meth:`_get_dataset_execution` helper for the dataset-history
  surface.

Before the deriva.bag migration this class duplicated ~400 lines
of bag-opening / ORM-building / table-lookup logic that's now
inherited from :class:`BagDatabase`. After the migration it's
~80 lines of deriva-ml-specific behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deriva.bag.database import BagDatabase
from deriva.core.ermrest_model import Model
from sqlalchemy import select
from sqlalchemy.orm import Session

from deriva_ml.core.definitions import ML_SCHEMA, RID, _get_domain_schemas
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetMinid, DatasetVersion
from deriva_ml.model.catalog import DerivaModel

logger = logging.getLogger(__name__)


class DatabaseModel(BagDatabase, DerivaModel):
    """DerivaML bag-backed database with dataset-version tracking.

    Combines :class:`~deriva.bag.database.BagDatabase` (generic bag
    SQLAlchemy access) with :class:`~deriva_ml.model.catalog.DerivaModel`
    (deriva-ml schema-analysis methods) and adds the dataset-version
    surface specific to deriva-ml datasets.

    Args:
        minid: :class:`DatasetMinid` describing the source bag —
            carries the dataset RID, version, and the bag's checksum.
            The ``minid.dataset_rid`` becomes :attr:`dataset_rid`.
        bag_path: Path to the BDBag directory (parent of ``data/``).
        dbase_path: Base directory for SQLite database files. A
            per-bag subdirectory is created under
            ``{dbase_path}/databases/{bag_cache_dir}/`` matching the
            on-disk layout established by ``BagCache``.

    Attributes:
        minid (DatasetMinid): The MINID this database was opened from.
        dataset_rid (RID): Primary dataset RID in this bag.
        bag_path (Path): On-disk bag directory.
        bag_rids (dict[RID, DatasetVersion]): Every dataset RID in
            the bag mapped to its version.
        dataset_table: The ``Dataset`` table from the ERMrest model.

    Plus everything :class:`BagDatabase` provides
    (``engine``, ``metadata``, ``Base``, ``schemas``,
    ``list_tables``, ``find_table``, ``get_table_contents``,
    ``get_orm_class_by_name``, ``get_orm_class_for_table``,
    ``is_association_table``, ``get_association_class``,
    ``dispose``, context-manager support) and everything
    :class:`DerivaModel` provides (``find_features``,
    ``is_asset``, ``is_vocabulary``, etc.).

    Example:
        >>> # Typical use through Execution / Dataset:
        >>> # db = DatabaseModel(minid, bag_path, working_dir)
        >>> # version = db.dataset_version("ABC123")
        >>> # for row in db.get_table_contents("Image"):
        >>> #     print(row["Filename"])
    """

    def __init__(
        self,
        minid: DatasetMinid,
        bag_path: Path,
        dbase_path: Path,
    ):
        self._logger = logging.getLogger("deriva_ml")
        self.minid = minid
        self.dataset_rid = minid.dataset_rid
        self.bag_path = Path(bag_path)

        # Parse the bag's schema.json into an ERMrest Model so we
        # know which deriva-ml schemas to include. The schemas list
        # is ``[*domain_schemas, ml_schema]`` so DerivaModel's
        # schema classification works the same way it did before
        # the bag-backed refactor.
        schema_file = self.bag_path / "data" / "schema.json"
        model = Model.fromfile("file-system", schema_file)
        ml_schema = ML_SCHEMA
        domain_schemas = _get_domain_schemas(
            model.schemas.keys(), ml_schema
        )
        schemas = [*domain_schemas, ml_schema]

        # BagDatabase's own __init__ creates the SQLite mirror,
        # loads the bag's CSVs, and sets up the ORM via
        # SchemaBuilder + DataLoader internally. dbase_path is
        # passed under a ``databases/`` subdirectory so the per-bag
        # SQLite cache doesn't pollute the working directory root
        # (matching the layout BagCache uses).
        database_dir = Path(dbase_path) / "databases"
        BagDatabase.__init__(
            self,
            bag_path=self.bag_path,
            database_dir=database_dir,
            schemas=schemas,
        )

        # DerivaModel.__init__ has to run *after* BagDatabase
        # because some DerivaModel methods walk ``self.model``,
        # which is set by BagDatabase's __init__.
        DerivaModel.__init__(
            self,
            model=model,
            ml_schema=ml_schema,
            domain_schemas=domain_schemas,
        )

        self.dataset_table = model.schemas[ml_schema].tables["Dataset"]

        # Build the dataset-RID → version mapping from the bag's
        # Dataset_Version table. Done at open time so callers can
        # resolve versions without touching the live catalog.
        self._build_bag_rids()

        self._logger.info(
            "Created DerivaML database for dataset %s in %s",
            self.dataset_rid,
            self.database_dir,
        )

    # ------------------------------------------------------------------
    # Dataset-version tracking
    # ------------------------------------------------------------------

    def _build_bag_rids(self) -> None:
        """Index the bag's Dataset_Version rows by dataset RID.

        Populates ``self.bag_rids`` so :meth:`dataset_version` and
        :meth:`rid_lookup` can answer questions without re-reading
        the SQLite. The highest version is kept per RID (a bag may
        contain multiple historical versions of nested datasets).
        """
        self.bag_rids: dict[RID, DatasetVersion] = {}

        dataset_version_table = self.metadata.tables.get(
            f"{self.ml_schema}.Dataset_Version"
        )
        if dataset_version_table is None:
            return

        with self.engine.connect() as conn:
            result = conn.execute(
                select(
                    dataset_version_table.c.Dataset,
                    dataset_version_table.c.Version,
                )
            )
            for rid, version_str in result:
                version = DatasetVersion.parse(version_str)
                # Keep the highest version per RID.
                if rid not in self.bag_rids or version > self.bag_rids[rid]:
                    self.bag_rids[rid] = version

    def dataset_version(
        self, dataset_rid: RID | None = None
    ) -> DatasetVersion:
        """Return the version of a dataset in this bag.

        Args:
            dataset_rid: RID to look up. ``None`` defaults to the
                bag's primary dataset (``self.dataset_rid``).

        Returns:
            :class:`DatasetVersion` for the named dataset.

        Raises:
            DerivaMLException: If the RID is not in this bag.
        """
        rid = dataset_rid or self.dataset_rid
        if rid not in self.bag_rids:
            raise DerivaMLException(
                f"Dataset RID {rid} is not in this bag"
            )
        return self.bag_rids[rid]

    def rid_lookup(self, dataset_rid: RID) -> DatasetVersion | None:
        """Return the version for a dataset RID, raising if absent.

        Distinct from :meth:`dataset_version`: this method is
        positively-named ("look up the RID"), and the historical
        signature returns ``DatasetVersion | None`` rather than
        always-``DatasetVersion``. In practice it raises on miss
        rather than returning ``None`` — matching pre-migration
        behavior the callers depend on.

        Args:
            dataset_rid: RID to look up.

        Returns:
            :class:`DatasetVersion` for the RID.

        Raises:
            DerivaMLException: If the RID is not in this bag.
        """
        if dataset_rid in self.bag_rids:
            return self.bag_rids[dataset_rid]
        raise DerivaMLException(
            f"Dataset {dataset_rid} not found in this bag"
        )

    def _get_dataset_execution(
        self, dataset_rid: str
    ) -> dict[str, Any] | None:
        """Return the Dataset_Version row for ``(dataset_rid, current_version)``.

        Used by the dataset-history surface to surface the
        :class:`Execution` that produced a particular dataset
        version. The result is a plain dict matching the
        ``Dataset_Version`` row columns (notably ``Execution``,
        ``Description``, ``Snapshot``).

        Args:
            dataset_rid: Dataset RID to look up.

        Returns:
            The matching ``Dataset_Version`` row as a dict, or
            ``None`` if either the RID isn't in this bag or no
            row matches the version we hold for it.
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

__all__ = ["DatabaseModel"]

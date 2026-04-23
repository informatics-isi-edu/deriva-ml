"""Per-feature denormalization cache for DatasetBag.

Bags are immutable after materialization. This module provides ``BagFeatureCache``,
which populates a per-feature SQLite cache table on first access and reads from it
on all subsequent accesses. The cache is the read path for
``DatasetBag.feature_values``.

The cache tables live in the bag's existing SQLite database, using the naming
convention ``_feature_cache_{feature_table_name}``. Because bags do not mutate
after materialization, each cache table is populated exactly once — populate-on-
first-access, then always-from-cache.

See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md
§"Bag denormalization cache — feature read path".
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from sqlalchemy import Column, MetaData, String, Text, Table, inspect, select
from sqlalchemy.orm import Session

from deriva_ml.core.exceptions import DerivaMLDataError, DerivaMLException
from deriva_ml.feature import FeatureRecord

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag

logger = logging.getLogger(__name__)

_CACHE_TABLE_PREFIX = "_feature_cache_"


class BagFeatureCache:
    """Read feature records from a bag's per-feature denormalization cache.

    On first access per feature, the cache projects the bag's source feature-table
    rows into a new SQLite cache table keyed by a generated ``_cache_rowid``.
    Subsequent reads query that cache table directly without re-scanning the
    source. Because bags are immutable snapshots, the cache is also immutable
    once populated.

    Cache tables are named ``_feature_cache_{feature_table_name}`` and are stored
    in the same SQLite database file as the rest of the bag data.

    Args:
        bag: The ``DatasetBag`` whose SQLite database to use.

    Example:
        >>> cache = BagFeatureCache(bag)
        >>> for rec in cache.fetch_feature_records("Image", "Glaucoma"):
        ...     print(rec.Image, rec.Glaucoma)
    """

    def __init__(self, bag: "DatasetBag") -> None:
        """Initialise the cache for the given bag.

        Args:
            bag: The DatasetBag whose engine and model will be used for
                all cache reads and writes.
        """
        self._bag = bag
        self._engine = bag.engine
        # Private metadata instance for cache tables only (avoids polluting
        # the bag's metadata with our synthetic tables).
        self._metadata = MetaData()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_feature_records(
        self,
        table: str,
        feature_name: str,
    ) -> Iterable[FeatureRecord]:
        """Yield FeatureRecord instances for (table, feature_name) from cache.

        On the first call for a given ``(table, feature_name)`` pair, the method
        populates a new SQLite cache table from the bag's source feature table.
        Subsequent calls read directly from that cache table.

        Args:
            table: Name of the target table the feature is defined on (e.g.
                ``"Image"``).
            feature_name: Name of the feature to retrieve (e.g. ``"Glaucoma"``).

        Yields:
            FeatureRecord instances with typed fields matching the feature
            definition.

        Raises:
            DerivaMLException: If the feature does not exist on ``table``.
            DerivaMLDataError: If the source feature table is missing from the
                bag (corrupt or incomplete extraction).

        Example:
            >>> cache = BagFeatureCache(bag)
            >>> records = list(cache.fetch_feature_records("Image", "Label"))
            >>> assert all(isinstance(r, FeatureRecord) for r in records)
        """
        feat = self._bag.model.lookup_feature(table, feature_name)
        record_class = feat.feature_record_class()
        cache_table_name = _CACHE_TABLE_PREFIX + feat.feature_table.name

        cache_table = self._ensure_cache_populated(feat, cache_table_name, record_class)

        field_names = set(record_class.model_fields.keys())
        with self._engine.connect() as conn:
            rows = conn.execute(select(cache_table)).mappings().all()

        for row in rows:
            filtered = {k: v for k, v in dict(row).items() if k in field_names}
            yield record_class(**filtered)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_cache_populated(
        self,
        feat,
        cache_table_name: str,
        record_class: type,
    ):
        """Return the cache table, creating and populating it if it does not exist.

        Args:
            feat: The Feature object describing the feature.
            cache_table_name: SQLite table name for the cache.
            record_class: Pydantic FeatureRecord class for this feature.

        Returns:
            The SQLAlchemy ``Table`` object for the cache table.

        Raises:
            DerivaMLDataError: If the source feature table is missing from the bag.
        """
        inspector = inspect(self._engine)
        if cache_table_name in inspector.get_table_names():
            # Cache already exists — reflect it and return.
            cache_table = Table(
                cache_table_name, self._metadata, autoload_with=self._engine
            )
            return cache_table

        # Source table must exist in the bag.
        try:
            source_table = self._bag.model.find_table(feat.feature_table.name)
        except (KeyError, DerivaMLException) as exc:
            bag_path = self._bag.model.bag_path
            raise DerivaMLDataError(
                f"Feature source table '{feat.feature_table.name}' is missing from "
                f"the bag at '{bag_path}'. Re-extract the bag to fix this."
            ) from exc

        # Build the cache table: one TEXT column per FeatureRecord field plus a PK.
        cache_columns = [
            Column("_cache_rowid", String, primary_key=True),
        ]
        for name in record_class.model_fields.keys():
            # Store everything as TEXT; Pydantic reifies types at FeatureRecord
            # construction time, so we don't need to match the source schema.
            cache_columns.append(Column(name, Text))

        cache_table = Table(cache_table_name, self._metadata, *cache_columns)
        cache_table.create(self._engine, checkfirst=True)

        # Read source rows and insert into cache.
        field_names = set(record_class.model_fields.keys())
        with Session(self._engine) as session:
            rows = session.execute(select(source_table)).mappings().all()

        with self._engine.begin() as conn:
            for i, raw in enumerate(rows):
                row_data = {k: v for k, v in dict(raw).items() if k in field_names}
                row_data["_cache_rowid"] = str(i)
                conn.execute(cache_table.insert().values(**row_data))

        logger.info(
            "BagFeatureCache: populated '%s' with %d rows", cache_table_name, len(rows)
        )
        return cache_table

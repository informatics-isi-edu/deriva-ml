import logging
from pathlib import Path
from typing import Dict, List, Any, Set, TypeVar, Type
import sqlite3
import deriva.core.ermrest_model as em
import os
import json

T = TypeVar("T")


class DatabaseModelMeta(type):
    _paths_loaded: bool = False

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Ensures the paths are only loaded once for all instances of the class.

        Args:
            *args: Positional arguments for class instantiation.
            **kwargs: Keyword arguments for class instantiation.

        Returns:
            Instance of the class with loaded paths.
        """
        obj = super().__call__(*args, **kwargs)
        if not cls._paths_loaded:
            obj._load_model()  # type: ignore
            cls._paths_loaded = True
        return obj


class DatabaseModel(metaclass=DatabaseModelMeta):
    """Handles database operations and model management for the deriva-ml package."""

    _rid_map: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def rid_lookup(cls, rid: str) -> Dict[str, Any]:
        """Looks up an Asset by RID.

        Args:
            rid: Resource identifier string.

        Returns:
            Dictionary containing asset information.

        Raises:
            ValueError: If the RID is not found in the database.
        """
        if rid not in cls._rid_map:
            raise ValueError(f"RID {rid} not found in the database")
        return cls._rid_map[rid]

    def __init__(
        self,
        model: Any,
        dataset_rid: str | None = None,
        minid: str | None = None,
        bag_path: str | Path | None = None,
        dbase_file: str | Path | None = None,
    ) -> None:
        """Initializes the DatabaseModel.

        Args:
            model: The model instance.
            dataset_rid: Dataset resource identifier.
            minid: Minimal identifier.
            bag_path: Path to the bag.
            dbase_file: Path to the database file.
        """
        self.bag_path: Path | None = Path(bag_path) if bag_path else None
        self.minid: str | None = minid
        self.dataset_rid: str | None = dataset_rid
        self.dbase_file: Path | None = Path(dbase_file) if dbase_file else None
        self.dbase: sqlite3.Connection | None = None
        self._logger: logging.Logger = logging.getLogger(__name__)
        self.ml_schema: str = model.ml_schema
        self.dataset_table: str = f"{self.ml_schema}.dataset"
        self.bag_rids: Set[str] = set()

    def _load_model(self) -> None:
        """Loads the model from the database if the database file exists."""
        if self.dbase_file and os.path.exists(self.dbase_file):
            self._load_sqlite()

    def _load_sqlite(self) -> None:
        """Loads data from SQLite database and populates the RID map."""
        if not self.dbase_file:
            return

        self.dbase = sqlite3.connect(str(self.dbase_file))
        self.dbase.row_factory = sqlite3.Row

        tables: List[str] = self.list_tables()
        for table in tables:
            rows: List[Dict[str, Any]] = self.get_table_as_dict(table)
            for row in rows:
                if "RID" in row:
                    rid: str = row["RID"]
                    self._rid_map[rid] = row
                    if table == self.dataset_table:
                        self.bag_rids.add(rid)

    def _localize_asset_table(
        self, table_name: str, schema: str, table: em.Table
    ) -> None:
        """Localizes an asset table by processing its JSON metadata files.

        Args:
            table_name: Name of the table.
            schema: Schema name.
            table: Table instance from ermrest_model.
        """
        if not self._is_asset(table):
            return

        local_path: Path = (
            Path(self.bag_path) / schema / table_name if self.bag_path else Path()
        )
        if not local_path.exists():
            return

        for filename in os.listdir(local_path):
            if filename.endswith(".json"):
                with open(local_path / filename) as f:
                    asset_metadata = json.load(f)
                    self._localize_asset(asset_metadata)

    def _is_asset(self, table: em.Table) -> bool:
        """Checks if a table is an asset table.

        Args:
            table: Table instance to check.

        Returns:
            True if the table is an asset table, False otherwise.
        """
        asset_columns: List[str] = [
            col.name
            for col in table.columns
            if col.type.typename == "text"
            and any(
                "asset" in tag.lower()
                for tag in getattr(col, "annotations", {}).get("tag", [])
            )
        ]
        return bool(asset_columns)

    def _localize_asset(self, asset_metadata: Dict[str, Any]) -> None:
        """Localizes an asset by updating the RID map.

        Args:
            asset_metadata: Dictionary containing asset metadata.
        """
        if "RID" not in asset_metadata:
            return

        rid: str = asset_metadata["RID"]
        if rid in self._rid_map:
            self._rid_map[rid].update(asset_metadata)
        else:
            self._rid_map[rid] = asset_metadata

    def list_tables(self) -> List[str]:
        """Lists all tables in the database.

        Returns:
            List of table names.
        """
        if not self.dbase:
            return []

        cursor: sqlite3.Cursor = self.dbase.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]

    def get_dataset(self, rid: str) -> Dict[str, Any]:
        """Retrieves dataset information by RID.

        Args:
            rid: Resource identifier.

        Returns:
            Dictionary containing dataset information.

        Raises:
            ValueError: If the RID is not found.
        """
        return self.rid_lookup(rid)

    def dataset_version(self, rid: str) -> str | None:
        """Gets dataset version.

        Args:
            rid: Resource identifier.

        Returns:
            Version string if found, None otherwise.
        """
        try:
            dataset = self.get_dataset(rid)
            return dataset.get("Version")
        except ValueError:
            return None

    def find_datasets(
        self, name: str | None = None, version: str | None = None
    ) -> List[Dict[str, Any]]:
        """Finds datasets matching the given criteria.

        Args:
            name: Dataset name to search for.
            version: Dataset version to search for.

        Returns:
            List of matching datasets.
        """
        datasets: List[Dict[str, Any]] = []
        for rid in self.bag_rids:
            dataset = self.get_dataset(rid)
            if name and dataset.get("Name") != name:
                continue
            if version and dataset.get("Version") != version:
                continue
            datasets.append(dataset)
        return datasets

    def get_table_as_dict(self, table_name: str) -> List[Dict[str, Any]]:
        """Gets table contents as list of dictionaries.

        Args:
            table_name: Name of the table.

        Returns:
            List of row dictionaries.
        """
        if not self.dbase:
            return []

        cursor: sqlite3.Cursor = self.dbase.cursor()
        cursor.execute(f'SELECT * FROM "{table_name}"')
        return [dict(row) for row in cursor.fetchall()]

    def normalize_table_name(self, table_name: str) -> str:
        """Normalizes a table name by ensuring proper schema prefix.

        Args:
            table_name: Table name to normalize.

        Returns:
            Normalized table name.

        Raises:
            ValueError: If the table name format is invalid.
        """
        parts: List[str] = table_name.split(".")
        if len(parts) == 1:
            return f"{self.ml_schema}.{parts[0]}"
        elif len(parts) == 2:
            return table_name
        else:
            raise ValueError(f"Invalid table name: {table_name}")

    def delete_database(self) -> None:
        """Deletes the database file and closes the connection."""
        if self.dbase:
            self.dbase.close()
            self.dbase = None
        if self.dbase_file and os.path.exists(self.dbase_file):
            os.remove(self.dbase_file)

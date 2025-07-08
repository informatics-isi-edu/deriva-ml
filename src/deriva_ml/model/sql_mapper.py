from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from deriva_ml.model.database import DatabaseModel

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class SQLMapper:
    def __init__(self, database: "DatabaseModel", table: str) -> None:
        table_name = database.normalize_table_name(table)
        schema, table = table_name.split(":")

        with database.dbase as dbase:
            self.col_names = [c[1] for c in dbase.execute(f'PRAGMA table_info("{table_name}")').fetchall()]

        self.boolean_columns = [
            self.col_names.index(c.name)
            for c in database.model.schemas[schema].tables[table].columns
            if c.type.typename == "boolean"
        ]
        self.time_columns = [
            self.col_names.index(c.name)
            for c in database.model.schemas[schema].tables[table].columns
            if c.type.typename in ["ermrest_rct", "ermrest_rmt"]
        ]

    def _map_value(self, idx: int, v: Any) -> Any:
        """
        Return a new value based on `data` where, for each index in `idxs`,
        """
        tf_map = {"t": True, "f": False}
        if idx in self.boolean_columns:
            return tf_map.get(v, v)
        if idx in self.time_columns:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f+00").replace(tzinfo=timezone.utc).isoformat()
        return v

    def transform_tuple(self, data: Sequence[Any]) -> Any:
        return dict(zip(self.col_names, tuple(self._map_value(i, v) for i, v in enumerate(data))))

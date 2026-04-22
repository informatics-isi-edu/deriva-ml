"""Compute structural diffs between two ERMrest /schema payloads.

This module is cache-agnostic. It takes two plain dicts (cached,
live), walks them deterministically, and emits a frozen
:class:`SchemaDiff` Pydantic model.

V1 dimensions: schemas (add/remove), tables (add/remove), columns
(add/remove + type change), foreign keys (add/remove). Out of scope
for V1: non-FK keys, annotations, ACLs, column nullability and
defaults, comments.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AddedTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str


class RemovedTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str


class AddedColumn(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    column: str
    type: str


class RemovedColumn(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    column: str


class ColumnTypeChange(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    column: str
    cached_type: str
    live_type: str


class AddedForeignKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    columns: list[str]
    referenced_schema: str
    referenced_table: str
    referenced_columns: list[str]


class RemovedForeignKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    columns: list[str]
    referenced_schema: str
    referenced_table: str
    referenced_columns: list[str]


class SchemaDiff(BaseModel):
    """Structured diff between a cached schema and a live schema.

    All list fields are deterministically ordered. An empty
    :class:`SchemaDiff` means the two payloads are structurally
    equivalent under the V1 dimensions.
    """

    model_config = ConfigDict(frozen=True)

    added_schemas: list[str]
    removed_schemas: list[str]
    added_tables: list[AddedTable]
    removed_tables: list[RemovedTable]
    added_columns: list[AddedColumn]
    removed_columns: list[RemovedColumn]
    column_type_changes: list[ColumnTypeChange]
    added_fkeys: list[AddedForeignKey]
    removed_fkeys: list[RemovedForeignKey]

    def is_empty(self) -> bool:
        """True iff no differences were found."""
        return not (
            self.added_schemas
            or self.removed_schemas
            or self.added_tables
            or self.removed_tables
            or self.added_columns
            or self.removed_columns
            or self.column_type_changes
            or self.added_fkeys
            or self.removed_fkeys
        )

    def render(self) -> str:
        """Return a human-readable multi-line summary.

        Empty diff returns the empty string. Useful for log messages
        and paste-into-ticket output.
        """
        if self.is_empty():
            return ""
        lines: list[str] = []
        for s in self.added_schemas:
            lines.append(f"+ schema {s}")
        for s in self.removed_schemas:
            lines.append(f"- schema {s}")
        for t in self.added_tables:
            lines.append(f"+ table {t.schema}.{t.table}")
        for t in self.removed_tables:
            lines.append(f"- table {t.schema}.{t.table}")
        for c in self.added_columns:
            lines.append(f"+ column {c.schema}.{c.table}.{c.column} ({c.type})")
        for c in self.removed_columns:
            lines.append(f"- column {c.schema}.{c.table}.{c.column}")
        for c in self.column_type_changes:
            lines.append(
                f"~ column {c.schema}.{c.table}.{c.column}: "
                f"{c.cached_type} \u2192 {c.live_type}"
            )
        for fk in self.added_fkeys:
            lines.append(
                f"+ fkey {fk.schema}.{fk.table}({','.join(fk.columns)}) "
                f"\u2192 {fk.referenced_schema}.{fk.referenced_table}"
                f"({','.join(fk.referenced_columns)})"
            )
        for fk in self.removed_fkeys:
            lines.append(
                f"- fkey {fk.schema}.{fk.table}({','.join(fk.columns)}) "
                f"\u2192 {fk.referenced_schema}.{fk.referenced_table}"
                f"({','.join(fk.referenced_columns)})"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


# --- walker ---------------------------------------------------------------


def _schemas(payload: dict) -> dict:
    return payload.get("schemas", {})


def _tables(schema_payload: dict) -> dict:
    return schema_payload.get("tables", {})


def _col_map(table_payload: dict) -> dict[str, str]:
    """Return {column_name: typename} for a table payload."""
    out: dict[str, str] = {}
    for c in table_payload.get("column_definitions", []):
        out[c["name"]] = c.get("type", {}).get("typename", "")
    return out


def _fkey_key(fk: dict) -> tuple:
    """Stable comparison key for a foreign-key definition.

    (fk columns sorted, ref schema, ref table, ref columns sorted).
    """
    fk_cols = sorted(c["column_name"] for c in fk.get("foreign_key_columns", []))
    ref_cols_raw = fk.get("referenced_columns", [])
    ref_cols = sorted(c["column_name"] for c in ref_cols_raw)
    ref_schema = ref_cols_raw[0]["schema_name"] if ref_cols_raw else ""
    ref_table = ref_cols_raw[0]["table_name"] if ref_cols_raw else ""
    return (tuple(fk_cols), ref_schema, ref_table, tuple(ref_cols))


def _fkey_detail(fk: dict) -> tuple[list[str], str, str, list[str]]:
    fk_cols = sorted(c["column_name"] for c in fk.get("foreign_key_columns", []))
    ref_cols_raw = fk.get("referenced_columns", [])
    ref_cols = sorted(c["column_name"] for c in ref_cols_raw)
    ref_schema = ref_cols_raw[0]["schema_name"] if ref_cols_raw else ""
    ref_table = ref_cols_raw[0]["table_name"] if ref_cols_raw else ""
    return fk_cols, ref_schema, ref_table, ref_cols


def compute_diff(cached: dict, live: dict) -> SchemaDiff:
    """Compare two ERMrest ``/schema`` payloads.

    Args:
        cached: Payload stored in the local schema cache.
        live: Payload fetched from the live catalog.

    Returns:
        A :class:`SchemaDiff`. Empty iff the two payloads are
        structurally equivalent under V1 dimensions.
    """
    cached_schemas = _schemas(cached)
    live_schemas = _schemas(live)

    cached_names = set(cached_schemas)
    live_names = set(live_schemas)
    added_schemas = sorted(live_names - cached_names)
    removed_schemas = sorted(cached_names - live_names)

    added_tables: list[AddedTable] = []
    removed_tables: list[RemovedTable] = []
    added_columns: list[AddedColumn] = []
    removed_columns: list[RemovedColumn] = []
    column_type_changes: list[ColumnTypeChange] = []
    added_fkeys: list[AddedForeignKey] = []
    removed_fkeys: list[RemovedForeignKey] = []

    for schema_name in sorted(cached_names & live_names):
        cached_tables = _tables(cached_schemas[schema_name])
        live_tables = _tables(live_schemas[schema_name])

        for t_name in sorted(set(live_tables) - set(cached_tables)):
            added_tables.append(AddedTable(schema=schema_name, table=t_name))
        for t_name in sorted(set(cached_tables) - set(live_tables)):
            removed_tables.append(RemovedTable(schema=schema_name, table=t_name))

        for t_name in sorted(set(cached_tables) & set(live_tables)):
            cached_cols = _col_map(cached_tables[t_name])
            live_cols = _col_map(live_tables[t_name])
            for col in sorted(set(live_cols) - set(cached_cols)):
                added_columns.append(
                    AddedColumn(
                        schema=schema_name, table=t_name,
                        column=col, type=live_cols[col],
                    )
                )
            for col in sorted(set(cached_cols) - set(live_cols)):
                removed_columns.append(
                    RemovedColumn(schema=schema_name, table=t_name, column=col)
                )
            for col in sorted(set(cached_cols) & set(live_cols)):
                if cached_cols[col] != live_cols[col]:
                    column_type_changes.append(
                        ColumnTypeChange(
                            schema=schema_name, table=t_name, column=col,
                            cached_type=cached_cols[col],
                            live_type=live_cols[col],
                        )
                    )

            # Foreign keys
            cached_fks = cached_tables[t_name].get("foreign_keys", []) or []
            live_fks = live_tables[t_name].get("foreign_keys", []) or []
            cached_keyed = {_fkey_key(fk): fk for fk in cached_fks}
            live_keyed = {_fkey_key(fk): fk for fk in live_fks}
            for k in sorted(set(live_keyed) - set(cached_keyed)):
                cols, rs, rt, rcs = _fkey_detail(live_keyed[k])
                added_fkeys.append(AddedForeignKey(
                    schema=schema_name, table=t_name,
                    columns=cols, referenced_schema=rs,
                    referenced_table=rt, referenced_columns=rcs,
                ))
            for k in sorted(set(cached_keyed) - set(live_keyed)):
                cols, rs, rt, rcs = _fkey_detail(cached_keyed[k])
                removed_fkeys.append(RemovedForeignKey(
                    schema=schema_name, table=t_name,
                    columns=cols, referenced_schema=rs,
                    referenced_table=rt, referenced_columns=rcs,
                ))

    return SchemaDiff(
        added_schemas=added_schemas,
        removed_schemas=removed_schemas,
        added_tables=added_tables,
        removed_tables=removed_tables,
        added_columns=added_columns,
        removed_columns=removed_columns,
        column_type_changes=column_type_changes,
        added_fkeys=added_fkeys,
        removed_fkeys=removed_fkeys,
    )

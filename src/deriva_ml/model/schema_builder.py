"""Create SQLAlchemy ORM from Deriva catalog model.

This module provides the SchemaBuilder class which creates a SQLAlchemy ORM
from a Deriva Model object. This is Phase 1 of the two-phase pattern:

1. Phase 1 (SchemaBuilder): Create ORM structure without data
2. Phase 2 (DataLoader): Fill database from a data source

The Model object can come from either:
- A live catalog: catalog.getCatalogModel()
- A schema.json file: Model.fromfile("file-system", schema_file)

Example:
    # From catalog
    model = catalog.getCatalogModel()
    builder = SchemaBuilder(model, schemas=['domain', 'deriva-ml'])
    orm = builder.build()

    # From file
    model = Model.fromfile("file-system", "schema.json")
    builder = SchemaBuilder(model, schemas=['domain', 'deriva-ml'])
    orm = builder.build()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, Type

from dateutil import parser
from deriva.core.ermrest_model import Column as DerivaColumn
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import Table as DerivaTable
from deriva.core.ermrest_model import Type as DerivaType
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    create_engine,
    event,
    inspect,
    select,
)
from sqlalchemy import Column as SQLColumn
from sqlalchemy import ForeignKeyConstraint as SQLForeignKeyConstraint
from sqlalchemy import Table as SQLTable
from sqlalchemy import UniqueConstraint as SQLUniqueConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.ext.automap import AutomapBase, automap_base
from sqlalchemy.orm import backref, foreign, relationship
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import TypeDecorator

logger = logging.getLogger(__name__)


# =============================================================================
# Type converters for loading CSV string data into SQLite with proper types
# =============================================================================

class ERMRestBoolean(TypeDecorator):
    """Convert ERMrest boolean strings to Python bool."""
    impl = Boolean
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> bool | None:
        if value in ("Y", "y", 1, True, "t", "T"):
            return True
        elif value in ("N", "n", 0, False, "f", "F"):
            return False
        elif value is None or value == "":
            return None
        raise ValueError(f"Invalid boolean value: {value!r}")


class StringToFloat(TypeDecorator):
    """Convert string to float, handling empty strings."""
    impl = Float
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> float | None:
        if value == "" or value is None:
            return None
        return float(value)


class StringToInteger(TypeDecorator):
    """Convert string to integer, handling empty strings."""
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> int | None:
        if value == "" or value is None:
            return None
        return int(value)


class StringToDateTime(TypeDecorator):
    """Convert string to datetime, handling empty strings."""
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value == "" or value is None:
            return None
        return parser.parse(value)


class StringToDate(TypeDecorator):
    """Convert string to date, handling empty strings."""
    impl = Date
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value == "" or value is None:
            return None
        return parser.parse(value).date()


# =============================================================================
# SchemaORM - Container for SQLAlchemy ORM components
# =============================================================================

class SchemaORM:
    """Container for SQLAlchemy ORM components.

    Provides access to the ORM structure and utility methods for
    table/class lookup. This is the result of Phase 1 (SchemaBuilder).

    Attributes:
        engine: SQLAlchemy Engine for database connections.
        metadata: SQLAlchemy MetaData with table definitions.
        Base: SQLAlchemy automap base for ORM classes.
        model: ERMrest Model the ORM was built from.
        schemas: List of schema names included.
        use_schemas: Whether schema prefixes are used (False for in-memory).
    """

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        Base: AutomapBase,
        model: Model,
        schemas: list[str],
        class_prefix: str,
        use_schemas: bool = True,
    ):
        """Initialize SchemaORM container.

        Args:
            engine: SQLAlchemy Engine.
            metadata: SQLAlchemy MetaData with tables.
            Base: Automap base with ORM classes.
            model: Source ERMrest Model.
            schemas: Schemas that were included.
            class_prefix: Prefix used for ORM class names.
            use_schemas: Whether schema prefixes are used (False for in-memory).
        """
        self.engine = engine
        self.metadata = metadata
        self.Base = Base
        self.model = model
        self.schemas = schemas
        self._class_prefix = class_prefix
        self._use_schemas = use_schemas
        self._disposed = False

    def list_tables(self) -> list[str]:
        """List all tables in the database.

        Returns:
            List of fully-qualified table names (schema.table), sorted.
        """
        tables = list(self.metadata.tables.keys())
        tables.sort()
        return tables

    def find_table(self, table_name: str) -> SQLTable:
        """Find a table by name.

        Handles both schema.table format and schema_table format (for in-memory databases).

        Args:
            table_name: Table name, with or without schema prefix.
                Can be "schema.table", "schema_table", or just "table".

        Returns:
            SQLAlchemy Table object.

        Raises:
            KeyError: If table not found.
        """
        # Try exact match first
        if table_name in self.metadata.tables:
            return self.metadata.tables[table_name]

        # Try converting schema.table to schema_table format (for in-memory)
        if "." in table_name and not self._use_schemas:
            converted_name = table_name.replace(".", "_").replace("-", "_")
            if converted_name in self.metadata.tables:
                return self.metadata.tables[converted_name]

        # Try matching just the table name part
        for full_name, table in self.metadata.tables.items():
            # Handle . separator (file-based)
            if "." in full_name and full_name.split(".")[-1] == table_name:
                return table
            # Handle _ separator (in-memory) - match suffix after first _
            if "_" in full_name and "." not in full_name:
                # Check if table_name matches the part after schema prefix
                parts = full_name.split("_", 1)
                if len(parts) > 1 and parts[1] == table_name:
                    return table
                # Also check if it ends with the table name
                if full_name.endswith(f"_{table_name}"):
                    return table

        raise KeyError(f"Table {table_name} not found")

    def get_orm_class(self, table_name: str) -> Any | None:
        """Get the ORM class for a table by name.

        Args:
            table_name: Table name, with or without schema prefix.

        Returns:
            SQLAlchemy ORM class for the table.

        Raises:
            KeyError: If table not found.
        """
        sql_table = self.find_table(table_name)
        return self.get_orm_class_for_table(sql_table)

    def get_orm_class_for_table(self, table: SQLTable | DerivaTable | str) -> Any | None:
        """Get the ORM class for a table.

        Args:
            table: SQLAlchemy Table, Deriva Table, or table name.

        Returns:
            SQLAlchemy ORM class, or None if not found.
        """
        if isinstance(table, DerivaTable):
            # Try schema.table format first (file-based), then schema_table (in-memory)
            table_key = f"{table.schema.name}.{table.name}"
            table = self.metadata.tables.get(table_key)
            if table is None and not self._use_schemas:
                # Try underscore format for in-memory databases
                table_key = f"{table.schema.name}_{table.name}".replace("-", "_")
                table = self.metadata.tables.get(table_key)
        if isinstance(table, str):
            table = self.find_table(table)
        if table is None:
            return None

        for mapper in self.Base.registry.mappers:
            if mapper.persist_selectable is table or table in mapper.tables:
                return mapper.class_
        return None

    def get_table_contents(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Retrieve all rows from a table as dictionaries.

        Args:
            table: Table name (with or without schema prefix).

        Yields:
            Dictionary for each row with column names as keys.
        """
        sql_table = self.find_table(table)
        with self.engine.connect() as conn:
            result = conn.execute(select(sql_table))
            for row in result.mappings():
                yield dict(row)

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

        An association table links two or more tables through foreign keys,
        with a composite unique key covering those foreign keys.

        Args:
            table_class: SQLAlchemy ORM class to check.
            min_arity: Minimum number of foreign keys (default 2).
            max_arity: Maximum number of foreign keys (default 2).
            unqualified: If True, reject associations with extra key columns.
            pure: If True, reject associations with extra non-key columns.
            no_overlap: If True, reject associations with shared FK columns.
            return_fkeys: If True, return the foreign keys instead of arity.

        Returns:
            If return_fkeys=False: Integer arity if association, False otherwise.
            If return_fkeys=True: Set of foreign keys if association, False otherwise.
        """
        if min_arity < 2:
            raise ValueError("An association cannot have arity < 2")
        if max_arity is not None and max_arity < min_arity:
            raise ValueError("max_arity cannot be less than min_arity")

        mapper = inspect(table_class).mapper
        system_cols = {"RID", "RCT", "RMT", "RCB", "RMB"}

        non_sys_cols = {
            col.name for col in mapper.columns if col.name not in system_cols
        }

        unique_columns = [
            {c.name for c in constraint.columns}
            for constraint in inspect(table_class).local_table.constraints
            if isinstance(constraint, SQLUniqueConstraint)
        ]

        non_sys_key_colsets = {
            frozenset(uc)
            for uc in unique_columns
            if uc.issubset(non_sys_cols) and len(uc) > 1
        }

        if not non_sys_key_colsets:
            return False

        # Choose longest compound key
        row_key = sorted(non_sys_key_colsets, key=lambda s: len(s), reverse=True)[0]
        foreign_keys = list(inspect(table_class).relationships.values())

        covered_fkeys = {
            fkey for fkey in foreign_keys
            if {c.name for c in fkey.local_columns}.issubset(row_key)
        }
        covered_fkey_cols = set()

        if len(covered_fkeys) < min_arity:
            return False
        if max_arity is not None and len(covered_fkeys) > max_arity:
            return False

        for fkey in covered_fkeys:
            fkcols = {c.name for c in fkey.local_columns}
            if no_overlap and fkcols.intersection(covered_fkey_cols):
                return False
            covered_fkey_cols.update(fkcols)

        if unqualified and row_key.difference(covered_fkey_cols):
            return False

        if pure and non_sys_cols.difference(row_key):
            return False

        return covered_fkeys if return_fkeys else len(covered_fkeys)

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
        for _, left_rel in inspect(left_cls).relationships.items():
            mid_cls = left_rel.mapper.class_
            is_assoc = self.is_association_table(mid_cls, return_fkeys=True)

            if not is_assoc:
                continue

            assoc_local_columns_left = list(is_assoc)[0].local_columns
            assoc_local_columns_right = list(is_assoc)[1].local_columns

            found_left = found_right = False

            for r in inspect(left_cls).relationships.values():
                remote_side = list(r.remote_side)[0]
                if remote_side in assoc_local_columns_left:
                    found_left = r
                if remote_side in assoc_local_columns_right:
                    found_left = r
                    # Swap if backwards
                    assoc_local_columns_left, assoc_local_columns_right = (
                        assoc_local_columns_right,
                        assoc_local_columns_left,
                    )

            for r in inspect(right_cls).relationships.values():
                remote_side = list(r.remote_side)[0]
                if remote_side in assoc_local_columns_right:
                    found_right = r

            if found_left and found_right:
                return mid_cls, found_left.class_attribute, found_right.class_attribute

        return None

    def dispose(self) -> None:
        """Dispose of SQLAlchemy resources.

        Call this when done with the database to properly clean up connections.
        After calling dispose(), the instance should not be used further.
        """
        if self._disposed:
            return

        if hasattr(self, "Base") and self.Base is not None:
            self.Base.registry.dispose()
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.dispose()

        self._disposed = True

    def __del__(self) -> None:
        """Cleanup resources when garbage collected."""
        self.dispose()

    def __enter__(self) -> "SchemaORM":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - dispose resources."""
        self.dispose()
        return False


# =============================================================================
# SchemaBuilder - Creates ORM from Deriva Model
# =============================================================================

class SchemaBuilder:
    """Creates SQLAlchemy ORM from a Deriva catalog model.

    Phase 1 of the two-phase database creation pattern. This class handles
    only schema/ORM creation - no data loading.

    The Model can come from either a live catalog or a schema.json file:
    - From catalog: model = catalog.getCatalogModel()
    - From file: model = Model.fromfile("file-system", "path/to/schema.json")

    Example:
        # Create ORM from catalog model
        model = catalog.getCatalogModel()
        builder = SchemaBuilder(model, schemas=['domain', 'deriva-ml'])
        orm = builder.build()

        # Create ORM from schema file
        model = Model.fromfile("file-system", "schema.json")
        builder = SchemaBuilder(model, schemas=['domain'], database_path="local.db")
        orm = builder.build()

        # Use the ORM
        ImageClass = orm.get_orm_class("Image")
        with Session(orm.engine) as session:
            images = session.query(ImageClass).all()

        # Clean up
        orm.dispose()
    """

    # Type mapping from ERMrest to SQLAlchemy
    _TYPE_MAP = {
        "boolean": ERMRestBoolean,
        "date": StringToDate,
        "float4": StringToFloat,
        "float8": StringToFloat,
        "int2": StringToInteger,
        "int4": StringToInteger,
        "int8": StringToInteger,
        "json": JSON,
        "jsonb": JSON,
        "timestamptz": StringToDateTime,
        "timestamp": StringToDateTime,
    }

    def __init__(
        self,
        model: Model,
        schemas: list[str],
        database_path: Path | str = ":memory:",
    ):
        """Initialize the schema builder.

        Args:
            model: ERMrest Model object (from catalog or schema.json file).
            schemas: List of schema names to include in the ORM.
            database_path: Path to SQLite database file. Use ":memory:" for
                in-memory database (default). If a Path or string is provided,
                separate .db files will be created for each schema.
        """
        self.model = model
        self.schemas = schemas
        self.database_path = Path(database_path) if database_path != ":memory:" else database_path

        # Will be set during build()
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self.Base: AutomapBase | None = None
        self._class_prefix: str = ""

    @staticmethod
    def _sql_type(deriva_type: DerivaType) -> TypeEngine:
        """Map ERMrest type to SQLAlchemy type with CSV string conversion.

        Args:
            deriva_type: ERMrest type object.

        Returns:
            SQLAlchemy type class.
        """
        return SchemaBuilder._TYPE_MAP.get(deriva_type.typename, String)

    def _is_key_column(self, column: DerivaColumn, table: DerivaTable) -> bool:
        """Check if column is the primary key (RID)."""
        return column in [key.unique_columns[0] for key in table.keys] and column.name == "RID"

    def build(self) -> SchemaORM:
        """Build the SQLAlchemy ORM structure.

        Creates SQLite tables from the ERMrest schema and generates
        ORM classes via SQLAlchemy automap.

        Returns:
            SchemaORM object containing engine, metadata, Base, and utilities.

        Note:
            In-memory databases (database_path=":memory:") do not support
            SQLite schema attachments, so all tables will be created in a
            single database without schema prefixes in table names.
        """
        # Create unique prefix for ORM class names
        self._class_prefix = f"_{id(self)}_"

        # Determine if we're using in-memory or file-based database
        self._use_schemas = self.database_path != ":memory:"

        # Create engine
        if self.database_path == ":memory:":
            self.engine = create_engine("sqlite:///:memory:", future=True)
        else:
            # Ensure the database path exists
            if isinstance(self.database_path, Path):
                if self.database_path.suffix == ".db":
                    # Single file path
                    self.database_path.parent.mkdir(parents=True, exist_ok=True)
                    main_db = self.database_path
                else:
                    # Directory path
                    self.database_path.mkdir(parents=True, exist_ok=True)
                    main_db = self.database_path / "main.db"
            else:
                main_db = Path(self.database_path)
                main_db.parent.mkdir(parents=True, exist_ok=True)

            self.engine = create_engine(f"sqlite:///{main_db.resolve()}", future=True)

            # Attach schema-specific databases
            event.listen(self.engine, "connect", self._attach_schemas)

        self.metadata = MetaData()
        self.Base = automap_base(metadata=self.metadata)

        # Build the schema
        self._create_tables()

        logger.info(
            "Built ORM for schemas %s with %d tables",
            self.schemas,
            len(self.metadata.tables),
        )

        return SchemaORM(
            engine=self.engine,
            metadata=self.metadata,
            Base=self.Base,
            model=self.model,
            schemas=self.schemas,
            class_prefix=self._class_prefix,
            use_schemas=self._use_schemas,
        )

    def _attach_schemas(self, dbapi_conn, _conn_record):
        """Attach schema-specific SQLite databases."""
        cur = dbapi_conn.cursor()
        db_dir = self.database_path if self.database_path.is_dir() else self.database_path.parent
        for schema in self.schemas:
            schema_file = (db_dir / f"{schema}.db").resolve()
            cur.execute(f"ATTACH DATABASE '{schema_file}' AS '{schema}'")
        cur.close()

    def _create_tables(self) -> None:
        """Create SQLite tables from the ERMrest schema."""

        def col(model, name: str):
            """Get column from ORM class, handling both attribute and table column access."""
            try:
                return getattr(model, name).property.columns[0]
            except AttributeError:
                return model.__table__.c[name]

        def guess_attr_name(col_name: str) -> str:
            """Generate relationship attribute name from column name."""
            return col_name[:-3] if col_name.lower().endswith("_id") else col_name

        def make_table_name(schema_name: str, table_name: str) -> str:
            """Generate table name, including schema prefix if using schemas."""
            if self._use_schemas:
                return f"{schema_name}.{table_name}"
            else:
                # For in-memory, use underscore separator to avoid conflicts
                return f"{schema_name}_{table_name}"

        database_tables: list[SQLTable] = []

        for schema_name in self.schemas:
            if schema_name not in self.model.schemas:
                logger.warning(f"Schema {schema_name} not found in model")
                continue

            for table in self.model.schemas[schema_name].tables.values():
                database_columns: list[SQLColumn] = []

                for c in table.columns:
                    database_column = SQLColumn(
                        name=c.name,
                        type_=self._sql_type(c.type),
                        comment=c.comment,
                        default=c.default,
                        primary_key=self._is_key_column(c, table),
                        nullable=c.nullok,
                    )
                    database_columns.append(database_column)

                # Use schema prefix only for file-based databases
                if self._use_schemas:
                    database_table = SQLTable(
                        table.name, self.metadata, *database_columns, schema=schema_name
                    )
                else:
                    # For in-memory, embed schema in table name
                    full_name = f"{schema_name}_{table.name}".replace("-", "_")
                    database_table = SQLTable(
                        full_name, self.metadata, *database_columns
                    )

                # Add unique constraints
                for key in table.keys:
                    key_columns = [c.name for c in key.unique_columns]
                    database_table.append_constraint(
                        SQLUniqueConstraint(*key_columns, name=key.name[1])
                    )

                # Add foreign key constraints (within same schema only for now)
                for fk in table.foreign_keys:
                    if fk.pk_table.schema.name not in self.schemas:
                        continue
                    if fk.pk_table.schema.name != schema_name:
                        continue

                    # Build reference column names
                    if self._use_schemas:
                        refcols = [
                            f"{schema_name}.{c.table.name}.{c.name}"
                            for c in fk.referenced_columns
                        ]
                    else:
                        # For in-memory, use the embedded schema name
                        ref_table_name = f"{schema_name}_{fk.pk_table.name}".replace("-", "_")
                        refcols = [
                            f"{ref_table_name}.{c.name}"
                            for c in fk.referenced_columns
                        ]

                    database_table.append_constraint(
                        SQLForeignKeyConstraint(
                            columns=[f"{c.name}" for c in fk.foreign_key_columns],
                            refcolumns=refcols,
                            name=fk.name[1],
                            comment=fk.comment,
                        )
                    )

                database_tables.append(database_table)

        # Create all tables
        with self.engine.begin() as conn:
            self.metadata.create_all(conn, tables=database_tables, checkfirst=True)

        # Configure ORM class naming
        def name_for_scalar_relationship(_base, local_cls, referred_cls, constraint):
            cols = list(constraint.columns) if constraint is not None else []
            if len(cols) == 1:
                name = cols[0].key
                if name in {c.key for c in local_cls.__table__.columns}:
                    name += "_rel"
                return name
            return constraint.name or referred_cls.__name__.lower()

        def name_for_collection_relationship(_base, local_cls, referred_cls, constraint):
            backref_name = constraint.name.replace("_fkey", "_collection")
            return backref_name or (referred_cls.__name__.lower() + "_collection")

        def classname_for_table(_base, tablename, table):
            return self._class_prefix + tablename.replace(".", "_").replace("-", "_")

        # Build ORM mappings
        self.Base.prepare(
            self.engine,
            name_for_scalar_relationship=name_for_scalar_relationship,
            name_for_collection_relationship=name_for_collection_relationship,
            classname_for_table=classname_for_table,
            reflect=True,
        )

        # Add cross-schema relationships
        for schema_name in self.schemas:
            if schema_name not in self.model.schemas:
                continue

            for table in self.model.schemas[schema_name].tables.values():
                for fk in table.foreign_keys:
                    if fk.pk_table.schema.name not in self.schemas:
                        continue
                    if fk.pk_table.schema.name == schema_name:
                        continue

                    table_name = make_table_name(schema_name, table.name)
                    table_class = self._get_orm_class_by_name(table_name)
                    foreign_key_column_name = fk.foreign_key_columns[0].name
                    foreign_key_column = col(table_class, foreign_key_column_name)

                    referenced_table_name = make_table_name(fk.pk_table.schema.name, fk.pk_table.name)
                    referenced_class = self._get_orm_class_by_name(referenced_table_name)
                    referenced_column = col(referenced_class, fk.referenced_columns[0].name)

                    relationship_attr = guess_attr_name(foreign_key_column_name)
                    backref_attr = fk.name[1].replace("_fkey", "_collection")

                    # Check if relationship already exists
                    existing_attr = getattr(table_class, relationship_attr, None)
                    from sqlalchemy.orm import RelationshipProperty
                    from sqlalchemy.orm.attributes import InstrumentedAttribute

                    is_relationship = isinstance(existing_attr, InstrumentedAttribute) and isinstance(
                        existing_attr.property, RelationshipProperty
                    )
                    if not is_relationship:
                        setattr(
                            table_class,
                            relationship_attr,
                            relationship(
                                referenced_class,
                                foreign_keys=[foreign_key_column],
                                primaryjoin=foreign(foreign_key_column) == referenced_column,
                                backref=backref(backref_attr, viewonly=True),
                                viewonly=True,
                            ),
                        )

        # Configure mappers
        self.Base.registry.configure()

    def _get_orm_class_by_name(self, table_name: str) -> Any | None:
        """Get ORM class by table name (internal use during build).

        Handles both schema.table format (file-based) and schema_table format (in-memory).
        """
        # Try exact match first
        if table_name in self.metadata.tables:
            sql_table = self.metadata.tables[table_name]
        else:
            # For in-memory databases, table names use underscore separator
            # Try converting schema.table to schema_table format
            if "." in table_name and not self._use_schemas:
                converted_name = table_name.replace(".", "_").replace("-", "_")
                if converted_name in self.metadata.tables:
                    sql_table = self.metadata.tables[converted_name]
                else:
                    sql_table = None
            else:
                # Try matching just the table name part
                sql_table = None
                for full_name, table in self.metadata.tables.items():
                    # Handle both . and _ separators
                    table_part = full_name.split(".")[-1] if "." in full_name else full_name.split("_", 1)[-1] if "_" in full_name else full_name
                    if table_part == table_name or full_name.endswith(f"_{table_name}"):
                        sql_table = table
                        break

        if sql_table is None:
            raise KeyError(f"Table {table_name} not found")

        for mapper in self.Base.registry.mappers:
            if mapper.persist_selectable is sql_table or sql_table in mapper.tables:
                return mapper.class_
        return None

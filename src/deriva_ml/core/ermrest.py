"""ERMrest data models for DerivaML.

This module provides Pydantic models that represent ERMrest catalog structures. These models are used
throughout DerivaML for defining and manipulating catalog elements like tables, columns, and keys.

Classes:
    FileUploadState: Tracks the state of file uploads.
    VocabularyTerm: Represents terms in controlled vocabularies.
    ColumnDefinition: Defines columns in tables.
    KeyDefinition: Defines primary and unique keys.
    ForeignKeyDefinition: Defines foreign key relationships.
    TableDefinition: Defines complete table structures.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import deriva.core.ermrest_model as em
from deriva.core.ermrest_model import builtin_types
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_serializer,
)

from .constants import RID
from .enums import BuiltinTypes, UploadState

# Pydantic warnings suppression
warnings.filterwarnings("ignore", message='Field name "schema"', category=Warning, module="pydantic")
warnings.filterwarnings(
    "ignore",
    message="fields may not start with an underscore",
    category=Warning,
    module="pydantic",
)


class FileUploadState(BaseModel):
    """Tracks the state and result of a file upload operation.

    Attributes:
        state (UploadState): Current state of the upload (success, failed, etc.).
        status (str): Detailed status message.
        result (Any): Upload result data, if any.
        rid (RID | None): Resource identifier of the uploaded file, if successful.
    """
    state: UploadState
    status: str
    result: Any

    @computed_field
    @property
    def rid(self) -> RID | None:
        return self.result and self.result["RID"]


class VocabularyTerm(BaseModel):
    """Represents a term in a controlled vocabulary.

    A vocabulary term is a standardized entry in a controlled vocabulary table. Each term has
    a primary name, optional synonyms, and identifiers for cross-referencing.

    Attributes:
        name (str): Primary name of the term.
        synonyms (list[str] | None): Alternative names for the term.
        id (str): CURIE (Compact URI) identifier.
        uri (str): Full URI for the term.
        description (str): Explanation of the term's meaning.
        rid (str): Resource identifier in the catalog.

    Example:
        >>> term = VocabularyTerm(
        ...     Name="epithelial",
        ...     Synonyms=["epithelium"],
        ...     ID="tissue:0001",
        ...     URI="http://example.org/tissue/0001",
        ...     Description="Epithelial tissue type",
        ...     RID="1-abc123"
        ... )
    """
    name: str = Field(alias="Name")
    synonyms: list[str] | None = Field(alias="Synonyms")
    id: str = Field(alias="ID")
    uri: str = Field(alias="URI")
    description: str = Field(alias="Description")
    rid: str = Field(alias="RID")

    class Config:
        extra = "ignore"


class ColumnDefinition(BaseModel):
    """Defines a column in an ERMrest table.

    Provides a Pydantic model for defining columns with their types, constraints, and metadata.
    Maps to deriva_py's Column.define functionality.

    Attributes:
        name (str): Name of the column.
        type (BuiltinTypes): ERMrest data type for the column.
        nullok (bool): Whether NULL values are allowed. Defaults to True.
        default (Any): Default value for the column.
        comment (str | None): Description of the column's purpose.
        acls (dict): Access control lists.
        acl_bindings (dict): Dynamic access control bindings.
        annotations (dict): Additional metadata annotations.

    Example:
        >>> col = ColumnDefinition(
        ...     name="score",
        ...     type=BuiltinTypes.float4,
        ...     nullok=False,
        ...     comment="Confidence score between 0 and 1"
        ... )
    """
    name: str
    type: BuiltinTypes
    nullok: bool = True
    default: Any = None
    comment: str | None = None
    acls: dict = Field(default_factory=dict)
    acl_bindings: dict = Field(default_factory=dict)
    annotations: dict = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def extract_type_name(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return BuiltinTypes(value["typename"])
        else:
            return value

    @model_serializer()
    def serialize_column_definition(self):
        return em.Column.define(
            self.name,
            builtin_types[self.type.value],
            nullok=self.nullok,
            default=self.default,
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations,
        )


class KeyDefinition(BaseModel):
    """Defines a key constraint in an ERMrest table.

    Provides a Pydantic model for defining primary keys and unique constraints.
    Maps to deriva_py's Key.define functionality.

    Attributes:
        colnames (Iterable[str]): Names of columns that form the key.
        constraint_names (Iterable[str]): Names for the key constraints.
        comment (str | None): Description of the key's purpose.
        annotations (dict): Additional metadata annotations.

    Example:
        >>> key = KeyDefinition(
        ...     colnames=["id", "version"],
        ...     constraint_names=["unique_id_version"],
        ...     comment="Unique identifier with version"
        ... )
    """
    colnames: Iterable[str]
    constraint_names: Iterable[str]
    comment: str | None = None
    annotations: dict = Field(default_factory=dict)

    @model_serializer()
    def serialize_key_definition(self):
        return em.Key.define(
            colnames=self.colnames,
            constraint_names=self.constraint_names,
            comment=self.comment,
            annotations=self.annotations,
        )


class ForeignKeyDefinition(BaseModel):
    """Defines a foreign key relationship between tables.

    Provides a Pydantic model for defining foreign key constraints with referential actions
    and metadata. Maps to deriva_py's ForeignKey.define functionality.

    Attributes:
        colnames (Iterable[str]): Names of columns in the referencing table.
        pk_sname (str): Schema name of the referenced table.
        pk_tname (str): Name of the referenced table.
        pk_colnames (Iterable[str]): Names of columns in the referenced table.
        constraint_names (Iterable[str]): Names for the foreign key constraints.
        on_update (str): Action on update of referenced row. Defaults to "NO ACTION".
        on_delete (str): Action on delete of referenced row. Defaults to "NO ACTION".
        comment (str | None): Description of the relationship.
        acls (dict): Access control lists.
        acl_bindings (dict): Dynamic access control bindings.
        annotations (dict): Additional metadata annotations.

    Example:
        >>> fk = ForeignKeyDefinition(
        ...     colnames=["dataset_id"],
        ...     pk_sname="core",
        ...     pk_tname="dataset",
        ...     pk_colnames=["id"],
        ...     on_delete="CASCADE"
        ... )
    """
    colnames: Iterable[str]
    pk_sname: str
    pk_tname: str
    pk_colnames: Iterable[str]
    constraint_names: Iterable[str] = Field(default_factory=list)
    on_update: str = "NO ACTION"
    on_delete: str = "NO ACTION"
    comment: str | None = None
    acls: dict[str, Any] = Field(default_factory=dict)
    acl_bindings: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)

    @model_serializer()
    def serialize_fk_definition(self):
        return em.ForeignKey.define(
            fk_colnames=self.colnames,
            pk_sname=self.pk_sname,
            pk_tname=self.pk_tname,
            pk_colnames=self.pk_colnames,
            on_update=self.on_update,
            on_delete=self.on_delete,
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations,
        )


class TableDefinition(BaseModel):
    """Defines a complete table structure in ERMrest.

    Provides a Pydantic model for defining tables with their columns, keys, and relationships.
    Maps to deriva_py's Table.define functionality.

    Attributes:
        name (str): Name of the table.
        column_defs (Iterable[ColumnDefinition]): Column definitions.
        key_defs (Iterable[KeyDefinition]): Key constraint definitions.
        fkey_defs (Iterable[ForeignKeyDefinition]): Foreign key relationship definitions.
        comment (str | None): Description of the table's purpose.
        acls (dict): Access control lists.
        acl_bindings (dict): Dynamic access control bindings.
        annotations (dict): Additional metadata annotations.

    Example:
        >>> table = TableDefinition(
        ...     name="experiment",
        ...     column_defs=[
        ...         ColumnDefinition(name="id", type=BuiltinTypes.text),
        ...         ColumnDefinition(name="date", type=BuiltinTypes.date)
        ...     ],
        ...     comment="Experimental data records"
        ... )
    """
    name: str
    column_defs: Iterable[ColumnDefinition]
    key_defs: Iterable[KeyDefinition] = Field(default_factory=list)
    fkey_defs: Iterable[ForeignKeyDefinition] = Field(default_factory=list)
    comment: str | None = None
    acls: dict = Field(default_factory=dict)
    acl_bindings: dict = Field(default_factory=dict)
    annotations: dict = Field(default_factory=dict)

    @model_serializer()
    def serialize_table_definition(self):
        return em.Table.define(
            tname=self.name,
            column_defs=[c.model_dump() for c in self.column_defs],
            key_defs=[k.model_dump() for k in self.key_defs],
            fkey_defs=[fk.model_dump() for fk in self.fkey_defs],
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations,
        )

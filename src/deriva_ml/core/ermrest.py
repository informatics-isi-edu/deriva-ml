"""
Pydantic models used throughout the DerivaML package.
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
    state: UploadState
    status: str
    result: Any

    @computed_field
    @property
    def rid(self) -> RID | None:
        return self.result and self.result["RID"]


class VocabularyTerm(BaseModel):
    """An entry in a vocabulary table.

    Attributes:
       name: Name of vocabulary term
       synonyms: List of alternative names for the term
       id: CURI identifier for the term
       uri: Unique URI for the term.
       description: A description of the term meaning
       rid: Resource identifier assigned to the term

    Args:

    Returns:

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
    """Pydantic model for deriva_py Column.define"""

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
    """Pydantic model for deriva_py ForeignKey.define"""

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

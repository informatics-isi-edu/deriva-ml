"""
Shared definitions that are used in different DerivaML modules.
"""


import warnings
from enum import Enum
from typing import Any, Iterable, Optional, Annotated

import deriva.core.ermrest_model as em
from deriva.core.ermrest_model import builtin_types
from pydantic import BaseModel, model_serializer, Field, computed_field

ML_SCHEMA = 'deriva-ml'

# We are going to use schema as a field name and this collides with method in pydantic base class
warnings.filterwarnings('ignore',
                        message='Field name "schema"',
                        category=Warning,
                        module='pydantic')

warnings.filterwarnings('ignore',
                        message='fields may not start with an underscore',
                        category=Warning,
                        module='pydantic')


rid_regex = r"^(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+)$"
RID = Annotated[str, Field(pattern=rid_regex)]

DerivaSystemColumns = ['RCT', 'RMT', 'RCB', 'RMB']

# For some reason, deriva-py doesn't use the proper enum class!!
class UploadState(Enum):
    success = 0
    failed = 1
    pending = 2
    running = 3
    paused = 4
    aborted = 5
    cancelled = 6
    timeout = 7

class StrEnum(str, Enum):
    pass

class FileUploadState(BaseModel):
    state: UploadState
    status: str
    result: Any

    @computed_field
    @property
    def rid(self) -> Optional[RID]:
        return self.result and self.result['RID']

class Status(StrEnum):
    """Enumeration class defining execution status.

    Attributes:
        running: Execution is currently running.
        pending: Execution is pending.
        completed: Execution has been completed successfully.
        failed: Execution has failed.

    """
    running = 'Running'
    pending = 'Pending'
    completed = 'Completed'
    failed = 'Failed'


class BuiltinTypes(Enum):
    text = builtin_types.text
    int2 = builtin_types.int2
    jsonb = builtin_types.json
    float8 = builtin_types.float8
    timestamp = builtin_types.timestamp
    int8 = builtin_types.int8
    boolean = builtin_types.boolean
    json = builtin_types.json
    float4 = builtin_types.float4
    int4 = builtin_types.int4
    timestamptz = builtin_types.timestamptz
    date = builtin_types.date
    ermrest_rid = builtin_types.ermrest_rid
    ermrest_rcb = builtin_types.ermrest_rcb
    ermrest_rmb = builtin_types.ermrest_rmb
    ermrest_rct = builtin_types.ermrest_rct
    ermrest_rmt = builtin_types.ermrest_rmt
    markdown = builtin_types.markdown
    longtext = builtin_types.longtext
    ermrest_curie = builtin_types.ermrest_curie
    ermrest_uri = builtin_types.ermrest_uri
    color_rgb_hex = builtin_types.color_rgb_hex
    serial2 = builtin_types.serial2
    serial4 = builtin_types.serial4
    serial8 = builtin_types.serial8

class MLVocab(StrEnum):
    """Names of controlled vocabulary for various types within DerivaML."""
    dataset_type = 'Dataset_Type'
    workflow_type = 'Workflow_Type'
    execution_asset_type = 'Execution_Asset_Type'
    execution_metadata_type = 'Execution_Metadata_Type'

class ExecMetadataVocab(StrEnum):
    """
    Predefined execution metatadata types.
    """
    execution_config = 'Execution_Config'
    runtime_env = 'Runtime_Env'

class ColumnDefinition(BaseModel):
    """Pydantic model for deriva_py Column.define"""

    name: str
    type: BuiltinTypes
    nullok: bool = True
    default: Any = None
    comment: str = None
    acls: dict = Field(default_factory=dict)
    acl_bindings: dict = Field(default_factory=dict)
    annotations: dict = Field(default_factory=dict)

    @model_serializer()
    def serialize_column_definition(self):
        return em.Column.define(
            cname=self.name,
            ctype=self.type.value,
            nullok=self.nullok,
            default=self.default,
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations)


class KeyDefinition(BaseModel):
    colnames: Iterable[str]
    constraint_names: Iterable[str]
    comment: Optional[str] = None
    annotations: dict = Field(default_factory=dict)

    @model_serializer()
    def serialize_key_definition(self):
        return em.Key.define(
            colnames=self.colnames,
            constraint_names=self.constraint_names,
            comment=self.comment,
            annotations=self.annotations
        )


class ForeignKeyDefinition(BaseModel):
    """Pydantic model for deriva_py ForeignKey.define"""
    colnames: Iterable[str]
    pk_sname: str
    pk_tname: str
    pk_colnames: Iterable[str]
    constraint_names: Iterable[str] = Field(default_factory=list)
    on_update: str = 'NO ACTION'
    on_delete: str = 'NO ACTION'
    comment: str = None
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
            annotations=self.annotations
        )


class TableDefinition(BaseModel):
    name: str
    column_defs: Iterable[ColumnDefinition]
    key_defs: Iterable[KeyDefinition] = Field(default_factory=list)
    fkey_defs: Iterable[ForeignKeyDefinition] = Field(default_factory=list)
    comment: str = None
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
            annotations=self.annotations)


class DerivaMLException(Exception):
    """Exception class specific to DerivaML module.

    Args:
        msg (str): Optional message for the exception.
    """

    def __init__(self, msg=''):
        super().__init__(msg)
        self._msg = msg

"""
Shared definitions that are used in different DerivaML modules.
"""

from __future__ import annotations

# Standard library imports
import json
import warnings
from datetime import date
from enum import Enum
from pathlib import Path
from socket import gethostname
from typing import Any, Iterable, Generator, Iterator, List, Annotated, NewType
from urllib.parse import urlparse

# Third-party imports
from pydantic import (
    BaseModel,
    model_serializer,
    Field,
    computed_field,
    field_validator,
    ValidationError,
)

# Deriva imports
import deriva.core.ermrest_model as em
from deriva.core.ermrest_model import builtin_types
import deriva.core.utils.hash_utils as hash_utils

# Constants
ML_SCHEMA = "deriva-ml"
DRY_RUN_RID = "0000"

# Pydantic warnings suppression
warnings.filterwarnings(
    "ignore", message='Field name "schema"', category=Warning, module="pydantic"
)
warnings.filterwarnings(
    "ignore",
    message="fields may not start with an underscore",
    category=Warning,
    module="pydantic",
)

# RID definitions
rid_part = r"(?P<rid>(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+))"
snapshot_part = r"(?:@(?P<snapshot>(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+)))?"
rid_regex = f"^{rid_part}{snapshot_part}$"
RID = NewType("RID", Annotated[str, Field(pattern=rid_regex)])

DerivaSystemColumns = ["RID", "RCT", "RMT", "RCB", "RMB"]


# For some reason, deriva-py doesn't use the proper enum class!!
class UploadState(Enum):
    """State of file upload"""

    success = 0
    failed = 1
    pending = 2
    running = 3
    paused = 4
    aborted = 5
    cancelled = 6
    timeout = 7


class BaseStrEnum(str, Enum):
    """Base class for string enums in Python 3.10"""

    pass


class FileUploadState(BaseModel):
    state: UploadState
    status: str
    result: Any

    @computed_field
    @property
    def rid(self) -> RID | None:
        return self.result and self.result["RID"]


class Status(BaseStrEnum):
    """Enumeration class defining execution status.

    Attributes:
        running: Execution is currently running.
        pending: Execution is pending.
        completed: Execution has been completed successfully.
        failed: Execution has failed.

    """

    initializing = "Initializing"
    created = "Created"
    pending = "Pending"
    running = "Running"
    aborted = "Aborted"
    completed = "Completed"
    failed = "Failed"


class BuiltinTypes(Enum):
    text = builtin_types.text.typename
    int2 = builtin_types.int2.typename
    jsonb = builtin_types.json.typename
    float8 = builtin_types.float8.typename
    timestamp = builtin_types.timestamp.typename
    int8 = builtin_types.int8.typename
    boolean = builtin_types.boolean.typename
    json = builtin_types.json.typename
    float4 = builtin_types.float4.typename
    int4 = builtin_types.int4.typename
    timestamptz = builtin_types.timestamptz.typename
    date = builtin_types.date.typename
    ermrest_rid = builtin_types.ermrest_rid.typename
    ermrest_rcb = builtin_types.ermrest_rcb.typename
    ermrest_rmb = builtin_types.ermrest_rmb.typename
    ermrest_rct = builtin_types.ermrest_rct.typename
    ermrest_rmt = builtin_types.ermrest_rmt.typename
    markdown = builtin_types.markdown.typename
    longtext = builtin_types.longtext.typename
    ermrest_curie = builtin_types.ermrest_curie.typename
    ermrest_uri = builtin_types.ermrest_uri.typename
    color_rgb_hex = builtin_types.color_rgb_hex.typename
    serial2 = builtin_types.serial2.typename
    serial4 = builtin_types.serial4.typename
    serial8 = builtin_types.serial8.typename


class FileSpec(BaseModel):
    """An entry into the File table

    Attributes:
        url: The File url to the url.
        description: The description of the file.
    """

    url: str
    description: str | None = ""
    md5: str
    length: int

    @field_validator("url")
    @classmethod
    def validate_file_url(cls, v):
        """Examine the provided URL.  If it's a local path, convert it into a tag URL."""
        url_parts = urlparse(v)
        if url_parts.scheme == "tag":
            # Already a tag URL, so just return it.
            return v
        elif (not url_parts.scheme) or url_parts.scheme == "file":
            # There is no scheme part tof the URL, or it is a file URL, so it is a local file path, so convert to a tag URL.
            return f"tag://{gethostname()},{date.today()}:file://{url_parts.path}"
        else:
            raise ValidationError("url is not a file URL")

    @model_serializer()
    def serialize_filespec(self):
        return {
            "URL": self.url,
            "Description": self.description,
            "MD5": self.md5,
            "Length": self.length,
        }

    @staticmethod
    def create_filespecs(
        path: Path | str, description: str
    ) -> Generator["FileSpec", None, None]:
        """Given a file or directory, generate the sequence of corresponding FileSpecs sutable to create a File table

        Arguments:
            path: Path to the file or directory.
            description: The description of the file(s)

        Returns:
            An iterable of FileSpecs for each file in the directory.
        """
        path = Path(path)

        def list_all_files(p) -> List[Path]:
            files = [p] if path.is_file() else list(Path(p).rglob("*"))
            return [f for f in files if f.is_file()]

        def create_spec(p: Path, description: str) -> FileSpec:
            hashes = hash_utils.compute_file_hashes(
                p, hashes=frozenset(["md5", "sha256"])
            )
            md5 = hashes["md5"][0]
            return FileSpec(
                length=path.stat().st_size,
                md5=md5,
                description=description,
                url=p.as_posix(),
            )

        return (create_spec(file, description) for file in list_all_files(path))

    @staticmethod
    def read_filespec(path: Path | str) -> Iterator[FileSpec]:
        """Get FileSpecs from a JSON lines file.

        Arguments:
         path: Path to the .jsonl file (string or Path).

        Yields:
             A FileSpec object.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield FileSpec(**json.loads(line))


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


class MLVocab(BaseStrEnum):
    """Names of controlled vocabulary for various types within DerivaML."""

    dataset_type = "Dataset_Type"
    workflow_type = "Workflow_Type"
    file_type = "File_Type"
    asset_type = "Asset_Type"
    asset_role = "Asset_Role"


class MLAsset(BaseStrEnum):
    execution_metadata = "Execution_Metadata"
    execution_asset = "Execution_Asset"


class ExecMetadataType(BaseStrEnum):
    """
    Predefined execution metadata types.
    """

    execution_config = "Execution_Config"
    runtime_env = "Runtime_Env"


class ExecAssetType(BaseStrEnum):
    """
    Predefined execution metadata types.
    """

    input_file = "Input_File"
    output_file = "Output_File"
    notebook_output = "Notebook_Output"


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


class DerivaMLException(Exception):
    """Exception class specific to DerivaML module.

    Args:
        msg (str): Optional message for the exception.
    """

    def __init__(self, msg=""):
        super().__init__(msg)
        self._msg = msg

"""ERMrest data models for DerivaML.

This module provides models that represent ERMrest catalog structures. These models are used
throughout DerivaML for defining and manipulating catalog elements like tables, columns, and keys.

The core definition classes (ColumnDef, KeyDef, ForeignKeyDef, TableDef) are now provided by
`deriva.core.typed` and re-exported here for backwards compatibility.

Classes:
    FileUploadState: Tracks the state of file uploads.
    VocabularyTerm: Represents terms in controlled vocabularies.
    ColumnDefinition: Alias for ColumnDef from deriva.core.typed.
    KeyDefinition: Alias for KeyDef from deriva.core.typed.
    ForeignKeyDefinition: Alias for ForeignKeyDef from deriva.core.typed.
    TableDefinition: Alias for TableDef from deriva.core.typed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol

# Import and re-export typed definitions from deriva.core.typed
from deriva.core.typed import (
    AssetTableDef,
    AssociationTableDef,
    ColumnDef,
    ForeignKeyDef,
    KeyDef,
    SchemaDef,
    TableDef,
    VocabularyTableDef,
)
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
)

from .constants import RID
from .enums import UploadState

# Re-export all typed classes for convenience
__all__ = [
    # New typed definitions from deriva.core.typed
    "ColumnDef",
    "KeyDef",
    "ForeignKeyDef",
    "TableDef",
    "VocabularyTableDef",
    "AssetTableDef",
    "AssociationTableDef",
    "SchemaDef",
    # Legacy aliases for backwards compatibility
    "ColumnDefinition",
    "KeyDefinition",
    "ForeignKeyDefinition",
    "TableDefinition",
    # DerivaML-specific classes
    "FileUploadState",
    "UploadProgress",
    "UploadCallback",
    "VocabularyTerm",
    "VocabularyTermHandle",
]

# Pydantic warnings suppression
warnings.filterwarnings("ignore", message='Field name "schema"', category=Warning, module="pydantic")
warnings.filterwarnings(
    "ignore",
    message="fields may not start with an underscore",
    category=Warning,
    module="pydantic",
)


# =============================================================================
# Compatibility Aliases
# =============================================================================
# These aliases maintain backwards compatibility with existing DerivaML code
# that uses the old Pydantic-based class names.

ColumnDefinition = ColumnDef
"""Alias for ColumnDef from deriva.core.typed.

This maintains backwards compatibility with existing DerivaML code.
New code should use ColumnDef directly.
"""

KeyDefinition = KeyDef
"""Alias for KeyDef from deriva.core.typed.

This maintains backwards compatibility with existing DerivaML code.
New code should use KeyDef directly.
"""

ForeignKeyDefinition = ForeignKeyDef
"""Alias for ForeignKeyDef from deriva.core.typed.

This maintains backwards compatibility with existing DerivaML code.
New code should use ForeignKeyDef directly.
"""

TableDefinition = TableDef
"""Alias for TableDef from deriva.core.typed.

This maintains backwards compatibility with existing DerivaML code.
New code should use TableDef directly.
"""


# =============================================================================
# DerivaML-Specific Classes
# =============================================================================


class FileUploadState(BaseModel):
    """Tracks the state and result of a file upload operation.

    Attributes:
        state (UploadState): Current state of the upload (success, failed, etc.).
        status (str): Detailed status message.
        result (Any): Upload result data, if any.
    """
    state: UploadState
    status: str
    result: Any

    @computed_field
    @property
    def rid(self) -> RID | None:
        return self.result and self.result["RID"]


@dataclass
class UploadProgress:
    """Progress information for file uploads.

    This dataclass is passed to upload callbacks to report progress during
    file upload operations.

    Attributes:
        file_path: Path to the file being uploaded.
        file_name: Name of the file being uploaded.
        bytes_completed: Number of bytes uploaded so far.
        bytes_total: Total number of bytes to upload.
        percent_complete: Percentage of upload completed (0-100).
        phase: Current phase of the upload operation.
        message: Human-readable status message.
    """
    file_path: str = ""
    file_name: str = ""
    bytes_completed: int = 0
    bytes_total: int = 0
    percent_complete: float = 0.0
    phase: str = ""
    message: str = ""


class UploadCallback(Protocol):
    """Protocol for upload progress callbacks.

    Implement this protocol to receive progress updates during file uploads.
    The callback is invoked with an UploadProgress object containing current
    upload state information.

    Example:
        >>> def my_callback(progress: UploadProgress) -> None:
        ...     print(f"Uploading {progress.file_name}: {progress.percent_complete:.1f}%")
        ...
        >>> execution.upload_execution_outputs(progress_callback=my_callback)
    """
    def __call__(self, progress: UploadProgress) -> None:
        """Called with upload progress information.

        Args:
            progress: Current upload progress state.
        """
        ...


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
    _name: str = PrivateAttr()
    _synonyms: list[str] | None = PrivateAttr()
    _description: str = PrivateAttr()
    id: str = Field(alias="ID")
    uri: str = Field(alias="URI")
    rid: str = Field(alias="RID")

    def __init__(self, **data):
        # Extract fields that will be private attrs before calling super
        name = data.pop("Name", None) or data.pop("name", None)
        synonyms = data.pop("Synonyms", None) or data.pop("synonyms", None)
        description = data.pop("Description", None) or data.pop("description", None)
        super().__init__(**data)
        self._name = name
        self._synonyms = synonyms
        self._description = description

    @property
    def name(self) -> str:
        """Primary name of the term."""
        return self._name

    @property
    def synonyms(self) -> tuple[str, ...]:
        """Alternative names for the term (immutable)."""
        return tuple(self._synonyms or [])

    @property
    def description(self) -> str:
        """Explanation of the term's meaning."""
        return self._description

    class Config:
        extra = "ignore"


class VocabularyTermHandle(VocabularyTerm):
    """A VocabularyTerm with methods to modify it in the catalog.

    This class extends VocabularyTerm to provide mutable access to vocabulary
    terms. Changes made through property setters are persisted to the catalog.

    The `synonyms` property returns a tuple (immutable) to prevent accidental
    modification without catalog update. To modify synonyms, assign a new
    tuple/list to the property.

    Attributes:
        Inherits all attributes from VocabularyTerm.

    Example:
        >>> term = ml.lookup_term("Dataset_Type", "Training")
        >>> term.description = "Data used for model training"
        >>> term.synonyms = ("Train", "TrainingData")
        >>> term.delete()
    """

    _ml: Any = PrivateAttr()
    _table: str = PrivateAttr()

    def __init__(self, ml: Any, table: str, **data):
        """Initialize a VocabularyTermHandle.

        Args:
            ml: DerivaML instance for catalog operations.
            table: Name of the vocabulary table containing this term.
            **data: Term data (Name, Synonyms, Description, ID, URI, RID).
        """
        super().__init__(**data)
        self._ml = ml
        self._table = table

    @property
    def description(self) -> str:
        """Explanation of the term's meaning."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        """Update the term's description in the catalog.

        Args:
            value: New description for the term.
        """
        self._ml._update_term_description(self._table, self.name, value)
        self._description = value

    @property
    def synonyms(self) -> tuple[str, ...]:
        """Alternative names for the term (immutable).

        Returns a tuple to prevent accidental modification without catalog update.
        To modify synonyms, assign a new tuple/list to this property.
        """
        return tuple(self._synonyms or [])

    @synonyms.setter
    def synonyms(self, value: list[str] | tuple[str, ...]) -> None:
        """Replace all synonyms for this term in the catalog.

        Args:
            value: New list of synonyms (replaces all existing synonyms).
        """
        new_synonyms = list(value)
        self._ml._update_term_synonyms(self._table, self.name, new_synonyms)
        self._synonyms = new_synonyms

    def delete(self) -> None:
        """Delete this term from the vocabulary.

        Raises:
            DerivaMLException: If the term is currently in use by other records.
        """
        self._ml.delete_term(self._table, self.name)

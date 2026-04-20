"""Auxiliary classes for dataset versioning, history, and configuration.

This module defines VersionPart, DatasetVersion, DatasetHistory,
DatasetMinid, DatasetSpec, and DatasetSpecConfig -- the value objects
used throughout DerivaML to represent dataset versions, provenance
records, and hydra-zen configuration entries.
"""

from enum import Enum
from pprint import pformat
from typing import Any, Optional, SupportsInt

from hydra_zen import hydrated_dataclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    conlist,
    field_serializer,
    field_validator,
    model_validator,
)
from semver import Version

from deriva_ml.core.definitions import RID

try:
    from icecream import ic

    ic.configureOutput(
        includeContext=True,
        argToStringFunction=lambda x: pformat(x.model_dump() if hasattr(x, "model_dump") else x, width=80, depth=10),
    )
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class VersionPart(Enum):
    """Simple enumeration for semantic versioning.

    Attributes:
        major (int): Major version number
        minor (int): Minor version number
        patch (int): Patch version number

    """

    major = "major"
    minor = "minor"
    patch = "patch"


class DatasetVersion(Version):
    """Represent the version associated with a dataset using semantic versioning.

    Methods:
        replace(major, minor, patch): Replace the major and minor versions
    """

    def __init__(self, major: SupportsInt, minor: SupportsInt = 0, patch: SupportsInt = 0) -> None:
        """Initialize a DatasetVersion object.

        Args:
            major: Major version number. Used to indicate schema changes.
            minor: Minor version number.  Used to indicate additional members added, or change in member values.
            patch: Patch number of the dataset.  Used to indicate minor clean-up and edits
        """
        super().__init__(major, minor, patch)

    def to_dict(self) -> dict[str, Any]:
        """

        Returns:
            dictionary of version information

        """
        return {"major": self.major, "minor": self.minor, "patch": self.patch}

    def to_tuple(self) -> tuple[int, int, int]:
        """

        Returns:
            tuple of version information

        """
        return self.major, self.minor, self.patch

    @classmethod
    def parse(cls, version: str, optional_minor_an_path: bool = False) -> "DatasetVersion":
        """Parse a semantic version string into a DatasetVersion.

        Args:
            version: A semantic version string (e.g., ``"1.2.3"``).
            optional_minor_an_path: Unused; kept for API compatibility with
                :meth:`semver.Version.parse`.

        Returns:
            A new DatasetVersion corresponding to the parsed string.

        Raises:
            ValueError: If *version* is not a valid semantic version string.
        """
        v = Version.parse(version)
        return DatasetVersion(v.major, v.minor, v.patch)

    def increment_version(self, component: VersionPart) -> "DatasetVersion":
        """Return a new DatasetVersion with the specified component incremented.

        Follows standard semantic versioning rules: incrementing a higher-order
        component resets all lower-order components to zero.

        Args:
            component: Which part of the version to bump (major, minor, or patch).

        Returns:
            A new DatasetVersion with the requested component incremented.
        """
        match component:
            case VersionPart.major:
                return self.bump_major()
            case VersionPart.minor:
                return self.bump_minor()
            case VersionPart.patch:
                return self.bump_patch()
            case _:
                return self


class DatasetHistory(BaseModel):
    """
    Class representing a dataset history.

    Attributes:
        dataset_version (DatasetVersion): A DatasetVersion object which captures the semantic versioning of the dataset.
        dataset_rid (RID): The RID of the dataset.
        version_rid (RID): The RID of the version record for the dataset in the Dataset_Version table.
        minid (str): The URL that represents the handle of the dataset bag.  This will be None if a MINID has not
                     been created yet.
        snapshot (str): Catalog snapshot ID of when the version record was created.
    """

    dataset_version: DatasetVersion
    dataset_rid: RID
    version_rid: RID
    execution_rid: Optional[RID] = None
    description: str | None = ""
    minid: str | None = None
    spec_hash: str | None = None
    snapshot: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("execution_rid", mode="before")
    @classmethod
    def _default_execution_rid(cls, v: str | None) -> str | None:
        return None if v == "" else v

    @field_validator("description", mode="after")
    def _default_description(cls, v: str | None) -> str:
        return v or ""


class DatasetMinid(BaseModel):
    """Represent information about a MINID that refers to a dataset

    Attributes:
        dataset_version (DatasetVersion): A DatasetVersion object which captures the semantic versioning of the dataset.
        metadata (dict): A dictionary containing metadata from the MINID landing page.
        minid (str): The URL that represents the handle of the MINID associated with the dataset.
        bag_url (str): The URL to the dataset bag
        identifier (str): The identifier of the MINID in CURI form
        landing_page (str): The URL to the landing page of the MINID
        version_rid (str): RID of the dataset version.
        checksum (str): The checksum of the MINID in SHA256 form

    """

    dataset_version: DatasetVersion
    metadata: dict[str, str | int] = {}
    minid: str = Field(alias="compact_uri", default=None)
    bag_url: str = Field(alias="location")
    identifier: Optional[str] = None
    landing_page: Optional[str] = None
    version_rid: RID = Field(alias="RID")
    checksum: str = Field(alias="checksums", default="")

    @computed_field
    @property
    def dataset_rid(self) -> str:
        rid_parts = self.version_rid.split("@")
        return rid_parts[0]

    @computed_field
    @property
    def dataset_snapshot(self) -> str:
        return self.version_rid.split("@")[1]

    @model_validator(mode="before")
    @classmethod
    def insert_metadata(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if "metadata" in data:
                data = data | data["metadata"]
        return data

    @field_validator("bag_url", mode="before")
    @classmethod
    def convert_location_to_str(cls, value: list[str] | str) -> str:
        return value[0] if isinstance(value, list) else value

    @field_validator("checksum", mode="before")
    @classmethod
    def convert_checksum_to_value(cls, checksums: list[dict]) -> str:
        checksum_value = ""
        for checksum in checksums:
            if checksum.get("function") == "sha256":
                checksum_value = checksum.get("value")
                break
        return checksum_value

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasetSpec(BaseModel):
    """Represent a dataset_table in an execution configuration dataset_table list

    Attributes:
        rid (RID): A dataset_table RID
        materialize (bool): If False do not materialize datasets, only download table data, no assets.  Defaults to True
        version (DatasetVersion): The version of the dataset.  Should follow semantic versioning.
        exclude_tables (set[str] | None): Optional set of table names to exclude from FK path
            traversal during bag export. Tables in this set will not be visited, pruning branches
            of the FK graph. Useful for avoiding query timeouts on large tables.
        timeout (tuple[int, int] | None): Optional (connect_timeout, read_timeout) in seconds
            for network requests during bag download. Defaults to (10, 610) if not specified.
            Increase read_timeout for large datasets with deep FK joins.
    """

    rid: RID
    version: DatasetVersion | conlist(item_type=int, min_length=3, max_length=3) | tuple[int, int, int] | str
    materialize: bool = True
    description: str = ""
    exclude_tables: set[str] | None = None
    timeout: tuple[int, int] | None = None
    fetch_concurrency: int = 8

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("version", mode="before")
    @classmethod
    def version_field_validator(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return DatasetVersion(**v)
        elif isinstance(v, str):
            return DatasetVersion.parse(v)
        elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) == 3:
            return DatasetVersion(int(v[0]), int(v[1]), int(v[2]))
        else:
            return v

    @model_validator(mode="before")
    @classmethod
    def _check_bare_rid(cls, data: Any) -> dict[str, str | bool]:
        # If you are just given a string, assume it's a rid and put into dict for further validation.
        return {"rid": data} if isinstance(data, str) else data

    @field_serializer("version")
    def serialize_version(self, version: DatasetVersion) -> dict[str, Any]:
        return version.to_dict()

    @classmethod
    def from_shorthand(cls, s: str) -> "DatasetSpec":
        """Parse ``'RID@version'`` into a :class:`DatasetSpec`.

        Used by the :meth:`DerivaML.create_execution` kwargs form so
        callers can write ``datasets=["1-XYZ@1.0.0"]`` instead of
        instantiating a full ``DatasetSpec`` by hand. Accepts both
        ``'RID'`` (bare RID; version defaults to ``0.0.0``) and
        ``'RID@version'`` (semantic version string).

        Args:
            s: The shorthand string. Must contain at most one ``'@'``
                separator and a non-empty RID portion.

        Returns:
            A :class:`DatasetSpec` instance.

        Raises:
            ValueError: If the string is empty or contains more than
                one ``'@'`` separator.

        Example:
            Parse a shorthand with explicit version::

                >>> spec = DatasetSpec.from_shorthand("1-XYZ@2.0.0")
                >>> spec.rid
                '1-XYZ'
                >>> str(spec.version)
                '2.0.0'

            Parse a bare RID (version defaults to ``0.0.0``)::

                >>> spec = DatasetSpec.from_shorthand("1-XYZ")
                >>> spec.rid
                '1-XYZ'
        """
        if not s:
            raise ValueError("empty dataset shorthand")
        parts = s.split("@")
        if len(parts) == 1:
            return cls(rid=parts[0], version="0.0.0")
        if len(parts) == 2:
            return cls(rid=parts[0], version=parts[1])
        raise ValueError(
            f"dataset shorthand has too many '@' separators: {s!r}"
        )


# Interface for hydra-zen
@hydrated_dataclass(DatasetSpec)
class DatasetSpecConfig:
    """Hydra-zen configuration dataclass for :class:`DatasetSpec`.

    Use this in hydra-zen ``store()`` calls and configuration modules to
    specify dataset inputs.  When instantiated by hydra-zen, it produces a
    :class:`DatasetSpec` instance.

    Attributes:
        rid: Dataset RID (e.g., ``"28CT"``).
        version: Semantic version string (e.g., ``"0.21.0"``).
        materialize: If False, download only table metadata, not asset files.
        description: Human-readable description of the dataset's role in this config.
        exclude_tables: Optional table names to exclude from FK path traversal
            during bag export.
        timeout: Optional ``[connect_timeout, read_timeout]`` in seconds for
            network requests during bag download.
        fetch_concurrency: Number of concurrent fetch threads for asset download.
    """

    rid: str
    version: str
    materialize: bool = True
    description: str = ""
    exclude_tables: list[str] | None = None
    timeout: list[int] | None = None
    fetch_concurrency: int = 8

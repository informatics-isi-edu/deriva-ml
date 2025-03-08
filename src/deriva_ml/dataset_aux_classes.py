"""
THis module defines the DataSet class with is used to manipulate n
"""

from datetime import datetime
from .deriva_definitions import RID
from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    Field,
    computed_field,
    model_validator,
    field_serializer,
)

from semver import Version
from typing import Optional, Any, SupportsInt


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

    def __init__(
        self, major: SupportsInt, minor: SupportsInt = 0, patch: SupportsInt = 0
    ):
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
    def parse(cls, version: str, optional_minor_an_path=False) -> "DatasetVersion":
        v = Version.parse(version)
        return DatasetVersion(v.major, v.minor, v.patch)

    def increment_version(self, component: VersionPart) -> "DatasetVersion":
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
        timestamp (datetime): The timestamp of when the  dataset was created.
    """

    dataset_version: DatasetVersion
    dataset_rid: RID
    version_rid: RID
    execution_rid: Optional[RID] = None
    description: str = ""
    minid: Optional[str] = None
    timestamp: Optional[datetime] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    metadata: dict[str, str | int]
    minid: str = Field(alias="compact_uri")
    bag_url: str = Field(alias="location")
    identifier: str
    landing_page: str
    version_rid: RID = Field(alias="Dataset_RID")
    checksum: str = Field(alias="checksums", default="")

    @computed_field
    @property
    def dataset_rid(self) -> str:
        return self.version_rid.split("@")[0]

    @computed_field
    @property
    def dataset_snapshot(self) -> str:
        return self.version_rid.split("@")[1]

    @model_validator(mode="before")
    @classmethod
    def insert_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "metadata" in data:
                data = data | data["metadata"]
        return data

    @field_validator("bag_url", mode="before")
    @classmethod
    def convert_location_to_str(cls, value: list[str]) -> str:
        return value[0]

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
        materialize (bool): If False, do not materialize datasets, only download table data, no assets.  Defaults to True
        version (DatasetVersion): The version of the dataset.  Should follow semantic versioning.
    """

    rid: RID
    materialize: bool = True
    version: DatasetVersion

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("version", mode="before")
    @classmethod
    def version_field_validator(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return DatasetVersion(**v)
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

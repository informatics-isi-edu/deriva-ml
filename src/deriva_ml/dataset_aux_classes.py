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
    validate_call,
)

from semver import Version
from typing import Optional, Any


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

    Attributes:
        major (int): Major version number
        minor (int): Minor version number
        patch (int): Patch version number

    Methods:
        to_dict(): Convert a DatasetVersion object to a dict.
        to_tuple(): Convert a DatasetVersion object to a tuple.
        replace(major, minor, patch): Replace the major and minor versions
        parse(str): Parse the string into a DatasetVersion object.
    """

    @validate_call
    def __init__(self, major: int, minor: int, patch: int):
        """Initialize a DatasetVersion object.

        Args:
            major (int | str)
        """
        super().__init__(major, minor, patch)


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
        checksum (str): The checksum of the MINID in SHA256 form
        dataset_rid (str): The RID of the dataset.
        dataset_snapshot (str): The ERMRest catalog snapshot for the dataset version

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

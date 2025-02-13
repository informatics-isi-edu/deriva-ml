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
    model_serializer,
)

from semver import Version
from typing import Optional, Any


class VersionPart(Enum):
    """Simple enumeration for semantic versioning."""

    major = "major"
    minor = "minor"
    patch = "patch"


class DatasetVersion(Version):
    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)

    @model_serializer()
    def model_dump(self):
        print("model_dump ")
        return self.to_dict()


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
    def dataset_rid(self) -> int:
        return self.version_rid.split("@")[0]

    @computed_field
    @property
    def dataset_snapshot(self) -> int:
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

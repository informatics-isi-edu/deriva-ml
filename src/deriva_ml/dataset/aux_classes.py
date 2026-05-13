"""Auxiliary classes for dataset versioning, history, and configuration.

This module defines VersionPart, DatasetVersion, DatasetHistory,
DatasetMinid, DatasetSpec, and DatasetSpecConfig -- the value objects
used throughout DerivaML to represent dataset versions, provenance
records, and hydra-zen configuration entries.
"""

from enum import Enum
from typing import Any, Optional, SupportsInt

from hydra_zen import hydrated_dataclass
from packaging.version import Version
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

from deriva_ml.core.definitions import RID
from deriva_ml.core.validation import VALIDATION_CONFIG


class VersionPart(Enum):
    """Names the component of a dataset version to advance on release.

    DerivaML uses a ``major.minor.patch`` release segment within the broader
    PEP 440 version space (see ADR-0004). Picking a ``VersionPart`` selects
    which component is incremented when a dev period is promoted to a
    released version.

    Attributes:
        major: Schema-altering changes that break backward compatibility.
        minor: Additive changes — new members, new feature values, new
            annotations.
        patch: Small clean-ups and edits that don't change the dataset's
            shape.
    """

    major = "major"
    minor = "minor"
    patch = "patch"


class DatasetVersion(Version):
    """A PEP 440 version associated with a dataset.

    Released versions are written as ``"MAJOR.MINOR.PATCH"`` (e.g., ``"0.4.0"``).
    Dev versions use the ``setuptools-scm``-compatible post-release form
    ``"<last_release>.post1.devN"`` (e.g., ``"0.4.0.post1.dev3"``) — they sort
    *after* the last release and *before* the next, and are queryable via
    :attr:`is_devrelease`. See ADR-0004 for the rationale behind PEP 440 over
    semver pre-release suffixes.

    The wire format for released versions is unchanged from the previous
    semver-backed implementation: a string like ``"0.4.0"`` parses and
    serialises identically.

    Example:
        Construct from positional integers (release-segment form):

            >>> v = DatasetVersion(0, 4, 0)
            >>> str(v)
            '0.4.0'
            >>> v.is_devrelease
            False

        Construct from a string (any PEP 440 form):

            >>> dev = DatasetVersion.parse("0.4.0.post1.dev3")
            >>> dev.is_devrelease
            True
            >>> DatasetVersion(0, 4, 0) < dev < DatasetVersion(0, 5, 0)
            True

        Advance the release-segment for a release:

            >>> DatasetVersion(0, 4, 0).next_release(VersionPart.minor)
            <Version('0.5.0')>
    """

    def __init__(
        self,
        major: SupportsInt,
        minor: SupportsInt = 0,
        patch: SupportsInt = 0,
    ) -> None:
        """Construct a released ``DatasetVersion`` from a release-segment tuple.

        For PEP 440 forms beyond ``MAJOR.MINOR.PATCH`` (post-release, dev,
        local, etc.), use :meth:`parse` with the canonical string form.

        Args:
            major: Major version number. Schema-altering changes.
            minor: Minor version number. Additive changes.
            patch: Patch number. Small clean-ups and edits.
        """
        super().__init__(f"{int(major)}.{int(minor)}.{int(patch)}")

    @property
    def patch(self) -> int:
        """The patch component of the release segment.

        ``packaging.Version`` exposes this as :attr:`micro`. ``patch`` is
        kept on ``DatasetVersion`` because it matches the ``VersionPart``
        vocabulary and the column meaning.
        """
        return self.micro

    def to_dict(self) -> dict[str, int]:
        """Serialise the release segment as a ``{major, minor, patch}`` dict.

        Used by :class:`DatasetSpec`'s field serializer for hydra-zen
        round-tripping. Pre-release / post-release / dev / local segments
        are *not* preserved in this form — it represents only the
        release-segment tuple. Use ``str(self)`` for a lossless serialisation.

        Returns:
            A dict with integer ``major``, ``minor``, and ``patch`` fields.

        Example:
            >>> DatasetVersion(1, 2, 3).to_dict()
            {'major': 1, 'minor': 2, 'patch': 3}
        """
        return {"major": self.major, "minor": self.minor, "patch": self.patch}

    @classmethod
    def parse(cls, version: str) -> "DatasetVersion":
        """Parse a PEP 440 version string into a ``DatasetVersion``.

        Args:
            version: A PEP 440 version string. Released forms like
                ``"1.2.3"`` and dev forms like ``"1.2.3.post1.dev4"`` are
                both accepted.

        Returns:
            A new ``DatasetVersion`` corresponding to the parsed string.

        Raises:
            packaging.version.InvalidVersion: If *version* is not a valid
                PEP 440 version string.

        Example:
            >>> str(DatasetVersion.parse("0.4.0"))
            '0.4.0'
            >>> DatasetVersion.parse("0.4.0.post1.dev3").is_devrelease
            True
        """
        # __new__ on the parent does the parse; we just need to return a
        # subclass instance with the same internal state.
        v = Version(version)
        instance = cls.__new__(cls)
        Version.__init__(instance, str(v))
        return instance

    def next_release(self, bump: "VersionPart") -> "DatasetVersion":
        """Return the next released ``DatasetVersion`` after this one.

        Applies a release-segment bump to ``(major, minor, patch)`` and
        discards any post-release / dev / local segments — the new value
        is always a clean released version. Higher-order bumps reset
        lower-order components to zero, matching the standard
        ``major.minor.patch`` convention.

        Args:
            bump: Which part of the release segment to advance.

        Returns:
            A new released ``DatasetVersion`` with the requested
            component advanced.

        Example:
            >>> DatasetVersion(0, 4, 0).next_release(VersionPart.minor)
            <Version('0.5.0')>
            >>> DatasetVersion(0, 4, 7).next_release(VersionPart.major)
            <Version('1.0.0')>
            >>> DatasetVersion.parse("0.4.0.post1.dev3").next_release(
            ...     VersionPart.minor
            ... )
            <Version('0.5.0')>
        """
        match bump:
            case VersionPart.major:
                return DatasetVersion(self.major + 1, 0, 0)
            case VersionPart.minor:
                return DatasetVersion(self.major, self.minor + 1, 0)
            case VersionPart.patch:
                return DatasetVersion(self.major, self.minor, self.patch + 1)
            case _:  # pragma: no cover - defensive; VersionPart is closed
                raise ValueError(f"unknown VersionPart: {bump!r}")


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

    model_config = VALIDATION_CONFIG

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
    minid: str | None = Field(alias="compact_uri", default=None)
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
    def dataset_snapshot(self) -> str | None:
        # ``version_rid`` is ``{rid}`` or ``{rid}@{snapshot}`` per
        # the validation pattern. The unsnapped form has no
        # ``@`` segment; surface that as ``None`` rather than
        # IndexError-ing on the missing split component.
        parts = self.version_rid.split("@", 1)
        return parts[1] if len(parts) == 2 else None

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

    model_config = VALIDATION_CONFIG


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

    model_config = VALIDATION_CONFIG

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
        raise ValueError(f"dataset shorthand has too many '@' separators: {s!r}")


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

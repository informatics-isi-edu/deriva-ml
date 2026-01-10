"""File specification utilities for DerivaML.

This module provides the FileSpec class for creating and managing file metadata
in the Deriva catalog. FileSpec objects represent files with their checksums,
sizes, and type classifications, ready for insertion into the File table.

Key Features:
    - Automatic MD5 checksum computation
    - URL normalization (local paths converted to tag URIs)
    - Support for file type classification
    - Batch processing of directories
    - JSONL serialization/deserialization

Example:
    Create FileSpec from a local file:
        >>> specs = list(FileSpec.create_filespecs(
        ...     path="/data/images/sample.png",
        ...     description="Sample image",
        ...     file_types=["Image", "PNG"]
        ... ))

    Read FileSpecs from a JSONL file:
        >>> specs = list(FileSpec.read_filespec("files.jsonl"))
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from socket import gethostname
from typing import Callable, Generator
from urllib.parse import urlparse

import deriva.core.utils.hash_utils as hash_utils
from pydantic import BaseModel, Field, conlist, field_validator, validate_call


class FileSpec(BaseModel):
    """Specification for a file to be added to the Deriva catalog.

    Represents file metadata required for creating entries in the File table.
    Handles URL normalization, ensuring local file paths are converted to
    tag URIs that uniquely identify the file's origin.

    Attributes:
        url: File location as URL or local path. Local paths are converted to tag URIs.
        md5: MD5 checksum for integrity verification.
        length: File size in bytes.
        description: Optional description of the file's contents or purpose.
        file_types: List of file type classifications from the Asset_Type vocabulary.

    Note:
        The 'File' type is automatically added to file_types if not present when
        using create_filespecs().

    Example:
        >>> spec = FileSpec(
        ...     url="/data/results.csv",
        ...     md5="d41d8cd98f00b204e9800998ecf8427e",
        ...     length=1024,
        ...     description="Analysis results",
        ...     file_types=["CSV", "Data"]
        ... )
    """

    url: str = Field(alias="URL", validation_alias="url")
    md5: str = Field(alias="MD5", validation_alias="md5")
    length: int = Field(alias="Length", validation_alias="length")
    description: str | None = Field(default="", alias="Description", validation_alias="description")
    file_types: conlist(str) | None = []

    @field_validator("url")
    @classmethod
    def validate_file_url(cls, url: str) -> str:
        """Examine the provided URL. If it's a local path, convert it into a tag URL.

        Args:
            url: The URL to validate and potentially convert

        Returns:
            The validated/converted URL

        Raises:
            ValidationError: If the URL is not a file URL
        """
        url_parts = urlparse(url)
        if url_parts.scheme == "tag":
            # Already a tag URL, so just return it.
            return url
        elif (not url_parts.scheme) or url_parts.scheme == "file":
            # There is no scheme part of the URL, or it is a file URL, so it is a local file path.
            # Convert to a tag URL.
            return f"tag://{gethostname()},{date.today()}:file://{url_parts.path}"
        else:
            raise ValueError("url is not a file URL")

    @classmethod
    def create_filespecs(
        cls, path: Path | str, description: str, file_types: list[str] | Callable[[Path], list[str]] | None = None
    ) -> Generator[FileSpec, None, None]:
        """Generate FileSpec objects for a file or directory.

        Creates FileSpec objects with computed MD5 checksums for each file found.
        For directories, recursively processes all files. The 'File' type is
        automatically prepended to file_types if not already present.

        Args:
            path: Path to a file or directory. If directory, all files are processed recursively.
            description: Description to apply to all generated FileSpecs.
            file_types: Either a static list of file types, or a callable that takes a Path
                and returns a list of types for that specific file. Allows dynamic type
                assignment based on file extension, content, etc.

        Yields:
            FileSpec: A specification for each file with computed checksums and metadata.

        Example:
            Static file types:
                >>> specs = FileSpec.create_filespecs("/data/images", "Images", ["Image"])

            Dynamic file types based on extension:
                >>> def get_types(path):
                ...     ext = path.suffix.lower()
                ...     return {"png": ["PNG", "Image"], ".jpg": ["JPEG", "Image"]}.get(ext, [])
                >>> specs = FileSpec.create_filespecs("/data", "Mixed files", get_types)
        """
        path = Path(path)
        file_types = file_types or []
        # Convert static list to callable for uniform handling
        file_types_fn = file_types if callable(file_types) else lambda _x: file_types

        def create_spec(file_path: Path) -> FileSpec:
            """Create a FileSpec for a single file with computed hashes."""
            hashes = hash_utils.compute_file_hashes(file_path, hashes=frozenset(["md5", "sha256"]))
            md5 = hashes["md5"][0]
            type_list = file_types_fn(file_path)
            return FileSpec(
                length=path.stat().st_size,
                md5=md5,
                description=description,
                url=file_path.as_posix(),
                # Ensure 'File' type is always included
                file_types=type_list if "File" in type_list else ["File"] + type_list,
            )

        # Handle both single files and directories (recursive)
        files = [path] if path.is_file() else [f for f in Path(path).rglob("*") if f.is_file()]
        return (create_spec(file) for file in files)

    @staticmethod
    def read_filespec(path: Path | str) -> Generator[FileSpec, None, None]:
        """Read FileSpec objects from a JSON Lines file.

        Parses a JSONL file where each line is a JSON object representing a FileSpec.
        Empty lines are skipped. This is useful for batch processing pre-computed
        file specifications.

        Args:
            path: Path to the .jsonl file containing FileSpec data.

        Yields:
            FileSpec: Parsed FileSpec object for each valid line.

        Example:
            >>> for spec in FileSpec.read_filespec("files.jsonl"):
            ...     print(f"{spec.url}: {spec.md5}")
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield FileSpec(**json.loads(line))


# =============================================================================
# Pydantic Workaround
# =============================================================================
# Workaround for Pydantic's validate_call decorator not working directly with
# classmethods that have forward references. We extract the underlying function,
# wrap it with validate_call, and re-create the classmethod.
_raw = FileSpec.create_filespecs.__func__
FileSpec.create_filespecs = classmethod(validate_call(_raw))

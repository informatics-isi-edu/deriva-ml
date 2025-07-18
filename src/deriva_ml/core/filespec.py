"""
File-related utility functions for DerivaML.
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
    """An entry into the File table

    Attributes:
        url: The File url to the url.
        description: The description of the file.
        md5: The MD5 hash of the file.
        length: The length of the file in bytes.
        file_types: A list of file types.  Each files_type should be a defined term in MLVocab.file_type vocabulary.
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
        """Given a file or directory, generate the sequence of corresponding FileSpecs suitable to create a File table.

        Args:
            path: Path to the file or directory.
            description: The description of the file(s)
            file_types: A list of file types or a function that takes a file path and returns a list of file types.

        Returns:
            An iterable of FileSpecs for each file in the directory.
        """

        path = Path(path)
        file_types = file_types or []
        file_types_fn = file_types if callable(file_types) else lambda _x: file_types

        def create_spec(file_path: Path) -> FileSpec:
            hashes = hash_utils.compute_file_hashes(file_path, hashes=frozenset(["md5", "sha256"]))
            md5 = hashes["md5"][0]
            type_list = file_types_fn(file_path)
            return FileSpec(
                length=path.stat().st_size,
                md5=md5,
                description=description,
                url=file_path.as_posix(),
                file_types=type_list if "File" in type_list else ["File"] + type_list,
            )

        files = [path] if path.is_file() else [f for f in Path(path).rglob("*") if f.is_file()]
        return (create_spec(file) for file in files)

    @staticmethod
    def read_filespec(path: Path | str) -> Generator[FileSpec, None, None]:
        """Get FileSpecs from a JSON lines file.

        Args:
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


# Hack round pydantic validate_call and forward reference.
_raw = FileSpec.create_filespecs.__func__
# wrap it with validate_call, then re‐make it a classmethod
FileSpec.create_filespecs = classmethod(validate_call(_raw))

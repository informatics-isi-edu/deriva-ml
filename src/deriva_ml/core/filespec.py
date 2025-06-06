"""
File-related utility functions for DerivaML.
"""

import json
from datetime import date
from pathlib import Path
from socket import gethostname
from typing import Generator, List
from urllib.parse import urlparse

import deriva.core.utils.hash_utils as hash_utils
from pydantic import BaseModel, ValidationError, field_validator, model_serializer


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
    def validate_file_url(cls, url) -> str:
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
            # There is no scheme part of the URL, or it is a file URL, so it is a local file path, so convert to a tag URL.
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

    @classmethod
    def create_filespecs(cls, path: Path | str, description: str) -> Generator["FileSpec", None, None]:
        """Given a file or directory, generate the sequence of corresponding FileSpecs suitable to create a File table.

        Args:
            path: Path to the file or directory.
            description: The description of the file(s)

        Returns:
            An iterable of FileSpecs for each file in the directory.
        """

        path = Path(path)

        def list_all_files(path: Path) -> List[Path]:
            """List all files in a directory or return a single file.

            Args:
                path: Path to file or directory

            Returns:
                List of Path objects for all files
            """
            files = [path] if path.is_file() else list(Path(path).rglob("*"))
            return [f for f in files if f.is_file()]

        return (cls.create_spec(file, description) for file in list_all_files(path))

    @classmethod
    def create_spec(cls, path: Path, description: str) -> "FileSpec":
        """Create a FileSpec for a single file.

        Args:
            path: Path to the file
            description: Description of the file

        Returns:
            FileSpec object for the file
        """
        hashes = hash_utils.compute_file_hashes(path, hashes=frozenset(["md5", "sha256"]))
        md5 = hashes["md5"][0]
        return FileSpec(
            length=path.stat().st_size,
            md5=md5,
            description=description,
            url=path.as_posix(),
        )

    @staticmethod
    def read_filespec(path: Path | str) -> Generator["FileSpec", None, None]:
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

"""
Tests for the validation utilities module.
"""

import pytest
from pydantic import ValidationError


def test_validate_file_url():
    """Test file URL validation."""
    # Test tag URL
    tag_url = "tag://hostname,2024-01-01:file:///path/to/file.txt"
    assert validate_file_url(tag_url) == tag_url

    # Test file path
    file_path = "/path/to/file.txt"
    converted = validate_file_url(file_path)
    assert converted.startswith("tag://")
    assert converted.endswith(file_path)

    # Test file URL
    file_url = "file:///path/to/file.txt"
    converted = validate_file_url(file_url)
    assert converted.startswith("tag://")
    assert converted.endswith("/path/to/file.txt")

    # Test invalid URL
    with pytest.raises(ValidationError):
        validate_file_url("http://example.com/file.txt")

"""
Tests for the core models module.
"""

import pytest
from core.filespec import create_filespecs
from pydantic import ValidationError

from deriva_ml.core.ermrest import FileSpec, VocabularyTerm


def test_file_spec_validation(tmp_path):
    """Test FileSpec validation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Test valid file URL
    specs = list(create_filespecs(test_file, "Test file"))
    assert len(specs) == 1
    spec = specs[0]
    assert spec.url.startswith("tag://")
    assert spec.description == "Test file"
    assert spec.length == len("test content")

    # Test invalid URL
    with pytest.raises(ValidationError):
        FileSpec(url="http://example.com/file.txt", description="Invalid URL", md5="123", length=0)


def test_vocabulary_term():
    """Test VocabularyTerm model."""
    term = VocabularyTerm(
        Name="Test Term",
        Synonyms=["test", "term"],
        ID="TEST:001",
        URI="http://example.com/test",
        Description="A test term",
        RID="1234",
    )

    assert term.name == "Test Term"
    assert term.synonyms == ["test", "term"]
    assert term.id == "TEST:001"
    assert term.uri == "http://example.com/test"
    assert term.description == "A test term"
    assert term.rid == "1234"


def test_file_spec_serialization(tmp_path):
    """Test FileSpec serialization."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Create FileSpec
    specs = list(create_filespecs(test_file, "Test file"))
    spec = specs[0]

    # Test serialization
    serialized = spec.model_dump_json()
    deserialized = FileSpec.model_validate_json(serialized)

    assert deserialized.url == spec.url
    assert deserialized.description == spec.description
    assert deserialized.md5 == spec.md5
    assert deserialized.length == spec.length


def test_file_spec_read_write(tmp_path):
    """Test reading and writing FileSpecs to JSONL."""
    # Create test files
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("content 1")
    file2.write_text("content 2")

    # Create FileSpecs
    specs = list(create_filespecs(tmp_path, "Test files"))
    assert len(specs) == 2

    # Write to JSONL
    jsonl_file = tmp_path / "specs.jsonl"
    with jsonl_file.open("w") as f:
        for spec in specs:
            f.write(spec.model_dump_json() + "\n")

    # Read back
    read_specs = list(FileSpec.read_filespec(jsonl_file))
    assert len(read_specs) == 2

    # Compare
    for original, read in zip(specs, read_specs):
        assert read.url == original.url
        assert read.description == original.description
        assert read.md5 == original.md5
        assert read.length == original.length

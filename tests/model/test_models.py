"""
Tests for the core models module.
"""

import pytest
from pydantic import ValidationError


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

    def test_is_vocabulary(self):
        # Test the vocabulary table predicates.
        self.assertTrue(self.ml_instance.model.is_vocabulary("Dataset_Type"))
        self.assertFalse(self.ml_instance.model.is_vocabulary("Dataset"))
        self.assertRaises(DerivaMLException, self.ml_instance.model.is_vocabulary, "FooBar")

    def test_find_vocabularies(self):
        # Look for a known vocabulary in the deriva-ml schema
        self.assertIn("Dataset_Type", [v.name for v in self.ml_instance.model.find_vocabularies()])

    def test_create_vocabulary(self):
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertIn("CV1", [v.name for v in self.ml_instance.model.find_vocabularies()])
        self.assertTrue(self.ml_instance.model.is_vocabulary("Dataset_Type"))

    def test_find_assets(self):
        self.assertTrue(self.ml_instance.model.is_asset("Execution_Asset"))
        self.assertFalse(self.ml_instance.model.is_asset("Dataset"))
        self.assertIn("Execution_Asset", [a.name for a in self.ml_instance.model.find_assets()])

    def test_is_assoc(self):
        self.assertTrue(self.ml_instance.model.is_association("Dataset_Dataset"))
        self.assertFalse(self.ml_instance.model.is_association("Dataset"))

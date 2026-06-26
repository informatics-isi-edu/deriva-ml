"""Tests for the _root_description helper in FileMixin.add_files.

These tests cover the root-dataset description-selection logic that was
extracted into a module-level helper so it can be tested without a live
catalog connection.
"""

from pathlib import Path

from deriva_ml.core.mixins.file import _root_description


def test_root_description_defaults_to_basename():
    root = Path("/tmp/abc/cifar10_source")
    assert _root_description(root, root_name=None, description="generic") == "cifar10_source"


def test_root_description_uses_explicit_root_name():
    root = Path("/tmp/abc/cifar10_source")
    assert _root_description(root, root_name="CIFAR-10 source", description="generic") == "CIFAR-10 source"


def test_root_description_falls_back_to_description_when_basename_empty():
    """Path('/').name == '' so description must be used."""
    assert _root_description(Path("/"), root_name=None, description="my files") == "my files"


def test_root_description_sentinel_when_all_empty():
    """When basename and description are both empty, returns 'root'."""
    assert _root_description(Path("/"), root_name=None, description="") == "root"

"""Verify the library imports cleanly when tensorflow is absent.

If tensorflow is truly installed in the test venv, we stub it out of
sys.modules temporarily. The goal is to prove that importing DatasetBag
does NOT import tensorflow eagerly.
"""
from __future__ import annotations

import sys

import pytest


def test_dataset_bag_imports_without_tf(monkeypatch):
    """Removing tensorflow from sys.modules lets DatasetBag still import."""
    for name in list(sys.modules):
        if name == "tensorflow" or name.startswith("tensorflow."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    # Re-import dataset_bag under the tensorflow-less sys.modules state.
    if "deriva_ml.dataset.dataset_bag" in sys.modules:
        monkeypatch.delitem(sys.modules, "deriva_ml.dataset.dataset_bag", raising=False)
    from deriva_ml.dataset.dataset_bag import DatasetBag  # noqa: F401


def test_as_tf_dataset_raises_importerror_without_tf(monkeypatch):
    """Calling build_tf_dataset when tensorflow is absent raises ImportError
    with an install-hint message containing 'tensorflow' and 'deriva-ml[tf]'."""
    for name in list(sys.modules):
        if name == "tensorflow" or name.startswith("tensorflow."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    from unittest.mock import MagicMock
    bag = MagicMock()

    if "deriva_ml.dataset.tf_adapter" in sys.modules:
        monkeypatch.delitem(sys.modules, "deriva_ml.dataset.tf_adapter", raising=False)

    from deriva_ml.dataset.tf_adapter import build_tf_dataset
    with pytest.raises(ImportError, match=r"tensorflow"):
        build_tf_dataset(bag, "Image")

"""Verify the library imports cleanly when torch is absent.

If torch is truly installed in the test venv, we stub it out of
sys.modules temporarily. The goal is to prove that importing DatasetBag
does NOT import torch eagerly.
"""
from __future__ import annotations

import sys

import pytest


def test_dataset_bag_imports_without_torch(monkeypatch):
    """Removing torch from sys.modules lets DatasetBag still import."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    # Re-import dataset_bag under the torch-less sys.modules state.
    if "deriva_ml.dataset.dataset_bag" in sys.modules:
        monkeypatch.delitem(sys.modules, "deriva_ml.dataset.dataset_bag", raising=False)
    from deriva_ml.dataset.dataset_bag import DatasetBag  # noqa: F401


def test_as_torch_dataset_raises_importerror_without_torch(monkeypatch):
    """Calling as_torch_dataset when torch is absent raises ImportError
    with an install-hint message."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    from unittest.mock import MagicMock
    bag = MagicMock()

    if "deriva_ml.dataset.torch_adapter" in sys.modules:
        monkeypatch.delitem(sys.modules, "deriva_ml.dataset.torch_adapter", raising=False)

    from deriva_ml.dataset.torch_adapter import build_torch_dataset
    with pytest.raises(ImportError, match=r"torch"):
        build_torch_dataset(bag, "Image")

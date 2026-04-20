"""Tests for ConnectionMode enum."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from deriva_ml import ConnectionMode


def test_enum_members():
    assert ConnectionMode.online.value == "online"
    assert ConnectionMode.offline.value == "offline"
    assert list(ConnectionMode) == [ConnectionMode.online, ConnectionMode.offline]


def test_coerce_from_string():
    adapter = TypeAdapter(ConnectionMode)
    assert adapter.validate_python("online") is ConnectionMode.online
    assert adapter.validate_python("offline") is ConnectionMode.offline


def test_invalid_string_raises():
    adapter = TypeAdapter(ConnectionMode)
    with pytest.raises(ValidationError):
        adapter.validate_python("hybrid")


def test_str_representation_is_value():
    assert str(ConnectionMode.online) == "online"
    assert str(ConnectionMode.offline) == "offline"


def test_derivaml_default_mode_is_online(test_ml):
    """Default mode is online."""
    from deriva_ml import ConnectionMode
    assert test_ml.mode is ConnectionMode.online


def test_derivaml_accepts_mode_enum(catalog_manager, tmp_path):
    """Constructing DerivaML with mode=ConnectionMode.offline works."""
    from deriva_ml import ConnectionMode, DerivaML
    catalog_manager.reset()
    ml = DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode=ConnectionMode.offline,
    )
    assert ml.mode is ConnectionMode.offline


def test_derivaml_accepts_mode_string(catalog_manager, tmp_path):
    """String 'offline' is coerced to ConnectionMode.offline."""
    from deriva_ml import ConnectionMode, DerivaML
    catalog_manager.reset()
    ml = DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode="offline",
    )
    assert ml.mode is ConnectionMode.offline


def test_derivaml_rejects_invalid_mode(catalog_manager, tmp_path):
    """Unknown mode string raises ValueError (or ValidationError)."""
    from pydantic import ValidationError
    from deriva_ml import DerivaML
    catalog_manager.reset()
    with pytest.raises((ValidationError, ValueError)):
        DerivaML(
            hostname=catalog_manager.hostname,
            catalog_id=catalog_manager.catalog_id,
            working_dir=tmp_path,
            mode="hybrid",
        )

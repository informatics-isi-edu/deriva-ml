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

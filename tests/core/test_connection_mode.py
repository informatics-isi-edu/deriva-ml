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
    """Constructing DerivaML with mode=ConnectionMode.offline works
    when a schema cache has been populated by a prior online run.

    S4 (offline-mode init) made offline mode require a pre-populated
    <working_dir>/schema-cache.json; before S4 offline mode silently
    did network work anyway. This test now verifies the two-step
    online-populate-then-offline-load contract.
    """
    from deriva_ml import ConnectionMode, DerivaML
    catalog_manager.reset()
    # Step 1: online run populates the schema cache in tmp_path.
    DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode=ConnectionMode.online,
    )
    # Step 2: offline run reads the cache, skips all network.
    ml = DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode=ConnectionMode.offline,
    )
    assert ml.mode is ConnectionMode.offline


def test_derivaml_accepts_mode_string(catalog_manager, tmp_path):
    """String 'offline' is coerced to ConnectionMode.offline.

    Same two-step pattern as the enum test above — online run
    populates the cache, then offline run loads it.
    """
    from deriva_ml import ConnectionMode, DerivaML
    catalog_manager.reset()
    DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode="online",
    )
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


def test_derivaml_config_accepts_mode():
    """DerivaMLConfig mirrors DerivaML.__init__'s mode parameter.

    Hydra-zen instantiation goes through this config class; without
    mode here, offline mode would be unreachable via DerivaML.instantiate().
    """
    from unittest.mock import patch

    from deriva_ml import ConnectionMode
    from deriva_ml.core.config import DerivaMLConfig

    # DerivaMLConfig's model validator touches HydraConfig; stub it so we
    # can construct the config outside a Hydra run (same pattern used in
    # tests/core/test_hydra_zen_config.py).
    with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
        mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_output"

        # Default is online.
        c1 = DerivaMLConfig(hostname="h", catalog_id="1")
        assert ConnectionMode(c1.mode) is ConnectionMode.online

        # Accepts enum.
        c2 = DerivaMLConfig(hostname="h", catalog_id="1", mode=ConnectionMode.offline)
        assert ConnectionMode(c2.mode) is ConnectionMode.offline

        # Accepts string (pydantic coerces to StrEnum).
        c3 = DerivaMLConfig(hostname="h", catalog_id="1", mode="offline")
        assert ConnectionMode(c3.mode) is ConnectionMode.offline

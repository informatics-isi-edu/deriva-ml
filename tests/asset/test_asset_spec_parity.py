"""Field-parity check for :class:`AssetSpec` and :class:`AssetSpecConfig`.

``AssetSpec`` (regular Python) and ``AssetSpecConfig`` (hydra-zen
dataclass interface) are documented as the two equivalent ways to
declare an asset. They must stay field-for-field equivalent:
anything expressible via ``AssetSpec`` must also be expressible via
``AssetSpecConfig`` and vice versa.

Pre-fix, ``AssetSpecConfig`` was missing the ``asset_role`` field
that ``AssetSpec`` already had. Users configuring assets through a
hydra-zen store had no way to mark an asset as an ``Output`` —
silent feature loss surfaced only when someone tried to express
the role through the config interface and discovered it was
ignored.

This test enforces the parity contract as a CI gate. Modelled on
``tests/test_dataset_like_signature_parity.py`` (audit §3.C).
"""

from __future__ import annotations

import dataclasses

from deriva_ml.asset.aux_classes import AssetSpec, AssetSpecConfig


def test_assetspec_assetspecconfig_field_parity() -> None:
    """Every ``AssetSpec`` field appears on ``AssetSpecConfig`` (same name & default).

    Strict-direction parity: ``AssetSpec`` is the canonical model;
    ``AssetSpecConfig`` is its hydra-zen surface and must mirror it.
    Going the other way is technically allowed (the config can carry
    purely-presentation hints like ``_target_``) but we don't see a
    use case yet; flag it explicitly if it ever comes up.
    """
    spec_fields = AssetSpec.model_fields
    config_fields = {f.name: f for f in dataclasses.fields(AssetSpecConfig)}

    missing = sorted(set(spec_fields) - set(config_fields))
    assert not missing, (
        f"AssetSpecConfig is missing fields that AssetSpec declares: "
        f"{missing}. The hydra-zen interface must mirror the Python "
        f"interface; otherwise hydra-zen users silently can't express "
        f"those asset semantics. The pre-fix gap that prompted this "
        f"test: ``asset_role`` lived on AssetSpec but not on "
        f"AssetSpecConfig, so configs had no way to mark an asset "
        f"as an Output."
    )

    # Defaults must match. ``model_fields[name].default`` is
    # PydanticUndefined for required fields; ``dataclasses.fields``
    # uses ``dataclasses.MISSING`` — translate both to a sentinel
    # so the comparison stays clean.
    REQUIRED = object()

    def _spec_default(name: str):
        d = spec_fields[name].default
        # Pydantic: ``PydanticUndefined`` means "required".
        from pydantic_core import PydanticUndefined

        return REQUIRED if d is PydanticUndefined else d

    def _config_default(name: str):
        d = config_fields[name].default
        return REQUIRED if d is dataclasses.MISSING else d

    mismatched = {
        name: (_spec_default(name), _config_default(name))
        for name in spec_fields
        if _spec_default(name) != _config_default(name)
    }
    assert not mismatched, (
        f"Default-value mismatches between AssetSpec and AssetSpecConfig: "
        f"{mismatched}. A drifted default means a hydra-zen-configured "
        f"asset behaves differently from one declared in Python."
    )


def test_assetspecconfig_carries_asset_role() -> None:
    """``AssetSpecConfig`` exposes ``asset_role`` with default ``'Input'``.

    Direct regression for the bug. Even with the parity test
    above, an explicit positive assertion makes the failure mode
    obvious in CI output: if the field disappears, this test names
    it.
    """
    config_field_names = {f.name for f in dataclasses.fields(AssetSpecConfig)}
    assert "asset_role" in config_field_names, (
        "AssetSpecConfig must declare ``asset_role`` for parity with "
        "AssetSpec. Without it, hydra-zen configs can't express "
        "Output assets."
    )

    cfg = AssetSpecConfig(rid="3JSE")
    assert cfg.asset_role == "Input", (
        f"AssetSpecConfig.asset_role default should be 'Input' "
        f"(matching AssetSpec); got {cfg.asset_role!r}."
    )

    # And it accepts a non-default value.
    cfg_out = AssetSpecConfig(rid="3JSE", asset_role="Output")
    assert cfg_out.asset_role == "Output"

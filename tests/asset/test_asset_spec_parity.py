"""Contract check for :class:`AssetSpec` and :class:`AssetSpecConfig`.

``AssetSpec`` (regular Python) is the canonical runtime model for an asset
reference. ``AssetSpecConfig`` (hydra-zen dataclass interface) is its
*configuration* surface — it exposes only the attributes that are meaningful
to configure.

The two are intentionally **not** field-for-field identical. ``AssetSpec``
carries ``asset_role`` (``Input`` / ``Output``), but that role is determined
by **context**, never configured:

- An asset referenced in an execution's input configuration is an *input*.
- Assets written via ``commit_output_assets`` are *outputs*.

So ``asset_role`` must NOT appear on ``AssetSpecConfig``: there is no scenario
where a config author legitimately marks a config-declared asset as an Output,
and exposing it as a configurable ``Literal``-typed field additionally broke
structured-config registration under omegaconf < 2.4 (which cannot serialize
``Literal`` annotations). This test pins that contract as a CI gate.
"""

from __future__ import annotations

import dataclasses

from deriva_ml.asset.aux_classes import AssetSpec, AssetSpecConfig

# Fields that exist on AssetSpec but are intentionally NOT configurable.
# asset_role is set by context (input config vs. commit_output_assets), so it
# is excluded from the hydra-zen configuration surface by design.
CONTEXT_DETERMINED_FIELDS = {"asset_role"}


def test_config_exposes_configurable_fields_with_matching_defaults() -> None:
    """Every *configurable* AssetSpec field appears on AssetSpecConfig.

    Parity holds for all fields EXCEPT the context-determined ones
    (``asset_role``). A configurable AssetSpec field missing from the config
    would be silent feature loss; a context-determined field present on the
    config would invite authors to set something the system ignores (and
    reintroduce the omegaconf ``Literal`` break).
    """
    spec_fields = AssetSpec.model_fields
    config_fields = {f.name: f for f in dataclasses.fields(AssetSpecConfig)}

    expected_configurable = set(spec_fields) - CONTEXT_DETERMINED_FIELDS

    missing = sorted(expected_configurable - set(config_fields))
    assert not missing, (
        f"AssetSpecConfig is missing configurable fields AssetSpec declares: "
        f"{missing}. The hydra-zen interface must expose every field a config "
        f"author can meaningfully set (e.g. ``rid``, ``cache``)."
    )

    # Defaults must match for the shared configurable fields.
    REQUIRED = object()

    def _spec_default(name: str):
        from pydantic_core import PydanticUndefined

        d = spec_fields[name].default
        return REQUIRED if d is PydanticUndefined else d

    def _config_default(name: str):
        d = config_fields[name].default
        return REQUIRED if d is dataclasses.MISSING else d

    mismatched = {
        name: (_spec_default(name), _config_default(name))
        for name in expected_configurable
        if _spec_default(name) != _config_default(name)
    }
    assert not mismatched, (
        f"Default-value mismatches between AssetSpec and AssetSpecConfig: "
        f"{mismatched}. A drifted default means a hydra-zen-configured asset "
        f"behaves differently from one declared in Python."
    )


def test_config_does_not_expose_asset_role() -> None:
    """``AssetSpecConfig`` must NOT expose ``asset_role`` (context-determined).

    Direct regression guard: asset role is set by where the asset is used
    (input config vs. ``commit_output_assets``), not by the config author.
    Re-adding ``asset_role`` here would also reintroduce the omegaconf
    ``Literal`` structured-config break.
    """
    config_field_names = {f.name for f in dataclasses.fields(AssetSpecConfig)}
    assert "asset_role" not in config_field_names, (
        "AssetSpecConfig must not declare ``asset_role`` — an asset's role is "
        "determined by context (input configuration vs. commit_output_assets), "
        "not configured. See AssetSpecConfig docstring."
    )


def test_config_instantiates_to_assetspec_with_input_default() -> None:
    """A configured asset instantiates to an AssetSpec defaulting to Input.

    ``rid`` + ``cache`` round-trip; the runtime ``asset_role`` takes its
    ``"Input"`` default (the role is then set by the consuming/producing
    operation).
    """
    from hydra_zen import instantiate

    spec = instantiate(AssetSpecConfig(rid="3JSE", cache=True))
    assert isinstance(spec, AssetSpec)
    assert spec.rid == "3JSE"
    assert spec.cache is True
    assert spec.asset_role == "Input"

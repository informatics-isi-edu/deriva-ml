"""Contract check for :class:`AssetSpec` and :class:`AssetSpecConfig`.

``AssetSpec`` (regular Python) is the canonical runtime model for an asset
reference. ``AssetSpecConfig`` (hydra-zen dataclass interface) is its
*configuration* surface — it exposes only the attributes that are meaningful
to configure.

Asset **role is determined by context, never specified** (the same rule as
datasets, whose role is structural):

- An asset referenced in an execution's input configuration is an *input*.
- Assets written via ``commit_output_assets`` are *outputs*.

So neither ``AssetSpec`` nor ``AssetSpecConfig`` carries an ``asset_role``
field — it was removed entirely (it had been a no-op runtime field never read
to set the edge). The two are now full-parity on configurable fields. This
test pins that contract as a CI gate.
"""

from __future__ import annotations

import dataclasses

from deriva_ml.asset.aux_classes import AssetSpec, AssetSpecConfig

# Fields that exist on AssetSpec but are intentionally NOT configurable.
# (Empty: asset_role was removed entirely — role is context-derived, never a
# field on either AssetSpec or AssetSpecConfig. AssetSpec and AssetSpecConfig
# are now full-parity on configurable fields.)
CONTEXT_DETERMINED_FIELDS: set[str] = set()


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


def test_assetspec_has_no_asset_role_field() -> None:
    """``AssetSpec`` must NOT carry an ``asset_role`` field at all.

    Asset role is determined by context (consuming an input vs. producing via
    ``commit_output_assets``) and is never specified by the caller — the same
    rule as datasets, whose role is structural. ``asset_role`` was previously a
    no-op runtime field (never read to set the edge), which only invited
    callers to set something the system ignores. Removed entirely.
    """
    assert "asset_role" not in AssetSpec.model_fields, (
        "AssetSpec must not declare ``asset_role`` — role is context-derived, "
        "never user-specified."
    )


def test_assetspec_rejects_asset_role_kwarg() -> None:
    """Passing ``asset_role`` to ``AssetSpec`` is rejected (not silently
    ignored) — a typo or a stale call that names a role fails loudly."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AssetSpec(rid="1-ABCD", asset_role="Output")


def test_config_does_not_expose_asset_role() -> None:
    """``AssetSpecConfig`` must NOT expose ``asset_role`` (context-determined).

    Direct regression guard: asset role is set by where the asset is used
    (input config vs. ``commit_output_assets``), not by the config author.
    """
    config_field_names = {f.name for f in dataclasses.fields(AssetSpecConfig)}
    assert "asset_role" not in config_field_names, (
        "AssetSpecConfig must not declare ``asset_role`` — an asset's role is "
        "determined by context (input configuration vs. commit_output_assets), "
        "not configured. See AssetSpecConfig docstring."
    )


def test_config_instantiates_to_assetspec() -> None:
    """A configured asset instantiates to an AssetSpec.

    ``rid`` + ``cache`` round-trip. There is no ``asset_role`` field — role is
    set by the consuming/producing operation, never carried on the spec.
    """
    from hydra_zen import instantiate

    spec = instantiate(AssetSpecConfig(rid="3JSE", cache=True))
    assert isinstance(spec, AssetSpec)
    assert spec.rid == "3JSE"
    assert spec.cache is True
    assert not hasattr(spec, "asset_role")

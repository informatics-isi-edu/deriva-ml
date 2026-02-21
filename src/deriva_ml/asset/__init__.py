"""Asset management module for DerivaML.

This module provides classes for managing assets (files) in a Deriva catalog:

- Asset: Live catalog access to asset records
- AssetFilePath: Extended Path for staging files during execution
- AssetSpec: Specification for asset references in configurations
"""

from .asset import Asset
from .aux_classes import AssetFilePath, AssetSpec, AssetSpecConfig

__all__ = [
    "Asset",
    "AssetFilePath",
    "AssetSpec",
    "AssetSpecConfig",
]

"""Asset management module for DerivaML.

This module provides classes for managing assets (files) in a Deriva catalog:

- Asset: Live catalog access to asset records
- AssetFilePath: Extended Path for staging files during execution
- AssetSpec: Specification for asset references in configurations
- AssetRecord: Base class for typed asset metadata
- AssetManifest: Persistent JSON manifest for crash-safe asset tracking
"""

from .asset import Asset
from .asset_record import AssetRecord, asset_record_class
from .aux_classes import AssetFilePath, AssetSpec, AssetSpecConfig, LocalFile, LocalFileConfig
from .manifest import AssetEntry, AssetManifest

__all__ = [
    "Asset",
    "AssetEntry",
    "AssetFilePath",
    "AssetManifest",
    "AssetRecord",
    "AssetSpec",
    "AssetSpecConfig",
    "LocalFile",
    "LocalFileConfig",
    "asset_record_class",
]

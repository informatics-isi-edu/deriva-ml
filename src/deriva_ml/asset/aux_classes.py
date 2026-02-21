"""Auxiliary classes for asset management in DerivaML.

This module defines helper classes for asset operations including:
- AssetFilePath: Extended Path for in-flight asset staging
- AssetSpec: Specification for asset references in configurations
- AssetSpecConfig: Hydra-zen config interface for AssetSpec
"""

from pathlib import Path
from typing import Any

from hydra_zen import hydrated_dataclass
from pydantic import BaseModel, ConfigDict, model_validator

from deriva_ml.core.definitions import RID


class AssetFilePath(Path):
    """Extended Path class for managing asset files during execution.

    Represents a file path with additional metadata about its role as an asset
    in the catalog. This class extends the standard Path class to include
    information about the asset's catalog representation and type.

    This is primarily used during execution for staging files before upload
    or after download. For catalog-backed asset operations, use the Asset class.

    Attributes:
        asset_table: Name of the asset table in the catalog (e.g., "Image", "Model").
        file_name: Name of the local file containing the asset.
        asset_metadata: Additional columns beyond URL, Length, and checksum.
        asset_types: Terms from the Asset_Type controlled vocabulary.
        asset_rid: Resource Identifier if uploaded to an asset table.

    Example:
        >>> path = AssetFilePath(
        ...     "/path/to/file.txt",
        ...     asset_table="Execution_Asset",
        ...     file_name="results.txt",
        ...     asset_metadata={"version": "1.0"},
        ...     asset_types=["Model_File"]
        ... )
    """

    def __init__(
        self,
        asset_path: str | Path,
        asset_table: str,
        file_name: str,
        asset_metadata: dict[str, Any],
        asset_types: list[str] | str,
        asset_rid: RID | None = None,
    ):
        """Initialize an AssetFilePath instance.

        Args:
            asset_path: Local path to the asset file.
            asset_table: Name of the asset table in the catalog.
            file_name: Name of the local file.
            asset_metadata: Additional metadata columns.
            asset_types: One or more asset type terms.
            asset_rid: Optional Resource Identifier if already in catalog.
        """
        super().__init__(asset_path)
        self.asset_table = asset_table
        self.file_name = file_name
        self.asset_metadata = asset_metadata
        self.asset_types = asset_types if isinstance(asset_types, list) else [asset_types]
        self.asset_rid = asset_rid

    def with_segments(self, *pathsegments):
        """Return a plain Path for derived path operations.

        Path methods like resolve(), with_name(), parent, etc. internally call
        type(self)(*args) to create new path objects. Since AssetFilePath requires
        extra constructor arguments, this would fail. Returning a plain Path
        avoids the issue while preserving correct path behavior.
        """
        return Path(*pathsegments)

    # Backward compatibility alias
    @property
    def asset_name(self) -> str:
        """Alias for asset_table (backward compatibility)."""
        return self.asset_table


class AssetSpec(BaseModel):
    """Specification for an asset in execution configurations.

    Used to reference assets as inputs to executions, similar to how
    DatasetSpec is used for datasets. Supports optional checksum-based
    caching for large assets like model weights.

    Attributes:
        rid: Resource Identifier of the asset.
        asset_role: Role of the asset ("Input" or "Output"). Defaults to "Input".
        cache: If True, cache the downloaded asset by MD5 checksum in the
            DerivaML cache directory. Cached assets are reused across executions
            when the checksum matches, avoiding repeated downloads of large files.

    Example:
        >>> spec = AssetSpec(rid="3JSE")
        >>> spec = AssetSpec(rid="3JSE", cache=True)  # enable caching
    """

    rid: RID
    asset_role: str = "Input"
    cache: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _check_bare_rid(cls, data: Any) -> dict[str, str | bool]:
        """Allow bare RID string as shorthand."""
        return {"rid": data} if isinstance(data, str) else data


# Interface for hydra-zen
@hydrated_dataclass(AssetSpec)
class AssetSpecConfig:
    """Hydra-zen configuration interface for AssetSpec.

    Use in hydra-zen store definitions to specify assets with caching:

        >>> from hydra_zen import store
        >>> asset_store = store(group="assets")
        >>> asset_store(
        ...     [AssetSpecConfig(rid="6-EPNR", cache=True)],
        ...     name="cached_weights",
        ... )
    """

    rid: str
    cache: bool = False

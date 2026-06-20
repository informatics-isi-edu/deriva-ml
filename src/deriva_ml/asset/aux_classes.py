"""Auxiliary classes for asset management in DerivaML.

This module defines helper classes for asset operations including:
- AssetFilePath: Extended Path for in-flight asset staging with manifest-backed metadata
- AssetSpec: Specification for asset references in configurations
- AssetSpecConfig: Hydra-zen config interface for AssetSpec
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra_zen import hydrated_dataclass
from pydantic import BaseModel, ConfigDict, model_validator

from deriva_ml.core.definitions import RID
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.validation import VALIDATION_CONFIG

if TYPE_CHECKING:
    from deriva_ml.asset.asset_record import AssetRecord
    from deriva_ml.asset.manifest import AssetManifest

logger = get_logger(__name__)


class AssetFilePath(Path):
    """Extended Path class for managing asset files during execution.

    Represents a file path with additional metadata about its role as an asset
    in the catalog. Metadata is backed by a persistent JSON manifest for crash
    safety. Metadata can be set incrementally after creation using the
    ``metadata`` property or ``set_asset_types()`` method.

    Attributes:
        asset_table: Name of the asset table in the catalog (e.g., "Image", "Model").
        file_name: Name of the local file containing the asset.
        asset_metadata: Additional columns beyond URL, Length, and checksum.
        asset_types: Terms from the Asset_Type controlled vocabulary.
        asset_rid: Resource Identifier if uploaded to an asset table.

    Example:
        >>> path = exe.asset_file_path("Image", "scan.jpg")  # doctest: +SKIP
        >>> # Set typed metadata via AssetRecord
        >>> ImageAsset = ml.asset_record_class("Image")  # doctest: +SKIP
        >>> path.metadata = ImageAsset(Subject="2-DEF", Acquisition_Date="2026-01-15")  # doctest: +SKIP
        >>> path.set_asset_types(["Training_Data"])  # doctest: +SKIP
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
        # Optional manifest reference — set by Execution.asset_file_path()
        self._manifest: AssetManifest | None = None
        self._manifest_key: str | None = None

    def _bind_manifest(self, manifest: AssetManifest, key: str) -> None:
        """Bind this path to a manifest for write-through updates.

        Called internally by Execution.asset_file_path().
        """
        self._manifest = manifest
        self._manifest_key = key

    @property
    def metadata(self) -> dict[str, Any]:
        """Current metadata dict. If bound to manifest, reads from manifest.

        Uses a point-query (:meth:`AssetManifest.get_asset`) rather than the
        full ``manifest.assets`` dict — accessing this attribute in a hot loop
        otherwise becomes N×SELECT-all against SQLite.
        """
        if self._manifest and self._manifest_key:
            entry = self._manifest.get_asset(self._manifest_key)
            if entry:
                return dict(entry.metadata)
        return dict(self.asset_metadata)

    @metadata.setter
    def metadata(self, record: AssetRecord | dict[str, Any]) -> None:
        """Set metadata from an AssetRecord or dict.

        If an AssetRecord is provided, uses model_dump() to extract values.
        Updates both the local attribute and the manifest (write-through + fsync).

        Args:
            record: An AssetRecord instance or dict of column → value.
        """
        if hasattr(record, "model_dump"):
            # It's a Pydantic model (AssetRecord subclass)
            metadata_dict = {k: v for k, v in record.model_dump().items() if v is not None}
        else:
            metadata_dict = dict(record)

        self.asset_metadata = metadata_dict

        if self._manifest and self._manifest_key:
            self._manifest.update_asset_metadata(self._manifest_key, metadata_dict)
            logger.debug(f"Updated manifest metadata for {self._manifest_key}")

    def set_asset_types(self, types: list[str]) -> None:
        """Set asset types. Updates both local attribute and manifest.

        Args:
            types: List of terms from the Asset_Type controlled vocabulary.
        """
        self.asset_types = list(types)

        if self._manifest and self._manifest_key:
            self._manifest.update_asset_types(self._manifest_key, self.asset_types)
            logger.debug(f"Updated manifest asset_types for {self._manifest_key}")

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
    """Specification for an asset consumed as an input to an execution.

    Used to reference assets as inputs to executions, similar to how
    DatasetSpec is used for datasets. Supports optional checksum-based
    caching for large assets like model weights.

    An asset's **role is determined by context, never by this spec**: an asset
    referenced here (in an execution's input configuration) is an *input*;
    assets a run *produces* are written via ``commit_output_assets`` as
    *outputs*. There is intentionally no ``asset_role`` field — role is not a
    caller choice (the same rule as datasets, whose role is structural). The
    model forbids extra fields, so a stray ``asset_role=`` is rejected rather
    than silently ignored.

    Attributes:
        rid: Resource Identifier of the asset.
        cache: If True, cache the downloaded asset by MD5 checksum in the
            DerivaML cache directory. Cached assets are reused across executions
            when the checksum matches, avoiding repeated downloads of large files.

    Example:
        >>> spec = AssetSpec(rid="3JSE")
        >>> spec = AssetSpec(rid="3JSE", cache=True)  # enable caching
    """

    rid: RID
    cache: bool = False

    model_config = ConfigDict(**VALIDATION_CONFIG, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _check_bare_rid(cls, data: Any) -> dict[str, str | bool]:
        """Allow bare RID string as shorthand."""
        return {"rid": data} if isinstance(data, str) else data


# Interface for hydra-zen
@hydrated_dataclass(AssetSpec)
class AssetSpecConfig:
    """Hydra-zen configuration interface for ``AssetSpec``.

    Exposes the asset attributes that are meaningful to *configure*: the
    asset ``rid`` and whether to ``cache`` it. There is no role to configure —
    asset role is determined by **context**, never specified: an asset
    referenced in an execution's input configuration is an *input*, and assets
    written via ``commit_output_assets`` are *outputs*. (``AssetSpec`` carries
    no ``asset_role`` field at all, so the config has nothing to mirror; full
    field parity holds.)

    Use in hydra-zen store definitions to specify assets, optionally cached:

        >>> from hydra_zen import store  # doctest: +SKIP
        >>> asset_store = store(group="assets")  # doctest: +SKIP
        >>> asset_store(  # doctest: +SKIP
        ...     [AssetSpecConfig(rid="6-EPNR", cache=True)],
        ...     name="cached_weights",
        ... )

    Attributes:
        rid: Resource Identifier of the asset. Mirrors ``AssetSpec.rid``.
        cache: If True, cache the downloaded asset by MD5 checksum in
            the DerivaML cache directory. Mirrors ``AssetSpec.cache``.
    """

    rid: str
    cache: bool = False

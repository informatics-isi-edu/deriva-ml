"""Dynamically generated Pydantic models for typed asset metadata.

Follows the same pattern as ``Feature.feature_record_class()`` in feature.py.
Each asset table gets a dynamically generated Pydantic model with fields
matching the table's metadata columns (everything beyond the standard
Filename, URL, Length, MD5, Description, and system columns).

Example:
    >>> ImageAsset = ml.asset_record_class("Image")
    >>> record = ImageAsset(Subject="2-DEF", Acquisition_Date="2026-01-15")
    >>> path = exe.asset_file_path("Image", "scan001.jpg", metadata=record)
"""

from __future__ import annotations

from typing import Any, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel, create_model

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


class AssetRecord(BaseModel):
    """Base class for dynamically generated asset metadata models.

    Subclasses are created by ``asset_record_class()`` with fields derived
    from the asset table's metadata columns. Fields are typed according
    to their database column type and nullable columns are Optional.

    Use ``model_dump()`` to serialize for manifest storage.
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


def _map_column_type(typename: str) -> Type:
    """Map an ERMrest column type to a Python type."""
    match typename:
        case "text" | "markdown" | "longtext":
            return str
        case "int2" | "int4" | "int8" | "serial2" | "serial4" | "serial8":
            return int
        case "float4" | "float8" | "numeric":
            return float
        case "boolean":
            return bool
        case "date" | "timestamp" | "timestamptz":
            return str
        case "json" | "jsonb":
            return Any
        case _:
            return str


def asset_record_class(model: DerivaModel, asset_table_name: str) -> type[AssetRecord]:
    """Create a dynamically generated Pydantic model for an asset table's metadata.

    The returned class is a subclass of AssetRecord with fields derived from
    the asset table's metadata columns (non-system, non-standard-asset columns).

    Args:
        model: DerivaModel instance for schema introspection.
        asset_table_name: Name of the asset table (e.g., "Image", "Model").

    Returns:
        An AssetRecord subclass with validated fields matching the table's metadata.

    Raises:
        DerivaMLTableTypeError: If the table is not an asset table.
    """
    table = model.name_to_table(asset_table_name)
    metadata_col_names = model.asset_metadata(asset_table_name)

    # Build field definitions from catalog column metadata
    fields: dict[str, tuple] = {}
    for col in table.columns:
        if col.name not in metadata_col_names:
            continue

        python_type = _map_column_type(col.type.typename)

        if col.nullok:
            fields[col.name] = (Optional[python_type], col.default if col.default else None)
        else:
            fields[col.name] = (python_type, ...)

    class_name = f"{asset_table_name}AssetRecord"

    record_class = create_model(
        class_name,
        __base__=AssetRecord,
        __doc__=(
            f"Typed metadata for the {asset_table_name} asset table.\n"
            f"Metadata columns: {', '.join(sorted(metadata_col_names))}"
        ),
        **fields,
    )

    return record_class

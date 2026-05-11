"""Catalog provenance metadata.

This module owns the catalog-level provenance annotation —
how the catalog was created, by whom, and (for clones) with what
parameters. The annotation is stored under the
``tag:deriva-ml.org,2025:catalog-provenance`` URL on the catalog
model.

The implementation lived in ``catalog/clone.py`` before the
deriva.bag migration. It's been lifted into its own module because
provenance is orthogonal to the clone *algorithm*: a fresh catalog
gets a ``CatalogCreationMethod.CREATE`` provenance row regardless
of whether the data movement happened via the legacy clone or the
new bag pipeline.

Public surface:

- :class:`CatalogCreationMethod`: enum tagging *how* the catalog
  was made.
- :class:`CloneDetails`: structural record of the clone parameters
  (only populated when ``creation_method == CLONE``).
- :class:`CatalogProvenance`: the full annotation payload.
- :func:`set_catalog_provenance`: write the annotation.
- :func:`get_catalog_provenance`: read it back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from deriva.core import ErmrestCatalog
from deriva.core.utils.core_utils import urlquote

logger = logging.getLogger(__name__)


#: ERMrest annotation URL used to attach the provenance payload to
#: a catalog model. Kept stable across the bag-migration so older
#: catalogs read by newer code (and vice versa) interoperate.
CATALOG_PROVENANCE_URL = "tag:deriva-ml.org,2025:catalog-provenance"


class CatalogCreationMethod(Enum):
    """How a catalog was created."""

    CLONE = "clone"
    """Cloned from another catalog (legacy or bag-pipeline)."""

    CREATE = "create"
    """Created programmatically (e.g., ``create_catalog``)."""

    SCHEMA = "schema"
    """Created from a schema definition (no row data)."""

    UNKNOWN = "unknown"
    """Pre-existing catalog or unknown provenance."""


@dataclass
class CloneDetails:
    """Clone-specific provenance details.

    Populated only when :attr:`CatalogProvenance.creation_method`
    is :attr:`CatalogCreationMethod.CLONE`. Carries the source
    catalog reference plus the parameters / statistics that
    describe what the clone did.

    The field set is wide because the legacy clone reported on
    many distinct knobs (orphan strategy, asset mode, schema
    exclusions, row counts, oversized truncations, FK pruning).
    Most fields are unused by the new bag-pipeline clone — its
    full record lives in the bag's ``metadata/`` provenance file
    — but the schema stays in place so older clone artifacts
    parse correctly.
    """

    source_hostname: str
    source_catalog_id: str
    source_snapshot: str | None = None
    source_schema_url: str | None = None
    orphan_strategy: str = "fail"
    truncate_oversized: bool = False
    prune_hidden_fkeys: bool = False
    schema_only: bool = False
    asset_mode: str = "refs"
    exclude_schemas: list[str] = field(default_factory=list)
    exclude_objects: list[str] = field(default_factory=list)
    add_ml_schema: bool = False
    copy_annotations: bool = True
    copy_policy: bool = True
    reinitialize_dataset_versions: bool = True
    rows_copied: int = 0
    rows_skipped: int = 0
    skipped_rids: list[str] = field(default_factory=list)
    truncated_count: int = 0
    orphan_rows_removed: int = 0
    orphan_rows_nullified: int = 0
    fkeys_pruned: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the annotation."""
        result = {
            "source_hostname": self.source_hostname,
            "source_catalog_id": self.source_catalog_id,
            "source_snapshot": self.source_snapshot,
            "source_schema_url": self.source_schema_url,
            "orphan_strategy": self.orphan_strategy,
            "truncate_oversized": self.truncate_oversized,
            "prune_hidden_fkeys": self.prune_hidden_fkeys,
            "schema_only": self.schema_only,
            "asset_mode": self.asset_mode,
            "exclude_schemas": self.exclude_schemas,
            "exclude_objects": self.exclude_objects,
            "add_ml_schema": self.add_ml_schema,
            "copy_annotations": self.copy_annotations,
            "copy_policy": self.copy_policy,
            "reinitialize_dataset_versions": self.reinitialize_dataset_versions,
            "rows_copied": self.rows_copied,
            "rows_skipped": self.rows_skipped,
            "truncated_count": self.truncated_count,
            "orphan_rows_removed": self.orphan_rows_removed,
            "orphan_rows_nullified": self.orphan_rows_nullified,
            "fkeys_pruned": self.fkeys_pruned,
        }
        if self.skipped_rids:
            result["skipped_rids"] = self.skipped_rids
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CloneDetails":
        """Reconstruct from the dict form stored on the annotation."""
        return cls(
            source_hostname=data.get("source_hostname", ""),
            source_catalog_id=data.get("source_catalog_id", ""),
            source_snapshot=data.get("source_snapshot"),
            source_schema_url=data.get("source_schema_url"),
            orphan_strategy=data.get("orphan_strategy", "fail"),
            truncate_oversized=data.get("truncate_oversized", False),
            prune_hidden_fkeys=data.get("prune_hidden_fkeys", False),
            schema_only=data.get("schema_only", False),
            asset_mode=data.get("asset_mode", "refs"),
            exclude_schemas=data.get("exclude_schemas", []),
            exclude_objects=data.get("exclude_objects", []),
            add_ml_schema=data.get("add_ml_schema", False),
            copy_annotations=data.get("copy_annotations", True),
            copy_policy=data.get("copy_policy", True),
            reinitialize_dataset_versions=data.get(
                "reinitialize_dataset_versions", True
            ),
            rows_copied=data.get("rows_copied", 0),
            rows_skipped=data.get("rows_skipped", 0),
            skipped_rids=data.get("skipped_rids", []),
            truncated_count=data.get("truncated_count", 0),
            orphan_rows_removed=data.get("orphan_rows_removed", 0),
            orphan_rows_nullified=data.get("orphan_rows_nullified", 0),
            fkeys_pruned=data.get("fkeys_pruned", 0),
        )


@dataclass
class CatalogProvenance:
    """Catalog-level provenance annotation payload.

    Stored on the catalog model under
    :data:`CATALOG_PROVENANCE_URL`. Records who/when/how/with-what
    a catalog came into being.
    """

    creation_method: CatalogCreationMethod
    created_at: str
    hostname: str
    catalog_id: str
    created_by: str | None = None
    name: str | None = None
    description: str | None = None
    workflow_url: str | None = None
    workflow_version: str | None = None
    clone_details: CloneDetails | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "creation_method": self.creation_method.value,
            "created_at": self.created_at,
            "hostname": self.hostname,
            "catalog_id": self.catalog_id,
            "created_by": self.created_by,
            "name": self.name,
            "description": self.description,
            "workflow_url": self.workflow_url,
            "workflow_version": self.workflow_version,
        }
        if self.clone_details:
            result["clone_details"] = self.clone_details.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogProvenance":
        clone_details = None
        if data.get("clone_details"):
            clone_details = CloneDetails.from_dict(data["clone_details"])
        method_str = data.get("creation_method", "unknown")
        try:
            creation_method = CatalogCreationMethod(method_str)
        except ValueError:
            creation_method = CatalogCreationMethod.UNKNOWN
        return cls(
            creation_method=creation_method,
            created_at=data.get("created_at", ""),
            hostname=data.get("hostname", ""),
            catalog_id=data.get("catalog_id", ""),
            created_by=data.get("created_by"),
            name=data.get("name"),
            description=data.get("description"),
            workflow_url=data.get("workflow_url"),
            workflow_version=data.get("workflow_version"),
            clone_details=clone_details,
        )

    @property
    def is_clone(self) -> bool:
        """``True`` when the catalog was cloned from another."""
        return (
            self.creation_method == CatalogCreationMethod.CLONE
            and self.clone_details is not None
        )


def set_catalog_provenance(
    catalog: ErmrestCatalog,
    name: str | None = None,
    description: str | None = None,
    workflow_url: str | None = None,
    workflow_version: str | None = None,
    creation_method: CatalogCreationMethod = CatalogCreationMethod.CREATE,
) -> CatalogProvenance:
    """Attach a :class:`CatalogProvenance` annotation to a catalog.

    Args:
        catalog: The catalog to annotate.
        name: Optional human-readable catalog name.
        description: Optional description of the catalog's purpose.
        workflow_url: Optional URL of the workflow / script that
            created the catalog. The deriva-ml convention is the
            git remote URL of the producer repository.
        workflow_version: Optional version tag of the workflow.
        creation_method: How the catalog was created. Defaults to
            :attr:`CatalogCreationMethod.CREATE`.

    Returns:
        The :class:`CatalogProvenance` object that was written.
    """
    created_by = None
    try:
        session_info = catalog.get("/authn/session").json()
        if session_info and "client" in session_info:
            client = session_info["client"]
            created_by = client.get("display_name") or client.get("id")
    except Exception as e:
        logger.debug(f"Could not retrieve session info for provenance: {e}")

    try:
        catalog_info = catalog.get("/").json()
        hostname = catalog_info.get("meta", {}).get("host", "")
        catalog_id = str(catalog.catalog_id)
    except Exception as e:
        logger.debug(f"Could not retrieve catalog info for provenance: {e}")
        hostname = ""
        catalog_id = str(catalog.catalog_id)

    provenance = CatalogProvenance(
        creation_method=creation_method,
        created_at=datetime.now(timezone.utc).isoformat(),
        hostname=hostname,
        catalog_id=catalog_id,
        created_by=created_by,
        name=name,
        description=description,
        workflow_url=workflow_url,
        workflow_version=workflow_version,
    )

    _write_provenance_annotation(catalog, provenance)
    return provenance


def get_catalog_provenance(
    catalog: ErmrestCatalog,
) -> CatalogProvenance | None:
    """Return the catalog's provenance annotation, or ``None``."""
    try:
        model = catalog.getCatalogModel()
        provenance_data = model.annotations.get(CATALOG_PROVENANCE_URL)
        if provenance_data:
            return CatalogProvenance.from_dict(provenance_data)
    except Exception as e:
        logger.debug(f"Could not get catalog provenance: {e}")
    return None


def _write_provenance_annotation(
    catalog: ErmrestCatalog,
    provenance: CatalogProvenance,
) -> None:
    """Write the annotation to the catalog. Best-effort."""
    try:
        catalog.put(
            f"/annotation/{urlquote(CATALOG_PROVENANCE_URL)}",
            json=provenance.to_dict(),
        )
        logger.info("Set catalog provenance annotation")
    except Exception as e:
        logger.warning(f"Failed to set catalog provenance annotation: {e}")


__all__ = [
    "CATALOG_PROVENANCE_URL",
    "CatalogCreationMethod",
    "CatalogProvenance",
    "CloneDetails",
    "get_catalog_provenance",
    "set_catalog_provenance",
]

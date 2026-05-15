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

from datetime import datetime, timezone
from enum import Enum

from deriva.core import ErmrestCatalog
from pydantic import BaseModel, Field

from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.validation import VALIDATION_CONFIG

logger = get_logger(__name__)
#: ERMrest annotation URL used to attach the provenance payload to
#: a catalog model. Kept stable across the bag-migration so older
#: catalogs read by newer code (and vice versa) interoperate.
CATALOG_PROVENANCE_URL = "tag:deriva-ml.org,2025:catalog-provenance"


class CatalogCreationMethod(str, Enum):
    """How a catalog was created.

    Inherits from ``str`` so values serialize naturally inside the
    Pydantic annotation payload without needing custom encoders.
    """

    CLONE = "clone"
    """Cloned from another catalog (legacy or bag-pipeline)."""

    CREATE = "create"
    """Created programmatically (e.g., ``create_catalog``)."""

    SCHEMA = "schema"
    """Created from a schema definition (no row data)."""

    UNKNOWN = "unknown"
    """Pre-existing catalog or unknown provenance."""


class CloneDetails(BaseModel):
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

    The localization-leg fields (``assets_localized``,
    ``assets_localized_at``, ``asset_source_hostname``,
    ``assets_copied``, ``assets_skipped``, ``assets_failed``)
    record the split-phase asset-copy step
    (:func:`~deriva_ml.catalog.localize.localize_assets`). They
    are populated when phase two completes — phase one (clone via
    bag) leaves them at their defaults.

    Example:
        >>> details = CloneDetails(
        ...     source_hostname="src.example.org",
        ...     source_catalog_id="1",
        ...     orphan_strategy="delete",
        ... )
        >>> details.source_hostname
        'src.example.org'
        >>> details.assets_localized
        False
    """

    model_config = VALIDATION_CONFIG

    source_hostname: str
    source_catalog_id: str
    source_snapshot: str | None = None
    source_schema_url: str | None = None
    orphan_strategy: str = "fail"
    truncate_oversized: bool = False
    prune_hidden_fkeys: bool = False
    schema_only: bool = False
    asset_mode: str = "refs"
    exclude_schemas: list[str] = Field(default_factory=list)
    exclude_objects: list[str] = Field(default_factory=list)
    add_ml_schema: bool = False
    copy_annotations: bool = True
    copy_policy: bool = True
    reinitialize_dataset_versions: bool = True
    rows_copied: int = 0
    rows_skipped: int = 0
    skipped_rids: list[str] = Field(default_factory=list)
    truncated_count: int = 0
    orphan_rows_removed: int = 0
    orphan_rows_nullified: int = 0
    fkeys_pruned: int = 0

    # Split-phase asset-localization leg (phase 2). Populated by
    # localize_assets() after the bytes have been moved server-to-
    # server. Phase 1 (clone_via_bag) leaves these at defaults.
    assets_localized: bool = False
    assets_localized_at: str | None = None  # ISO8601 UTC
    asset_source_hostname: str | None = None
    assets_copied: int = 0
    assets_skipped: int = 0
    assets_failed: int = 0


class CatalogProvenance(BaseModel):
    """Catalog-level provenance annotation payload.

    Stored on the catalog model under
    :data:`CATALOG_PROVENANCE_URL`. Records who/when/how/with-what
    a catalog came into being.

    Example:
        >>> prov = CatalogProvenance(
        ...     creation_method=CatalogCreationMethod.CREATE,
        ...     created_at="2026-05-15T12:00:00+00:00",
        ...     hostname="example.org",
        ...     catalog_id="42",
        ... )
        >>> prov.is_clone
        False
    """

    model_config = VALIDATION_CONFIG

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

    @property
    def is_clone(self) -> bool:
        """``True`` when the catalog was cloned from another."""
        # use_enum_values=True stores the value, not the enum member,
        # so compare against the string form.
        return (
            self.creation_method == CatalogCreationMethod.CLONE.value
            and self.clone_details is not None
        )


def set_catalog_provenance(
    catalog: ErmrestCatalog,
    name: str | None = None,
    description: str | None = None,
    workflow_url: str | None = None,
    workflow_version: str | None = None,
    creation_method: CatalogCreationMethod = CatalogCreationMethod.CREATE,
    clone_details: CloneDetails | None = None,
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
        clone_details: Clone-specific provenance details. Should
            be supplied when ``creation_method == CLONE``; ignored
            (left as ``None``) otherwise.

    Returns:
        The :class:`CatalogProvenance` object that was written.

    Example:
        >>> from deriva_ml.catalog.provenance import (  # doctest: +SKIP
        ...     set_catalog_provenance, CatalogCreationMethod, CloneDetails,
        ... )
        >>> set_catalog_provenance(  # doctest: +SKIP
        ...     catalog,
        ...     creation_method=CatalogCreationMethod.CLONE,
        ...     clone_details=CloneDetails(
        ...         source_hostname="src.example.org",
        ...         source_catalog_id="1",
        ...     ),
        ... )
    """
    created_by = None
    try:
        session_info = catalog.get("/authn/session").json()
        if session_info and "client" in session_info:
            client = session_info["client"]
            created_by = client.get("display_name") or client.get("id")
    except Exception as e:
        logger.debug("Could not retrieve session info for provenance: %s", e)

    try:
        catalog_info = catalog.get("/").json()
        hostname = catalog_info.get("meta", {}).get("host", "")
        catalog_id = str(catalog.catalog_id)
    except Exception as e:
        logger.debug("Could not retrieve catalog info for provenance: %s", e)
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
        clone_details=clone_details,
    )

    _write_provenance_annotation(catalog, provenance)
    return provenance


def get_catalog_provenance(
    catalog: ErmrestCatalog,
) -> CatalogProvenance | None:
    """Return the catalog's provenance annotation, or ``None``.

    Args:
        catalog: The catalog to read.

    Returns:
        Parsed :class:`CatalogProvenance` if the annotation is
        present and well-formed; ``None`` otherwise.

    Example:
        >>> prov = get_catalog_provenance(catalog)  # doctest: +SKIP
        >>> if prov is not None and prov.is_clone:  # doctest: +SKIP
        ...     print(prov.clone_details.source_hostname)
    """
    try:
        model = catalog.getCatalogModel()
        provenance_data = model.annotations.get(CATALOG_PROVENANCE_URL)
        if provenance_data:
            return CatalogProvenance.model_validate(provenance_data)
    except Exception as e:
        logger.debug("Could not get catalog provenance: %s", e)
    return None


def _write_provenance_annotation(
    catalog: ErmrestCatalog,
    provenance: CatalogProvenance,
) -> None:
    """Write the annotation to the catalog. Best-effort.

    Uses the deriva-py model API
    (``model.annotations[URL] = ...`` + ``model.apply()``) rather
    than a raw ``catalog.put`` so the in-memory model stays in
    sync with the catalog state — subsequent ``getCatalogModel()``
    calls in the same process see the new value without an extra
    round-trip.
    """
    try:
        model = catalog.getCatalogModel()
        model.annotations[CATALOG_PROVENANCE_URL] = provenance.model_dump(mode="json")
        model.apply()
        logger.info("Set catalog provenance annotation")
    except Exception as e:
        logger.warning("Failed to set catalog provenance annotation: %s", e)


__all__ = [
    "CATALOG_PROVENANCE_URL",
    "CatalogCreationMethod",
    "CatalogProvenance",
    "CloneDetails",
    "get_catalog_provenance",
    "set_catalog_provenance",
]

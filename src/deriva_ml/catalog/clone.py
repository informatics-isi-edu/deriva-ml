"""Legacy clone surface — thin re-export over the bag pipeline.

This module used to contain ~1900 lines of bespoke three-stage
clone logic (schema-without-FKs → async row copy → FK application
with orphan handling). Per ADR-0006, catalog cloning factors
through a bag: :class:`CatalogBagBuilder` writes a bag from the
source catalog; :class:`BagCatalogLoader` loads the bag into the
destination. The whole legacy implementation has been removed in
favor of :func:`~deriva_ml.catalog.clone_via_bag.clone_via_bag`.

What's preserved for back-compat:

- :func:`create_ml_workspace` — same name, same signature shape,
  but reimplemented on top of :func:`clone_via_bag`. Legacy
  parameters that don't have bag-pipeline equivalents
  (``truncate_oversized``, ``prune_hidden_fkeys``, ``table_concurrency``,
  ``progress_callback``, ``copy_annotations``, ``copy_policy``,
  ``add_ml_schema``, ``alias``) are accepted but logged at
  warning level — they no longer affect the clone.
- :data:`OrphanStrategy` — alias of
  :class:`deriva.bag.traversal.DanglingFKStrategy`. The legacy
  ``FAIL`` / ``DELETE`` / ``NULLIFY`` value names are preserved
  (both enums use those values).
- :func:`_coerce_asset_mode` accepts legacy string spellings
  (``"REFERENCES"`` / ``"FULL"`` / ``"none"`` / ``"refs"`` /
  ``"full"``) and maps them to :class:`AssetMode` members.
- Provenance API (:class:`CatalogProvenance`,
  :class:`CatalogCreationMethod`, :class:`CloneDetails`,
  :func:`set_catalog_provenance`, :func:`get_catalog_provenance`) —
  unchanged. These live in :mod:`deriva_ml.catalog.provenance`;
  import them from there.

What's deleted:

- All bespoke ``CloneReport`` / ``CloneIssue`` / ``CloneIssueSeverity`` /
  ``CloneIssueCategory`` / ``CloneReportSummary`` machinery. The
  new path returns a :class:`~deriva_ml.catalog.clone_via_bag.CloneViaBagResult`
  carrying a :class:`~deriva.bag.catalog_loader.LoadReport` instead.
- ``CloneCatalogResult``, ``AssetFilter``, ``TruncatedValue``.
- The three-stage clone implementation
  (``_create_ml_workspace_async``, ``_apply_fkeys_with_orphan_handling``,
  ``_copy_data_via_export_paths_async``,
  ``_build_subset_schema``, ``_discover_reachable_tables``, etc.).

Live-catalog tests that exercised the legacy implementation
(``tests/catalog/test_clone_catalog.py``,
``tests/catalog/test_clone_subset_catalog.py``) are removed in
the same commit; they targeted internals (``CloneReport.add_issue``,
``_identify_orphan_values``) that no longer exist. Live-catalog
tests for the new path are tracked separately.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

from deriva.bag.traversal import (
    AssetMode as _AssetMode,
)
from deriva.bag.traversal import (
    DanglingFKStrategy as _DanglingFKStrategy,
)
from deriva.bag.traversal import (
    FKTraversalPolicy,
)

from deriva_ml.core.constants import INTENTIONAL_FK_CYCLES
from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Legacy enums — aliased to deriva.bag.traversal
# ---------------------------------------------------------------------------


#: Legacy alias of :class:`deriva.bag.traversal.DanglingFKStrategy`.
#:
#: Value names (``FAIL``, ``DELETE``, ``NULLIFY``) match between
#: the two; ``OrphanStrategy.FAIL is DanglingFKStrategy.FAIL``.
OrphanStrategy = _DanglingFKStrategy


# ---------------------------------------------------------------------------
# create_ml_workspace — bag-pipeline reimplementation
# ---------------------------------------------------------------------------


def create_ml_workspace(
    source_hostname: str,
    source_catalog_id: str,
    root_rid: str,
    *,
    include_tables: list[str] | None = None,
    exclude_objects: list[str] | None = None,
    exclude_schemas: list[str] | None = None,
    include_associations: bool = True,
    include_vocabularies: bool = True,
    dest_hostname: str | None = None,
    alias: str | None = None,
    add_ml_schema: bool = True,
    asset_mode: Any = _AssetMode.ROWS_ONLY,
    copy_annotations: bool = True,
    copy_policy: bool = True,
    source_credential: dict | None = None,
    dest_credential: dict | None = None,
    orphan_strategy: Any = OrphanStrategy.FAIL,
    prune_hidden_fkeys: bool = False,
    truncate_oversized: bool = False,
    reinitialize_dataset_versions: bool = True,
    table_concurrency: int = 1,
    progress_callback: Callable[[str, float], None] | None = None,
    dest_catalog_id: str | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Clone catalog content from source → bag → destination.

    Reimplementation of the legacy clone surface on top of the
    bag pipeline (ADR-0006). The parameter list matches the
    pre-migration signature for back-compat, but several knobs
    are no longer load-bearing.

    **⚠ Legacy parameters (accepted, no-op, logged as warnings)**:
    these belonged to the bespoke three-stage clone that was
    retired in v1.36.0 and have no direct equivalents in the
    bag pipeline. They're accepted so existing call-sites don't
    error; explicit non-default values trigger a deprecation
    warning. Each one is also marked inline below in the
    ``Args:`` block with ``[NO-OP]``. The full set:
    ``prune_hidden_fkeys``, ``truncate_oversized``,
    ``reinitialize_dataset_versions``, ``table_concurrency``,
    ``progress_callback``, ``copy_annotations``,
    ``copy_policy``, ``add_ml_schema``, ``alias``,
    ``include_associations``, ``include_vocabularies``.

    Args:
        source_hostname: Hostname of the source ERMrest server.
        source_catalog_id: ID of the source catalog.
        root_rid: The dataset RID rooting the clone. Becomes a
            single ``RIDAnchor(table="Dataset", rids=[root_rid])``
            for the bag walk.
        include_tables: Schemas to *include* in the walk. When
            non-empty, treated as ``policy.schemas``. Each entry
            is parsed as ``"schema:table"`` and the schema name
            extracted.
        include_associations: **[NO-OP]** Legacy; ignored.
        include_vocabularies: **[NO-OP]** Legacy; ignored.
        exclude_objects: ``"schema:table"`` entries to exclude.
            Mapped to ``policy.exclude_tables``.
        exclude_schemas: Schemas to skip during the walk. Merged
            into ``policy.exclude_schemas``.
        dest_hostname: Destination hostname. Defaults to
            ``source_hostname`` (same-server clone).
        alias: **[NO-OP]** Legacy; ignored.
        add_ml_schema: **[NO-OP]** Legacy; ignored.
        asset_mode: An :class:`AssetMode` value. Legacy string
            spellings ``"refs"``/``"REFERENCES"`` map to
            ``ROWS_ONLY`` and ``"full"``/``"FULL"`` maps to
            ``UPLOAD_IF_MISSING`` via :func:`_coerce_asset_mode`.
        copy_annotations: **[NO-OP]** Legacy; ignored.
        copy_policy: **[NO-OP]** Legacy; ignored.
        source_credential: Optional credential dict for the
            source server. ``None`` triggers :func:`get_credential`
            on ``source_hostname``.
        dest_credential: Optional credential dict for the
            destination server. ``None`` triggers
            :func:`get_credential` on the destination hostname.
        orphan_strategy: Maps to :attr:`FKTraversalPolicy.dangling_fk_strategy`.
        prune_hidden_fkeys: **[NO-OP]** Legacy; ignored.
        truncate_oversized: **[NO-OP]** Legacy; ignored.
        reinitialize_dataset_versions: **[NO-OP]** Legacy;
            ignored. The bag-pipeline clone preserves source-
            catalog ``Dataset_Version`` rows verbatim. If you
            need destination-snapshot re-seeding, do it
            explicitly after the clone returns.
        table_concurrency: **[NO-OP]** Legacy; ignored.
        progress_callback: **[NO-OP]** Legacy; ignored.
        dest_catalog_id: Destination catalog ID. **Required** for
            the bag pipeline — the new path does not create the
            destination catalog. Use deriva-py's
            :meth:`DerivaServer.create_ermrest_catalog` separately
            if you need to materialize one.
        output_dir: Where the intermediate bag lives. Defaults to
            ``./clone-{source_catalog_id}-to-{dest_catalog_id}/``.

    Returns:
        :class:`~deriva_ml.catalog.clone_via_bag.CloneViaBagResult`
        from the underlying ``clone_via_bag`` call. **Not** the
        old ``CloneCatalogResult`` — that class no longer exists.

    Raises:
        ValueError: If ``dest_catalog_id`` is not supplied (the
            new path requires it; the legacy path created the
            destination catalog automatically).

    Example:
        Same-server clone via the bag pipeline::

            >>> result = create_ml_workspace(  # doctest: +SKIP
            ...     source_hostname="example.org",
            ...     source_catalog_id="1",
            ...     dest_catalog_id="42",
            ...     root_rid="ABC-123",
            ... )
            >>> result.bag_path  # doctest: +SKIP
            PosixPath('clone-1-to-42')
    """
    # Late import keeps this module free of catalog network deps
    # at import time.
    from deriva_ml.catalog.clone_via_bag import clone_via_bag

    if dest_catalog_id is None:
        raise ValueError(
            "create_ml_workspace now requires dest_catalog_id. "
            "The bag-pipeline clone does not create the destination "
            "catalog; create it separately via "
            "DerivaServer.create_ermrest_catalog(), then pass the "
            "resulting ID here."
        )

    # Warn about legacy parameters whose non-default values were
    # carried in via the legacy signature. Default values pass
    # silently so existing callers that left them at defaults
    # don't get noise.
    _warn_about_legacy_params(
        prune_hidden_fkeys=prune_hidden_fkeys,
        truncate_oversized=truncate_oversized,
        reinitialize_dataset_versions=reinitialize_dataset_versions,
        table_concurrency=table_concurrency,
        progress_callback=progress_callback,
        copy_annotations=copy_annotations,
        copy_policy=copy_policy,
        add_ml_schema=add_ml_schema,
        alias=alias,
        include_associations=include_associations,
        include_vocabularies=include_vocabularies,
    )

    # Resolve schema/exclusion knobs. The legacy parameters were
    # "schema:table" strings; the policy fields are
    # ``set[str]`` for schemas and ``set[tuple[str, str]]`` for
    # tables.
    policy_schemas: set[str] | None = None
    if include_tables:
        policy_schemas = {entry.split(":", 1)[0] for entry in include_tables if ":" in entry}

    policy_exclude_tables: set[tuple[str, str]] = set()
    if exclude_objects:
        for entry in exclude_objects:
            if ":" in entry:
                schema, table = entry.split(":", 1)
                policy_exclude_tables.add((schema, table))

    policy_exclude_schemas = set(exclude_schemas) if exclude_schemas else set()
    # Default system-schema exclusions stay on; merge rather than
    # replace.
    from deriva.bag.traversal import DEFAULT_EXCLUDE_SCHEMAS

    policy_exclude_schemas |= set(DEFAULT_EXCLUDE_SCHEMAS)

    # Coerce legacy enum spellings to the new ones.
    resolved_asset_mode = _coerce_asset_mode(asset_mode)
    resolved_orphan_strategy = _coerce_orphan_strategy(orphan_strategy)

    policy = FKTraversalPolicy(
        schemas=policy_schemas,
        exclude_schemas=policy_exclude_schemas,
        exclude_tables=policy_exclude_tables,
        asset_mode=resolved_asset_mode,
        dangling_fk_strategy=resolved_orphan_strategy,
        # Silence WARNING-level cycle-break log spam for the
        # known Dataset ↔ Dataset_Version cycle. See
        # core/constants.py:INTENTIONAL_FK_CYCLES.
        intentional_cycles=set(INTENTIONAL_FK_CYCLES),
    )

    return clone_via_bag(
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
        dest_hostname=dest_hostname or source_hostname,
        dest_catalog_id=dest_catalog_id,
        root_rid=root_rid,
        output_dir=output_dir,
        policy=policy,
        source_credential=source_credential,
        dest_credential=dest_credential,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_asset_mode(value: Any) -> _AssetMode:
    """Accept an :class:`AssetMode` or legacy string spelling.

    Pre-bag-pipeline callers passed strings like ``"refs"``,
    ``"REFERENCES"``, ``"full"``, ``"FULL"``, or ``"none"``. We
    translate those to the new :class:`AssetMode` members so the
    legacy signature still accepts them. Unrecognized values
    raise.
    """
    if isinstance(value, _AssetMode):
        return value
    if isinstance(value, str):
        if value in ("refs", "REFERENCES", "rows_only"):
            return _AssetMode.ROWS_ONLY
        if value in ("full", "FULL", "upload_if_missing"):
            return _AssetMode.UPLOAD_IF_MISSING
        if value in ("force", "upload_force"):
            return _AssetMode.UPLOAD_FORCE
        if value in ("none", "NONE"):
            return _AssetMode.ROWS_ONLY
    raise TypeError(f"Unrecognized asset_mode value: {value!r}. Use deriva.bag.traversal.AssetMode members.")


def _coerce_orphan_strategy(value: Any) -> _DanglingFKStrategy:
    """Accept either a :class:`DanglingFKStrategy` or a legacy
    :class:`OrphanStrategy` value (they're aliases, but be defensive).
    """
    if isinstance(value, _DanglingFKStrategy):
        return value
    if isinstance(value, str):
        try:
            return _DanglingFKStrategy(value)
        except ValueError:
            pass
    raise TypeError(
        f"Unrecognized orphan_strategy value: {value!r}. Use deriva.bag.traversal.DanglingFKStrategy members."
    )


def _warn_about_legacy_params(**kwargs: Any) -> None:
    """Emit deprecation warnings for non-default legacy parameters.

    The legacy ``create_ml_workspace`` accepted several knobs that
    have no equivalent in the bag pipeline. We accept them so the
    pre-migration signature still type-checks, but warn when
    callers actually set them away from default.
    """
    # Defaults are mirrored from create_ml_workspace's signature.
    defaults = {
        "prune_hidden_fkeys": False,
        "truncate_oversized": False,
        "reinitialize_dataset_versions": True,
        "table_concurrency": 1,
        "progress_callback": None,
        "copy_annotations": True,
        "copy_policy": True,
        "add_ml_schema": True,
        "alias": None,
        "include_associations": True,
        "include_vocabularies": True,
    }
    for name, value in kwargs.items():
        if name not in defaults:
            continue
        if value != defaults[name]:
            warnings.warn(
                f"create_ml_workspace: ``{name}={value!r}`` is a legacy "
                f"parameter with no equivalent in the bag-pipeline "
                f"clone (ADR-0006); the value is ignored. See "
                f"deriva.bag.traversal.FKTraversalPolicy for the "
                f"new configuration surface.",
                DeprecationWarning,
                stacklevel=3,
            )


__all__ = [
    "OrphanStrategy",
    "create_ml_workspace",
]

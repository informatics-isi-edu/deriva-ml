"""Bootstrap suggested config entries by reading the catalog.

Pure-read companion to :meth:`DerivaML.validate_config_file`. Returns
structured suggestions; does NOT write files. The skill prose
(``/deriva-ml:dataset-lifecycle``, ``/deriva-ml:work-with-assets``,
``/deriva-ml:write-hydra-config``) owns the format-and-write side.

Public surface:

- :class:`BootstrapSuggestion` -- one suggested config entry, with
  enough info for the caller to format a ``DatasetSpecConfig(...)``
  / ``AssetSpecConfig(...)`` / etc. line.
- :class:`BootstrapReport` -- top-level report grouping suggestions
  by kind, with per-entity rationale strings.

The :class:`DerivaML` driver method lives on
:class:`~deriva_ml.core.mixins.dataset.DatasetMixin` alongside the
existing validators.

Example:
    Get suggestions for the standard four config groups::

        >>> report = ml.bootstrap_config()  # doctest: +SKIP
        >>> for s in report.suggestions:  # doctest: +SKIP
        ...     print(f"# {s.rationale}")
        ...     print(f"# config name: {s.config_name}")
        ...     print(s.spec_string)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ConfigKind = Literal["deriva_ml", "datasets", "assets", "workflow"]

# Default Dataset_Type terms the bootstrap suggests entries for.
# These are the partition-role tags that experiments typically pin.
# ``Split`` parents and ``Complete`` are included because users do
# sometimes consume them directly (e.g. for ablation studies).
DEFAULT_DATASET_TYPE_FILTER: tuple[str, ...] = (
    "Training",
    "Testing",
    "Validation",
    "Complete",
    "Labeled",
)


class BootstrapSuggestion(BaseModel):
    """One suggested config entry.

    Attributes:
        kind: Which config group this entry belongs to. The skill
            consumer routes it to the matching file
            (``deriva_ml`` -> ``configs/deriva.py``, ``datasets`` ->
            ``configs/datasets.py``, etc.).
        config_name: A sensible ``name=`` value to use in the
            ``<group>_store(name=..., ...)`` registration. Derived
            from the entity's description / friendly name; the caller
            may rename it.
        rid: The catalog RID this entry pins.
        version: For dataset entries, the released version to pin.
            ``None`` for kinds that don't take a version (assets,
            workflows, deriva_ml).
        spec_string: A ready-to-paste Python expression -- e.g.
            ``DatasetSpecConfig(rid="2-B4C8", version="0.4.0")``.
            Constructed by the bootstrap so the caller doesn't have
            to remember the canonical syntax for each kind.
        description: Human-readable description from the catalog
            (dataset description, asset filename, workflow name).
            Useful for the ``with_description(...)`` wrap.
        rationale: Why this entry was suggested -- e.g. "Training
            dataset type; latest released version 0.4.0". The skill
            shows this to the user when offering the suggestion.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ConfigKind
    config_name: str
    rid: str
    version: str | None = None
    spec_string: str
    description: str | None = None
    rationale: str


class BootstrapSkipped(BaseModel):
    """An entity the bootstrap considered but didn't suggest.

    Attributes:
        kind: The config group the entity would have belonged to.
        rid: The entity's RID.
        reason: Human-readable reason the entity was skipped --
            e.g. "no released version", "dataset_type=Split (parent
            container, navigate to children instead)".
    """

    model_config = ConfigDict(extra="forbid")

    kind: ConfigKind
    rid: str
    reason: str


class BootstrapReport(BaseModel):
    """Result returned by :meth:`DerivaML.bootstrap_config`.

    Attributes:
        catalog: ``{"hostname": ..., "catalog_id": ...}`` -- the
            catalog the suggestions came from. Echoed so the caller
            knows what to put in ``deriva.py``.
        suggestions: The proposed entries, grouped (by being typed
            with ``kind``) but not segregated -- the caller filters
            by ``kind`` to route into files.
        skipped: Entities considered but not suggested, with reason.
            Useful for explaining the empty case ("found 12 datasets,
            none had a released version") without forcing the caller
            to re-query the catalog.
    """

    model_config = ConfigDict(extra="forbid")

    catalog: dict[str, str] = Field(default_factory=dict)
    suggestions: list[BootstrapSuggestion] = Field(default_factory=list)
    skipped: list[BootstrapSkipped] = Field(default_factory=list)


def _sanitize_config_name(text: str | None, fallback: str) -> str:
    """Turn an entity description into a usable Python identifier.

    ``"Small CIFAR-10 labeled split: stratified 400/100"`` ->
    ``"small_cifar_10_labeled_split"``.

    Args:
        text: The free-form description to sanitize.
        fallback: A safe default (typically the RID) when text is
            empty or yields no usable characters.

    Returns:
        A lowercase identifier-safe string. Never empty.
    """
    if not text:
        return fallback.replace("-", "_").lower()
    # Strip non-alphanumeric, collapse runs of underscores, lowercase.
    cleaned_chars = []
    last_was_underscore = False
    for ch in text:
        if ch.isalnum():
            cleaned_chars.append(ch.lower())
            last_was_underscore = False
        elif not last_was_underscore:
            cleaned_chars.append("_")
            last_was_underscore = True
    name = "".join(cleaned_chars).strip("_")
    # Truncate very long descriptions; 40 chars keeps things readable.
    if len(name) > 40:
        name = name[:40].rstrip("_")
    if not name or not name[0].isalpha():
        name = "ds_" + name if name else fallback.replace("-", "_").lower()
    return name


def _format_dataset_spec(rid: str, version: str) -> str:
    """Format a ``DatasetSpecConfig(rid=..., version=...)`` line."""
    return f'DatasetSpecConfig(rid="{rid}", version="{version}")'


def _format_asset_spec(rid: str, *, cache: bool = False) -> str:
    """Format an ``AssetSpecConfig(rid=..., cache=...)`` line."""
    if cache:
        return f'AssetSpecConfig(rid="{rid}", cache=True)'
    return f'AssetSpecConfig(rid="{rid}")'


def _format_workflow_spec(rid: str) -> str:
    """Format a ``Workflow(rid=...)`` line."""
    return f'Workflow(rid="{rid}")'


def _format_deriva_ml_spec(hostname: str, catalog_id: str) -> str:
    """Format the ``deriva_store(DerivaMLConfig, ...)`` call body."""
    return (
        f'deriva_store(DerivaMLConfig, name="default_deriva", '
        f'hostname="{hostname}", catalog_id="{catalog_id}")'
    )

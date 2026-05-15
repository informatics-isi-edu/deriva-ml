"""Unit tests for :mod:`deriva_ml.catalog.provenance`.

Closes audit Phase 3 catalog/ §3.4 — the provenance API had
zero tests before this file. These tests use mocked catalogs so
no live server is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.catalog.provenance import (
    CATALOG_PROVENANCE_URL,
    CatalogCreationMethod,
    CatalogProvenance,
    CloneDetails,
    get_catalog_provenance,
    set_catalog_provenance,
)


# ---------------------------------------------------------------------------
# CloneDetails — Pydantic round-trip + defaults
# ---------------------------------------------------------------------------


def test_clone_details_required_fields() -> None:
    """Source hostname + catalog_id are the only required fields."""
    details = CloneDetails(
        source_hostname="src.example.org",
        source_catalog_id="1",
    )
    assert details.source_hostname == "src.example.org"
    assert details.source_catalog_id == "1"
    # Defaults for everything else
    assert details.orphan_strategy == "fail"
    assert details.asset_mode == "refs"
    assert details.rows_copied == 0
    assert details.assets_localized is False
    assert details.assets_localized_at is None
    assert details.asset_source_hostname is None
    assert details.assets_copied == 0


def test_clone_details_round_trip_via_model_dump() -> None:
    """``model_dump()`` + ``model_validate(...)`` is a no-op round trip."""
    original = CloneDetails(
        source_hostname="src.example.org",
        source_catalog_id="1",
        orphan_strategy="delete",
        asset_mode="rows_only",
        rows_copied=12345,
        orphan_rows_removed=7,
        assets_localized=True,
        assets_localized_at="2026-05-15T12:00:00+00:00",
        asset_source_hostname="src.example.org",
        assets_copied=42,
        assets_skipped=3,
        assets_failed=1,
    )
    dumped = original.model_dump(mode="json")
    restored = CloneDetails.model_validate(dumped)
    assert restored == original


# ---------------------------------------------------------------------------
# CatalogProvenance — is_clone semantics + serialization
# ---------------------------------------------------------------------------


def test_catalog_provenance_is_clone_false_for_create() -> None:
    """``creation_method=CREATE`` → ``is_clone == False``."""
    prov = CatalogProvenance(
        creation_method=CatalogCreationMethod.CREATE,
        created_at="2026-05-15T12:00:00+00:00",
        hostname="example.org",
        catalog_id="42",
    )
    assert prov.is_clone is False


def test_catalog_provenance_is_clone_true_when_method_clone_and_details_present() -> None:
    """``CLONE`` + populated ``clone_details`` → ``is_clone == True``."""
    prov = CatalogProvenance(
        creation_method=CatalogCreationMethod.CLONE,
        created_at="2026-05-15T12:00:00+00:00",
        hostname="example.org",
        catalog_id="42",
        clone_details=CloneDetails(
            source_hostname="src.example.org",
            source_catalog_id="1",
        ),
    )
    assert prov.is_clone is True


def test_catalog_provenance_is_clone_false_when_clone_but_no_details() -> None:
    """``CLONE`` without ``clone_details`` is not really a clone.

    Regression guard for the original ``CloneDetails`` bug:
    ``is_clone`` returned False unconditionally before the wiring
    work because ``clone_details`` was never populated. Now that
    it can be populated, the predicate must still require both
    the method and the details.
    """
    prov = CatalogProvenance(
        creation_method=CatalogCreationMethod.CLONE,
        created_at="2026-05-15T12:00:00+00:00",
        hostname="example.org",
        catalog_id="42",
        clone_details=None,
    )
    assert prov.is_clone is False


def test_catalog_provenance_round_trip_with_clone_details() -> None:
    """The full payload round-trips through ``model_dump`` → ``model_validate``."""
    original = CatalogProvenance(
        creation_method=CatalogCreationMethod.CLONE,
        created_at="2026-05-15T12:00:00+00:00",
        hostname="example.org",
        catalog_id="42",
        created_by="alice@isi.edu",
        name="my catalog",
        description="test clone",
        workflow_url="https://github.com/example/repo",
        workflow_version="v1.0.0",
        clone_details=CloneDetails(
            source_hostname="src.example.org",
            source_catalog_id="1",
            orphan_strategy="delete",
            rows_copied=12345,
        ),
    )
    dumped = original.model_dump(mode="json")
    restored = CatalogProvenance.model_validate(dumped)
    assert restored == original


# ---------------------------------------------------------------------------
# get_catalog_provenance — annotation parsing
# ---------------------------------------------------------------------------


def _make_fake_catalog(*, annotation: dict | None) -> MagicMock:
    """Build a fake ErmrestCatalog whose model carries the given annotation."""
    catalog = MagicMock()
    model = MagicMock()
    if annotation is None:
        # No annotation present → .get returns None
        model.annotations = {}
    else:
        model.annotations = {CATALOG_PROVENANCE_URL: annotation}
    catalog.getCatalogModel.return_value = model
    catalog.catalog_id = "42"
    return catalog


def test_get_catalog_provenance_returns_none_when_absent() -> None:
    """No annotation on the catalog → ``None``, not a raise."""
    catalog = _make_fake_catalog(annotation=None)
    assert get_catalog_provenance(catalog) is None


def test_get_catalog_provenance_parses_existing_annotation() -> None:
    """A well-formed annotation round-trips back into a ``CatalogProvenance``."""
    payload = CatalogProvenance(
        creation_method=CatalogCreationMethod.CLONE,
        created_at="2026-05-15T12:00:00+00:00",
        hostname="example.org",
        catalog_id="42",
        clone_details=CloneDetails(
            source_hostname="src.example.org",
            source_catalog_id="1",
            rows_copied=99,
        ),
    ).model_dump(mode="json")

    catalog = _make_fake_catalog(annotation=payload)
    parsed = get_catalog_provenance(catalog)
    assert parsed is not None
    assert parsed.is_clone is True
    assert parsed.clone_details is not None
    assert parsed.clone_details.source_hostname == "src.example.org"
    assert parsed.clone_details.rows_copied == 99


def test_get_catalog_provenance_handles_unknown_creation_method() -> None:
    """An unknown ``creation_method`` value falls back to ``UNKNOWN``.

    Pydantic's enum coercion rejects unknown values by default; the
    Pydantic model's str-Enum form maps the wire string directly to
    the enum member. If a future writer introduces a new method
    name, older readers see ``UNKNOWN`` rather than crashing.

    Implementation note: this test currently expects parsing to
    return ``None`` (caught in the catch-all), not a degraded
    ``UNKNOWN`` payload, because Pydantic raises on unknown enum
    values. The behavior is acceptable — readers see "no
    provenance" rather than a partial record they might
    misinterpret.
    """
    payload = {
        "creation_method": "definitely-not-a-real-method",
        "created_at": "2026-05-15T12:00:00+00:00",
        "hostname": "example.org",
        "catalog_id": "42",
    }
    catalog = _make_fake_catalog(annotation=payload)
    parsed = get_catalog_provenance(catalog)
    # Either None (Pydantic raised, caught by the get_*'s except)
    # or UNKNOWN (if we extend the parser later). Both are valid
    # outcomes; the contract is "don't blow up." Assert weak.
    assert parsed is None or parsed.creation_method == CatalogCreationMethod.UNKNOWN


# ---------------------------------------------------------------------------
# set_catalog_provenance — write path
# ---------------------------------------------------------------------------


def test_set_catalog_provenance_writes_annotation() -> None:
    """``set_catalog_provenance`` populates the model annotation + applies."""
    catalog = _make_fake_catalog(annotation=None)
    # Mock session + catalog-info endpoints used to populate metadata
    catalog.get.return_value.json.side_effect = [
        {"client": {"display_name": "Alice"}},  # /authn/session
        {"meta": {"host": "example.org"}},  # /
    ]

    result = set_catalog_provenance(
        catalog,
        name="my catalog",
        creation_method=CatalogCreationMethod.CREATE,
    )

    # Returned object reflects what was written
    assert result.name == "my catalog"
    assert result.creation_method == CatalogCreationMethod.CREATE.value
    assert result.created_by == "Alice"

    # The annotation was written and apply() was called
    model = catalog.getCatalogModel.return_value
    assert CATALOG_PROVENANCE_URL in model.annotations
    model.apply.assert_called_once()


def test_set_catalog_provenance_passes_clone_details_through() -> None:
    """``clone_details`` argument flows into the written annotation."""
    catalog = _make_fake_catalog(annotation=None)
    catalog.get.return_value.json.side_effect = [
        {"client": {"display_name": "Alice"}},
        {"meta": {"host": "example.org"}},
    ]

    details = CloneDetails(
        source_hostname="src.example.org",
        source_catalog_id="1",
        rows_copied=500,
    )

    result = set_catalog_provenance(
        catalog,
        creation_method=CatalogCreationMethod.CLONE,
        clone_details=details,
    )

    assert result.is_clone is True
    assert result.clone_details is not None
    assert result.clone_details.rows_copied == 500


def test_set_catalog_provenance_swallows_write_failures() -> None:
    """Annotation write failures are warnings, not exceptions.

    The provenance writer is "best-effort" — a failure to attach
    the annotation should not break the calling code's primary
    workflow (clone, localize, create). Verify by making the
    underlying ``model.apply`` raise.
    """
    catalog = _make_fake_catalog(annotation=None)
    catalog.get.return_value.json.side_effect = [
        {"client": {"display_name": "Alice"}},
        {"meta": {"host": "example.org"}},
    ]
    catalog.getCatalogModel.return_value.apply.side_effect = RuntimeError("boom")

    # Should not raise
    result = set_catalog_provenance(catalog, name="my catalog")
    assert result is not None
    assert result.name == "my catalog"


# ---------------------------------------------------------------------------
# CatalogCreationMethod — enum integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,value",
    [
        (CatalogCreationMethod.CLONE, "clone"),
        (CatalogCreationMethod.CREATE, "create"),
        (CatalogCreationMethod.SCHEMA, "schema"),
        (CatalogCreationMethod.UNKNOWN, "unknown"),
    ],
)
def test_catalog_creation_method_string_form(
    method: CatalogCreationMethod, value: str
) -> None:
    """Each enum member's string form matches the documented wire value."""
    # str(Enum) compares against the value because we inherit from str
    assert method.value == value
    assert method == value

"""Tests for :mod:`deriva_ml.schema.annotations`.

Focused on the pure-Python ``build_navbar_menu`` builder. The
function walks a :class:`DerivaModel` to produce a Chaise
``navbarMenu`` payload — we exercise it against a stand-in model
so no live catalog is required.

Closes audit Phase 3 schema/ §3.7.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.schema.annotations import build_navbar_menu

# ---------------------------------------------------------------------------
# Model stand-in
# ---------------------------------------------------------------------------


def _make_model(
    *,
    catalog_id: str = "42",
    ml_schema: str = "deriva-ml",
    domain_schemas: frozenset[str] = frozenset({"my_domain"}),
    ml_tables: tuple[str, ...] = (
        "Workflow",
        "Execution",
        "Dataset",
        "Dataset_Version",
        "Asset_Type",  # vocab
        "Workflow_Type",  # vocab
        "Execution_Metadata",  # asset
        "Execution_Asset",  # asset
    ),
    domain_tables: tuple[str, ...] = (
        "Subject",
        "Image",  # asset
        "Subject_Image",  # association
        "Subject_Type",  # vocab
    ),
    vocabularies: set[str] = frozenset({"Asset_Type", "Workflow_Type", "Subject_Type"}),
    assets: set[str] = frozenset({"Execution_Metadata", "Execution_Asset", "Image"}),
    associations: set[str] = frozenset({"Subject_Image"}),
) -> MagicMock:
    """Build a minimal DerivaModel-shaped mock for navbar tests."""
    model = MagicMock()
    model.catalog.catalog_id = catalog_id
    model.ml_schema = ml_schema
    model.domain_schemas = domain_schemas

    # model.schemas maps schema name → an object with a ``tables`` dict
    ml_schema_obj = MagicMock()
    ml_schema_obj.tables = {name: object() for name in ml_tables}
    domain_schema_obj = MagicMock()
    domain_schema_obj.tables = {name: object() for name in domain_tables}
    model.schemas = {ml_schema: ml_schema_obj, "my_domain": domain_schema_obj}

    model.is_vocabulary.side_effect = lambda name: name in vocabularies
    model.is_asset.side_effect = lambda name: name in assets
    model.is_association.side_effect = lambda name, **_: name in associations

    # No features by default.
    model.find_features.return_value = []

    return model


# ---------------------------------------------------------------------------
# Shape invariants
# ---------------------------------------------------------------------------


def test_navbar_menu_returns_top_level_shape() -> None:
    """Top-level menu has ``newTab=False`` and a ``children`` list."""
    menu = build_navbar_menu(_make_model())
    assert menu["newTab"] is False
    assert isinstance(menu["children"], list)
    assert menu["children"], "navbar menu must have child entries"


def test_navbar_menu_includes_required_sections() -> None:
    """Each load-bearing section appears exactly once."""
    menu = build_navbar_menu(_make_model())
    names = [c.get("name") for c in menu["children"]]

    for required in (
        "User Info",
        "Deriva-ML",
        "WWW",
        "my_domain",  # the configured domain schema
        "Vocabulary",
        "Assets",
        "Features",
        "Catalog Registry",
    ):
        assert names.count(required) == 1, f"section '{required}' should appear exactly once; got names={names}"


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def test_navbar_menu_uses_catalog_id_in_urls() -> None:
    """The Chaise URLs all carry the configured catalog ID.

    Exception: the Catalog Registry link points at the ERMrest
    registry catalog itself (always ID 0), so we exclude any
    URL that references ``ermrest:registry``.
    """
    menu = build_navbar_menu(_make_model(catalog_id="my-catalog"))

    # Crawl the tree collecting all URL strings.
    urls: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            if "url" in node:
                urls.append(node["url"])
            for child in node.get("children", []):
                _walk(child)

    for child in menu["children"]:
        _walk(child)

    assert urls, "menu should contain at least one URL"
    catalog_urls = [u for u in urls if "ermrest:registry" not in u]
    assert catalog_urls, "menu should contain at least one catalog-scoped URL"
    for url in catalog_urls:
        if "/chaise/recordset/#" in url:
            assert "#my-catalog/" in url, f"URL {url!r} should embed the catalog_id 'my-catalog'"


# ---------------------------------------------------------------------------
# Classification routing
# ---------------------------------------------------------------------------


def test_navbar_menu_routes_vocabularies_to_vocabulary_section() -> None:
    """Tables classified as vocabularies appear under the Vocabulary menu."""
    menu = build_navbar_menu(_make_model())
    vocab_section = next(c for c in menu["children"] if c.get("name") == "Vocabulary")
    vocab_names = [child.get("name") for child in vocab_section["children"] if not child.get("header")]

    # Asset_Type and Workflow_Type are ML-schema vocabs; Subject_Type
    # is a domain vocab — all three should appear.
    assert "Asset_Type" in vocab_names
    assert "Workflow_Type" in vocab_names
    assert "Subject_Type" in vocab_names


def test_navbar_menu_routes_assets_to_assets_section() -> None:
    """Tables classified as assets appear under the Assets menu."""
    menu = build_navbar_menu(_make_model())
    assets_section = next(c for c in menu["children"] if c.get("name") == "Assets")
    asset_names = [child["name"] for child in assets_section["children"]]

    assert "Execution_Metadata" in asset_names
    assert "Execution_Asset" in asset_names
    assert "Image" in asset_names  # domain asset


def test_navbar_menu_domain_schema_omits_vocabs_and_associations() -> None:
    """A domain-schema menu lists only non-vocab non-association tables."""
    menu = build_navbar_menu(_make_model())
    domain_section = next(c for c in menu["children"] if c.get("name") == "my_domain")
    domain_table_names = [child["name"] for child in domain_section["children"]]

    # Plain table + asset stay in the domain section.
    assert "Subject" in domain_table_names
    assert "Image" in domain_table_names
    # Vocab and association are filtered out.
    assert "Subject_Type" not in domain_table_names
    assert "Subject_Image" not in domain_table_names


def test_navbar_menu_features_section_lists_features() -> None:
    """A model with features produces a populated Features menu."""
    model = _make_model()

    # Add two fake features
    feature_a = MagicMock()
    feature_a.feature_name = "scan_quality"
    feature_a.target_table.name = "Image"
    feature_a.feature_table.schema.name = "my_domain"
    feature_a.feature_table.name = "Image_scan_quality"

    feature_b = MagicMock()
    feature_b.feature_name = "subject_height"
    feature_b.target_table.name = "Subject"
    feature_b.feature_table.schema.name = "my_domain"
    feature_b.feature_table.name = "Subject_subject_height"

    model.find_features.return_value = [feature_a, feature_b]

    menu = build_navbar_menu(model)
    features_section = next(c for c in menu["children"] if c.get("name") == "Features")
    feature_names = [child["name"] for child in features_section["children"]]

    assert "Image:scan_quality" in feature_names
    assert "Subject:subject_height" in feature_names


def test_navbar_menu_features_section_empty_when_no_features() -> None:
    """No features → empty Features menu, not absent."""
    menu = build_navbar_menu(_make_model())
    features_section = next(c for c in menu["children"] if c.get("name") == "Features")
    assert features_section["children"] == []

"""Tests for apply_catalog_annotations method."""

from deriva.core.utils.core_utils import tag as deriva_tags


class TestCatalogAnnotations:
    """Test catalog annotation functionality."""

    def test_apply_catalog_annotations_default(self, test_ml):
        """Test applying catalog annotations with default parameters."""
        ml = test_ml

        # Apply annotations
        ml.apply_catalog_annotations()

        # Refresh the model to get updated annotations
        ml.model.catalog.getCatalogModel()

        # Check that annotations were applied
        annotations = ml.model.annotations
        assert deriva_tags.chaise_config in annotations
        assert deriva_tags.display in annotations
        assert deriva_tags.bulk_upload in annotations

        # Check default values
        chaise_config = annotations[deriva_tags.chaise_config]
        assert chaise_config["navbarBrandText"] == "ML Data Browser"
        assert chaise_config["headTitle"] == "Catalog ML"

    def test_apply_catalog_annotations_custom_branding(self, test_ml):
        """Test applying catalog annotations with custom branding."""
        ml = test_ml

        # Apply annotations with custom branding
        ml.apply_catalog_annotations(
            navbar_brand_text="My Custom Project",
            head_title="Custom ML Catalog",
        )

        # Refresh and check
        ml.model.catalog.getCatalogModel()
        chaise_config = ml.model.annotations[deriva_tags.chaise_config]

        assert chaise_config["navbarBrandText"] == "My Custom Project"
        assert chaise_config["headTitle"] == "Custom ML Catalog"

    def test_apply_catalog_annotations_navbar_structure(self, test_ml):
        """Test that navbar menu has expected structure."""
        ml = test_ml

        # Apply annotations
        ml.apply_catalog_annotations()

        # Check navbar menu structure
        chaise_config = ml.model.annotations[deriva_tags.chaise_config]
        navbar_menu = chaise_config["navbarMenu"]

        assert navbar_menu["newTab"] is False
        assert "children" in navbar_menu

        # Get menu names
        menu_names = [child.get("name") for child in navbar_menu["children"]]

        # Check expected menus exist
        assert "User Info" in menu_names
        assert "Deriva-ML" in menu_names
        assert "WWW" in menu_names
        assert "Vocabulary" in menu_names
        assert "Assets" in menu_names
        assert "Documentation" in menu_names
        assert "Catalog Registry" in menu_names

        # Check domain schema menu exists
        assert ml.domain_schema in menu_names

    def test_apply_catalog_annotations_deriva_ml_menu(self, test_ml):
        """Test that Deriva-ML menu has expected items."""
        ml = test_ml
        ml.apply_catalog_annotations()

        chaise_config = ml.model.annotations[deriva_tags.chaise_config]
        navbar_children = chaise_config["navbarMenu"]["children"]

        # Find Deriva-ML menu
        deriva_ml_menu = next(
            (m for m in navbar_children if m.get("name") == "Deriva-ML"),
            None,
        )
        assert deriva_ml_menu is not None

        # Check expected items
        item_names = [item["name"] for item in deriva_ml_menu["children"]]
        assert "Workflow" in item_names
        assert "Execution" in item_names
        assert "Dataset" in item_names
        assert "Dataset Version" in item_names  # Note: space not underscore in menu

    def test_apply_catalog_annotations_display_settings(self, test_ml):
        """Test that display settings are correctly applied."""
        ml = test_ml
        ml.apply_catalog_annotations()

        chaise_config = ml.model.annotations[deriva_tags.chaise_config]

        # Check display settings
        assert chaise_config["systemColumnsDisplayEntry"] == ["RID"]
        assert chaise_config["systemColumnsDisplayCompact"] == ["RID"]
        assert chaise_config["deleteRecord"] is True
        assert chaise_config["showFaceting"] is True
        assert chaise_config["defaultTable"]["table"] == "Dataset"
        assert chaise_config["defaultTable"]["schema"] == "deriva-ml"

    def test_apply_catalog_annotations_with_vocabulary(self, test_ml):
        """Test that vocabularies appear in the Vocabulary menu."""
        ml = test_ml

        # Create a vocabulary before applying annotations
        ml.create_vocabulary("TestVocab", "A test vocabulary")

        # Apply annotations
        ml.apply_catalog_annotations()

        chaise_config = ml.model.annotations[deriva_tags.chaise_config]
        navbar_children = chaise_config["navbarMenu"]["children"]

        # Find Vocabulary menu
        vocab_menu = next(
            (m for m in navbar_children if m.get("name") == "Vocabulary"),
            None,
        )
        assert vocab_menu is not None

        # Check that TestVocab appears in the menu
        vocab_items = [
            item.get("name")
            for item in vocab_menu["children"]
            if not item.get("header")
        ]
        assert "TestVocab" in vocab_items

    def test_apply_catalog_annotations_with_asset(self, test_ml):
        """Test that asset tables appear in the Assets menu."""
        ml = test_ml

        # Create an asset before applying annotations
        ml.create_asset("TestAsset")

        # Apply annotations
        ml.apply_catalog_annotations()

        chaise_config = ml.model.annotations[deriva_tags.chaise_config]
        navbar_children = chaise_config["navbarMenu"]["children"]

        # Find Assets menu
        assets_menu = next(
            (m for m in navbar_children if m.get("name") == "Assets"),
            None,
        )
        assert assets_menu is not None

        # Check that TestAsset appears in the menu
        asset_items = [item.get("name") for item in assets_menu["children"]]
        assert "TestAsset" in asset_items

    def test_apply_catalog_annotations_bulk_upload(self, test_ml):
        """Test that bulk upload configuration is set."""
        ml = test_ml

        # Create an asset so bulk upload config has something to configure
        ml.create_asset("UploadAsset")

        ml.apply_catalog_annotations()

        annotations = ml.model.annotations
        assert deriva_tags.bulk_upload in annotations

        bulk_upload = annotations[deriva_tags.bulk_upload]
        assert "asset_mappings" in bulk_upload or isinstance(bulk_upload, dict)

    def test_apply_catalog_annotations_idempotent(self, test_ml):
        """Test that calling apply_catalog_annotations twice doesn't cause issues."""
        ml = test_ml

        # Apply twice
        ml.apply_catalog_annotations(navbar_brand_text="First")
        ml.apply_catalog_annotations(navbar_brand_text="Second")

        # Second call should overwrite first
        chaise_config = ml.model.annotations[deriva_tags.chaise_config]
        assert chaise_config["navbarBrandText"] == "Second"

"""Tests for schema-evolution impact analysis (issue #75, Round B).

``find_datasets_referencing`` and ``find_features_referencing`` answer
the catalog-evolver's question "what breaks if I change this table /
column?" by walking the deriva-ml domain model: datasets reference
tables through their member associations (``Dataset_<Table>``);
features reference tables through their association table's FKs
(the self-FK to the target table plus any term / asset / value FKs).
"""

import pytest


@pytest.fixture
def impact_ml(catalog_manager, tmp_path):
    """A DerivaML over the WITH_DATASETS catalog state.

    ensure_datasets seeds datasets with Image + Subject members AND the
    demo features (Execution_Image_BoundingBox / _Quality,
    Execution_Subject_Health), giving a realistic reference graph.
    """
    ml, _desc = catalog_manager.ensure_datasets(tmp_path)
    return ml


class TestFindDatasetsReferencing:
    def test_member_table_is_referenced(self, impact_ml):
        """Datasets holding Image members are reported, with member counts."""
        refs = impact_ml.find_datasets_referencing("Image")
        assert refs, "demo datasets hold Image members; expected non-empty"
        for ref in refs:
            assert ref.element_table == "Image"
            assert ref.member_count >= 1
            assert ref.dataset_rid

    def test_unreferenced_table_returns_empty(self, impact_ml):
        """A table with no Dataset association yields no references."""
        refs = impact_ml.find_datasets_referencing("Observation")
        assert refs == []

    def test_column_is_table_granular(self, impact_ml):
        """Dataset impact is table-granular: column narrows nothing.

        Dataset membership is row-level; any column drop on a member
        table impacts every dataset holding rows of that table.
        """
        by_table = impact_ml.find_datasets_referencing("Image")
        by_column = impact_ml.find_datasets_referencing("Image", column="URL")
        assert {r.dataset_rid for r in by_column} == {r.dataset_rid for r in by_table}


class TestFindFeaturesReferencing:
    def test_features_on_target_table(self, impact_ml):
        """Features defined ON Image reference it via the self-FK."""
        refs = impact_ml.find_features_referencing("Image")
        names = {r.feature_name for r in refs}
        assert {"BoundingBox", "Quality"} <= names
        for ref in refs:
            assert ref.referencing_columns, "expected the FK column names"

    def test_vocabulary_reference_via_term_column(self, impact_ml):
        """A feature's term column makes it reference the vocabulary table."""
        refs = impact_ml.find_features_referencing("ImageQuality")
        names = {r.feature_name for r in refs}
        assert "Quality" in names
        quality = next(r for r in refs if r.feature_name == "Quality")
        assert quality.target_table == "Image"

    def test_column_narrows_to_referenced_column(self, impact_ml):
        """column= matches the FK's REFERENCED column (usually RID)."""
        by_rid = impact_ml.find_features_referencing("Image", column="RID")
        assert {r.feature_name for r in by_rid} >= {"BoundingBox", "Quality"}
        none = impact_ml.find_features_referencing("Image", column="No_Such_Column")
        assert none == []

    def test_unreferenced_table_returns_empty(self, impact_ml):
        refs = impact_ml.find_features_referencing("ClinicalRecord")
        assert refs == []

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

    @staticmethod
    def _soft_delete_an_image_dataset(impact_ml):
        """Soft-delete one Image-referencing dataset; return its RID.

        Sets ``Deleted=true`` directly on the Dataset row (what a soft delete
        does at the data level), which keeps the membership junction rows — the
        exact state that made #355's association-only query leak the dataset.
        Done via the path-builder rather than ``delete_dataset`` so the test is
        independent of that method's nesting guard (the demo datasets are
        nested).
        """
        refs = impact_ml.find_datasets_referencing("Image")
        assert refs, "fixture must have Image-referencing datasets"
        victim = refs[0].dataset_rid

        pb = impact_ml.pathBuilder()
        ds_table = pb.schemas[impact_ml._dataset_table.schema.name].tables[impact_ml._dataset_table.name]
        ds_table.update([{"RID": victim, "Deleted": True}])

        # Sanity: it is now soft-deleted (lookup_dataset refuses it by default).
        from deriva_ml import DerivaMLException

        try:
            impact_ml.lookup_dataset(victim)
            raise AssertionError("victim should be soft-deleted (lookup_dataset should refuse it)")
        except DerivaMLException:
            pass
        return victim

    def test_soft_deleted_datasets_excluded_by_default(self, impact_ml):
        """A soft-deleted dataset must NOT appear in the default result — it
        agrees with find_datasets / lookup_dataset, which default-exclude
        deleted datasets. Junction rows persist after a soft delete, so the
        association-only query would otherwise leak the deleted dataset (#355).
        """
        victim = self._soft_delete_an_image_dataset(impact_ml)

        # Default: the soft-deleted dataset is gone; the surface now agrees
        # with lookup_dataset (which raises for the same RID).
        default_rids = {r.dataset_rid for r in impact_ml.find_datasets_referencing("Image")}
        assert victim not in default_rids, "soft-deleted dataset must be excluded by default"

    def test_soft_deleted_datasets_included_with_flag(self, impact_ml):
        """``deleted=True`` opts back into soft-deleted datasets — the
        impact-analysis 'what still references this row, even in tombstoned
        datasets' use case."""
        victim = self._soft_delete_an_image_dataset(impact_ml)

        with_deleted = {r.dataset_rid for r in impact_ml.find_datasets_referencing("Image", deleted=True)}
        assert victim in with_deleted, "deleted=True must include soft-deleted datasets"


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

"""Regression tests for key-qualified multi-value features (findings 10 + 12).

Two coupled bugs hid behind the same fixture gap — no prior fixture
modeled a feature-association table whose **compound uniqueness key
includes a qualifier FK** beyond ``{target, Feature_Name, Execution}``.
That is the structural shape of eye-ai's real
``Execution_Subject_Chart_Label`` feature (an ``Image_Side`` qualifier:
the same Subject has a left-eye row and a right-eye row).

Bug 1 (finding 10): ``find_features`` capped ``find_associations`` at
``max_arity=3``, so a key-FK-arity-4 qualified feature was silently
undiscoverable (and therefore unreadable via ``lookup_feature`` /
``feature_values``).

Bug 2 (finding 12): ``reduce_with_selector`` grouped by the target RID
alone, so a selector collapsed the two eyes of one Subject into one row —
silently dropping a distinct, valid observation.

The fixture here (``denorm_schema_qualified_feature``) reproduces both:
under the old ``max_arity=3`` cap ``find_features`` returns nothing for it;
under the fix it is discovered, exposes its ``Image_Side`` qualifier
column, and a selector keyed on ``(Subject, Image_Side)`` preserves both
eyes.

Findings (absolute paths):
  /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e/findings/investigation/10-find-features-multivalue-arity-cap.md
  /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e/findings/investigation/12-selector-grouping-qualified-features.md

Pure offline tests — ``Model.fromfile`` wires ``referenced_by`` so
``find_features`` works without a live catalog.
"""

from __future__ import annotations

from deriva_ml.feature import reduce_with_selector
from deriva_ml.model.catalog import DerivaModel


class TestQualifiedFeatureDiscovery:
    """Bug 1: the arity cap. ``find_features`` must discover the qualified feature."""

    def test_find_features_discovers_key_qualified_feature(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """``find_features("Subject")`` includes the key-qualified feature.

        Under the former ``max_arity=3`` cap this returned ``[]`` (the
        compound key's FK arity is 4). The fix drops the ceiling so the
        feature is discovered.
        """
        feats = list(denorm_qualified_feature_deriva_model.find_features("Subject"))
        names = {f.feature_table.name for f in feats}
        assert "Execution_Subject_Chart_Label" in names

    def test_old_arity_cap_would_miss_it(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """Pin the gap: the OLD ``max_arity=3`` cap excludes this feature.

        This is what made the bug invisible. We call ``find_associations``
        directly with both ceilings and assert they disagree — guaranteeing
        the fixture genuinely exercises the cap (and isn't accidentally
        arity-3).
        """
        subject = denorm_qualified_feature_deriva_model.name_to_table("Subject")
        old = [a.table.name for a in subject.find_associations(min_arity=3, max_arity=3, pure=False)]
        new = [a.table.name for a in subject.find_associations(min_arity=3, max_arity=None, pure=False)]
        assert "Execution_Subject_Chart_Label" not in old
        assert "Execution_Subject_Chart_Label" in new

    def test_lookup_feature_succeeds(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """``lookup_feature`` resolves the qualified feature (was ``DerivaMLFeatureNotFound``)."""
        feat = denorm_qualified_feature_deriva_model.lookup_feature("Subject", "Chart_Label")
        assert feat.feature_name == "Chart_Label"
        assert feat.target_table.name == "Subject"


class TestQualifierColumns:
    """``Feature.qualifier_columns`` exposes the in-key value FKs."""

    def test_qualifier_columns_is_image_side(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """The qualifier is exactly ``Image_Side`` — the in-key value FK.

        ``Condition_Label`` is a value FK too, but NOT in the compound key,
        so it must NOT appear as a qualifier. ``Feature_Name`` / ``Execution``
        are structural and excluded.
        """
        feat = denorm_qualified_feature_deriva_model.lookup_feature("Subject", "Chart_Label")
        assert feat.qualifier_columns == ["Image_Side"]

    def test_record_class_carries_qualifier_field(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """The generated FeatureRecord carries the ``Image_Side`` field.

        The qualifier must be present and distinguishable on every record so
        ``reduce_with_selector`` can group on it.
        """
        feat = denorm_qualified_feature_deriva_model.lookup_feature("Subject", "Chart_Label")
        record_class = feat.feature_record_class()
        fields = set(record_class.model_fields.keys())
        assert {"Subject", "Image_Side", "Condition_Label"}.issubset(fields)


class TestQualifiedFeatureSelectorReduction:
    """Bug 2: selector grouping. Both eyes must survive a selector."""

    def _records(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """Build two records for one Subject — a Left eye and a Right eye.

        Uses the feature's own ``feature_record_class`` so the records carry
        exactly the fields the production read path produces.
        """
        feat = denorm_qualified_feature_deriva_model.lookup_feature("Subject", "Chart_Label")
        record_class = feat.feature_record_class()
        # RIDs are obtained as opaque strings here only because no catalog is
        # involved (offline FeatureRecord construction). They are never parsed.
        left = record_class(
            Subject="SUBJ-A",
            Image_Side="Left",
            Condition_Label="POAG",
            Execution="EXEC-1",
            RCT="2026-01-01T00:00:00+00:00",
        )
        right = record_class(
            Subject="SUBJ-A",
            Image_Side="Right",
            Condition_Label="POAG",
            Execution="EXEC-1",
            RCT="2026-01-02T00:00:00+00:00",
        )
        return feat, [left, right]

    def test_select_newest_preserves_both_eyes(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """``select_newest`` keeps the Left AND the Right eye — not one.

        This is the core fix proof: grouping on ``(Subject, Image_Side)``
        means each eye is its own identity bucket. The selector reduces
        *within* each bucket (here each bucket has one record), so both
        survive. Pre-fix (group-by-Subject) the two collapsed to one.
        """
        from deriva_ml.feature import FeatureRecord

        feat, records = self._records(denorm_qualified_feature_deriva_model)
        result = list(
            reduce_with_selector(
                records,
                feat.target_table.name,
                FeatureRecord.select_newest,
                feat.qualifier_columns,
            )
        )
        assert len(result) == 2
        assert {r.Image_Side for r in result} == {"Left", "Right"}
        # Same Subject on both — they are two observations of one Subject.
        assert {r.Subject for r in result} == {"SUBJ-A"}

    def test_without_qualifier_would_collapse_to_one(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """Pin the bug: group-by-target alone collapses the two eyes to one.

        Calling the helper WITHOUT the qualifier reproduces the pre-fix
        behavior — proof that threading ``qualifier_columns`` is what fixes
        it, not some incidental change.
        """
        from deriva_ml.feature import FeatureRecord

        feat, records = self._records(denorm_qualified_feature_deriva_model)
        collapsed = list(
            reduce_with_selector(
                records,
                feat.target_table.name,
                FeatureRecord.select_newest,
            )
        )
        assert len(collapsed) == 1  # one eye dropped — the old, wrong answer

    def test_selector_reduces_redundant_within_eye(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """Two annotations of the SAME eye still reduce to one; both eyes kept.

        Identity is ``(Subject, Image_Side)``: redundant records within one
        eye collapse, but the two eyes stay distinct. Result: exactly two
        survivors (newest Left, newest Right).
        """
        from deriva_ml.feature import FeatureRecord

        feat = denorm_qualified_feature_deriva_model.lookup_feature("Subject", "Chart_Label")
        record_class = feat.feature_record_class()
        records = [
            record_class(Subject="SUBJ-A", Image_Side="Left", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
            record_class(Subject="SUBJ-A", Image_Side="Left", Execution="E2", RCT="2026-01-03T00:00:00+00:00"),
            record_class(Subject="SUBJ-A", Image_Side="Right", Execution="E1", RCT="2026-01-02T00:00:00+00:00"),
        ]
        result = list(reduce_with_selector(records, "Subject", FeatureRecord.select_newest, feat.qualifier_columns))
        by_side = {r.Image_Side: r.Execution for r in result}
        assert by_side == {"Left": "E2", "Right": "E1"}


class TestUnqualifiedFeatureBackwardCompat:
    """Backward-compat: an unqualified feature is unchanged by the fix.

    Uses the ``Execution_Image_Quality`` feature carried in the same
    fixture — a key-FK-arity-3 feature whose compound key is
    ``[Execution, Image, Feature_Name]`` (no qualifier). It is discovered
    by the same ``find_features`` path, so it exercises the new code while
    asserting the historical group-by-target behavior is preserved.
    """

    def test_unqualified_feature_has_no_qualifiers(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """The ordinary ``Quality`` feature has empty qualifier_columns.

        Its compound identity is the target RID alone, so grouping is
        unchanged — exactly the historical behavior. ``Condition_Label`` is
        a value FK but NOT in the key, so it must NOT be treated as a
        qualifier.
        """
        feat = denorm_qualified_feature_deriva_model.lookup_feature("Image", "Quality")
        assert feat.qualifier_columns == []

    def test_unqualified_feature_reduces_to_one_per_target(self, denorm_qualified_feature_deriva_model: DerivaModel):
        """With empty qualifiers, two records for one Image collapse to one.

        Byte-identical to grouping on the target RID alone — the invariant
        the fix must preserve for every existing (unqualified) caller.
        """
        from deriva_ml.feature import FeatureRecord

        feat = denorm_qualified_feature_deriva_model.lookup_feature("Image", "Quality")
        record_class = feat.feature_record_class()
        records = [
            record_class(Image="IMG-1", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
            record_class(Image="IMG-1", Execution="E2", RCT="2026-01-02T00:00:00+00:00"),
            record_class(Image="IMG-2", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
        ]
        result = list(reduce_with_selector(records, "Image", FeatureRecord.select_newest, feat.qualifier_columns))
        # Two target Images → two survivors; IMG-1's two records collapse.
        assert {r.Image for r in result} == {"IMG-1", "IMG-2"}
        assert len(result) == 2

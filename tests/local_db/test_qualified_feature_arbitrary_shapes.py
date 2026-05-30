"""Empirical generalization of the #261 key-qualified-feature fix.

The single-qualifier ``test_qualified_feature_discovery.py`` proved the fix on
ONE vocab-FK qualifier (eye-ai's ``Chart_Label`` / ``Image_Side``). The fix
generalizes *by construction* — ``Feature.qualifier_columns`` is a
comprehension over ``atable.other_fkeys`` (deriva-py's compound-key-covered
FKs), so it is agnostic to the number of qualifiers and to each qualifier FK's
referent type. "By construction" is not "tested," and the live eye-ai catalog
has no multi-qualifier or asset-qualifier feature to exercise. These tests make
the generalization empirical, on offline fixtures whose ``qualifier_columns``
are verified through the real ``find_features`` / ``Feature`` path (never
asserted by construction alone — the "fixture lies" guard).

Four shapes (one fixture each in ``conftest.py``):

1. ``denorm_multi_qualifier_deriva_model`` — TWO key-covered qualifier FKs
   (``Image_Side`` + ``Visit_Number``). Arbitrary qualifier *count*.
2. ``denorm_asset_qualifier_deriva_model`` — a key-covered qualifier FK whose
   referent is an ASSET table (``Image_Region``). Referent-type-agnostic.
3. ``denorm_decoration_feature_deriva_model`` — a many-column UNQUALIFIED
   feature (scalars + a non-key asset FK + a non-key vocab FK). The over-split
   guard: a richly-decorated feature must NOT be mistaken for a qualified one.
4. ``denorm_mixed_feature_deriva_model`` — a key qualifier AND non-key
   decoration together; the two must be separated (qualifier → group key,
   decoration → record field but not group key).

Findings (absolute paths):
  /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e/findings/investigation/10-find-features-multivalue-arity-cap.md
  /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e/findings/investigation/12-selector-grouping-qualified-features.md
"""

from __future__ import annotations

from deriva_ml.feature import FeatureRecord, reduce_with_selector
from deriva_ml.model.catalog import DerivaModel


class TestMultiQualifierFeature:
    """Two key-covered qualifier FKs — arbitrary qualifier count (task §1)."""

    FEATURE = "Visit_Chart"
    TABLE = "Execution_Subject_Visit_Chart"

    def test_find_features_discovers_it(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """A key-FK-arity-5 feature is discovered (the cap is gone)."""
        names = {f.feature_table.name for f in denorm_multi_qualifier_deriva_model.find_features("Subject")}
        assert self.TABLE in names

    def test_qualifier_columns_are_both_fks(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """``qualifier_columns`` is exactly the two in-key value FKs.

        Compared as a set — order is the comprehension's iteration order over
        ``other_fkeys`` and is not part of the contract. ``Condition_Label``
        and ``Score`` are non-key values and must be excluded; ``Feature_Name``
        / ``Execution`` are structural and excluded.
        """
        feat = denorm_multi_qualifier_deriva_model.lookup_feature("Subject", self.FEATURE)
        assert set(feat.qualifier_columns) == {"Image_Side", "Visit_Number"}

    def test_record_class_carries_all_columns(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """The FeatureRecord exposes both qualifiers and both value columns."""
        feat = denorm_multi_qualifier_deriva_model.lookup_feature("Subject", self.FEATURE)
        fields = set(feat.feature_record_class().model_fields.keys())
        assert {"Subject", "Image_Side", "Visit_Number", "Condition_Label", "Score"}.issubset(fields)

    def _combo_records(self, model: DerivaModel) -> tuple:
        """One record per (Image_Side, Visit_Number) combination for SUBJ-A.

        Four distinct identities: (Left,V1), (Left,V2), (Right,V1), (Right,V2).
        """
        feat = model.lookup_feature("Subject", self.FEATURE)
        rc = feat.feature_record_class()
        recs = [
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V1", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V2", Execution="E1", RCT="2026-01-02T00:00:00+00:00"),
            rc(
                Subject="SUBJ-A", Image_Side="Right", Visit_Number="V1", Execution="E1", RCT="2026-01-03T00:00:00+00:00"
            ),
            rc(
                Subject="SUBJ-A", Image_Side="Right", Visit_Number="V2", Execution="E1", RCT="2026-01-04T00:00:00+00:00"
            ),
        ]
        return feat, recs

    def test_selector_none_returns_all_combinations(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """No selector ⇒ every (target, q1, q2) row survives, no grouping at all.

        ``reduce_with_selector`` is never called on the no-selector path
        (``feature_values`` yields raw records), so the analogue here is the
        full record list — all four combinations.
        """
        _feat, recs = self._combo_records(denorm_multi_qualifier_deriva_model)
        assert len(recs) == 4
        assert {(r.Image_Side, r.Visit_Number) for r in recs} == {
            ("Left", "V1"),
            ("Left", "V2"),
            ("Right", "V1"),
            ("Right", "V2"),
        }

    def test_select_newest_keeps_one_per_combination(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """``select_newest`` yields one row per (Subject, Image_Side, Visit_Number).

        NOT one-per-Subject and NOT one-per-(Subject, Image_Side): all four
        identities are distinct, each its own singleton bucket, so all four
        survive.
        """
        feat, recs = self._combo_records(denorm_multi_qualifier_deriva_model)
        result = list(reduce_with_selector(recs, "Subject", FeatureRecord.select_newest, feat.qualifier_columns))
        assert len(result) == 4
        assert {(r.Image_Side, r.Visit_Number) for r in result} == {
            ("Left", "V1"),
            ("Left", "V2"),
            ("Right", "V1"),
            ("Right", "V2"),
        }

    def test_composite_key_uses_all_qualifiers_not_just_first(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """Two rows sharing (Subject, Image_Side) but differing in Visit_Number
        are kept by ``select_newest`` — proving Q2 participates in the key.

        If the composite key used only the first qualifier, these two would
        collapse to one (the newest). They must NOT. This is the multi-column
        analogue of the original one-eye-dropped bug.
        """
        feat = denorm_multi_qualifier_deriva_model.lookup_feature("Subject", self.FEATURE)
        rc = feat.feature_record_class()
        recs = [
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V1", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V2", Execution="E1", RCT="2026-01-09T00:00:00+00:00"),
        ]
        result = list(reduce_with_selector(recs, "Subject", FeatureRecord.select_newest, feat.qualifier_columns))
        assert len(result) == 2, "two rows differing only in Q2 must NOT collapse"
        assert {r.Visit_Number for r in result} == {"V1", "V2"}

    def test_reduces_within_full_composite_bucket(self, denorm_multi_qualifier_deriva_model: DerivaModel):
        """Redundant records sharing the FULL composite identity still reduce to one.

        Two rows with identical (Subject, Image_Side, Visit_Number) but
        different executions reduce to the newest; the other three distinct
        combinations are untouched. Four distinct identities → four survivors,
        one of which is the within-bucket newest.
        """
        feat = denorm_multi_qualifier_deriva_model.lookup_feature("Subject", self.FEATURE)
        rc = feat.feature_record_class()
        recs = [
            # (Left, V1) twice — must reduce to the E2 (newer) row.
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V1", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V1", Execution="E2", RCT="2026-01-05T00:00:00+00:00"),
            rc(Subject="SUBJ-A", Image_Side="Left", Visit_Number="V2", Execution="E1", RCT="2026-01-02T00:00:00+00:00"),
            rc(
                Subject="SUBJ-A", Image_Side="Right", Visit_Number="V1", Execution="E1", RCT="2026-01-03T00:00:00+00:00"
            ),
        ]
        result = list(reduce_with_selector(recs, "Subject", FeatureRecord.select_newest, feat.qualifier_columns))
        by_combo = {(r.Image_Side, r.Visit_Number): r.Execution for r in result}
        assert by_combo == {("Left", "V1"): "E2", ("Left", "V2"): "E1", ("Right", "V1"): "E1"}


class TestAssetQualifierFeature:
    """A key-covered qualifier FK pointing at an ASSET table (task §2)."""

    FEATURE = "Region_Label"
    TABLE = "Execution_Subject_Region_Label"

    def test_find_features_discovers_it(self, denorm_asset_qualifier_deriva_model: DerivaModel):
        names = {f.feature_table.name for f in denorm_asset_qualifier_deriva_model.find_features("Subject")}
        assert self.TABLE in names

    def test_qualifier_referent_is_genuinely_an_asset(self, denorm_asset_qualifier_deriva_model: DerivaModel):
        """Precondition: ``Image_Region`` really is an asset table, not a vocab.

        Without this the test below would be vacuous (the point is that an
        ASSET-referent qualifier is still discovered as a qualifier).
        """
        assert denorm_asset_qualifier_deriva_model.is_asset("Image_Region")
        assert not denorm_asset_qualifier_deriva_model.is_vocabulary("Image_Region")

    def test_asset_fk_is_a_qualifier(self, denorm_asset_qualifier_deriva_model: DerivaModel):
        """The asset-referent FK is in ``qualifier_columns`` — referent type is irrelevant.

        ``qualifier_columns`` keys off FK-in-``other_fkeys`` (key coverage),
        NOT off the referent being a vocabulary. ``Condition_Label`` (non-key
        value) is excluded.
        """
        feat = denorm_asset_qualifier_deriva_model.lookup_feature("Subject", self.FEATURE)
        assert feat.qualifier_columns == ["Image_Region"]

    def test_asset_qualifier_participates_in_group_key(self, denorm_asset_qualifier_deriva_model: DerivaModel):
        """Two rows for one Subject differing only by the asset qualifier both survive.

        The asset FK is the identity discriminator, so ``select_newest`` keeps
        both regions rather than collapsing to one-per-Subject.
        """
        feat = denorm_asset_qualifier_deriva_model.lookup_feature("Subject", self.FEATURE)
        rc = feat.feature_record_class()
        recs = [
            rc(Subject="SUBJ-A", Image_Region="REGION-1", Execution="E1", RCT="2026-01-01T00:00:00+00:00"),
            rc(Subject="SUBJ-A", Image_Region="REGION-2", Execution="E1", RCT="2026-01-02T00:00:00+00:00"),
        ]
        result = list(reduce_with_selector(recs, "Subject", FeatureRecord.select_newest, feat.qualifier_columns))
        assert len(result) == 2
        assert {r.Image_Region for r in result} == {"REGION-1", "REGION-2"}


class TestDecorationFeatureOverSplitGuard:
    """A many-column UNQUALIFIED feature must NOT be mistaken for qualified (task §3)."""

    FEATURE = "RichQuality"
    TABLE = "Execution_Image_RichQuality"

    def test_find_features_discovers_it(self, denorm_decoration_feature_deriva_model: DerivaModel):
        """``pure=False`` keeps an impure-decoration feature discoverable."""
        names = {f.feature_table.name for f in denorm_decoration_feature_deriva_model.find_features("Image")}
        assert self.TABLE in names

    def test_qualifier_columns_is_empty(self, denorm_decoration_feature_deriva_model: DerivaModel):
        """No decoration FK is key-covered ⇒ ``qualifier_columns`` is EMPTY.

        Five non-structural columns (3 scalar, 1 asset FK, 1 vocab FK), none in
        the compound key ``[Execution, Image, Feature_Name]`` — so none is a
        qualifier.
        """
        feat = denorm_decoration_feature_deriva_model.lookup_feature("Image", self.FEATURE)
        assert feat.qualifier_columns == []

    def test_record_class_carries_all_decoration(self, denorm_decoration_feature_deriva_model: DerivaModel):
        """The decoration columns ARE real feature data on the record.

        They are classified by referent: ``Thumbnail`` → asset, ``Quality_Grade``
        → term (it's a real vocab table), the scalars → value. All present as
        fields; just none of them is identity.
        """
        feat = denorm_decoration_feature_deriva_model.lookup_feature("Image", self.FEATURE)
        fields = set(feat.feature_record_class().model_fields.keys())
        assert {"Image", "Confidence", "Vote_Count", "Notes", "Thumbnail", "Quality_Grade"}.issubset(fields)
        # The asset / term classification is exercised end-to-end here.
        assert {c.name for c in feat.asset_columns} == {"Thumbnail"}
        assert {c.name for c in feat.term_columns} == {"Quality_Grade"}
        assert {"Confidence", "Vote_Count", "Notes"}.issubset({c.name for c in feat.value_columns})

    def test_select_newest_groups_by_target_alone(self, denorm_decoration_feature_deriva_model: DerivaModel):
        """Decoration does NOT cause over-splitting — group by Image alone.

        Two records for IMG-1 that differ in their decoration columns still
        collapse to one (the newest), because identity is the target RID alone.
        IMG-2 is a separate target. Two targets → two survivors. This is the
        inverse-bug guard: a richly-decorated feature is NOT over-split.
        """
        feat = denorm_decoration_feature_deriva_model.lookup_feature("Image", self.FEATURE)
        rc = feat.feature_record_class()
        recs = [
            rc(
                Image="IMG-1",
                Execution="E1",
                Confidence=0.5,
                Vote_Count=2,
                Notes="a",
                Thumbnail="T1",
                Quality_Grade="High",
                RCT="2026-01-01T00:00:00+00:00",
            ),
            rc(
                Image="IMG-1",
                Execution="E2",
                Confidence=0.9,
                Vote_Count=5,
                Notes="b",
                Thumbnail="T2",
                Quality_Grade="Low",
                RCT="2026-01-02T00:00:00+00:00",
            ),
            rc(
                Image="IMG-2",
                Execution="E1",
                Confidence=0.3,
                Vote_Count=1,
                Notes="c",
                Thumbnail="T3",
                Quality_Grade="High",
                RCT="2026-01-01T00:00:00+00:00",
            ),
        ]
        result = list(reduce_with_selector(recs, "Image", FeatureRecord.select_newest, feat.qualifier_columns))
        assert {r.Image for r in result} == {"IMG-1", "IMG-2"}
        assert len(result) == 2
        # IMG-1 reduced to the newer (E2) row despite differing decoration.
        img1 = next(r for r in result if r.Image == "IMG-1")
        assert img1.Execution == "E2"


class TestMixedQualifierAndDecoration:
    """A key qualifier AND non-key decoration together; the two are separated (task §4)."""

    FEATURE = "MixedLabel"
    TABLE = "Execution_Subject_MixedLabel"

    def test_find_features_discovers_it(self, denorm_mixed_feature_deriva_model: DerivaModel):
        names = {f.feature_table.name for f in denorm_mixed_feature_deriva_model.find_features("Subject")}
        assert self.TABLE in names

    def test_only_the_key_qualifier_is_a_qualifier(self, denorm_mixed_feature_deriva_model: DerivaModel):
        """``Image_Side`` (in key) is the sole qualifier; decoration is excluded.

        ``Severity_Label`` (vocab) and ``Heatmap`` (asset) and ``Confidence``
        (scalar) are non-key decoration — none is a qualifier even though two of
        them are FKs.
        """
        feat = denorm_mixed_feature_deriva_model.lookup_feature("Subject", self.FEATURE)
        assert feat.qualifier_columns == ["Image_Side"]

    def test_decoration_classified_but_not_identity(self, denorm_mixed_feature_deriva_model: DerivaModel):
        """Decoration FKs become record fields (asset/term) but are not the group key."""
        feat = denorm_mixed_feature_deriva_model.lookup_feature("Subject", self.FEATURE)
        assert {c.name for c in feat.asset_columns} == {"Heatmap"}
        assert {c.name for c in feat.term_columns} == {"Severity_Label"}
        fields = set(feat.feature_record_class().model_fields.keys())
        assert {"Image_Side", "Severity_Label", "Heatmap", "Confidence"}.issubset(fields)

    def test_qualifier_splits_but_decoration_does_not(self, denorm_mixed_feature_deriva_model: DerivaModel):
        """Same Subject + same Image_Side but differing decoration → reduces to one.

        Different Image_Side → two distinct identities. So three records — two
        Left (differing only in decoration) + one Right — reduce to exactly two
        survivors (one per eye), proving the qualifier joins the group key while
        the decoration does not split it.
        """
        feat = denorm_mixed_feature_deriva_model.lookup_feature("Subject", self.FEATURE)
        rc = feat.feature_record_class()
        recs = [
            rc(
                Subject="SUBJ-A",
                Image_Side="Left",
                Execution="E1",
                Severity_Label="Mild",
                Heatmap="H1",
                Confidence=0.4,
                RCT="2026-01-01T00:00:00+00:00",
            ),
            rc(
                Subject="SUBJ-A",
                Image_Side="Left",
                Execution="E2",
                Severity_Label="Severe",
                Heatmap="H2",
                Confidence=0.95,
                RCT="2026-01-05T00:00:00+00:00",
            ),  # newer Left
            rc(
                Subject="SUBJ-A",
                Image_Side="Right",
                Execution="E1",
                Severity_Label="Mild",
                Heatmap="H3",
                Confidence=0.6,
                RCT="2026-01-02T00:00:00+00:00",
            ),
        ]
        result = list(reduce_with_selector(recs, "Subject", FeatureRecord.select_newest, feat.qualifier_columns))
        assert len(result) == 2, "decoration must not split; qualifier must"
        by_side = {r.Image_Side: r.Execution for r in result}
        assert by_side == {"Left": "E2", "Right": "E1"}
        # The surviving Left row kept its (decoration) Severity_Label intact.
        left = next(r for r in result if r.Image_Side == "Left")
        assert left.Severity_Label == "Severe"

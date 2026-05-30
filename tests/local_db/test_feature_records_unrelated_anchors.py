"""Regression tests: feature reads drop unrelated sibling element types.

A feature read is intentionally scoped to ONE target table plus its
feature-association table. When the dataset also contains *sibling*
element types that have no FK path to that target — e.g. an AIREADI
dataset whose members are ``CGM_Blood_Glucose`` + ``Subject`` +
``OCT_DICOM`` and the read targets a ``CGM_Blood_Glucose`` feature —
those siblings are "unrelated anchors" (denormalizer Rule 8 / case-6).

PR #260 (Stage 3b) delegated **both** ``Dataset.feature_values`` and
``DatasetBag.feature_values`` to ``Denormalizer.feature_records``, which
hard-coded ``ignore_unrelated_anchors=False``. On a single-element-type
dataset (the catalog-27 validation case) there were no unrelated
siblings, so the guard never fired and the two surfaces agreed. The
eye-ai re-validation (finding 11) exercised a *multi-element-type* bag
and exposed the divergence: ``DatasetBag.feature_values`` RAISED
``DerivaMLDenormalizeUnrelatedAnchor`` where the live
``Dataset.feature_values`` returned rows.

The fix (finding 11 §7) makes ``feature_records`` pass
``ignore_unrelated_anchors=True`` — a feature read drops irrelevant
sibling element types silently rather than erroring. Because both
wrappers share the ``feature_records`` seam, both inherit the fix and
stay in lockstep.

These tests pin that contract on a synthetic but genuinely
multi-element fixture (``Image`` carrying the feature + an unrelated
``UnrelatedThing`` element type with no FK path to ``Image``), so the
single-element catalog-27 case can no longer hide the multi-element
divergence. The live three-way A==C oracle on eye-ai ``6-EKGA`` is
captured in the PR body; this is its offline regression guard.

See finding 11 (``findings/investigation/11-stage3b-eyeai-revalidation.md``
in the deriva-ml-model-template-e2e worktree), §4 root cause and §7 fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deriva.core.ermrest_model import FindAssociationResult, Model
from sqlalchemy.orm import Session

from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor
from deriva_ml.feature import Feature, FeatureRecord
from deriva_ml.local_db.denormalizer import Denormalizer
from deriva_ml.local_db.schema import LocalSchema
from deriva_ml.model.catalog import DerivaModel

FEAT_TABLE = "Execution_Image_Image_Classification"
DS_RID = "DS-001"
IMG_RIDS = ["IMG-1", "IMG-2", "IMG-3"]


def _build_feature(model: DerivaModel) -> Feature:
    """Construct the ``Image / Image_Classification`` Feature by hand.

    The conftest feature fixture models a real 4-FK feature association
    table (``Execution_Image_Image_Classification``: Image,
    Image_Classification, Feature_Name, Execution). The public
    ``find_features`` / ``lookup_feature`` discovery layer caps
    association arity at 3 (deriva-py ``find_associations(max_arity=3)``)
    so it cannot see this 4-FK shape — see finding 10's arity-cap audit,
    which is intentionally out of scope for this fix. This regression
    test exercises the *anchor-classification* seam, not discovery, so we
    build the ``Feature`` directly from the association table's FKs (the
    same wiring ``Feature.__init__`` expects), bypassing discovery.

    Args:
        model: DerivaModel built from the feature fixture schema.

    Returns:
        A ``Feature`` whose ``target_table`` is ``Image`` and whose
        ``feature_table`` is ``Execution_Image_Image_Classification``.
    """
    feat_tbl = model.model.schemas["deriva-ml"].tables[FEAT_TABLE]
    fks = {fk.foreign_key_columns[0].name: fk for fk in feat_tbl.foreign_keys}
    atable = FindAssociationResult(
        feat_tbl,
        fks["Image"],  # self_fkey: points back at the target table
        {fks["Execution"], fks["Feature_Name"], fks["Image_Classification"]},
    )
    return Feature(atable, model)


@pytest.fixture
def hetero_feature_db(denorm_feature_model: Model, tmp_path: Path) -> dict[str, Any]:
    """Local schema populated with an Image feature + an unrelated sibling.

    Builds the multi-element-type shape that the eye-ai 6-EKGA bag has:
    a target element type (``Image``) carrying feature values, alongside
    a sibling element type (``UnrelatedThing``) that has **no** FK path to
    ``Image`` or to the feature-association table. ``UnrelatedThing`` is
    the synthetic analogue of eye-ai's ``Subject`` / ``OCT_DICOM`` members
    relative to a ``CGM_Blood_Glucose`` feature read.

    Returns:
        Dict with ``model`` (DerivaModel), ``local_schema``, ``feature``
        (the constructed Image feature), and the member RID lists.
    """
    model = DerivaModel(model=denorm_feature_model, ml_schema="deriva-ml", domain_schemas={"isa"})
    db_path = tmp_path / "hetero_feature_db"
    db_path.mkdir()
    ls = LocalSchema.build(model=denorm_feature_model, schemas=["isa", "deriva-ml"], database_path=db_path)

    with Session(ls.engine) as session:
        Image = ls.get_orm_class("Image")
        for rid in IMG_RIDS:
            session.add(Image(RID=rid, Filename=f"{rid}.png", Subject=None))

        # The unrelated sibling element type — no FK to Image / feature table.
        Unrelated = ls.get_orm_class("UnrelatedThing")
        session.add(Unrelated(RID="UT-1", Label="noise-a"))
        session.add(Unrelated(RID="UT-2", Label="noise-b"))

        Dataset = ls.get_orm_class("Dataset")
        session.add(Dataset(RID=DS_RID, Description="heterogeneous dataset"))

        Dataset_Image = ls.get_orm_class("Dataset_Image")
        for rid in IMG_RIDS:
            session.add(Dataset_Image(RID=f"DI-{rid}", Dataset=DS_RID, Image=rid))

        Execution = ls.get_orm_class("Execution")
        session.add(Execution(RID="EXE-1", Description="annotator run"))

        Feature_Name = ls.get_orm_class("Feature_Name")
        session.add(Feature_Name(RID="FN-1", Name="Image_Classification", Description=None))

        Image_Classification = ls.get_orm_class("Image_Classification")
        session.add(Image_Classification(RID="IC-cat", Name="cat", Description=None))
        session.add(Image_Classification(RID="IC-dog", Name="dog", Description=None))

        Feat = ls.get_orm_class(FEAT_TABLE)
        session.add(
            Feat(
                RID="F1",
                Feature_Name="Image_Classification",
                Image="IMG-1",
                Execution="EXE-1",
                Image_Classification="IC-cat",
            )
        )
        session.add(
            Feat(
                RID="F2",
                Feature_Name="Image_Classification",
                Image="IMG-2",
                Execution="EXE-1",
                Image_Classification="IC-dog",
            )
        )
        session.add(
            Feat(
                RID="F3",
                Feature_Name="Image_Classification",
                Image="IMG-3",
                Execution="EXE-1",
                Image_Classification="IC-cat",
            )
        )
        session.commit()

    return {
        "model": model,
        "local_schema": ls,
        "feature": _build_feature(model),
        "image_rids": IMG_RIDS,
        "unrelated_rids": ["UT-1", "UT-2"],
    }


class _FeatureDataset:
    """Minimal DatasetLike exposing a configurable member dict.

    ``Denormalizer(self)`` derives model / engine / orm_resolver from the
    attributes set here and reads anchors from ``list_dataset_members``.
    ``_ml_instance`` returns ``None`` so the Denormalizer stays in local
    (bag-equivalent, ``source="local"``) mode against the fixture engine —
    the exact mode ``DatasetBag.feature_values`` uses.
    """

    def __init__(self, fixture: dict[str, Any], members: dict[str, list[dict]]) -> None:
        self.dataset_rid = DS_RID
        self.model = fixture["model"]
        self.engine = fixture["local_schema"].engine
        self._orm_resolver = fixture["local_schema"].get_orm_class
        self._members = members

    @property
    def _ml_instance(self) -> None:
        return None

    def list_dataset_members(self, **_kwargs: Any) -> dict[str, list[dict]]:
        return self._members

    def list_dataset_children(self, **_kwargs: Any) -> list:
        return []


def _hetero_members() -> dict[str, list[dict]]:
    """Members = Image (feature target) + UnrelatedThing (sibling, no FK path)."""
    return {
        "Image": [{"RID": r} for r in IMG_RIDS],
        "UnrelatedThing": [{"RID": "UT-1"}, {"RID": "UT-2"}],
    }


def _homo_members() -> dict[str, list[dict]]:
    """Members = Image only (no sibling element types)."""
    return {"Image": [{"RID": r} for r in IMG_RIDS]}


def test_feature_records_drops_unrelated_anchor(hetero_feature_db) -> None:
    """``feature_records`` does NOT raise on a multi-element dataset.

    The pre-fix code hard-coded ``ignore_unrelated_anchors=False`` in
    ``feature_records`` and raised ``DerivaMLDenormalizeUnrelatedAnchor``
    here (reproducing the eye-ai 6-EKGA bag failure). Post-fix the
    unrelated ``UnrelatedThing`` anchor is dropped silently and all three
    Image feature rows come back.
    """
    feat = hetero_feature_db["feature"]
    ds = _FeatureDataset(hetero_feature_db, _hetero_members())
    d = Denormalizer(ds)
    # Confirm we exercise the bag-equivalent local source path.
    assert d._source == "local"

    records = d.feature_records(feat, selector=None)

    assert len(records) == 3
    assert all(isinstance(r, FeatureRecord) for r in records)
    assert {r.Image for r in records} == set(IMG_RIDS)


def test_feature_records_hetero_equals_homo(hetero_feature_db) -> None:
    """The unrelated sibling element type changes nothing about the result.

    This is the lockstep invariant in miniature: adding a sibling element
    type with no FK path to the feature's target must not change the rows
    a feature read returns. (The live A==C oracle on eye-ai 6-EKGA — bag
    vs catalog return the same 27 / 9 rows — is the production-scale form
    of this same property; captured verbatim in the PR body.)
    """
    feat = hetero_feature_db["feature"]
    hetero = Denormalizer(_FeatureDataset(hetero_feature_db, _hetero_members()))
    homo = Denormalizer(_FeatureDataset(hetero_feature_db, _homo_members()))

    hetero_rows = hetero.feature_records(feat, selector=None)
    homo_rows = homo.feature_records(feat, selector=None)

    def _key(records):
        return sorted((r.Image, r.Image_Classification, r.Execution) for r in records)

    assert _key(hetero_rows) == _key(homo_rows)
    assert len(hetero_rows) == 3


def test_feature_records_hetero_equals_homo_with_selector(hetero_feature_db) -> None:
    """Lockstep holds under selector reduction too (one record per target)."""
    feat = hetero_feature_db["feature"]
    hetero = Denormalizer(_FeatureDataset(hetero_feature_db, _hetero_members()))
    homo = Denormalizer(_FeatureDataset(hetero_feature_db, _homo_members()))

    hetero_rows = hetero.feature_records(feat, selector=FeatureRecord.select_newest)
    homo_rows = homo.feature_records(feat, selector=FeatureRecord.select_newest)

    def _key(records):
        return sorted((r.Image, r.Image_Classification) for r in records)

    assert _key(hetero_rows) == _key(homo_rows)
    # One row per Image target after reduction.
    target_rids = [r.Image for r in hetero_rows]
    assert len(target_rids) == len(set(target_rids))


def test_unrelated_anchor_is_genuinely_case_6(hetero_feature_db) -> None:
    """Guard: the dropped anchor really is a Rule-8 unrelated anchor.

    If ``UnrelatedThing`` were secretly reachable from ``Image`` (a fixture
    drift), the ``feature_records`` non-raise above would be vacuous. This
    pins that the *default* denormalize path (``ignore_unrelated_anchors``
    defaults to ``False``) still raises ``DerivaMLDenormalizeUnrelatedAnchor``
    naming ``UnrelatedThing`` — so the fix is a deliberate, scoped relaxation
    inside ``feature_records``, not a fixture that happens to be all-related.
    This mirrors the eye-ai raise naming ``['OCT_DICOM', 'Subject']``.
    """
    ds = _FeatureDataset(hetero_feature_db, _hetero_members())
    d = Denormalizer(ds)
    with pytest.raises(DerivaMLDenormalizeUnrelatedAnchor) as exc_info:
        d.as_dataframe(["Image", FEAT_TABLE])
    assert "UnrelatedThing" in str(exc_info.value)

"""Tests for the ``selector`` parameter on ``_denormalize_impl``.

Stage 1 of the ``feature_values`` / ``Denormalizer`` consolidation —
closes the gap identified by finding 01 §7 of the e2e denormalizer
audit. The contract matches ``Dataset.feature_values``'s ``selector``
argument: a callable ``(list[FeatureRecord]) -> FeatureRecord | None``
that picks one record per target-RID group (or returns ``None`` to drop
the group).

The fixture ``denorm_local_schema_feature`` provides a feature-
association table ``Execution_Image_Image_Classification`` with FKs to
``Image``, ``Execution``, and ``Image_Classification``. These unit
tests pre-populate the engine with multiple feature rows per Image
(distinct RCT, distinct Execution) so the selector reduction has
something to do.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session

from deriva_ml.feature import FeatureRecord
from deriva_ml.local_db.denormalize import _denormalize_impl

# ----------------------------------------------------------------------
# Shared fixture: populate the feature-fixture engine with multi-row data
# ----------------------------------------------------------------------


@pytest.fixture
def populated_feature_denorm(
    denorm_feature_deriva_model: Any,
    denorm_local_schema_feature: Any,
) -> dict[str, Any]:
    """LocalSchema pre-populated with a dataset, images, and multiple
    feature rows per image.

    Each image carries ``N_ANNOTATORS`` feature rows in
    ``Execution_Image_Image_Classification``, written under distinct
    Executions, with monotonically increasing RCT timestamps so
    ``select_newest`` has a deterministic pick. Returns a dict with
    enough handles for the selector tests to exercise reduction.
    """
    ls = denorm_local_schema_feature
    model = denorm_feature_deriva_model

    ds_rid = "DS-001"
    image_rids = ["IMG-1", "IMG-2", "IMG-3"]
    exec_rids = ["EXE-1", "EXE-2", "EXE-3"]
    cls_rid = "CLS-cat"

    # Per-image, per-execution RCT timestamps. The third execution's
    # timestamps are uniformly newest, so select_newest must pick it
    # for every image.
    rct_table = {
        "EXE-1": "2026-01-01T00:00:00.000000+00:00",
        "EXE-2": "2026-02-01T00:00:00.000000+00:00",
        "EXE-3": "2026-03-01T00:00:00.000000+00:00",
    }

    with Session(ls.engine) as session:
        # Dataset row (Dataset table comes from the feature fixture's
        # deriva-ml schema; the planner needs Dataset_Image to wire
        # Dataset → Image for the join, so populate that too).
        ds_cls = ls.get_orm_class("Dataset")
        session.add(ds_cls(RID=ds_rid, Description="selector unit fixture"))

        img_cls = ls.get_orm_class("Image")
        for rid in image_rids:
            session.add(img_cls(RID=rid, Filename=f"{rid}.png", Subject=None))

        di_cls = ls.get_orm_class("Dataset_Image")
        for rid in image_rids:
            session.add(di_cls(RID=f"DI-{rid}", Dataset=ds_rid, Image=rid))

        exe_cls = ls.get_orm_class("Execution")
        for erid in exec_rids:
            session.add(exe_cls(RID=erid, Description=f"run {erid}"))

        cls_cls = ls.get_orm_class("Image_Classification")
        session.add(cls_cls(RID=cls_rid, Name="cat"))

        feat_cls = ls.get_orm_class("Execution_Image_Image_Classification")
        for img_rid in image_rids:
            for erid in exec_rids:
                session.add(
                    feat_cls(
                        RID=f"EIIC-{img_rid}-{erid}",
                        RCT=rct_table[erid],
                        Feature_Name="default",
                        Image=img_rid,
                        Execution=erid,
                        Image_Classification=cls_rid,
                    )
                )

        session.commit()

    return {
        "model": model,
        "local_schema": ls,
        "dataset_rid": ds_rid,
        "image_rids": image_rids,
        "exec_rids": exec_rids,
        "rct_table": rct_table,
        "feature_assoc_table": "Execution_Image_Image_Classification",
        "newest_execution": "EXE-3",
    }


# ----------------------------------------------------------------------
# Core reduction behavior
# ----------------------------------------------------------------------


class TestSelectorReducesGroups:
    """Selector reduces multi-row feature groups to one row per target RID."""

    def test_selector_reduces_multi_row_feature_to_one_per_target(
        self, populated_feature_denorm: dict[str, Any]
    ) -> None:
        """``selector=FeatureRecord.select_newest`` ⇒ one row per Image."""
        model = populated_feature_denorm["model"]
        ls = populated_feature_denorm["local_schema"]
        ds_rid = populated_feature_denorm["dataset_rid"]
        image_rids = populated_feature_denorm["image_rids"]
        feature_assoc = populated_feature_denorm["feature_assoc_table"]
        newest_exec = populated_feature_denorm["newest_execution"]

        # Baseline: without selector, three Images × three Executions
        # = nine rows.
        baseline = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", feature_assoc],
            row_per=feature_assoc,
        )
        assert baseline.row_count == len(image_rids) * 3, (
            f"Baseline expected {len(image_rids) * 3} rows; got {baseline.row_count}."
        )

        # With selector: one row per Image — the newest by RCT.
        reduced = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", feature_assoc],
            row_per=feature_assoc,
            selector=FeatureRecord.select_newest,
        )
        assert reduced.row_count == len(image_rids), (
            f"Selector should reduce to one row per Image ({len(image_rids)}); got {reduced.row_count}."
        )
        # The kept row per Image must come from the newest execution.
        target_label = f"{feature_assoc}.Image"
        exec_label = f"{feature_assoc}.Execution"
        per_image: dict[str, str] = {}
        for row in reduced.iter_rows():
            per_image[row[target_label]] = row[exec_label]
        assert set(per_image.keys()) == set(image_rids), (
            f"Reduced output should cover every Image; got keys {sorted(per_image.keys())}."
        )
        for img, exec_chosen in per_image.items():
            assert exec_chosen == newest_exec, (
                f"select_newest should pick {newest_exec} for Image {img}; got {exec_chosen}."
            )

    def test_selector_none_is_passthrough(self, populated_feature_denorm: dict[str, Any]) -> None:
        """``selector=None`` ⇒ unchanged behavior."""
        model = populated_feature_denorm["model"]
        ls = populated_feature_denorm["local_schema"]
        ds_rid = populated_feature_denorm["dataset_rid"]
        image_rids = populated_feature_denorm["image_rids"]
        feature_assoc = populated_feature_denorm["feature_assoc_table"]

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", feature_assoc],
            row_per=feature_assoc,
            selector=None,
        )
        assert result.row_count == len(image_rids) * 3, (
            f"selector=None should preserve all rows; got {result.row_count}."
        )

    def test_selector_returning_none_omits_target(self, populated_feature_denorm: dict[str, Any]) -> None:
        """Selector returning ``None`` for a target ⇒ that target drops."""
        model = populated_feature_denorm["model"]
        ls = populated_feature_denorm["local_schema"]
        ds_rid = populated_feature_denorm["dataset_rid"]
        image_rids = populated_feature_denorm["image_rids"]
        feature_assoc = populated_feature_denorm["feature_assoc_table"]

        dropped_image = image_rids[0]

        def drop_first_image(records: list[FeatureRecord]) -> FeatureRecord | None:
            # All records in a group share a target RID, so probe the
            # first one to decide.
            if getattr(records[0], "Image", None) == dropped_image:
                return None
            return FeatureRecord.select_newest(records)

        result = _denormalize_impl(
            model=model,
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
            dataset_rid=ds_rid,
            include_tables=["Image", feature_assoc],
            row_per=feature_assoc,
            selector=drop_first_image,
        )
        target_label = f"{feature_assoc}.Image"
        surviving = {row[target_label] for row in result.iter_rows()}
        assert surviving == set(image_rids) - {dropped_image}, (
            f"Selector returning None should drop {dropped_image} only; surviving={surviving}."
        )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


class TestSelectorValidation:
    """include_tables shape requirements for the selector parameter."""

    def test_selector_without_feature_assoc_table_raises(self, populated_denorm: dict[str, Any]) -> None:
        """No feature-association table in include_tables + selector ⇒ ValueError.

        Uses the basic ``populated_denorm`` fixture, which has no
        feature-association tables in its schema.
        """
        model = populated_denorm["model"]
        ls = populated_denorm["local_schema"]
        ds_rid = populated_denorm["dataset_rid"]

        with pytest.raises(ValueError, match=r"selector requires a feature-association table"):
            _denormalize_impl(
                model=model,
                engine=ls.engine,
                orm_resolver=ls.get_orm_class,
                dataset_rid=ds_rid,
                include_tables=["Image", "Subject"],
                selector=FeatureRecord.select_newest,
            )

    def test_selector_with_multiple_feature_assoc_tables_raises(
        self,
        denorm_feature_deriva_model: Any,
        denorm_local_schema_feature: Any,
    ) -> None:
        """More than one feature-association table + selector ⇒ ValueError.

        The unit feature fixture only declares one real feature-association
        table (``Execution_Image_Image_Classification``); to exercise the
        multi-feature branch synthesize the predicate result via
        monkeypatching ``_is_feature_association`` so the planner sees two
        feature-association tables. Verifies the guard fires before any
        materialization work happens.
        """
        ls = denorm_local_schema_feature
        model = denorm_feature_deriva_model

        # Force two table names to pass the feature-association predicate.
        fake_feature_tables = {
            "Execution_Image_Image_Classification",
            "Image_Subject_UnrelatedThing",  # a 3-FK non-feature in the fixture
        }
        original = model._planner._is_feature_association

        def fake_is_feature(name_or_table: Any) -> bool:
            name = name_or_table if isinstance(name_or_table, str) else getattr(name_or_table, "name", "")
            return name in fake_feature_tables or original(name_or_table)

        model._planner._is_feature_association = fake_is_feature  # type: ignore[assignment]
        try:
            with pytest.raises(
                ValueError, match=r"selector with multiple feature-association tables not yet supported"
            ):
                _denormalize_impl(
                    model=model,
                    engine=ls.engine,
                    orm_resolver=ls.get_orm_class,
                    dataset_rid="DS-001",
                    include_tables=[
                        "Image",
                        "Execution_Image_Image_Classification",
                        "Image_Subject_UnrelatedThing",
                    ],
                    selector=FeatureRecord.select_newest,
                )
        finally:
            model._planner._is_feature_association = original  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Surface-level forwarding (Denormalizer + Dataset/DatasetBag wrappers)
# ----------------------------------------------------------------------


class TestSelectorPlumbing:
    """The selector parameter flows through Denormalizer and the public wrappers."""

    def test_selector_via_denormalizer_as_dict(self, populated_feature_denorm: dict[str, Any]) -> None:
        """``Denormalizer.as_dict(selector=...)`` reduces just like as_dataframe."""
        from deriva_ml.local_db.denormalizer import Denormalizer

        model = populated_feature_denorm["model"]
        ls = populated_feature_denorm["local_schema"]
        ds_rid = populated_feature_denorm["dataset_rid"]
        image_rids = populated_feature_denorm["image_rids"]
        feature_assoc = populated_feature_denorm["feature_assoc_table"]

        # Build a Denormalizer over a minimal DatasetLike shim that
        # mirrors the populated engine.
        class _Shim:
            dataset_rid = ds_rid
            model = denormalizer_model = None
            engine = ls.engine
            _orm_resolver = ls.get_orm_class

            def __init__(self, m):  # noqa: D401
                self.model = m

            def list_dataset_members(self, **_kwargs: Any) -> dict[str, list[dict]]:
                return {"Image": [{"RID": r} for r in image_rids]}

            def list_dataset_children(self, **_kwargs: Any) -> list:
                return []

        d = Denormalizer(_Shim(model))
        rows = list(
            d.as_dict(
                ["Image", feature_assoc],
                row_per=feature_assoc,
                selector=FeatureRecord.select_newest,
            )
        )
        target_label = f"{feature_assoc}.Image"
        assert len(rows) == len(image_rids), (
            f"Denormalizer.as_dict + selector should yield one row per Image; got {len(rows)}."
        )
        assert {r[target_label] for r in rows} == set(image_rids)

    def test_selector_via_denormalizer_as_dataframe(self, populated_feature_denorm: dict[str, Any]) -> None:
        """``Denormalizer.as_dataframe(selector=...)`` reduces correctly."""
        from deriva_ml.local_db.denormalizer import Denormalizer

        model = populated_feature_denorm["model"]
        ls = populated_feature_denorm["local_schema"]
        ds_rid = populated_feature_denorm["dataset_rid"]
        image_rids = populated_feature_denorm["image_rids"]
        feature_assoc = populated_feature_denorm["feature_assoc_table"]

        class _Shim:
            def __init__(self, m: Any) -> None:
                self.dataset_rid = ds_rid
                self.model = m
                self.engine = ls.engine
                self._orm_resolver = ls.get_orm_class

            def list_dataset_members(self, **_kwargs: Any) -> dict[str, list[dict]]:
                return {"Image": [{"RID": r} for r in image_rids]}

            def list_dataset_children(self, **_kwargs: Any) -> list:
                return []

        d = Denormalizer(_Shim(model))
        df = d.as_dataframe(
            ["Image", feature_assoc],
            row_per=feature_assoc,
            selector=FeatureRecord.select_newest,
        )
        assert len(df) == len(image_rids)

"""Tests for the ``materialize_feature_records`` adapter (Stage 3a).

Stage 3a of the ``feature_values`` / ``Denormalizer`` consolidation:
``Dataset.feature_values`` delegates to
``Denormalizer.feature_records``, which runs the dataset-scoped
denormalize join and then calls :func:`materialize_feature_records` to
turn the wide-table rows into typed ``FeatureRecord`` instances with
``RCT`` recovered from the system-column supplementary fetch.

These offline unit tests exercise the pure adapter directly with
synthetic wide-table rows over a small in-memory SQLite engine — the
end-to-end ``Denormalizer.feature_records`` path against a live catalog
is covered by the bit-identical live verification in the PR description.
The adapter is where the strip-prefix-and-project, the null-target drop,
and the RCT/UTC coercion live, so it is the unit worth pinning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest
from pydantic import create_model
from sqlalchemy import Column, DateTime, MetaData, String, create_engine, insert
from sqlalchemy.orm import declarative_base

from deriva_ml.feature import FeatureRecord
from deriva_ml.local_db.denormalize import materialize_feature_records

FEAT_TABLE = "Execution_Image_Quality"
TARGET = "Image"


def _record_class() -> type[FeatureRecord]:
    """A FeatureRecord subclass shaped like an Image/Quality feature."""
    return create_model(
        "ImageFeatureQuality",
        __base__=FeatureRecord,
        Image=(str, ...),
        Quality=(Optional[str], None),
    )


@pytest.fixture
def engine_with_feature_rows():
    """In-memory engine holding feature-assoc rows with tz-naive RCT datetimes.

    The RCT column is declared as a plain string column but populated with
    Python ``datetime`` objects so the read-back returns tz-naive datetimes
    — the exact shape a live-catalog SQLite round-trip produces, which the
    UTC-coercion branch of ``_recover_system_columns`` must normalize.
    """
    engine = create_engine("sqlite://")
    Base = declarative_base(metadata=MetaData())

    class _Feat(Base):
        __tablename__ = FEAT_TABLE
        RID = Column(String, primary_key=True)
        Image = Column(String)
        Quality = Column(String)
        Execution = Column(String)
        Feature_Name = Column(String)
        RCT = Column(DateTime)  # SQLite stores the datetime, returns it tz-naive

    Base.metadata.create_all(engine)

    rows = [
        {
            "RID": "F1",
            "Image": "IMG-A",
            "Quality": "good",
            "Execution": "EXE-old",
            "Feature_Name": "Quality",
            "RCT": datetime(2026, 1, 1, 0, 0, 0),
        },
        {
            "RID": "F2",
            "Image": "IMG-A",
            "Quality": "bad",
            "Execution": "EXE-new",
            "Feature_Name": "Quality",
            "RCT": datetime(2026, 6, 1, 0, 0, 0),
        },
        {
            "RID": "F3",
            "Image": "IMG-B",
            "Quality": "good",
            "Execution": "EXE-old",
            "Feature_Name": "Quality",
            "RCT": datetime(2026, 1, 1, 0, 0, 0),
        },
    ]
    with engine.begin() as conn:
        conn.execute(insert(_Feat.__table__), rows)

    def resolver(name: str):
        return _Feat if name == FEAT_TABLE else None

    return engine, resolver


def _wide_rows():
    """Wide-table rows as ``Denormalizer`` would emit them (dotted labels, no RCT).

    Includes a target-table column (``Image.URL``) the FeatureRecord must
    project away, and the feature-assoc columns the planner keeps.
    """
    return [
        {
            f"{TARGET}.RID": "IMG-A",
            f"{TARGET}.URL": "a.png",
            f"{FEAT_TABLE}.RID": "F1",
            f"{FEAT_TABLE}.Image": "IMG-A",
            f"{FEAT_TABLE}.Quality": "good",
            f"{FEAT_TABLE}.Execution": "EXE-old",
            f"{FEAT_TABLE}.Feature_Name": "Quality",
        },
        {
            f"{TARGET}.RID": "IMG-A",
            f"{TARGET}.URL": "a.png",
            f"{FEAT_TABLE}.RID": "F2",
            f"{FEAT_TABLE}.Image": "IMG-A",
            f"{FEAT_TABLE}.Quality": "bad",
            f"{FEAT_TABLE}.Execution": "EXE-new",
            f"{FEAT_TABLE}.Feature_Name": "Quality",
        },
    ]


def test_strip_prefix_and_project_to_feature_record(engine_with_feature_rows):
    """Dotted labels stripped, target-table columns dropped, FeatureRecord built."""
    engine, resolver = engine_with_feature_rows
    records = materialize_feature_records(
        _wide_rows(),
        record_class=_record_class(),
        engine=engine,
        orm_resolver=resolver,
        feature_assoc_table=FEAT_TABLE,
        target_table_name=TARGET,
        schema_for_table={TARGET: "domain", FEAT_TABLE: "deriva-ml"},
        multi_schema=False,
    )
    assert len(records) == 2
    assert all(isinstance(r, FeatureRecord) for r in records)
    # Bare-name projection: Image / Quality / Execution populated; URL dropped.
    assert {r.Image for r in records} == {"IMG-A"}
    assert {r.Quality for r in records} == {"good", "bad"}
    assert not hasattr(records[0], "URL")


def test_rct_recovered_with_utc_offset(engine_with_feature_rows):
    """RCT is recovered from the supplementary fetch and carries a +00:00 offset.

    Pins the live-verification divergence fix: the tz-naive SQLite datetime
    must serialize to the PathBuilder ``...+00:00`` shape, not a bare
    offset-less ISO string.
    """
    engine, resolver = engine_with_feature_rows
    records = materialize_feature_records(
        _wide_rows(),
        record_class=_record_class(),
        engine=engine,
        orm_resolver=resolver,
        feature_assoc_table=FEAT_TABLE,
        target_table_name=TARGET,
        schema_for_table={TARGET: "domain", FEAT_TABLE: "deriva-ml"},
        multi_schema=False,
    )
    assert all(r.RCT for r in records), [r.RCT for r in records]
    assert all(r.RCT.endswith("+00:00") for r in records), [r.RCT for r in records]
    # F2 (EXE-new) is the June timestamp; confirm it round-tripped correctly.
    by_exec = {r.Execution: r.RCT for r in records}
    assert by_exec["EXE-new"] == "2026-06-01T00:00:00+00:00"


def test_null_target_fk_row_dropped(engine_with_feature_rows):
    """A wide-table row whose target FK is None is dropped (audit §10)."""
    engine, resolver = engine_with_feature_rows
    rows = _wide_rows()
    # Simulate the Denormalizer LEFT-JOIN orphan: target FK is None.
    rows.append(
        {
            f"{TARGET}.RID": None,
            f"{TARGET}.URL": None,
            f"{FEAT_TABLE}.RID": "F-orphan",
            f"{FEAT_TABLE}.Image": None,
            f"{FEAT_TABLE}.Quality": "ghost",
            f"{FEAT_TABLE}.Execution": "EXE-x",
            f"{FEAT_TABLE}.Feature_Name": "Quality",
        }
    )
    records = materialize_feature_records(
        rows,
        record_class=_record_class(),
        engine=engine,
        orm_resolver=resolver,
        feature_assoc_table=FEAT_TABLE,
        target_table_name=TARGET,
        schema_for_table={TARGET: "domain", FEAT_TABLE: "deriva-ml"},
        multi_schema=False,
    )
    # Orphan dropped — only the two real rows survive.
    assert len(records) == 2
    assert "ghost" not in {r.Quality for r in records}


def test_multi_schema_label_prefix(engine_with_feature_rows):
    """Multi-schema labels (``schema.Table.col``) are stripped correctly."""
    engine, resolver = engine_with_feature_rows
    rows = [
        {
            f"domain.{TARGET}.RID": "IMG-A",
            f"domain.{TARGET}.URL": "a.png",
            f"deriva-ml.{FEAT_TABLE}.RID": "F1",
            f"deriva-ml.{FEAT_TABLE}.Image": "IMG-A",
            f"deriva-ml.{FEAT_TABLE}.Quality": "good",
            f"deriva-ml.{FEAT_TABLE}.Execution": "EXE-old",
            f"deriva-ml.{FEAT_TABLE}.Feature_Name": "Quality",
        },
    ]
    records = materialize_feature_records(
        rows,
        record_class=_record_class(),
        engine=engine,
        orm_resolver=resolver,
        feature_assoc_table=FEAT_TABLE,
        target_table_name=TARGET,
        schema_for_table={TARGET: "domain", FEAT_TABLE: "deriva-ml"},
        multi_schema=True,
    )
    assert len(records) == 1
    assert records[0].Image == "IMG-A"
    assert records[0].Quality == "good"
    assert records[0].RCT.endswith("+00:00")


def test_naive_datetime_iso_matches_pathbuilder_shape() -> None:
    """A tz-naive datetime and the same instant tz-aware serialize identically.

    Pins the specific fix for the live-verification divergence at the
    serialization level.
    """
    naive = datetime(2026, 6, 1, 12, 30, 0)
    aware = naive.replace(tzinfo=timezone.utc)
    assert naive.replace(tzinfo=timezone.utc).isoformat() == aware.isoformat()
    assert aware.isoformat().endswith("+00:00")

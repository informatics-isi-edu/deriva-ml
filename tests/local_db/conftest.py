"""Shared fixtures for local_db tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from deriva.core.ermrest_model import Model
from sqlalchemy.orm import Session

from deriva_ml.local_db.schema import LocalSchema
from deriva_ml.model.catalog import DerivaModel


@pytest.fixture
def canned_bag_schema(tmp_path: Path) -> Path:
    """Write a minimal valid bag-style schema.json into a temp dir.

    Schema has:
      - schema 'isa' with tables 'Subject' and 'Image'
      - schema 'deriva-ml' with table 'Dataset'
      - Image has FK to Subject and FK to Dataset
    """
    schema_doc = _base_schema_doc()
    out = tmp_path / "schema.json"
    out.write_text(json.dumps(schema_doc))
    return out


@pytest.fixture
def canned_bag_model(canned_bag_schema: Path) -> Model:
    """Load the canned bag schema as a deriva Model."""
    return Model.fromfile("file-system", canned_bag_schema)


# ---------------------------------------------------------------------------
# Extended schema with Dataset_Image association table (for denormalize tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def denorm_schema(tmp_path: Path) -> Path:
    """Canned schema with a Dataset_Image association table.

    FK graph: Dataset <-- Dataset_Image --> Image --> Subject
    """
    doc = _base_schema_doc()
    # Add Dataset_Image to the deriva-ml schema
    doc["schemas"]["deriva-ml"]["tables"]["Dataset_Image"] = {
        "table_name": "Dataset_Image",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Dataset", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Dataset_Image",
                        "column_name": "Dataset",
                    }
                ],
                "referenced_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Dataset",
                        "column_name": "RID",
                    }
                ],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Dataset_Image",
                        "column_name": "Image",
                    }
                ],
                "referenced_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Image",
                        "column_name": "RID",
                    }
                ],
            },
        ],
    }
    out = tmp_path / "schema.json"
    out.write_text(json.dumps(doc))
    return out


@pytest.fixture
def denorm_model(denorm_schema: Path) -> Model:
    """ERMrest Model loaded from the extended schema."""
    return Model.fromfile("file-system", denorm_schema)


@pytest.fixture
def denorm_deriva_model(denorm_model: Model) -> DerivaModel:
    """DerivaModel instance for join planning."""
    return DerivaModel(
        model=denorm_model,
        ml_schema="deriva-ml",
        domain_schemas={"isa"},
    )


# ---------------------------------------------------------------------------
# Diamond-schema fixture (for planner rule tests that need multiple FK paths)
# ---------------------------------------------------------------------------


@pytest.fixture
def denorm_schema_diamond(tmp_path: Path) -> Path:
    """Canned schema with a diamond between Image and Subject.

    Extends :func:`denorm_schema` by adding an ``Observation`` table and
    an ``Image.Observation`` FK, creating two FK paths from Image to
    Subject:

    - ``Image → Subject``        (direct FK)
    - ``Image → Observation → Subject``  (indirect via Observation)

    Used for testing Rule 6 (path ambiguity) in the denormalization
    planner. Kept separate from :func:`denorm_schema` so that the
    non-diamond integration tests in ``test_denormalize.py`` continue
    to exercise the simple chain.
    """
    doc = _base_schema_doc()

    # Add Dataset_Image association
    doc["schemas"]["deriva-ml"]["tables"]["Dataset_Image"] = {
        "table_name": "Dataset_Image",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Dataset", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Dataset_Image",
                        "column_name": "Dataset",
                    }
                ],
                "referenced_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Dataset",
                        "column_name": "RID",
                    }
                ],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Dataset_Image",
                        "column_name": "Image",
                    }
                ],
                "referenced_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Image",
                        "column_name": "RID",
                    }
                ],
            },
        ],
    }

    # Add Observation table in isa schema
    doc["schemas"]["isa"]["tables"]["Observation"] = {
        "table_name": "Observation",
        "schema_name": "isa",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "Date", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Observation",
                        "column_name": "Subject",
                    }
                ],
                "referenced_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Subject",
                        "column_name": "RID",
                    }
                ],
            },
        ],
    }

    # Extend Image with Observation FK (creates the diamond)
    image = doc["schemas"]["isa"]["tables"]["Image"]
    image["column_definitions"].append({"name": "Observation", "type": {"typename": "text"}, "nullok": True})
    image["foreign_keys"].append(
        {
            "foreign_key_columns": [
                {
                    "schema_name": "isa",
                    "table_name": "Image",
                    "column_name": "Observation",
                }
            ],
            "referenced_columns": [
                {
                    "schema_name": "isa",
                    "table_name": "Observation",
                    "column_name": "RID",
                }
            ],
        }
    )

    out = tmp_path / "schema.json"
    out.write_text(json.dumps(doc))
    return out


@pytest.fixture
def denorm_diamond_model(denorm_schema_diamond: Path) -> Model:
    """ERMrest Model with diamond FK paths (Image → Subject has two routes)."""
    return Model.fromfile("file-system", denorm_schema_diamond)


@pytest.fixture
def denorm_diamond_deriva_model(denorm_diamond_model: Model) -> DerivaModel:
    """DerivaModel with the diamond fixture (Observation intermediate)."""
    return DerivaModel(
        model=denorm_diamond_model,
        ml_schema="deriva-ml",
        domain_schemas={"isa"},
    )


@pytest.fixture
def denorm_local_schema(denorm_model: Model, tmp_path: Path) -> LocalSchema:
    """LocalSchema with tables from the extended schema."""
    db_path = tmp_path / "denorm_db"
    db_path.mkdir()
    return LocalSchema.build(
        model=denorm_model,
        schemas=["isa", "deriva-ml"],
        database_path=db_path,
    )


@pytest.fixture
def denorm_diamond_local_schema(denorm_diamond_model: Model, tmp_path: Path) -> LocalSchema:
    """LocalSchema with tables from the diamond-schema fixture.

    Mirrors :func:`denorm_local_schema` but builds against the diamond
    model (includes ``Observation`` + ``Image.Observation`` FK).
    """
    db_path = tmp_path / "denorm_diamond_db"
    db_path.mkdir()
    return LocalSchema.build(
        model=denorm_diamond_model,
        schemas=["isa", "deriva-ml"],
        database_path=db_path,
    )


@pytest.fixture
def populated_denorm(
    denorm_deriva_model: DerivaModel,
    denorm_local_schema: LocalSchema,
) -> dict[str, Any]:
    """LocalSchema pre-populated with test data for denormalization.

    Returns a dict with:
        model: DerivaModel
        local_schema: LocalSchema
        dataset_rid: str
        subject_rids: list[str]
        image_rids: list[str]
    """
    ls = denorm_local_schema
    engine = ls.engine

    ds_rid = "DS-001"
    subj_rids = ["SUBJ-A", "SUBJ-B"]
    img_rids = ["IMG-1", "IMG-2", "IMG-3"]

    with Session(engine) as session:
        # Insert subjects
        subj_cls = ls.get_orm_class("Subject")
        for rid, name in zip(subj_rids, ["Alice", "Bob"]):
            session.add(subj_cls(RID=rid, Name=name))

        # Insert images (IMG-3 has NULL subject for LEFT JOIN testing)
        img_cls = ls.get_orm_class("Image")
        session.add(img_cls(RID="IMG-1", Filename="a.png", Subject="SUBJ-A"))
        session.add(img_cls(RID="IMG-2", Filename="b.png", Subject="SUBJ-B"))
        session.add(img_cls(RID="IMG-3", Filename="c.png", Subject=None))

        # Insert dataset
        ds_cls = ls.get_orm_class("Dataset")
        session.add(ds_cls(RID=ds_rid, Description="test dataset"))

        # Insert associations
        di_cls = ls.get_orm_class("Dataset_Image")
        for img_rid in img_rids:
            session.add(
                di_cls(
                    RID=f"DI-{img_rid}",
                    Dataset=ds_rid,
                    Image=img_rid,
                )
            )

        session.commit()

    return {
        "model": denorm_deriva_model,
        "local_schema": ls,
        "dataset_rid": ds_rid,
        "subject_rids": subj_rids,
        "image_rids": img_rids,
    }


@pytest.fixture
def populated_denorm_diamond(
    denorm_diamond_deriva_model: DerivaModel,
    denorm_diamond_local_schema: LocalSchema,
) -> dict[str, Any]:
    """LocalSchema pre-populated with diamond-schema test data.

    Mirrors :func:`populated_denorm` but uses the diamond fixture, which
    adds an ``Observation`` table and an ``Image.Observation`` FK (creating
    two FK paths from Image to Subject).

    Returns a dict with:
        model: DerivaModel (diamond)
        local_schema: LocalSchema
        dataset_rid: str
        subject_rids: list[str]
        observation_rids: list[str]
        image_rids: list[str]
    """
    ls = denorm_diamond_local_schema
    engine = ls.engine

    ds_rid = "DS-001"
    subj_rids = ["SUBJ-A", "SUBJ-B"]
    obs_rids = ["OBS-1", "OBS-2"]
    img_rids = ["IMG-1", "IMG-2", "IMG-3"]

    with Session(engine) as session:
        # Insert subjects
        subj_cls = ls.get_orm_class("Subject")
        for rid, name in zip(subj_rids, ["Alice", "Bob"]):
            session.add(subj_cls(RID=rid, Name=name))

        # Insert observations (OBS-1 -> SUBJ-A, OBS-2 -> SUBJ-B)
        obs_cls = ls.get_orm_class("Observation")
        session.add(obs_cls(RID="OBS-1", Subject="SUBJ-A", Date="2026-01-01"))
        session.add(obs_cls(RID="OBS-2", Subject="SUBJ-B", Date="2026-01-02"))

        # Insert images
        # IMG-1 -> SUBJ-A + OBS-1, IMG-2 -> SUBJ-B + OBS-2, IMG-3 with NULL FKs
        img_cls = ls.get_orm_class("Image")
        session.add(img_cls(RID="IMG-1", Filename="a.png", Subject="SUBJ-A", Observation="OBS-1"))
        session.add(img_cls(RID="IMG-2", Filename="b.png", Subject="SUBJ-B", Observation="OBS-2"))
        session.add(img_cls(RID="IMG-3", Filename="c.png", Subject=None, Observation=None))

        # Insert dataset
        ds_cls = ls.get_orm_class("Dataset")
        session.add(ds_cls(RID=ds_rid, Description="test diamond dataset"))

        # Insert associations
        di_cls = ls.get_orm_class("Dataset_Image")
        for img_rid in img_rids:
            session.add(
                di_cls(
                    RID=f"DI-{img_rid}",
                    Dataset=ds_rid,
                    Image=img_rid,
                )
            )

        session.commit()

    return {
        "model": denorm_diamond_deriva_model,
        "local_schema": ls,
        "dataset_rid": ds_rid,
        "subject_rids": subj_rids,
        "observation_rids": obs_rids,
        "image_rids": img_rids,
    }


# ---------------------------------------------------------------------------
# Feature-association fixture (issue #174)
# ---------------------------------------------------------------------------


@pytest.fixture
def denorm_schema_feature(tmp_path: Path) -> Path:
    """Canned schema with a DerivaML feature-association table.

    Extends :func:`denorm_schema` by adding:

    - ``Execution`` table in the ``deriva-ml`` schema (provenance).
    - ``Image_Classification`` vocabulary table in the ``isa`` schema
      (the value table — i.e. the term).
    - ``Execution_Image_Image_Classification`` association table with
      three FKs (Image, Execution, Image_Classification) and a
      ``Feature_Name`` column. This is the canonical shape of a
      DerivaML feature value table — see ``DerivaModel.find_features``
      for the runtime predicate.

    Plus a *non-feature* 3-FK association,
    ``Image_Subject_UnrelatedThing``, with three domain FKs but no
    FK to ``Execution`` — used to verify
    :meth:`_is_feature_association` correctly rejects 3-FK associations
    that aren't features.

    Used by the planner-rules tests in
    :mod:`tests.local_db.test_planner_rules` to exercise the
    transparency widening introduced for issue #174.
    """
    doc = _base_schema_doc()

    # Add Dataset_Image association (same as denorm_schema)
    doc["schemas"]["deriva-ml"]["tables"]["Dataset_Image"] = {
        "table_name": "Dataset_Image",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Dataset", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Dataset_Image", "column_name": "Dataset"}
                ],
                "referenced_columns": [{"schema_name": "deriva-ml", "table_name": "Dataset", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Dataset_Image", "column_name": "Image"}
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Image", "column_name": "RID"}],
            },
        ],
    }

    # Add Execution table (deriva-ml schema)
    doc["schemas"]["deriva-ml"]["tables"]["Execution"] = {
        "table_name": "Execution",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Description", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [],
    }

    # Add Image_Classification vocabulary term table (isa schema)
    doc["schemas"]["isa"]["tables"]["Image_Classification"] = {
        "table_name": "Image_Classification",
        "schema_name": "isa",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Name", "type": {"typename": "text"}, "nullok": False},
            {"name": "Description", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [{"unique_columns": ["RID"]}, {"unique_columns": ["Name"]}],
        "foreign_keys": [],
    }

    # Add Execution_Image_Image_Classification feature-assoc table.
    # Three domain FKs: Image, Execution, Image_Classification.
    # Includes a Feature_Name column to match the DerivaML runtime
    # predicate (DerivaModel.find_features's is_feature check).
    doc["schemas"]["deriva-ml"]["tables"]["Execution_Image_Image_Classification"] = {
        "table_name": "Execution_Image_Image_Classification",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Feature_Name", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image_Classification", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Image_Image_Classification",
                        "column_name": "Image",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Image", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Image_Image_Classification",
                        "column_name": "Execution",
                    }
                ],
                "referenced_columns": [{"schema_name": "deriva-ml", "table_name": "Execution", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Image_Image_Classification",
                        "column_name": "Image_Classification",
                    }
                ],
                "referenced_columns": [
                    {"schema_name": "isa", "table_name": "Image_Classification", "column_name": "RID"}
                ],
            },
        ],
    }

    # Add a non-feature 3-FK association (no FK to Execution) to verify
    # the predicate rejects 3-FK shapes that aren't features.
    doc["schemas"]["isa"]["tables"]["Image_Subject_UnrelatedThing"] = {
        "table_name": "Image_Subject_UnrelatedThing",
        "schema_name": "isa",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "UnrelatedThing", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Image_Subject_UnrelatedThing",
                        "column_name": "Image",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Image", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Image_Subject_UnrelatedThing",
                        "column_name": "Subject",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Subject", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "isa",
                        "table_name": "Image_Subject_UnrelatedThing",
                        "column_name": "UnrelatedThing",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "UnrelatedThing", "column_name": "RID"}],
            },
        ],
    }

    # Add a 4-FK association to verify the predicate rejects multi-way
    # associations (should NOT be transparent).
    doc["schemas"]["isa"]["tables"]["FourWayAssoc"] = {
        "table_name": "FourWayAssoc",
        "schema_name": "isa",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "UnrelatedThing", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [
            {
                "foreign_key_columns": [{"schema_name": "isa", "table_name": "FourWayAssoc", "column_name": "Image"}],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Image", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [{"schema_name": "isa", "table_name": "FourWayAssoc", "column_name": "Subject"}],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Subject", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {"schema_name": "isa", "table_name": "FourWayAssoc", "column_name": "UnrelatedThing"}
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "UnrelatedThing", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {"schema_name": "isa", "table_name": "FourWayAssoc", "column_name": "Execution"}
                ],
                "referenced_columns": [{"schema_name": "deriva-ml", "table_name": "Execution", "column_name": "RID"}],
            },
        ],
    }

    out = tmp_path / "schema.json"
    out.write_text(json.dumps(doc))
    return out


@pytest.fixture
def denorm_feature_model(denorm_schema_feature: Path) -> Model:
    """ERMrest Model with a DerivaML feature-association table."""
    return Model.fromfile("file-system", denorm_schema_feature)


@pytest.fixture
def denorm_feature_deriva_model(denorm_feature_model: Model) -> DerivaModel:
    """DerivaModel for the feature-association fixture."""
    return DerivaModel(
        model=denorm_feature_model,
        ml_schema="deriva-ml",
        domain_schemas={"isa"},
    )


@pytest.fixture
def denorm_local_schema_feature(denorm_feature_model: Model, tmp_path: Path) -> LocalSchema:
    """LocalSchema built from the feature-association fixture.

    Same shape as :func:`denorm_local_schema`, but against the schema
    that includes ``Execution``, ``Image_Classification``, and the
    ``Execution_Image_Image_Classification`` feature-association table.
    Used by the row-completeness invariant tests (SC-06 / RB-02) that
    need two element paths converging on ``Image``.
    """
    db_path = tmp_path / "denorm_feature_db"
    db_path.mkdir()
    return LocalSchema.build(
        model=denorm_feature_model,
        schemas=["isa", "deriva-ml"],
        database_path=db_path,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _base_schema_doc() -> dict:
    """Return the base schema document (Subject, Image, Dataset)."""
    return {
        "schemas": {
            "isa": {
                "tables": {
                    "Subject": {
                        "table_name": "Subject",
                        "schema_name": "isa",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Name", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [
                            {"unique_columns": ["RID"]},
                        ],
                        "foreign_keys": [],
                    },
                    "Image": {
                        "table_name": "Image",
                        "schema_name": "isa",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Filename", "type": {"typename": "text"}, "nullok": True},
                            {"name": "Subject", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [
                            {
                                "foreign_key_columns": [
                                    {"schema_name": "isa", "table_name": "Image", "column_name": "Subject"}
                                ],
                                "referenced_columns": [
                                    {"schema_name": "isa", "table_name": "Subject", "column_name": "RID"}
                                ],
                            }
                        ],
                    },
                    "UnrelatedThing": {
                        "table_name": "UnrelatedThing",
                        "schema_name": "isa",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Label", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [],
                    },
                },
            },
            "deriva-ml": {
                "tables": {
                    "Dataset": {
                        "table_name": "Dataset",
                        "schema_name": "deriva-ml",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Description", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [],
                    },
                },
            },
        },
    }

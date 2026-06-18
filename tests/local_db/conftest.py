"""Shared fixtures for local_db tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from deriva.core.ermrest_model import Model, tag
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

    Used for testing Path ambiguity in the denormalization
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
    - ``Feature_Name`` vocabulary table in the ``deriva-ml`` schema
      (one of the two feature marker FK targets).
    - ``Image_Classification`` vocabulary table in the ``isa`` schema
      (the value table — i.e. the term).
    - ``Execution_Image_Image_Classification`` association table with
      **four** domain FKs (Image, Image_Classification, Feature_Name,
      Execution). This is the canonical real shape of a DerivaML
      feature value table — exactly what ``create_feature`` produces.
      The Feature_Name FK + Execution FK are the structural markers
      ``_is_feature_association`` keys off; see
      ``DerivaModel.find_features`` for the equivalent runtime
      predicate.

    Plus a *non-feature* 3-FK association,
    ``Image_Subject_UnrelatedThing``, with three domain FKs but no
    FK to ``Execution`` or ``Feature_Name`` — used to verify
    :meth:`_is_feature_association` correctly rejects associations
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

    # Add Feature_Name vocabulary table (deriva-ml schema). Every
    # DerivaML feature-association table carries an FK to this vocab —
    # it's one of the two structural markers
    # ``_is_feature_association`` keys off (the other is the Execution
    # FK). See ``DerivaModel.find_features``'s is_feature predicate and
    # ``FeatureMixin.create_feature`` (which always adds this FK).
    doc["schemas"]["deriva-ml"]["tables"]["Feature_Name"] = {
        "table_name": "Feature_Name",
        "schema_name": "deriva-ml",
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
    # Four domain FKs — the canonical real DerivaML feature shape:
    #   Image                (feature target)
    #   Image_Classification (value: vocab term)
    #   Feature_Name         (-> deriva-ml.Feature_Name vocab)
    #   Execution            (-> deriva-ml.Execution, provenance)
    # The Feature_Name FK + Execution FK are the two structural markers
    # ``_is_feature_association`` keys off (mirroring
    # ``DerivaModel.find_features``'s is_feature check). This matches
    # what ``FeatureMixin.create_feature`` actually produces; an earlier
    # version of this fixture modeled Feature_Name as a plain text
    # column and gave the table only 3 FKs, which is why the 4-FK
    # predicate bug shipped green (see finding 09).
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
                        "column_name": "Feature_Name",
                    }
                ],
                "referenced_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Feature_Name", "column_name": "Name"}
                ],
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
def denorm_schema_feature_diamond(denorm_schema_feature: Path, tmp_path: Path) -> Path:
    """Feature fixture + a *second* independent FK path target↔value.

    Builds the "diamond-with-feature-bridge" shape (finding 09 §7.1,
    §10 limitation #2): ``Image`` reaches ``Image_Classification`` via
    **two** routes —

    1. the transparent feature bridge
       ``Execution_Image_Image_Classification`` (4-FK feature), and
    2. a direct ``Image.Image_Classification`` FK.

    Under the old (broken) predicate the feature bridge was opaque, so
    Path ambiguity saw only the direct path and planned silently. Under the
    Option E2 predicate the bridge is transparent and hops in
    ``_is_downstream_chain``, so Path ambiguity now sees two competing
    downstream chains from ``Image`` to ``Image_Classification`` and
    must raise ``DerivaMLDenormalizeAmbiguousPath``. This fixture pins
    that intentional new behavior.
    """
    # Start from the JSON the feature fixture already wrote, then add a
    # direct Image → Image_Classification FK to create the second path.
    doc = json.loads(denorm_schema_feature.read_text())
    image = doc["schemas"]["isa"]["tables"]["Image"]
    image["column_definitions"].append({"name": "Image_Classification", "type": {"typename": "text"}, "nullok": True})
    image["foreign_keys"].append(
        {
            "foreign_key_columns": [
                {"schema_name": "isa", "table_name": "Image", "column_name": "Image_Classification"}
            ],
            "referenced_columns": [{"schema_name": "isa", "table_name": "Image_Classification", "column_name": "RID"}],
        }
    )

    out = tmp_path / "schema_feature_diamond.json"
    out.write_text(json.dumps(doc))
    return out


@pytest.fixture
def denorm_feature_diamond_deriva_model(denorm_schema_feature_diamond: Path) -> DerivaModel:
    """DerivaModel for the diamond-with-feature-bridge fixture."""
    return DerivaModel(
        model=Model.fromfile("file-system", denorm_schema_feature_diamond),
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
# Key-qualified feature fixture (findings 10 + 12)
# ---------------------------------------------------------------------------


@pytest.fixture
def denorm_schema_qualified_feature(tmp_path: Path) -> Path:
    """Canned schema with a *key-qualified* multi-value feature.

    This is the structural shape that no prior fixture modeled — and the
    gap that hid two coupled bugs (findings 10 + 12). It mirrors eye-ai's
    real ``Execution_Subject_Chart_Label`` feature: a feature-association
    table whose **compound uniqueness key includes a qualifier FK** beyond
    ``{target, Feature_Name, Execution}``. The same target (Subject) can
    therefore have two rows distinguished only by the qualifier.

    Tables added on top of :func:`_base_schema_doc`:

    - ``Execution`` and ``Feature_Name`` (``deriva-ml`` schema) — the two
      structural feature markers.
    - ``Image_Side`` vocabulary (``isa`` schema) — the qualifier's target
      (terms: ``Left`` / ``Right``).
    - ``Condition_Label`` vocabulary (``isa`` schema) — an ordinary value
      term (NOT in the key).
    - ``Execution_Subject_Chart_Label`` feature-association table with FKs
      to ``Subject`` (target), ``Execution``, ``Feature_Name``,
      ``Image_Side`` (qualifier), and ``Condition_Label`` (value). Its
      compound key is
      ``[Execution, Subject, Feature_Name, Image_Side]`` — exactly the
      eye-ai shape. The ``Image_Side`` FK is *in the key* (a qualifier);
      ``Condition_Label`` is *not* (a value).

    Because the key-FK arity is 4, ``find_associations(max_arity=3)`` —
    the former ``find_features`` cap — silently excludes this table.
    ``find_associations(max_arity=None)`` discovers it. The fixture thus
    fails the old code and passes the new code, which is the whole point.
    """
    doc = _base_schema_doc()

    # Execution (deriva-ml).
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

    # Feature_Name vocabulary (deriva-ml).
    doc["schemas"]["deriva-ml"]["tables"]["Feature_Name"] = {
        "table_name": "Feature_Name",
        "schema_name": "deriva-ml",
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

    # Image_Side vocabulary (isa) — the QUALIFIER's target (Left/Right).
    doc["schemas"]["isa"]["tables"]["Image_Side"] = {
        "table_name": "Image_Side",
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

    # Condition_Label vocabulary (isa) — an ordinary VALUE term (NOT in key).
    doc["schemas"]["isa"]["tables"]["Condition_Label"] = {
        "table_name": "Condition_Label",
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

    # The key-qualified feature-association table. Its compound key
    # includes Image_Side (the qualifier), giving key-FK arity 4.
    doc["schemas"]["deriva-ml"]["tables"]["Execution_Subject_Chart_Label"] = {
        "table_name": "Execution_Subject_Chart_Label",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            # Default on Feature_Name is what Feature.feature_name reads.
            {
                "name": "Feature_Name",
                "type": {"typename": "text"},
                "nullok": False,
                "default": "Chart_Label",
            },
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image_Side", "type": {"typename": "text"}, "nullok": False},
            {"name": "Condition_Label", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            # The compound IDENTITY key — Image_Side IN the key makes it a
            # qualifier (key-FK arity 4, the eye-ai Chart_Label shape).
            {"unique_columns": ["Execution", "Subject", "Feature_Name", "Image_Side"]},
        ],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Subject_Chart_Label",
                        "column_name": "Subject",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Subject", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Subject_Chart_Label",
                        "column_name": "Execution",
                    }
                ],
                "referenced_columns": [{"schema_name": "deriva-ml", "table_name": "Execution", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Subject_Chart_Label",
                        "column_name": "Feature_Name",
                    }
                ],
                "referenced_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Feature_Name", "column_name": "Name"}
                ],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Subject_Chart_Label",
                        "column_name": "Image_Side",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Image_Side", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Subject_Chart_Label",
                        "column_name": "Condition_Label",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Condition_Label", "column_name": "RID"}],
            },
        ],
    }

    # An ORDINARY (unqualified) feature alongside the qualified one, so a
    # single fixture proves both the fix AND the backward-compat invariant
    # via the same ``find_features`` path. Compound key is
    # ``[Execution, Image, Feature_Name]`` (key-FK arity 3, no qualifier),
    # so ``Feature.qualifier_columns`` must be empty and a selector must
    # reduce to one-per-Image — unchanged from before the fix.
    doc["schemas"]["deriva-ml"]["tables"]["Execution_Image_Quality"] = {
        "table_name": "Execution_Image_Quality",
        "schema_name": "deriva-ml",
        "column_definitions": [
            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
            {"name": "RCT", "type": {"typename": "timestamptz"}},
            {"name": "RMT", "type": {"typename": "timestamptz"}},
            {"name": "RCB", "type": {"typename": "text"}},
            {"name": "RMB", "type": {"typename": "text"}},
            {"name": "Feature_Name", "type": {"typename": "text"}, "nullok": False, "default": "Quality"},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            {"name": "Condition_Label", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            # No qualifier in the key — the target (Image) alone is identity.
            {"unique_columns": ["Execution", "Image", "Feature_Name"]},
        ],
        "foreign_keys": [
            {
                "foreign_key_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Execution_Image_Quality", "column_name": "Image"}
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Image", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Execution_Image_Quality", "column_name": "Execution"}
                ],
                "referenced_columns": [{"schema_name": "deriva-ml", "table_name": "Execution", "column_name": "RID"}],
            },
            {
                "foreign_key_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Execution_Image_Quality", "column_name": "Feature_Name"}
                ],
                "referenced_columns": [
                    {"schema_name": "deriva-ml", "table_name": "Feature_Name", "column_name": "Name"}
                ],
            },
            {
                "foreign_key_columns": [
                    {
                        "schema_name": "deriva-ml",
                        "table_name": "Execution_Image_Quality",
                        "column_name": "Condition_Label",
                    }
                ],
                "referenced_columns": [{"schema_name": "isa", "table_name": "Condition_Label", "column_name": "RID"}],
            },
        ],
    }

    out = tmp_path / "schema_qualified_feature.json"
    out.write_text(json.dumps(doc))
    return out


@pytest.fixture
def denorm_qualified_feature_model(denorm_schema_qualified_feature: Path) -> Model:
    """ERMrest Model with a key-qualified feature-association table."""
    return Model.fromfile("file-system", denorm_schema_qualified_feature)


@pytest.fixture
def denorm_qualified_feature_deriva_model(denorm_qualified_feature_model: Model) -> DerivaModel:
    """DerivaModel for the key-qualified feature fixture."""
    return DerivaModel(
        model=denorm_qualified_feature_model,
        ml_schema="deriva-ml",
        domain_schemas={"isa"},
    )


# ---------------------------------------------------------------------------
# Arbitrary-shape qualified-feature fixtures (PR #261 generalization proof)
# ---------------------------------------------------------------------------
#
# The single-qualifier ``denorm_schema_qualified_feature`` above proved the
# #261 fix on ONE vocab-FK qualifier (the eye-ai Chart_Label / Image_Side
# shape). The fix generalizes *by construction* — ``Feature.qualifier_columns``
# is a comprehension over ``atable.other_fkeys`` (the compound-key-covered
# FKs), so it is agnostic to the number of qualifiers and to the referent
# type of each qualifier FK. These fixtures make that generalization
# *empirical*: arbitrary qualifier counts, an asset-table qualifier referent,
# and the inverse — a many-column unqualified feature that must NOT be
# mistaken for a qualified one (the over-split guard).
#
# The structural distinction the whole fix hinges on: a qualifier FK's column
# is part of a multi-column uniqueness KEY (→ it appears in deriva-py's
# ``covered_fkeys`` / ``other_fkeys``); a decoration FK is NOT in any compound
# key (→ it never appears there). Each fixture below is verified in its test
# to actually produce the intended ``qualifier_columns`` via the real
# ``find_features`` / ``Feature`` path — never asserted by construction alone.


def _qf_sys_cols() -> list[dict[str, Any]]:
    """The five ERMrest system columns every table carries."""
    return [
        {"name": "RID", "type": {"typename": "text"}, "nullok": False},
        {"name": "RCT", "type": {"typename": "timestamptz"}},
        {"name": "RMT", "type": {"typename": "timestamptz"}},
        {"name": "RCB", "type": {"typename": "text"}},
        {"name": "RMB", "type": {"typename": "text"}},
    ]


def _qf_simple_vocab(name: str, schema: str = "isa") -> dict[str, Any]:
    """A name-keyed reference table (Name/Description only).

    NOT a deriva-py *vocabulary* (which requires ID/URI/Synonyms of specific
    types) — so an FK to it that is NOT key-covered classifies as a
    ``value_column``, mirroring the existing fixtures' Condition_Label shape.
    Used where the referent's vocab-vs-value classification is irrelevant and
    only the FK's key-coverage matters.
    """
    return {
        "table_name": name,
        "schema_name": schema,
        "column_definitions": _qf_sys_cols()
        + [
            {"name": "Name", "type": {"typename": "text"}, "nullok": False},
            {"name": "Description", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [{"unique_columns": ["RID"]}, {"unique_columns": ["Name"]}],
        "foreign_keys": [],
    }


def _qf_real_vocab(name: str, schema: str = "isa") -> dict[str, Any]:
    """A proper deriva-py vocabulary table (ID/URI/Name/Description/Synonyms).

    Recognized by ``DerivaModel.is_vocabulary``, so a non-key-covered FK to it
    classifies as a ``term_column`` on the feature. Used for the decoration
    vocab FK so the over-split guard can assert the FeatureRecord carries it
    as a real term field (feature data, just not identity).
    """
    return {
        "table_name": name,
        "schema_name": schema,
        "column_definitions": _qf_sys_cols()
        + [
            {"name": "ID", "type": {"typename": "ermrest_curie"}, "nullok": False},
            {"name": "URI", "type": {"typename": "ermrest_uri"}, "nullok": False},
            {"name": "Name", "type": {"typename": "text"}, "nullok": False},
            {"name": "Description", "type": {"typename": "markdown"}, "nullok": False},
            {"name": "Synonyms", "type": {"typename": "text[]"}, "nullok": True},
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            {"unique_columns": ["Name"]},
            {"unique_columns": ["ID"]},
        ],
        "foreign_keys": [],
    }


def _qf_asset(name: str, schema: str = "isa") -> dict[str, Any]:
    """A proper deriva-py asset table (URL/Filename/Length/MD5 + asset tag).

    Recognized by ``DerivaModel.is_asset``. Used both as a *qualifier* referent
    (fixture 2 — proving the qualifier derivation is referent-type-agnostic)
    and as a *decoration* referent (fixtures 3 + 4 — a non-key asset FK that
    becomes a record asset field but not part of identity).
    """
    return {
        "table_name": name,
        "schema_name": schema,
        "column_definitions": _qf_sys_cols()
        + [
            {
                "name": "URL",
                "type": {"typename": "text"},
                "nullok": False,
                "annotations": {tag.asset: {}},
            },
            {"name": "Filename", "type": {"typename": "text"}, "nullok": True},
            {"name": "Length", "type": {"typename": "int8"}, "nullok": False},
            {"name": "MD5", "type": {"typename": "text"}, "nullok": False},
        ],
        "keys": [{"unique_columns": ["RID"]}],
        "foreign_keys": [],
    }


def _qf_fk(table: str, col: str, ref_schema: str, ref_table: str, ref_col: str = "RID") -> dict[str, Any]:
    """An outbound single-column FK from ``deriva-ml.<table>.<col>`` to a referent."""
    return {
        "foreign_key_columns": [{"schema_name": "deriva-ml", "table_name": table, "column_name": col}],
        "referenced_columns": [{"schema_name": ref_schema, "table_name": ref_table, "column_name": ref_col}],
    }


def _qf_base_doc() -> dict[str, Any]:
    """Base schema for the arbitrary-shape fixtures: Subject, Image, Execution, Feature_Name.

    ``_base_schema_doc`` carries Dataset + UnrelatedThing that these fixtures
    don't need; this trimmed base keeps each fixture's intent legible.
    """
    return {
        "schemas": {
            "isa": {
                "tables": {
                    "Subject": {
                        "table_name": "Subject",
                        "schema_name": "isa",
                        "column_definitions": _qf_sys_cols()
                        + [{"name": "Name", "type": {"typename": "text"}, "nullok": True}],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [],
                    },
                    "Image": {
                        "table_name": "Image",
                        "schema_name": "isa",
                        "column_definitions": _qf_sys_cols()
                        + [{"name": "Filename", "type": {"typename": "text"}, "nullok": True}],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [],
                    },
                }
            },
            "deriva-ml": {
                "tables": {
                    "Execution": {
                        "table_name": "Execution",
                        "schema_name": "deriva-ml",
                        "column_definitions": _qf_sys_cols()
                        + [{"name": "Description", "type": {"typename": "text"}, "nullok": True}],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [],
                    },
                    "Feature_Name": _qf_simple_vocab("Feature_Name", "deriva-ml"),
                }
            },
        }
    }


def _qf_model(doc: dict[str, Any], tmp_path: Path, name: str) -> DerivaModel:
    """Write ``doc`` to ``tmp_path/<name>.json`` and build a DerivaModel."""
    out = tmp_path / f"{name}.json"
    out.write_text(json.dumps(doc))
    model = Model.fromfile("file-system", out)
    return DerivaModel(model=model, ml_schema="deriva-ml", domain_schemas={"isa"})


@pytest.fixture
def denorm_multi_qualifier_deriva_model(tmp_path: Path) -> DerivaModel:
    """A feature with TWO key-covered qualifier FKs (``Image_Side`` + ``Visit_Number``).

    The compound uniqueness key is
    ``[Execution, Subject, Feature_Name, Image_Side, Visit_Number]`` (key-FK
    arity 5). The same Subject therefore has rows distinguished by a
    *combination* of two qualifiers — (Left, V1), (Left, V2), (Right, V1), …
    ``Condition_Label`` and ``Score`` are non-key value columns. Proves the
    qualifier derivation handles an arbitrary qualifier count, not just one.
    """
    doc = _qf_base_doc()
    doc["schemas"]["isa"]["tables"]["Image_Side"] = _qf_simple_vocab("Image_Side")
    doc["schemas"]["isa"]["tables"]["Visit_Number"] = _qf_simple_vocab("Visit_Number")
    doc["schemas"]["isa"]["tables"]["Condition_Label"] = _qf_simple_vocab("Condition_Label")
    t = "Execution_Subject_Visit_Chart"
    doc["schemas"]["deriva-ml"]["tables"][t] = {
        "table_name": t,
        "schema_name": "deriva-ml",
        "column_definitions": _qf_sys_cols()
        + [
            {"name": "Feature_Name", "type": {"typename": "text"}, "nullok": False, "default": "Visit_Chart"},
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image_Side", "type": {"typename": "text"}, "nullok": False},
            {"name": "Visit_Number", "type": {"typename": "text"}, "nullok": False},
            {"name": "Condition_Label", "type": {"typename": "text"}, "nullok": True},
            {"name": "Score", "type": {"typename": "float8"}, "nullok": True},
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            # BOTH qualifiers are in the key — the composite identity is
            # (Subject, Image_Side, Visit_Number) per execution.
            {"unique_columns": ["Execution", "Subject", "Feature_Name", "Image_Side", "Visit_Number"]},
        ],
        "foreign_keys": [
            _qf_fk(t, "Subject", "isa", "Subject"),
            _qf_fk(t, "Execution", "deriva-ml", "Execution"),
            _qf_fk(t, "Feature_Name", "deriva-ml", "Feature_Name", "Name"),
            _qf_fk(t, "Image_Side", "isa", "Image_Side"),
            _qf_fk(t, "Visit_Number", "isa", "Visit_Number"),
            _qf_fk(t, "Condition_Label", "isa", "Condition_Label"),
        ],
    }
    return _qf_model(doc, tmp_path, "multi_qualifier")


@pytest.fixture
def denorm_asset_qualifier_deriva_model(tmp_path: Path) -> DerivaModel:
    """A feature whose key-covered qualifier FK points to an ASSET table.

    ``Image_Region`` is a proper asset table and is *in* the compound key
    ``[Execution, Subject, Feature_Name, Image_Region]``. Proves
    ``qualifier_columns`` keys off FK-in-``other_fkeys`` membership, NOT off
    the referent being a vocabulary — the derivation is referent-type-agnostic.
    ``Condition_Label`` is a non-key value column.
    """
    doc = _qf_base_doc()
    doc["schemas"]["isa"]["tables"]["Image_Region"] = _qf_asset("Image_Region")
    doc["schemas"]["isa"]["tables"]["Condition_Label"] = _qf_simple_vocab("Condition_Label")
    t = "Execution_Subject_Region_Label"
    doc["schemas"]["deriva-ml"]["tables"][t] = {
        "table_name": t,
        "schema_name": "deriva-ml",
        "column_definitions": _qf_sys_cols()
        + [
            {"name": "Feature_Name", "type": {"typename": "text"}, "nullok": False, "default": "Region_Label"},
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image_Region", "type": {"typename": "text"}, "nullok": False},
            {"name": "Condition_Label", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            # The qualifier FK referent is an ASSET table, not a vocabulary.
            {"unique_columns": ["Execution", "Subject", "Feature_Name", "Image_Region"]},
        ],
        "foreign_keys": [
            _qf_fk(t, "Subject", "isa", "Subject"),
            _qf_fk(t, "Execution", "deriva-ml", "Execution"),
            _qf_fk(t, "Feature_Name", "deriva-ml", "Feature_Name", "Name"),
            _qf_fk(t, "Image_Region", "isa", "Image_Region"),
            _qf_fk(t, "Condition_Label", "isa", "Condition_Label"),
        ],
    }
    return _qf_model(doc, tmp_path, "asset_qualifier")


@pytest.fixture
def denorm_decoration_feature_deriva_model(tmp_path: Path) -> DerivaModel:
    """A decoration-heavy UNQUALIFIED feature — the over-split guard.

    ``Execution_Image_RichQuality`` has many non-key columns: three scalar
    metadata columns (``Confidence`` float, ``Vote_Count`` int, ``Notes``
    text), an asset FK (``Thumbnail``), and a vocab FK (``Quality_Grade``) —
    NONE in the compound key ``[Execution, Image, Feature_Name]`` (impure
    decoration; ``pure=False`` keeps it discoverable). Its identity is the
    target ``Image`` alone, so ``qualifier_columns`` must be EMPTY and a
    selector must reduce to one-row-per-Image. Proves a many-column feature is
    NOT mistaken for a qualified one (the inverse of the qualifier bug).
    """
    doc = _qf_base_doc()
    doc["schemas"]["isa"]["tables"]["Thumbnail"] = _qf_asset("Thumbnail")
    doc["schemas"]["isa"]["tables"]["Quality_Grade"] = _qf_real_vocab("Quality_Grade")
    t = "Execution_Image_RichQuality"
    doc["schemas"]["deriva-ml"]["tables"][t] = {
        "table_name": t,
        "schema_name": "deriva-ml",
        "column_definitions": _qf_sys_cols()
        + [
            {"name": "Feature_Name", "type": {"typename": "text"}, "nullok": False, "default": "RichQuality"},
            {"name": "Image", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            # Scalar metadata — real feature data, NOT identity, NOT in key.
            {"name": "Confidence", "type": {"typename": "float8"}, "nullok": True},
            {"name": "Vote_Count", "type": {"typename": "int4"}, "nullok": True},
            {"name": "Notes", "type": {"typename": "text"}, "nullok": True},
            # Asset FK + vocab FK — decoration, NOT in key.
            {"name": "Thumbnail", "type": {"typename": "text"}, "nullok": True},
            {"name": "Quality_Grade", "type": {"typename": "text"}, "nullok": True},
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            # No decoration FK in the key — the target (Image) alone is identity.
            {"unique_columns": ["Execution", "Image", "Feature_Name"]},
        ],
        "foreign_keys": [
            _qf_fk(t, "Image", "isa", "Image"),
            _qf_fk(t, "Execution", "deriva-ml", "Execution"),
            _qf_fk(t, "Feature_Name", "deriva-ml", "Feature_Name", "Name"),
            _qf_fk(t, "Thumbnail", "isa", "Thumbnail"),
            _qf_fk(t, "Quality_Grade", "isa", "Quality_Grade"),
        ],
    }
    return _qf_model(doc, tmp_path, "decoration_feature")


@pytest.fixture
def denorm_mixed_feature_deriva_model(tmp_path: Path) -> DerivaModel:
    """A feature with a key qualifier AND non-key decoration columns together.

    ``Execution_Subject_MixedLabel`` puts ``Image_Side`` *in* the compound key
    ``[Execution, Subject, Feature_Name, Image_Side]`` (a qualifier) while
    carrying ``Severity_Label`` (vocab), ``Heatmap`` (asset), and
    ``Confidence`` (scalar) as non-key decoration. Proves the two are
    correctly separated: the qualifier joins the group key; the decoration
    becomes record fields but does NOT split the group.
    """
    doc = _qf_base_doc()
    doc["schemas"]["isa"]["tables"]["Image_Side"] = _qf_simple_vocab("Image_Side")
    doc["schemas"]["isa"]["tables"]["Severity_Label"] = _qf_real_vocab("Severity_Label")
    doc["schemas"]["isa"]["tables"]["Heatmap"] = _qf_asset("Heatmap")
    t = "Execution_Subject_MixedLabel"
    doc["schemas"]["deriva-ml"]["tables"][t] = {
        "table_name": t,
        "schema_name": "deriva-ml",
        "column_definitions": _qf_sys_cols()
        + [
            {"name": "Feature_Name", "type": {"typename": "text"}, "nullok": False, "default": "MixedLabel"},
            {"name": "Subject", "type": {"typename": "text"}, "nullok": False},
            {"name": "Execution", "type": {"typename": "text"}, "nullok": False},
            {"name": "Image_Side", "type": {"typename": "text"}, "nullok": False},  # qualifier (IN key)
            {"name": "Severity_Label", "type": {"typename": "text"}, "nullok": True},  # decoration vocab
            {"name": "Heatmap", "type": {"typename": "text"}, "nullok": True},  # decoration asset
            {"name": "Confidence", "type": {"typename": "float8"}, "nullok": True},  # decoration scalar
        ],
        "keys": [
            {"unique_columns": ["RID"]},
            # Only Image_Side is in the key; Severity_Label/Heatmap/Confidence are not.
            {"unique_columns": ["Execution", "Subject", "Feature_Name", "Image_Side"]},
        ],
        "foreign_keys": [
            _qf_fk(t, "Subject", "isa", "Subject"),
            _qf_fk(t, "Execution", "deriva-ml", "Execution"),
            _qf_fk(t, "Feature_Name", "deriva-ml", "Feature_Name", "Name"),
            _qf_fk(t, "Image_Side", "isa", "Image_Side"),
            _qf_fk(t, "Severity_Label", "isa", "Severity_Label"),
            _qf_fk(t, "Heatmap", "isa", "Heatmap"),
        ],
    }
    return _qf_model(doc, tmp_path, "mixed_feature")


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


# ---------------------------------------------------------------------------
# Demo-catalog planner fixture (for FK-reachable-route tests)
# ---------------------------------------------------------------------------
#
# The demo catalog schema reaches ``Image`` two distinct ways:
#   - Dataset -> Dataset_Image -> Image              (direct membership)
#   - Dataset -> Dataset_Subject -> Subject -> Image (FK-reachable, since
#     ``Image.Subject`` references ``Subject``)
# Tests that exercise multi-route emission load the *real* demo schema rather
# than a canned shape so both association tables are present.

# Absolute path to the demo catalog schema shipped with the dataset tests.
_DEMO_CATALOG_SCHEMA = Path(__file__).resolve().parents[1] / "dataset" / "demo-catalog-schema.json"
_EYE_AI_CATALOG_SCHEMA = Path(__file__).resolve().parents[1] / "dataset" / "eye-ai-catalog-schema.json"


class _DenormDatasetStub:
    """Minimal DatasetLike stand-in for catalog-free planner tests.

    ``_prepare_wide_table`` never dereferences the ``dataset`` argument in the
    catalog-free path (it only uses the model and ``dataset_rid``), but later
    consumers expect a ``DatasetLike`` exposing ``dataset_rid`` and a
    children accessor. This stub provides just those two members so the same
    fixture can be reused as the planner pipeline grows.

    Attributes:
        dataset_rid: The dataset RID this stub represents.

    Example:
        >>> stub = _DenormDatasetStub("1-FAKE")  # illustrative placeholder id
        >>> stub.list_dataset_children(recurse=True)
        []
    """

    def __init__(self, dataset_rid: str) -> None:
        self.dataset_rid = dataset_rid

    def list_dataset_children(self, recurse: bool = False) -> list:
        """Return this dataset's child datasets (none, for planning tests)."""
        return []


@pytest.fixture
def demo_catalog_planner() -> tuple[Any, _DenormDatasetStub, str]:
    """Build a DenormalizePlanner over the real demo-catalog schema.

    Mirrors the ``denorm_deriva_model`` construction pattern (load a schema
    JSON into a deriva ``Model``, wrap it in :class:`DerivaModel`) but uses the
    shipped demo schema so both the ``Dataset_Image`` membership association
    and the ``Dataset_Subject`` FK-reachable route to ``Image`` are present.

    Returns:
        A ``(planner, dataset_stub, dataset_rid)`` tuple where *planner* is the
        ``DerivaModel._planner``, *dataset_stub* is a catalog-free
        :class:`_DenormDatasetStub`, and *dataset_rid* is a freshly minted
        opaque RID-shaped string (never a hard-coded literal).

    Example:
        >>> planner, stub, rid = demo_catalog_planner  # doctest: +SKIP
        >>> join_tables, _cols, _multi = planner._prepare_wide_table(  # doctest: +SKIP
        ...     stub, rid, ["Image"], row_per="Image"
        ... )
    """
    model = Model.fromfile("file-system", _DEMO_CATALOG_SCHEMA)
    deriva_model = DerivaModel(
        model=model,
        ml_schema="deriva-ml",
        domain_schemas={"test-schema"},
    )
    # RIDs are opaque: synthesize a unique placeholder rather than embedding a
    # literal. The planner never treats this as a live catalog RID in the
    # catalog-free path; it only flows through the join plan as an identifier.
    dataset_rid = f"1-{uuid.uuid4().hex[:8].upper()}"
    dataset_stub = _DenormDatasetStub(dataset_rid)
    return deriva_model._planner, dataset_stub, dataset_rid


@pytest.fixture
def eye_ai_planner() -> tuple[Any, _DenormDatasetStub, str]:
    """Catalog-free planner over the committed eye-ai schema fixture.

    The eye-ai schema has the multi-route topology that exposed the #320
    JOIN-order regression: ``Subject`` and ``Observation`` are reachable from
    ``Dataset`` via membership associations (e.g. ``Subject_Dataset``), while
    ``Image`` is FK-reachable through ``Observation`` (``Image.Observation``).
    A request over ``[Subject, Observation, Image]`` therefore yields routes
    whose ``Dataset -> ... -> Image`` prefix places ``Observation``/``Subject``
    *before* ``Image`` — the shape that makes a subtree edge's ON clause
    reference a not-yet-joined table.

    Returns ``(planner, dataset_stub, dataset_rid)`` mirroring
    :func:`demo_catalog_planner`.
    """
    model = Model.fromfile("file-system", _EYE_AI_CATALOG_SCHEMA)
    ml_schema = "deriva-ml"
    domain_schemas = {s for s in model.schemas.keys() if s != ml_schema}
    deriva_model = DerivaModel(
        model=model,
        ml_schema=ml_schema,
        domain_schemas=domain_schemas,
    )
    dataset_rid = f"1-{uuid.uuid4().hex[:8].upper()}"
    dataset_stub = _DenormDatasetStub(dataset_rid)
    return deriva_model._planner, dataset_stub, dataset_rid

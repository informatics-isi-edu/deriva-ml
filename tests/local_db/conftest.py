"""Shared fixtures for local_db tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deriva.core.ermrest_model import Model


@pytest.fixture
def canned_bag_schema(tmp_path: Path) -> Path:
    """Write a minimal valid bag-style schema.json into a temp dir.

    Schema has:
      - schema 'isa' with tables 'Subject' and 'Image'
      - schema 'deriva-ml' with table 'Dataset'
      - Image has FK to Subject and FK to Dataset
    """
    schema_doc = {
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
    out = tmp_path / "schema.json"
    out.write_text(json.dumps(schema_doc))
    return out


@pytest.fixture
def canned_bag_model(canned_bag_schema: Path) -> Model:
    """Load the canned bag schema as a deriva Model."""
    return Model.fromfile("file-system", canned_bag_schema)

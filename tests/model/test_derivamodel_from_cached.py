"""Unit test for DerivaModel.from_cached — offline model construction."""
from __future__ import annotations


def test_from_cached_constructs_model_without_network():
    """from_cached(schema_dict) builds a DerivaModel using a
    CatalogStub; no network call is made."""
    from deriva_ml.core.catalog_stub import CatalogStub
    from deriva_ml.model.catalog import DerivaModel

    # Minimal ermrest /schema dict — just enough to instantiate
    # deriva-py's Model without blowing up.
    schema_dict = {
        "schemas": {
            "deriva-ml": {
                "schema_name": "deriva-ml",
                "tables": {},
                "annotations": {},
                "comment": None,
            },
        },
        "acls": {},
        "annotations": {},
    }

    model = DerivaModel.from_cached(
        schema_dict,
        catalog=CatalogStub(),
        ml_schema="deriva-ml",
        domain_schemas=None,
        default_schema=None,
    )
    assert model is not None
    # The ml_schema name propagates.
    assert model.ml_schema == "deriva-ml"

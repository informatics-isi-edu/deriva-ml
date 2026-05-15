"""Integration test for ``create_ml_schema``'s CASCADE behavior.

Closes audit Phase 3 schema/ §3.1 — the destructive
"DROP+CASCADE if schema exists" branch in ``create_ml_schema``
had no regression test. A silent behavior change (e.g., losing
the CASCADE guard, or accidentally CASCADE-ing a different
schema name) could destroy data without anyone noticing.

This test verifies that re-running ``create_ml_schema`` against
a catalog that already has the schema:

1. Drops the existing schema with CASCADE (any pre-existing
   rows in deriva-ml tables disappear).
2. Re-creates the schema fresh (the canonical tables are
   present again, empty).

Pattern follows ``test_vocab_fk_convention.py`` — create a fresh
catalog, introspect, delete on teardown. Requires DERIVA_HOST.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_create_ml_schema_drops_existing_with_cascade() -> None:
    """A second create_ml_schema call drops + recreates the schema.

    Setup: create a fresh catalog with the ML schema initialized
    (which also seeds the standard vocabulary terms).

    Action: call create_ml_schema again on the same catalog.

    Expected:
    - The catalog still has the deriva-ml schema after the call.
    - The Dataset_Type vocabulary table (initialized after the
      first create_ml_schema by initialize_ml_schema) is **empty**
      — the CASCADE dropped its rows along with the schema, and
      the second create_ml_schema did not re-seed (initialize is
      a separate step).

    If the CASCADE guard were silently removed, the second call
    would fail with "schema already exists" or similar.

    If the function silently no-op'd on existing schemas, the
    Dataset_Type rows would still be present and the assert
    would fail.
    """
    from deriva_ml.schema.create_schema import (
        create_ml_catalog,
        create_ml_schema,
        initialize_ml_schema,
    )

    catalog = create_ml_catalog(
        hostname="localhost", project_name="s1b_cascade_test",
    )
    try:
        model = catalog.getCatalogModel()

        # Seed the vocabulary terms so we have rows to be CASCADE-dropped.
        initialize_ml_schema(model, schema_name="deriva-ml")
        catalog = model.catalog  # refresh handle
        seeded_model = catalog.getCatalogModel()
        dataset_type_before = seeded_model.schemas["deriva-ml"].tables["Dataset_Type"]
        pb = catalog.getPathBuilder()
        seeded_rows = list(
            pb.schemas["deriva-ml"].tables["Dataset_Type"].entities().fetch()
        )
        assert seeded_rows, (
            "initialize_ml_schema should have seeded Dataset_Type — "
            "if this fails the test premise is broken, not CASCADE."
        )
        _ = dataset_type_before  # quieten unused-var

        # Action: re-run create_ml_schema. The destructive branch
        # in line 320-322 of create_schema.py should DROP the
        # existing schema with CASCADE before recreating.
        create_ml_schema(catalog, schema_name="deriva-ml")

        # The schema exists again, fresh.
        recreated_model = catalog.getCatalogModel()
        assert "deriva-ml" in recreated_model.schemas, (
            "create_ml_schema must leave the schema present after a "
            "drop+recreate cycle."
        )
        assert "Dataset_Type" in recreated_model.schemas["deriva-ml"].tables, (
            "Dataset_Type table must be re-created after CASCADE drop."
        )

        # The rows are gone — CASCADE took them.
        pb = catalog.getPathBuilder()
        post_rows = list(
            pb.schemas["deriva-ml"].tables["Dataset_Type"].entities().fetch()
        )
        assert post_rows == [], (
            f"Expected CASCADE to drop all Dataset_Type rows; got {len(post_rows)} survivors."
        )

    finally:
        catalog.delete_ermrest_catalog(really=True)

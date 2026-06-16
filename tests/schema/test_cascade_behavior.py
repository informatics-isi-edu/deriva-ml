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
2. Re-creates the schema fresh and re-seeds standard vocabulary
   (since the create path now calls ``initialize_ml_schema``
   internally — see ``create_schema.py`` line 410).

Pattern follows ``test_vocab_fk_convention.py`` — create a fresh
catalog, introspect, delete on teardown. Requires DERIVA_HOST.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_create_ml_schema_drops_existing_with_cascade() -> None:
    """A second create_ml_schema call drops + recreates the schema.

    Setup: create a fresh catalog with the ML schema initialized.
    Seed a **sentinel** vocabulary term in Dataset_Type that is
    *not* in the canonical re-seed list — this is the witness
    for "the existing schema's rows were dropped." If
    ``create_ml_schema`` actually CASCADEs, the sentinel must
    disappear.

    Action: call create_ml_schema again on the same catalog.

    Expected:
    - The catalog still has the deriva-ml schema after the call.
    - Dataset_Type exists and contains the canonical re-seeded
      terms (since ``create_ml_schema`` calls
      ``initialize_ml_schema`` at line 410).
    - The sentinel term is **gone** — it was dropped by CASCADE
      and not in the canonical re-seed set.

    If the CASCADE guard were silently removed, the second call
    would fail with "schema already exists" or similar.

    If the function silently no-op'd on existing schemas, the
    sentinel term would still be present and the assert would
    fail.
    """
    from deriva_ml.schema.create_schema import (
        create_ml_catalog,
        create_ml_schema,
    )

    sentinel_name = "CASCADE_SENTINEL_001"

    catalog = create_ml_catalog(
        hostname="localhost",
        project_name="s1b_cascade_test",
    )
    try:
        # create_ml_catalog runs create_ml_schema + initialize_ml_schema,
        # so Dataset_Type already has the canonical seeded terms.
        pb = catalog.getPathBuilder()
        dataset_type = pb.schemas["deriva-ml"].tables["Dataset_Type"]
        seeded_rows = list(dataset_type.entities().fetch())
        assert seeded_rows, (
            "create_ml_catalog should have seeded Dataset_Type — if this fails the test premise is broken, not CASCADE."
        )

        # Plant a sentinel that is NOT in the canonical re-seed list.
        # Survives only if the second create_ml_schema does *not*
        # actually CASCADE.
        dataset_type.insert(
            [{"Name": sentinel_name, "Description": "CASCADE witness"}],
            defaults={"ID", "URI"},
        )
        pre_rows = list(dataset_type.entities().fetch())
        names_pre = {r["Name"] for r in pre_rows}
        assert sentinel_name in names_pre, "Sentinel insert failed; test premise broken."

        # Action: re-run create_ml_schema. The destructive branch
        # in create_schema.py:335-337 should DROP the existing
        # schema with CASCADE before recreating; line 410 then
        # re-seeds the canonical vocabulary.
        create_ml_schema(catalog, schema_name="deriva-ml")

        # The schema exists again, fresh.
        recreated_model = catalog.getCatalogModel()
        assert "deriva-ml" in recreated_model.schemas, (
            "create_ml_schema must leave the schema present after a drop+recreate cycle."
        )
        assert "Dataset_Type" in recreated_model.schemas["deriva-ml"].tables, (
            "Dataset_Type table must be re-created after CASCADE drop."
        )

        # The canonical terms are re-seeded; the sentinel is gone.
        pb = catalog.getPathBuilder()
        post_rows = list(pb.schemas["deriva-ml"].tables["Dataset_Type"].entities().fetch())
        names_post = {r["Name"] for r in post_rows}
        assert sentinel_name not in names_post, (
            f"CASCADE failed to drop sentinel term {sentinel_name!r}: "
            f"still present after create_ml_schema. The destructive "
            f"drop branch may have silently regressed."
        )
        # Canonical re-seed happened (sanity check on the new behavior).
        assert "Complete" in names_post, (
            "Expected canonical Dataset_Type re-seed (e.g., 'Complete') "
            "after create_ml_schema; got "
            f"{sorted(names_post)}."
        )

    finally:
        catalog.delete_ermrest_catalog(really=True)

"""Planner emits ALL reachable Dataset->element routes (membership + FK-reachable),
not just the preferred membership association. Regression for subject-partitioned
feature_values returning 0 (denormalize-fk-reachable-paths spec)."""

from __future__ import annotations

import uuid

import pytest

from deriva_ml.core.exceptions import DerivaMLDenormalizeAmbiguousPath


def test_prepare_wide_table_emits_membership_and_fk_routes(demo_catalog_planner):
    """The demo schema reaches Image two ways:
      - Dataset -> Dataset_Image -> Image              (membership)
      - Dataset -> Dataset_Subject -> Subject -> Image (FK-reachable)
    Both must appear as separate join_tables entries so the consumer unions them.
    """
    planner, dataset_stub, dataset_rid = demo_catalog_planner
    join_tables, _cols, _multi = planner._prepare_wide_table(dataset_stub, dataset_rid, ["Image"], row_per="Image")
    # join_tables values are (path, join_conditions, join_types); path is a list of
    # table-NAME strings starting with "Dataset". The membership/FK-reachable hop
    # rides on the *association table* (path[1]); collect every table name that
    # appears anywhere in any route so the assertion is robust to the exact index.
    all_path_tables = {name for (path, _jc, _jt) in join_tables.values() for name in path}
    assert "Dataset_Image" in all_path_tables, f"membership route missing: {all_path_tables}"
    assert "Dataset_Subject" in all_path_tables, f"FK-reachable route missing: {all_path_tables}"


def test_only_row_per_element_drives_routes(demo_catalog_planner):
    """Multi-element requests must emit row-driving routes for the ``row_per``
    element ONLY — never for the hoisted upstream elements.

    Regression for the wide-table cartesian bug (#322): after #318 retained
    every ``Dataset -> ... -> endpoint`` route, ``_prepare_wide_table`` keyed
    one ``join_tables`` entry per *endpoint* — including non-``row_per``
    include-tables. The consumer SELECTs the full column set for every route,
    so a ``Subject#*`` route (which never joins ``Image``) cross-joins
    ``Image``'s columns and produces a cartesian product.

    For ``[Image, Subject]`` the sink auto-infers to ``Image`` (Image is
    downstream of Subject via ``Image.Subject``). ``Subject``'s columns must be
    HOISTED into the ``Image#*`` routes' shared subtree, not emitted as their
    own row-driving routes. So every ``join_tables`` key must name the
    ``row_per`` element ``Image`` — zero ``Subject#`` routes.

    (The live-catalog analog is ``[Image, Observation]`` on the demo catalog,
    where ``Image.Observation`` plays the role ``Image.Subject`` plays here;
    pinned end-to-end in ``tests/dataset/test_denormalize.py``.)
    """
    planner, dataset_stub, dataset_rid = demo_catalog_planner
    # row_per auto-infers to Image (furthest-downstream of the pair, via Image.Subject).
    join_tables, _cols, _multi = planner._prepare_wide_table(
        dataset_stub, dataset_rid, ["Image", "Subject"]
    )
    # Keys are "{element}#{route_index}"; the element prefix is the part before "#".
    elements = {key.split("#", 1)[0] for key in join_tables}
    assert elements == {"Image"}, (
        "only the row_per element (Image) may drive rows; a non-row_per route "
        f"would cross-join the full column set. Got route elements: {sorted(elements)}"
    )
    # Every retained Image route must still cover Subject (so its columns are
    # hoisted, not dropped) — the FK-reachable union must not lose data.
    for key, (path, _jc, _jt) in join_tables.items():
        assert "Subject" in path, (
            f"route {key} dropped Subject from its join path {path}; "
            "Subject columns would be NULL instead of hoisted"
        )


def test_genuine_column_ambiguity_still_raises(denorm_diamond_deriva_model):
    """Multiple Dataset->element ROUTES are unioned (not an error), but a genuine
    row_per<->include-table COLUMN ambiguity must still raise after the multi-route change.

    Reuses the diamond fixture / ambiguous pair from
    ``tests.local_db.test_planner_rules.TestPrepareWideTableIntegration``:
    the diamond schema reaches ``Subject`` from ``Image`` via two FK paths
    (direct ``Image -> Subject`` and indirect ``Image -> Observation -> Subject``),
    a genuine column ambiguity the caller must resolve with ``via=``. The
    Task-3 multi-route union (membership + FK-reachable Dataset routes) is a
    different path-class and must NOT suppress this Rule-6 guard.
    """
    model = denorm_diamond_deriva_model
    # RIDs are opaque: synthesize a unique placeholder rather than embedding a
    # literal. The guard raises in Phase 0 before any catalog/RID dereference,
    # so the value only needs to be a distinct identifier.
    dataset_rid = f"1-{uuid.uuid4().hex[:8].upper()}"
    with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
        model._planner._prepare_wide_table(
            dataset=None,
            dataset_rid=dataset_rid,
            include_tables=["Image", "Subject"],
            row_per="Image",
        )

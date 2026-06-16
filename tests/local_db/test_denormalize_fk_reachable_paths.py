"""Planner emits ALL reachable Dataset->element routes (membership + FK-reachable),
not just the preferred membership association. Regression for subject-partitioned
feature_values returning 0 (denormalize-fk-reachable-paths spec)."""

from __future__ import annotations


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

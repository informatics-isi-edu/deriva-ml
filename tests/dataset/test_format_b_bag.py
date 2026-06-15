"""Tests for Format-B (rid-set) bag generation."""

from deriva_ml.dataset.bag_builder import _rid_sets_from_reachability


def test_rid_sets_from_reachability_tuple_keys_and_drops_vocab():
    # reached: (schema, table) -> [fk_path, ...]  (paths irrelevant here)
    reached = {
        ("eye-ai", "Image"): [()],
        ("eye-ai", "Subject"): [()],
        ("deriva-ml", "Asset_Role"): [()],  # vocab
    }
    rids_by_table = {
        "Image": {"r1", "r2"},
        "Subject": {"s1"},
        "Asset_Role": {"Input", "Output"},  # bare names; vocab
    }
    vocab_tables = {("deriva-ml", "Asset_Role")}
    result = _rid_sets_from_reachability(reached, rids_by_table, vocab_tables)
    # Tuple-keyed, vocab dropped, RID lists sorted for determinism.
    assert result == {
        ("eye-ai", "Image"): ["r1", "r2"],
        ("eye-ai", "Subject"): ["s1"],
    }


def test_rid_sets_from_reachability_missing_table_is_empty_list():
    """A reached non-vocab table with no RID-set entry maps to []."""
    reached = {("S", "T"): [()]}
    result = _rid_sets_from_reachability(reached, {}, set())
    assert result == {("S", "T"): []}

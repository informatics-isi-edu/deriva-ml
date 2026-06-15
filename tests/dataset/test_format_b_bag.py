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


def test_compute_rid_sets_method_exists():
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    assert hasattr(DatasetBagBuilder, "_compute_rid_sets")


def test_compute_rid_sets_carries_from_model_and_reachability():
    """The shared helper owns the from_model fix and the reachability call."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    src = inspect.getsource(DatasetBagBuilder._compute_rid_sets)
    assert "from_model" in src
    assert "compute_reachability" in src
    assert "_rid_sets_from_reachability" in src


def test_estimate_delegates_to_compute_rid_sets():
    """estimate_bag_size delegates the reachability assembly (no longer inlines
    its own from_model fetch closure)."""
    import inspect

    from deriva_ml.dataset.dataset import Dataset

    src = inspect.getsource(Dataset.estimate_bag_size)
    assert "_compute_rid_sets" in src
    # The from_model closure now lives in the shared helper, not the estimate.
    assert "datapath.from_model" not in src


def test_catalog_bag_builder_accepts_rid_sets():
    """_catalog_bag_builder forwards an opt-in rid_sets to CatalogBagBuilder."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    sig = inspect.signature(DatasetBagBuilder._catalog_bag_builder)
    assert "rid_sets" in sig.parameters
    assert sig.parameters["rid_sets"].default is None


def test_build_bag_uses_rid_sets():
    """build_bag computes rid_sets and passes them to the CatalogBagBuilder."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    src = inspect.getsource(DatasetBagBuilder.build_bag)
    assert "_compute_rid_sets" in src
    assert "rid_sets=" in src

"""Unit tests for :mod:`deriva_ml.catalog.clone_via_bag`.

The full clone-via-bag flow requires two live ERMrest catalogs
(source + destination) and is exercised by integration tests. The
unit tests here focus on the *wrapper* logic — argument
validation, the convenience ``root_rid`` → ``RIDAnchor`` mapping,
and default ``output_dir`` derivation — which doesn't need a
network connection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deriva.bag.anchors import RIDAnchor, TableAnchor
from deriva.bag.traversal import (
    AssetMode,
    DanglingFKStrategy,
    FKTraversalPolicy,
    VocabExport,
)

from deriva_ml.catalog.clone_via_bag import (
    CloneViaBagResult,
    clone_via_bag,
)


def test_clone_via_bag_requires_anchors_or_root_rid() -> None:
    """ValueError when neither anchors nor root_rid is supplied."""
    with pytest.raises(ValueError, match="anchors.*root_rid"):
        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
        )


def test_clone_via_bag_root_rid_maps_to_rid_anchor(tmp_path: Path) -> None:
    """``root_rid="ABC"`` becomes ``RIDAnchor(table="Dataset", rids=["ABC"])``.

    We patch the heavyweight pieces (catalog connection, builder,
    loader) so the test exercises the parameter-mapping path
    without touching the network.
    """
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()
    fake_load_report = MagicMock()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        # Configure builder to return a known bag path.
        builder_instance = MockBuilder.return_value
        builder_instance.build.return_value = fake_bag_path
        # Configure loader to be context-manager-shaped and return
        # our fake report.
        loader_instance = MagicMock()
        loader_instance.run.return_value = fake_load_report
        MockLoader.return_value.__enter__.return_value = loader_instance

        result = clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC123",
            output_dir=tmp_path,
        )

        # The builder must have been called with a single
        # RIDAnchor on Dataset.
        _, kwargs = MockBuilder.call_args
        anchors = kwargs["anchors"]
        assert len(anchors) == 1
        anchor = anchors[0]
        assert isinstance(anchor, RIDAnchor)
        assert anchor.table == "Dataset"
        assert anchor.rids == ["ABC123"]

    assert isinstance(result, CloneViaBagResult)
    assert result.source_catalog_id == "1"
    assert result.dest_catalog_id == "42"
    assert result.bag_path == fake_bag_path
    assert result.load_report is fake_load_report


def test_clone_via_bag_passes_through_explicit_anchors(
    tmp_path: Path,
) -> None:
    """When anchors are explicitly supplied, root_rid is ignored."""
    custom_anchors = [
        TableAnchor(table="Subject"),
        RIDAnchor(table="Image", rids=["I1", "I2"]),
    ]
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            anchors=custom_anchors,
            output_dir=tmp_path,
        )

        _, kwargs = MockBuilder.call_args
        # ``clone_via_bag`` may rewrap the list (e.g., to expand
        # nested-dataset anchors) — non-Dataset anchors should
        # survive content-equivalent.
        assert kwargs["anchors"] == custom_anchors


def test_clone_via_bag_default_output_dir(tmp_path: Path) -> None:
    """``output_dir=None`` derives a name from the catalog IDs."""
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
        patch(
            "deriva_ml.catalog.clone_via_bag.Path.cwd",
            return_value=tmp_path,
        ),
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
        )

        _, kwargs = MockBuilder.call_args
        expected = tmp_path / "clone-1-to-42"
        assert kwargs["output_dir"] == expected


def test_clone_via_bag_passes_policy_through(tmp_path: Path) -> None:
    """Caller-supplied FKTraversalPolicy reaches the builder + loader.

    ``clone_via_bag`` may merge deriva-ml clone defaults
    (``vocab_export=FULL``, ``terminal_tables={Execution,
    Workflow}``, ``dangling_fk_strategy=DELETE``) into a policy
    that left those fields at library defaults. To verify the
    caller's explicit choices pass through, this test supplies a
    policy that customizes every merge-eligible field — the
    merge is a no-op and the same instance reaches both endpoints.
    """
    # Include every merge-eligible field so the merge in
    # clone_via_bag is a no-op and the caller's policy instance
    # reaches the builder / loader unchanged (by identity). Fields
    # currently merge-eligible: vocab_export, terminal_tables,
    # dangling_fk_strategy, intentional_cycles. If you add another
    # to the merge predicate, set it explicitly here too.
    policy = FKTraversalPolicy(
        asset_mode=AssetMode.ROWS_ONLY,
        dangling_fk_strategy=DanglingFKStrategy.NULLIFY,
        vocab_export=VocabExport.FULL,
        terminal_tables={("deriva-ml", "Execution")},
        intentional_cycles=set(),
    )
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            policy=policy,
            output_dir=tmp_path,
        )

        # Both builder and loader receive the same policy instance.
        _, b_kwargs = MockBuilder.call_args
        assert b_kwargs["policy"] is policy
        _, l_kwargs = MockLoader.call_args
        assert l_kwargs["policy"] is policy


def test_clone_via_bag_merges_defaults_into_partial_policy(
    tmp_path: Path,
) -> None:
    """A caller's policy missing clone-required fields gets them merged in.

    Caller-supplied policies that left ``vocab_export``,
    ``terminal_tables``, or ``dangling_fk_strategy`` at the
    library defaults pick up deriva-ml's clone defaults
    (``VocabExport.FULL``, ``{Execution, Workflow}``,
    ``DanglingFKStrategy.DELETE``). The caller's explicit
    choices for other fields (``asset_mode`` here) survive.
    """
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    # Caller explicitly sets asset_mode only. Everything else is
    # left at library defaults — the merge replaces them with
    # clone defaults.
    user_policy = FKTraversalPolicy(asset_mode=AssetMode.ROWS_ONLY)

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            policy=user_policy,
            output_dir=tmp_path,
        )

        _, b_kwargs = MockBuilder.call_args
        merged = b_kwargs["policy"]

        # User's explicit choice survives.
        assert merged.asset_mode == AssetMode.ROWS_ONLY
        # Clone defaults fill in for what user left at library default.
        assert merged.vocab_export == VocabExport.FULL
        assert merged.dangling_fk_strategy == DanglingFKStrategy.DELETE
        assert ("deriva-ml", "Execution") in merged.terminal_tables
        assert ("deriva-ml", "Workflow") in merged.terminal_tables
        # The known Dataset ↔ Dataset_Version cycle is merged in
        # so the bag-load doesn't spam WARNING-level cycle-break
        # messages for the deriva-ml core schema's intentional
        # cycle. See core/constants.py:INTENTIONAL_FK_CYCLES.
        cycle_table_sets = {frozenset(c) for c in merged.intentional_cycles}
        assert (
            frozenset({"deriva-ml.Dataset", "deriva-ml.Dataset_Version"})
            in cycle_table_sets
        )


def test_clone_via_bag_preserves_explicit_empty_cycles(tmp_path: Path) -> None:
    """A caller-supplied ``intentional_cycles=set()`` is not overwritten.

    Regression guard: ``intentional_cycles`` is merge-eligible per
    the model_fields_set predicate. A caller who deliberately
    passes an empty set (or any explicit value) should keep that
    choice — the merge only fills in when the field wasn't
    supplied.
    """
    user_policy = FKTraversalPolicy(
        asset_mode=AssetMode.ROWS_ONLY,
        intentional_cycles=set(),  # explicit empty
    )
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            policy=user_policy,
            output_dir=tmp_path,
        )

        _, b_kwargs = MockBuilder.call_args
        merged = b_kwargs["policy"]

        # Explicit empty set survives — no default cycles injected.
        assert merged.intentional_cycles == set()


def test_clone_via_bag_preserves_explicit_fail_strategy(
    tmp_path: Path,
) -> None:
    """Explicit ``DanglingFKStrategy.FAIL`` survives the merge.

    Regression test for the audit §6.2 finding: the old merge
    used value comparison (``policy.dangling_fk_strategy ==
    DanglingFKStrategy.FAIL``) which silently overrode an
    explicit FAIL because FAIL is also the library default. The
    fix uses ``model_fields_set`` to distinguish "left at
    default" from "explicitly chose FAIL."

    A DBA running a strict-mode clone (any orphan should abort)
    must be able to pass ``dangling_fk_strategy=FAIL`` and have
    it stick.
    """
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    user_policy = FKTraversalPolicy(
        dangling_fk_strategy=DanglingFKStrategy.FAIL,
    )

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            policy=user_policy,
            output_dir=tmp_path,
        )

        _, b_kwargs = MockBuilder.call_args
        merged = b_kwargs["policy"]

        # Explicit FAIL must NOT be overridden by the clone default.
        assert merged.dangling_fk_strategy == DanglingFKStrategy.FAIL


def test_clone_via_bag_preserves_explicit_referenced_only_vocab(
    tmp_path: Path,
) -> None:
    """Explicit ``VocabExport.REFERENCED_ONLY`` survives the merge.

    Regression test for the audit §6.2 finding: same trap as
    explicit FAIL, applied to ``vocab_export``. ``REFERENCED_ONLY``
    is the library default; the old value-comparison merge
    silently overrode it with ``FULL``. The ``model_fields_set``
    form respects the explicit choice.
    """
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    user_policy = FKTraversalPolicy(
        vocab_export=VocabExport.REFERENCED_ONLY,
    )

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential",
            return_value={},
        ),
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            policy=user_policy,
            output_dir=tmp_path,
        )

        _, b_kwargs = MockBuilder.call_args
        merged = b_kwargs["policy"]

        # Explicit REFERENCED_ONLY survives.
        assert merged.vocab_export == VocabExport.REFERENCED_ONLY


def test_clone_via_bag_uses_get_credential_when_creds_absent(
    tmp_path: Path,
) -> None:
    """``source_credential=None`` triggers ``get_credential(hostname)``."""
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential"
        ) as mock_creds,
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        mock_creds.return_value = {"fake": "creds"}
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            output_dir=tmp_path,
        )

        # get_credential should have been called once per host.
        hostnames = [c.args[0] for c in mock_creds.call_args_list]
        assert "src.example.org" in hostnames
        assert "dst.example.org" in hostnames


def test_clone_via_bag_uses_explicit_credentials(tmp_path: Path) -> None:
    """Caller-supplied credentials bypass ``get_credential``."""
    fake_bag_path = tmp_path / "fake-bag"
    fake_bag_path.mkdir()

    with (
        patch("deriva_ml.catalog.clone_via_bag.DerivaServer"),
        patch(
            "deriva_ml.catalog.clone_via_bag.get_credential"
        ) as mock_creds,
        patch(
            "deriva_ml.catalog.clone_via_bag.CatalogBagBuilder"
        ) as MockBuilder,
        patch(
            "deriva_ml.catalog.clone_via_bag.BagCatalogLoader"
        ) as MockLoader,
    ):
        MockBuilder.return_value.build.return_value = fake_bag_path
        loader = MagicMock()
        loader.run.return_value = MagicMock()
        MockLoader.return_value.__enter__.return_value = loader

        clone_via_bag(
            source_hostname="src.example.org",
            source_catalog_id="1",
            dest_hostname="dst.example.org",
            dest_catalog_id="42",
            root_rid="ABC",
            source_credential={"src": "x"},
            dest_credential={"dst": "y"},
            output_dir=tmp_path,
        )

        # When both credentials are explicit, get_credential is
        # never consulted.
        mock_creds.assert_not_called()


# ---------------------------------------------------------------------------
# Nested-dataset anchor expansion
# ---------------------------------------------------------------------------


def _make_mock_catalog_with_dataset_dataset_rows(
    rows_by_seed: dict[tuple[str, ...], list[dict[str, str]]],
) -> MagicMock:
    """Mock that returns predetermined Dataset_Dataset rows per query.

    Mirrors the implementation's datapath call shape:

        pb = catalog.getPathBuilder()
        dd = pb.schemas["deriva-ml"].tables["Dataset_Dataset"]
        rows = dd.filter(dd.Dataset.in_(sorted(frontier))).entities().fetch()

    Args:
        rows_by_seed: ``{tuple(sorted(seed_rids)): rows}`` mapping —
            the helper looks up by the comma-sorted seed list as a
            tuple. Each ``rows`` is a list of ``{"Nested_Dataset": rid}``.
    """
    catalog = MagicMock()

    # The end of the chain — .fetch() — returns rows keyed off the
    # last seen seed list. We capture the seeds during the .in_(...)
    # call so .fetch() can look them up.
    def fetch_side_effect():
        seeds = tuple(sorted(catalog._current_seeds))
        return rows_by_seed.get(seeds, [])

    def in_side_effect(seeds):
        catalog._current_seeds = list(seeds)
        return MagicMock()  # the in_(...) predicate object

    def filter_side_effect(_predicate):
        return catalog._chain

    pb = MagicMock()
    catalog.getPathBuilder.return_value = pb

    dd_table = MagicMock()
    pb.schemas = {"deriva-ml": MagicMock()}
    pb.schemas["deriva-ml"].tables = {"Dataset_Dataset": dd_table}

    chain = MagicMock()
    catalog._chain = chain
    dd_table.filter.side_effect = filter_side_effect
    chain.entities.return_value = chain  # entities() returns self
    chain.fetch.side_effect = fetch_side_effect

    dd_table.Dataset.in_.side_effect = in_side_effect

    return catalog


def test_expand_nested_dataset_anchors_collects_transitive_children() -> None:
    """Anchored Dataset 'P' pulls in nested 'C1', 'C2', and 'C1's nested 'G1'."""
    from deriva_ml.catalog.clone_via_bag import (
        _expand_nested_dataset_anchors,
    )

    catalog = _make_mock_catalog_with_dataset_dataset_rows(
        {
            ("P",): [
                {"Nested_Dataset": "C1"},
                {"Nested_Dataset": "C2"},
            ],
            ("C1", "C2"): [
                {"Nested_Dataset": "G1"},
            ],
            ("G1",): [],
        }
    )
    anchors = [RIDAnchor(table="Dataset", rids=["P"])]
    expanded = _expand_nested_dataset_anchors(anchors, catalog)
    assert len(expanded) == 1
    assert isinstance(expanded[0], RIDAnchor)
    # All four datasets — root plus two children plus one grandchild.
    assert set(expanded[0].rids) == {"P", "C1", "C2", "G1"}


def test_expand_nested_dataset_anchors_leaves_non_dataset_unchanged() -> None:
    """Non-Dataset anchors and non-RID anchor kinds are passed through."""
    from deriva_ml.catalog.clone_via_bag import (
        _expand_nested_dataset_anchors,
    )

    catalog = _make_mock_catalog_with_dataset_dataset_rows({})
    anchors = [
        TableAnchor(table="Subject"),
        RIDAnchor(table="Image", rids=["I1"]),
    ]
    expanded = _expand_nested_dataset_anchors(anchors, catalog)
    assert len(expanded) == 2
    assert expanded[0] == anchors[0]
    assert expanded[1] == anchors[1]


def test_expand_nested_dataset_anchors_no_children() -> None:
    """A leaf Dataset (no nested children) is returned with just its own RID."""
    from deriva_ml.catalog.clone_via_bag import (
        _expand_nested_dataset_anchors,
    )

    catalog = _make_mock_catalog_with_dataset_dataset_rows(
        {("L",): []}
    )
    anchors = [RIDAnchor(table="Dataset", rids=["L"])]
    expanded = _expand_nested_dataset_anchors(anchors, catalog)
    assert set(expanded[0].rids) == {"L"}

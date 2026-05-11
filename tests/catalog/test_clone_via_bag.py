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
        patch(
            "deriva_ml.catalog.clone_via_bag.DerivaServer"
        ) as MockServer,
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
        assert kwargs["anchors"] is custom_anchors


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
    """Caller-supplied FKTraversalPolicy reaches the builder + loader."""
    policy = FKTraversalPolicy(
        asset_mode=AssetMode.ROWS_ONLY,
        dangling_fk_strategy=DanglingFKStrategy.NULLIFY,
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

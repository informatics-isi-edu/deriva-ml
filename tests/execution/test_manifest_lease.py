"""Unit tests for lease_manifest_pending_assets."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _fake_catalog(token_to_rid_map: dict[str, str]):
    """Return a catalog whose POST to ERMrest_RID_Lease returns the
    server-assigned RIDs for the tokens in ``token_to_rid_map``."""
    catalog = MagicMock()

    def fake_post(path: str, json):
        assert path == "/entity/public:ERMrest_RID_Lease"
        response = MagicMock()
        response.json.return_value = [
            {"ID": tok_entry["ID"], "RID": token_to_rid_map[tok_entry["ID"]]}
            for tok_entry in json
        ]
        return response

    catalog.post.side_effect = fake_post
    return catalog


def _fake_manifest(entries: dict):
    """Build a fake AssetManifest whose pending_assets() returns the given
    dict.

    Both the legacy per-key ``set_asset_rid`` and the bulk
    ``set_asset_rids_batch`` are wired to mutate the same shared entries
    dict so tests can assert on ``entries[key].rid`` regardless of which
    write path the production code uses. ``call_count`` on each remains
    available so tests can assert which path fired.
    """
    manifest = MagicMock()
    manifest.pending_assets.return_value = entries

    def fake_set_asset_rid(key, rid):
        entries[key].rid = rid

    def fake_set_asset_rids_batch(items):
        for key, rid in items:
            entries[key].rid = rid

    manifest.set_asset_rid.side_effect = fake_set_asset_rid
    manifest.set_asset_rids_batch.side_effect = fake_set_asset_rids_batch
    return manifest


def _entry(asset_table: str, rid=None):
    from deriva_ml.asset.manifest import AssetEntry
    return AssetEntry(asset_table=asset_table, schema="test-schema", rid=rid)


def test_empty_manifest_is_noop():
    from deriva_ml.execution.manifest_lease import lease_manifest_pending_assets
    catalog = MagicMock()
    manifest = _fake_manifest({})
    lease_manifest_pending_assets(catalog, manifest)
    catalog.post.assert_not_called()


def test_all_entries_already_have_rid_is_noop():
    from deriva_ml.execution.manifest_lease import lease_manifest_pending_assets
    catalog = MagicMock()
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", rid="1-ABC"),
        "Image/b.png": _entry("Image", rid="1-DEF"),
    })
    lease_manifest_pending_assets(catalog, manifest)
    catalog.post.assert_not_called()


def test_rids_assigned_to_entries_missing_rid(monkeypatch):
    from deriva_ml.execution import manifest_lease
    from deriva_ml.execution.manifest_lease import lease_manifest_pending_assets

    # Deterministic tokens — patch generate_lease_token to return fixed tokens.
    tokens_generated = ["tok-a", "tok-b"]
    monkeypatch.setattr(
        manifest_lease, "generate_lease_token",
        lambda: tokens_generated.pop(0),
    )

    catalog = _fake_catalog({"tok-a": "1-NEW-A", "tok-b": "1-NEW-B"})
    entries = {
        "Image/a.png": _entry("Image"),
        "Image/b.png": _entry("Image"),
    }
    manifest = _fake_manifest(entries)

    lease_manifest_pending_assets(catalog, manifest)

    assert entries["Image/a.png"].rid == "1-NEW-A"
    assert entries["Image/b.png"].rid == "1-NEW-B"
    # The bulk path fires once, not per-entry — perf fix in
    # set_asset_rids_batch consolidates N writebacks into one
    # SQLite transaction.
    assert manifest.set_asset_rids_batch.call_count == 1
    items = manifest.set_asset_rids_batch.call_args.args[0]
    assert sorted(items) == sorted(
        [("Image/a.png", "1-NEW-A"), ("Image/b.png", "1-NEW-B")]
    )


def test_mixed_entries_only_missing_rids_leased(monkeypatch):
    from deriva_ml.execution import manifest_lease
    from deriva_ml.execution.manifest_lease import lease_manifest_pending_assets

    monkeypatch.setattr(
        manifest_lease, "generate_lease_token",
        lambda: "tok-fresh",
    )
    catalog = _fake_catalog({"tok-fresh": "1-LEASED"})

    entries = {
        "Image/a.png": _entry("Image", rid="1-ALREADY"),
        "Image/b.png": _entry("Image"),
    }
    manifest = _fake_manifest(entries)

    lease_manifest_pending_assets(catalog, manifest)

    assert entries["Image/a.png"].rid == "1-ALREADY"  # unchanged
    assert entries["Image/b.png"].rid == "1-LEASED"
    # set_asset_rids_batch called once with only the one missing entry.
    assert manifest.set_asset_rids_batch.call_count == 1
    items = manifest.set_asset_rids_batch.call_args.args[0]
    assert items == [("Image/b.png", "1-LEASED")]

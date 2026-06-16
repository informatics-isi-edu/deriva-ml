"""Unit tests for the three internal helpers in ``clone_via_bag.py``.

Closes audit P1 1.2: ``_materialize_bag_dir``, ``_materialize_bag_assets``,
and ``_record_clone_provenance`` had zero direct coverage. The
existing ``tests/catalog/test_clone_via_bag.py`` covers the wrapper-
level surface (policy merge, anchor expansion, credential lookup,
nested-dataset expansion) but every uncovered branch in these
three helpers (zip fallback, ambiguous extraction, missing
fetch.txt, the orphan-stat aggregation loop, the orphan-strategy
str-Enum coercion, the broad-except provenance guard) is unit-
testable with a small filesystem + a ``MagicMock`` LoadReport.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deriva.bag.traversal import AssetMode, DanglingFKStrategy, FKTraversalPolicy

from deriva_ml.catalog.clone_via_bag import (
    _materialize_bag_assets,
    _materialize_bag_dir,
    _record_clone_provenance,
)


# ---------------------------------------------------------------------------
# _materialize_bag_dir
# ---------------------------------------------------------------------------


class TestMaterializeBagDir:
    """Exercise every branch of ``_materialize_bag_dir``.

    The function decides between three outcomes:
      * Bag directory already exists on disk → return it as-is.
      * Bag directory absent but a sibling ``.zip`` exists → extract
        the zip and return the unpacked directory.
      * Neither exists, OR extraction yields an ambiguous set of
        candidate directories → raise ``FileNotFoundError``.

    All three were previously uncovered.
    """

    def test_returns_directory_when_already_unpacked(self, tmp_path: Path):
        """If the bag directory exists on disk, the function is a no-op."""
        bag = tmp_path / "my-bag"
        bag.mkdir()
        (bag / "bagit.txt").touch()  # token bag marker
        result = _materialize_bag_dir(bag)
        assert result == bag

    def test_extracts_zip_when_directory_missing(self, tmp_path: Path):
        """If only ``{bag}.zip`` exists, extract and return the unpacked dir.

        ``CatalogBagBuilder`` hard-codes ``bag_archiver=zip`` and
        removes the unpacked directory after archiving, so the
        bag-pipeline arrives at this helper expecting it to recover
        from the zip.
        """
        bag = tmp_path / "my-bag"  # directory will not exist
        # Build a ``{bag}.zip`` that holds a single top-level dir
        # matching ``bag.name`` — the canonical bdbag layout.
        zip_path = bag.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            # bdbag emits one top-level dir + nested files.
            zf.writestr(f"{bag.name}/bagit.txt", "BagIt-Version: 1.0\n")
            zf.writestr(f"{bag.name}/data/.keep", "")

        result = _materialize_bag_dir(bag)
        assert result == bag
        assert (bag / "bagit.txt").exists()

    def test_raises_when_neither_directory_nor_zip_exists(self, tmp_path: Path):
        """Both directory and zip absent → ``FileNotFoundError`` (loud)."""
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match=r"Bag missing.*no.*fallback"):
            _materialize_bag_dir(missing)

    def test_raises_when_zip_yields_ambiguous_candidates(self, tmp_path: Path):
        """A zip that produces multiple top-level dirs has no clear bag.

        bdbag's contract is one top-level dir per zip; a violation
        of that means we can't pick which one is the bag, so we
        refuse rather than guess. This branch was uncovered before
        — a regression that loosens the heuristic (e.g., to "pick
        the first one") would silently load arbitrary contents.
        """
        bag = tmp_path / "my-bag"  # not pre-extracted
        zip_path = bag.with_suffix(".zip")
        # Build a zip with TWO top-level dirs, neither named
        # ``my-bag``. The extraction will leave both behind in
        # ``tmp_path`` and the helper can't pick one.
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("first-dir/bagit.txt", "")
            zf.writestr("second-dir/bagit.txt", "")
        with pytest.raises(FileNotFoundError, match=r"Could not locate bag directory"):
            _materialize_bag_dir(bag)


# ---------------------------------------------------------------------------
# _materialize_bag_assets
# ---------------------------------------------------------------------------


class TestMaterializeBagAssets:
    """Exercise the skip-vs-materialize guard in ``_materialize_bag_assets``.

    The function short-circuits when ``fetch.txt`` is absent or
    empty. Otherwise it calls ``bdb.materialize(bag_path)``. Both
    branches were uncovered before.
    """

    def test_skips_when_fetch_txt_absent(self, tmp_path: Path):
        """No ``fetch.txt`` → no-op (no import of ``bdb``)."""
        bag = tmp_path / "my-bag"
        bag.mkdir()
        # No fetch.txt at all. The helper should return without
        # importing or calling ``bdb.materialize``.
        with patch("bdbag.bdbag_api.materialize") as mock_materialize:
            _materialize_bag_assets(bag)
        mock_materialize.assert_not_called()

    def test_skips_when_fetch_txt_empty(self, tmp_path: Path):
        """Empty ``fetch.txt`` → no-op.

        A zero-byte ``fetch.txt`` is the bdbag convention for
        "no remote payloads to fetch"; the helper must not call
        ``bdb.materialize`` (which would otherwise do a wasted
        roundtrip).
        """
        bag = tmp_path / "my-bag"
        bag.mkdir()
        (bag / "fetch.txt").touch()  # zero-byte
        with patch("bdbag.bdbag_api.materialize") as mock_materialize:
            _materialize_bag_assets(bag)
        mock_materialize.assert_not_called()

    def test_calls_materialize_when_fetch_txt_has_entries(self, tmp_path: Path):
        """Non-empty ``fetch.txt`` → ``bdb.materialize(bag_path)`` is called."""
        bag = tmp_path / "my-bag"
        bag.mkdir()
        # Non-empty fetch.txt — one line is enough.
        (bag / "fetch.txt").write_text("https://example.org/hatrac/x\t1024\tdata/asset/x.bin\n")
        with patch("bdbag.bdbag_api.materialize") as mock_materialize:
            _materialize_bag_assets(bag)
        mock_materialize.assert_called_once_with(str(bag))


# ---------------------------------------------------------------------------
# _record_clone_provenance
# ---------------------------------------------------------------------------


def _make_load_report(
    *,
    total_rows: int = 100,
    per_table_orphans: dict[str, dict[str, int]] | None = None,
) -> MagicMock:
    """Build a minimal LoadReport-shaped mock.

    The helper reads:
      - ``report.total_rows_inserted`` (an int)
      - ``report.table_stats`` — a dict from table-name to a
        per-table stats object that ``getattr`` reads for
        ``rows_skipped_orphan`` and ``rows_nullified_orphan``
        (both default to 0 when absent).
    """
    report = MagicMock()
    report.total_rows_inserted = total_rows
    table_stats: dict[str, MagicMock] = {}
    for table_name, counts in (per_table_orphans or {}).items():
        ts = MagicMock(spec=[])  # spec=[] so getattr defaults work
        for key, value in counts.items():
            setattr(ts, key, value)
        table_stats[table_name] = ts
    report.table_stats = table_stats
    return report


class TestRecordCloneProvenance:
    """Exercise ``_record_clone_provenance`` against mock catalogs + reports.

    The helper does three things:
      1. Aggregates orphan counters across tables.
      2. Coerces the policy's str-Enums (``dangling_fk_strategy``,
         ``asset_mode``) into the lowercase suffix.
      3. Calls ``set_catalog_provenance`` with a built ``CloneDetails``.

    Each branch was uncovered before.
    """

    def test_aggregates_orphan_stats_across_tables(self):
        """Per-table orphan counters are summed."""
        report = _make_load_report(
            total_rows=200,
            per_table_orphans={
                "Image": {"rows_skipped_orphan": 5, "rows_nullified_orphan": 0},
                "Subject": {"rows_skipped_orphan": 3, "rows_nullified_orphan": 2},
                "Observation": {"rows_skipped_orphan": 0, "rows_nullified_orphan": 7},
            },
        )
        policy = FKTraversalPolicy(
            dangling_fk_strategy=DanglingFKStrategy.NULLIFY,
            asset_mode=AssetMode.ROWS_ONLY,
        )
        dest_catalog = MagicMock()

        with patch("deriva_ml.catalog.clone_via_bag.set_catalog_provenance") as mock_set:
            _record_clone_provenance(
                dest_catalog=dest_catalog,
                source_hostname="src.example.org",
                source_catalog_id="42",
                policy=policy,
                report=report,
            )

        mock_set.assert_called_once()
        clone_details = mock_set.call_args.kwargs["clone_details"]
        assert clone_details.rows_copied == 200
        assert clone_details.orphan_rows_removed == 5 + 3 + 0
        assert clone_details.orphan_rows_nullified == 0 + 2 + 7

    def test_str_enum_coercion_uses_lowercase_suffix(self):
        """Policy str-Enums are coerced to the lowercase suffix.

        ``DanglingFKStrategy.NULLIFY`` → ``"nullify"``;
        ``AssetMode.UPLOAD_IF_MISSING`` → ``"upload_if_missing"``.
        Pre-fix this was uncovered; a regression in
        ``str(policy.X).split(".")[-1].lower()`` would silently
        write the full ``"DanglingFKStrategy.NULLIFY"`` string
        into the annotation.
        """
        report = _make_load_report(total_rows=10)
        policy = FKTraversalPolicy(
            dangling_fk_strategy=DanglingFKStrategy.NULLIFY,
            asset_mode=AssetMode.UPLOAD_IF_MISSING,
        )
        dest_catalog = MagicMock()

        with patch("deriva_ml.catalog.clone_via_bag.set_catalog_provenance") as mock_set:
            _record_clone_provenance(
                dest_catalog=dest_catalog,
                source_hostname="src.example.org",
                source_catalog_id="42",
                policy=policy,
                report=report,
            )

        clone_details = mock_set.call_args.kwargs["clone_details"]
        assert clone_details.orphan_strategy == "nullify"
        assert clone_details.asset_mode == "upload_if_missing"

    def test_swallows_unexpected_report_shape(self):
        """A report missing attributes does not abort the clone.

        The helper's broad ``except Exception`` is the safety net
        for "set_catalog_provenance failed" / "LoadReport shape
        regressed" — provenance writes are best-effort and must
        not break a successful clone. Pre-fix the broad-except
        branch was untested.
        """
        broken_report = MagicMock()
        # No total_rows_inserted, no table_stats — getattr falls
        # through; the explicit attribute reads raise AttributeError.
        del broken_report.table_stats
        del broken_report.total_rows_inserted

        policy = FKTraversalPolicy()
        dest_catalog = MagicMock()

        # The function should not raise — it logs and continues.
        with patch("deriva_ml.catalog.clone_via_bag.set_catalog_provenance") as mock_set:
            _record_clone_provenance(
                dest_catalog=dest_catalog,
                source_hostname="src.example.org",
                source_catalog_id="42",
                policy=policy,
                report=broken_report,
            )

        # The set call never happened (we crashed before getting
        # to it) — which is fine, the helper swallowed the error.
        mock_set.assert_not_called()

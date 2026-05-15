"""Unit tests for ``bag_commit`` helpers with mocked collaborators.

The audit at
``docs/design/deriva-ml-audit-2026-05-phase3-execution.md`` §C.1
flagged that the bag-commit pipeline (the production upload path)
is only tested end-to-end through ``Execution.upload_execution_outputs``.
When something breaks, the failure surface is the full pipeline,
not the individual function — bug locality is poor.

This module pins the highest-leverage invariants of the bag-commit
helpers at the function level, using lightweight mocks instead of
a live catalog. Scope per the audit's #10 pragmatic subset (β):

1. :func:`bag_commit.report_to_asset_map` — pure-function shape
   conversion from a manifest + report to the legacy return shape.
   Covers the ``keys=None`` (full-manifest) vs ``keys=<list>``
   (per-call subset) distinction load-bearing for action #14's
   property-based ``Execution.uploaded_assets``.

2. :func:`bag_commit._add_asset_rows_to_bag` — the lease-batching
   invariant: one ``post_lease_batch`` call per association type,
   passing exactly the right number of tokens.

The remaining bag-commit functions
(:func:`bag_commit._add_staged_feature_rows_to_bag`,
:func:`bag_commit.load_execution_bag`) are out of this PR's scope;
their full mock infrastructure is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deriva_ml.asset.manifest import AssetEntry

# ---------------------------------------------------------------------------
# Lightweight mocks for bag_commit collaborators
# ---------------------------------------------------------------------------


@dataclass
class _FakeSchema:
    """Stand-in for the ``schema`` accessor on a deriva-py Table."""

    name: str


@dataclass
class _FakeTable:
    """Stand-in for a deriva-py Table object.

    ``report_to_asset_map`` only reads ``.schema.name``; the bag-commit
    code in ``_add_asset_rows_to_bag`` additionally reads ``.name``.
    """

    name: str
    schema: _FakeSchema


class _FakeModel:
    """Stand-in for ``DerivaModel`` exposing what ``bag_commit`` uses.

    - ``name_to_table(name)`` → ``_FakeTable``. Raises ``KeyError`` for
      unknown names so the defensive ``try/except`` in
      ``report_to_asset_map`` can be exercised.
    - ``asset_metadata(name)`` → ``set[str]`` of declared metadata
      column names.
    - ``find_association(table, partner)`` →
      ``(assoc_table, asset_fk_col, partner_fk_col)``. The bag-commit
      function reads ``assoc_table.name`` only.
    """

    def __init__(
        self,
        *,
        tables: dict[str, _FakeTable] | None = None,
        metadata_cols: dict[str, set[str]] | None = None,
        associations: dict[tuple[str, str], tuple[_FakeTable, str, str]] | None = None,
    ):
        self._tables = tables or {}
        self._metadata_cols = metadata_cols or {}
        self._associations = associations or {}

    def name_to_table(self, name: str) -> _FakeTable:
        if name not in self._tables:
            raise KeyError(f"unknown table: {name}")
        return self._tables[name]

    def asset_metadata(self, name: str) -> set[str]:
        return self._metadata_cols.get(name, set())

    def find_association(self, table_name: str, partner_name: str):
        key = (table_name, partner_name)
        if key not in self._associations:
            raise KeyError(f"no association: {table_name} ↔ {partner_name}")
        return self._associations[key]


class _FakeCatalog:
    """Catalog stub. Pure marker — only used to identify the catalog
    instance in mocked ``post_lease_batch`` calls.
    """


class _FakeMLObject:
    """Stand-in for the DerivaML instance held at ``execution._ml_object``."""

    def __init__(self):
        self.catalog = _FakeCatalog()


class _FakeManifest:
    """Stand-in for :class:`AssetManifest` exposing only ``.assets``."""

    def __init__(self, entries: dict[str, AssetEntry]):
        self.assets = entries


@dataclass
class _FakeExecution:
    """Stand-in for :class:`Execution` exposing the fields bag_commit reads.

    Carries ``_model``, ``_ml_object``, ``_working_dir``, and
    ``execution_rid``. No lifecycle behaviour.
    """

    _model: _FakeModel
    _working_dir: Path
    execution_rid: str
    _ml_object: _FakeMLObject


class _MockBagBuilder:
    """Stand-in for :class:`BagBuilder` recording ``add_row``,
    ``add_rows``, and ``add_asset`` calls so tests can assert on them.

    ``_add_asset_rows_to_bag`` calls ``bb.add_rows(table_name,
    list_of_dicts)`` for each table it populates and ``bb.add_asset(
    table_name, rid, src, link=True)`` for each asset file. Each call
    is appended to the corresponding list.
    """

    def __init__(self):
        self.add_rows_calls: list[tuple[str, list[dict]]] = []
        self.add_asset_calls: list[tuple[str, str, Path]] = []

    def add_rows(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        self.add_rows_calls.append((table_name, list(rows)))

    def add_asset(self, table_name: str, rid: str, src: Path, link: bool = False) -> None:
        # The bag-commit code always passes ``link=True``; the kwarg
        # is captured implicitly via the call shape (we don't assert
        # on it here — that's deriva-py's concern, not deriva-ml's).
        del link
        self.add_asset_calls.append((table_name, rid, src))


# ---------------------------------------------------------------------------
# report_to_asset_map — shape and filtering
# ---------------------------------------------------------------------------


def _make_execution(tmp_path: Path) -> _FakeExecution:
    """Build a minimal ``_FakeExecution`` with one asset table registered.

    ``Image`` lives in schema ``deriva-ml`` and declares a single
    metadata column ``Caption``. Adjust per test if you need more.
    """
    model = _FakeModel(
        tables={"Image": _FakeTable(name="Image", schema=_FakeSchema(name="deriva-ml"))},
        metadata_cols={"Image": {"Caption"}},
    )
    return _FakeExecution(
        _model=model,
        _working_dir=tmp_path,
        execution_rid="EXE-A",
        _ml_object=_FakeMLObject(),
    )


def test_report_to_asset_map_keys_none_returns_full_manifest(tmp_path):
    """``keys=None`` projects every ``status="uploaded"`` entry."""
    from deriva_ml.execution.bag_commit import report_to_asset_map

    execution = _make_execution(tmp_path)
    manifest = _FakeManifest(
        {
            "Image/a.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-A",
                status="uploaded",
            ),
            "Image/b.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-B",
                status="uploaded",
            ),
        }
    )

    result = report_to_asset_map(
        execution=execution,
        report=None,  # type: ignore[arg-type]  # unused when keys=None
        manifest=manifest,
        keys=None,
    )

    assert set(result) == {"deriva-ml/Image"}
    rids = sorted(p.asset_rid for p in result["deriva-ml/Image"])
    assert rids == ["ASSET-A", "ASSET-B"]


def test_report_to_asset_map_keys_filters_to_subset(tmp_path):
    """``keys=[...]`` returns only the listed entries."""
    from deriva_ml.execution.bag_commit import report_to_asset_map

    execution = _make_execution(tmp_path)
    manifest = _FakeManifest(
        {
            "Image/a.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-A",
                status="uploaded",
            ),
            "Image/b.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-B",
                status="uploaded",
            ),
        }
    )

    result = report_to_asset_map(
        execution=execution,
        report=None,  # type: ignore[arg-type]
        manifest=manifest,
        keys=["Image/a.jpg"],
    )

    assert set(result) == {"deriva-ml/Image"}
    rids = [p.asset_rid for p in result["deriva-ml/Image"]]
    assert rids == ["ASSET-A"]


def test_report_to_asset_map_skips_non_uploaded(tmp_path):
    """``status != "uploaded"`` entries (pending/failed) are excluded."""
    from deriva_ml.execution.bag_commit import report_to_asset_map

    execution = _make_execution(tmp_path)
    manifest = _FakeManifest(
        {
            "Image/done.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-OK",
                status="uploaded",
            ),
            "Image/wait.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid=None,
                status="pending",
            ),
            "Image/bad.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-BAD",
                status="failed",
            ),
        }
    )

    result = report_to_asset_map(
        execution=execution,
        report=None,  # type: ignore[arg-type]
        manifest=manifest,
        keys=None,
    )
    rids = [p.asset_rid for p in result["deriva-ml/Image"]]
    assert rids == ["ASSET-OK"]


def test_report_to_asset_map_skips_malformed_keys(tmp_path):
    """Manifest keys missing the ``/`` separator are silently skipped."""
    from deriva_ml.execution.bag_commit import report_to_asset_map

    execution = _make_execution(tmp_path)
    manifest = _FakeManifest(
        {
            "Image/good.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-OK",
                status="uploaded",
            ),
            "no-slash-here": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-MAL",
                status="uploaded",
            ),
        }
    )

    result = report_to_asset_map(
        execution=execution,
        report=None,  # type: ignore[arg-type]
        manifest=manifest,
        keys=None,
    )
    assert set(result) == {"deriva-ml/Image"}
    rids = [p.asset_rid for p in result["deriva-ml/Image"]]
    assert rids == ["ASSET-OK"]


def test_report_to_asset_map_skips_unknown_tables(tmp_path):
    """Entries whose table isn't in the model are silently skipped.

    Defensive shape — if the manifest carries an asset for a table
    that's been removed from the schema (rare, but possible across
    catalog reinitialisation), the report should keep going for the
    other entries rather than crash.
    """
    from deriva_ml.execution.bag_commit import report_to_asset_map

    execution = _make_execution(tmp_path)
    manifest = _FakeManifest(
        {
            "Image/known.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-A",
                status="uploaded",
            ),
            "Ghost/gone.jpg": AssetEntry(
                asset_table="Ghost",
                schema="deriva-ml",
                rid="ASSET-X",
                status="uploaded",
            ),
        }
    )

    result = report_to_asset_map(
        execution=execution,
        report=None,  # type: ignore[arg-type]
        manifest=manifest,
        keys=None,
    )
    assert set(result) == {"deriva-ml/Image"}


def test_report_to_asset_map_filters_metadata_to_declared_cols(tmp_path):
    """Per-entry metadata is filtered to the columns the table declares."""
    from deriva_ml.execution.bag_commit import report_to_asset_map

    execution = _make_execution(tmp_path)
    manifest = _FakeManifest(
        {
            "Image/a.jpg": AssetEntry(
                asset_table="Image",
                schema="deriva-ml",
                rid="ASSET-A",
                status="uploaded",
                metadata={"Caption": "hello", "NotDeclared": "drop me"},
            ),
        }
    )

    result = report_to_asset_map(
        execution=execution,
        report=None,  # type: ignore[arg-type]
        manifest=manifest,
        keys=None,
    )
    [path] = result["deriva-ml/Image"]
    assert path.asset_metadata == {"Caption": "hello"}


# ---------------------------------------------------------------------------
# _add_asset_rows_to_bag — lease-batching + add_rows wiring
# ---------------------------------------------------------------------------


def test_add_asset_rows_leases_one_batch_per_association(tmp_path, monkeypatch):
    """Two pending assets (no asset-types) → one ``post_lease_batch``
    call for the Execution association, two tokens; ``add_rows`` is
    called for the asset table and the Execution association table.

    Pins audit §C.1's lease-batching invariant. Without this guard,
    a regression that switches to a per-entry POST would multiply
    catalog round-trips.
    """
    from deriva_ml.execution import bag_commit

    # ----- model: Image + Image_Execution association -----
    image = _FakeTable(name="Image", schema=_FakeSchema(name="deriva-ml"))
    image_execution = _FakeTable(name="Image_Execution", schema=_FakeSchema(name="deriva-ml"))
    model = _FakeModel(
        tables={"Image": image},
        metadata_cols={"Image": set()},
        associations={
            ("Image", "Execution"): (image_execution, "Image", "Execution"),
            # Asset_Type is consulted but with an empty type-map the
            # association lookup still has to succeed; provide it.
            ("Image", "Asset_Type"): (
                _FakeTable(name="Image_Asset_Type", schema=_FakeSchema(name="deriva-ml")),
                "Image",
                "Asset_Type",
            ),
        },
    )
    execution = _FakeExecution(
        _model=model,
        _working_dir=tmp_path,
        execution_rid="EXE-A",
        _ml_object=_FakeMLObject(),
    )

    # ----- two pending asset files on disk -----
    # The flat-asset-dir layout is
    # ``{working_dir}/{exe_rid}/asset/Image/``.
    flat_dir = tmp_path / "EXE-A" / "asset" / "Image"
    flat_dir.mkdir(parents=True)
    (flat_dir / "a.jpg").write_bytes(b"alpha")
    (flat_dir / "b.jpg").write_bytes(b"beta")

    entries = [
        ("a.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-A", status="pending")),
        ("b.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-B", status="pending")),
    ]

    # ----- stub the asset-type-map reader to return no types -----
    monkeypatch.setattr(bag_commit, "_read_asset_type_map", lambda execution, asset_table: {})

    # ----- stub the flat_asset_dir helper to point at our tmp tree -----
    monkeypatch.setattr(
        bag_commit,
        "flat_asset_dir",
        lambda working_dir, execution_rid, table_name: working_dir / execution_rid / "asset" / table_name,
    )

    # ----- stub post_lease_batch: deterministic token → RID mapping -----
    lease_calls: list[list[str]] = []

    def _fake_post_lease_batch(*, catalog, tokens):
        lease_calls.append(list(tokens))
        return {token: f"LEASED-{i}" for i, token in enumerate(tokens)}

    monkeypatch.setattr(
        "deriva_ml.execution.rid_lease.post_lease_batch",
        _fake_post_lease_batch,
    )

    # ----- run -----
    bb = _MockBagBuilder()
    final_staged = bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]  # duck-typed
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=2,
    )

    # ----- assertions -----
    assert final_staged == 2

    # Exactly one batched lease call for the Execution association
    # (no Asset_Type pairs → no second batch).
    assert len(lease_calls) == 1
    assert len(lease_calls[0]) == 2  # one token per entry

    # ``add_rows`` was called for the Image table (asset rows) and
    # the Image_Execution association table.
    add_rows_tables = [t for t, _rows in bb.add_rows_calls]
    assert "Image" in add_rows_tables
    assert "Image_Execution" in add_rows_tables
    # Two asset rows, two execution-association rows.
    image_rows = next(rows for t, rows in bb.add_rows_calls if t == "Image")
    exec_rows = next(rows for t, rows in bb.add_rows_calls if t == "Image_Execution")
    assert len(image_rows) == 2
    assert len(exec_rows) == 2

    # Every association row carries the leased RID (not None).
    assert all(row["RID"].startswith("LEASED-") for row in exec_rows)
    # Every association row carries ``Asset_Role="Output"``.
    assert all(row["Asset_Role"] == "Output" for row in exec_rows)

    # ``bb.add_asset`` was called once per entry.
    assert len(bb.add_asset_calls) == 2
    asset_rids = sorted(rid for _table, rid, _src in bb.add_asset_calls)
    assert asset_rids == ["ASSET-A", "ASSET-B"]


def test_add_asset_rows_with_types_leases_two_batches(tmp_path, monkeypatch):
    """Pending assets with asset-types → two ``post_lease_batch`` calls:
    one for the Execution association, one for the Asset_Type pairs.
    """
    from deriva_ml.execution import bag_commit

    image = _FakeTable(name="Image", schema=_FakeSchema(name="deriva-ml"))
    image_execution = _FakeTable(name="Image_Execution", schema=_FakeSchema(name="deriva-ml"))
    image_asset_type = _FakeTable(name="Image_Asset_Type", schema=_FakeSchema(name="deriva-ml"))
    model = _FakeModel(
        tables={"Image": image},
        metadata_cols={"Image": set()},
        associations={
            ("Image", "Execution"): (image_execution, "Image", "Execution"),
            ("Image", "Asset_Type"): (image_asset_type, "Image", "Asset_Type"),
        },
    )
    execution = _FakeExecution(
        _model=model,
        _working_dir=tmp_path,
        execution_rid="EXE-A",
        _ml_object=_FakeMLObject(),
    )

    flat_dir = tmp_path / "EXE-A" / "asset" / "Image"
    flat_dir.mkdir(parents=True)
    (flat_dir / "a.jpg").write_bytes(b"alpha")

    entries = [
        ("a.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-A", status="pending")),
    ]

    # One entry with two asset types → two (entry, type) pairs.
    monkeypatch.setattr(
        bag_commit,
        "_read_asset_type_map",
        lambda execution, asset_table: {"a.jpg": ["TypeX", "TypeY"]},
    )
    monkeypatch.setattr(
        bag_commit,
        "flat_asset_dir",
        lambda working_dir, execution_rid, table_name: working_dir / execution_rid / "asset" / table_name,
    )

    lease_calls: list[list[str]] = []

    def _fake_post_lease_batch(*, catalog, tokens):
        lease_calls.append(list(tokens))
        return {token: f"LEASED-{i}" for i, token in enumerate(tokens)}

    monkeypatch.setattr(
        "deriva_ml.execution.rid_lease.post_lease_batch",
        _fake_post_lease_batch,
    )

    bb = _MockBagBuilder()
    bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=1,
    )

    # Two batched calls: one for Execution association, one for the
    # two Asset_Type pairs. Each one's token count matches the row
    # count it leased for.
    assert len(lease_calls) == 2
    assert len(lease_calls[0]) == 1  # Execution association: one row
    assert len(lease_calls[1]) == 2  # Asset_Type: two pairs

    add_rows_tables = [t for t, _rows in bb.add_rows_calls]
    assert "Image_Asset_Type" in add_rows_tables
    type_rows = next(rows for t, rows in bb.add_rows_calls if t == "Image_Asset_Type")
    assert len(type_rows) == 2
    assert sorted(r["Asset_Type"] for r in type_rows) == ["TypeX", "TypeY"]


def test_add_asset_rows_skips_missing_files(tmp_path, monkeypatch):
    """Asset files that don't exist on disk are skipped with a log
    warning; the function returns the count of files actually staged.

    Defensive shape — a manifest entry whose backing file vanished
    (manual cleanup, disk issue) shouldn't take down the whole
    commit. The other assets still upload.
    """
    from deriva_ml.execution import bag_commit

    image = _FakeTable(name="Image", schema=_FakeSchema(name="deriva-ml"))
    image_execution = _FakeTable(name="Image_Execution", schema=_FakeSchema(name="deriva-ml"))
    image_asset_type = _FakeTable(name="Image_Asset_Type", schema=_FakeSchema(name="deriva-ml"))
    model = _FakeModel(
        tables={"Image": image},
        metadata_cols={"Image": set()},
        associations={
            ("Image", "Execution"): (image_execution, "Image", "Execution"),
            ("Image", "Asset_Type"): (image_asset_type, "Image", "Asset_Type"),
        },
    )
    execution = _FakeExecution(
        _model=model,
        _working_dir=tmp_path,
        execution_rid="EXE-A",
        _ml_object=_FakeMLObject(),
    )

    flat_dir = tmp_path / "EXE-A" / "asset" / "Image"
    flat_dir.mkdir(parents=True)
    # only one of two entries has a file on disk
    (flat_dir / "present.jpg").write_bytes(b"alpha")

    entries = [
        ("present.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-A", status="pending")),
        ("missing.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-B", status="pending")),
    ]

    monkeypatch.setattr(bag_commit, "_read_asset_type_map", lambda execution, asset_table: {})
    monkeypatch.setattr(
        bag_commit,
        "flat_asset_dir",
        lambda working_dir, execution_rid, table_name: working_dir / execution_rid / "asset" / table_name,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.rid_lease.post_lease_batch",
        lambda *, catalog, tokens: {token: f"LEASED-{i}" for i, token in enumerate(tokens)},
    )

    bb = _MockBagBuilder()
    final_staged = bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=2,
    )

    # Only the present file gets staged.
    assert final_staged == 1
    assert len(bb.add_asset_calls) == 1
    image_rows = next(rows for t, rows in bb.add_rows_calls if t == "Image")
    assert len(image_rows) == 1
    assert image_rows[0]["RID"] == "ASSET-A"

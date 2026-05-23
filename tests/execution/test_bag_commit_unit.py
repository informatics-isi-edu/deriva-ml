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
    - ``asset_metadata_sorted(name)`` → ``list[str]`` of declared
      metadata column names in alphabetic order. Mirrors the
      production ``DerivaModel`` helper that pins the sort
      invariant; ``bag_commit`` reads through this method.
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

    def asset_metadata_sorted(self, name: str) -> list[str]:
        return sorted(self.asset_metadata(name))

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


def test_add_asset_rows_reserves_one_batch_when_no_types(tmp_path, monkeypatch):
    """Two pending assets, no user-supplied asset-types → two reservations:
    one for Execution association rows, one for the auto-added
    ``Output_File`` directional tag.

    The Output_File auto-add is the "deriva-ml assigns Input/Output
    to every execution asset" rule — every asset gets a directional
    tag even when the user supplies no content types.

    Post Ex-batch, ``_add_asset_rows_to_bag`` doesn't POST to
    ERMrest_RID_Lease at all — it reserves tokens against the
    shared aggregator that the driver flushes once at end of
    commit. This pins:

    1. Zero direct POSTs from this function.
    2. The Execution-association rows land in ``deferred_emits``
       carrying placeholder tokens (the driver resolves them
       after flush).
    3. Every entry gets one ``Output_File`` Asset_Type row.
    4. The on-disk asset rows still go straight to ``bb.add_rows``
       (no RID rewrite needed — their RIDs come from the
       manifest, not from the lease).
    """
    from deriva_ml.execution import bag_commit
    from deriva_ml.execution.rid_lease import LeaseAggregator

    image = _FakeTable(name="Image", schema=_FakeSchema(name="deriva-ml"))
    image_execution = _FakeTable(name="Image_Execution", schema=_FakeSchema(name="deriva-ml"))
    model = _FakeModel(
        tables={"Image": image},
        metadata_cols={"Image": set()},
        associations={
            ("Image", "Execution"): (image_execution, "Image", "Execution"),
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

    flat_dir = tmp_path / "EXE-A" / "asset" / "Image"
    flat_dir.mkdir(parents=True)
    (flat_dir / "a.jpg").write_bytes(b"alpha")
    (flat_dir / "b.jpg").write_bytes(b"beta")

    entries = [
        ("a.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-A", status="pending")),
        ("b.jpg", AssetEntry(asset_table="Image", schema="deriva-ml", rid="ASSET-B", status="pending")),
    ]

    monkeypatch.setattr(bag_commit, "_read_asset_type_map", lambda execution, asset_table: {})
    monkeypatch.setattr(
        bag_commit,
        "flat_asset_dir",
        lambda working_dir, execution_rid, table_name: working_dir / execution_rid / "asset" / table_name,
    )

    # Make ``post_lease_batch`` raise so we catch any accidental
    # in-function POST. Only the driver's single flush should
    # POST — and this test never calls flush.
    def _post_must_not_be_called(*, catalog, tokens):
        raise AssertionError(
            "post_lease_batch must not be called inside _add_asset_rows_to_bag; "
            "the driver's lease_agg.flush() owns the single round trip."
        )

    monkeypatch.setattr(
        "deriva_ml.execution.rid_lease.post_lease_batch",
        _post_must_not_be_called,
    )

    bb = _MockBagBuilder()
    lease_agg = LeaseAggregator()
    deferred_emits: list = []

    final_staged = bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=2,
        lease_agg=lease_agg,
        deferred_emits=deferred_emits,
    )

    assert final_staged == 2

    # Four tokens reserved on the aggregator:
    # - 2 for Execution-association rows (one per entry)
    # - 2 for Output_File Asset_Type rows (auto-added one per entry,
    #   even though the user supplied no content types). This pins
    #   the "every execution asset gets Input_File or Output_File"
    #   rule for the no-user-types case.
    assert len(lease_agg._tokens) == 4  # noqa: SLF001 — pinning internal state by design

    # Asset rows still emitted directly to ``bb.add_rows`` (they
    # use the manifest's RID, not a leased one).
    add_rows_tables = [t for t, _ in bb.add_rows_calls]
    assert "Image" in add_rows_tables
    image_rows = next(rows for t, rows in bb.add_rows_calls if t == "Image")
    assert len(image_rows) == 2

    # The Image_Execution association rows were DEFERRED — they
    # carry placeholder tokens in the RID slot. The driver's
    # post-flush walk resolves them.
    deferred_tables = [t for t, _ in deferred_emits]
    assert "Image_Execution" in deferred_tables
    exec_rows = next(rows for t, rows in deferred_emits if t == "Image_Execution")
    assert len(exec_rows) == 2
    # RID slot holds reserved tokens (UUID4-ish strings), not
    # leased catalog RIDs. The set of exec-row tokens is a SUBSET
    # of all reserved tokens (the aggregator also holds tokens
    # for the auto-added Output_File rows).
    assert set(row["RID"] for row in exec_rows).issubset(set(lease_agg._tokens))
    # ``Asset_Role`` is still pinned to "Output".
    assert all(row["Asset_Role"] == "Output" for row in exec_rows)

    # Two Output_File Asset_Type rows landed (one per entry) even
    # though the user supplied no content types. This is the
    # deriva-ml-assigns-Input/Output directional-tag rule.
    assert "Image_Asset_Type" in deferred_tables
    type_rows = next(rows for t, rows in deferred_emits if t == "Image_Asset_Type")
    assert len(type_rows) == 2
    assert all(row["Asset_Type"] == "Output_File" for row in type_rows)
    assert {row["Image"] for row in type_rows} == {"ASSET-A", "ASSET-B"}

    # Asset hardlinks still happen directly.
    assert len(bb.add_asset_calls) == 2


def test_add_asset_rows_with_types_reserves_both_associations(tmp_path, monkeypatch):
    """Pending assets WITH asset-types → two ``lease_agg.reserve`` calls
    on the aggregator (Execution + Asset_Type), no inline POSTs.

    The two reservations accumulate on the same aggregator —
    pre-fix this was two separate ``post_lease_batch`` POSTs.
    """
    from deriva_ml.execution import bag_commit
    from deriva_ml.execution.rid_lease import LeaseAggregator

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

    def _post_must_not_be_called(*, catalog, tokens):
        raise AssertionError("post_lease_batch must not be called inside _add_asset_rows_to_bag")

    monkeypatch.setattr(
        "deriva_ml.execution.rid_lease.post_lease_batch",
        _post_must_not_be_called,
    )

    bb = _MockBagBuilder()
    lease_agg = LeaseAggregator()
    deferred_emits: list = []
    bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=1,
        lease_agg=lease_agg,
        deferred_emits=deferred_emits,
    )

    # 1 token for the exec association + 3 tokens for the three
    # (asset, type) pairs — TypeX, TypeY, AND the auto-added
    # Output_File directional tag — totalling 4. All accumulated
    # on one aggregator — the driver will flush them in ONE POST,
    # which is the whole point of Ex-batch.
    assert len(lease_agg._tokens) == 4  # noqa: SLF001

    # Both deferred emits land — Image_Execution and Image_Asset_Type.
    deferred_tables = [t for t, _ in deferred_emits]
    assert "Image_Execution" in deferred_tables
    assert "Image_Asset_Type" in deferred_tables
    type_rows = next(rows for t, rows in deferred_emits if t == "Image_Asset_Type")
    # Three tags: TypeX, TypeY, and the auto-added Output_File
    # directional tag (the canonical "deriva-ml assigns
    # Input/Output to every execution asset" rule).
    assert len(type_rows) == 3
    assert sorted(r["Asset_Type"] for r in type_rows) == ["Output_File", "TypeX", "TypeY"]


def test_add_asset_rows_does_not_duplicate_explicit_output_file_tag(tmp_path, monkeypatch):
    """User-supplied ``Output_File`` is honored without duplication.

    When the caller explicitly passes
    ``ExecAssetType.output_file.value`` (e.g., a pre-Phase-3
    workflow that still types its outputs manually), the
    auto-add machinery must NOT emit a second ``Output_File``
    Asset_Type row.

    Mirrors the dedup contract in
    ``asset_upload.update_asset_execution_table`` Output branch
    (see lines 584-593 of ``asset_upload.py``). This pin guards
    against a future regression where the bag-commit path
    diverges from the canonical Output-tag handling.
    """
    from deriva_ml.execution import bag_commit
    from deriva_ml.execution.rid_lease import LeaseAggregator

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

    # Caller explicitly tags this asset as Model_File + Output_File.
    monkeypatch.setattr(
        bag_commit,
        "_read_asset_type_map",
        lambda execution, asset_table: {"a.jpg": ["Model_File", "Output_File"]},
    )
    monkeypatch.setattr(
        bag_commit,
        "flat_asset_dir",
        lambda working_dir, execution_rid, table_name: working_dir / execution_rid / "asset" / table_name,
    )

    bb = _MockBagBuilder()
    lease_agg = LeaseAggregator()
    deferred_emits: list = []
    bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=1,
        lease_agg=lease_agg,
        deferred_emits=deferred_emits,
    )

    # Output_File appears exactly once in the type rows.
    type_rows = next(rows for t, rows in deferred_emits if t == "Image_Asset_Type")
    output_file_rows = [r for r in type_rows if r["Asset_Type"] == "Output_File"]
    assert len(output_file_rows) == 1, (
        f"Output_File must appear exactly once; got {len(output_file_rows)} rows. "
        f"Full tag set: {sorted(r['Asset_Type'] for r in type_rows)}"
    )
    # Model_File is preserved too.
    assert sorted(r["Asset_Type"] for r in type_rows) == ["Model_File", "Output_File"]


def test_add_asset_rows_skips_missing_files(tmp_path, monkeypatch):
    """Asset files that don't exist on disk are skipped with a log
    warning; the function returns the count of files actually staged.

    Defensive shape — a manifest entry whose backing file vanished
    (manual cleanup, disk issue) shouldn't take down the whole
    commit. The other assets still upload.
    """
    from deriva_ml.execution import bag_commit
    from deriva_ml.execution.rid_lease import LeaseAggregator

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

    bb = _MockBagBuilder()
    lease_agg = LeaseAggregator()
    deferred_emits: list = []
    final_staged = bag_commit._add_asset_rows_to_bag(
        bb=bb,
        execution=execution,  # type: ignore[arg-type]
        asset_table_name="Image",
        entries=entries,
        progress_callback=None,
        staged_so_far=0,
        total_assets=2,
        lease_agg=lease_agg,
        deferred_emits=deferred_emits,
    )

    # Only the present file gets staged.
    assert final_staged == 1
    assert len(bb.add_asset_calls) == 1
    image_rows = next(rows for t, rows in bb.add_rows_calls if t == "Image")
    assert len(image_rows) == 1
    assert image_rows[0]["RID"] == "ASSET-A"

"""Unit tests for :meth:`DerivaML.validate_config_file` and :meth:`validate_config_directory`.

These tests stub the dataset/asset/workflow validation primitives the
new methods compose, so the suite exercises the integration boundary
(AST -> per-kind buckets -> per-RID validators -> aggregated report)
without needing a live catalog. The primitives themselves
(``validate_dataset_specs``, ``_validate_asset_spec``,
``_validate_workflow_rid``) have their own unit suites.

Coverage targets:

- A clean config with one entry of each kind returns ``all_valid=True``.
- A config with a bad dataset RID surfaces ``rid_not_found`` with the
  parsed entry echoed back.
- A config with a missing ``version=`` kwarg surfaces
  ``version_missing`` (a diagnostic that doesn't exist on the
  underlying spec validators).
- A config with an unresolvable RID (``rid=some_func()``) surfaces
  ``rid_unresolvable`` without calling the catalog primitives.
- A DerivaMLConfig entry whose hostname doesn't match the connection
  surfaces ``catalog_hostname_mismatch``.
- A directory walk skips ``__pycache__`` and dot-prefixed paths.
- A directory walk continues past a single unparseable file.
"""

from __future__ import annotations

from pathlib import Path

from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.config import ConfigValidationReport
from deriva_ml.core.mixins.dataset import DatasetMixin
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.validation import (
    AssetSpecResult,
    DatasetSpecResult,
    DatasetSpecValidationReport,
    WorkflowSpecResult,
)

# ---------------------------------------------------------------------------
# Stub mixin: provides just enough surface for the tested methods.
# ---------------------------------------------------------------------------


class _StubMixin(DatasetMixin):
    """A DatasetMixin instance with stubbed primitives.

    The primitives this mixin's ``validate_config_file`` calls into are
    ``validate_dataset_specs``, ``_validate_asset_spec``, and
    ``_validate_workflow_rid``. We patch them via instance methods so
    the tests can inject canned per-RID responses without going near
    the real catalog code paths.
    """

    def __init__(
        self,
        *,
        host_name: str = "data.example.org",
        catalog_id: str = "42",
        dataset_results_by_rid: dict[str, DatasetSpecResult] | None = None,
        asset_results_by_rid: dict[str, AssetSpecResult] | None = None,
        workflow_results_by_rid: dict[str, WorkflowSpecResult] | None = None,
    ) -> None:
        self.host_name = host_name
        self.catalog_id = catalog_id
        self._dataset_results = dataset_results_by_rid or {}
        self._asset_results = asset_results_by_rid or {}
        self._workflow_results = workflow_results_by_rid or {}

    # The mixin's ``validate_dataset_specs`` walks ``ml.resolve_rid``
    # etc.; override to return canned results based on RID.
    def validate_dataset_specs(self, specs, **_):  # type: ignore[override]
        results = []
        for s in specs:
            canned = self._dataset_results.get(s.rid)
            if canned is None:
                # Default: spec is valid (we don't know the rid yet, so
                # assume well-formed).
                results.append(DatasetSpecResult(spec=s, valid=True, resolved_version=str(s.version)))
            else:
                # Echo the spec back with the canned outcome's reasons.
                results.append(
                    DatasetSpecResult(
                        spec=s,
                        valid=canned.valid,
                        reasons=list(canned.reasons),
                        actual_table=canned.actual_table,
                        available_versions=canned.available_versions,
                        dataset_name=canned.dataset_name,
                    )
                )
        return DatasetSpecValidationReport(
            all_valid=all(r.valid for r in results),
            results=results,
        )

    def _validate_asset_spec(self, spec):  # type: ignore[override]
        canned = self._asset_results.get(spec.rid)
        if canned is None:
            return AssetSpecResult(spec=spec, valid=True, asset_table="Image")
        return AssetSpecResult(
            spec=spec,
            valid=canned.valid,
            reasons=list(canned.reasons),
            actual_table=canned.actual_table,
            asset_table=canned.asset_table,
            filename=canned.filename,
        )

    def _validate_workflow_rid(self, rid):  # type: ignore[override]
        canned = self._workflow_results.get(rid)
        if canned is None:
            return WorkflowSpecResult(rid=rid, valid=True, workflow_name="canned-workflow")
        return canned


def _write(tmp_path: Path, source: str, name: str = "configs.py") -> Path:
    p = tmp_path / name
    p.write_text(source)
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_clean_config_returns_all_valid(tmp_path: Path) -> None:
    src = """
datasets_store(name="train", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))
assets_store(name="w", spec=AssetSpecConfig(rid="1-BBBB", cache=True))
workflow_store(name="t", spec=Workflow(workflow_rid="1-CCCC"))
deriva_store(DerivaMLConfig, hostname="data.example.org", catalog_id="42")
"""
    path = _write(tmp_path, src)
    ml = _StubMixin()
    report = ml.validate_config_file(path)

    assert isinstance(report, ConfigValidationReport)
    assert report.file_count == 1
    assert report.entry_count == 4
    assert report.all_valid is True
    assert all(r.valid for r in report.results)
    assert [r.entry.entry_kind for r in report.results] == [
        "DatasetSpecConfig",
        "AssetSpecConfig",
        "Workflow",
        "DerivaMLConfig",
    ]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_dataset_rid_not_found(tmp_path: Path) -> None:
    src = 'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-NOPE", version="0.1.0"))\n'
    path = _write(tmp_path, src)
    ml = _StubMixin(
        dataset_results_by_rid={
            "1-NOPE": DatasetSpecResult(
                spec=DatasetSpec(rid="1-NOPE", version="0.1.0"),
                valid=False,
                reasons=["rid_not_found"],
            ),
        },
    )
    report = ml.validate_config_file(path)
    assert report.all_valid is False
    assert len(report.results) == 1
    r = report.results[0]
    assert r.valid is False
    assert "rid_not_found" in r.reasons
    assert r.entry.rid == "1-NOPE"


def test_dataset_version_not_found_surfaces_available_versions(tmp_path: Path) -> None:
    src = 'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="9.9.9"))\n'
    path = _write(tmp_path, src)
    ml = _StubMixin(
        dataset_results_by_rid={
            "1-AAAA": DatasetSpecResult(
                spec=DatasetSpec(rid="1-AAAA", version="9.9.9"),
                valid=False,
                reasons=["version_not_found"],
                available_versions=["0.4.0", "0.3.0", "0.2.0"],
            ),
        },
    )
    report = ml.validate_config_file(path)
    r = report.results[0]
    assert r.valid is False
    assert "version_not_found" in r.reasons
    assert r.available_versions == ["0.4.0", "0.3.0", "0.2.0"]


def test_missing_version_yields_version_missing(tmp_path: Path) -> None:
    """A DatasetSpecConfig with no ``version=`` kwarg yields the
    ``version_missing`` diagnostic instead of the misleading
    ``version_not_found``."""
    src = 'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA"))\n'
    path = _write(tmp_path, src)
    ml = _StubMixin(
        dataset_results_by_rid={
            # Default valid -- the canned RID exists, but the entry has
            # no version, so the wrapper rewrites to version_missing.
            "1-AAAA": DatasetSpecResult(
                spec=DatasetSpec(rid="1-AAAA", version="0.0.0"),
                valid=False,
                reasons=["version_not_found"],
                available_versions=["0.1.0"],
            ),
        },
    )
    report = ml.validate_config_file(path)
    r = report.results[0]
    assert r.valid is False
    assert "version_missing" in r.reasons
    # The version_not_found inner reason has been rewritten.
    assert "version_not_found" not in r.reasons


def test_unresolvable_rid_does_not_call_catalog(tmp_path: Path) -> None:
    """An ``rid=some_func()`` shape surfaces ``rid_unresolvable`` without
    making any catalog calls (the AST couldn't extract a RID to look up)."""
    src = """
def get_rid():
    return "1-ZZZZ"
datasets_store(name="x", spec=DatasetSpecConfig(rid=get_rid(), version="0.1.0"))
"""
    path = _write(tmp_path, src)
    # Empty stub map -- if anyone called validate_dataset_specs with a
    # canned-rid lookup, the default-valid branch would trip a False
    # positive on our assertion.
    ml = _StubMixin()
    # Wrap validate_dataset_specs so we can assert it wasn't called.
    real_validate = ml.validate_dataset_specs
    call_count = {"n": 0}

    def counting_validate(specs, **kw):  # type: ignore[no-redef]
        call_count["n"] += 1
        return real_validate(specs, **kw)

    ml.validate_dataset_specs = counting_validate  # type: ignore[method-assign]
    report = ml.validate_config_file(path)

    assert call_count["n"] == 0, "no catalog call should happen for unresolvable RIDs"
    assert len(report.results) == 1
    assert report.results[0].valid is False
    assert "rid_unresolvable" in report.results[0].reasons


def test_deriva_ml_hostname_mismatch(tmp_path: Path) -> None:
    src = """
deriva_store(DerivaMLConfig, hostname="other.example.org", catalog_id="42")
"""
    path = _write(tmp_path, src)
    ml = _StubMixin(host_name="data.example.org", catalog_id="42")
    report = ml.validate_config_file(path)
    r = report.results[0]
    assert r.valid is False
    assert "catalog_hostname_mismatch" in r.reasons


def test_deriva_ml_catalog_id_mismatch(tmp_path: Path) -> None:
    src = """
deriva_store(DerivaMLConfig, hostname="data.example.org", catalog_id="999")
"""
    path = _write(tmp_path, src)
    ml = _StubMixin(host_name="data.example.org", catalog_id="42")
    report = ml.validate_config_file(path)
    r = report.results[0]
    assert r.valid is False
    assert "catalog_id_mismatch" in r.reasons


# ---------------------------------------------------------------------------
# Asset / workflow failure modes
# ---------------------------------------------------------------------------


def test_asset_not_found(tmp_path: Path) -> None:
    src = 'assets_store(name="w", spec=AssetSpecConfig(rid="1-NOPE"))\n'
    path = _write(tmp_path, src)
    ml = _StubMixin(
        asset_results_by_rid={
            "1-NOPE": AssetSpecResult(
                spec=AssetSpec(rid="1-NOPE"),
                valid=False,
                reasons=["rid_not_found"],
            ),
        },
    )
    report = ml.validate_config_file(path)
    r = report.results[0]
    assert r.valid is False
    assert "rid_not_found" in r.reasons


def test_workflow_not_a_workflow(tmp_path: Path) -> None:
    src = 'workflow_store(name="t", spec=Workflow(workflow_rid="1-WRNG"))\n'
    path = _write(tmp_path, src)
    ml = _StubMixin(
        workflow_results_by_rid={
            "1-WRNG": WorkflowSpecResult(
                rid="1-WRNG",
                valid=False,
                reasons=["not_a_workflow"],
                actual_table="Dataset",
            ),
        },
    )
    report = ml.validate_config_file(path)
    r = report.results[0]
    assert r.valid is False
    assert "not_a_workflow" in r.reasons
    assert r.actual_table == "Dataset"


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


def test_unparseable_file_recorded_in_parse_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "this is not python (((\n")
    ml = _StubMixin()
    report = ml.validate_config_file(path)
    assert report.file_count == 0
    assert report.entry_count == 0
    assert report.all_valid is False
    assert len(report.parse_errors) == 1
    assert "syntax" in report.parse_errors[0].message.lower() or report.parse_errors[0].message


def test_missing_file_recorded_in_parse_errors(tmp_path: Path) -> None:
    ml = _StubMixin()
    report = ml.validate_config_file(tmp_path / "nope.py")
    assert report.file_count == 0
    assert report.parse_errors


# ---------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------


def test_directory_walk_includes_subdirs(tmp_path: Path) -> None:
    (tmp_path / "datasets.py").write_text(
        'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))\n'
    )
    subdir = tmp_path / "dev"
    subdir.mkdir()
    (subdir / "datasets.py").write_text(
        'datasets_store(name="y", spec=DatasetSpecConfig(rid="1-BBBB", version="0.2.0"))\n'
    )
    ml = _StubMixin()
    report = ml.validate_config_directory(tmp_path)
    assert report.entry_count == 2


def test_directory_walk_skips_pycache(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "bogus.py").write_text(
        "this is not python (((\n"  # Would be a parse error if visited.
    )
    (tmp_path / "datasets.py").write_text(
        'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))\n'
    )
    ml = _StubMixin()
    report = ml.validate_config_directory(tmp_path)
    # Only the real file should have been visited.
    assert report.file_count == 1
    assert report.parse_errors == []
    assert report.entry_count == 1


def test_directory_walk_continues_past_unparseable_file(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("this is not python (((\n")
    (tmp_path / "good.py").write_text(
        'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))\n'
    )
    ml = _StubMixin()
    report = ml.validate_config_directory(tmp_path)
    # The good file's entry was still found.
    assert report.entry_count == 1
    # The broken file is in parse_errors.
    assert len(report.parse_errors) == 1
    # ...and the report says NOT all valid because of the parse error.
    assert report.all_valid is False


def test_directory_walk_missing_dir_returns_parse_error(tmp_path: Path) -> None:
    ml = _StubMixin()
    report = ml.validate_config_directory(tmp_path / "nope")
    assert report.file_count == 0
    assert len(report.parse_errors) == 1
    assert "not found" in report.parse_errors[0].message


def test_directory_walk_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "datasets.py").write_text(
        'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))\n'
    )
    subdir = tmp_path / "dev"
    subdir.mkdir()
    (subdir / "datasets.py").write_text(
        'datasets_store(name="y", spec=DatasetSpecConfig(rid="1-BBBB", version="0.2.0"))\n'
    )
    ml = _StubMixin()
    report = ml.validate_config_directory(tmp_path, recursive=False)
    # Only the top-level file is visited.
    assert report.entry_count == 1
    assert report.results[0].entry.rid == "1-AAAA"

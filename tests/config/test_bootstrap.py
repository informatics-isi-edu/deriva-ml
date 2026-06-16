"""Unit tests for :meth:`DerivaML.bootstrap_config`.

These tests stub the catalog-read primitives the method composes
(``find_datasets``, ``list_asset_tables``, ``list_assets``,
``find_workflows``) so the suite runs without a live catalog. The
stubs are intentionally minimal -- shape-of-data only -- because the
bootstrap method's value-add is *what it suggests*, not how it reads
the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deriva_ml.config import BootstrapReport
from deriva_ml.config.bootstrap import _sanitize_config_name
from deriva_ml.core.mixins.dataset import DatasetMixin

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubDataset:
    dataset_rid: str
    dataset_types: list[str] = field(default_factory=list)
    current_version: str | None = "0.1.0"
    description: str = ""


@dataclass
class _StubAsset:
    asset_rid: str
    filename: str = ""


@dataclass
class _StubAssetTable:
    name: str


@dataclass
class _StubWorkflow:
    workflow_rid: str
    name: str = ""


class _StubMixin(DatasetMixin):
    """Minimal :class:`DatasetMixin` for bootstrap tests."""

    def __init__(
        self,
        *,
        host_name: str = "data.example.org",
        catalog_id: str = "42",
        datasets: list[_StubDataset] | None = None,
        asset_tables: list[_StubAssetTable] | None = None,
        assets_by_table: dict[str, list[_StubAsset]] | None = None,
        workflows: list[_StubWorkflow] | None = None,
    ) -> None:
        self.host_name = host_name
        self.catalog_id = catalog_id
        self._datasets = datasets or []
        self._asset_tables = asset_tables or []
        self._assets_by_table = assets_by_table or {}
        self._workflows = workflows or []

    # Stub the catalog-read primitives. Each replaces the real mixin
    # method without invoking ermrest.
    def find_datasets(self, deleted: bool = False, sort=None):  # type: ignore[override]
        return list(self._datasets)

    def list_asset_tables(self):  # type: ignore[override]
        return list(self._asset_tables)

    def list_assets(self, table: Any):  # type: ignore[override]
        name = table.name if hasattr(table, "name") else str(table)
        return list(self._assets_by_table.get(name, []))

    def find_workflows(self, sort=None):  # type: ignore[override]
        return list(self._workflows)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_bootstrap_returns_one_per_kind_minimum() -> None:
    """A catalog with content of every kind produces at least one entry per kind."""
    ml = _StubMixin(
        datasets=[
            _StubDataset(
                dataset_rid="1-AAAA",
                dataset_types=["Training", "Labeled"],
                current_version="0.4.0",
                description="CIFAR-10 training images",
            ),
        ],
        asset_tables=[_StubAssetTable(name="Image")],
        assets_by_table={
            "Image": [_StubAsset(asset_rid="1-BBBB", filename="model_weights.pt")],
        },
        workflows=[_StubWorkflow(workflow_rid="1-CCCC", name="train_cifar10")],
    )
    report = ml.bootstrap_config()

    assert isinstance(report, BootstrapReport)
    assert report.catalog == {"hostname": "data.example.org", "catalog_id": "42"}
    kinds = [s.kind for s in report.suggestions]
    assert "deriva_ml" in kinds
    assert "datasets" in kinds
    assert "assets" in kinds
    assert "workflow" in kinds


def test_dataset_suggestion_pins_released_version() -> None:
    ml = _StubMixin(
        datasets=[
            _StubDataset(
                dataset_rid="1-AAAA",
                dataset_types=["Training"],
                current_version="0.4.0",
                description="training set",
            ),
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"])
    assert len(report.suggestions) == 1
    s = report.suggestions[0]
    assert s.kind == "datasets"
    assert s.rid == "1-AAAA"
    assert s.version == "0.4.0"
    assert 'rid="1-AAAA"' in s.spec_string
    assert 'version="0.4.0"' in s.spec_string


def test_dataset_with_dev_version_is_skipped() -> None:
    ml = _StubMixin(
        datasets=[
            _StubDataset(
                dataset_rid="1-AAAA",
                dataset_types=["Training"],
                current_version="0.4.0.post1.dev3",  # dev label
                description="training set",
            ),
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"])
    assert report.suggestions == []
    assert len(report.skipped) == 1
    assert report.skipped[0].rid == "1-AAAA"
    assert "dev label" in report.skipped[0].reason


def test_dataset_with_no_current_version_is_skipped() -> None:
    ml = _StubMixin(
        datasets=[
            _StubDataset(
                dataset_rid="1-AAAA",
                dataset_types=["Training"],
                current_version=None,
            ),
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"])
    assert report.suggestions == []
    assert len(report.skipped) == 1
    assert report.skipped[0].reason == "no current version"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_dataset_type_filter_default_excludes_non_matching() -> None:
    """Default filter is Training/Testing/Validation/Complete/Labeled."""
    ml = _StubMixin(
        datasets=[
            _StubDataset(dataset_rid="1-AA", dataset_types=["Training"], current_version="0.1.0"),
            _StubDataset(dataset_rid="1-BB", dataset_types=["Split"], current_version="0.1.0"),  # excluded
            _StubDataset(dataset_rid="1-CC", dataset_types=["Foo"], current_version="0.1.0"),  # excluded
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"])
    rids = [s.rid for s in report.suggestions]
    assert rids == ["1-AA"]
    skipped_rids = [s.rid for s in report.skipped]
    assert "1-BB" in skipped_rids
    assert "1-CC" in skipped_rids


def test_dataset_type_filter_empty_means_all_types() -> None:
    """``dataset_type_filter=[]`` includes every type."""
    ml = _StubMixin(
        datasets=[
            _StubDataset(dataset_rid="1-AA", dataset_types=["Split"], current_version="0.1.0"),
            _StubDataset(dataset_rid="1-BB", dataset_types=["Foo"], current_version="0.1.0"),
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"], dataset_type_filter=[])
    rids = sorted(s.rid for s in report.suggestions)
    assert rids == ["1-AA", "1-BB"]


def test_dataset_type_filter_custom() -> None:
    """A custom filter narrows to exactly the requested types."""
    ml = _StubMixin(
        datasets=[
            _StubDataset(dataset_rid="1-AA", dataset_types=["Training"], current_version="0.1.0"),
            _StubDataset(dataset_rid="1-BB", dataset_types=["Validation"], current_version="0.1.0"),
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"], dataset_type_filter=["Validation"])
    rids = [s.rid for s in report.suggestions]
    assert rids == ["1-BB"]


# ---------------------------------------------------------------------------
# Kinds gating
# ---------------------------------------------------------------------------


def test_kinds_gates_what_is_returned() -> None:
    ml = _StubMixin(
        datasets=[_StubDataset(dataset_rid="1-AA", dataset_types=["Training"])],
        asset_tables=[_StubAssetTable(name="Image")],
        assets_by_table={"Image": [_StubAsset(asset_rid="1-BB")]},
        workflows=[_StubWorkflow(workflow_rid="1-CC")],
    )
    report = ml.bootstrap_config(kinds=["datasets"])
    assert {s.kind for s in report.suggestions} == {"datasets"}


def test_kinds_none_returns_all_four() -> None:
    ml = _StubMixin(
        datasets=[],
        asset_tables=[],
        workflows=[],
    )
    # Even with empty content, the deriva_ml suggestion always fires
    # (it's catalog-state-independent -- just pins the connection).
    report = ml.bootstrap_config()
    assert {s.kind for s in report.suggestions} == {"deriva_ml"}


# ---------------------------------------------------------------------------
# Asset suggestions
# ---------------------------------------------------------------------------


def test_assets_skip_builtin_metadata_tables() -> None:
    """``Execution_Metadata`` and ``Execution_Asset`` are skipped."""
    ml = _StubMixin(
        asset_tables=[
            _StubAssetTable(name="Image"),
            _StubAssetTable(name="Execution_Metadata"),
            _StubAssetTable(name="Execution_Asset"),
        ],
        assets_by_table={
            "Image": [_StubAsset(asset_rid="1-AA", filename="img.png")],
            "Execution_Metadata": [_StubAsset(asset_rid="1-XX", filename="config.json")],
            "Execution_Asset": [_StubAsset(asset_rid="1-YY", filename="weights.pt")],
        },
    )
    report = ml.bootstrap_config(kinds=["assets"])
    rids = [s.rid for s in report.suggestions]
    assert rids == ["1-AA"]
    skipped_rids = [s.rid for s in report.skipped]
    assert "Execution_Metadata" in skipped_rids
    assert "Execution_Asset" in skipped_rids


# ---------------------------------------------------------------------------
# Workflow suggestions
# ---------------------------------------------------------------------------


def test_workflow_with_no_rid_is_silently_dropped() -> None:
    """An in-memory Workflow without a catalog RID isn't suggested."""
    ml = _StubMixin(
        workflows=[
            _StubWorkflow(workflow_rid="1-AA", name="trainer"),
            _StubWorkflow(workflow_rid=None, name="just-in-memory"),  # type: ignore[arg-type]
        ],
    )
    report = ml.bootstrap_config(kinds=["workflow"])
    rids = [s.rid for s in report.suggestions]
    assert rids == ["1-AA"]


# ---------------------------------------------------------------------------
# config_name sanitization
# ---------------------------------------------------------------------------


def test_sanitize_config_name_uses_description() -> None:
    assert _sanitize_config_name("CIFAR-10 Training Set!", "fallback") == "cifar_10_training_set"


def test_sanitize_config_name_falls_back_when_empty() -> None:
    assert _sanitize_config_name("", "1-XYZW") == "1_xyzw"


def test_sanitize_config_name_falls_back_when_unusable() -> None:
    assert _sanitize_config_name("!!!", "1-AB") == "1_ab"


def test_sanitize_config_name_prepends_ds_when_starts_with_digit() -> None:
    # "8 little cats" sanitizes to "8_little_cats"; we prepend "ds_" so
    # the result is a valid Python identifier.
    name = _sanitize_config_name("8 little cats", "fallback")
    assert name.startswith("ds_")


def test_sanitize_config_name_truncates_long_strings() -> None:
    long_desc = "a" * 80
    name = _sanitize_config_name(long_desc, "fb")
    assert len(name) <= 40


# ---------------------------------------------------------------------------
# Spec string formatting
# ---------------------------------------------------------------------------


def test_dataset_spec_string_is_ready_to_paste() -> None:
    ml = _StubMixin(
        datasets=[
            _StubDataset(
                dataset_rid="1-AA",
                dataset_types=["Training"],
                current_version="0.4.0",
            ),
        ],
    )
    report = ml.bootstrap_config(kinds=["datasets"])
    s = report.suggestions[0]
    assert s.spec_string == 'DatasetSpecConfig(rid="1-AA", version="0.4.0")'


def test_asset_spec_string_is_ready_to_paste() -> None:
    ml = _StubMixin(
        asset_tables=[_StubAssetTable(name="Image")],
        assets_by_table={"Image": [_StubAsset(asset_rid="1-BB", filename="x.png")]},
    )
    report = ml.bootstrap_config(kinds=["assets"])
    s = report.suggestions[0]
    assert s.spec_string == 'AssetSpecConfig(rid="1-BB")'


def test_workflow_spec_string_is_ready_to_paste() -> None:
    ml = _StubMixin(workflows=[_StubWorkflow(workflow_rid="1-CC")])
    report = ml.bootstrap_config(kinds=["workflow"])
    s = report.suggestions[0]
    assert s.spec_string == 'Workflow(workflow_rid="1-CC")'


def test_deriva_ml_spec_string_uses_connection_state() -> None:
    ml = _StubMixin(host_name="example.com", catalog_id="99")
    report = ml.bootstrap_config(kinds=["deriva_ml"])
    s = report.suggestions[0]
    assert 'hostname="example.com"' in s.spec_string
    assert 'catalog_id="99"' in s.spec_string

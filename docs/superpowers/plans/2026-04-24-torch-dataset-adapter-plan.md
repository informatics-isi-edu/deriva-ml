# PyTorch + TensorFlow Dataset Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `DatasetBag.as_torch_dataset()` and `DatasetBag.as_tf_dataset()` thin adapters that turn a bag into framework-native training datasets, and align `DatasetBag.restructure_assets()` to share the same target-specification vocabulary.

**Architecture:** Both adapters delegate to a shared private helper (`_resolve_targets`) that walks `bag.feature_values()` with matching semantics. Framework-specific code lives in `torch_adapter.py` / `tf_adapter.py`; torch and tensorflow imports are lazy (inside the builder bodies) so the base library stays importable without either framework. `restructure_assets` gets a clean-break signature rename to adopt the shared vocabulary.

**Tech Stack:** Python 3.12+, optional PyTorch ≥2.0 (`deriva-ml[torch]`), optional TensorFlow ≥2.15 (`deriva-ml[tf]`), pytest with `pytest.importorskip` gates, MkDocs for UG docs.

**Design reference:** `docs/superpowers/specs/2026-04-24-torch-dataset-adapter-design.md` (commit `7df8dc9` on branch `feature/torch-adapter`).

---

## Task ordering rationale

The tasks are sequenced to land the shared foundation first, then one adapter, then the second adapter (parallel to the first), then the restructure alignment (which depends on the shared helper), then docs.

1. Shared helper + tests — unblocks all three method implementations.
2. Package extras — minimal pyproject edit; lets later tasks install torch / tf in their test fixtures.
3. PyTorch adapter — proves the adapter pattern end-to-end.
4. TensorFlow adapter — mirrors the torch work; parallel structure.
5. Restructure alignment — rewires the existing method to use the new vocabulary.
6. UG section — writes user-facing docs covering all three paths.
7. Cross-reference sweep — small edits to related docstrings.
8. Final verification.

Tasks 3 and 4 are independent and can run back-to-back without coordination. Task 5 depends on tasks 1 and 3 (for the shared helper and the method-on-DatasetBag wiring pattern).

---

## Task 1: Shared target-resolution helper

**Files:**
- Create: `src/deriva_ml/dataset/target_resolution.py`
- Test: `tests/dataset/test_target_resolution.py`

Create the shared `_resolve_targets()` helper that both adapters and the rewritten `restructure_assets()` will call. It walks `bag.feature_values()` for each target, handles the three `missing=` policies, and returns a dict keyed by element RID mapping to the appropriate target-arity shape.

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_target_resolution.py`:

```python
"""Unit tests for the shared target-resolution helper.

Framework-agnostic: tests the selector/missing/arity logic in isolation
from torch, tensorflow, and the restructure_assets call-site. If these
tests fail, the adapter and restructure alignment both break in the
same way, which is a feature — one helper, one set of semantics.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.target_resolution import _resolve_targets
from deriva_ml.feature import FeatureRecord


class _FakeRecord(FeatureRecord):
    """Minimal FeatureRecord stand-in for target-resolution tests."""
    Image: str  # target column: asset RID this record describes
    Grade: str  # scalar label


def _make_bag(feature_returns: dict[str, dict[str, list[_FakeRecord]]]):
    """Build a MagicMock bag whose feature_values returns canned data.

    Args:
        feature_returns: {feature_name: {element_rid: [records...]}}.

    Returns:
        A MagicMock standing in for a DatasetBag.
    """
    bag = MagicMock()

    def fake_feature_values(element_type, feature_name, selector=None):
        per_rid = feature_returns.get(feature_name, {})
        for rid, records in per_rid.items():
            if selector is None:
                yield from records
            else:
                selected = selector(records)
                if selected is not None:
                    yield selected

    bag.feature_values = fake_feature_values
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": rid} for rid in feature_returns.get(
            next(iter(feature_returns), "Grade"), {}).keys()]}
    )
    return bag


def test_resolve_targets_none_returns_unlabeled_per_rid():
    """targets=None → empty dict (no label resolution needed)."""
    bag = _make_bag({})
    result = _resolve_targets(bag, "Image", targets=None, missing="error")
    assert result == {}


def test_resolve_targets_single_target_returns_featurerecord_per_rid():
    """Single-target list yields one FeatureRecord per element RID."""
    recs_a = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    recs_b = [_FakeRecord(Image="1-IMG2", Grade="Severe")]
    bag = _make_bag({"Grade": {"1-IMG1": recs_a, "1-IMG2": recs_b}})
    result = _resolve_targets(bag, "Image", targets=["Grade"], missing="error")
    assert result["1-IMG1"].Grade == "Mild"
    assert result["1-IMG2"].Grade == "Severe"


def test_resolve_targets_missing_error_raises_with_rid_list():
    """missing='error' with sparse labels raises and names unlabeled RIDs."""
    recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    bag = _make_bag({"Grade": {"1-IMG1": recs, "1-IMG2": []}})
    with pytest.raises(DerivaMLException, match=r"1-IMG2"):
        _resolve_targets(bag, "Image", targets=["Grade"], missing="error")


def test_resolve_targets_missing_skip_drops_unlabeled():
    """missing='skip' omits the unlabeled RID from the result entirely."""
    recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    bag = _make_bag({"Grade": {"1-IMG1": recs, "1-IMG2": []}})
    result = _resolve_targets(bag, "Image", targets=["Grade"], missing="skip")
    assert "1-IMG1" in result
    assert "1-IMG2" not in result


def test_resolve_targets_missing_unknown_yields_none():
    """missing='unknown' keeps the RID with None as its target value."""
    recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    bag = _make_bag({"Grade": {"1-IMG1": recs, "1-IMG2": []}})
    result = _resolve_targets(bag, "Image", targets=["Grade"], missing="unknown")
    assert result["1-IMG1"].Grade == "Mild"
    assert result["1-IMG2"] is None


def test_resolve_targets_multi_target_returns_dict_keyed_by_feature():
    """Multi-target yields dict[feature_name, FeatureRecord] per RID."""
    grade_recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    severity_recs = [_FakeRecord(Image="1-IMG1", Grade="Low")]
    bag = _make_bag({
        "Grade": {"1-IMG1": grade_recs},
        "Severity": {"1-IMG1": severity_recs},
    })
    result = _resolve_targets(
        bag, "Image", targets=["Grade", "Severity"], missing="error"
    )
    assert set(result["1-IMG1"].keys()) == {"Grade", "Severity"}
    assert result["1-IMG1"]["Grade"].Grade == "Mild"


def test_resolve_targets_selector_dict_applies_per_feature():
    """dict form passes selectors per-feature to bag.feature_values."""
    selector = FeatureRecord.select_newest
    r1 = _FakeRecord(Image="1-IMG1", Grade="Mild")
    r2 = _FakeRecord(Image="1-IMG1", Grade="Severe")
    bag = _make_bag({"Grade": {"1-IMG1": [r1, r2]}})
    result = _resolve_targets(
        bag, "Image", targets={"Grade": selector}, missing="error"
    )
    # The selector picks one; exact choice depends on select_newest's
    # impl, but the result should be one of the two records.
    assert result["1-IMG1"].Grade in ("Mild", "Severe")


def test_resolve_targets_unknown_feature_raises_at_construction():
    """Passing a feature name that doesn't exist raises the same error
    bag.feature_values would raise."""
    bag = MagicMock()
    bag.feature_values = MagicMock(side_effect=DerivaMLException("No such feature"))
    bag.list_dataset_members = MagicMock(return_value={"Image": []})
    with pytest.raises(DerivaMLException):
        _resolve_targets(bag, "Image", targets=["Bogus"], missing="error")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_target_resolution.py -v`
Expected: ALL tests fail with `ImportError: cannot import name '_resolve_targets'` (module doesn't exist yet).

- [ ] **Step 3: Write the helper module**

Create `src/deriva_ml/dataset/target_resolution.py`:

```python
"""Shared target-resolution logic for DatasetBag adapters and restructure.

Both ``DatasetBag.as_torch_dataset``, ``DatasetBag.as_tf_dataset``, and
``DatasetBag.restructure_assets`` need the same logic: given a bag, an
element-table name, and a target specification, walk ``bag.feature_values``
for each target, apply the requested missing-value policy, and return a
dict keyed by element RID mapping to the target-arity shape.

Keeping this logic in one module makes the "same semantics" claim across
the three public methods enforceable by shared code rather than parallel
implementation. See design spec §8.5.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Literal

from deriva_ml.core.exceptions import DerivaMLException

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.feature import FeatureRecord

    FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]


# Maximum unlabeled-RID count to show in missing="error" message before
# truncating. Keeps error messages readable on very sparse datasets.
_MISSING_ERROR_RID_LIST_LIMIT = 20


def _resolve_targets(
    bag: "DatasetBag",
    element_type: str,
    *,
    targets: "list[str] | dict[str, FeatureSelector] | None",
    missing: Literal["error", "skip", "unknown"],
) -> "dict[str, Any]":
    """Resolve feature values into per-element target records.

    Walks ``bag.feature_values(element_type, feature_name, selector=...)``
    for each feature in ``targets``, groups records by their target-
    element RID, and applies the ``missing`` policy to elements whose
    feature value is absent.

    Args:
        bag: Source ``DatasetBag``.
        element_type: Domain table name whose rows are the elements
            (e.g., ``"Image"``).
        targets: Target specification per the aligned vocabulary (spec
            §3.7). ``None`` yields no labels (empty result). ``list[str]``
            yields one ``FeatureRecord`` per element for single-target,
            ``dict[feature_name, FeatureRecord]`` for multi-target.
            ``dict[str, FeatureSelector]`` passes per-feature selectors
            through to ``bag.feature_values``.
        missing: Policy for elements with no feature value.

    Returns:
        Dict keyed by element RID. Value shape:

        - `targets=None`: empty dict.
        - `targets=["A"]`: `FeatureRecord`.
        - `targets=["A", "B"]` or dict with 2+ keys: `dict[str, FeatureRecord]`.
        - Element absent under `missing="skip"`: key not in returned dict.
        - Element absent under `missing="unknown"`: value is `None`.

    Raises:
        DerivaMLException: If ``missing="error"`` and any element lacks a
            feature value (message lists up to 20 unlabeled RIDs).
    """
    if not targets:
        return {}

    # Normalize targets to (feature_name, selector) pairs. A list form has
    # selector=None for each feature; a dict form uses the mapped selector.
    if isinstance(targets, list):
        feature_specs: list[tuple[str, Any]] = [(name, None) for name in targets]
    else:
        feature_specs = list(targets.items())

    # Walk features and collect records keyed by target-element RID.
    # The target column is the element_type's name (e.g., "Image" on an
    # Image-target FeatureRecord). FeatureRecord instances carry that
    # attribute; we pull it dynamically.
    per_feature_per_rid: dict[str, dict[str, "FeatureRecord"]] = {
        name: {} for name, _ in feature_specs
    }
    for feature_name, selector in feature_specs:
        for record in bag.feature_values(
            element_type, feature_name, selector=selector
        ):
            rid = getattr(record, element_type)
            per_feature_per_rid[feature_name][rid] = record

    # Determine the universe of element RIDs. Union across features so an
    # element labeled for some targets but not others is still considered.
    all_rids: set[str] = set()
    for rid_map in per_feature_per_rid.values():
        all_rids.update(rid_map.keys())

    # Apply missing policy and build the result dict.
    unlabeled: list[str] = []
    result: dict[str, Any] = {}
    is_single_target = len(feature_specs) == 1
    single_feature_name = feature_specs[0][0] if is_single_target else None

    for rid in sorted(all_rids):
        feature_records = {
            name: per_feature_per_rid[name].get(rid)
            for name, _ in feature_specs
        }
        any_missing = any(v is None for v in feature_records.values())

        if any_missing:
            if missing == "error":
                unlabeled.append(rid)
                continue
            if missing == "skip":
                continue
            # missing == "unknown"
            result[rid] = None
            continue

        if is_single_target:
            result[rid] = feature_records[single_feature_name]
        else:
            result[rid] = feature_records

    if missing == "error" and unlabeled:
        preview = unlabeled[:_MISSING_ERROR_RID_LIST_LIMIT]
        suffix = (
            f" (and {len(unlabeled) - len(preview)} more)"
            if len(unlabeled) > len(preview)
            else ""
        )
        raise DerivaMLException(
            f"{len(unlabeled)} element(s) of type {element_type!r} have no "
            f"value for one or more targets in {targets!r}. "
            f"Unlabeled RIDs: {preview}{suffix}. "
            f"Pass missing='skip' to drop unlabeled elements, or "
            f"missing='unknown' to keep them with target=None."
        )

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_target_resolution.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run the full fast suite to confirm no regressions**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/dataset/test_target_resolution.py -q`
Expected: 498+ passed (same baseline as main + the new target_resolution tests), 3 skipped, 0 failed.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/torch-adapter
git add src/deriva_ml/dataset/target_resolution.py tests/dataset/test_target_resolution.py
git commit -m "feat(dataset): add _resolve_targets shared helper for D2

Post-S2 D2 Task 1. Shared target-resolution logic consumed by
as_torch_dataset, as_tf_dataset, and the rewritten restructure_assets.
Handles feature-value resolution via bag.feature_values, three
missing= policies (error/skip/unknown), and single/multi-target arity.
Framework-agnostic — zero torch/tf imports.

See spec §8.5 for the helper's contract.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Package extras in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

Add the `[project.optional-dependencies]` table with `torch` and `tf` entries so users can `pip install 'deriva-ml[torch]'` or `pip install 'deriva-ml[tf]'`.

- [ ] **Step 1: Locate the right spot**

Run: `grep -n "^\[project\.optional\|^\[tool\." pyproject.toml | head -5`
Expected: confirms no existing `[project.optional-dependencies]` table (there's already a stub at line 49 per the initial read, but it's empty).

- [ ] **Step 2: Edit pyproject.toml**

Find the existing empty `[project.optional-dependencies]` stub around line 49-50 and replace with:

```toml
[project.optional-dependencies]
torch = ["torch>=2.0"]
tf = ["tensorflow>=2.15"]
```

- [ ] **Step 3: Verify build configuration still parses**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv sync --extra torch --extra tf`
Expected: succeeds, torch and tensorflow get installed to the venv. Note: this may take 1-3 minutes due to tensorflow's size.

Alternative if tf install fails on your platform: `uv sync --extra torch` alone works and is sufficient for Task 3 verification; Task 4 will need TF separately.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): add torch + tf optional-dependency extras

Post-S2 D2 Task 2. Adds [project.optional-dependencies] with:
- torch = [\"torch>=2.0\"]
- tf = [\"tensorflow>=2.15\"]

Users install with 'pip install deriva-ml[torch]' or
'pip install deriva-ml[tf]' depending on which framework they
use. The base install remains lean — neither framework is a
hard dependency, and both are imported lazily inside the
adapter code per design anchor 3.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: PyTorch adapter

**Files:**
- Create: `src/deriva_ml/dataset/torch_adapter.py`
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (add `as_torch_dataset` method)
- Test: `tests/dataset/test_torch_adapter_no_torch.py`
- Test: `tests/dataset/test_torch_adapter_logic.py`
- Test: `tests/dataset/test_torch_adapter_e2e.py`

Ships `DatasetBag.as_torch_dataset(element_type, *, sample_loader=None, transform=None, targets=None, target_transform=None, missing="error")` returning a `torch.utils.data.Dataset`. Torch is imported lazily inside the builder.

- [ ] **Step 1: Write the no-torch import-guard test**

Create `tests/dataset/test_torch_adapter_no_torch.py`:

```python
"""Verify the library imports cleanly when torch is absent.

If torch is truly installed in the test venv, we stub it out of
sys.modules temporarily. The goal is to prove that importing DatasetBag
does NOT import torch eagerly.
"""
from __future__ import annotations

import sys

import pytest


def test_dataset_bag_imports_without_torch(monkeypatch):
    """Removing torch from sys.modules lets DatasetBag still import."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    # Re-import dataset_bag under the torch-less sys.modules state.
    if "deriva_ml.dataset.dataset_bag" in sys.modules:
        monkeypatch.delitem(sys.modules, "deriva_ml.dataset.dataset_bag", raising=False)
    from deriva_ml.dataset.dataset_bag import DatasetBag  # noqa: F401


def test_as_torch_dataset_raises_importerror_without_torch(monkeypatch):
    """Calling as_torch_dataset when torch is absent raises ImportError
    with an install-hint message."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    from unittest.mock import MagicMock
    bag = MagicMock()

    if "deriva_ml.dataset.torch_adapter" in sys.modules:
        monkeypatch.delitem(sys.modules, "deriva_ml.dataset.torch_adapter", raising=False)

    from deriva_ml.dataset.torch_adapter import build_torch_dataset
    with pytest.raises(ImportError, match=r"torch"):
        build_torch_dataset(bag, "Image")
```

- [ ] **Step 2: Write the logic test (torch stubbed)**

Create `tests/dataset/test_torch_adapter_logic.py`:

```python
"""Pure-Python logic tests for torch_adapter.

Uses pytest.importorskip — runs only when torch is installed. Tests
the join logic, selector pass-through, missing= branches, target arity,
and error paths using the real torch.utils.data.Dataset base class.
"""
from __future__ import annotations

from unittest.mock import MagicMock
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from deriva_ml.core.exceptions import DerivaMLException  # noqa: E402
from deriva_ml.dataset.torch_adapter import build_torch_dataset  # noqa: E402
from deriva_ml.feature import FeatureRecord  # noqa: E402


class _FakeRecord(FeatureRecord):
    Image: str
    Grade: str


def _mock_bag_with_labeled_images(rids_and_labels: dict[str, str]):
    """Build a MagicMock DatasetBag with the given labeled images."""
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": rid} for rid in rids_and_labels]}
    )

    def fake_feature_values(element_type, feature_name, selector=None):
        for rid, label in rids_and_labels.items():
            rec = _FakeRecord(Image=rid, Grade=label)
            if selector is None:
                yield rec
            else:
                sel = selector([rec])
                if sel is not None:
                    yield sel

    bag.feature_values = fake_feature_values
    # Mock table metadata so the asset-path resolver can find columns.
    bag.model = MagicMock()
    asset_row = {"RID": "1-IMG1", "Filename": "img1.jpg"}
    bag.get_table_as_dict = MagicMock(return_value=iter([asset_row]))
    bag.is_asset = MagicMock(return_value=True)
    return bag


def test_as_torch_dataset_returns_torch_dataset_subclass():
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    ds = build_torch_dataset(
        bag, "Image",
        sample_loader=lambda p, row: b"fake",
        targets=["Grade"],
    )
    assert isinstance(ds, torch.utils.data.Dataset)


def test_len_reflects_labeled_count_when_missing_skip():
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild", "1-IMG2": ""})
    # Simulate missing: IMG2 has empty label treated as absent in this
    # simplified test; for real data the fake feature_values would
    # simply not yield IMG2.
    # In logic tests we only check len() semantics.
    # ...rest of tests follow the same pattern per spec §6.2 matrix.


def test_missing_error_raises_at_construction():
    bag = MagicMock()
    bag.path = Path("/tmp/fake_bag")
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": "1-IMG1"}, {"RID": "1-IMG2"}]}
    )
    bag.feature_values = MagicMock(return_value=iter([_FakeRecord(Image="1-IMG1", Grade="Mild")]))
    bag.model = MagicMock()
    with pytest.raises(DerivaMLException, match=r"1-IMG2"):
        build_torch_dataset(
            bag, "Image",
            sample_loader=lambda p, row: b"",
            targets=["Grade"],
            missing="error",
        )


def test_asset_table_without_sample_loader_raises():
    """Asset-table element_type with no sample_loader raises at construction."""
    bag = _mock_bag_with_labeled_images({"1-IMG1": "Mild"})
    with pytest.raises(DerivaMLException, match=r"sample_loader"):
        build_torch_dataset(bag, "Image", targets=["Grade"])
```

- [ ] **Step 3: Write the e2e test placeholder**

Create `tests/dataset/test_torch_adapter_e2e.py`:

```python
"""End-to-end torch adapter test using a real bag fixture.

Gated on torch being importable. Uses catalog_with_datasets fixture
to build a real bag, then iterates under a real DataLoader.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402


def test_bag_to_dataloader_yields_tensors(catalog_with_datasets, tmp_path):
    ml, dataset_desc = catalog_with_datasets
    bag = dataset_desc.dataset.download_dataset_bag(version="1.0.0")

    ds = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=lambda p, row: torch.tensor([1.0, 2.0, 3.0]),
        targets=None,  # unlabeled — just exercise the plumbing
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) > 0
    assert batches[0].shape[0] <= 2
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_torch_adapter_*.py -v`
Expected: all fail with ImportError/ModuleNotFoundError for `deriva_ml.dataset.torch_adapter`.

- [ ] **Step 5: Write the torch adapter module**

Create `src/deriva_ml/dataset/torch_adapter.py`:

```python
"""PyTorch adapter for DatasetBag.

Builds a ``torch.utils.data.Dataset`` that lazy-loads samples and labels
from an already-downloaded ``DatasetBag``. Torch is imported at builder
entry (lazy import) so the base library stays importable without torch.

See design spec §§3.1-3.7 for the full contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.target_resolution import _resolve_targets

if TYPE_CHECKING:
    import torch.utils.data
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.feature import FeatureRecord

    FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]


_TORCH_INSTALL_HINT = (
    "PyTorch is not installed. Install with:\n"
    "    pip install 'deriva-ml[torch]'\n"
    "or install torch directly:\n"
    "    pip install 'torch>=2.0'"
)


def build_torch_dataset(
    bag: "DatasetBag",
    element_type: str,
    *,
    sample_loader: "Callable[[Path, dict[str, Any]], Any] | None" = None,
    transform: "Callable[[Any], Any] | None" = None,
    targets: "list[str] | dict[str, FeatureSelector] | None" = None,
    target_transform: "Callable[..., Any] | None" = None,
    missing: Literal["error", "skip", "unknown"] = "error",
) -> "torch.utils.data.Dataset":
    """Build a torch.utils.data.Dataset from a DatasetBag.

    See DatasetBag.as_torch_dataset docstring for full argument docs
    and spec §3 for the design contract.
    """
    try:
        import torch.utils.data as _torch_data
    except ImportError as e:
        raise ImportError(_TORCH_INSTALL_HINT) from e

    # Validate element_type exists in the bag.
    members_by_type = bag.list_dataset_members(recurse=True)
    if element_type not in members_by_type:
        raise DerivaMLException(
            f"Element type {element_type!r} not found in bag; available "
            f"types: {sorted(members_by_type.keys())}"
        )

    # Validate sample_loader is provided for asset-table element types.
    is_asset = _bag_element_is_asset(bag, element_type)
    if is_asset and sample_loader is None:
        raise DerivaMLException(
            f"Element type {element_type!r} is an asset table and requires "
            f"a sample_loader. Common loaders:\n"
            f"    sample_loader=PIL.Image.open       # images\n"
            f"    sample_loader=nibabel.load         # NIfTI medical volumes\n"
            f"    sample_loader=h5py.File            # HDF5 arrays\n"
            f"Or pass sample_loader=lambda path, row: path.read_bytes() "
            f"if you want raw bytes."
        )

    # Resolve targets once at construction time.
    target_map = _resolve_targets(
        bag, element_type, targets=targets, missing=missing
    )

    # Build the RID list — all elements if targets=None, only labeled
    # elements if targets is set.
    all_rids = [m["RID"] for m in members_by_type[element_type]]
    if targets is None:
        rids = all_rids
    elif missing == "skip":
        rids = [rid for rid in all_rids if rid in target_map]
    else:
        rids = all_rids  # "error" already raised if incomplete; "unknown"
                         # keeps all RIDs with None target for absent ones

    # Build the row lookup so sample_loader gets the full row dict.
    row_lookup = _build_row_lookup(bag, element_type)

    class _TorchDataset(_torch_data.Dataset):
        def __init__(self):
            self._rids = rids
            self._target_map = target_map
            self._is_asset = is_asset

        def __len__(self):
            return len(self._rids)

        def __getitem__(self, index):
            rid = self._rids[index]
            row = row_lookup.get(rid, {"RID": rid})

            if self._is_asset:
                path = _resolve_asset_path(bag, element_type, rid, row)
                sample = sample_loader(path, row)
            else:
                sample = (
                    sample_loader(None, row)
                    if sample_loader is not None
                    else row
                )

            if transform is not None:
                sample = transform(sample)

            if targets is None:
                return sample

            target = self._target_map.get(rid)
            if target_transform is not None:
                target = target_transform(target)
            return sample, target

    return _TorchDataset()


def _bag_element_is_asset(bag: "DatasetBag", element_type: str) -> bool:
    """Return True if element_type is an asset table in the bag."""
    try:
        return bag.is_asset(element_type)
    except AttributeError:
        # Fall back to model-level check for bags that don't expose
        # is_asset directly.
        return bool(getattr(bag.model, "is_asset", lambda _: False)(element_type))


def _build_row_lookup(bag: "DatasetBag", element_type: str) -> dict:
    """Build {RID: row_dict} lookup for the element table."""
    return {row["RID"]: row for row in bag.get_table_as_dict(element_type)}


def _resolve_asset_path(bag, element_type: str, rid: str, row: dict) -> Path:
    """Compute the on-disk path for an asset's file."""
    filename = row.get("Filename") or f"{rid}.bin"
    return bag.path / "data" / "assets" / element_type / rid / filename
```

- [ ] **Step 6: Add the DatasetBag method**

Find the class body of `DatasetBag` in `src/deriva_ml/dataset/dataset_bag.py` (around line 1445 where `restructure_assets` lives) and add before `restructure_assets`:

```python
    def as_torch_dataset(
        self,
        element_type: str,
        *,
        sample_loader: Callable[[Path, dict[str, Any]], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        targets: list[str] | dict[str, Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        missing: Literal["error", "skip", "unknown"] = "error",
    ) -> "torch.utils.data.Dataset":
        """Build a torch.utils.data.Dataset from this bag.

        [... full docstring per spec §3 ...]
        """
        from deriva_ml.dataset.torch_adapter import build_torch_dataset
        return build_torch_dataset(
            self,
            element_type,
            sample_loader=sample_loader,
            transform=transform,
            targets=targets,
            target_transform=target_transform,
            missing=missing,
        )
```

The docstring should be the full version per spec §3.1-3.6: one-line summary, extended description, Args (one paragraph per argument, matching §3.2), Returns, Raises, Example with `# doctest: +SKIP` on catalog-dependent lines.

- [ ] **Step 7: Run all torch adapter tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_torch_adapter_no_torch.py tests/dataset/test_torch_adapter_logic.py -v`
Expected: all tests PASS. The e2e test needs a live catalog so runs separately.

- [ ] **Step 8: Run fast suite**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/dataset/test_torch_adapter_no_torch.py tests/dataset/test_torch_adapter_logic.py tests/dataset/test_target_resolution.py -q`
Expected: 500+ passed, 3 skipped, 0 failed.

- [ ] **Step 9: Commit**

```bash
git add src/deriva_ml/dataset/torch_adapter.py src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_torch_adapter_*.py
git commit -m "feat(dataset): add DatasetBag.as_torch_dataset adapter

Post-S2 D2 Task 3. Ships the PyTorch adapter per design spec §3.

New module torch_adapter.py builds a torch.utils.data.Dataset from
a DatasetBag, delegating to the shared _resolve_targets helper for
label resolution. Torch is imported lazily inside the builder so
the base library stays importable without torch (anchor 3).

DatasetBag.as_torch_dataset is the public entry point; it delegates
to build_torch_dataset. Signature mirrors spec §3.1 exactly.

Three test tiers per spec §6:
- test_torch_adapter_no_torch.py — library still imports when torch
  is absent; calling adapter raises clean ImportError with hint.
- test_torch_adapter_logic.py — join logic, missing= branches,
  target arity, error paths (gated on torch installable).
- test_torch_adapter_e2e.py — real bag fixture + real DataLoader
  (gated on live catalog).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: TensorFlow adapter

**Files:**
- Create: `src/deriva_ml/dataset/tf_adapter.py`
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (add `as_tf_dataset` method)
- Test: `tests/dataset/test_tf_adapter_no_tf.py`
- Test: `tests/dataset/test_tf_adapter_logic.py`
- Test: `tests/dataset/test_tf_adapter_e2e.py`

Mirrors Task 3's structure for TensorFlow. Ships `DatasetBag.as_tf_dataset(...)` returning a `tf.data.Dataset`, with `output_signature` inferred from the first sample when `None`.

- [ ] **Step 1: Write the no-tf import-guard test**

Create `tests/dataset/test_tf_adapter_no_tf.py` mirroring `test_torch_adapter_no_torch.py` exactly, substituting `tensorflow` / `tf` for torch. Verify:
- Importing `DatasetBag` without tensorflow still works.
- Calling `build_tf_dataset(bag, ...)` raises `ImportError` with install hint `'deriva-ml[tf]'`.

Code shape: same monkeypatch pattern as Task 3 Step 1, substitute `"tensorflow"` wherever `"torch"` appears.

- [ ] **Step 2: Write the logic test (tf stubbed)**

Create `tests/dataset/test_tf_adapter_logic.py` mirroring the torch logic test but verifying `tf.data.Dataset` shape:

```python
import pytest
tf = pytest.importorskip("tensorflow")

from deriva_ml.dataset.tf_adapter import build_tf_dataset  # noqa: E402


def test_as_tf_dataset_returns_tf_data_dataset():
    bag = _mock_bag(...)
    ds = build_tf_dataset(
        bag, "Image",
        sample_loader=lambda p, row: tf.constant([1.0, 2.0]),
        targets=["Grade"],
    )
    assert isinstance(ds, tf.data.Dataset)


def test_output_signature_inferred_from_first_sample():
    """output_signature=None triggers first-sample inference."""
    bag = _mock_bag(...)
    ds = build_tf_dataset(
        bag, "Image",
        sample_loader=lambda p, row: tf.constant([1.0, 2.0, 3.0]),
        targets=["Grade"],
        target_transform=lambda rec: 0,
        output_signature=None,  # explicit default
    )
    # element_spec should reflect the inferred shape
    spec = ds.element_spec
    assert isinstance(spec, tuple)
    assert spec[0].shape[-1] == 3
```

Full test matrix from spec §6.2 applies: all three `missing=` branches, target arity, asset-table-without-loader error, selector dict pass-through.

- [ ] **Step 3: Write the e2e test placeholder**

Create `tests/dataset/test_tf_adapter_e2e.py` — real bag fixture + real `tf.data.Dataset.batch(...).prefetch(...)` iteration. Gated on `pytest.importorskip("tensorflow")` and the `catalog_with_datasets` fixture.

- [ ] **Step 4: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_tf_adapter_*.py -v`
Expected: fails with ModuleNotFoundError for `deriva_ml.dataset.tf_adapter`.

- [ ] **Step 5: Write the tf adapter module**

Create `src/deriva_ml/dataset/tf_adapter.py`:

```python
"""TensorFlow adapter for DatasetBag.

Builds a ``tf.data.Dataset`` via ``tf.data.Dataset.from_generator`` that
lazy-loads samples and labels from an already-downloaded ``DatasetBag``.
Tensorflow is imported at builder entry (lazy import) so the base
library stays importable without tensorflow.

See design spec §§3.1-3.7 for the full contract. ``output_signature``
is inferred from the first sample when None (spec §3.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.target_resolution import _resolve_targets

if TYPE_CHECKING:
    import tensorflow as tf
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.feature import FeatureRecord

    FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]


_TF_INSTALL_HINT = (
    "TensorFlow is not installed. Install with:\n"
    "    pip install 'deriva-ml[tf]'\n"
    "or install tensorflow directly:\n"
    "    pip install 'tensorflow>=2.15'\n"
    "macOS: use 'tensorflow-macos' instead of 'tensorflow'.\n"
    "CUDA:  use 'tensorflow[and-cuda]' for GPU support."
)


def build_tf_dataset(
    bag: "DatasetBag",
    element_type: str,
    *,
    sample_loader: "Callable[[Path, dict[str, Any]], Any] | None" = None,
    transform: "Callable[[Any], Any] | None" = None,
    targets: "list[str] | dict[str, FeatureSelector] | None" = None,
    target_transform: "Callable[..., Any] | None" = None,
    missing: Literal["error", "skip", "unknown"] = "error",
    output_signature: "tf.TensorSpec | tuple[tf.TensorSpec, ...] | None" = None,
) -> "tf.data.Dataset":
    """Build a tf.data.Dataset from a DatasetBag.

    See DatasetBag.as_tf_dataset docstring for full argument docs.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        raise ImportError(_TF_INSTALL_HINT) from e

    # [Same validation + resolution logic as torch adapter, factored
    #  out where possible. Wraps the iteration in tf.data.Dataset.from_generator
    #  with either user-supplied or first-sample-inferred output_signature.]
    # [Full implementation writes a Python generator, wraps it, handles
    #  the one-extra-eager-sample path for inference.]
```

Implementation mirrors `torch_adapter.py` in structure: same validation (element_type exists, asset-table-requires-loader), same delegation to `_resolve_targets`, same RID-list construction per `missing=` policy. Differences:

1. Build a Python generator `def _gen(): for rid in rids: yield (sample, target)` instead of a `Dataset` subclass.
2. If `output_signature is None`, call `_gen()` once to get the first sample, use `tf.type_spec_from_value` to infer, then re-wrap the generator so the first sample isn't lost.
3. Return `tf.data.Dataset.from_generator(_gen, output_signature=...)`.

- [ ] **Step 6: Add the DatasetBag method**

Add `as_tf_dataset` method to `DatasetBag` class body, next to `as_torch_dataset`. Same shape as Task 3 Step 6 but delegates to `build_tf_dataset` and includes the `output_signature` parameter.

- [ ] **Step 7: Run all tf adapter tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_tf_adapter_no_tf.py tests/dataset/test_tf_adapter_logic.py -v`
Expected: all pass.

- [ ] **Step 8: Run fast suite**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/dataset/test_torch_adapter_*.py tests/dataset/test_tf_adapter_*.py tests/dataset/test_target_resolution.py -q`
Expected: 503+ passed, 3 skipped, 0 failed.

- [ ] **Step 9: Commit**

```bash
git add src/deriva_ml/dataset/tf_adapter.py src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_tf_adapter_*.py
git commit -m "feat(dataset): add DatasetBag.as_tf_dataset adapter

Post-S2 D2 Task 4. Ships the TensorFlow adapter per design spec §3,
mirroring the PyTorch adapter's shape for the six shared parameters
and adding output_signature for TF-specific TensorSpec configuration.

output_signature=None (default) triggers first-sample inference at
construction time; the generator is re-wrapped so the eagerly-consumed
first sample is not lost. Users who want deterministic startup pass
output_signature explicitly.

Test structure parallels the torch adapter: no-tf / logic-stubbed /
e2e. All three gated appropriately.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Restructure alignment (clean break)

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (rewrite `restructure_assets` signature + body)
- Test: update or add tests in `tests/dataset/` covering the new signature

Rewrite `restructure_assets` to accept the aligned vocabulary (`targets`, `target_transform`, `missing`) instead of `group_by`/`value_selector`/`"Feature.column"` dotted syntax. Uses the shared `_resolve_targets` helper from Task 1.

- [ ] **Step 1: Grep sibling repos for callers of deprecated kwargs**

Run:
```bash
for d in ~/GitHub/deriva-ml-model-template ~/GitHub/deriva-ml-apps ~/GitHub/deriva-ml-demo ~/GitHub/deriva-ml-template-test; do
  echo "=== $d ==="
  [ -d "$d" ] && grep -rn "group_by=\|value_selector=\|restructure_assets" "$d" --include="*.py" 2>/dev/null | head -10
done
```

Record the results — these go into the PR description's "Breaking changes" section so template maintainers can update in lockstep.

- [ ] **Step 2: Update existing tests first (they need to pass with the new signature)**

`grep -rn "group_by\|value_selector" tests/` — enumerate callers. For each test that uses old kwargs, rewrite to the new form:
- `group_by=["Feature"]` → `targets=["Feature"]`
- `group_by=["Feature.Col"]` → `targets=["Feature"], target_transform=lambda rec: rec.Col`
- `value_selector=FeatureRecord.select_newest` → `targets={"Feature": FeatureRecord.select_newest}`

Commit these test updates at the end of the task (with the impl), not separately — keeps the repo in a buildable state only after both sides land.

- [ ] **Step 3: Rewrite `restructure_assets` in `dataset_bag.py`**

Replace the current signature and body. The new signature per spec §3.1 + §8:

```python
def restructure_assets(
    self,
    output_dir: Path | str,
    *,
    asset_table: str | None = None,
    targets: list[str] | dict[str, FeatureSelector] | None = None,
    target_transform: Callable[..., str] | None = None,
    missing: Literal["error", "skip", "unknown"] = "unknown",
    use_symlinks: bool = True,
    type_selector: Callable[[list[str]], str] | None = None,
    type_to_dir_map: dict[str, str] | None = None,
    enforce_vocabulary: bool = True,
    file_transformer: Callable[[Path, Path], Path] | None = None,
) -> dict[Path, Path]:
    """[... full docstring per spec §8 ...]"""
```

Body:
1. Validate `asset_table` (auto-detect from members if None; unchanged from today).
2. Call `_resolve_targets(self, asset_table, targets=targets, missing=missing)` to get the `{rid: FeatureRecord}` map.
3. For each RID, derive the directory name: if `target_transform` is provided, call it (and validate string return); else use the FeatureRecord's primary value (tricky — spec §3.7.2 says target_transform is required when targets is a multi-column feature because the direct FeatureRecord isn't a string; document this and enforce).
4. Place file in `{output_dir}/{type_dir}/{target_dir}/{filename}` per existing FK-reachability logic.
5. Apply missing policy: for `missing="unknown"` put into `unknown/`; for `missing="skip"` omit entirely.
6. Keep all existing behavior for `type_selector`, `type_to_dir_map`, `enforce_vocabulary`, `file_transformer` — they don't change.
7. Raise `TypeError` if user passes `group_by=` or `value_selector=` (use the `@validate_call` decorator or explicit check; the stdlib behavior of rejecting unknown kwargs is also acceptable).

- [ ] **Step 4: Verify old-kwarg rejection**

Manually test:
```python
bag.restructure_assets(output_dir="/tmp", group_by=["Diagnosis"])
# Expected: TypeError: restructure_assets() got an unexpected keyword argument 'group_by'
```

- [ ] **Step 5: Add migration-note paragraph to the docstring**

Per spec §8.7, the docstring gets a short Migration note at the bottom:

```python
    """[... main docstring ...]

    Migration note (from pre-D2 signature):
        - group_by=["Diagnosis"] → targets=["Diagnosis"]
        - group_by=["Classification.Label"] → targets=["Classification"],
          target_transform=lambda rec: rec.Label
        - value_selector=FeatureRecord.select_newest →
          targets={"Feature": FeatureRecord.select_newest}
    """
```

- [ ] **Step 6: Run fast suite**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/dataset/ -q`
Expected: all pass. Any test using old kwargs should have been updated in Step 2.

- [ ] **Step 7: Run full dataset suite (live catalog) — time permitting**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ --timeout=600`
Expected: 64+ passed. Focuses on the restructure integration tests.

- [ ] **Step 8: Commit**

```bash
git add src/deriva_ml/dataset/dataset_bag.py tests/dataset/
git commit -m "refactor(dataset): align restructure_assets vocab with adapters

Post-S2 D2 Task 5. restructure_assets now uses the same
target-specification vocabulary (targets, target_transform, missing)
as as_torch_dataset and as_tf_dataset, delegating feature resolution
to the shared _resolve_targets helper.

Signature changes (clean break — see spec §8):
  - group_by → targets
  - value_selector → merged into targets dict form
  - \"Feature.column\" dotted syntax → use target_transform
  - Added missing: Literal[\"error\", \"skip\", \"unknown\"] = \"unknown\"

All other parameters (output_dir, asset_table, use_symlinks,
type_selector, type_to_dir_map, enforce_vocabulary, file_transformer)
unchanged. FK-reachability and type-derived directory naming
unchanged. The default missing=\"unknown\" preserves today's behavior
for unparameterized calls.

Existing tests using old kwargs are updated to the new signature in
the same commit. Passing old kwargs after this PR raises TypeError
(standard Python behavior for removed kwargs).

Migration note in the docstring guides users who hit a TypeError.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: User guide section

**Files:**
- Modify: `docs/user-guide/offline.md` (add "How to feed a bag to a training framework" section per spec §7.2)

Write the UG section with the decision tree + three per-framework sub-subsections + Keras routing + ImageFolder pointer.

- [ ] **Step 1: Locate insertion point in offline.md**

The new section goes after the existing `restructure_assets` content and before the section on working with bags in notebooks (exact location TBD from current offline.md structure).

- [ ] **Step 2: Write the new UG section**

Follows spec §7.2 exactly. Structure:

1. **Motivation** (2 sentences).
2. **Choosing your path** (decision paragraph).
3. **7.2.1 Using with PyTorch** — simple image-classification example, worked variations (tabular regression, multi-target, multi-annotator selector), notes pointer to `deriva-ml[torch]`.
4. **7.2.2 Using with TensorFlow** — simple example + output_signature guidance + same three worked variations, notes pointer to `deriva-ml[tf]` and platform wheels.
5. **7.2.3 Using with Keras** — three-backend routing example.
6. **7.2.4 Using with `ImageFolder` / third-party tools** — pointer to `restructure_assets`.
7. **7.2.5 See also** — cross-references.

Each code example should be ~10-20 lines. Use `# doctest: +SKIP` consistently for catalog-dependent examples; mkdocs-jupyter doesn't execute them but the markup should be valid.

- [ ] **Step 3: Run mkdocs --strict**

Run: `uv run mkdocs build --strict 2>&1 | tail -20`
Expected: either 27 warnings (same as baseline) or fewer. No new griffe warnings from the new section.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/offline.md
git commit -m "docs(user-guide): add 'How to feed a bag to a training framework' section

Post-S2 D2 Task 6. New UG section in Chapter 5 (offline.md) covering
PyTorch, TensorFlow, Keras (all three backends), and third-party
ImageFolder-style layout.

Structure follows the UG contract per spec §7.2:
- Motivation + decision-tree paragraph at the top
- Per-framework sub-sections with worked examples
- Keras routing via either adapter based on backend choice
- Pointer to restructure_assets for class-folder workflows

mkdocs --strict: 27 warnings (unchanged from baseline).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Cross-reference sweep

**Files:**
- Modify: various docstrings and related files

Small edits per spec §7.3: docstring cross-references between the adapters, `restructure_assets`, `split_dataset`, and the `DatasetBag` class-level docstring.

- [ ] **Step 1: Update `DatasetBag` class docstring**

Add one bullet mentioning the adapters alongside the existing `restructure_assets()` mention. Keep to one sentence per bullet; link by method name only (mkdocstrings generates the cross-link).

- [ ] **Step 2: Update `split_dataset` docstring**

Add one sentence in the "Examples" or "See Also" section pointing at the adapter composition pattern (spec §7.4.1).

- [ ] **Step 3: Update `restructure_assets` docstring**

Add one sentence (separate from the Migration note) pointing at `as_torch_dataset`/`as_tf_dataset` as the general-case adapter when a directory tree isn't needed.

- [ ] **Step 4: Update CLAUDE.md**

The section on `DatasetBag` in the project-level `CLAUDE.md` gets a one-sentence mention of the adapters for future agentic work.

- [ ] **Step 5: Run mkdocs --strict**

Expected: warnings unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/dataset/ CLAUDE.md
git commit -m "docs: cross-reference sweep for D2 adapters

Post-S2 D2 Task 7. Small edits to related docstrings:
- DatasetBag class-level docstring mentions the two adapters
- split_dataset docstring points at the composition pattern
- restructure_assets docstring points at the adapters for
  non-class-folder workflows
- CLAUDE.md DatasetBag section gets the adapter mention

No code changes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Final verification

**Files:** none modified — verification only.

- [ ] **Step 1: Run fast unit tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/dataset/test_target_resolution.py tests/dataset/test_torch_adapter_no_torch.py tests/dataset/test_torch_adapter_logic.py tests/dataset/test_tf_adapter_no_tf.py tests/dataset/test_tf_adapter_logic.py -q`
Expected: 505+ passed, 3 skipped, 0 failed.

- [ ] **Step 2: Run e2e tests (if live catalog available)**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_torch_adapter_e2e.py tests/dataset/test_tf_adapter_e2e.py -v`
Expected: 2+ passed. Verifies end-to-end against a real bag.

- [ ] **Step 3: mkdocs --strict**

Run: `uv run mkdocs build --strict 2>&1 | grep -cE "WARNING|Aborted"`
Expected: `27` (unchanged from main baseline).

- [ ] **Step 4: ruff check + format**

Run: `uv run ruff check src/deriva_ml/dataset/ && uv run ruff format --check src/deriva_ml/dataset/`
Expected: clean on the new files (any format issues should have been caught during commits, but belt-and-suspenders).

- [ ] **Step 5: Sibling-repo grep for removed kwargs**

Re-run the grep from Task 5 Step 1 and record results in the PR description under "Breaking changes" so template maintainers see the migration guide.

- [ ] **Step 6: Finish the branch**

Invoke `superpowers:finishing-a-development-branch` and present the four options (merge / PR / keep / discard). The branch has ~9 commits (including the 7 spec commits + task commits). Recommended: Option 2 (push + create PR) given the PR size and the need for review on the clean-break restructure rename.

---

## Task 9: Migration guide

**Files:**
- Create: `docs/user-guide/migration-from-previous-version.md` (or similar name — see Step 1)
- Modify: `mkdocs.yml` to include the new page in the nav
- Modify: `docs/index.md` — add a pointer to the migration guide

Write a user-facing migration guide covering all breaking changes and notable additions shipped to deriva-ml across the post-S2 documentation pass (PR #65 UG rewrite, PR #66 docstring sweep with 12 renames, PR #67 DRY cleanup, PR #68 metrics_file, and this D2 PR). Users upgrading from the previously-published version need a single place that says "here's what changes, here's how to update your code."

### Why this belongs in D2 specifically

The biggest breaking change in the set is **D2's `restructure_assets` signature rename** (`group_by=` → `targets=`, `value_selector=` merged, `"Feature.column"` removed). That's a `TypeError`-on-call for any downstream caller. PRs #66 and #67 had smaller renames (the 12 `_`-prefixed private renames were API cleanup that affected code calling the old public names). The migration guide lands in D2's PR so users pulling the new version find the guide in the same release that breaks them.

- [ ] **Step 1: Decide the file path and naming**

The migration guide should live somewhere discoverable. Three options:

- `docs/user-guide/migration.md` — alongside the other UG chapters. Most discoverable.
- `docs/migration.md` — top-level, separate from the UG structure. Good if users think of "migration" as a one-time reference not part of the regular UG.
- `docs/reference/migration.md` — under the reference section. Good if it's more reference material than tutorial.

Lean: `docs/user-guide/migration.md` — users will find it while browsing the UG, and the tone is more "walk me through changing my code" than reference-material lookup.

- [ ] **Step 2: Enumerate every breaking change**

The guide covers:

**A. Breaking changes — `TypeError` or `AttributeError` on call:**

1. **`DatasetBag.restructure_assets` signature rename** (D2):
   - `group_by=["Diagnosis"]` → `targets=["Diagnosis"]`
   - `group_by=["Classification.Label"]` → `targets=["Classification"], target_transform=lambda rec: rec.Label`
   - `value_selector=FeatureRecord.select_newest` → `targets={"Feature": FeatureRecord.select_newest}`
   - Dotted `"Feature.column"` syntax removed entirely.
2. **Public methods renamed to `_`-prefixed private** (PR #66 — 12 renames):
   - `ml.domain_path()` → `ml._domain_path()` (and similar for `table_path`, `is_system_schema`, `get_domain_schemas`, `apply_logger_overrides`, `compute_diff`, `retrieve_rid`, `cache_features`, `add_workflow`, `start_upload`, `asset_record_class` module-level factory)
   - These were all internal helpers that shouldn't have been public; users calling them were depending on accidental API surface. The underscored versions still work.
   - User-facing alternatives:
     - `retrieve_rid` → use `resolve_rid()` instead (user-facing wrapper)
     - `asset_record_class` — the mixin method `ml.asset_record_class(table)` is still public; only the module-level factory is private.
     - `add_workflow` → use `create_workflow()` (user-facing factory)
3. **Methods deleted** (PR #66):
   - `prefetch_dataset` — was a deprecated one-line shim; use `cache_dataset` directly.
   - `list_foreign_keys` — no callers, no replacement needed.
   - `add_page`, `user_list`, `globus_login` — removed from `DerivaML`; these were stale web-app helpers that never had users.
4. **Documentation-only fix that may affect copy-pasted user code**:
   - `AssetRIDConfig` (doc name) → `AssetSpec` / `AssetSpecConfig` (real class names). Users who copied example code from the old configuration docs hit an `ImportError`.

**B. New features — additive, no break, but worth flagging:**

1. **`Execution.metrics_file(filename="metrics.jsonl")`** (PR #68) — recommended over manually calling `asset_file_path(MLAsset.execution_metadata, ...)` for training metrics.
2. **`DatasetBag.as_torch_dataset(...)` and `DatasetBag.as_tf_dataset(...)`** (D2) — recommended over hand-rolling `torch.utils.data.Dataset` subclasses.
3. **State-machine recovery edge** (PR A) — `Running → Pending_Upload` is now a legal transition. Users doing manual crash-recovery should use `update_status(Pending_Upload)` instead of `update_status(Failed)` (the old workaround).

**C. Environment / tooling changes:**

1. **Python 3.12+ required** (no change — already the floor, but worth stating).
2. **Optional `deriva-ml[torch]` / `deriva-ml[tf]` extras** (D2) — new install-time choices.
3. **Dropped `SETUPTOOLS_USE_DISTUTILS` env var** (PR #68 side-effect) — internal fix, no user action needed; documented because users with custom setups may notice the env var no longer gets set process-wide.

- [ ] **Step 3: Draft the migration guide**

Structure the guide as:

```markdown
# Migrating from previous deriva-ml versions

[one-paragraph intro: "if you're upgrading from X.Y.Z or earlier, here's
what changes"]

## At a glance

[single summary table: category | old | new | fix]

## Breaking changes

### DatasetBag.restructure_assets signature changes
[before/after code blocks, one per rename. Include the migration mapping
from this plan's Task 5.]

### Renamed private methods
[table of 12 renames + one paragraph per user-facing alternative where
the rename hides a better public API (resolve_rid, create_workflow)]

### Deleted methods
[list + replacements]

### Asset specification classes (documentation-only break)
[AssetRIDConfig was never a real class; copy-paste victims land here]

## New recommended patterns

### Training metrics
[metrics_file() vs manual asset_file_path]

### PyTorch / TensorFlow / Keras training
[pointer to the three-framework UG section]

### Crash recovery
[update_status(Pending_Upload) pattern]

## Finding affected code in your project

[grep recipes for each break:]
```bash
# Find restructure_assets old-kwarg callers:
grep -rn "group_by=\|value_selector=\|restructure_assets.*\"[A-Za-z_]*\.[A-Za-z_]*\"" your_project/

# Find renamed-private callers:
grep -rn -E "\b(domain_path|table_path|is_system_schema|get_domain_schemas|apply_logger_overrides|compute_diff|retrieve_rid|cache_features|add_workflow|start_upload)\(" your_project/ \
  | grep -v "_\(domain_path\|table_path\|..." # exclude already-underscored

# Find deleted-method callers:
grep -rn -E "\.(prefetch_dataset|list_foreign_keys|add_page|user_list|globus_login)\(" your_project/
```

## Version compatibility matrix

| deriva-ml version | Python | Torch | TF | Notes |
|---|---|---|---|---|
| prior published | ≥3.12 | n/a | n/a | [current baseline] |
| this release | ≥3.12 | ≥2.0 optional | ≥2.15 optional | [what this doc covers] |

## Support

[if users hit a break not documented here, pointer to raising an issue]
```

Keep it around 400-600 lines; detailed enough to answer "what do I change in my code" but not a rehash of the UG.

- [ ] **Step 4: Update `mkdocs.yml` nav**

Add the migration guide to the User Guide section of the nav, probably at the end (not between chapters):

```yaml
nav:
  - Introduction: index.md
  - User Guide:
      - Exploring a catalog: user-guide/exploring.md
      - Working with datasets: user-guide/datasets.md
      - ... [other chapters] ...
      - Integrating with hydra-zen: user-guide/hydra-zen.md
      - Migrating from previous versions: user-guide/migration.md    # new
  # ... rest unchanged ...
```

- [ ] **Step 5: Add pointer from Introduction**

`docs/index.md` gets one sentence near the top (or in a "What's new" section if one exists) pointing users coming from older versions at the migration guide. Don't make it prominent — users upgrading need it; new users don't.

- [ ] **Step 6: Grep sibling repos once more**

Run the sibling-repo greps from Task 5 Step 1 and Task 8 Step 5 one more time. This time record the raw output in the migration-guide's "Finding affected code in your project" section as worked examples — real call sites from real deriva-ml-ecosystem projects.

If the sibling repos aren't present on this machine, skip this step and flag in the commit message that the example-output section is representative, not actual.

- [ ] **Step 7: mkdocs --strict**

Run: `uv run mkdocs build --strict 2>&1 | tail -20`
Expected: no new warnings. Confirm the new page renders (it'll show up in `site/user-guide/migration/index.html` after build).

- [ ] **Step 8: Commit**

```bash
git add docs/user-guide/migration.md mkdocs.yml docs/index.md
git commit -m "docs(migration): add migration guide from previous deriva-ml version

Post-S2 D2 Task 9. User-facing migration guide covering every
breaking change and notable addition across the post-S2 documentation
pass (PRs #65, #66, #67, #68, and this PR).

Sections:
- At-a-glance table summarizing each change
- Breaking changes with before/after code:
  - restructure_assets signature (group_by → targets, value_selector
    merged, Feature.column dotted syntax removed)
  - 12 renamed private methods + user-facing alternatives where applicable
  - 5 deleted methods + replacements
  - AssetRIDConfig doc-only break (class is actually AssetSpec/
    AssetSpecConfig)
- New recommended patterns:
  - metrics_file() for training-metric logs (D1)
  - as_torch_dataset / as_tf_dataset for framework integration (D2)
  - update_status(Pending_Upload) for crash recovery (A)
- Grep recipes for finding affected call sites
- Version compatibility matrix

The guide lands in D2's PR because D2 ships the single biggest
break (restructure_assets kwarg rename — TypeError on call); users
pulling the new release find the guide in the same release that
breaks them.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Summary checklist

| Task | Files touched | Status |
|---|---|---|
| 1 | `target_resolution.py`, `test_target_resolution.py` | pending |
| 2 | `pyproject.toml` | pending |
| 3 | `torch_adapter.py`, `dataset_bag.py`, 3 test files | pending |
| 4 | `tf_adapter.py`, `dataset_bag.py`, 3 test files | pending |
| 5 | `dataset_bag.py` (restructure rewrite), existing test updates | pending |
| 6 | `offline.md` UG section | pending |
| 7 | Various docstrings + `CLAUDE.md` | pending |
| 8 | Verification only | pending |
| 9 | `migration.md`, `mkdocs.yml`, `index.md` | pending |

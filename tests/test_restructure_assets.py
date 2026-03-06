"""Tests for DatasetBag.restructure_assets, focusing on file_transformer and manifest return."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deriva_ml.dataset.dataset_bag import DatasetBag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASSET_RID = "1-ABC"
DATASET_RID = "1-DS1"


def _make_bag(tmp_path: Path, assets: list[dict]) -> MagicMock:
    """Return a MagicMock[DatasetBag] with internal methods pre-configured."""
    bag = MagicMock(spec=DatasetBag)
    bag._detect_asset_table.return_value = "Image"
    bag._build_dataset_type_path_map.return_value = {DATASET_RID: ["train"]}
    bag._get_asset_dataset_mapping.return_value = {a["RID"]: DATASET_RID for a in assets}
    bag._load_feature_values_cache.return_value = {}
    bag._get_reachable_assets.return_value = assets
    return bag


def _call(bag: MagicMock, output_dir: Path, **kwargs) -> dict[Path, Path]:
    """Call restructure_assets as an unbound method so the mock is used as self."""
    return DatasetBag.restructure_assets(bag, output_dir=output_dir, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def src_file(tmp_path: Path) -> Path:
    """A single source file on disk."""
    f = tmp_path / "source" / "image.jpg"
    f.parent.mkdir()
    f.write_bytes(b"fake image content")
    return f


@pytest.fixture()
def src_dcm(tmp_path: Path) -> Path:
    """A single DICOM source file on disk."""
    f = tmp_path / "source" / "scan.dcm"
    f.parent.mkdir()
    f.write_bytes(b"fake dicom content")
    return f


# ---------------------------------------------------------------------------
# Default behaviour (regression: manifest now returned instead of Path)
# ---------------------------------------------------------------------------

def test_symlink_returns_manifest(tmp_path: Path, src_file: Path) -> None:
    """Default symlink mode returns a manifest mapping src to the symlink."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": str(src_file)}]
    bag = _make_bag(tmp_path, assets)

    manifest = _call(bag, out_dir)

    assert isinstance(manifest, dict)
    assert len(manifest) == 1
    actual = manifest[src_file]
    assert actual.is_symlink()
    assert actual.resolve() == src_file.resolve()
    assert actual.name == src_file.name


def test_copy_returns_manifest(tmp_path: Path, src_file: Path) -> None:
    """use_symlinks=False copies files and returns a manifest."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": str(src_file)}]
    bag = _make_bag(tmp_path, assets)

    manifest = _call(bag, out_dir, use_symlinks=False)

    assert len(manifest) == 1
    actual = manifest[src_file]
    assert not actual.is_symlink()
    assert actual.exists()
    assert actual.read_bytes() == b"fake image content"


def test_no_assets_returns_empty_manifest(tmp_path: Path) -> None:
    """When there are no assets the manifest is empty."""
    out_dir = tmp_path / "output"
    bag = _make_bag(tmp_path, [])

    manifest = _call(bag, out_dir)

    assert manifest == {}


def test_missing_file_skipped(tmp_path: Path) -> None:
    """Assets whose Filename does not exist on disk are skipped."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": "/nonexistent/path/image.jpg"}]
    bag = _make_bag(tmp_path, assets)

    manifest = _call(bag, out_dir)

    assert manifest == {}


# ---------------------------------------------------------------------------
# file_transformer
# ---------------------------------------------------------------------------

def test_transformer_called_with_src_and_suggested_dest(tmp_path: Path, src_file: Path) -> None:
    """file_transformer receives (src_path, suggested_dest_path)."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": str(src_file)}]
    bag = _make_bag(tmp_path, assets)

    calls: list[tuple[Path, Path]] = []

    def transformer(src: Path, dest: Path) -> Path:
        calls.append((src, dest))
        dest.write_bytes(b"transformed")
        return dest

    _call(bag, out_dir, file_transformer=transformer)

    assert len(calls) == 1
    got_src, got_dest = calls[0]
    assert got_src == src_file
    # Suggested dest keeps original filename, placed under the type/group dirs
    assert got_dest.name == src_file.name
    assert got_dest.is_relative_to(out_dir)


def test_transformer_return_path_used_in_manifest(tmp_path: Path, src_file: Path) -> None:
    """The path returned by file_transformer is the value in the manifest."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": str(src_file)}]
    bag = _make_bag(tmp_path, assets)

    def transformer(src: Path, dest: Path) -> Path:
        dest.write_bytes(b"out")
        return dest

    manifest = _call(bag, out_dir, file_transformer=transformer)

    assert manifest[src_file].exists()
    assert manifest[src_file].read_bytes() == b"out"


def test_transformer_extension_change_reflected_in_manifest(tmp_path: Path, src_dcm: Path) -> None:
    """When transformer changes the extension the manifest maps src.dcm -> dest.png."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": str(src_dcm)}]
    bag = _make_bag(tmp_path, assets)

    def dcm_to_png(src: Path, dest: Path) -> Path:
        png = dest.with_suffix(".png")
        png.write_bytes(b"png data")
        return png

    manifest = _call(bag, out_dir, file_transformer=dcm_to_png)

    assert len(manifest) == 1
    actual = manifest[src_dcm]
    assert actual.suffix == ".png"
    assert actual.exists()
    assert actual.read_bytes() == b"png data"
    # Original .dcm should NOT have been created at the suggested path
    assert not (actual.parent / src_dcm.name).exists()


def test_transformer_replaces_symlink_copy_logic(tmp_path: Path, src_file: Path) -> None:
    """When file_transformer is provided, use_symlinks is ignored and no symlink is created."""
    out_dir = tmp_path / "output"
    assets = [{"RID": ASSET_RID, "Filename": str(src_file)}]
    bag = _make_bag(tmp_path, assets)

    transformer_calls = 0

    def transformer(src: Path, dest: Path) -> Path:
        nonlocal transformer_calls
        transformer_calls += 1
        dest.write_bytes(b"converted")
        return dest

    # use_symlinks=True would normally create a symlink; transformer should override that
    manifest = _call(bag, out_dir, use_symlinks=True, file_transformer=transformer)

    assert transformer_calls == 1
    actual = manifest[src_file]
    assert not actual.is_symlink()
    assert actual.read_bytes() == b"converted"


def test_split_parent_transparent_in_type_path(tmp_path: Path, src_file: Path) -> None:
    """Split-type parent datasets are transparent: their type doesn't add a path component.

    When a type_to_dir_map is provided and a dataset's type is NOT in the map
    (e.g. "Split"), that dataset is treated as a structural container and does
    not contribute a directory component. Its children (Training, Testing) still
    get their own clean paths via their own types.
    """
    out_dir = tmp_path / "output"
    SPLIT_RID = "1-SPL"
    TRAIN_RID = "1-TRN"

    assets = [{"RID": ASSET_RID, "Filename": str(src_file)}]
    bag = _make_bag(tmp_path, assets)
    bag._build_dataset_type_path_map.return_value = {TRAIN_RID: ["train"]}
    bag._get_asset_dataset_mapping.return_value = {ASSET_RID: TRAIN_RID}

    manifest = _call(
        bag, out_dir,
        type_to_dir_map={"Training": "train", "Testing": "test"},
    )

    assert len(manifest) == 1
    actual = manifest[src_file]
    # Relative path from output_dir should be train/filename, NOT split/train/filename
    rel = actual.relative_to(out_dir)
    assert rel.parts[0] == "train", f"Expected first dir 'train', got {rel}"
    assert len(rel.parts) == 2, f"Expected 2 path parts (dir/file), got {rel.parts}"


def test_transformer_multiple_assets(tmp_path: Path) -> None:
    """Transformer is called once per asset; manifest has one entry per asset."""
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    files = []
    for i in range(3):
        f = src_dir / f"scan_{i}.dcm"
        f.write_bytes(f"content {i}".encode())
        files.append(f)

    out_dir = tmp_path / "output"
    assets = [{"RID": f"1-A{i}", "Filename": str(files[i])} for i in range(3)]
    bag = _make_bag(tmp_path, assets)
    # Fix the dataset mapping for all three RIDs
    bag._get_asset_dataset_mapping.return_value = {a["RID"]: DATASET_RID for a in assets}

    converted: list[Path] = []

    def transformer(src: Path, dest: Path) -> Path:
        out = dest.with_suffix(".png")
        out.write_bytes(b"png")
        converted.append(out)
        return out

    manifest = _call(bag, out_dir, file_transformer=transformer)

    assert len(manifest) == 3
    assert set(manifest.keys()) == set(files)
    for src, actual in manifest.items():
        assert actual.suffix == ".png"
        assert actual.exists()

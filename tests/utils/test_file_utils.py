"""
Tests for the file utilities module.
"""

from core.filespec import create_filespecs, create_spec, list_all_files


def test_list_all_files(tmp_path):
    """Test listing files in a directory."""
    # Create test files
    (tmp_path / "file1.txt").write_text("content 1")
    (tmp_path / "file2.txt").write_text("content 2")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("content 3")

    # Test listing all files
    files = list_all_files(tmp_path)
    assert len(files) == 3
    assert all(f.is_file() for f in files)
    assert set(f.name for f in files) == {"file1.txt", "file2.txt", "file3.txt"}

    # Test listing single file
    single_file = tmp_path / "file1.txt"
    files = list_all_files(single_file)
    assert len(files) == 1
    assert files[0].name == "file1.txt"


def test_create_spec(tmp_path):
    """Test creating a FileSpec for a single file."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Create spec
    spec = create_spec(test_file, "Test file")

    assert spec.url.endswith("test.txt")
    assert spec.description == "Test file"
    assert spec.length == len("test content")
    assert len(spec.md5) == 32  # MD5 hash is 32 characters


def test_create_filespecs(tmp_path):
    """Test creating FileSpecs for multiple files."""
    # Create test files
    (tmp_path / "file1.txt").write_text("content 1")
    (tmp_path / "file2.txt").write_text("content 2")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("content 3")

    # Create specs
    specs = list(create_filespecs(tmp_path, "Test files"))

    assert len(specs) == 3
    assert all(spec.description == "Test files" for spec in specs)
    assert len({spec.md5 for spec in specs}) == 3  # All files should have different MD5s

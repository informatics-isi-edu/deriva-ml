"""Unit tests for local_db.paths — pure path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from deriva_ml.local_db import paths as p


class TestWorkspaceRoot:
    def test_encodes_host_and_catalog(self, tmp_path: Path) -> None:
        root = p.workspace_root(tmp_path, "example.org", "42")
        assert root == tmp_path / "catalogs" / "example.org__42"

    def test_sanitises_unsafe_hostname(self, tmp_path: Path) -> None:
        # Path separators and other unsafe chars are replaced with '_'.
        root = p.workspace_root(tmp_path, "a/b:c", "1")
        assert root == tmp_path / "catalogs" / "a_b_c__1"

    def test_numeric_catalog_id_coerced_to_str(self, tmp_path: Path) -> None:
        root = p.workspace_root(tmp_path, "example.org", 42)
        assert root.name == "example.org__42"


class TestWorkingDbPath:
    def test_under_workspace_root(self, tmp_path: Path) -> None:
        db = p.working_db_path(tmp_path, "example.org", "42")
        assert db == tmp_path / "catalogs" / "example.org__42" / "working"


class TestWorkingDir:
    def test_working_dir_is_directory_not_file(self, tmp_path: Path) -> None:
        d = p.working_db_path(tmp_path, "example.org", "42")
        assert d.name == "working"
        assert not d.name.endswith(".db")

    def test_working_dir_under_workspace_root(self, tmp_path: Path) -> None:
        d = p.working_db_path(tmp_path, "example.org", "42")
        assert d == tmp_path / "catalogs" / "example.org__42" / "working"


class TestWorkingMainDbPath:
    def test_main_db_inside_working_dir(self, tmp_path: Path) -> None:
        main = p.working_main_db_path(tmp_path, "example.org", "42")
        assert main == tmp_path / "catalogs" / "example.org__42" / "working" / "main.db"


class TestSliceDir:
    def test_slice_dir_under_workspace(self, tmp_path: Path) -> None:
        d = p.slice_dir(tmp_path, "example.org", "42", "abc123")
        assert d == tmp_path / "catalogs" / "example.org__42" / "slices" / "abc123"

    def test_slice_id_sanitised(self, tmp_path: Path) -> None:
        d = p.slice_dir(tmp_path, "example.org", "42", "a/b")
        assert d == tmp_path / "catalogs" / "example.org__42" / "slices" / "a_b"


class TestSliceDbPath:
    def test_slice_db_file_under_slice_dir(self, tmp_path: Path) -> None:
        db = p.slice_db_path(tmp_path, "example.org", "42", "abc123")
        assert db == (tmp_path / "catalogs" / "example.org__42" / "slices" / "abc123" / "slice.db")


class TestSanitiseComponent:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("foo", "foo"),
            ("foo.bar", "foo.bar"),
            ("a/b", "a_b"),
            ("a:b", "a_b"),
            ("..", "__"),
            ("", "_"),
        ],
    )
    def test_sanitise(self, raw: str, expected: str) -> None:
        assert p._sanitise_component(raw) == expected

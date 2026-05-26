"""Unit tests for :func:`deriva_ml.config.parse_config_file`.

The walker is a pure-Python AST visitor with no catalog dependency,
so these tests are fast and don't need any fixtures beyond temp
files. Coverage targets:

- Each recognized constructor shape (bare, ``builds(...)`` wrapper,
  ``with_description(...)`` wrapper, ``store(<Class>, ...)``).
- Module-level constant resolution.
- Unresolvable RIDs surface as ``rid=None`` (not as an exception).
- Syntax errors return a :class:`ConfigFileParseError`, not an
  exception.
- Comments are skipped.
- ``__pycache__`` paths are not visited (tested indirectly via
  ``validate_config_directory``).
"""

from __future__ import annotations

from pathlib import Path

from deriva_ml.config import parse_config_file
from deriva_ml.config.validation import ConfigFileParseError


def _write(tmp_path: Path, source: str, name: str = "configs.py") -> Path:
    p = tmp_path / name
    p.write_text(source)
    return p


# ---------------------------------------------------------------------------
# Bare-constructor shapes
# ---------------------------------------------------------------------------


def test_bare_dataset_spec_config(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
from deriva_ml.dataset import DatasetSpecConfig
datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_kind == "DatasetSpecConfig"
    assert e.rid == "1-AAAA"
    assert e.version == "0.1.0"
    assert e.line == 3


def test_bare_asset_spec_config(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
assets_store(name="w", spec=AssetSpecConfig(rid="1-BBBB", cache=True))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_kind == "AssetSpecConfig"
    assert e.rid == "1-BBBB"
    assert e.cache is True


def test_bare_workflow(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
workflow_store(name="t", spec=Workflow(workflow_rid="1-CCCC"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    assert entries[0].entry_kind == "Workflow"
    assert entries[0].rid == "1-CCCC"


# ---------------------------------------------------------------------------
# Wrapper shapes
# ---------------------------------------------------------------------------


def test_with_description_wrapper(tmp_path: Path) -> None:
    """``with_description(DatasetSpecConfig(...), "...")`` unwraps to the inner spec."""
    path = _write(
        tmp_path,
        """
from deriva_ml.execution import with_description
datasets_store(
    name="x",
    spec=with_description(
        DatasetSpecConfig(rid="1-AAAA", version="0.2.0"),
        "training set",
    ),
)
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    # Should emit exactly ONE entry, not two -- the inner call is
    # claimed by the wrapper's classification.
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_kind == "DatasetSpecConfig"
    assert e.rid == "1-AAAA"
    assert e.version == "0.2.0"


def test_builds_with_known_class(tmp_path: Path) -> None:
    """``builds(Workflow, workflow_rid=...)`` is treated as the Workflow class."""
    path = _write(
        tmp_path,
        """
from hydra_zen import builds
workflow_store(name="t", spec=builds(Workflow, workflow_rid="1-DDDD"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_kind == "Workflow"
    assert e.rid == "1-DDDD"


def test_builds_with_unsuffixed_spec_name(tmp_path: Path) -> None:
    """``builds(DatasetSpec, ...)`` (no ``Config`` suffix) normalizes to ``DatasetSpecConfig``."""
    path = _write(
        tmp_path,
        """
datasets_store(name="x", spec=builds(DatasetSpec, rid="1-EEEE", version="0.3.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_kind == "DatasetSpecConfig"
    assert e.rid == "1-EEEE"
    assert e.version == "0.3.0"


def test_deriva_store_positional_class(tmp_path: Path) -> None:
    """``deriva_store(DerivaMLConfig, hostname=..., catalog_id=...)`` is recognized."""
    path = _write(
        tmp_path,
        """
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="data.example.org",
    catalog_id="42",
)
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_kind == "DerivaMLConfig"
    assert e.hostname == "data.example.org"
    assert e.catalog_id == "42"
    # `name=` is a hydra-zen store kwarg, not a DerivaMLConfig field.
    # We intentionally don't capture it -- the validator doesn't need it.


def test_deriva_store_int_catalog_id(tmp_path: Path) -> None:
    """``catalog_id=`` is sometimes a literal int; normalize to str."""
    path = _write(
        tmp_path,
        """
deriva_store(
    DerivaMLConfig,
    hostname="localhost",
    catalog_id=7,
)
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert entries[0].catalog_id == "7"


# ---------------------------------------------------------------------------
# Module-level constant resolution
# ---------------------------------------------------------------------------


def test_resolves_module_level_constants(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
TRAINING_RID = "1-AAAA"
TEST_RID: str = "1-BBBB"
datasets_store(name="train", spec=DatasetSpecConfig(rid=TRAINING_RID, version="0.1.0"))
datasets_store(name="test", spec=DatasetSpecConfig(rid=TEST_RID, version="0.1.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    rids = [e.rid for e in entries]
    assert rids == ["1-AAAA", "1-BBBB"]


def test_unresolvable_constant_returns_none(tmp_path: Path) -> None:
    """Constants not found in the module-level name map yield ``rid=None``."""
    path = _write(
        tmp_path,
        """
def get_rid():
    return "1-XYZW"
datasets_store(name="x", spec=DatasetSpecConfig(rid=get_rid(), version="0.1.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    assert entries[0].rid is None


def test_unresolvable_local_assignment(tmp_path: Path) -> None:
    """Names assigned inside functions don't pollute the module-level map."""
    path = _write(
        tmp_path,
        """
def _wrap():
    INNER = "1-NOPE"
    return DatasetSpecConfig(rid=INNER, version="0.1.0")
datasets_store(name="x", spec=DatasetSpecConfig(rid=INNER, version="0.1.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    # Both entries are emitted (the one inside the function AND the
    # one at module scope). Neither resolves -- INNER is not at
    # module scope.
    for e in entries:
        assert e.rid is None


# ---------------------------------------------------------------------------
# Negative / edge cases
# ---------------------------------------------------------------------------


def test_comments_with_constructor_text_are_ignored(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
# DatasetSpecConfig(rid="FAKE-RID", version="0.0.0") in a comment
datasets_store(name="x", spec=DatasetSpecConfig(rid="1-REAL", version="0.1.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    assert entries[0].rid == "1-REAL"


def test_syntax_error_returns_parse_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
this is not python code (((
""",
    )
    entries, err = parse_config_file(path)
    assert entries == []
    assert err is not None
    assert isinstance(err, ConfigFileParseError)
    assert err.line is not None


def test_missing_file_returns_parse_error(tmp_path: Path) -> None:
    entries, err = parse_config_file(tmp_path / "nope.py")
    assert entries == []
    assert err is not None
    assert "nope.py" in err.file


def test_empty_file_returns_empty_entries(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    entries, err = parse_config_file(path)
    assert err is None
    assert entries == []


def test_no_known_constructors_returns_empty(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
from hydra_zen import builds
some_other_store(SomeOtherConfig, name="x")
do_something(SomethingElse, rid="1-AAAA")
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert entries == []


def test_missing_version_kwarg(tmp_path: Path) -> None:
    """Missing version is captured as ``version=None``; validator handles it."""
    path = _write(
        tmp_path,
        """
datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert len(entries) == 1
    assert entries[0].rid == "1-AAAA"
    assert entries[0].version is None


def test_snippet_captures_source_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
datasets_store(name="x", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    assert entries[0].snippet is not None
    assert "DatasetSpecConfig" in entries[0].snippet


def test_multiple_entries_in_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
datasets_store(name="a", spec=DatasetSpecConfig(rid="1-AAAA", version="0.1.0"))
datasets_store(name="b", spec=DatasetSpecConfig(rid="1-BBBB", version="0.2.0"))
datasets_store(name="c", spec=DatasetSpecConfig(rid="1-CCCC", version="0.3.0"))
""",
    )
    entries, err = parse_config_file(path)
    assert err is None
    rids = [e.rid for e in entries]
    assert rids == ["1-AAAA", "1-BBBB", "1-CCCC"]
    assert [e.line for e in entries] == [2, 3, 4]

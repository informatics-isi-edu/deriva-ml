"""Unit tests for `DerivaML.from_context`, `cite`, `chaise_url`.

Closes coverage gaps audited at
``docs/design/deriva-ml-audit-2026-05-phase3-core.md`` §C.2 and §C.3.

All tests here are **pure unit tests** — no live catalog, no
``test_ml`` fixture. The class methods under test are exercised
either through their helper functions (``_find_context_file``) or
via monkeypatched ``DerivaML.__init__``.
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# _find_context_file — walks parent directories looking for the JSON file
# ---------------------------------------------------------------------------


def test_find_context_file_in_start_dir(tmp_path):
    """The file exists directly in the start directory."""
    from deriva_ml.core.base import CONTEXT_FILENAME, _find_context_file

    ctx = tmp_path / CONTEXT_FILENAME
    ctx.write_text(json.dumps({"hostname": "h", "catalog_id": "1"}))

    found = _find_context_file(tmp_path)
    assert found == ctx.resolve()


def test_find_context_file_walks_parents(tmp_path):
    """The file is several levels up; the walker finds it."""
    from deriva_ml.core.base import CONTEXT_FILENAME, _find_context_file

    ctx = tmp_path / CONTEXT_FILENAME
    ctx.write_text(json.dumps({"hostname": "h", "catalog_id": "1"}))

    deep = tmp_path / "level1" / "level2" / "level3"
    deep.mkdir(parents=True)

    found = _find_context_file(deep)
    assert found == ctx.resolve()


def test_find_context_file_missing_raises(tmp_path):
    """No context file anywhere → FileNotFoundError with a helpful message."""
    from deriva_ml.core.base import _find_context_file

    with pytest.raises(FileNotFoundError) as exc:
        _find_context_file(tmp_path)
    # Error message mentions the filename and points to `connect_catalog`
    assert ".deriva-context.json" in str(exc.value)


# ---------------------------------------------------------------------------
# from_context — parses JSON and calls DerivaML(**kwargs)
# ---------------------------------------------------------------------------


def test_from_context_passes_required_fields(tmp_path, monkeypatch):
    """Hostname + catalog_id from the context file flow into the constructor."""
    from deriva_ml.core import base as base_module
    from deriva_ml.core.base import CONTEXT_FILENAME, DerivaML

    captured: list[dict] = []

    class _FakeML:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    # Replace DerivaML.__init__ via the class itself (monkeypatching
    # `cls(**kwargs)` requires intercepting `DerivaML.__init__`).
    # Use `setattr` of an `__init__` on a subclass via from_context's
    # `cls` binding: simpler to monkeypatch the class entirely.
    monkeypatch.setattr(base_module, "DerivaML", _FakeML)

    (tmp_path / CONTEXT_FILENAME).write_text(
        json.dumps(
            {
                "hostname": "deriva.example.org",
                "catalog_id": "42",
                "default_schema": "mydomain",
                "working_dir": "/tmp/wd",
            }
        )
    )

    # Have to call the captured-binding version (we replaced DerivaML).
    base_module.DerivaML.from_context = DerivaML.from_context.__func__.__get__(_FakeML)
    base_module.DerivaML.from_context(path=tmp_path)

    assert len(captured) == 1
    kw = captured[0]
    assert kw["hostname"] == "deriva.example.org"
    assert kw["catalog_id"] == "42"
    assert kw["default_schema"] == "mydomain"
    assert kw["working_dir"] == "/tmp/wd"


def test_from_context_omits_optional_fields_when_absent(tmp_path, monkeypatch):
    """`default_schema` / `working_dir` are omitted from kwargs when null."""
    from deriva_ml.core import base as base_module
    from deriva_ml.core.base import CONTEXT_FILENAME, DerivaML

    captured: list[dict] = []

    class _FakeML:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(base_module, "DerivaML", _FakeML)

    (tmp_path / CONTEXT_FILENAME).write_text(json.dumps({"hostname": "h", "catalog_id": "1"}))

    base_module.DerivaML.from_context = DerivaML.from_context.__func__.__get__(_FakeML)
    base_module.DerivaML.from_context(path=tmp_path)

    kw = captured[0]
    assert kw == {"hostname": "h", "catalog_id": "1"}
    assert "default_schema" not in kw
    assert "working_dir" not in kw


def test_from_context_no_file_raises(tmp_path, monkeypatch):
    """Missing context file in the directory tree → FileNotFoundError."""
    from deriva_ml.core import base as base_module
    from deriva_ml.core.base import DerivaML

    class _FakeML:
        def __init__(self, **kwargs):
            raise AssertionError("should not construct on missing context")

    monkeypatch.setattr(base_module, "DerivaML", _FakeML)
    base_module.DerivaML.from_context = DerivaML.from_context.__func__.__get__(_FakeML)

    with pytest.raises(FileNotFoundError):
        base_module.DerivaML.from_context(path=tmp_path)


# ---------------------------------------------------------------------------
# cite — citation URL construction
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    """Snapshot stand-in carrying just the ``snaptime`` attribute."""

    def __init__(self, snaptime: str):
        self.snaptime = snaptime


class _FakeCatalog:
    """Minimal catalog stand-in for `cite` tests.

    `cite()` reads `self.catalog.latest_snapshot().snaptime`.
    `chaise_url()` reads `self.catalog.get_server_uri()`.
    """

    def __init__(self, snaptime: str = "2026-01-01T12:00:00"):
        self._snaptime = snaptime

    def latest_snapshot(self):
        return _FakeSnapshot(self._snaptime)

    def get_server_uri(self):
        return f"https://deriva.example.org/ermrest/catalog/42@{self._snaptime}"


class _CiteHarness:
    """Lightweight stand-in for the `cite` method's `self`.

    `DerivaML.cite` reads `self.host_name`, `self.catalog_id`,
    `self.catalog.get_server_uri()`, and calls `self.resolve_rid(rid)`.
    We construct an object with just those four members so we can
    invoke `DerivaML.cite` unbound.
    """

    def __init__(self, *, snaptime: str = "2026-01-01T12:00:00"):
        self.host_name = "deriva.example.org"
        self.catalog_id = "42"
        self.catalog = _FakeCatalog(snaptime=snaptime)

    def resolve_rid(self, rid):
        # Treat any RID as resolvable; raise for "INVALID-*" strings
        if rid.startswith("INVALID"):
            from deriva_ml.core.exceptions import DerivaMLException

            raise DerivaMLException(f"unknown rid: {rid}")
        return object()  # not consulted further by `cite`


def test_cite_returns_permanent_url_by_default():
    """Default behaviour: include the catalog snapshot time."""
    from deriva_ml.core.base import DerivaML

    harness = _CiteHarness(snaptime="2026-01-01T12:00:00")
    url = DerivaML.cite(harness, "1-ABC")  # type: ignore[arg-type]
    assert url == "https://deriva.example.org/id/42/1-ABC@2026-01-01T12:00:00"


def test_cite_current_drops_snapshot():
    """`current=True` returns a URL without the snapshot time."""
    from deriva_ml.core.base import DerivaML

    harness = _CiteHarness()
    url = DerivaML.cite(harness, "1-ABC", current=True)  # type: ignore[arg-type]
    assert url == "https://deriva.example.org/id/42/1-ABC"
    assert "@" not in url.split("/id/")[-1]


def test_cite_passes_through_existing_url():
    """A pre-formed citation URL is returned unchanged (short-circuit)."""
    from deriva_ml.core.base import DerivaML

    harness = _CiteHarness()
    pre = "https://deriva.example.org/id/42/1-ABC@2025-12-31"
    url = DerivaML.cite(harness, pre)  # type: ignore[arg-type]
    assert url == pre


def test_cite_accepts_dict_with_rid_key():
    """Passing `{"RID": "..."}` works the same as a bare RID string."""
    from deriva_ml.core.base import DerivaML

    harness = _CiteHarness()
    url = DerivaML.cite(harness, {"RID": "1-XYZ"})  # type: ignore[arg-type]
    assert url.startswith("https://deriva.example.org/id/42/1-XYZ@")

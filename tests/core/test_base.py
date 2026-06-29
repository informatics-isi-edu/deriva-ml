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
from unittest.mock import patch

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


def test_cite_dry_run_sentinel_returns_placeholder_string():
    """Dry-run sentinel RID ``"0000"`` short-circuits to a non-link string.

    ``run_notebook(dry_run=True)`` (and related dry-run paths) hand back
    ``DRY_RUN_RID`` ("0000") as the execution RID because no execution
    row was created on the catalog. Notebook templates commonly embed
    the cite output in ``[label]({url})`` markdown. Resolving "0000"
    against the catalog 404s and crashes the informational header cell,
    so ``cite`` must guard the sentinel before reaching the catalog.

    The placeholder is a bare string (no scheme) so a markdown link
    renderer leaves it as plain text rather than rendering a clickable
    link that would resolve to a 404.
    """
    from deriva_ml.core.base import DerivaML
    from deriva_ml.core.constants import DRY_RUN_RID

    harness = _CiteHarness()

    def _explode(_rid):  # pragma: no cover - sentinel must not call this
        raise AssertionError("resolve_rid must not be invoked for the dry-run sentinel")

    harness.resolve_rid = _explode  # type: ignore[assignment]

    out = DerivaML.cite(harness, DRY_RUN_RID)  # type: ignore[arg-type]
    assert out == f"dry-run (rid={DRY_RUN_RID})"
    assert "http" not in out  # not a clickable link


def test_cite_dry_run_sentinel_via_dict():
    """Same short-circuit when the sentinel arrives wrapped in ``{"RID": ...}``."""
    from deriva_ml.core.base import DerivaML
    from deriva_ml.core.constants import DRY_RUN_RID

    harness = _CiteHarness()
    out = DerivaML.cite(harness, {"RID": DRY_RUN_RID})  # type: ignore[arg-type]
    assert out == f"dry-run (rid={DRY_RUN_RID})"


# ---------------------------------------------------------------------------
# catalog_snapshot — kwarg forwarding
# ---------------------------------------------------------------------------
#
# ``DerivaML.catalog_snapshot(version)`` must construct the snapshot
# instance with every connection-shaping kwarg the original
# instance carries. Earlier versions forwarded only the two logging
# levels and the hostname — every other kwarg the user had passed
# to the original ``DerivaML(...)`` (working_dir, cache_dir,
# domain_schemas, default_schema, project_name, ml_schema, s3_bucket,
# use_minid, credential, clean_execution_dir, mode) got silently
# re-defaulted. The snapshot then had a different working_dir from
# its parent, re-fetched credentials from disk, and re-auto-detected
# domain schemas — all observable behaviour drifts.
#
# This test pins the forwarding contract by monkey-patching
# ``DerivaML.__init__`` to capture the kwargs the snapshot call
# uses, then asserts each one matches the parent.


def test_catalog_snapshot_forwards_connection_kwargs():
    """Every connection-shaping kwarg on `self` is forwarded to the snapshot.

    Catches regressions where ``catalog_snapshot`` drops a kwarg
    when DerivaML grows a new constructor parameter.
    """
    from deriva_ml.core.base import DerivaML
    from deriva_ml.core.connection_mode import ConnectionMode

    # Build a fake "parent" DerivaML with known kwarg state. We
    # don't actually construct one (would need a live catalog);
    # we forge an instance and populate the attributes that
    # ``catalog_snapshot`` reads.
    parent = DerivaML.__new__(DerivaML)
    parent.host_name = "src.example.org"
    parent.domain_schemas = {"my_domain"}
    parent.default_schema = "my_domain"
    parent.project_name = "myproj"
    parent.cache_dir = "/tmp/mycache"
    parent.working_dir = "/tmp/mywd"
    parent.ml_schema = "deriva-ml-custom"
    parent._logging_level = 30  # logging.WARNING
    parent._deriva_logging_level = 30
    parent.credential = {"cookie": "secret"}
    parent.s3_bucket = "s3://my-bucket"
    parent.use_minid = True
    parent.clean_execution_dir = False
    parent._mode = ConnectionMode.online
    # Attributes catalog_snapshot reads for the schema-reuse + memoization
    # fast path (perf spec §3.1): the parsed schema it forwards, and the
    # per-instance snapshot cache.
    parent._schema_json = {"schemas": {}}
    parent._snapshot_cache = {}

    # Capture the kwargs passed to the inner DerivaML(...) call.
    captured: dict = {}

    def fake_init(self, hostname, catalog_id, **kwargs):
        captured["hostname"] = hostname
        captured["catalog_id"] = catalog_id
        captured.update(kwargs)
        # Stub out the rest so __init__ doesn't try to connect.
        raise _StopInit("captured")

    class _StopInit(Exception):
        pass

    import contextlib

    with patch.object(DerivaML, "__init__", fake_init):
        with contextlib.suppress(_StopInit):
            DerivaML.catalog_snapshot(parent, "1@SNAP")  # type: ignore[arg-type]

    # Hostname + the version snapshot ID are positional.
    assert captured["hostname"] == "src.example.org"
    assert captured["catalog_id"] == "1@SNAP"

    # Every connection-shaping kwarg the parent carries must have
    # arrived at the snapshot. If a new kwarg is added to DerivaML
    # and catalog_snapshot doesn't forward it, this list — and the
    # docstring on catalog_snapshot — needs to be updated together.
    assert captured["domain_schemas"] == {"my_domain"}
    assert captured["default_schema"] == "my_domain"
    assert captured["project_name"] == "myproj"
    assert captured["cache_dir"] == "/tmp/mycache"
    assert captured["working_dir"] == "/tmp/mywd"
    assert captured["ml_schema"] == "deriva-ml-custom"
    assert captured["logging_level"] == 30
    assert captured["deriva_logging_level"] == 30
    assert captured["credential"] == {"cookie": "secret"}
    assert captured["s3_bucket"] == "s3://my-bucket"
    assert captured["use_minid"] is True
    assert captured["clean_execution_dir"] is False
    assert captured["mode"] is ConnectionMode.online
    # The schema-reuse fast path (perf spec §3.1) forwards the parent's
    # already-parsed schema so the snapshot skips its own /schema fetch.
    assert captured["reuse_schema_json"] == {"schemas": {}}


# ---------------------------------------------------------------------------
# is_authenticated / whoami — session check via the server-level
# get_authn_session() (GET /authn/session). 200+client → authenticated;
# 401/404 → not. Pure unit tests: the catalog binding is mocked so both
# the success and no-session paths are deterministic with no live server.
# ---------------------------------------------------------------------------


def _ml_with_mock_catalog(catalog):
    """A DerivaML instance with only its ``catalog`` attribute set.

    Built via ``__new__`` to bypass ``__init__`` (no network) — these tests
    exercise only the auth methods, which touch ``self.catalog``.
    """
    from deriva_ml.core.base import DerivaML

    ml = DerivaML.__new__(DerivaML)
    ml.catalog = catalog
    return ml


def test_whoami_returns_client_identity_when_session_valid():
    """whoami() returns the server's ``client`` identity dict on a 200."""
    from unittest.mock import MagicMock

    client = {
        "id": "https://auth.globus.org/abc",
        "display_name": "user@example.org",
        "full_name": "Example User",
        "email": "user@example.org",
        "identities": [],
    }
    resp = MagicMock()
    resp.json.return_value = {"client": client, "seconds_remaining": 3600}
    catalog = MagicMock()
    catalog.get_authn_session.return_value = resp

    ml = _ml_with_mock_catalog(catalog)
    who = ml.whoami()

    assert who == client
    catalog.get_authn_session.assert_called_once()


def test_whoami_returns_none_when_no_session():
    """whoami() returns None when /authn/session 404s (no session)."""
    from unittest.mock import MagicMock

    from requests.exceptions import HTTPError

    err = HTTPError()
    err.response = MagicMock(status_code=404)
    catalog = MagicMock()
    catalog.get_authn_session.side_effect = err

    ml = _ml_with_mock_catalog(catalog)
    assert ml.whoami() is None


def test_is_authenticated_true_when_session_valid():
    """is_authenticated() is True when a session resolves."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.json.return_value = {"client": {"id": "x"}}
    catalog = MagicMock()
    catalog.get_authn_session.return_value = resp

    ml = _ml_with_mock_catalog(catalog)
    assert ml.is_authenticated() is True


def test_is_authenticated_false_on_401_and_404():
    """is_authenticated() is False for both 401 and 404 (no session)."""
    from unittest.mock import MagicMock

    from requests.exceptions import HTTPError

    for status in (401, 404):
        err = HTTPError()
        err.response = MagicMock(status_code=status)
        catalog = MagicMock()
        catalog.get_authn_session.side_effect = err
        ml = _ml_with_mock_catalog(catalog)
        assert ml.is_authenticated() is False, f"status {status} should be not-authenticated"


def test_is_authenticated_reraises_unexpected_http_error():
    """A non-auth HTTPError (e.g. 500) propagates — not swallowed as 'not authed'."""
    from unittest.mock import MagicMock

    from requests.exceptions import HTTPError

    err = HTTPError()
    err.response = MagicMock(status_code=500)
    catalog = MagicMock()
    catalog.get_authn_session.side_effect = err

    ml = _ml_with_mock_catalog(catalog)
    with pytest.raises(HTTPError):
        ml.is_authenticated()

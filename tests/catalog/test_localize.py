"""Unit tests for :mod:`deriva_ml.catalog.localize`.

These exercise pure helpers and result-object shape. The full
``localize_assets`` integration flow (Hatrac fetch + push,
catalog update, provenance write) requires a live Deriva catalog
and is left to a separate integration-test fixture.

Audit Phase 3 catalog/ §3.2 — ``localize_assets`` had zero
tests before this file.
"""

from __future__ import annotations

import pytest

from deriva_ml.catalog.localize import LocalizeResult, _extract_hatrac_path


# ---------------------------------------------------------------------------
# LocalizeResult — Pydantic round-trip + defaults
# ---------------------------------------------------------------------------


def test_localize_result_defaults() -> None:
    """All count fields start at zero and the lists start empty."""
    result = LocalizeResult()
    assert result.assets_processed == 0
    assert result.assets_skipped == 0
    assert result.assets_failed == 0
    assert result.errors == []
    assert result.localized_assets == []


def test_localize_result_round_trip() -> None:
    """``model_dump`` / ``model_validate`` is a no-op round trip."""
    original = LocalizeResult(
        assets_processed=10,
        assets_skipped=2,
        assets_failed=1,
        errors=["asset-7 failed: timeout"],
        localized_assets=[
            ("1-ABC", "https://src/hatrac/x", "https://dst/hatrac/y"),
        ],
    )
    dumped = original.model_dump(mode="json")
    restored = LocalizeResult.model_validate(dumped)
    assert restored == original


def test_localize_result_increment_counts() -> None:
    """Counts can be incremented field-by-field during a run."""
    result = LocalizeResult()
    result.assets_processed += 1
    result.assets_skipped += 2
    result.assets_failed += 1
    assert result.assets_processed == 1
    assert result.assets_skipped == 2
    assert result.assets_failed == 1


# ---------------------------------------------------------------------------
# _extract_hatrac_path — pure path-parsing helper
# ---------------------------------------------------------------------------


def test_extract_hatrac_path_absolute_https_url() -> None:
    """Standard absolute HTTPS URL → ``/hatrac/...`` path."""
    url = "https://src.example.org/hatrac/Image/abc123.jpg"
    assert _extract_hatrac_path(url) == "/hatrac/Image/abc123.jpg"


def test_extract_hatrac_path_absolute_http_url() -> None:
    """HTTP also works."""
    url = "http://src.example.org/hatrac/Image/abc123.jpg"
    assert _extract_hatrac_path(url) == "/hatrac/Image/abc123.jpg"


def test_extract_hatrac_path_relative_url() -> None:
    """Pure path (no scheme) — assumed already a hatrac path."""
    assert _extract_hatrac_path("/hatrac/Image/abc123.jpg") == "/hatrac/Image/abc123.jpg"


def test_extract_hatrac_path_nested_namespace() -> None:
    """Deeper namespace hierarchies are preserved."""
    url = "https://host/hatrac/MyTable/2024/05/abc123.dat"
    assert _extract_hatrac_path(url) == "/hatrac/MyTable/2024/05/abc123.dat"


def test_extract_hatrac_path_non_hatrac_url() -> None:
    """A URL that's not hatrac-rooted returns None."""
    assert _extract_hatrac_path("https://host/objects/abc123.jpg") is None


def test_extract_hatrac_path_empty_path() -> None:
    """Just a hostname with no path → None."""
    assert _extract_hatrac_path("https://host") is None


@pytest.mark.parametrize(
    "url",
    [
        "ftp://src.example.org/hatrac/Image/abc.jpg",
        "file:///hatrac/Image/abc.jpg",
        "s3://bucket/hatrac/key",
    ],
)
def test_extract_hatrac_path_rejects_non_http_schemes(url: str) -> None:
    """Audit §6.6 — non-HTTP(S) schemes are rejected.

    The downstream HatracStore would fail anyway with a confusing
    error; rejecting at the parse boundary is friendlier.
    """
    assert _extract_hatrac_path(url) is None


def test_extract_hatrac_path_hatrac_in_middle_of_path() -> None:
    """``/foo/hatrac/...`` returns from the /hatrac/ position.

    Real-world prefixed URLs (e.g., reverse-proxy paths) still
    yield a valid hatrac path.
    """
    url = "https://host/some/prefix/hatrac/Image/abc.jpg"
    assert _extract_hatrac_path(url) == "/hatrac/Image/abc.jpg"

"""Unit tests for ``deriva_ml.core.sort.resolve_sort``.

Pure-Python -- no catalog, no fixtures.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.sort import SortSpec, resolve_sort


def test_resolve_sort_none_returns_none():
    """sort=None means no sort applied; helper returns None."""
    result = resolve_sort(None, lambda p: "should-not-be-called", object())
    assert result is None


def test_resolve_sort_true_calls_default():
    """sort=True invokes the method-supplied default callable."""
    captured_path = []

    def default_callable(path):
        captured_path.append(path)
        return "default-keys"

    path = object()
    result = resolve_sort(True, default_callable, path)
    assert captured_path == [path]
    assert result == ["default-keys"]


def test_resolve_sort_callable_runs_user_callable():
    """sort=callable invokes the user callable, ignores the default."""
    user_calls = []
    default_calls = []

    def user_callable(path):
        user_calls.append(path)
        return "user-keys"

    def default_callable(path):
        default_calls.append(path)
        return "default-keys"

    path = object()
    result = resolve_sort(user_callable, default_callable, path)
    assert user_calls == [path]
    assert default_calls == []
    assert result == ["user-keys"]


def test_resolve_sort_wraps_single_value_in_list():
    """A single column key is wrapped so the result is always list-shaped."""
    result = resolve_sort(lambda p: "RCT-desc", lambda p: "ignored", object())
    assert result == ["RCT-desc"]


def test_resolve_sort_passes_through_list():
    """A list/tuple is returned as a list (tuple is normalized)."""
    result_list = resolve_sort(lambda p: ["A", "B"], lambda p: "ignored", object())
    result_tuple = resolve_sort(lambda p: ("A", "B"), lambda p: "ignored", object())
    assert result_list == ["A", "B"]
    assert result_tuple == ["A", "B"]


def test_resolve_sort_rejects_invalid_type():
    """sort must be None, True, or callable; other values raise TypeError."""
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        resolve_sort(42, lambda p: "ignored", object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        resolve_sort("RCT", lambda p: "ignored", object())  # type: ignore[arg-type]


def test_resolve_sort_rejects_false():
    """sort=False is NOT accepted -- only True is the sentinel.

    Rationale: ``False`` is ambiguous ("don't sort" overlaps with
    ``None``); we keep the sentinel narrow to the documented value.
    """
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        resolve_sort(False, lambda p: "ignored", object())


def test_sort_spec_type_alias_exists():
    """SortSpec is exported for callers that want to type-annotate."""
    # Just import-time check; nothing to assert at runtime
    assert SortSpec is not None

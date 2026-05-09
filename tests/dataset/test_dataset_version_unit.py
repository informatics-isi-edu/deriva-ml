"""Unit tests for ``DatasetVersion``'s PEP 440 behavior.

These tests exercise ``DatasetVersion`` in isolation — no catalog, no
network, no fixtures. They verify the wire-format invariants and the
PEP 440 semantics that ADR-0004 commits to:

* Released-version strings (``"M.m.p"``) round-trip identically to
  their previous semver-backed form.
* Dev versions use the ``setuptools-scm``-compatible
  ``"<release>.post1.devN"`` form.
* Ordering is correct: ``release < release.post1.devN < next_release``.
* The ``is_devrelease`` property correctly identifies dev versions.
* ``next_release`` produces a clean released version regardless of
  whether the source is a release or a dev.
"""

from __future__ import annotations

import pytest

from deriva_ml.dataset.aux_classes import DatasetVersion, VersionPart


class TestReleasedConstruction:
    """Constructing a released version from positional integers."""

    def test_str_round_trips(self):
        assert str(DatasetVersion(0, 4, 0)) == "0.4.0"
        assert str(DatasetVersion(1, 2, 3)) == "1.2.3"
        assert str(DatasetVersion(0, 0, 0)) == "0.0.0"

    def test_default_minor_and_patch_are_zero(self):
        assert str(DatasetVersion(2)) == "2.0.0"
        assert str(DatasetVersion(2, 5)) == "2.5.0"

    def test_components_accessible(self):
        v = DatasetVersion(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_patch_is_alias_for_micro(self):
        v = DatasetVersion(1, 2, 3)
        # ``packaging`` exposes the third component as ``micro``;
        # ``DatasetVersion`` adds ``patch`` to match the ``VersionPart``
        # vocabulary. Both must agree.
        assert v.patch == v.micro

    def test_released_is_not_devrelease(self):
        assert DatasetVersion(0, 4, 0).is_devrelease is False


class TestParsing:
    """Parsing via ``DatasetVersion.parse``."""

    def test_parse_released_form(self):
        v = DatasetVersion.parse("0.4.0")
        assert isinstance(v, DatasetVersion)
        assert str(v) == "0.4.0"
        assert v.is_devrelease is False

    def test_parse_dev_form(self):
        v = DatasetVersion.parse("0.4.0.post1.dev3")
        assert isinstance(v, DatasetVersion)
        assert str(v) == "0.4.0.post1.dev3"
        assert v.is_devrelease is True

    def test_parse_invalid_raises(self):
        from packaging.version import InvalidVersion

        with pytest.raises(InvalidVersion):
            DatasetVersion.parse("not-a-version")


class TestOrdering:
    """PEP 440 ordering must match the dev-versioning model."""

    def test_dev_sorts_after_its_anchor_release(self):
        anchor = DatasetVersion(0, 4, 0)
        dev = DatasetVersion.parse("0.4.0.post1.dev3")
        assert anchor < dev

    def test_dev_sorts_before_next_release(self):
        dev = DatasetVersion.parse("0.4.0.post1.dev3")
        next_release = DatasetVersion(0, 5, 0)
        assert dev < next_release

    def test_devN_advances_in_order(self):
        d1 = DatasetVersion.parse("0.4.0.post1.dev1")
        d2 = DatasetVersion.parse("0.4.0.post1.dev2")
        d10 = DatasetVersion.parse("0.4.0.post1.dev10")
        assert d1 < d2 < d10

    def test_max_picks_dev_over_anchor(self):
        # ``current_version`` relies on this: max([released, dev]) → dev
        assert max(
            [
                DatasetVersion(0, 4, 0),
                DatasetVersion.parse("0.4.0.post1.dev1"),
            ]
        ) == DatasetVersion.parse("0.4.0.post1.dev1")


class TestNextRelease:
    """``next_release`` produces a clean released version."""

    @pytest.mark.parametrize(
        "start, bump, expected",
        [
            (DatasetVersion(0, 4, 0), VersionPart.major, "1.0.0"),
            (DatasetVersion(0, 4, 0), VersionPart.minor, "0.5.0"),
            (DatasetVersion(0, 4, 0), VersionPart.patch, "0.4.1"),
            # Higher-order bumps reset lower-order components.
            (DatasetVersion(0, 4, 7), VersionPart.major, "1.0.0"),
            (DatasetVersion(0, 4, 7), VersionPart.minor, "0.5.0"),
            (DatasetVersion(0, 4, 7), VersionPart.patch, "0.4.8"),
        ],
    )
    def test_release_bumps(self, start, bump, expected):
        assert str(start.next_release(bump)) == expected

    def test_dev_promotes_to_clean_release(self):
        # The dev label makes no forward claim; release picks fresh.
        dev = DatasetVersion.parse("0.4.0.post1.dev3")
        assert str(dev.next_release(VersionPart.minor)) == "0.5.0"
        assert str(dev.next_release(VersionPart.major)) == "1.0.0"
        assert str(dev.next_release(VersionPart.patch)) == "0.4.1"

    def test_result_is_dataset_version(self):
        # next_release must return a DatasetVersion, not a packaging.Version.
        v = DatasetVersion(0, 4, 0).next_release(VersionPart.minor)
        assert isinstance(v, DatasetVersion)

    def test_result_is_not_devrelease(self):
        dev = DatasetVersion.parse("0.4.0.post1.dev3")
        promoted = dev.next_release(VersionPart.minor)
        assert promoted.is_devrelease is False


class TestSerialization:
    """``to_dict`` round-trips with the ``DatasetSpec`` serializer."""

    def test_to_dict_released(self):
        assert DatasetVersion(1, 2, 3).to_dict() == {
            "major": 1,
            "minor": 2,
            "patch": 3,
        }

    def test_to_dict_drops_postdev_segments(self):
        # to_dict captures only the release segment — by design. Use
        # ``str(v)`` for a lossless serialisation. Documented in the
        # to_dict docstring.
        dev = DatasetVersion.parse("0.4.0.post1.dev3")
        assert dev.to_dict() == {"major": 0, "minor": 4, "patch": 0}

    def test_dict_roundtrip_via_kwargs(self):
        # ``DatasetSpec.version_field_validator`` constructs via
        # ``DatasetVersion(**v)`` from a dict.
        d = {"major": 1, "minor": 2, "patch": 3}
        assert str(DatasetVersion(**d)) == "1.2.3"

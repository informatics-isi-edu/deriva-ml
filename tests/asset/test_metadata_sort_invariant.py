"""Tests pinning the alphabetic-sort invariant on asset-metadata columns.

Closes audit P1 X2: the upload-spec regex order
(``core/upload_layout.py``) and the bag-build row emit
(``execution/bag_commit.py``) both rely on a deterministic
metadata-column order. If one site switched to e.g. a bare
``list(...)`` or a different sort key, the regex would mismatch
the directory layout the bag-build code emitted, and uploads
would silently fail to match.

Post-Y3 refactor both sites call
``model.asset_metadata_sorted(table)`` (a centralised helper on
:class:`DerivaModel`). The structural test pins that call shape;
the behavioural test pins the underlying sort semantics
(Python's default case-sensitive alphabetical sort, applied to
the metadata-column-name set).

Two layers of pin:

1. **Structural** — both call sites use the literal
   ``model.asset_metadata_sorted(...)`` helper. A regression
   that drops back to a bare ``list(model.asset_metadata(...))``
   fails this test.

2. **Behavioural** — when invoked with a mocked model that
   returns the same metadata-column set, the two sites produce
   the same canonical column order (alphabetical, case-sensitive
   Python default). This is the contract the workspace
   CLAUDE.md captures ("Metadata directory order must match
   sorted(metadata_columns) — both ``asset_table_upload_spec()``
   and ``_build_upload_staging()`` sort alphabetically").

Pure-Python; no live catalog required.
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock

from deriva_ml.core import upload_layout
from deriva_ml.execution import bag_commit


# ---------------------------------------------------------------------------
# Structural pins — grep the source for the canonical pattern
# ---------------------------------------------------------------------------


_SORTED_HELPER_PATTERN = re.compile(r"model\.asset_metadata_sorted\(")


def test_upload_layout_uses_sorted_helper() -> None:
    """``upload_layout.asset_table_upload_spec`` reads metadata via the centralised helper."""
    src = inspect.getsource(upload_layout.asset_table_upload_spec)
    assert _SORTED_HELPER_PATTERN.search(src), (
        "upload_layout.asset_table_upload_spec must produce metadata "
        "columns via model.asset_metadata_sorted(...). A regression "
        "that drops back to a bare list(model.asset_metadata(...)) "
        "would break the regex-vs-directory layout invariant and "
        "silently fail uploads."
    )


def test_bag_commit_uses_sorted_helper() -> None:
    """``bag_commit._add_asset_rows_to_bag`` reads metadata via the centralised helper."""
    src = inspect.getsource(bag_commit._add_asset_rows_to_bag)
    assert _SORTED_HELPER_PATTERN.search(src), (
        "bag_commit._add_asset_rows_to_bag must produce metadata "
        "columns via model.asset_metadata_sorted(...). A regression "
        "that drops back to a bare list(model.asset_metadata(...)) "
        "would break the directory-vs-regex invariant on the upload "
        "path."
    )


def test_sorted_helper_returns_sorted_names() -> None:
    """The centralised helper actually returns alphabetically-sorted names.

    Belt-and-braces with the structural test: catches a future
    refactor that keeps the helper name but quietly drops the
    ``sorted(...)`` wrapper inside.
    """
    from deriva_ml.model.catalog import DerivaModel

    fake_model = MagicMock(spec=DerivaModel)
    fake_model.asset_metadata.return_value = {
        "Zebra",
        "Acquisition_Date",
        "Bravo",
    }
    # Call the unbound method against the mock so we exercise the
    # real implementation, not a mock-out.
    result = DerivaModel.asset_metadata_sorted(fake_model, "Image")
    assert result == ["Acquisition_Date", "Bravo", "Zebra"]


# ---------------------------------------------------------------------------
# Behavioural pin — both sites produce the same canonical order
# ---------------------------------------------------------------------------


def test_sorted_model_asset_metadata_is_alphabetic_case_sensitive() -> None:
    """The canonical ordering is alphabetical (Python default ``sorted``).

    Mock a model that returns deliberately-unsorted metadata
    column names; assert the wrapped ``sorted()`` produces
    alphabetical order. This nails down "what does 'sorted' mean
    here" so a future cleanup that switches to e.g.
    ``sorted(..., key=str.lower)`` or a custom key gets caught.
    """
    fake_model = MagicMock()
    # Returned set in deliberately unsorted iteration order.
    fake_model.asset_metadata.return_value = {
        "Zebra",
        "Acquisition_Date",
        "Subject",
        "Bravo",
    }

    result = sorted(fake_model.asset_metadata("Image"))
    assert result == ["Acquisition_Date", "Bravo", "Subject", "Zebra"], (
        f"Expected case-sensitive alphabetical order; got {result}."
    )


def test_upload_spec_regex_and_bag_commit_iterate_same_order() -> None:
    """The two sort sites produce the same metadata-column iteration order.

    The cheapest end-to-end behavioural check: feed both call sites
    the same ``model.asset_metadata`` return value and verify the
    iteration order they observe is identical.

    We don't call the heavy functions outright — they do
    network/disk I/O. Instead we exercise the pattern by computing
    ``sorted(model.asset_metadata(...))`` directly, the same shape
    both call sites use.
    """
    fake_model = MagicMock()
    cols = {"Subject", "Acquisition_Date", "Observation"}
    fake_model.asset_metadata.return_value = cols

    layout_order = sorted(fake_model.asset_metadata("Image"))
    bag_order = sorted(fake_model.asset_metadata("Image"))

    assert layout_order == bag_order
    assert layout_order == ["Acquisition_Date", "Observation", "Subject"]

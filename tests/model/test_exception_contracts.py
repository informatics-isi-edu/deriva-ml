"""Regression tests for tightened exception contracts in ``model/catalog.py``.

Alignment-audit Batch G (**G3**) tightened
:meth:`deriva_ml.model.catalog.DerivaModel.name_to_table` to raise the
documented subclass :class:`DerivaMLTableNotFound` instead of the bare
:class:`DerivaMLException` when the named table doesn't exist in any
searchable schema. This propagates to every caller — most notably
:meth:`DerivaModel.lookup_feature`, whose docstring promises
``DerivaMLTableNotFound`` for an unknown target table.

The audit chose to tighten ``name_to_table`` itself (rather than only
``lookup_feature``) because no caller in ``src/`` catches a *sibling*
subclass or branches on the exact exception type — every catcher that
can intercept a ``name_to_table`` failure uses ``except
DerivaMLException`` (the base), which still catches the subclass.

Each test asserts the **exact** raised type (``excinfo.type is
DerivaMLTableNotFound``), so it would fail on pre-change code (which
raised the bare ``DerivaMLException``). The model is built from the
checked-in demo schema JSON — no live catalog required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deriva.core.ermrest_model import Model

from deriva_ml.core.exceptions import (
    DerivaMLException,
    DerivaMLFeatureNotFound,
    DerivaMLTableNotFound,
)
from deriva_ml.model.catalog import DerivaModel

# The demo schema ships ``deriva-ml`` (ML schema), ``test-schema``
# (domain), and an ``Image`` table with features defined on it.
_DEMO_SCHEMA = Path(__file__).parent.parent / "dataset" / "demo-catalog-schema.json"


@pytest.fixture
def demo_model() -> DerivaModel:
    """A ``DerivaModel`` built from the checked-in demo catalog schema."""
    model = Model.fromfile("file-system", str(_DEMO_SCHEMA))
    return DerivaModel(model=model, ml_schema="deriva-ml", domain_schemas={"test-schema"})


class TestNameToTableRaisesTableNotFound:
    """``name_to_table`` on an unknown name raises ``DerivaMLTableNotFound``."""

    def test_raises_exactly_table_not_found(self, demo_model: DerivaModel) -> None:
        with pytest.raises(DerivaMLTableNotFound) as excinfo:
            demo_model.name_to_table("DefinitelyNotATable")

        # Exact-type assertion: distinguishes the post-change subclass
        # from the pre-change bare ``DerivaMLException``.
        assert excinfo.type is DerivaMLTableNotFound
        assert isinstance(excinfo.value, DerivaMLTableNotFound)
        # Backward-compatible: still a DerivaMLException.
        assert isinstance(excinfo.value, DerivaMLException)
        # Structured field carries the offending name.
        assert excinfo.value.table_name == "DefinitelyNotATable"

    def test_existing_table_still_resolves(self, demo_model: DerivaModel) -> None:
        """Sanity: a real table name resolves without raising."""
        assert demo_model.name_to_table("Image").name == "Image"


class TestLookupFeaturePropagatesTableNotFound:
    """``lookup_feature`` on an unknown table propagates ``DerivaMLTableNotFound``.

    This is the documented contract the audit was closing: the
    ``lookup_feature`` docstring promises ``DerivaMLTableNotFound`` for
    an unknown target table, but pre-change the bare
    ``DerivaMLException`` leaked out of ``name_to_table``.
    """

    def test_unknown_table_raises_table_not_found(self, demo_model: DerivaModel) -> None:
        with pytest.raises(DerivaMLTableNotFound) as excinfo:
            demo_model.lookup_feature("DefinitelyNotATable", "AnyFeature")

        assert excinfo.type is DerivaMLTableNotFound
        assert isinstance(excinfo.value, DerivaMLException)

    def test_known_table_unknown_feature_raises_feature_not_found(self, demo_model: DerivaModel) -> None:
        """A valid table with a missing feature raises ``DerivaMLFeatureNotFound``
        (the other documented branch) — confirms the table path succeeded."""
        with pytest.raises(DerivaMLFeatureNotFound) as excinfo:
            demo_model.lookup_feature("Image", "NoSuchFeatureXYZ")

        assert excinfo.type is DerivaMLFeatureNotFound
        assert isinstance(excinfo.value, DerivaMLException)

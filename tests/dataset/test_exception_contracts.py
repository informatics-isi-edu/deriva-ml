"""Regression tests for tightened exception contracts in ``dataset.py``.

Alignment-audit Batch G tightened two raise sites in
:mod:`deriva_ml.dataset.dataset` to raise the documented subclass of
:class:`DerivaMLException` instead of the bare base:

- **G1** — :meth:`Dataset.add_dataset_members` raises
  :class:`DerivaMLCycleError` (not bare ``DerivaMLException``) when a
  member addition would create a cycle in the nested-dataset graph.
  Documented in ``docs/user-guide/datasets.md`` and the method
  docstring.
- **G2** — :meth:`Dataset.release` raises
  :class:`DerivaMLValidationError` (not bare ``DerivaMLException``)
  when called on a dataset with no dev period. Documented in
  ``docs/adr/0003-dataset-dev-versioning-model.md``.

Each test asserts the **exact** raised type (``excinfo.type is
<Subclass>``), so it would fail on pre-change code (which raised the
bare ``DerivaMLException``) even though ``pytest.raises(<Subclass>)``
alone would not distinguish a base from a subclass. The tests are
pure unit tests: the ``Dataset`` instance is built via ``__new__`` and
only the minimal surface each raise path touches is stubbed, so no
live catalog is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deriva_ml.core.exceptions import (
    DerivaMLCycleError,
    DerivaMLException,
    DerivaMLValidationError,
)
from deriva_ml.dataset.dataset import Dataset

# ---------------------------------------------------------------------------
# G2 — release() on a dataset with no dev period
# ---------------------------------------------------------------------------


class TestReleaseNoDevRowRaisesValidationError:
    """``release()`` with no dev period raises ``DerivaMLValidationError``.

    Per ADR-0003, ``release`` on a dataset with no dev row is a
    validation error — the caller must ``mark_dev()`` first.
    """

    def test_raises_exactly_validation_error(self, monkeypatch) -> None:
        ds = Dataset.__new__(Dataset)
        ds.dataset_rid = "1-ABCD"
        ds.dataset_history = lambda: []  # type: ignore[method-assign]
        # ``current_version`` is a property on the class; patch it so
        # the error-message f-string evaluates without a catalog.
        monkeypatch.setattr(Dataset, "current_version", property(lambda self: "0.1.0"), raising=True)

        with pytest.raises(DerivaMLValidationError) as excinfo:
            ds.release(bump="minor", description="noop")

        # Exact-type assertion: this is what distinguishes the
        # post-change subclass from the pre-change bare base. Pre-fix,
        # ``excinfo.type`` was ``DerivaMLException`` and this assert
        # would fail.
        assert excinfo.type is DerivaMLValidationError
        assert isinstance(excinfo.value, DerivaMLValidationError)
        # Backward-compatibility: still a DerivaMLException for
        # ``except DerivaMLException`` catchers.
        assert isinstance(excinfo.value, DerivaMLException)
        assert "no dev period" in str(excinfo.value)


# ---------------------------------------------------------------------------
# G1 — add_dataset_members cycle case
# ---------------------------------------------------------------------------


class TestAddDatasetMembersCycleRaisesCycleError:
    """Adding a self-referential dataset member raises ``DerivaMLCycleError``.

    The simplest cycle is a self-reference: adding the dataset's own
    RID as a (nested-dataset) member. ``check_dataset_cycle`` returns
    True because the cycle-RID set always contains ``self.dataset_rid``.
    """

    def _cyclic_dataset(self) -> Dataset:
        """Build a stub ``Dataset`` that routes a self-RID member to the
        Dataset table and detects the cycle without any catalog access."""
        ds = Dataset.__new__(Dataset)
        ds.dataset_rid = "1-SELF"

        # A sentinel object that stands in for the Dataset table. Both
        # ``_dataset_table`` and the resolved ``rid_info.table`` point at
        # this same object so the ``rid_info.table == self._dataset_table``
        # identity check holds.
        dataset_table_sentinel = object()

        # ``_dataset_table`` is a property on the class reading from
        # ``_ml_instance``; build an ``_ml_instance`` whose model returns
        # the sentinel for ``schemas[ml_schema].tables["Dataset"]``.
        tables = {"Dataset": dataset_table_sentinel}
        schema = SimpleNamespace(tables=tables)
        model = SimpleNamespace(
            schemas={"deriva-ml": schema},
            # ``name_to_table`` is called for each association-map key to
            # build candidate_tables; return the sentinel (value unused
            # for the cycle assertion).
            name_to_table=lambda name: dataset_table_sentinel,
        )

        # ``resolve_rids`` returns ``{rid: rid_info}``. The rid_info must
        # route to the Dataset association table (table_name in the map)
        # and resolve to the Dataset table sentinel with the self-RID.
        rid_info = SimpleNamespace(
            table_name="Dataset",
            table=dataset_table_sentinel,
            rid=ds.dataset_rid,
        )
        ml_instance = SimpleNamespace(
            ml_schema="deriva-ml",
            model=model,
            resolve_rids=lambda members, candidate_tables=None: {ds.dataset_rid: rid_info},
        )
        ds._ml_instance = ml_instance

        # The association map must include "Dataset" so the member
        # passes the "is a registered element type" check.
        ds._element_to_association_map = lambda: {"Dataset": "Nested_Dataset"}  # type: ignore[method-assign]
        # No descendants → cycle-RID set is exactly {self.dataset_rid},
        # so adding the self-RID trips the cycle check.
        ds.list_dataset_children = lambda recurse=False: []  # type: ignore[method-assign]
        return ds

    def test_raises_exactly_cycle_error(self) -> None:
        ds = self._cyclic_dataset()

        with pytest.raises(DerivaMLCycleError) as excinfo:
            # ``validate=False`` skips the duplicate-member pre-check
            # (which would query ``list_dataset_members``); the cycle
            # check runs regardless of ``validate``.
            ds.add_dataset_members([ds.dataset_rid], validate=False)

        # Exact-type assertion distinguishing post-change subclass from
        # the pre-change bare ``DerivaMLException``.
        assert excinfo.type is DerivaMLCycleError
        assert isinstance(excinfo.value, DerivaMLCycleError)
        assert isinstance(excinfo.value, DerivaMLException)
        # Structured field carries the nodes involved in the cycle.
        assert ds.dataset_rid in excinfo.value.cycle_nodes
        assert "cycle" in str(excinfo.value).lower()

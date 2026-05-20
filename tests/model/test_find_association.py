"""Integration tests for ``DerivaModel.find_association`` typed exceptions.

These tests exercise the real catalog model rather than mocks, because
``find_association`` walks association tables via ``Table.find_associations``
which has no clean stub surface. The "no association" case is exercised
against the unmodified demo schema. The "multiple associations" case
requires building a second association table at test time, because
``Table.find_associations`` only recognises tables that carry a composite
key over their FK columns — a property absent from the demo schema's
deliberately redundant ``Image_Dataset_Legacy`` table.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.definitions import ColumnDefinition, TableDefinition
from deriva_ml.core.enums import BuiltinTypes
from deriva_ml.core.exceptions import (
    AmbiguousAssociationException,
    DerivaMLException,
    NoAssociationException,
)


class TestFindAssociationTypedExceptions:
    """``find_association`` surfaces failure modes via typed exceptions.

    Before issue #180, both failure modes raised the bare
    :class:`DerivaMLException`, forcing callers to string-match the
    error message to distinguish them. The typed subclasses replace
    that contract.
    """

    def test_no_association_raises_no_association_exception(self, test_ml):
        """Two domain tables with no association raise ``NoAssociationException``.

        ``Subject`` and ``ClinicalRecord`` exist in the demo schema but
        are not linked by any association table — ``ClinicalRecord`` is
        linked to ``Observation`` via ``ClinicalRecord_Observation``,
        which in turn links to ``Subject``, but there is no direct
        ``Subject``-``ClinicalRecord`` association table.
        """
        with pytest.raises(NoAssociationException) as exc_info:
            test_ml.model.find_association("Subject", "ClinicalRecord")
        # Carries the structured field rather than burying it in the message.
        assert exc_info.value.table1 == "Subject"
        assert exc_info.value.table2 == "ClinicalRecord"

    def test_ambiguous_association_raises_ambiguous_exception(self, test_ml):
        """Two true association tables between the same pair raise ``AmbiguousAssociationException``.

        Build the ambiguity at test time: create two source tables and
        register them as dataset element types under different names.
        ``_define_association`` produces a real association definition
        (composite key over the FK columns), so both tables are detected
        by ``Table.find_associations``.
        """
        ml = test_ml
        # Create a fresh source table to associate with Dataset twice.
        source_table = ml.model.create_table(
            TableDefinition(
                name="AmbiguousAssocSource",
                columns=[ColumnDefinition(name="Name", type=BuiltinTypes.text)],
            )
        )

        # First association: the auto-generated Dataset_<SourceTable>.
        ml.add_dataset_element_type(source_table)

        # Second association: a manually named association table linking
        # the same two endpoints. ``_define_association`` builds a real
        # association (composite key over the FK columns) so it surfaces
        # via ``Table.find_associations``.
        dataset_table = ml.model.name_to_table("Dataset")
        second_def = ml.model._define_association(
            associates=[dataset_table, source_table],
            table_name="Dataset_AmbiguousAssocSource_Alt",
        )
        ml.model.create_table(second_def)

        with pytest.raises(AmbiguousAssociationException) as exc_info:
            ml.model.find_association(source_table, dataset_table)
        assert exc_info.value.table1 == source_table.name
        assert exc_info.value.table2 == dataset_table.name
        assert exc_info.value.count >= 2

    def test_no_association_still_caught_by_base_exception(self, test_ml):
        """Legacy ``except DerivaMLException:`` callers still catch the new subclass.

        Inheritance is not a backwards-compat shim — it's the correct
        Pythonic relationship. This test pins the contract so a future
        refactor that reparents the exception away from
        ``DerivaMLException`` is caught.
        """
        with pytest.raises(DerivaMLException):
            test_ml.model.find_association("Subject", "ClinicalRecord")

    def test_ambiguous_association_still_caught_by_base_exception(self, test_ml):
        """Same as above, for the ambiguous case."""
        ml = test_ml
        source_table = ml.model.create_table(
            TableDefinition(
                name="AmbiguousAssocSourceBase",
                columns=[ColumnDefinition(name="Name", type=BuiltinTypes.text)],
            )
        )
        ml.add_dataset_element_type(source_table)

        dataset_table = ml.model.name_to_table("Dataset")
        second_def = ml.model._define_association(
            associates=[dataset_table, source_table],
            table_name="Dataset_AmbiguousAssocSourceBase_Alt",
        )
        ml.model.create_table(second_def)

        with pytest.raises(DerivaMLException):
            ml.model.find_association(source_table, dataset_table)

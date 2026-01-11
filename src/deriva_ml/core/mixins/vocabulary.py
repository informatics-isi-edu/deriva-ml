"""Vocabulary management mixin for DerivaML.

This module provides the VocabularyMixin class which handles vocabulary
term operations including adding, looking up, and listing terms in
controlled vocabulary tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import Table
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import (
    DerivaMLException,
    DerivaMLInvalidTerm,
    DerivaMLTableTypeError,
)

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


class VocabularyMixin:
    """Mixin providing vocabulary/term management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - pathBuilder(): method returning catalog path builder

    Methods:
        add_term: Add a new term to a vocabulary table
        lookup_term: Find a term by name or synonym
        list_vocabulary_terms: List all terms in a vocabulary table
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    pathBuilder: Callable[[], Any]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_term(
        self,
        table: str | Table,
        term_name: str,
        description: str,
        synonyms: list[str] | None = None,
        exists_ok: bool = True,
    ) -> VocabularyTerm:
        """Adds a term to a vocabulary table.

        Creates a new standardized term with description and optional synonyms in a vocabulary table.
        Can either create a new term or return an existing one if it already exists.

        Args:
            table: Vocabulary table to add term to (name or Table object).
            term_name: Primary name of the term (must be unique within vocabulary).
            description: Explanation of term's meaning and usage.
            synonyms: Alternative names for the term.
            exists_ok: If True, return the existing term if found. If False, raise error.

        Returns:
            VocabularyTerm: Object representing the created or existing term.

        Raises:
            DerivaMLException: If a term exists and exists_ok=False, or if the table is not a vocabulary table.

        Examples:
            Add a new tissue type:
                >>> term = ml.add_term(
                ...     table="tissue_types",
                ...     term_name="epithelial",
                ...     description="Epithelial tissue type",
                ...     synonyms=["epithelium"]
                ... )

            Attempt to add an existing term:
                >>> term = ml.add_term("tissue_types", "epithelial", "...", exists_ok=True)
        """
        # Initialize an empty synonyms list if None
        synonyms = synonyms or []

        # Get table reference and validate if it is a vocabulary table
        table = self.model.name_to_table(table)
        pb = self.pathBuilder()
        if not (self.model.is_vocabulary(table)):
            raise DerivaMLTableTypeError("vocabulary", table.name)

        # Get schema and table names for path building
        schema_name = table.schema.name
        table_name = table.name

        try:
            # Attempt to insert a new term
            term_id = VocabularyTerm.model_validate(
                pb.schemas[schema_name]
                .tables[table_name]
                .insert(
                    [
                        {
                            "Name": term_name,
                            "Description": description,
                            "Synonyms": synonyms,
                        }
                    ],
                    defaults={"ID", "URI"},
                )[0]
            )
        except DataPathException:
            # Term exists - look it up or raise an error
            term_id = self.lookup_term(table, term_name)
            if not exists_ok:
                raise DerivaMLInvalidTerm(table.name, term_name, msg="term already exists")
        return term_id

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """Finds a term in a vocabulary table.

        Searches for a term in the specified vocabulary table, matching either the primary name
        or any of its synonyms.

        Args:
            table: Vocabulary table to search in (name or Table object).
            term_name: Name or synonym of the term to find.

        Returns:
            VocabularyTerm: The matching vocabulary term.

        Raises:
            DerivaMLVocabularyException: If the table is not a vocabulary table, or term is not found.

        Examples:
            Look up by primary name:
                >>> term = ml.lookup_term("tissue_types", "epithelial")
                >>> print(term.description)

            Look up by synonym:
                >>> term = ml.lookup_term("tissue_types", "epithelium")
        """
        # Get and validate vocabulary table reference
        vocab_table = self.model.name_to_table(table)
        if not self.model.is_vocabulary(vocab_table):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Get schema and table paths
        schema_name, table_name = vocab_table.schema.name, vocab_table.name
        schema_path = self.pathBuilder().schemas[schema_name]

        # Search for term by name or synonym
        for term in schema_path.tables[table_name].entities().fetch():
            if term_name == term["Name"] or (term["Synonyms"] and term_name in term["Synonyms"]):
                return VocabularyTerm.model_validate(term)

        # Term not found
        raise DerivaMLInvalidTerm(table_name, term_name)

    def list_vocabulary_terms(self, table: str | Table) -> list[VocabularyTerm]:
        """Lists all terms in a vocabulary table.

        Retrieves all terms, their descriptions, and synonyms from a controlled vocabulary table.

        Args:
            table: Vocabulary table to list terms from (name or Table object).

        Returns:
            list[VocabularyTerm]: List of vocabulary terms with their metadata.

        Raises:
            DerivaMLException: If table doesn't exist or is not a vocabulary table.

        Examples:
            >>> terms = ml.list_vocabulary_terms("tissue_types")
            >>> for term in terms:
            ...     print(f"{term.name}: {term.description}")
            ...     if term.synonyms:
            ...         print(f"  Synonyms: {', '.join(term.synonyms)}")
        """
        # Get path builder and table reference
        pb = self.pathBuilder()
        table = self.model.name_to_table(table.value if isinstance(table, MLVocab) else table)

        # Validate table is a vocabulary table
        if not (self.model.is_vocabulary(table)):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Fetch and convert all terms to VocabularyTerm objects
        return [VocabularyTerm(**v) for v in pb.schemas[table.schema.name].tables[table.name].entities().fetch()]

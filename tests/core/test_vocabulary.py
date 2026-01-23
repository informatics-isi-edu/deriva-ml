import pytest

from deriva_ml import DerivaMLException, DerivaMLInvalidTerm
from deriva_ml.core.definitions import VocabularyTerm, VocabularyTermHandle


class TestVocabulary:
    def test_vocabulary_create(self, test_ml):
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV1", "A vocab")
        assert next((t for t in ml_instance.model.find_vocabularies() if t.name == "CV1"), None)

        # Check new vocabulary
        assert ml_instance.model.is_vocabulary("CV1")

        # Check for non-vocabulary
        assert not ml_instance.model.is_vocabulary("Dataset")

        # Check for non-existent table
        with pytest.raises(DerivaMLException):
            ml_instance.model.is_vocabulary("FooBar")

        # Check for duplicate
        with pytest.raises(DerivaMLException):
            ml_instance.create_vocabulary("CV1", "A vocab")

    def test_vocabulary_term(self):
        """Test VocabularyTerm model."""
        term = VocabularyTerm(
            Name="Test Term",
            Synonyms=["test", "term"],
            ID="TEST:001",
            URI="http://example.com/test",
            Description="A test term",
            RID="1234",
        )

        assert term.name == "Test Term"
        assert term.synonyms == ("test", "term")  # Returns tuple, not list
        assert term.id == "TEST:001"
        assert term.uri == "http://example.com/test"
        assert term.description == "A test term"
        assert term.rid == "1234"

    def test_add_term(self, test_ml):
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV2", "A vocab")

        # Check for new term
        assert len(ml_instance.list_vocabulary_terms("CV2")) == 0
        ml_instance.add_term("CV2", "T1", description="A vocab")
        assert len(ml_instance.list_vocabulary_terms("CV2")) == 1
        assert ml_instance.lookup_term("CV2", "T1").name == "T1"
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.lookup_term("CV2", "T2")

        # Check for repeat add
        ml_instance.add_term("CV2", "T1", description="A vocab")
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.add_term("CV2", "T1", description="A vocab", exists_ok=False)

    def test_add_term_synonyms(self, test_ml):
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV3", "A vocab")

        ml_instance.add_term("CV3", "T3", synonyms=["S1", "S2"], description="A vocab")
        assert ml_instance.lookup_term("CV3", "S1").name == "T3"
        # Check synonyms

    def test_vocabulary_cache(self, test_ml):
        """Test that vocabulary term lookups are cached for performance."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_Cache", "Test caching")

        # Add multiple terms
        ml_instance.add_term("CV_Cache", "Term1", description="First term")
        ml_instance.add_term("CV_Cache", "Term2", description="Second term", synonyms=["Alias2"])
        ml_instance.add_term("CV_Cache", "Term3", description="Third term")

        # Clear any existing cache
        ml_instance.clear_vocabulary_cache("CV_Cache")

        # First lookup should populate the cache
        term1 = ml_instance.lookup_term("CV_Cache", "Term1")
        assert term1.name == "Term1"

        # Subsequent lookups should use cache (we can verify by checking cache exists)
        cache = ml_instance._get_vocab_cache()
        cache_key = (ml_instance.model.default_schema, "CV_Cache")
        assert cache_key in cache, "Cache should be populated after first lookup"

        # Lookup by synonym should work from cache
        term2 = ml_instance.lookup_term("CV_Cache", "Alias2")
        assert term2.name == "Term2"

        # Lookup another term should work from cache
        term3 = ml_instance.lookup_term("CV_Cache", "Term3")
        assert term3.name == "Term3"

        # Clear cache and verify it's empty
        ml_instance.clear_vocabulary_cache("CV_Cache")
        assert cache_key not in ml_instance._get_vocab_cache()

        # Lookup should still work (repopulates cache)
        term1_again = ml_instance.lookup_term("CV_Cache", "Term1")
        assert term1_again.name == "Term1"
        assert cache_key in ml_instance._get_vocab_cache()

    def test_add_synonym(self, test_ml):
        """Test adding synonyms to an existing term via property setter."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_Syn", "Test synonyms")

        # Add a term without synonyms
        ml_instance.add_term("CV_Syn", "MainTerm", description="A term")
        term = ml_instance.lookup_term("CV_Syn", "MainTerm")
        assert term.synonyms == ()  # Empty tuple

        # Verify term is a VocabularyTermHandle
        assert isinstance(term, VocabularyTermHandle)

        # Add a synonym via property setter
        term.synonyms = ("Alias1",)
        assert "Alias1" in term.synonyms

        # Verify lookup by synonym works (need fresh lookup to test cache invalidation)
        found_term = ml_instance.lookup_term("CV_Syn", "Alias1")
        assert found_term.name == "MainTerm"

        # Add another synonym (must include existing ones)
        term = ml_instance.lookup_term("CV_Syn", "MainTerm")
        term.synonyms = term.synonyms + ("Alias2",)
        assert "Alias1" in term.synonyms
        assert "Alias2" in term.synonyms

        # Adding existing synonym is handled by caller (no-op pattern)
        term = ml_instance.lookup_term("CV_Syn", "MainTerm")
        current = term.synonyms
        if "Alias1" not in current:
            term.synonyms = current + ("Alias1",)
        assert term.synonyms.count("Alias1") == 1

    def test_remove_synonym(self, test_ml):
        """Test removing synonyms from an existing term via property setter."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_RemSyn", "Test remove synonyms")

        # Add a term with synonyms
        ml_instance.add_term("CV_RemSyn", "MainTerm", description="A term", synonyms=["Alias1", "Alias2"])
        term = ml_instance.lookup_term("CV_RemSyn", "MainTerm")
        assert "Alias1" in term.synonyms
        assert "Alias2" in term.synonyms

        # Remove a synonym via property setter
        term.synonyms = tuple(s for s in term.synonyms if s != "Alias1")
        assert "Alias1" not in term.synonyms
        assert "Alias2" in term.synonyms

        # Verify lookup by removed synonym no longer works
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.lookup_term("CV_RemSyn", "Alias1")

        # Lookup by remaining synonym should still work
        found_term = ml_instance.lookup_term("CV_RemSyn", "Alias2")
        assert found_term.name == "MainTerm"

        # Removing non-existent synonym is handled by caller (no-op pattern)
        term = ml_instance.lookup_term("CV_RemSyn", "MainTerm")
        term.synonyms = tuple(s for s in term.synonyms if s != "NonExistent")
        assert "Alias2" in term.synonyms

    def test_vocabulary_term_handle(self, test_ml):
        """Test VocabularyTermHandle class and its property setters."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_Handle", "Test VocabularyTermHandle")

        # Add a term - should return VocabularyTermHandle
        term = ml_instance.add_term("CV_Handle", "HandleTest", description="Original description")
        assert isinstance(term, VocabularyTermHandle)
        assert term.description == "Original description"

        # Update description via property setter
        term.description = "Updated description"
        assert term.description == "Updated description"

        # Verify description persisted to catalog
        fresh_term = ml_instance.lookup_term("CV_Handle", "HandleTest")
        assert fresh_term.description == "Updated description"

    def test_vocabulary_term_handle_delete(self, test_ml):
        """Test VocabularyTermHandle.delete() method."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_HandleDel", "Test VocabularyTermHandle delete")

        # Add a term
        term = ml_instance.add_term("CV_HandleDel", "ToDelete", description="A term")

        # Delete via handle method
        term.delete()

        # Verify it's gone
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.lookup_term("CV_HandleDel", "ToDelete")

    def test_cache_invalidation_on_synonym_update(self, test_ml):
        """Test that cache is properly invalidated when synonyms are updated."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_CacheInv", "Test cache invalidation")

        # Add a term with synonyms
        ml_instance.add_term("CV_CacheInv", "CacheTerm", description="Test", synonyms=["OldAlias"])

        # Lookup to populate cache
        term = ml_instance.lookup_term("CV_CacheInv", "OldAlias")
        assert term.name == "CacheTerm"

        # Update synonyms (removes OldAlias, adds NewAlias)
        term.synonyms = ("NewAlias",)

        # Old alias should no longer work (cache was invalidated)
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.lookup_term("CV_CacheInv", "OldAlias")

        # New alias should work
        found = ml_instance.lookup_term("CV_CacheInv", "NewAlias")
        assert found.name == "CacheTerm"

    def test_synonyms_returns_tuple(self, test_ml):
        """Test that synonyms property returns tuple to prevent accidental mutation."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_Tuple", "Test tuple return")

        ml_instance.add_term("CV_Tuple", "TupleTerm", description="Test", synonyms=["a", "b"])
        term = ml_instance.lookup_term("CV_Tuple", "TupleTerm")

        # Should be a tuple, not list
        assert isinstance(term.synonyms, tuple)
        assert term.synonyms == ("a", "b")

        # Cannot append to tuple (would raise AttributeError)
        with pytest.raises(AttributeError):
            term.synonyms.append("c")

    def test_delete_term_unused(self, test_ml):
        """Test deleting an unused term."""
        ml_instance = test_ml
        ml_instance.create_vocabulary("CV_Del", "Test delete")

        # Add a term
        ml_instance.add_term("CV_Del", "ToDelete", description="A term to delete")
        assert ml_instance.lookup_term("CV_Del", "ToDelete").name == "ToDelete"

        # Delete the term
        ml_instance.delete_term("CV_Del", "ToDelete")

        # Verify it's gone
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.lookup_term("CV_Del", "ToDelete")

    def test_delete_term_in_use(self, test_ml):
        """Test that deleting a term in use raises an exception."""
        ml_instance = test_ml
        from deriva_ml import MLVocab
        from deriva_ml.execution.execution import ExecutionConfiguration

        # Add workflow and dataset types for the execution
        ml_instance.add_term(MLVocab.workflow_type, "TestWorkflow", description="Test workflow")
        ml_instance.add_term(MLVocab.dataset_type, "InUseType", description="A type that will be in use")

        # Create an execution and dataset using this type
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="TestWorkflow",
            description="Test workflow",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(dataset_types=["InUseType"], description="Test dataset")

        # Verify the type is associated with the dataset
        assert "InUseType" in dataset.dataset_types

        # Attempt to delete the term should fail
        with pytest.raises(DerivaMLException) as exc_info:
            ml_instance.delete_term(MLVocab.dataset_type, "InUseType")

        assert "referenced by" in str(exc_info.value)
        assert "InUseType" in str(exc_info.value)

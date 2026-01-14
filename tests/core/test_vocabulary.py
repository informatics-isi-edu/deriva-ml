import pytest

from deriva_ml import DerivaMLException, DerivaMLInvalidTerm
from deriva_ml.core.definitions import VocabularyTerm


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
        assert term.synonyms == ["test", "term"]
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
        cache_key = (ml_instance.model.domain_schema, "CV_Cache")
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

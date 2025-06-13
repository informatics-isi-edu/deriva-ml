

import pytest

from deriva_ml import DerivaMLException, DerivaMLInvalidTerm


class TestVocabulary:

    def test_vocabulary_create(self, test_ml_catalog):
        ml_instance = test_ml_catalog
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

    def test_add_term(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.create_vocabulary("CV2", "A vocab")

        # Check for new term
        assert len(ml_instance.list_vocabulary_terms("CV2")) == 0
        term = ml_instance.add_term("CV2", "T1", description="A vocab")
        assert len(ml_instance.list_vocabulary_terms("CV2")) == 1
        assert ml_instance.lookup_term('CV2', 'T1').name == "T1"
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.lookup_term('CV2', 'T2')

        # Check for repeat add
        ml_instance.add_term("CV2", "T1", description="A vocab")
        with pytest.raises(DerivaMLInvalidTerm):
            ml_instance.add_term("CV2", "T1", description="A vocab", exists_ok=False)


    def test_add_term_synonyms(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.create_vocabulary("CV3", "A vocab")

        term = ml_instance.add_term("CV3", "T3", synonyms=["S1", "S2"], description = "A vocab")
        assert ml_instance.lookup_term('CV3', 'S1').name == "T3"
        # Check synonums




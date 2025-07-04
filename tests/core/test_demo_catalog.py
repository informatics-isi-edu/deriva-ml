from deriva_ml import MLVocab


class TestVocabulary:
    def test_demo_vocabularies(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        vocabs = ml_instance.model.find_vocabularies()
        assert len([v for v in vocabs if v.schema.name == ml_instance.ml_schema]) == len(MLVocab)
        # SubjectHeath and ImageQuality
        assert len([v for v in vocabs if v.schema.name == ml_instance.domain_schema]) == 2
        assert len(vocabs) == len(MLVocab) + 2
        assert all(map(ml_instance.model.is_vocabulary, vocabs))

    def test_dataset_relationships(self, test_ml_catalog_populated):
        """Test dataset relationship management."""
        ml_instance = test_ml_catalog_populated
        datasets = ml_instance.find_datasets()
        assert len(datasets) == 7
        double_nested_dataset = [d for d in datasets if "Complete" in d["Dataset_Type"]][0]["RID"]
        nested_datasets = [
            d["RID"] for e in ml_instance.list_dataset_members(double_nested_dataset).values() for d in e
        ]
        assert 2 == ml_instance._dataset_nesting_depth()
        assert set(ml_instance.list_dataset_children(double_nested_dataset)) == set(nested_datasets)
        assert set([d["RID"] for d in datasets]) == {double_nested_dataset}.union(
            set(ml_instance.list_dataset_children(double_nested_dataset, recurse=True))
        )

        # Check parents and children.
        assert len(ml_instance.list_dataset_children(nested_datasets[0])) == 2
        assert double_nested_dataset == ml_instance.list_dataset_parents(nested_datasets[0])[0]

        # Verify relationship
        children = ml_instance.list_dataset_children(nested_datasets[0])
        assert len(children) == 2
        parents = ml_instance.list_dataset_parents(children[0])
        assert any([nested_datasets[0] == p for p in parents])

    def test_demo_datasets(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        datasets = ml_instance.find_datasets()

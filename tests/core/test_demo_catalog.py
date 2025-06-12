class TestVocabulary:
    def test_demo_vocabularies(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        vocabs = ml_instance.model.find_vocabularies()
        assert len([v for v in vocabs if v.schema.name == ml_instance.ml_schema]) == 6
        assert len([v for v in vocabs if v.schema.name == ml_instance.domain_schema]) == 2
        assert len(vocabs) == 8
        assert all(map(ml_instance.model.is_vocabulary, vocabs))

    def test_demo_terms(self, test_ml_catalog_populated):
        pass

    def test_demo_assets(self, test_ml_catalog_populated):
        pass

    def test_dataset_relationships(self, test_ml_catalog_populated):
        """Test dataset relationship management."""
        ml_instance = test_ml_catalog_populated
        datasets = ml_instance.find_datasets()
        double_nested_dataset = [d for d in datasets if "Complete" in d["Dataset_Type"]][0]["RID"]
        nested_datasets = [
            d["RID"] for e in ml_instance.list_dataset_members(double_nested_dataset).values() for d in e
        ]
        assert 2 == ml_instance._dataset_nesting_depth()
        assert set(nested_datasets) == set(ml_instance.list_dataset_children(double_nested_dataset))
        assert set([d["RID"] for d in datasets]) == {double_nested_dataset}.union(
            set(ml_instance.list_dataset_children(double_nested_dataset, recurse=True))
        )

        # Check parents and children.
        assert 2 == len(ml_instance.list_dataset_children(nested_datasets[0]))
        assert double_nested_dataset == ml_instance.list_dataset_parents(nested_datasets[0])[0]

        # Verify relationship
        children = ml_instance.list_dataset_children(nested_datasets[0])

        parents = ml_instance.get_dataset_parents(children[0])
        assert any(p["RID"] == p["RID"] for p in parents)

    def test_demo_datasets(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        datasets = ml_instance.find_datasets()

    def test_demo_files(self, test_ml_catalog_populated, shared_tmp_path):
        pass

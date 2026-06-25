"""The Directory_Dataset satellite table exists with the expected shape."""


class TestDirectoryDatasetTable:
    def test_directory_dataset_table_shape(self, test_ml):
        model = test_ml.model.model
        ml_schema = test_ml.ml_schema
        assert "Directory_Dataset" in model.schemas[ml_schema].tables, (
            "Directory_Dataset satellite table must exist in the deriva-ml schema"
        )
        table = model.schemas[ml_schema].tables["Directory_Dataset"]
        colnames = {c.name for c in table.columns}
        # System columns plus the two payload columns.
        assert {"Dataset", "Path"} <= colnames

        # FK Dataset -> Dataset.RID exists.
        fk_targets = {
            (fk.pk_table.name, tuple(c.name for c in fk.foreign_key_columns))
            for fk in table.foreign_keys
        }
        assert ("Dataset", ("Dataset",)) in fk_targets, (
            "Directory_Dataset.Dataset must be an FK to Dataset.RID"
        )

    def test_directory_dataset_one_row_per_dataset(self, test_ml):
        """A key on Dataset enforces at most one Directory_Dataset row per dataset."""
        table = test_ml.model.model.schemas[test_ml.ml_schema].tables["Directory_Dataset"]
        key_colsets = {tuple(sorted(c.name for c in k.unique_columns)) for k in table.keys}
        assert ("Dataset",) in key_colsets, "expected a uniqueness key on the Dataset column"

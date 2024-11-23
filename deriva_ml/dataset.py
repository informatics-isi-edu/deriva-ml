from typing import Iterable, Any
from deriva.core.ermrest_model import Model, Table

class Dataset:
    def __init__(self, model: Model, domain_schema):

        self.table = model.schemas['deriva-ml'].tables['Dataset']
        self.domain_schema = domain_schema
        self.model = model


    def add_dataset_element_type(self, element: str | Table) -> Table:
        """
        A dataset is a heterogeneous collection of object, each of which comes from a different table. This
        routine makes it possible to add objects from the specified table to a dataset.

        :param element: Name or the table or table object that is to be added to the dataset.
        :return: The table object that was added to the dataset.
        """
        # Add table to map
        element_table = self._get_table(element)
        table = self.model.schemas[self.domain_schema].create_table(
            Table.define_association([self, element_table]))
        self.model = self.catalog.getCatalogModel()
        self.dataset_table.annotations.update(
            generate_dataset_annotations(self.model))
        self.model.apply()
        return table


    def create_dataset(self, ds_type: str | list[str], description: str) -> RID:
        """
        Create a new dataset from the specified list of RIDs.
        :param ds_type: One or more dataset types.  Must be a term from the DatasetType controlled vocabulary.
        :param description:  Description of the dataset.
        :return: New dataset RID.
        """
        # Create the entry for the new dataset and get its RID.
        ds_types = [ds_type] if isinstance(ds_type, str) else ds_type
        for ds_type in ds_types:
            if not self.lookup_term(MLVocab.dataset_type, ds_type):
                raise DerivaMLException(f'Dataset type must be a vocabulary term.')
        dataset_table_path = (
            self.pathBuilder.schemas[self.dataset_table.schema.name].tables)[self.dataset_table.name]
        dataset = dataset_table_path.insert([{'Description': description, MLVocab.dataset_type: ds_type}])[0]['RID']

        # Get the name of the association table between dataset and dataset_type.
        atable = next(self.model.schemas[self.ml_schema].tables[MLVocab.dataset_type].find_associations()).name
        self.pathBuilder.schemas[self.ml_schema].tables[atable].insert(
            [{MLVocab.dataset_type: ds_type, 'Dataset': dataset} for ds_type in ds_types])
        return dataset

    def find_datasets(self) -> Iterable[dict[str, Any]]:
        """
        Returns a list of currently available datasets.
        :return:
        """
        # Get datapath to all the tables we will need: Dataset, DatasetType and the association table.
        pb = self.pathBuilder
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        atable = next(self.model.schemas[self.ml_schema].tables[MLVocab.dataset_type].find_associations()).name
        ml_path = pb.schemas[self.ml_schema]
        dataset_type_path = ml_path.tables[MLVocab.dataset_type]
        atable_path = ml_path.tables[atable]

        # Get a list of all the dataset_type values associated with this dataset.
        datasets = []
        for dataset in dataset_path.entities().fetch():
            ds_types = atable_path.filter(atable_path.Dataset == dataset['RID']).attributes(
                atable_path.Dataset_Type).fetch()
            datasets.append(dataset | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in ds_types]})
        return datasets

    def delete_dataset(self, dataset_rid: RID) -> None:
        """
        Delete a dataset from the catalog.
        :param dataset_rid:  RID of the dataset to delete.
        :return:
        """
        # Get association table entries for this dataset
        # Delete association table entries
        pb = self.catalog.getPathBuilder()
        for assoc_table in self.find_associations(self):
            schema_path = pb.schemas[assoc_table.table.schema.name]
            table_path = schema_path.tables[assoc_table.name]
            dataset_column_path = table_path.columns[assoc_table.self_fkey.columns[0].name]
            dataset_entries = table_path.filter(dataset_column_path == dataset_rid)
            try:
                dataset_entries.delete()
            except DataPathException:
                pass

        # Delete dataset.
        dataset_path = pb.schemas[self.schema.name].tables[self.name]
        dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).delete()

    def list_dataset_element_types(self) -> Iterable[Table]:
        """
        Return the list of tables that can be included as members of a dataset.
        :return: An iterable of Table objects that can be included as an element of a dataset.
        """

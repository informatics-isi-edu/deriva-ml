"""Dataset management mixin for DerivaML.

This module provides the DatasetMixin class which handles
dataset operations including finding, creating, looking up,
deleting, and managing dataset elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

from deriva.core.ermrest_model import Table
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import RID, MLVocab
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLTableTypeError
from deriva_ml.dataset.aux_classes import DatasetSpec

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.dataset.dataset_bag import DatasetBag
    from deriva_ml.model.catalog import DerivaModel


class DatasetMixin:
    """Mixin providing dataset management operations.

    This mixin requires the host class to have:
        - model: DerivaModel instance
        - ml_schema: str - name of the ML schema
        - domain_schema: str - name of the domain schema
        - s3_bucket: str | None - S3 bucket URL for dataset storage
        - use_minid: bool - whether to use MINIDs
        - pathBuilder(): method returning catalog path builder
        - _dataset_table: property returning the Dataset table

    Methods:
        find_datasets: List all datasets in the catalog
        create_dataset: Create a new dataset
        lookup_dataset: Look up a dataset by RID or spec
        delete_dataset: Delete a dataset
        list_dataset_element_types: List types that can be added to datasets
        add_dataset_element_type: Add a new element type to datasets
        download_dataset_bag: Download a dataset as a bag
    """

    # Type hints for IDE support - actual attributes/methods from host class
    model: "DerivaModel"
    ml_schema: str
    domain_schema: str
    s3_bucket: str | None
    use_minid: bool
    pathBuilder: Callable[[], Any]

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table. Must be provided by host class."""
        raise NotImplementedError

    def find_datasets(self, deleted: bool = False) -> Iterable["Dataset"]:
        """Returns a list of currently available datasets.

        Arguments:
            deleted: If True, included the datasets that have been deleted.

        Returns:
             list of currently available datasets.
        """
        # Import here to avoid circular imports
        from deriva_ml.dataset.dataset import Dataset

        # Get datapath to the Dataset table
        pb = self.pathBuilder()
        dataset_path = pb.schemas[self._dataset_table.schema.name].tables[self._dataset_table.name]

        if deleted:
            filtered_path = dataset_path
        else:
            filtered_path = dataset_path.filter(
                (dataset_path.Deleted == False) | (dataset_path.Deleted == None)  # noqa: E711, E712
            )

        # Create Dataset objects - dataset_types is now a property that fetches from catalog
        datasets = []
        for dataset in filtered_path.entities().fetch():
            datasets.append(
                Dataset(
                    self,  # type: ignore[arg-type]
                    dataset_rid=dataset["RID"],
                    description=dataset["Description"],
                )
            )
        return datasets

    def lookup_dataset(self, dataset: RID | DatasetSpec, deleted: bool = False) -> "Dataset":
        """Looks up a dataset by RID or DatasetSpec.

        Arguments:
            dataset: Dataset RID or DatasetSpec to look up.
            deleted: If True, included the datasets that have been deleted.

        Returns:
            A Dataset object for the specified dataset RID or DatasetSpec.
        """
        if isinstance(dataset, DatasetSpec):
            dataset_rid = dataset.rid
        else:
            dataset_rid = dataset

        try:
            return [ds for ds in self.find_datasets(deleted=deleted) if ds.dataset_rid == dataset_rid][0]
        except IndexError:
            raise DerivaMLException(f"Dataset {dataset_rid} not found.")

    def delete_dataset(self, dataset: "Dataset", recurse: bool = False) -> None:
        """Delete a dataset from the catalog.

        Args:
            dataset: The dataset to delete.
            recurse: If True, delete the dataset along with any nested datasets. (Default value = False)
        """
        # Get association table entries for this dataset_table
        # Delete association table entries
        dataset_rid = dataset.dataset_rid
        if not self.model.is_dataset_rid(dataset.dataset_rid):
            raise DerivaMLException("Dataset_rid is not a dataset.")

        if parents := dataset.list_dataset_parents():
            raise DerivaMLException(f'Dataset "{dataset}" is in a nested dataset: {parents}.')

        pb = self.pathBuilder()
        dataset_path = pb.schemas[self._dataset_table.schema.name].tables[self._dataset_table.name]

        rid_list = [dataset_rid] + (dataset.list_dataset_children() if recurse else [])
        dataset_path.update([{"RID": r, "Deleted": True} for r in rid_list])

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of entities that can be added to a dataset.

        Returns:
            An iterable of Table objects that can be included as an element of a dataset.
        """

        def domain_table(table: Table) -> bool:
            return table.schema.name == self.domain_schema or table.name == self._dataset_table.name

        return [t for a in self._dataset_table.find_associations() if domain_table(t := a.other_fkeys.pop().pk_table)]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_element_type(self, element: str | Table) -> Table:
        """Makes it possible to add objects from the specified table to a dataset.

        A dataset is a heterogeneous collection of objects, each of which comes from a different table.
        This routine adds the specified table as a valid element type for datasets.

        Args:
            element: Name of the table or table object that is to be added to the dataset.

        Returns:
            The table object that was added to the dataset.
        """
        # Import here to avoid circular imports
        from deriva_ml.dataset.catalog_graph import CatalogGraph

        # Add table to map
        element_table = self.model.name_to_table(element)
        atable_def = Table.define_association([self._dataset_table, element_table])
        try:
            table = self.model.schemas[self.model.domain_schema].create_table(atable_def)
        except ValueError as e:
            if "already exists" in str(e):
                table = self.model.name_to_table(atable_def["table_name"])
            else:
                raise e

        # self.model = self.catalog.getCatalogModel()
        annotations = CatalogGraph(self, s3_bucket=self.s3_bucket, use_minid=self.use_minid).generate_dataset_download_annotations()  # type: ignore[arg-type]
        self._dataset_table.annotations.update(annotations)
        self.model.model.apply()
        return table

    def download_dataset_bag(
        self,
        dataset: DatasetSpec,
    ) -> "DatasetBag":
        """Downloads a dataset to the local filesystem and creates a MINID if needed.

        Downloads a dataset specified by DatasetSpec to the local filesystem. If the dataset doesn't have
        a MINID (Minimal Viable Identifier), one will be created. The dataset can optionally be associated
        with an execution record.

        Args:
            dataset: Specification of the dataset to download, including version and materialization options.

        Returns:
            DatasetBag: Object containing:
                - path: Local filesystem path to downloaded dataset
                - rid: Dataset's Resource Identifier
                - minid: Dataset's Minimal Viable Identifier

        Examples:
            Download with default options:
                >>> spec = DatasetSpec(rid="1-abc123")
                >>> bag = ml.download_dataset_bag(dataset=spec)
                >>> print(f"Downloaded to {bag.path}")

            Download with execution tracking:
                >>> bag = ml.download_dataset_bag(
                ...     dataset=DatasetSpec(rid="1-abc123", materialize=True),
                ...     execution_rid="1-xyz789"
                ... )
        """
        if not self.model.is_dataset_rid(dataset.rid):
            raise DerivaMLTableTypeError("Dataset", dataset.rid)
        ds = self.lookup_dataset(dataset)
        return ds.download_dataset_bag(
            version=dataset.version,
            materialize=dataset.materialize,
            use_minid=self.use_minid,
        )

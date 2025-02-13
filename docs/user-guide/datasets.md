#  Datasets

When working with ML models, it is often convenient to collect various input data into named, identifiable collections: A dataset
DerivaML provides a very flexible set of mechinisms for creating and manipulating datasets.

A dataset is a collection of objects that appear within a DerivaML catalog. Datasets can be heterogenious, containing 
sets of different object types. Its possible that you can have an object referenced twice, for example a collection of 
subjects of a study and a collection of observations.  DerivaML sorts out these repeated references and makes it possible
to view them from all of the possible paths.  DerivaML datasets allow reproducible grouping of objects to be provided to ML code

As with any other object in DerivaML, each individual dataset is identifed by its *Resource Identifier* or RID.
In addition, a dataset may have one or more dataset types, and also a version

## Dataset Types

Dataset types are assigned from a controlled vocabulary called `MLVocab.dataset_type`. You can define new dataset types
as you need:
```
from deriva_ml import MLVocab
...
ml_instance.add_term(MLVocab.dataset_type, "DemoSet", description="A test dataset_table")
```
When you create a dataset, you can provide as many dataset types as required to streamline orginizing and discovering
them in your code. 

## Creating Datasets

Its important to know how a dataset was created, so the most common way to create a dataset is within an execution:
```aiignore
# Now lets create model configuration for our program.
api_workflow = Workflow(
    name="API Workflow",
    url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/Notebooks/DerivaML%20Dataset.ipynb",
    workflow_type="Create Dataset Notebook"
)

dataset_execution = ml_instance.create_execution(
    ExecutionConfiguration(
        workflow=api_workflow,
        description="Our Sample Workflow instance")
)

subject_dataset = dataset_execution.create_dataset(
    dataset_types=['DemoSet', 'Subject'], 
    description="A subject dataset_table")
image_dataset = dataset_execution.create_dataset(
    dataset_types=['DemoSet', 'Image'], description="A image training dataset_table")
```
However, sometimes for purposes of bootstraping a catalog you might want to just create a dataset without tracking:
```aiignore
subject_dataset = dataset_execution.create_dataset(
   dataset_types=['DemoSet', 'Subject'], 
   description="A subject dataset_table")
```

## Dataset element types

In DerivaML, a dataet may consist of many different types of objects. In generally, any element of a domain model may be included in a dataset. Howerver, 
Its important to know how a 
Datasets can contain elements from the domain model. or other datasets.

## Adding members to a dataset
Dataset members, type and description.

## Nested Datasets
Listing datasets
Adding new element types to the dataset

Adding members to a dataset

Listing members of a dataset

# Dataset Versioning

Every dataset is assigned a version number which will change over the lifetime of the dataset.DerivaML uses semantic versioning to differntiate between versions of the dataset over time. A semantic version consists of three parts. Each part is an integer seperated by a dot:`major.minor.patch`. 0  The major part will change when there is a schema change to any object included in the dataset.  The minor part will change when new elements are added to a dataset, and the patch parch changes for minor alterations, such as adding or changing a comment or data cleaning.

DerivaML will automatically assign an initial version of `0.1.0` when a dataaset is first created, and increment the
minor part of the version number whenever new elements are added. It is up to the DerivaML user to otherwise increment the dataset version.

The version of a dataset can be incremented by the method: [`DerivaML.increment_version()`][deriva_ml.dataset.Dataset.increment_dataset_version]
The current version of a dataset can be returned: [`DerivaML.dataset_version()`][deriva_ml.dataset.Dataset.dataset_version], as can the version history of a dataset: [`DerivaML.dataset_history`][deriva_ml.dataset.Dataset.dataset_history].

Dataset versions must be specified when a dataset is downloaded to a compute platform for processing by ML code.
It is important to know that the values in the dataset are the values that were in place at the time the dataset was created.  This is true for the current dataset as well.  If you want to use the current values in a catalog, you must create a new dataset as part of the execution configuration. This is easily accomplished using the `increment_version` method.

# Downloading Datasets

Datasets are automatically downloaded as part of creating a new execution.  
Downloading datasets for the ML code.  Caching

MINIDs

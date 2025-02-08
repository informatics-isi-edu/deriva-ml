Version 1.2.0

- Dataset versioning with semantic versioning. Note that the current dataset version does *NOT* have the current catalog values, but rather the values at the time the dataset was created. 
To get the current values you must increment the dataset version number.  Please consult online documentation for more information on dataset and versioning.
- Streamlined create_execution.  Now all datasets are automatically downloaded and instance variable has databag classes. You no longer need to explictly create dataset_bdbag. 
- Significant performance improvement on cached dataset access and initial download
- Automatic creation of MINID for every dataset download
- Added method to restore an existing execution from local disk.

Version 1.1.4
- Fixed error when creating DatasetBag on windows platform.

Version 1.1.1

- Removed restriction on nested datasets so that now any level of nesting can be accomidated.
- Fixed bug in nested dataset download.
- Added additional methods to DatasetBag to make it easear to explore datasets.
- Added `datasets` instance variable to Execution object which has Dataset objects for all of the datasets listed in the configuration.
- Added option to DatasetBag init to provide a dataset RID or a path.  If the dataset has already been loaded, or the dataset is nested, this will return the assocated DatasetBag object.

